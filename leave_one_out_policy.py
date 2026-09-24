import json


INPUT_FILE = "evidence_policy_benchmark.json"

MINIMUM_QUALITY = 0.80
MINIMUM_EVIDENCE = 1.0

FALLBACK_BUDGET = 3


def quality_index(judgement):

    if judgement is None:
        return 0.0

    return (
        judgement["groundedness"]
        + judgement["relevance"]
        + judgement["completeness"]
    ) / 6.0


def load_data():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def extract_rows(data):

    rows = []

    if not isinstance(data, list):
        raise ValueError(
            "Expected a top-level list."
        )

    for question_result in data:

        question = question_result.get(
            "question",
            ""
        )

        budgets = question_result.get(
            "budgets",
            {}
        )

        for budget, result in budgets.items():

            judgement = result.get(
                "judgement"
            )

            rows.append({
                "question": question,
                "budget": int(budget),

                "quality": quality_index(
                    judgement
                ),

                "evidence_recall":
                    result.get(
                        "evidence_recall",
                        0.0
                    ),

                "context_characters":
                    result.get(
                        "context_characters",
                        0
                    ),

                "latency_seconds":
                    result.get(
                        "generation_latency_seconds",
                        0.0
                    )
            })

    return rows


def learn_policy(
    training_rows
):

    budgets = {}

    for row in training_rows:

        budget = row["budget"]

        if (
            row["quality"] >= MINIMUM_QUALITY
            and
            row["evidence_recall"]
            >= MINIMUM_EVIDENCE
        ):

            budgets.setdefault(
                budget,
                []
            ).append(row)

    if not budgets:

        return {
            "budget": FALLBACK_BUDGET,
            "reason": "fallback_no_training_success"
        }

    candidates = []

    for budget, rows in budgets.items():

        average_quality = (
            sum(
                row["quality"]
                for row in rows
            )
            / len(rows)
        )

        average_context = (
            sum(
                row["context_characters"]
                for row in rows
            )
            / len(rows)
        )

        average_evidence = (
            sum(
                row["evidence_recall"]
                for row in rows
            )
            / len(rows)
        )

        candidates.append({
            "budget": budget,
            "average_quality":
                average_quality,
            "average_context_characters":
                average_context,
            "average_evidence_recall":
                average_evidence
        })

    # Choose the smallest budget that satisfies
    # the learned quality/evidence requirements.

    candidates.sort(
        key=lambda row: (
            row["budget"],
            row["average_context_characters"]
        )
    )

    selected = candidates[0]

    return {
        "budget": selected["budget"],
        "reason": "learned_from_other_questions",
        "average_quality":
            selected["average_quality"],
        "average_context_characters":
            selected["average_context_characters"],
        "average_evidence_recall":
            selected["average_evidence_recall"]
    }


def find_question_rows(
    rows,
    question
):

    return [
        row
        for row in rows
        if row["question"] == question
    ]


def find_budget_result(
    rows,
    budget
):

    matches = [
        row
        for row in rows
        if row["budget"] == budget
    ]

    if not matches:

        return None

    return matches[0]


def main():

    data = load_data()

    rows = extract_rows(data)

    questions = sorted(
        set(
            row["question"]
            for row in rows
        )
    )

    print(
        f"Questions: {len(questions)}"
    )

    results = []

    for test_question in questions:

        training_rows = [
            row
            for row in rows
            if row["question"]
            != test_question
        ]

        test_rows = find_question_rows(
            rows,
            test_question
        )

        learned_policy = learn_policy(
            training_rows
        )

        predicted_budget = (
            learned_policy["budget"]
        )

        actual_result = (
            find_budget_result(
                test_rows,
                predicted_budget
            )
        )

        if actual_result is None:

            # If the learned budget was not available
            # for the test question, use the largest
            # available budget as a conservative fallback.

            test_rows_sorted = sorted(
                test_rows,
                key=lambda row: row["budget"]
            )

            actual_result = (
                test_rows_sorted[-1]
                if test_rows_sorted
                else None
            )

            evaluation_budget = (
                actual_result["budget"]
                if actual_result
                else None
            )

        else:

            evaluation_budget = predicted_budget

        if actual_result is None:

            continue

        baseline_rows = [
            row
            for row in test_rows
            if row["budget"] == FALLBACK_BUDGET
        ]

        baseline = (
            baseline_rows[0]
            if baseline_rows
            else None
        )

        adaptive_quality = (
            actual_result["quality"]
        )

        baseline_quality = (
            baseline["quality"]
            if baseline
            else 0.0
        )

        adaptive_context = (
            actual_result["context_characters"]
        )

        baseline_context = (
            baseline["context_characters"]
            if baseline
            else 0
        )

        adaptive_evidence = (
            actual_result["evidence_recall"]
        )

        baseline_evidence = (
            baseline["evidence_recall"]
            if baseline
            else 0.0
        )

        results.append({
            "question": test_question,

            "learned_budget":
                predicted_budget,

            "evaluated_budget":
                evaluation_budget,

            "policy_reason":
                learned_policy["reason"],

            "adaptive_quality":
                adaptive_quality,

            "baseline_quality":
                baseline_quality,

            "quality_delta": (
                adaptive_quality
                - baseline_quality
            ),

            "adaptive_evidence_recall":
                adaptive_evidence,

            "baseline_evidence_recall":
                baseline_evidence,

            "evidence_loss": (
                adaptive_evidence
                < baseline_evidence
            ),

            "adaptive_context_characters":
                adaptive_context,

            "baseline_context_characters":
                baseline_context,

            "context_reduction": (
                1
                - adaptive_context
                / baseline_context
                if baseline_context > 0
                else 0.0
            )
        })

        print()
        print(
            test_question
        )

        print(
            "  learned budget:",
            predicted_budget
        )

        print(
            "  evaluated budget:",
            evaluation_budget
        )

        print(
            "  quality delta:",
            round(
                adaptive_quality
                - baseline_quality,
                3
            )
        )

        print(
            "  context reduction:",
            round(
                (
                    1
                    - adaptive_context
                    / baseline_context
                )
                if baseline_context > 0
                else 0.0,
                3
            )
        )

    total = len(results)

    if total:

        average_quality_delta = (
            sum(
                result["quality_delta"]
                for result in results
            )
            / total
        )

        average_context_reduction = (
            sum(
                result["context_reduction"]
                for result in results
            )
            / total
        )

        evidence_loss_count = sum(
            result["evidence_loss"]
            for result in results
        )

    else:

        average_quality_delta = 0.0
        average_context_reduction = 0.0
        evidence_loss_count = 0

    report = {
        "config": {
            "minimum_quality":
                MINIMUM_QUALITY,
            "minimum_evidence":
                MINIMUM_EVIDENCE,
            "fallback_budget":
                FALLBACK_BUDGET
        },

        "summary": {
            "questions": total,

            "average_quality_delta":
                average_quality_delta,

            "average_context_reduction":
                average_context_reduction,

            "evidence_loss_count":
                evidence_loss_count,

            "evidence_loss_rate": (
                evidence_loss_count / total
                if total
                else 0.0
            )
        },

        "questions": results
    }

    with open(
        "leave_one_out_policy.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 70)
    print("Leave-One-Question-Out Evaluation")
    print("=" * 70)

    print(
        "Questions:",
        total
    )

    print(
        "Average quality delta:",
        round(
            average_quality_delta,
            3
        )
    )

    print(
        "Average context reduction:",
        round(
            average_context_reduction,
            3
        )
    )

    print(
        "Evidence loss rate:",
        round(
            (
                evidence_loss_count / total
                if total
                else 0.0
            ),
            3
        )
    )

    print()
    print(
        "Saved leave_one_out_policy.json"
    )


if __name__ == "__main__":
    main()