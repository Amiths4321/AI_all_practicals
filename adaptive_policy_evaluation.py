import json


INPUT_FILE = "empirical_context_benchmark.json"


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


def classify_adaptive_result(result):

    fixed = result["fixed"]
    adaptive = result["empirical"]

    fixed_quality = quality_index(
        fixed["judgement"]
    )

    adaptive_quality = quality_index(
        adaptive["judgement"]
    )

    fixed_evidence = fixed[
        "evidence_recall"
    ]

    adaptive_evidence = adaptive[
        "evidence_recall"
    ]

    fixed_context = fixed[
        "context_characters"
    ]

    adaptive_context = adaptive[
        "context_characters"
    ]

    # --------------------------------------------------
    # Evidence loss
    # --------------------------------------------------

    evidence_loss = (
        adaptive_evidence
        < fixed_evidence
    )

    # --------------------------------------------------
    # Quality loss
    # --------------------------------------------------

    quality_loss = (
        adaptive_quality
        < fixed_quality
    )

    # --------------------------------------------------
    # Context savings
    # --------------------------------------------------

    context_saved = (
        adaptive_context
        < fixed_context
    )

    # --------------------------------------------------
    # Overall classification
    # --------------------------------------------------

    if evidence_loss:

        classification = "evidence_loss"

    elif quality_loss:

        classification = "quality_loss"

    elif context_saved:

        classification = "efficient_success"

    else:

        classification = "no_savings"

    return {
        "classification": classification,
        "fixed_quality": fixed_quality,
        "adaptive_quality": adaptive_quality,
        "quality_delta": (
            adaptive_quality
            - fixed_quality
        ),
        "fixed_evidence": fixed_evidence,
        "adaptive_evidence": adaptive_evidence,
        "context_saved": context_saved,
        "context_reduction": (
            1
            - adaptive_context / fixed_context
            if fixed_context > 0
            else 0.0
        )
    }


def aggregate(results):

    classifications = {}

    for result in results:

        analysis = classify_adaptive_result(
            result
        )

        category = analysis[
            "classification"
        ]

        classifications[category] = (
            classifications.get(
                category,
                0
            ) + 1
        )

    total = len(results)

    efficient_success = (
        classifications.get(
            "efficient_success",
            0
        )
    )

    evidence_loss = (
        classifications.get(
            "evidence_loss",
            0
        )
    )

    quality_loss = (
        classifications.get(
            "quality_loss",
            0
        )
    )

    no_savings = (
        classifications.get(
            "no_savings",
            0
        )
    )

    context_reductions = []
    quality_deltas = []

    for result in results:

        analysis = classify_adaptive_result(
            result
        )

        context_reductions.append(
            analysis["context_reduction"]
        )

        quality_deltas.append(
            analysis["quality_delta"]
        )

    return {
        "questions": total,

        "classification_counts":
            classifications,

        "efficient_success_rate": (
            efficient_success / total
            if total
            else 0.0
        ),

        "evidence_loss_rate": (
            evidence_loss / total
            if total
            else 0.0
        ),

        "quality_loss_rate": (
            quality_loss / total
            if total
            else 0.0
        ),

        "no_savings_rate": (
            no_savings / total
            if total
            else 0.0
        ),

        "average_context_reduction": (
            sum(context_reductions)
            / len(context_reductions)
            if context_reductions
            else 0.0
        ),

        "average_quality_delta": (
            sum(quality_deltas)
            / len(quality_deltas)
            if quality_deltas
            else 0.0
        )
    }


def main():

    data = load_data()

    results = data["results"]

    print(
        f"Evaluating {len(results)} questions."
    )

    question_analysis = []

    for result in results:

        analysis = classify_adaptive_result(
            result
        )

        question_analysis.append({
            "question":
                result["question"],
            **analysis
        })

    summary = aggregate(
        results
    )

    report = {
        "summary": summary,
        "questions": question_analysis
    }

    with open(
        "adaptive_policy_evaluation.json",
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
    print("Adaptive Policy Evaluation")
    print("=" * 70)

    print(
        "Questions:",
        summary["questions"]
    )

    print(
        "Efficient success rate:",
        f"{summary['efficient_success_rate']:.3f}"
    )

    print(
        "Evidence loss rate:",
        f"{summary['evidence_loss_rate']:.3f}"
    )

    print(
        "Quality loss rate:",
        f"{summary['quality_loss_rate']:.3f}"
    )

    print(
        "No savings rate:",
        f"{summary['no_savings_rate']:.3f}"
    )

    print(
        "Average context reduction:",
        f"{summary['average_context_reduction']:.3f}"
    )

    print(
        "Average quality delta:",
        f"{summary['average_quality_delta']:.3f}"
    )

    print()
    print(
        "Saved adaptive_policy_evaluation.json"
    )


if __name__ == "__main__":
    main()