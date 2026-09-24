import json


INPUT_FILE = "unseen_budget_comparison.json"


def average(values):

    return (
        sum(values) / len(values)
        if values
        else 0.0
    )


def main():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    questions = data["questions"]
    budgets = data["budgets"]

    summary = {}

    for budget in budgets:

        rows = [
            question["fixed"][str(budget)]
            for question in questions
        ]

        summary[str(budget)] = {
            "average_context_characters":
                average([
                    row["context_characters"]
                    for row in rows
                ]),

            "average_latency_seconds":
                average([
                    row["latency_seconds"]
                    for row in rows
                ]),

            "average_evidence_recall":
                average([
                    row["evidence_recall"]
                    for row in rows
                ]),

            "average_quality":
                average([
                    row["quality_index"]
                    for row in rows
                ])
        }

    adaptive_rows = [
        question["adaptive"]["result"]
        for question in questions
    ]

    summary["adaptive"] = {
        "average_context_characters":
            average([
                row["context_characters"]
                for row in adaptive_rows
            ]),

        "average_latency_seconds":
            average([
                row["latency_seconds"]
                for row in adaptive_rows
            ]),

        "average_evidence_recall":
            average([
                row["evidence_recall"]
                for row in adaptive_rows
            ]),

        "average_quality":
            average([
                row["quality_index"]
                for row in adaptive_rows
            ])
    }

    # ----------------------------------------------
    # Compare adaptive policy to each fixed budget.
    # ----------------------------------------------

    comparisons = {}

    for budget in budgets:

        fixed = summary[str(budget)]
        adaptive = summary["adaptive"]

        comparisons[str(budget)] = {
            "context_reduction": (
                1
                - adaptive[
                    "average_context_characters"
                ]
                / fixed[
                    "average_context_characters"
                ]
                if fixed[
                    "average_context_characters"
                ] > 0
                else 0.0
            ),

            "quality_delta":
                adaptive["average_quality"]
                - fixed["average_quality"],

            "evidence_delta":
                adaptive["average_evidence_recall"]
                - fixed["average_evidence_recall"],

            "latency_delta":
                adaptive["average_latency_seconds"]
                - fixed["average_latency_seconds"]
        }

    report = {
        "summary": summary,
        "adaptive_vs_fixed": comparisons
    }

    with open(
        "unseen_budget_analysis.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2
        )

    print()
    print("=" * 70)
    print("Unseen Question Budget Analysis")
    print("=" * 70)

    for budget in budgets:

        row = summary[str(budget)]

        print(
            f"Budget {budget}: "
            f"context={row['average_context_characters']:.1f}, "
            f"quality={row['average_quality']:.3f}, "
            f"evidence={row['average_evidence_recall']:.3f}"
        )

    adaptive = summary["adaptive"]

    print()
    print(
        f"Adaptive: "
        f"context={adaptive['average_context_characters']:.1f}, "
        f"quality={adaptive['average_quality']:.3f}, "
        f"evidence={adaptive['average_evidence_recall']:.3f}"
    )

    print()
    print("Adaptive vs fixed:")

    for budget in budgets:

        row = comparisons[str(budget)]

        print(
            f"  vs {budget}: "
            f"context_reduction="
            f"{row['context_reduction']:.3f}, "
            f"quality_delta="
            f"{row['quality_delta']:.3f}, "
            f"evidence_delta="
            f"{row['evidence_delta']:.3f}"
        )

    print()
    print(
        "Saved unseen_budget_analysis.json"
    )


if __name__ == "__main__":
    main()