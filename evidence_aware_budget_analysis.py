import json


INPUT_FILE = "evidence_aware_budget_sweep.json"
OUTPUT_FILE = "evidence_aware_budget_analysis.json"

BUDGETS = [1, 2, 3, 5, 10]

STRATEGIES = [
    "top_n",
    "evidence_aware"
]


def average(values):

    values = list(values)

    if not values:
        return 0.0

    return sum(values) / len(values)


def load_results():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def aggregate(results):

    output = {}

    for strategy in STRATEGIES:

        output[strategy] = {}

        for budget in BUDGETS:

            rows = []

            for result in results:

                data = (
                    result["strategies"]
                    .get(strategy, {})
                    .get(str(budget))
                )

                if data:
                    rows.append(data)

            if not rows:
                continue

            groundedness = average(
                row["judgement"]["groundedness"]
                for row in rows
            )

            relevance = average(
                row["judgement"]["relevance"]
                for row in rows
            )

            completeness = average(
                row["judgement"]["completeness"]
                for row in rows
            )

            unsupported = average(
                row["judgement"]["unsupported_claims"]
                for row in rows
            )

            quality_index = (
                groundedness
                + relevance
                + completeness
            ) / 6

            output[strategy][str(budget)] = {
                "questions": len(rows),

                "average_context_characters":
                    average(
                        row["context_characters"]
                        for row in rows
                    ),

                "average_evidence_recall":
                    average(
                        row["evidence_recall"]
                        for row in rows
                    ),

                "average_latency_seconds":
                    average(
                        row[
                            "generation_latency_seconds"
                        ]
                        for row in rows
                    ),

                "average_groundedness":
                    groundedness,

                "average_relevance":
                    relevance,

                "average_completeness":
                    completeness,

                "average_unsupported_claims":
                    unsupported,

                "quality_index":
                    quality_index
            }

    return output


def main():

    results = load_results()

    aggregated = aggregate(
        results
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            aggregated,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 110)
    print("TOP-N VS EVIDENCE-AWARE")
    print("=" * 110)

    for budget in BUDGETS:

        print()
        print(
            f"Budget: {budget}"
        )

        for strategy in STRATEGIES:

            data = aggregated[
                strategy
            ].get(str(budget))

            if not data:
                continue

            print(
                f"  {strategy:16} | "
                f"context="
                f"{data['average_context_characters']:6.0f} | "
                f"evidence="
                f"{data['average_evidence_recall']:.3f} | "
                f"latency="
                f"{data['average_latency_seconds']:.2f}s | "
                f"quality="
                f"{data['quality_index']:.3f}"
            )

    print()
    print(
        f"Saved results to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()