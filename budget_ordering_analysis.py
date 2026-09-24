import json


INPUT_FILE = "budget_ordering_benchmark.json"
OUTPUT_FILE = "budget_ordering_analysis.json"

BUDGETS = [1, 2, 3, 5, 10]

STRATEGIES = [
    "original",
    "evidence_first",
    "evidence_last"
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

            quality = (
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
                    quality
            }

    return output


def calculate_ordering_deltas(
    aggregated
):

    deltas = {}

    for budget in BUDGETS:

        original = (
            aggregated["original"]
            .get(str(budget))
        )

        if not original:
            continue

        deltas[str(budget)] = {}

        for strategy in [
            "evidence_first",
            "evidence_last"
        ]:

            current = (
                aggregated[strategy]
                .get(str(budget))
            )

            if not current:
                continue

            deltas[str(budget)][strategy] = {
                "quality_delta":
                    current["quality_index"]
                    - original["quality_index"],

                "groundedness_delta":
                    current["average_groundedness"]
                    - original["average_groundedness"],

                "completeness_delta":
                    current["average_completeness"]
                    - original["average_completeness"],

                "evidence_recall_delta":
                    current["average_evidence_recall"]
                    - original["average_evidence_recall"],

                "latency_delta":
                    current["average_latency_seconds"]
                    - original["average_latency_seconds"]
            }

    return deltas


def main():

    results = load_results()

    aggregated = aggregate(
        results
    )

    deltas = calculate_ordering_deltas(
        aggregated
    )

    report = {
        "aggregated": aggregated,
        "ordering_deltas_vs_original": deltas
    }

    with open(
        OUTPUT_FILE,
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
    print("=" * 110)
    print("CONTEXT ORDERING × BUDGET ANALYSIS")
    print("=" * 110)

    for budget in BUDGETS:

        print()
        print(
            f"Budget: {budget}"
        )

        for strategy in STRATEGIES:

            data = (
                aggregated[strategy]
                .get(str(budget))
            )

            if not data:
                continue

            print(
                f"  {strategy:16} | "
                f"evidence="
                f"{data['average_evidence_recall']:.3f} | "
                f"ground="
                f"{data['average_groundedness']:.2f} | "
                f"complete="
                f"{data['average_completeness']:.2f} | "
                f"quality="
                f"{data['quality_index']:.3f} | "
                f"latency="
                f"{data['average_latency_seconds']:.2f}s"
            )

    print()
    print("=" * 110)
    print("DELTAS VS ORIGINAL ORDER")
    print("=" * 110)

    for budget, strategies in deltas.items():

        print()
        print(
            f"Budget: {budget}"
        )

        for strategy, delta in strategies.items():

            print(
                f"  {strategy:16} | "
                f"quality={delta['quality_delta']:+.3f} | "
                f"ground={delta['groundedness_delta']:+.3f} | "
                f"complete={delta['completeness_delta']:+.3f} | "
                f"evidence={delta['evidence_recall_delta']:+.3f} | "
                f"latency={delta['latency_delta']:+.2f}s"
            )

    print()
    print(
        f"Saved results to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()