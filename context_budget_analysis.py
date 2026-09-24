import json


INPUT_FILE = "context_budget_sweep.json"
OUTPUT_FILE = "context_budget_analysis.json"

BUDGETS = [1, 2, 3, 5, 10]


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

    aggregated = {}

    for budget in BUDGETS:

        rows = []

        for result in results:

            data = result["budgets"].get(
                str(budget)
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

        unsupported_claims = average(
            row["judgement"]["unsupported_claims"]
            for row in rows
        )

        quality_index = (
            groundedness
            + relevance
            + completeness
        ) / 6

        aggregated[budget] = {
            "questions": len(rows),

            "average_documents":
                average(
                    row["documents_used"]
                    for row in rows
                ),

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
                    row["generation_latency_seconds"]
                    for row in rows
                ),

            "average_groundedness":
                groundedness,

            "average_relevance":
                relevance,

            "average_completeness":
                completeness,

            "average_unsupported_claims":
                unsupported_claims,

            "quality_index":
                quality_index
        }

    return aggregated


def calculate_marginal_benefit(aggregated):

    marginal = {}

    for index in range(1, len(BUDGETS)):

        previous_budget = BUDGETS[index - 1]
        current_budget = BUDGETS[index]

        if (
            previous_budget not in aggregated
            or current_budget not in aggregated
        ):
            continue

        previous = aggregated[
            previous_budget
        ]

        current = aggregated[
            current_budget
        ]

        additional_documents = (
            current_budget
            - previous_budget
        )

        additional_context = (
            current["average_context_characters"]
            - previous["average_context_characters"]
        )

        additional_latency = (
            current["average_latency_seconds"]
            - previous["average_latency_seconds"]
        )

        evidence_gain = (
            current["average_evidence_recall"]
            - previous["average_evidence_recall"]
        )

        quality_gain = (
            current["quality_index"]
            - previous["quality_index"]
        )

        marginal[str(current_budget)] = {
            "from_budget": previous_budget,
            "to_budget": current_budget,

            "additional_documents":
                additional_documents,

            "additional_context_characters":
                additional_context,

            "additional_latency_seconds":
                additional_latency,

            "evidence_recall_gain":
                evidence_gain,

            "quality_index_gain":
                quality_gain,

            "evidence_gain_per_document": (
                evidence_gain
                / additional_documents
                if additional_documents
                else 0.0
            ),

            "quality_gain_per_document": (
                quality_gain
                / additional_documents
                if additional_documents
                else 0.0
            )
        }

    return marginal


def main():

    results = load_results()

    aggregated = aggregate(
        results
    )

    marginal = calculate_marginal_benefit(
        aggregated
    )

    report = {
        "budgets": aggregated,
        "marginal_benefit": marginal
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
    print("CONTEXT BUDGET ANALYSIS")
    print("=" * 110)

    print()
    print(
        f"{'Budget':8}"
        f"{'Context':12}"
        f"{'Evidence':12}"
        f"{'Latency':12}"
        f"{'Ground':10}"
        f"{'Complete':10}"
        f"{'Quality':10}"
    )

    print("-" * 110)

    for budget in BUDGETS:

        if budget not in aggregated:
            continue

        data = aggregated[budget]

        print(
            f"{budget:<8}"
            f"{data['average_context_characters']:10.0f}  "
            f"{data['average_evidence_recall']:10.3f}  "
            f"{data['average_latency_seconds']:10.2f}s  "
            f"{data['average_groundedness']:8.2f}  "
            f"{data['average_completeness']:8.2f}  "
            f"{data['quality_index']:8.3f}"
        )

    print()
    print("=" * 110)
    print("MARGINAL BENEFIT")
    print("=" * 110)

    for budget, data in marginal.items():

        print()
        print(
            f"{data['from_budget']} "
            f"-> "
            f"{data['to_budget']} documents"
        )

        print(
            f"  Additional context: "
            f"{data['additional_context_characters']:.0f} chars"
        )

        print(
            f"  Additional latency: "
            f"{data['additional_latency_seconds']:.3f}s"
        )

        print(
            f"  Evidence recall gain: "
            f"{data['evidence_recall_gain']:+.3f}"
        )

        print(
            f"  Quality gain: "
            f"{data['quality_index_gain']:+.3f}"
        )

        print(
            f"  Quality gain/document: "
            f"{data['quality_gain_per_document']:+.4f}"
        )

    print()
    print(
        f"Saved report to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()