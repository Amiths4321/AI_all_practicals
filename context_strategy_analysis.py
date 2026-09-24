import json


INPUT_FILE = "context_strategy_benchmark.json"
OUTPUT_FILE = "context_strategy_analysis.json"

STRATEGIES = [
    "original",
    "top_n",
    "evidence_aware",
    "evidence_first",
    "evidence_last",
    "sentence"
]


def load_results():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def average(values):
    values = list(values)

    if not values:
        return 0.0
    return sum(values) / len(values)


def analyze_strategy(results, strategy):
    rows = []

    for result in results:
        data = result["strategies"].get(strategy)

        if not data:
            continue

        judgement = data.get("judgement")

        if not judgement:
            continue

        rows.append({
            "context_characters": data["context_characters"],
            "latency": data["generation_latency_seconds"],

            "retrieval_evidence_recall": data.get(
                "retrieval_evidence_recall",
                result.get("retrieval_evidence_recall", 0.0)
            ),

            "strategy_evidence_recall": data.get(
                "strategy_evidence_recall",
                0.0
            ),

            "groundedness": judgement["groundedness"],
            "relevance": judgement["relevance"],
            "completeness": judgement["completeness"],
            "unsupported_claims": judgement["unsupported_claims"]
        })

    if not rows:
        return {}

    average_context = average(row["context_characters"] for row in rows)
    average_latency = average(row["latency"] for row in rows)

    average_retrieval_evidence = average(
        row["retrieval_evidence_recall"] for row in rows
    )

    average_strategy_evidence = average(
        row["strategy_evidence_recall"] for row in rows
    )

    average_groundedness = average(row["groundedness"] for row in rows)
    average_relevance = average(row["relevance"] for row in rows)
    average_completeness = average(row["completeness"] for row in rows)
    average_unsupported = average(row["unsupported_claims"] for row in rows)

    # groundedness / relevance / completeness are each on a 0-2 scale,
    # so dividing their sum by 6 normalizes the index to roughly 0-1.
    quality_index = (
        average_groundedness
        + average_relevance
        + average_completeness
    ) / 6

    return {
        "questions": len(rows),
        "average_context_characters": average_context,
        "average_latency_seconds": average_latency,
        "average_retrieval_evidence_recall": average_retrieval_evidence,
        "average_strategy_evidence_recall": average_strategy_evidence,
        "average_groundedness": average_groundedness,
        "average_relevance": average_relevance,
        "average_completeness": average_completeness,
        "average_unsupported_claims": average_unsupported,
        "quality_index": quality_index
    }


def main():
    results = load_results()

    analysis = {}

    for strategy in STRATEGIES:
        analysis[strategy] = analyze_strategy(results, strategy)

    # Use original context as the baseline.
    original_context = analysis["original"].get(
        "average_context_characters",
        0
    )

    original_evidence = analysis["original"].get(
        "average_strategy_evidence_recall",
        0.0
    )

    for strategy, metrics in analysis.items():
        if not metrics:
            continue

        if original_context > 0:
            metrics["context_reduction_vs_original"] = (
                1
                - metrics["average_context_characters"]
                / original_context
            )
        else:
            metrics["context_reduction_vs_original"] = 0.0

        if original_evidence > 0:
            metrics["evidence_retention_vs_original"] = (
                metrics["average_strategy_evidence_recall"]
                / original_evidence
            )
        else:
            metrics["evidence_retention_vs_original"] = 0.0

    # Retrieval evidence recall is independent of the context strategy,
    # since retrieval happens before strategy selection — so it's rolled
    # up once here from each result's top-level field rather than per
    # strategy. NOTE: fixed to read "retrieval_evidence_recall" (the key
    # actually used elsewhere in this file and in the source data) —
    # the original code read "evidence_recall" here, which doesn't
    # exist on these result objects and would have silently produced 0.0.
    evidence_recalls = [
        result.get("retrieval_evidence_recall", 0.0)
        for result in results
    ]

    report = {
        "questions": len(results),
        "average_retrieval_evidence_recall": average(evidence_recalls),
        "strategies": analysis
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print()
    print("Context Strategy Analysis")
    print("=" * 80)

    print(f"Questions: {len(results)}")
    print(
        "Average retrieval evidence recall:",
        round(report["average_retrieval_evidence_recall"], 3)
    )

    print()

    for strategy, metrics in analysis.items():
        if not metrics:
            continue

        print(f"Strategy: {strategy}")
        print(f"  Context chars: {metrics['average_context_characters']:.1f}")
        print(f"  Context reduction: {metrics['context_reduction_vs_original']:.1%}")
        print(f"  Latency: {metrics['average_latency_seconds']:.3f}s")
        print(f"  Retrieval evidence recall: {metrics['average_retrieval_evidence_recall']:.3f}")
        print(f"  Strategy evidence recall: {metrics['average_strategy_evidence_recall']:.3f}")
        print(f"  Evidence retention vs original: {metrics['evidence_retention_vs_original']:.1%}")
        print(f"  Groundedness: {metrics['average_groundedness']:.2f}")
        print(f"  Relevance: {metrics['average_relevance']:.2f}")
        print(f"  Completeness: {metrics['average_completeness']:.2f}")
        print(f"  Unsupported claims: {metrics['average_unsupported_claims']:.2f}")
        print(f"  Quality index: {metrics['quality_index']:.3f}")
        print()


if __name__ == "__main__":
    main()