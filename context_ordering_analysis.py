import json
from statistics import mean


INPUT_FILE = "context_ordering_results.json"
OUTPUT_FILE = "context_ordering_analysis.json"

STRATEGIES = [
    "original",
    "evidence_first",
    "evidence_last",
    "alternating"
]


def average(values):
    return mean(values) if values else 0.0


def build_report(results):

    grouped = {
        strategy: {
            "groundedness": [],
            "completeness": [],
            "relevance": [],
            "unsupported_claims": [],
            "latency": [],
            "evidence_indices": []
        }
        for strategy in STRATEGIES
    }

    for question_result in results:

        for strategy, data in (
            question_result["results"].items()
        ):

            judgement = data["judgement"]

            grouped[strategy]["groundedness"].append(
                judgement["groundedness"]
            )

            grouped[strategy]["completeness"].append(
                judgement["completeness"]
            )

            grouped[strategy]["relevance"].append(
                judgement["relevance"]
            )

            grouped[strategy]["unsupported_claims"].append(
                judgement["unsupported_claims"]
            )

            grouped[strategy]["latency"].append(
                data["latency_seconds"]
            )

            grouped[strategy]["evidence_indices"].append(
                data["evidence_index"]
            )

    report = {}

    for strategy, data in grouped.items():

        if not data["groundedness"]:
            continue

        groundedness = average(
            data["groundedness"]
        )

        completeness = average(
            data["completeness"]
        )

        relevance = average(
            data["relevance"]
        )

        unsupported = average(
            data["unsupported_claims"]
        )

        report[strategy] = {
            "questions": len(
                data["groundedness"]
            ),

            "average_groundedness":
                groundedness,

            "average_completeness":
                completeness,

            "average_relevance":
                relevance,

            "average_unsupported_claims":
                unsupported,

            "average_latency_seconds":
                average(data["latency"]),

            "average_evidence_index":
                average(data["evidence_indices"]),

            "quality_index": (
                groundedness +
                completeness +
                relevance
            ) / 6
        }

    return report


def print_report(report):

    print()
    print("=" * 120)
    print("CONTEXT ORDERING ANALYSIS")
    print("=" * 120)

    print(
        f"{'Strategy':<20}"
        f"{'Questions':>10}"
        f"{'Ground':>10}"
        f"{'Complete':>10}"
        f"{'Relevant':>10}"
        f"{'Unsupported':>12}"
        f"{'Latency':>12}"
        f"{'EvidenceIdx':>14}"
        f"{'Quality':>10}"
    )

    print("-" * 120)

    for strategy in STRATEGIES:

        if strategy not in report:
            continue

        metrics = report[strategy]

        print(
            f"{strategy:<20}"
            f"{metrics['questions']:>10}"
            f"{metrics['average_groundedness']:>10.2f}"
            f"{metrics['average_completeness']:>10.2f}"
            f"{metrics['average_relevance']:>10.2f}"
            f"{metrics['average_unsupported_claims']:>12.2f}"
            f"{metrics['average_latency_seconds']:>12.3f}"
            f"{metrics['average_evidence_index']:>14.2f}"
            f"{metrics['quality_index']:>10.3f}"
        )

    print("=" * 120)


def calculate_differences(report):

    if "original" not in report:
        return {}

    baseline = report["original"]

    differences = {}

    for strategy, metrics in report.items():

        if strategy == "original":
            continue

        differences[strategy] = {
            "groundedness_delta":
                metrics["average_groundedness"]
                - baseline["average_groundedness"],

            "completeness_delta":
                metrics["average_completeness"]
                - baseline["average_completeness"],

            "relevance_delta":
                metrics["average_relevance"]
                - baseline["average_relevance"],

            "unsupported_claims_delta":
                metrics["average_unsupported_claims"]
                - baseline["average_unsupported_claims"],

            "latency_delta_seconds":
                metrics["average_latency_seconds"]
                - baseline["average_latency_seconds"],

            "quality_index_delta":
                metrics["quality_index"]
                - baseline["quality_index"]
        }

    return differences


def main():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        results = json.load(f)

    if not results:
        raise ValueError(
            "No context-ordering results found."
        )

    report = build_report(results)

    differences = calculate_differences(
        report
    )

    output = {
        "strategies": report,
        "relative_to_original": differences
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    print_report(report)

    print()
    print("RELATIVE TO ORIGINAL")
    print("-" * 80)

    for strategy, delta in differences.items():

        print()
        print(strategy)

        for metric, value in delta.items():

            print(
                f"  {metric}: {value:+.3f}"
            )

    print()
    print(
        f"Saved analysis to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()