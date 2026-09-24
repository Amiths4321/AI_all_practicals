import json
from statistics import mean


INPUT_FILE = "sentence_compression_benchmark.json"
OUTPUT_FILE = "unified_compression_report.json"


def average(values):
    return mean(values) if values else 0.0


def build_report(results):

    strategies = {}

    for result in results:

        for strategy, data in result["strategies"].items():

            if strategy not in strategies:
                strategies[strategy] = {
                    "contexts": [],
                    "latencies": [],
                    "groundedness": [],
                    "completeness": [],
                    "relevance": [],
                    "unsupported_claims": []
                }

            strategies[strategy]["contexts"].append(
                data["context_characters"]
            )

            strategies[strategy]["latencies"].append(
                data["latency_seconds"]
            )

            judgement = data.get("judgement")

            if judgement:

                strategies[strategy]["groundedness"].append(
                    judgement["groundedness"]
                )

                strategies[strategy]["completeness"].append(
                    judgement["completeness"]
                )

                strategies[strategy]["relevance"].append(
                    judgement["relevance"]
                )

                strategies[strategy]["unsupported_claims"].append(
                    judgement["unsupported_claims"]
                )

    report = {}

    for strategy, data in strategies.items():

        groundedness = average(
            data["groundedness"]
        )

        completeness = average(
            data["completeness"]
        )

        relevance = average(
            data["relevance"]
        )

        # Normalized descriptive quality index.
        quality_index = (
            groundedness +
            completeness +
            relevance
        ) / 6

        report[strategy] = {
            "questions": len(
                data["contexts"]
            ),

            "average_context_characters":
                average(data["contexts"]),

            "average_latency_seconds":
                average(data["latencies"]),

            "average_groundedness":
                groundedness,

            "average_completeness":
                completeness,

            "average_relevance":
                relevance,

            "average_unsupported_claims":
                average(
                    data["unsupported_claims"]
                ),

            "quality_index":
                quality_index
        }

    return report


def calculate_reductions(report):

    if "full" not in report:
        print(
            "Warning: 'full' strategy not found."
        )
        return

    baseline = report["full"]

    baseline_context = (
        baseline["average_context_characters"]
    )

    baseline_latency = (
        baseline["average_latency_seconds"]
    )

    for strategy, metrics in report.items():

        context = (
            metrics["average_context_characters"]
        )

        latency = (
            metrics["average_latency_seconds"]
        )

        if baseline_context > 0:
            metrics["context_reduction"] = (
                1 -
                context / baseline_context
            )
        else:
            metrics["context_reduction"] = 0.0

        if baseline_latency > 0:
            metrics["latency_reduction"] = (
                1 -
                latency / baseline_latency
            )
        else:
            metrics["latency_reduction"] = 0.0


def print_report(report):

    print()
    print("=" * 120)
    print("UNIFIED COMPRESSION BENCHMARK")
    print("=" * 120)

    print(
        f"{'Strategy':<18}"
        f"{'Context':>12}"
        f"{'Latency':>12}"
        f"{'Ground':>10}"
        f"{'Complete':>10}"
        f"{'Relevant':>10}"
        f"{'Unsupported':>12}"
        f"{'Quality':>10}"
        f"{'Ctx Red.':>10}"
        f"{'Lat Red.':>10}"
    )

    print("-" * 120)

    for strategy, metrics in report.items():

        print(
            f"{strategy:<18}"
            f"{metrics['average_context_characters']:>12.1f}"
            f"{metrics['average_latency_seconds']:>12.3f}"
            f"{metrics['average_groundedness']:>10.2f}"
            f"{metrics['average_completeness']:>10.2f}"
            f"{metrics['average_relevance']:>10.2f}"
            f"{metrics['average_unsupported_claims']:>12.2f}"
            f"{metrics['quality_index']:>10.3f}"
            f"{metrics.get('context_reduction', 0):>9.1%}"
            f"{metrics.get('latency_reduction', 0):>9.1%}"
        )

    print("=" * 120)


def main():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        results = json.load(f)

    if not results:
        raise ValueError(
            "No benchmark results found."
        )

    report = build_report(results)

    calculate_reductions(report)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    print_report(report)

    print()
    print(
        f"Saved report to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()