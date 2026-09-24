import json
from statistics import mean


INPUT_FILE = "compression_sweep.json"
OUTPUT_FILE = "compression_efficiency.json"


def average(values):
    return mean(values) if values else 0.0


def build_report(results):

    grouped = {}

    for result in results:
        max_documents = result["max_documents"]

        if max_documents not in grouped:
            grouped[max_documents] = {
                "context_chars": [],
                "latency": [],
                "groundedness": [],
                "completeness": [],
                "relevance": [],
                "unsupported_claims": []
            }

        grouped[max_documents]["context_chars"].append(
            result["context_characters"]
        )

        grouped[max_documents]["latency"].append(
            result["latency_seconds"]
        )

        judgement = result.get("judgement")

        if judgement:
            grouped[max_documents]["groundedness"].append(
                judgement["groundedness"]
            )

            grouped[max_documents]["completeness"].append(
                judgement["completeness"]
            )

            grouped[max_documents]["relevance"].append(
                judgement["relevance"]
            )

            grouped[max_documents]["unsupported_claims"].append(
                judgement["unsupported_claims"]
            )

    report = {}

    for max_documents in sorted(grouped):

        data = grouped[max_documents]

        groundedness = average(data["groundedness"])
        completeness = average(data["completeness"])
        relevance = average(data["relevance"])
        unsupported = average(data["unsupported_claims"])
        context = average(data["context_chars"])
        latency = average(data["latency"])

        # Simple quality index.
        #
        # Maximum:
        # groundedness   = 2
        # completeness   = 2
        # relevance      = 2
        #
        # Normalize to 0-1.
        quality = (
            groundedness +
            completeness +
            relevance
        ) / 6

        report[str(max_documents)] = {
            "max_documents": max_documents,
            "average_context_chars": context,
            "average_latency_seconds": latency,
            "average_groundedness": groundedness,
            "average_completeness": completeness,
            "average_relevance": relevance,
            "average_unsupported_claims": unsupported,
            "quality_index": quality
        }

    return report


def calculate_baseline(report):

    if "10" not in report:
        return

    baseline = report["10"]

    baseline_context = baseline["average_context_chars"]
    baseline_latency = baseline["average_latency_seconds"]

    for metrics in report.values():

        context = metrics["average_context_chars"]
        latency = metrics["average_latency_seconds"]

        if baseline_context > 0:
            metrics["context_reduction"] = (
                1 - context / baseline_context
            )
        else:
            metrics["context_reduction"] = 0.0

        if baseline_latency > 0:
            metrics["latency_reduction"] = (
                1 - latency / baseline_latency
            )
        else:
            metrics["latency_reduction"] = 0.0


def print_report(report):

    print()
    print("=" * 100)
    print("COMPRESSION EFFICIENCY REPORT")
    print("=" * 100)

    header = (
        f"{'Docs':>6}"
        f"{'Context':>12}"
        f"{'Latency':>12}"
        f"{'Ground':>10}"
        f"{'Complete':>10}"
        f"{'Relevant':>10}"
        f"{'Quality':>10}"
        f"{'Ctx Red.':>10}"
        f"{'Lat Red.':>10}"
    )

    print(header)
    print("-" * 100)

    for key in sorted(
        report,
        key=lambda x: int(x)
    ):

        m = report[key]

        print(
            f"{m['max_documents']:>6}"
            f"{m['average_context_chars']:>12.1f}"
            f"{m['average_latency_seconds']:>12.3f}"
            f"{m['average_groundedness']:>10.2f}"
            f"{m['average_completeness']:>10.2f}"
            f"{m['average_relevance']:>10.2f}"
            f"{m['quality_index']:>10.3f}"
            f"{m.get('context_reduction', 0):>9.1%}"
            f"{m.get('latency_reduction', 0):>9.1%}"
        )

    print("=" * 100)


def main():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        results = json.load(f)

    if not results:
        raise ValueError(
            "compression_sweep.json contains no results."
        )

    report = build_report(results)

    calculate_baseline(report)

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