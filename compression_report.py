import json
from statistics import mean


INPUT_FILE = "compression_generation_results.json"
OUTPUT_FILE = "compression_tradeoff_report.json"


def average(values):
    return mean(values) if values else 0.0


def build_report(results):
    strategies = {}

    for result in results:
        for strategy, data in result["strategies"].items():

            if strategy not in strategies:
                strategies[strategy] = {
                    "latencies": [],
                    "context_chars": [],
                    "groundedness": [],
                    "relevance": [],
                    "completeness": [],
                    "unsupported_claims": []
                }

            strategies[strategy]["latencies"].append(
                data["latency_seconds"]
            )

            strategies[strategy]["context_chars"].append(
                data["context_characters"]
            )

            judgement = data.get("judgement")

            if judgement:
                strategies[strategy]["groundedness"].append(
                    judgement["groundedness"]
                )

                strategies[strategy]["relevance"].append(
                    judgement["relevance"]
                )

                strategies[strategy]["completeness"].append(
                    judgement["completeness"]
                )

                strategies[strategy]["unsupported_claims"].append(
                    judgement["unsupported_claims"]
                )

    report = {}

    for strategy, data in strategies.items():
        report[strategy] = {
            "questions": len(data["latencies"]),

            "average_latency_seconds":
                average(data["latencies"]),

            "average_context_chars":
                average(data["context_chars"]),

            "average_groundedness":
                average(data["groundedness"]),

            "average_relevance":
                average(data["relevance"]),

            "average_completeness":
                average(data["completeness"]),

            "average_unsupported_claims":
                average(data["unsupported_claims"])
        }

    return report


def print_report(report):
    print("\n=== COMPRESSION TRADE-OFF REPORT ===\n")

    for strategy, metrics in report.items():
        print(f"Strategy: {strategy}")
        print(f"  Questions:              {metrics['questions']}")
        print(f"  Avg context chars:      {metrics['average_context_chars']:.1f}")
        print(f"  Avg latency:            {metrics['average_latency_seconds']:.3f}s")
        print(f"  Avg groundedness:       {metrics['average_groundedness']:.2f}/2")
        print(f"  Avg relevance:           {metrics['average_relevance']:.2f}/2")
        print(f"  Avg completeness:        {metrics['average_completeness']:.2f}/2")
        print(f"  Avg unsupported claims:  {metrics['average_unsupported_claims']:.2f}/2")
        print()


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not results:
        raise ValueError("No compression results found.")

    report = build_report(results)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_report(report)

    print(f"Saved report to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()