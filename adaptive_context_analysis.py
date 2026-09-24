import json


INPUT_FILE = "adaptive_context_benchmark.json"


def quality_index(judgement):

    if judgement is None:
        return 0.0

    return (
        judgement["groundedness"]
        + judgement["relevance"]
        + judgement["completeness"]
    ) / 6.0


def analyze(results):

    fixed = []
    adaptive = []

    for result in results:

        fixed_result = result["fixed"]
        adaptive_result = result["adaptive"]

        fixed_judgement = fixed_result["judgement"]
        adaptive_judgement = adaptive_result["judgement"]

        fixed.append({
            "context_characters":
                fixed_result["context_characters"],

            "latency":
                fixed_result["latency_seconds"],

            "quality":
                quality_index(fixed_judgement)
        })

        adaptive.append({
            "context_characters":
                adaptive_result["context_characters"],

            "latency":
                adaptive_result["latency_seconds"],

            "quality":
                quality_index(adaptive_judgement)
        })

    def average(items, key):

        values = [
            item[key]
            for item in items
        ]

        return (
            sum(values) / len(values)
            if values
            else 0.0
        )

    fixed_context = average(
        fixed,
        "context_characters"
    )

    adaptive_context = average(
        adaptive,
        "context_characters"
    )

    fixed_latency = average(
        fixed,
        "latency"
    )

    adaptive_latency = average(
        adaptive,
        "latency"
    )

    fixed_quality = average(
        fixed,
        "quality"
    )

    adaptive_quality = average(
        adaptive,
        "quality"
    )

    if fixed_context:

        context_reduction = (
            1
            - adaptive_context / fixed_context
        )

    else:

        context_reduction = 0.0

    return {
        "fixed": {
            "average_context_characters":
                fixed_context,

            "average_latency_seconds":
                fixed_latency,

            "average_quality_index":
                fixed_quality
        },

        "adaptive": {
            "average_context_characters":
                adaptive_context,

            "average_latency_seconds":
                adaptive_latency,

            "average_quality_index":
                adaptive_quality
        },

        "context_reduction":
            context_reduction,

        "quality_delta":
            adaptive_quality - fixed_quality,

        "latency_delta":
            adaptive_latency - fixed_latency
    }


def main():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    report = analyze(
        data["results"]
    )

    with open(
        "adaptive_context_analysis.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2
        )

    print()
    print("Adaptive Context Analysis")
    print("=" * 50)

    print(
        "Fixed context:",
        report["fixed"]["average_context_characters"]
    )

    print(
        "Adaptive context:",
        report["adaptive"]["average_context_characters"]
    )

    print(
        "Context reduction:",
        report["context_reduction"]
    )

    print(
        "Quality delta:",
        report["quality_delta"]
    )

    print(
        "Latency delta:",
        report["latency_delta"]
    )


if __name__ == "__main__":
    main()