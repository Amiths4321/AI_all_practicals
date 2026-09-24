import json
from statistics import mean


INPUT_FILE = "lost_middle_benchmark.json"
OUTPUT_FILE = "lost_middle_analysis.json"


POSITIONS = [
    "beginning",
    "quarter",
    "middle",
    "three_quarters",
    "end"
]


def average(values):
    return mean(values) if values else 0.0


def build_analysis(results):

    grouped = {}

    for result in results:

        context_size = result["context_size"]
        position = result["position"]

        key = (
            context_size,
            position
        )

        if key not in grouped:
            grouped[key] = {
                "groundedness": [],
                "completeness": [],
                "relevance": [],
                "unsupported_claims": [],
                "latency": []
            }

        judgement = result["judgement"]

        grouped[key]["groundedness"].append(
            judgement["groundedness"]
        )

        grouped[key]["completeness"].append(
            judgement["completeness"]
        )

        grouped[key]["relevance"].append(
            judgement["relevance"]
        )

        grouped[key]["unsupported_claims"].append(
            judgement["unsupported_claims"]
        )

        grouped[key]["latency"].append(
            result["latency_seconds"]
        )

    analysis = {}

    for context_size in sorted(
        set(result["context_size"] for result in results)
    ):

        analysis[str(context_size)] = {}

        for position in POSITIONS:

            key = (
                context_size,
                position
            )

            if key not in grouped:
                continue

            data = grouped[key]

            analysis[str(context_size)][position] = {
                "groundedness": average(
                    data["groundedness"]
                ),
                "completeness": average(
                    data["completeness"]
                ),
                "relevance": average(
                    data["relevance"]
                ),
                "unsupported_claims": average(
                    data["unsupported_claims"]
                ),
                "latency_seconds": average(
                    data["latency"]
                )
            }

    return analysis


def print_matrix(
    analysis,
    metric
):

    print()
    print("=" * 100)
    print(metric.upper())
    print("=" * 100)

    print(
        f"{'Context':>10}"
        f"{'Beginning':>14}"
        f"{'Quarter':>14}"
        f"{'Middle':>14}"
        f"{'75%':>14}"
        f"{'End':>14}"
    )

    print("-" * 100)

    for context_size in sorted(
        analysis,
        key=lambda value: int(value)
    ):

        row = (
            f"{context_size:>10}"
        )

        for position in POSITIONS:

            value = analysis[
                context_size
            ][position][metric]

            row += f"{value:>14.2f}"

        print(row)


def calculate_position_range(
    analysis,
    metric
):

    print()
    print(
        f"POSITION RANGE FOR {metric.upper()}"
    )

    for context_size in sorted(
        analysis,
        key=lambda value: int(value)
    ):

        values = [
            analysis[context_size][position][metric]
            for position in POSITIONS
        ]

        minimum = min(values)
        maximum = max(values)

        print(
            f"{context_size:>3} documents: "
            f"min={minimum:.2f}, "
            f"max={maximum:.2f}, "
            f"range={maximum - minimum:.2f}"
        )


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

    analysis = build_analysis(
        results
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            analysis,
            f,
            indent=2
        )

    print_matrix(
        analysis,
        "groundedness"
    )

    print_matrix(
        analysis,
        "completeness"
    )

    print_matrix(
        analysis,
        "relevance"
    )

    print_matrix(
        analysis,
        "unsupported_claims"
    )

    calculate_position_range(
        analysis,
        "completeness"
    )

    print()
    print(
        f"Saved analysis to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()