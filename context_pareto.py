import json


INPUT_FILE = "unified_context_report.json"
OUTPUT_FILE = "context_pareto.json"


def load_report():
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def dominates(a, b):
    """
    Return True if strategy A dominates strategy B.

    Lower is better:
        context size
        latency

    Higher is better:
        evidence retention
        quality index
    """

    a_no_worse = (
        a["average_context_characters"]
        <= b["average_context_characters"]
        and
        a["average_latency_seconds"]
        <= b["average_latency_seconds"]
        and
        a["evidence_retention_vs_original"]
        >= b["evidence_retention_vs_original"]
        and
        a["quality_index"]
        >= b["quality_index"]
    )

    a_strictly_better = (
        a["average_context_characters"]
        < b["average_context_characters"]
        or
        a["average_latency_seconds"]
        < b["average_latency_seconds"]
        or
        a["evidence_retention_vs_original"]
        > b["evidence_retention_vs_original"]
        or
        a["quality_index"]
        > b["quality_index"]
    )

    return (
        a_no_worse
        and a_strictly_better
    )


def main():

    report = load_report()

    strategies = report["strategies"]

    dominated_by = {
        strategy: []
        for strategy in strategies
    }

    # ---------------------------------------------------------
    # Compare every pair of strategies
    # ---------------------------------------------------------

    strategy_names = list(
        strategies.keys()
    )

    for strategy_a in strategy_names:

        for strategy_b in strategy_names:

            if strategy_a == strategy_b:
                continue

            if dominates(
                strategies[strategy_b],
                strategies[strategy_a]
            ):
                dominated_by[strategy_a].append(
                    strategy_b
                )

    # ---------------------------------------------------------
    # Pareto frontier
    # ---------------------------------------------------------

    pareto_strategies = [
        strategy
        for strategy in strategy_names
        if not dominated_by[strategy]
    ]

    # ---------------------------------------------------------
    # Build detailed output
    # ---------------------------------------------------------

    details = {}

    for strategy in strategy_names:

        metrics = strategies[strategy]

        details[strategy] = {
            "average_context_characters":
                metrics[
                    "average_context_characters"
                ],

            "average_latency_seconds":
                metrics[
                    "average_latency_seconds"
                ],

            "evidence_retention_vs_original":
                metrics[
                    "evidence_retention_vs_original"
                ],

            "average_strategy_evidence_recall":
                metrics[
                    "average_strategy_evidence_recall"
                ],

            "quality_index":
                metrics[
                    "quality_index"
                ],

            "failure_rate":
                metrics[
                    "failure_rate"
                ],

            "dominated_by":
                dominated_by[strategy],

            "on_pareto_frontier":
                strategy in pareto_strategies
        }

    output = {
        "pareto_frontier":
            pareto_strategies,

        "strategies":
            details
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ---------------------------------------------------------
    # Console output
    # ---------------------------------------------------------

    print()
    print("=" * 100)
    print("CONTEXT STRATEGY PARETO ANALYSIS")
    print("=" * 100)

    print()
    print(
        "Pareto frontier:"
    )

    for strategy in pareto_strategies:
        print(
            f"  * {strategy}"
        )

    print()
    print(
        "Strategy details:"
    )

    print("-" * 100)

    for strategy, data in details.items():

        print()
        print(strategy)

        print(
            f"  Context: "
            f"{data['average_context_characters']:.0f} chars"
        )

        print(
            f"  Latency: "
            f"{data['average_latency_seconds']:.2f}s"
        )

        print(
            f"  Evidence retention: "
            f"{data['evidence_retention_vs_original']:.1%}"
        )

        print(
            f"  Evidence recall: "
            f"{data['average_strategy_evidence_recall']:.3f}"
        )

        print(
            f"  Quality index: "
            f"{data['quality_index']:.3f}"
        )

        print(
            f"  Failure rate: "
            f"{data['failure_rate']:.1%}"
        )

        if data["dominated_by"]:
            print(
                "  Dominated by: "
                + ", ".join(
                    data["dominated_by"]
                )
            )
        else:
            print(
                "  Dominated by: none"
            )

    print()
    print(
        f"Saved results to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()