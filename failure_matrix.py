import json


INPUT_FILE = "context_failure_analysis.json"
OUTPUT_FILE = "failure_matrix.json"


def load_results():
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def main():

    report = load_results()

    results = report["question_results"]

    # ---------------------------------------------------------
    # Build question -> strategy -> failure type
    # ---------------------------------------------------------

    matrix = {}

    strategies = []

    for item in results:

        question = item["question"]
        strategy = item["strategy"]

        if strategy not in strategies:
            strategies.append(strategy)

        if question not in matrix:
            matrix[question] = {}

        matrix[question][strategy] = (
            item["failure_type"]
        )

    # ---------------------------------------------------------
    # Count failures by question
    # ---------------------------------------------------------

    question_summary = {}

    for question, strategy_results in matrix.items():

        counts = {}

        for failure_type in strategy_results.values():

            counts[failure_type] = (
                counts.get(
                    failure_type,
                    0
                ) + 1
            )

        failures = sum(
            count
            for failure_type, count in counts.items()
            if failure_type != "success"
        )

        total = len(strategy_results)

        question_summary[question] = {
            "strategies_tested": total,
            "failures": failures,
            "successes": (
                total - failures
            ),
            "failure_rate": (
                failures / total
                if total
                else 0.0
            ),
            "failure_types": counts
        }

    # ---------------------------------------------------------
    # Strategy summary
    # ---------------------------------------------------------

    strategy_summary = {}

    for strategy in strategies:

        strategy_items = [
            item
            for item in results
            if item["strategy"] == strategy
        ]

        total = len(strategy_items)

        failures = sum(
            item["failure_type"] != "success"
            for item in strategy_items
        )

        context_losses = sum(
            item["failure_type"] == "context_loss"
            for item in strategy_items
        )

        retrieval_failures = sum(
            item["failure_type"] == "retrieval_failure"
            for item in strategy_items
        )

        generation_failures = sum(
            item["failure_type"]
            in {
                "generation_ungrounded",
                "generation_incomplete",
                "generation_irrelevant",
                "unsupported_claims"
            }
            for item in strategy_items
        )

        strategy_summary[strategy] = {
            "questions": total,
            "failures": failures,
            "failure_rate": (
                failures / total
                if total
                else 0.0
            ),
            "context_loss_count":
                context_losses,
            "retrieval_failure_count":
                retrieval_failures,
            "generation_failure_count":
                generation_failures
        }

    # ---------------------------------------------------------
    # Save report
    # ---------------------------------------------------------

    output = {
        "strategies": strategies,
        "matrix": matrix,
        "question_summary": question_summary,
        "strategy_summary": strategy_summary
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
    # Console matrix
    # ---------------------------------------------------------

    print()
    print("=" * 120)
    print("FAILURE ATTRIBUTION MATRIX")
    print("=" * 120)

    header = (
        f"{'Question':45}"
        + "".join(
            f"{strategy:22}"
            for strategy in strategies
        )
    )

    print(header)
    print("-" * 120)

    for question, strategy_results in matrix.items():

        display_question = question[:43]

        row = f"{display_question:45}"

        for strategy in strategies:

            failure_type = strategy_results.get(
                strategy,
                "-"
            )

            row += f"{failure_type:22}"

        print(row)

    print()
    print("=" * 80)
    print("STRATEGY SUMMARY")
    print("=" * 80)

    for strategy, summary in strategy_summary.items():

        print()
        print(strategy)

        print(
            f"  Failure rate: "
            f"{summary['failure_rate']:.1%}"
        )

        print(
            f"  Retrieval failures: "
            f"{summary['retrieval_failure_count']}"
        )

        print(
            f"  Context losses: "
            f"{summary['context_loss_count']}"
        )

        print(
            f"  Generation failures: "
            f"{summary['generation_failure_count']}"
        )

    print()
    print(
        f"Saved results to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()