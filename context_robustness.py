import json


INPUT_FILE = "failure_matrix.json"
OUTPUT_FILE = "context_robustness.json"


def load_results():
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def main():

    report = load_results()

    matrix = report["matrix"]

    question_results = {}

    for question, strategy_results in matrix.items():

        total_strategies = len(strategy_results)

        successes = sum(
            failure_type == "success"
            for failure_type in strategy_results.values()
        )

        failures = total_strategies - successes

        unique_failure_types = set(
            failure_type
            for failure_type in strategy_results.values()
            if failure_type != "success"
        )

        # A question is strategy-sensitive when
        # some strategies succeed and others fail.
        strategy_sensitive = (
            successes > 0
            and failures > 0
        )

        # A question is consistently difficult when
        # every strategy fails.
        consistently_failed = (
            failures == total_strategies
            and total_strategies > 0
        )

        # A question is robust when every strategy succeeds.
        fully_robust = (
            successes == total_strategies
            and total_strategies > 0
        )

        question_results[question] = {
            "strategies_tested": total_strategies,
            "successful_strategies": successes,
            "failed_strategies": failures,

            "success_rate": (
                successes / total_strategies
                if total_strategies
                else 0.0
            ),

            "strategy_sensitive": strategy_sensitive,
            "consistently_failed": consistently_failed,
            "fully_robust": fully_robust,

            "failure_types": sorted(
                unique_failure_types
            )
        }

    # ---------------------------------------------------------
    # Aggregate robustness statistics
    # ---------------------------------------------------------

    total_questions = len(question_results)

    strategy_sensitive_count = sum(
        item["strategy_sensitive"]
        for item in question_results.values()
    )

    consistently_failed_count = sum(
        item["consistently_failed"]
        for item in question_results.values()
    )

    fully_robust_count = sum(
        item["fully_robust"]
        for item in question_results.values()
    )

    average_success_rate = (
        sum(
            item["success_rate"]
            for item in question_results.values()
        )
        / total_questions
        if total_questions
        else 0.0
    )

    summary = {
        "questions": total_questions,

        "strategy_sensitive_questions":
            strategy_sensitive_count,

        "strategy_sensitive_rate": (
            strategy_sensitive_count / total_questions
            if total_questions
            else 0.0
        ),

        "consistently_failed_questions":
            consistently_failed_count,

        "consistently_failed_rate": (
            consistently_failed_count / total_questions
            if total_questions
            else 0.0
        ),

        "fully_robust_questions":
            fully_robust_count,

        "fully_robust_rate": (
            fully_robust_count / total_questions
            if total_questions
            else 0.0
        ),

        "average_strategy_success_rate":
            average_success_rate
    }

    output = {
        "summary": summary,
        "questions": question_results
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
    print("=" * 80)
    print("CONTEXT STRATEGY ROBUSTNESS")
    print("=" * 80)

    print(
        f"Questions: "
        f"{summary['questions']}"
    )

    print(
        f"Strategy-sensitive: "
        f"{summary['strategy_sensitive_questions']} "
        f"({summary['strategy_sensitive_rate']:.1%})"
    )

    print(
        f"Consistently failed: "
        f"{summary['consistently_failed_questions']} "
        f"({summary['consistently_failed_rate']:.1%})"
    )

    print(
        f"Fully robust: "
        f"{summary['fully_robust_questions']} "
        f"({summary['fully_robust_rate']:.1%})"
    )

    print(
        f"Average strategy success rate: "
        f"{summary['average_strategy_success_rate']:.1%}"
    )

    print()
    print("-" * 80)
    print("QUESTION DETAILS")
    print("-" * 80)

    for question, data in question_results.items():

        print()
        print(question)

        print(
            f"  Success rate: "
            f"{data['success_rate']:.1%}"
        )

        print(
            f"  Successful strategies: "
            f"{data['successful_strategies']}"
        )

        print(
            f"  Failed strategies: "
            f"{data['failed_strategies']}"
        )

        if data["strategy_sensitive"]:
            print(
                "  Classification: strategy-sensitive"
            )

        elif data["consistently_failed"]:
            print(
                "  Classification: consistently failed"
            )

        elif data["fully_robust"]:
            print(
                "  Classification: fully robust"
            )

        else:
            print(
                "  Classification: mixed"
            )

        if data["failure_types"]:
            print(
                "  Failure types: "
                + ", ".join(
                    data["failure_types"]
                )
            )

    print()
    print(
        f"Saved results to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()