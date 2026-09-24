import json


BENCHMARK_FILE = "context_strategy_benchmark.json"
FAILURE_FILE = "context_failure_analysis.json"
ROBUSTNESS_FILE = "context_robustness.json"

OUTPUT_FILE = "unified_context_report.json"


STRATEGIES = [
    "original",
    "top_n",
    "evidence_aware",
    "evidence_first",
    "evidence_last",
    "sentence"
]


def load_json(filename):
    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def average(values):

    values = list(values)

    if not values:
        return 0.0

    return sum(values) / len(values)


def main():

    benchmark = load_json(
        BENCHMARK_FILE
    )

    failure_report = load_json(
        FAILURE_FILE
    )

    robustness_report = load_json(
        ROBUSTNESS_FILE
    )

    # ---------------------------------------------------------
    # Strategy-level aggregation
    # ---------------------------------------------------------

    strategy_metrics = {}

    for strategy in STRATEGIES:

        rows = []

        for result in benchmark:

            data = result["strategies"].get(
                strategy
            )

            if not data:
                continue

            judgement = data.get(
                "judgement"
            )

            if not judgement:
                continue

            rows.append({
                "context_characters":
                    data["context_characters"],

                "latency":
                    data["generation_latency_seconds"],

                "retrieval_evidence":
                    data.get(
                        "retrieval_evidence_recall",
                        result.get(
                            "retrieval_evidence_recall",
                            0.0
                        )
                    ),

                "strategy_evidence":
                    data.get(
                        "strategy_evidence_recall",
                        0.0
                    ),

                "groundedness":
                    judgement["groundedness"],

                "relevance":
                    judgement["relevance"],

                "completeness":
                    judgement["completeness"],

                "unsupported_claims":
                    judgement["unsupported_claims"]
            })

        if not rows:
            continue

        strategy_metrics[strategy] = {
            "questions":
                len(rows),

            "average_context_characters":
                average(
                    row["context_characters"]
                    for row in rows
                ),

            "average_latency_seconds":
                average(
                    row["latency"]
                    for row in rows
                ),

            "average_retrieval_evidence_recall":
                average(
                    row["retrieval_evidence"]
                    for row in rows
                ),

            "average_strategy_evidence_recall":
                average(
                    row["strategy_evidence"]
                    for row in rows
                ),

            "average_groundedness":
                average(
                    row["groundedness"]
                    for row in rows
                ),

            "average_relevance":
                average(
                    row["relevance"]
                    for row in rows
                ),

            "average_completeness":
                average(
                    row["completeness"]
                    for row in rows
                ),

            "average_unsupported_claims":
                average(
                    row["unsupported_claims"]
                    for row in rows
                )
        }

    # ---------------------------------------------------------
    # Context reduction relative to original
    # ---------------------------------------------------------

    original_context = strategy_metrics[
        "original"
    ]["average_context_characters"]

    original_evidence = strategy_metrics[
        "original"
    ]["average_strategy_evidence_recall"]

    for strategy, metrics in strategy_metrics.items():

        if original_context > 0:

            metrics[
                "context_reduction_vs_original"
            ] = (
                1
                - metrics[
                    "average_context_characters"
                ]
                / original_context
            )

        else:

            metrics[
                "context_reduction_vs_original"
            ] = 0.0

        if original_evidence > 0:

            metrics[
                "evidence_retention_vs_original"
            ] = (
                metrics[
                    "average_strategy_evidence_recall"
                ]
                / original_evidence
            )

        else:

            metrics[
                "evidence_retention_vs_original"
            ] = 0.0

        # Descriptive normalized quality index.
        metrics["quality_index"] = (
            metrics["average_groundedness"]
            + metrics["average_relevance"]
            + metrics["average_completeness"]
        ) / 6

    # ---------------------------------------------------------
    # Failure statistics
    # ---------------------------------------------------------

    failure_counts = (
        failure_report.get(
            "strategy_failure_counts",
            {}
        )
    )

    for strategy, metrics in strategy_metrics.items():

        failures = failure_counts.get(
            strategy,
            {}
        )

        total = metrics["questions"]

        failure_count = sum(
            failures.values()
        )

        metrics["failure_count"] = (
            failure_count
        )

        metrics["failure_rate"] = (
            failure_count / total
            if total
            else 0.0
        )

        metrics["context_loss_count"] = (
            failures.get(
                "context_loss",
                0
            )
        )

        metrics["retrieval_failure_count"] = (
            failures.get(
                "retrieval_failure",
                0
            )
        )

        metrics["generation_failure_count"] = sum(
            failures.get(
                failure_type,
                0
            )
            for failure_type in [
                "generation_ungrounded",
                "generation_incomplete",
                "generation_irrelevant",
                "unsupported_claims"
            ]
        )

    # ---------------------------------------------------------
    # Robustness
    # ---------------------------------------------------------

    robustness_questions = (
        robustness_report.get(
            "questions",
            {}
        )
    )

    total_questions = len(
        robustness_questions
    )

    strategy_sensitive = sum(
        data["strategy_sensitive"]
        for data in robustness_questions.values()
    )

    fully_robust = sum(
        data["fully_robust"]
        for data in robustness_questions.values()
    )

    consistently_failed = sum(
        data["consistently_failed"]
        for data in robustness_questions.values()
    )

    robustness_summary = {
        "questions":
            total_questions,

        "strategy_sensitive_questions":
            strategy_sensitive,

        "strategy_sensitive_rate": (
            strategy_sensitive / total_questions
            if total_questions
            else 0.0
        ),

        "fully_robust_questions":
            fully_robust,

        "fully_robust_rate": (
            fully_robust / total_questions
            if total_questions
            else 0.0
        ),

        "consistently_failed_questions":
            consistently_failed,

        "consistently_failed_rate": (
            consistently_failed / total_questions
            if total_questions
            else 0.0
        )
    }

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    report = {
        "summary": {
            "questions":
                total_questions,

            "strategies":
                len(strategy_metrics)
        },

        "robustness":
            robustness_summary,

        "strategies":
            strategy_metrics
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ---------------------------------------------------------
    # Console report
    # ---------------------------------------------------------

    print()
    print("=" * 120)
    print("UNIFIED CONTEXT STRATEGY REPORT")
    print("=" * 120)

    print()

    header = (
        f"{'Strategy':18}"
        f"{'Context':12}"
        f"{'Reduction':12}"
        f"{'Evidence':12}"
        f"{'Latency':12}"
        f"{'Ground':10}"
        f"{'Complete':10}"
        f"{'Failures':10}"
    )

    print(header)
    print("-" * 120)

    for strategy, metrics in strategy_metrics.items():

        print(
            f"{strategy:18}"
            f"{metrics['average_context_characters']:10.0f}  "
            f"{metrics['context_reduction_vs_original']:10.1%}  "
            f"{metrics['average_strategy_evidence_recall']:10.3f}  "
            f"{metrics['average_latency_seconds']:10.2f}s  "
            f"{metrics['average_groundedness']:8.2f}  "
            f"{metrics['average_completeness']:8.2f}  "
            f"{metrics['failure_rate']:8.1%}"
        )

    print()
    print("ROBUSTNESS")
    print("-" * 60)

    print(
        f"Strategy-sensitive questions: "
        f"{strategy_sensitive}/{total_questions} "
        f"({robustness_summary['strategy_sensitive_rate']:.1%})"
    )

    print(
        f"Fully robust questions: "
        f"{fully_robust}/{total_questions} "
        f"({robustness_summary['fully_robust_rate']:.1%})"
    )

    print(
        f"Consistently failed questions: "
        f"{consistently_failed}/{total_questions} "
        f"({robustness_summary['consistently_failed_rate']:.1%})"
    )

    print()
    print(
        f"Saved report to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()