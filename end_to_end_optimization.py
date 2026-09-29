import json
from pathlib import Path


BENCHMARK_FILE = Path(
    "end_to_end_benchmark.json"
)

ANALYTICS_FILE = Path(
    "experiment_analysis_report.json"
)

OUTPUT_FILE = Path(
    "optimization_analysis.json"
)


def load_json(path):
    if not path.exists():
        return {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def safe_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def analyze_benchmark(benchmark):
    summary = benchmark.get(
        "summary",
        {},
    )

    return {
        "questions": summary.get(
            "questions",
            0,
        ),

        "answer_rate": safe_number(
            summary.get("answer_rate")
        ),

        "abstention_rate": safe_number(
            summary.get("abstention_rate")
        ),

        "error_rate": safe_number(
            summary.get("error_rate")
        ),

        "evidence_recall": safe_number(
            summary.get(
                "mean_context_evidence_recall"
            )
        ),

        "latency_seconds": safe_number(
            summary.get(
                "mean_latency_seconds"
            )
        ),

        "generation_cache_hit_rate":
            safe_number(
                summary.get(
                    "generation_cache_hit_rate"
                )
            ),

        "judge_cache_hit_rate":
            safe_number(
                summary.get(
                    "judge_cache_hit_rate"
                )
            ),
    }


def build_tradeoffs(metrics):
    tradeoffs = []

    evidence = metrics["evidence_recall"]
    latency = metrics["latency_seconds"]

    if evidence is not None:
        if evidence >= 0.90:
            tradeoffs.append(
                "Evidence coverage is high."
            )
        elif evidence >= 0.80:
            tradeoffs.append(
                "Evidence coverage is moderate."
            )
        else:
            tradeoffs.append(
                "Evidence coverage is below the "
                "target range."
            )

    if latency is not None:
        if latency <= 5:
            tradeoffs.append(
                "Average latency is relatively low."
            )
        elif latency <= 15:
            tradeoffs.append(
                "Average latency is moderate."
            )
        else:
            tradeoffs.append(
                "Average latency is relatively high."
            )

    if (
        metrics["generation_cache_hit_rate"]
        is not None
    ):
        if (
            metrics[
                "generation_cache_hit_rate"
            ] > 0
        ):
            tradeoffs.append(
                "Generation caching is being used."
            )
        else:
            tradeoffs.append(
                "No generation cache hits were "
                "observed in this benchmark."
            )

    if metrics["error_rate"] is not None:
        if metrics["error_rate"] > 0:
            tradeoffs.append(
                "Some requests failed and should "
                "be investigated."
            )
        else:
            tradeoffs.append(
                "No controller errors were observed."
            )

    return tradeoffs


def main():

    benchmark = load_json(
        BENCHMARK_FILE
    )

    analytics = load_json(
        ANALYTICS_FILE
    )

    metrics = analyze_benchmark(
        benchmark
    )

    report = {
        "optimization_version": 1,

        "benchmark_metrics": metrics,

        "tradeoffs": build_tradeoffs(
            metrics
        ),

        "experiment_context": {
            "experiment_count":
                analytics.get(
                    "experiment_count"
                ),

            "available":
                bool(analytics),
        },

        "optimization_principles": [
            "Do not optimize a single metric in isolation.",
            "Preserve evidence coverage while reducing latency where possible.",
            "Treat context size as a cost as well as a quality mechanism.",
            "Use caching when it reduces repeated computation without changing answer correctness.",
            "Use Pareto analysis to identify trade-offs rather than selecting an overall winner.",
        ],
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    print(
        "End-to-end optimization analysis complete."
    )

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()