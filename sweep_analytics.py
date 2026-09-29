from pathlib import Path
import json
import statistics


BASE_DIR = Path(__file__).resolve().parent

REGISTRY_FILE = (
    BASE_DIR / "experiment_registry.json"
)

OUTPUT_FILE = (
    BASE_DIR / "sweep_analytics.json"
)


def load_registry():

    with open(
        REGISTRY_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def extract_scalar(value):

    if isinstance(value, (int, float)):
        return float(value)

    return None


def get_experiment_metrics(experiment):

    summary = experiment.get(
        "summary",
        {}
    )

    failure = experiment.get(
        "metrics",
        {}
    ).get(
        "failure_monitoring",
        {}
    )

    failure_summary = failure.get(
        "summary",
        {}
    )

    return {
        "evidence_recall": extract_scalar(
            summary.get("evidence_recall")
        ),
        "quality": extract_scalar(
            summary.get("quality")
        ),
        "latency": extract_scalar(
            summary.get(
                "average_latency_seconds"
            )
        ),
        "context": extract_scalar(
            summary.get(
                "average_context_characters"
            )
        ),
        "success_rate": extract_scalar(
            failure_summary.get(
                "success_rate"
            )
        ),
        "failure_rate": extract_scalar(
            failure_summary.get(
                "failure_rate"
            )
        ),
    }


def average(values):

    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return statistics.mean(values)


def summarize(experiments):

    metric_names = [
        "evidence_recall",
        "quality",
        "latency",
        "context",
        "success_rate",
        "failure_rate",
    ]

    summary = {}

    for metric in metric_names:

        values = []

        for experiment in experiments:

            metrics = get_experiment_metrics(
                experiment
            )

            value = metrics[metric]

            if value is not None:
                values.append(value)

        summary[metric] = {
            "count": len(values),
            "mean": average(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    return summary


def dominates(a, b):

    """
    A dominates B when A is:
      - at least as good on all objectives
      - strictly better on at least one

    Higher is better:
      evidence_recall
      quality
      success_rate

    Lower is better:
      latency
      context
      failure_rate
    """

    a_metrics = get_experiment_metrics(a)
    b_metrics = get_experiment_metrics(b)

    objectives = [
        ("evidence_recall", True),
        ("quality", True),
        ("latency", False),
        ("context", False),
        ("success_rate", True),
        ("failure_rate", False),
    ]

    better_or_equal = True
    strictly_better = False

    for metric, higher_is_better in objectives:

        av = a_metrics[metric]
        bv = b_metrics[metric]

        if av is None or bv is None:
            return False

        if higher_is_better:

            if av < bv:
                better_or_equal = False

            if av > bv:
                strictly_better = True

        else:

            if av > bv:
                better_or_equal = False

            if av < bv:
                strictly_better = True

    return (
        better_or_equal
        and strictly_better
    )


def pareto_frontier(experiments):

    frontier = []

    for candidate in experiments:

        dominated = False

        for other in experiments:

            if (
                candidate["experiment_id"]
                == other["experiment_id"]
            ):
                continue

            if dominates(
                other,
                candidate
            ):

                dominated = True
                break

        if not dominated:
            frontier.append(candidate)

    return frontier


def parameter_analysis(experiments):

    parameters = [
        "chunk_size",
        "overlap",
        "rerank_k",
        "default_budget",
    ]

    result = {}

    for parameter in parameters:

        groups = {}

        for experiment in experiments:

            value = experiment.get(
                "configuration",
                {}
            ).get(parameter)

            if value is None:
                continue

            groups.setdefault(
                str(value),
                []
            )

            metrics = (
                get_experiment_metrics(
                    experiment
                )
            )

            groups[str(value)].append(
                metrics
            )

        parameter_result = {}

        for value, records in groups.items():

            parameter_result[value] = {
                "count": len(records),

                "evidence_recall": average([
                    x["evidence_recall"]
                    for x in records
                ]),

                "quality": average([
                    x["quality"]
                    for x in records
                ]),

                "latency": average([
                    x["latency"]
                    for x in records
                ]),

                "context": average([
                    x["context"]
                    for x in records
                ]),

                "success_rate": average([
                    x["success_rate"]
                    for x in records
                ]),
            }

        result[parameter] = (
            parameter_result
        )

    return result


def main():

    registry = load_registry()

    experiments = registry.get(
        "experiments",
        []
    )

    valid = []

    for experiment in experiments:

        metrics = get_experiment_metrics(
            experiment
        )

        if any(
            value is not None
            for value in metrics.values()
        ):

            valid.append(experiment)

    frontier = pareto_frontier(
        valid
    )

    analytics = {
        "experiment_count": len(
            experiments
        ),

        "valid_experiment_count": len(
            valid
        ),

        "summary": summarize(
            valid
        ),

        "pareto_frontier": [
            item["experiment_id"]
            for item in frontier
        ],

        "pareto_count": len(
            frontier
        ),

        "parameter_analysis": (
            parameter_analysis(valid)
        ),
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            analytics,
            f,
            indent=2
        )

    print("=" * 70)
    print("SWEEP ANALYTICS")
    print("=" * 70)

    print(
        f"Experiments: "
        f"{len(experiments)}"
    )

    print(
        f"Valid:       "
        f"{len(valid)}"
    )

    print(
        f"Pareto:      "
        f"{len(frontier)}"
    )

    print()
    print("Pareto configurations:")

    for experiment in frontier:

        print(
            f"  "
            f"{experiment['experiment_id']}"
        )

    print()
    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()