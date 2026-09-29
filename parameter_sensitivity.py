from pathlib import Path
import json
import math
import statistics


BASE_DIR = Path(__file__).resolve().parent

REGISTRY_FILE = (
    BASE_DIR / "experiment_registry.json"
)

OUTPUT_FILE = (
    BASE_DIR / "parameter_sensitivity.json"
)


PARAMETERS = [
    "chunk_size",
    "overlap",
    "rerank_k",
    "default_budget",
]

METRICS = [
    "evidence_recall",
    "quality",
    "latency",
    "context",
    "success_rate",
]


def load_registry():

    with open(
        REGISTRY_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def extract_metrics(experiment):

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
        "evidence_recall": summary.get(
            "evidence_recall"
        ),

        "quality": summary.get(
            "quality"
        ),

        "latency": summary.get(
            "average_latency_seconds"
        ),

        "context": summary.get(
            "average_context_characters"
        ),

        "success_rate": failure_summary.get(
            "success_rate"
        ),
    }


def pearson_correlation(xs, ys):

    pairs = [
        (x, y)
        for x, y in zip(xs, ys)
        if (
            isinstance(x, (int, float))
            and isinstance(y, (int, float))
            and math.isfinite(x)
            and math.isfinite(y)
        )
    ]

    if len(pairs) < 3:
        return None

    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]

    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)

    numerator = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in pairs
    )

    denominator_x = math.sqrt(
        sum(
            (x - mean_x) ** 2
            for x in xs
        )
    )

    denominator_y = math.sqrt(
        sum(
            (y - mean_y) ** 2
            for y in ys
        )
    )

    denominator = (
        denominator_x *
        denominator_y
    )

    if denominator == 0:
        return None

    return numerator / denominator


def group_means(experiments, parameter):

    groups = {}

    for experiment in experiments:

        configuration = experiment.get(
            "configuration",
            {}
        )

        parameter_value = configuration.get(
            parameter
        )

        if parameter_value is None:
            continue

        metrics = extract_metrics(
            experiment
        )

        groups.setdefault(
            str(parameter_value),
            []
        ).append(metrics)

    result = {}

    for value, records in groups.items():

        result[value] = {}

        for metric in METRICS:

            values = [
                record[metric]
                for record in records
                if isinstance(
                    record[metric],
                    (int, float)
                )
            ]

            result[value][metric] = {
                "count": len(values),

                "mean": (
                    statistics.mean(values)
                    if values
                    else None
                ),

                "stdev": (
                    statistics.stdev(values)
                    if len(values) >= 2
                    else None
                ),
            }

    return result


def parameter_correlations(experiments):

    result = {}

    for parameter in PARAMETERS:

        xs = []

        for experiment in experiments:

            value = experiment.get(
                "configuration",
                {}
            ).get(parameter)

            xs.append(value)

        result[parameter] = {}

        for metric in METRICS:

            ys = [
                extract_metrics(
                    experiment
                )[metric]
                for experiment in experiments
            ]

            result[parameter][metric] = (
                pearson_correlation(
                    xs,
                    ys
                )
            )

    return result


def rank_effects(correlations):

    result = {}

    for parameter, metrics in correlations.items():

        ranked = []

        for metric, correlation in metrics.items():

            if correlation is None:
                continue

            ranked.append({
                "metric": metric,
                "correlation": correlation,
                "absolute_correlation": abs(
                    correlation
                ),
            })

        ranked.sort(
            key=lambda item:
            item["absolute_correlation"],
            reverse=True
        )

        result[parameter] = ranked

    return result


def main():

    registry = load_registry()

    experiments = registry.get(
        "experiments",
        []
    )

    correlations = (
        parameter_correlations(
            experiments
        )
    )

    analysis = {
        "experiment_count": len(
            experiments
        ),

        "parameters": PARAMETERS,

        "metrics": METRICS,

        "correlations": correlations,

        "effect_strength": rank_effects(
            correlations
        ),

        "group_means": {
            parameter: group_means(
                experiments,
                parameter
            )
            for parameter in PARAMETERS
        },
    }

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

    print("=" * 70)
    print("PARAMETER SENSITIVITY")
    print("=" * 70)

    print(
        f"Experiments analyzed: "
        f"{len(experiments)}"
    )

    print()

    for parameter in PARAMETERS:

        print(
            f"{parameter}:"
        )

        for item in analysis[
            "effect_strength"
        ][parameter]:

            print(
                f"  "
                f"{item['metric']:<20}"
                f"{item['correlation']:+.3f}"
            )

        print()

    print(
        f"Output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()