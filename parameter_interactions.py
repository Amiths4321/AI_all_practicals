import json
from pathlib import Path
from itertools import combinations


REGISTRY_FILE = Path("experiment_registry.json")
OUTPUT_FILE = Path("parameter_interactions.json")


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
    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metric(experiment, metric):
    metrics = experiment.get("metrics", {})

    value = metrics.get(metric)

    if value is None:
        summary = experiment.get("summary", {})
        value = summary.get(metric)

    if isinstance(value, dict):
        value = value.get("mean")

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def group_by_pair(experiments, parameter_a, parameter_b, metric):
    groups = {}

    for experiment in experiments:
        config = experiment.get("configuration", {})

        a = config.get(parameter_a)
        b = config.get(parameter_b)

        value = extract_metric(experiment, metric)

        if a is None or b is None or value is None:
            continue

        key = (a, b)

        groups.setdefault(key, []).append(value)

    return {
        f"{a}__{b}": {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
        for (a, b), values in groups.items()
    }


def interaction_strength(groups):
    """
    Measure how much the observed pair combinations vary.

    This is descriptive rather than causal.
    """

    means = [
        group["mean"]
        for group in groups.values()
        if group["count"] > 0
    ]

    if len(means) < 2:
        return 0.0

    return max(means) - min(means)


def main():
    registry = load_registry()
    experiments = registry.get("experiments", [])

    results = {
        "experiment_count": len(experiments),
        "parameter_pairs": {},
    }

    for parameter_a, parameter_b in combinations(PARAMETERS, 2):

        pair_name = f"{parameter_a}__{parameter_b}"

        results["parameter_pairs"][pair_name] = {
            "parameters": [
                parameter_a,
                parameter_b,
            ],
            "metrics": {},
        }

        for metric in METRICS:

            groups = group_by_pair(
                experiments,
                parameter_a,
                parameter_b,
                metric,
            )

            results["parameter_pairs"][pair_name]["metrics"][metric] = {
                "groups": groups,
                "interaction_strength": interaction_strength(groups),
            }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Parameter interaction analysis complete.")
    print(f"Experiments: {len(experiments)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()