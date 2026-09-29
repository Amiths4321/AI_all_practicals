import json
from pathlib import Path
from datetime import datetime


REGISTRY_FILE = Path("experiment_registry.json")
ANALYTICS_FILE = Path("sweep_analytics.json")
SENSITIVITY_FILE = Path("parameter_sensitivity.json")
INTERACTIONS_FILE = Path("parameter_interactions.json")

OUTPUT_FILE = Path("experiment_analysis_report.json")


def load_json(path):
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metric_summary(experiments, metric):
    values = []

    for experiment in experiments:
        metrics = experiment.get("metrics", {})
        summary = experiment.get("summary", {})

        value = metrics.get(metric)

        if value is None:
            value = summary.get(metric)

        if isinstance(value, dict):
            value = value.get("mean")

        try:
            values.append(float(value))
        except (TypeError, ValueError):
            pass

    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
        }

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def build_report():
    registry = load_json(REGISTRY_FILE)
    analytics = load_json(ANALYTICS_FILE)
    sensitivity = load_json(SENSITIVITY_FILE)
    interactions = load_json(INTERACTIONS_FILE)

    experiments = registry.get("experiments", [])

    metrics = [
        "evidence_recall",
        "quality",
        "latency",
        "context",
        "success_rate",
        "failure_rate",
    ]

    report = {
        "report_version": 1,
        "generated_at": datetime.now().isoformat(),

        "experiment_count": len(experiments),

        "dataset": {
            "registry_file": str(REGISTRY_FILE),
            "analytics_file": str(ANALYTICS_FILE),
            "sensitivity_file": str(SENSITIVITY_FILE),
            "interactions_file": str(INTERACTIONS_FILE),
        },

        "overall_metrics": {},

        "sweep_analytics": analytics,

        "parameter_sensitivity": sensitivity,

        "parameter_interactions": interactions,

        "interpretation_notes": [
            "Metrics describe observed experiment results.",
            "Parameter sensitivity measures association, not causation.",
            "Parameter interaction analysis describes differences across parameter combinations.",
            "Pareto analysis identifies non-dominated configurations without selecting an overall winner.",
        ],
    }

    for metric in metrics:
        report["overall_metrics"][metric] = metric_summary(
            experiments,
            metric,
        )

    return report


def main():
    report = build_report()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Experiment analysis report generated.")
    print(f"Experiments: {report['experiment_count']}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()