import json
from datetime import datetime
from pathlib import Path


HEALTH_FILE = Path("production_health.json")
BASELINE_FILE = Path("production_health_baseline.json")
OUTPUT_FILE = Path("drift_report.json")


# Maximum acceptable absolute change for rates.
RATE_DRIFT_THRESHOLD = 0.10


def load_json(path):
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def extract_quality(report):
    return report.get("quality", {})


def calculate_drift(baseline, current):
    baseline_quality = extract_quality(baseline)
    current_quality = extract_quality(current)

    drift = {}

    metrics = [
        "success_rate",
        "retrieval_failure_rate",
        "abstention_rate",
        "unsupported_claim_rate",
        "ungrounded_rate",
        "irrelevant_rate",
        "incomplete_rate",
    ]

    for metric in metrics:
        baseline_value = baseline_quality.get(metric)
        current_value = current_quality.get(metric)

        if baseline_value is None or current_value is None:
            continue

        change = current_value - baseline_value

        drift[metric] = {
            "baseline": baseline_value,
            "current": current_value,
            "change": change,
            "absolute_change": abs(change),
            "drift_detected": (
                abs(change) > RATE_DRIFT_THRESHOLD
            ),
        }

    return drift


def build_report(baseline, current):
    drift = calculate_drift(
        baseline,
        current,
    )

    detected = [
        metric
        for metric, result in drift.items()
        if result["drift_detected"]
    ]

    return {
        "drift_detection_version": 1,
        "timestamp": datetime.now().isoformat(),

        "baseline_timestamp": baseline.get(
            "timestamp"
        ),

        "current_timestamp": current.get(
            "timestamp"
        ),

        "threshold": RATE_DRIFT_THRESHOLD,

        "metrics": drift,

        "drift_detected": bool(detected),

        "drifted_metrics": detected,

        "status": (
            "drift_detected"
            if detected
            else "stable"
        ),
    }


def main():
    current = load_json(HEALTH_FILE)

    if not current:
        print(
            "ERROR: production_health.json "
            "does not exist or is empty."
        )
        return

    if not BASELINE_FILE.exists():
        save_json(
            BASELINE_FILE,
            current,
        )

        print(
            "Baseline created:"
            f" {BASELINE_FILE}"
        )
        print(
            "Run the detector again after "
            "new monitoring data is available."
        )
        return

    baseline = load_json(BASELINE_FILE)

    report = build_report(
        baseline,
        current,
    )

    save_json(
        OUTPUT_FILE,
        report,
    )

    print("Drift detection complete.")
    print(f"Status: {report['status']}")
    print(f"Output: {OUTPUT_FILE}")

    if report["drifted_metrics"]:
        print("Drifted metrics:")

        for metric in report["drifted_metrics"]:
            print(f"  - {metric}")


if __name__ == "__main__":
    main()