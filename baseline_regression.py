import json
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"


# Allowed regression thresholds
MAX_EVIDENCE_RECALL_DROP = 0.05
MAX_QUALITY_DROP = 0.05
MAX_LATENCY_INCREASE = 0.25
MAX_CONTEXT_INCREASE = 0.25


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_previous_experiment():
    experiments = [
        path
        for path in EXPERIMENTS_DIR.iterdir()
        if path.is_dir()
    ]

    experiments.sort(
        key=lambda path: path.name,
        reverse=True
    )

    for experiment in experiments:
        report = (
            experiment /
            "final_adaptive_evaluation.json"
        )

        manifest = (
            experiment /
            "experiment_manifest.json"
        )

        if report.exists() and manifest.exists():
            return experiment

    return None


def extract_metrics(report):
    summary = report.get("summary", {})

    return {
        "evidence_recall": float(
            summary.get(
                "adaptive_average_evidence_recall",
                summary.get(
                    "adaptive_evidence_recall",
                    0.0
                )
            )
        ),
        "quality": float(
            summary.get(
                "adaptive_average_quality_index",
                summary.get(
                    "adaptive_quality",
                    0.0
                )
            )
        ),
        "latency": float(
            summary.get(
                "adaptive_average_latency_seconds",
                0.0
            )
        ),
        "context": float(
            summary.get(
                "adaptive_average_context_characters",
                0.0
            )
        )
    }


def compare(current, baseline):
    evidence_drop = (
        baseline["evidence_recall"]
        - current["evidence_recall"]
    )

    quality_drop = (
        baseline["quality"]
        - current["quality"]
    )

    if baseline["latency"] > 0:
        latency_change = (
            current["latency"]
            - baseline["latency"]
        ) / baseline["latency"]
    else:
        latency_change = 0.0

    if baseline["context"] > 0:
        context_change = (
            current["context"]
            - baseline["context"]
        ) / baseline["context"]
    else:
        context_change = 0.0

    checks = {
        "evidence_recall": {
            "baseline": baseline["evidence_recall"],
            "current": current["evidence_recall"],
            "change": -evidence_drop,
            "passed": (
                evidence_drop
                <= MAX_EVIDENCE_RECALL_DROP
            )
        },
        "quality": {
            "baseline": baseline["quality"],
            "current": current["quality"],
            "change": -quality_drop,
            "passed": (
                quality_drop
                <= MAX_QUALITY_DROP
            )
        },
        "latency": {
            "baseline": baseline["latency"],
            "current": current["latency"],
            "change": latency_change,
            "passed": (
                latency_change
                <= MAX_LATENCY_INCREASE
            )
        },
        "context": {
            "baseline": baseline["context"],
            "current": current["context"],
            "change": context_change,
            "passed": (
                context_change
                <= MAX_CONTEXT_INCREASE
            )
        }
    }

    return {
        "passed": all(
            check["passed"]
            for check in checks.values()
        ),
        "checks": checks
    }


def main():
    print("=" * 70)
    print("BASELINE REGRESSION CHECK")
    print("=" * 70)

    previous = find_previous_experiment()

    if previous is None:
        print()
        print(
            "No previous valid experiment found."
        )
        print(
            "Baseline comparison skipped."
        )
        return 0

    current_report = (
        BASE_DIR /
        "final_adaptive_evaluation.json"
    )

    if not current_report.exists():
        print(
            "Current evaluation report not found."
        )
        return 1

    baseline_report = (
        previous /
        "final_adaptive_evaluation.json"
    )

    current = extract_metrics(
        load_json(current_report)
    )

    baseline = extract_metrics(
        load_json(baseline_report)
    )

    result = compare(
        current,
        baseline
    )

    print()
    print(
        "Baseline:",
        previous.name
    )

    for name, check in result["checks"].items():
        print()
        print(name)

        print(
            "  baseline:",
            round(check["baseline"], 4)
        )

        print(
            "  current:",
            round(check["current"], 4)
        )

        print(
            "  change:",
            round(check["change"], 4)
        )

        print(
            "  status:",
            "PASS" if check["passed"]
            else "FAIL"
        )

    output = {
        "baseline_experiment":
            previous.name,
        "current_experiment":
            "current",
        "passed":
            result["passed"],
        "checks":
            result["checks"]
    }

    output_file = (
        BASE_DIR /
        "baseline_regression.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file,
            indent=2
        )

    print()

    if result["passed"]:
        print("=" * 70)
        print("BASELINE REGRESSION CHECK PASSED")
        print("=" * 70)
        return 0

    print("=" * 70)
    print("BASELINE REGRESSION DETECTED")
    print("=" * 70)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())