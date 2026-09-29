import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


OUTPUT_FILE = Path(
    "final_validation_report.json"
)


COMMANDS = [
    (
        "regression_tests",
        ["python", "regression_tests.py"],
    ),
    (
        "validate_pipeline",
        ["python", "validate_pipeline.py"],
    ),
    (
        "quality_gate",
        ["python", "quality_gate.py"],
    ),
    (
        "production_evaluation",
        ["python", "production_evaluation.py"],
    ),
    (
        "production_monitor",
        ["python", "production_monitor.py"],
    ),
    (
        "drift_detector",
        ["python", "drift_detector.py"],
    ),
    (
        "end_to_end_optimization",
        ["python", "end_to_end_optimization.py"],
    ),
]


def run_command(name, command):

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    return {
        "name": name,
        "command": command,
        "return_code": result.returncode,
        "passed": result.returncode == 0,
    }


def main():

    started = datetime.now().isoformat()

    results = []

    for name, command in COMMANDS:

        result = run_command(
            name,
            command,
        )

        results.append(result)

        # Continue running all checks so that
        # the final report shows every failure.
        #
        # Do not stop at the first failure.

    passed = all(
        item["passed"]
        for item in results
    )

    report = {
        "validation_version": 1,

        "started_at": started,

        "finished_at":
            datetime.now().isoformat(),

        "overall_status": (
            "PASSED"
            if passed
            else "FAILED"
        ),

        "checks": results,

        "failed_checks": [
            item["name"]
            for item in results
            if not item["passed"]
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

    print()
    print("=" * 60)
    print(
        f"FINAL VALIDATION: "
        f"{report['overall_status']}"
    )
    print("=" * 60)

    if report["failed_checks"]:
        print("Failed checks:")

        for name in report[
            "failed_checks"
        ]:
            print(f"  - {name}")

    print(
        f"Report: {OUTPUT_FILE}"
    )

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()