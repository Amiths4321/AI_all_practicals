import json
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


REQUIRED_FILES = [
    "config.json",
    "evaluation_data.json",
    "empirical_context_policy.json",
    "adaptive_rag_report.json",
    "final_adaptive_evaluation.json",
    "failure_monitoring.json",
]


def check_files():
    print("\n[1] Checking required files...")

    missing = []

    for filename in REQUIRED_FILES:
        path = BASE_DIR / filename

        if path.exists():
            print(f"  OK   {filename}")
        else:
            print(f"  MISS {filename}")
            missing.append(filename)

    return missing


def check_json_files():
    print("\n[2] Checking JSON files...")

    failures = []

    for filename in REQUIRED_FILES:
        path = BASE_DIR / filename

        if not path.exists():
            continue

        try:
            with open(path, "r", encoding="utf-8") as file:
                json.load(file)

            print(f"  OK   {filename}")

        except Exception as error:
            print(f"  FAIL {filename}: {error}")
            failures.append(filename)

    return failures


def check_regression_tests():
    print("\n[3] Running regression tests...")

    result = subprocess.run(
        [sys.executable, "regression_tests.py"],
        cwd=BASE_DIR
    )

    if result.returncode == 0:
        print("  PASS regression tests")
        return True

    print("  FAIL regression tests")
    return False


def check_adaptive_report():
    print("\n[4] Checking adaptive report...")

    path = BASE_DIR / "adaptive_rag_report.json"

    if not path.exists():
        print("  FAIL adaptive report missing")
        return False

    with open(path, "r", encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report, dict):
        print("  FAIL report is not an object")
        return False

    required = [
        "config",
        "summary",
        "results"
    ]

    for field in required:
        if field not in report:
            print(f"  FAIL missing field: {field}")
            return False

    results = report["results"]

    if not isinstance(results, list):
        print("  FAIL results is not a list")
        return False

    print(f"  OK   {len(results)} question results")

    return True


def check_failure_monitoring():
    print("\n[5] Checking failure monitoring...")

    path = BASE_DIR / "failure_monitoring.json"

    if not path.exists():
        print("  FAIL failure monitoring report missing")
        return False

    with open(path, "r", encoding="utf-8") as file:
        report = json.load(file)

    required = [
        "summary",
        "category_counts",
        "question_results"
    ]

    for field in required:
        if field not in report:
            print(f"  FAIL missing field: {field}")
            return False

    print("  OK   failure monitoring schema")

    return True

def run_api_tests():

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "test_api.py",
            "-v"
        ],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:

        print(result.stderr)

        raise RuntimeError(
            "API contract tests failed"
        )
def main():
    print("=" * 70)
    print("RAG PIPELINE VALIDATION")
    print("=" * 70)

    missing_files = check_files()

    if missing_files:
        print("\nVALIDATION FAILED")
        print("Missing files:")
        for filename in missing_files:
            print(f"  - {filename}")
        return 1

    json_failures = check_json_files()

    if json_failures:
        print("\nVALIDATION FAILED")
        return 1

    if not check_regression_tests():
        print("\nVALIDATION FAILED")
        return 1

    if not check_adaptive_report():
        print("\nVALIDATION FAILED")
        return 1

    if not check_failure_monitoring():
        print("\nVALIDATION FAILED")
        return 1

    print("\n" + "=" * 70)
    print("VALIDATION PASSED")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())