import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


# Quality thresholds
MIN_EVIDENCE_RECALL = 0.80
MAX_RETRIEVAL_FAILURE_RATE = 0.40
MAX_UNGROUNDED_RATE = 0.20
MIN_SUCCESS_RATE = 0.50


def load_json(filename):
    path = BASE_DIR / filename

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def check_failure_monitoring():
    report = load_json(
        "failure_monitoring.json"
    )

    summary = report["summary"]

    questions = summary.get("questions", 0)

    if questions == 0:
        return {
            "passed": False,
            "reason": "no_questions"
        }

    retrieval_failures = summary.get(
        "retrieval_failures",
        0
    )

    ungrounded_answers = summary.get(
        "ungrounded_answers",
        0
    )

    successes = summary.get(
        "successes",
        0
    )

    retrieval_failure_rate = (
        retrieval_failures / questions
    )

    ungrounded_rate = (
        ungrounded_answers / questions
    )

    success_rate = (
        successes / questions
    )

    checks = {
        "retrieval_failure_rate": {
            "value": retrieval_failure_rate,
            "limit": MAX_RETRIEVAL_FAILURE_RATE,
            "passed": (
                retrieval_failure_rate
                <= MAX_RETRIEVAL_FAILURE_RATE
            )
        },
        "ungrounded_rate": {
            "value": ungrounded_rate,
            "limit": MAX_UNGROUNDED_RATE,
            "passed": (
                ungrounded_rate
                <= MAX_UNGROUNDED_RATE
            )
        },
        "success_rate": {
            "value": success_rate,
            "limit": MIN_SUCCESS_RATE,
            "passed": (
                success_rate
                >= MIN_SUCCESS_RATE
            )
        }
    }

    return {
        "passed": all(
            item["passed"]
            for item in checks.values()
        ),
        "checks": checks
    }


def check_evidence_recall():
    report = load_json(
        "adaptive_rag_report.json"
    )

    results = report.get(
        "results",
        []
    )

    recalls = []

    for result in results:
        recall = result.get(
            "retrieval_evidence_recall"
        )

        if recall is not None:
            recalls.append(
                float(recall)
            )

    if not recalls:
        return {
            "passed": False,
            "reason": "no_evidence_recall_data"
        }

    average_recall = (
        sum(recalls) / len(recalls)
    )

    return {
        "passed": (
            average_recall
            >= MIN_EVIDENCE_RECALL
        ),
        "average_evidence_recall":
            average_recall,
        "minimum_required":
            MIN_EVIDENCE_RECALL
    }


def run_quality_gate():
    failure_check = (
        check_failure_monitoring()
    )

    evidence_check = (
        check_evidence_recall()
    )

    passed = (
        failure_check["passed"]
        and evidence_check["passed"]
    )

    return {
        "passed": passed,
        "failure_monitoring":
            failure_check,
        "evidence_recall":
            evidence_check
    }


def main():
    print("=" * 70)
    print("RAG QUALITY GATE")
    print("=" * 70)

    report = run_quality_gate()

    print()
    print(
        "Evidence recall:",
        round(
            report["evidence_recall"].get(
                "average_evidence_recall",
                0.0
            ),
            3
        )
    )

    print(
        "Evidence threshold:",
        MIN_EVIDENCE_RECALL
    )

    failure_checks = report[
        "failure_monitoring"
    ].get("checks", {})

    for name, check in failure_checks.items():
        print(
            f"{name}:",
            round(check["value"], 3),
            "PASS" if check["passed"]
            else "FAIL"
        )

    print()

    if report["passed"]:
        print("=" * 70)
        print("QUALITY GATE PASSED")
        print("=" * 70)
        return 0

    print("=" * 70)
    print("QUALITY GATE FAILED")
    print("=" * 70)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())