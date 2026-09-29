import json
from datetime import datetime
from pathlib import Path


ADAPTIVE_REPORT = Path("adaptive_rag_report.json")
FAILURE_REPORT = Path("failure_monitoring.json")

OUTPUT_FILE = Path("production_health.json")


def load_json(path):
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_rate(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def build_health_report():
    adaptive = load_json(ADAPTIVE_REPORT)
    failure = load_json(FAILURE_REPORT)

    summary = failure.get("summary", {})

    questions = summary.get("questions", 0)
    successes = summary.get("successes", 0)

    retrieval_failures = summary.get(
        "retrieval_failures",
        0,
    )

    abstentions = summary.get(
        "abstentions",
        0,
    )

    unsupported_claims = summary.get(
        "unsupported_claims",
        0,
    )

    ungrounded_answers = summary.get(
        "ungrounded_answers",
        0,
    )

    irrelevant_answers = summary.get(
        "irrelevant_answers",
        0,
    )

    incomplete_answers = summary.get(
        "incomplete_answers",
        0,
    )

    health = {
        "monitoring_version": 1,

        "timestamp": datetime.now().isoformat(),

        "volume": {
            "questions": questions,
        },

        "quality": {
            "success_rate": safe_rate(
                successes,
                questions,
            ),

            "retrieval_failure_rate": safe_rate(
                retrieval_failures,
                questions,
            ),

            "abstention_rate": safe_rate(
                abstentions,
                questions,
            ),

            "unsupported_claim_rate": safe_rate(
                unsupported_claims,
                questions,
            ),

            "ungrounded_rate": safe_rate(
                ungrounded_answers,
                questions,
            ),

            "irrelevant_rate": safe_rate(
                irrelevant_answers,
                questions,
            ),

            "incomplete_rate": safe_rate(
                incomplete_answers,
                questions,
            ),
        },

        "failure_categories": summary.get(
            "category_counts",
            {},
        ),

        "status": "unknown",
    }

    # Basic health classification.
    #
    # These thresholds are intentionally explicit
    # so they can later be moved into config.

    quality = health["quality"]

    if quality["questions"] if False else False:
        pass

    if quality["retrieval_failure_rate"] > 0.40:
        health["status"] = "degraded"

    elif quality["ungrounded_rate"] > 0.20:
        health["status"] = "degraded"

    elif quality["success_rate"] < 0.50:
        health["status"] = "degraded"

    else:
        health["status"] = "healthy"

    health["source_files"] = {
        "adaptive_report": str(ADAPTIVE_REPORT),
        "failure_report": str(FAILURE_REPORT),
    }

    return health


def main():
    report = build_health_report()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Production health report generated.")
    print(f"Status: {report['status']}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()