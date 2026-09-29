import json
from pathlib import Path
from datetime import datetime


REVIEW_FILE = Path("manual_review_queue.json")
OUTPUT_FILE = Path("dashboard_data.json")


def load_json_file(file_path):

    if not file_path.exists():
        return {}

    try:
        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            content = file.read().strip()

            if not content:
                return {}

            return json.loads(content)

    except json.JSONDecodeError:
        return {}


def build_review_metrics():

    queue = load_json_file(REVIEW_FILE)

    if not isinstance(queue, list):
        queue = []

    total = len(queue)

    pending = sum(
        1
        for item in queue
        if item.get("status") == "PENDING"
    )

    approved = sum(
        1
        for item in queue
        if item.get("status") == "APPROVED"
    )

    rejected = sum(
        1
        for item in queue
        if item.get("status") == "REJECTED"
    )

    return {
        "total_reviews": total,
        "pending": pending,
        "approved": approved,
        "rejected": rejected
    }


def build_dashboard():

    review_metrics = build_review_metrics()

    dashboard = {

        "generated_at":
            datetime.now().isoformat(),

        "system": {
            "name": "Banking RAG System",
            "status": "ACTIVE"
        },

        "rag": {
            "retrieval": {
                "enabled": True,
                "method": "Hybrid + RRF"
            },

            "reranking": {
                "enabled": True
            },

            "citation_validation": {
                "enabled": True
            },

            "human_review": {
                "enabled": True
            }
        },

        "review_metrics": review_metrics,

        "performance": {
            "retrieval_seconds": None,
            "reranking_seconds": None,
            "llm_seconds": None,
            "total_seconds": None
        }
    }

    return dashboard


if __name__ == "__main__":

    dashboard = build_dashboard()

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dashboard,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n==============================")
    print("DASHBOARD DATA")
    print("==============================")

    print(
        "Total Reviews:",
        dashboard["review_metrics"]["total_reviews"]
    )

    print(
        "Pending:",
        dashboard["review_metrics"]["pending"]
    )

    print(
        "Approved:",
        dashboard["review_metrics"]["approved"]
    )

    print(
        "Rejected:",
        dashboard["review_metrics"]["rejected"]
    )

    print(
        "\nSaved:",
        OUTPUT_FILE
    )

    print("==============================")