import json
from pathlib import Path
from datetime import datetime


AUDIT_FILE = Path("audit_metrics.json")
REVIEW_FILE = Path("review_operations_report.json")
OUTPUT_FILE = Path("dashboard_data.json")


def load_json(file_path):

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

        print(
            "Invalid JSON:",
            file_path
        )

        return {}


def build_dashboard():

    audit_metrics = load_json(
        AUDIT_FILE
    )

    review_metrics = load_json(
        REVIEW_FILE
    )

    dashboard = {

        "generated_at":
            datetime.now().isoformat(),

        "system": {

            "name":
                "Banking RAG System",

            "status":
                "ACTIVE"
        },

        "rag": {

            "total_audits":
                audit_metrics.get(
                    "total_audits",
                    0
                ),

            "completed":
                audit_metrics.get(
                    "completed",
                    0
                ),

            "manual_review":
                audit_metrics.get(
                    "manual_review",
                    0
                ),

            "citation":
                audit_metrics.get(
                    "citation",
                    {}
                ),

            "performance":
                audit_metrics.get(
                    "performance",
                    {}
                )
        },

        "human_review": {

            "total_reviews":
                review_metrics.get(
                    "total_reviews",
                    0
                ),

            "pending":
                review_metrics.get(
                    "pending",
                    0
                ),

            "approved":
                review_metrics.get(
                    "approved",
                    0
                ),

            "rejected":
                review_metrics.get(
                    "rejected",
                    0
                ),

            "resolved_reviews":
                review_metrics.get(
                    "resolved_reviews",
                    0
                ),

            "average_resolution_minutes":
                review_metrics.get(
                    "average_resolution_minutes",
                    0
                ),

            "review_reasons":
                review_metrics.get(
                    "review_reasons",
                    {}
                )
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
    print("UNIFIED DASHBOARD")
    print("==============================")

    print(
        "Total Audits:",
        dashboard["rag"]["total_audits"]
    )

    print(
        "Completed:",
        dashboard["rag"]["completed"]
    )

    print(
        "Manual Review:",
        dashboard["rag"]["manual_review"]
    )

    print(
        "Pending Human Reviews:",
        dashboard["human_review"]["pending"]
    )

    print(
        "Approved:",
        dashboard["human_review"]["approved"]
    )

    print(
        "Rejected:",
        dashboard["human_review"]["rejected"]
    )

    print(
        "Average Resolution:",
        dashboard["human_review"][
            "average_resolution_minutes"
        ],
        "minutes"
    )

    print(
        "\nDashboard saved:",
        OUTPUT_FILE
    )

    print("==============================")