import json
from pathlib import Path
from datetime import datetime


audit_files = sorted(
    Path(".").glob("audit_*.json")
)

if not audit_files:
    print("No audit files found.")
    exit()


reports = []

for file_path in audit_files:

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            reports.append(data)

    except Exception as e:

        print(
            "Could not read:",
            file_path.name,
            "|",
            e
        )


total = len(reports)

completed = 0
manual_review = 0

citation_valid = 0
citation_invalid = 0

conflict_count = 0


for report in reports:

    decision = report.get(
        "decision",
        {}
    )

    status = decision.get(
        "status"
    )

    if status == "COMPLETED":
        completed += 1

    elif status == "MANUAL_REVIEW":
        manual_review += 1

    citation = report.get(
        "citation_validation",
        {}
    )

    if citation.get("valid") is True:
        citation_valid += 1

    elif citation:
        citation_invalid += 1

    conflicts = report.get(
        "conflicts",
        []
    )

    if conflicts:
        conflict_count += 1


if total:

    completion_rate = (
        completed / total * 100
    )

    review_rate = (
        manual_review / total * 100
    )

else:

    completion_rate = 0
    review_rate = 0


audit_summary = {

    "report_generated":
        datetime.now().isoformat(),

    "total_audits":
        total,

    "summary": {

        "completed":
            completed,

        "manual_review":
            manual_review,

        "completion_rate":
            round(
                completion_rate,
                2
            ),

        "review_rate":
            round(
                review_rate,
                2
            ),

        "citation_valid":
            citation_valid,

        "citation_invalid":
            citation_invalid,

        "policy_conflicts":
            conflict_count
    },

    "audits":
        reports
}


filename = (
    "complete_audit_report_"
    + datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    + ".json"
)


with open(
    filename,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        audit_summary,
        file,
        indent=4,
        ensure_ascii=False
    )


print("\n======================================")
print("COMPLETE RAG AUDIT REPORT")
print("======================================")

print("Total audits:", total)
print("Completed:", completed)
print("Manual review:", manual_review)

print(
    "Completion rate:",
    round(completion_rate, 2),
    "%"
)

print(
    "Review rate:",
    round(review_rate, 2),
    "%"
)

print(
    "Valid citations:",
    citation_valid
)

print(
    "Invalid citations:",
    citation_invalid
)

print(
    "Policy conflicts:",
    conflict_count
)

print("\nReport saved:", filename)