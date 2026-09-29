import re


def validate_citations(answer, evidence):

    valid_ids = {
        item["evidence_id"]
        for item in evidence
    }

    cited_ids = re.findall(
        r"\[(E\d+)\]",
        answer
    )

    invalid_ids = [
        evidence_id
        for evidence_id in cited_ids
        if evidence_id not in valid_ids
    ]

    missing_citations = []

    if not cited_ids:
        missing_citations.append(
            "NO_EVIDENCE_ID"
        )

    return {
        "valid": len(invalid_ids) == 0
                 and len(cited_ids) > 0,

        "cited_ids": cited_ids,

        "valid_ids": list(valid_ids),

        "invalid_ids": invalid_ids,

        "missing_citations": missing_citations
    }