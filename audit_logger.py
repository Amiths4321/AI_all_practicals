import json
from datetime import datetime


def save_audit_result(
    query,
    rewritten_query,
    product,
    answer,
    confidence,
    review_required,
    sources,
    conflicts=None,
    review_reason=None,
    performance=None,
    status=None,
    request_id=None,
    citation_validation=None,
    citation_support=None,
):

    audit_result = {
        "timestamp": datetime.now().isoformat(),

        "query": {
            "original": query,
            "rewritten": rewritten_query,
            "product": product
        },

        "decision": {
            "status": status,
            "confidence_score": confidence["score"],
            "confidence_level": confidence["level"],
            "review_required": review_required,
            "review_reason": review_reason
        },
        "citation_validation": citation_validation or {},
        
        "citation_support": citation_support or {},
        "performance": performance or {},
        
        "answer": answer,

        "conflicts": conflicts or [],

        "sources": sources,

        "status": status,

        "request_id": None
    }

    filename = (
        "audit_"
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
            audit_result,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nAudit result saved: {filename}"
    )