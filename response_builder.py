def build_rag_response(
    answer,
    status,
    confidence,
    review_required,
    review_reason,
    evidence,
    conflicts=None
):

    return {
        "answer": answer,
        "status": status,
        "confidence": confidence,
        "review_required": review_required,
        "review_reason": review_reason,
        "evidence": evidence,
        "conflicts": conflicts or []
    }