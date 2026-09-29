def get_rag_status(
    quality,
    decision
):

    if not quality["valid"]:
        return {
            "status": "MANUAL_REVIEW",
            "reason": quality["reason"]
        }

    add_to_review_queue(
        request_id=request_id,
        query=query,
        product=product,
        reason=quality["reason"],
        answer=answer,
        evidence=evidence
    )

    if decision["decision"] == "MANUAL_REVIEW":
        return {
            "status": "MANUAL_REVIEW",
            "reason": decision["reason"]
        }

    add_to_review_queue(
        request_id=request_id,
        query=query,
        product=product,
        reason=decision["reason"],
        answer=answer,
        evidence=evidence
    )

    return {
        "status": "COMPLETED",
        "reason": None
    }