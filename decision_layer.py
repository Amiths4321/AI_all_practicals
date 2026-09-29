def make_retrieval_decision(confidence, conflicts):

    if conflicts:
        return {
            "decision": "MANUAL_REVIEW",
            "reason": "POLICY_CONFLICT"
        }

    if confidence["level"] == "LOW":
        return {
            "decision": "MANUAL_REVIEW",
            "reason": "LOW_RETRIEVAL_CONFIDENCE"
        }

    return {
        "decision": "PROCEED",
        "reason": None
    }