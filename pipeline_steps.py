from hybrid_retriever import hybrid_retrieve
from reranker import rerank
from confidence import calculate_confidence
from conflict_detector import detect_policy_conflict
from retrieval_quality import check_retrieval_quality
from decision_layer import make_retrieval_decision


def retrieve_and_rerank(query, product):

    candidates = hybrid_retrieve(
        query,
        product
    )

    if not candidates:
        return {
            "candidates": [],
            "results": [],
            "confidence": {
                "score": 0.0,
                "margin": 0.0,
                "level": "LOW"
            },
            "conflicts": [],
            "quality": {
                "valid": False,
                "reason": "NO_CANDIDATES"
            },
            "decision": {
                "decision": "MANUAL_REVIEW",
                "reason": "NO_CANDIDATES"
            }
        }

    results = rerank(
        query,
        candidates,
        top_k=3
    )

    confidence = calculate_confidence(
        results
    )

    conflicts = detect_policy_conflict(
        results
    )

    quality = check_retrieval_quality(
        results
    )

    decision = make_retrieval_decision(
        confidence,
        conflicts
    )

    return {
        "candidates": candidates,
        "results": results,
        "confidence": confidence,
        "conflicts": conflicts,
        "quality": quality,
        "decision": decision
    }