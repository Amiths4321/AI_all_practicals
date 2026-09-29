from hybrid_retriever import hybrid_retrieve
from reranker import rerank
from llm import generate_answer
from query_rewriter import rewrite_query
from confidence import calculate_confidence
from audit_logger import save_audit_result
from conflict_detector import detect_policy_conflict
from decision_layer import make_retrieval_decision
from retrieval_debug import save_retrieval_debug
from performance_tracker import start_timer, elapsed_time
from retrieval_quality import check_retrieval_quality
from rag_status import get_rag_status
from request_id import generate_request_id
from evidence_builder import build_evidence
from citation_validator import validate_citations
from citation_support_validator import validate_citation_support
from review_queue import add_to_review_queue


def run_rag(
    query,
    product,
    conversation_history=None
):

    # --------------------------------
    # 1. Query rewriting
    # --------------------------------
    request_id = generate_request_id()

    print("\n==============================")
    print("REQUEST")
    print("==============================")
    print("Request ID:", request_id)
    rewritten_query = rewrite_query(
        query,
        product,
        conversation_history
    )

    print("\n==============================")
    print("QUERY REWRITE")
    print("==============================")
    print("Original Query:")
    print(query)
    print("\nRewritten Query:")
    print(rewritten_query)

    # --------------------------------
    # 2. Hybrid retrieval
    # --------------------------------

    retrieval_start = start_timer()

    candidates = hybrid_retrieve(
        rewritten_query,
        product
    )

    retrieval_time = elapsed_time(retrieval_start)

    # --------------------------------
    # 3. No retrieval results
    # --------------------------------

    if not candidates:

        answer = "No sufficiently relevant information found."
        confidence = {
            "score": 0.0,
            "level": "LOW"
        }
        performance = {
            "retrieval_seconds": retrieval_time,
            "reranking_seconds": 0.0
        }
        add_to_review_queue(
            request_id=request_id,
            query=query,
            product=product,
            reason="NO_CANDIDATES",
            answer=answer,
            evidence=[]
        )

        save_audit_result(
            query=query,
            rewritten_query=rewritten_query,
            product=product,
            answer=answer,
            confidence=confidence,
            review_required=False,
            sources=[],
            conflicts=[],
            performance=performance,
            request_id=request_id
                )

        return {
            "answer": answer,
            "confidence": confidence,
            "review_required": True,
            "sources": [],
            "performance": performance
        }

        # --------------------------------
    # 4. Reranking & Quality Check
    # --------------------------------

    rerank_start = start_timer()

    results = rerank(
        rewritten_query,
        candidates,
        top_k=3
    )

    rerank_time = elapsed_time(rerank_start)

    quality = check_retrieval_quality(results)

    performance = {
        "retrieval_seconds": retrieval_time,
        "reranking_seconds": rerank_time
    }

    confidence = calculate_confidence(results)

    conflicts = detect_policy_conflict(results)

    # --------------------------------
    # 5. Build structured evidence
    # --------------------------------

    evidence = build_evidence(results)

    # --------------------------------
    # 6. Generate LLM answer
    # --------------------------------

    llm_start = start_timer()

    answer = generate_answer(
        query,
        evidence
    )

    llm_time = elapsed_time(llm_start)

    performance["llm_seconds"] = llm_time

    performance["total_seconds"] = round(
        performance["retrieval_seconds"]
        + performance["reranking_seconds"]
        + performance["llm_seconds"],
        4
    )

    # --------------------------------
    # 7. Validate citations
    # --------------------------------

    citation_result = validate_citations(
        answer,
        evidence
    )

    support_result = validate_citation_support(
        answer,
        evidence
    )

    citation_valid = (
        citation_result["valid"]
        and support_result["valid"]
    )

    # --------------------------------
    # 8. Invalid citation → review
    # --------------------------------

    if not citation_valid:

        review_answer = (
            "The generated answer could not be "
            "validated against the retrieved evidence. "
            "Manual review is required."
        )
        add_to_review_queue(
            request_id=request_id,
            query=query,
            product=product,
            reason="INVALID_CITATION",
            answer=answer,
            evidence=evidence
        )
        
        save_audit_result(
            query=query,
            rewritten_query=rewritten_query,
            product=product,
            answer=answer,
            confidence=confidence,
            review_required=True,
            sources=evidence,
            conflicts=conflicts,
            review_reason="INVALID_CITATION",
            performance=performance,
            request_id=request_id,
            citation_validation=citation_result,
            citation_support=support_result
        )

        add_to_review_queue(
            request_id=request_id,
            query=query,
            product=product,
            reason="NO_CANDIDATES",
            answer=answer,
            evidence=[]
        )
        
        return {
            "answer": review_answer,
            "status": "MANUAL_REVIEW",
            "review_required": True,
            "review_reason": "INVALID_CITATION",
            "confidence": confidence,
            "evidence": evidence,
            "citation_validation": citation_result,
            "citation_support": support_result,
            "conflicts": conflicts,
            "performance": performance
        }

    # --------------------------------
    # 9. Successful audit
    # --------------------------------

    save_audit_result(
        query=query,
        rewritten_query=rewritten_query,
        product=product,
        answer=answer,
        confidence=confidence,
        review_required=False,
        sources=evidence,
        conflicts=conflicts,
        review_reason=None,
        performance=performance,
        request_id=request_id,
        citation_validation=citation_result,
        citation_support=support_result
    )

    # --------------------------------
    # 10. Final response
    # --------------------------------

    return {
        "answer": answer,
        "status": "COMPLETED",
        "review_required": False,
        "review_reason": None,
        "confidence": confidence,
        "evidence": evidence,
        "citation_validation": citation_result,
        "citation_support": support_result,
        "conflicts": conflicts,
        "performance": performance
    }

# --------------------------------
# Command-line runner
# --------------------------------

if __name__ == "__main__":

    question = input("\nEnter your question: ").strip()

    product = "home_loan"

    result = run_rag(
        query=question,
        product=product
    )

    print("\n==============================")
    print("FINAL RESULT")
    print("==============================")

    print("\nAnswer:")
    print(result["answer"])

    print("\nStatus:")
    print(result.get("status"))

    print("\nConfidence:")
    print(result.get("confidence"))

    print("\nPerformance:")
    print(result.get("performance"))