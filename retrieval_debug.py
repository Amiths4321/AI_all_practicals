import json
from datetime import datetime


def save_retrieval_debug(
    request_id,
    query,
    rewritten_query,
    candidates,
    results,
    confidence,
    conflicts,
    decision
):

    debug_data = {
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,

        "query": {
            "original": query,
            "rewritten": rewritten_query
        },

        "retrieval": {
            "candidate_count": len(candidates),
            "reranked_count": len(results)
        },

        "candidates": [],

        "reranked_results": [],

        "confidence": confidence,

        "conflicts": conflicts,

        "decision": decision
    }

    # Candidates
    for rank, result in enumerate(candidates, start=1):

        document = result.get("document", {})

        debug_data["candidates"].append({
            "rank": rank,
            "document": document.get("document"),
            "product": document.get("product"),
            "section": document.get("section"),
            "chunk_id": document.get("chunk_id"),
            "rrf_score": result.get("rrf_score"),
            "text": document.get("text")
        })

    # Reranked results
    for rank, result in enumerate(results, start=1):

        document = result.get("document", {})

        debug_data["reranked_results"].append({
            "rank": rank,
            "document": document.get("document"),
            "product": document.get("product"),
            "section": document.get("section"),
            "chunk_id": document.get("chunk_id"),
            "reranker_score": result.get("reranker_score"),
            "text": document.get("text")
        })

    filename = (
        "retrieval_debug_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(
            debug_data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nRetrieval debug saved: {filename}")
    print("\n==============================")
    print("RAG RETRIEVAL DEBUG")
    print("==============================")
    print("Request ID:", request_id)