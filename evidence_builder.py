def build_evidence(results):

    evidence = []

    for rank, result in enumerate(results, start=1):

        document = result.get("document", {})

        evidence.append({
            "evidence_id": f"E{rank:03d}",
            "rank": rank,
            "document": document.get("document"),
            "section": document.get("section"),
            "chunk_id": document.get("chunk_id"),
            "product": document.get("product"),
            "policy_version": document.get("policy_version"),
            "effective_date": document.get("effective_date"),
            "text": document.get("text"),
            "reranker_score": result.get(
                "reranker_score"
            )
        })

    return evidence