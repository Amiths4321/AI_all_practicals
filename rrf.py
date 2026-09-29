def reciprocal_rank_fusion(
    vector_results,
    keyword_results,
    k=60
):

    scores = {}
    documents = {}

    # -----------------------------
    # VECTOR RESULTS
    # -----------------------------

    for rank, result in enumerate(
        vector_results,
        start=1
    ):

        doc_id = result["id"]

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + 1 / (k + rank)
        )

        documents[doc_id] = result["document"]

    # -----------------------------
    # KEYWORD RESULTS
    # -----------------------------

    for rank, result in enumerate(
        keyword_results,
        start=1
    ):

        doc_id = result["id"]

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + 1 / (k + rank)
        )

        # Keep the document information
        # already stored by vector search
        if doc_id not in documents:
            documents[doc_id] = result["document"]

    # -----------------------------
    # SORT
    # -----------------------------

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    results = []

    for doc_id in ranked_ids:

        results.append({
            "id": doc_id,
            "rrf_score": scores[doc_id],
            "document": documents[doc_id]
        })

    return results