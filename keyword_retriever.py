from rank_bm25 import BM25Okapi


def keyword_search(query, documents, top_k=5):

    tokenized_documents = [
        document["text"].lower().split()
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)

    results = []

    for index, score in enumerate(scores):

        results.append({
            "id": index,
            "score": float(score),
            "document": documents[index]
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]