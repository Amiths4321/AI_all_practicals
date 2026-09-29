from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(MODEL_NAME)


def rerank(query, results, top_k=3):

    if not results:
        return []

    pairs = []

    for result in results:

        document = result.get("document", {})

        text = document.get("text", "")

        pairs.append([
            query,
            text
        ])


    # Debug: show exactly what goes into reranker
    print("\n==============================")
    print("RERANKER INPUT")
    print("==============================")

    for i, pair in enumerate(pairs, start=1):

        print(f"\nCandidate {i}")
        print("Query:", pair[0])
        print("Document:")
        print(pair[1])


    scores = reranker.predict(pairs)


    # Attach scores
    reranked_results = []

    for result, score in zip(
        results,
        scores
    ):

        result["reranker_score"] = float(score)

        reranked_results.append(result)


    # Sort highest score first
    reranked_results.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )


    return reranked_results[:top_k]