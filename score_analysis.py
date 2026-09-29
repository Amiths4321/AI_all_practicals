def calculate_score_margin(results):

    if not results:
        return 0.0

    scores = [
        float(result.get("reranker_score", 0.0))
        for result in results
    ]

    scores.sort(reverse=True)

    if len(scores) < 2:
        return 0.0

    best_score = scores[0]
    second_score = scores[1]

    return best_score - second_score