from score_analysis import calculate_score_margin


def calculate_confidence(results):

    if not results:
        return {
            "score": 0.0,
            "margin": 0.0,
            "level": "LOW"
        }

    scores = [
        float(result.get("reranker_score", 0.0))
        for result in results
    ]

    best_score = max(scores)

    margin = calculate_score_margin(results)

    if margin >= 1.0:
        level = "HIGH"

    elif margin >= 0.2:
        level = "MEDIUM"

    else:
        level = "LOW"

    return {
        "score": float(best_score),
        "margin": float(margin),
        "level": level
    }