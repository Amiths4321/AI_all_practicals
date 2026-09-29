def evaluate_answer(answer, expected_phrases):

    answer_lower = answer.lower()

    matched = []

    for phrase in expected_phrases:
        if phrase.lower() in answer_lower:
            matched.append(phrase)

    if len(matched) == len(expected_phrases):
        return {
            "correct": True,
            "matched": matched,
            "missing": []
        }

    missing = [
        phrase
        for phrase in expected_phrases
        if phrase not in matched
    ]

    return {
        "correct": False,
        "matched": matched,
        "missing": missing
    }