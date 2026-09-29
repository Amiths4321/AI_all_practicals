def validate_rewrite(original_query, rewritten_query):

    if not rewritten_query:
        return {
            "valid": False,
            "reason": "EMPTY_REWRITE"
        }

    original_words = set(
        original_query.lower().split()
    )

    rewritten_words = set(
        rewritten_query.lower().split()
    )

    # Important intent words that should normally
    # survive the rewrite.

    intent_terms = [
        "income",
        "age",
        "documents",
        "document",
        "tenure",
        "interest",
        "rate",
        "eligibility",
        "eligible"
    ]

    original_intent = [
        word
        for word in intent_terms
        if word in original_words
    ]

    rewritten_intent = [
        word
        for word in intent_terms
        if word in rewritten_words
    ]

    if original_intent:

        if not any(
            term in rewritten_intent
            for term in original_intent
        ):
            return {
                "valid": False,
                "reason": "INTENT_CHANGED"
            }

    return {
        "valid": True,
        "reason": None
    }