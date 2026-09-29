import re


def evaluate_grounding(answer, context):

    answer_sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?]", answer)
        if sentence.strip()
    ]

    context_lower = context.lower()

    unsupported_sentences = []

    for sentence in answer_sentences:

        sentence_lower = sentence.lower()

        # Ignore very short sentences
        if len(sentence_lower.split()) < 3:
            continue

        # Check whether important words from the
        # answer sentence appear in the evidence.

        words = [
            word
            for word in re.findall(
                r"\b[a-zA-Z]+\b",
                sentence_lower
            )
            if len(word) > 3
        ]

        matched_words = [
            word
            for word in words
            if word in context_lower
        ]

        # Require reasonable overlap with evidence.
        if words and len(matched_words) / len(words) < 0.5:
            unsupported_sentences.append(sentence)

    if unsupported_sentences:

        return {
            "grounded": False,
            "reason": "UNSUPPORTED_CLAIM",
            "unsupported_sentences":
                unsupported_sentences
        }

    return {
        "grounded": True,
        "reason": None,
        "unsupported_sentences": []
    }