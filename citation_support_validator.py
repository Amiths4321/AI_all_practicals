import re


def validate_citation_support(answer, evidence):

    evidence_map = {
        item["evidence_id"]: item["text"]
        for item in evidence
    }

    cited_ids = re.findall(
        r"\[(E\d+)\]",
        answer
    )

    results = []

    for evidence_id in cited_ids:

        if evidence_id not in evidence_map:
            results.append({
                "evidence_id": evidence_id,
                "supported": False,
                "reason": "EVIDENCE_ID_NOT_FOUND"
            })
            continue

        evidence_text = evidence_map[
            evidence_id
        ].lower()

        # Remove the citation from the answer
        clean_answer = re.sub(
            r"\[E\d+\]",
            "",
            answer
        ).lower()

        # Extract meaningful words
        answer_words = set(
            word
            for word in re.findall(
                r"\b[a-zA-Z]+\b",
                clean_answer
            )
            if len(word) > 3
        )

        evidence_words = set(
            word
            for word in re.findall(
                r"\b[a-zA-Z]+\b",
                evidence_text
            )
            if len(word) > 3
        )

        overlap = (
            answer_words & evidence_words
        )

        if answer_words:

            overlap_ratio = (
                len(overlap)
                / len(answer_words)
            )

        else:

            overlap_ratio = 0.0

        supported = overlap_ratio >= 0.4

        results.append({
            "evidence_id": evidence_id,
            "supported": supported,
            "overlap_ratio": round(
                overlap_ratio,
                4
            ),
            "reason": None
                if supported
                else "INSUFFICIENT_EVIDENCE_OVERLAP"
        })

    return {
        "valid": all(
            item["supported"]
            for item in results
        ) if results else False,

        "citations": results
    }