import json


def load_json(filename):
    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def source_recall(
    retrieved_sources,
    relevant_sources
):
    relevant_sources = set(
        relevant_sources
    )

    if not relevant_sources:
        return 1.0

    retrieved_sources = set(
        retrieved_sources
    )

    found = retrieved_sources.intersection(
        relevant_sources
    )

    return len(found) / len(
        relevant_sources
    )


def source_precision(
    retrieved_sources,
    relevant_sources
):
    if not retrieved_sources:
        return 0.0

    relevant_sources = set(
        relevant_sources
    )

    relevant_count = sum(
        1
        for source in retrieved_sources
        if source in relevant_sources
    )

    return relevant_count / len(
        retrieved_sources
    )


def evidence_recall(
    retrieved_documents,
    required_evidence
):
    """
    Basic evidence-recall metric.

    Each required evidence item is represented
    by keywords/phrases. We check whether those
    terms appear in the retrieved context.

    This is intentionally simple and transparent.
    """

    if not required_evidence:
        return 1.0

    context = " ".join(
        document["document"].lower()
        for document in retrieved_documents
    )

    found = 0

    for evidence in required_evidence:
        evidence_lower = evidence.lower()

        if evidence_lower in context:
            found += 1

    return found / len(
        required_evidence
    )


def evaluate_question(
    evaluation_item,
    retrieved_documents
):
    retrieved_sources = [
        document["metadata"].get(
            "source"
        )
        for document in retrieved_documents
    ]

    relevant_sources = evaluation_item.get(
        "relevant_sources",
        []
    )

    required_evidence = evaluation_item.get(
        "required_evidence",
        []
    )

    return {
        "source_precision": source_precision(
            retrieved_sources,
            relevant_sources
        ),

        "source_recall": source_recall(
            retrieved_sources,
            relevant_sources
        ),

        "evidence_recall": evidence_recall(
            retrieved_documents,
            required_evidence
        )
    }