def source_retrieval_rate(results):
    if not results:
        return 0.0

    correct = 0

    for item in results:
        relevant_sources = set(
            item["relevant_sources"]
        )

        retrieved_sources = set(
            item["retrieved_sources"]
        )

        if retrieved_sources.intersection(
            relevant_sources
        ):
            correct += 1

    return correct / len(results)


def answer_contains_reference_terms(
    generated_answer,
    reference_answer
):
    generated_words = set(
        generated_answer.lower().split()
    )

    reference_words = set(
        reference_answer.lower().split()
    )

    if not reference_words:
        return False

    overlap = (
        generated_words.intersection(
            reference_words
        )
    )

    return (
        len(overlap)
        / len(reference_words)
    )


def average_reference_overlap(results):
    if not results:
        return 0.0

    scores = []

    for item in results:
        score = answer_contains_reference_terms(
            item["generated_answer"],
            item["reference_answer"]
        )

        scores.append(score)

    return sum(scores) / len(scores)

print("CONTEXT:")

for document in retrieved:
    print(
        f"[{document['metadata'].get('source')}]"
    )

    print(
        document["document"]
    )

    print()

def citation_source_valid(
    answer,
    retrieved_documents
):
    retrieved_sources = {
        item["metadata"].get("source")
        for item in retrieved_documents
    }

    answer_lower = answer.lower()

    cited_sources = []

    for source in retrieved_sources:
        if source.lower() in answer_lower:
            cited_sources.append(source)

    return cited_sources
    
        