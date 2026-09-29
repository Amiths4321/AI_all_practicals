def analyze_retrieval_failure(
    candidates,
    results,
    expected_document,
    expected_section
):

    # 1. Nothing retrieved
    if not candidates:
        return {
            "status": "FAILED",
            "stage": "RETRIEVAL",
            "reason": "NO_CANDIDATES"
        }

    # Check whether expected source
    # entered the candidate set.
    candidate_found = False

    for result in candidates:
        document = result.get("document", {})

        if (
            document.get("document") == expected_document
            and document.get("section") == expected_section
        ):
            candidate_found = True
            break

    # Expected source was never retrieved
    if not candidate_found:
        return {
            "status": "FAILED",
            "stage": "RETRIEVAL",
            "reason": "EXPECTED_SOURCE_NOT_RETRIEVED"
        }

    # 2. Candidates exist but reranking produced nothing
    if not results:
        return {
            "status": "FAILED",
            "stage": "RERANKING",
            "reason": "NO_RERANKED_RESULTS"
        }

    # 3. Expected source entered candidates
    # but disappeared after reranking.
    reranked_found = False

    for result in results:
        document = result.get("document", {})

        if (
            document.get("document") == expected_document
            and document.get("section") == expected_section
        ):
            reranked_found = True
            break

    if not reranked_found:
        return {
            "status": "FAILED",
            "stage": "RERANKING",
            "reason": "RERANKER_DROPPED_EXPECTED_SOURCE"
        }

    # 4. Everything worked
    return {
        "status": "PASSED",
        "stage": "RETRIEVAL_AND_RERANKING",
        "reason": None
    }