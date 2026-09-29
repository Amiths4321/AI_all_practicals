def check_retrieval_quality(
    results,
    expected_document=None,
    expected_section=None,
    k=3
):
    if not results:
        return {
            "valid": False,
            "reason": "NO_RESULTS"
        }

    top_results = results[:k]

    # Check whether the expected source was retrieved
    source_found = False

    if expected_document and expected_section:
        for result in top_results:
            document = result.get("document", {})

            if (
                document.get("document") == expected_document
                and document.get("section") == expected_section
            ):
                source_found = True
                break

    if expected_document and expected_section:
        if source_found:
            return {
                "valid": True,
                "reason": "EXPECTED_SOURCE_FOUND"
            }

        return {
            "valid": False,
            "reason": "EXPECTED_SOURCE_NOT_FOUND"
        }

    # Normal production mode
    return {
        "valid": True,
        "reason": "RESULTS_AVAILABLE"
    }