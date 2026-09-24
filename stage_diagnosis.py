import json


DETAILS_FILE = "experiment_details.json"


def load_results():
    with open(
        DETAILS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def get_ids(results):
    return {
        item["id"]
        for item in results
    }


def diagnose(question_result):
    expected_sources = set(
        question_result.get("expected_sources", [])
    )

    vector_ids = get_ids(
        question_result["vector_results"]
    )

    keyword_ids = get_ids(
        question_result["keyword_results"]
    )

    hybrid_ids = get_ids(
        question_result["hybrid_results"]
    )

    reranked_ids = get_ids(
        question_result["reranked_results"]
    )

    if not expected_sources:
        return "unanswerable_or_no_expected_source"

    def contains_expected_source(results):
        for item in results:
            source = (
                item.get("metadata", {})
                .get("source")
            )

            if source in expected_sources:
                return True

        return False

    vector_has = contains_expected_source(
        question_result["vector_results"]
    )

    keyword_has = contains_expected_source(
        question_result["keyword_results"]
    )

    hybrid_has = contains_expected_source(
        question_result["hybrid_results"]
    )

    reranked_has = contains_expected_source(
        question_result["reranked_results"]
    )

    if not vector_has and not keyword_has:
        return "retrieval_failure"

    if keyword_has and not vector_has:
        if hybrid_has:
            return "vector_retrieval_missed_but_bm25_recovered"
        return "hybrid_failed_to_recover_bm25"

    if vector_has and not hybrid_has:
        return "hybrid_rrf_lost_vector_result"

    if hybrid_has and not reranked_has:
        return "reranker_removed_expected_source"

    if reranked_has:
        return "evidence_available_after_reranking"

    return "unknown"


def main():
    experiments = load_results()

    print("=" * 80)
    print("RETRIEVAL STAGE DIAGNOSIS")
    print("=" * 80)

    counts = {}

    for experiment in experiments:

        for result in experiment["question_results"]:

            # Add expected source information if available.
            result["expected_sources"] = (
                result.get("expected_sources", [])
            )

            diagnosis = diagnose(result)

            counts[diagnosis] = (
                counts.get(diagnosis, 0) + 1
            )

            print()
            print(
                f"Question: {result['question']}"
            )
            print(
                f"Diagnosis: {diagnosis}"
            )

    print()
    print("=" * 80)
    print("DIAGNOSIS COUNTS")
    print("=" * 80)

    for diagnosis, count in sorted(
        counts.items()
    ):
        print(
            f"{diagnosis:<45} {count}"
        )


if __name__ == "__main__":
    main()