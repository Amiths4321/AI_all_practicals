import json


DETAILS_FILE = "experiment_details.json"


def load_results():
    with open(DETAILS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def contains_expected_source(results, expected_sources):
    expected_sources = set(expected_sources)

    for item in results:
        metadata = item.get("metadata") or {}
        source = metadata.get("source")

        if source in expected_sources:
            return True

    return False


def main():
    experiments = load_results()

    stage_counts = {
        "vector": [0, 0],
        "bm25": [0, 0],
        "hybrid": [0, 0],
        "reranker": [0, 0]
    }

    total_answerable = 0

    for experiment in experiments:
        for result in experiment["question_results"]:

            if not result["answerable"]:
                continue

            expected_sources = result.get(
                "expected_sources",
                []
            )

            if not expected_sources:
                continue

            total_answerable += 1

            stages = {
                "vector": result["vector_results"],
                "bm25": result["keyword_results"],
                "hybrid": result["hybrid_results"],
                "reranker": result["reranked_results"]
            }

            for stage, documents in stages.items():

                stage_counts[stage][1] += 1

                if contains_expected_source(
                    documents,
                    expected_sources
                ):
                    stage_counts[stage][0] += 1

    print("=" * 70)
    print("RETRIEVAL STAGE RECALL")
    print("=" * 70)

    print(
        f"Answerable cases evaluated: "
        f"{total_answerable}"
    )

    print()

    for stage, (found, total) in stage_counts.items():

        recall = found / total if total else 0

        print(
            f"{stage:<12} "
            f"{found:>4}/{total:<4} "
            f"recall={recall:.3f}"
        )


if __name__ == "__main__":
    main()