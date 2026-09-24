import json
from collections import defaultdict


DETAILS_FILE = "experiment_details.json"


def load_results():
    with open(
        DETAILS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def classify(result):
    answerable = result["answerable"]
    gate_passed = result["gate_passed"]
    evidence_recall = result["evidence_recall"]

    if answerable and gate_passed:
        if evidence_recall >= 1.0:
            return "supported_answer"
        return "partial_evidence"

    if answerable and not gate_passed:
        return "over_abstention"

    if not answerable and gate_passed:
        return "false_answer"

    if not answerable and not gate_passed:
        return "correct_abstention"

    return "unknown"


def main():
    experiments = load_results()

    questions = defaultdict(list)

    for experiment in experiments:
        config = {
            "chunk_size": experiment["chunk_size"],
            "vector_k": experiment["vector_k"],
            "hybrid_k": experiment["hybrid_k"],
            "rerank_k": experiment["rerank_k"],
            "threshold": experiment["evidence_threshold"]
        }

        for result in experiment["question_results"]:
            question = result["question"]

            questions[question].append({
                "category": classify(result),
                "evidence_recall": result["evidence_recall"],
                "gate_passed": result["gate_passed"],
                "config": config
            })

    print("=" * 80)
    print("QUESTION DIFFICULTY ANALYSIS")
    print("=" * 80)

    for question, results in questions.items():

        total = len(results)

        supported = sum(
            r["category"] == "supported_answer"
            for r in results
        )

        partial = sum(
            r["category"] == "partial_evidence"
            for r in results
        )

        over_abstention = sum(
            r["category"] == "over_abstention"
            for r in results
        )

        false_answers = sum(
            r["category"] == "false_answer"
            for r in results
        )

        correct_abstention = sum(
            r["category"] == "correct_abstention"
            for r in results
        )

        avg_evidence = sum(
            r["evidence_recall"]
            for r in results
        ) / total

        print()
        print("-" * 80)
        print(question)
        print("-" * 80)

        print(f"Configurations tested: {total}")
        print(f"Supported answers:      {supported}/{total}")
        print(f"Partial evidence:       {partial}/{total}")
        print(f"Over-abstention:        {over_abstention}/{total}")
        print(f"False answers:          {false_answers}/{total}")
        print(f"Correct abstention:     {correct_abstention}/{total}")
        print(f"Average evidence recall: {avg_evidence:.3f}")


if __name__ == "__main__":
    main()