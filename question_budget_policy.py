import json


INPUT_FILE = "evidence_aware_budget_sweep.json"


def quality_index(judgement):
    if judgement is None:
        return 0.0

    return (
        judgement["groundedness"]
        + judgement["relevance"]
        + judgement["completeness"]
    ) / 6.0


def load_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_rows(data):
    """
    Flatten the nested benchmark structure into one row per
    (question, budget) combination under the evidence_aware strategy.
    """
    if not isinstance(data, dict):
        return []

    rows = []

    for question_result in data.get("results", []):
        question = question_result.get("question", "")

        strategies = question_result.get("strategies", {})
        evidence_aware = strategies.get("evidence_aware", {})

        for budget, result in evidence_aware.items():
            if not isinstance(result, dict):
                continue

            judgement = result.get("judgement")

            rows.append({
                "question": question,
                "budget": int(budget),
                "quality": quality_index(judgement),
                "evidence_recall": result.get(
                    "strategy_evidence_recall",
                    result.get("evidence_recall", 0.0)
                ),
                "context_characters": result.get("context_characters", 0),
                "latency_seconds": result.get(
                    "generation_latency_seconds",
                    result.get("latency_seconds", 0.0)
                )
            })

    return rows


def choose_budget(rows, minimum_quality=0.80, minimum_evidence=1.0):
    """
    Pick the row with the smallest budget that clears both the
    quality and evidence-recall thresholds. Returns None if no
    row for this question ever clears the bar.
    """
    valid = [
        row
        for row in rows
        if row["quality"] >= minimum_quality
        and row["evidence_recall"] >= minimum_evidence
    ]

    if not valid:
        return None

    # Smallest budget first; break ties on smaller context.
    valid.sort(key=lambda row: (row["budget"], row["context_characters"]))

    return valid[0]


def build_policy(rows):
    """
    Group rows by question and choose the smallest budget that
    meets the quality/evidence bar for each question independently.
    """
    questions = {}

    for row in rows:
        questions.setdefault(row["question"], []).append(row)

    policy = {}

    for question, question_rows in questions.items():
        selected = choose_budget(question_rows)

        if selected is None:
            policy[question] = {
                "budget": None,
                "reason": "no_successful_budget"
            }
        else:
            policy[question] = {
                "budget": selected["budget"],
                "reason": "smallest_successful_budget",
                "quality": selected["quality"],
                "evidence_recall": selected["evidence_recall"],
                "context_characters": selected["context_characters"],
                "latency_seconds": selected["latency_seconds"]
            }

    return policy


def main():
    data = load_data()
    rows = extract_rows(data)

    print(f"Loaded {len(rows)} benchmark rows.")

    policy = build_policy(rows)

    print()
    print("=" * 70)
    print("Question-Specific Budget Policy")
    print("=" * 70)

    for question, decision in policy.items():
        print()
        print(question)
        print("  budget:", decision["budget"])
        print("  reason:", decision["reason"])

    output = {
        "source_file": INPUT_FILE,
        "policy": policy
    }

    with open("question_budget_policy.json", "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print()
    print("Saved question_budget_policy.json")


if __name__ == "__main__":
    main()