import json

from pairwise_judge import pairwise_judge


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_case(case):
    result = pairwise_judge(
        question=case["question"],
        context=case["context"],
        answer_a=case["answer_a"],
        answer_b=case["answer_b"]
    )

    expected = case["expected_winner"]
    actual = result["winner"]

    return {
        "expected": expected,
        "actual": actual,
        "correct": expected == actual,
        "judge": result
    }


def main():

    dataset = load_dataset(
        "pairwise_dataset.json"
    )

    results = []

    for index, case in enumerate(dataset, start=1):

        print(f"\nCase {index}")
        print("-" * 60)

        result = run_case(case)

        results.append(result)

        print("Expected:", result["expected"])
        print("Actual:  ", result["actual"])
        print("Correct: ", result["correct"])

        print(
            "Reason:",
            result["judge"]["reason"]
        )

    correct = sum(
        result["correct"]
        for result in results
    )

    total = len(results)

    accuracy = correct / total if total else 0.0

    print("\n")
    print("=" * 60)
    print("PAIRWISE JUDGE RELIABILITY")
    print("=" * 60)

    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.3f}")

def swap_winner(winner):
    if winner == "A":
        return "B"

    if winner == "B":
       return "A"

    return "TIE"

def run_position_test(case):

    first = pairwise_judge(
        question=case["question"],
          context=case["context"],
          answer_a=case["answer_a"],
          answer_b=case["answer_b"]
        )

    second = pairwise_judge(
        question=case["question"],
            context=case["context"],
        answer_a=case["answer_b"],
            answer_b=case["answer_a"]
        )

    normalized_second = swap_winner(
        second["winner"]
        )

    return {
        "first_winner": first["winner"],
        "second_winner_normalized": normalized_second,
        "consistent": (
            first["winner"]
                == normalized_second
        )
    }
if __name__ == "__main__":
    main()