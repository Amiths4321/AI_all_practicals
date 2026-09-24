import json
from collections import defaultdict


INPUT_FILE = "evidence_aware_budget_sweep.json"


def quality_index(judgement):
    if judgement is None:
        return 0.0

    return (
        judgement["groundedness"]
        + judgement["relevance"]
        + judgement["completeness"]
    ) / 6.0


def load_results():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def extract_rows(data):

    # If data is already a list, return it directly
    if isinstance(data, list):
        return data

  # If it's a dictionary, look for the relevant key safely
    if isinstance(data, dict):
        # Try common key names or fallback to empty list
        return data.get("results", data.get("questions", data.get("data", []))))    

    return []

    rows = []

    # Supports the common structure:
    #
    # {
    #   "results": [
    #       {
    #           "question": "...",
    #           "strategies": {...}
    #       }
    #   ]
    # }

    results = data.get("results", [])

    for question_result in results:

        question = question_result.get(
            "question",
            ""
        )

        strategies = question_result.get(
            "strategies",
            {}
        )

        for strategy, budgets in strategies.items():

            if not isinstance(budgets, dict):
                continue

            for budget, result in budgets.items():

                if not isinstance(result, dict):
                    continue

                judgement = result.get(
                    "judgement"
                )

                quality = quality_index(
                    judgement
                )

                rows.append({
                    "question": question,
                    "strategy": strategy,
                    "budget": int(budget),
                    "context_characters": result.get(
                        "context_characters",
                        0
                    ),
                    "latency_seconds": result.get(
                        "generation_latency_seconds",
                        result.get(
                            "latency_seconds",
                            0.0
                        )
                    ),
                    "quality": quality,
                    "evidence_recall": result.get(
                        "strategy_evidence_recall",
                        result.get(
                            "evidence_recall",
                            0.0
                        )
                    )
                })

    return rows


def build_policy(rows):

    grouped = defaultdict(list)

    for row in rows:

        if row.get("strategy") != "evidence_aware":
            continue

        grouped[row.get("budget")].append(row)

    policy = []

    for budget, items in sorted(
        grouped.items()
    ):

        average_quality = (
            sum(
                item["quality"]
                for item in items
            ) / len(items)
        )

        average_context = (
            sum(
                item["context_characters"]
                for item in items
            ) / len(items)
        )

        average_latency = (
            sum(
                item["latency_seconds"]
                for item in items
            ) / len(items)
        )

        average_evidence = (
            sum(
                item["evidence_recall"]
                for item in items
            ) / len(items)
        )

        policy.append({
            "budget": budget,
            "questions": len(items),
            "average_quality": average_quality,
            "average_context_characters":
                average_context,
            "average_latency_seconds":
                average_latency,
            "average_evidence_recall":
                average_evidence
        })

    return policy


def choose_empirical_budget(
    policy,
    minimum_quality=0.80,
    minimum_evidence_recall=1.0
):

    candidates = [
        row
        for row in policy
        if row["average_quality"] >= minimum_quality
        and row["average_evidence_recall"]
            >= minimum_evidence_recall
    ]

    if not candidates:

        return {
            "budget": None,
            "reason": "no_budget_meets_requirements"
        }

    # Prefer the smallest context that
    # satisfies the requirements.

    candidates.sort(
        key=lambda row: (
            row["average_context_characters"],
            row["budget"]
        )
    )

    selected = candidates[0]

    return {
        "budget": selected["budget"],
        "reason": "empirically_selected",
        "average_quality":
            selected["average_quality"],
        "average_evidence_recall":
            selected["average_evidence_recall"],
        "average_context_characters":
            selected["average_context_characters"]
    }


def main():

    data = load_results()

    rows = extract_rows(data)

    print(
        f"Loaded {len(rows)} benchmark rows."
    )

    if not rows:

        print(
            "No compatible benchmark rows found."
        )

        return

    policy = build_policy(rows)

    print()
    print("Empirical Budget Policy")
    print("=" * 70)

    for row in policy:

        print(
            f"Budget {row['budget']}: "
            f"quality={row['average_quality']:.3f}, "
            f"evidence={row['average_evidence_recall']:.3f}, "
            f"context={row['average_context_characters']:.0f}, "
            f"latency={row['average_latency_seconds']:.3f}s"
        )

    decision = choose_empirical_budget(
        policy,
        minimum_quality=0.80,
        minimum_evidence_recall=1.0
    )

    print()
    print("Selected policy:")
    print(decision)

    output = {
        "source_file": INPUT_FILE,
        "policy": policy,
        "selected_budget": decision
    }

    with open(
        "empirical_context_policy.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )

    print()
    print(
        "Saved empirical_context_policy.json"
    )


if __name__ == "__main__":
    main()