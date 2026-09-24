import json
from statistics import mean


INPUT_FILE = "exact_evidence_position.json"
OUTPUT_FILE = "position_sensitivity.json"

POSITIONS = [
    "beginning",
    "quarter",
    "middle",
    "three_quarters",
    "end"
]


def average(values):
    return mean(values) if values else 0.0


def calculate_question_metrics(question_result):

    results = question_result["results"]

    groundedness = [
        results[position]["judgement"]["groundedness"]
        for position in POSITIONS
        if position in results
    ]

    completeness = [
        results[position]["judgement"]["completeness"]
        for position in POSITIONS
        if position in results
    ]

    relevance = [
        results[position]["judgement"]["relevance"]
        for position in POSITIONS
        if position in results
    ]

    unsupported_claims = [
        results[position]["judgement"]["unsupported_claims"]
        for position in POSITIONS
        if position in results
    ]

    return {
        "average_groundedness":
            average(groundedness),

        "groundedness_range":
            max(groundedness) - min(groundedness),

        "average_completeness":
            average(completeness),

        "completeness_range":
            max(completeness) - min(completeness),

        "average_relevance":
            average(relevance),

        "relevance_range":
            max(relevance) - min(relevance),

        "average_unsupported_claims":
            average(unsupported_claims),

        "unsupported_claims_range":
            max(unsupported_claims) -
            min(unsupported_claims)
    }


def build_report(results):

    questions = []

    for question_result in results:

        metrics = calculate_question_metrics(
            question_result
        )

        questions.append({
            "question":
                question_result["question"],

            "evidence_id":
                question_result["evidence_id"],

            "evidence_similarity":
                question_result[
                    "evidence_similarity"
                ],

            **metrics
        })

    return questions


def aggregate_questions(question_reports):

    if not question_reports:
        return {}

    return {
        "questions": len(question_reports),

        "average_groundedness":
            average([
                item["average_groundedness"]
                for item in question_reports
            ]),

        "average_groundedness_range":
            average([
                item["groundedness_range"]
                for item in question_reports
            ]),

        "average_completeness":
            average([
                item["average_completeness"]
                for item in question_reports
            ]),

        "average_completeness_range":
            average([
                item["completeness_range"]
                for item in question_reports
            ]),

        "average_relevance":
            average([
                item["average_relevance"]
                for item in question_reports
            ]),

        "average_relevance_range":
            average([
                item["relevance_range"]
                for item in question_reports
            ]),

        "average_unsupported_claims":
            average([
                item["average_unsupported_claims"]
                for item in question_reports
            ]),

        "average_unsupported_claims_range":
            average([
                item["unsupported_claims_range"]
                for item in question_reports
            ])
    }


def print_report(
    question_reports,
    aggregate
):

    print()
    print("=" * 100)
    print("POSITION SENSITIVITY ANALYSIS")
    print("=" * 100)

    print()

    for item in question_reports:

        print(
            f"Question: {item['question']}"
        )

        print(
            f"  Evidence similarity: "
            f"{item['evidence_similarity']:.3f}"
        )

        print(
            f"  Groundedness: "
            f"{item['average_groundedness']:.2f} "
            f"(range {item['groundedness_range']:.2f})"
        )

        print(
            f"  Completeness: "
            f"{item['average_completeness']:.2f} "
            f"(range {item['completeness_range']:.2f})"
        )

        print(
            f"  Relevance: "
            f"{item['average_relevance']:.2f} "
            f"(range {item['relevance_range']:.2f})"
        )

        print(
            f"  Unsupported claims: "
            f"{item['average_unsupported_claims']:.2f} "
            f"(range {item['unsupported_claims_range']:.2f})"
        )

        print()

    print("-" * 100)
    print("OVERALL")
    print("-" * 100)

    for key, value in aggregate.items():

        if isinstance(value, float):
            print(
                f"{key}: {value:.3f}"
            )
        else:
            print(
                f"{key}: {value}"
            )


def main():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        results = json.load(f)

    if not results:
        raise ValueError(
            "No results found."
        )

    question_reports = (
        build_report(results)
    )

    aggregate = aggregate_questions(
        question_reports
    )

    output = {
        "questions": question_reports,
        "aggregate": aggregate
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    print_report(
        question_reports,
        aggregate
    )

    print()
    print(
        f"Saved report to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()