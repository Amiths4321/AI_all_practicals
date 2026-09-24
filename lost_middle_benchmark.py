import json
import time

from generation import generate_answer
from rag_judge import judge_answer


OUTPUT_FILE = "lost_middle_benchmark.json"

CONTEXT_SIZES = [5, 10, 20]

POSITIONS = [
    "beginning",
    "quarter",
    "middle",
    "three_quarters",
    "end"
]


QUESTION = "Who approves annual leave?"

RELEVANT_DOCUMENT = {
    "id": "relevant",
    "document": (
        "Annual leave requires manager approval "
        "before the leave is taken."
    ),
    "metadata": {
        "source": "leave_policy.txt"
    }
}


def make_noise_document(index):

    return {
        "id": f"noise_{index}",
        "document": (
            f"General company information item {index}. "
            "Employees should follow normal workplace "
            "procedures and company guidelines."
        ),
        "metadata": {
            "source": "employee_handbook.txt"
        }
    }


def build_documents(context_size):

    documents = []

    for index in range(
        context_size - 1
    ):
        documents.append(
            make_noise_document(index)
        )

    documents.append(
        RELEVANT_DOCUMENT
    )

    return documents


def position_index(
    total_documents,
    position
):

    if position == "beginning":
        return 0

    if position == "quarter":
        return total_documents // 4

    if position == "middle":
        return total_documents // 2

    if position == "three_quarters":
        return (
            3 * total_documents // 4
        )

    if position == "end":
        return total_documents - 1

    raise ValueError(
        f"Unknown position: {position}"
    )


def place_relevant_document(
    documents,
    position
):

    documents = [
        document
        for document in documents
        if document["id"] != "relevant"
    ]

    index = position_index(
        len(documents) + 1,
        position
    )

    documents.insert(
        index,
        RELEVANT_DOCUMENT
    )

    return documents


def build_context(documents):

    parts = []

    for index, item in enumerate(
        documents,
        start=1
    ):

        source = item["metadata"].get(
            "source",
            "unknown"
        )

        parts.append(
            f"[SOURCE {index}: {source}]\n"
            f"{item['document']}"
        )

    return "\n\n".join(parts)


def run_experiment():

    results = []

    for context_size in CONTEXT_SIZES:

        base_documents = build_documents(
            context_size
        )

        for position in POSITIONS:

            documents = place_relevant_document(
                base_documents,
                position
            )

            actual_position = next(
                index
                for index, document
                in enumerate(documents)
                if document["id"] == "relevant"
            )

            context = build_context(
                documents
            )

            print()
            print(
                f"Context={context_size} | "
                f"Position={position} | "
                f"Index={actual_position}"
            )

            start = time.perf_counter()

            answer = generate_answer(
                question=QUESTION,
                documents=documents
            )

            latency = (
                time.perf_counter()
                - start
            )

            judgement = judge_answer(
                question=QUESTION,
                context=context,
                answer=answer
            )

            results.append({
                "context_size": context_size,
                "position": position,
                "relevant_index": actual_position,
                "context_characters": len(
                    context
                ),
                "latency_seconds": latency,
                "answer": answer,
                "judgement": judgement
            })

            print(
                f"  Groundedness: "
                f"{judgement['groundedness']}/2"
            )

            print(
                f"  Completeness: "
                f"{judgement['completeness']}/2"
            )

            print(
                f"  Relevance: "
                f"{judgement['relevance']}/2"
            )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print()
    print(
        f"Saved results to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    run_experiment()