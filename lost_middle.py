import json
import time

from generation import generate_answer
from rag_judge import judge_answer


OUTPUT_FILE = "lost_middle_results.json"


def build_context(documents):
    parts = []

    for index, item in enumerate(documents, start=1):
        metadata = item.get("metadata", {}) or {}

        source = metadata.get(
            "source",
            "unknown"
        )

        parts.append(
            f"[SOURCE {index}: {source}]\n"
            f"{item['document']}"
        )

    return "\n\n".join(parts)


def reorder_documents(
    documents,
    relevant_index,
    position
):
    """
    Move the relevant document to the beginning,
    middle, or end while keeping all other
    documents unchanged.
    """

    documents = documents.copy()

    relevant_document = documents.pop(
        relevant_index
    )

    if position == "beginning":
        documents.insert(
            0,
            relevant_document
        )

    elif position == "end":
        documents.append(
            relevant_document
        )

    elif position == "middle":

        middle_index = len(documents) // 2

        documents.insert(
            middle_index,
            relevant_document
        )

    else:
        raise ValueError(
            f"Unknown position: {position}"
        )

    return documents


def run_experiment(
    question,
    documents,
    relevant_index
):

    results = {}

    for position in [
        "beginning",
        "middle",
        "end"
    ]:

        ordered_documents = reorder_documents(
            documents=documents,
            relevant_index=relevant_index,
            position=position
        )

        context = build_context(
            ordered_documents
        )

        start = time.perf_counter()

        answer = generate_answer(
            question=question,
            documents=ordered_documents
        )

        latency = (
            time.perf_counter()
            - start
        )

        judgement = judge_answer(
            question=question,
            context=context,
            answer=answer
        )

        results[position] = {
            "answer": answer,
            "context_characters": len(
                context
            ),
            "latency_seconds": latency,
            "judgement": judgement
        }

    return results


def main():

    # Small controlled experiment.
    #
    # The relevant evidence is deliberately
    # identical in every position.

    question = (
        "Who approves annual leave?"
    )

    documents = [

        {
            "id": "noise_1",
            "document": (
                "Employees should maintain "
                "accurate personal information "
                "in company systems."
            ),
            "metadata": {
                "source": "employee_handbook.txt"
            }
        },

        {
            "id": "relevant",
            "document": (
                "Annual leave requires manager "
                "approval before the leave is taken."
            ),
            "metadata": {
                "source": "leave_policy.txt"
            }
        },

        {
            "id": "noise_2",
            "document": (
                "Employees are expected to follow "
                "normal workplace attendance procedures."
            ),
            "metadata": {
                "source": "employee_handbook.txt"
            }
        },

        {
            "id": "noise_3",
            "document": (
                "Employees should submit required "
                "administrative requests through "
                "the appropriate company system."
            ),
            "metadata": {
                "source": "employee_handbook.txt"
            }
        },

        {
            "id": "noise_4",
            "document": (
                "Managers may communicate operational "
                "updates to members of their teams."
            ),
            "metadata": {
                "source": "employee_handbook.txt"
            }
        }
    ]

    results = run_experiment(
        question=question,
        documents=documents,
        relevant_index=1
    )

    for position, result in results.items():

        judgement = result["judgement"]

        print()
        print("=" * 70)
        print(f"POSITION: {position}")
        print("=" * 70)

        print(
            f"Context characters: "
            f"{result['context_characters']}"
        )

        print(
            f"Latency: "
            f"{result['latency_seconds']:.3f}s"
        )

        print(
            f"Groundedness: "
            f"{judgement['groundedness']}/2"
        )

        print(
            f"Relevance: "
            f"{judgement['relevance']}/2"
        )

        print(
            f"Completeness: "
            f"{judgement['completeness']}/2"
        )

        print(
            f"Unsupported claims: "
            f"{judgement['unsupported_claims']}/2"
        )

        print()
        print("Answer:")
        print(result["answer"])

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "question": question,
                "results": results
            },
            f,
            indent=2
        )

    print()
    print(
        f"Saved results to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()