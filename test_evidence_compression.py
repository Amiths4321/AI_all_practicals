from rag_benchmark import load_json, build_chunks

from semantic_evidence import SemanticEvidenceEvaluator

from evidence_compression import (
    EvidenceAwareCompressor
)


def main():

    documents = load_json(
        "documents.json"
    )

    evaluation_data = load_json(
        "evaluation_data.json"
    )

    chunks = build_chunks(
        documents,
        chunk_size=500,
        overlap=100
    )

    evaluator = SemanticEvidenceEvaluator()

    compressor = EvidenceAwareCompressor(
        evaluator=evaluator,
        threshold=0.60
    )

    for case in evaluation_data:

        required_evidence = case.get(
            "required_evidence",
            []
        )

        if not required_evidence:
            continue

        selected = compressor.compress(
            required_evidence=required_evidence,
            reranked_documents=[
                {
                    "id": chunk["id"],
                    "document": chunk["document"],
                    "metadata": chunk["metadata"]
                }
                for chunk in chunks
            ],
            max_documents=3
        )

        print()
        print("=" * 70)
        print(case["question"])
        print("=" * 70)

        print("Required evidence:")

        for evidence in required_evidence:
            print(f"- {evidence}")

        print("\nSelected documents:")

        for document in selected:
            print(
                f"- {document['id']} | "
                f"{document['metadata'].get('source')}"
            )


if __name__ == "__main__":
    main()