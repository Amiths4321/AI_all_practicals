from semantic_evidence import (
    SemanticEvidenceEvaluator
)

semantic_evaluator = (
    SemanticEvidenceEvaluator()
)

evaluator = SemanticEvidenceEvaluator()

hybrid_retriever = HybridRetriever(
    chunks
)

reranker = DocumentReranker()

semantic_evaluator = (
    SemanticEvidenceEvaluator()
)

tests = [
    (
        "manager approval",
        "Annual leave must be approved by the employee's manager."
    ),
    (
        "medical documentation",
        "Employees may be required to submit a medical certificate."
    ),
    (
        "remote work",
        "Annual leave must be approved by the employee's manager."
    )
]


for evidence, document in tests:

    score = evaluator.score(
        evidence,
        document
    )

    print()
    print("Evidence:")
    print(evidence)

    print("Document:")
    print(document)

    print(
        f"Similarity: {score:.4f}"
    )