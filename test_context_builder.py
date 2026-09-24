from context_builder import ContextBuilder


documents = [
    {
        "id": "1",
        "document": "Annual leave requires manager approval.",
        "metadata": {
            "source": "leave_policy.txt"
        }
    },
    {
        "id": "2",
        "document": "Employees should submit requests in advance.",
        "metadata": {
            "source": "leave_policy.txt"
        }
    },
    {
        "id": "3",
        "document": "Employees should maintain accurate records.",
        "metadata": {
            "source": "employee_handbook.txt"
        }
    }
]


builder = ContextBuilder()


question = "Who approves annual leave?"

required_evidence = [
    "annual leave approval authority"
]


for strategy in [
    "original",
    "top_n",
    "evidence_aware",
    "evidence_first",
    "evidence_last",
    "sentence"
]:

    result = builder.build(
        question=question,
        reranked_documents=documents,
        strategy=strategy,
        required_evidence=required_evidence,
        max_documents=2
    )

    print()
    print("=" * 60)
    print(strategy)
    print("=" * 60)

    for item in result:
        print(
            item["document"]
        )