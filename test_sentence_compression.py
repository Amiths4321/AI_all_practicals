from sentence_compression import SentenceCompressor


documents = [
    {
        "id": "policy1_0",
        "document": (
            "Employees must submit annual leave requests "
            "through the leave system. "
            "Annual leave requires manager approval. "
            "Requests should normally be submitted in advance. "
            "Employees should check their remaining leave balance."
        ),
        "metadata": {
            "source": "leave_policy.txt"
        }
    }
]


compressor = SentenceCompressor(
    threshold=0.45
)


question = "Who approves annual leave?"


compressed = compressor.compress(
    question=question,
    documents=documents
)


print("\nSelected sentences:\n")

for item in compressed:
    print(
        f"{item['similarity']:.3f} | "
        f"{item['sentence']}"
    )


reconstructed = compressor.reconstruct_documents(
    compressed
)


print("\nReconstructed documents:\n")

for document in reconstructed:
    print(document["document"])