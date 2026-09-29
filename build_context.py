from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. Embedding model
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 2. Qdrant
# -----------------------------
client = QdrantClient(":memory:")

collection_name = "banking_policies"

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)


# -----------------------------
# 3. Knowledge base
# -----------------------------
chunks = [
    {
        "chunk_id": 1,
        "section": "Eligibility",
        "text": "Applicants must have a minimum monthly income of ₹50,000."
    },
    {
        "chunk_id": 2,
        "section": "Eligibility",
        "text": "Applicants must be at least 21 years old."
    },
    {
        "chunk_id": 3,
        "section": "Eligibility",
        "text": "The maximum applicant age at loan maturity is 65 years."
    },
    {
        "chunk_id": 4,
        "section": "Documents",
        "text": "Applicants must provide valid identity proof."
    },
    {
        "chunk_id": 5,
        "section": "Documents",
        "text": "Applicants must provide address proof."
    },
    {
        "chunk_id": 6,
        "section": "Documents",
        "text": "Applicants must provide income proof."
    }
]


# -----------------------------
# 4. Create embeddings
# -----------------------------
texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(texts)


# -----------------------------
# 5. Store in Qdrant
# -----------------------------
points = []

for chunk, embedding in zip(chunks, embeddings):

    points.append(
        PointStruct(
            id=chunk["chunk_id"],
            vector=embedding.tolist(),
            payload={
                "chunk_id": chunk["chunk_id"],
                "document": "banking_policy.txt",
                "section": chunk["section"],
                "text": chunk["text"]
            }
        )
    )


client.upsert(
    collection_name=collection_name,
    points=points
)


# -----------------------------
# 6. User query
# -----------------------------
query = "What salary is required for a home loan?"

query_vector = model.encode(query).tolist()


# -----------------------------
# 7. Retrieve Top-3
# -----------------------------
results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=3
).points


# -----------------------------
# 8. Build context
# -----------------------------
context_parts = []

for rank, result in enumerate(results, start=1):

    payload = result.payload

    context_parts.append(
        f"""
Source {rank}
Document: {payload['document']}
Section: {payload['section']}
Chunk ID: {payload['chunk_id']}
Similarity Score: {result.score:.4f}

Evidence:
{payload['text']}
"""
    )


context = "\n".join(context_parts)


# -----------------------------
# 9. Display final context
# -----------------------------
print("\n==============================")
print("CONTEXT FOR LLM")
print("==============================")

print(context)