from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Local in-memory Qdrant
client = QdrantClient(":memory:")

collection_name = "banking_policies"

# Create collection
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

# Policy chunks
chunks = [
    "Applicants must have a minimum monthly income of ₹50,000.",
    "Applicants must be at least 21 years old.",
    "The maximum applicant age at loan maturity is 65 years.",
    "Applicants must provide valid identity proof.",
    "Applicants must provide address proof.",
    "Applicants must provide income proof."
]

# Create embeddings
embeddings = model.encode(chunks)

# Store vectors in Qdrant
points = []

for i, embedding in enumerate(embeddings):
    points.append(
        PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={
                "chunk_id": i + 1,
                "document": "banking_policy.txt",
                "text": chunks[i]
            }
        )
    )

client.upsert(
    collection_name=collection_name,
    points=points
)

print("Documents stored in Qdrant.")


# User query
query = "What salary is required for a home loan?"

# Create query embedding
query_vector = model.encode(query).tolist()

# Search Qdrant
results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=3
).points

print("\nTop 3 Retrieved Chunks:")

for rank, result in enumerate(results, start=1):
    print(f"\nRank: {rank}")
    print(f"Score: {result.score:.4f}")
    print(f"Chunk ID: {result.payload['chunk_id']}")
    print(f"Text: {result.payload['text']}")