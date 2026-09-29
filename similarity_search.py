import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = [
    "Applicants must have a minimum monthly income of ₹50,000.",
    "Applicants must be at least 21 years old.",
    "The maximum applicant age at loan maturity is 65 years.",
    "Applicants must provide valid identity proof.",
    "Applicants must provide address proof.",
    "Applicants must provide income proof."
]

# Create chunk embeddings
chunk_embeddings = model.encode(chunks)

# User question
query = "What salary is required for a home loan?"

# Create query embedding
query_embedding = model.encode([query])[0]


# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# Compare query with every chunk
scores = []

for i, chunk_embedding in enumerate(chunk_embeddings):
    score = cosine_similarity(
        query_embedding,
        chunk_embedding
    )

    scores.append((i, score))


# Sort highest similarity first
scores.sort(
    key=lambda x: x[1],
    reverse=True
)


print("Query:", query)

print("\nResults:")

for index, score in scores:
    print(
        f"{score:.4f} -> {chunks[index]}"
    )