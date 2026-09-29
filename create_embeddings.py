from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

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

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)

print("\nFirst embedding:")
print(embeddings[0])