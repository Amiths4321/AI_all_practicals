import os

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama


print("\n==============================")
print("RAG HEALTH CHECK")
print("==============================")


# 1. Qdrant
try:
    client = QdrantClient(path="qdrant_storage")

    if client.collection_exists("banking_policies"):
        print("Qdrant:        OK")
    else:
        print("Qdrant:        COLLECTION MISSING")

    client.close()

except Exception as e:
    print("Qdrant:        FAILED")
    print("Error:", e)


# 2. Embedding model
try:
    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Embeddings:    OK")

except Exception as e:
    print("Embeddings:    FAILED")
    print("Error:", e)


# 3. Ollama
try:
    models = ollama.list()

    print("Ollama:        OK")

except Exception as e:
    print("Ollama:        FAILED")
    print("Error:", e)


# 4. Data folder
if os.path.exists("data"):
    print("Data folder:   OK")
else:
    print("Data folder:   MISSING")


# 5. Qdrant storage
if os.path.exists("qdrant_storage"):
    print("Qdrant storage: OK")
else:
    print("Qdrant storage: MISSING")


print("==============================")
print("HEALTH CHECK COMPLETE")
print("==============================")