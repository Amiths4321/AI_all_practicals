from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)


# ==========================================
# 1. Qdrant
# ==========================================

client = QdrantClient(
    path="qdrant_storage"
)

collection_name = "banking_policies"


if not client.collection_exists(collection_name):

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

    print("Collection created.")

else:

    print("Collection already exists.")


# ==========================================
# 2. Load embedding model
# ==========================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# 3. Read all documents
# ==========================================

data_folder = Path("data")

documents = []


for file_path in data_folder.glob("*.txt"):

    # --------------------------------------
    # Determine product
    # --------------------------------------

    if "banking_policy" in file_path.name:

        product = "home_loan"

    elif "personal_loan" in file_path.name:

        product = "personal_loan"

    else:

        product = "unknown"


    # --------------------------------------
    # Determine policy version
    # --------------------------------------

    if file_path.name == "banking_policy.txt":

        policy_version = "v1.0"
        effective_date = "2026-01-01"

    elif file_path.name == "banking_policy_v2.txt":

        policy_version = "v2.0"
        effective_date = "2026-06-01"

    elif file_path.name == "personal_loan_policy.txt":

        policy_version = "v1.0"
        effective_date = "2026-01-01"

    else:

        policy_version = "v1.0"
        effective_date = "2026-01-01"


    # --------------------------------------
    # Read file
    # --------------------------------------

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()


    # --------------------------------------
    # Chunk document
    # --------------------------------------

    raw_chunks = [
        chunk.strip()
        for chunk in text.split("\n\n")
        if chunk.strip()
    ]


    # --------------------------------------
    # Create document records
    # --------------------------------------

    for chunk in raw_chunks:

        first_line = chunk.split("\n")[0].strip()


        if first_line.endswith(":"):

            section = first_line[:-1]

        else:

            section = "General"


        documents.append({

            "document": file_path.name,

            "product": product,

            "document_type": "loan_policy",

            "policy_version": policy_version,

            "effective_date": effective_date,

            "section": section,

            "text": chunk

        })


print(
    "\nDocuments found:",
    len(list(data_folder.glob("*.txt")))
)

print(
    "Chunks created:",
    len(documents)
)


# ==========================================
# 4. Create embeddings
# ==========================================

texts = [
    document["text"]
    for document in documents
]


embeddings = model.encode(texts)


print(
    "Embedding shape:",
    embeddings.shape
)


# ==========================================
# 5. Prepare Qdrant points
# ==========================================

points = []


for i, (document, embedding) in enumerate(
    zip(documents, embeddings),
    start=1
):

    points.append(

        PointStruct(

            id=i,

            vector=embedding.tolist(),

            payload={

                "chunk_id": i,

                "document":
                    document["document"],

                "product":
                    document["product"],

                "document_type":
                    document["document_type"],

                "policy_version":
                    document["policy_version"],

                "effective_date":
                    document["effective_date"],

                "section":
                    document["section"],

                "text":
                    document["text"]

            }

        )
    )


# ==========================================
# 6. Store in Qdrant
# ==========================================

client.upsert(

    collection_name=collection_name,

    points=points

)


print(
    "\nAll documents successfully indexed."
)


# ==========================================
# 7. Close Qdrant
# ==========================================

client.close()