with open("data/banking_policy.txt", "r", encoding="utf-8") as file:
    text = file.read()

chunks = text.split("\n\n")

documents = []

for i, chunk in enumerate(chunks):
    documents.append({
        "chunk_id": i + 1,
        "document": "banking_policy.txt",
        "section": chunk.split(":")[0].strip(),
        "text": chunk.strip()
    })

for doc in documents:
    print("\n", doc)