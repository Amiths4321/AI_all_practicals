import json
import os

from chunking import chunk_text

file_path = "documents.json"

if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
  try:
    with open(file_path, "r", encoding="utf-8") as file:
      source_documents = json.load(file)
  except json.JSONDecodeError:
    print(
        f"⚠️ Warning: '{file_path}' is not valid JSON. Loading an empty list"
        " instead."
    )
    source_documents = []
else:
  print(f"⚠️ Warning: '{file_path}' is empty or missing. Initializing empty list.")
  source_documents = []
  
  with open(file_path, "w", encoding="utf-8") as file:
    json.dump(source_documents, file)
# --------------------------------------------------
# 1. LOAD SOURCE DOCUMENTS
# --------------------------------------------------

with open(
    "documents.json",
    "r",
    encoding="utf-8"
) as file:

    source_documents = json.load(file)


# --------------------------------------------------
# 2. CHUNK CONFIGURATIONS
# --------------------------------------------------

chunk_configs = [
    {
        "chunk_size": 200,
        "overlap": 40
    },
    {
        "chunk_size": 300,
        "overlap": 60
    },
    {
        "chunk_size": 500,
        "overlap": 100
    },
    {
        "chunk_size": 800,
        "overlap": 160
    }
]


# --------------------------------------------------
# 3. BENCHMARK
# --------------------------------------------------

print("\n" + "=" * 70)

print("CHUNKING BENCHMARK")

print("=" * 70)


for config in chunk_configs:

    chunk_size = config[
        "chunk_size"
    ]

    overlap = config[
        "overlap"
    ]

    total_chunks = 0

    total_characters = 0

    largest_chunk = 0

    smallest_chunk = None


    for item in source_documents:

        text = item["text"]

        chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        total_chunks += len(chunks)

        for chunk in chunks:

            length = len(chunk)

            total_characters += length

            largest_chunk = max(
                largest_chunk,
                length
            )

            if (
                smallest_chunk is None
                or length < smallest_chunk
            ):

                smallest_chunk = length


    average_chunk_size = (
        total_characters
        / total_chunks
        if total_chunks
        else 0
    )


    print("\nConfiguration:")

    print(
        f"Chunk size: {chunk_size}"
    )

    print(
        f"Overlap: {overlap}"
    )

    print(
        f"Total chunks: {total_chunks}"
    )

    print(
        f"Average characters: "
        f"{average_chunk_size:.1f}"
    )

    print(
        f"Smallest chunk: "
        f"{smallest_chunk}"
    )

    print(
        f"Largest chunk: "
        f"{largest_chunk}"
    )