import json
import os
import shutil

import chromadb

from chunking import chunk_text


CHUNK_CONFIGS = [
    {"chunk_size": 200, "overlap": 40},
    {"chunk_size": 300, "overlap": 60},
    {"chunk_size": 500, "overlap": 100},
    {"chunk_size": 800, "overlap": 160},
]

TOP_K = 5
DB_ROOT = "./benchmark_chroma"


def load_documents():
  file_path = "documents.json"
  if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
    print(
        f"⚠️ Warning: '{file_path}' is missing or empty. Initializing with `[]`."
    )
    with open(file_path, "w", encoding="utf-8") as file:
      json.dump([], file)

  with open(file_path, "r", encoding="utf-8") as file:
    try:
      return json.load(file)
    except json.JSONDecodeError:
      print(
          f"⚠️ Warning: '{file_path}' contains invalid JSON. Resetting to empty"
          " list."
      )
      return []


def load_evaluation_data():
  file_path = "rag_eval.json"
  if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
    print(
        f"⚠️ Warning: '{file_path}' is missing or empty. Initializing with `[]`."
    )
    with open(file_path, "w", encoding="utf-8") as file:
      json.dump([], file)

  with open(file_path, "r", encoding="utf-8") as file:
    try:
      return json.load(file)
    except json.JSONDecodeError:
      print(
          f"⚠️ Warning: '{file_path}' contains invalid JSON. Resetting to empty"
          " list."
      )
      return []


def build_chunks(documents, chunk_size, overlap):
    chunks = []

    for document in documents:
        source = document["source"]
        text = document["text"]

        document_chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk_index, chunk in enumerate(document_chunks):
            chunks.append({
                "id": f"{source}_chunk_{chunk_index}",
                "document": chunk,
                "metadata": {
                    "source": source,
                    "chunk_index": chunk_index
                }
            })

    return chunks


def create_collection(chunks, collection_name):
    client = chromadb.PersistentClient(
        path=os.path.join(DB_ROOT, collection_name)
    )

    collection = client.get_or_create_collection(
        name="documents"
    )

    if collection.count() > 0:
        collection.delete(
            where={}
        )

    collection.add(
        ids=[item["id"] for item in chunks],
        documents=[item["document"] for item in chunks],
        metadatas=[item["metadata"] for item in chunks]
    )

    return collection


def retrieve(collection, question, top_k):
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    ids = results["ids"][0]

    return ids


def precision_at_k(retrieved_ids, relevant_ids, k):
    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    relevant_count = sum(
        1
        for doc_id in retrieved
        if doc_id in relevant_ids
    )

    return relevant_count / len(retrieved)


def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 1.0

    retrieved = retrieved_ids[:k]

    relevant_count = sum(
        1
        for doc_id in retrieved
        if doc_id in relevant_ids
    )

    return relevant_count / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1 / rank

    return 0.0


def evaluate_collection(collection, evaluation_data):
    precisions = []
    recalls = []
    reciprocal_ranks = []

    for item in evaluation_data:
        question = item["question"]
        relevant_sources = set(item["relevant_sources"])

        retrieved_ids = retrieve(
            collection,
            question,
            TOP_K
        )

        precision = precision_at_k(
            retrieved_ids,
            relevant_ids,
            TOP_K
        )

        recall = recall_at_k(
            retrieved_ids,
            relevant_ids,
            TOP_K
        )

        rr = reciprocal_rank(
            retrieved_ids,
            relevant_ids
        )
        retrieved = collection.get(
            ids=retrieved_ids,
            include=["metadatas"]
        )

        retrieved_sources = [
            metadata.get("source")
            for metadata in retrieved["metadatas"]
        ]

        precision = precision_at_k(
            retrieved_sources,
            relevant_sources,
            TOP_K
        )

        recall = recall_at_k(
            retrieved_sources,
            relevant_sources,
            TOP_K
        )

        rr = reciprocal_rank(
            retrieved_sources,
            relevant_sources
        )
        precisions.append(precision)
        recalls.append(recall)
        reciprocal_ranks.append(rr)

    return {
        "precision_at_5": sum(precisions) / len(precisions),
        "recall_at_5": sum(recalls) / len(recalls),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks)
    }


def main():
    documents = load_documents()
    evaluation_data = load_evaluation_data()

    if os.path.exists(DB_ROOT):
        shutil.rmtree(DB_ROOT)

    os.makedirs(DB_ROOT)

    benchmark_results = []

    for config in CHUNK_CONFIGS:
        chunk_size = config["chunk_size"]
        overlap = config["overlap"]

        print()
        print("=" * 60)
        print(f"Testing chunk_size={chunk_size}, overlap={overlap}")
        print("=" * 60)

        chunks = build_chunks(
            documents,
            chunk_size=chunk_size,
            overlap=overlap
        )

        print(f"Total chunks: {len(chunks)}")

        collection_name = f"chunks_{chunk_size}_{overlap}"

        collection = create_collection(
            chunks,
            collection_name
        )

        metrics = evaluate_collection(
            collection,
            evaluation_data
        )

        result = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "total_chunks": len(chunks),
            **metrics
        }

        benchmark_results.append(result)

        print(f"Precision@5: {metrics['precision_at_5']:.3f}")
        print(f"Recall@5:    {metrics['recall_at_5']:.3f}")
        print(f"MRR:         {metrics['mrr']:.3f}")

    print()
    print("\nFINAL BENCHMARK")
    print("=" * 80)

    print(
        f"{'Chunk':>8} "
        f"{'Overlap':>8} "
        f"{'Chunks':>8} "
        f"{'P@5':>8} "
        f"{'R@5':>8} "
        f"{'MRR':>8}"
    )

    print("-" * 80)

    for result in benchmark_results:
        print(
            f"{result['chunk_size']:>8} "
            f"{result['overlap']:>8} "
            f"{result['total_chunks']:>8} "
            f"{result['precision_at_5']:>8.3f} "
            f"{result['recall_at_5']:>8.3f} "
            f"{result['mrr']:>8.3f}"
        )


if __name__ == "__main__":
    main()