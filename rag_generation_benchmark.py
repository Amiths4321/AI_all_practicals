import json
import os

# Disable Chroma telemetry before importing/creating Chroma clients.
os.environ["ANONYMIZED_TELEMETRY"] = "false"

import chromadb
from chromadb.config import Settings

from chunking import chunk_text
from hybrid_search import HybridRetriever
from reranking import DocumentReranker
from generation import generate_answer
from rag_judge import judge_answer


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 500
OVERLAP = 100

RETRIEVAL_K = 10
RERANK_K = 5

DB_PATH = "./generation_benchmark_chroma"


# ============================================================
# LOAD JSON
# ============================================================

def load_json(filename):
    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


# ============================================================
# BUILD CHUNKS
# ============================================================

def build_chunks(documents):
    chunks = []

    for document in documents:
        source = document["source"]

        chunks_for_document = chunk_text(
            document["text"],
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP
        )

        for index, chunk in enumerate(
            chunks_for_document
        ):
            chunks.append({
                "id": f"{source}_chunk_{index}",
                "document": chunk,
                "metadata": {
                    "source": source,
                    "chunk_index": index
                }
            })

    return chunks


# ============================================================
# CREATE CHROMA COLLECTION
# ============================================================

def create_collection(chunks):
    client = chromadb.PersistentClient(
        path=DB_PATH,
        settings=Settings(
            anonymized_telemetry=False
        )
    )

    # Delete the previous benchmark collection.
    try:
        client.delete_collection(
            name="documents"
        )
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="documents"
    )

    if chunks:
        collection.add(
            ids=[
                item["id"]
                for item in chunks
            ],
            documents=[
                item["document"]
                for item in chunks
            ],
            metadatas=[
                item["metadata"]
                for item in chunks
            ]
        )

    return collection


# ============================================================
# VECTOR SEARCH
# ============================================================

def vector_search(
    collection,
    question,
    top_k
):
    available = collection.count()

    if available == 0:
        return []

    actual_k = min(
        top_k,
        available
    )

    results = collection.query(
        query_texts=[question],
        n_results=actual_k
    )

    output = []

    for doc_id, document, metadata in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0]
    ):
        output.append({
            "id": doc_id,
            "document": document,
            "metadata": metadata or {}
        })

    return output

def build_context(documents):
    parts = []

    for document in documents:
        source = document["metadata"].get(
            "source",
            "unknown"
        )

        parts.append(
            f"[SOURCE: {source}]\n"
            f"{document['document']}"
        )

    return "\n\n".join(parts)
# ============================================================
# HYBRID + RERANKING
# ============================================================

def retrieve_documents(
    question,
    collection,
    hybrid_retriever,
    reranker
):
    vector_results = vector_search(
        collection,
        question,
        RETRIEVAL_K
    )

    if not vector_results:
        return []

    hybrid_results = hybrid_retriever.search(
        question=question,
        vector_results=vector_results,
        top_k=RETRIEVAL_K
    )

    if not hybrid_results:
        return []

    reranked = reranker.rerank(
        question=question,
        documents=hybrid_results,
        top_n=RERANK_K
    )

    return reranked


# ============================================================
# EVIDENCE CHECK
# ============================================================

def source_overlap(
    retrieved_documents,
    relevant_sources
):
    retrieved_sources = {
        item["metadata"].get("source")
        for item in retrieved_documents
    }

    relevant_sources = set(
        relevant_sources
    )

    overlap = retrieved_sources.intersection(
        relevant_sources
    )

    print()
    print("Evidence check:")
    print("  Retrieved:", retrieved_sources)
    print("  Expected: ", relevant_sources)
    print("  Matching: ", overlap)

    return bool(overlap)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    documents = load_json(
        "documents.json"
    )

    evaluation_data = load_json(
        "generation_eval.json"
    )

    # --------------------------------------------------------
    # Build chunks
    # --------------------------------------------------------

    chunks = build_chunks(
        documents
    )

    print(
        f"Total documents: {len(documents)}"
    )

    print(
        f"Total chunks: {len(chunks)}"
    )

    for chunk in chunks:
        print(
            chunk["id"],
            "=>",
            chunk["metadata"]["source"]
        )

    print()

    # --------------------------------------------------------
    # Create Chroma collection
    # --------------------------------------------------------

    collection = create_collection(
        chunks
    )

    print(
        f"Chroma collection contains "
        f"{collection.count()} chunks."
    )

    print()

    # --------------------------------------------------------
    # Initialize retrieval components
    # --------------------------------------------------------

    hybrid_retriever = HybridRetriever(
        chunks
    )

    reranker = DocumentReranker()

    # --------------------------------------------------------
    # Evaluate questions
    # --------------------------------------------------------

    results = []

    for index, item in enumerate(
        evaluation_data,
        start=1
    ):
        question = item["question"]

        print("=" * 80)
        print(
            f"QUESTION {index}: {question}"
        )
        print("=" * 80)

        # -----------------------------------------------
        # Retrieve + rerank
        # -----------------------------------------------

        retrieved = retrieve_documents(
            question,
            collection,
            hybrid_retriever,
            reranker
        )

        # -----------------------------------------------
        # Debug retrieval
        # -----------------------------------------------

        print()
        print("EXPECTED SOURCES:")
        print(
            item["relevant_sources"]
        )

        print()
        print("RETRIEVED SOURCES:")
        print([
            document["metadata"].get("source")
            for document in retrieved
        ])

        # -----------------------------------------------
        # Evidence gate
        # -----------------------------------------------

        has_evidence = source_overlap(
            retrieved,
            item["relevant_sources"]
        )

        # -----------------------------------------------
        # Generate answer
        # -----------------------------------------------

        if not has_evidence:
            answer = (
                "I don't have enough information "
                "in the provided documents."
            )
        else:
            answer = generate_answer(
                question,
                retrieved
            )
          context = build_context(
          retrieved
          )

          judgement = judge_answer(
                    question,
                    context,
                    answer
          )

          print()
          print("JUDGE:")
          print(
                    json.dumps(
                    judgement,
                    indent=2
          )
          )

          results.append({
                    "question": question,

                    "reference_answer":
                    item["answer"],

                    "generated_answer":
                    answer,

                    "retrieved_sources": [
                    document["metadata"].get(
                              "source"
                    )
                    for document in retrieved
                    ],

                    "judge": judgement
                    })

        # -----------------------------------------------
        # Display answer
        # -----------------------------------------------

        print()
        print("ANSWER:")
        print(answer)

        print()
        print("SOURCES:")

        for document in retrieved:
            print(
                "-",
                document["metadata"].get(
                    "source"
                )
            )

        # -----------------------------------------------
        # Save result
        # -----------------------------------------------

        results.append({
            "question": question,
            "reference_answer": item["answer"],
            "generated_answer": answer,
            "retrieved_sources": [
                document["metadata"].get(
                    "source"
                )
                for document in retrieved
            ]
        })

        print()
        judged_results = [
          item
          for item in results
          if item["judge"] is not None
        ]
        
        if judged_results:

          average_groundedness = (
          sum(
                    item["judge"]["groundedness"]
                    for item in judged_results
          )
          / len(judged_results)
          )

          average_relevance = (
          sum(
                    item["judge"]["relevance"]
                    for item in judged_results
          )
          / len(judged_results)
          )

          average_completeness = (
          sum(
                    item["judge"]["completeness"]
                    for item in judged_results
          )
          / len(judged_results)
          )

          average_unsupported = (
          sum(
                    item["judge"]["unsupported_claims"]
                    for item in judged_results
          )
          / len(judged_results)
          )

          else:

          average_groundedness = 0.0
          average_relevance = 0.0
          average_completeness = 0.0
          average_unsupported = 0.0

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    with open(
        "generation_results.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        "Saved generation_results.json"
    )

def detect_disagreement(
    evidence_recall,
    judgement
):
    if (
        evidence_recall < 1.0
        and judgement["groundedness"] == 2
    ):
        return True

    if (
        evidence_recall == 1.0
        and judgement["groundedness"] == 0
    ):
        return True

    return False

    disagreement = detect_disagreement(
          semantic_evidence_recall,
          judgement
          )       
    "judge_disagreement": disagreement   
# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()