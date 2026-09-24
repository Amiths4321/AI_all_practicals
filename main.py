from ingestion import create_chroma_collection
from retrieval import retrieve_documents
from reranking import rerank_documents
from generation import generate_answer
from hybrid_search import hybrid_search
from reranking import DocumentReranker

# --------------------------------------------------
# 1. BUILD / LOAD KNOWLEDGE BASE
# --------------------------------------------------

chroma_client, collection = create_chroma_collection()


# --------------------------------------------------
# 2. QUESTION
# --------------------------------------------------

question = "Who approves annual leave?"


# --------------------------------------------------
# 3. RETRIEVE
# --------------------------------------------------

retrieval = retrieve_documents(
    collection=collection,
    question=question,
    n_results=5
)

ids = retrieval["ids"]
documents = retrieval["documents"]
metadatas = retrieval["metadatas"]


candidates = []

for document_id, document in zip(
    ids,
    documents
):

    candidates.append(
        {
            "id": document_id,
            "document": document
        }
    )


vector_results = []

for document_id, document in zip(
    ids,
    documents
):

    vector_results.append(
        {
            "id": document_id,
            "document": document
        }
    )


hybrid_results = hybrid_search(
    question=question,
    vector_results=vector_results,
    candidates=candidates,
    top_k=5
)

print("\nHYBRID RESULTS:")

for item in hybrid_results:

    print(
        f"\nHybrid score: "
        f"{item['hybrid_score']:.6f}"
    )

    print(
        f"Vector rank: "
        f"{item['vector_rank']}"
    )

    print(
        f"Keyword rank: "
        f"{item['keyword_rank']}"
    )

    print(
        item["document"]
    )

# --------------------------------------------------
# 4. RERANK
# --------------------------------------------------

reranked = rerank_documents(
    question=question,
    documents=documents,
    metadatas=metadatas,
    top_n=2
)

confidence = calculate_confidence(
    reranked
)

print(
    "\nTOP SCORE:",
    confidence["confidence"]
)

print(
    "SCORE MARGIN:",
    confidence["margin"]
)


# --------------------------------------------------
# 5. GENERATE ANSWER
# --------------------------------------------------

answer = generate_answer(
    question=question,
    context=reranked
)


# --------------------------------------------------
# 6. DISPLAY QUESTION
# --------------------------------------------------

print("\n" + "=" * 60)

print("QUESTION:")
print(question)


# --------------------------------------------------
# 7. DISPLAY RETRIEVED DOCUMENTS
# --------------------------------------------------

print("\nRETRIEVED DOCUMENTS:")

for document, metadata in zip(
    documents,
    metadatas
):

    if metadata is None:
        metadata = {}

    source = metadata.get(
        "source",
        "unknown"
    )

    page = metadata.get(
        "page",
        "N/A"
    )

    chunk_index = metadata.get(
        "chunk_index",
        "unknown"
    )

    print(
        f"\nSource: {source}"
    )

    print(
        f"Page: {page}"
    )

    print(
        f"Chunk: {chunk_index}"
    )

    print(document)


# --------------------------------------------------
# 8. DISPLAY RERANKED DOCUMENTS
# --------------------------------------------------

print("\nRERANKED:")

for item in reranked:

    metadata = item.get(
        "metadata",
        {}
    )

    if metadata is None:
        metadata = {}

    source = metadata.get(
        "source",
        "unknown"
    )

    page = metadata.get(
        "page"
    )

    print(
        f"\nScore: {item['score']:.4f}"
    )

    if page is not None:

        print(
            f"Source: {source} — Page {page}"
        )

    else:

        print(
            f"Source: {source}"
        )

    print(
        item["document"]
    )


# --------------------------------------------------
# 9. DISPLAY ANSWER
# --------------------------------------------------

print("\nANSWER:")
print(answer)


# --------------------------------------------------
# 10. DISPLAY SOURCES
# --------------------------------------------------

print("\nSOURCES:")

seen_sources = set()

source_number = 1

for item in reranked:

    metadata = item.get(
        "metadata",
        {}
    )

    if metadata is None:
        metadata = {}

    source = metadata.get(
        "source",
        "unknown"
    )

    page = metadata.get(
        "page"
    )

    if page is not None:

        source_text = (
            f"{source} — Page {page}"
        )

    else:

        source_text = source

    if source_text in seen_sources:
        continue

    seen_sources.add(
        source_text
    )

    print(
        f"{source_number}. {source_text}"
    )

    source_number += 1

from ingestion import chunk_text


sample_text = """
Annual Leave Policy

Employees can apply for leave through the HR portal.

Annual leave requires manager approval.

Employees must inform their manager before taking leave.

Medical Leave Policy

Medical leave may require supporting documents.
"""


chunks = chunk_text(
    sample_text,
    chunk_size=20,
    overlap=5
)


print("\n" + "=" * 60)

print("CHUNKING TEST")

for index, chunk in enumerate(
    chunks,
    start=1
):

    print(
        f"\nCHUNK {index}:"
    )

    print(chunk)

all_data = collection.get(
    include=[
        "documents",
        "metadatas"
    ]
)

all_documents = []

for document_id, document, metadata in zip(
    all_data["ids"],
    all_data["documents"],
    all_data["metadatas"]
):

    all_documents.append(
        {
            "id": document_id,
            "document": document,
            "metadata": metadata or {}
        }
    )

from hybrid_search import HybridRetriever


hybrid_retriever = HybridRetriever(
    all_documents
)

vector_results = []

for document_id, document, metadata in zip(
    retrieval["ids"],
    retrieval["documents"],
    retrieval["metadatas"]
):

    vector_results.append(
        {
            "id": document_id,
            "document": document,
            "metadata": metadata or {}
        }
    )

hybrid_results = hybrid_retriever.search(
    question=question,
    vector_results=vector_results,
    top_k=10
)

print("\n" + "=" * 60)

print("HYBRID RETRIEVAL")

for rank, item in enumerate(
    hybrid_results,
    start=1
):

    print(
        f"\nRank: {rank}"
    )

    print(
        f"ID: {item['id']}"
    )

    print(
        f"RRF score: "
        f"{item['rrf_score']:.6f}"
    )

    print(
        f"Vector rank: "
        f"{item['vector_rank']}"
    )

    print(
        f"Keyword rank: "
        f"{item['keyword_rank']}"
    )

    print(
        item["document"]
    )

reranker = DocumentReranker()

reranked = reranker.rerank(
    question=question,
    documents=hybrid_results,
    top_n=3
)

relevant_documents = filter_by_relevance(
    reranked,
    threshold=0.0
)

if not relevant_documents:

    answer = (
        "I don't have enough information."
    )

else:

    answer = generate_answer(
        question=question,
        context=relevant_documents
    )


print("\n" + "=" * 60)

print("RERANKED RESULTS")

for rank, item in enumerate(
    reranked,
    start=1
):

    print(
        f"\nRank: {rank}"
    )

    print(
        f"ID: {item['id']}"
    )

    print(
        f"RRF score: "
        f"{item['rrf_score']:.6f}"
    )

    print(
        f"Reranker score: "
        f"{item['rerank_score']:.4f}"
    )

    print(
        item["document"]
    )

answer = generate_answer(
    question=question,
    context=reranked
)

def calculate_confidence(
    reranked_documents
):

    if not reranked_documents:

        return {
            "confidence": 0.0,
            "margin": 0.0
        }

    top_score = (
        reranked_documents[0][
            "rerank_score"
        ]
    )

    if len(reranked_documents) == 1:

        margin = top_score

    else:

        second_score = (
            reranked_documents[1][
                "rerank_score"
            ]
        )

        margin = (
            top_score
            - second_score
        )

    return {
        "confidence": top_score,
        "margin": margin
    }

def has_sufficient_evidence(
    reranked_documents,
    minimum_score=0.0,
    minimum_margin=0.0
):

    if not reranked_documents:
        return False

    top_score = (
        reranked_documents[0][
            "rerank_score"
        ]
    )

    if top_score < minimum_score:
        return False

    if len(reranked_documents) >= 2:

        second_score = (
            reranked_documents[1][
                "rerank_score"
            ]
        )

        margin = (
            top_score
            - second_score
        )

        if margin < minimum_margin:
            return False

    return True

if has_sufficient_evidence(
    reranked_documents=reranked,
    minimum_score=0.0,
    minimum_margin=0.0
):

    answer = generate_answer(
        question=question,
        context=reranked
    )

else:

    answer = (
        "I don't have enough information."
    )