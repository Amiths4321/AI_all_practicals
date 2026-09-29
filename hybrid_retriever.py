from retriever import retrieve
from keyword_retriever import keyword_search
from document_loader import load_documents
from rrf import reciprocal_rank_fusion


TOP_K = 5


def make_doc_id(document):
    return (
        f"{document.get('document')}|"
        f"{document.get('section')}|"
        f"{document.get('text')}"
    )


def get_latest_documents(documents, product):

    product_documents = [
        document
        for document in documents
        if document["product"] == product
    ]

    if not product_documents:
        return []

    dates = [
        document.get("effective_date")
        for document in product_documents
        if document.get("effective_date")
    ]

    if not dates:
        return product_documents

    latest_date = max(dates)

    return [
        document
        for document in product_documents
        if document.get("effective_date") == latest_date
    ]


def hybrid_retrieve(query, product):

    # --------------------------------
    # VECTOR SEARCH
    # --------------------------------

    vector_results = retrieve(query, product)

    vector_documents = []

    for result in vector_results:

        payload = result.payload

        document = {
            "document": payload.get("document"),
            "product": payload.get("product"),
            "document_type": payload.get("document_type"),
            "policy_version": payload.get("policy_version"),
            "effective_date": payload.get("effective_date"),
            "section": payload.get("section"),
            "chunk_id": payload.get("chunk_id"),
            "text": payload.get("text")
        }

        vector_documents.append({
            "id": make_doc_id(document),
            "score": result.score,
            "document": document
        })

    # --------------------------------
    # LOAD DOCUMENTS
    # --------------------------------

    all_documents = load_documents()

    # --------------------------------
    # KEEP ONLY LATEST POLICY
    # --------------------------------

    latest_documents = get_latest_documents(
        all_documents,
        product
    )

    print("\n==============================")
    print("HYBRID POLICY FILTER")
    print("==============================")

    print("Product:", product)

    if latest_documents:

        print(
            "Latest Effective Date:",
            latest_documents[0].get("effective_date")
        )

        print(
            "Documents for keyword search:",
            len(latest_documents)
        )

    # --------------------------------
    # KEYWORD SEARCH
    # --------------------------------

    keyword_results = keyword_search(
        query,
        latest_documents,
        top_k=TOP_K
    )

    keyword_documents = []

    for result in keyword_results:

        document = result["document"]

        keyword_documents.append({
            "id": make_doc_id(document),
            "score": result["score"],
            "document": document
        })

    # --------------------------------
    # RRF
    # --------------------------------

    fused_results = reciprocal_rank_fusion(
        vector_documents,
        keyword_documents
    )

    return fused_results[:TOP_K]