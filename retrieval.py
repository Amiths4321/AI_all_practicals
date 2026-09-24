def retrieve_documents(
    collection,
    question,
    n_results=5
):

    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )

    return {
        "ids": results["ids"][0],
        "documents": results["documents"][0]
    }

def retrieve_documents(
    collection,
    question,
    n_results=5
):

    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )

    return {
        "ids": results["ids"][0],
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0]
    }