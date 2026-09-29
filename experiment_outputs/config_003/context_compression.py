def compress_context(
    reranked_documents,
    max_documents=3
):
    if not reranked_documents:
        return []

    return reranked_documents[:max_documents]


def compression_stats(
    original_documents,
    compressed_documents
):
    original_chars = sum(
        len(item["document"])
        for item in original_documents
    )

    compressed_chars = sum(
        len(item["document"])
        for item in compressed_documents
    )

    if original_chars == 0:
        reduction = 0.0
    else:
        reduction = (
            1
            - compressed_chars / original_chars
        )

    return {
        "original_documents": len(
            original_documents
        ),
        "compressed_documents": len(
            compressed_documents
        ),
        "original_chars": original_chars,
        "compressed_chars": compressed_chars,
        "compression_ratio": reduction
    }