import adaptive_rag


if __name__ == "__main__":
    previous = adaptive_rag.rollback_index()

    print("Rollback successful")
    print(f"Collection: {previous['collection']}")
    print(f"Fingerprint: {previous['fingerprint']}")
    print(f"Chunks: {previous['chunk_count']}")