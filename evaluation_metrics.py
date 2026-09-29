# evaluation_metrics.py

def calculate_precision_at_k(retrieved_ids, relevant_ids, k=3):
    """Calculates Precision@K."""
    if not retrieved_ids or k == 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / min(k, len(top_k))


def calculate_recall_at_k(retrieved_ids, relevant_ids, k=3):
    """Calculates Recall@K."""
    if not relevant_ids:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(relevant_set)