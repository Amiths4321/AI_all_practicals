def precision_at_k(
    retrieved_ids,
    relevant_ids,
    k
):

    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    relevant_count = sum(
        1
        for doc_id in retrieved
        if doc_id in relevant_ids
    )

    return relevant_count / len(retrieved)


def recall_at_k(
    retrieved_ids,
    relevant_ids,
    k
):

    if not relevant_ids:
        return 1.0

    retrieved = retrieved_ids[:k]

    relevant_count = sum(
        1
        for doc_id in retrieved
        if doc_id in relevant_ids
    )

    return (
        relevant_count
        / len(relevant_ids)
    )


def f1_score(
    precision,
    recall
):

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


def reciprocal_rank(
    retrieved_ids,
    relevant_ids
):

    for rank, doc_id in enumerate(
        retrieved_ids,
        start=1
    ):

        if doc_id in relevant_ids:

            return 1 / rank

    return 0.0


def mean_reciprocal_rank(
    reciprocal_ranks
):

    if not reciprocal_ranks:
        return 0.0

    return sum(
        reciprocal_ranks
    ) / len(
        reciprocal_ranks
    )

def evaluate_retrieval(
    retrieved_ids,
    relevant_ids,
    k=5
):

    precision = precision_at_k(
        retrieved_ids,
        relevant_ids,
        k
    )

    recall = recall_at_k(
        retrieved_ids,
        relevant_ids,
        k
    )

    f1 = f1_score(
        precision,
        recall
    )

    rr = reciprocal_rank(
        retrieved_ids,
        relevant_ids
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "reciprocal_rank": rr
    }

def evaluate_ranked_results(
    retrieved_ids,
    relevant_ids,
    k=3
):

    precision = precision_at_k(
        retrieved_ids,
        relevant_ids,
        k
    )

    recall = recall_at_k(
        retrieved_ids,
        relevant_ids,
        k
    )

    f1 = f1_score(
        precision,
        recall
    )

    rr = reciprocal_rank(
        retrieved_ids,
        relevant_ids
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": rr
    }

