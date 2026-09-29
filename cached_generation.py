from cache import (
    make_key,
    cache_get,
    cache_set
)

from generation import generate_answer


def cached_generate_answer(
    question,
    documents,
    model_name="qwen2.5vl:latest"
):

    cache_data = {
        "question": question,
        "documents": [
            {
                "id": item["id"],
                "document": item["document"]
            }
            for item in documents
        ],
        "model": model_name
    }

    key = make_key(
        "generation",
        cache_data
    )

    cached = cache_get(key)

    if cached is not None:
        return {
            **cached,
            "cache_hit": True
        }

    answer = generate_answer(
        question,
        documents
    )

    result = {
        "answer": answer
    }

    cache_set(
        key,
        result
    )

    return {
        **result,
        "cache_hit": False
    }