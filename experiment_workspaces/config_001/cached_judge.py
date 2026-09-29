from cache import (
    make_key,
    cache_get,
    cache_set
)

from rag_judge import judge_answer


def cached_judge_answer(
    question,
    context,
    answer,
    model_name="qwen2.5vl:latest"
):

    cache_data = {
        "question": question,
        "context": context,
        "answer": answer,
        "model": model_name
    }

    key = make_key(
        "judge",
        cache_data
    )

    cached = cache_get(key)

    if cached is not None:
        return {
            **cached,
            "cache_hit": True
        }

    judgement = judge_answer(
        question=question,
        context=context,
        answer=answer
    )

    cache_set(
        key,
        judgement
    )

    return {
        **judgement,
        "cache_hit": False
    }