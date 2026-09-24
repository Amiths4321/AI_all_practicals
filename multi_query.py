import json
import re
import requests


OLLAMA_URL = "http://10.22.39.192:11434"
MODEL_NAME = "qwen2.5vl:latest"


def generate_queries(question, num_queries=3):
    prompt = f"""
Generate {num_queries} different search queries for the
following question.

Rules:
1. Preserve the original meaning.
2. Each query should use somewhat different wording.
3. Keep important policy names, entities, and terminology.
4. Do not answer the question.
5. Do not invent facts.
6. Return ONLY a JSON array of strings.

Question:
{question}

Example format:
[
    "query one",
    "query two",
    "query three"
]
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=3600
    )

    response.raise_for_status()

    raw = response.json()["response"].strip()

    # Remove optional markdown fences.
    raw = re.sub(
        r"^```json\s*",
        "",
        raw,
        flags=re.IGNORECASE
    )

    raw = re.sub(
        r"\s*```$",
        "",
        raw
    )

    start = raw.find("[")
    end = raw.rfind("]")

    if start == -1 or end == -1:
        raise ValueError(
            "Model did not return a JSON array."
        )

    queries = json.loads(
        raw[start:end + 1]
    )

    if not isinstance(queries, list):
        raise ValueError(
            "Generated queries are not a list."
        )

    queries = [
        query.strip()
        for query in queries
        if isinstance(query, str)
        and query.strip()
    ]

    if not queries:
        raise ValueError(
            "No valid queries were generated."
        )

    return queries[:num_queries]


if __name__ == "__main__":

    question = input("Question: ")

    queries = generate_queries(question)

    print("\nGenerated queries:")

    for index, query in enumerate(
        queries,
        start=1
    ):
        print(f"{index}. {query}")