import requests


OLLAMA_URL = "http://10.22.39.192:11434"
MODEL_NAME = "qwen2.5vl:latest"


def rewrite_query(question):
    prompt = f"""
Rewrite the following user question into a concise
retrieval query.

Rules:
1. Preserve the original meaning.
2. Keep important names, policies, entities, and terminology.
3. Do not answer the question.
4. Do not invent information.
5. Return ONLY the rewritten query.

Original question:
{question}

Rewritten query:
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

    return response.json()["response"].strip()


if __name__ == "__main__":

    question = input("Question: ")

    rewritten = rewrite_query(question)

    print("\nOriginal:")
    print(question)

    print("\nRewritten:")
    print(rewritten)