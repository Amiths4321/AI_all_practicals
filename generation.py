import requests


OLLAMA_URL = "http://10.22.39.192:11434"
MODEL_NAME = "qwen2.5vl:latest"


def build_context(documents):
    context_parts = []

    for index, item in enumerate(documents, start=1):
        source = item["metadata"].get(
            "source",
            "unknown"
        )

        context_parts.append(
            f"[SOURCE {index}: {source}]\n"
            f"{item['document']}"
        )

    return "\n\n".join(context_parts)


def generate_answer(question, documents):
    context = build_context(documents)

    prompt = f"""
You are a retrieval-grounded assistant.

Answer the question using ONLY the supplied context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the context does not contain enough evidence, say:
   "I don't have enough information in the provided documents."
4. Keep the answer concise.
5. Cite the source names you used.

Question:
{question}

Context:
{context}

Answer:
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

    return response.json()["response"]