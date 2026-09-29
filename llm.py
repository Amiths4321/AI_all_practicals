import ollama

MODEL_NAME = "llama3.2"

from prompt import build_rag_prompt


def generate_answer(query, evidence):

    prompt = build_rag_prompt(
        query,
        evidence
    )

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]