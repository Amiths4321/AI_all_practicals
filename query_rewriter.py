import ollama

MODEL_NAME = "llama3.2"


def rewrite_query(
    query,
    product,
    conversation_history=None
):
    # No history → current query is already standalone
    if not conversation_history:
        return query

    history_text = ""

    for item in conversation_history:
        history_text += (
            f"User: {item['question']}\n"
            f"Assistant: {item['answer']}\n"
        )

    prompt = f"""
You are a search query rewriting assistant for a banking RAG system.

Your task is to rewrite the user's current question into ONE
standalone search query suitable for policy retrieval.

Loan Product:
{product}

Previous Conversation:
{history_text}

Current Question:
{query}

Rules:
1. Resolve vague references such as "it", "this", "that", "its", "their".
2. Use the previous conversation when necessary.
3. Preserve the user's original intent.
4. Do not answer the question.
5. Do not add information that is not present in the conversation.
6. Include the loan product when useful.
7. Return ONLY the rewritten search query.
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    rewritten_query = response["message"]["content"].strip()

    return rewritten_query