def build_rag_prompt(query, evidence):

    evidence_text = ""

    for item in evidence:

        evidence_text += f"""
Evidence ID: {item["evidence_id"]}
Document: {item["document"]}
Section: {item["section"]}
Policy Version: {item["policy_version"]}
Effective Date: {item["effective_date"]}

Evidence:
{item["text"]}

--------------------------------
"""

    prompt = f"""
You are a banking policy assistant.

Answer the user's question using ONLY the evidence provided.

STRICT RULES:

1. Do not use outside knowledge.
2. Do not guess.
3. Do not invent policy information.
4. Every factual statement must be supported by the evidence.
5. Cite the Evidence ID that supports the answer.
6. If the evidence is insufficient, say:
   "The available policy information is insufficient to answer this question."
7. Keep the answer concise.
8. Do not create Evidence IDs.

User Question:
{query}

Retrieved Evidence:

{evidence_text}

Return:

Answer:
<answer>

Evidence:
<Evidence ID>
"""

    return prompt