import json
import re
import requests


OLLAMA_URL = "http://10.22.39.192:11434"
MODEL_NAME = "qwen2.5vl:latest"


def build_pairwise_prompt(question, context, answer_a, answer_b):
    return f"""
You are evaluating two answers produced by RAG systems.

Evaluate ONLY against the supplied context.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Compare the two answers on:

1. groundedness
   Which answer is better supported by the supplied context?

2. relevance
   Which answer addresses the question more directly?

3. completeness
   Which answer contains more of the important information supported
   by the context?

4. unsupported_claims
   Which answer contains fewer claims that are not supported by
   the context?

Choose the overall better answer.

Return ONLY valid JSON.

Required format:

{{
    "winner": "A",
    "groundedness_winner": "A",
    "relevance_winner": "A",
    "completeness_winner": "A",
    "unsupported_claims_winner": "A",
    "reason": "brief explanation"
}}

Valid winner values are:

"A"
"B"
"TIE"
"""


def extract_json(text):
    text = text.strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("Judge did not return JSON.")

    return json.loads(text[start:end + 1])


def validate_winner(value):
    if value not in ["A", "B", "TIE"]:
        raise ValueError(
            f"Invalid winner: {value}"
        )


def validate_judgement(judgement):
    required_fields = [
        "winner",
        "groundedness_winner",
        "relevance_winner",
        "completeness_winner",
        "unsupported_claims_winner",
        "reason"
    ]

    for field in required_fields:
        if field not in judgement:
            raise ValueError(
                f"Missing judge field: {field}"
            )

    winner_fields = [
        "winner",
        "groundedness_winner",
        "relevance_winner",
        "completeness_winner",
        "unsupported_claims_winner"
    ]

    for field in winner_fields:
        validate_winner(judgement[field])

    if not isinstance(judgement["reason"], str):
        raise ValueError("reason must be a string.")

    return judgement


def pairwise_judge(question, context, answer_a, answer_b):

    prompt = build_pairwise_prompt(
        question=question,
        context=context,
        answer_a=answer_a,
        answer_b=answer_b
    )

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

    raw_output = response.json()["response"]

    judgement = extract_json(raw_output)

    return validate_judgement(judgement)