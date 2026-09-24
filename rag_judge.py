import json
import re
import requests


OLLAMA_URL = "http://10.22.39.192:11434"
MODEL_NAME = "qwen2.5vl:latest"


def build_judge_prompt(
    question,
    context,
    answer
):
    return f"""
You are evaluating a retrieval-augmented generation answer.

Evaluate ONLY against the supplied context.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Evaluate these dimensions:

1. groundedness:
   Are the factual claims in the answer supported by the context?

2. relevance:
   Does the answer directly address the question?

3. completeness:
   Does the answer include the important information needed
   to answer the question based on the context?

4. unsupported_claims:
   Does the answer contain factual claims that cannot be supported
   by the context?

Use integer scores from 0 to 2:

0 = poor
1 = partial
2 = strong

Return ONLY valid JSON.

Required format:

{{
    "groundedness": 0,
    "relevance": 0,
    "completeness": 0,
    "unsupported_claims": 0,
    "reason": "brief explanation"
}}
"""


def extract_json(text):
    text = text.strip()

    # Remove markdown code fences if the model adds them.
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
        raise ValueError(
            "Judge did not return JSON."
        )

    return json.loads(
        text[start:end + 1]
    )


def validate_judgement(judgement):
    required_fields = [
        "groundedness",
        "relevance",
        "completeness",
        "unsupported_claims",
        "reason"
    ]

    for field in required_fields:
        if field not in judgement:
            raise ValueError(
                f"Missing judge field: {field}"
            )

    for field in [
        "groundedness",
        "relevance",
        "completeness",
        "unsupported_claims"
    ]:
        value = judgement[field]

        if not isinstance(value, int):
            raise ValueError(
                f"{field} must be an integer."
            )

        if value not in [0, 1, 2]:
            raise ValueError(
                f"{field} must be 0, 1, or 2."
            )

    if not isinstance(
        judgement["reason"],
        str
    ):
        raise ValueError(
            "reason must be a string."
        )

    return judgement


def judge_answer(
    question,
    context,
    answer
):
    prompt = build_judge_prompt(
        question,
        context,
        answer
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

    judgement = extract_json(
        raw_output
    )

    return validate_judgement(
        judgement
    )