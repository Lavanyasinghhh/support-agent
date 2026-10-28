"""Answer generation.

If OPENAI_API_KEY is set, the answer is written by an LLM that is told to use
ONLY the retrieved help-centre passages. Without a key, the bot runs fully
offline and answers with the best-matching passage itself (extractive mode).
Both modes cite their sources.
"""
import os

import requests

SYSTEM_PROMPT = (
    "You are a customer support assistant for Koala Gear, an Australian outdoor gear store. "
    "Answer the customer's question using ONLY the help-centre passages provided. "
    "If the passages don't contain the answer, reply exactly: NOT_IN_CONTEXT. "
    "Keep answers to 2-3 friendly sentences. Never invent prices, dates or policies."
)


def generate_answer(question: str, passages: list[str]) -> str | None:
    """Returns the answer text, or None if the context doesn't cover it."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return " ".join(passages[:2])  # extractive fallback: quote the best passages

    context = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
            ],
        },
        timeout=30,
    )
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"].strip()
    return None if "NOT_IN_CONTEXT" in answer else answer
