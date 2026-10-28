"""Input and output safety checks.

- redact_pii: masks card numbers, emails and phone numbers before anything is
  logged or sent to a language model.
- is_prompt_injection: flags messages that try to override the bot's rules.
"""
import re

CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"(?:\+?61|0)[2-478](?:[ -]?\d){8}\b")

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"disregard (your|the) (rules|instructions|policy)",
    r"you are now",
    r"system prompt",
    r"pretend (to be|you are)",
    r"act as (an? )?(admin|developer|manager)",
    r"(approve|issue) (a |the )?refund (anyway|regardless|without)",
]
INJECTION = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Return the redacted text and the kinds of PII that were found."""
    found = []
    for name, pattern, mask in [
        ("card", CARD, "[CARD]"),
        ("email", EMAIL, "[EMAIL]"),
        ("phone", PHONE, "[PHONE]"),
    ]:
        if pattern.search(text):
            found.append(name)
            text = pattern.sub(mask, text)
    return text, found


def is_prompt_injection(text: str) -> bool:
    return bool(INJECTION.search(text))
