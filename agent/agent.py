"""The support agent: decides what to do with each customer message.

Flow for every message:
  1. Guardrails: redact personal details, block prompt-injection attempts.
  2. Route: classify the intent (order status, refund, policy question, ...).
  3. Act: call a tool (order lookup / refund) or answer from the help centre.
  4. Check confidence: if the agent isn't sure, hand over to a human instead
     of guessing. Wrong answers cost more than a hand-off.
"""
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .guardrails import is_prompt_injection, redact_pii
from .llm import generate_answer
from .retriever import Retriever
from .router import IntentRouter
from .tools import ORDER_ID, OrderTools, extract_order_id

DATA = Path(__file__).resolve().parent.parent / "data"

INTENT_THRESHOLD = 0.35     # below this, the router's guess isn't trusted
RETRIEVAL_THRESHOLD = 0.12  # below this, the help centre doesn't cover the question


@dataclass
class Response:
    reply: str
    intent: str
    action: str                  # what the agent did, e.g. "lookup_order", "escalate"
    escalated: bool = False
    sources: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)  # e.g. "pii_card", "prompt_injection"
    confidence: float = 0.0


class SupportAgent:
    def __init__(self, data_dir: Path = DATA, today: date | None = None):
        self.retriever = Retriever(data_dir / "kb")
        self.router = IntentRouter(data_dir / "intents.csv")
        self.tools = OrderTools(data_dir / "orders.json", today=today)
        # The models above are loaded once and shared by every conversation.
        # The only per-conversation state is "what are we waiting for", kept
        # per session id behind a lock so concurrent users can't see each
        # other's state.
        self._pending: dict[str, str] = {}
        self._lock = threading.Lock()

    def _get_pending(self, session_id: str) -> str | None:
        with self._lock:
            return self._pending.get(session_id)

    def _set_pending(self, session_id: str, intent: str | None) -> None:
        with self._lock:
            if intent is None:
                self._pending.pop(session_id, None)
            else:
                self._pending[session_id] = intent

    def handle(self, message: str, session_id: str = "default") -> Response:
        text, pii = redact_pii(message)
        flags = [f"pii_{p}" for p in pii]
        prefix = ""
        if "card" in pii:
            prefix = ("For your security, please don't share card numbers in chat. "
                      "I've removed it from this conversation. ")

        if is_prompt_injection(text):
            return Response(
                reply=("I can only help with Koala Gear orders, returns and store policies, "
                       "and I can't change those rules. What can I help you with?"),
                intent="blocked", action="refuse", flags=flags + ["prompt_injection"])

        order_id = extract_order_id(text)
        intent, conf = self.router.predict(text)

        # Customer replied with (mostly) just an order number after we asked
        # for one, e.g. "KG1001" or "it's KG1001": continue what they asked for.
        pending = self._get_pending(session_id)
        if pending and order_id:
            leftover = ORDER_ID.sub("", text).split()
            if len(leftover) <= 3 or conf < 0.6:
                intent, conf = pending, 1.0

        if conf < INTENT_THRESHOLD:
            # Unsure. If there's an order number, a status lookup is the safest
            # action (it can't change anything). Otherwise try the help centre,
            # which has its own confidence check.
            intent = "order_status" if order_id else "policy_question"

        resp = self._act(intent, text, order_id, session_id)
        resp.reply = prefix + resp.reply
        resp.flags = flags + resp.flags
        resp.confidence = round(conf, 3)
        return resp

    def _act(self, intent: str, text: str, order_id: str | None, session_id: str) -> Response:
        if intent in ("order_status", "refund_request"):
            if not order_id:
                self._set_pending(session_id, intent)
                return Response("Sure, can you give me your order number? It starts with KG, "
                                "for example KG1234.", intent, "ask_order_id")
            self._set_pending(session_id, None)
            if intent == "order_status":
                result = self.tools.get_order_status(order_id)
                return Response(result["message"], intent, "lookup_order")
            result = self.tools.request_refund(order_id)
            if result["decision"] == "needs_human":
                return Response(result["message"], intent, "escalate", escalated=True)
            return Response(result["message"], intent, f"refund_{result['decision']}")

        if intent == "human_request":
            return self._escalate(session_id, intent, "No problem, I'm connecting you with a team member now.")

        if intent == "greeting":
            return Response("Hi! I can help with orders, returns, shipping and other store "
                            "questions. What do you need?", intent, "small_talk")

        if intent == "out_of_scope":
            return Response("Sorry, I can only help with Koala Gear orders and store questions.",
                            intent, "decline")

        # policy_question: answer from the help centre, or hand over if not covered.
        hits = self.retriever.search(text, k=3)
        if hits[0].score < RETRIEVAL_THRESHOLD:
            return self._escalate(session_id, intent, "I'm not sure about that one, so I've passed your "
                                  "question to a team member who'll reply by email.")
        # Keep passages that score close to the best one; in offline mode the
        # answer is these passages themselves.
        top = hits[0].score
        passages = [h.chunk.text for h in hits if h.score >= 0.85 * top]
        answer = generate_answer(text, passages)
        if answer is None:
            return self._escalate(session_id, intent, "I couldn't find that in our help centre, so I've "
                                  "passed your question to a team member.")
        sources = list(dict.fromkeys(h.chunk.source for h in hits if h.score >= RETRIEVAL_THRESHOLD))
        return Response(answer, intent, "answer_from_kb", sources=sources)

    def _escalate(self, session_id: str, intent: str, reply: str) -> Response:
        self._set_pending(session_id, None)
        return Response(reply, intent, "escalate", escalated=True)
