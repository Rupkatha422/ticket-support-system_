"""
LLM-based ticket triage classifier (Groq -- Llama 3.1 8B Instant).

Two backends behind a common interface:

* `GroqClassifier` -- calls the Groq Chat Completions API. Activated
  automatically when `GROQ_API_KEY` is set in the environment. Uses
  Llama 3.1 8B Instant by default (very fast, very cheap).
* `MockClassifier` -- a deterministic rule-based stand-in that mimics a
  well-prompted LLM. Used when no API key is available so the rest of
  the pipeline is still runnable.

Both classifiers expose:

    .classify(text) -> TriageResult(category, urgency, confidence,
                                    rationale, latency_ms,
                                    input_tokens, output_tokens)

Latency and token usage are tracked so the evaluator can compute
cost / latency tables.

Run as:
    python -m llm.llm_classifier "I want a refund for my broken phone"
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    GROQ_MODEL,
    GROQ_BASE_URL,
    LLM_PRICING_PER_M_TOKENS,
    URGENCIES,
)
from llm.prompts import (  # noqa: E402
    build_system_prompt,
    build_messages,
    _load_categories,
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class TriageResult:
    category: str
    urgency: str
    confidence: float
    rationale: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    backend: str = ""
    raw: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# JSON post-processing -- tolerant to ```json fences and minor noise
# ---------------------------------------------------------------------------
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_json(text: str) -> dict:
    text = text.strip()
    # Strip code fences if the model emitted them anyway
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJECT_RE.search(text)
        if m:
            return json.loads(m.group(0))
        raise


def _normalize(payload: dict, allowed_categories: Sequence[str]) -> dict:
    """Snap LLM outputs to the allowlists; keep raw confidence."""
    cat = str(payload.get("category", "")).strip().upper()
    if cat not in allowed_categories:
        # If the model invented a class, fall back to closest substring match
        for c in allowed_categories:
            if c in cat or cat in c:
                cat = c
                break
        else:
            cat = "FEEDBACK" if "FEEDBACK" in allowed_categories else allowed_categories[0]

    urg = str(payload.get("urgency", "")).strip().lower()
    if urg not in URGENCIES:
        urg = "medium"

    try:
        conf = float(payload.get("confidence", 0.7))
    except (TypeError, ValueError):
        conf = 0.7
    conf = max(0.0, min(1.0, conf))

    rationale = str(payload.get("rationale", ""))[:500]
    return {
        "category": cat, "urgency": urg,
        "confidence": conf, "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Groq backend
# ---------------------------------------------------------------------------
class GroqClassifier:
    backend = "groq"

    def __init__(self, model: str = GROQ_MODEL, api_key: str | None = None):
        from groq import Groq  # type: ignore

        self.model = model
        self.client = Groq(
            api_key=api_key or os.environ.get("GROQ_API_KEY"),
            base_url=GROQ_BASE_URL,
        )
        self.categories = _load_categories()
        self.system_prompt = build_system_prompt(self.categories)

    def _to_openai_messages(self, ticket_text: str) -> list[dict]:
        """Groq uses OpenAI-style messages: system goes in role=system."""
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(build_messages(ticket_text, self.categories))
        return messages

    def classify(self, text: str) -> TriageResult:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self._to_openai_messages(text),
            max_tokens=200,
            temperature=0,
            # Force JSON output -- Groq supports OpenAI-style response_format
            response_format={"type": "json_object"},
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        raw = resp.choices[0].message.content or ""
        payload = _coerce_json(raw)
        norm = _normalize(payload, self.categories)

        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0

        return TriageResult(
            **norm,
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            backend=self.backend,
            raw=raw,
        )


# ---------------------------------------------------------------------------
# Mock backend -- deterministic, illustrative, runs without API keys
# ---------------------------------------------------------------------------
class MockClassifier:
    """Rule-based stand-in for a well-prompted LLM.

    It applies a set of keyword rules and adds a small amount of
    realistic latency (sleep) so the evaluator's latency comparison
    isn't trivially zero. It also reports synthetic token counts so the
    cost extrapolation works.
    """

    backend = "mock"

    PATTERNS = [
        # REFUND
        (r"\b(refund|money back|reimburse)\b",                "REFUND",          "high"),
        # CANCEL
        (r"\b(cancel|cancellation)\b",                        "CANCEL",          "high"),
        # ACCOUNT - high urgency lockouts
        (r"\b(forgot|reset|recover|locked out).*\b(password|account)\b", "ACCOUNT", "high"),
        (r"\b(delete account|close account|remove my account)\b", "ACCOUNT",     "high"),
        (r"\b(complaint|file a complaint|unhappy|terrible|awful|furious)\b",
         "FEEDBACK", "high"),
        (r"\b(payment failed|declined|charged twice|double charge|money was deducted)\b",
         "PAYMENT", "high"),
        # ORDER tracking / changes
        (r"\b(track|where is) my (order|package|shipment)\b",  "ORDER",          "medium"),
        (r"\b(change|modify|update).*\border\b",               "ORDER",          "medium"),
        (r"\b(place|complete|make).*\border\b",                "ORDER",          "medium"),
        # SHIPPING ADDRESS
        (r"\b(shipping|delivery) address\b",                   "SHIPPING_ADDRESS","medium"),
        (r"\b(reroute|change.*address|new address)\b",         "SHIPPING_ADDRESS","medium"),
        # DELIVERY
        (r"\b(delivery (options|time|period)|how long .* deliver|express delivery|same.day)\b",
         "DELIVERY", "medium"),
        # INVOICE
        (r"\b(invoice|receipt|billing statement)\b",           "INVOICE",        "medium"),
        # PAYMENT methods (low/medium)
        (r"\b(payment method|how can i pay|paypal|apple pay|bank transfer)\b",
         "PAYMENT", "low"),
        # ACCOUNT - low urgency edits
        (r"\b(create|sign ?up|register).*\baccount\b",         "ACCOUNT",        "low"),
        (r"\b(edit|update|change).*\b(profile|email|account information|account settings)\b",
         "ACCOUNT", "low"),
        (r"\bswitch.*\baccount\b",                             "ACCOUNT",        "low"),
        # CONTACT
        (r"\b(human (agent|representative)|live (agent|person)|talk to (a )?(human|person|someone))\b",
         "CONTACT", "low"),
        (r"\b(contact|reach|phone number|support email).*\bsupport\b",
         "CONTACT", "low"),
        (r"\bcontact (us|you|customer service)\b",             "CONTACT",        "low"),
        # NEWSLETTER
        (r"\bnewsletter|promotional emails|weekly emails\b",   "NEWSLETTER",     "low"),
        # FEEDBACK / review
        (r"\b(review|leave.*feedback|share.*feedback)\b",      "FEEDBACK",       "low"),
        # REFUND policy (low)
        (r"\b(refund policy|return policy|money.back guarantee)\b", "REFUND",    "low"),
    ]

    # Groq Llama 3.1 8B is unusually fast (~150-300ms). We pick a value
    # that makes the latency comparison interesting but not unrealistic.
    SIMULATED_LATENCY_MS = 200

    def __init__(self):
        self.categories = _load_categories()

    def classify(self, text: str) -> TriageResult:
        t0 = time.perf_counter()
        time.sleep(self.SIMULATED_LATENCY_MS / 1000.0)

        low = text.lower()
        for pattern, category, urgency in self.PATTERNS:
            if re.search(pattern, low):
                conf = 0.88
                rationale = f"Matched pattern indicating {category.lower()} with {urgency} urgency."
                break
        else:
            category, urgency, conf = "FEEDBACK", "low", 0.40
            rationale = "No strong pattern matched; defaulting to general feedback."

        if category not in self.categories:
            category = self.categories[0]

        latency_ms = (time.perf_counter() - t0) * 1000
        # Synthetic token counts roughly matching a Groq Llama 3.1 8B call
        input_tokens = max(50, int(len(text.split()) * 1.3) + 350)  # +system prompt
        output_tokens = 60

        return TriageResult(
            category=category,
            urgency=urgency,
            confidence=conf,
            rationale=rationale,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            backend=self.backend,
            raw=json.dumps({
                "category": category, "urgency": urgency,
                "confidence": conf, "rationale": rationale,
            }),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_classifier() -> "GroqClassifier | MockClassifier":
    """Pick the Groq backend if an API key is present, else the mock."""
    if os.environ.get("GROQ_API_KEY"):
        try:
            return GroqClassifier()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[llm] Groq backend unavailable ({exc}); falling back to mock.",
                file=sys.stderr,
            )
    return MockClassifier()


def estimate_cost_usd(input_tokens: int, output_tokens: int, model: str = GROQ_MODEL) -> float:
    pricing = LLM_PRICING_PER_M_TOKENS.get(model)
    if not pricing:
        return 0.0
    return (
        input_tokens / 1_000_000 * pricing["input"]
        + output_tokens / 1_000_000 * pricing["output"]
    )


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "I'd like to cancel my order, the shipping is taking too long!"
    clf = get_classifier()
    res = clf.classify(text)
    print(json.dumps(res.to_dict(), indent=2))
