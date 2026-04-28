"""
Prompt engineering for the LLM-based ticket triage classifier.

Design notes:

1. The system prompt pins the *exact* category and urgency label sets.
   This prevents the model from inventing new classes and keeps the
   output directly comparable to the classical model's predictions.

2. Output is constrained to a single JSON object. The schema is shown
   in the prompt and the model is instructed to emit *only* JSON with
   no surrounding prose, which makes parsing trivial and reliable.

3. We include three calibrated few-shot examples spanning low / medium
   / high urgency so the model picks up on the urgency rubric defined
   in `config.py::URGENCY_RULES`.

4. A short reasoning field ("rationale") is requested. It is used by
   the routing dashboard for explainability but does not affect
   classification.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from config import CATEGORIES_FILE, URGENCIES


def _load_categories() -> list[str]:
    if CATEGORIES_FILE.exists():
        return json.loads(CATEGORIES_FILE.read_text())["categories"]
    # Sensible default if data hasn't been loaded yet
    return [
        "ACCOUNT", "CANCEL", "CONTACT", "DELIVERY", "FEEDBACK",
        "INVOICE", "NEWSLETTER", "ORDER", "PAYMENT", "REFUND",
        "SHIPPING_ADDRESS",
    ]


SYSTEM_PROMPT_TEMPLATE = """You are an expert customer-support ticket triage system.
Your job is to read a customer's message and classify it for routing.

You must output exactly one JSON object — no prose, no markdown fences, no commentary.

Schema:
{{
  "category": "<one of: {categories_list}>",
  "urgency":  "<one of: {urgencies_list}>",
  "confidence": <number between 0 and 1>,
  "rationale": "<one short sentence explaining the choice>"
}}

Category definitions:
- ACCOUNT: anything about creating, deleting, editing, switching accounts, password recovery, registration problems
- CANCEL: cancelling an existing order
- CONTACT: how to reach support, contacting human agents
- DELIVERY: delivery options, expected delivery times (NOT tracking a specific order — that's ORDER)
- FEEDBACK: complaints, reviews, general feedback
- INVOICE: viewing, downloading, or requesting invoices
- NEWSLETTER: newsletter subscribe / unsubscribe
- ORDER: placing, changing, tracking specific orders
- PAYMENT: payment methods, payment failures, double charges
- REFUND: refund policy, requesting refunds, tracking refunds
- SHIPPING_ADDRESS: adding or changing a shipping address

Urgency rubric:
- high: refunds, cancellations, account lockouts, complaints, payment failures
- medium: active orders, billing/invoice questions, shipping address changes, delivery questions
- low: general FAQ, subscriptions, contact info, reviews, account edits

If the message is ambiguous, pick the most likely category and lower the confidence.
Always emit valid JSON. Always pick from the allowlists above."""


FEW_SHOT_EXAMPLES = [
    {
        "user": "I want my money back for the broken headphones I received last week!!",
        "assistant": json.dumps({
            "category": "REFUND",
            "urgency": "high",
            "confidence": 0.96,
            "rationale": "Customer is explicitly requesting a refund for a defective product.",
        }),
    },
    {
        "user": "Hi, can you tell me which payment methods you accept at checkout?",
        "assistant": json.dumps({
            "category": "PAYMENT",
            "urgency": "low",
            "confidence": 0.93,
            "rationale": "General informational question about accepted payment methods.",
        }),
    },
    {
        "user": "Where is my order #4421? It was supposed to arrive yesterday.",
        "assistant": json.dumps({
            "category": "ORDER",
            "urgency": "medium",
            "confidence": 0.94,
            "rationale": "Customer is asking to track a specific late order.",
        }),
    },
]


def build_system_prompt(categories: Sequence[str] | None = None) -> str:
    cats = list(categories) if categories else _load_categories()
    return SYSTEM_PROMPT_TEMPLATE.format(
        categories_list=", ".join(cats),
        urgencies_list=", ".join(URGENCIES),
    )


def build_messages(ticket_text: str, categories: Sequence[str] | None = None) -> list[dict]:
    """Return Anthropic-style messages array for one ticket."""
    msgs: list[dict] = []
    for ex in FEW_SHOT_EXAMPLES:
        msgs.append({"role": "user", "content": ex["user"]})
        msgs.append({"role": "assistant", "content": ex["assistant"]})
    msgs.append({"role": "user", "content": ticket_text})
    return msgs
