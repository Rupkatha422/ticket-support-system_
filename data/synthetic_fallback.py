"""
Representative-data fallback for environments where Hugging Face is
unreachable (e.g. corporate proxies, sandboxed evaluators).

The generator produces tickets that mirror the Bitext customer-support
schema -- (instruction, category, intent) -- using the official Bitext
category and intent taxonomy. Templates are paraphrased per row to
introduce realistic vocabulary variation.

This is *not* a substitute for the real dataset, but it's enough to
exercise the full pipeline end-to-end. When the real dataset is
available, `data.load_data` will prefer it automatically.
"""
from __future__ import annotations

import random
from typing import Iterable

import pandas as pd

# The official Bitext (category, intent) taxonomy used in the
# customer-support dataset. 11 categories, 27 intents.
TAXONOMY = {
    "ACCOUNT": [
        "create_account", "delete_account", "edit_account",
        "switch_account", "recover_password", "registration_problems",
    ],
    "CANCEL": ["cancel_order"],
    "CONTACT": ["contact_customer_service", "contact_human_agent"],
    "DELIVERY": ["delivery_options", "delivery_period"],
    "FEEDBACK": ["complaint", "review"],
    "INVOICE": ["check_invoice", "get_invoice"],
    "NEWSLETTER": ["newsletter_subscription"],
    "ORDER": ["place_order", "change_order", "track_order"],
    "PAYMENT": ["check_payment_methods", "payment_issue"],
    "REFUND": ["check_refund_policy", "get_refund", "track_refund"],
    "SHIPPING_ADDRESS": ["change_shipping_address", "set_up_shipping_address"],
}

# Per-intent templates. Each template uses {q} for an opener and {p}
# for a politeness/urgency tag, sampled at generation time.
TEMPLATES: dict[str, list[str]] = {
    "create_account": [
        "{q} I want to create a new account on your platform",
        "{q} how can I sign up for an account?",
        "I'd like to register, what's the process? {p}",
        "Can you help me open a customer account? {p}",
    ],
    "delete_account": [
        "{q} I want to delete my account permanently",
        "Please remove my account from your system. {p}",
        "How do I close my account? I no longer want to use the service.",
        "{q} I need my account deleted right away",
    ],
    "edit_account": [
        "{q} I need to update the email on my profile",
        "How can I change my account information?",
        "I'd like to edit my personal details. {p}",
        "{q} can you help me modify my account settings?",
    ],
    "switch_account": [
        "{q} I want to switch to a different account",
        "How do I log into a secondary account?",
        "Can I have two accounts and switch between them? {p}",
        "{q} I need to use a different user profile",
    ],
    "recover_password": [
        "{q} I forgot my password and can't log in",
        "I need to reset my password, please help. {p}",
        "Locked out of my account, password recovery isn't working!",
        "{q} the password reset email never arrived",
    ],
    "registration_problems": [
        "{q} I'm getting an error when trying to register",
        "The signup form keeps rejecting my email. {p}",
        "Why can't I create my account? It says invalid input.",
        "{q} I keep getting 'username already taken' but I'm new",
    ],
    "cancel_order": [
        "{q} I need to cancel my order immediately",
        "Please cancel order #1234 before it ships. {p}",
        "I changed my mind, can I cancel my recent purchase?",
        "{q} cancel my order asap, I ordered the wrong item",
    ],
    "contact_customer_service": [
        "{q} how do I contact your support team?",
        "What is your customer service phone number?",
        "I need to talk to someone about my order. {p}",
        "{q} is there a support email I can use?",
    ],
    "contact_human_agent": [
        "{q} I want to speak with a human agent, not a bot",
        "Please connect me to a real person. {p}",
        "Can a live representative help me? The chatbot isn't enough.",
        "{q} I need a human agent right now",
    ],
    "delivery_options": [
        "{q} what shipping methods are available?",
        "Do you offer express delivery? I'm in a hurry.",
        "Can I choose same-day shipping? {p}",
        "{q} what are my delivery options for this order?",
    ],
    "delivery_period": [
        "{q} how long will my order take to arrive?",
        "What's the typical delivery time to my country?",
        "When can I expect my package? {p}",
        "{q} estimated delivery for standard shipping?",
    ],
    "complaint": [
        "{q} I'm extremely unhappy with the service",
        "This is unacceptable, I want to file a formal complaint! {p}",
        "Your support has been awful and I'm furious about it",
        "{q} I want to lodge a complaint about the recent experience",
    ],
    "review": [
        "{q} how can I leave a review for the product I bought?",
        "I'd like to share my feedback on a recent purchase",
        "Where do I post a product review? {p}",
        "{q} I want to write a review of my experience",
    ],
    "check_invoice": [
        "{q} I need to look up an old invoice",
        "Where can I view my past invoices? {p}",
        "Can you show me invoice number 8842?",
        "{q} I'm trying to find a copy of my invoice",
    ],
    "get_invoice": [
        "{q} please send me the invoice for my recent order",
        "I need an invoice for accounting. {p}",
        "Can you email me a PDF invoice for order #5500?",
        "{q} I'd like to download an invoice",
    ],
    "newsletter_subscription": [
        "{q} how do I subscribe to your newsletter?",
        "I'd like to receive your weekly emails. {p}",
        "Can I unsubscribe from the newsletter please?",
        "{q} sign me up for promotional emails",
    ],
    "place_order": [
        "{q} I'd like to place a new order",
        "How do I order this product? {p}",
        "I'm ready to buy, what's next?",
        "{q} can you help me complete my purchase?",
    ],
    "change_order": [
        "{q} I need to modify my existing order",
        "Can I add another item to order #4422? {p}",
        "Please change the size on my recent order before it ships",
        "{q} I want to update the quantity in my order",
    ],
    "track_order": [
        "{q} where is my order? It hasn't arrived yet",
        "Tracking number says delivered but I never got it. {p}",
        "Can you tell me the status of order #9001?",
        "{q} how do I track my shipment?",
    ],
    "check_payment_methods": [
        "{q} which payment methods do you accept?",
        "Do you take Apple Pay or only credit cards? {p}",
        "Can I pay with PayPal? Is bank transfer an option?",
        "{q} what are my payment options at checkout?",
    ],
    "payment_issue": [
        "{q} my payment was declined but my card is fine",
        "I was charged twice for the same order! {p}",
        "Help, the payment failed but money was deducted!",
        "{q} I'm having trouble paying, the page errors out",
    ],
    "check_refund_policy": [
        "{q} what is your refund policy?",
        "How many days do I have to return an item? {p}",
        "Can I get a full refund if I'm not satisfied?",
        "{q} do you offer money-back guarantees?",
    ],
    "get_refund": [
        "{q} I want to get a refund for my purchase",
        "Please refund order #3300, the item arrived broken. {p}",
        "I need my money back for this defective product!",
        "{q} how do I request a refund?",
    ],
    "track_refund": [
        "{q} I'm still waiting for my refund, where is it?",
        "It's been 10 days and no refund yet. {p}",
        "Can you check on the status of my refund?",
        "{q} when will my refund be processed?",
    ],
    "change_shipping_address": [
        "{q} I need to change the shipping address on my order",
        "Wrong address on my order, please update it! {p}",
        "Can you reroute my package to a new address?",
        "{q} how do I edit the delivery address?",
    ],
    "set_up_shipping_address": [
        "{q} how do I add a new shipping address to my account?",
        "I want to save a second address for deliveries",
        "Can I set up multiple addresses on my profile? {p}",
        "{q} adding a new address before my next order",
    ],
}

OPENERS = [
    "Hi,", "Hello,", "Hey there,", "Good morning,", "Quick question -",
    "I have a question:", "Could you help -", "Excuse me,", "Hi team,", "",
]
POLITE_TAGS = [
    "Thanks!", "Please assist.", "I'd appreciate a quick reply.",
    "This is urgent.", "Thanks in advance.", "Looking forward to hearing back.",
    "Cheers.", "Many thanks.", "", "",
]


_NOISE_DROP_PROB = 0.08
_NOISE_TYPO_PROB = 0.06
_NOISE_LOWER_PROB = 0.30


def _add_noise(text: str, rng: random.Random) -> str:
    """Inject realistic noise: dropped words, character typos, casing."""
    words = text.split()
    if rng.random() < _NOISE_DROP_PROB and len(words) > 3:
        idx = rng.randrange(len(words))
        words.pop(idx)
    out = []
    for w in words:
        if rng.random() < _NOISE_TYPO_PROB and len(w) > 3:
            i = rng.randrange(1, len(w) - 1)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]  # swap two chars
        out.append(w)
    s = " ".join(out)
    if rng.random() < _NOISE_LOWER_PROB:
        s = s.lower()
    return s


def _render_template(template: str, rng: random.Random) -> str:
    raw = template.format(
        q=rng.choice(OPENERS), p=rng.choice(POLITE_TAGS),
    ).strip().replace("  ", " ")
    return _add_noise(raw, rng)


def generate(
    n_per_intent: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame with columns: instruction, category, intent."""
    rng = random.Random(seed)
    rows: list[dict] = []
    for category, intents in TAXONOMY.items():
        for intent in intents:
            templates = TEMPLATES[intent]
            for _ in range(n_per_intent):
                t = rng.choice(templates)
                rows.append({
                    "instruction": _render_template(t, rng),
                    "category": category,
                    "intent": intent,
                })
    rng.shuffle(rows)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    print(df.head())
    print(f"\nTotal rows: {len(df)}")
    print(df["category"].value_counts())
