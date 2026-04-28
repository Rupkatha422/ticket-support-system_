"""
Central configuration for the support ticket triage project.

All paths are anchored to the project root so the code can be run
from anywhere (CLI, Streamlit, notebook).
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

RAW_DATASET_CSV = DATA_DIR / "bitext_tickets.csv"
PROCESSED_DATASET_CSV = DATA_DIR / "tickets_processed.csv"

CATEGORY_MODEL_PATH = MODELS_DIR / "category_clf.joblib"
URGENCY_MODEL_PATH = MODELS_DIR / "urgency_clf.joblib"

METRICS_JSON = RESULTS_DIR / "comparison_metrics.json"
PREDICTIONS_CSV = RESULTS_DIR / "predictions.csv"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
HF_DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

# Column names in the raw dataset
COL_TEXT = "instruction"          # the customer message
COL_CATEGORY = "category"         # high level intent (10-ish classes)
COL_INTENT = "intent"             # finer-grained intent (~27 classes)

# We derive an "urgency" label since the dataset doesn't ship one. The
# mapping below is a defensible heuristic curated from the Bitext
# category/intent taxonomy: account-locked or refund-related issues are
# treated as high urgency, billing problems as medium, and general FAQ
# style queries as low.
URGENCY_RULES = {
    "high": {
        "categories": {"REFUND", "CANCEL"},
        "intents": {
            "get_refund", "track_refund", "check_refund_policy",
            "cancel_order", "delete_account", "complaint",
            "recover_password", "payment_issue",
        },
    },
    "medium": {
        "categories": {"ORDER", "PAYMENT", "SHIPPING_ADDRESS", "DELIVERY"},
        "intents": {
            "track_order", "change_order", "place_order",
            "check_invoice", "get_invoice", "check_payment_methods",
            "change_shipping_address", "set_up_shipping_address",
            "delivery_options", "delivery_period",
        },
    },
    # everything else is low
}

# ---------------------------------------------------------------------------
# Model hyper-params
# ---------------------------------------------------------------------------
TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=100_000,
)
LR_PARAMS = dict(
    C=4.0,
    max_iter=2000,
    n_jobs=-1,
    solver="saga",
)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# Groq -- OpenAI-compatible inference endpoint, very fast for small Llamas.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# Sample size for the LLM evaluation slice. Running the LLM over 27k
# tickets is expensive; we run it on a stratified slice and extrapolate.
LLM_EVAL_SAMPLE_SIZE = int(os.environ.get("LLM_EVAL_SAMPLE_SIZE", "300"))

# Approximate Groq pricing (USD per 1M tokens) for cost estimation.
# Update if the published pricing changes -- see https://groq.com/pricing
LLM_PRICING_PER_M_TOKENS = {
    "llama-3.1-8b-instant":   {"input": 0.05, "output": 0.08},
    "llama-3.1-70b-versatile":{"input": 0.59, "output": 0.79},
    "llama-3.3-70b-versatile":{"input": 0.59, "output": 0.79},
    "mixtral-8x7b-32768":     {"input": 0.24, "output": 0.24},
    "gemma2-9b-it":           {"input": 0.20, "output": 0.20},
}

# ---------------------------------------------------------------------------
# Categories the LLM is allowed to emit. Filled in by the data loader so the
# label set matches the trained classical model exactly.
# ---------------------------------------------------------------------------
CATEGORIES_FILE = DATA_DIR / "categories.json"
URGENCIES = ["low", "medium", "high"]
