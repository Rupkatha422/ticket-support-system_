"""
Download the Bitext customer-support dataset from Hugging Face,
derive an urgency label from its category/intent taxonomy, and cache
the result as a CSV for the rest of the pipeline.

Run as a module so relative imports work:

    python -m data.load_data
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Make the project root importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    HF_DATASET_NAME,
    COL_TEXT,
    COL_CATEGORY,
    COL_INTENT,
    URGENCY_RULES,
    RAW_DATASET_CSV,
    PROCESSED_DATASET_CSV,
    CATEGORIES_FILE,
    URGENCIES,
)


def derive_urgency(category: str, intent: str) -> str:
    """Map a Bitext (category, intent) pair to an urgency level.

    The rule order matters: HIGH wins over MEDIUM wins over LOW.
    """
    cat = (category or "").upper()
    itn = (intent or "").lower()

    if cat in URGENCY_RULES["high"]["categories"] or itn in URGENCY_RULES["high"]["intents"]:
        return "high"
    if cat in URGENCY_RULES["medium"]["categories"] or itn in URGENCY_RULES["medium"]["intents"]:
        return "medium"
    return "low"


def load_from_huggingface() -> pd.DataFrame:
    """Pull the Bitext dataset using the `datasets` library."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_NAME, split="train")
    df = ds.to_pandas()
    return df


def load_from_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_from_synthetic_fallback() -> pd.DataFrame:
    """Generate a representative dataset that mirrors the Bitext schema.

    Used when network access to Hugging Face is unavailable so the
    pipeline can still be exercised end-to-end.
    """
    from data.synthetic_fallback import generate
    return generate(n_per_intent=120)


def main() -> None:
    if RAW_DATASET_CSV.exists():
        print(f"[load_data] Using cached raw CSV at {RAW_DATASET_CSV}")
        df = load_from_csv(RAW_DATASET_CSV)
    else:
        print(f"[load_data] Downloading {HF_DATASET_NAME} from Hugging Face ...")
        try:
            df = load_from_huggingface()
            df.to_csv(RAW_DATASET_CSV, index=False)
            print(f"[load_data] Cached raw dataset -> {RAW_DATASET_CSV}")
        except Exception as exc:  # noqa: BLE001
            print(f"[load_data] HF download failed ({exc.__class__.__name__}: {exc})")
            print("[load_data] Falling back to representative synthetic data.")
            print("[load_data] Re-run on a network with HF access for the real 27k rows.")
            df = load_from_synthetic_fallback()
            df.to_csv(RAW_DATASET_CSV, index=False)
            print(f"[load_data] Wrote synthetic raw dataset -> {RAW_DATASET_CSV}")

    print(f"[load_data] Raw rows: {len(df):,}  cols: {list(df.columns)}")

    # Some Bitext mirrors use slightly different column names. Normalize.
    rename_map = {}
    for canonical, candidates in {
        COL_TEXT: ["instruction", "Instruction", "utterance", "text"],
        COL_CATEGORY: ["category", "Category"],
        COL_INTENT: ["intent", "Intent"],
    }.items():
        for c in candidates:
            if c in df.columns and c != canonical:
                rename_map[c] = canonical
                break
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {COL_TEXT, COL_CATEGORY, COL_INTENT}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Dataset missing required columns: {missing}")

    df = df[[COL_TEXT, COL_CATEGORY, COL_INTENT]].dropna()
    df[COL_TEXT] = df[COL_TEXT].astype(str).str.strip()
    df = df[df[COL_TEXT].str.len() > 0].reset_index(drop=True)

    df["urgency"] = [
        derive_urgency(c, i) for c, i in zip(df[COL_CATEGORY], df[COL_INTENT])
    ]

    # Persist a clean processed version
    df.to_csv(PROCESSED_DATASET_CSV, index=False)
    print(f"[load_data] Wrote processed dataset -> {PROCESSED_DATASET_CSV}")

    # Persist the canonical category list for the LLM prompt
    categories = sorted(df[COL_CATEGORY].astype(str).str.upper().unique().tolist())
    CATEGORIES_FILE.write_text(
        json.dumps({"categories": categories, "urgencies": URGENCIES}, indent=2)
    )
    print(f"[load_data] Wrote categories -> {CATEGORIES_FILE}")

    # Quick stats
    print("\n[load_data] Category distribution:")
    print(df[COL_CATEGORY].value_counts().to_string())
    print("\n[load_data] Urgency distribution:")
    print(df["urgency"].value_counts().to_string())


if __name__ == "__main__":
    main()
