"""
Train TF-IDF + multinomial Logistic Regression pipelines on the
processed Bitext dataset for two tasks:

  1. Category classification (11-way)
  2. Urgency classification (3-way)

Both models are saved to `models/` as joblib pickles. A held-out test
split is also persisted so the LLM evaluator scores on identical data.

Run as:
    python -m classical.train_classical
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    PROCESSED_DATASET_CSV,
    CATEGORY_MODEL_PATH,
    URGENCY_MODEL_PATH,
    DATA_DIR,
    RESULTS_DIR,
    TFIDF_PARAMS,
    LR_PARAMS,
    RANDOM_STATE,
    TEST_SIZE,
    COL_TEXT,
    COL_CATEGORY,
)


TEST_SPLIT_PATH = DATA_DIR / "test_split.csv"
TRAIN_SPLIT_PATH = DATA_DIR / "train_split.csv"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf", LogisticRegression(**LR_PARAMS)),
    ])


def time_predictions(pipe: Pipeline, X) -> tuple[np.ndarray, dict]:
    """Run prediction and capture per-row latency stats."""
    # warm up
    pipe.predict(X[:1])

    timings_ms = []
    for x in X:
        t0 = time.perf_counter()
        pipe.predict([x])
        timings_ms.append((time.perf_counter() - t0) * 1000)
    timings_ms = np.array(timings_ms)
    preds = pipe.predict(X)
    return preds, {
        "p50_ms": float(np.percentile(timings_ms, 50)),
        "p95_ms": float(np.percentile(timings_ms, 95)),
        "mean_ms": float(np.mean(timings_ms)),
    }


def report(y_true, y_pred, label: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"\n=== Classical {label} ===")
    print(f"accuracy:    {acc:.4f}")
    print(f"macro F1:    {macro_f1:.4f}")
    print(f"weighted F1: {weighted_f1:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": classification_report(
            y_true, y_pred, zero_division=0, output_dict=True,
        ),
    }


def main() -> None:
    if not PROCESSED_DATASET_CSV.exists():
        raise SystemExit(
            "Processed dataset not found. Run `python -m data.load_data` first."
        )
    df = pd.read_csv(PROCESSED_DATASET_CSV)
    print(f"[train] Loaded {len(df):,} rows")

    # Stratify on category since urgency is derived from it
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[COL_CATEGORY].astype(str).str.upper(),
    )
    train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
    test_df.to_csv(TEST_SPLIT_PATH, index=False)
    print(f"[train] Train rows: {len(train_df):,}  test rows: {len(test_df):,}")

    X_train = train_df[COL_TEXT].astype(str).values
    X_test = test_df[COL_TEXT].astype(str).values

    # Normalize category labels to upper case for consistency with the
    # LLM allowlist
    y_cat_train = train_df[COL_CATEGORY].astype(str).str.upper().values
    y_cat_test = test_df[COL_CATEGORY].astype(str).str.upper().values
    y_urg_train = train_df["urgency"].astype(str).values
    y_urg_test = test_df["urgency"].astype(str).values

    # ---- Category model ---------------------------------------------------
    print("\n[train] Fitting category pipeline ...")
    cat_pipe = build_pipeline()
    t0 = time.perf_counter()
    cat_pipe.fit(X_train, y_cat_train)
    cat_train_secs = time.perf_counter() - t0
    print(f"[train] Trained category model in {cat_train_secs:.2f}s")

    cat_preds, cat_latency = time_predictions(cat_pipe, X_test)
    cat_metrics = report(y_cat_test, cat_preds, "category")
    cat_metrics["train_seconds"] = cat_train_secs
    cat_metrics["latency_ms"] = cat_latency
    cat_metrics["confusion_matrix"] = confusion_matrix(
        y_cat_test, cat_preds, labels=sorted(set(y_cat_train))
    ).tolist()
    cat_metrics["labels"] = sorted(set(y_cat_train))

    joblib.dump(cat_pipe, CATEGORY_MODEL_PATH)
    print(f"[train] Saved -> {CATEGORY_MODEL_PATH}")

    # ---- Urgency model ----------------------------------------------------
    print("\n[train] Fitting urgency pipeline ...")
    urg_pipe = build_pipeline()
    t0 = time.perf_counter()
    urg_pipe.fit(X_train, y_urg_train)
    urg_train_secs = time.perf_counter() - t0
    print(f"[train] Trained urgency model in {urg_train_secs:.2f}s")

    urg_preds, urg_latency = time_predictions(urg_pipe, X_test)
    urg_metrics = report(y_urg_test, urg_preds, "urgency")
    urg_metrics["train_seconds"] = urg_train_secs
    urg_metrics["latency_ms"] = urg_latency
    urg_metrics["confusion_matrix"] = confusion_matrix(
        y_urg_test, urg_preds, labels=["low", "medium", "high"]
    ).tolist()
    urg_metrics["labels"] = ["low", "medium", "high"]

    joblib.dump(urg_pipe, URGENCY_MODEL_PATH)
    print(f"[train] Saved -> {URGENCY_MODEL_PATH}")

    # ---- Persist metrics --------------------------------------------------
    out = RESULTS_DIR / "classical_metrics.json"
    out.write_text(json.dumps(
        {"category": cat_metrics, "urgency": urg_metrics}, indent=2,
    ))
    print(f"\n[train] Wrote metrics -> {out}")


if __name__ == "__main__":
    main()
