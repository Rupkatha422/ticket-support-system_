"""
Side-by-side evaluation of the classical ML pipeline and the LLM
classifier on the same held-out test split.

Outputs:
  results/comparison_metrics.json  — accuracy, latency, cost, F1
  results/predictions.csv          — per-row predictions for both models

Run as:
    python -m evaluate.compare
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    CATEGORY_MODEL_PATH,
    URGENCY_MODEL_PATH,
    DATA_DIR,
    METRICS_JSON,
    PREDICTIONS_CSV,
    LLM_EVAL_SAMPLE_SIZE,
    GROQ_MODEL,
    COL_TEXT,
    COL_CATEGORY,
    URGENCIES,
    RANDOM_STATE,
)
from llm.llm_classifier import get_classifier, estimate_cost_usd  # noqa: E402


TEST_SPLIT_PATH = DATA_DIR / "test_split.csv"


def _percentile(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else 0.0


def evaluate_classical(test_df: pd.DataFrame) -> dict:
    cat_clf = joblib.load(CATEGORY_MODEL_PATH)
    urg_clf = joblib.load(URGENCY_MODEL_PATH)

    X = test_df[COL_TEXT].astype(str).values
    y_cat = test_df[COL_CATEGORY].astype(str).str.upper().values
    y_urg = test_df["urgency"].astype(str).values

    # Time per-row for fair latency comparison
    cat_lat, urg_lat = [], []
    cat_pred, urg_pred = [], []

    cat_clf.predict(X[:1])
    urg_clf.predict(X[:1])

    for x in tqdm(X, desc="classical"):
        t0 = time.perf_counter()
        cp = cat_clf.predict([x])[0]
        cat_lat.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        up = urg_clf.predict([x])[0]
        urg_lat.append((time.perf_counter() - t0) * 1000)

        cat_pred.append(cp)
        urg_pred.append(up)

    return {
        "category_pred": cat_pred,
        "urgency_pred": urg_pred,
        "category_accuracy": float(accuracy_score(y_cat, cat_pred)),
        "category_macro_f1": float(f1_score(y_cat, cat_pred, average="macro")),
        "urgency_accuracy": float(accuracy_score(y_urg, urg_pred)),
        "urgency_macro_f1": float(f1_score(y_urg, urg_pred, average="macro")),
        "latency_ms": {
            "category": {
                "p50": _percentile(cat_lat, 50),
                "p95": _percentile(cat_lat, 95),
                "mean": float(np.mean(cat_lat)),
            },
            "urgency": {
                "p50": _percentile(urg_lat, 50),
                "p95": _percentile(urg_lat, 95),
                "mean": float(np.mean(urg_lat)),
            },
        },
        "cost_usd_per_1k_tickets": 0.0,  # compute only
        "category_confusion_matrix": confusion_matrix(
            y_cat, cat_pred, labels=sorted(set(list(y_cat) + cat_pred))
        ).tolist(),
        "category_labels": sorted(set(list(y_cat) + cat_pred)),
        "urgency_confusion_matrix": confusion_matrix(
            y_urg, urg_pred, labels=URGENCIES,
        ).tolist(),
    }


def evaluate_llm(eval_df: pd.DataFrame) -> dict:
    clf = get_classifier()
    backend = clf.backend
    print(f"[evaluate] LLM backend: {backend}")

    X = eval_df[COL_TEXT].astype(str).values
    y_cat = eval_df[COL_CATEGORY].astype(str).str.upper().values
    y_urg = eval_df["urgency"].astype(str).values

    cat_pred, urg_pred, lats = [], [], []
    in_toks, out_toks = 0, 0
    rationales = []
    confidences = []

    for x in tqdm(X, desc=f"llm[{backend}]"):
        try:
            res = clf.classify(x)
        except Exception as exc:  # noqa: BLE001
            print(f"[evaluate] LLM error on a row, falling back to FEEDBACK/low: {exc}")
            from llm.llm_classifier import TriageResult
            res = TriageResult(
                category="FEEDBACK", urgency="low",
                confidence=0.0, rationale=f"error: {exc}",
                latency_ms=0.0, backend=backend,
            )
        cat_pred.append(res.category)
        urg_pred.append(res.urgency)
        lats.append(res.latency_ms)
        in_toks += res.input_tokens
        out_toks += res.output_tokens
        rationales.append(res.rationale)
        confidences.append(res.confidence)

    n = len(X)
    cost_per_call = estimate_cost_usd(in_toks / max(n, 1), out_toks / max(n, 1), GROQ_MODEL)
    cost_per_1k = cost_per_call * 1_000

    return {
        "category_pred": cat_pred,
        "urgency_pred": urg_pred,
        "rationale": rationales,
        "confidence": confidences,
        "category_accuracy": float(accuracy_score(y_cat, cat_pred)),
        "category_macro_f1": float(f1_score(y_cat, cat_pred, average="macro", zero_division=0)),
        "urgency_accuracy": float(accuracy_score(y_urg, urg_pred)),
        "urgency_macro_f1": float(f1_score(y_urg, urg_pred, average="macro", zero_division=0)),
        "latency_ms": {
            "category": {
                "p50": _percentile(lats, 50),
                "p95": _percentile(lats, 95),
                "mean": float(np.mean(lats)),
            },
            # LLM does both tasks in one call, so re-use timings
            "urgency": {
                "p50": _percentile(lats, 50),
                "p95": _percentile(lats, 95),
                "mean": float(np.mean(lats)),
            },
        },
        "cost_usd_per_1k_tickets": cost_per_1k,
        "model": GROQ_MODEL if backend == "groq" else "mock",
        "backend": backend,
        "tokens": {"input": in_toks, "output": out_toks},
        "category_confusion_matrix": confusion_matrix(
            y_cat, cat_pred,
            labels=sorted(set(list(y_cat) + cat_pred)),
        ).tolist(),
        "category_labels": sorted(set(list(y_cat) + cat_pred)),
        "urgency_confusion_matrix": confusion_matrix(
            y_urg, urg_pred, labels=URGENCIES,
        ).tolist(),
        "n_evaluated": n,
    }


def main() -> None:
    if not TEST_SPLIT_PATH.exists():
        raise SystemExit(
            "Test split not found. Run `python -m classical.train_classical` first."
        )

    test_df = pd.read_csv(TEST_SPLIT_PATH)
    print(f"[evaluate] Test rows: {len(test_df):,}")

    # Stratified slice for the LLM (running thousands of API calls is
    # expensive). The classical model is evaluated on the full test set.
    if len(test_df) > LLM_EVAL_SAMPLE_SIZE:
        llm_eval_df = (
            test_df.groupby(COL_CATEGORY, group_keys=False)
            .apply(lambda g: g.sample(
                min(len(g), max(1, LLM_EVAL_SAMPLE_SIZE // test_df[COL_CATEGORY].nunique())),
                random_state=RANDOM_STATE,
            ))
            .reset_index(drop=True)
        )
    else:
        llm_eval_df = test_df.copy()
    print(f"[evaluate] LLM eval slice: {len(llm_eval_df):,}")

    classical_metrics = evaluate_classical(test_df)
    llm_metrics = evaluate_llm(llm_eval_df)

    # ---- Persist per-row predictions -------------------------------------
    out_rows = []
    for i, row in test_df.reset_index(drop=True).iterrows():
        out_rows.append({
            "instruction": row[COL_TEXT],
            "true_category": str(row[COL_CATEGORY]).upper(),
            "true_urgency": row["urgency"],
            "classical_category": classical_metrics["category_pred"][i],
            "classical_urgency": classical_metrics["urgency_pred"][i],
        })
    pred_df = pd.DataFrame(out_rows)

    # Merge LLM predictions where applicable. We rename the LLM frame's
    # text column to "instruction" so the merge key matches what
    # `pred_df` uses, and drop_duplicates so the join doesn't multiply
    # rows when synthetic templates collide.
    llm_df = llm_eval_df[[COL_TEXT]].rename(columns={COL_TEXT: "instruction"}).copy()
    llm_df["llm_category"] = llm_metrics["category_pred"]
    llm_df["llm_urgency"] = llm_metrics["urgency_pred"]
    llm_df["llm_rationale"] = llm_metrics["rationale"]
    llm_df["llm_confidence"] = llm_metrics["confidence"]
    llm_df = llm_df.drop_duplicates(subset=["instruction"], keep="first")
    pred_df = pred_df.merge(llm_df, how="left", on="instruction")
    pred_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"[evaluate] Wrote predictions -> {PREDICTIONS_CSV}")

    # ---- Drop the prediction arrays from the JSON report ------------------
    for d in (classical_metrics, llm_metrics):
        d.pop("category_pred", None)
        d.pop("urgency_pred", None)
        d.pop("rationale", None)
        d.pop("confidence", None)

    METRICS_JSON.write_text(json.dumps({
        "classical": classical_metrics,
        "llm": llm_metrics,
    }, indent=2))
    print(f"[evaluate] Wrote metrics  -> {METRICS_JSON}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<28} {'Classical':>15} {'LLM':>15}")
    print("-" * 60)
    print(f"{'Category accuracy':<28} {classical_metrics['category_accuracy']:>15.4f} {llm_metrics['category_accuracy']:>15.4f}")
    print(f"{'Category macro-F1':<28} {classical_metrics['category_macro_f1']:>15.4f} {llm_metrics['category_macro_f1']:>15.4f}")
    print(f"{'Urgency accuracy':<28} {classical_metrics['urgency_accuracy']:>15.4f} {llm_metrics['urgency_accuracy']:>15.4f}")
    print(f"{'Urgency macro-F1':<28} {classical_metrics['urgency_macro_f1']:>15.4f} {llm_metrics['urgency_macro_f1']:>15.4f}")
    print(f"{'Latency p50 (ms)':<28} {classical_metrics['latency_ms']['category']['p50']:>15.2f} {llm_metrics['latency_ms']['category']['p50']:>15.2f}")
    print(f"{'Latency p95 (ms)':<28} {classical_metrics['latency_ms']['category']['p95']:>15.2f} {llm_metrics['latency_ms']['category']['p95']:>15.2f}")
    print(f"{'Cost / 1k tickets (USD)':<28} {classical_metrics['cost_usd_per_1k_tickets']:>15.4f} {llm_metrics['cost_usd_per_1k_tickets']:>15.4f}")


if __name__ == "__main__":
    main()
