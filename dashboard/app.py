"""
Streamlit triage dashboard.

Three views:

1. Live triage      — paste a ticket, see what each model routes it to.
2. Model comparison — accuracy, latency, cost, confusion matrices.
3. Routing decisions — sample of tickets and the queue each would go to.

Run:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make the project root importable when launched via `streamlit run`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    CATEGORY_MODEL_PATH,
    URGENCY_MODEL_PATH,
    METRICS_JSON,
    PREDICTIONS_CSV,
    PROCESSED_DATASET_CSV,
    URGENCIES,
)
from llm.llm_classifier import get_classifier  # noqa: E402


# ---------------------------------------------------------------------------
# Page setup & cached resources
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Support Ticket Triage — ML vs LLM",
    page_icon="🎫",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_classical_models():
    if not CATEGORY_MODEL_PATH.exists() or not URGENCY_MODEL_PATH.exists():
        return None, None
    return joblib.load(CATEGORY_MODEL_PATH), joblib.load(URGENCY_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_llm_classifier():
    return get_classifier()


@st.cache_data(show_spinner=False)
def load_metrics():
    if not METRICS_JSON.exists():
        return None
    return json.loads(METRICS_JSON.read_text())


@st.cache_data(show_spinner=False)
def load_predictions():
    if not PREDICTIONS_CSV.exists():
        return None
    return pd.read_csv(PREDICTIONS_CSV)


@st.cache_data(show_spinner=False)
def load_dataset():
    if not PROCESSED_DATASET_CSV.exists():
        return None
    return pd.read_csv(PROCESSED_DATASET_CSV)


# ---------------------------------------------------------------------------
# Routing rules — map (category, urgency) -> queue
# ---------------------------------------------------------------------------
def route_to_queue(category: str, urgency: str) -> str:
    cat = (category or "").upper()
    urg = (urgency or "").lower()

    if urg == "high":
        if cat in {"REFUND", "PAYMENT"}:
            return "Tier-2 Billing (urgent)"
        if cat in {"CANCEL", "ORDER"}:
            return "Order Operations (urgent)"
        if cat == "ACCOUNT":
            return "Account Recovery"
        if cat == "FEEDBACK":
            return "Customer Success Manager"
        return "Senior Support Lead"

    if urg == "medium":
        if cat in {"ORDER", "DELIVERY", "SHIPPING_ADDRESS"}:
            return "Order Operations"
        if cat in {"INVOICE", "PAYMENT"}:
            return "Billing"
        return "Tier-1 Support"

    # low
    if cat in {"NEWSLETTER", "CONTACT", "FEEDBACK"}:
        return "Self-service / FAQ"
    return "Tier-1 Support"


# ---------------------------------------------------------------------------
# Sidebar — navigation & status
# ---------------------------------------------------------------------------
st.sidebar.title("🎫 Triage")
view = st.sidebar.radio(
    "View",
    ["Live triage", "Model comparison", "Routing decisions"],
    index=0,
)

cat_clf, urg_clf = load_classical_models()
llm = load_llm_classifier()
metrics = load_metrics()

with st.sidebar.expander("Pipeline status", expanded=True):
    st.write(f"Classical models: {'✅' if cat_clf and urg_clf else '⚠️ run `train_classical`'}")
    st.write(f"LLM backend: **{llm.backend}**")
    st.write(f"Metrics file: {'✅' if metrics else '⚠️ run `evaluate.compare`'}")

# ---------------------------------------------------------------------------
# View 1 — Live triage
# ---------------------------------------------------------------------------
if view == "Live triage":
    st.title("Live ticket triage")
    st.write(
        "Paste any customer message below. Both models classify it side-by-side."
    )

    examples = [
        "I want a refund for my broken phone, it arrived dead on arrival!",
        "Hi, what payment methods do you accept?",
        "Where is my order? It was supposed to arrive yesterday.",
        "I forgot my password and I'm completely locked out of my account",
        "Please cancel order #4422 immediately, I don't need it anymore",
    ]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex[:30] + "…", help=ex, use_container_width=True):
            st.session_state["ticket_text"] = ex

    text = st.text_area(
        "Customer message",
        value=st.session_state.get("ticket_text",
            "I want a refund for my broken phone, it arrived dead on arrival!"),
        height=100,
    )

    if st.button("Triage", type="primary"):
        if not (cat_clf and urg_clf):
            st.error("Classical models not found — run `python -m classical.train_classical`")
        else:
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Classical ML")
                t0 = time.perf_counter()
                cat_pred = cat_clf.predict([text])[0]
                urg_pred = urg_clf.predict([text])[0]
                lat = (time.perf_counter() - t0) * 1000
                # Probabilities
                cat_proba = cat_clf.predict_proba([text])[0]
                cat_classes = cat_clf.classes_
                top = sorted(zip(cat_classes, cat_proba), key=lambda x: -x[1])[:3]

                queue = route_to_queue(cat_pred, urg_pred)
                st.metric("Category", cat_pred)
                st.metric("Urgency", urg_pred.upper())
                st.metric("Latency", f"{lat:.2f} ms")
                st.success(f"➡️ Route to: **{queue}**")
                with st.expander("Top-3 category probabilities"):
                    for c, p in top:
                        st.write(f"- {c}: {p:.3f}")

            with c2:
                st.subheader(f"LLM ({llm.backend})")
                with st.spinner("Calling LLM …"):
                    res = llm.classify(text)
                queue = route_to_queue(res.category, res.urgency)
                st.metric("Category", res.category)
                st.metric("Urgency", res.urgency.upper())
                st.metric("Latency", f"{res.latency_ms:.0f} ms")
                st.metric("Confidence", f"{res.confidence:.2f}")
                st.success(f"➡️ Route to: **{queue}**")
                st.caption(f"💭 {res.rationale}")
                if res.input_tokens or res.output_tokens:
                    st.caption(
                        f"Tokens — input: {res.input_tokens}, output: {res.output_tokens}"
                    )

# ---------------------------------------------------------------------------
# View 2 — Model comparison
# ---------------------------------------------------------------------------
elif view == "Model comparison":
    st.title("Classical ML vs. LLM — head-to-head")

    if not metrics:
        st.warning("Run `python -m evaluate.compare` to generate the metrics report.")
        st.stop()

    cm = metrics["classical"]
    lm = metrics["llm"]

    st.subheader("Headline metrics")
    cols = st.columns(4)
    cols[0].metric("Category accuracy",
                   f"{cm['category_accuracy']:.1%}",
                   f"{(lm['category_accuracy']-cm['category_accuracy'])*100:+.1f} pp LLM")
    cols[1].metric("Urgency accuracy",
                   f"{cm['urgency_accuracy']:.1%}",
                   f"{(lm['urgency_accuracy']-cm['urgency_accuracy'])*100:+.1f} pp LLM")
    cols[2].metric("Latency p50 (ms)",
                   f"{cm['latency_ms']['category']['p50']:.2f}",
                   f"LLM: {lm['latency_ms']['category']['p50']:.0f}")
    cols[3].metric("Cost / 1k tickets",
                   f"${cm['cost_usd_per_1k_tickets']:.4f}",
                   f"LLM: ${lm['cost_usd_per_1k_tickets']:.4f}")

    # ---- Bar chart: accuracy & F1 -----------------------------------------
    st.subheader("Accuracy & macro-F1")
    df_acc = pd.DataFrame([
        {"model": "Classical", "metric": "category accuracy", "value": cm["category_accuracy"]},
        {"model": "Classical", "metric": "category macro-F1", "value": cm["category_macro_f1"]},
        {"model": "Classical", "metric": "urgency accuracy",  "value": cm["urgency_accuracy"]},
        {"model": "Classical", "metric": "urgency macro-F1",  "value": cm["urgency_macro_f1"]},
        {"model": "LLM",       "metric": "category accuracy", "value": lm["category_accuracy"]},
        {"model": "LLM",       "metric": "category macro-F1", "value": lm["category_macro_f1"]},
        {"model": "LLM",       "metric": "urgency accuracy",  "value": lm["urgency_accuracy"]},
        {"model": "LLM",       "metric": "urgency macro-F1",  "value": lm["urgency_macro_f1"]},
    ])
    fig_acc = px.bar(df_acc, x="metric", y="value", color="model",
                     barmode="group", range_y=[0, 1.05])
    st.plotly_chart(fig_acc, use_container_width=True)

    # ---- Latency comparison -----------------------------------------------
    st.subheader("Latency (per ticket)")
    df_lat = pd.DataFrame([
        {"model": "Classical", "stat": "p50", "ms": cm["latency_ms"]["category"]["p50"]},
        {"model": "Classical", "stat": "p95", "ms": cm["latency_ms"]["category"]["p95"]},
        {"model": "LLM",       "stat": "p50", "ms": lm["latency_ms"]["category"]["p50"]},
        {"model": "LLM",       "stat": "p95", "ms": lm["latency_ms"]["category"]["p95"]},
    ])
    fig_lat = px.bar(df_lat, x="stat", y="ms", color="model",
                     barmode="group", log_y=True,
                     labels={"ms": "milliseconds (log scale)"})
    st.plotly_chart(fig_lat, use_container_width=True)

    # ---- Cost extrapolation -----------------------------------------------
    st.subheader("Estimated cost")
    volumes = [1_000, 10_000, 100_000, 1_000_000]
    cost_rows = []
    for v in volumes:
        cost_rows.append({"volume": f"{v:,} tickets",
                          "model": "Classical",
                          "cost_usd": cm["cost_usd_per_1k_tickets"] * v / 1000})
        cost_rows.append({"volume": f"{v:,} tickets",
                          "model": "LLM",
                          "cost_usd": lm["cost_usd_per_1k_tickets"] * v / 1000})
    cost_df = pd.DataFrame(cost_rows)
    fig_cost = px.bar(cost_df, x="volume", y="cost_usd", color="model",
                      barmode="group", labels={"cost_usd": "USD"})
    st.plotly_chart(fig_cost, use_container_width=True)
    st.caption(f"LLM model: `{lm.get('model', 'mock')}`. Pricing per `config.py`.")

    # ---- Confusion matrices ----------------------------------------------
    st.subheader("Urgency confusion matrices")
    cc1, cc2 = st.columns(2)
    for name, m, container in [("Classical", cm, cc1), ("LLM", lm, cc2)]:
        cm_arr = np.array(m["urgency_confusion_matrix"])
        fig = go.Figure(data=go.Heatmap(
            z=cm_arr, x=URGENCIES, y=URGENCIES,
            text=cm_arr, texttemplate="%{text}",
            colorscale="Blues",
        ))
        fig.update_layout(title=name, xaxis_title="predicted", yaxis_title="true",
                          height=350)
        container.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# View 3 — Routing decisions
# ---------------------------------------------------------------------------
elif view == "Routing decisions":
    st.title("Routing decisions")
    preds = load_predictions()
    if preds is None:
        st.warning("No predictions file. Run `python -m evaluate.compare` first.")
        st.stop()

    required_cols = {
        "instruction", "true_category", "true_urgency",
        "classical_category", "classical_urgency",
    }
    missing = required_cols - set(preds.columns)
    if missing:
        st.error(
            f"`predictions.csv` is missing columns: {sorted(missing)}.\n\n"
            "Re-run `python -m evaluate.compare` to regenerate the file."
        )
        st.stop()

    preds = preds.copy()
    preds["classical_queue"] = [
        route_to_queue(c, u)
        for c, u in zip(preds["classical_category"], preds["classical_urgency"])
    ]
    preds["llm_queue"] = [
        route_to_queue(c, u) if pd.notna(c) else "—"
        for c, u in zip(preds["llm_category"], preds["llm_urgency"])
    ]
    preds["agree"] = preds["classical_queue"] == preds["llm_queue"]

    st.subheader("Filters")
    c1, c2, c3 = st.columns(3)
    with c1:
        urgency_filter = st.multiselect("Urgency", URGENCIES, default=URGENCIES)
    with c2:
        agreement_filter = st.selectbox(
            "Agreement", ["all", "agree only", "disagree only"], index=0
        )
    with c3:
        only_with_llm = st.checkbox(
            "Only rows with LLM predictions", value=True,
            help="Limits to the LLM evaluation slice.",
        )

    view_df = preds[preds["true_urgency"].isin(urgency_filter)]
    if only_with_llm:
        view_df = view_df[view_df["llm_category"].notna()]
    if agreement_filter == "agree only":
        view_df = view_df[view_df["agree"]]
    elif agreement_filter == "disagree only":
        view_df = view_df[~view_df["agree"]]

    st.write(f"**{len(view_df):,}** tickets match.")
    st.dataframe(
        view_df[[
            "instruction", "true_category", "true_urgency",
            "classical_category", "classical_urgency", "classical_queue",
            "llm_category", "llm_urgency", "llm_queue",
            "llm_rationale", "llm_confidence",
        ]],
        use_container_width=True,
        height=520,
    )

    # Queue distribution
    st.subheader("Routing queue distribution")
    qc = (
        view_df.melt(
            id_vars=["instruction"],
            value_vars=["classical_queue", "llm_queue"],
            var_name="model",
            value_name="queue",
        )
        .dropna(subset=["queue"])
    )
    qc["model"] = qc["model"].str.replace("_queue", "")
    fig_q = px.histogram(qc, x="queue", color="model", barmode="group")
    fig_q.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_q, use_container_width=True)
