# Support Ticket Triage: Classical ML vs. LLM

A side-by-side comparison of two approaches to customer-support ticket
classification:

1. **Classical ML** &mdash; TF-IDF features + multinomial Logistic Regression
   trained on the full Bitext customer-support dataset.
2. **LLM** &mdash; **Groq Llama 3.1 8B Instant** with an engineered system
   prompt that returns structured JSON (category, urgency, confidence,
   reasoning).

The comparison answers the question every team eventually asks: *"Should
we train a model or just call an LLM?"* The pipeline measures
**accuracy, latency, and cost** for both approaches, then surfaces the
results in a Streamlit triage dashboard.

## Project layout

```
ticket support system/
├── config.py                   # central configuration / paths
├── requirements.txt
├── data/
│   └── load_data.py            # downloads Bitext from Hugging Face
├── classical/
│   └── train_classical.py      # TF-IDF + LR for category & urgency
├── llm/
│   ├── prompts.py              # the engineered system prompt
│   └── llm_classifier.py       # Groq client + deterministic mock
├── evaluate/
│   └── compare.py              # accuracy / latency / cost report
├── dashboard/
│   └── app.py                  # Streamlit UI
├── models/                     # saved sklearn pipelines
└── results/                    # comparison_metrics.json, predictions.csv
```

## Quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. download the dataset (~3 MB) and derive labels
python -m data.load_data

# 3. train the classical pipelines (category + urgency)
python -m classical.train_classical

# 4. (optional) export your Groq key for real LLM eval
#    free key: https://console.groq.com/keys
export GROQ_API_KEY=gsk_...

# 5. evaluate & compare
python -m evaluate.compare

# 6. launch the dashboard
streamlit run dashboard/app.py
```

If `GROQ_API_KEY` is not set, the LLM module automatically falls
back to a deterministic rule-based mock so the whole pipeline still
runs end-to-end. Groq's Llama 3.1 8B Instant is *very* cheap and *very*
fast (typical p50 ~150-300ms, ~$0.05 / $0.08 per 1M input/output tokens),
which makes it a good real-world LLM baseline against the classical model.

## What gets measured

For each approach the comparison records:

| Metric              | Classical ML                 | LLM (Claude)                            |
|---------------------|------------------------------|-----------------------------------------|
| Category accuracy   | macro-F1 + weighted accuracy | same, on the same held-out split        |
| Urgency accuracy    | macro-F1 + weighted accuracy | same                                    |
| Latency (p50, p95)  | per-prediction wall clock    | per-API-call wall clock (Groq is ~10x faster than other hosted LLMs) |
| Cost / 1k tickets   | ~0 (compute only)            | extrapolated from Groq token usage & pricing |

## Urgency labels

Bitext does not ship urgency labels, so we derive them from the
category/intent taxonomy in `config.py::URGENCY_RULES`:

* **high** &mdash; refunds, cancellations, account lockouts, complaints
* **medium** &mdash; active orders, payments, shipping, delivery
* **low** &mdash; everything else (FAQ, contact, newsletter, feedback)

This rule-based mapping serves as the ground truth for both models.

## Engineered prompt

`llm/prompts.py` contains the system prompt. Highlights:

* Constrains the output to **JSON only**, with a fixed schema.
* Pins the category list and urgency list so the model can't hallucinate
  new classes.
* Includes 3 hand-picked few-shot examples covering high / medium / low
  urgency.
* Asks for a 1-sentence rationale, useful for the routing dashboard.

## Dashboard

The Streamlit app has three views:

1. **Live triage** &mdash; paste a ticket, see what each classifier routes it to,
   side-by-side, with the LLM's reasoning.
2. **Model comparison** &mdash; bar charts of accuracy / latency / cost,
   confusion matrices, and per-class F1.
3. **Routing decisions** &mdash; sample of tickets + the queue each model would
   send them to (e.g. `BILLING:high` &rarr; "Tier-2 Billing").
