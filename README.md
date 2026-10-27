# Cost-Aware Model Router

HCLTech hackathon, Case 6. Every query is sent to the **cheapest model that can answer it correctly**
(small → medium → large). Exact and similar questions are answered from a **safe cache**, and an
answer the model is **unsure** about is escalated to a bigger model. The router is trained offline from
every model's answers on the dataset, and it keeps learning from 👍/👎.

Results on the held-out test split (102 queries, **mock mode**, placeholder prices):

| Strategy | Accuracy | Cost per correct answer | Savings vs always-large | Avg latency |
|---|---|---|---|---|
| always-small | 65.7% | 0.000049 | 95.0% | 296 ms |
| always-medium | 74.5% | 0.000248 | 71.4% | 929 ms |
| always-large | 95.1% | 0.000679 | 0% | 2629 ms |
| large + cache | 95.1% | 0.000628 | 7.6% | 2463 ms |
| **ours** | **94.1%** | **0.000488** (−28%) | **28.9%** | **1924 ms** |

Cache safety: 7 of 8 paraphrases reused, 0 of 3 number-changed questions reused, 0 wrong cache hits.
These numbers come from mock models. Rerun `scripts/eval.py` with real models for real numbers.
They vary a little from run to run: the small model's self-consistency samples are random, and a fresh
build re-creates the search index. Across our runs, "ours" scored 92–94% accuracy with 23–29% savings.

---

## Run it

Python 3.11+ (developed on 3.13) and Node 18+. On macOS, LightGBM needs `brew install libomp`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
(cd frontend && npm install)
cp .env.example .env            # MOCK_LLM=1 by default: no API keys, no cost
```

**Offline: build everything** (about 1 minute in mock mode):

```bash
export MOCK_LLM=1
python scripts/setup.py         # all questions × 3 models → data/model_runs.csv + Chroma "examples"
python scripts/train.py         # LightGBM router → data/router.pkl
python scripts/eval.py          # test split: baselines vs ours → eval_results + printed table
```

**Online: the demo** (3 terminals):

```bash
MOCK_LLM=1 uvicorn backend.app:app --port 8000      # API
cd frontend && npm run dev                          # chat UI   → http://localhost:5173
streamlit run report/app.py                         # report    → http://localhost:8501
```

**Learning from feedback, and housekeeping:**

```bash
python scripts/feedback.py              # apply new 👍/👎 (cache + kNN examples)
python scripts/feedback.py --retrain    # ...then retrain; activates only if it beats the current router on tune
python scripts/reset_live.py            # start the demo from zero (clears live requests + answer cache)
```

### Mock vs real models

| | Mock (`MOCK_LLM=1`) | Real (`MOCK_LLM=0`) |
|---|---|---|
| Model call | fake answer built from the reference answer | LiteLLM call to the model in `data/prices.yaml` |
| Correct how often | small 60% / medium 80% / large 95% on average, **lower on hard questions** (so the router has a signal to learn) | whatever the models really do |
| Tokens, latency | fake but realistic, no sleeping | real |
| Cost | none | real API spend |
| Questions outside `queries.csv` | placeholder answer | real answer |

`MOCK_FAIL_RATE=0.15` makes 15% of mock calls fail, which shows the retry and fallback in the report.

**Switching to real models:** put the LiteLLM model names and prices in `data/prices.yaml` and the API
keys in `.env`, then set `MOCK_LLM=0`. Delete `data/model_runs.csv` (the mock answers), then rerun
`setup.py` (resumable: it skips work already done), `train.py` and `eval.py`. `setup.py` makes about
3 calls per question per model: 1 answer, plus 2 extra samples for the small model on mcq/math.

**Organizers' data:** replace `data/queries.csv` with their file, which uses the same columns: `id, query,
reference_answer, task_type (mcq|math|text), difficulty, subject, split, cache_group, cache_expected`.
If `split` is missing, it is read from `data/splits.json`. Then rerun the offline steps.

---

## Architecture

Each part is tagged **Gen AI** (an LLM call), **ML** (a trained or embedding model) or **Rules** (deterministic code).

### Online: `backend/pipeline.py`, one query

| # | Step | Tag | File | What it does |
|---|---|---|---|---|
| 1 | Clean | Rules | `clean.py` | lowercase + trim; mask emails and phone numbers (`has_pii`); extract numbers, names, negation, MCQ options, time words; guess task type |
| 2 | L1 cache (exact) | Rules | `cache.py` | sha256(clean text + options) looked up in Chroma `cache`. A hit costs 0 |
| 3 | Embed | ML | `embed.py` | `BAAI/bge-small-en-v1.5` on CPU → 384-d vector, computed once and reused in steps 4 and 5 |
| 4 | L2 cache (similar) | Rules | `cache.py` | top-3 neighbours; reused only if **all 7 safety rules** pass: similarity ≥ 0.92, same numbers, names, negation, options and task type, entry confident and query not time-sensitive |
| 5 | Choose model | ML | `router.py` | features = embedding + regex features + kNN success rates of small/medium on the 5 most similar labelled questions. Two calibrated LightGBM models give P_small and P_medium: small if P_small ≥ τ_small, else medium if P_medium ≥ τ_medium, else large. Fallback: kNN rates alone, then rules |
| 6 | Call model | Gen AI | `llm.py` | LiteLLM with a per-task prompt that ends in `Final answer:`. On an error: retry once, then the next bigger model. Tokens × price = cost |
| 7 | Unsure check | Rules + Gen AI | `uncertainty.py` | unsure if no valid final answer, a hedge word, or (small model, mcq/math only) 2 extra samples at temperature 0.7 disagree. Unsure → escalate, at most 2 jumps |
| 8 | Save | Rules | `pipeline.py`, `db.py` | request + every LLM call → SQLite. Cache write only if sure, no PII and not time-sensitive |

Every request also gets a `trace_id`, a timer per step, cost from `prices.yaml`, and a **daily budget cap**.
Once the cap is reached, the small model is forced and the routing card says so.
Everything the UI shows comes back as the "routing card": answer, model, why, cache, confidence,
escalations, cost and per-step latency.

### Offline

| Script | Tag | What it does |
|---|---|---|
| `scripts/setup.py` | Gen AI | runs every question on all 3 models; grades deterministically (letter / number with tolerance / normalized text); records the unsure check; puts **train** questions only into Chroma `examples` |
| `scripts/train.py` | ML | trains 2 LightGBM classifiers (isotonic calibration); sweeps τ on tune + out-of-fold train (about 400 questions) with escalation replayed; picks the cheapest τ whose accuracy, minus 1 standard error, is ≥ 95% of always-large; saves `router.pkl` and a `router_versions` row |
| `scripts/eval.py` | – | **test split only**. Baselines come from `model_runs.csv` (no new calls); "large + cache" and "ours" run the real pipeline from an empty cache; saves 225 τ points for the Pareto curve |
| `scripts/feedback.py` | ML | 👎 → that answer's cache entry is never reused. Feedback on new questions → labelled `examples`, so routing changes at once. `--retrain` activates a new router only if it is no worse on accuracy and cost on tune, and better on one |

### API (`backend/app.py`)

`POST /query {query, task_type?}` → routing card · `POST /feedback {trace_id, thumbs}` ·
`GET /stats` → spend, spend if always-large, savings %, routing mix, cache hit rate · `GET /samples` → demo questions

### Storage

- **SQLite `data/app.db`:** `prices`, `requests` (with `source`: `live` or `eval:*`, so evaluation never
  counts toward live stats or the budget), `llm_calls`, `feedback`, `eval_results`, `router_versions`
- **Chroma `data/chroma/`:** `cache` (answers + safety metadata) and `examples` (labelled questions for kNN)
- **Files:** `data/model_runs.csv`, `data/router.pkl`

### Settings

All thresholds are in `backend/config.py`: cache similarity, kNN K, default τ, quality target, tune
safety margin, hedge words, number of samples, maximum escalations, daily budget and mock accuracies.
Prices are in `data/prices.yaml`.

---

## Notes

- **macOS:** torch and LightGBM each ship their own OpenMP library and segfault together. `backend/__init__.py`
  imports LightGBM first and sets `OMP_NUM_THREADS=1`, and every entry point goes through it.
- **Reproducible:** Chroma searches with a high `search_ef` (effectively exact on a few hundred items) and
  LightGBM is seeded, so `train.py` gives the same router every run. The small model's samples at temperature
  0.7 are random, so `ours` varies slightly between eval runs.
- **Dataset:** `data/queries.csv` is a stand-in built from MMLU + GSM8K plus paraphrase, near-duplicate and
  number-changed cache tests (see `data/README.md` and `scripts/build_dataset.py`).

```
backend/   config, db, clean, embed, cache, router, llm, uncertainty, pipeline, app (+ data.py loaders)
scripts/   setup, train, eval, feedback, reset_live
frontend/  React + Vite + Recharts chat UI
report/    Streamlit + Plotly report
data/      queries.csv, prices.yaml, model_runs.csv, router.pkl, app.db, chroma/
```
