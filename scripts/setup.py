# Offline step [Gen AI]: run every train/tune/test question on all 3 models, grade the answers
# against the reference, save data/model_runs.csv (resumable), and load TRAIN questions into the
# Chroma "examples" collection with their per-model correct labels (used for kNN features).
# It also records the unsure check (step 7) per answer so escalation can be replayed offline.
# Usage: MOCK_LLM=1 python scripts/setup.py [--limit N]
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from backend import cache, config, llm, uncertainty
from backend.clean import clean
from backend.data import load_queries
from backend.embed import embed


def run_models(queries, limit):
    done = set()
    if config.MODEL_RUNS_CSV.exists():
        old = pd.read_csv(config.MODEL_RUNS_CSV, dtype={"id": str})
        done = set(zip(old["id"], old["model"]))
    todo = [r for r in queries.itertuples() if any((r.id, t) not in done for t in config.TIERS)]
    todo = todo[:limit] if limit else todo
    print(f"{len(done)} runs already done, {len(todo)} questions to run (mock={config.MOCK_LLM})")

    for n, r in enumerate(todo, 1):
        text = clean(r.query, r.task_type)["redacted"]  # exactly what the pipeline sends
        rows = []
        for tier in config.TIERS:
            if (r.id, tier) in done:
                continue
            res = llm.ask(tier, text, r.task_type, fallback=False)
            last = res["calls"][-1]
            # same unsure check as the pipeline, so tuning/eval can replay escalation offline
            unsure, _, extra = uncertainty.check(tier, text, r.task_type, res)
            rows.append({"id": r.id, "model": tier, "answer": res["answer"],
                         "correct": llm.grade(r.task_type, res["answer"], r.reference_answer),
                         "input_tokens": last["input_tokens"], "output_tokens": last["output_tokens"],
                         "cost": last["cost"], "ms": round(last["ms"], 1), "ok": res["ok"],
                         "unsure": unsure, "check_cost": sum(c["cost"] for c in extra),
                         "check_ms": max((c["ms"] for c in extra), default=0.0)})
        pd.DataFrame(rows).to_csv(config.MODEL_RUNS_CSV, mode="a", index=False,
                                  header=not config.MODEL_RUNS_CSV.exists())
        if n % 50 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}")


def build_examples(queries):
    """TRAIN split only: never put tune/test questions into examples."""
    runs = pd.read_csv(config.MODEL_RUNS_CSV, dtype={"id": str})
    correct = runs.pivot_table(index="id", columns="model", values="correct", aggfunc="last")
    train = queries[(queries.split == "train") & queries.id.isin(correct.dropna().index)]
    cleaned = [clean(q, t) for q, t in zip(train["query"], train["task_type"])]
    vectors = embed([c["text"] for c in cleaned])
    metas = [{f"{t}_correct": int(correct.loc[i, t]) for t in config.TIERS} | {"subject": s, "task_type": tt}
             for i, s, tt in zip(train["id"], train["subject"], train["task_type"])]
    cache.collection("examples").upsert(ids=list(train["id"]), embeddings=vectors,
                                        documents=[c["text"] for c in cleaned], metadatas=metas)
    print(f"examples collection: {cache.collection('examples').count()} train questions")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only run this many new questions")
    args = ap.parse_args()
    queries = load_queries()
    queries = queries[queries.split.isin(["train", "tune", "test"])]
    run_models(queries, args.limit)
    build_examples(queries)
    runs = pd.read_csv(config.MODEL_RUNS_CSV)
    print("accuracy per model:", runs.groupby("model")["correct"].mean().round(3).to_dict())
