# Evaluate on the TEST split only and write results to eval_results (+ print a table).
#  - always-small / always-medium / always-large: computed from model_runs.csv (no new API calls)
#  - large + cache, and ours: run the real pipeline, starting from an empty cache
#  - τ sweep points (cost vs accuracy, routing + escalation, no cache) for the Pareto curve
# Usage: MOCK_LLM=1 python scripts/eval.py
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend  # noqa: F401  (sets the OpenMP fix before LightGBM / torch load)
import pandas as pd

from backend import cache, config, db, llm, pipeline, router
from backend.data import load_queries, load_runs
from scripts.train import build_features, sweep


def metrics(name, correct, total_cost, avg_ms, hits=0, wrong_hits=0, tau=(None, None)):
    n_correct = sum(correct)
    return {"strategy": name, "accuracy": n_correct / len(correct), "total_cost": total_cost,
            "cost_per_correct": total_cost / n_correct if n_correct else None,
            "avg_ms": avg_ms, "cache_hit_rate": hits / len(correct),
            "wrong_hit_rate": wrong_hits / hits if hits else 0.0,
            "tau_small": tau[0], "tau_medium": tau[1]}


def run_pipeline(name, test, force_tier=None):
    """Run every test query through the real pipeline, in file order (originals before paraphrases)."""
    cache.clear()
    rows = []
    for r in test.itertuples():
        out = pipeline.run(r.query, r.task_type, force_tier=force_tier, source=f"eval:{name}")
        rows.append({"id": r.id, "cache_expected": r.cache_expected, "cache": out["cache"],
                     "model": out["model_used"], "escalations": out["escalations"], "cost": out["cost"],
                     "ms": out["total_ms"], "correct": llm.grade(r.task_type, out["answer"], r.reference_answer)})
    cache.clear()  # don't leave test answers in the live cache
    df = pd.DataFrame(rows)
    hits = df[df.cache != "none"]
    return df, metrics(name, df.correct.tolist(), df.cost.sum(), df.ms.mean(), len(hits), int((~hits.correct).sum()))


def main():
    queries, runs = load_queries(), load_runs()
    test = queries[(queries.split == "test") & queries.id.isin(runs["correct"].index)]
    print(f"TEST split: {len(test)} queries  (mock={config.MOCK_LLM}, router={router.version()})\n")

    results = [metrics(f"always-{t}", runs["correct"].loc[test.id, t].tolist(), runs["cost"].loc[test.id, t].sum(),
                       runs["ms"].loc[test.id, t].mean()) for t in config.TIERS]
    large_cache_df, m = run_pipeline("large+cache", test, force_tier="large")
    results.append(m)
    ours_df, m = run_pipeline("ours", test)
    b = router.bundle()
    m["tau_small"], m["tau_medium"] = (b["tau_small"], b["tau_medium"]) if b else (None, None)
    results.append(m)

    large_cost = results[2]["total_cost"]
    for r in results:
        r["savings_pct"] = 100 * (1 - r["total_cost"] / large_cost)

    # Pareto points: every (τ_small, τ_medium) on test, with escalation replayed from model_runs.csv
    points = []
    if b:
        X = build_features(test, exclude_group=False)
        points = sweep(b["models"]["small"].predict_proba(X)[:, 1], b["models"]["medium"].predict_proba(X)[:, 1],
                       list(test.id), runs)

    run_ts = datetime.now().isoformat(timespec="seconds")
    for r in results:
        db.insert("eval_results", {"run_ts": run_ts, **r})
    for p in points:
        n_correct = p["accuracy"] * len(test)
        db.insert("eval_results", {"run_ts": run_ts, "strategy": "sweep", "accuracy": p["accuracy"],
                                   "total_cost": p["cost"], "cost_per_correct": p["cost"] / n_correct if n_correct else None,
                                   "savings_pct": 100 * (1 - p["cost"] / large_cost),
                                   "tau_small": p["tau_small"], "tau_medium": p["tau_medium"]})

    table = pd.DataFrame(results).set_index("strategy")
    table["accuracy"] = (100 * table["accuracy"]).round(1).astype(str) + "%"
    table["savings_pct"] = table["savings_pct"].round(1).astype(str) + "%"
    table["cache_hit_rate"] = (100 * table["cache_hit_rate"]).round(1).astype(str) + "%"
    table["wrong_hit_rate"] = (100 * table["wrong_hit_rate"]).round(1).astype(str) + "%"
    table["avg_ms"] = table["avg_ms"].round(0).astype(int)
    for col in ["total_cost", "cost_per_correct"]:
        table[col] = table[col].map(lambda v: f"{v:.6f}")
    cols = ["accuracy", "total_cost", "cost_per_correct", "savings_pct", "avg_ms", "cache_hit_rate", "wrong_hit_rate"]
    print(f"Costs in {config.CURRENCY}\n" + table[cols].to_string())

    print("\nOURS details")
    print("  final model mix:", dict(Counter(ours_df.model)))
    print("  escalations:    ", dict(Counter(ours_df.escalations)))
    for label, df in [("large+cache", large_cache_df), ("ours", ours_df)]:
        para, changed = df[df.cache_expected == "reuse"], df[df.cache_expected == "no_reuse"]
        print(f"  cache test [{label}]: paraphrases reused {(para.cache != 'none').sum()}/{len(para)}, "
              f"number-changed wrongly reused {(changed.cache != 'none').sum()}/{len(changed)}")
    print(f"\nsaved {len(results)} strategies + {len(points)} τ sweep points to eval_results (run_ts={run_ts})")


if __name__ == "__main__":
    main()
