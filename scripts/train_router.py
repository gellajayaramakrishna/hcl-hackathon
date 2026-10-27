"""Step 2 of training: learn to predict, from the query text alone, which models will get it right.

Uses the labels from scripts/run_models.py and the splits from data/splits.json:
  train -> fit one classifier per tier: P(tier answers correctly | query)
  tune  -> pick the probability threshold with the best accuracy/cost trade-off
  test  -> report the final router against fixed baselines, once

Output: models/router.joblib (load with src.router.Router.load) and models/router_report.json

Usage:
    python scripts/train_router.py
    python scripts/train_router.py --cost-weight 1.0   # care more about cost
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_loader import load_queries  # noqa: E402
from src.features import build_classifier  # noqa: E402
from src.llm_client import load_models  # noqa: E402
from src.router import ConstantClassifier, Router  # noqa: E402
from src.runs import load_runs  # noqa: E402

THRESHOLDS = np.round(np.arange(0.30, 0.96, 0.05), 2)


def evaluate(choices, ids, correct, cost) -> dict:
    """Accuracy and total cost if each query is sent to the chosen tier (looked up from real runs)."""
    picked_correct = [correct.at[i, t] for i, t in zip(ids, choices)]
    picked_cost = [cost.at[i, t] for i, t in zip(ids, choices)]
    return {"accuracy": float(np.mean(picked_correct)), "cost_usd": float(np.sum(picked_cost)),
            "tier_share": {t: choices.count(t) / len(choices) for t in correct.columns}}


def score(result, max_cost, cost_weight) -> float:
    """What threshold tuning maximises: accuracy minus a penalty for cost.
    Cost is measured relative to always using the largest model, so both terms are 0-1."""
    return result["accuracy"] - cost_weight * result["cost_usd"] / max_cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", default=ROOT / "data" / "queries.csv")
    parser.add_argument("--splits", default=ROOT / "data" / "splits.json")
    parser.add_argument("--runs", default=ROOT / "data" / "model_runs.jsonl")
    parser.add_argument("--models", default=ROOT / "config" / "models.yaml")
    parser.add_argument("--out-dir", default=ROOT / "models")
    parser.add_argument("--cost-weight", type=float, default=0.5,
                        help="0 = accuracy only; higher = willing to lose more accuracy to save cost")
    args = parser.parse_args()

    tiers = list(load_models(args.models))  # cheapest first
    queries = load_queries(args.queries).set_index("id")
    correct, cost = load_runs(args.runs, tiers)
    splits = json.loads(Path(args.splits).read_text())
    ids = {name: [i for i in members if i in correct.index] for name, members in splits.items()}
    print("queries with runs for every tier: " + ", ".join(f"{k}={len(v)}" for k, v in ids.items()))
    if min(len(v) for v in ids.values()) == 0:
        sys.exit("some split has no fully-labelled queries -- finish scripts/run_models.py first")

    # Label summary: how often each tier is right, and the "cheapest correct tier" per query.
    cheapest = correct.apply(lambda r: next((t for t in tiers if r[t] == 1), "none"), axis=1)
    print("\naccuracy per tier (all labelled queries): "
          + ", ".join(f"{t}={correct[t].mean():.1%}" for t in tiers))
    print("cheapest correct tier: " + ", ".join(f"{k}={v}" for k, v in cheapest.value_counts().items()))

    # --- train: one classifier per tier ---------------------------------------------------
    train_text = queries.loc[ids["train"], "query"].tolist()
    classifiers = {}
    for tier in tiers:
        y = correct.loc[ids["train"], tier].astype(int).to_numpy()
        if y.min() == y.max():  # tier always right (or always wrong): nothing to learn
            classifiers[tier] = ConstantClassifier(float(y[0]))
        else:
            classifiers[tier] = build_classifier().fit(train_text, y)
    router = Router(tiers, classifiers, threshold=0.5)

    # --- tune: choose the threshold ------------------------------------------------------
    tune_text = queries.loc[ids["tune"], "query"].tolist()
    max_cost_tune = cost.loc[ids["tune"], tiers[-1]].sum()
    print(f"\nthreshold sweep on tune (cost_weight={args.cost_weight}):")
    print(f"{'thr':>5} {'acc':>7} {'cost $':>10} {'score':>7}  share " + "/".join(tiers))
    best = None
    for thr in THRESHOLDS:
        result = evaluate(router.choose(tune_text, thr), ids["tune"], correct, cost)
        s = score(result, max_cost_tune, args.cost_weight)
        share = "/".join(f"{result['tier_share'][t]:.0%}" for t in tiers)
        print(f"{thr:5.2f} {result['accuracy']:7.1%} {result['cost_usd']:10.5f} {s:7.3f}  {share}")
        if best is None or s > best[0]:
            best = (s, float(thr))
    router.threshold = best[1]
    print(f"chosen threshold: {router.threshold}")

    # --- test: compare against baselines, once -------------------------------------------
    test_ids = ids["test"]
    test_text = queries.loc[test_ids, "query"].tolist()
    max_cost_test = cost.loc[test_ids, tiers[-1]].sum()
    report = {f"always_{t}": evaluate([t] * len(test_ids), test_ids, correct, cost) for t in tiers}
    # Oracle = perfect knowledge of the cheapest correct tier: the best any router could do.
    oracle = [cheapest[i] if cheapest[i] != "none" else tiers[0] for i in test_ids]
    report["oracle"] = evaluate(oracle, test_ids, correct, cost)
    report["router"] = evaluate(router.choose(test_text), test_ids, correct, cost)

    print(f"\ntest split ({len(test_ids)} queries):")
    print(f"{'policy':14s} {'acc':>7} {'cost $':>10} {'score':>7}")
    for name, r in report.items():
        r["score"] = score(r, max_cost_test, args.cost_weight)
        print(f"{name:14s} {r['accuracy']:7.1%} {r['cost_usd']:10.5f} {r['score']:7.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    router.save(out_dir / "router.joblib")
    report["meta"] = {"threshold": router.threshold, "cost_weight": args.cost_weight,
                      "tiers": tiers, "n": {k: len(v) for k, v in ids.items()}}
    (out_dir / "router_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out_dir / 'router.joblib'} and router_report.json")


if __name__ == "__main__":
    main()
