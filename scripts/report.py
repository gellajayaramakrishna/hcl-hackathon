"""Write a full training report: data, model runs (accuracy, latency, tokens, cost),
difficulty analysis, classifier quality and router results vs baselines.

Everything is computed from the saved files, so the report always matches them.

Usage:
    python scripts/report.py --models config/models_2tier.yaml \
        --runs data/model_runs_2tier.jsonl --router models/2tier/router.joblib \
        --out reports/training_report_2tier.md
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_loader import load_queries  # noqa: E402
from src.llm_client import load_models  # noqa: E402
from src.router import Router  # noqa: E402


def md(df: pd.DataFrame) -> str:
    return df.to_markdown() if not df.empty else "_none_"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", default=ROOT / "data" / "queries.csv")
    parser.add_argument("--splits", default=ROOT / "data" / "splits.json")
    parser.add_argument("--models", default=ROOT / "config" / "models.yaml")
    parser.add_argument("--runs", default=ROOT / "data" / "model_runs.jsonl")
    parser.add_argument("--router", default=ROOT / "models" / "router.joblib")
    parser.add_argument("--out", default=ROOT / "reports" / "training_report.md")
    args = parser.parse_args()

    models = load_models(args.models)
    tiers = list(models)
    queries = load_queries(args.queries).set_index("id")
    splits = json.loads(Path(args.splits).read_text())
    split_of = {i: name for name, ids in splits.items() for i in ids}
    runs = pd.read_json(args.runs, lines=True, dtype={"id": str}).drop_duplicates(["id", "tier"], keep="last")
    runs = runs.join(queries[["task_type", "difficulty", "source"]], on="id")
    runs["split"] = runs.id.map(split_of)
    out = []

    # ---------------------------------------------------------------- 1. data
    out.append("# Router training report\n")
    out.append(f"Models: {', '.join(f'**{t}** = `{m['name']}`' for t, m in models.items())}  \n"
               f"Runs file: `{Path(args.runs).name}` ({len(runs)} runs)\n")
    out.append("## 1. Data\n")
    q = queries.assign(split=queries.index.map(split_of))
    out.append(md(pd.crosstab([q.task_type, q.difficulty], q.split, margins=True, margins_name="total")) + "\n")
    out.append("\n" + md(pd.crosstab(q.source, q.cache_expected, margins=True, margins_name="total")) + "\n")

    # ---------------------------------------------------------------- 2. model runs
    out.append("\n## 2. Model runs (all 500 queries)\n")
    g = runs.groupby(["tier", "task_type"])
    stats = pd.DataFrame({
        "n": g.size(),
        "accuracy": g.correct.mean(),
        "unparsed_answer": g.answer.apply(lambda s: s.isna().mean()),
        "tokens_in_avg": g.tokens_in.mean(),
        "tokens_out_avg": g.tokens_out.mean(),
        "hit_output_cap": g.tokens_out.apply(lambda s: (s >= 512).mean()),
        "latency_avg_s": g.latency_s.mean(),
        "latency_p50_s": g.latency_s.median(),
        "latency_p95_s": g.latency_s.quantile(0.95),
        "latency_total_min": g.latency_s.sum() / 60,
        "cost_total_usd": g.cost_usd.sum(),
        "cost_per_query_usd": g.cost_usd.mean(),
    })
    stats["cost_per_correct_usd"] = stats.cost_total_usd / (stats.accuracy * stats.n)
    out.append(md(stats.reindex(tiers, level=0).round(5)) + "\n")

    overall = runs.groupby("tier").agg(accuracy=("correct", "mean"), latency_total_min=("latency_s", lambda s: s.sum() / 60),
                                       cost_total_usd=("cost_usd", "sum")).reindex(tiers)
    out.append("\nPer model, both task types together:\n\n" + md(overall.round(5)) + "\n")
    out.append(f"\nTotal labelling time: **{runs.latency_s.sum() / 60:.1f} min** for {len(runs)} calls; "
               f"total (placeholder) cost: **${runs.cost_usd.sum():.4f}**.\n")

    acc = runs.pivot_table(index=["task_type", "difficulty"], columns="tier", values="correct", aggfunc="mean")
    out.append("\nAccuracy by proxy difficulty:\n\n" + md(acc[tiers].round(3)) + "\n")

    wide = runs.pivot(index="id", columns="tier", values="correct").astype(int)
    pattern = wide[tiers].astype(str).agg("".join, axis=1).value_counts()
    pattern.index = [" / ".join(f"{t}={'right' if c == '1' else 'wrong'}" for t, c in zip(tiers, p)) for p in pattern.index]
    out.append("\nWho gets each query right:\n\n" + md(pattern.rename("queries").to_frame()) + "\n")

    # ---------------------------------------------------------------- 3. classifier quality
    router = Router.load(args.router)
    out.append("\n## 3. Classifier quality (AUC: 0.5 = random, 1.0 = perfect)\n")
    rows = {}
    for name, ids in splits.items():
        ids = [i for i in ids if i in wide.index]
        probs = router.probabilities(queries.loc[ids, "query"].tolist())
        rows[name] = {t: roc_auc_score(wide.loc[ids, t], probs[t]) if wide.loc[ids, t].nunique() > 1 else np.nan
                      for t in tiers}
    out.append(md(pd.DataFrame(rows).T.round(3)) + "\n")
    out.append("\nA big drop from train to tune/test means overfitting (memorising the training queries).\n")

    # ---------------------------------------------------------------- 4. router vs baselines
    test = [i for i in splits["test"] if i in wide.index]
    text = queries.loc[test, "query"].tolist()
    start = time.perf_counter()
    choice = router.choose(text)
    router_ms = (time.perf_counter() - start) / len(text) * 1000
    lookup = runs.set_index(["id", "tier"])
    cheapest = wide.apply(lambda r: next((t for t in tiers if r[t] == 1), tiers[0]), axis=1)
    policies = {f"always_{t}": [t] * len(test) for t in tiers}
    policies["oracle"] = [cheapest[i] for i in test]
    policies["router"] = choice
    max_cost = sum(lookup.loc[(i, tiers[-1]), "cost_usd"] for i in test)
    report = json.loads((Path(args.router).parent / "router_report.json").read_text())
    cost_weight = report["meta"]["cost_weight"]

    table = {}
    for name, picks in policies.items():
        sel = lookup.loc[list(zip(test, picks))]
        table[name] = {
            "accuracy": sel.correct.mean(),
            "acc_mcq": sel.correct[np.array(queries.loc[test, "task_type"]) == "mcq"].mean(),
            "acc_math": sel.correct[np.array(queries.loc[test, "task_type"]) == "math"].mean(),
            "cost_usd": sel.cost_usd.sum(),
            "cost_vs_always_large": sel.cost_usd.sum() / max_cost,
            "latency_total_s": sel.latency_s.sum(),
            "latency_avg_s": sel.latency_s.mean(),
            "score": sel.correct.mean() - cost_weight * sel.cost_usd.sum() / max_cost,
            **{f"share_{t}": picks.count(t) / len(picks) for t in tiers},
        }
    out.append(f"\n## 4. Router vs baselines (test split, {len(test)} queries)\n")
    out.append(f"Threshold {router.threshold} (chosen on tune), score = accuracy - {cost_weight} x cost/cost(always largest). "
               f"Router decision time: {router_ms:.2f} ms per query (not included in latency below).\n\n")
    out.append(md(pd.DataFrame(table).T.round(4)) + "\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out))
    print("\n".join(out))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
