"""Step 1b: measure difficulty from model behaviour, and check it against the proxy label.

Run after scripts/run_models.py. For every query:
  empirical_difficulty = easy (cheapest model right) / medium (only a middle model or above)
                         / hard (only the largest model, or none)
  fail_rate            = fraction of models that got it wrong (0.0 ... 1.0)

Then it answers "is our build-time proxy label any good?":
  * proxy label vs empirical label (agreement table)
  * average fail rate per proxy label -- should rise from easy to hard
  * for math, rank correlation between the operation count and the fail rate

Output: data/empirical_difficulty.csv
Usage:  python scripts/empirical_difficulty.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_loader import load_queries  # noqa: E402
from src.difficulty import count_operations, empirical_difficulty, fail_rate  # noqa: E402
from src.llm_client import load_models  # noqa: E402
from src.runs import load_runs  # noqa: E402

LEVELS = ["easy", "medium", "hard"]


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation: +1 = when one goes up the other always goes up, 0 = no relation."""
    return a.rank().corr(b.rank())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", default=ROOT / "data" / "queries.csv")
    parser.add_argument("--runs", default=ROOT / "data" / "model_runs.jsonl")
    parser.add_argument("--models", default=ROOT / "config" / "models.yaml")
    parser.add_argument("--out", default=ROOT / "data" / "empirical_difficulty.csv")
    args = parser.parse_args()

    tiers = list(load_models(args.models))  # cheapest first
    queries = load_queries(args.queries).set_index("id")
    correct, _ = load_runs(args.runs, tiers)
    df = queries.loc[correct.index, ["task_type", "difficulty", "source", "solution"]].copy()
    df = df.rename(columns={"difficulty": "proxy_difficulty"})

    rows = correct.astype(bool).to_dict("index")  # id -> {tier: True/False}
    df["empirical_difficulty"] = [empirical_difficulty(rows[i], tiers) for i in df.index]
    df["fail_rate"] = [round(fail_rate(rows[i]), 3) for i in df.index]
    df["n_operations"] = [count_operations(s) if s else None for s in df["solution"]]
    for t in tiers:
        df[f"correct_{t}"] = correct[t].astype(int)

    out_cols = ["task_type", "proxy_difficulty", "n_operations", "empirical_difficulty",
                "fail_rate", *[f"correct_{t}" for t in tiers]]
    df[out_cols].to_csv(args.out, index_label="id")
    print(f"wrote {len(df)} rows to {args.out}")

    # Compare on base items only: derived items are near-copies and would count twice.
    base = df[df.source != "derived"]
    print(f"\ncomparing proxy vs empirical on {len(base)} base queries")
    for task, part in base.groupby("task_type"):
        print(f"\n=== {task} ===")
        print("empirical difficulty: " + ", ".join(
            f"{k}={v}" for k, v in part.empirical_difficulty.value_counts().reindex(LEVELS, fill_value=0).items()))

        proxy_levels = [lv for lv in LEVELS if lv in set(part.proxy_difficulty)]
        table = pd.crosstab(part.proxy_difficulty, part.empirical_difficulty)
        print("\nrows = proxy label, columns = empirical label")
        print(table.reindex(index=proxy_levels, columns=LEVELS, fill_value=0).to_string())

        mean_fail = part.groupby("proxy_difficulty").fail_rate.mean().reindex(proxy_levels)
        print("\naverage fail rate by proxy label (should increase): "
              + ", ".join(f"{k}={v:.2f}" for k, v in mean_fail.items()))
        ordinal = part.proxy_difficulty.map({lv: i for i, lv in enumerate(LEVELS)})
        print(f"rank correlation proxy label vs fail rate: {spearman(ordinal, part.fail_rate):+.2f}")

        if task == "math":
            ops = part.n_operations.astype(float)
            by_ops = part.groupby(ops).fail_rate.agg(["mean", "size"])
            print("\nfail rate by number of operations:")
            print(by_ops.rename(columns={"mean": "fail_rate", "size": "n"}).round(2).to_string())
            print(f"rank correlation n_operations vs fail rate: {spearman(ops, part.fail_rate):+.2f}")

    print("\nHow to read this: a clearly positive correlation (e.g. above +0.3) means the proxy "
          "tracks real model failures; near 0 means the proxy is not measuring difficulty.")


if __name__ == "__main__":
    main()
