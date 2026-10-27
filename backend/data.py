# Load queries.csv. If it has no "split" column, take the split from data/splits.json.
import json

import pandas as pd

from backend import config


def load_queries():
    df = pd.read_csv(config.QUERIES_CSV, dtype=str).fillna("")
    if "split" not in df.columns or (df["split"] == "").all():
        splits = json.loads(config.SPLITS_JSON.read_text())
        df["split"] = df["id"].map({i: name for name, ids in splits.items() for i in ids}).fillna("")
    return df


def load_runs():
    """model_runs.csv as tables indexed by id with one column per tier (small/medium/large):
    correct, cost, unsure, check_cost (extra samples), ms, check_ms."""
    runs = pd.read_csv(config.MODEL_RUNS_CSV, dtype={"id": str, "answer": str})
    return {col: runs.pivot(index="id", columns="model", values=col)[config.TIERS]
            for col in ["correct", "cost", "unsure", "check_cost", "ms", "check_ms"]}
