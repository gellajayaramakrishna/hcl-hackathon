"""Read data/model_runs.jsonl (written by scripts/run_models.py) into id x tier tables."""
import pandas as pd


def load_runs(path, tiers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two id x tier tables: was each tier correct (1.0/0.0), and what did it cost (USD).
    Only queries that every tier has answered are kept."""
    runs = pd.read_json(path, lines=True, dtype={"id": str})
    runs = runs.drop_duplicates(["id", "tier"], keep="last")
    correct = runs.pivot(index="id", columns="tier", values="correct").astype(float)
    cost = runs.pivot(index="id", columns="tier", values="cost_usd")
    missing = [t for t in tiers if t not in correct.columns]
    if missing:
        raise ValueError(f"no runs for tiers {missing}; run scripts/run_models.py first")
    complete = correct[tiers].notna().all(axis=1)
    return correct.loc[complete, tiers], cost.loc[complete, tiers]
