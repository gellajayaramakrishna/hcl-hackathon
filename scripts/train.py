# Offline step [ML]: train the router. Two LightGBM classifiers (will small / medium be correct?),
# calibrated with isotonic regression, then sweep τ_small / τ_medium and keep the cheapest setting
# whose accuracy is ≥ QUALITY_TARGET × always-large accuracy. The sweep uses TUNE plus out-of-fold
# predictions on TRAIN (~400 questions instead of ~100, so τ is less driven by luck). TEST is never used.
# The sweep replays escalation (step 7) from model_runs.csv, so τ is tuned for the real pipeline.
# Usage: python scripts/train.py
import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend  # noqa: F401  (sets OMP_NUM_THREADS before LightGBM / torch load)
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold, cross_val_predict

from backend import config, db, router
from backend.clean import clean
from backend.data import load_queries, load_runs
from backend.embed import embed

GRID = np.round(np.arange(0.30, 1.0001, 0.05), 2)


def groups_of(df):
    """A question and its paraphrase / number-changed copies share one group."""
    return df["cache_group"].where(df["cache_group"] != "", df["id"]).to_numpy()


def build_features(df, exclude_group):
    """Feature matrix for the rows of df. exclude_group: a train row must not see itself
    (or its paraphrases) among its kNN neighbours, since they are in "examples"."""
    cleaned = [clean(q, t) for q, t in zip(df["query"], df["task_type"])]
    vectors = embed([c["text"] for c in cleaned])
    groups = groups_of(df)
    rows = []
    for qid, g, c, v in zip(df["id"], groups, cleaned, vectors):
        exclude = set(df["id"][groups == g]) if exclude_group else set()
        rows.append(router.features(c, v, router.knn_stats(v, exclude_ids=exclude)))
    return np.array(rows)


def make_model():
    lgbm = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                          colsample_bytree=0.5, random_state=42, verbose=-1)
    return CalibratedClassifierCV(lgbm, method="isotonic", cv=5)


def replay(start, runs, qid):
    """Replay steps 6-7 offline for one question: start at `start`, escalate while unsure.
    Returns (final tier, correct, cost, escalations)."""
    t, cost, jumps = config.TIERS.index(start), 0.0, 0
    while True:
        tier = config.TIERS[t]
        cost += runs["cost"].at[qid, tier] + runs["check_cost"].at[qid, tier]
        if not runs["unsure"].at[qid, tier] or tier == "large" or jumps == config.MAX_ESCALATIONS:
            return tier, bool(runs["correct"].at[qid, tier]), cost, jumps
        t, jumps = t + 1, jumps + 1


def sweep(p_small, p_medium, ids, runs):
    """Accuracy and cost (with escalation) of every (τ_small, τ_medium) pair."""
    points = []
    for ts in GRID:
        for tm in GRID:
            starts = [router.pick(ps, pm, ts, tm)[0] for ps, pm in zip(p_small, p_medium)]
            results = [replay(s, runs, qid) for s, qid in zip(starts, ids)]
            points.append({"tau_small": float(ts), "tau_medium": float(tm),
                           "accuracy": float(np.mean([r[1] for r in results])),
                           "cost": float(sum(r[2] for r in results)),
                           "mix": {t: starts.count(t) for t in config.TIERS}})
    return points


def choose_tau(points, large_acc, n):
    """Cheapest point whose accuracy, minus TUNE_SAFETY_SE standard errors, meets the target."""
    def lower_bound(acc):
        return acc - config.TUNE_SAFETY_SE * np.sqrt(acc * (1 - acc) / n)
    ok = [p for p in points if lower_bound(p["accuracy"]) >= config.QUALITY_TARGET * large_acc]
    if not ok:  # target unreachable: take the most accurate setting
        return max(points, key=lambda p: (p["accuracy"], -p["cost"]))
    return min(ok, key=lambda p: (p["cost"], -p["accuracy"]))


def train_router():
    """Train on TRAIN, tune τ on TUNE + out-of-fold TRAIN. Returns (bundle, chosen point)."""
    queries = load_queries()
    runs = load_runs()
    correct = runs["correct"]
    train = queries[(queries.split == "train") & queries.id.isin(correct.index)]
    tune = queries[(queries.split == "tune") & queries.id.isin(correct.index)]

    X_train, X_tune = build_features(train, exclude_group=True), build_features(tune, exclude_group=False)
    folds = GroupKFold(5)
    models, probs = {}, {}
    for t in ["small", "medium"]:
        y = correct.loc[train.id, t].astype(int).to_numpy()
        models[t] = make_model().fit(X_train, y)
        oof = cross_val_predict(make_model(), X_train, y, groups=groups_of(train), cv=folds,
                                method="predict_proba")[:, 1]  # each train row scored by a model that never saw it
        probs[t] = np.concatenate([oof, models[t].predict_proba(X_tune)[:, 1]])

    ids = list(train.id) + list(tune.id)
    large_acc = correct.loc[ids, "large"].mean()
    best = choose_tau(sweep(probs["small"], probs["medium"], ids, runs), large_acc, len(ids))

    print(f"train={len(train)} tune={len(tune)} sweep set={len(ids)} (out-of-fold train + tune)")
    print(f"sweep always-large: accuracy={large_acc:.3f} cost={runs['cost'].loc[ids, 'large'].sum():.5f}")
    print(f"sweep router:       accuracy={best['accuracy']:.3f} cost={best['cost']:.5f} "
          f"τ_small={best['tau_small']} τ_medium={best['tau_medium']} mix={best['mix']}")
    version = "lgbm-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    bundle = {"version": version, "models": models,
              "tau_small": best["tau_small"], "tau_medium": best["tau_medium"]}
    return bundle, best


def activate(bundle, best):
    """Save router.pkl and mark this version active in router_versions."""
    config.ROUTER_PKL.write_bytes(pickle.dumps(bundle))
    with db.connect() as conn:
        conn.execute("UPDATE router_versions SET is_active = 0")
    db.insert("router_versions", {"version": bundle["version"], "ts": datetime.now().isoformat(timespec="seconds"),
                                  "tune_accuracy": best["accuracy"], "tune_cost": best["cost"], "is_active": 1})
    router.reload()
    print(f"saved {config.ROUTER_PKL.name}, active version {bundle['version']}")


if __name__ == "__main__":
    activate(*train_router())
