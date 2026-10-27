# Learn from 👍/👎 feedback (only rows not processed before):
#  - 👎 on any answer → its cache entry is marked confident=False (never reused again)
#  - feedback on model answers + escalation results → new labeled rows in Chroma "examples",
#    which changes the kNN features (and so routing) immediately. Questions from queries.csv are
#    skipped: they already have exact labels, and tune/test must never enter "examples".
#  - --retrain: train a new router, activate it only if it beats the current one on the TUNE split
# Usage: python scripts/feedback.py [--retrain]
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend  # noqa: F401  (sets the OpenMP fix before LightGBM / torch load)

from backend import cache, config, db, router
from backend.clean import clean
from backend.data import load_queries
from backend.embed import embed

CURSOR = config.DATA / "feedback_cursor.txt"  # id of the last processed feedback row


def labels_for(final, tried, thumbs):
    """Correct/incorrect per tier from one piece of feedback, or None if a label is unknown."""
    up = thumbs == 1
    labels = {}
    for t in config.TIERS:
        if t == final:
            labels[t] = int(up)
        elif config.TIERS.index(t) < config.TIERS.index(final):
            # smaller model: tried → it was unsure and escalated (count as wrong);
            # not tried → wrong if even the bigger model failed, unknown if it succeeded
            labels[t] = 0 if t in tried or not up else None
        else:
            labels[t] = 1  # bigger model never tried: assume it would be right
    return None if None in labels.values() else labels


def process_new_feedback():
    last = int(CURSOR.read_text()) if CURSOR.exists() else 0
    rows = db.query("""SELECT f.id, f.thumbs, r.trace_id, r.query, r.model_used, r.cache_type, r.cache_key
                       FROM feedback f JOIN requests r USING (trace_id) WHERE f.id > ? ORDER BY f.id""", (last,))
    dataset = {clean(q)["text"] for q in load_queries()["query"]}
    unconfident, added, skipped, from_dataset = 0, 0, 0, 0
    for r in rows:
        c = clean(r["query"])
        if r["thumbs"] == -1:  # stop reusing this answer, whether it came from the cache or a model
            cache.mark_unconfident(r["cache_key"] or cache.key(c))
            unconfident += 1
        if r["cache_type"] != "none":
            continue  # no model ran, so nothing to learn about the tiers
        if c["text"] in dataset:
            from_dataset += 1
            continue
        tried = {x["model"] for x in db.query("SELECT model FROM llm_calls WHERE trace_id = ? AND ok = 1",
                                              (r["trace_id"],))}
        labels = labels_for(r["model_used"], tried, r["thumbs"])
        if labels is None:
            skipped += 1
            continue
        cache.collection("examples").upsert(
            ids=[f"fb-{r['trace_id']}"], embeddings=[embed(c["text"])], documents=[c["text"]],
            metadatas=[{f"{t}_correct": v for t, v in labels.items()} | {"subject": "feedback", "task_type": c["task_type"]}])
        added += 1
    if rows:
        CURSOR.write_text(str(rows[-1]["id"]))
    print(f"{len(rows)} new feedback rows: {unconfident} answers marked not reusable, "
          f"{added} examples added, {skipped} skipped (not enough information), "
          f"{from_dataset} skipped (already in queries.csv)")


def tune_score(bundle, X_tune, ids, runs):
    """Accuracy and cost of a router (its models + τ) on the TUNE split, with escalation replayed."""
    from scripts.train import replay
    p_small = bundle["models"]["small"].predict_proba(X_tune)[:, 1]
    p_medium = bundle["models"]["medium"].predict_proba(X_tune)[:, 1]
    results = [replay(router.pick(ps, pm, bundle["tau_small"], bundle["tau_medium"])[0], runs, qid)
               for ps, pm, qid in zip(p_small, p_medium, ids)]
    return sum(r[1] for r in results) / len(results), sum(r[2] for r in results)


def retrain():
    from backend.data import load_queries, load_runs
    from scripts.train import activate, build_features, train_router
    current = router.bundle()
    new, best = train_router()
    queries, runs = load_queries(), load_runs()
    tune = queries[(queries.split == "tune") & queries.id.isin(runs["correct"].index)]
    X_tune = build_features(tune, exclude_group=False)
    new_acc, new_cost = tune_score(new, X_tune, list(tune.id), runs)
    print(f"new router on tune:     accuracy={new_acc:.3f} cost={new_cost:.5f}")
    if current:
        cur_acc, cur_cost = tune_score(current, X_tune, list(tune.id), runs)
        print(f"current router on tune: accuracy={cur_acc:.3f} cost={cur_cost:.5f}")
        # activate only on a clear win: not worse on either, better on at least one
        if not (new_acc >= cur_acc and new_cost <= cur_cost and (new_acc, new_cost) != (cur_acc, cur_cost)):
            print(f"kept current router {current['version']} (new one does not beat it)")
            return
    activate(new, best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true", help="retrain and activate the router if it is better")
    args = ap.parse_args()
    process_new_feedback()
    if args.retrain:
        retrain()
