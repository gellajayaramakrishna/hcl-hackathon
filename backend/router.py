# Step 5: choose the model tier for a query.
# Uses data/router.pkl (2 calibrated LightGBM classifiers) if present; otherwise falls back to
# kNN success rates from the "examples" collection; if that is empty too, to simple rules.
import pickle

from backend import cache, config

_bundle = None
_loaded_mtime = None  # router.pkl is re-read when the file changes (e.g. after feedback.py --retrain)


def bundle():
    """router.pkl contents, or None if it does not exist."""
    global _bundle, _loaded_mtime
    mtime = config.ROUTER_PKL.stat().st_mtime if config.ROUTER_PKL.exists() else None
    if mtime != _loaded_mtime:
        _bundle = pickle.loads(config.ROUTER_PKL.read_bytes()) if mtime else None
        _loaded_mtime = mtime
    return _bundle


def reload():
    global _loaded_mtime
    _loaded_mtime = -1


def version():
    b = bundle()
    if b:
        return b["version"]
    return "knn-fallback" if cache.collection("examples").count() else "rules"


def knn_stats(embedding, exclude_ids=()):
    """Success rate of each tier on the K most similar labeled questions (None if no examples)."""
    col = cache.collection("examples")
    n = col.count()
    if n == 0:
        return None
    res = col.query(query_embeddings=[embedding], n_results=min(config.KNN_K + 5, n))
    # sort by (distance, id) so ties (near-duplicate questions) come back in a stable order
    ranked = sorted(zip(res["distances"][0], res["ids"][0], res["metadatas"][0]),
                    key=lambda r: (round(r[0], 6), r[1]))
    hits = [(m, d) for d, i, m in ranked if i not in exclude_ids][:config.KNN_K]
    stats = {t: sum(m[f"{t}_correct"] for m, _ in hits) / len(hits) for t in config.TIERS}
    stats["sim"] = sum(1 - d for _, d in hits) / len(hits)
    return stats


def features(c, embedding, knn):
    """Embedding + simple regex features + kNN features -> one flat list of numbers."""
    knn = knn or {"small": 0.5, "medium": 0.5, "large": 0.5, "sim": 0.0}
    text = c["text"]
    return list(embedding) + [
        len(text), len(text.split()), sum(ch.isdigit() for ch in text), len(c["numbers"]),
        int(bool(c["options"])), c["math_words"], int(c["negation"]),
        int(c["task_type"] == "mcq"), int(c["task_type"] == "math"), int(c["task_type"] == "text"),
        knn["small"], knn["medium"], knn["large"], knn["sim"],
    ]


def pick(p_small, p_medium, tau_small, tau_medium):
    """Return (tier, why)."""
    if p_small >= tau_small:
        return "small", f"P_small={p_small:.2f} ≥ {tau_small:.2f} → small"
    if p_medium >= tau_medium:
        return "medium", f"P_small={p_small:.2f} < {tau_small:.2f}, P_medium={p_medium:.2f} ≥ {tau_medium:.2f} → medium"
    return "large", f"P_small={p_small:.2f} < {tau_small:.2f}, P_medium={p_medium:.2f} < {tau_medium:.2f} → large"


def rule_choose(c):
    """Last resort when there is no router.pkl and no examples: length + math words."""
    words = len(c["text"].split())
    if c["task_type"] == "math" and c["math_words"] >= 4 or words > 150:
        return "large", f"rules: {words} words, {c['math_words']} math words → large"
    if c["task_type"] == "math" or words > 60:
        return "medium", f"rules: {words} words, {c['math_words']} math words → medium"
    return "small", f"rules: short ({words} words), few math words → small"


def choose(c, embedding):
    """Return (tier, why)."""
    knn = knn_stats(embedding)
    b = bundle()
    if b:
        x = [features(c, embedding, knn)]
        p_small = b["models"]["small"].predict_proba(x)[0, 1]
        p_medium = b["models"]["medium"].predict_proba(x)[0, 1]
        tier, why = pick(p_small, p_medium, b["tau_small"], b["tau_medium"])
        return tier, why
    if knn:
        tier, why = pick(knn["small"], knn["medium"], config.TAU_SMALL, config.TAU_MEDIUM)
        return tier, "kNN fallback: " + why
    return rule_choose(c)
