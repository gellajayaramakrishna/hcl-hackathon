# Steps 2 + 4: answer cache in a Chroma collection "cache".
# L1 = exact match on a sha256 key. L2 = similar query, reused only if all 7 safety rules pass.
import hashlib

import chromadb

from backend import config

_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def collection(name="cache"):
    # high search_ef = (near-)exact search on small collections, so results are stable run to run
    return _client.get_or_create_collection(
        name, metadata={"hnsw:space": "cosine", "hnsw:search_ef": 200, "hnsw:construction_ef": 200})


def key(c):
    return hashlib.sha256((c["text"] + "|" + "|".join(c["options"])).encode()).hexdigest()


def lookup_exact(k):
    got = collection().get(ids=[k])
    if got["ids"] and got["metadatas"][0]["confident"]:
        return {"key": k, **got["metadatas"][0]}
    return None


def _join(items):
    return "|".join(items)


def _check_rules(c, meta, sim):
    """Return None if the entry is safe to reuse, else the first rule that failed."""
    if sim < config.CACHE_SIM_THRESHOLD:
        return f"similarity {sim:.3f} < {config.CACHE_SIM_THRESHOLD}"
    if _join(c["numbers"]) != meta["numbers"]:
        return "different numbers"
    if _join(c["names"]) != meta["names"]:
        return "different names"
    if c["negation"] != meta["negation"]:
        return "different negation"
    if _join(c["options"]) != meta["options"]:
        return "different options"
    if c["task_type"] != meta["task_type"]:
        return "different task type"
    if not meta["confident"] or c["time_sensitive"]:
        return "entry not confident or query time-sensitive"
    return None


def lookup_similar(c, embedding):
    """Top-k nearest entries; return (hit or None, note explaining the decision)."""
    col = collection()
    if col.count() == 0:
        return None, "cache empty"
    res = col.query(query_embeddings=[embedding], n_results=min(config.CACHE_TOP_K, col.count()))
    notes = []
    for k, dist, meta in zip(res["ids"][0], res["distances"][0], res["metadatas"][0]):
        sim = 1 - dist
        failed = _check_rules(c, meta, sim)
        if failed is None:
            return {"key": k, "similarity": sim, **meta}, f"similar entry (sim={sim:.3f}), all 7 rules pass"
        notes.append(failed)
    return None, "no safe match: " + "; ".join(notes)


def store(k, c, embedding, answer, model):
    collection().upsert(ids=[k], embeddings=[embedding], documents=[c["text"]], metadatas=[{
        "answer": answer, "model": model, "numbers": _join(c["numbers"]), "names": _join(c["names"]),
        "negation": c["negation"], "options": _join(c["options"]), "task_type": c["task_type"],
        "confident": True}])


def mark_unconfident(k):
    col = collection()
    got = col.get(ids=[k])
    if got["ids"]:
        col.update(ids=[k], metadatas=[{**got["metadatas"][0], "confident": False}])


def clear():
    if "cache" in [col.name for col in _client.list_collections()]:
        _client.delete_collection("cache")
