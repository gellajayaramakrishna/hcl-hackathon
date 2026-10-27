# Runs steps 1→8 for one query and returns the answer plus a "routing card"
# (model, why, cache type, cost, per-step latency, trace_id).
import json
import time
import uuid
from datetime import datetime

from backend import cache, config, db, llm, router, uncertainty
from backend.clean import clean
from backend.embed import embed


def _ms(start):
    return round((time.perf_counter() - start) * 1000, 1)


def run(query, task_type=None, force_tier=None, source="live"):
    """force_tier: skip the router (used by eval for the "large + cache" baseline).
    source: "live" for real users; eval runs use "eval:<strategy>" so they don't count in /stats or budget."""
    trace_id = str(uuid.uuid4())
    steps = {}
    card = {"trace_id": trace_id, "cache": "none", "cache_key": None, "escalations": 0,
            "confidence": "n/a", "cost": 0.0, "currency": config.CURRENCY, "source": source}

    # 1. clean
    t = time.perf_counter()
    c = clean(query, task_type)
    k = cache.key(c)
    steps["clean"] = _ms(t)
    card["task_type"] = c["task_type"]

    # 2. L1 exact cache
    t = time.perf_counter()
    hit = cache.lookup_exact(k)
    steps["cache_exact"] = _ms(t)
    if hit:
        return _finish(card, steps, c, hit, "exact", "exact match in cache")

    # 3. embed (once, reused by steps 4 and 5)
    t = time.perf_counter()
    emb = embed(c["text"])
    steps["embed"] = _ms(t)

    # 4. L2 similar cache
    t = time.perf_counter()
    hit, note = cache.lookup_similar(c, emb)
    steps["cache_similar"] = _ms(t)
    if hit:
        return _finish(card, steps, c, hit, "similar", note)

    # 5. choose model (budget cap forces small)
    t = time.perf_counter()
    over_budget = source == "live" and db.today_spend() >= config.DAILY_BUDGET
    if over_budget:
        tier, why = "small", f"daily budget {config.DAILY_BUDGET} {config.CURRENCY} reached → forced small"
    elif force_tier:
        tier, why = force_tier, f"forced {force_tier}"
    else:
        tier, why = router.choose(c, emb)
    steps["route"] = _ms(t)

    # 6 + 7. call model, check if unsure, escalate to the next bigger model (max 2 jumps)
    calls, steps["llm"], steps["uncertainty"] = [], 0.0, 0.0
    while True:
        result = llm.ask(tier, c["redacted"], c["task_type"])  # retry + fallback inside
        calls += result["calls"]
        steps["llm"] += sum(x["ms"] for x in result["calls"])
        if result["tier"] != tier:
            why += f"; {tier} failed → fell back to {result['tier']}"

        unsure, reason, extra = uncertainty.check(result["tier"], c["redacted"], c["task_type"], result)
        calls += extra
        steps["uncertainty"] += max((x["ms"] for x in extra), default=0.0)  # samples run in parallel
        if not unsure:
            break
        if result["tier"] == "large" or card["escalations"] == config.MAX_ESCALATIONS or over_budget:
            why += f"; {result['tier']} unsure ({reason}), no escalation left"
            break
        tier = config.TIERS[config.TIERS.index(result["tier"]) + 1]
        why += f"; {result['tier']} unsure ({reason}) → {tier}"
        card["escalations"] += 1
    steps["llm"], steps["uncertainty"] = round(steps["llm"], 1), round(steps["uncertainty"], 1)
    sure = not unsure

    # 8. save
    t = time.perf_counter()
    if sure and not c["has_pii"] and not c["time_sensitive"]:
        cache.store(k, c, emb, result["answer"], result["tier"])
    steps["save"] = _ms(t)

    card.update(answer=result["answer"] or result["text"].strip()[-300:] or "(no answer)",
                model_used=result["tier"], why=why, confidence="sure" if sure else "unsure",
                cost=sum(x["cost"] for x in calls))
    return _save(card, steps, query, calls)


def _finish(card, steps, c, hit, cache_type, why):
    """Return a cached answer (cost 0)."""
    card.update(answer=hit["answer"], model_used="cache", why=f"{why} (answered by {hit['model']})",
                cache=cache_type, cache_key=hit["key"], confidence="sure")
    return _save(card, steps, c["query"], [])


def _save(card, steps, query, calls):
    card["latency_ms"] = steps
    card["total_ms"] = round(sum(steps.values()), 1)
    db.insert("requests", {
        "trace_id": card["trace_id"], "ts": datetime.now().isoformat(timespec="seconds"),
        "query": query, "answer": card["answer"], "model_used": card["model_used"],
        "cache_type": card["cache"], "cache_key": card["cache_key"], "escalations": card["escalations"],
        "total_cost": card["cost"], "total_ms": card["total_ms"], "router_version": router.version(),
        "source": card["source"], "latency_json": json.dumps(steps)})
    for x in calls:
        db.insert("llm_calls", {"trace_id": card["trace_id"], "model": x["model"],
                                "input_tokens": x["input_tokens"], "output_tokens": x["output_tokens"],
                                "cost": x["cost"], "ms": x["ms"], "ok": x["ok"]})
    return card


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(config.QUERIES_CSV, dtype=str).set_index("id")
    para = df[df.cache_expected == "reuse"].iloc[0]
    changed = df[df.cache_expected == "no_reuse"].iloc[0]
    samples = [
        (df.loc[para.cache_group, "query"], df.loc[para.cache_group, "task_type"], "original"),
        (para.query, para.task_type, "paraphrase (should hit similar cache)"),
        (df.loc[changed.cache_group, "query"], changed.task_type, "original"),
        (changed.query, changed.task_type, "number changed (must NOT hit cache)"),
        ("My email is jo@x.com. What is the latest version of Python?", None, "PII + time-sensitive"),
        (df.loc[para.cache_group, "query"], df.loc[para.cache_group, "task_type"], "repeat (exact cache)"),
    ]
    cache.clear()
    for q, tt, label in samples:
        r = run(q, tt)
        print(f"\n[{label}] {q[:70]!r}")
        print(f"  answer={r['answer']!r} model={r['model_used']} cache={r['cache']} "
              f"cost=${r['cost']:.6f} total={r['total_ms']}ms\n  why: {r['why']}")
