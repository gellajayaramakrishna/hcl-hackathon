# FastAPI routes: POST /query, POST /feedback, GET /stats, GET /samples (demo questions for the UI).
# Run: uvicorn backend.app:app --reload --port 8000
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend import config, db, llm, pipeline
from backend.data import load_queries, load_runs
from backend.embed import embed

app = FastAPI(title="Cost-Aware Model Router")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


class Query(BaseModel):
    query: str
    task_type: str | None = None  # optional: mcq | math | text (guessed if missing)


class Feedback(BaseModel):
    trace_id: str
    thumbs: int  # +1 or -1


@app.on_event("startup")
def warm_up():
    embed("warm up")  # load bge-small now so the first query is fast


@app.post("/query")
def query(q: Query):
    if not q.query.strip():
        raise HTTPException(400, "empty query")
    if q.task_type not in (None, "mcq", "math", "text"):
        raise HTTPException(400, "task_type must be mcq, math or text")
    return pipeline.run(q.query, q.task_type)


@app.post("/feedback")
def feedback(f: Feedback):
    if f.thumbs not in (1, -1):
        raise HTTPException(400, "thumbs must be 1 or -1")
    if not db.query("SELECT 1 FROM requests WHERE trace_id = ?", (f.trace_id,)):
        raise HTTPException(404, "unknown trace_id")
    db.insert("feedback", {"trace_id": f.trace_id, "thumbs": f.thumbs,
                           "ts": datetime.now().isoformat(timespec="seconds")})
    return {"ok": True}


@app.get("/stats")
def stats():
    reqs = db.query("SELECT trace_id, model_used, cache_type, total_cost FROM requests WHERE source = 'live'")
    # tokens of the final successful call per request, priced as if the large model answered
    last_calls = db.query("""SELECT trace_id, input_tokens, output_tokens FROM llm_calls
                             WHERE id IN (SELECT MAX(l.id) FROM llm_calls l JOIN requests r USING (trace_id)
                                          WHERE l.ok = 1 AND r.source = 'live' GROUP BY l.trace_id)""")
    large_costs = [llm.cost("large", c["input_tokens"], c["output_tokens"]) for c in last_calls]
    avg_large = sum(large_costs) / len(large_costs) if large_costs else 0.0
    cache_hits = sum(r["cache_type"] != "none" for r in reqs)
    # a cache hit would still have cost one large call without our system
    if_always_large = sum(large_costs) + cache_hits * avg_large
    spend = sum(r["total_cost"] for r in reqs)

    mix = {t: 0 for t in config.TIERS + ["cache"]}
    for r in reqs:
        mix[r["model_used"]] = mix.get(r["model_used"], 0) + 1
    return {
        "requests": len(reqs),
        "currency": config.CURRENCY,
        "total_spend": spend,
        "if_always_large": if_always_large,
        "savings_pct": 100 * (1 - spend / if_always_large) if if_always_large else 0.0,
        "routing_mix": mix,
        "cache_hit_rate": cache_hits / len(reqs) if reqs else 0.0,
        "today_spend": db.today_spend(),
        "daily_budget": config.DAILY_BUDGET,
        "mock_mode": config.MOCK_LLM,
    }


@app.get("/samples")
def samples():
    """A few TEST questions for the demo: easy/hard, math, and a paraphrase + number-changed pair."""
    q = load_queries()
    test = q[q.split == "test"].set_index("id")
    # prefer pairs whose original the models answer confidently: unsure answers are never cached
    confident = set(test.index)
    if config.MODEL_RUNS_CSV.exists():
        unsure = load_runs()["unsure"]
        confident = set(unsure.index[~unsure["medium"].astype(bool) & ~unsure["large"].astype(bool)])
    picks = [("Easy MCQ", test[(test.task_type == "mcq") & (test.difficulty == "easy")].iloc[0]),
             ("Hard MCQ", test[(test.task_type == "mcq") & (test.difficulty == "hard")].iloc[0]),
             ("Easy math", test[(test.task_type == "math") & (test.difficulty == "easy")].iloc[0]),
             ("Hard math", test[(test.task_type == "math") & (test.difficulty == "hard")].iloc[0])]
    for kind, label in [("reuse", "paraphrase (cache should reuse)"), ("no_reuse", "number changed (cache must not reuse)")]:
        derived = test[(test.cache_expected == kind) & test.cache_group.isin(test.index)]
        # skip near-duplicates (only spacing/case changed): a real rewording shows the similar cache
        norm = lambda t: " ".join(t.lower().split()).rstrip("?.! ")  # noqa: E731
        derived = derived[[norm(r.query) != norm(test.loc[r.cache_group, "query"]) for r in derived.itertuples()]]
        derived = derived.sort_values("cache_group", key=lambda g: ~g.isin(confident))  # confident first
        if len(derived):
            d = derived.iloc[0]
            picks += [(f"Original of {label.split(' (')[0]}", test.loc[d.cache_group]), (label.capitalize(), d)]
    return [{"label": label, "query": r.query, "task_type": r.task_type} for label, r in picks]
