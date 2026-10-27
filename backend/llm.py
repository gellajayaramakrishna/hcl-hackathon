# Step 6: call a model through LiteLLM (or fake it in mock mode), with retry, fallback to a
# bigger model, cost from prices.yaml, plus answer parsing and grading helpers.
import random
import re
import time
from fractions import Fraction

import litellm
import pandas as pd

from backend import config

litellm.suppress_debug_info = True

PROMPTS = {
    "mcq": "Answer this multiple-choice question. Think briefly, then end with a line\n"
           "'Final answer: <letter>' where <letter> is one of A, B, C, D.\n\n{query}",
    "math": "Solve this math problem step by step. End with a line\n"
            "'Final answer: <number>' containing only the number.\n\n{query}",
    "text": "Answer the question concisely. End with a line 'Final answer: <answer>'.\n\n{query}",
}


# ---------- answer parsing and grading ----------

def parse_answer(task_type, text):
    """Return the answer after the last 'Final answer:', or None if missing/invalid."""
    found = re.findall(r"final answer\s*[:\-]\s*(.+)", text or "", re.I)
    if not found:
        return None
    raw = found[-1].strip().strip("*$ .")
    if task_type == "mcq":
        m = re.fullmatch(r"\(?([A-Da-d])\)?(?:[).:].*)?", raw)
        return m.group(1).upper() if m else None
    if task_type == "math":
        m = re.fullmatch(r"-?\d[\d,]*(?:\.\d+)?", raw)
        return m.group(0).replace(",", "") if m else None
    return raw or None


def normalize_text(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def grade(task_type, answer, reference):
    """Deterministic grading: letter match, number match with tolerance, or normalized text."""
    if answer is None:
        return False
    if task_type == "mcq":
        return answer.upper() == str(reference).strip().upper()
    if task_type == "math":
        try:
            a, r = Fraction(str(answer)), Fraction(str(reference).replace(",", ""))
            return abs(a - r) <= Fraction(1, 1000) * max(1, abs(r))
        except (ValueError, ZeroDivisionError):
            return False
    return normalize_text(answer) == normalize_text(reference)


def cost(tier, input_tokens, output_tokens):
    p = config.PRICES["models"][tier]
    return (input_tokens * p["input_per_1m"] + output_tokens * p["output_per_1m"]) / 1_000_000


# ---------- mock mode ----------

_references = None


def _reference_for(query):
    """Look the query up in queries.csv so mock answers can be right or wrong.
    Returns (reference_answer, difficulty) or (None, None)."""
    global _references
    if _references is None:
        df = pd.read_csv(config.QUERIES_CSV, dtype=str).fillna("")
        _references = {q.strip().lower(): (ref, diff) for q, ref, diff
                       in zip(df["query"], df["reference_answer"], df["difficulty"])}
    return _references.get(query.strip().lower(), (None, None))


def _mock_call(tier, query, task_type, temperature):
    # temperature 0 -> same answer every time (so setup.py and the pipeline agree)
    if random.random() < config.MOCK_FAIL_RATE:
        raise RuntimeError("mock API error")
    rng = random.Random(f"{tier}|{query}") if temperature == 0 else random.Random()
    ref, difficulty = _reference_for(query)
    p_correct = config.MOCK_ACCURACY[tier].get(difficulty, config.MOCK_ACCURACY_UNKNOWN[tier])
    correct = rng.random() < p_correct
    if ref is None:
        ref = {"mcq": "A", "math": "42", "text": "mock answer"}[task_type]
    if correct:
        answer = ref
    elif task_type == "mcq":
        answer = rng.choice([c for c in "ABCD" if c != ref.upper()])
    elif task_type == "math":
        answer = str(Fraction(ref) + rng.choice([-10, -2, -1, 1, 2, 10]))
    else:
        answer = "unknown"
    hedge = "I think " if not correct and rng.random() < 0.2 else ""
    text = f"{hedge}the answer follows from the question.\nFinal answer: {answer}"
    out_tokens = rng.randint(*{"mcq": (10, 40), "math": (60, 250), "text": (20, 120)}[task_type])
    ms = config.MOCK_LATENCY_MS[tier] * rng.uniform(0.7, 1.3)
    return text, len(query) // 4 + 40, out_tokens, ms


# ---------- real call ----------

def _real_call(tier, query, task_type, temperature):
    start = time.perf_counter()
    resp = litellm.completion(
        model=config.PRICES["models"][tier]["litellm_model"],
        messages=[{"role": "user", "content": PROMPTS[task_type].format(query=query)}],
        temperature=temperature,
        max_tokens=config.MAX_TOKENS[task_type],
        timeout=config.LLM_TIMEOUT_S,
    )
    text = resp.choices[0].message.content or ""
    ms = (time.perf_counter() - start) * 1000
    return text, resp.usage.prompt_tokens, resp.usage.completion_tokens, ms


def ask(tier, query, task_type, temperature=0.0, fallback=True):
    """Call `tier`; on error retry once, then move to the next bigger tier (if fallback).
    Returns {tier, text, answer, ok, calls}; `calls` holds one record per attempt for llm_calls."""
    call_fn = _mock_call if config.MOCK_LLM else _real_call
    calls = []
    tiers = config.TIERS[config.TIERS.index(tier):] if fallback else [tier]
    for t in tiers:
        for _attempt in range(2):
            start = time.perf_counter()
            try:
                text, tin, tout, ms = call_fn(t, query, task_type, temperature)
            except Exception as e:  # network/API errors: log a failed call and try again
                calls.append({"model": t, "input_tokens": 0, "output_tokens": 0, "cost": 0.0,
                              "ms": (time.perf_counter() - start) * 1000, "ok": 0, "error": str(e)[:200]})
                continue
            calls.append({"model": t, "input_tokens": tin, "output_tokens": tout,
                          "cost": cost(t, tin, tout), "ms": ms, "ok": 1})
            return {"tier": t, "text": text, "answer": parse_answer(task_type, text),
                    "ok": True, "calls": calls}
    return {"tier": tier, "text": "", "answer": None, "ok": False, "calls": calls}


if __name__ == "__main__":
    df = pd.read_csv(config.QUERIES_CSV, dtype=str)
    row = df[df.task_type == "math"].iloc[0]
    print(f"MOCK_LLM={config.MOCK_LLM}  query id={row.id}  reference={row.reference_answer}")
    for tier in config.TIERS:
        r = ask(tier, row.query, row.task_type)
        c = r["calls"][-1]
        print(f"{tier:6} -> answer={r['answer']!s:8} correct={grade('math', r['answer'], row.reference_answer)!s:5} "
              f"tokens={c['input_tokens']}/{c['output_tokens']} cost=${c['cost']:.6f} ms={c['ms']:.0f} attempts={len(r['calls'])}")
