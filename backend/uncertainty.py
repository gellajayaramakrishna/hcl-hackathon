# Step 7: decide if an answer is UNSURE (rules + extra samples), so the pipeline can escalate.
# Unsure if: no valid "Final answer:", a hedge word, or (small model, mcq/math only) the answer
# disagrees with 2 extra samples at temperature 0.7.
from concurrent.futures import ThreadPoolExecutor

from backend import config, llm


def check(tier, query, task_type, result):
    """Return (unsure, reason, extra_calls)."""
    if result["answer"] is None:
        return True, "no valid final answer", []
    text = result["text"].lower()
    for word in config.HEDGE_WORDS:
        if word in text:
            return True, f"hedge word '{word}'", []
    if tier != "small" or task_type not in ("mcq", "math"):
        return False, "sure", []

    # self-consistency: 2 more samples in parallel
    with ThreadPoolExecutor(config.EXTRA_SAMPLES) as pool:
        samples = list(pool.map(lambda _: llm.ask(tier, query, task_type, config.SAMPLE_TEMPERATURE,
                                                  fallback=False), range(config.EXTRA_SAMPLES)))
    calls = [c for s in samples for c in s["calls"]]
    answers = [result["answer"]] + [s["answer"] for s in samples]
    if len(set(answers)) > 1:
        return True, f"samples disagree ({', '.join(str(a) for a in answers)})", calls
    return False, f"{len(answers)} samples agree", calls
