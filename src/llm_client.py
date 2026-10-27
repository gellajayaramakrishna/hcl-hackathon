"""Minimal client for the models in config/models.yaml (Ollama for local development)."""
import json
import time
import urllib.request
from pathlib import Path

import yaml

OLLAMA_URL = "http://localhost:11434/api/chat"
SEED = 42

# Instructions per task type, and an output-token cap so cost stays bounded.
PROMPTS = {
    "mcq": ("Answer the multiple-choice question. Reply with only the letter of the correct "
            "option (A, B, C or D).", 32),
    "math": ("Solve the problem step by step. End your reply with a final line of the form "
             "'Answer: <number>'.", 512),
}


def load_models(path) -> dict:
    """{tier: config} in price order, e.g. {'small': {...}, 'medium': {...}, 'large': {...}}."""
    models = yaml.safe_load(Path(path).read_text())["models"]
    return dict(sorted(models.items(), key=lambda kv: kv[1]["price_out_per_1k"]))


def cost_usd(model: dict, tokens_in: int, tokens_out: int) -> float:
    return tokens_in / 1000 * model["price_in_per_1k"] + tokens_out / 1000 * model["price_out_per_1k"]


def ask(model: dict, task_type: str, query: str, timeout: int = 300) -> dict:
    """Send one query and return the reply text, token counts, latency and cost."""
    if model["provider"] != "ollama":
        raise NotImplementedError(f"provider {model['provider']!r} is not wired up yet")
    instruction, max_tokens = PROMPTS.get(task_type, PROMPTS["math"])
    body = {
        "model": model["name"],
        "messages": [{"role": "system", "content": instruction},
                     {"role": "user", "content": query}],
        "stream": False,
        # temperature 0 + fixed seed -> the same reply every run (reproducible labels)
        "options": {"temperature": 0, "seed": SEED, "num_predict": max_tokens},
    }
    request = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    tokens_in, tokens_out = result.get("prompt_eval_count", 0), result.get("eval_count", 0)
    return {
        "reply": result["message"]["content"],
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_s": round(time.perf_counter() - start, 3),
        "cost_usd": cost_usd(model, tokens_in, tokens_out),
    }
