"""Step 1 of training: run every model tier on every query and record whether it was right.

These runs ARE the training labels: "difficulty" for the router means "which models get
this query right", which we can only learn by asking the models.

Output: data/model_runs.jsonl, one line per (query, tier). The script is resumable: pairs
already in the file are skipped, so you can stop it (Ctrl-C) and start it again.

Usage:
    ollama serve                             # in another terminal
    ollama pull qwen2.5:0.5b qwen2.5:3b qwen2.5:7b
    python scripts/run_models.py             # all 500 queries x 3 tiers
    python scripts/run_models.py --limit 10  # quick smoke test
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_loader import load_queries  # noqa: E402
from src.grading import extract_answer, is_correct  # noqa: E402
from src.llm_client import ask, load_models  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", default=ROOT / "data" / "queries.csv")
    parser.add_argument("--models", default=ROOT / "config" / "models.yaml")
    parser.add_argument("--out", default=ROOT / "data" / "model_runs.jsonl")
    parser.add_argument("--tiers", nargs="+", help="only run these tiers, e.g. --tiers small")
    parser.add_argument("--limit", type=int, help="only the first N queries (for testing)")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    if args.limit:
        queries = queries.head(args.limit)
    models = load_models(args.models)
    tiers = args.tiers or list(models)

    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            run = json.loads(line)
            done.add((run["id"], run["tier"]))

    todo = [(row, t) for t in tiers for row in queries.itertuples() if (row.id, t) not in done]
    print(f"{len(done)} runs already saved, {len(todo)} to go")

    # Tier-by-tier, so Ollama keeps a single model loaded in memory at a time.
    with out.open("a") as f:
        for n, (row, tier) in enumerate(todo, start=1):
            result = ask(models[tier], row.task_type, row.query)
            record = {
                "id": row.id, "tier": tier, "model": models[tier]["name"],
                "answer": extract_answer(row.task_type, result["reply"]),
                "correct": is_correct(row.task_type, result["reply"], row.reference_answer),
                **result,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()  # every result is on disk immediately, so a crash loses nothing
            print(f"[{n}/{len(todo)}] {row.id} {tier:6s} {'OK ' if record['correct'] else 'BAD'} "
                  f"{result['latency_s']:.1f}s  answer={record['answer']} ref={row.reference_answer}")


if __name__ == "__main__":
    main()
