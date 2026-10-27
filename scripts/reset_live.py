# Start a demo from zero: delete LIVE requests (+ their llm_calls and feedback) and empty the answer
# cache. Keeps eval results, router versions, router.pkl, model_runs.csv and the "examples" collection.
# Usage: python scripts/reset_live.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend  # noqa: F401

from backend import cache, config, db

with db.connect() as conn:
    live = "SELECT trace_id FROM requests WHERE source = 'live'"
    calls = conn.execute(f"DELETE FROM llm_calls WHERE trace_id IN ({live})").rowcount
    fb = conn.execute(f"DELETE FROM feedback WHERE trace_id IN ({live})").rowcount
    reqs = conn.execute("DELETE FROM requests WHERE source = 'live'").rowcount
cache.clear()
(config.DATA / "feedback_cursor.txt").unlink(missing_ok=True)
print(f"deleted {reqs} live requests, {calls} llm_calls, {fb} feedback rows; answer cache emptied")
