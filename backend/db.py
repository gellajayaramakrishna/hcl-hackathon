# SQLite setup (6 tables) and a few small helpers to insert rows and run queries.
import sqlite3
from datetime import date

from backend import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (model TEXT PRIMARY KEY, input_per_1m REAL, output_per_1m REAL);
CREATE TABLE IF NOT EXISTS requests (
    trace_id TEXT PRIMARY KEY, ts TEXT, query TEXT, answer TEXT, model_used TEXT,
    cache_type TEXT, cache_key TEXT, escalations INTEGER, total_cost REAL, total_ms REAL,
    router_version TEXT, source TEXT, latency_json TEXT);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, model TEXT, input_tokens INTEGER,
    output_tokens INTEGER, cost REAL, ms REAL, ok INTEGER);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT, thumbs INTEGER, ts TEXT);
CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_ts TEXT, strategy TEXT, accuracy REAL,
    total_cost REAL, cost_per_correct REAL, savings_pct REAL, avg_ms REAL,
    cache_hit_rate REAL, wrong_hit_rate REAL, tau_small REAL, tau_medium REAL);
CREATE TABLE IF NOT EXISTS router_versions (
    version TEXT PRIMARY KEY, ts TEXT, tune_accuracy REAL, tune_cost REAL, is_active INTEGER);
"""


def connect():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init():
    """Create tables and (re)load prices from prices.yaml."""
    with connect() as conn:
        conn.executescript(SCHEMA)
        for tier, m in config.PRICES["models"].items():
            conn.execute("INSERT OR REPLACE INTO prices VALUES (?, ?, ?)",
                         (tier, m["input_per_1m"], m["output_per_1m"]))


def insert(table, row):
    cols = ", ".join(row)
    marks = ", ".join("?" * len(row))
    with connect() as conn:
        conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))


def query(sql, params=()):
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def today_spend():
    rows = query("SELECT COALESCE(SUM(total_cost), 0) AS s FROM requests WHERE source = 'live' AND ts LIKE ?",
                 (date.today().isoformat() + "%",))
    return rows[0]["s"]


init()
