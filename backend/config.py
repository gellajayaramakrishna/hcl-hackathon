# All settings for the router in one place: paths, thresholds, budget, model tiers and prices.
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
load_dotenv(ROOT / ".env")  # API keys for LiteLLM

# --- files ---
QUERIES_CSV = DATA / "queries.csv"
SPLITS_JSON = DATA / "splits.json"  # used when queries.csv has no "split" column
PRICES_YAML = DATA / "prices.yaml"
DB_PATH = DATA / "app.db"
CHROMA_DIR = DATA / "chroma"
ROUTER_PKL = DATA / "router.pkl"
MODEL_RUNS_CSV = DATA / "model_runs.csv"

# --- mock mode: MOCK_LLM=1 fakes model calls (no API cost) ---
MOCK_LLM = os.getenv("MOCK_LLM", "0") == "1"
# P(correct) by tier and question difficulty; averages ~60% / 80% / 95% on our dataset.
# Difficulty-aware so the router has a real signal to learn in mock mode.
MOCK_ACCURACY = {"small": {"easy": 0.85, "medium": 0.60, "hard": 0.35},
                 "medium": {"easy": 0.95, "medium": 0.80, "hard": 0.65},
                 "large": {"easy": 0.99, "medium": 0.96, "hard": 0.90}}
MOCK_ACCURACY_UNKNOWN = {"small": 0.60, "medium": 0.80, "large": 0.95}  # query not in queries.csv
MOCK_LATENCY_MS = {"small": 300, "medium": 900, "large": 2500}
MOCK_FAIL_RATE = float(os.getenv("MOCK_FAIL_RATE", "0"))  # e.g. 0.1 = 10% of mock calls error (shows retry/fallback)

# --- model tiers, cheapest first ---
TIERS = ["small", "medium", "large"]
PRICES = yaml.safe_load(PRICES_YAML.read_text())
CURRENCY = PRICES.get("currency", "USD")

# --- LLM call ---
MAX_TOKENS = {"mcq": 256, "math": 512, "text": 256}
LLM_TIMEOUT_S = 60

# --- budget ---
DAILY_BUDGET = 1.00  # in CURRENCY; when reached, force the small model

# --- cache ---
CACHE_SIM_THRESHOLD = 0.92
CACHE_TOP_K = 3

# --- embeddings ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# --- router ---
KNN_K = 5
TAU_SMALL = 0.70   # defaults; train.py overwrites them with tuned values in router.pkl
TAU_MEDIUM = 0.70
QUALITY_TARGET = 0.95  # router accuracy must be >= 95% of always-large accuracy
TUNE_SAFETY_SE = 1.0   # tune accuracy must clear the target by this many standard errors
                       # (the tune split is small, so a setting that only just passes may be luck)

# --- uncertainty / escalation ---
HEDGE_WORDS = ["i think", "maybe", "possibly", "not sure", "unclear"]
EXTRA_SAMPLES = 2       # extra samples for the small model on mcq/math
SAMPLE_TEMPERATURE = 0.7
MAX_ESCALATIONS = 2
