"""Load the query set -- either our stand-in dataset or the organizers' file.

When the organizers' 500 queries arrive, only COLUMN_MAP below should need
editing: map *their* column names to *our* names and the rest of the pipeline
keeps working unchanged.
"""
from pathlib import Path

import pandas as pd

# organizers' column name -> our column name. Edit this when the real file arrives.
# Names are matched after lowercasing and stripping whitespace.
COLUMN_MAP = {
    "question": "query",
    "prompt": "query",
    "answer": "reference_answer",
    "reference": "reference_answer",
    "expected_answer": "reference_answer",
    "query_id": "id",
    "qid": "id",
}

REQUIRED_COLUMNS = ["query", "reference_answer"]

# Columns the organizers' file may not have, with the value to fill in.
OPTIONAL_DEFAULTS = {
    "task_type": "unknown",
    "difficulty": "unknown",
    "subject": "",
    "source": "organizers",
    "solution": "",
    "cache_group": "",
    "cache_expected": "n/a",
}

COLUMN_ORDER = ["id", "query", "reference_answer", *OPTIONAL_DEFAULTS]


def load_queries(path) -> pd.DataFrame:
    """Read a .csv, .jsonl or .json query file into a DataFrame with our standard columns.

    Every value is kept as a string so answers like "007" or "1250" are not
    silently turned into numbers.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        # keep_default_na=False: an empty cell stays "" instead of becoming NaN
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True, dtype=False)
    elif suffix == ".json":
        df = pd.read_json(path, dtype=False)
    else:
        raise ValueError(f"unsupported file type: {path}")

    # Normalise headers, then rename the organizers' columns to ours.
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns {missing}; update COLUMN_MAP")

    if "id" not in df.columns:
        df["id"] = [f"q{i:04d}" for i in range(1, len(df) + 1)]
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    df = df.fillna("").astype(str)
    extra = [c for c in df.columns if c not in COLUMN_ORDER]  # keep unknown columns at the end
    return df[COLUMN_ORDER + extra]
