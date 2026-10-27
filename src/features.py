"""Features the router computes from the query text ALONE.

At run time the router only sees the query -- never the subject, the solution or the
difficulty label -- so everything here must be derivable from the raw text.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Words that tend to signal specialist or multi-step questions.
HARD_HINTS = re.compile(r"\b(prove|integral|derivative|matrix|eigen|statute|plaintiff|defendant|"
                        r"hearsay|diagnosis|regression|equilibrium|valid|entail|group|ring|field)\b", re.I)


def handcrafted(queries) -> np.ndarray:
    """One row of simple, explainable numbers per query."""
    rows = []
    for q in queries:
        words = q.split()
        rows.append([
            np.log1p(len(q)),                                   # length (log: 100 vs 2000 chars)
            np.log1p(len(words)),
            len(re.findall(r"\d+(?:\.\d+)?", q)),                # how many numbers
            len(re.findall(r"[.!?](?:\s|$)", q)),                # sentence count
            np.mean([len(w) for w in words]) if words else 0,    # long words ~ technical text
            float("\nA) " in q),                                  # multiple choice?
            len(re.findall(r"[=+\-*/^√∑∫]", q)),                  # math symbols
            len(HARD_HINTS.findall(q)),
        ])
    return np.array(rows, dtype=float)


def build_classifier() -> Pipeline:
    """Text -> P(this model answers correctly).

    TF-IDF captures *what* the question is about (law vocabulary, algebra notation...);
    the handcrafted numbers capture its *shape* (length, how many numbers...).
    Logistic regression keeps it fast, small and easy to explain.
    """
    features = FeatureUnion([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000,
                                  sublinear_tf=True)),
        ("shape", Pipeline([("calc", FunctionTransformer(handcrafted)),
                            ("scale", StandardScaler())])),
    ])
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])
