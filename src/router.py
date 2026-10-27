"""The trained router: pick the cheapest model that is likely to answer correctly."""
import joblib
import numpy as np


class ConstantClassifier:
    """Stand-in when a tier was right on every (or no) training query: always predicts that rate.
    Logistic regression cannot be trained on a single class."""

    def __init__(self, p_correct: float):
        self.p_correct = p_correct

    def predict_proba(self, queries):
        p = np.full(len(queries), self.p_correct)
        return np.column_stack([1 - p, p])


class Router:
    """One classifier per model tier predicts P(that tier answers correctly | query).

    Routing rule: walk the tiers from cheapest to most expensive and take the first one
    whose predicted probability is >= threshold. If none qualifies, use the largest model.
    """

    def __init__(self, tiers: list[str], classifiers: dict, threshold: float):
        self.tiers = tiers                # cheapest first
        self.classifiers = classifiers    # tier -> fitted sklearn model
        self.threshold = threshold

    def probabilities(self, queries) -> dict[str, list[float]]:
        return {t: self.classifiers[t].predict_proba(list(queries))[:, 1].tolist() for t in self.tiers}

    def choose(self, queries, threshold: float | None = None) -> list[str]:
        threshold = self.threshold if threshold is None else threshold
        probs = self.probabilities(queries)
        choices = []
        for i in range(len(queries)):
            tier = next((t for t in self.tiers[:-1] if probs[t][i] >= threshold), self.tiers[-1])
            choices.append(tier)
        return choices

    def save(self, path) -> None:
        joblib.dump({"tiers": self.tiers, "classifiers": self.classifiers,
                     "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path) -> "Router":
        return cls(**joblib.load(path))
