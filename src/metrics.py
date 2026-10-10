import numpy as np
from sklearn.metrics import cohen_kappa_score

def weighted_kappa(ratings_a, ratings_b):
    if len(ratings_a) != len(ratings_b) or len(ratings_a) < 2:
        raise ValueError("At least two paired ratings are required")
    return float(cohen_kappa_score(ratings_a, ratings_b, weights="quadratic"))

def bootstrap_mean_ci(values, iterations=2000, confidence=0.95, seed=42):
    values = np.asarray(values, dtype=float)
    if values.size == 0: raise ValueError("Cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(iterations)])
    alpha = 1-confidence
    lo, hi = np.quantile(means, [alpha/2, 1-alpha/2])
    return float(values.mean()), float(lo), float(hi)
