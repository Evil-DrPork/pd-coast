"""Small-data ensemble and human-tail prototype branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, float), -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def _rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    if len(x) <= 1:
        return np.full_like(x, 0.5)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks / (len(x) - 1)


@dataclass
class HumanTailPrototype:
    max_features: int = 64

    def fit(self, x: np.ndarray, y: np.ndarray) -> "HumanTailPrototype":
        x, y = np.asarray(x, float), np.asarray(y, int)
        # Select features with stable univariate separation in the training fold.
        h, b = x[y == 0], x[y == 1]
        scale = np.std(x, axis=0) + 1e-6
        effect = np.abs((b.mean(0) - h.mean(0)) / scale)
        self.indices_ = np.argsort(-effect, kind="mergesort")[: min(self.max_features, x.shape[1])]
        self.scaler_ = StandardScaler().fit(x[:, self.indices_])
        z = self.scaler_.transform(x[:, self.indices_])
        zh, zb = z[y == 0], z[y == 1]
        self.human_ = LedoitWolf().fit(zh)
        self.bot_ = LedoitWolf().fit(zb)
        dh = self.human_.mahalanobis(zh)
        db = self.bot_.mahalanobis(zh)
        margin = dh - db
        self.center_ = float(np.median(margin))
        self.scale_ = float(np.std(margin) + 1e-6)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler_.transform(np.asarray(x, float)[:, self.indices_])
        margin = self.human_.mahalanobis(z) - self.bot_.mahalanobis(z)
        p = _sigmoid((margin - self.center_) / self.scale_)
        return np.column_stack([1 - p, p])


class V3Ensemble:
    """Three regularized tabular learners plus a human-tail density branch."""

    def __init__(self, seed: int = 44) -> None:
        self.seed = int(seed)
        self.logistic = make_pipeline(
            VarianceThreshold(1e-10), RobustScaler(),
            LogisticRegression(C=0.12, class_weight="balanced", max_iter=3000, random_state=seed),
        )
        self.extra = ExtraTreesClassifier(
            n_estimators=350, max_depth=7, min_samples_leaf=8,
            max_features=0.45, class_weight="balanced", n_jobs=-1,
            random_state=seed,
        )
        self.hist = make_pipeline(
            VarianceThreshold(1e-10),
            HistGradientBoostingClassifier(
                max_iter=180, learning_rate=0.035, max_leaf_nodes=9,
                min_samples_leaf=18, l2_regularization=4.0,
                random_state=seed,
            ),
        )
        self.prototype = HumanTailPrototype(max_features=64)
        self.branch_weights_ = np.asarray([0.45, 0.25, 0.20, 0.10], float)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "V3Ensemble":
        for model in (self.logistic, self.extra, self.hist, self.prototype):
            model.fit(x, y)
        return self

    def branch_scores(self, x: np.ndarray) -> np.ndarray:
        cols = []
        for model in (self.logistic, self.extra, self.hist, self.prototype):
            p = model.predict_proba(x)
            cols.append(np.asarray(p[:, 1], float))
        return np.column_stack(cols)

    def raw_score(self, x: np.ndarray) -> np.ndarray:
        b = self.branch_scores(x)
        # Probability blend is used across separately processed requests. The
        # inference mapper may apply a monotone batch transformation afterward.
        return np.clip(b @ self.branch_weights_, 1e-5, 1 - 1e-5)


def choose_weights(oof_branches: np.ndarray, y: np.ndarray, scorer) -> np.ndarray:
    """Small nonnegative grid selected on honest OOF predictions."""
    candidates = []
    grid = np.arange(0, 1.01, 0.1)
    # Prototype remains a small tail specialist; distribute the rest over the
    # three general learners without a large meta-model.
    for wp in (0.0, 0.1, 0.2):
        remaining = 1.0 - wp
        for a in grid:
            for b in grid:
                c = 1.0 - a - b
                if c < -1e-9:
                    continue
                w = np.asarray([remaining * a, remaining * b, remaining * c, wp])
                score = oof_branches @ w
                m = scorer(y, score)
                objective = 0.6 * m["ap_score"] + 0.4 * m["bot_recall"]
                candidates.append((objective, w))
    return max(candidates, key=lambda z: z[0])[1]

