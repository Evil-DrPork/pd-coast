"""Tabular base learners owned by the separate large-chunk challenger.

The definitions intentionally live here instead of importing ``model_v3.model``
because that module also imports the optional neural runtime.  This challenger
must remain trainable and serveable with only the tabular dependencies.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`*",
    category=UserWarning,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class HumanTailPrototype:
    max_features: int = 64

    def fit(self, x: np.ndarray, y: np.ndarray) -> "HumanTailPrototype":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)
        human = x[y == 0]
        bot = x[y == 1]
        scale = np.std(x, axis=0) + 1e-6
        effect = np.abs((bot.mean(axis=0) - human.mean(axis=0)) / scale)
        self.indices_ = np.argsort(-effect, kind="mergesort")[: min(self.max_features, x.shape[1])]
        self.scaler_ = StandardScaler().fit(x[:, self.indices_])
        z = self.scaler_.transform(x[:, self.indices_])
        z_human = z[y == 0]
        z_bot = z[y == 1]
        self.human_ = LedoitWolf().fit(z_human)
        self.bot_ = LedoitWolf().fit(z_bot)
        margin = self.human_.mahalanobis(z_human) - self.bot_.mahalanobis(z_human)
        self.center_ = float(np.median(margin))
        self.scale_ = float(np.std(margin) + 1e-6)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler_.transform(np.asarray(x, dtype=float)[:, self.indices_])
        margin = self.human_.mahalanobis(z) - self.bot_.mahalanobis(z)
        probability = _sigmoid((margin - self.center_) / self.scale_)
        return np.column_stack([1.0 - probability, probability])


class TabularBaseEnsemble:
    """The stable four tabular model families, fitted on challenger views."""

    def __init__(self, seed: int = 44) -> None:
        self.seed = int(seed)
        self.logistic = make_pipeline(
            VarianceThreshold(1e-10),
            RobustScaler(),
            LogisticRegression(
                C=0.12,
                class_weight="balanced",
                max_iter=3000,
                random_state=seed,
            ),
        )
        self.extra = ExtraTreesClassifier(
            n_estimators=350,
            max_depth=7,
            min_samples_leaf=8,
            max_features=0.45,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        self.hist = make_pipeline(
            VarianceThreshold(1e-10),
            HistGradientBoostingClassifier(
                max_iter=180,
                learning_rate=0.035,
                max_leaf_nodes=9,
                min_samples_leaf=18,
                l2_regularization=4.0,
                random_state=seed,
            ),
        )
        self.prototype = HumanTailPrototype(max_features=64)
        self.branch_weights_ = np.asarray([0.45, 0.25, 0.20, 0.10], dtype=float)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TabularBaseEnsemble":
        for model in (self.logistic, self.extra, self.hist, self.prototype):
            model.fit(x, y)
        return self

    def branch_scores(self, x: np.ndarray) -> np.ndarray:
        columns = []
        for model in (self.logistic, self.extra, self.hist, self.prototype):
            columns.append(np.asarray(model.predict_proba(x)[:, 1], dtype=float))
        return np.column_stack(columns)


def choose_tabular_weights(oof_branches: np.ndarray, y: np.ndarray, scorer) -> np.ndarray:
    """Small nonnegative grid, matching the stable v3 selection constraints."""

    candidates = []
    grid = np.arange(0.0, 1.01, 0.10)
    for prototype_weight in (0.0, 0.1, 0.2):
        remaining = 1.0 - prototype_weight
        for logistic_weight in grid:
            for extra_weight in grid:
                hist_weight = 1.0 - logistic_weight - extra_weight
                if hist_weight < -1e-9:
                    continue
                weights = np.asarray(
                    [
                        remaining * logistic_weight,
                        remaining * extra_weight,
                        remaining * hist_weight,
                        prototype_weight,
                    ],
                    dtype=float,
                )
                detail = scorer(y, oof_branches @ weights)
                objective = 0.60 * detail["ap_score"] + 0.40 * detail["bot_recall"]
                candidates.append((objective, weights))
    return max(candidates, key=lambda item: item[0])[1]
