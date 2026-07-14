"""Production inference wrapper for Poker44 V4 coherent artifacts."""

from __future__ import annotations

import hashlib
import hmac
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np

from detection_model.model_v3.calibration import apply_mapper
from detection_model.model_v3.schema import clean_hand

from .features import (
    FEATURE_IMPLEMENTATION_SHA256,
    FEATURE_NAMES,
    FEATURE_SCHEMA_SHA256,
    matrix_for_chunks,
)
from .mapping import chunk_tie_key, exact_rank_map
from .model import BRANCH_NAMES, blend_branches
from .provenance import DETECTOR_RUNTIME_SHA256, RUNTIME_LIBRARY_VERSIONS


def _allow_runtime_mismatch() -> bool:
    """Return whether the operator explicitly accepts an unsupported runtime."""
    return os.getenv("P44_ALLOW_RUNTIME_MISMATCH", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _runtime_mismatch(message: str) -> None:
    """Reject runtime drift unless the operator enabled the canary override."""
    if not _allow_runtime_mismatch():
        raise ValueError(message)
    warnings.warn(
        f"{message}. Continuing because P44_ALLOW_RUNTIME_MISMATCH=1; "
        "compare scores against the qualified artifact before promotion",
        RuntimeWarning,
        stacklevel=3,
    )


class Poker44V4Detector:
    """Load, validate and serve a V4 real-only coherent ensemble."""

    def __init__(self, artifact: Dict[str, Any]) -> None:
        if artifact.get("artifact_version") != 4:
            raise ValueError("not a Poker44 V4 artifact")
        if artifact.get("architecture") != "coherent_real_rank_robust_v2":
            raise ValueError("V4 architecture mismatch; retrain the artifact")
        if list(artifact.get("feature_names") or []) != FEATURE_NAMES:
            raise ValueError("V4 feature schema mismatch; retrain the artifact")
        if artifact.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256:
            raise ValueError("V4 feature schema fingerprint mismatch; retrain the artifact")
        if artifact.get("feature_implementation_sha256") != FEATURE_IMPLEMENTATION_SHA256:
            raise ValueError("V4 feature implementation mismatch; retrain the artifact")
        artifact_runtime = artifact.get("detector_runtime_sha256")
        if artifact_runtime != DETECTOR_RUNTIME_SHA256:
            _runtime_mismatch(
                "V4 detector runtime mismatch: "
                f"artifact={artifact_runtime!r} runtime={DETECTOR_RUNTIME_SHA256!r}"
            )
        self.artifact = artifact
        self.model = artifact["model"]
        if tuple(getattr(self.model, "branch_names", ())) != BRANCH_NAMES:
            raise ValueError("V4 branch schema mismatch; retrain the artifact")
        artifact_versions = artifact.get("runtime_library_versions")
        if artifact_versions != RUNTIME_LIBRARY_VERSIONS:
            expected = artifact_versions if isinstance(artifact_versions, dict) else {}
            names = sorted(set(expected) | set(RUNTIME_LIBRARY_VERSIONS))
            differences = ", ".join(
                f"{name}: artifact={expected.get(name)!r} "
                f"runtime={RUNTIME_LIBRARY_VERSIONS.get(name)!r}"
                for name in names
                if expected.get(name) != RUNTIME_LIBRARY_VERSIONS.get(name)
            )
            _runtime_mismatch(
                "V4 runtime library mismatch"
                + (f" ({differences})" if differences else "")
            )
        self.mapper = dict(artifact["mapper"])
        reference = artifact.get("feature_reference")
        if not isinstance(reference, dict):
            raise ValueError("V4 feature reference is missing; retrain the artifact")
        self.feature_reference = {
            name: np.asarray(reference.get(name), dtype=float)
            for name in ("q01", "q25", "median", "q75", "q99")
        }
        if any(
            values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all()
            for values in self.feature_reference.values()
        ):
            raise ValueError("V4 feature reference is invalid; retrain the artifact")
        self.batch_top_fraction = float(artifact.get("batch_top_fraction", 0.10))
        self.blend_mode = str(artifact.get("blend_mode", "probability"))
        if self.blend_mode not in {"probability", "rank"}:
            raise ValueError("V4 blend mode must be probability or rank")
        if not 0.0 < self.batch_top_fraction < 1.0:
            raise ValueError("V4 batch_top_fraction must be between zero and one")
        if not {"cut", "scale"}.issubset(self.mapper) or float(self.mapper["scale"]) <= 0.0:
            raise ValueError("V4 mapper is invalid")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> "Poker44V4Detector":
        artifact_path = Path(path).expanduser()
        if expected_sha256:
            expected = str(expected_sha256).strip().lower()
            digest = hashlib.sha256()
            with artifact_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            actual = digest.hexdigest()
            if len(expected) != 64 or not hmac.compare_digest(actual, expected):
                raise ValueError(
                    "V4 artifact SHA-256 mismatch before deserialization: "
                    f"expected={expected} actual={actual}"
                )
        return cls(joblib.load(artifact_path))

    @staticmethod
    def _clean(chunks: Sequence[Sequence[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        return [
            [clean_hand(hand) for hand in (chunk or []) if isinstance(hand, dict)]
            for chunk in chunks
        ]

    def _raw_from_clean(
        self,
        clean: List[List[Dict[str, Any]]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matrix = matrix_for_chunks(clean)
        branches = self.model.branch_scores(matrix)
        # Percentile fusion is meaningful only for a validator-sized request;
        # single-chunk calls retain calibrated probability behavior.
        mode = self.blend_mode if len(clean) >= 8 else "probability"
        keys = [chunk_tie_key(chunk) for chunk in clean]
        raw = blend_branches(branches, self.model.branch_weights_, mode, tie_keys=keys)
        return raw, branches, matrix

    def map_scores(
        self,
        raw: Sequence[float] | np.ndarray,
        *,
        chunks: Sequence[Sequence[Dict[str, Any]]] | None = None,
        batch_map: bool = True,
    ) -> np.ndarray:
        values = np.asarray(raw, dtype=float)
        if batch_map and len(values) >= 8:
            keys = None if chunks is None else [chunk_tie_key(chunk) for chunk in chunks]
            return exact_rank_map(values, self.batch_top_fraction, tie_keys=keys)
        return apply_mapper(values, self.mapper)

    def predict_chunks(
        self,
        chunks: List[List[Dict[str, Any]]],
        *,
        batch_map: bool = True,
        return_diagnostics: bool = False,
    ):
        if not chunks:
            return ([], {}) if return_diagnostics else []
        clean = self._clean(chunks)
        raw, branches, matrix = self._raw_from_clean(clean)
        scores = self.map_scores(raw, chunks=clean, batch_map=batch_map)
        scores = np.nan_to_num(scores, nan=0.5, posinf=0.99, neginf=0.01)
        output = [round(float(np.clip(value, 0.01, 0.99)), 8) for value in scores]
        if not return_diagnostics:
            return output
        lower = self.feature_reference["q01"]
        upper = self.feature_reference["q99"]
        outside = (matrix < lower) | (matrix > upper)
        training_iqr = np.maximum(
            self.feature_reference["q75"] - self.feature_reference["q25"],
            1e-9,
        )
        median_shift = np.abs(
            np.median(matrix, axis=0) - self.feature_reference["median"]
        ) / training_iqr
        diagnostics = {
            "chunk_count": len(clean),
            "hand_counts": [len(chunk) for chunk in clean],
            "blend_mode": self.blend_mode if len(clean) >= 8 else "probability",
            "branch_names": list(BRANCH_NAMES),
            "branch_disagreement_mean": float(np.std(branches, axis=1).mean()),
            "raw_min": float(raw.min()),
            "raw_max": float(raw.max()),
            "raw_std": float(raw.std()),
            "feature_outside_q01_q99_rate": float(outside.mean()),
            "features_majority_outside_q01_q99": int(np.sum(outside.mean(axis=0) > 0.5)),
            "median_feature_shift_iqr": float(np.median(median_shift)),
            "p90_feature_shift_iqr": float(np.quantile(median_shift, 0.90)),
            "batch_mapping": bool(batch_map and len(clean) >= 8),
            "positive_count": int(np.sum(scores >= 0.5)),
            "positive_rate": float(np.mean(scores >= 0.5)),
        }
        return output, diagnostics

    def predict_chunk(self, chunk: List[Dict[str, Any]]) -> float:
        return self.predict_chunks([chunk], batch_map=False)[0]

    def raw_scores(self, chunks: List[List[Dict[str, Any]]]) -> np.ndarray:
        if not chunks:
            return np.zeros(0, dtype=float)
        clean = self._clean(chunks)
        raw, _, _ = self._raw_from_clean(clean)
        return raw
