"""Reproducible fingerprints for the complete V4 inference implementation."""

from __future__ import annotations

import hashlib
import platform
from importlib import metadata
from pathlib import Path


def _normalized_source_bytes(path: Path) -> bytes:
    """Return text bytes with checkout-independent line endings."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def detector_runtime_sha256() -> str:
    project_root = Path(__file__).resolve().parents[2]
    dependencies = (
        Path(__file__).resolve(),
        project_root / "detection_model" / "model_v4" / "features.py",
        project_root / "detection_model" / "model_v4" / "mapping.py",
        project_root / "detection_model" / "model_v4" / "model.py",
        project_root / "detection_model" / "model_v4" / "inference.py",
        project_root / "detection_model" / "model_v3" / "features.py",
        project_root / "detection_model" / "model_v3" / "schema.py",
        project_root / "detection_model" / "model_v3" / "calibration.py",
    )
    digest = hashlib.sha256()
    for path in dependencies:
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        payload = _normalized_source_bytes(path)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


DETECTOR_RUNTIME_SHA256 = detector_runtime_sha256()


def runtime_library_versions() -> dict[str, str]:
    """Versions required for supported sklearn/joblib artifact loading."""
    return {
        "python": platform.python_version(),
        "numpy": metadata.version("numpy"),
        "scipy": metadata.version("scipy"),
        "scikit-learn": metadata.version("scikit-learn"),
        "joblib": metadata.version("joblib"),
    }


RUNTIME_LIBRARY_VERSIONS = runtime_library_versions()
