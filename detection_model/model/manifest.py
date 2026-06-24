from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model_manifest(
    repo_url: str,
    repo_commit: str,
    artifact_path: str | Path,
    model_name: str = "p44-hybrid-chunk-transformer",
    model_version: str = "1.0.0",
    framework: str = "PyTorch",
    license_name: str = "MIT",
    training_data_sources: List[str] | None = None,
) -> Dict[str, Any]:
    training_data_sources = training_data_sources or ["Poker44 public miner benchmark", "local legal synthetic simulations"]
    return {
        "open_source": True,
        "repo_url": repo_url,
        "repo_commit": repo_commit,
        "model_name": model_name,
        "model_version": model_version,
        "framework": framework,
        "license": license_name,
        "training_data_statement": "Trained only on public benchmark data and legal local synthetic simulations; no validator-only evaluation batches or leaked live payloads were used.",
        "training_data_sources": training_data_sources,
        "data_attestation": "No /internal/eval/current data, validator-served chunks, hidden labels, live provider SQL data, or memorized payload hashes were used for training.",
        "artifact_sha256": sha256_file(artifact_path) if Path(artifact_path).exists() else None,
        "implementation_sha256": None,
    }
