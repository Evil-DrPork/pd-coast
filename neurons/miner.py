"""Poker44 miner using the V3/V4 order-invariant bot detectors.

Pipeline per validator chunk:
    chunk -> order-invariant hand/chunk features (actions sorted by action_id)
          -> selected V3/V4 detector (regularized ensemble + monotone batch
             boundary mapping)
          -> one calibrated risk_score per chunk

An optional top-K cap forces only the K highest-scoring chunks above the 0.5
boundary (FPR protection). Competition deployments should set
``P44_REQUIRE_MODEL=1`` so a missing, mismatched, or corrupt artifact stops the
process instead of silently serving heuristic scores.
"""

import hashlib
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Tuple

import bittensor as bt
from dotenv import load_dotenv

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

load_dotenv()

MODEL_REPO_PATH = "detection_model"

try:
    from detection_model.model_v3.inference import Poker44V3Detector
    MODEL_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - keep the miner alive on import error
    Poker44V3Detector = None
    MODEL_IMPORT_ERROR = str(exc)

try:
    from detection_model.model_v3_large.inference import Poker44V3LargeDetector
    LARGE_MODEL_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - stable v3 must remain independently usable
    Poker44V3LargeDetector = None
    LARGE_MODEL_IMPORT_ERROR = str(exc)

try:
    from detection_model.model_v4.inference import Poker44V4Detector
    V4_MODEL_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - stable v3 must remain independently usable
    Poker44V4Detector = None
    V4_MODEL_IMPORT_ERROR = str(exc)


def _sha256_file(path: str | Path) -> str:
    path = Path(path).expanduser()
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_paths(paths: list[str | Path]) -> list[Path]:
    return [p for p in (Path(x).expanduser() for x in paths) if p.exists() and p.is_file()]


def _git(args: list[str], repo_root: Path) -> str:
    """Run a git command in repo_root, returning stripped stdout or "" on failure."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:  # pragma: no cover - git missing / not a repo
        return ""
    if out.returncode != 0:
        return ""
    return out.stdout.strip()


def _git_commit(repo_root: Path) -> str:
    """Current HEAD commit hash (manifest policy requires a real git commit)."""
    return _git(["rev-parse", "HEAD"], repo_root)


def _normalize_repo_url(url: str) -> str:
    """Normalize an origin URL to a public https form, stripping creds/.git suffix."""
    url = url.strip()
    if not url:
        return ""
    if url.startswith("git@"):  # git@github.com:owner/repo.git
        host, _, path = url[len("git@"):].partition(":")
        url = f"https://{host}/{path}"
    elif url.startswith("ssh://git@"):
        url = "https://" + url[len("ssh://git@"):]
    if url.endswith(".git"):
        url = url[: -len(".git")]
    return url


def _git_remote_url(repo_root: Path) -> str:
    """Public URL of the origin remote, normalized to https without a .git suffix."""
    return _normalize_repo_url(_git(["remote", "get-url", "origin"], repo_root))


class Miner(BaseMinerNeuron):
    """Score each DetectionSynapse chunk with a trained Poker44 detector."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("Poker44 trained-model miner started")

        repo_root = Path(__file__).resolve().parents[1]
        model_repo_root = Path(MODEL_REPO_PATH).expanduser()

        self.model_path = os.getenv("P44_MODEL_PATH", "detection_model/artifacts/p44_v3.joblib")
        self.prediction_threshold = float(os.getenv("P44_PREDICTION_THRESHOLD", "0.5"))
        self.require_model = os.getenv("P44_REQUIRE_MODEL", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        # Top-K bot cap. When enabled, only the K highest-scoring chunks in a
        # synapse are pushed above the 0.5 boundary (classified bot); every other
        # chunk is forced below 0.5 (human). This hard-caps false positives to
        # protect the validator's FPR cliff. P44_TOP_K is an absolute count;
        # P44_TOP_K_FRAC (0..1) is a batch-relative fraction used when the
        # absolute count is unset (more robust to varying synapse sizes). Both
        # default to disabled (0) -> scores pass through unchanged.
        self.top_k = int(os.getenv("P44_TOP_K", "0"))
        self.top_k_frac = float(os.getenv("P44_TOP_K_FRAC", "0"))
        try:
            self.latency_warn_seconds = float(
                os.getenv("P44_LATENCY_WARN_SECONDS", "15")
            )
        except (TypeError, ValueError):
            self.latency_warn_seconds = 15.0

        self.detector = None
        self.model_manifest = self._build_model_manifest(repo_root, model_repo_root)
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)

        self._log_manifest_startup(repo_root)
        self._load_trained_model()
        bt.logging.info(f"Axon created: {self.axon}")

    # ------------------------------------------------------------------ setup

    def _build_model_manifest(self, repo_root: Path, model_repo_root: Path) -> dict:
        model_artifact_path = Path(self.model_path).expanduser()
        detector_variant = os.getenv("P44_DETECTOR_VARIANT", "v3").strip().lower()
        is_large = detector_variant in {"large", "v3_large"}
        is_v4 = detector_variant in {"v4", "v4_coherent", "coherent"}
        implementation_paths = [
            Path(__file__).resolve(),
            model_repo_root / "model_v3" / "inference.py",
            model_repo_root / "model_v3" / "features.py",
            model_repo_root / "model_v3" / "schema.py",
            model_repo_root / "model_v3" / "model.py",
            model_repo_root / "model_v3" / "neural.py",
            model_repo_root / "model_v3" / "calibration.py",
            model_repo_root / "model_v3" / "metrics.py",
            model_repo_root / "model_v3" / "train.py",
        ]
        if is_large:
            implementation_paths.extend(
                [
                    model_repo_root / "model_v3_large" / "inference.py",
                    model_repo_root / "model_v3_large" / "features.py",
                    model_repo_root / "model_v3_large" / "model.py",
                    model_repo_root / "model_v3_large" / "tabular.py",
                    model_repo_root / "model_v3_large" / "augmentation.py",
                ]
            )
        if is_v4:
            implementation_paths.extend(
                [
                    model_repo_root / "model_v4" / "inference.py",
                    model_repo_root / "model_v4" / "features.py",
                    model_repo_root / "model_v4" / "model.py",
                    model_repo_root / "model_v4" / "mapping.py",
                    model_repo_root / "model_v4" / "provenance.py",
                    model_repo_root / "model_v4" / "requirements.txt",
                    model_repo_root / "model_v4" / "train.py",
                ]
            )
        implementation_files = _existing_paths(implementation_paths)
        artifact_sha256 = os.getenv("POKER44_MODEL_ARTIFACT_SHA256", _sha256_file(model_artifact_path))

        # Auto-derive the git identity so the manifest is transparent-compliant
        # out of the box (env vars still take precedence for overrides).
        repo_url = os.getenv("P44_MANIFEST_REPO_URL") or _git_remote_url(repo_root)
        repo_commit = (
            os.getenv("P44_MANIFEST_REPO_COMMIT")
            or os.getenv("P44_MODEL_REPO_COMMIT")
            or _git_commit(repo_root)
        )

        return build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=implementation_files,
            defaults={
                "open_source": True,
                "model_name": os.getenv(
                    "P44_MANIFEST_MODEL_NAME",
                    (
                        "p44-v4-coherent-rank-robust"
                        if is_v4
                        else "p44-v3-large-multiscale"
                        if is_large
                        else "p44-v3-order-free-ensemble"
                    ),
                ),
                "model_version": os.getenv(
                    "P44_MANIFEST_MODEL_VERSION",
                    "4.1.0" if is_v4 else "3.1.0" if is_large else "3.0.0",
                ),
                "framework": os.getenv(
                    "P44_MANIFEST_FRAMEWORK",
                    (
                        "scikit-learn-coherent-rank-robust-ensemble"
                        if is_v4
                        else "scikit-learn-order-free-ensemble"
                    ),
                ),
                "license": os.getenv("P44_MANIFEST_LICENSE", "MIT"),
                "repo_url": repo_url,
                "repo_commit": repo_commit,
                "artifact_url": os.getenv("P44_MANIFEST_ARTIFACT_URL", ""),
                "artifact_sha256": artifact_sha256,
                "model_card_url": os.getenv("P44_MANIFEST_MODEL_CARD_URL", ""),
                "training_data_statement": (
                    "Trained only on the public Poker44 benchmark, canonicalized through the "
                    "validator's public miner-payload transform to match the served distribution. "
                    "No validator-only evaluation data, hidden labels, or leaked validator payloads "
                    "were used."
                ),
                "training_data_sources": [
                    "public Poker44 benchmark chunks (api.poker44.net)",
                    "canonicalized copies of the public benchmark (validator payload_view transform)",
                ],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data, live eval "
                    "batches, hidden validator labels, or any private validator data."
                ),
                "data_attestation": (
                    "All training data is the public Poker44 benchmark (api.poker44.net) and "
                    "canonicalized copies derived from it. No private, scraped, or validator-side "
                    "data is used; the published repo and commit reproduce the full model flow."
                ),
                "inference_mode": "remote",
                "notes": (
                    f"Poker44 {'coherent real-data challenger' if is_v4 else 'large-chunk challenger' if is_large else 'model_v3 detector'}. "
                    "Distributional order-invariant chunk features -> "
                    "regularized linear/tree ensemble plus human-tail prototype -> monotone "
                    "query-batch boundary mapper. All hands are used; action order is preserved "
                    "only within each hand and no cross-hand order is assumed. "
                    + (
                        "Training includes 50-75 and 80-100 hand same-label public-benchmark "
                        "mixtures plus deterministic permutation-invariant micro-bags. "
                        if is_large else ""
                    )
                    + (
                        "Training uses real public chunks only, broad chronological history, "
                        "and exact within-chunk behavioral-coherence signatures; no synthetic "
                        "cross-source merges or micro-bag pseudo-labels are used. Request-relative "
                        "feature-rank branches protect ranking under numeric domain shift. "
                        if is_v4 else ""
                    )
                    + f"Local artifact: {model_artifact_path}"
                ),
            },
        )

    def _load_trained_model(self) -> None:
        variant = os.getenv("P44_DETECTOR_VARIANT", "v3").strip().lower()
        if variant in {"v4", "v4_coherent", "coherent"}:
            detector_class = Poker44V4Detector
            import_error = V4_MODEL_IMPORT_ERROR
        elif variant in {"large", "v3_large"}:
            detector_class = Poker44V3LargeDetector
            import_error = LARGE_MODEL_IMPORT_ERROR
        elif variant in {"v3", "stable", "tabular"}:
            detector_class = Poker44V3Detector
            import_error = MODEL_IMPORT_ERROR
        else:
            message = f"Unknown P44_DETECTOR_VARIANT={variant!r}"
            if self.require_model:
                raise ValueError(message)
            bt.logging.error(f"{message}. Miner will use the heuristic fallback.")
            return

        if detector_class is None:
            message = f"Could not import detector variant {variant!r}: {import_error}"
            if self.require_model:
                raise RuntimeError(message)
            bt.logging.error(message)
            bt.logging.error("Miner will use the heuristic fallback.")
            return

        model_path = Path(self.model_path).expanduser()
        if not model_path.exists():
            message = f"Model artifact not found: {model_path}"
            if self.require_model:
                raise FileNotFoundError(message)
            bt.logging.error(f"{message}. Using heuristic fallback.")
            return

        try:
            expected_sha256 = os.getenv("POKER44_MODEL_ARTIFACT_SHA256", "").strip().lower()
            actual_sha256 = _sha256_file(model_path).lower()
            if expected_sha256:
                if len(expected_sha256) != 64 or any(
                    character not in "0123456789abcdef" for character in expected_sha256
                ):
                    raise ValueError("POKER44_MODEL_ARTIFACT_SHA256 must be 64 hex characters")
                if actual_sha256 != expected_sha256:
                    raise ValueError(
                        "model artifact SHA-256 mismatch before deserialization: "
                        f"expected={expected_sha256} actual={actual_sha256}"
                    )
            bt.logging.info(f"Loading {variant} ensemble from: {model_path}")
            self.detector = detector_class.load(model_path)
            env_threshold = os.getenv("P44_PREDICTION_THRESHOLD")
            if env_threshold is not None:
                self.prediction_threshold = float(env_threshold)
            bt.logging.info(f"{variant} detector loaded successfully")
            bt.logging.info(
                f"features: {len(self.detector.artifact.get('feature_names', []))} | "
                f"training chunks: {self.detector.artifact.get('training_count')} | "
                f"threshold={self.prediction_threshold}"
            )
        except Exception as exc:
            if self.require_model:
                raise RuntimeError(f"Required trained model failed to load: {exc}") from exc
            bt.logging.error(f"Failed to load trained model: {exc}. Using heuristic fallback.")
            self.detector = None

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"digest={self.manifest_digest} "
            f"open_source={self.model_manifest.get('open_source')}"
        )

    # ---------------------------------------------------------------- scoring

    def _predict_chunks(self, chunks: list[list[dict]]) -> list[float]:
        """Whole-chunk, order-invariant scoring via the selected detector."""
        if self.detector is None:
            return [self.score_chunk(chunk) for chunk in chunks]
        return self.detector.predict_chunks(chunks)

    def _finalize_score(self, score: float) -> float:
        """Final value sent to the validator.

        V3/V4 detectors already apply their monotone mapper, so only finite
        clamping and rounding are required here.
        """
        score = max(0.0, min(1.0, float(score)))
        if (
            Poker44V3Detector is not None
            and isinstance(self.detector, Poker44V3Detector)
        ) or (
            Poker44V4Detector is not None
            and isinstance(self.detector, Poker44V4Detector)
        ):
            return round(score, 6)
        threshold = self.prediction_threshold
        if threshold <= 0.0 or threshold >= 1.0:
            return round(score, 6)
        if score <= threshold:
            calibrated = (score / threshold) * 0.5
        else:
            calibrated = 0.5 + ((score - threshold) / (1.0 - threshold)) * 0.5
        return round(max(0.0, min(1.0, calibrated)), 6)

    # ------------------------------------------------------------------ top-k

    def _resolve_top_k(self, n: int) -> int:
        """Effective K for a batch of ``n`` chunks (0 = disabled / no-op)."""
        if self.top_k and self.top_k > 0:
            return min(self.top_k, n)
        if self.top_k_frac and self.top_k_frac > 0.0:
            return max(1, min(n, int(round(self.top_k_frac * n))))
        return 0

    def _apply_top_k(self, scores: list[float]) -> list[float]:
        """Force only the top-K scores above 0.5, everyone else below.

        Ranking is preserved exactly (the validator's average-precision term is
        rank-based, so it is unaffected). Membership in the "bot" set is decided
        by rank — not by a value threshold — so ties across the boundary can
        never leak an extra positive: the batch ends with *exactly* K chunks at
        risk >= 0.5. Within each band the original score spacing is kept via a
        min-max rescale so the miner still emits a graded, monotone signal.
        """
        n = len(scores)
        k = self._resolve_top_k(n)
        if k <= 0 or k >= n:
            return scores  # disabled, or no "extra" chunks to demote -> no-op

        order = sorted(range(n), key=lambda i: (scores[i], i), reverse=True)
        bot_idx = set(order[:k])
        bot_vals = [scores[i] for i in order[:k]]
        hum_vals = [scores[i] for i in order[k:]]
        b_lo, b_hi = min(bot_vals), max(bot_vals)
        h_lo, h_hi = min(hum_vals), max(hum_vals)

        out: list[float] = []
        for i, s in enumerate(scores):
            if i in bot_idx:
                frac = 0.0 if b_hi == b_lo else (s - b_lo) / (b_hi - b_lo)
                out.append(round(0.5 + 1e-6 + frac * (0.5 - 1e-6), 6))  # (0.5, 1.0]
            else:
                frac = 0.0 if h_hi == h_lo else (s - h_lo) / (h_hi - h_lo)
                out.append(round(frac * (0.5 - 1e-6), 6))  # [0.0, 0.5)
        return out

    # ----------------------------------------------------------------- serve

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Return one risk_score per chunk (close to 1 = bot-like)."""
        chunks = synapse.chunks or []
        bt.logging.info(f"Received synapse from validator hotkey: {synapse.dendrite.hotkey}")

        if not chunks:
            synapse.risk_scores = []
            synapse.predictions = []
            synapse.model_manifest = dict(self.model_manifest)
            return synapse

        request_started = time.perf_counter()
        try:
            if self.detector is None:
                bt.logging.warning("Trained model not loaded. Using heuristic fallback.")
                raw_scores = [self.score_chunk(chunk) for chunk in chunks]
            else:
                raw_scores = self._predict_chunks(chunks)

            if len(raw_scores) != len(chunks):
                raise ValueError(f"Wrong score count: chunks={len(chunks)}, scores={len(raw_scores)}")

            scores = [self._finalize_score(s) for s in raw_scores]
            scores = self._apply_top_k(scores)
            synapse.risk_scores = scores
            synapse.predictions = [s >= 0.5 for s in scores]
            synapse.model_manifest = dict(self.model_manifest)

            effective_k = self._resolve_top_k(len(scores))
            elapsed = time.perf_counter() - request_started
            bt.logging.info(
                f"Scored {len(chunks)} chunks with "
                f"{'trained ensemble' if self.detector else 'heuristic fallback'} | "
                f"latency={elapsed:.3f}s | "
                f"top_k={effective_k or 'off'} bots={sum(s >= 0.5 for s in scores)} | "
                f"preview={scores}"
            )
            if self.latency_warn_seconds > 0 and elapsed >= self.latency_warn_seconds:
                bt.logging.warning(
                    f"Inference latency sentry: {elapsed:.3f}s exceeded "
                    f"P44_LATENCY_WARN_SECONDS={self.latency_warn_seconds:.3f}s"
                )
            return synapse

        except Exception as exc:
            elapsed = time.perf_counter() - request_started
            bt.logging.error(f"Inference failed after {elapsed:.3f}s: {exc}")
            # The validator scores risk_scores, not the advisory predictions.
            # A neutral-below-boundary response cannot manufacture false bots.
            fallback = [0.49 for _ in chunks]
            synapse.risk_scores = fallback
            synapse.predictions = [False for _ in fallback]
            synapse.model_manifest = dict(self.model_manifest)
            return synapse

    # --------------------------------------------------------- heuristic fallback

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _score_hand(cls, hand: dict) -> float:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}

        action_counts = Counter(action.get("action_type") for action in actions)
        meaningful = max(1, sum(action_counts.get(k, 0) for k in ("call", "check", "bet", "raise", "fold")))
        call_ratio = action_counts.get("call", 0) / meaningful
        check_ratio = action_counts.get("check", 0) / meaningful
        fold_ratio = action_counts.get("fold", 0) / meaningful
        raise_ratio = action_counts.get("raise", 0) / meaningful
        street_depth = len(streets) / 3.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0
        player_count_signal = (6 - min(len(players), 6)) / 4.0 if players else 0.0

        score = 0.0
        score += 0.32 * street_depth
        score += 0.22 * showdown_flag
        score += 0.18 * cls._clamp01(call_ratio / 0.35)
        score += 0.12 * cls._clamp01(check_ratio / 0.30)
        score += 0.08 * cls._clamp01(player_count_signal)
        score -= 0.18 * cls._clamp01(fold_ratio / 0.55)
        score -= 0.10 * cls._clamp01(raise_ratio / 0.20)
        return cls._clamp01(score)

    @classmethod
    def score_chunk(cls, chunk: list[dict]) -> float:
        if not chunk:
            return 0.5
        hand_scores = [cls._score_hand(hand) for hand in chunk]
        return round(cls._clamp01(sum(hand_scores) / len(hand_scores)), 6)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 v3 trained-model miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
