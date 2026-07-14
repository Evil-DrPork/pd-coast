from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from detection_model.model_v3.calibration import fit_fixed_mapper
from detection_model.model_v3.schema import Chunk, load_chunks
from detection_model.model_v4.features import (
    BASE_FEATURE_NAMES,
    COHERENT_FEATURE_NAMES,
    FEATURE_IMPLEMENTATION_SHA256,
    FEATURE_NAMES,
    FEATURE_SCHEMA_SHA256,
    _normalized_source_bytes as normalized_feature_source,
    matrix_for_chunks,
)
from detection_model.model_v4.inference import Poker44V4Detector
from detection_model.model_v4.mapping import chunk_tie_key, exact_rank_map, positive_count
from detection_model.model_v4.model import (
    BRANCH_NAMES,
    CoherentEnsemble,
    ModelConfig,
    blend_branches,
    percentile_feature_matrix,
)
from detection_model.model_v4.provenance import (
    DETECTOR_RUNTIME_SHA256,
    RUNTIME_LIBRARY_VERSIONS,
    _normalized_source_bytes as normalized_runtime_source,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRUSTED_DATED_DATA = (
    REPO_ROOT / "detection_model" / "data_v3" / "benchmark_chunks_2026-05-26.json"
)


def _generated_fixture_chunks() -> list[Chunk]:
    """Small deterministic fallback so clean-checkout CI never skips V4."""
    chunks: list[Chunk] = []
    for label in (0, 1):
        for chunk_index in range(4):
            hands = []
            for hand_index in range(8):
                hero = 1 + ((chunk_index + hand_index) % 3)
                if label:
                    kinds = ("raise", "call", "raise", "fold")
                    amounts = (2.0, 2.0, 6.0, 0.0)
                else:
                    rotations = (
                        ("check", "call", "fold", "check"),
                        ("call", "check", "bet", "fold"),
                        ("fold", "check", "call", "check"),
                    )
                    kinds = rotations[(chunk_index + hand_index) % len(rotations)]
                    amounts = (0.0, 1.0 + 0.25 * hand_index, 2.5, 0.0)
                actions = []
                pot = 1.5
                for action_index, (kind, amount) in enumerate(zip(kinds, amounts), start=1):
                    after = pot + amount
                    actions.append(
                        {
                            "action_id": action_index,
                            "action_type": kind,
                            "actor_seat": 1 + ((hero + action_index) % 3),
                            "amount": amount * 0.02,
                            "normalized_amount_bb": amount,
                            "pot_before": pot * 0.02,
                            "pot_after": after * 0.02,
                            "street": "preflop" if action_index < 3 else "flop",
                        }
                    )
                    pot = after
                hands.append(
                    {
                        "metadata": {
                            "ante": 0.0,
                            "bb": 0.02,
                            "button_seat": 1 + (hand_index % 3),
                            "hero_seat": hero,
                            "max_seats": 3,
                            "sb": 0.01,
                        },
                        "players": [
                            {"seat": seat, "starting_stack": 1.8 + 0.1 * seat}
                            for seat in (1, 2, 3)
                        ],
                        "actions": actions,
                        "streets": [{"street": "preflop"}, {"street": "flop"}],
                    }
                )
            chunks.append(Chunk(len(chunks), hands, label, "fixture-2026-01-01"))
    return chunks


def _balanced_subset(chunks: list[Chunk], per_label: int = 4) -> list[Chunk]:
    selected: list[Chunk] = []
    for label in (0, 1):
        candidates = [chunk for chunk in chunks if chunk.label == label]
        if len(candidates) < per_label:
            raise AssertionError(f"trusted fixture has fewer than {per_label} label={label} chunks")
        selected.extend(candidates[:per_label])
    return selected


class ModelV4CoherentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trusted_chunks: list[Chunk]
        if TRUSTED_DATED_DATA.exists():
            cls.trusted_chunks = _balanced_subset(load_chunks(TRUSTED_DATED_DATA))
        else:
            cls.trusted_chunks = _generated_fixture_chunks()

    def _require_trusted_chunks(self) -> list[Chunk]:
        return self.trusted_chunks

    def test_feature_schema_is_unique_finite_and_excludes_leakage_fields(self) -> None:
        chunks = self._require_trusted_chunks()
        self.assertEqual(FEATURE_NAMES[: len(BASE_FEATURE_NAMES)], BASE_FEATURE_NAMES)
        self.assertEqual(len(FEATURE_NAMES), len(set(FEATURE_NAMES)))
        self.assertTrue(COHERENT_FEATURE_NAMES)

        forbidden = (
            "chunk_id",
            "hand_id",
            "source",
            "date",
            "label",
            "is_bot",
            "outcome",
            "winner",
            "payout",
            "hole",
            "uid",
            "hand_count",
            "n_hands",
        )
        leaked = {
            name: token
            for name in FEATURE_NAMES
            for token in forbidden
            if token in name.lower()
        }
        self.assertEqual(leaked, {})

        matrix = matrix_for_chunks([chunks[0].hands, []])
        self.assertEqual(matrix.shape, (2, len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(matrix).all())

    def test_provenance_normalizes_windows_and_unix_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            windows = Path(directory) / "windows.py"
            unix = Path(directory) / "unix.py"
            windows.write_bytes(b"alpha\r\nbeta\r\n")
            unix.write_bytes(b"alpha\nbeta\n")
            self.assertEqual(normalized_feature_source(windows), normalized_feature_source(unix))
            self.assertEqual(normalized_runtime_source(windows), normalized_runtime_source(unix))

    def test_features_and_tie_key_are_exactly_hand_permutation_invariant(self) -> None:
        chunk = self._require_trusted_chunks()[0].hands
        reversed_chunk = list(reversed(chunk))
        first = matrix_for_chunks([chunk])
        second = matrix_for_chunks([reversed_chunk])
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(chunk_tie_key(chunk), chunk_tie_key(reversed_chunk))

    def test_tie_key_ignores_private_and_identifier_fields(self) -> None:
        import copy

        chunk = self._require_trusted_chunks()[0].hands
        changed = copy.deepcopy(chunk)
        changed[0]["hand_id"] = "private-id"
        changed[0]["outcome"] = {"winners": ["secret"], "payouts": [999]}
        changed[0].setdefault("metadata", {})["rng_seed_commitment"] = "secret"
        if changed[0].get("players"):
            changed[0]["players"][0]["player_uid"] = "secret-player"
            changed[0]["players"][0]["hole_cards"] = ["As", "Ah"]
        self.assertEqual(chunk_tie_key(chunk), chunk_tie_key(changed))

    def test_exact_mapping_has_exact_budget_and_preserves_rank(self) -> None:
        raw = np.asarray(
            [0.17, 0.91, 0.34, 0.72, 0.08, 0.56, 0.99, 0.43, 0.63, 0.81] * 4,
            dtype=float,
        )
        # Make every raw value unique so strict order preservation is testable.
        raw += np.arange(raw.size, dtype=float) * 1e-5
        keys = [f"chunk-{index:02d}" for index in range(raw.size)]
        fraction = 0.125
        mapped = exact_rank_map(raw, fraction, tie_keys=keys)

        expected = positive_count(len(raw), fraction)
        self.assertEqual(expected, 5)
        self.assertEqual(int(np.sum(mapped >= 0.5)), expected)
        self.assertTrue(np.isfinite(mapped).all())
        self.assertTrue(np.array_equal(np.argsort(raw), np.argsort(mapped)))

    def test_exact_mapping_ties_are_request_permutation_invariant(self) -> None:
        raw = np.asarray([0.9, 0.2, 0.9, 0.7, 0.2, 0.9, 0.7, 0.4], dtype=float)
        keys = ["f", "a", "b", "h", "c", "d", "g", "e"]
        permutation = np.asarray([6, 2, 7, 0, 4, 1, 5, 3], dtype=int)

        first = exact_rank_map(raw, 0.25, tie_keys=keys)
        second = exact_rank_map(
            raw[permutation],
            0.25,
            tie_keys=[keys[index] for index in permutation],
        )
        by_key_first = dict(zip(keys, first))
        by_key_second = dict(zip((keys[index] for index in permutation), second))
        self.assertEqual(by_key_first, by_key_second)
        self.assertEqual(sum(value >= 0.5 for value in by_key_first.values()), 2)

    def test_percentile_blend_ties_are_request_permutation_invariant(self) -> None:
        branches = np.asarray(
            [
                [0.8, 0.2, 0.5],
                [0.8, 0.7, 0.1],
                [0.3, 0.7, 0.5],
                [0.3, 0.2, 0.9],
                [0.8, 0.2, 0.9],
                [0.3, 0.7, 0.1],
            ],
            dtype=float,
        )
        weights = np.asarray([0.45, 0.25, 0.30], dtype=float)
        keys = ["k5", "k1", "k3", "k0", "k4", "k2"]
        permutation = np.asarray([4, 1, 5, 0, 3, 2], dtype=int)

        first = blend_branches(branches, weights, "rank", tie_keys=keys)
        second = blend_branches(
            branches[permutation],
            weights,
            "rank",
            tie_keys=[keys[index] for index in permutation],
        )
        by_key_first = dict(zip(keys, first))
        by_key_second = dict(zip((keys[index] for index in permutation), second))
        for key in keys:
            self.assertEqual(by_key_first[key], by_key_second[key])

    def test_percentile_features_are_tie_and_request_permutation_invariant(self) -> None:
        matrix = np.asarray(
            [[4.0, 1.0], [2.0, 3.0], [4.0, 2.0], [1.0, 3.0], [2.0, 0.0]],
            dtype=float,
        )
        permutation = np.asarray([3, 0, 4, 2, 1], dtype=int)
        first = percentile_feature_matrix(matrix)
        second = percentile_feature_matrix(matrix[permutation])
        restored = np.empty_like(second)
        restored[permutation] = second
        self.assertTrue(np.array_equal(first, restored))
        self.assertEqual(first[0, 0], first[2, 0])
        self.assertEqual(first[1, 0], first[4, 0])

    def test_tiny_trusted_v4_artifact_roundtrip(self) -> None:
        chunks = self._require_trusted_chunks()
        hands = [chunk.hands for chunk in chunks]
        labels = np.asarray([chunk.label for chunk in chunks], dtype=int)
        self.assertEqual(set(labels.tolist()), {0, 1})
        self.assertEqual(len({chunk.source_date for chunk in chunks}), 1)

        matrix = matrix_for_chunks(hands)
        config = ModelConfig(trees=5, hist_iterations=5, max_depth=3, learning_rate=0.1)
        model = CoherentEnsemble(seed=44, config=config).fit(matrix, labels)
        self.assertEqual(model.coherent_extra.n_estimators, 5)
        self.assertEqual(model.coherent_forest.n_estimators, 5)
        subset = np.asarray([0, 1, 2, 4, 5, 7], dtype=int)
        full_branches = model.branch_scores(matrix)
        rebased = model.request_rebased_branches(
            matrix[subset],
            full_branches[subset],
        )
        direct = model.branch_scores(matrix[subset])
        self.assertTrue(np.allclose(rebased, direct, rtol=0.0, atol=1e-12))
        mapper = fit_fixed_mapper(model.probability_score(matrix), labels)
        feature_reference = {
            "q01": np.quantile(matrix, 0.01, axis=0),
            "q25": np.quantile(matrix, 0.25, axis=0),
            "median": np.quantile(matrix, 0.50, axis=0),
            "q75": np.quantile(matrix, 0.75, axis=0),
            "q99": np.quantile(matrix, 0.99, axis=0),
        }
        artifact = {
            "artifact_version": 4,
            "architecture": "coherent_real_rank_robust_v2",
            "model": model,
            "feature_names": list(FEATURE_NAMES),
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "feature_implementation_sha256": FEATURE_IMPLEMENTATION_SHA256,
            "detector_runtime_sha256": DETECTOR_RUNTIME_SHA256,
            "runtime_library_versions": dict(RUNTIME_LIBRARY_VERSIONS),
            "branch_names": list(BRANCH_NAMES),
            "feature_reference": feature_reference,
            "blend_mode": "rank",
            "mapper": mapper,
            "batch_top_fraction": 0.125,
            "training_count": len(chunks),
            "synthetic_training_count": 0,
        }

        before, before_diagnostics = Poker44V4Detector(artifact).predict_chunks(
            hands, return_diagnostics=True
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trusted_v4_roundtrip.joblib"
            joblib.dump(artifact, path)
            loaded = Poker44V4Detector.load(path)
            after, after_diagnostics = loaded.predict_chunks(hands, return_diagnostics=True)

        self.assertEqual(before, after)
        self.assertEqual(before_diagnostics, after_diagnostics)
        self.assertEqual(len(after), len(chunks))
        self.assertTrue(np.isfinite(np.asarray(after, dtype=float)).all())
        self.assertEqual(after_diagnostics["positive_count"], 1)
        self.assertIn("feature_outside_q01_q99_rate", after_diagnostics)
        self.assertIn("median_feature_shift_iqr", after_diagnostics)

    def test_detector_rejects_wrong_version_and_feature_or_branch_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a Poker44 V4 artifact"):
            Poker44V4Detector({"artifact_version": 3})

        with self.assertRaisesRegex(ValueError, "feature schema mismatch"):
            Poker44V4Detector(
                {
                    "artifact_version": 4,
                    "architecture": "coherent_real_rank_robust_v2",
                    "feature_names": FEATURE_NAMES[:-1],
                }
            )

        with self.assertRaisesRegex(ValueError, "branch schema mismatch"):
            Poker44V4Detector(
                {
                    "artifact_version": 4,
                    "architecture": "coherent_real_rank_robust_v2",
                    "feature_names": list(FEATURE_NAMES),
                    "model": SimpleNamespace(branch_names=("wrong",)),
                    "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                    "feature_implementation_sha256": FEATURE_IMPLEMENTATION_SHA256,
                    "detector_runtime_sha256": DETECTOR_RUNTIME_SHA256,
                    "runtime_library_versions": dict(RUNTIME_LIBRARY_VERSIONS),
                }
            )

        with self.assertRaisesRegex(ValueError, "schema fingerprint mismatch"):
            Poker44V4Detector(
                {
                    "artifact_version": 4,
                    "architecture": "coherent_real_rank_robust_v2",
                    "feature_names": list(FEATURE_NAMES),
                    "feature_schema_sha256": "wrong",
                }
            )

        with self.assertRaisesRegex(ValueError, "feature implementation mismatch"):
            Poker44V4Detector(
                {
                    "artifact_version": 4,
                    "architecture": "coherent_real_rank_robust_v2",
                    "feature_names": list(FEATURE_NAMES),
                    "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                    "feature_implementation_sha256": "wrong",
                }
            )

        with self.assertRaisesRegex(ValueError, "detector runtime mismatch"):
            Poker44V4Detector(
                {
                    "artifact_version": 4,
                    "architecture": "coherent_real_rank_robust_v2",
                    "feature_names": list(FEATURE_NAMES),
                    "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                    "feature_implementation_sha256": FEATURE_IMPLEMENTATION_SHA256,
                    "detector_runtime_sha256": "wrong",
                }
            )

        with self.assertRaisesRegex(ValueError, "runtime library mismatch"):
            Poker44V4Detector(
                {
                    "artifact_version": 4,
                    "architecture": "coherent_real_rank_robust_v2",
                    "feature_names": list(FEATURE_NAMES),
                    "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                    "feature_implementation_sha256": FEATURE_IMPLEMENTATION_SHA256,
                    "detector_runtime_sha256": DETECTOR_RUNTIME_SHA256,
                    "model": SimpleNamespace(branch_names=BRANCH_NAMES),
                    "runtime_library_versions": {"wrong": "0"},
                }
            )


if __name__ == "__main__":
    unittest.main()
