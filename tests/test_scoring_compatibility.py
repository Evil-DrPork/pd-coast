from __future__ import annotations

import unittest

import numpy as np

from detection_model.model.scoring import reward_metrics, validator_reward
from poker44.score.scoring import reward as authoritative_reward


class ScoringCompatibilityTests(unittest.TestCase):
    def test_default_wrapper_exactly_matches_authoritative_reward(self) -> None:
        labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
        scores = np.asarray([0.12, 0.91, 0.35, 0.76, 0.51, 0.64, 0.22, 0.88])
        expected_reward, expected_details = authoritative_reward(scores, labels)
        actual_reward, actual_details = validator_reward(scores, labels)
        self.assertEqual(actual_reward, expected_reward)
        self.assertEqual(actual_details, expected_details)

    def test_reward_metrics_exposes_current_rank_and_threshold_components(self) -> None:
        labels = [0, 1, 0, 1]
        scores = [0.10, 0.90, 0.20, 0.80]
        metrics = reward_metrics(labels, scores)
        self.assertAlmostEqual(metrics["validator_reward"], 1.0)
        self.assertAlmostEqual(metrics["validator_ap_score"], 1.0)
        self.assertAlmostEqual(metrics["validator_bot_recall"], 1.0)
        self.assertAlmostEqual(metrics["hard_fpr"], 0.0)
        self.assertAlmostEqual(metrics["hard_bot_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
