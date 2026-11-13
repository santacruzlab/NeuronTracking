"""Checks for identifiable and unidentifiable learning curves and setting adjustment."""
import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_session_learning import fit_decay, assess_curve, regression_trend, signed_angle


class LearningTests(unittest.TestCase):
    def test_decay_recovery_and_units(self):
        x = np.arange(16)
        y = 1 + 0.8 * np.exp(-x / 3)
        np.testing.assert_allclose(fit_decay(x, y), [1, 0.8, 3], atol=1e-5)
        rec = assess_curve(x, y, np.random.default_rng(1), draws=100)
        self.assertEqual(rec["status"], "estimable")
        self.assertAlmostEqual(rec["tau_trials"], 24, places=3)
        self.assertAlmostEqual(rec["rate_per_set"], 1 / 3, places=4)

    def test_flat_curve_does_not_report_learning_rate(self):
        rec = assess_curve(np.arange(16), np.ones(16), np.random.default_rng(1), draws=100)
        self.assertIn("weak_amplitude", rec["status"])
        self.assertTrue(np.isnan(rec["tau_sets"]))

    def test_fast_unresolved_curve_does_not_report_rate(self):
        x = np.arange(16)
        rec = assess_curve(x, 1 + np.exp(-x / 0.1), np.random.default_rng(1), draws=100)
        self.assertIn("tau_at_resolution_or_bound", rec["status"])
        self.assertTrue(np.isnan(rec["tau_sets"]))

    def test_setting_change_is_not_mislabeled_as_session_trend(self):
        x = np.arange(40)
        regimes = ["small"] * 20 + ["large"] * 20
        y = np.r_[np.ones(20), np.full(20, 2)]
        result = regression_trend(x, y, regimes, True, np.random.default_rng(1))
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["slope"], 0, places=10)
        unadjusted = regression_trend(x, y, regimes, False, np.random.default_rng(1))
        self.assertGreater(unadjusted["slope"], 0)
        self.assertEqual(signed_angle(310), -50)
        self.assertEqual(signed_angle(270), -90)


if __name__ == "__main__":
    unittest.main()
