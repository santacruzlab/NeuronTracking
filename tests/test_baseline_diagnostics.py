import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from baseline_diagnostics import bootstrap_means, summarize


class BaselineDiagnosticsTests(unittest.TestCase):
    def inputs(self):
        settings = pd.DataFrame([dict(file="test.hdf", monkey="airp", date="2021-01-01",
                                      rotation_signed_deg=50, catalog_rotation_signed_deg=50,
                                      rotation_matches_catalog=True)])
        trials = pd.DataFrame([dict(file="test.hdf", block=2, set_number=i // 8 + 1,
                                    trial=i + 1, duration_s=2 + (i % 8) / 8, valid=True)
                               for i in range(24)])
        sets = pd.DataFrame([dict(file="test.hdf", block=2, set_number=i, valid=True,
                                  eight_targets_once=True) for i in (1, 2, 3)])
        attempts = pd.DataFrame([dict(file="test.hdf", block=1, valid_timing=True,
                                      outcome="success", error_clamp=0, perturbation=0,
                                      reward_number=i + 1, target_deg=(i % 8) * 45,
                                      duration_s=1 + (i % 8) / 8) for i in range(128)])
        return trials, sets, attempts, settings

    def run_summary(self, inputs):
        return summarize(*inputs, 100, np.random.default_rng(1))

    def test_normalization_is_invariant_to_duration_units(self):
        inputs = self.inputs()
        original = self.run_summary(inputs)
        inputs[0]["duration_s"] *= 1000
        inputs[2]["duration_s"] *= 1000
        scaled = self.run_summary(inputs)
        for column in ("ratio", "z_mean", "relative_sem", "cv", "ratio_lo_b4", "ratio_hi_b8"):
            np.testing.assert_allclose(original[column], scaled[column])
        self.assertTrue((original.ratio > 1).all())
        self.assertTrue((original.z_mean > 0).all())

    def test_invalid_first_set_is_not_replaced_by_later_trials(self):
        inputs = self.inputs()
        inputs[1].loc[0, "valid"] = False
        result = self.run_summary(inputs)
        self.assertFalse(result.early_valid.any())
        self.assertTrue(result.mean_s.isna().all())
        self.assertTrue(result.ratio.isna().all())

    def test_baseline_excludes_perturbed_trials_and_requires_coverage(self):
        inputs = self.inputs()
        inputs[2].loc[inputs[2].target_deg == 0, "perturbation"] = 1
        result = self.run_summary(inputs)
        self.assertTrue(result.early_valid.all())
        self.assertFalse(result.baseline_valid.any())
        self.assertTrue(result.ratio.isna().all())

    def test_full_length_circular_block_has_no_mean_resampling_variation(self):
        # An eight-trial block cannot estimate early uncertainty from only eight trials.
        means = bootstrap_means(np.arange(8), 100, 8, np.random.default_rng(2))
        np.testing.assert_allclose(means, 3.5)


if __name__ == "__main__":
    unittest.main()
