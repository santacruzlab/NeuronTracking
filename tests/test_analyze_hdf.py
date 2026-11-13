"""Synthetic HDF checks for trial boundaries and session comparability."""
import sys
import tempfile
import unittest
import csv
from argparse import Namespace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import tables

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyzeHDF import analyze_sessions, summarize_sets, compare_windows
from trial_utils import get_two_target_rewarded_trials


class TimingTests(unittest.TestCase):
    def extract(self, messages, blocks=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.hdf"
            task = np.zeros(500, dtype=[("block_type", "i4")])
            task["block_type"] = 2 if blocks is None else blocks
            msgs = np.array(messages, dtype=[("msg", "S32"), ("time", "i4")])
            with tables.open_file(path, "w") as hdf:
                hdf.create_table("/", "task", obj=task)
                hdf.create_table("/", "task_msgs", obj=msgs)
            return get_two_target_rewarded_trials(path, 2)

    def test_first_target_includes_both_targets_and_excludes_premove(self):
        tr = self.extract([(b"wait", 0), (b"premove", 1), (b"target", 8),
                           (b"hold", 9), (b"targ_transition", 10),
                           (b"target", 11), (b"hold", 114),
                           (b"targ_transition", 126), (b"reward", 127)])
        np.testing.assert_allclose(tr["duration"], [119 / 60])
        np.testing.assert_array_equal(tr["row_first_target"], [8])
        np.testing.assert_array_equal(tr["row_second_target"], [11])

    def test_failed_attempt_does_not_supply_next_reward_start(self):
        tr = self.extract([(b"premove", 1), (b"hold_penalty", 30),
                           (b"wait", 40), (b"reward", 100),
                           (b"wait", 110), (b"premove", 120), (b"target", 125),
                           (b"hold", 130), (b"targ_transition", 135), (b"target", 140),
                           (b"hold", 160), (b"targ_transition", 179), (b"reward", 180)])
        self.assertTrue(np.isnan(tr["duration"][0]))
        self.assertEqual(tr["row_first_target"][0], -1)
        self.assertEqual(tr["duration"][1], 55 / 60)

    def test_retry_uses_new_premove(self):
        tr = self.extract([(b"premove", 1), (b"timeout_penalty", 20),
                           (b"premove", 40), (b"target", 45), (b"hold", 50),
                           (b"targ_transition", 55), (b"target", 60), (b"hold", 80),
                           (b"targ_transition", 99), (b"reward", 100)])
        np.testing.assert_allclose(tr["duration"], [55 / 60])

    def test_single_target_and_interrupted_sequences_are_invalid(self):
        tr = self.extract([(b"target", 1), (b"hold", 5), (b"targ_transition", 10), (b"reward", 11),
                           (b"target", 20), (b"hold", 25), (b"targ_transition", 30),
                           (b"wait", 31), (b"target", 35), (b"hold", 40),
                           (b"targ_transition", 45), (b"reward", 46)])
        self.assertEqual(len(tr["duration"]), 2)
        self.assertTrue(np.all(np.isnan(tr["duration"])))

    def test_exact_sequence_does_not_require_premove(self):
        tr = self.extract([(b"target", 10), (b"hold", 20), (b"targ_transition", 30),
                           (b"target", 40), (b"hold", 60), (b"targ_transition", 69), (b"reward", 70)])
        np.testing.assert_allclose(tr["duration"], [1])

    def test_nonmonotonic_sequence_is_invalid(self):
        tr = self.extract([(b"target", 10), (b"hold", 20), (b"targ_transition", 15),
                           (b"target", 40), (b"hold", 60), (b"targ_transition", 69), (b"reward", 70)])
        self.assertTrue(np.isnan(tr["duration"][0]))

    def test_block_crossing_is_invalid_and_other_block_is_excluded(self):
        blocks = np.full(500, 2)
        blocks[80:] = 3
        tr = self.extract([(b"premove", 20), (b"target", 25), (b"hold", 30),
                           (b"targ_transition", 35), (b"target", 40), (b"hold", 90),
                           (b"targ_transition", 99), (b"reward", 100),
                           (b"wait", 110), (b"premove", 120), (b"target", 125),
                           (b"hold", 130), (b"targ_transition", 135), (b"target", 140),
                           (b"hold", 160), (b"targ_transition", 179), (b"reward", 180)], blocks)
        self.assertEqual(len(tr["duration"]), 1)
        self.assertTrue(np.isnan(tr["duration"][0]))

    def test_set_means_and_first_six_summary(self):
        d = summarize_sets(np.repeat(np.arange(1, 8), 8), 6)
        np.testing.assert_allclose(d["means"], np.arange(1, 8))
        self.assertEqual(d["early_mean"], 3.5)
        self.assertAlmostEqual(d["early_sem"], np.std(np.arange(1, 7), ddof=1) / np.sqrt(6))

    def test_incomplete_window_and_leftovers(self):
        d = summarize_sets(np.ones(47), 6)
        self.assertEqual(d["n_sets"], 5)
        self.assertEqual(d["leftover"], 7)
        self.assertTrue(np.isnan(d["early_mean"]))
        empty = summarize_sets([], 6)
        self.assertEqual(empty["n_sets"], 0)

    def test_invalid_trial_preserves_set_position(self):
        durations = np.ones(48)
        durations[2] = np.nan
        d = summarize_sets(durations, 6)
        self.assertTrue(np.isnan(d["means"][0]))
        np.testing.assert_allclose(d["means"][1:], 1)
        self.assertTrue(np.isnan(d["early_mean"]))

    def test_target_balance_preserves_set_positions(self):
        angles = np.tile(np.arange(8) * 45, 2).astype(float)
        angles[0] = angles[1]
        d = summarize_sets(np.ones(16), 2, angles)
        np.testing.assert_array_equal(d["balanced"], [False, True])
        self.assertTrue(np.isnan(d["means"][0]))
        self.assertEqual(d["means"][1], 1)
        self.assertTrue(np.isnan(d["early_mean"]))

    def test_pipeline_exports_short_sessions_and_both_spacings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for session, n_trials in [(1, 8), (2, 16)]:
                task = np.zeros(500, dtype=[("block_type", "i4"), ("target", "f8", (3,))])
                task["block_type"] = 2
                messages = []
                for j in range(n_trials):
                    angle = np.deg2rad((j % 8) * 45)
                    task["target"][j * 20:(j + 1) * 20] = [np.cos(angle), 0, np.sin(angle)]
                    messages.extend([(b"wait", j * 20), (b"premove", j * 20 + 1),
                                     (b"target", j * 20 + 2), (b"hold", j * 20 + 3),
                                     (b"targ_transition", j * 20 + 4),
                                     (b"target", j * 20 + 5), (b"hold", j * 20 + 15),
                                     (b"targ_transition", j * 20 + 18),
                                     (b"reward", j * 20 + 19)])
                msgs = np.array(messages, dtype=[("msg", "S32"), ("time", "i4")])
                with tables.open_file(root / f"airp20220101_{session:02d}_te{session}.hdf", "w") as hdf:
                    hdf.create_table("/", "task", obj=task)
                    hdf.create_table("/", "task_msgs", obj=msgs)
            for spacing in ["even", "calendar"]:
                out = root / spacing
                args = Namespace(hdf_dir=root, out_dir=out, monkeys=["airp", "braz"],
                                 blocks=[2, 3], first_sets=2, sensitivity_sets=3,
                                 late_start=25, min_late_sets=8, spacing=spacing, show=False)
                analyze_sessions(args)
                with (out / "session_summary.csv").open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 4)
                self.assertEqual(rows[0]["status"], "incomplete_or_invalid_early_window")
                self.assertEqual(rows[1]["status"], "no_rewards")
                self.assertEqual(rows[2]["status"], "ok")
                self.assertAlmostEqual(float(rows[2]["early_mean_s"]), 17 / 60)
                self.assertTrue(np.isnan(float(rows[2]["sensitivity_mean_s"])))
                self.assertTrue(np.isnan(float(rows[2]["early_late_ratio"])))
                self.assertTrue((out / f"block2_windows2-3_{spacing}.png").exists())
                self.assertTrue((out / f"block2_first2_{spacing}.png").exists())
                self.assertTrue((out / f"block3_first2_{spacing}.png").exists())
                with (out / "trial_times.csv").open(newline="") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 24)
                with (out / "set_means.csv").open(newline="") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 3)

    def test_late_reference_counts_valid_sets_and_keeps_fixed_sensitivity_window(self):
        means = np.ones(33)
        means[:3] = [3, 2, 1]
        means[24] = np.nan
        d = compare_windows(means)
        self.assertEqual(d["sensitivity_mean"], 2)
        self.assertEqual(d["n_late_sets"], 8)
        self.assertEqual(d["late_mean"], 1)
        self.assertEqual(d["late_sem"], 0)
        means[25] = np.nan
        self.assertTrue(np.isnan(compare_windows(means)["late_mean"]))
        means[1] = np.nan
        self.assertTrue(np.isnan(compare_windows(means)["sensitivity_mean"]))


if __name__ == "__main__":
    unittest.main()
