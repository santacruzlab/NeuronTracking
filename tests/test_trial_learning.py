import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from trial_learning import extract_attempts, trajectory_metrics, assign_phases, gls_projector, half_life
from simulate_learning import sample_coefficients, diagnostics


class TrialLearningTests(unittest.TestCase):
    def test_success_failures_and_censoring_are_separate(self):
        task = np.zeros(110, dtype=[("block_type", "i4", (1,)), ("cursor", "f8", (3,)),
                                   ("target", "f8", (3,)), ("error_clamp", "i4", (1,)), ("pert", "i4", (1,))])
        task["block_type"] = 2
        task["target"][4:25, 0] = 10
        task["target"][34:60, 0] = 10
        task["target"][94:, 0] = 10
        msgs = np.array([(b"premove", 1), (b"target", 2), (b"hold", 3), (b"targ_transition", 3), (b"target", 4),
                         (b"hold", 10), (b"targ_transition", 12), (b"reward", 13),
                         (b"premove", 30), (b"target", 31), (b"target", 34), (b"timeout_penalty", 54),
                         (b"premove", 70), (b"target", 71), (b"hold_penalty", 73),
                         (b"premove", 90), (b"target", 94), (b"None", 100)],
                        dtype=[("msg", "S32"), ("time", "i4")])
        attempts = extract_attempts(task, msgs)
        self.assertEqual([r["outcome"] for r in attempts], ["success", "failure", "failure", "censored"])
        self.assertAlmostEqual(attempts[0]["duration_s"], 11 / 60)
        self.assertEqual(attempts[0]["row_first_target"], 2)
        self.assertTrue(attempts[0]["sequence_valid"])
        self.assertAlmostEqual(attempts[0]["peripheral_reach_s"], 6 / 60)
        self.assertTrue(np.isnan(attempts[1]["peripheral_reach_s"]))
        self.assertFalse(attempts[2]["peripheral"])
        self.assertFalse(attempts[3]["valid_timing"])

    def test_reward_requires_exact_two_target_sequence(self):
        task = np.zeros(30, dtype=[("block_type", "i4", (1,)), ("cursor", "f8", (3,)),
                                  ("target", "f8", (3,)), ("error_clamp", "i4", (1,)), ("pert", "i4", (1,))])
        task["block_type"] = 2
        task["target"][5:, 0] = 10
        missing = [(b"premove", 1), (b"target", 2), (b"hold", 3),
                   (b"target", 5), (b"hold", 10), (b"targ_transition", 11), (b"reward", 12)]
        dtype = [("msg", "S32"), ("time", "i4")]
        bad = extract_attempts(task, np.array(missing, dtype=dtype))[0]
        self.assertFalse(bad["sequence_valid"])
        self.assertTrue(np.isnan(bad["duration_s"]))
        complete = missing[1:3] + [(b"targ_transition", 4)] + missing[3:]
        good = extract_attempts(task, np.array(complete, dtype=dtype))[0]
        self.assertTrue(good["sequence_valid"])
        self.assertAlmostEqual(good["duration_s"], 10 / 60)

    def test_valid_sequence_with_stale_target_position_keeps_timing(self):
        task = np.zeros(30, dtype=[("block_type", "i4", (1,)), ("cursor", "f8", (3,)),
                                  ("target", "f8", (3,)), ("error_clamp", "i4", (1,)), ("pert", "i4", (1,))])
        task["block_type"] = 2
        msgs = np.array([(b"premove", 1), (b"target", 2), (b"hold", 3),
                         (b"targ_transition", 4), (b"target", 4), (b"hold", 10),
                         (b"targ_transition", 11), (b"reward", 12)],
                        dtype=[("msg", "S32"), ("time", "i4")])
        trial = extract_attempts(task, msgs)[0]
        self.assertTrue(trial["valid_timing"])
        self.assertTrue(trial["peripheral"])
        self.assertFalse(trial["target_position_valid"])
        self.assertTrue(np.isnan(trial["target_deg"]))
        self.assertAlmostEqual(trial["duration_s"], 10 / 60)

    def test_trajectory_geometry(self):
        straight = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        r = trajectory_metrics(straight, [2, 0, 0], 0, 2)
        self.assertEqual(r["straightness"], 1)
        self.assertEqual(r["initial_error_deg"], 0)
        bent = np.array([[0, 0, 0], [1, 0, 1], [2, 0, 0]])
        r = trajectory_metrics(bent, [2, 0, 0], 0, 2)
        self.assertAlmostEqual(r["straightness"], 1 / np.sqrt(2))
        self.assertAlmostEqual(r["initial_error_deg"], 45)

    def test_phase_exposure_retains_disputed_session_position(self):
        rows = pd.DataFrame([dict(file=f"braz2022010{i}_01_te{i}.hdf", monkey="braz", date=f"2022-01-0{i}",
                                  catalog_rotation_signed_deg=a, rotation_signed_deg=a,
                                  rotation_matches_catalog=i != 1) for i, a in [(1, -50), (2, -50), (3, 90)]])
        phases = assign_phases(rows)
        np.testing.assert_array_equal(phases.exposure_number, [1, 2, 1])
        np.testing.assert_array_equal(phases.exposure, [0, 1, 0])
        self.assertTrue(phases.iloc[0].disputed)

    def test_gls_projector_matches_direct_covariance(self):
        x = np.arange(12, dtype=float)
        X = np.column_stack([np.ones(12), x])
        Z = np.column_stack([np.ones(12), np.exp(-x / 8)])
        groups = np.repeat(np.arange(3), 4)
        D = np.diag([0.2, 0.1])
        projection, covariance = gls_projector(X, Z, groups, D, 0.5)
        V = np.eye(12) * 0.5
        for g in range(3):
            ix = np.flatnonzero(groups == g)
            V[np.ix_(ix, ix)] += Z[ix] @ D @ Z[ix].T
        vi = np.linalg.inv(V)
        cov = np.linalg.inv(X.T @ vi @ X)
        np.testing.assert_allclose(covariance, cov, atol=1e-10)
        np.testing.assert_allclose(projection, cov @ X.T @ vi, atol=1e-10)
        self.assertAlmostEqual(half_life(1, 0), 8 * np.log(2))
        self.assertTrue(np.isnan(half_life(-1, 1)))

    def test_simulation_distinguishes_starting_cost_from_rate(self):
        t = np.tile(np.arange(64), 6)
        groups = np.repeat(np.arange(6), 64)
        exposure = np.repeat(np.linspace(0, 1, 6), 64)
        fast, slow = np.exp(-t / 8), np.exp(-t / 32)
        X = np.column_stack([np.ones(len(t)), exposure, fast, slow, exposure * fast, exposure * slow])
        Z = np.column_stack([np.ones(len(t)), fast])
        projector, covariance = gls_projector(X, Z, groups, np.zeros((2, 2)), 0.05)
        ctx = dict(X=X, Z=Z, groups=groups, exposure=exposure, t=t, beta=np.array([0, 0, .2, .1, 0, 0]),
                   names=np.array(["Intercept", "exposure", "fast", "slow", "exposure:fast", "exposure:slow"]),
                   random_cov=np.zeros((2, 2)), residual=np.zeros(len(t)),
                   projector=projector, fixed_cov=covariance)
        rng = np.random.default_rng(7)
        null = diagnostics(sample_coefficients(ctx, 0, 0, 0, 3, rng), ctx)
        self.assertLess(null["joint"].max(), 1e-10)
        cost = diagnostics(sample_coefficients(ctx, .5, 0, 0, 3, rng), ctx)
        self.assertTrue(np.all(cost["gap_change"] < -0.1))
        np.testing.assert_allclose(cost["rate_z"], 0, atol=1e-8)
        rate = diagnostics(sample_coefficients(ctx, 0, .25, 0, 3, rng), ctx)
        self.assertTrue(np.all(rate["rate_z"] > 0))


if __name__ == "__main__":
    unittest.main()
