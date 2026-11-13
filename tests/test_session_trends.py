"""Check calendar gaps, fit recovery and residual-block inference."""
import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fit_session_trends import fit_models, predict, blocked_cv, block_linear_bootstrap, prepare_rows


class TrendTests(unittest.TestCase):
    def test_exponential_recovers_known_curve(self):
        x = np.linspace(0, 1, 30)
        y = 1.2 + 0.8 * np.exp(-x / 0.2)
        fitted = fit_models(x, y)
        np.testing.assert_allclose(fitted["exponential"], [1.2, 0.8, 0.2], rtol=1e-4)
        np.testing.assert_allclose(predict("exponential", fitted["exponential"], x), y, atol=1e-5)

    def test_block_bootstrap_preserves_exact_linear_slope(self):
        x = np.linspace(0, 1, 20)
        y = 3 - 0.5 * x
        for length in [4, 6]:
            draws = block_linear_bootstrap(x, y, length, np.random.default_rng(7), draws=50)
            np.testing.assert_allclose(draws[:, 0], 3, atol=1e-12)
            np.testing.assert_allclose(draws[:, 1], -0.5, atol=1e-12)
        cv = blocked_cv(x, y)
        self.assertLess(cv["linear"], 1e-10)
        self.assertGreater(cv["constant"], 0.1)

    def test_missing_session_positions_and_same_day_sessions(self):
        rows = []
        for date, session, value in [("2022-01-01", 1, "2"), ("2022-01-01", 2, "nan"),
                                     ("2022-01-11", 1, "1")]:
            rows.append(dict(monkey="airp", block="2", file=f"airp{date.replace('-', '')}_{session:02d}_te1.hdf",
                             date=date, early_mean_s=value, sensitivity_mean_s=value))
        x, y, origin = prepare_rows(rows, "airp", 2, "early_mean_s", "even")
        np.testing.assert_array_equal(x, [0, 2])
        np.testing.assert_array_equal(y, [2, 1])
        x, _, _ = prepare_rows(rows, "airp", 2, "early_mean_s", "calendar")
        np.testing.assert_array_equal(x, [0, 10])
        rows[1]["early_mean_s"] = rows[1]["sensitivity_mean_s"] = "2"
        x, _, _ = prepare_rows(rows, "airp", 2, "early_mean_s", "calendar")
        np.testing.assert_array_equal(x, [0, 0, 10])


if __name__ == "__main__":
    unittest.main()
