"""Fit recorded-session and calendar trends from analyzeHDF session_summary.csv.

Unweighted fits give each session equal influence. Constant, linear and
decreasing exponential fits are compared by holding out five contiguous groups
of sessions. Linear intervals use circular blocks of adjacent observed residuals
(lengths 4 and 6), with dates held fixed. These exploratory intervals assume
approximately stationary residual dependence in recorded-session order; they
are not simultaneous intervals across all metrics. No causal inference or
within-session learning-rate inference is made.
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import theilslopes


ROOT = Path(__file__).resolve().parents[1]
METRICS = [("early_mean_s", "Early: first 2 sets", "s"),
           ("sensitivity_mean_s", "Early: first 3 sets", "s"),
           ("late_mean_s", "Late: sets 25+", "s"),
           ("early_late_ratio", "First 2 / late", "ratio"),
           ("sensitivity_late_ratio", "First 3 / late", "ratio")]


def number(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return np.nan


def fit_models(x, y):
    """x is scaled to the observed span; exponential tau is in span units."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    linear = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    upper = max(10 * np.max(y), 1.0)
    best = None
    for tau in [0.05, 0.2, 1.0]:
        a = max(float(np.median(y[-max(3, len(y) // 4):])), 1e-6)
        b = max(float(np.mean(y[:3]) - a), 0.01)
        fit = least_squares(lambda p: p[0] + p[1] * np.exp(-x / p[2]) - y,
                            [a, min(b, upper), tau], bounds=([0, 0, 0.01], [upper, upper, 4]),
                            max_nfev=1000)
        if best is None or fit.cost < best.cost:
            best = fit
    return {"constant": np.array([np.mean(y)]), "linear": linear,
            "exponential": best.x}


def predict(model, p, x):
    if model == "constant":
        return np.full(len(x), p[0])
    if model == "linear":
        return p[0] + p[1] * x
    return p[0] + p[1] * np.exp(-np.asarray(x) / p[2])


def blocked_cv(x, y, folds=5):
    errors = {name: [] for name in ["constant", "linear", "exponential"]}
    for test in np.array_split(np.arange(len(x)), min(folds, len(x) // 3)):
        train = np.setdiff1d(np.arange(len(x)), test)
        fits = fit_models(x[train], y[train])
        for name, p in fits.items():
            errors[name].extend((predict(name, p, x[test]) - y[test]) ** 2)
    return {name: float(np.sqrt(np.mean(e))) for name, e in errors.items()}


def block_linear_bootstrap(x, y, block_length, rng, draws=10000):
    """Resample residual blocks; retain the actual x positions, including gaps."""
    design = np.column_stack([np.ones(len(x)), x])
    inverse = np.linalg.pinv(design)
    beta = inverse @ y
    residual = y - design @ beta
    residual -= residual.mean()
    residual *= np.sqrt(len(x) / (len(x) - 2))
    starts = rng.integers(0, len(x), size=(draws, int(np.ceil(len(x) / block_length))))
    ix = (starts[:, :, None] + np.arange(block_length)) % len(x)
    ix = ix.reshape(draws, -1)[:, :len(x)]
    return (inverse @ ((design @ beta)[None, :] + residual[ix]).T).T


def prepare_rows(rows, monkey, block, metric, spacing):
    # Assign positions before dropping missing values so sessions do not shift.
    files = sorted({r["file"] for r in rows if r["monkey"] == monkey},
                   key=lambda f: (datetime.strptime(f[4:12], "%Y%m%d"),
                                  int(f.split("_")[1]), f))
    positions = {f: k for k, f in enumerate(files)}
    origin = datetime.strptime(files[0][4:12], "%Y%m%d")
    subset = sorted([r for r in rows if r["monkey"] == monkey and int(r["block"]) == block],
                    key=lambda r: positions[r["file"]])
    x, y = [], []
    for r in subset:
        value = number(r, metric)
        # Direct sensitivity comparisons use the same early-window cohort.
        if metric in ["early_mean_s", "sensitivity_mean_s"]:
            eligible = np.all(np.isfinite([number(r, "early_mean_s"), number(r, "sensitivity_mean_s")]))
        elif metric.endswith("ratio"):
            eligible = np.all(np.isfinite([number(r, "early_late_ratio"), number(r, "sensitivity_late_ratio")]))
        else:
            eligible = True
        if eligible and np.isfinite(value):
            date = datetime.strptime(r["date"], "%Y-%m-%d")
            x.append(positions[r["file"]] if spacing == "even" else (date - origin).days)
            y.append(value)
    return np.asarray(x, float), np.asarray(y, float), origin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "plots/premove_session_sets/session_summary.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "plots/session_trends")
    args = parser.parse_args()
    with args.summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if any(int(r["requested_sets"]) != 2 or int(r["sensitivity_sets"]) != 3
           or int(r["late_start_set"]) != 25 for r in rows if r.get("early_mean_s")):
        raise ValueError("This analysis expects primary=2, sensitivity=3, late start=25")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261013)
    summaries, predictions = [], []
    for block in [2, 3]:
        for spacing in ["even", "calendar"]:
            fig, axes = plt.subplots(2, 5, figsize=(23, 9), squeeze=False)
            for row, monkey in enumerate(["airp", "braz"]):
                for col, (metric, label, units) in enumerate(METRICS):
                    ax = axes[row, col]
                    x, y, origin = prepare_rows(rows, monkey, block, metric, spacing)
                    if len(x) < 10 or np.ptp(x) == 0:
                        ax.set_title(f"{monkey}: insufficient sessions")
                        continue
                    x0, span = x.min(), np.ptp(x)
                    scaled = (x - x0) / span
                    fits = fit_models(scaled, y)
                    cv = blocked_cv(scaled, y)
                    boots = block_linear_bootstrap(scaled, y, 4, rng)
                    sensitivity = block_linear_bootstrap(scaled, y, 6, rng)
                    lo, hi = np.percentile(boots[:, 1], [2.5, 97.5]) / span
                    lo6, hi6 = np.percentile(sensitivity[:, 1], [2.5, 97.5]) / span
                    slope = fits["linear"][1] / span
                    robust = theilslopes(y, x)[0]
                    # Diagnostic only: retain all sessions in the main fits.
                    after = np.arange(len(x)) >= 4
                    later_scaled = (x[after] - x0) / span
                    later_fits = fit_models(later_scaled, y[after])
                    later_boot = block_linear_bootstrap(later_scaled, y[after], 4, rng)
                    later_lo, later_hi = np.percentile(later_boot[:, 1], [2.5, 97.5]) / span
                    sst = np.sum((y - y.mean()) ** 2)
                    r2 = 1 - np.sum((predict("linear", fits["linear"], scaled) - y) ** 2) / sst if sst > 0 else np.nan
                    p = fits["exponential"]
                    summaries.append(dict(monkey=monkey, block=block, spacing=spacing, metric=metric,
                                          n_sessions=len(y), x_units="session" if spacing == "even" else "day",
                                          slope=slope, slope_ci_lo=lo, slope_ci_hi=hi,
                                          slope_ci_lo_block6=lo6, slope_ci_hi_block6=hi6,
                                          theil_sen_slope=robust, fitted_change=fits["linear"][1],
                                          slope_excluding_first4=later_fits["linear"][1] / span,
                                          slope_excluding_first4_ci_lo=later_lo,
                                          slope_excluding_first4_ci_hi=later_hi,
                                          fitted_percent_change=100 * fits["linear"][1] / fits["linear"][0],
                                          linear_r2=r2, constant_cv_rmse=cv["constant"],
                                          linear_cv_rmse=cv["linear"], exponential_cv_rmse=cv["exponential"],
                                          best_cv_model=min(cv, key=cv.get), exponential_a=p[0], exponential_b=p[1],
                                          exponential_tau=p[2] * span,
                                          exponential_tau_at_bound=bool(p[2] < 0.011 or p[2] > 3.99)))
                    grid = np.linspace(x0, x.max(), 200)
                    g = (grid - x0) / span
                    plotted_x = x + 1 if spacing == "even" else mdates.date2num(origin) + x
                    plotted_grid = grid + 1 if spacing == "even" else mdates.date2num(origin) + grid
                    ax.plot(plotted_x, y, "o", ms=3, alpha=0.7, color="C0", label="Sessions")
                    for model, color, style in [("constant", "gray", ":"), ("linear", "C1", "-"),
                                                 ("exponential", "C2", "--")]:
                        line = predict(model, fits[model], g)
                        ax.plot(plotted_grid, line, style, color=color, lw=1.5, label=model)
                    bands = boots[:, 0, None] + boots[:, 1, None] * g
                    ax.fill_between(plotted_grid, *np.percentile(bands, [2.5, 97.5], axis=0), color="C1", alpha=0.15)
                    if units == "ratio":
                        ax.axhline(1, color="k", lw=0.6)
                    if spacing == "calendar":
                        loc = mdates.AutoDateLocator(minticks=3, maxticks=5)
                        ax.xaxis.set_major_locator(loc)
                        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
                    ax.set(title=f"{monkey}: {label}, n={len(y)}\nlinear change={fits['linear'][1]:+.3f}; CV best={min(cv, key=cv.get)}",
                           xlabel="Recorded session" if spacing == "even" else "Session date",
                           ylabel="Duration (s)" if units == "s" else "Early / late")
                    ax.grid(alpha=0.25)
                    if row == 0 and col == 0:
                        ax.legend(fontsize=7)
                    for j, gx in enumerate(grid):
                        predictions.append(dict(monkey=monkey, block=block, spacing=spacing, metric=metric,
                                                x=gx, linear=predict("linear", fits["linear"], g)[j],
                                                exponential=predict("exponential", p, g)[j]))
            fig.suptitle(f"Block {block}: {spacing} spacing; shaded = pointwise 95% linear block-bootstrap interval", fontsize=12)
            fig.tight_layout()
            fig.savefig(args.out_dir / f"block{block}_fits_{spacing}.png", dpi=180)
            plt.close(fig)
    for name, data in [("trend_summary.csv", summaries), ("fitted_curves.csv", predictions)]:
        with (args.out_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    for s in summaries:
        if s["block"] == 2:
            print(f"{s['monkey']} {s['spacing']} {s['metric']}: slope={s['slope']:+.5f} "
                  f"CI=[{s['slope_ci_lo']:+.5f},{s['slope_ci_hi']:+.5f}], "
                  f"change={s['fitted_percent_change']:+.1f}%, R2={s['linear_r2']:.2f}; CV={s['best_cv_model']}")
    print("Saved to", args.out_dir)


if __name__ == "__main__":
    main()
