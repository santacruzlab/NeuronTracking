"""Fit curves to the per-session normalized early-trial means (baseline_diagnostics plots).

Reads plots/baseline_diagnostics/session_diagnostics.csv (one row per session and
window) and, for each monkey and window (first 8/16/24 trials), fits to
normalized_mean vs days, weighted by 1/normalized_sem:
  linear        y = a + b*d
  exponential   y = c + A*exp(-d/tau)     (decay toward an asymptote)
  per-rotation  separate line inside each run of consecutive same-rotation sessions
Disputed-rotation sessions are left out of the fits (still drawn as black crosses).
Outputs go to plots/baseline_trend_fits/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "plots/baseline_diagnostics/session_diagnostics.csv"
OUT = ROOT / "plots/baseline_trend_fits"
COLORS = {-90: "C0", -50: "C1", 50: "C2", 90: "C3"}


def wls_line(d, y, sig):
    w = 1 / sig
    X = np.column_stack([np.ones_like(d), d])
    beta = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
    return beta


def aic(y, pred, sig, k):
    chi2 = np.sum(((y - pred) / sig) ** 2)
    return chi2 + 2 * k, chi2


def exp_fit(d, y, sig):
    f = lambda x, c, a, tau: c + a * np.exp(-x / tau)
    best = None
    for tau0 in (10, 30, 80):
        try:
            p, _ = curve_fit(f, d, y, p0=[y.min(), y.max() - y.min(), tau0], sigma=sig,
                             bounds=([0, -5, 1], [5, 10, 500]), maxfev=20000)
        except Exception:
            continue
        chi = np.sum(((y - f(d, *p)) / sig) ** 2)
        if best is None or chi < best[1]:
            best = (p, chi)
    return (f, best[0]) if best else (f, None)


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(SRC)
    df["rotation"] = df.rotation.astype(float)
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    for i, monkey in enumerate(["airp", "braz"]):
        for j, win in enumerate([8, 16, 24]):
            ax = axes[i, j]
            g = df[(df.monkey == monkey) & (df.window_trials == win)].sort_values("days")
            g = g[np.isfinite(g.normalized_mean) & np.isfinite(g.normalized_sem) & (g.normalized_sem > 0)]
            fit = g[~g.rotation_disputed.astype(bool)]
            d, y, s = fit.days.to_numpy(float), fit.normalized_mean.to_numpy(), fit.normalized_sem.to_numpy()
            for rot, sub in g.groupby("rotation"):
                ax.errorbar(sub.days, sub.normalized_mean, sub.normalized_sem, fmt="o", ms=3, capsize=2,
                            color=COLORS.get(int(rot), "k"), alpha=0.7,
                            label=f"{rot:+g} deg")
            dis = g[g.rotation_disputed.astype(bool)]
            ax.scatter(dis.days, dis.normalized_mean, marker="x", color="k", s=60, zorder=5)
            grid = np.linspace(d.min(), d.max(), 300)
            a, b = wls_line(d, y, s)
            ax.plot(grid, a + b * grid, "k--", lw=1.5, label="linear")
            lin_aic, lin_chi = aic(y, a + b * d, s, 2)
            f, p = exp_fit(d, y, s)
            exp_aic = np.nan
            if p is not None:
                ax.plot(grid, f(grid, *p), "k-", lw=2, label="exp. decay")
                exp_aic, _ = aic(y, f(d, *p), s, 3)
            # per-rotation-run lines
            fit = fit.assign(run=(fit.rotation != fit.rotation.shift()).cumsum())
            pr_chi, pr_k = 0.0, 0
            for run, sub in fit.groupby("run"):
                if len(sub) < 3:
                    continue
                dd, yy, ss = sub.days.to_numpy(float), sub.normalized_mean.to_numpy(), sub.normalized_sem.to_numpy()
                aa, bb = wls_line(dd, yy, ss)
                xs = np.array([dd.min(), dd.max()])
                ax.plot(xs, aa + bb * xs, color=COLORS.get(int(sub.rotation.iloc[0]), "k"), lw=3)
                pr_chi += np.sum(((yy - (aa + bb * dd)) / ss) ** 2)
                pr_k += 2
                rows.append(dict(monkey=monkey, window=win, model="per-rotation line", rotation=sub.rotation.iloc[0],
                                 n_sessions=len(sub), intercept=aa, slope_per_day=bb))
            rows.append(dict(monkey=monkey, window=win, model="linear", n_sessions=len(d), intercept=a,
                             slope_per_day=b, chi2=lin_chi, aic=lin_aic))
            rows.append(dict(monkey=monkey, window=win, model="exp decay", n_sessions=len(d),
                             asymptote=p[0] if p is not None else np.nan, amplitude=p[1] if p is not None else np.nan,
                             tau_days=p[2] if p is not None else np.nan, aic=exp_aic))
            rows.append(dict(monkey=monkey, window=win, model="per-rotation lines (all runs)",
                             n_sessions=len(d), chi2=pr_chi, aic=pr_chi + 2 * pr_k))
            ax.axhline(1, color="gray", ls=":", lw=1)
            ax.set(title=f"{monkey}: first {win} trials (n={len(g)})", xlabel="Days since first recorded session")
            if j == 0:
                ax.set_ylabel("Mean duration / block 1 mean")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Normalized early-trial means with fits: dashed = linear, black = exponential decay, "
                 "thick colored = line per rotation run (AIC in baseline_trend_fits.csv)")
    fig.tight_layout()
    fig.savefig(OUT / "baseline_trend_fits.png", dpi=150)
    pd.DataFrame(rows).to_csv(OUT / "baseline_trend_fits.csv", index=False)
    summary = pd.DataFrame(rows)
    print(summary[summary.model != "per-rotation line"].round(3).to_string())


if __name__ == "__main__":
    main()
