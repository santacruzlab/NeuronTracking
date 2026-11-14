"""Best exponential decay fitted separately inside each perturbation (rotation) run.

For every monkey and window (first 8/16/24 trials) the sessions are split into runs of
consecutive sessions with the same recorded rotation. Inside each run, with d0 = day of
the run's first session, we fit

    y = c + A * exp(-(d - d0) / tau)        (weighted by 1/normalized_sem)

by a multi-start search over tau (log-spaced 0.5-300 days, each start refined by
bounded least squares) and keep the lowest-chi2 solution. Constant and straight-line
fits are made for the same run and compared with AIC; the exponential is only called
supported when it beats both. Disputed-rotation sessions are excluded from the fits.
Outputs: plots/baseline_exp_per_rotation/{exp_per_rotation.png, exp_per_rotation.csv}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "plots/baseline_diagnostics/session_diagnostics.csv"
OUT = ROOT / "plots/baseline_exp_per_rotation"
COLORS = {-90: "C0", -50: "C1", 50: "C2", 90: "C3"}
MIN_SESSIONS = 6


def chi2_of(y, pred, s):
    return float(np.sum(((y - pred) / s) ** 2))


def fit_exp(x, y, s):
    """Multi-start weighted fit of c + A*exp(-x/tau); x measured from run start."""
    best = None
    span = max(x.max(), 1.0)
    for tau0 in np.geomspace(0.5, 300, 25):
        for sign in (1, -1):
            c0, a0 = float(np.median(y[-3:])), sign * float(np.ptp(y))
            res = least_squares(lambda p: (p[0] + p[1] * np.exp(-x / p[2]) - y) / s,
                                [c0, a0, tau0], bounds=([0, -5, 0.5], [5, 5, 300]))
            if best is None or res.cost < best.cost:
                best = res
    return best.x, 2 * best.cost


def fit_line(x, y, s):
    w = 1 / s
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
    return beta, chi2_of(y, X @ beta, s)


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(SRC)
    df = df[np.isfinite(df.normalized_mean) & np.isfinite(df.normalized_sem) & (df.normalized_sem > 0)]
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    for i, monkey in enumerate(["airp", "braz"]):
        for j, win in enumerate([8, 16, 24]):
            ax = axes[i, j]
            g = df[(df.monkey == monkey) & (df.window_trials == win)].sort_values("days")
            ok = g[~g.rotation_disputed.astype(bool)].copy()
            ok["run"] = (ok.rotation != ok.rotation.shift()).cumsum()
            for rot, sub in g.groupby("rotation"):
                ax.errorbar(sub.days, sub.normalized_mean, sub.normalized_sem, fmt="o", ms=3, capsize=2,
                            color=COLORS.get(int(rot), "k"), alpha=0.55, label=f"{rot:+g} deg")
            for run, sub in ok.groupby("run"):
                rot = int(sub.rotation.iloc[0])
                if len(sub) < MIN_SESSIONS:
                    continue
                d0 = sub.days.min()
                x = (sub.days - d0).to_numpy(float)
                y, s = sub.normalized_mean.to_numpy(), sub.normalized_sem.to_numpy()
                (c, a, tau), chi_e = fit_exp(x, y, s)
                const = np.sum(y / s ** 2) / np.sum(1 / s ** 2)
                chi_c = chi2_of(y, const, s)
                (b0, b1), chi_l = fit_line(x, y, s)
                aic = dict(const=chi_c + 2, line=chi_l + 4, exp=chi_e + 6)
                best = min(aic, key=aic.get)
                grid = np.linspace(0, x.max(), 200)
                # Both fits are drawn for every run, whichever wins by AIC.
                ax.plot(d0 + grid, c + a * np.exp(-grid / tau), color=COLORS[rot], lw=3, ls="-")
                ax.plot(d0 + grid, b0 + b1 * grid, color=COLORS[rot], lw=2.2, ls=(0, (4, 2)),
                        path_effects=[pe.withStroke(linewidth=3.6, foreground="black", alpha=0.5)])
                rows.append(dict(monkey=monkey, window=win, rotation=rot, n_sessions=len(sub),
                                 first_day=d0, last_day=sub.days.max(), asymptote_c=c, amplitude_A=a,
                                 tau_days=tau, start_value=c + a, half_life_days=tau * np.log(2),
                                 chi2_exp=chi_e, chi2_line=chi_l, chi2_const=chi_c,
                                 aic_exp=aic["exp"], aic_line=aic["line"], aic_const=aic["const"],
                                 best_by_aic=best, tau_at_bound=bool(tau < 0.51 or tau > 299)))
            ax.axhline(1, color="gray", ls=":", lw=1)
            ax.set(title=f"{monkey}: first {win} trials", xlabel="Days since first recorded session")
            if j == 0:
                ax.set_ylabel("Mean duration / block 1 mean")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Fits per rotation run: solid = best exponential decay, dashed (dark outline) = straight line; "
                 "AIC winner per run is in exp_per_rotation.csv")
    fig.tight_layout()
    fig.savefig(OUT / "exp_per_rotation.png", dpi=150)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "exp_per_rotation.csv", index=False)
    cols = ["monkey", "window", "rotation", "n_sessions", "start_value", "asymptote_c", "tau_days",
            "half_life_days", "best_by_aic", "tau_at_bound"]
    print(res[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
