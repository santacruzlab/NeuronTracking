"""Conditional recoverability checks for the trial-level curve model.

Keep actual trial positions, targets, phase/exposure counts, gaps and missing
completed reaches. Resample real conditional residuals in eight-trial blocks
and draw session random effects, independently or with lag-one correlation 0.5.
GLS holds fitted covariance parameters fixed: this is a conditional diagnostic,
not power for the earlier accepted-tau pipeline or a full refitted MixedLM.
An independent null set checks empirically calibrated detection thresholds.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.proportion import proportion_confint

from trial_learning import half_life, OUT

SCENARIOS = [("null", 0, 0), ("cost25", 0.25, 0), ("cost50", 0.5, 0),
             ("speed25", 0, 0.25), ("speed50", 0, 0.5), ("both50", 0.5, 0.5)]


def sample_coefficients(ctx, cost_change, speed_change, rho, draws, rng):
    X, Z, groups = ctx["X"], ctx["Z"], ctx["groups"]
    beta = ctx["beta"].copy()
    names = list(ctx["names"])
    for name in ["fast", "slow", "exposure", "exposure:fast", "exposure:slow"]:
        beta[names.index(name)] = 0
    base = X @ beta
    exposure, t = ctx["exposure"], ctx["t"]
    # Use each animal's fitted starting excess; tau starts at 16 trials.
    starting_gap = max(float(ctx["beta"][names.index("fast")] + ctx["beta"][names.index("slow")]), 0.10)
    tau = ctx.get("tau_trials", 16)
    true_curve = starting_gap * (1 - cost_change * exposure) * np.exp(-t / (tau * (1 - speed_change * exposure)))
    samples = np.tile((base + true_curve)[:, None], (1, draws))
    eig, vec = np.linalg.eigh(ctx["random_cov"])
    root = vec @ np.diag(np.sqrt(np.maximum(eig, 0)))
    previous = np.zeros((draws, root.shape[0]))
    for j, group in enumerate(np.unique(groups)):
        ix = np.flatnonzero(groups == group)
        residual = ctx["residual"][ix].copy()
        residual -= residual.mean()
        starts = rng.integers(0, len(ix), size=(draws, int(np.ceil(len(ix) / 8))))
        indices = ((starts[:, :, None] + np.arange(8)) % len(ix)).reshape(draws, -1)[:, :len(ix)]
        innovation = rng.normal(size=(draws, root.shape[0])) @ root.T
        effects = innovation if j == 0 else rho * previous + np.sqrt(1 - rho ** 2) * innovation
        samples[ix] += residual[indices].T + Z[ix] @ effects.T
        previous = effects
    return (ctx["projector"] @ samples).T


def diagnostics(coefficients, ctx):
    names = list(ctx["names"])
    f, s, ef, es = [names.index(k) for k in ["fast", "slow", "exposure:fast", "exposure:slow"]]
    covariance = ctx["fixed_cov"]
    change = coefficients[:, [ef, es]]
    ci = covariance[np.ix_([ef, es], [ef, es])]
    joint = np.einsum("bi,ij,bj->b", change, np.linalg.pinv(ci), change)
    amplitude = change.sum(axis=1)
    amplitude_se = np.sqrt(ci.sum())
    a0 = coefficients[:, f] + coefficients[:, s]
    a1 = a0 + amplitude
    fraction = coefficients[:, f] / a0
    last_fraction = (coefficients[:, f] + coefficients[:, ef]) / a1
    gradient = np.zeros_like(coefficients)
    gradient[:, f] = (coefficients[:, s] + coefficients[:, es]) / a1 ** 2 - coefficients[:, s] / a0 ** 2
    gradient[:, s] = -(coefficients[:, f] + coefficients[:, ef]) / a1 ** 2 + coefficients[:, f] / a0 ** 2
    gradient[:, ef] = (coefficients[:, s] + coefficients[:, es]) / a1 ** 2
    gradient[:, es] = -(coefficients[:, f] + coefficients[:, ef]) / a1 ** 2
    se = np.sqrt(np.maximum(np.einsum("bi,ij,bj->b", gradient, covariance, gradient), 1e-15))
    valid = (a0 > 0.05) & (a1 > 0.05) & (fraction >= 0) & (fraction <= 1) & (last_fraction >= 0) & (last_fraction <= 1)
    rate_z = np.where(valid, (last_fraction - fraction) / se, np.nan)
    return dict(joint=joint, gap_z=amplitude / amplitude_se, rate_z=rate_z,
                gap_change=amplitude, rate_valid=valid,
                half_first=np.array([half_life(b[f], b[s]) for b in coefficients]),
                half_last=np.array([half_life(b[f] + b[ef], b[s] + b[es]) for b in coefficients]))


def sample_batched(ctx, cost, speed, rho, draws, rng):
    return np.concatenate([sample_coefficients(ctx, cost, speed, rho, min(1000, draws - start), rng)
                           for start in range(0, draws, 1000)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--tau-trials", type=float, default=16)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.draws < 100 or args.tau_trials <= 0:
        parser.error("Use at least 100 Monte Carlo datasets and a positive starting tau")
    suffix = "" if args.tau_trials == 16 else f"_tau{args.tau_trials:g}"
    rng = np.random.default_rng(20261013)
    rows, calibration = [], []
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
    for i, monkey in enumerate(["airp", "braz"]):
        with np.load(args.out_dir / f"simulation_context_{monkey}.npz") as context:
            ctx = {key: context[key] for key in context.files}
        ctx["tau_trials"] = args.tau_trials
        for j, rho in enumerate([0, 0.5]):
            null = diagnostics(sample_batched(ctx, 0, 0, rho, 10000, rng), ctx)
            threshold = dict(joint=float(np.percentile(null["joint"], 95)),
                             gap_z=float(np.percentile(null["gap_z"], 5)),
                             rate_z=float(np.nanpercentile(null["rate_z"], 95)))
            calibration.append(dict(monkey=monkey, session_correlation=rho, **threshold))
            plot_rows = []
            for scenario, cost, speed in SCENARIOS:
                datasets = 5000 if scenario == "null" else args.draws
                d = diagnostics(sample_batched(ctx, cost, speed, rho, datasets, rng), ctx)
                detected = dict(curve=d["joint"] > threshold["joint"],
                                initial_cost=d["gap_z"] < threshold["gap_z"],
                                shape=d["rate_z"] > threshold["rate_z"])
                for measure, success in detected.items():
                    lo, hi = proportion_confint(int(success.sum()), datasets, method="wilson")
                    rows.append(dict(monkey=monkey, session_correlation=rho, scenario=scenario,
                                     cost_reduction=cost, tau_reduction=speed, measure=measure,
                                     detected_fraction=float(success.mean()), ci_lo=lo, ci_hi=hi,
                                     datasets=datasets, rate_shape_available_fraction=float(d["rate_valid"].mean()),
                                     median_half_life_first=float(np.nanmedian(d["half_first"])),
                                     median_half_life_last=float(np.nanmedian(d["half_last"])),
                                     true_half_life_first=args.tau_trials * np.log(2),
                                     true_half_life_last=args.tau_trials * (1 - speed) * np.log(2),
                                     nominal_curve_detection_fraction=float((d["joint"] > chi2.ppf(0.95, 2)).mean())))
                plot_rows.append({k: float(v.mean()) for k, v in detected.items()})
            for k, (name, label) in enumerate([("curve", "Any curve change"), ("initial_cost", "Smaller initial cost"),
                                               ("shape", "Faster shape proxy")]):
                axes[i, j].bar(np.arange(len(SCENARIOS)) + (k - 1) * 0.25,
                               [r[name] for r in plot_rows], width=0.25, label=label)
            axes[i, j].set_xticks(np.arange(len(SCENARIOS)), [s[0] for s in SCENARIOS], rotation=30)
            axes[i, j].set(title=f"{monkey}: neighboring-session correlation={rho}",
                           ylabel="Detected fraction (unavailable shapes count as undetected)", ylim=(0, 1))
            axes[i, j].grid(axis="y", alpha=0.2)
            if i == 0 and j == 0:
                axes[i, j].legend(fontsize=8)
    fig.suptitle(f"Conditional recoverability: starting tau={args.tau_trials:g} trials")
    fig.tight_layout()
    fig.savefig(args.out_dir / f"recoverability{suffix}.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(args.out_dir / f"simulation_detection{suffix}.csv", index=False)
    pd.DataFrame(calibration).to_csv(args.out_dir / f"simulation_calibration{suffix}.csv", index=False)
    with (args.out_dir / f"simulation_assumptions{suffix}.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(seed=20261013, datasets_per_scenario=args.draws,
                       population_initial_log_excess="Fitted first-exposure excess, with floor 0.10",
                       population_tau_trials=args.tau_trials, residual_block_trials=8, session_correlations=[0, 0.5],
                       calibration="10000 simulated null datasets set thresholds; 5000 independent null datasets test them",
                       rate_measure="Change in fast-basis fraction, requiring nonnegative fitted shape weights",
                       limitations=["Covariance estimates fixed; not full MixedLM refitting",
                                    "Conditions on observed completed reaches and target/failure patterns",
                                    "Does not simulate missing decoder changes or metadata errors",
                                    "Power for curve contrasts, not the earlier accepted-tau filter",
                                    "Nonnegative shape weights can restrict recovery at basis boundaries"]), handle, indent=2)
    print(pd.DataFrame(rows).query("scenario in ['null','speed50','cost50','both50']").to_string(index=False))
    print("Saved recoverability checks to", args.out_dir)


if __name__ == "__main__":
    main()
