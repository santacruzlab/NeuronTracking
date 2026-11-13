"""Diagnose early windows without selecting for a session-trend p value.

Consumes analyzeHDF CSVs, checks eight-target balance against the original HDF,
and compares early means with a fixed late reference (sets 25 onward, at least
8 late sets). Bootstrap intervals resample sessions, not individual trials.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tables
from trial_utils import rewarded_sequence_rows, TIMING_DEFINITION

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "plots" / "premove_session_sets"
OUTPUT = ROOT / "plots" / "window_selection"
MAX_X = 11
LATE_START = 25
MIN_LATE_SETS = 8
N_BOOT = 2000
MIN_RETAINED_EXCESS = 0.70  # Exploratory tradeoff, not a universal optimum.


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def target_angles_and_conditions(path):
    with tables.open_file(path, "r") as hdf:
        tbl = hdf.root.task
        blocks = tbl.read(field="block_type").ravel()
        target = tbl.read(field="target").reshape(len(blocks), -1)
        msgs = hdf.root.task_msgs[:]
        pert = tbl.read(field="pert").ravel()
        clamp = tbl.read(field="error_clamp").ravel()
    names = np.array([m.decode() for m in msgs["msg"]])
    rows = msgs["time"].astype(int)
    angles = {}
    for i in np.flatnonzero(names == "reward"):
        first, second = rewarded_sequence_rows(names, rows, i)
        if 0 <= first <= second <= rows[i] < len(target):
            p = target[second]
            angles[int(rows[i])] = int(np.round(np.degrees(np.arctan2(p[2], p[0])))) % 360
    conditions = []
    for block in np.unique(blocks):
        mask = blocks == block
        conditions.append(dict(file=path.name, block=int(block),
                               perturbation_values=json.dumps(np.unique(pert[mask]).tolist()),
                               error_clamp_values=json.dumps(np.unique(clamp[mask]).tolist())))
    return angles, conditions


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = read_csv(INPUT / "session_summary.csv")
    trials = read_csv(INPUT / "trial_times.csv")
    by_trial = {}
    for tr in trials:
        by_trial.setdefault((tr["file"], int(tr["block"])), []).append(tr)
    angles_by_file, conditions = {}, []
    for filename in sorted({s["file"] for s in summaries}):
        angles, cond = target_angles_and_conditions(ROOT / "data" / "HDF" / filename)
        angles_by_file[filename] = angles
        conditions.extend(cond)
    save_csv(OUTPUT / "block_conditions.csv", conditions)

    groups = {(m, b): [] for m in ["airp", "braz"] for b in [2, 3]}
    balance_rows = []
    for s in summaries:
        key = (s["file"], int(s["block"]))
        ts = by_trial.get(key, [])
        n = len(ts) // 8
        durations = np.array([float(t["duration_s"]) for t in ts[:n * 8]]).reshape(n, 8)
        angles = np.array([angles_by_file[s["file"]].get(int(t["row_reward"]), -1)
                           for t in ts[:n * 8]]).reshape(n, 8)
        balanced = np.array([np.all(a >= 0) and len(np.unique(a)) == 8 for a in angles])
        means = durations.mean(axis=1)
        means[~balanced] = np.nan
        for k, valid in enumerate(balanced):
            balance_rows.append(dict(file=s["file"], monkey=s["monkey"], block=key[1],
                                     set_number=k + 1, eight_targets_once=bool(valid)))
        late = means[LATE_START - 1:]
        late = late[np.isfinite(late)]
        reference = float(late.mean()) if len(late) >= MIN_LATE_SETS else np.nan
        groups[(s["monkey"], key[1])].append(dict(file=s["file"], means=means,
                                                 durations=durations, baseline=reference))
    save_csv(OUTPUT / "target_balance.csv", balance_rows)
    print("Target balance:", sum(r["eight_targets_once"] for r in balance_rows), "/", len(balance_rows))

    rng = np.random.default_rng(20261013)
    metrics, coverage, late_checks = [], [], []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for row, monkey in enumerate(["airp", "braz"]):
        for col, block in enumerate([2, 3]):
            recs = groups[(monkey, block)]
            # One fixed cohort for every X, avoiding cohort shifts as X grows.
            cohort = [d for d in recs if len(d["means"]) >= MAX_X
                      and np.all(np.isfinite(d["means"][:MAX_X])) and np.isfinite(d["baseline"])]
            M = np.array([d["means"][:MAX_X] for d in cohort])
            baseline = np.array([d["baseline"] for d in cohort])
            Q = M / baseline[:, None]
            ix = rng.integers(0, len(cohort), size=(N_BOOT, len(cohort)))
            med_curve = np.median(Q, axis=0)
            boot_curves = np.median(Q[ix], axis=1)
            lo, hi = np.percentile(boot_curves, [2.5, 97.5], axis=0)
            ax = axes[row, col]
            xvalues = np.arange(1, MAX_X + 1)
            ax.plot(xvalues, med_curve, "o-")
            ax.fill_between(xvalues, lo, hi, alpha=0.2)
            ax.axhline(1, color="k", ls="--", lw=0.8)
            ax.set(title=f"{monkey}, block {block}: n={len(cohort)} sessions",
                   xlabel="Set number (8 rewarded trials)", ylabel="Set mean / late mean")
            ax.grid(alpha=0.3)
            for X in range(1, MAX_X + 1):
                eligible = [d for d in recs if len(d["means"]) >= X
                            and np.all(np.isfinite(d["means"][:X]))]
                coverage.append(dict(monkey=monkey, block=block, X=X,
                                     eligible_sessions=len(eligible), total_sessions=len(recs)))
                early = M[:, :X].mean(axis=1)
                ratios = early / baseline
                excess = ratios - 1
                first_excess = Q[:, 0] - 1
                initial_cost = float(np.median(first_excess))
                # Retention is meaningful only for an initial slowdown.
                retention = np.median(excess) / initial_cost if initial_cost > 0 else np.nan
                boot_ratio = np.median(ratios[ix], axis=1)
                # A descriptive SE; between-set variation includes adaptation.
                rel_sem = (M[:, :X].std(axis=1, ddof=1) / np.sqrt(X) / early
                           if X > 1 else np.full(len(M), np.nan))
                metrics.append(dict(monkey=monkey, block=block, X=X, trials=8 * X,
                                    n_fixed_cohort=len(cohort), eligible_sessions=len(eligible),
                                    median_early_s=float(np.median(early)),
                                    median_late_s=float(np.median(baseline)),
                                    median_early_late_ratio=float(np.median(ratios)),
                                    initial_excess_positive=initial_cost > 0,
                                    ratio_ci_lo=float(np.percentile(boot_ratio, 2.5)),
                                    ratio_ci_hi=float(np.percentile(boot_ratio, 97.5)),
                                    retained_initial_excess=float(retention),
                                    median_relative_sem_across_sets=float(np.median(rel_sem))))
            full_late = [d for d in recs if len(d["means"]) >= 40
                         and np.all(np.isfinite(d["means"][24:40]))]
            drift = np.array([d["means"][32:40].mean() / d["means"][24:32].mean() - 1
                              for d in full_late])
            late_checks.append(dict(monkey=monkey, block=block, n_sessions=len(full_late),
                                    median_late_change=float(np.median(drift)),
                                    q25=float(np.percentile(drift, 25)),
                                    q75=float(np.percentile(drift, 75))))
    fig.tight_layout()
    fig.savefig(OUTPUT / "normalized_early_curves.png", dpi=180)
    plt.close(fig)
    save_csv(OUTPUT / "candidate_windows.csv", metrics)
    save_csv(OUTPUT / "window_coverage.csv", coverage)
    save_csv(OUTPUT / "late_reference_check.csv", late_checks)
    # Use primary block 2 in both animals. Prefer the widest early window that
    # retains the specified fraction of initial slowdown, without looking at
    # correlations with session order or SFC.
    sensitivity = {}
    for threshold in [0.60, MIN_RETAINED_EXCESS, 0.80]:
        acceptable = [X for X in range(1, MAX_X + 1)
                      if all(next(m for m in metrics if m["monkey"] == monkey
                                  and m["block"] == 2 and m["X"] == X)
                             ["retained_initial_excess"] >= threshold
                             for monkey in ["airp", "braz"])]
        sensitivity[str(threshold)] = max(acceptable) if acceptable else None
    selected = sensitivity[str(MIN_RETAINED_EXCESS)]
    reference_sensitivity = {}
    for name, reducer in [("late_mean", np.mean), ("late_median", np.median)]:
        retained = {}
        for monkey in ["airp", "braz"]:
            cohort = [d for d in groups[(monkey, 2)] if len(d["means"]) >= MAX_X
                      and np.all(np.isfinite(d["means"][:MAX_X])) and np.isfinite(d["baseline"])]
            early_sets = np.array([d["means"][:MAX_X] for d in cohort])
            ref = np.array([reducer(d["means"][24:][np.isfinite(d["means"][24:])]) for d in cohort])
            initial = np.median(early_sets[:, 0] / ref - 1)
            retained[monkey] = [float(np.median(early_sets[:, :X].mean(axis=1) / ref - 1) / initial)
                                for X in range(1, MAX_X + 1)]
        acceptable = [X for X in range(1, MAX_X + 1)
                      if all(retained[m][X - 1] >= MIN_RETAINED_EXCESS for m in retained)]
        reference_sensitivity[name] = max(acceptable) if acceptable else None
    selection = dict(selected_X=selected, selected_trials=8 * selected if selected else None,
                     timing_definition=TIMING_DEFINITION,
                     primary_block=2, min_retained_initial_excess=MIN_RETAINED_EXCESS,
                     threshold_sensitivity=sensitivity,
                     late_reference_sensitivity=reference_sensitivity,
                     criterion="Widest window retaining >=70% of initial slowdown in both animals; exploratory",
                     reference="Mean of valid sets 25 onward, with at least 8 valid late sets",
                     uncertainty="2000 session bootstrap draws; relative set SEM is descriptive and includes adaptation",
                     exclusions="Require balanced eight-target sets and finite timing; fixed cohort valid through set 11",
                     limitation="Does not establish sufficient spike counts or SFC reliability, or test meta-learning")
    with (OUTPUT / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2, allow_nan=False)
    decision_fig, decision_axes = plt.subplots(1, 2, figsize=(11, 4))
    for monkey in ["airp", "braz"]:
        rows = [m for m in metrics if m["monkey"] == monkey and m["block"] == 2]
        xs = [m["X"] for m in rows]
        decision_axes[0].plot(xs, [100 * m["retained_initial_excess"] for m in rows], "o-", label=monkey)
        decision_axes[1].plot(xs, [100 * m["median_relative_sem_across_sets"] for m in rows], "o-", label=monkey)
    decision_axes[0].axhline(100 * MIN_RETAINED_EXCESS, color="k", ls="--", label="70% criterion")
    decision_axes[0].set(ylabel="Initial excess retained (%)", title="Block 2: dilution of early adaptation")
    decision_axes[1].set(ylabel="Median relative SEM across sets (%)", title="Descriptive variability (includes adaptation)")
    for ax in decision_axes:
        ax.axvline(selected, color="gray", ls=":")
        ax.set_xlabel("First X sets (8 trials per set)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    decision_fig.tight_layout()
    decision_fig.savefig(OUTPUT / "window_tradeoff.png", dpi=180)
    plt.close(decision_fig)
    print("Selection:", json.dumps(selection))
    for m in metrics:
        if m["X"] in [1, 2, 3, 4, 6, 8, 11]:
            print(f"{m['monkey']} B{m['block']} X={m['X']:2}: "
                  f"early/late={m['median_early_late_ratio']:.3f}; "
                  f"retain={m['retained_initial_excess']:.1%}; "
                  f"relative SEM={m['median_relative_sem_across_sets']:.1%}")
    print("Late checks:", json.dumps(late_checks))
    print("Saved to", OUTPUT)


if __name__ == "__main__":
    main()
