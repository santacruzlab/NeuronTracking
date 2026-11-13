"""Plot session averages with trial min/max bars and within-session variance over time."""
import argparse
import csv
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-sets", type=int, default=2)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "plots/premove_session_sets")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "plots/two_set_trial_variation")
    args = parser.parse_args()
    if args.first_sets < 1:
        parser.error("--first-sets must be positive")
    sets = {(r["file"], int(r["set_number"])): r
            for r in read_rows(args.source_dir / "set_means.csv") if r["block"] == "2"}
    trials = defaultdict(list)
    for row in read_rows(args.source_dir / "trial_times.csv"):
        if row["block"] == "2" and 1 <= int(row["set_number"]) <= args.first_sets:
            trials[row["file"]].append(row)
    sessions = read_rows(args.source_dir / "session_summary.csv")
    n = 8 * args.first_sets
    exported = []
    fig, axes = plt.subplots(2, 1, figsize=(17, 9), layout="constrained")
    for ax, animal, color in zip(axes, ("airp", "braz"), ("C0", "C1")):
        cohort = sorted((s for s in sessions if s["block"] == "2" and s["monkey"] == animal),
                        key=lambda s: (s["date"], s["session"], s["file"]))
        first_date = date.fromisoformat(cohort[0]["date"])
        records = []
        for session in cohort:
            selected = [sets.get((session["file"], k)) for k in range(1, args.first_sets + 1)]
            rows = trials[session["file"]]
            durations = np.asarray([float(r["duration_s"]) for r in rows])
            valid = (all(s is not None and s["valid"] == "True" and s["eight_targets_once"] == "True"
                         for s in selected)
                     and len(rows) == n and all(r["valid"] == "True" for r in rows)
                     and bool(np.isfinite(durations).all()))
            record = dict(monkey=animal, block=2, session=session["session"], file=session["file"],
                          date=session["date"],
                          days_since_first_session=(date.fromisoformat(session["date"]) - first_date).days,
                          first_sets=args.first_sets, n_trials=len(rows), valid=valid,
                          mean_s=float(durations.mean()) if valid else np.nan,
                          variance_within_trials_s2=float(durations.var(ddof=1)) if valid else np.nan,
                          min_trial_s=float(durations.min()) if valid else np.nan,
                          max_trial_s=float(durations.max()) if valid else np.nan)
            records.append(record)
        exported.extend(records)
        points = [r for r in records if r["valid"]]
        ax.errorbar([r["days_since_first_session"] for r in points], [r["mean_s"] for r in points],
                    yerr=[[r["mean_s"] - r["min_trial_s"] for r in points],
                          [r["max_trial_s"] - r["mean_s"] for r in points]], fmt="o", linestyle="none",
                    color=color, markersize=4, elinewidth=1, capsize=3, alpha=0.85)
        ax.set(title=f"{animal}, block 2: {len(points)}/{len(cohort)} valid session averages",
               xlabel="Days since animal's first recorded session (day 0)",
               ylabel="Mean first target to reward (s)")
        ax.set_xlim(-1, max(r["days_since_first_session"] for r in records) + 1)
        ax.grid(alpha=0.25)
    fig.suptitle(f"First {args.first_sets} sets ({n} rewarded trials) averaged within each session\n"
                 f"Bars: lowest to highest time among the {n} individual trials; all selected target-balanced sets must be valid",
                 fontsize=12)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"block2_first{args.first_sets}_days_scatter_trial_range"
    for suffix in ("png", "pdf"):
        fig.savefig(args.out_dir / f"{stem}.{suffix}", dpi=180)
    plt.close(fig)
    with (args.out_dir / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    variance_fig, variance_axes = plt.subplots(2, 1, figsize=(17, 9), sharey=True,
                                             layout="constrained")
    for ax, animal, color in zip(variance_axes, ("airp", "braz"), ("C0", "C1")):
        records = [r for r in exported if r["monkey"] == animal]
        points = [r for r in records if r["valid"]]
        ax.scatter([r["days_since_first_session"] for r in points],
                   [r["variance_within_trials_s2"] for r in points],
                   color=color, s=30, alpha=0.85)
        ax.set(title=f"{animal}, block 2: {len(points)}/{len(records)} valid sessions",
               xlabel="Days since animal's first recorded session (day 0)",
               ylabel="Within-session trial variance (s²)")
        ax.set_xlim(-1, max(r["days_since_first_session"] for r in records) + 1)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
    variance_fig.suptitle(
        f"Trial-time variance over time: first {args.first_sets} sets ({n} rewarded trials)\n"
        f"Each point = sample variance across the {n} individual trial durations (denominator {n - 1})",
        fontsize=12)
    variance_stem = f"block2_first{args.first_sets}_days_scatter_trial_variance"
    for suffix in ("png", "pdf"):
        variance_fig.savefig(args.out_dir / f"{variance_stem}.{suffix}", dpi=180)
    plt.close(variance_fig)
    print("Saved trial min/max and variance scatterplots, PDFs and data to", args.out_dir)


if __name__ == "__main__":
    main()
