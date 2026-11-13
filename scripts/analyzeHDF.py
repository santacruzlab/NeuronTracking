"""Compare first-target-to-reward times session by session, in sets of 8.

Run directly for the first 2 sets with evenly spaced sessions. Use
--first-sets X to choose the early window and --spacing calendar after reviewing
the even plots.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tables
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from trial_utils import (HDF_FS, SET_SIZE, list_sessions, get_two_target_rewarded_trials,
                         TIMING_DEFINITION, REWARDED_SEQUENCE)

ROOT = Path(__file__).resolve().parents[1]
FIRST_SETS = 2  # Early window selected by scripts/select_trial_window.py diagnostics.


def summarize_sets(duration, first_sets, target_angles=None):
    """Keep set numbers; flag invalid timing and, when supplied, target balance."""
    duration = np.asarray(duration, float)
    n_sets, leftover = divmod(len(duration), SET_SIZE)
    grouped = duration[:n_sets * SET_SIZE].reshape(n_sets, SET_SIZE)
    means = grouped.mean(axis=1)
    sems = grouped.std(axis=1, ddof=1) / np.sqrt(SET_SIZE)
    balanced = np.ones(n_sets, dtype=bool)
    if target_angles is not None:
        angles = np.asarray(target_angles[:n_sets * SET_SIZE]).reshape(n_sets, SET_SIZE)
        balanced = np.array([np.all(np.isfinite(a)) and len(np.unique(a)) == SET_SIZE
                             for a in angles], dtype=bool)
        means[~balanced] = np.nan
        sems[~balanced] = np.nan
    # Require the entire early window for a comparable session-level value.
    early = means[:first_sets]
    complete = n_sets >= first_sets and np.all(np.isfinite(early))
    return dict(means=means, sems=sems, balanced=balanced, n_sets=n_sets, leftover=leftover,
                early_mean=float(early.mean()) if complete else np.nan,
                early_sem=(float(early.std(ddof=1) / np.sqrt(first_sets))
                           if complete and first_sets > 1 else np.nan))


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare_windows(means, sensitivity_sets=3, late_start=25, min_late_sets=8):
    """Keep the sensitivity window fixed; require enough valid late sets."""
    means = np.asarray(means, float)
    early = means[:sensitivity_sets]
    complete = len(means) >= sensitivity_sets and np.all(np.isfinite(early))
    late = means[late_start - 1:]
    late = late[np.isfinite(late)]
    late_complete = len(late) >= min_late_sets
    return dict(sensitivity_mean=float(early.mean()) if complete else np.nan,
                sensitivity_sem=(float(early.std(ddof=1) / np.sqrt(sensitivity_sets))
                                 if complete and sensitivity_sets > 1 else np.nan),
                n_late_sets=len(late),
                late_mean=float(late.mean()) if late_complete else np.nan,
                late_sem=(float(late.std(ddof=1) / np.sqrt(len(late)))
                          if late_complete and len(late) > 1 else np.nan))


def plot_window_comparison(sessions, results, args):
    for block in args.blocks:
        fig, axes = plt.subplots(len(args.monkeys), 2,
                                 figsize=(15, 4.5 * len(args.monkeys)), squeeze=False)
        for r, monkey in enumerate(args.monkeys):
            ordered = sessions[monkey]
            recs = [results.get((s["path"].name, block)) for s in ordered]
            raw_ax, ratio_ax = axes[r]
            x = session_axis(raw_ax, ordered, args.spacing)
            session_axis(ratio_ax, ordered, args.spacing)
            for key, label, color in [("early", f"First {args.first_sets} sets", "C0"),
                                      ("sensitivity", f"First {args.sensitivity_sets} sets", "C1"),
                                      ("late", f"Late: set {args.late_start}+", "C2")]:
                y = [d[key + "_mean"] if d else np.nan for d in recs]
                err = [d[key + "_sem"] if d else np.nan for d in recs]
                raw_ax.errorbar(x, y, yerr=err, fmt="o-", ms=3, lw=1, capsize=2,
                                label=f"{label} (n={np.sum(np.isfinite(y))})", color=color)
            # Both normalized windows use exactly the same eligible sessions.
            paired = [d is not None and np.all(np.isfinite(
                [d["early_mean"], d["sensitivity_mean"], d["late_mean"]]))
                and d["late_mean"] > 0 for d in recs]
            for key, label, color in [("early", f"First {args.first_sets} / late", "C0"),
                                      ("sensitivity", f"First {args.sensitivity_sets} / late", "C1")]:
                ratio_ax.plot(x, [d[key + "_mean"] / d["late_mean"] if ok else np.nan
                                  for d, ok in zip(recs, paired)], "o-", ms=3, lw=1,
                              label=label, color=color)
            ratio_ax.axhline(1, color="k", ls="--", lw=0.8)
            ratio_ax.set_ylabel("Early mean / same-session late mean")
            ratio_ax.set_title(f"{monkey}, block {block}: matched sessions (n={sum(paired)})")
            raw_ax.set_title(f"{monkey}, block {block}: early and late performance")
            raw_ax.legend(fontsize=8)
            ratio_ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out_dir / f"block{block}_windows{args.first_sets}-{args.sensitivity_sets}_{args.spacing}.png", dpi=180)
        if not args.show:
            plt.close(fig)


def session_axis(ax, sessions, spacing):
    if spacing == "even":
        x = np.arange(1, len(sessions) + 1)
        ax.set_xticks(x)
        ax.set_xticklabels([s["label"] for s in sessions], rotation=60,
                           ha="right", fontsize=7)
        ax.set_xlabel("Session (chronological, evenly spaced)")
    else:
        # Actual dates: same-day sessions share an x coordinate, with no
        # invented time of day. Their session IDs remain in the CSVs.
        x = [s["date"] for s in sessions]
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.set_xlabel("Session date")
    ax.set_ylabel("Mean first target to reward (s)")
    ax.grid(alpha=0.3)
    return x


def plot_sessions(sessions, results, args):
    for block in args.blocks:
        fig, axes = plt.subplots(len(args.monkeys), 2,
                                 figsize=(15, 4.5 * len(args.monkeys)), squeeze=False)
        curves, curve_axes = plt.subplots(len(args.monkeys), 1,
                                         figsize=(10, 4 * len(args.monkeys)), squeeze=False)
        for r, monkey in enumerate(args.monkeys):
            ordered = sessions[monkey]
            ax, summary_ax = axes[r]
            x = session_axis(ax, ordered, args.spacing)
            session_axis(summary_ax, ordered, args.spacing)
            recs = [results.get((s["path"].name, block)) for s in ordered]
            for k in range(args.first_sets):
                means = [d["means"][k] if d and k < d["n_sets"] else np.nan for d in recs]
                sems = [d["sems"][k] if d and k < d["n_sets"] else np.nan for d in recs]
                ax.errorbar(x, means, yerr=sems, fmt="o-", ms=3, lw=1,
                            capsize=2, label=f"Set {k + 1}")
            ax.set_title(f"{monkey}, block {block}: each initial set of 8")
            ax.legend(fontsize=8)
            summary_ax.errorbar(x, [d["early_mean"] if d else np.nan for d in recs],
                                yerr=[d["early_sem"] if d else np.nan for d in recs],
                                fmt="o-", capsize=3)
            n = sum(d is not None and np.isfinite(d["early_mean"]) for d in recs)
            summary_ax.set_title(f"{monkey}, block {block}: first {args.first_sets} sets "
                                 f"({n}/{len(ordered)} sessions complete)")
            ca = curve_axes[r, 0]
            for j, d in enumerate(recs):
                if d and d["n_sets"]:
                    ca.plot(np.arange(1, d["n_sets"] + 1), d["means"], lw=0.8,
                            alpha=0.65, color=plt.cm.viridis(j / max(len(recs) - 1, 1)))
            ca.axvspan(0.5, args.first_sets + 0.5, color="orange", alpha=0.15)
            ca.set(title=f"{monkey}, block {block}: one curve per session (dark = earlier)",
                   xlabel="Trial set (8 consecutive rewards)", ylabel="Mean first target to reward (s)")
            ca.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / f"block{block}_first{args.first_sets}_{args.spacing}.png", dpi=180)
        curves.tight_layout()
        curves.savefig(args.out_dir / f"block{block}_all_set_curves.png", dpi=180)
        if not args.show:
            plt.close(fig)
            plt.close(curves)
    if args.show:
        plt.show()


def analyze_sessions(args):
    sessions = list_sessions(args.hdf_dir, args.monkeys)
    if not any(sessions.values()):
        raise ValueError(f"No matching airp/braz HDF sessions found in {args.hdf_dir}")
    results, trial_rows, set_rows, session_rows = {}, [], [], []
    for monkey, ordered in sessions.items():
        print(f"{monkey}: {len(ordered)} sessions")
        for session in ordered:
            for block in args.blocks:
                common = dict(monkey=monkey, block=block, session=session["label"],
                              file=session["path"].name, date=session["date"].date().isoformat())
                try:
                    trials = get_two_target_rewarded_trials(session["path"], block)
                except (OSError, tables.HDF5ExtError, tables.NoSuchNodeError, KeyError, ValueError) as err:
                    session_rows.append(dict(common, status=f"read_error: {err}"))
                    print(f"  SKIP {session['path'].name}, block {block}: {err}")
                    continue
                d = summarize_sets(trials["duration"], args.first_sets, trials["target_angle"])
                d.update(compare_windows(d["means"], args.sensitivity_sets,
                                         args.late_start, args.min_late_sets))
                results[(session["path"].name, block)] = d
                for j, duration in enumerate(trials["duration"]):
                    trial_rows.append(dict(common, trial=j + 1, set_number=j // SET_SIZE + 1,
                                           row_first_target=trials["row_first_target"][j],
                                           row_second_target=trials["row_second_target"][j],
                                           row_reward=trials["row_reward"][j], duration_s=duration,
                                           target_angle_degrees=trials["target_angle"][j],
                                           valid=bool(np.isfinite(duration)),
                                           invalid_reason=trials["invalid_reason"][j]))
                for k, mean in enumerate(d["means"]):
                    set_rows.append(dict(common, set_number=k + 1, mean_s=mean, sem_s=d["sems"][k],
                                         n_trials=SET_SIZE, selected=k < args.first_sets,
                                         eight_targets_once=bool(d["balanced"][k]),
                                         valid=bool(np.isfinite(mean))))
                status = ("no_rewards" if not len(trials["duration"]) else
                          "ok" if np.isfinite(d["early_mean"]) else "incomplete_or_invalid_early_window")
                session_rows.append(dict(common, status=status, n_rewards=len(trials["duration"]),
                                         invalid_trials=int(np.sum(~np.isfinite(trials["duration"]))),
                                         complete_sets=d["n_sets"], leftover_trials=d["leftover"],
                                         unbalanced_sets=int(np.sum(~d["balanced"])),
                                         requested_sets=args.first_sets, early_mean_s=d["early_mean"],
                                         early_sem_across_sets_s=d["early_sem"],
                                         sensitivity_sets=args.sensitivity_sets,
                                         sensitivity_mean_s=d["sensitivity_mean"],
                                         sensitivity_sem_across_sets_s=d["sensitivity_sem"],
                                         late_start_set=args.late_start, n_valid_late_sets=d["n_late_sets"],
                                         late_mean_s=d["late_mean"], late_sem_across_sets_s=d["late_sem"],
                                         early_late_ratio=(d["early_mean"] / d["late_mean"]
                                                           if d["late_mean"] > 0 else np.nan),
                                         sensitivity_late_ratio=(d["sensitivity_mean"] / d["late_mean"]
                                                                 if d["late_mean"] > 0 else np.nan),
                                         sensitivity_minus_primary_s=d["sensitivity_mean"] - d["early_mean"]))
                print(f"  {session['path'].name}, block {block}: {d['n_sets']} sets, "
                      f"{d['leftover']} leftover trials; first {args.first_sets} mean={d['early_mean']:.3f} s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "analysis_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(timing_definition=TIMING_DEFINITION, required_sequence=REWARDED_SEQUENCE,
                       task_clock_hz=HDF_FS, set_size=SET_SIZE, first_sets=args.first_sets,
                       sensitivity_sets=args.sensitivity_sets, late_start_set=args.late_start,
                       min_valid_late_sets=args.min_late_sets, invalid_rewards_preserve_positions=True), handle, indent=2)
    base = ["monkey", "block", "session", "file", "date"]
    write_csv(args.out_dir / "trial_times.csv", base + ["trial", "set_number", "row_first_target",
              "row_second_target", "row_reward", "duration_s", "target_angle_degrees", "valid", "invalid_reason"], trial_rows)
    write_csv(args.out_dir / "set_means.csv", base + ["set_number", "mean_s", "sem_s",
              "n_trials", "selected", "eight_targets_once", "valid"], set_rows)
    write_csv(args.out_dir / "session_summary.csv", base + ["status", "n_rewards", "invalid_trials",
              "complete_sets", "leftover_trials", "unbalanced_sets", "requested_sets", "early_mean_s",
              "early_sem_across_sets_s", "sensitivity_sets", "sensitivity_mean_s",
              "sensitivity_sem_across_sets_s", "late_start_set", "n_valid_late_sets",
              "late_mean_s", "late_sem_across_sets_s", "early_late_ratio",
              "sensitivity_late_ratio", "sensitivity_minus_primary_s"], session_rows)
    plot_window_comparison(sessions, results, args)
    plot_sessions(sessions, results, args)
    print(f"Saved to {args.out_dir}")


def inspect_hdf(path):
    """Retain the original single-file overview as an explicit option."""
    with tables.open_file(path, "r") as hdf:
        print(hdf)
        task, msgs = hdf.root.task[:], hdf.root.task_msgs[:]
    print("Task fields:", task.dtype.names)
    print("Rows:", len(task), f"-> {len(task) / HDF_FS:.1f} s")
    print(task[:3])
    print("First 15 task messages:", msgs[:15])
    states, counts = np.unique(msgs["msg"], return_counts=True)
    print("State counts:", dict(zip(states, counts)))
    if "cursor" in task.dtype.names:
        cur = task["cursor"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        t = np.arange(len(cur)) / HDF_FS
        ax[0].plot(t, cur[:, 0], label="x")
        ax[0].plot(t, cur[:, 2], label="z")
        ax[0].set(xlabel="Time (s)", ylabel="Cursor position", title="Cursor vs time")
        ax[0].legend()
        ax[1].plot(cur[:, 0], cur[:, 2], lw=0.5)
        ax[1].set(xlabel="x", ylabel="z", title="Cursor trajectory", aspect="equal")
        fig.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf-dir", type=Path, default=ROOT / "data" / "HDF")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "plots" / "premove_session_sets")
    parser.add_argument("--monkeys", nargs="+", choices=["airp", "braz"], default=["airp", "braz"])
    parser.add_argument("--blocks", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--first-sets", type=int, default=FIRST_SETS)
    parser.add_argument("--sensitivity-sets", type=int, default=3)
    parser.add_argument("--late-start", type=int, default=25)
    parser.add_argument("--min-late-sets", type=int, default=8)
    parser.add_argument("--spacing", choices=["even", "calendar"], default="even")
    parser.add_argument("--show", action="store_true", help="Display saved figures interactively")
    parser.add_argument("--inspect", type=Path, help="Inspect a single HDF instead of comparing sessions")
    args = parser.parse_args()
    if min(args.first_sets, args.sensitivity_sets, args.late_start, args.min_late_sets) < 1:
        parser.error("Set counts and --late-start must be at least 1")
    if args.late_start <= max(args.first_sets, args.sensitivity_sets):
        parser.error("--late-start must follow both early windows")
    if args.inspect:
        inspect_hdf(args.inspect)
    else:
        if not args.show:
            plt.switch_backend("Agg")
        analyze_sessions(args)


if __name__ == "__main__":
    main()
