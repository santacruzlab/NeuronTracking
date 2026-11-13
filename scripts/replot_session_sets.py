"""Replot saved validated set means with another early window or spacing.

This does not reread HDF data. Use analyzeHDF.py first after any timing change.
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from analyzeHDF import plot_sessions, plot_window_comparison, compare_windows, write_csv
from trial_utils import TIMING_DEFINITION, REWARDED_SEQUENCE

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "plots/premove_session_sets")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "plots/premove_session_sets")
    parser.add_argument("--first-sets", type=int, default=2)
    parser.add_argument("--spacing", choices=["even", "calendar"], default="even")
    args = parser.parse_args()
    args.monkeys, args.blocks, args.show = ["airp", "braz"], [2, 3], False
    args.sensitivity_sets, args.late_start, args.min_late_sets = 3, 25, 8
    if not 1 <= args.first_sets < args.late_start:
        parser.error("--first-sets must be between 1 and 24")
    def read(name):
        with (args.source_dir / name).open(newline="") as handle:
            return list(csv.DictReader(handle))
    saved, set_rows = read("session_summary.csv"), read("set_means.csv")
    groups = {}
    for row in set_rows:
        groups.setdefault((row["file"], int(row["block"])), []).append(row)
    sessions = {monkey: [] for monkey in args.monkeys}
    results, summaries = {}, []
    for monkey in args.monkeys:
        unique = {s["file"]: s for s in saved if s["monkey"] == monkey}
        for s in sorted(unique.values(), key=lambda s: (s["date"], s["session"], s["file"])):
            sessions[monkey].append(dict(path=Path(s["file"]), label=s["session"],
                                         date=datetime.fromisoformat(s["date"])))
    for s in saved:
        rows = sorted(groups.get((s["file"], int(s["block"])), []), key=lambda r: int(r["set_number"]))
        n = int(s["complete_sets"]) if s.get("complete_sets") else 0
        means, sems = np.full(n, np.nan), np.full(n, np.nan)
        for row in rows:
            k = int(row["set_number"]) - 1
            if row["valid"] == "True" and row["eight_targets_once"] == "True":
                means[k], sems[k] = float(row["mean_s"]), float(row["sem_s"])
        early = means[:args.first_sets]
        valid = len(early) == args.first_sets and np.all(np.isfinite(early))
        d = dict(means=means, sems=sems, n_sets=n,
                 early_mean=float(early.mean()) if valid else np.nan,
                 early_sem=float(early.std(ddof=1) / np.sqrt(len(early))) if valid and len(early) > 1 else np.nan)
        d.update(compare_windows(means, args.sensitivity_sets, args.late_start, args.min_late_sets))
        results[(s["file"], int(s["block"]))] = d
        summary = dict(s)
        summary.update(requested_sets=args.first_sets, early_mean_s=d["early_mean"],
                       early_sem_across_sets_s=d["early_sem"],
                       early_late_ratio=d["early_mean"] / d["late_mean"] if d["late_mean"] > 0 else np.nan,
                       sensitivity_minus_primary_s=d["sensitivity_mean"] - d["early_mean"])
        if not s["status"].startswith("read_error") and s.get("n_rewards") != "0":
            summary["status"] = "ok" if valid else "incomplete_or_invalid_early_window"
        summaries.append(summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.source_dir / "trial_times.csv").open(newline="") as handle:
        if "row_first_target" not in next(csv.reader(handle)):
            raise ValueError("Source trial times must first be regenerated with two-target timing")
    with (args.out_dir / "analysis_settings.json").open("w") as handle:
        json.dump(dict(timing_definition=TIMING_DEFINITION, required_sequence=REWARDED_SEQUENCE,
                       first_sets=args.first_sets, spacing=args.spacing,
                       source_dir=str(args.source_dir), task_clock_hz=60, set_size=8,
                       sensitivity_sets=3, late_start_set=25, min_valid_late_sets=8,
                       invalid_rewards_preserve_positions=True), handle, indent=2)
    if (args.out_dir.resolve() != args.source_dir.resolve()
            or any(s.get("requested_sets") != str(args.first_sets) for s in saved if not s["status"].startswith("read_error"))):
        write_csv(args.out_dir / "session_summary.csv", list(saved[0]), summaries)
    plot_window_comparison(sessions, results, args)
    plot_sessions(sessions, results, args)
    print(f"Replotted first {args.first_sets} sets ({args.spacing}) from saved set means into {args.out_dir}")


if __name__ == "__main__":
    main()
