"""Plot X consecutive set means per session as one chronological line per animal.

Consumes saved behavioral summaries without rereading HDF files. Each session
occupies exactly X positions, including gaps for missing or invalid sets.
Companion plots average those X sets into one point per session, requiring
all X sets to be valid.
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

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-sets", type=int, default=6)
    parser.add_argument("--start-set", type=int, default=1)
    parser.add_argument("--spacing", choices=["even", "calendar", "days"], default="even")
    parser.add_argument("--scatter", action="store_true", help="Draw points without connecting lines")
    parser.add_argument("--blocks", nargs="+", choices=[2, 3], type=int, default=[2, 3])
    parser.add_argument("--source-dir", type=Path, default=ROOT / "plots/premove_session_sets")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "plots/session_set_sequence")
    args = parser.parse_args()
    if args.first_sets < 1 or args.start_set < 1:
        parser.error("--first-sets and --start-set must be positive")
    last_set = args.start_set + args.first_sets - 1
    window_label = (f"first {args.first_sets} sets" if args.start_set == 1
                    else f"sets {args.start_set}-{last_set}")
    stem = (f"first{args.first_sets}" if args.start_set == 1
            else f"sets{args.start_set}-{last_set}")
    if args.spacing != "even":
        stem += f"_{args.spacing}"
    if args.scatter:
        stem += "_scatter"
    with (args.source_dir / "set_means.csv").open(newline="") as handle:
        sets = {(row["file"], row["block"], int(row["set_number"])): row
                for row in csv.DictReader(handle)}
    with (args.source_dir / "session_summary.csv").open(newline="") as handle:
        sessions = list(csv.DictReader(handle))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    exported, averages = [], []
    for block in map(str, args.blocks):
        fig, axes = plt.subplots(2, 1, figsize=(17, 9), squeeze=False)
        mean_fig, mean_axes = plt.subplots(2, 1, figsize=(17, 9), squeeze=False)
        for ax, mean_ax, monkey in zip(axes[:, 0], mean_axes[:, 0], ["airp", "braz"]):
            cohort = sorted([s for s in sessions if s["monkey"] == monkey and s["block"] == block],
                            key=lambda s: (s["date"], s["session"], s["file"]))
            values, session_means = [], []
            dates = mdates.date2num([datetime.fromisoformat(s["date"]) for s in cohort])
            elapsed_days = dates - dates[0] if len(dates) else dates
            time_positions = elapsed_days if args.spacing == "days" else dates
            for j, session in enumerate(cohort):
                start = j * args.first_sets + 1
                if j % 2 == 0 and args.spacing == "even":
                    ax.axvspan(start - 0.5, start + args.first_sets - 0.5, color="0.94", zorder=0)
                if j and args.spacing == "even":
                    ax.axvline(start - 0.5, color="0.8", lw=0.5, zorder=1)
                for k in range(args.start_set, last_set + 1):
                    row = sets.get((session["file"], block, k))
                    valid = row is not None and row["valid"] == "True" and row["eight_targets_once"] == "True"
                    value = float(row["mean_s"]) if valid else np.nan
                    values.append(value)
                    exported.append(dict(monkey=monkey, block=block, session=session["session"],
                                         file=session["file"], session_position=j + 1, set_number=k,
                                         point_position=len(values), mean_s=value,
                                         date=session["date"],
                                         days_since_first_session=int(elapsed_days[j]),
                                         sem_s=float(row["sem_s"]) if valid else np.nan,
                                         valid=bool(valid and np.isfinite(value))))
                window = np.asarray(values[-args.first_sets:])
                complete = bool(np.all(np.isfinite(window)))
                mean = float(window.mean()) if complete else np.nan
                sem = (float(window.std(ddof=1) / np.sqrt(args.first_sets))
                       if complete and args.first_sets > 1 else np.nan)
                session_means.append(mean)
                averages.append(dict(monkey=monkey, block=block, session=session["session"],
                                     file=session["file"], session_position=j + 1,
                                     first_sets=args.first_sets, n_valid_sets=int(np.isfinite(window).sum()),
                                     start_set=args.start_set, end_set=last_set, date=session["date"],
                                     days_since_first_session=int(elapsed_days[j]),
                                     mean_s=mean, sem_across_sets_s=sem, valid=complete))
            sequence_positions = (np.repeat(time_positions, args.first_sets) if args.spacing != "even"
                                  else np.arange(1, len(values) + 1))
            if args.scatter:
                ax.scatter(sequence_positions, values, color="C0", s=18)
            else:
                ax.plot(sequence_positions, values, "o-", color="C0", ms=2.7, lw=1)
            centers = np.arange(len(cohort)) * args.first_sets + (args.first_sets + 1) / 2
            if args.spacing == "even":
                ax.set_xticks(centers)
                ax.set_xticklabels([s["session"] for s in cohort], rotation=60, ha="right", fontsize=7)
            ax.set(title=f"{monkey}, block {block}: {window_label} per session",
                   ylabel="Mean first target to reward (s)",
                   xlabel=f"Sessions in chronological order; within each session: sets {args.start_set} to {last_set}")
            ax.grid(axis="y", alpha=0.25)
            if values and args.spacing == "even":
                ax.set_xlim(0.5, len(values) + 0.5)
            print(f"{monkey}, block {block}: {len(cohort)} sessions, {len(values)} positions, "
                  f"{np.isfinite(values).sum()} valid set means")
            positions = time_positions if args.spacing != "even" else np.arange(1, len(cohort) + 1)
            if args.scatter:
                mean_ax.scatter(positions, session_means, color="C0", s=24)
            else:
                mean_ax.plot(positions, session_means, "o-", color="C0", ms=4, lw=1.2)
            if args.spacing == "even":
                mean_ax.set_xticks(positions)
                mean_ax.set_xticklabels([s["session"] for s in cohort], rotation=60, ha="right", fontsize=7)
            mean_ax.set(title=f"{monkey}, block {block}: {np.isfinite(session_means).sum()}/{len(cohort)} valid session averages",
                        ylabel="Mean first target to reward (s)", xlabel="Session (chronological, evenly spaced)")
            mean_ax.grid(alpha=0.25)
            if len(cohort) and args.spacing == "even":
                mean_ax.set_xlim(0.5, len(cohort) + 0.5)
            if args.spacing == "calendar":
                for calendar_ax in [ax, mean_ax]:
                    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
                    calendar_ax.xaxis.set_major_locator(locator)
                    calendar_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
                    calendar_ax.tick_params(axis="x", labelrotation=0, labelsize=9)
                    calendar_ax.set_xlabel("Recording date (actual calendar spacing)")
                    calendar_ax.autoscale_view()
            elif args.spacing == "days":
                for day_ax in [ax, mean_ax]:
                    day_ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    day_ax.set_xlabel("Days since animal's first recorded session (day 0)")
                    if len(elapsed_days):
                        day_ax.set_xlim(-1, max(float(elapsed_days[-1]), 1) + 1)
        sequence_note = ("Within-session points share the same day; actual gaps between sessions are preserved"
                         if args.spacing != "even" else "Vertical boundaries separate sessions")
        plot_label = "Unconnected points per animal" if args.scatter else "One continuous line per animal"
        fig.suptitle(f"{plot_label}; each point = 8 rewarded trials\n"
                     f"{sequence_note}; missing or invalid sets remain gaps", fontsize=12)
        fig.tight_layout()
        path = args.out_dir / f"block{block}_{stem}_sequence.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        print("Saved", path)
        mean_fig.suptitle(f"{window_label.capitalize()} averaged within each session "
                          f"({8 * args.first_sets} rewarded trials per point)\n"
                          f"{'Unconnected points' if args.scatter else 'One line'} per animal; "
                          "all selected sets must be valid; missing sessions remain gaps", fontsize=12)
        mean_fig.tight_layout()
        mean_path = args.out_dir / f"block{block}_{stem}_session_averages.png"
        mean_fig.savefig(mean_path, dpi=180)
        plt.close(mean_fig)
        print("Saved", mean_path)
    with (args.out_dir / f"{stem}_sequence.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    with (args.out_dir / f"{stem}_session_averages.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(averages[0]))
        writer.writeheader()
        writer.writerows(averages)


if __name__ == "__main__":
    main()
