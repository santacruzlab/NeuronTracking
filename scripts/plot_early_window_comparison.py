"""Generate first-X-set scatter figures and a single-page comparison sheet."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots/shifted_early_sets/window_comparison_1_3_to_11"
WINDOWS = [1, *range(3, 12)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-sets", nargs="+", type=int, default=WINDOWS,
                        help="Early window sizes in sets of 8 rewarded trials")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if any(count < 1 for count in args.first_sets):
        parser.error("--first-sets must contain positive integers")
    windows = list(dict.fromkeys(args.first_sets))
    out = args.out_dir.resolve()
    window_description = ", ".join(str(count) for count in windows)
    out.mkdir(parents=True, exist_ok=True)
    for count in windows:
        subprocess.run([sys.executable, "-B", str(ROOT / "scripts/plot_session_set_sequence.py"),
                        "--first-sets", str(count), "--blocks", "2", "--spacing", "days", "--scatter",
                        "--out-dir", str(out)], cwd=ROOT, check=True)

    fig, axes = plt.subplots(len(windows), 4, figsize=(22, 3.2 * len(windows)), sharex=True,
                             squeeze=False, layout="constrained")
    y_ranges = {}
    data = {}
    for count in windows:
        for kind in ["session_averages", "sequence"]:
            with (out / f"first{count}_days_scatter_{kind}.csv").open(newline="") as handle:
                data[count, kind] = list(csv.DictReader(handle))
    for monkey in ["airp", "braz"]:
        values = [float(r["mean_s"]) for rows in data.values() for r in rows
                  if r["monkey"] == monkey and r["valid"] == "True"]
        low, high = min(values), max(values)
        margin = (high - low) * 0.05
        y_ranges[monkey] = low - margin, high + margin
    for row_index, count in enumerate(windows):
        for column, (kind, monkey) in enumerate([
                ("session_averages", "airp"), ("session_averages", "braz"),
                ("sequence", "airp"), ("sequence", "braz")]):
            rows = [r for r in data[count, kind] if r["monkey"] == monkey]
            ax = axes[row_index, column]
            ax.scatter([int(r["days_since_first_session"]) for r in rows],
                       [float(r["mean_s"]) for r in rows], s=9,
                       color="C0" if monkey == "airp" else "C1", alpha=0.8)
            valid = sum(r["valid"] == "True" for r in rows)
            label = "session averages" if kind == "session_averages" else "individual set means"
            window = "Set 1" if count == 1 else f"First {count} sets"
            ax.set_title(f"{monkey} | {window} ({8 * count} trials) | {label}\n{valid}/{len(rows)} valid points", fontsize=10)
            ax.set_xlim(-1, 113)
            ax.set_ylim(*y_ranges[monkey])
            ax.grid(alpha=0.25)
            ax.tick_params(labelbottom=True, labelsize=8)
            ax.set_xlabel("Days since first session", fontsize=9)
            ax.set_ylabel("First target to reward (s)", fontsize=9)
    fig.suptitle(f"Block 2: first {window_description} trial sets\n"
                 "Left: one average per session | Right: individual set means\n"
                 "Each set = 8 rewarded trials; actual day gaps preserved; missing values omitted. "
                 "Session averages require every selected set to be valid.", fontsize=15)
    fig.savefig(out / "all_windows_one_page.pdf")
    fig.savefig(out / "all_windows_one_page.png", dpi=150)
    plt.close(fig)

    cards = []
    for count in windows:
        label = "First set" if count == 1 else f"First {count} sets"
        figures = []
        for kind, title in [("session_averages", "Session averages"), ("sequence", "Individual set means")]:
            name = f"block2_first{count}_days_scatter_{kind}.png"
            figures.append(f'<figure><figcaption>{title}</figcaption><a href="{name}">'
                           f'<img src="{name}" alt="{label}: {title}" loading="lazy"></a></figure>')
        cards.append(f'<section><h2>{label} ({8 * count} rewarded trials)</h2><div class="pair">'
                     + "".join(figures) + '</div></section>')
    html = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Early trial-set windows</title><style>'
            'body{font:16px system-ui;margin:24px;background:#f5f6f8;color:#18202a}'
            'section{background:white;padding:16px;margin:20px 0;border-radius:10px}'
            '.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}'
            'figure{margin:0}img{width:100%;height:auto}figcaption{font-weight:600;margin:8px 0}'
            '@media(max-width:900px){.pair{grid-template-columns:1fr}}</style>'
            '<h1>Block 2: early trial-set windows</h1>'
            f'<p>First {window_description} sets. Days since each animal\'s first session; '
            'scatter points without connecting lines. Missing/invalid values remain missing.</p>'
            '<p><a href="all_windows_one_page.pdf">Download the single-page comparison PDF</a></p>'
            + "".join(cards) + '</html>')
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "settings.json").write_text(json.dumps(dict(block=2, first_sets=windows,
        rewarded_trials=[8 * count for count in windows],
        spacing="Days since each animal's first recorded session", scatter=True,
        start_set=1, require_complete_window_for_average=True,
        timing="First target to reward, exact two-target seven-event sequence"), indent=2))
    print("Saved individual plots and single-page comparisons to", out, flush=True)


if __name__ == "__main__":
    main()
