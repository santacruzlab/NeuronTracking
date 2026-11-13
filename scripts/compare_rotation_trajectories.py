"""Compare first three perturbed Brazos sessions and an agreed +50 degree reference."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tables

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots/rotation_trajectory_check"
FILES = ["braz20220315_07_te90.hdf", "braz20220316_06_te96.hdf", "braz20220318_07_te108.hdf",
         "braz20220507_04_te391.hdf"]
COLORS = ["C3", "C0", "C2", "C4"]


def align_path(cursor, target):
    """Translate start to zero; rotate desired direction to +x, preserving handedness."""
    points = np.asarray(cursor, float)[:, [0, 2]]
    desired = np.asarray(target, float)[[0, 2]] - points[0]
    distance = np.linalg.norm(desired)
    if distance <= 0 or not np.isfinite(points).all():
        raise ValueError("Non-finite path or zero target distance")
    forward = desired / distance
    left = np.array([-forward[1], forward[0]])
    aligned = (points - points[0]) @ np.column_stack([forward, left]) / distance
    return aligned, distance


def metrics(path):
    # Positive lateral deviation is on the CCW side of the desired direction.
    # Signed swept area instead measures the orientation of the path closed to its start.
    moving = np.flatnonzero(np.linalg.norm(path, axis=1) >= 0.2)
    angle = np.degrees(np.arctan2(path[moving[0], 1], path[moving[0], 0])) if len(moving) else np.nan
    area = 0.5 * np.sum(path[:-1, 0] * path[1:, 1] - path[:-1, 1] * path[1:, 0])
    return dict(mean_lateral=path[:, 1].mean(), heading_at_20pct_deg=angle, signed_swept_area=area)


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    attempts = pd.read_csv(ROOT / "plots/trial_learning/attempts.csv")
    settings = pd.read_csv(ROOT / "plots/within_session_learning/task_settings.csv").set_index("file")
    paths, rows, samples = {}, [], []
    titles = []
    for file in FILES:
        selected = attempts[(attempts.file == file) & (attempts.block == 2) & attempts.peripheral
                            & (attempts.perturbation == 1) & (attempts.error_clamp == 0)].sort_values(
                                "peripheral_attempt").head(16)
        cfg = settings.loc[file]
        titles.append(f"{cfg.date}: HDF {cfg.rotation_signed_deg:+g}°, catalog {cfg.catalog_rotation_signed_deg:+g}°")
        with tables.open_file(ROOT / "data/HDF" / file, "r") as hdf:
            for _, trial in selected.iterrows():
                start = int(trial.row_target)
                stop = int(trial.row_enter) if trial.row_enter >= start else int(trial.row_stop)
                data = hdf.root.task.read(start, stop + 1)
                # A target value can lag its message by one frame. Use the first peripheral
                # target in the first 0.2 s; retain the full cursor path from message onset.
                targets = data["target"][:13]
                peripheral = np.flatnonzero(np.linalg.norm(targets[:, [0, 2]], axis=1) > 1e-6)
                if not len(peripheral):
                    raise ValueError(f"No peripheral target at onset: {file}, {start}")
                target = targets[peripheral[0]]
                path, distance = align_path(data["cursor"], target)
                number = int(trial.peripheral_attempt)
                angle = int(round(np.degrees(np.arctan2(target[2], target[0])))) % 360
                paths[file, number] = (path, data["cursor"][:, [0, 2]], target[[0, 2]], trial.outcome, angle)
                rows.append(dict(file=file, date=cfg.date, hdf_rotation=cfg.rotation_signed_deg,
                                 catalog_rotation=cfg.catalog_rotation_signed_deg, attempt=number,
                                 outcome=trial.outcome, target_deg=angle, row_start=start, row_end=stop,
                                 endpoint="first_hold" if trial.row_enter >= start else "attempt_stop",
                                 target_distance=distance, **metrics(path)))
                samples.extend(dict(file=file, attempt=number, frame=i, time_s=i / 60,
                                    x=float(raw[0]), z=float(raw[1]), forward=float(p[0]), lateral=float(p[1]))
                               for i, (raw, p) in enumerate(zip(data["cursor"][:, [0, 2]], path)))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "trajectory_metrics.csv", index=False)
    pd.DataFrame(samples).to_csv(OUT / "trajectory_samples.csv", index=False)

    fig, axes = plt.subplots(3, len(FILES), figsize=(18, 12), sharex=True, sharey=True, layout="constrained")
    for col, file in enumerate(FILES):
        for row, number in enumerate((1, 2, 3)):
            ax = axes[row, col]
            _, raw, target, outcome, angle = paths[file, number]
            ax.plot(raw[:, 0], raw[:, 1], color=COLORS[col])
            ax.scatter(*raw[0], color="black", s=15, label="Start")
            ax.scatter(*target, marker="*", s=100, color="orange", label="Target")
            ax.plot([raw[0, 0], target[0]], [raw[0, 1], target[1]], "--", color="gray", alpha=0.5)
            for k in range(30, len(raw), 30):
                ax.annotate("", xy=raw[k], xytext=raw[k-6], arrowprops=dict(arrowstyle="->", color=COLORS[col]))
            ax.set(title=f"{titles[col]}\nAttempt {number}, target {angle}°, {outcome}", xlabel="Cursor x", ylabel="Cursor z")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
    fig.suptitle("First three peripheral attempts in each session (including failures)\nPaths end at first target hold, or attempt stop if no hold; arrows show time direction")
    save(fig, "first_three_attempts")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True, layout="constrained")
    for ax, angle in zip(axes.flat, range(0, 360, 45)):
        for color, file, title in zip(COLORS, FILES, titles):
            matches = frame[(frame.file == file) & (frame.target_deg == angle)].sort_values("attempt")
            if matches.empty:
                continue
            number = int(matches.iloc[0].attempt)
            path = paths[file, number][0]
            rotation = settings.loc[file, "rotation_signed_deg"]
            label = f"{title[:10]} ({rotation:+g}°, attempt {number})"
            ax.plot(path[:, 0], path[:, 1], color=color,
                    linestyle="--" if file == FILES[-1] else "-", label=label)
            ax.scatter(*path[-1], s=15, color=color)
        ax.scatter(1, 0, marker="*", s=80, color="orange")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set(title=f"Target {angle}°: first attempt among first 16", xlabel="Progress / target distance", ylabel="CCW-side deviation / distance")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle("Matched target directions: translated and rotated, with no reflection\nAbove zero = counterclockwise side of desired direction; below = clockwise side")
    save(fig, "matched_targets")

    summary = []
    for file, group in frame.groupby("file", sort=False):
        for n in (3, 16):
            d = group[group.attempt <= n]
            summary.append(dict(file=file, first_attempts=n, n=len(d),
                                median_lateral=d.mean_lateral.median(),
                                ccw_side_trials=int((d.mean_lateral > 0).sum()),
                                cw_side_trials=int((d.mean_lateral < 0).sum()),
                                median_heading_deg=d.heading_at_20pct_deg.median()))
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "direction_summary.csv", index=False)
    (OUT / "analysis_settings.json").write_text(json.dumps(dict(files=FILES,
        attempts="Original first 16 peripheral perturbed non-clamped attempts, retaining failures",
        plane="x horizontal, z vertical; screen mapping has not been independently verified",
        metric="Time-mean lateral displacement / desired target distance; positive is CCW side",
        heading="Direction from start at first displacement >=20% of initial target distance",
        limitations=["Descriptive trajectory comparison cannot establish actual applied rotation",
                     "Side of the target line differs from clockwise/counterclockwise turning along an arc",
                     "Attempts and target histories differ across sessions", "No labels or source data changed"]), indent=2))
    html = '<!doctype html><meta charset="utf-8"><title>Rotation trajectory check</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto}img{width:100%}td,th{padding:8px}</style>'
    html += '<h1>First three perturbed Brazos sessions</h1><p>March 15 is disputed (+50° HDF, −50° catalog). March 16 and March 18 both record −50°. Comparisons retain failed attempts.</p>'
    html += '<p><strong>Purple dashed reference: May 7, 2022 (te391).</strong> This is the first '
    html += 'Brazos session where both the HDF and catalog agree on +50°. It uses the same first-16-attempt '
    html += 'window and target alignment as the original sessions.</p>'
    html += '<p><strong>The early paths bend predominantly to opposite sides.</strong> '
    html += 'March 15 has positive mean lateral deviation in all 16 early attempts; '
    html += 'March 16 and March 18 have negative mean lateral deviation in 13/16 and 14/16 attempts. '
    html += 'This is evidence against treating March 15 as behaviorally equivalent to the next two −50° sessions. '
    html += 'It is consistent with the differing HDF label but does not independently prove the applied angle.</p>'
    reference = summary[(summary.file == FILES[-1]) & (summary.first_attempts == 16)].iloc[0]
    html += (f'<p>The agreed +50° reference has positive mean lateral deviation in '
             f'{int(reference.ccw_side_trials)}/{int(reference.n)} early attempts and negative deviation in '
             f'{int(reference.cw_side_trials)}/{int(reference.n)}. This reference occurs later in training; '
             'movement differences may also reflect adaptation and session history.</p>')
    html += '<h2>Direction summary</h2>' + summary.to_html(index=False, float_format=lambda x: f"{x:.3f}")
    for stem in ("first_three_attempts", "matched_targets"):
        html += f'<h2>{stem.replace("_", " ")}</h2><a href="{stem}.pdf">PDF</a><img src="{stem}.png">'
    html += '<p>Positive lateral deviation means the counterclockwise side of the target direction in the stored x-z plane. It does not directly specify the applied rotation sign. Paths, voluntary corrections, adaptation and decoder behavior can all affect curvature. See analysis_settings.json and trajectory_metrics.csv for definitions.</p>'
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
