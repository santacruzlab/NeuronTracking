"""Early block 2 SEM, block 1 normalization, variability and metadata inventory."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tables
from analyze_session_learning import regression_trend

ROOT = Path(__file__).resolve().parents[1]


def describe(values):
    values = np.asarray(values, float)
    if len(values) < 2 or not np.isfinite(values).all():
        return dict(mean=np.nan, sd=np.nan, sem=np.nan)
    sd = values.std(ddof=1)
    return dict(mean=values.mean(), sd=sd, sem=sd / np.sqrt(len(values)))


def bootstrap_means(values, draws, block, rng):
    """Circular moving blocks retain local ordering; conditional on observed trials."""
    values = np.asarray(values, float)
    starts = rng.integers(len(values), size=(draws, int(np.ceil(len(values) / block))))
    indices = ((starts[..., None] + np.arange(block)) % len(values)).reshape(draws, -1)
    return values[indices[:, :len(values)]].mean(axis=1)


def inventory(settings, hdf_dir, out_dir):
    rows = []
    for file in settings.file:
        path = hdf_dir / file
        record = dict(file=file, status="missing", nodes="", fields="", relevant_attributes="")
        if path.exists():
            with tables.open_file(path, "r") as hdf:
                nodes, fields, attrs = [], {}, {}
                for node in hdf.walk_nodes():
                    nodes.append(node._v_pathname)
                    if isinstance(node, tables.Table):
                        fields[node._v_pathname] = node.colnames
                    for key in node._v_attrs._f_list():
                        if any(term in key.lower() for term in ("pert", "rot", "decoder", "task", "assist", "log")):
                            attrs[node._v_pathname + ":" + key] = str(getattr(node._v_attrs, key))
                record.update(status="ok", nodes=json.dumps(nodes), fields=json.dumps(fields),
                              relevant_attributes=json.dumps(attrs, sort_keys=True))
        rows.append(record)
    pd.DataFrame(rows).to_csv(out_dir / "hdf_metadata_inventory.csv", index=False)
    # Search local data filenames only; no claim about logs outside this repository.
    candidates = [str(p.relative_to(ROOT)) for p in (ROOT / "data").rglob("*")
                  if p.is_file() and (p.suffix.lower() in (".log", ".txt", ".json", ".yaml", ".yml")
                                      or "behavior" in p.name.lower())]
    (out_dir / "local_log_candidates.json").write_text(json.dumps(candidates, indent=2))


def summarize(trials, sets, attempts, settings, draws, rng):
    records = []
    for _, session in settings.iterrows():
        a = attempts[attempts.file == session.file]
        base = a[(a.block == 1) & a.valid_timing & (a.outcome == "success")
                 & (a.error_clamp == 0) & (a.perturbation == 0)].sort_values("reward_number").tail(128)
        baseline = base.duration_s.to_numpy()
        baseline_valid = (len(base) >= 64 and base.target_deg.nunique() == 8
                          and np.isfinite(baseline).all() and (baseline > 0).all())
        b = describe(baseline) if baseline_valid else describe([])
        for window in (1, 2, 3):
            s = sets[(sets.file == session.file) & (sets.block == 2)
                     & sets.set_number.between(1, window)]
            t = trials[(trials.file == session.file) & (trials.block == 2)
                       & trials.set_number.between(1, window)].sort_values("trial")
            values = t.duration_s.to_numpy()
            valid = (len(s) == window and s.valid.all() and s.eight_targets_once.all()
                     and len(t) == 8 * window and t.valid.all()
                     and np.isfinite(values).all() and (values > 0).all())
            d = describe(values) if valid else describe([])
            rec = dict(file=session.file, monkey=session.monkey, date=session.date,
                       window_trials=8 * window, early_valid=valid, baseline_valid=baseline_valid,
                       n_early=len(t), n_baseline=len(base), baseline_target_counts=json.dumps(
                           {str(k): int(v) for k, v in base.target_deg.value_counts().sort_index().items()}),
                       mean_s=d["mean"], sd_s=d["sd"], sem_s=d["sem"], variance_s2=d["sd"] ** 2,
                       cv=d["sd"] / d["mean"], relative_sem=d["sem"] / d["mean"],
                       baseline_mean_s=b["mean"], baseline_sd_s=b["sd"], baseline_sem_s=b["sem"],
                       ratio=d["mean"] / b["mean"],
                       normalized_mean=d["mean"] / b["mean"],
                       normalized_sem=d["sem"] / b["mean"],
                       z_mean=(d["mean"] - b["mean"]) / b["sd"] if b["sd"] > 0 else np.nan,
                       z_sem=d["sem"] / b["sd"] if b["sd"] > 0 else np.nan,
                       rotation=session.rotation_signed_deg,
                       catalog_rotation=session.catalog_rotation_signed_deg,
                       rotation_disputed=not session.rotation_matches_catalog)
            for block in (4, 8):
                lo, hi = np.nan, np.nan
                if valid and baseline_valid:
                    ratios = (bootstrap_means(values, draws, block, rng)
                              / bootstrap_means(baseline, draws, block, rng))
                    lo, hi = np.quantile(ratios, [0.025, 0.975])
                rec[f"ratio_lo_b{block}"], rec[f"ratio_hi_b{block}"] = lo, hi
            records.append(rec)
    frame = pd.DataFrame(records)
    dates = pd.to_datetime(frame.date)
    frame["days"] = (dates - dates.groupby(frame.monkey).transform("min")).dt.days
    return frame


def plot(frame, out_dir):
    specs = [("mean_s", "Mean duration (s); bars = trial SEM", None),
             ("normalized_mean", "Mean duration / session block 1 mean; bars = normalized trial SEM", 1),
             ("ratio", "Mean / block 1 mean; bars = 95% block-bootstrap interval", 1),
             ("z_mean", "Mean baseline z-score; bars = conditional SEM", 0),
             ("variance_s2", "Trial sample variance (s squared)", None),
             ("relative_sem", "Trial SEM / mean", None), ("cv", "Trial SD / mean", None)]
    angles = sorted(frame.rotation.dropna().unique())
    colors = dict(zip(angles, plt.get_cmap("tab10").colors))
    for metric, label, reference in specs:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True, layout="constrained")
        for row, animal in enumerate(("airp", "braz")):
            for col, window in enumerate((8, 16, 24)):
                ax = axes[row, col]
                group = frame[(frame.monkey == animal) & (frame.window_trials == window)]
                for angle in angles:
                    points = group[(group.rotation == angle) & group[metric].notna()]
                    if points.empty:
                        continue
                    err = None
                    if metric == "mean_s":
                        err = points.sem_s
                    elif metric == "normalized_mean":
                        err = points.normalized_sem
                    elif metric == "z_mean":
                        err = points.z_sem
                    elif metric == "ratio":
                        # Plot intervals independently: percentile intervals need not contain the point estimate.
                        ax.vlines(points.days, points.ratio_lo_b4, points.ratio_hi_b4,
                                  color=colors[angle], alpha=0.6, linewidth=1)
                    ax.errorbar(points.days, points[metric], yerr=err, fmt="o", ms=3,
                                color=colors[angle], capsize=2, label=f"{angle:+g} deg")
                    disputed = points[points.rotation_disputed]
                    ax.scatter(disputed.days, disputed[metric], marker="x", color="black", s=70, zorder=5)
                if reference is not None:
                    ax.axhline(reference, color="gray", linestyle="--", linewidth=1)
                ax.set(title=f"{animal}: first {window} trials (n={group[metric].notna().sum()})",
                       xlabel="Days since first recorded session",
                       ylabel=("Mean / baseline mean" if metric == "normalized_mean" else metric) if col == 0 else "")
                ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(label + "\nColors: recorded HDF rotation; black cross: disputed catalog label", fontsize=12)
        for extension in ("png", "pdf"):
            fig.savefig(out_dir / f"{metric}.{extension}", dpi=160)
        plt.close(fig)


def trends(frame, out_dir, rng):
    rows = []
    for (animal, window), group in frame.groupby(["monkey", "window_trials"]):
        for metric in ("mean_s", "ratio", "z_mean"):
            for variant in ("unadjusted", "hdf_rotation", "catalog_rotation", "include_disputed_hdf"):
                selected = group[group[metric].notna()].copy()
                if variant != "include_disputed_hdf":
                    selected = selected[~selected.rotation_disputed]
                selected["session_number"] = selected.file.str.split("_").str[1].astype(int)
                selected = selected.sort_values(["date", "session_number", "file"])
                angle = "catalog_rotation" if variant == "catalog_rotation" else "rotation"
                result = regression_trend(selected.days.to_numpy(float), selected[metric].to_numpy(float),
                                          selected[angle].astype(str).tolist(), variant != "unadjusted", rng)
                rows.append(dict(monkey=animal, window_trials=window, metric=metric, variant=variant, **result))
    pd.DataFrame(rows).to_csv(out_dir / "exploratory_calendar_trends.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "plots/baseline_diagnostics")
    parser.add_argument("--hdf-dir", type=Path, default=ROOT / "data/HDF")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    settings = pd.read_csv(ROOT / "plots/within_session_learning/task_settings.csv")
    attempts = pd.read_csv(ROOT / "plots/trial_learning/attempts.csv")
    trials = pd.read_csv(ROOT / "plots/premove_session_sets/trial_times.csv")
    sets = pd.read_csv(ROOT / "plots/premove_session_sets/set_means.csv")
    frame = summarize(trials, sets, attempts, settings, args.bootstrap, np.random.default_rng(20261030))
    frame.to_csv(args.out_dir / "session_diagnostics.csv", index=False)
    # Fixed cohort for comparing precision across trial windows.
    common = frame.groupby("file").early_valid.all()
    matched = frame[frame.file.isin(common[common].index)]
    matched.groupby(["monkey", "window_trials"]).agg(
        n_sessions=("file", "size"), median_relative_sem=("relative_sem", "median"),
        median_cv=("cv", "median"), median_ratio=("ratio", "median")
    ).to_csv(args.out_dir / "matched_window_summary.csv")
    attempts.groupby(["monkey", "file", "block", "perturbation", "error_clamp"], dropna=False).size().rename(
        "n_attempts").to_csv(args.out_dir / "recorded_perturbation_counts.csv")
    plot(frame, args.out_dir)
    trends(frame, args.out_dir, np.random.default_rng(20261031))
    inventory(settings, args.hdf_dir, args.out_dir)
    config = dict(seed=20261030, bootstrap=args.bootstrap, blocks=[4, 8],
                  baseline="Last 128 valid successful non-clamped pert=0 block 1 trials; minimum 64 and all 8 directions",
                  baseline_weighting="Pooled trial mean, not equal direction weights; target counts exported",
                  early="Exact original first 1/2/3 target-balanced sets; invalid positions never replaced",
                  sem="Sample SD / sqrt(n); descriptive, assumes independent trials for precision interpretation",
                  normalized_mean="Early trial mean / same-session block 1 mean; SEM divided by the same baseline mean, conditional on baseline estimate",
                  z_score="(Block 2 duration - block 1 mean) / block 1 sample SD",
                  ratio_uncertainty="Independent circular block resampling of early and baseline trials; 4/8-trial sensitivity",
                  limitations=["Adaptation, target effects and serial dependence contribute to variability",
                               "Very few early blocks make bootstrap intervals exploratory",
                               "With 8 early trials, block length 8 resamples have a constant early mean; that sensitivity cannot measure numerator uncertainty",
                               "Z-score SEM conditions on baseline mean and SD",
                               "Larger early windows also include more adaptation, not just more samples",
                               "Perturbation codes and rotation angles do not identify unrecorded settings",
                               "Metadata inventory reads current HDF; analysis CSVs must match those source files",
                               "Calendar trends reuse 5000-draw residual-block intervals of lengths 4/6; exploratory, no multiplicity correction",
                               "Rotation-adjusted trends assume a common slope across regimes confounded with chronology"])
    (args.out_dir / "analysis_settings.json").write_text(json.dumps(config, indent=2))
    summary = pd.read_csv(args.out_dir / "matched_window_summary.csv")
    html = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Baseline diagnostics</title>',
            '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:16px}img{width:100%}td,th{padding:8px;text-align:right}p{line-height:1.5}</style>',
            '<h1>Trial-count and baseline diagnostics</h1>',
            '<p>Compare 8, 16 and 24 original rewarded trials in block 2. Baseline: last 128 valid, '
            'unperturbed, non-clamped successes in block 1 (minimum 64, all eight directions). '
            'Ratios above 1 and z-scores above 0 indicate slower performance than the session baseline.</p>',
            '<h2>Precision on the same sessions across all three windows</h2>',
            summary.to_html(index=False, float_format=lambda x: f"{x:.4f}"),
            '<p>Trial SEM is descriptive: adaptation, target differences and dependence contribute to variability. '
            'Increasing the window changes the adaptation period as well as the number of observations. '
            'These checks cannot establish that trial count causes an animal difference.</p>']
    for metric in ("mean_s", "normalized_mean", "ratio", "z_mean", "variance_s2", "relative_sem", "cv"):
        html.append(f'<h2>{metric}</h2><a href="{metric}.pdf">Download PDF</a><img src="{metric}.png" alt="{metric} by animal and early window">')
    html.extend(['<h2>Methods and limitations</h2>',
                 '<p>Normalized mean bars are trial SEM divided by the same session baseline mean. '
                 'They condition on the estimated baseline; the separate ratio plot also includes baseline uncertainty.</p>',
                 '<p>Ratio bars show 95% percentile intervals using independent circular four-trial block '
                 'resamples of early and baseline trials. Eight-trial block sensitivity is in the CSV. '
                 'At an eight-trial window, eight-trial blocks cannot estimate numerator uncertainty. '
                 'Z-score SEM holds the estimated baseline mean and SD fixed. Baseline means pool trials; '
                 'direction counts are exported to reveal unequal target representation.</p>',
                 '<p>Colors identify HDF rotations; the black cross marks the disputed first braz session. '
                 'Calendar trend CSVs exclude it except in an explicit sensitivity model. Trends are '
                 'exploratory and assume a common slope when adjusting for rotations that vary with chronology.</p>',
                 '<p>The metadata inventory covers all HDF nodes, table fields, and relevant attributes. '
                 'The perturbation table samples recorded attempt anchors, not every task row. '
                 'A binary pert code does not identify five perturbation types. Local filename searches '
                 'cover the repository data directory; external behavior logs remain unverified.</p>',
                 '<p><a href="session_diagnostics.csv">Session data</a> · '
                 '<a href="exploratory_calendar_trends.csv">Exploratory slopes and intervals</a> · '
                 '<a href="hdf_metadata_inventory.csv">HDF inventory</a> · '
                 '<a href="analysis_settings.json">Full settings</a></p></html>'])
    (args.out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(matched.groupby(["monkey", "window_trials"]).relative_sem.median().to_string())
    print("Saved", args.out_dir)


if __name__ == "__main__":
    main()
