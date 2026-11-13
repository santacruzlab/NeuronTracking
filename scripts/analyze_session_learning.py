"""Block 2: per-session learning fits, task-setting audit and longitudinal trends.

Fit the same first 16 sets in each session; late reference uses valid sets 25+.
Missing sets keep their original positions. Tau is reported only after explicit
signal, fit, resolution, robust-fit and bootstrap precision checks. Comparisons
across sessions are conditional on this selection. Task-regime fixed effects
are exploratory because task settings and chronology are strongly associated.
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import tables
from scipy.optimize import least_squares

from sessions import AIRPORT_SESSIONS, AIRPORT_ROTATION, BRAZOS_SESSIONS, BRAZOS_ROTATION
from trial_utils import TIMING_DEFINITION

ROOT = Path(__file__).resolve().parents[1]
CATALOG = dict(zip(AIRPORT_SESSIONS + BRAZOS_SESSIONS, AIRPORT_ROTATION + BRAZOS_ROTATION))
SETTINGS = ["rot_angle_deg", "hold_time", "premove_time", "target_radius", "timeout_time",
            "reward_time", "assist_level", "assist_level_time", "window_size"]
TAU_BOUNDS = (0.25, 32.0)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def signed_angle(value):
    return float((float(value) + 180) % 360 - 180)


def audit_settings(summaries, out_dir, reuse=False):
    destination = out_dir / "task_settings.csv"
    if reuse and destination.exists():
        return {r["file"]: r for r in read_csv(destination)}
    rows = []
    for s in summaries:
        rec = dict(file=s["file"], monkey=s["monkey"], session=s["session"], date=s["date"])
        with tables.open_file(ROOT / "data/HDF" / s["file"], "r") as hdf:
            tbl = hdf.root.task
            available = tbl._v_attrs._f_list()
            for key in SETTINGS:
                value = getattr(tbl._v_attrs, key) if key in available else None
                rec[key] = json.dumps(np.asarray(value).tolist()) if value is not None else "unknown"
            rec["rotation_signed_deg"] = (signed_angle(float(rec["rot_angle_deg"]))
                                            if rec["rot_angle_deg"] != "unknown" else np.nan)
            expected = CATALOG.get(Path(s["file"]).stem)
            rec["catalog_rotation_signed_deg"] = signed_angle(expected) if expected is not None else np.nan
            rec["rotation_matches_catalog"] = bool(rec["rotation_signed_deg"] == rec["catalog_rotation_signed_deg"])
            blocks = tbl.read(field="block_type").ravel()
            assist = tbl.read(field="assist_level").reshape(len(blocks), -1)[blocks == 2]
            rec["block2_assist_min"] = float(assist.min()) if assist.size else np.nan
            rec["block2_assist_max"] = float(assist.max()) if assist.size else np.nan
            refs = {}
            for node in [hdf.root, tbl]:
                for key in node._v_attrs._f_list():
                    if "decoder" in key.lower():
                        refs[node._v_pathname + ":" + key] = str(getattr(node._v_attrs, key))
            rec["decoder_reference"] = json.dumps(refs, sort_keys=True) if refs else "unknown"
        # Do not treat decoder_state (the changing control state) as decoder identity.
        rec["regime"] = json.dumps({k: rec[k] for k in ["rotation_signed_deg", "hold_time", "premove_time",
                                                     "target_radius", "timeout_time", "reward_time",
                                                     "block2_assist_min", "block2_assist_max"]}, sort_keys=True)
        rows.append(rec)
        print(f"Settings {len(rows)}/{len(summaries)}: {s['file']} rotation={rec['rotation_signed_deg']}", flush=True)
    save_csv(destination, rows)
    return {r["file"]: r for r in rows}


def decay(p, x):
    return p[0] + p[1] * np.exp(-np.asarray(x) / p[2])


def fit_decay(x, y, robust=False, start=None):
    a = max(float(np.median(y[-4:])), 0.1)
    b = max(float(np.mean(y[:2]) - a), 0.05)
    upper = max(float(y.max()) * 10, 5)
    starts = [start] if start is not None else [[a, b, tau] for tau in [1, 4, 12]]
    best = None
    for initial in starts:
        fitted = least_squares(lambda p: decay(p, x) - y, initial,
                               bounds=([0.01, 0, TAU_BOUNDS[0]], [upper, upper, TAU_BOUNDS[1]]),
                               loss="soft_l1" if robust else "linear", f_scale=0.1, max_nfev=800)
        if best is None or fitted.cost < best.cost:
            best = fitted
    return best.x


def aicc(sse, n, parameters):
    # Include the estimated residual variance as a parameter.
    k = parameters + 1
    return n * np.log(max(sse / n, 1e-15)) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def assess_curve(x, y, rng, draws=300):
    """Normalized set means; x retains original set positions (set 1 = 0)."""
    p = fit_decay(x, y)
    robust = fit_decay(x, y, robust=True)
    fitted = decay(p, x)
    residual = y - fitted
    noise = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    improvement = aicc(np.sum((y - y.mean()) ** 2), len(y), 1) - aicc(np.sum(residual ** 2), len(y), 3)
    flags = []
    if p[1] <= max(0.05, 2 * noise):
        flags.append("weak_amplitude")
    if improvement < 4:
        flags.append("flat_model_competitive")
    if p[2] <= 0.5 or p[2] >= 31.5:
        flags.append("tau_at_resolution_or_bound")
    if max(p[2] / robust[2], robust[2] / p[2]) > 2:
        flags.append("outlier_sensitive")
    lo, hi, amplitude_lo = np.nan, np.nan, np.nan
    boot = []
    if not flags and draws:
        centered = residual - residual.mean()
        for _ in range(draws):
            starts = rng.integers(0, len(y), size=int(np.ceil(len(y) / 2)))
            ix = ((starts[:, None] + np.arange(2)) % len(y)).ravel()[:len(y)]
            by = np.maximum(fitted + centered[ix], 0.01)
            boot.append(fit_decay(x, by, start=p))
        boot = np.asarray(boot)
        lo, hi = np.percentile(boot[:, 2], [2.5, 97.5])
        amplitude_lo = np.percentile(boot[:, 1], 2.5)
        if lo <= 0.5 or hi >= 31.5 or hi / lo > 4:
            flags.append("imprecise_tau")
        if amplitude_lo <= 0.05:
            flags.append("uncertain_amplitude")
    accepted = not flags and draws > 0
    return dict(status="estimable" if accepted else "|".join(flags) or "bootstrap_not_run",
                fitted_a=p[0], fitted_b=p[1], tau_candidate_sets=p[2], robust_tau_sets=robust[2],
                residual_mad=noise, delta_aicc_flat_minus_decay=improvement,
                tau_ci_lo_sets=lo, tau_ci_hi_sets=hi, amplitude_ci_lo=amplitude_lo,
                tau_sets=float(p[2]) if accepted else np.nan,
                tau_trials=float(8 * p[2]) if accepted else np.nan,
                rate_per_set=float(1 / p[2]) if accepted else np.nan)


def regression_trend(x, y, regimes, adjusted, rng):
    span = np.ptp(x)
    z = (x - x.min()) / span
    columns = [np.ones(len(x)), z]
    if adjusted:
        levels = sorted(set(regimes))
        columns.extend(np.asarray(regimes) == level for level in levels[1:])
    design = np.column_stack(columns).astype(float)
    if len(x) < 2 * design.shape[1] + 5 or np.linalg.matrix_rank(design) < design.shape[1]:
        return dict(status="insufficient_data", n=len(x), n_parameters=design.shape[1])
    inverse = np.linalg.pinv(design)
    beta = inverse @ y
    residual = y - design @ beta
    residual -= residual.mean()
    residual *= np.sqrt(len(y) / (len(y) - design.shape[1]))
    intervals = []
    for size in [4, 6]:
        starts = rng.integers(0, len(y), size=(5000, int(np.ceil(len(y) / size))))
        ix = ((starts[:, :, None] + np.arange(size)) % len(y)).reshape(5000, -1)[:, :len(y)]
        boot = (inverse @ ((design @ beta)[None, :] + residual[ix]).T).T
        intervals.append(np.percentile(boot[:, 1], [2.5, 97.5]) / span)
    return dict(status="ok", n=len(y), n_parameters=design.shape[1], slope=beta[1] / span,
                slope_ci_lo=intervals[0][0], slope_ci_hi=intervals[0][1],
                slope_ci_lo_block6=intervals[1][0], slope_ci_hi_block6=intervals[1][1],
                change_over_span=beta[1], design_condition_number=np.linalg.cond(design))


def trend_plots(records, out_dir, rng):
    metrics = [("early_mean_s", "Early duration (s)"), ("learning_amount_fraction", "Early excess / late"),
               ("tau_sets", "Learning tau (sets; smaller = faster)"), ("late_mean_s", "Late duration (s)")]
    output = []
    for spacing in ["even", "calendar"]:
        fig, axes = plt.subplots(2, 4, figsize=(18, 9), squeeze=False)
        for r, monkey in enumerate(["airp", "braz"]):
            all_recs = [d for d in records if d["monkey"] == monkey]
            dates = np.array([mdates.date2num(datetime.strptime(d["date"], "%Y-%m-%d")) for d in all_recs])
            xpos = np.arange(len(all_recs), dtype=float) if spacing == "even" else dates - dates.min()
            for c, (key, label) in enumerate(metrics):
                ax = axes[r, c]
                values = np.array([d.get(key, np.nan) for d in all_recs], float)
                valid = np.isfinite(values)
                x, y = xpos[valid], values[valid]
                regimes = [d["regime"] for d, ok in zip(all_recs, valid) if ok]
                # Log tau for fitting positive rates spanning different scales.
                fitted_y = np.log(y) if key == "tau_sets" else y
                for angle, color in zip([-90, -50, 50, 90], ["C0", "C1", "C2", "C3"]):
                    mask = valid & np.array([d["rotation_signed_deg"] == angle for d in all_recs])
                    ax.scatter((xpos[mask] + 1 if spacing == "even" else dates[mask]), values[mask],
                               color=color, s=22, label=f"{angle:+d} deg")
                for adjustment in ["none", "rotation_hdf", "rotation_catalog", "full_task_regime"]:
                    adjusted = adjustment != "none"
                    if adjustment == "rotation_hdf":
                        groups = [str(d["rotation_signed_deg"]) for d, ok in zip(all_recs, valid) if ok]
                    elif adjustment == "rotation_catalog":
                        groups = [str(d["catalog_rotation_signed_deg"]) for d, ok in zip(all_recs, valid) if ok]
                    else:
                        groups = regimes
                    result = (regression_trend(x, fitted_y, groups, adjusted, rng)
                              if len(y) >= 10 and np.ptp(x) > 0 else dict(status="insufficient_data", n=len(y)))
                    output.append(dict(monkey=monkey, spacing=spacing, metric=key,
                                       transform="log" if key == "tau_sets" else "identity",
                                       adjustment=adjustment, **result))
                    if not adjusted and result["status"] == "ok":
                        grid = np.linspace(x.min(), x.max(), 100)
                        coefficient = np.polyfit(x, fitted_y, 1)
                        line = np.polyval(coefficient, grid)
                        ax.plot(grid + 1 if spacing == "even" else dates.min() + grid,
                                np.exp(line) if key == "tau_sets" else line, color="k", lw=1.3,
                                label="Unadjusted trend")
                ax.set(title=f"{monkey}: n={len(y)}", ylabel=label,
                       xlabel="Recorded session" if spacing == "even" else "Session date")
                if spacing == "calendar":
                    loc = mdates.AutoDateLocator(minticks=3, maxticks=5)
                    ax.xaxis.set_major_locator(loc)
                    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
                if key == "learning_amount_fraction":
                    ax.axhline(0, color="gray", ls="--", lw=0.7)
                if key == "tau_sets":
                    ax.set_yscale("log")
                    # Show rejected candidate rates as crosses, keeping the
                    # selective availability of reliable rates visible.
                    rejected = np.array([not np.isfinite(d.get("tau_sets", np.nan))
                                         and np.isfinite(d.get("tau_candidate_sets", np.nan)) for d in all_recs])
                    candidates = np.array([d.get("tau_candidate_sets", np.nan) for d in all_recs])
                    ax.scatter(xpos[rejected] + 1 if spacing == "even" else dates[rejected],
                               candidates[rejected], marker="x", color="gray", s=16, alpha=0.4,
                               label="Rejected candidate (not in trend)")
                    ax.set_title(f"{monkey}: {len(y)} reliable rates / {len(all_recs)} sessions")
                    ax.legend(fontsize=6)
                ax.grid(alpha=0.25)
                if r == 0 and c == 0:
                    ax.legend(fontsize=7)
        fig.suptitle("Block 2: learning metrics across sessions; colors show rotation settings")
        fig.tight_layout()
        fig.savefig(out_dir / f"learning_metrics_{spacing}.png", dpi=180)
        plt.close(fig)
    save_csv(out_dir / "learning_trends.csv", output)
    return output


def fit_check_plots(records, curves, out_dir, fit_sets):
    for monkey in ["airp", "braz"]:
        recs = [d for d in records if d["monkey"] == monkey]
        for page, start in enumerate(range(0, len(recs), 12), 1):
            fig, axes = plt.subplots(3, 4, figsize=(16, 10), squeeze=False)
            for ax, rec in zip(axes.ravel(), recs[start:start + 12]):
                means = curves[rec["file"]]
                x = np.arange(1, len(means) + 1)
                ax.plot(x, means, "o-", ms=2, lw=0.6, color="gray")
                if np.isfinite(rec.get("fitted_a", np.nan)):
                    gx = np.linspace(1, fit_sets, 100)
                    p = [rec["fitted_a"], rec["fitted_b"], rec["tau_candidate_sets"]]
                    ax.plot(gx, decay(p, gx - 1) * rec["late_mean_s"], color="C0")
                if np.isfinite(rec["late_mean_s"]):
                    ax.axhline(rec["late_mean_s"], color="C2", ls="--", lw=0.7)
                ax.axvspan(0.5, 2.5, color="orange", alpha=0.15)
                ax.axvline(fit_sets + 0.5, color="gray", ls=":")
                ax.set(title=f"{rec['session']} ({rec['rotation_signed_deg']:+g} deg)\n{rec['status']}",
                       xlabel="Trial set", ylabel="Duration (s)")
                ax.title.set_fontsize(8)
            for ax in axes.ravel()[len(recs[start:start + 12]):]:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / f"{monkey}_fit_checks_{page}.png", dpi=160)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-sets", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--reuse-settings", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "plots/within_session_learning")
    args = parser.parse_args()
    if not 12 <= args.fit_sets < 25 or args.bootstrap < 100:
        parser.error("Use 12-24 fit sets and at least 100 bootstrap draws")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "analysis_settings.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(fit_sets=args.fit_sets, sensitivity_fit_sets=[12, 24],
                       timing_definition=TIMING_DEFINITION,
                       early_sets=2, late_start_set=25, min_valid_late_sets=8,
                       bootstrap_draws=args.bootstrap, within_session_residual_block_length=2,
                       tau_bounds_sets=list(TAU_BOUNDS), seed=20261013,
                       rate_acceptance=dict(min_valid_fit_sets=12, require_first_two_sets=True,
                                            min_amplitude_fraction=0.05, min_amplitude_residual_mad_multiple=2,
                                            min_delta_aicc=4, tau_resolution_floor_sets=0.5,
                                            max_robust_tau_disagreement_factor=2, max_tau_ci_ratio=4),
                       trend_bootstrap_draws=5000, trend_residual_block_lengths=[4, 6],
                       limitations=["Rate trends condition on passing the fit checks",
                                    "Task settings and chronology are associated",
                                    "Decoder identity unavailable from audited HDF attributes",
                                    "Exploratory intervals; no multiplicity correction"]), handle, indent=2)
    summaries = [r for r in read_csv(ROOT / "plots/premove_session_sets/session_summary.csv") if r["block"] == "2"]
    summaries.sort(key=lambda r: (r["monkey"], r["date"], int(r["file"].split("_")[1])))
    settings = audit_settings(summaries, args.out_dir, args.reuse_settings)
    set_rows = [r for r in read_csv(ROOT / "plots/premove_session_sets/set_means.csv") if r["block"] == "2"]
    curves = {}
    for s in summaries:
        data = sorted([r for r in set_rows if r["file"] == s["file"]], key=lambda r: int(r["set_number"]))
        curves[s["file"]] = np.array([float(r["mean_s"]) for r in data])
    records = []
    rng = np.random.default_rng(20261013)
    for s in summaries:
        def value(key):
            return float(s[key]) if s.get(key) else np.nan
        cfg = settings[s["file"]]
        rec = dict(file=s["file"], monkey=s["monkey"], session=s["session"], date=s["date"],
                   fit_sets=args.fit_sets, bootstrap_draws=args.bootstrap,
                   early_mean_s=value("early_mean_s"), late_mean_s=value("late_mean_s"),
                   rotation_signed_deg=float(cfg["rotation_signed_deg"]), regime=cfg["regime"])
        rec["catalog_rotation_signed_deg"] = float(cfg["catalog_rotation_signed_deg"])
        rec["learning_amount_s"] = rec["early_mean_s"] - rec["late_mean_s"]
        rec["learning_amount_fraction"] = (rec["learning_amount_s"] / rec["late_mean_s"]
                                           if rec["late_mean_s"] > 0 else np.nan)
        means = curves[s["file"]]
        first = means[:args.fit_sets]
        valid = np.isfinite(first)
        rec["n_valid_fit_sets"] = int(valid.sum())
        if len(first) < args.fit_sets or valid.sum() < 12 or not np.all(np.isfinite(first[:2])):
            rec["status"] = "insufficient_or_missing_early_sets"
        elif not rec["late_mean_s"] > 0:
            rec["status"] = "missing_late_reference"
        else:
            x, y = np.arange(args.fit_sets)[valid], first[valid] / rec["late_mean_s"]
            rec.update(assess_curve(x, y, rng, args.bootstrap))
            # Common-window sensitivity; these candidates are diagnostic, not accepted rates.
            for size in [12, 24]:
                sample = means[:size]
                ok = np.isfinite(sample)
                if len(sample) == size and ok.sum() >= 10:
                    p = fit_decay(np.arange(size)[ok], sample[ok] / rec["late_mean_s"])
                    rec[f"tau_candidate_first{size}"] = p[2]
        rec.setdefault("tau_sets", np.nan)
        records.append(rec)
        print(f"Fit {len(records)}/{len(summaries)} {s['file']}: {rec['status']}", flush=True)
    save_csv(args.out_dir / "session_learning_metrics.csv", records)
    fit_check_plots(records, curves, args.out_dir, args.fit_sets)
    trends = trend_plots(records, args.out_dir, rng)
    for monkey in ["airp", "braz"]:
        rr = [r for r in records if r["monkey"] == monkey]
        print(f"{monkey}: {sum(r['status'] == 'estimable' for r in rr)}/{len(rr)} rates estimable")
    for t in trends:
        if t["metric"] in ["tau_sets", "learning_amount_fraction"]:
            if t["adjustment"] in ["none", "rotation_hdf", "rotation_catalog"]:
                print(json.dumps(t))
    print("Saved to", args.out_dir)


if __name__ == "__main__":
    main()
