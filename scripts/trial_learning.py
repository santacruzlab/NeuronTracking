"""Attempt-level learning, within-rotation exposure and repeated-measure models.

Primary analysis excludes disputed rotation labels, keeps failed/censored
attempts in the outcome tables, and models completed durations separately.
The curve basis is flexible regression, not a mechanistic two-rate model.
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import tables
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.optimize import brentq
from statsmodels.stats.multitest import multipletests
from analyze_session_learning import regression_trend
from trial_utils import rewarded_sequence_rows, TIMING_DEFINITION

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots/trial_learning"


def trajectory_metrics(cursor, target, onset, stop, fs=60):
    points = np.asarray(cursor[onset:stop + 1])[:, [0, 2]]
    if len(points) < 2 or not np.all(np.isfinite(points)):
        return dict(path_length=np.nan, straightness=np.nan, initial_error_deg=np.nan)
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    chord = float(np.linalg.norm(points[-1] - points[0]))
    desired = np.asarray(target)[[0, 2]] - points[0]
    displacements = points[1:min(len(points), int(0.2 * fs) + 1)] - points[0]
    moving = np.flatnonzero(np.linalg.norm(displacements, axis=1) >= 0.1)
    angle = np.nan
    if len(moving) and np.linalg.norm(desired) > 0:
        movement = displacements[moving[0]]
        angle = float(np.degrees(np.arctan2(desired[0] * movement[1] - desired[1] * movement[0],
                                           np.dot(desired, movement))))
    return dict(path_length=length, straightness=min(chord / length, 1) if length > 0 else np.nan,
                initial_error_deg=angle)


def extract_attempts(task, msgs, blocks=(1, 2)):
    """Each premove is an attempt; penalties/reward terminate it; others censor."""
    names = [m.decode() if isinstance(m, bytes) else str(m) for m in msgs["msg"]]
    rows = msgs["time"].astype(int)
    n = len(task)
    output, pending, count, reward_count, peripheral_count = [], None, {}, {}, {}

    def finish(index, outcome):
        nonlocal pending
        if pending is None:
            return
        start_index, start = pending
        pending = None
        if not 0 <= start < n:
            return
        block = int(np.asarray(task["block_type"][start]).ravel()[0])
        if block not in blocks:
            return
        stop = min(int(rows[index]), n - 1)
        if stop < start:
            return
        count[block] = count.get(block, 0) + 1
        target_indices = [j for j in range(start_index + 1, index + 1)
                          if names[j] == "target" and 0 <= rows[j] < n
                          and np.linalg.norm(task["target"][rows[j]][[0, 2]]) > 1e-6]
        ti = target_indices[-1] if target_indices else None
        first_target, second_target = rewarded_sequence_rows(names, rows, index) if outcome == "success" else (-1, -1)
        if second_target >= 0:
            # Message timing remains valid even if the position field still
            # records the center on a shared transition/target timestamp.
            ti = index - 3
        onset = int(rows[ti]) if ti is not None else -1
        holds = ([j for j in range(ti + 1, index + 1) if names[j] == "hold"]
                 if ti is not None else [])
        enter = min(int(rows[holds[0]]), n - 1) if holds else -1
        cross_block = np.any(np.asarray(task["block_type"][start:stop + 1]).ravel() != block)
        valid = not cross_block and rows[index] < n and outcome != "censored"
        sequence_valid = (first_target >= start and first_target <= second_target <= stop
                          and second_target == onset) if outcome == "success" else False
        if outcome == "success":
            valid = valid and sequence_valid
        timing_start = first_target if outcome == "success" else start
        if outcome == "success":
            reward_count[block] = reward_count.get(block, 0) + 1
        if ti is not None:
            peripheral_count[block] = peripheral_count.get(block, 0) + 1
        anchor = onset if onset >= 0 else start
        target = task["target"][anchor]
        target_valid = (onset >= 0 and np.all(np.isfinite(target[[0, 2]]))
                        and np.linalg.norm(target[[0, 2]]) > 1e-6)
        angle = (int(np.round(np.degrees(np.arctan2(target[2], target[0])))) % 360
                 if target_valid else np.nan)
        metrics = trajectory_metrics(task["cursor"], target, onset,
                                     enter if enter >= onset else stop) if onset >= 0 else dict(
                                         path_length=np.nan, straightness=np.nan, initial_error_deg=np.nan)
        if not target_valid:
            metrics["initial_error_deg"] = np.nan
        output.append(dict(block=block, attempt=count[block], outcome=outcome, terminal_state=names[index],
                           row_premove=start, row_first_target=first_target, row_target=onset, row_enter=enter, row_stop=stop,
                           sequence_valid=bool(sequence_valid), target_position_valid=bool(target_valid),
                           valid_timing=bool(valid), block_crossing=bool(cross_block),
                           peripheral=onset >= 0, target_deg=angle,
                           peripheral_attempt=peripheral_count.get(block, 0) if onset >= 0 else np.nan,
                           reward_number=reward_count[block] if outcome == "success" else np.nan,
                           duration_s=(stop - timing_start) / 60 if valid else np.nan,
                           observed_time_s=(stop - start) / 60,
                           peripheral_reach_s=(enter - onset) / 60 if valid and enter >= onset >= 0 else np.nan,
                           pre_peripheral_s=(onset - timing_start) / 60 if valid and onset >= 0 else np.nan,
                           error_clamp=int(np.asarray(task["error_clamp"][anchor]).ravel()[0]),
                           perturbation=int(np.asarray(task["pert"][anchor]).ravel()[0]), **metrics))

    for i, name in enumerate(names):
        if name == "premove":
            finish(i, "censored")
            pending = (i, int(rows[i]))
        elif name == "target" and pending is None:
            # The successful seven-event sequence does not require a premove.
            pending = (i, int(rows[i]))
        elif name == "reward":
            finish(i, "success")
        elif name.endswith("_penalty"):
            finish(i, "failure")
        elif name in ["wait", "None"]:
            finish(i, "censored")
    if pending is not None and len(msgs):
        finish(len(msgs) - 1, "censored")
    return output


def read_task(path):
    """Read needed fields in chunks so compressed spike arrays are not retained."""
    with tables.open_file(path, "r") as hdf:
        tbl = hdf.root.task
        fields = ["block_type", "cursor", "target", "error_clamp", "pert"]
        task = np.empty(tbl.nrows, dtype=[(k, tbl.dtype.fields[k][0]) for k in fields])
        for start in range(0, tbl.nrows, 8192):
            chunk = tbl.read(start, min(start + 8192, tbl.nrows))
            for k in fields:
                task[k][start:start + len(chunk)] = chunk[k]
        return task, hdf.root.task_msgs[:]


def assign_phases(settings, source="catalog"):
    frame = settings.copy()
    frame["session_number"] = frame.file.str.split("_").str[1].astype(int)
    frame = frame.sort_values(["monkey", "date", "session_number", "file"])
    angle_col = "catalog_rotation_signed_deg" if source == "catalog" else "rotation_signed_deg"
    records = []
    for monkey, group in frame.groupby("monkey", sort=False):
        previous, phase, exposure, last_date = None, 0, 0, None
        for _, row in group.iterrows():
            angle = float(row[angle_col])
            date = pd.Timestamp(row["date"])
            if angle != previous:
                phase, exposure = phase + 1, 0
            exposure += 1
            records.append(dict(file=row["file"], monkey=monkey, date=row["date"], rotation=angle,
                                phase=f"{monkey}_{source}_{phase}", exposure_number=exposure,
                                gap_days=0 if last_date is None else (date - last_date).days,
                                disputed=not str(row["rotation_matches_catalog"]).lower() == "true"))
            previous, last_date = angle, date
    frame = pd.DataFrame(records)
    frame["phase_size"] = frame.groupby("phase")["file"].transform("count")
    frame["exposure"] = (frame.exposure_number - 1) / np.maximum(frame.phase_size - 1, 1)
    return frame


def gls_projector(X, Z, groups, random_cov, scale):
    """Fixed estimated covariance GLS; also used for conditional power simulations."""
    eig, vec = np.linalg.eigh(random_cov)
    root = vec @ np.diag(np.sqrt(np.maximum(eig, 0)))
    score = np.zeros((X.shape[1], len(X)))
    for group in np.unique(groups):
        ix = np.flatnonzero(groups == group)
        U = Z[ix] @ root
        vinv = (np.eye(len(ix)) - U @ np.linalg.solve(np.eye(U.shape[1]) + U.T @ U / scale, U.T) / scale) / scale
        score[:, ix] = X[ix].T @ vinv
    covariance = np.linalg.pinv(score @ X)
    return covariance @ score, covariance


def half_life(fast, slow):
    """Half-life of a nonnegative fitted basis; unavailable for invalid shapes."""
    if fast < -1e-8 or slow < -1e-8 or fast + slow <= 0.05:
        return np.nan
    fast, slow = max(fast, 0), max(slow, 0)
    return brentq(lambda t: fast * np.exp(-t / 8) + slow * np.exp(-t / 32) - (fast + slow) / 2, 0, 128)


def model_data(attempts, phases, outcome, include_disputed=False):
    data = attempts.merge(phases, on=["file", "monkey", "date"], validate="many_to_one")
    keep = ((data.block == 2) & data.peripheral & data.valid_timing & (data.outcome == "success")
            & data.target_deg.notna() & (data.error_clamp == 0)
            & (data.peripheral_attempt <= 128) & (data[outcome] > 0))
    if not include_disputed:
        keep &= ~data.disputed
    data = data.loc[keep].copy()
    data["log_y"] = np.log(data[outcome])
    data["t"] = data.peripheral_attempt - 1
    data["fast"], data["slow"] = np.exp(-data.t / 8), np.exp(-data.t / 32)
    data["target_deg"] = data.target_deg.astype(int).astype(str)
    data["gap_z"] = (data.gap_days - data.gap_days.mean()) / max(data.gap_days.std(), 1)
    return data


def fit_mixed(data, monkey, outcome, variant, out_dir, export_context=False):
    formula = "log_y ~ C(target_deg) + C(phase) + gap_z + exposure * (fast + slow)"
    if variant == "phase_specific_shape":
        formula = "log_y ~ C(target_deg) + C(phase) * (fast + slow) + gap_z + exposure + exposure:fast + exposure:slow"
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        model = smf.mixedlm(formula, data, groups=data.file, re_formula="1 + fast")
        result = model.fit(reml=True, method=["lbfgs", "bfgs"], maxiter=500, disp=False)
    names = list(result.fe_params.index)
    covariance = result.cov_params().loc[names, names].to_numpy()
    beta = result.fe_params.to_numpy()
    fast_weights, slow_weights = np.zeros(len(beta)), np.zeros(len(beta))
    fast_weights[names.index("fast")], slow_weights[names.index("slow")] = 1, 1
    for i, name in enumerate(names):
        if "C(phase)" in name and (":fast" in name or name.startswith("fast:")):
            fast_weights[i] = 1 / data.phase.nunique()
        if "C(phase)" in name and (":slow" in name or name.startswith("slow:")):
            slow_weights[i] = 1 / data.phase.nunique()
    first_fast, first_slow = fast_weights @ beta, slow_weights @ beta
    coefficients = [dict(monkey=monkey, outcome=outcome, variant=variant, term=name,
                         estimate=beta[i], se=np.sqrt(max(covariance[i, i], 0)),
                         converged=bool(result.converged)) for i, name in enumerate(names)]
    ix = [names.index("exposure:fast"), names.index("exposure:slow")]
    joint = beta[ix] @ np.linalg.pinv(covariance[np.ix_(ix, ix)]) @ beta[ix]
    from scipy.stats import chi2
    record = dict(monkey=monkey, outcome=outcome, variant=variant, sessions=data.file.nunique(),
                  trials=len(data), converged=bool(result.converged), joint_curve_p=float(chi2.sf(joint, 2)),
                  initial_excess_first=float(first_fast + first_slow),
                  initial_excess_last=float(first_fast + first_slow + beta[ix].sum()),
                  warning_count=len(notices), warnings=" | ".join(sorted({str(w.message) for w in notices})))
    record["half_life_first_trials"] = half_life(first_fast, first_slow)
    record["half_life_last_trials"] = half_life(first_fast + beta[ix[0]], first_slow + beta[ix[1]])
    gap_change = beta[ix].sum()
    gap_se = np.sqrt(max(covariance[np.ix_(ix, ix)].sum(), 0))
    record.update(initial_excess_change=gap_change, initial_excess_change_ci_lo=gap_change - 1.96 * gap_se,
                  initial_excess_change_ci_hi=gap_change + 1.96 * gap_se)
    rng = np.random.default_rng(20261013)
    eigen, vectors = np.linalg.eigh(covariance)
    draws = beta + rng.normal(size=(2000, len(beta))) @ (vectors @ np.diag(np.sqrt(np.maximum(eigen, 0)))).T
    f, s = names.index("fast"), names.index("slow")
    draw_fast, draw_slow = draws @ fast_weights, draws @ slow_weights
    changes = np.array([half_life(a + b[ix[0]], c + b[ix[1]]) - half_life(a, c)
                        for a, c, b in zip(draw_fast, draw_slow, draws)])
    available = np.isfinite(changes)
    record["shape_draw_valid_fraction"] = float(available.mean())
    lo, hi = np.percentile(changes[available], [2.5, 97.5]) if available.sum() >= 100 else (np.nan, np.nan)
    record.update(half_life_change_ci_lo=lo, half_life_change_ci_hi=hi,
                  half_life_interval_conditional_on_valid_shape=True)
    grid_rows = []
    for exposure in [0, 1]:
        for trial in range(1, 129):
            grid_rows.append(dict(exposure=exposure, t=trial - 1))
    grid = pd.DataFrame(grid_rows)
    for name in ["target_deg", "phase"]:
        grid[name] = data[name].iloc[0]
    grid["gap_z"] = 0
    grid["fast"], grid["slow"] = np.exp(-grid.t / 8), np.exp(-grid.t / 32)
    # Curves above their own fitted baseline isolate starting gap and shape.
    for exposure, values in grid.groupby("exposure"):
        excess = (first_fast + exposure * beta[ix[0]]) * values.fast
        excess += (first_slow + exposure * beta[ix[1]]) * values.slow
        grid.loc[values.index, "relative_typical_duration"] = np.exp(excess)
        bands = (draw_fast[:, None] + exposure * draws[:, ix[0], None]) * values.fast.to_numpy()
        bands += (draw_slow[:, None] + exposure * draws[:, ix[1], None]) * values.slow.to_numpy()
        bounds = np.percentile(np.exp(bands), [2.5, 97.5], axis=0)
        grid.loc[values.index, "ci_lo"] = bounds[0]
        grid.loc[values.index, "ci_hi"] = bounds[1]
    grid["monkey"], grid["outcome"], grid["variant"] = monkey, outcome, variant
    if export_context and result.converged:
        X, Z = result.model.exog, result.model.exog_re
        groups = pd.factorize(data.file, sort=False)[0]
        projector, fixed_cov = gls_projector(X, Z, groups, result.cov_re.to_numpy(), result.scale)
        np.savez_compressed(out_dir / f"simulation_context_{monkey}.npz", X=X, Z=Z, groups=groups,
                            beta=beta, names=np.asarray(names), residual=result.resid.to_numpy(),
                            random_cov=result.cov_re.to_numpy(), scale=result.scale,
                            exposure=data.exposure.to_numpy(), t=data.t.to_numpy(),
                            projector=projector, fixed_cov=fixed_cov)
    return record, coefficients, grid


def session_summaries(attempts, phases):
    records = []
    for _, p in phases.iterrows():
        d = attempts[attempts.file == p.file]
        b2 = d[(d.block == 2) & d.valid_timing & (d.error_clamp == 0)]
        per = b2[b2.peripheral]
        first_attempts = per[per.peripheral_attempt <= 128]
        successes = b2[b2.outcome == "success"]
        early = successes[successes.reward_number <= 16]
        late = successes[(successes.reward_number >= 193) & (successes.reward_number <= 320)]
        baseline = d[(d.block == 1) & d.valid_timing & (d.outcome == "success") & (d.error_clamp == 0)].tail(128)
        rec = dict(p)
        rec.update(attempts=len(b2), failures=int((b2.outcome == "failure").sum()),
                   censored=int(((d.block == 2) & (d.outcome == "censored")).sum()),
                   center_failures=int(((b2.outcome == "failure") & ~b2.peripheral).sum()),
                   success_fraction_first128=float((first_attempts.outcome == "success").mean()),
                   n_first128_attempts=len(first_attempts), n_fixed_late=len(late))
        for key in ["duration_s", "peripheral_reach_s", "pre_peripheral_s", "straightness", "path_length",
                    "initial_error_abs_deg"]:
            def balanced_mean(frame):
                grouped = frame.groupby("target_deg")[key].mean().dropna()
                return float(grouped.mean()) if len(grouped) == 8 else np.nan
            rec["early_" + key] = balanced_mean(early) if len(early) == 16 else np.nan
            rec["late_" + key] = balanced_mean(late) if len(late) >= 100 else np.nan
            rec["baseline_" + key] = balanced_mean(baseline) if len(baseline) >= 64 else np.nan
        rec["early_to_baseline_duration"] = rec["early_duration_s"] / rec["baseline_duration_s"]
        rec["early_to_fixed_late_duration"] = rec["early_duration_s"] / rec["late_duration_s"]
        if len(early) == 16:
            end = early.row_stop.max()
            preceding = b2[b2.row_premove <= end]
            rec["elapsed_s_to_16_rewards"] = (end - preceding.row_premove.min()) / 60
            rec["active_attempt_s_to_16_rewards"] = preceding.observed_time_s.sum()
        records.append(rec)
    return pd.DataFrame(records)


def summary_checks(summary, out_dir):
    metrics = ["early_duration_s", "early_peripheral_reach_s", "early_pre_peripheral_s",
               "early_straightness", "early_initial_error_abs_deg", "early_to_baseline_duration",
               "early_to_fixed_late_duration", "success_fraction_first128",
               "elapsed_s_to_16_rewards", "active_attempt_s_to_16_rewards"]
    rows = []
    rng = np.random.default_rng(20261013)
    for monkey in ["airp", "braz"]:
        eligible = summary[(summary.monkey == monkey) & ~summary.disputed]
        for metric in metrics:
            d = eligible[np.isfinite(eligible[metric])]
            if len(d) < 10 or d.exposure.nunique() < 2:
                result = dict(status="insufficient_data", n=len(d))
            else:
                result = regression_trend(d.exposure.to_numpy(), d[metric].to_numpy(),
                                          d.phase.tolist(), True, rng)
            rows.append(dict(monkey=monkey, metric=metric, comparison="Within-phase exposure, phase fixed effects",
                             **result))
    pd.DataFrame(rows).to_csv(out_dir / "summary_sensitivity_trends.csv", index=False)
    displayed = [("early_duration_s", "Early whole-trial duration (s)"),
                 ("early_peripheral_reach_s", "Early peripheral reach (s)"),
                 ("early_straightness", "Early path straightness (1 = straight)"),
                 ("early_initial_error_abs_deg", "Early initial direction error (degrees)")]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), squeeze=False)
    for i, monkey in enumerate(["airp", "braz"]):
        d = summary[(summary.monkey == monkey) & ~summary.disputed]
        for j, (metric, label) in enumerate(displayed):
            for phase, group in d.groupby("phase", sort=False):
                axes[i, j].plot(group.exposure_number, group[metric], "o-", ms=3, lw=0.8,
                                label=f"{group.rotation.iloc[0]:+g} deg")
            axes[i, j].set(title=monkey, ylabel=label, xlabel="Exposure number within rotation phase")
            axes[i, j].grid(alpha=0.2)
            if i == 0 and j == 0:
                axes[i, j].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "phase_performance_checks.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-attempts", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import scipy, statsmodels
    with (args.out_dir / "analysis_plan.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(primary="Within-rotation exposure change, analyzed separately per animal",
                       cohort="Exclude disputed rotation in primary; include both labels as sensitivity",
                       trial_window="First 128 peripheral attempts, including failed attempts in exposure count",
                       primary_outcomes=["First-target-to-reward completed duration", "Completed peripheral reach duration"],
                       timing_definition=TIMING_DEFINITION,
                       basis_trial_constants=[8, 32], random_effects="Session intercept and fast-basis slope",
                       fixed_effects="Target, phase, gap days, exposure, curve basis and exposure interactions",
                       sensitivity_models=["Phase-specific curve shapes", "Disputed rotation included under catalog and HDF labels"],
                       uncertainty="Mixed-model population intervals; independent session random effects assumed",
                       correction="Holm correction for four primary joint-curve tests; other checks exploratory",
                       early_window="First 16 rewards; target-stratified means", late_window="Rewards 193-320",
                       baseline="Last 128 unassisted, non-clamped successful block 1 trials",
                       warnings=["Not a mechanistic two-rate learning model", "Not prospectively registered",
                                 "Half-life intervals condition on valid positive curve shapes",
                                 "Decoder identity remains unknown", "Reward dwell lies after timing endpoint"],
                       versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                                     statsmodels=statsmodels.__version__)), handle, indent=2)
    settings = pd.read_csv(ROOT / "plots/within_session_learning/task_settings.csv")
    phases = assign_phases(settings)
    hdf_phases = assign_phases(settings, "hdf")
    phases.to_csv(args.out_dir / "phases_catalog.csv", index=False)
    hdf_phases.to_csv(args.out_dir / "phases_hdf.csv", index=False)
    destination = args.out_dir / "attempts.csv"
    if args.reuse_attempts:
        attempts = pd.read_csv(destination)
    else:
        rows = []
        for i, s in settings.iterrows():
            task, msgs = read_task(ROOT / "data/HDF" / s.file)
            found = extract_attempts(task, msgs)
            for row in found:
                row.update(file=s.file, monkey=s.monkey, date=s.date)
            rows.extend(found)
            print(f"Extract {i + 1}/{len(settings)} {s.file}: {len(found)} attempts", flush=True)
        attempts = pd.DataFrame(rows)
        attempts.to_csv(destination, index=False)
    attempts["initial_error_abs_deg"] = attempts.initial_error_deg.abs()
    attempts.to_csv(destination, index=False)
    summary = session_summaries(attempts, phases)
    summary.to_csv(args.out_dir / "trial_session_summary.csv", index=False)
    summary_checks(summary, args.out_dir)
    previous = pd.read_csv(ROOT / "plots/premove_session_sets/trial_times.csv")
    previous = previous[previous.block == 2]
    matched = previous.merge(attempts[(attempts.block == 2) & (attempts.outcome == "success")],
                             left_on=["file", "row_reward"], right_on=["file", "row_stop"],
                             how="left", suffixes=("_old", "_new"), validate="one_to_one")
    finite = np.isfinite(matched.duration_s_old) & np.isfinite(matched.duration_s_new)
    validity_mismatches = int((np.isfinite(matched.duration_s_old) != np.isfinite(matched.duration_s_new)).sum())
    mismatches = int((np.abs(matched.loc[finite, "duration_s_old"] - matched.loc[finite, "duration_s_new"]) > 1e-10).sum())
    pd.DataFrame([dict(prior_rewards=len(previous), matched_rewards=int(matched.outcome.notna().sum()),
                       compared_valid_durations=int(finite.sum()), duration_mismatches=mismatches,
                       validity_mismatches=validity_mismatches)]).to_csv(
                           args.out_dir / "extraction_qc.csv", index=False)
    if mismatches or validity_mismatches or matched.outcome.isna().any():
        raise ValueError("Attempt extraction disagrees with prior rewarded-trial timing")
    records, coefficients, grids = [], [], []
    for monkey in ["airp", "braz"]:
        for outcome in ["duration_s", "peripheral_reach_s"]:
            variants = [("primary", phases, False), ("phase_specific_shape", phases, False)]
            if monkey == "braz" and outcome == "peripheral_reach_s":
                variants += [("include_disputed_catalog", phases, True), ("include_disputed_hdf", hdf_phases, True)]
            for variant, phase_data, include in variants:
                data = model_data(attempts[attempts.monkey == monkey], phase_data, outcome, include)
                print(f"Model {monkey} {outcome} {variant}: {len(data)} trials", flush=True)
                try:
                    record, coeff, grid = fit_mixed(data, monkey, outcome, variant, args.out_dir,
                                                   export_context=outcome == "peripheral_reach_s" and variant == "primary")
                    records.append(record)
                    coefficients.extend(coeff)
                    grids.append(grid)
                except (ValueError, np.linalg.LinAlgError) as err:
                    records.append(dict(monkey=monkey, outcome=outcome, variant=variant, converged=False, error=str(err)))
    primary = [r for r in records if r["variant"] == "primary" and r.get("converged") and np.isfinite(r.get("joint_curve_p", np.nan))]
    corrected = multipletests([r["joint_curve_p"] for r in primary], method="holm")[1] if primary else []
    for r, p in zip(primary, corrected):
        r["joint_curve_p_holm"] = p
    pd.DataFrame(records).to_csv(args.out_dir / "mixed_model_summary.csv", index=False)
    pd.DataFrame(coefficients).to_csv(args.out_dir / "mixed_model_coefficients.csv", index=False)
    if grids:
        curves = pd.concat(grids)
        curves.to_csv(args.out_dir / "predicted_curves.csv", index=False)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
        for i, monkey in enumerate(["airp", "braz"]):
            for j, outcome in enumerate(["duration_s", "peripheral_reach_s"]):
                d = curves[(curves.monkey == monkey) & (curves.outcome == outcome) & (curves.variant == "primary")]
                for e, g in d.groupby("exposure"):
                    line, = axes[i, j].plot(g.t + 1, g.relative_typical_duration,
                                           label="First within-phase exposure" if e == 0 else "Last within-phase exposure")
                    axes[i, j].fill_between(g.t + 1, g.ci_lo, g.ci_hi, color=line.get_color(), alpha=0.15)
                axes[i, j].axhline(1, color="gray", ls="--")
                label = "First target to reward" if outcome == "duration_s" else "Peripheral reach"
                axes[i, j].set(title=f"{monkey}: {label}", xlabel="Peripheral attempt in block",
                               ylabel="Typical duration / fitted long-trial baseline")
                axes[i, j].legend(fontsize=8)
                axes[i, j].grid(alpha=0.2)
        fig.suptitle("Target-adjusted typical curves; shaded = model-based population 95% intervals")
        fig.tight_layout()
        fig.savefig(args.out_dir / "within_phase_learning_curves.png", dpi=180)
        plt.close(fig)
    # Failure trends retain unsuccessful attempts; do not impute their reach times.
    failure_rows = []
    for monkey in ["airp", "braz"]:
        d = attempts.merge(phases, on=["file", "monkey", "date"])
        d = d[(d.monkey == monkey) & (d.block == 2) & d.peripheral & d.valid_timing
              & (d.peripheral_attempt <= 128) & ~d.disputed].copy()
        d["success"] = (d.outcome == "success").astype(int)
        model = smf.glm("success ~ C(phase) + exposure + np.log1p(peripheral_attempt)", d,
                        family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d.file})
        ci = model.conf_int().loc["exposure"]
        failure_rows.append(dict(monkey=monkey, sessions=d.file.nunique(), attempts=len(d),
                                 exposure_log_odds=model.params["exposure"], ci_lo=ci.iloc[0], ci_hi=ci.iloc[1]))
    pd.DataFrame(failure_rows).to_csv(args.out_dir / "success_trends.csv", index=False)
    for r in records:
        print(json.dumps(r))
    print("Saved to", args.out_dir)


if __name__ == "__main__":
    main()
