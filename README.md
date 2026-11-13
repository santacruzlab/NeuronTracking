# The tracking algorithm for traditional microelectrode arrays (trackMEA)

## Setup
`pip install -e src`

## Current workflow

```powershell
py scripts/analyzeHDF.py
py scripts/fit_session_trends.py
py scripts/analyze_session_learning.py
py scripts/trial_learning.py
py scripts/simulate_learning.py
py scripts/select_trial_window.py
py scripts/plot_session_set_sequence.py --first-sets 2
py scripts/plot_session_set_sequence.py --first-sets 6
py -B -m unittest discover -s tests -v
```

These commands produce trial/set/session summaries, session trend fits, and
within-session learning fits with a task-setting audit, followed by the expanded
trial-level checks and conditional recoverability simulations. Results are under
`plots/`; see `plots/README.md` for the output map. Window-selection diagnostics
can be reproduced separately with `py scripts/select_trial_window.py`.

The behavioral analysis lives in `scripts/`, automated checks in `tests/`,
the tracking package in `src/trackMEA/`, and the original SFC/tracking workflows
in `scripts/` and `sri/`. Raw data, generated results, caches, local environments,
and package-install metadata are excluded from Git.

### Baseline normalization and trial-count diagnostics

Run `py scripts/baseline_diagnostics.py` after the behavioral, task-setting, and
trial-level extraction stages. Open `plots/baseline_diagnostics/index.html` for
raw means with trial SEM, same-session block 1 ratios, baseline z-scores,
variance, coefficient of variation, and relative SEM for the first 8/16/24
rewarded block 2 trials. Invalid original sets are never replaced. The common
cohort comparison includes 61 airp and 58 braz sessions.

Baseline is the last 128 valid successful non-clamped, unperturbed block 1
trials, requiring at least 64 and all eight target directions. This diagnostic
uses pooled trial means and exports target counts. Ratio intervals resample
both early and baseline trials in circular blocks of four, with block-eight
sensitivity saved in CSV. Eight-trial blocks cannot estimate numerator
uncertainty from an eight-trial window. Z-score SEM holds baseline estimates
fixed. Trial SEM includes adaptation and target effects; increasing the window
changes the behavior being averaged as well as the sample size.

The script inventories all 122 HDF files' nodes, fields and relevant attributes,
exports attempt-anchor perturbation codes, and searches local data filenames
for potential logs. The binary `pert` field does not identify additional
perturbation types. Exploratory calendar slopes use the existing residual-block
bootstrap with and without rotation adjustment; chronology and rotation are
confounded, and the disputed first braz label is retained as a sensitivity.
Run `py -B -m unittest discover -s tests -p test_baseline_diagnostics.py -v` for
normalization, invalid-window, baseline-coverage and resampling checks.

The October 15 rerun uses the exact two-target sequence described below.
The output folder name `premove_session_sets` is retained for compatibility,
but its current timings begin at the first target. Prior premove-based results
are preserved in `plots/archive_premove_20261015_150713/`.
In this workspace, the working analysis interpreter is
`.conda/analysis312/Scripts/python.exe`; it can replace `py` in these commands.

## Behavioral session comparisons

`scripts/analyzeHDF.py` compares airp and braz sessions chronologically, separately
for block types 2 and 3. Each rewarded trial is timed from **first target onset to
reward onset**, using the 60 Hz task clock, and must match this exact contiguous
message sequence: `target, hold, targ_transition, target, hold, targ_transition, reward`.
The interval includes both targets and excludes the preceding premove.
A set is 8 consecutive rewarded
trials in that block, with each peripheral target represented once. Failed attempts are not
included in the timing of a later successful attempt.

The default also compares the first 3 sets as a sensitivity check and the mean
of valid sets 25 onward as late performance (requires at least 8 valid late
sets). `block*_windows2-3_even.png` shows these averages across sessions in
seconds and both early/late ratios. The ratio panel uses identical eligible
sessions for the two early windows; missing data retain their session positions.
The raw panel uses all eligible sessions for each series and labels their counts.
Error bars in the raw panel are SEM across set means. Ratio curves have no
uncertainty bars. Additional columns in `session_summary.csv` record both
windows, the late reference, their ratios, and the three-set minus two-set
difference. Configure these with `--sensitivity-sets`, `--late-start`, and
`--min-late-sets`; the late range must follow both early windows.

```powershell
py scripts/analyzeHDF.py
py scripts/analyzeHDF.py --first-sets 10
py scripts/analyzeHDF.py --first-sets 2 --spacing calendar
```

The default compares the first 2 sets with evenly spaced sessions. Each initial
set has its own series across sessions, with SEM across its 8 trials. A second
panel compares the mean over the first X sets, with SEM across set means. A
session needs all X complete, valid sets for that second panel; missing values
remain gaps at the original session position. All-set learning curves show the
available sets per session and shade the selected early window.

Review the evenly spaced plots for session-to-session improvement first. If no
clear meta-learning pattern is apparent, rerun with `--spacing calendar` to
inspect elapsed calendar time. The script does not automatically declare or
test a meta-learning effect. Calendar spacing uses actual dates; sessions on
the same date share that coordinate.

Outputs go to `plots/premove_session_sets/`:

- `trial_times.csv`: every reward in the selected blocks, with first-target,
  second-target and reward rows, duration in seconds, validity and rejection reason.
- `set_means.csv`: averages and SEM for every complete set, with selected-window,
  eight-target balance, and validity flags.
- `session_summary.csv`: reward counts, available full sets, leftovers, unbalanced sets, invalid
  trials, first-X averages, and read-error/eligibility status.
- `block*_firstX_even.png` (or `calendar.png`) and `block*_all_set_curves.png`.

Incomplete final sets are excluded from set means. Missing or malformed two-target sequences or
trials crossing a block change retain their reward position with a NaN duration,
so later set numbers do not shift. A full set containing invalid timing gets a
NaN mean, as does a set lacking one visit to each of eight target directions.
Invalid sets retain their original positions and are not replaced by later
sets. Both successful and short sessions remain in the count summary.

Paths are relative to the repository, independent of the launching directory.
Use `--hdf-dir`, `--out-dir`, `--monkeys`, or `--blocks` to change the inputs;
`--show` displays figures. `--inspect path/to/session.hdf` retains the original
single-file overview. The behavioral scripts require NumPy, PyTables, and
Matplotlib. Run synthetic checks with `py -B tests/test_analyze_hdf.py`.

### Choosing the early window

The current exploratory choice is **X = 2 sets (16 rewarded trials)** for block
2 early performance. `py scripts/select_trial_window.py` reproduces the
diagnostics in `plots/window_selection/`, using the analysis CSVs plus the HDF
target positions. This script also requires those HDF files' `pert` and
`error_clamp` fields.

Candidate X values are 1 through 11. Comparisons use a fixed cohort with valid,
target-balanced first 11 sets and at least 8 valid late sets from set 25 onward:
57 airp sessions and 58 braz sessions in block 2. Each session's early average
is divided by its own mean over those late sets. The late mean is a reference,
not a fitted asymptote. Median change between sets 25-32 and 33-40 is below 0.5%
in both animals among eligible block 2 sessions, supporting its use here.

The decision criterion is the widest window retaining at least 70% of the
median first-set slowdown relative to that reference, in both animals. This
is an explicit exploratory tradeoff rather than a universal optimum. With the
first-target-to-reward timing, two sets retain 97.5% in airp and 73.9% in braz;
three retain 73.4% and 58.0%; six retain 60.3% and 41.7%. Threshold sensitivity gives
X=2 at 60% and 70%, and X=1 at 80%. Using the median instead of the mean
late reference also selects X=2. Relative SEM across the two set means
has median 5.6% for airp and 6.5% for braz; it includes adaptation and should not
be interpreted as a pure measurement-noise estimate. Session bootstrap
intervals use 2,000 resamples of the fixed cohort.

The first-two-set session comparison can include 61 airp and all 58 braz
sessions in block 2, since it does not require the extra late reference or
validity through set 11. Window selection never uses a session-trend p value
or an SFC correlation. Block 3 is analyzed separately: it does not show the
same initial slowdown and contains error-clamp periods. This behavioral window
does not establish that 16 trials provide enough spikes for reliable SFC.

### Trends across sessions and calendar dates

Run `py scripts/fit_session_trends.py` after `analyzeHDF.py`. It reads the saved
session summary, so fitting does not reread the HDF files. SciPy is required in
addition to the behavioral-script dependencies. Outputs in `plots/session_trends/`
include fits for both spacing choices, a
`trend_summary.csv`, and the fitted curves. Run checks with
`py -B tests/test_session_trends.py`.

For each animal/block, the script fits a constant, a straight line, and a
decreasing exponential `a + b*exp(-x/tau)` to early duration, late duration, and
early/late ratios. Fits give every session equal weight. Actual calendar dates
retain gaps and same-day sessions; recorded-session positions retain missing
observations. Two- and three-set windows use identical eligible sessions for
their direct comparisons. The exponential uses bounded
[SciPy least-squares fitting](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html).

Models are compared by RMSE over five contiguous held-out groups of sessions.
The lowest score is descriptive model preference; small differences should not
be treated as decisive evidence. This is held-out interpolation/extrapolation,
not a prospective forecast validation. Linear 95% intervals use 10,000 circular
residual-block bootstrap draws with block lengths 4 and 6. Dates are held fixed,
and residual dependence is modeled in adjacent observed-session order, assuming
approximately stationary residuals. Intervals are pointwise/exploratory and
are not adjusted for the multiple metrics, axes or animals inspected.

The table also records the Theil-Sen slope and a sensitivity analysis excluding
the first four eligible observations, with its own interval. Main fits retain
those observations. A negative slope indicates shorter durations or a smaller
early/late ratio; an interval spanning zero leaves a flat trend compatible with
these data. These session-level curves do not measure within-session learning
rate or establish causality. The exponential tau is in recorded sessions or days,
not trials within a block.

### Within-session learning and task-setting checks

Run `py scripts/analyze_session_learning.py` after the behavioral analysis.
It processes block 2 in all 122 sessions, with the same first 16 sets (128
rewarded trials) as the fitting window. A free-baseline exponential
`a + b*exp(-(set-1)/tau)` is fitted to each session's set means normalized by
its independently measured late mean (valid sets 25 onward, at least 8 sets).
The first-two-set mean remains the early performance metric. Original set
positions are retained when invalid sets are missing. Fits require at least
12 valid sets and valid first two sets. Candidate fits over 12 and 24 sets are
saved as window-sensitivity diagnostics; they do not replace the primary fit.

The table records early-minus-late duration and its fraction of the late mean.
A positive difference describes within-session improvement; a declining
difference across sessions can reflect less initial perturbation cost. It
does not by itself establish faster learning.

Rate acceptance requires amplitude above both 5% of late duration and twice
the residual MAD-based noise estimate, a decay-versus-flat AICc advantage of
at least 4, tau above the 0.5-set resolution floor and away from the upper
bound, and agreement within a factor of 2 with a robust soft-L1 fit. The
300-draw residual-block bootstrap interval must remain inside those rate
limits, span less than a factor of 4, and support a positive amplitude above
5%. These are explicit exploratory quality rules, not formal significance
tests. Rejected candidates are preserved with reasons; accepted `tau_sets`
and `tau_trials` are separate from the candidate parameter. Smaller tau
means faster decay; `rate_per_set = 1/tau_sets`.

The HDF audit checks rotation, actual block 2 assistance, hold/premove/reward
timings, target radius, timeout and available decoder references. Rotation is
checked against `scripts/sessions.py`. The audit found one discrepancy:
`braz20220315_07_te90.hdf` records +50 degrees while the catalog says -50.
Decoder identity is not provided by the inspected attributes; a changing
`decoder_state` is not a decoder-configuration fingerprint. Recorded reward
duration varies but is after reward onset, the endpoint of our timing metric;
it is retained as task context, not subtracted from trial duration.

Trend tables compare unadjusted, HDF-rotation-adjusted, catalog-rotation-adjusted,
and full task-regime fixed-effect models on both session and calendar axes.
Rotation changes occur in chronological groups, so adjusted slopes remain
model dependent and should not be interpreted causally. Log tau is used for
rate trends; intervals use 5,000 residual-block draws of lengths 4 and 6.
Models without sufficient residual degrees of freedom are marked unavailable.
Rate trends include only accepted sessions and can therefore be affected by
selection. Gray crosses in rate plots show the rejected candidate parameters.

Outputs in `plots/within_session_learning/` include:

- `session_learning_metrics.csv`: learning amount, fitted parameters, confidence
  intervals, accepted rates, sensitivity candidates and rejection reasons.
- `task_settings.csv` and `analysis_settings.json`: audit and analysis settings.
- `learning_trends.csv`: longitudinal coefficients and exploratory intervals.
- `learning_metrics_even.png`, `learning_metrics_calendar.png`, and fit-check
  pages covering every session.

Use `--fit-sets` (12-24) and `--bootstrap` (at least 100) to change settings.
`--reuse-settings` uses the prior audit without rereading HDF fields; use it
only when source data are unchanged. Run checks with
`py -B tests/test_session_learning.py`.

### Trial-level checks and recoverability

`py scripts/trial_learning.py` reads the HDF files and the saved task-setting
audit. This stage requires pandas and statsmodels in addition to the existing
dependencies (`py -m pip install statsmodels`). A contiguous run of one signed
rotation defines a phase. Exposure count includes every recorded session in
that phase, even when a disputed session is excluded from a model; progress is
scaled from 0 to 1 within each phase. Actual gaps between recorded sessions are
retained. Catalog and HDF phase definitions are both saved. The disputed first
braz session is excluded from the primary analysis, then included under each
label in sensitivity fits. Its actual rotation and decoder identity remain
unresolved; neither source is silently corrected.

Each premove starts an attempt. Reward or a penalty completes it; interruption
or end of recording censors it. Center and peripheral failures are separate.
Successful duration uses the same exact seven-event sequence as the set analysis,
from its first target through reward. A first target can also start an attempt
when a preceding premove is absent. Malformed successful sequences retain reward
positions but have invalid timing. Failure/censor observed time still starts at
the attempt onset, for active-time accounting.
Extraction checks compare all 29,951 block 2 rewards against the earlier timing
table. Only completed successful reaches enter the duration models; failures
remain in success-rate and time-to-16-reward outcomes. Timeout observations are
not treated as completed reaches. The primary window is the first 128
peripheral attempts, including failures when counting progress through a block.

Peripheral reach time ends at first target entry. Path straightness is endpoint
displacement divided by traveled distance in the x-z plane. Initial direction
error uses the first displacement of at least 0.1 task units in the first
200 ms; it is missing if movement is insufficient. Session summaries compare
the first 16 rewards with fixed late rewards 193-320 and with the last 128
non-clamped successful block 1 trials. Means give equal weight to each of the
eight target directions. Elapsed time to 16 rewards includes reward/wait dwell;
active-attempt time is also saved to separate that contribution.

Each animal is modeled separately using log duration, target and phase fixed
effects, gap days, phase exposure, two curve basis functions (8 and 32 trials),
and exposure-by-curve interactions. Session intercepts and fast-basis slopes
are random effects. This is a flexible regression basis, not a mechanistic
two-process model. The four primary joint curve-change tests receive Holm
correction. Population intervals assume the mixed-model covariance structure;
boundary warnings and convergence are retained. Half-life summaries are only
available for positive curve shapes, and their interval's valid-draw fraction
is recorded. A change in the starting gap does not establish a change in rate.
Sensitivity fits allow each rotation phase to have its own curve shape; the
initial-cost improvement remains visible under that specification.
[Mixed-model documentation](https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.from_formula.html).

Additional session-summary trends use phase fixed effects and residual-block
intervals of lengths 4 and 6. These trajectory, baseline and failure checks are
exploratory and not corrected as a family. Success GLM intervals cluster by
session. `analysis_plan.json` records the specification and library versions;
it documents this exploratory pass rather than a prospective registration.

`py scripts/simulate_learning.py` keeps the actual trial design and missing
completed reaches, resamples real residuals in eight-trial blocks, and simulates
known 25%/50% changes in initial log-duration cost, tau, or both. Initial excess
comes from the fitted model; starting tau is 16 trials. Session-effect
correlations 0 and 0.5 are examined. Estimated covariance parameters are held
fixed for fast GLS refitting, so these are conditional diagnostics, not power
for the original accepted-tau pipeline or a complete refitted mixed model.
Separate null datasets calibrate thresholds and check false positives; rate
detection uses a curve-shape proxy and records unavailable estimates.
Repeat with `py scripts/simulate_learning.py --tau-trials 24` to check dependence
on the starting curve; its outputs have a `_tau24` suffix. Halving tau from 16
to 8 reaches a basis boundary and often makes positive-shape estimates
unavailable. Starting at 24 substantially improves recovery of a simulated rate
change. Thus detection depends on the estimator and starting impairment, and
an inconclusive observed rate change does not establish absence of learning.

Outputs are in `plots/trial_learning/`: attempt and phase tables, extraction QC,
session summaries, model coefficients/contrasts, trajectory checks, predicted
population curves, failure trends, simulation results and assumptions. Start
with `within_phase_learning_curves.png`, `phase_performance_checks.png`, and
`recoverability.png`. Use `--reuse-attempts` only when source data are unchanged.
Run checks with `py -B tests/test_trial_learning.py`.
