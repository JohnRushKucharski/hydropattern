# Reporting metrics and unknown outcomes: TDD plan

**Status:** agreed design; R0 baseline and R1–R7 complete; R8–R10 not started.
Documentation sequence S3–S5 is complete; S4 is integrated into local
`main`, while S5 is integrated into local `main` at `b05b326`, without pushing.
R0 evidence was captured while working on `docs-reporting-r0`. R1–R7 are
committed on `reporting-metrics`, based on `b05b326`. R7 is complete; R8 is
next and requires user approval before implementation. No changes have
been merged to `main` or pushed.
This is separate from the pattern-correctness and user-documentation plans.
Preserve their work and decisions; coordinate documentation changes.
See [ADR 0004](../adr/0004-reporting-and-unknown-outcomes.md) and the
[user-documentation plan](user-documentation.md).
Work order relative to that plan, and the user pages each slice updates, are
defined in the [documentation and reporting sequence](2026-10-07-documentation-and-reporting-sequence.md):
R0 starts after the documentation site skeleton (sequence step S3) exists.
Terminology follows [`CONTEXT.md`](../../../CONTEXT.md). Since this plan was written,
`exclusive_event_window` became `exclusive_windows`. Nested frequency parts are
called intra-annual and interannual patterns; shared and un-nested code fields
stay neutral, while fields specific to the second part use `interannual_`.

## Agreed decisions

- A component event is a maximal uninterrupted run of final component
  success, not a physical hydrologic event or a frequency window.
- Summary fractions exclude unknown outcomes from the denominator.
  `[1, 0, unknown, unknown]` therefore gives 50%, not 25%.
  Report known and total timestep counts so limited coverage is visible.
  An all-unknown group has an undefined summary, not zero success.
- Fix unavailable moving-average and rate-of-change calculations before
  changing reporting. They must remain unknown rather than becoming
  success or failure through comparison, including `!=`.
- Preserve three-valued component logic: a definite unmet condition settles
  the combined condition as failure; all met conditions settle it as success;
  otherwise it is unknown. Failure-pattern inversion changes known verdicts
  only. Preserve terminal frequency's role in determining component outcomes.
- Audit duration and frequency consumers for loss of unknown outcomes.
- Assess frequency counts using all possible counts implied by unknown trials:
  two qualifying trials and one unknown allow counts 2 or 3. `>= 2` passes,
  `>= 3` is unknown, and `>= 4` fails.
- Unknown source trials are possible qualifying window anchors. Conditions
  accepting zero continue to anchor at every trial.
- With overlapping windows, definite successful coverage takes priority over
  uncertain coverage; a failed window does not erase other coverage.
  Length 3, `>= 1`, source `[unknown, 0, 0, 0]` gives
  `[unknown, unknown, unknown, 0]`; source `[unknown, 1, 0, 0]` gives
  `[unknown, 1, 1, 1]`.
- Exclusive windows preserve uncertainty about which anchors are suppressed.
  Outcomes are known only where every possible schedule agrees.
  Length 3, `>= 1`, source `[unknown, 1, 0, 0]` gives
  `[unknown, 1, 1, unknown]`: a qualifying first trial produces
  `[1, 1, 1, 0]`, while a non-qualifying first trial produces `[0, 1, 1, 1]`.
  Require exact outcomes and practical performance; do not rely on exhaustive
  enumeration of every unknown assignment for production evaluation.
- Duration considers possible run boundaries and lengths rather than treating
  unknown qualifying outcomes as definite run breaks. Preserve a timestep's
  known verdict only when all possible runs agree.
  Source `[0, 1, 1, unknown, 1, 0]` with duration `>= 3` gives
  `[0, unknown, unknown, unknown, unknown, 0]`: the unknown either separates
  non-qualifying runs of lengths 2 and 1 or joins a qualifying run of length 4.
  Whole-run duration assessment remains unchanged.
- Intra-annual probability conditions describe all observed trials in a
  complete water year, not only known trials. Assess possible qualifying
  fractions; preserve a known annual verdict only when all possibilities
  agree. For 12 monthly trials with one known qualifying, one known
  non-qualifying, and ten unknown, the possible fraction is 1/12 to 11/12;
  `>= 0.5` is unknown. Seven known qualifying trials out of 12 guarantee
  `>= 0.5` regardless of remaining unknowns.
  This intentionally replaces the original known-trial annual denominator
  policy, while descriptive summary fractions still exclude unknowns.
  Unknown annual trial outcomes must never be silently converted to zero.
- Apply the agreed uncertainty rules throughout nested frequency. Annual
  reduction for count/between intra-annual patterns must not declare failure when
  there are unknown outcomes and no definite success; the verdict remains
  unknown. Do not replace unknown annual trials with zero before interannual
  window evaluation.
- Retain partial water years in annual reports and mark each as partial or
  complete. Whole-record summaries include all recorded timesteps.
  Use configured water-year boundaries and ending-year labels consistently.
  For an October-start record from January 2020 through December 2021, report
  WY2020 (partial), WY2021 (complete), and WY2022 (partial).
  Calendar completeness and known-outcome coverage are separate: nested
  frequency remains unknown in partial years, but other conditions may have
  known outcomes there. Event-rate exposure is defined separately below.
- Component event counts preserve uncertainty rather than treating unknown
  outcomes as definite separators. Final component outcomes `[1, unknown, 1]`
  imply 1 or 2 component events; do not report 2 as definite.
  A fully known continuous success run crossing a water-year boundary remains
  one whole-record component event. Bounds APIs and annual attribution are
  defined below.
- For the MVP, event-count bounds use the final outcome array without retaining
  dependencies between unknown timesteps. The bounds may include impossible
  combinations and must be labelled and explained accordingly.
  Example: one uncertain frequency window can produce only `[0, 0, 0]` or
  `[1, 1, 1]`, so its actual event count is 0–1; stored
  `[unknown, unknown, unknown]` alone also permits `[1, 0, 1]`, giving MVP
  bounds 0–2. The user explicitly accepts this limitation for the MVP.
  Add an optional final TDD stage to retain dependencies and exclude these
  impossible combinations. This refinement is not required for MVP completion.
- Add explicit event-count and event-rate bounds methods. Existing scalar
  methods return a number only when lower and upper bounds agree; otherwise
  raise a clear error directing callers to the bounds method. Do not silently
  select an endpoint or midpoint. Apply consistent rules to lower-level public
  helpers and document the changed treatment of unknowns.
- Remove reciprocal summary support for the next release. Supported summary
  modes are `portion` and `percentage` only. Reject `return_period` with clear
  explanation and migration guidance; do not silently substitute another mode
  or retain an alias. Remove the Python enum member and update parser choices,
  plot labels/default color-map logic, examples, tests, and every consumer.
  Do not reinterpret reciprocal portion as inverse event rate.
- Expose component event-count/rate bounds through Python APIs and exported
  reporting details. Keep response-surface modes limited to `portion` and
  `percentage`; do not add event-statistic plotting modes in the MVP.
  Event statistics describe final component outcomes, not each characteristic
  diagnostic. Coverage details also apply to characteristic summaries.
- Default response-surface coloring uses red for lower final component
  success/non-failure fractions for both success- and failure-pattern
  configurations. Remove failure-pattern and reciprocal-specific reversal
  rules; preserve explicit custom colormaps. Explain that colors describe
  configured outcomes, not guaranteed ecological benefit.
- Attribute annual component events to the water year containing their first
  successful observed timestep, determining runs before water-year grouping.
  A continuous September 29–October 3 success run with an October boundary
  counts once in the prior water year, not again in the next.
  Count a run already successful at record start as one observed component
  event, without claiming that its physical onset occurred there.
  Annual event rates use these attributed counts and each year's observed
  exposure. Annual bounds need not sum to whole-record bounds.
- Preserve existing metric matrices in per-component workbooks and add a
  reporting-details sheet. Identify scenario, outcome column, and interval;
  include successful/known/total timestep counts, known-outcome coverage,
  and partial/complete/undetermined status. Component rows additionally include
  event-count bounds, exposure, event-rate bounds, and explicit availability
  status. Keep raw timestep CSV/Excel layouts unchanged.
  Export companion coverage data alongside response-surface grids.
- Undefined scenario metrics remain missing grid cells, never zero.
  Plot valid regions without interpolating across undefined cells; identify
  affected scenarios with a clear diagnostic.
  If no valid surface region remains, retain exported summaries/grid/coverage
  and explicitly report that plotting could not be completed. Test the
  installed renderer's minimum-data requirements rather than assuming every
  partially populated grid is renderable.
- Add a configurable minimum known-outcome coverage threshold for plotting,
  defaulting to 90%. Apply it to each scenario's whole-record component metric,
  not to the descriptive summary matrices or event statistics.
  Coverage is known component outcomes divided by recorded component timesteps.
  A below-threshold scenario becomes a plot gap, with an explicit reason in
  diagnostics and companion data; it is not converted to zero.
  Plot/grid exports must distinguish undefined metrics from defined metrics
  withheld because of low coverage.
  Configure `[output.plot] minimum_coverage = 0.9` in fractional units [0, 1],
  with matching Python option and CLI override `--minimum-coverage`.
  Exactly 90% passes; zero disables the cutoff but does not define an
  all-unknown metric. Keep separate from climate-canvas's rendering threshold.
  Validate finite numeric fractions and preserve CLI override/conflict rules.
- If `fillin = true` would coexist with scenarios withheld for undefined
  metrics or insufficient coverage, raise an explicit option-conflict error
  explaining that `fillin` must be disabled. The installed renderer fills
  missing cells globally and cannot distinguish these protected gaps.
  Do not silently override the user's option. Keep existing filling behavior
  for absent scenario combinations when no scenario is withheld.
- Default plot labels must identify fractions/percentages of known outcomes,
  and visible coverage information must disclose the cutoff and number of
  withheld scenarios. Preserve explicit custom labels; coverage information
  must not disappear merely because a custom axis label was supplied.
- Component event rate uses the whole observed record, including fractional
  water years, with its numerator assessed over the same interval.
  Propagate possible event counts into possible descriptive event rates.
  For 18 monthly observations and three definite events, exposure is 1.5 water
  years and rate is 2 events per water year.
  This intentionally replaces complete-water-year-only event-rate exposure;
  fractional-year conventions are defined below. It does not
  imply independence, recurrence probabilities, or a Poisson process.
- Monthly exposure is the number of observed monthly intervals divided by 12.
  Unknown component outcomes still occupy observed intervals.
- Adopt automatic optional-leap-day handling rather than requiring an explicit
  Gregorian/no-leap source setting. An otherwise complete daily water year
  has exposure denominator 366 if February 29 is present, or 365 if omitted.
  Accept the trade-off: a missing real leap-day observation is indistinguishable
  from deliberate synthetic omission and is accepted too.
  Extend daily cadence and annual completeness checks consistently; do not
  tolerate other missing days or collapse existing leap-day observations.
  Existing DOWY normalization remains unchanged.
  For each daily water year, use denominator 366 only when a February 29
  observation is present in that year's recorded rows; otherwise use 365,
  including partial years whose span never reaches February 29.
  This deliberately assumes omitted leap day for those partial years.
- With unsupported cadence, non-leap-day gaps, or insufficient information
  to establish cadence, keep observed-outcome counts and fractions available.
  Mark annual completeness as undetermined and issue a clear diagnostic.
  Reject time-based event-rate exposure and nested annual evaluation rather
  than guessing. Do not impose cadence requirements on plain observed-outcome
  summaries or conflate an unknown outcome with a missing timestamp.

## Dedicated documentation section

Add a user-facing section explaining unknown outcomes, linked from scientific
concepts, affected characteristic references, reporting guidance, and the
Python API. Coordinate placement with the user-documentation redesign.
Here and throughout this plan, documentation means plain-language explanations
with concrete examples in user documentation, not merely developer records,
docstrings, or this plan. Developer records support but do not replace them.

Explain unknown versus failure, `NaN`/`pd.NA` representations, rejected missing
input, unavailable calculation history, restricted rate denominators, partial
water years, component combination, and summary denominators. Include worked
tables for duration, frequency counts, uncertain anchors, overlapping windows,
and exclusivity using the agreed rules.
The user reference for `exclusive_windows` must include the agreed
exclusive-window example above, show both possible schedules, and explain why
the middle outcomes remain known while the first and last remain unknown.
The duration reference must explain uncertain run boundaries with the agreed
length-2/length-1 versus length-4 example. Include thresholds and bounded-run
examples demonstrating both unknown and settled outcomes.
Explain annual qualifying fractions separately from descriptive known-outcome
summary fractions, using the 12-month examples. Migration guidance must flag
the intentional annual-probability policy change.
Reporting guidance must show the October-start partial/complete example and
explain that whole-record summaries retain partial-year observations.
Explain whole-record event-rate exposure with the 18-month example and flag
the change from complete-year-only exposure in migration guidance.
Explain that normalized 365-label DOWY and actual observation counts are
different: February 28 and 29 share a seasonal label but remain separate
observed daily trials. Include leap-year exposure regression examples.
Explain automatic optional-leap-day handling in user data preparation,
water-year completeness, and event-rate guidance. State plainly that this
accepts both synthetic omission and missing real February 29 observations,
but not other gaps. Include complete daily records with and without leap day.
Include partial leap-containing years that never reach February 29 and explain
their agreed denominator of 365. Do not infer denominator from DOWY labels.
Explain which summaries remain available for unsupported schedules, why
completeness is undetermined, and why event rates or nested annual evaluation
cannot be supplied. Provide corrective guidance rather than silent fallback.
Event-count guidance must show the September–October example, explain
first-successful-timestep attribution, observed-record boundary treatment,
annual exposure, and why annual bounds need not add to total bounds.
Plotting guidance must show the 90% default, fractional TOML/CLI/Python values,
the inclusive cutoff, and disabling it with zero. Explain that reports retain
defined fractions even when plotting withholds them. Use `[1, unknown, 1]`
to show a 100% summary with 66.7% coverage that is not plotted by default.
Explain the conflict between `fillin` and protected gaps with corrective
instructions. Explain consistent red-for-lower-final-outcome default coloring
for both pattern types, without implying ecological benefit.

Do not imply that current evaluation already preserves all these unknowns:
moving-average and rate comparisons currently turn unavailable calculations
into binary outcomes. Short forward windows are not automatically unknown;
the existing record-end truncation policy remains distinct.

## Engineering constraints

These do not require further scientific policy decisions:

- Share canonical water-year labeling, cadence/completeness, exposure, and
  outcome-summary helpers rather than maintaining separate implementations
  in `Result` and formatters. Preserve historical ADR text.
- Propagate the configured water-year boundary through evaluation and scenario
  APIs into reporting. Direct library callers must supply or unambiguously
  establish that boundary; never silently assume January 1 for an October
  record. Reject inconsistent metadata explicitly.
- Use actual timestamps for calendar boundaries, retaining normalized DOWY
  for seasonal conditions. Test February-start boundaries where February 28
  and 29 share DOWY; they must not accidentally create two water years.
- Supported daily records may omit February 29 only. Monthly records may use
  month starts, month ends, or consistent supported midmonth dates.
  Duplicate/invalid timestamps and non-leap-day gaps do not become valid
  time-based exposure. An undetermined completeness status is not "partial".
- Whole-record portion is total successful outcomes divided by total known
  outcomes, not the unweighted average of annual portions. Denominators
  belong to each outcome column, since diagnostics can have different coverage.
  Preserve nested output's timestep-summary interpretation; do not silently
  replace it with an equally weighted fraction of water years.
- Bounds return values expose named `lower` and `upper` fields, typed as
  integers for counts and floats for rates. Definite bounds have equal
  endpoints. Scalar-method ambiguity errors identify the appropriate bounds
  method. Reject invalid outcome values instead of treating them as unknown.
- Annual bounds consider transitions in the full record, not independently
  sliced annual runs. Event-rate bounds divide count bounds by positive,
  cadence-supported exposure. Unknown outcomes occupy exposure; missing
  timestamps are a different problem.
- Preserve existing metric-matrix and raw-output layouts where agreed.
  Allocate the additive details sheet without colliding with an existing
  characteristic/component sheet name or Excel's naming limits.
- Companion grid-coverage data identifies scenario coordinates, raw summary,
  known/total counts, coverage, eligibility, and exclusion reason. The plotted
  grid uses the same eligibility mask as the figure.
- Use repository-standard explicit diagnostics. A requested unsupported rate
  must fail; automatic reporting details mark it unavailable with reason
  rather than substituting zero. Expected inability to render a plot must
  remain distinguishable from successful rendering.
- Ordinary interpolation and filling must not resurrect protected gaps.
  Test real renderer output in addition to mocked argument threading.
  Determine minimum renderable data in R0 and codify it before R9; do not
  invent an arbitrary valid-point count or accept a blank figure as success.
- Do not add event plotting modes, general missing-input imputation,
  arbitrary-cadence exposure, scientific independence assumptions, or
  dependency tracking to the required MVP.

## Required TDD sequence

Phase IDs below belong to this reporting plan, not the earlier phases 0–8.
For each slice: write failing acceptance tests, confirm the intended failure,
implement the smallest complete change, refactor, run relevant regressions,
and update directly related user documentation with worked examples on the
target pages listed in the sequence plan. Write only the behavior the slice
implements; keep the strict documentation build passing.

| Slice | Prerequisites | Test-first work and completion gate |
| --- | --- | --- |
| R0. Baseline and feasibility | None | Capture current API/output shapes and known behavior. Inventory all statistic, mode, plot-option, and documentation consumers. Probe installed renderer with partial, all-missing, constant, and sparse grids; record minimum renderability and protected-gap behavior. Measure representative daily/monthly evaluation and dense-unknown cases to establish performance baselines before choosing algorithms. Record breaking changes separately from preserved behavior. |
| R1. Unknown-preserving diagnostics | R0 | Preserve unavailable moving-average and rate results, including `!=`, startup, and denominator restrictions. Test smoothing plus look-back, positive/zero/negative restricted denominators, three-valued component combination, failure inversion, and terminal frequency dispatch. Existing fully known comparisons remain unchanged. |
| R2. Canonical water years and exposure | R0 | Wire configured boundaries across library/scenario/reporting surfaces. Test January/October/February boundaries, ending-year labels, daily/monthly calendars, monthly mid/end/start dates, optional leap day, partial-year denominators, and full-year exposure of exactly 1. Keep observed summaries available with undetermined completeness for unsupported cadence; reject unsupported time-based requests. No duplicate leap-day boundary or silent January fallback. |
| R3. Duration uncertainty | R1 | Verify possible run boundaries for thresholds and inclusive bounds; cover known runs on each side of unknowns, all-unknown runs, and record boundaries. Use the agreed length-2/length-1 versus length-4 fixture. Preserve whole-run behavior and settled failures, not only unknown propagation. **Complete and committed on `reporting-metrics`.** |
| R4. Forward frequency uncertainty | R1, R3 | Test possible counts, unknown anchors, zero-admitting predicates, all six operators, inclusive between bounds, overlap union, exclusive scheduling, and record-end truncation. Reproduce both agreed overlap examples and exclusive `[unknown, 1, 1, unknown]` result. Compare short inputs against a binary-completion oracle; production implementation must not enumerate every unknown assignment. **Complete and committed on `reporting-metrics`.** |
| R5. Nested annual uncertainty | R2, R4 | Test whole-year possible fractions, 12-month examples, definite annual thresholds, count/between annual reduction, unknown annual anchors, exclusive interannual schedules, annual broadcasting, and partial-year exclusion. Remove unknown-to-zero conversion. Preserve existing fully known nested examples. **Complete on `reporting-metrics`.** |
| R6. Summary fractions and mode removal | R2, R5 | Use per-column known denominators; test 50% example, zero-success versus all-unknown, empty groups, unequal annual coverage, and whole-record aggregation. Make `Result.frequency_table()` and formatter summaries agree on boundaries, counts, and percentages. Reject `return_period` with guidance; remove its enum and all consumers without retaining an alias. **Complete on `reporting-metrics`.** |
| R7. Event bounds and rates | R2, R5 | Add named count/rate bounds and scalar ambiguity errors. Test `[1, unknown, 1]` → 1–2, all-unknown patterns, definite zero/counts, conservative bounds from final arrays, record-start runs, cross-year attribution, partial-year exposure, unsupported cadence, and positive-exposure validation. Small exhaustive tests establish correct MVP bounds without claiming source-dependency exactness. **Complete on `reporting-metrics`.** |
| R8. Reporting details | R6, R7 | Preserve existing metric sheets/raw files; add successful/known/total counts, coverage, completeness, component event bounds, exposure, and availability reasons. Test scenario/column alignment, partial-year rows, sheet-name collisions, CSV/Excel behavior, and Python/CLI consistency. |
| R9. Coverage-aware response surfaces | R6, R8 | Add fractional `minimum_coverage` default 0.9 to TOML, CLI override/conflict resolution, and Python surfaces. Test exact 90%, below/above cutoff, zero cutoff, all-unknown, invalid/nonfinite/bool values, portion/percentage equivalence, protected gaps, `fillin` conflict, grid coverage data, explicit no-surface failure, custom labels/maps, and consistent default coloring. Preserve valid exported data on plotting failure. Verify actual rendered masks for interpolation on/off rather than testing only input arrays. |
| R10. Documentation and release readiness | R1–R9 | Complete dedicated unknown-outcome section and characteristic/reporting/plotting/API examples. Add executable fixtures for every agreed worked example. Explain breaking changes, optional leap-day trade-off, MVP bounds, coverage cutoff, reciprocal removal, and changed colors. Coordinate with the user-documentation redesign and next matching release; keep unreleased notice accurate. |

## R0 baseline record (2026-10-08)

Baseline was captured on branch `docs-reporting-r0` from S5 commit `b05b326`.
No scientific or application code changed during R0. The recorded environment
was Python 3.12.4, NumPy 2.5.1, pandas 2.3.3, Matplotlib 3.11.1,
climate-canvas 0.1.0, and SciPy 1.18.0.

### Current API and output shapes

- `evaluate_component()` returns a `Result` whose `df` contains the selected
  flow column, `dowy`, one diagnostic column per characteristic, and one final
  component-outcome column. Its time index is retained as a `DatetimeIndex`
  when supplied.
- `ScenarioResults.summary()` returns a DataFrame with scenario names as
  columns and `total` plus water-year labels as rows; without a component name,
  it returns a component-name-to-DataFrame mapping. `to_excel()` and `to_csv()`
  return the output directory and retain the raw timestep layouts.
- `Result.frequency_table()` returns one row by default, with `T`, each
  characteristic and its percentage, and the component and its percentage.
  With `by_water_years=True`, it adds groups based on calendar-year labels
  from `identify_water_years()`, not the configured water-year boundary.
- `Result.event_count()` returns an integer and treats each unknown outcome as
  a run break: `[1, unknown, 1]` currently returns 2. `Result.event_rate()`
  returns a float using `record_length_years()`, which counts complete,
  cadence-verified water years and rejects records without supported complete
  years. These scalar APIs cannot express uncertainty.
- `MetricMode` and `[output.metric].mode` currently accept `portion`,
  `percentage`, and `return_period`. Plot configuration contains `enabled`
  and `climate-canvas` options, including `fillin`; CLI exposes overrides for
  plotting, interpolation, display, threshold, color map/ticks, and filling.
  `--run-toml-options` rejects explicit output overrides. No
  `minimum_coverage` option exists in configuration, CLI, or Python APIs.
- Plotting requires scenario names in
  `_<precipitation_delta>_<temperature_delta>` form with at least two distinct
  values on each axis. The grid CSV represents missing scenario combinations
  with NaN; current summaries convert all-unknown outcomes to zero, so
  undefined outcome summaries do not reach the renderer. Output currently
  consists of raw result files, per-component summary workbooks, and—when
  plotting succeeds—grid CSV and PNG files. No reporting-details sheet or
  companion coverage file exists.

### Implementation and documentation consumers

- `hydropattern/patterns/core.py` owns component combination, `Result` output
  columns, `frequency_table()`, `event_count()`, and `event_rate()`.
  `patterns/characteristics.py` prepares duration/frequency inputs and
  comparison results; `patterns/water_year.py` supplies completeness,
  annual-reduction, and exposure helpers.
- `hydropattern/formatters.py` owns summary denominators, workbook/CSV output,
  response-surface summary values, color-map direction, and renderer calls.
  `hydropattern/scenarios.py` exposes in-memory summaries, export methods, and
  plotting to library callers.
- `hydropattern/parsing/specs.py` defines metric and plot option models;
  `hydropattern/parsing/options.py` parses TOML modes and plot settings.
  `hydropattern/cli.py` defines plot overrides and their precedence/conflict
  behavior. `hydropattern/scenario_grid.py` validates coordinates and creates
  grids with NaN for absent combinations.
- Existing user-facing consumers are `docs/user/guide/outputs.md`,
  `docs/user/guide/plotting.md`, `docs/user/reference/configuration.md`,
  `docs/user/reference/cli.md`, `docs/user/migration.md`,
  `docs/user/concepts/glossary.md`, and `docs/user/api/index.md`. Characteristic
  uncertainty explanations belong on the affected reference pages; the
  sequence plan maps R1–R10 to their specific user pages. No dedicated
  unknown-outcome page exists yet.

### Current summary and event behavior

Observed probes confirm that `compute_portion_series()` and
`Result.frequency_table()` divide successes by all recorded rows, including
unknowns. `[1, 0, unknown, unknown]` therefore yields 25% rather than the
agreed 50%; an all-unknown column yields zero rather than an undefined summary.
Each characteristic and component column computes its own success count but
uses the same all-row denominator. The known-outcome denominator change is
therefore a reporting behavior change, not a currently implemented policy.

`evaluate_component()` rejects missing values in input data. Unavailable
calculations created internally, such as moving-average startup values and
restricted rate denominators, pass through ordinary comparisons and become
binary outcomes; `!=` can mark an unavailable NaN calculation as success.
Duration and frequency eligibility preparation also converts unknown
predecessors into non-qualifying binary inputs.

`Result.identify_water_years()` labels rows by calendar year. Formatter
summaries instead accept `first_day_of_wy` and calculate ending-year labels.
`Result` does not retain configured water-year-boundary metadata. Current
event-rate exposure comes from `record_length_years()`, using only complete
cadence-supported water years; the R7 whole-record exposure policy must replace
this convention.

### Renderer feasibility

The installed renderer was exercised with 2-by-2 partial, all-missing,
constant, single-point, diagonal-two-point, and three-point grids, with
interpolation on/off and filling on/off where applicable. The hydropattern
scenario-grid guard separately requires two distinct coordinates per axis.

- A partial grid with multiple known values renders. A completely missing
  grid raises `ValueError: vmin, vcenter, vmax must increase monotonically`
  and writes no plot.
- The renderer's default `TwoSlopeNorm` rejects a constant finite grid.
  hydropattern's `_degenerate_range_norm()` supplies a `Normalize` escape
  hatch; through that wrapper, both constant grids and a single finite point
  render. A single finite point is only one colored cell, not a two-dimensional
  response region; successful function return alone is not a sufficient
  success criterion for R9.
- Sparse grids with two or three distinct known points render as sparse cells.
  Delaunay filling requires at least three non-collinear points. For a 3-by-3
  grid with one unknown center and eight known surrounding values, interpolation
  without `fillin` leaves 48 of 169 resampled cells finite and the center
  unknown; interpolation with `fillin=True` makes all 169 cells finite and
  fills the center with 0.5. The renderer has no protected-gap mask, confirming
  that R9 must reject `fillin=True` when scenarios are withheld.
- R9 must distinguish renderer acceptance from a meaningful nonblank
  response region, validate masks on actual rendered output, and explicitly
  report all-missing or otherwise unrenderable surfaces while retaining
  summaries and grid/coverage exports. Do not infer success from a PNG path
  alone.

### Performance baseline

Seven-run median timings were measured after imports on this Windows
environment. Each `evaluate_component()` workload used a varied-flow fixture
from `default_rng(41)`, with magnitude `> 2.5`, duration `[2, 30]`, and
un-nested frequency `>= 2` over a maximum 30-timestep window. The daily
10-calendar-year series had 3,653 rows; the monthly series had 120 rows.
Results were 12.727 ms daily and 2.401 ms monthly.

For dense-unknown baseline inputs, direct duration and frequency characteristic
functions received diagnostic arrays with 3,476 of 3,653 daily entries and
116 of 120 monthly entries unknown (generated with a 95% target probability).
Together they took 4.547 ms daily and 0.154 ms monthly. Neither output
contained NaN: current eligibility preparation turns unknown predecessors
into non-qualifying binary inputs. These timings establish a reference only;
they do not measure the uncertainty-preserving algorithms planned for R1–R5
or set a performance target.

### Behavior-change inventory

R1–R9 will change unknown propagation through comparisons, duration/frequency,
and annual evaluation; summary denominators and all-unknown handling; event
count/rate bounds and exposure; optional-leap-day/completeness handling;
removal of `return_period`; response-surface coverage eligibility, protected
gaps, and default coloring; and additive reporting details. Do not describe
these as implemented before their slices land.

Preserve fully known evaluation behavior and existing published arrays except
where the agreed calendar, reporting, exposure, mode-removal, or color changes
explicitly apply. Keep raw timestep CSV/Excel layouts and existing metric
matrices; add reporting details and companion coverage data without replacing
them. Preserve CLI explicit-override precedence and the `--run-toml-options`
conflict rule while extending plot options.

R1–R5 were implemented after R0. Do not implement optional R11 as an
implicit prerequisite or expand MVP while correcting unrelated issues.

## Implementation progress (2026-10-08)

- **R1 complete:** unavailable order-1 comparison inputs remain NaN, including
  moving-average startup and restricted rate denominators. Updated comparisons
  for `!=`, retained known verdict behavior, and updated existing expectations.
  Existing component-truth/failure-inversion/terminal-frequency tests pass.
- **R2 complete:** canonical ending-year labels now share one validated helper;
  `Result` stores/uses configured boundary metadata and scenario evaluation
  propagates it. Direct `Result` callers must supply a boundary; timestamp+DOWY
  callers may establish one unambiguously. February 28/29 no longer create two
  water years. Daily completeness permits omitted February 29 only. Added
  cadence-validated daily/monthly observed-exposure helper with the agreed
  365/366/12 denominators; `Result.event_rate()` remains on old complete-year
  exposure until R7.
- Focused reporting/pattern tests passed; full suite passed (705 tests),
  `mypy hydropattern/` passed, and strict MkDocs build passed.
- User-facing updates clarify unavailable independent comparisons and
  configured water-year labels/exposure. Full uncertainty and migration
  documentation remains assigned to R10.

- **R3 complete:** duration now combines preceding conditions using
  three-valued logic and evaluates all possible run boundaries. A diagnostic
  stays known only when every possible run agrees; definite failures still
  break runs. Thresholds, inclusive bounds, all-unknown inputs, record edges,
  and the agreed length-2/length-1 versus length-4 case are covered, including
  a short-input binary-completion oracle. Updated duration and evaluation-order
  pages; the dedicated unknown-outcomes page remains assigned to R10.
- Validation: `uv run pytest -q` passed (713 tests), `uv run mypy hydropattern/`,
  `uv run mkdocs build --strict`, and `git diff --check` passed.
- **R4 complete:** un-nested frequency now preserves unknown qualifying
  timesteps and possible anchors. Count ranges are assessed without enumerating
  binary assignments; overlapping windows combine possible/definite coverage,
  and exclusive windows propagate possible claim schedules. Tests cover all
  six comparison operators, inclusive bounds, zero-accepting predicates,
  truncation, the agreed overlap/exclusive examples, multiple preceding
  outcomes, and short binary-completion count oracles. Updated the frequency
  reference and glossary/index status.
- Validation: `uv run pytest -q` passed (723 tests), `uv run mypy hydropattern/`,
  `uv run mkdocs build --strict`, and `git diff --check` passed. Dense-unknown
  3,653-step evaluation with a 30-step window took 15.1 ms overlapping and
  31.9 ms exclusive in the R0 environment.

- **R4 correction during R5:** the original implementation treated overlapping
  counts and exclusive schedules independently, losing shared-trial correlations.
  Its short oracle compared only the first output for longer windows and missed
  `[1, unknown]`, `= 1`, N=2: the correct output is `[unknown, 1]`.
  Restored full-output checks and expanded the oracle to every ternary input
  of lengths 1–4, all six operators, inclusive bounds, every window length,
  and both overlap modes. Production uses reduced decision graphs rather than
  enumerating assignments; schedules discard consumed variables while
  retaining future constraints, with earliest trials ordered first. Monotone
  overlapping predicates retain their
  count-bound fast path. The earlier timing figures describe the superseded
  approximation, not the corrected exclusive evaluator.
  For a seeded 95%-unknown, 3,653-step record and N=30, corrected `>= 2`
  took approximately 15 ms overlapping / 1.1 s exclusive; `= 2` took 9.3 s /
  1.5 s. Exact bounded/equality overlap is materially slower on dense unknown
  records; no source assignments are enumerated and fully known behavior retains
  its fast path.

- **R5 complete:** shared three-valued conjunction retains unknown trials
  unless another preceding condition settles failure. Annual fractions assess
  every attainable qualifying count over all observed yearly timesteps;
  agreement yields a known verdict, disagreement yields NaN. Scalar annual
  ratios remain undefined when any trial is unknown. Annual count/between
  diagnostics reduce to 1 on any definite success, 0 only on all definite
  zeros, otherwise unknown. Unknown yearly trials reach the correlated
  interannual engine without conversion to zero; complete-year outcomes
  broadcast and partial years remain excluded. Monthly examples, six-operator
  fraction oracles, nested equality/zero/between conditions, exclusive
  schedules, component dispatch, and fully known regressions are covered.
  Updated frequency reference, migration notes, glossary, and index.
- Validation: `uv run pytest -q` passed (760 tests), `uv run mypy hydropattern/`,
  `uv run mkdocs build --strict`, and `git diff --check` passed.

R7 is complete; R8 is next and requires user approval. Do not start R8–R10
ahead of their dependencies or implement optional R11 without further
authorization.

## Validation and completion

- Use existing pytest and type-check tooling; add no dependency merely for
  planning or validation. Run targeted selectors per slice and full regressions
  at meaningful integration boundaries.
- Preserve fully known behavior except explicitly agreed calendar, reporting,
  exposure, mode-removal, and color changes. Existing published arrays remain
  regression fixtures.
- For local uncertainty algorithms, use exhaustive binary completions on short
  source arrays as test oracles, including all-unknown and mixed inputs.
  Also test long, densely unknown inputs against R0 baselines to detect
  exponential growth or unacceptable memory use.
- Validate exact numeric shapes and limits: bounds endpoints, inclusive 90%
  cutoff, coverage/summary denominators, 12-month exposure, and 365/366
  presence-based daily denominators. A plausible chart is not proof of masks.
- Confirm CLI precedence, explicit zero overrides, TOML-only conflict handling,
  direct Python calls, scenario APIs, and exported workbook/grid data agree.
- Expected plot failures must produce an explicit failure status while keeping
  already written summaries/grid/coverage. Do not claim every component plotted
  if some could not; preserve valid component outputs where feasible.
- MVP is complete when R0–R10 are implemented and documented, required checks
  pass, and no agreed behavior remains silently unsupported. R11 is optional.

## Code evidence

- `hydropattern/patterns/core.py`: comparison evaluation converts calculations
  into binary outcomes; component combination already preserves three-valued
  logic; `Result.identify_water_years()` currently labels calendar years;
  `Result.event_rate()` counts whole-record events but divides by complete years.
- `hydropattern/patterns/characteristics.py`: duration and frequency source
  preparation currently treats unknown preceding conditions as non-qualifying.
- `hydropattern/patterns/water_year.py`: shared complete-water-year detection
  and annual reduction helpers.
- `hydropattern/formatters.py`: summaries already use ending-year labels but
  divide by all rows, including unknown outcomes. Response surfaces use the
  whole-record summary metric, not component event counts. Their numeric guard
  currently rejects `pd.NA`; undefined-metric plotting needs explicit handling.
- `hydropattern/scenarios.py`, `hydropattern/cli.py`, and
  `hydropattern/parsing/specs.py`: scenario boundary metadata, public plotting
  calls, option override/conflict handling, and the current metric-mode enum.
- Installed climate-canvas exposes global `fillin` but no protected-gap mask.
  Its interpolation path can fill NaN regions when filling is enabled.
  Do not edit installed package files as an implementation shortcut.

### R6 completion record (2026-10-08)

Summary portions now divide each characteristic/component's successes by
that column's known outcomes. All-unknown and empty groups remain undefined;
whole-record totals aggregate success/known counts instead of averaging annual
portions. `Result.frequency_table()` now uses the configured water-year
boundary, ending-year integer labels, and per-column known-outcome percentages,
matching formatter summaries. `return_period` was removed from `MetricMode`
and default colormap selection; TOML parsing rejects it with replacement
guidance. Updated output/configuration/migration/glossary docs and current
example/developer guidance.

Validation: targeted formatter, metric-option, scenario, and user-documentation
tests passed; `uv run pytest -q` passed (761 tests), `uv run mypy hydropattern/`,
`uv run mkdocs build --strict`, and `git diff --check` passed.

### R7 completion record (2026-10-08)

Added named `EventCountBounds` and `EventRateBounds` results, public
`count_event_bounds()`, and `Result` whole-record and water-year-specific
bounds APIs. Scalar event count/rate methods now raise clear errors when their
bounds differ. Bounds use dynamic programming over final outcomes, allowing
each unknown timestep to vary independently; they intentionally may include
counts impossible under source-characteristic dependencies. Annual event
starts are counted from whole-record transitions and attributed to the
water year containing each start. Event rates use observed whole-record
exposure, including partial water years, and annual exposure, with supported
daily/monthly cadence required. Boundary inference uses timestamps plus DOWY
when direct `Result` callers omit metadata.

Updated output guidance, Python API guidance, and migration notes with
uncertain counts, conservative bounds, annual attribution, partial-year
exposure, and unsupported-cadence behavior.

Validation: `tests/test_event_count.py`, `tests/test_event_rate.py`, and
`tests/test_time_handling_phase5.py` passed; `uv run mypy hydropattern/`,
`uv run mkdocs build --strict`, and `git diff --check` passed. Full pytest
passed (771 tests).

R8 is next; wait for user approval before starting it.

## Reporting-versus-event-count example

For final component outcomes `[1, unknown, 1]`, the agreed descriptive portion
is 1.0, known-outcome coverage is 2/3, and the possible component event count
is 1 or 2. These are different statistics. User documentation must explain
that 100% describes known outcomes, not demonstrated success throughout the
record, and must make coverage discoverable alongside response-surface data.
An uncertain event count does not itself make the current plotting metric
undefined. Explicit bounds methods expose the uncertainty; scalar methods
reject ambiguous values. Under default 90% plot coverage, this scenario is
withheld unless the threshold is lowered.

Explain the MVP's potentially over-wide bounds with the concrete uncertain
frequency-window example. Do not describe them as exact counts allowed by the
original frequency evaluation.

## Optional R11: dependency-aware event bounds

After R0–R10, optionally retain relationships between unknown
outcomes through evaluation and use them to exclude impossible event-count
combinations. Start with a failing test for the uncertain three-timestep window:
MVP bounds are 0–2, but only 0–1 is possible. Define a practical representation
and performance gates before implementation; cover duration, overlapping and
exclusive windows, and nested annual evaluation. Handle results without
dependency information explicitly, never silently claiming exactness.
