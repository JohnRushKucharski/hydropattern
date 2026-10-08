# hydropattern

Evaluates hydrologic timeseries against configured flow-pattern components, and reports
results as Excel/CSV summaries and, optionally, response-surface plots across scenarios.

## Implementation context

Documentation sequence S1a–S5 is complete and integrated into local `main`
at `b05b326`. S6 / R0 baseline is recorded; R1–R7 are complete on
`reporting-metrics` (R3: `d40b640`, R4: `59e69eb`, R4 correction:
`e53612a`, R7: `27112d2`). R8 reporting details and R9 coverage-aware
response surfaces are complete; R1–R9 are committed on this branch. R10 and
optional R11 have not started. Nothing has been merged to `main` or pushed.
Wait for user approval before R10. Keep reporting slices separate on
`reporting-metrics`. See the
[sequence](docs/developer/plans/2026-10-07-documentation-and-reporting-sequence.md)
and its linked plans for scope, completion records, and prerequisites.

## Language

This file is the source of truth for domain terminology. Some entries capture
agreed terminology for future reporting behavior; their presence does not
mean that behavior is implemented. Check the documentation and reporting
plans for each work item's implementation status before describing it to users.

**Scenario**:
One data column in a `[timeseries]` input, representing one hydrologic trace/run to
evaluate independently against all configured components.
_Avoid_: Run, trace (as a synonym in code/docs).

**Scenario grid**:
A set of scenarios whose names encode two numeric axes using the `_x_y` naming
convention (e.g. `_0_1.5`), forming a 2D grid suitable for a response-surface plot.
Not every set of scenarios is a scenario grid — only ones matching this naming pattern.

**Precipitation delta**:
The first numeric value in a scenario-grid column name (e.g. the `0` in `_0_1.5`).
X-axis of the response-surface plot, in %.
_Avoid_: x, precip change.

**Temperature delta**:
The second numeric value in a scenario-grid column name (e.g. the `1.5` in `_0_1.5`).
Y-axis of the response-surface plot, in °C.
_Avoid_: y, temp change.

**Characteristic parameters**:
The values that define a characteristic's **condition**, such as a flow
threshold, a seasonal interval, or a duration.
_Avoid_: metrics (for characteristic parameters).

**Condition**:
A criterion used to determine whether a characteristic is satisfied.
For example, a magnitude condition may require flow to exceed a specified threshold.
_Avoid_: metric (for a characteristic condition).

**Summary metric**:
A single numerical summary of a component's outcomes for one **scenario**,
over a water year or the whole record, expressed as a portion or percentage.
Fractions use only known outcomes;
an interval with no known outcomes has no defined summary.
`portion` is the fraction of known outcomes marked as component success;
`percentage` is the same value multiplied by 100.
_Avoid_: metric (without qualification), score, value (too generic),
fraction of successful timesteps (omits the known-outcome denominator).

**Characteristic summary**:
The same known-outcome fraction computed for one characteristic's diagnostic
outcome column rather than the final component outcome. Each column has its
own known-outcome denominator and coverage.
_Avoid_: summary metric (reserved for final component outcomes).

**Unknown outcome**:
A characteristic or component outcome for which success or failure cannot be
determined from the available information; it is neither success nor failure.
_Avoid_: failure, zero (for an undetermined outcome).

**Characteristic outcome**:
A characteristic's success, failure, or unknown verdict at one timestep,
before characteristics are combined into a **component outcome**.
_Avoid_: component outcome (for a single characteristic's verdict).

**Component outcome**:
A component's final success, failure, or unknown verdict; for conditions
combined by conjunction, a definite unmet condition settles failure, all met
conditions settle success, and otherwise the verdict is unknown.
A failure-pattern component reverses known verdicts but leaves unknowns unknown.
_Avoid_: ecological success (for a configured component verdict).

**Known-outcome coverage**:
The fraction of recorded timesteps in an interval with a known outcome for
the characteristic or component being considered. It differs from the
fraction of known outcomes that are successful and from water-year completeness.
_Avoid_: success fraction, complete year (for known-outcome coverage).

**Qualifying timestep**:
An observation interval that satisfies the conditions being considered.
For a frequency characteristic, these are the combined conditions of its preceding characteristics.
_Avoid_: event (for an individual qualifying timestep).

**Qualifying water year**:
A water year whose **intra-annual pattern** condition is met; the unit counted
by an **interannual pattern**.
_Avoid_: successful year, event year.

**Trial**:
One unit counted by a frequency characteristic: a timestep for an
**un-nested frequency** or **intra-annual pattern**, a water year for an
**interannual pattern**. The term carries no statistical meaning; it does not
imply independent or random trials. User documentation names the concrete
unit (timestep or water year) and uses "trial" only when generalizing over both.
_Avoid_: period (too generic); Bernoulli trial (implies independence).

**Water year**:
A year-long reporting and evaluation interval beginning at the configured
calendar boundary, labelled by its ending calendar year. A partial water year
contains only part of that interval in the observed record.
_Avoid_: calendar year (unless the configured boundary is January 1).

**Optional leap day**:
February 29 may be present or deliberately absent in an otherwise daily
record. Its absence is accepted without distinguishing synthetic omission
from a missing real observation.
_Avoid_: general gap tolerance (only leap-day omission is accepted).

**Evaluation interval**:
The span of time considered when assessing a characteristic's condition.
A **frequency window**, **qualifying run**, or **look-back interval** is a specific kind of evaluation interval.
_Avoid_: event (for an evaluation interval).

**Frequency window**:
A forward evaluation interval over which qualifying timesteps, or qualifying
water years for interannual frequency, are counted.
_Avoid_: event window (when referring to the interval rather than a component event).

**Frequency window anchor**:
The trial at which a **frequency window** starts: a qualifying trial, or any
trial when the count condition accepts zero. An unknown trial is a possible
qualifying anchor; with overlapping windows, definite successful coverage
settles an outcome despite other uncertain windows.
_Avoid_: component event start (for a frequency-window anchor).

**Frequency window length**:
The configured maximum number of timesteps, or water years for interannual
frequency, in a **frequency window**; record boundaries can shorten the observed window.
_Avoid_: window period (for a count).

**Exclusive windows**:
The optional trailing boolean of a frequency pattern, `exclusive_windows`.
When false (default), every anchored **frequency window** is evaluated and
overlapping windows' successful coverage is combined (union). When true, a
window claims its span and later anchors inside it are skipped, so windows
never overlap. Each part of a nested frequency sets it independently.
_Avoid_: exclusive_event_window (former name); exclusive bounds (unrelated).

**Frequency count**:
The number of qualifying trials in a **frequency window**. Unknown trial
outcomes leave multiple possible counts; the window verdict is known only
when every possible count gives the same verdict.
_Avoid_: component event count (for qualifying-trial counts).

**Annual qualifying fraction**:
The fraction of all observed timesteps in a complete water year that qualify
for an intra-annual probability condition. Unknown trials leave possible fractions;
the annual condition is known only when all possibilities agree.
_Avoid_: known-outcome summary fraction (for this whole-year condition).

**Qualifying run**:
A maximal uninterrupted sequence of **qualifying timesteps**.
Its **qualifying run length** is assessed by a duration characteristic.
_Avoid_: event (without specifying whether the sequence is a component outcome).

**Qualifying run length**:
The number of consecutive timesteps in a **qualifying run**.
It is distinct from a **component-success run length**.
_Avoid_: qualifying run duration, period (for a timestep count).

**Duration threshold**:
A configured timestep count against which a **qualifying run length** is compared.
The comparison determines whether shorter, longer, or equal-length runs qualify.
_Avoid_: required duration period.

**Duration bounds**:
The configured lower and upper timestep counts defining an inclusive acceptable
range of **qualifying run lengths**.
_Avoid_: duration period.

**Duration characteristic**:
A characteristic that accepts or rejects an entire **qualifying run** according
to its run length and the configured **duration threshold** or **duration bounds**.
Unknown qualifying outcomes leave run boundaries uncertain; a duration verdict
is known only when all possible runs agree at that timestep.
_Avoid_: elapsed-duration tracking.

**Look-back interval**:
The span between the current timestep and the earlier timestep used in a
rate-of-change comparison; it does not imply that intervening timesteps qualify.
_Avoid_: event, duration (for this comparison span).

**Component event**:
A maximal uninterrupted occurrence of final component success, occupying a
**component success period**; it is not necessarily a distinct physical event such as a flood.
Also called a **component-success run** when discussing consecutive successful timesteps.
_Avoid_: event (without qualification), frequency window.

**Component event count**:
The number of **component events** in an assessed interval. Unknown component
outcomes can leave multiple possible counts rather than a definite count.
_Avoid_: observed success-run count (when claiming a definite event count).

**Component event rate**:
The **component event count** divided by the observed record's exposure in
water years, including partial years. It is descriptive, can be uncertain,
and does not imply independent events or a recurrence probability.
_Avoid_: recurrence interval, Poisson rate (for this descriptive summary).

**Component success period**:
The time interval occupied by a **component event**, from its first successful
timestep through its last.
_Avoid_: run length (for the interval rather than its length).

**Component-success run length**:
The number of timesteps in a **component success period**.
It is distinct from the **qualifying run length** assessed by a duration characteristic.
_Avoid_: component event duration, success period (for a timestep count).

**Calendar length**:
The length of a period expressed in calendar-time units, such as days or months,
rather than a count of timesteps.
_Avoid_: run length (when calendar time rather than timestep count is intended).

**Timestep interval**:
The spacing between successive observations; cadence describes its regularity,
such as daily or monthly sampling.
_Avoid_: timestep frequency (to distinguish sampling from the frequency characteristic).

**Un-nested frequency**:
A frequency characteristic with a single pattern, such as `[op, n, N]` or
`[min_n, max_n, N]`, evaluated over timesteps across the whole record; its
windows may cross water-year boundaries.
_Avoid_: intra-annual (for un-nested frequency), base pattern.

**Intra-annual pattern**:
The first part of a nested frequency,
`frequency = [[intra-annual pattern], [interannual pattern]]`, evaluated on
timesteps within each complete water year to decide whether that year is a
**qualifying water year**.
_Avoid_: base pattern, inner pattern, first pattern.

**Interannual pattern**:
The second part of a nested frequency, evaluated across water years on
the intra-annual pattern's qualifying water years.
_Avoid_: nested pattern, outer pattern, second pattern.
Code uses neutral names for fields shared with un-nested frequency and
`interannual_` names only for fields specific to the interannual part of a
nested frequency.
`CharacteristicSpec` uses `has_interannual_pattern` and `interannual_*` fields;
the generic characteristic marker is `is_terminal`.

## Example dialogue

> **Dev**: The great_lakes example has scenario columns like `_0_1.5`, `_5_3`. What are
> those?
> **Domain expert**: That's a **scenario grid** — the first number is **precipitation
> delta**, the second is **temperature delta**. Together with each scenario's
> **summary metric** (here, `portion`), that's exactly the x/y/z a response-surface plot needs.
> **Dev**: What if the grid is missing some combos, like `_10_1.5`?
> **Domain expert**: Then that grid cell is `NaN`. The interpolator will still fill in
> cells with all four neighboring corners present, and leave `NaN` gaps only where a
> corner is missing.
