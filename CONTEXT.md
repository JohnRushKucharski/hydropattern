# hydropattern

Evaluates hydrologic timeseries against configured flow-pattern components, and reports
results as Excel/CSV summaries and, optionally, response-surface plots across scenarios.

## Language

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
over a water year or the whole record, expressed as a portion, percentage,
or descriptive reciprocal of the portion.
_Avoid_: metric (without qualification), score, value (too generic).

**Qualifying timestep**:
An observation interval that satisfies the conditions being considered.
For a frequency characteristic, these are the combined conditions of its preceding characteristics.
_Avoid_: event (for an individual qualifying timestep).

**Trial**:
One unit counted in a frequency characteristic's denominator: a timestep for
un-nested and intra-annual (base) frequency patterns, a water year for interannual
(nested) frequency patterns.
_Avoid_: timestep, period (too generic outside this context).

**Evaluation interval**:
The span of time considered when assessing a characteristic's condition.
A **frequency window**, **qualifying run**, or **look-back interval** is a specific kind of evaluation interval.
_Avoid_: event (for an evaluation interval).

**Frequency window**:
A forward evaluation interval over which qualifying timesteps, or qualifying
water years for interannual frequency, are counted.
_Avoid_: event window (when referring to the interval rather than a component event).

**Frequency window length**:
The configured maximum number of timesteps, or water years for interannual
frequency, in a **frequency window**; record boundaries can shorten the observed window.
_Avoid_: window period (for a count).

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

**Base pattern**:
The un-nested frequency form (`[op, probability]`, `[op, n, N]`, or
`[min_n, max_n, N]`) evaluated first in a frequency characteristic — intra-annual
when nested, the whole characteristic when not.
_Avoid_: inner pattern, first pattern.

**Nested pattern**:
The optional second base pattern in `frequency = [<base pattern>, [nested pattern]]`,
evaluated on the base pattern's per-water-year qualifying outcomes across years
(interannual). Absent nested pattern means the frequency characteristic is un-nested.
_Avoid_: outer pattern, second pattern.

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
