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

**Metric**:
The single scalar per scenario derived from a component's `Result`, using the
configured `[metric]` mode (`portion` | `percentage` | `return_period`), summarized
over the whole record (the `'total'` row of `build_summary_sheet`). Z-axis of the
response-surface plot.
_Avoid_: score, value (too generic).

## Example dialogue

> **Dev**: The great_lakes example has scenario columns like `_0_1.5`, `_5_3`. What are
> those?
> **Domain expert**: That's a **scenario grid** — the first number is **precipitation
> delta**, the second is **temperature delta**. Together with each scenario's
> **metric** (here, `portion`), that's exactly the x/y/z a response-surface plot needs.
> **Dev**: What if the grid is missing some combos, like `_10_1.5`?
> **Domain expert**: Then that grid cell is `NaN`. The interpolator will still fill in
> cells with all four neighboring corners present, and leave `NaN` gaps only where a
> corner is missing.
