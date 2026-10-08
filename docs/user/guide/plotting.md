# Response-surface plotting

Plotting compares a component's whole-record summary across a scenario grid.
It requires multiple input columns whose names encode two numeric axes in
the form `_precipitation_delta_temperature_delta`, such as `_0_1.5`. There
must be at least two distinct values on each axis. An arbitrary set of
scenario names or one scenario alone is not a grid.

Each plotted value is the component's whole-record `portion` or `percentage`
of known outcomes. The default color scale uses red for lower fractions for
both success-pattern and failure-pattern components; colors describe the
configured component outcome, not ecological benefit. Axis/colorbar labels
name the selected summary scale. A plot title also reports the coverage
cutoff and number of scenarios withheld, even when you set a custom title or
axis labels.

By default, a scenario needs at least 90% known-outcome coverage to appear in
the plot. Coverage is known component outcomes divided by recorded component
timesteps. The cutoff applies only to plots: summaries and raw timestep data
retain every scenario. A scenario exactly at the cutoff is included. Set the
fraction in `[output.plot]`:

```toml
[output.plot]
enabled = true
minimum_coverage = 0.9
```

The equivalent CLI override is `--minimum-coverage 0.75`; valid values are
finite fractions from 0 through 1. Setting zero disables the coverage cutoff,
but an all-unknown scenario still has no defined summary and remains blank.
The renderer's `threshold` option is separate: it controls the color scale,
not scenario eligibility.

When a valid grid is used, hydropattern writes a summary grid CSV, a companion
`{component}_grid_coverage.csv`, and a PNG for each component. The summary
grid contains values only for eligible scenarios. The companion file has one
row per scenario, with precipitation and temperature coordinates, raw summary,
known/total counts, coverage, cutoff, eligibility, and exclusion reason.
Reasons distinguish a below-cutoff metric from a summary with no known
outcomes (or no recorded timesteps). Missing axis combinations remain blank
in the summary grid. Hydropattern warns with names and reasons for excluded
scenarios.

Interpolation can smooth eligible regions, but it must not fill a scenario
withheld for low coverage or an undefined summary. If any scenario is
withheld, `fillin = true` is rejected because the renderer's global fill-in
cannot distinguish protected gaps from ordinary missing grid combinations.
Disable fill-in to preserve those gaps. Fill-in remains available for absent
grid combinations when no scenario is withheld.

At least three non-collinear scenarios with defined summaries are needed to
form a two-dimensional response region. If fewer remain, plotting reports
`PLOT_NO_RENDERABLE_SURFACE`; summary outputs and both grid CSVs are retained.

## Enable plotting

Enable it on the command line:

```console
hydropattern run config.toml --plot
```

Or enable it in the configuration:

```toml
[output.plot]
enabled = true
minimum_coverage = 0.9

[output.plot.climate-canvas]
interpolate = true
show = false
title = "Seasonal flow response"
```

Use `--no-interp` to disable interpolation and `--show` to open an interactive
plot window as well as save the image. The `[output.plot.climate-canvas]`
table also accepts title and axis-label settings. See the
[CLI reference](../reference/cli.md) and
[configuration reference](../reference/configuration.md#output-settings).

An invalid set of scenario names raises `PLOT_INVALID_SCENARIO_GRID`; it does
not produce a plot with guessed axes. Output summaries remain available
independently of plotting.
