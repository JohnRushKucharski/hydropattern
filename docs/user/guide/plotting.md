# Response-surface plotting

Plotting compares a component's whole-record summary across a scenario grid.
It requires multiple input columns whose names encode two numeric axes in
the form `_precipitation_delta_temperature_delta`, such as `_0_1.5`. There
must be at least two distinct values on each axis. An arbitrary set of
scenario names or one scenario alone is not a grid.

When a valid grid is used, hydropattern writes a grid CSV and a PNG for each
component. A missing combination of precipitation and temperature values is
blank in the grid CSV; display of missing cells in a plot depends on the
renderer options.

## Enable plotting

Enable it on the command line:

```console
hydropattern run config.toml --plot
```

Or enable it in the configuration:

```toml
[output.plot]
enabled = true

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
independently of plotting. Plot-color interpretation and coverage eligibility
will be documented with their reporting implementation.
