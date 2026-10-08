# Configuration reference

The TOML configuration connects an input time series, one or more
components, and output choices. TOML is plain text: square-bracketed headers
name sections, and each setting is written as `name = value`.

## Input time series

```toml
[timeseries]
path = "data/flow.csv"
date_format = "%Y-%m-%d"
first_day_of_water_year = 1
sheet_name = 0
```

| Setting | Type | Default | Description |
|---|---|---|---|
| `path` | string | required | CSV or Excel file. Its columns are `time` followed by one or more numeric scenario columns. |
| `date_format` | string | inferred | Date parsing format for text dates, such as `%Y-%m-%d`. |
| `first_day_of_water_year` | integer | `1` | Calendar day-of-year (1–365) on which the water year begins. |
| `sheet_name` | string or integer | `0` | Excel sheet name or zero-based sheet position; ignored for CSV input. |

See [preparing data](../guide/preparing-data.md) for input layout,
working-folder behavior, and date guidance.

## Components and characteristic order

Each `[components.<name>]` section defines one component. Use ordered
characteristic tables for new configurations so the evaluation order is
explicit:

```toml
[components.sustained_flow]
success_pattern = true

[[components.sustained_flow.characteristics]]
type = "magnitude"
parameters = [">", 1]

[[components.sustained_flow.characteristics]]
type = "duration"
parameters = [">=", 2]
```

Each ordered table requires a `type` and a `parameters` array. The former
`metrics` key is rejected. See [migration guidance](../migration.md) for
before-and-after syntax.

Compact characteristic keys remain supported:

```toml
[components.sustained_flow]
magnitude = [">", 1]
duration = [">=", 2]
```

The compact form's order follows the order in the TOML table. Although the
current parser preserves that order, TOML does not promise ordering for
ordinary key/value pairs; hydropattern emits a portability warning. Ordered
tables state the sequence directly and do not have that warning. Do not mix
the two forms in one component.

### Component settings

| Setting | Type | Default | Description |
|---|---|---|---|
| `success_pattern` | boolean | `true` | With `true`, all conditions describe the configured pattern. With `false`, they describe a combined failure condition and the component reports its logical complement (non-failure). |

The `order` and `verbose` component settings are not supported. Characteristic
types, exact parameters, defaults, and units are described in the
[characteristic reference](../reference.md#characteristic-reference).

## Output settings

All output tables are optional. CLI options can override corresponding
settings; see the [command-line reference](cli.md).

```toml
[output]
directory = "results"
overwrite = true
excel = true

[output.metric]
mode = "portion"

[output.plot]
enabled = false
```

| Setting | Type | Default | Description |
|---|---|---|---|
| `[output].directory` | string | `{config_stem}_output` | Output folder, created if needed. |
| `[output].overwrite` | boolean | `true` | Replace existing files; set false to create numbered alternatives. |
| `[output].excel` | boolean | `true` | Write raw timestep results to Excel rather than separate CSV files. |
| `[output.metric].mode` | string | `portion` | Summary scale: `portion` (0–1) or `percentage` (0–100). |
| `[output.plot].enabled` | boolean | `false` | Create response-surface outputs for a valid scenario grid. |

`[output.plot.climate-canvas]` controls rendering details:

| Setting | Type | Default | Description |
|---|---|---|---|
| `interpolate` | boolean | `true` | Interpolate the plotted surface to a finer grid. |
| `show` | boolean | `false` | Also display a plot window interactively. |
| `title` | string | Component name | Plot title. |
| `xlabel`, `ylabel` | strings | Precipitation and temperature labels | Axis labels. |
| `zlabel` | string | Configured summary mode | Colorbar label. |
| `threshold` | number | Midpoint of the plotted range | Center of the diverging color scale. |
| `color_map` | string | `RdBu` | Matplotlib colormap name. |
| `color_map_ticks` | array of numbers | Renderer-selected | Optional colorbar tick positions. |
| `fillin` | boolean | `false` | Pass the fill-in setting through to the plot renderer. |

For the next release, use the `portion` or `percentage` mode shown above.
See [migration guidance](../migration.md) for removal of the former
`return_period` mode.

The summary-mode options above describe fully known outcomes. Summary
denominators for unknown outcomes will be updated with their reporting
implementation. See [output files](../guide/outputs.md).

For plotting options, see the [plotting guide](../guide/plotting.md). The
future coverage cutoff, protected gaps, and color interpretation are not
described here ahead of their implementation.
