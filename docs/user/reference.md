# hydropattern User Reference

This reference covers all characteristic metric parameters, their valid values, and the
parser error codes you may encounter. It supplements the inline comments in example
configuration files such as `examples/detailed.toml`.

---

## Configuration overview

A hydropattern configuration is a TOML file with two top-level sections:

```toml
[timeseries]
path = "data/flow.csv"
date_format = "%Y-%m-%d"

[components.my_component]
timing    = [305, 335]
magnitude = [">", 1.0]
```

Each component is defined under `[components.<name>]` and contains one or more
characteristic keys. The sections below document each characteristic and its valid
parameter ranges.

---

## Timeseries options

```toml
[timeseries]
path                     = "data/flow.csv"  # required
date_format              = "%Y-%m-%d"       # optional, defaults to ''
first_day_of_water_year  = 1                # optional, defaults to 1
sheet_name               = 0                # optional, defaults to 0 (Excel only)
```

| Key                        | Type          | Default | Required | Description |
|----------------------------|---------------|---------|----------|--------------|
| `path`                     | string        | —       | **yes**  | Path to a `*.csv` or `*.xlsx`/`*.xls` file with header row `time, <column_1>, ..., <column_n>`. Every column after `time` is treated as its own scenario (see [Response surface plots](#response-surface-plots---plot)). |
| `date_format`              | string        | `''`    | no       | `strftime`/`strptime` format code for the `time` column, e.g. `"%Y-%m-%d"`. Empty string (default) lets pandas auto-detect the format. |
| `first_day_of_water_year`  | integer       | `1`     | no       | Day-of-year (1–365) the water year starts on. `1` = 1 January. |
| `sheet_name`               | string or int | `0`     | no       | Excel sheet name or 0-based index to read. Ignored when `path` is a `*.csv` file. |

This section is parsed into a `TimeseriesSpec` (`hydropattern/parsers.py`), the single
source of truth for these defaults — see `parse_timeseries_spec`.

**Errors**
```toml
# PARSER_MISSING_SECTION: no [timeseries] section at all.

[timeseries]
date_format = "%Y-%m-%d"   # PARSER_MISSING_FIELD: 'path' is required.
```

---

## Characteristic parameters

### Timing

```toml
timing = [first_doy, last_doy]
```

Defines the calendar window during which the component is evaluated.

| Parameter   | Type    | Constraint         | Description |
|-------------|---------|-------------------|-------------|
| `first_doy` | integer | 1 ≤ value ≤ 366   | First calendar day-of-year (inclusive). |
| `last_doy`  | integer | 1 ≤ value ≤ 366   | Last calendar day-of-year (inclusive). |

**Notes**
- Day-of-year values use a 365-day base year. During leap years, 28 Feb and 29 Feb share the same day-of-year position.
- `first_doy == last_doy` is valid and evaluates exactly one day per year.
- `first_doy > last_doy` is valid and describes a cross-year (wrap-around) window.
  For example, `[335, 60]` matches 1 December through 1 March.

**Examples**
```toml
timing = [305, 335]   # 1 November – 1 December
timing = [180, 180]   # Single day (1 July)
timing = [335, 60]    # Wrap-around: December through February
```

---

### Magnitude

```toml
# Simple form
magnitude = [operator, value]
magnitude = [operator, value, ma_periods]

# Between form
magnitude = [min_value, max_value]
magnitude = [min_value, max_value, ma_periods]
```

Evaluates whether streamflow meets a threshold condition.

| Parameter    | Type          | Constraint      | Description |
|--------------|---------------|-----------------|-------------|
| `operator`   | string        | one of `<`, `<=`, `>`, `>=`, `=`, `!=` | Comparison operator. |
| `value`      | real number   | ≥ 0             | Threshold to compare flow against. |
| `min_value`  | real number   | ≥ 0             | Lower bound (between form). |
| `max_value`  | real number   | ≥ 0, > min_value | Upper bound (between form). |
| `ma_periods` | integer       | ≥ 1             | Optional. Moving average window in timesteps. Defaults to 1 (no smoothing). |

**Moving average formula**

When `ma_periods = k`:
```
y_t = 0                                         if t < k - 1
y_t = (x[t-k+1] + x[t-k+2] + ... + x[t]) / k  otherwise
```
The comparison is made against `y_t` rather than the raw value `x_t`.

**Examples**
```toml
magnitude = [">", 1.0]        # Flow > 1.0
magnitude = ["<", 1.0, 7]     # 7-day moving average < 1.0
magnitude = [0.5, 5.0]        # 0.5 < flow < 5.0 (between, exclusive)
```

---

### Duration

```toml
# Simple form
duration = [operator, time_steps]

# Between form
duration = [min_steps, max_steps]
```

Evaluates whether the number of consecutive timesteps meeting prior characteristic
conditions satisfies a threshold.

| Parameter    | Type    | Constraint             | Description |
|--------------|---------|------------------------|-------------|
| `operator`   | string  | one of `<`, `<=`, `>`, `>=`, `=`, `!=` | Comparison operator. |
| `time_steps` | integer | ≥ 1                    | Threshold number of consecutive timesteps. |
| `min_steps`  | integer | ≥ 1                    | Lower bound (between form). |
| `max_steps`  | integer | ≥ 1, > min_steps       | Upper bound (between form). |

**Examples**
```toml
duration = [">", 7]    # Condition must hold for more than 7 timesteps
duration = [3, 14]     # Condition holds for between 3 and 14 timesteps
```

---

### Rate of Change

```toml
# Simple form
rate_of_change = [operator, value]
rate_of_change = [operator, value, ma_periods]
rate_of_change = [operator, value, ma_periods, look_back]
rate_of_change = [operator, value, ma_periods, look_back, min]

# Between form
rate_of_change = [lower, upper]
rate_of_change = [lower, upper, ma_periods, look_back, min]
```

Evaluates the ratio of flow at time `t` relative to flow at time `t - look_back`.

| Parameter    | Type        | Constraint          | Description |
|--------------|-------------|---------------------|-------------|
| `operator`   | string      | one of `<`, `<=`, `>`, `>=`, `=`, `!=` | Comparison operator. |
| `value`      | real number | > 0                 | Threshold ratio. Must be positive (see note). |
| `lower`      | real number | > 0                 | Lower bound ratio (between form). |
| `upper`      | real number | > 0, > lower        | Upper bound ratio (between form). |
| `ma_periods` | integer     | ≥ 1                 | Optional. Moving average window. Defaults to 1. Must be the 3rd parameter. |
| `look_back`  | integer     | ≥ 1                 | Optional. Steps back for denominator. Defaults to 1. Must be the 4th parameter. |
| `min`        | real number | ≥ 0                 | Optional. Minimum allowed denominator `y[t-n]`. Defaults to 0. Must be the 5th parameter. |

**Ratio formula**

```
z_t = y_t / y_[t-n]
```

where `y` is the raw or moving-average series and `look_back = n`. The comparison is made
against `z_t`.

**`value` must be > 0** because `z_t` is always positive when evaluated (the ratio of two
positive flow values). A threshold ≤ 0 would never be meaningful.

> ⚠️ **Divide-by-zero warning**: When `min = 0` (the default), a denominator value
> `y[t-n] = 0` in the timeseries will raise a runtime error. If your data may contain
> zero-flow timesteps, set `min` to a small positive value (e.g. `0.001`) to replace
> zero denominators with that floor value.

**Parameter order is strict**: `ma_periods` is always 3rd, `look_back` always 4th,
`min` always 5th. You cannot provide `min` without also providing `ma_periods` and
`look_back`.

**Examples**
```toml
rate_of_change = [">", 2.0]              # Flow doubled since previous timestep
rate_of_change = [">", 2.0, 3]          # 3-day MA doubled since previous 3-day MA
rate_of_change = [">", 2.0, 1, 7]       # Flow doubled since 7 timesteps ago
rate_of_change = [">", 2.0, 1, 1, 0.1]  # Floor denominator at 0.1 to avoid divide-by-zero
```

---

### Frequency

> **Note**: Frequency characteristic validation is covered in a separate issue and is
> not yet fully enforced. See `examples/detailed.toml` for current usage guidance.

---

## Component options

```toml
[components.my_component]
verbose         = false  # Evaluate characteristics independently? Defaults to false.
success_pattern = true   # Present = all characteristics met? Defaults to true.
```

| Key              | Type    | Default | Description |
|------------------|---------|---------|-------------|
| `verbose`        | boolean | `false` | When `false`, each characteristic is only evaluated where all prior characteristics are met. When `true`, characteristics are evaluated independently. |
| `success_pattern`| boolean | `true`  | When `true`, the component is "present" when all characteristics are satisfied. When `false`, presence is indicated by characteristics *not* being satisfied (useful for describing failure states). |

---

## Output options

```toml
[output]
directory = "custom_output_dir/"  # optional; defaults to auto-derived '{config_stem}_output'
overwrite = true                  # optional; defaults to true
excel     = true                  # optional; defaults to true

[output.metric]
mode = "portion"                  # optional; "portion" (default) | "percentage" | "return_period"

[output.plot]
enabled = false                   # optional; defaults to false

[output.plot.climate-canvas]
interpolate = true                                  # optional; defaults to true
show        = false                                 # optional; defaults to false
title       = "My Custom Title"                     # optional; defaults to the component name
xlabel      = "Precipitation Delta (%)"             # optional; shown default
ylabel      = "Temperature Delta (C)"               # optional; shown default
zlabel      = "portion"                             # optional; defaults to [output.metric].mode
threshold   = 0.0                                   # optional; defaults to z-range midpoint
color_map   = "RdBu"                                # optional; matplotlib colormap name
color_map_ticks = [-2.0, 0.0, 2.0]                  # optional; explicit colorbar ticks
```

The entire `[output]` section is optional, as is every key within it and its nested
`[output.metric]`, `[output.plot]`, and `[output.plot.climate-canvas]` sections. Every key
mirrors a `run` CLI flag of the same behavior (see the table below); **an explicit CLI flag
always overrides the corresponding toml value**. When a CLI flag is omitted, the toml value
applies; when both are absent, the documented default applies.

| `[output]` key | Type | Default | Equivalent CLI flag |
|----------------|------|---------|----------------------|
| `directory`    | string | auto-derived `{config_stem}_output` | `--output-dir` |
| `overwrite`    | boolean | `true` | `--overwrite/--no-overwrite` |
| `excel`        | boolean | `true` | `--excel/--no-excel` |

### `--run-toml-options` / `--override-toml-options`

By default (`--override-toml-options`), any explicit CLI flag above always overrides its
corresponding `[output]` toml value, as described above. Passing `--run-toml-options`
instead reverses this: the program must run *exactly* as specified in the toml file's
`[output]` section, and none of the other output-related CLI flags (`--output-dir`,
`--plot/--no-plot`, `--excel/--no-excel`, `--overwrite/--no-overwrite`,
`--interp/--no-interp`, `--show/--no-show`, `--threshold`, `--color-map`,
`--color-map-ticks`) may be passed explicitly alongside it. Doing so raises a
`CLI_CONFLICTING_OPTIONS` error instead of silently ignoring or merging the conflicting values.

### `[output.metric]`

Controls the metric computed in the `{component}_summary.xlsx` summary sheets written by
the formatter (see `hydropattern/formatters.py`), and (when plotting) the response surface's
z-values. The section is optional; when absent, or when `mode` is omitted, the default is
`"portion"`. This option only affects the adapter/reporting layer — core compute contracts
(`Result`, `evaluate_component(s)`) are unchanged.

| Value            | Description | NA/zero policy |
|------------------|-------------|----------------|
| `"portion"`      | Fraction of timesteps in `[0.0, 1.0]` where the condition holds. | Zero successes → `0.0`. No timesteps in a water year → blank (NA). |
| `"percentage"`   | `portion * 100`, on a `[0, 100]` scale. | Same as portion. |
| `"return_period"`| `1 / portion` — average recurrence interval in water years. | Zero-success (undefined/infinite) and NA portions both → blank (NA), never `inf`. |

**Examples**
```toml
[output.metric]
mode = "percentage"

[output.metric]
mode = "return_period"
```

**Invalid configuration**
```toml
[output.metric]
mode = "average"      # PARSER_INVALID_VALUE: not one of portion/percentage/return_period
mode = 1               # PARSER_INVALID_VALUE: non-string mode

[output.metric]
threshold = 0.5        # PARSER_UNKNOWN_OPTION: 'threshold' is not a recognized key
```

> **Migration note**: the metric mode option previously lived at the top-level `[metric]`
> section. It has moved to `[output.metric]`; the old top-level `[metric]` is no longer read.

### `[output.plot]` and `[output.plot.climate-canvas]`

See [Response surface plots](#response-surface-plots---plot) below.

---

## Response surface plots (`--plot`)

```bash
hydropattern run config.toml --plot
hydropattern run config.toml --plot --no-interp
hydropattern run config.toml --plot --show
```

Plotting can also be enabled purely via the config file, with no CLI flag at all:

```toml
[output.plot]
enabled = true
```

The `run` command's `--plot` flag (or `[output.plot].enabled = true`) renders a 2D climate
response-surface plot per component, using scenario results as the z-axis. This requires the
timeseries's scenario columns (excluding the trailing `dowy` column) to encode a **scenario
grid**: each scenario column name must follow the `_<precip_delta>_<temp_delta>` convention
(e.g. `_0_1.5` → precipitation delta 0%, temperature delta 1.5°C), with at least two
distinct values on each axis. See `examples/great_lakes/example_1.toml` /
`examples/great_lakes/superior.xlsx` for a worked example.

For each component, `--plot` writes two files to the run's output directory:

| File | Contents |
|------|----------|
| `{component_name}_grid.csv` | The (precip_delta × temp_delta) grid of the component's `[output.metric]` value (`'total'` row, i.e. computed over the whole record), one row per temperature delta, one column per precipitation delta. Missing precip/temp combos are blank (NA). |
| `{component_name}_plot.png` | The rendered response-surface plot (imshow + contour), with missing grid cells shown as gaps. |

| Option | Default | `[output.plot]`/`[output.plot.climate-canvas]` equivalent | Description |
|--------|---------|------------------------------------------------------------|--------------|
| `--plot/--no-plot` | `false` | `[output.plot].enabled` | Enable response-surface plotting (requires a valid scenario grid; see above). |
| `--interp/--no-interp` | `true` | `[output.plot.climate-canvas].interpolate` | Bilinearly interpolate the plotted surface to a finer grid. Interpolation only fills cells where all four surrounding grid corners are present — gaps adjacent to a missing scenario remain blank. |
| `--show/--no-show` | `false` (not shown) | `[output.plot.climate-canvas].show` | Also open an interactive matplotlib window per component, in addition to saving the plot file. |
| `--threshold <float>` | midpoint of z-range | `[output.plot.climate-canvas].threshold` | Centers the diverging colormap at the provided z-value. |
| `--color-map <name>` | `"RdBu"` | `[output.plot.climate-canvas].color_map` | Matplotlib colormap name used for the response surface. When left at the default `"RdBu"`, hydropattern auto-reverses it to `"RdBu_r"` per component if `metric.mode = "return_period"` XOR the component's `success_pattern = false` (both together cancel out, keeping `"RdBu"`), so red always indicates less success. Explicit non-default colormaps are never auto-reversed. |
| `--color-map-ticks <float>` (repeatable) | climate-canvas automatic ticks | `[output.plot.climate-canvas].color_map_ticks` | Explicit colorbar tick values (repeat flag for multiple ticks). |
| — (toml only) | component name | `[output.plot.climate-canvas].title` | Plot title. Defaults to the component's name when unset. |
| — (toml only) | `"Precipitation Delta (%)"` | `[output.plot.climate-canvas].xlabel` | X-axis label. |
| — (toml only) | `"Temperature Delta (C)"` | `[output.plot.climate-canvas].ylabel` | Y-axis label. |
| — (toml only) | `[output.metric].mode` value | `[output.plot.climate-canvas].zlabel` | Colorbar label. Defaults to the configured metric mode (e.g. `"portion"`) when unset. |

As with all `[output]` keys, an explicit CLI flag (e.g. `--plot`, `--no-interp`) always
overrides the corresponding toml value; `title`/`xlabel`/`ylabel`/`zlabel` have no CLI
equivalent and can only be set via the toml file.

**Non-grid scenarios**: If `--plot` is used on a config whose scenario names don't form
a valid grid (e.g. a single-scenario timeseries, or names not matching the
`_<precip_delta>_<temp_delta>` convention), hydropattern raises a `HydropatternError`
with code `PLOT_INVALID_SCENARIO_GRID` (see [Plot error codes](#plot-error-codes)
below) instead of silently producing an empty or nonsensical plot.

---

## Parser error codes

These codes appear in the `code` field of a `HydropatternError` envelope.

| Code | Meaning | Common cause |
|------|---------|--------------|
| `PARSER_MISSING_SECTION` | A required top-level section is absent. | Config file missing `[timeseries]` or `[components]`. |
| `PARSER_MISSING_FIELD` | A required field or metrics list is absent or empty. | `timing = []`, missing `path` in timeseries. |
| `PARSER_INVALID_TYPE` | A parameter has the wrong Python type. | Float instead of integer for `time_steps`; non-string operator. |
| `PARSER_INVALID_VALUE` | A parameter has the right type but is out of range or has an illegal value. | `first_doy = 0`; `ma_periods = 0`; negative magnitude threshold. |
| `PARSER_UNKNOWN_CHARACTERISTIC` | A characteristic key is not recognised. | Typo in characteristic name, e.g. `magntiude`. |
| `PARSER_UNKNOWN_COMPARISON_SYMBOL` | An operator string is not in the valid set. | `"gt"` instead of `">"`. |
| `PARSER_UNKNOWN_OPTION` | A key in an options section (e.g. `[output]`, `[output.metric]`, `[output.plot]`, `[output.plot.climate-canvas]`) is not recognised. | `[output.metric]` table has a typo'd or unsupported key. |

---

## Plot error codes

These codes appear in the `code` field of a `HydropatternError` envelope raised by
`--plot` (see [Response surface plots](#response-surface-plots---plot) above). Unlike
parser errors, plot errors use `source: 'plot'` in the envelope.

| Code | Meaning | Common cause |
|------|---------|--------------|
| `PLOT_INVALID_SCENARIO_GRID` | Scenario column names don't form a valid precip/temp scenario grid. | Single-scenario timeseries; scenario names don't match `_<precip_delta>_<temp_delta>`; fewer than 2 distinct values on an axis. |

### Accessing error details programmatically

```python
from hydropattern.errors import HydropatternError

try:
    from hydropattern.parsers import timing_parser
    timing_parser([0, 100], order=1)
except HydropatternError as exc:
    print(exc.envelope.code)     # 'PARSER_INVALID_VALUE'
    print(exc.envelope.message)  # Human-readable description
    print(exc.envelope.context)  # {'metrics': [0, 100]}
    print(exc.envelope.source)   # 'parser'
```
