# hydropattern User Reference

This reference covers all characteristic parameters, their valid values, and the
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
- Timing uses the timestamp's calendar day-of-year, not day-of-water-year.
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
| `min_value`  | real number   | ≥ 0             | Lower bound (between form, inclusive). |
| `max_value`  | real number   | ≥ 0, > min_value | Upper bound (between form, inclusive). |
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
magnitude = [0.5, 5.0]        # 0.5 <= flow <= 5.0 (between, inclusive)
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
| `min_steps`  | integer | ≥ 1                    | Lower bound (between form, inclusive). |
| `max_steps`  | integer | ≥ 1, > min_steps       | Upper bound (between form, inclusive). |

**Examples**
```toml
duration = [">", 7]    # Condition must hold for more than 7 timesteps
duration = [3, 14]     # Condition holds for 3 to 14 timesteps, inclusive (3 <= n <= 14)
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
| `lower`      | real number | > 0                 | Lower bound ratio (between form, inclusive). |
| `upper`      | real number | > 0, > lower        | Upper bound ratio (between form, inclusive). |
| `ma_periods` | integer     | ≥ 1                 | Optional. Moving average window. Defaults to 1. Must be the 3rd parameter. |
| `look_back`  | integer     | ≥ 1                 | Optional. Steps back for denominator. Defaults to 1. Must be the 4th parameter. |
| `min`        | real number | ≥ 0                 | Optional. Minimum allowed denominator `y[t-n]`. Defaults to 0. Must be the 5th parameter. |

**Ratio formula**

```
z_t = y_t / y_[t-n]
```

where `y` is the raw or moving-average series and `look_back = n`. The comparison is made
against `z_t`.

**`value` must be > 0**. The denominator is only valid when it exceeds `min`;
the numerator may be zero or negative, so the resulting ratio is not necessarily
positive.

`min` is a strict denominator threshold, not a floor: a lagged value `y[t-n]`
is used only when `y[t-n] > min`. Otherwise that timestep's ratio is undefined
and the diagnostic is false. For example, `min = 0.1` does not replace a zero
denominator with `0.1`; it excludes zero and all other denominators `<= 0.1`.

**Parameter order is strict**: `ma_periods` is always 3rd, `look_back` always 4th,
`min` always 5th. You cannot provide `min` without also providing `ma_periods` and
`look_back`.

**Examples**
```toml
rate_of_change = [">", 2.0]              # Flow doubled since previous timestep
rate_of_change = [">", 2.0, 3]          # 3-day MA doubled since previous 3-day MA
rate_of_change = [">", 2.0, 1, 7]       # Flow doubled since 7 timesteps ago
rate_of_change = [">", 2.0, 1, 1, 0.1]  # Evaluate only when lagged flow > 0.1
```

---

### Frequency

Frequency evaluates forward windows from qualifying timesteps. A full window
can include later observations, so its result is retrospective rather than a
prediction available on the anchor timestep. At the end of the record, the
available part of a window is evaluated.

#### Timestep-window forms

```toml
# Count predicate
frequency = [operator, n, N, (exclusive_windows)]

# Inclusive count range
frequency = [min_n, max_n, N, (exclusive_windows)]
```

For an un-nested frequency, `N` is a count of input timesteps, not years. A
timestep is one day for daily input and one month for monthly input. For
interannual frequency, `N` is a count of complete water years.

| Field | Type and valid range | Meaning |
|---|---|---|
| `operator` | `<`, `<=`, `>`, `>=`, `=`, `!=` | Condition applied to the number of qualifying timesteps in a candidate window. |
| `n` | integer, `0 <= n <= N` | Count threshold for operator form. |
| `min_n`, `max_n` | integers, `0 <= min_n < max_n <= N` | Inclusive lower and upper count bounds. |
| `N` | positive integer | Maximum forward window length in timesteps or water years, depending on the form. |
| `exclusive_windows` | boolean, default `false` | `false`: combine the results of all qualifying overlapping windows. `true`: each qualifying window prevents later anchors within its N-timestep or N-water-year span, so the windows do not overlap. |

Impossible count predicates are rejected: for example, `< 0` and `> N` cannot
be satisfied by a count in `[0, N]`. Equality at zero or N and `>= N` are valid.
Between bounds include both endpoints.

A qualifying timestep is one where all preceding characteristic conditions
in the component are met. The frequency count is the number of qualifying
**timesteps**, not consecutive sequences or component events. A qualifying
frequency window marks every timestep it spans, including timesteps where the
preceding conditions are not met. If zero can satisfy the condition (for
example `= 0`, `< 1`, or `!= 1`), every timestep can start a window so absence
is observable. Otherwise, only qualifying timesteps can start windows.

With `exclusive_windows = false`, the results of overlapping qualifying
windows are combined, and a later window can extend the marked result. With
`true`, only a qualifying window suppresses later anchors; a window that does
not qualify does not suppress them. Anchors inside a claimed span cannot
extend it. An anchor at the span's final timestep is still suppressed; the
next timestep can anchor another window.

**Six timestep-window examples** (positions are zero-based; each row's output
is the frequency diagnostic and, for a positive component, the component
output):

| Qualifying timesteps | Frequency condition | Exclusive | Expected output |
|---|---|---:|---|
| `[1,0,0,0,0,0]` | `[">=",1,5]` | either | `[1,1,1,1,1,0]` |
| `[0,0,0,0,1,0]` | `[">=",1,5]` | either | `[0,0,0,0,1,1]` |
| `[0,0,0,0,0,1,0,0,0,0,0]` | `[">=",1,5]` | either | `[0,0,0,0,0,1,1,1,1,1,0]` |
| `[0,1,0,0,1,0,0,0,0,0]` | `[">=",1,5]` | `false` | `[0,1,1,1,1,1,1,1,1,0]` |
| `[0,1,0,0,1,0,0,0,0,0]` | `[">=",1,5]` | `true` | `[0,1,1,1,1,1,0,0,0,0]` |
| `[1,0,1,0,0,0,0]` | `[">=",2,5]` | `true` | `[1,1,1,1,1,0,0]` |

`examples/frequency.toml` is a runnable example of compact and ordered
configuration, default/explicit exclusivity, and two-part frequency:

```console
uv run python -m hydropattern run examples/frequency.toml --no-excel
```

#### Intra-annual and interannual frequency

A two-part frequency characteristic combines an intra-annual pattern with an
interannual pattern:

```toml
# Fraction of qualifying timesteps per complete water year,
# then at least one qualifying water year in each forward 2-water-year window.
frequency = [[">=", 0.5], [">=", 1, 2]]
```

For count and range patterns, trailing booleans control overlapping frequency
windows independently at each level. The intra-annual fraction form compares
one annual fraction for each complete water year instead of using a frequency
window, so its boolean has no effect. The interannual boolean controls windows
across complete water years.

For an intra-annual fraction form `[operator, p]`, the fraction is the number
of qualifying timesteps divided by the total number of timesteps in a complete
water year. The comparison is made once per complete water year, and its
verdict is broadcast across every timestep in that year; the comparison is not
additionally gated by whether the last timestep qualifies. This form does not
use a frequency window, so `exclusive_windows` has no effect. The interannual
count/range pattern counts qualifying water years in forward windows measured
in water years. A qualifying interannual window marks every complete water
year it spans, including years that did not qualify under the intra-annual
condition. If the interannual count condition accepts zero, every complete
water year can anchor a window; otherwise, only qualifying water years can
anchor one. At the end of the record, a partial interannual window can be
evaluated using the complete water years available, but a partial water year
does not count as a qualifying water year.

When timestamps are present, only cadence-verified complete daily or monthly
water years contribute to annual statistics; leading and trailing partial
years are excluded. Unsupported cadence, irregular gaps, invalid, duplicate,
or unordered timestamps raise an error for annual calculations. Direct
DataFrames without datetime timestamps can identify years only by DOWY resets,
so they cannot verify calendar completeness.

For four timesteps in each illustrative water year and
`frequency = [[">=", 0.5], [">=", 1, 2]]`:

```text
magnitude       [1,1,1,0, 0,0,0,0, 1,0,1,0]
intra-annual fraction [3/4,     0/4,     2/4    ]
intra-annual    [1,1,1,1, 0,0,0,0, 1,1,1,1]
interannual     [1,1,1,1, 1,1,1,1, 1,1,1,1]
component       [1,1,1,1, 1,1,1,1, 1,1,1,1]
```

If the second and third water years are swapped, the resulting arrays are:

```text
magnitude       [1,1,1,0, 1,0,1,0, 0,0,0,0]
intra-annual fraction [3/4,     2/4,     0/4    ]
intra-annual    [1,1,1,1, 1,1,1,1, 0,0,0,0]
interannual     [1,1,1,1, 1,1,1,1, 1,1,1,1]  # interannual union
component       [1,1,1,1, 1,1,1,1, 1,1,1,1]
interannual     [1,1,1,1, 1,1,1,1, 0,0,0,0]  # interannual exclusive
component       [1,1,1,1, 1,1,1,1, 0,0,0,0]
```

The intra-annual pattern qualifies the first two water years, giving
`[1,1,0]`. With overlapping interannual windows, `[">=",1,2]` marks all
three water years successful. With exclusive windows,
`[">=",1,2,true]` marks only the first two successful.

---

## Component options

```toml
[components.my_component]
success_pattern = true   # Present = all characteristics met? Defaults to true.
```

| Key              | Type    | Default | Description |
|------------------|---------|---------|-------------|
| `success_pattern`| boolean | `true`  | When `true`, the component is present when all conditions are satisfied. When `false`, the characteristics describe a combined failure condition and component output is its logical complement (non-failure). An unknown characteristic verdict remains unknown unless another condition determines the conjunction. |

Characteristic order is always inferred from sequence, never configured
explicitly (the `order` and `verbose` component options are unsupported).
Timing, magnitude, and rate-of-change
always report their own truth value regardless of position or preceding
characteristics; duration and frequency remain dependent on the conjunction
of their preceding characteristics. Components use either the compact
characteristic-key form (as above) or an ordered array-of-tables form:

```toml
[components.my_component]
success_pattern = true

[[components.my_component.characteristics]]
type = "magnitude"
parameters = [">", 1.0]

[[components.my_component.characteristics]]
type = "duration"
parameters = [">=", 2]
```

The compact form's characteristic order depends on TOML table key/value
iteration order, which TOML v1.0 does not formally guarantee (though this
project's `tomllib` and Python dict both preserve it); using it emits a
`UserWarning`. The ordered array form's order is TOML-guaranteed and does
not warn.
Each ordered characteristic table must include a `parameters` array. The
former field `metrics` is rejected, even if `parameters` is also present; see
the [migration notes](migration.md) for an example of the change.

For `success_pattern = false`, characteristics describe a combined failure
condition and the component reports its logical complement (non-failure), not
affirmative ecological success. Unknown values remain unknown unless a known
failure or success determines the conjunction.

### Python evaluation and result columns

`evaluate_component(data, component, data_column=0)` evaluates one zero-based
data-column position; the final DOWY column is not selectable. The default is
the first data column. `evaluate_components(data, components, data_column=0)`
forwards the same selection to each component. To evaluate several series,
call once for each index, or use scenario evaluation, which splits the input
series and evaluates each independently.

Each returned `Result.df` contains only the selected data column under its
original name, then `dowy`, one column per characteristic, and the component
column. `Result.dv_name` is the selected column's original name. Its index is
preserved; a `DatetimeIndex` is named `time`. Invalid indices, invalid final
DOWY values, and duplicate output column names raise `ValueError`.
Characteristic output columns use the characteristic type and parameter
label (for example `magnitude_gt5.0` or `frequency_ge1in5(union)`); the final
component column uses the configured component name.

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

Controls the summary calculated in the `{component}_summary.xlsx` summary sheets and,
when plotting, the response surface's z-values. The section is optional; when absent, or
when `mode` is omitted, the default is `"portion"`. This setting changes reported summary
values, not characteristic or component outcomes. The `Result` object returned by
`evaluate_component(s)` has the data-column selection and result shape described in
[Python evaluation and result columns](#python-evaluation-and-result-columns).

| Value            | Description | NA/zero policy |
|------------------|-------------|----------------|
| `"portion"`      | Fraction of timesteps in `[0.0, 1.0]` where the condition holds. | Zero successes → `0.0`. No timesteps in a water year → blank (NA). |
| `"percentage"`   | `portion * 100`, on a `[0, 100]` scale. | Same as portion. |
| `"return_period"`| Descriptive reciprocal `1 / portion`; it is not a Poisson recurrence probability or guaranteed mean recurrence interval. | Zero-success (undefined/infinite) and NA portions both → blank (NA), never `inf`. |

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

> **Migration note**: the summary mode setting previously lived at the top-level `[metric]`
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
distinct values on each axis. See `tests/test_scenario_grid.py` /
`tests/test_files/cli_smoke_grid_input.csv` for a worked example.

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
| — (toml only) | `[output.metric].mode` value | `[output.plot.climate-canvas].zlabel` | Colorbar label. Defaults to the configured summary mode (e.g. `"portion"`) when unset. |

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
| `PARSER_MISSING_FIELD` | A required field or characteristic parameters are absent or empty. | `timing = []`, missing `path` in timeseries. |
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
    print(exc.envelope.context)  # Additional error details, when available
    print(exc.envelope.source)   # 'parser'
```
