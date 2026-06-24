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
