# Issue 6 Behavior Delta Report: Deterministic Parameter Validation

## What changed

### 1. Empty/missing metrics: `IndexError` → `PARSER_MISSING_FIELD`

**Before**: Calling any characteristic parser with an empty list (e.g. `timing_parser([])`)
raised a bare Python `IndexError` with no structured context.

**After**: Raises `HydropatternError` with `code = PARSER_MISSING_FIELD` and
`context.characteristic = '<name>'`. This applies to all four characteristics:
`timing`, `magnitude`, `duration`, and `rate_of_change`.

---

### 2. Timing: range validation added

**Before**: Any two integers were accepted as day-of-year values.
`timing_parser([0, 400])` would parse without error.

**After**: Each day-of-year value must be in `[1, 366]`.
Out-of-range values raise `PARSER_INVALID_VALUE`.

---

### 3. Timing: single-day and wrap-around windows now supported

**Before**: `timing_parser([180, 180])` and `timing_parser([335, 60])` both raised
`PARSER_INVALID_VALUE` because they were passed to `between_parser`, which required
strictly ascending values.

**After**: Both are valid:
- `[180, 180]` — single-day window, evaluates exactly 1 day per year (`180 ≤ doy ≤ 180`).
- `[335, 60]` — cross-year wrap-around, evaluates `doy ≥ 335 OR doy ≤ 60`.

A new `timing_window_fx` function implements this logic and replaces the direct
`between_parser` call inside `timing_parser`.

---

### 4. Magnitude: value range validation added

**Before**: No lower bound was enforced on threshold values. `magnitude_parser(['>', -5.0])`
parsed successfully.

**After**: For simple form, `value` must be `≥ 0`. For between form, both `min_value` and
`max_value` must be `≥ 0`. Violations raise `PARSER_INVALID_VALUE`.

---

### 5. Magnitude / rate-of-change: `ma_periods` range validation added

**Before**: `ma_periods` was only type-checked (must be `int`). `magnitude_parser(['>', 1.0, 0])`
raised no error.

**After**: `ma_periods` must be an integer `≥ 1`. Zero or negative values raise
`PARSER_INVALID_VALUE`.

---

### 6. Duration: integer + range validation added

**Before**: Float values were accepted as `time_steps` (e.g. `duration_parser(['>', 1.5])`
parsed without error). Zero and negative values also passed.

**After**: `time_steps` (and both between values) must be integers `≥ 1`. Float values
raise `PARSER_INVALID_TYPE`; zero or negative values raise `PARSER_INVALID_VALUE`.

---

### 7. Rate of change: value positivity check added

**Before**: Non-positive threshold values were accepted.
`rate_of_change_parser(['>', 0.0])` parsed successfully.

**After**: The threshold `value` must be `> 0` (since `z_t` is always positive, a
non-positive threshold would never match any meaningful flow condition).
Violations raise `PARSER_INVALID_VALUE`.

---

### 8. Rate of change: `look_back` range validation added

**Before**: `look_back` was only type-checked. `rate_of_change_parser(['>', 1.0, 1, 0])`
raised no error.

**After**: `look_back` must be an integer `≥ 1`. Zero or negative values raise
`PARSER_INVALID_VALUE`.

---

### 9. Rate of change: optional `min` parameter now validated

**Before**: `validate_rate_of_change_metrics` rejected any metrics list longer than 4
elements, so the optional 5th parameter `min` could not be provided without triggering
`PARSER_INVALID_VALUE` (despite `rate_of_change_parser` reading `metrics[4]`).

**After**: Lists of length 2–5 are accepted. When `len == 5`, `min` must be a real number
`≥ 0`. Negative values raise `PARSER_INVALID_VALUE`.

---

## Winner determination

**All new checks are correct** and represent the intended behavior stated in
`examples/detailed.toml` notes and confirmed in the issue specification interview.

Rationale:
- Empty metrics previously caused opaque `IndexError`s. Structured errors with stable codes
  are essential for any caller (CLI, notebook, future GUI) to give actionable feedback.
- Range checks prevent nonsensical configurations from passing silently into computation.
- Timing wrap-around is an explicitly required feature for seasonal analysis that crosses
  the year boundary (e.g. wet-season onset in December–February).
- The `min` parameter fix closes a bug: the validator and the parser were out of sync on
  the maximum allowed list length.

---

## Focused synthetic test cases

### Empty metrics → `PARSER_MISSING_FIELD`
```python
from hydropattern.parsers import timing_parser
from hydropattern.errors import HydropatternError, ParserErrorCode

try:
    timing_parser([], order=1)
except HydropatternError as exc:
    assert exc.envelope.code == str(ParserErrorCode.MISSING_FIELD)
    assert exc.envelope.context['characteristic'] == 'timing'
```

### Timing wrap-around window
```python
import pandas as pd
from hydropattern.parsers import timing_parser

char = timing_parser([335, 60], order=1)
df = pd.DataFrame({'dowy': [335, 60, 180]})
result = char.fx(df)
assert result[0] == 1   # doy 335 is in window
assert result[1] == 1   # doy 60 is in window
assert result[2] == 0   # doy 180 is not in window
```

### Duration float → `PARSER_INVALID_TYPE`
```python
from hydropattern.parsers import duration_parser
from hydropattern.errors import HydropatternError, ParserErrorCode

try:
    duration_parser(['>', 1.5], order=2)
except HydropatternError as exc:
    assert exc.envelope.code == str(ParserErrorCode.INVALID_TYPE)
```

---

## Regression risk

- **Existing configs that used wrap-around timing** (`first_doy > last_doy`) previously
  failed at parse time. They will now succeed. This is a **fix**, not a regression.
- **Existing configs with `min` as 5th rate-of-change parameter** previously failed at
  parse time due to the length check bug. They will now succeed if `min ≥ 0`.
- **Existing configs with negative magnitude values or zero duration** previously parsed
  silently and would produce silent incorrect results. These now raise clear errors.
  Users with deliberately invalid configs (unlikely) will see new errors.
- All 80 previously passing tests continue to pass after this change.

---

## Affected code locations

- [`hydropattern/parsers.py`](../../../hydropattern/parsers.py) — all validation changes
- [`tests/test_parameter_validation.py`](../../../tests/test_parameter_validation.py) — new test file
- [`docs/user/reference.md`](../../user/reference.md) — new user documentation
