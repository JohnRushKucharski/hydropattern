# Behavior Delta Report — Issue #13: Parser and TOML Contract for Formatter/Metric Options

## 1. What changed in behavior

### Before (baseline)

The formatter (`hydropattern/formatters.py`, resolved in #12) computed exactly one
summary metric — **portion** (`n / T`, 0.0–1.0) — with no way to configure it from a
TOML file. `parsers.py` had no concept of formatter/metric options at all; it only
normalized `[components.*]` data into `Request`/`ComponentSpec`/`CharacteristicSpec`.
There was no TOML section, no validation, and no error contract for
formatter/metric behavior.

### After (new behavior)

- `parsers.py` gains a `MetricMode` enum (`PORTION`, `PERCENTAGE`, `RETURN_PERIOD`) and a
  frozen `MetricOptions` dataclass (`mode: MetricMode = MetricMode.PORTION`), both
  comparable via `==` like the existing `Request`/`ComponentSpec` contract.
- `parsers.parse_metric_options(data) -> MetricOptions` reads an **optional** top-level
  `[metric]` TOML section:
  ```toml
  [metric]
  mode = "portion"        # default; also "percentage" | "return_period"
  ```
  - Missing `[metric]` section → `MetricOptions()` (mode=`portion`).
  - Missing `mode` key within `[metric]` → defaults to `portion`.
  - `mode` not one of the three valid strings (or not a string) →
    `PARSER_INVALID_VALUE`.
  - `[metric]` present but not a table → `PARSER_INVALID_TYPE`.
  - Any other key inside `[metric]` → **new** `PARSER_UNKNOWN_OPTION` error code
    (added to `ParserErrorCode`), mirroring the existing
    `PARSER_UNKNOWN_CHARACTERISTIC` pattern for `[components]`.
- `hydropattern/formatters.py` adds `compute_metric_series(result, column, mode, first_day_of_wy)`,
  a thin transform layered on top of the existing `compute_portion_series` (which is
  unchanged):
  - `PORTION` → passthrough.
  - `PERCENTAGE` → `portion * 100`.
  - `RETURN_PERIOD` → `1 / portion`, with zero-success and NA portions both mapped to
    `pd.NA` (never `inf`), consistent with the locked NA/zero policy (CI-016).
  `build_summary_sheet`, `write_summary`, and `write_results` now accept an optional
  `mode`/`metric_mode` parameter (default `MetricMode.PORTION`) and thread it through to
  `compute_metric_series`.
- `hydropattern/cli.py` adds `load_metric_options(data)` (parallel to `load_components`/
  `load_timeseries`) and wires the parsed `MetricOptions.mode` through
  `write_output` → `write_results` → `write_summary`.

Core compute contracts (`Result`, `evaluate_component`, `evaluate_components`) are
**not modified**. All new logic lives in the parser/formatter adapter layer.

## 2. Which behavior is correct (new vs baseline)?

**New behavior is correct and strictly additive.**

- Baseline behavior (always compute `portion`) is preserved exactly as the default —
  omitting `[metric]` produces byte-identical summary output to before this change.
- The new `[metric]` section gives users a documented, validated way to switch summary
  units without touching code, satisfying issue #12's original "portion, percentage,
  return period" scope that the MVP slice intentionally deferred.
- Winner rationale: additive default-preserving change with a narrow, single new TOML
  key (`metric.mode`) — lowest risk path to close the gap between #12's MVP and the
  full PRD3 metric-pipeline scope, without pre-empting #21's separate `[output]`
  section (value domain/rounding/export format), which remains blocked until this
  parser contract lands.

## 3. Synthetic test case demonstrating the delta

```python
from hydropattern.parsers import parse_metric_options, MetricMode, MetricOptions

# Before: no such function/contract existed.

# After:
assert parse_metric_options({}) == MetricOptions(mode=MetricMode.PORTION)          # default
assert parse_metric_options({'metric': {'mode': 'percentage'}}).mode == MetricMode.PERCENTAGE
assert parse_metric_options(
    {'metric': {'mode': 'portion'}}
) == parse_metric_options({})  # explicit portion == default (equivalence)

# Invalid mode -> deterministic, machine-readable error.
from hydropattern.errors import HydropatternError, ParserErrorCode
try:
    parse_metric_options({'metric': {'mode': 'average'}})
except HydropatternError as exc:
    assert exc.envelope.code == ParserErrorCode.INVALID_VALUE

# Unknown key -> new PARSER_UNKNOWN_OPTION code.
try:
    parse_metric_options({'metric': {'threshold': 0.5}})
except HydropatternError as exc:
    assert exc.envelope.code == ParserErrorCode.UNKNOWN_OPTION
```

Formatter round-trip, verified against `examples/great_lakes/example_1.toml` +
`[metric] mode = "percentage"` run through the CLI: summary values scale from the
`[0, 1]` portion range to `[0, 100]`, e.g. `total` portion `0.98657` → `98.657` (%).

## 4. Regression risk notes

- **Low risk.** `MetricOptions()` (default `portion`) is used everywhere a caller does
  not pass an explicit mode, so all pre-existing call sites (`write_results`,
  `write_summary`, `build_summary_sheet`, CLI `run`) behave identically to before this
  change when `[metric]` is absent — confirmed by the full pre-existing test suite
  passing unchanged (224 tests, up from 199).
- `return_period`'s zero/NA handling reuses the same `pd.NA`-based policy already
  established for `portion` in #12 — no new NA-handling paradigm introduced.
- New `PARSER_UNKNOWN_OPTION` error code is additive to the `ParserErrorCode` enum;
  does not change the string value or behavior of any existing code.
- `formatters.py` gained one additional `# type: ignore[call-overload]` for the new
  `return_period` `.where(..., other=pd.NA)` call, matching the same pre-existing mypy
  stub limitation already present (and accepted) for the `portion` computation two
  lines above it. Baseline mypy error count is unchanged (4 errors, all pre-existing).
- `[output]` TOML section (value domain, rounding, export format — issue #21) and
  metadata binding (issue #14) are intentionally **out of scope** here; both remain
  blocked on this issue and can build on the `[metric]`-section precedent
  (optional section, per-key validation, `PARSER_UNKNOWN_OPTION` for stray keys).

## 5. Affected code locations

| File | Change |
|------|--------|
| `hydropattern/errors.py` | Added `ParserErrorCode.UNKNOWN_OPTION = 'PARSER_UNKNOWN_OPTION'`. |
| `hydropattern/parsers.py` | Added `MetricMode` enum, `MetricOptions` dataclass, `parse_metric_options()`. |
| `hydropattern/formatters.py` | Added `compute_metric_series()`; threaded `mode`/`metric_mode` through `build_summary_sheet`, `write_summary`, `write_results`. |
| `hydropattern/cli.py` | Added `load_metric_options()`; wired `MetricOptions.mode` into `write_output` → `write_results`. |
| `tests/test_metric_options.py` | New: 14 tests covering defaults, valid modes, invalid values/types, unknown keys, and equivalence. |
| `tests/test_formatters.py` | Added `TestComputeMetricSeries` (7 tests) + 1 `build_summary_sheet` mode-threading test. |
| `tests/test_cli.py` | Added 3 `load_metric_options` tests + 1 `write_output` metric-mode wiring test. |
| `docs/user/reference.md` | Documented `[metric]` section, valid values, NA/zero policy, and `PARSER_UNKNOWN_OPTION`. |
