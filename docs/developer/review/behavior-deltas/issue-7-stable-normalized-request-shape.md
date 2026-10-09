# Behavior Delta Report — Issue #7: Stable Normalized Internal Request Shape

## 1. What changed in behavior

### Before (baseline)
`parse_components(data)` directly builds executable `Characteristic` namedtuples
(each containing a callable closure `fx`) and returns `list[Component]`.  No
intermediate pure-data representation existed.  There was no way to compare two
configurations for equivalence without running them and diffing outputs.

### After (new behavior)
Two-stage pipeline:

1. `parse_request(data) → Request` — validates and normalizes input into a
   pure-data graph of frozen dataclasses (`Request`, `ComponentSpec`,
   `CharacteristicSpec`).  No closures.  Comparable via `==`.
2. `build_components(request) → list[Component]` — converts the `Request` into
   the same executable `Component` objects as before.

`parse_components` now delegates to `build_components(parse_request(data))`,
preserving the old call signature.

`cli.py`'s `load_components` now calls `parse_request` + `build_components`
directly instead of `parse_components`.

## 2. Which behavior is correct (new vs baseline)?

**New behavior is correct and strictly better.**

- The old pipeline mixed normalization and closure construction into one step,
  making it impossible to inspect or compare the parsed structure without
  executing it.
- The new pipeline separates concerns: normalization (parser contract) from
  compilation (builder step).
- All existing tests continue to pass; evaluation results are identical
  (proven by round-trip tests).

## 3. Synthetic test case demonstrating the delta

```python
from hydropattern.parsers import parse_request, build_components
from hydropattern.patterns import evaluate_components
import pandas as pd, numpy as np

# Two configs that differ only in operator whitespace.
config_a = {'comp': {'magnitude': ['>', 1.0]}}
config_b = {'comp': {'magnitude': [' > ', 1.0]}}

# Before: no way to compare configs without evaluating both.
# After: direct structural equality.
assert parse_request(config_a) == parse_request(config_b)  # NEW ✓

# Round-trip: results are identical.
df = pd.DataFrame({'flow': [1.0, 1.2, 0.8], 'dowy': [1.0, 2.0, 3.0]},
                  index=pd.to_datetime(['2020-01-01','2020-01-02','2020-01-03']))
df.index.name = 'time'

res_a = evaluate_components(df, build_components(parse_request(config_a)))
res_b = evaluate_components(df, build_components(parse_request(config_b)))

np.testing.assert_array_equal(res_a[0].df['comp'].values,
                               res_b[0].df['comp'].values)  # [0, 1, 0] for both
```

## 4. Regression risk notes

- **Low risk.**  `parse_components` is now a thin delegate — all existing call
  sites continue to work unchanged.
- The spec-builder functions `_timing_spec`, `_magnitude_spec`, etc. share the
  same validation functions as the old per-type parsers, so all existing
  validation error codes and messages are preserved exactly.
- The builder `_build_characteristic` calls the same `patterns.*_fx` functions
  as the old parsers, producing functionally identical closures.
- One subtle existing behavior preserved: `duration` BETWEEN comparison uses
  `patterns.comparison_fx('<', min, '>', max)` (matching `duration_parser`),
  NOT the symmetric `between_parser` form used by magnitude/rate_of_change.
  This is intentional reproduction of existing behavior, not a new bug.

## 5. Affected code locations

| File | Change |
|------|--------|
| `hydropattern/parsers.py` | Added `CharacteristicSpec`, `ComponentSpec`, `Request` dataclasses; added `_timing_spec`, `_magnitude_spec`, `_duration_spec`, `_rate_of_change_spec`, `_frequency_spec` spec builders; added `parse_request`, `_build_characteristic`, `build_components`; `parse_components` now delegates |
| `hydropattern/cli.py` | `load_components` updated to call `build_components(parse_request(...))` directly |
| `tests/test_stable_request_shape.py` | New: 43 tests covering equality, golden snapshots, equivalence, negatives, builder, and round-trip |
