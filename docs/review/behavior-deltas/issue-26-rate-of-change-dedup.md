# Issue #26: Collapse rate_of_change dup path — Behavior Delta Report

## What changed

`hydropattern/parsing/builders.py::_build_characteristic`'s RATE_OF_CHANGE branch no
longer reimplements name-formatting and fx-construction (`patterns.comparison_fx` /
`patterns.rate_of_change_fx` + manual `f'{label}_...'` string building). It now converts
the `CharacteristicSpec` back into a raw metrics list — `[operator, threshold]` or
`[min, max]`, followed by any non-default `ma_periods`/`look_back`/`min_val` (positional,
so trailing defaults are trimmed but interior ones are preserved) — and calls
`parsers.rate_of_change_parser` directly.

No comparison logic, moving-average handling, look-back semantics, or naming convention
changed.

## Winner determination

`parsers.rate_of_change_parser` is the winner — same rationale as #24/#25: older,
independently-tested (`tests/test_parameter_validation.py`), publicly documented in
`docs/user/reference.md`. The `builders.py` reimplementation was the newer, untested-in-
isolation duplicate from the `parsing/` package split (commit `1c86755`).

## Before / after example

Config: `{'comp': {'rate_of_change': ['>', 0.5, 1, 1, 0.2]}}` (simple form, non-default
`min_val=0.2`, default `ma_periods=1`/`look_back=1` retained for positional correctness)

- **Before**: `builders.py` built `comp_fx = patterns.comparison_fx('>', 0.5)` and
  `name = 'rate_of_change_>0.5'` locally, then called
  `patterns.rate_of_change_fx(comp_fx, order, ma_periods=1, look_back=1, min_val=0.2)`.
- **After**: `builders.py` builds `metrics = ['>', 0.5, 1, 1, 0.2]` and calls
  `parsers.rate_of_change_parser(metrics, order=1)`, which performs the identical
  construction internally.
- **Result**: `Characteristic.name` and evaluated `fx` output are byte-identical in both
  versions (proven by
  `tests/test_parser_builder_equivalence.py::TestRateOfChangeEquivalence`, which predates
  this change and passes unmodified against it).

## Risks

- Low. Only relocates construction path; underlying `comparison_fx`/`rate_of_change_fx`/
  naming logic in `parsers.py` is untouched.
- Positional-optional-args handling (ma_periods/look_back/min_val) is the trickiest part
  of this dedup — covered explicitly by
  `test_trailing_optional_args_trimmed_when_default` and
  `test_all_optional_args_forwarded_when_non_default` in the new test file, in addition
  to the #23 equivalence test.
- Same pre-existing `import_module`/`getattr` indirection risk as #24/#25 (deferred to
  #29): not statically typed, unchanged by this issue.

## Proving tests

- `tests/test_rate_of_change_dedup.py` (new): asserts `build_components` calls
  `parsers.rate_of_change_parser` with expected metrics/order for simple form, between
  form, trailing-default trimming, and full non-default forwarding.
- `tests/test_parser_builder_equivalence.py::TestRateOfChangeEquivalence` (from #23,
  unmodified): asserts direct-parser and spec-path rate_of_change construction remain
  output-identical after this change.
- Full suite: 435 passed (1 flaky, env-only `test_cli.py` failure reproduced only under
  full-suite run, passes in isolation — same pre-existing matplotlib/tkinter test-order
  flakiness seen in #24/#25, unrelated to this change).
- `mypy hydropattern/`: 0 errors.

## Affected code locations

- `hydropattern/parsing/builders.py` — `_build_characteristic`, RATE_OF_CHANGE branch
- `tests/test_rate_of_change_dedup.py` — new
