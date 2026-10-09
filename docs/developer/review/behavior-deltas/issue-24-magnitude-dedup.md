# Issue #24: Collapse magnitude dup path — Behavior Delta Report

## What changed

`hydropattern/parsing/builders.py::_build_characteristic`'s MAGNITUDE branch no longer
reimplements name-formatting and fx-construction (`patterns.comparison_fx` /
`patterns.magnitude_fx` + manual `f'{label}_...'` string building). It now converts the
`CharacteristicSpec` back into a raw metrics list and calls `parsers.magnitude_parser`
directly — the same function already used by the direct-parser path and by the CLI's
documented Python API (`docs/user/reference.md`).

No comparison logic, threshold semantics, moving-average handling, or naming convention
changed. The two code paths that previously computed the same thing two different ways
now compute it one way.

## Winner determination

`parsers.magnitude_parser` is the winner — it is the older, independently-tested,
publicly-exported function (used directly in `tests/test_parameter_validation.py`,
`tests/test_operator_normalization.py`, and documented in `docs/user/reference.md`).
The `builders.py` reimplementation was the newer, undocumented, untested-in-isolation
duplicate introduced during the `parsing/` package split (commit `1c86755`). Collapsing
onto the pre-existing, better-tested implementation is the safer direction.

## Before / after example

Config: `{'comp': {'magnitude': ['>', 5.0, 3]}}` (simple form, `ma_periods=3`)

- **Before**: `builders.py` built `comp_fx = patterns.comparison_fx('>', 5.0)` and
  `name = 'magnitude_>5.0'` locally, then called `patterns.magnitude_fx(comp_fx, order, 3)`.
- **After**: `builders.py` builds `metrics = ['>', 5.0, 3]` and calls
  `parsers.magnitude_parser(metrics, order=1)`, which performs the identical
  `comparison_fx`/`magnitude_fx`/name-construction internally.
- **Result**: `Characteristic.name` and evaluated `fx` output are byte-identical in both
  versions (proven by `tests/test_parser_builder_equivalence.py::TestMagnitudeEquivalence`,
  which predates this change and passes unmodified against it).

## Risks

- Low. The delegation only relocates *which* code path performs the construction; the
  underlying `comparison_fx`/`magnitude_fx`/naming logic in `parsers.py` is untouched.
- The remaining `import_module`/`getattr(parsers_module, 'magnitude_parser')` indirection
  (unchanged in this issue, deferred to #29) means this call is not statically typed —
  a rename of `magnitude_parser` in `parsers.py` would only fail at runtime, not at
  mypy-check time. This is a pre-existing risk, not introduced here.

## Proving tests

- `tests/test_magnitude_dedup.py` (new): asserts `build_components` calls
  `parsers.magnitude_parser` with the expected metrics/order for the simple form,
  between form, and non-default `ma_periods`.
- `tests/test_parser_builder_equivalence.py::TestMagnitudeEquivalence` (from #23,
  unmodified): asserts direct-parser and spec-path magnitude construction remain
  output-identical after this change.
- Full suite: 429 passed (1 pre-existing, unrelated `tkinter`/env failure in
  `test_cli.py::TestPlotComponents`, reproducible on `main` before this change too).
- `mypy hydropattern/`: 0 errors.

## Affected code locations

- `hydropattern/parsing/builders.py` — `_build_characteristic`, MAGNITUDE branch
- `tests/test_magnitude_dedup.py` — new
