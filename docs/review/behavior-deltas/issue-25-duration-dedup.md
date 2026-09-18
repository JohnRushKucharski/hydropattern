# Issue #25: Collapse duration dup path — Behavior Delta Report

## What changed

`hydropattern/parsing/builders.py::_build_characteristic`'s DURATION branch no longer
reimplements name-formatting and fx-construction (`patterns.comparison_fx` /
`patterns.duration_fx` + manual `f'{label}_...'` string building). It now converts the
`CharacteristicSpec` back into a raw `[operator, threshold]` or `[min, max]` metrics list
and calls `parsers.duration_parser` directly — the same function already used by the
direct-parser path and documented in `docs/user/reference.md`.

No comparison logic or naming convention changed. The two code paths that previously
computed the same thing two different ways now compute it one way.

## Winner determination

`parsers.duration_parser` is the winner — same rationale as #24: it is the older,
independently-tested, publicly-exported function (`tests/test_parameter_validation.py`),
while the `builders.py` reimplementation was the newer, untested-in-isolation duplicate
from the `parsing/` package split (commit `1c86755`).

## Before / after example

Config: `{'comp': {'magnitude': ['>', 5.0], 'duration': ['>', 1]}}` (simple form, duration
follows magnitude at order 2)

- **Before**: `builders.py` built `comp_fx = patterns.comparison_fx('>', 1)` and
  `name = 'duration_>1'` locally, then called `patterns.duration_fx(comp_fx, order=2)`.
- **After**: `builders.py` builds `metrics = ['>', 1]` and calls
  `parsers.duration_parser(metrics, order=2)`, which performs the identical
  `comparison_fx`/`duration_fx`/name-construction internally.
- **Result**: `Characteristic.name` and evaluated `fx` output are byte-identical in both
  versions (proven by `tests/test_parser_builder_equivalence.py::TestDurationEquivalence`,
  which predates this change and passes unmodified against it).

## Risks

- Low. Only relocates construction path; underlying `comparison_fx`/`duration_fx`/naming
  logic in `parsers.py` is untouched.
- Same pre-existing `import_module`/`getattr` indirection risk as #24 (deferred to #29):
  not statically typed, unchanged by this issue.

## Proving tests

- `tests/test_duration_dedup.py` (new): asserts `build_components` calls
  `parsers.duration_parser` with expected metrics/order for simple and between forms.
- `tests/test_parser_builder_equivalence.py::TestDurationEquivalence` (from #23,
  unmodified): asserts direct-parser and spec-path duration construction remain
  output-identical after this change.
- Full suite: 431 passed (1 flaky, env-only `test_cli.py` failure reproduced only
  under full-suite run, passes in isolation — pre-existing matplotlib/tkinter
  test-order flakiness in this environment, unrelated to this change).
- `mypy hydropattern/`: 0 errors.

## Affected code locations

- `hydropattern/parsing/builders.py` — `_build_characteristic`, DURATION branch
- `tests/test_duration_dedup.py` — new
