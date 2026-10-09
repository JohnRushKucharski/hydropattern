# Issue #27: Collapse timing dup path — Behavior Delta Report

## What changed

`hydropattern/parsing/builders.py::_build_characteristic`'s TIMING branch no longer
reimplements name-formatting and fx-construction (`timing_window_fx` +
`patterns.timing_fx` + manual `f'{label}_...'` string building). It now calls
`parsers.timing_parser` directly with `[first_doy, last_doy]`, the same function used by
the direct-parser path and documented in `docs/user/reference.md`.

No window-comparison logic (including cross-year wrap-around handling) or naming
convention changed.

## Winner determination

`parsers.timing_parser` is the winner — same rationale as #24/#25/#26: older,
independently-tested (`tests/test_parameter_validation.py`), publicly documented. The
`builders.py` reimplementation was the newer, untested-in-isolation duplicate from the
`parsing/` package split (commit `1c86755`). This is also the simplest of the five dedups
— timing metrics have no optional/positional arguments to reconstruct.

## Before / after example

Config: `{'comp': {'timing': [335, 60]}}` (cross-year wrap-around window)

- **Before**: `builders.py` called `timing_window_fx(335, 60)` locally and built
  `name = 'timing_335-60'`, then called `patterns.timing_fx(window_fx, order=1)`.
- **After**: `builders.py` calls `parsers.timing_parser([335, 60], order=1)`, which
  performs the identical `timing_window_fx`/`timing_fx`/name-construction internally.
- **Result**: `Characteristic.name` and evaluated `fx` output (including wrap-around
  day-of-year matching) are byte-identical in both versions (proven by
  `tests/test_parser_builder_equivalence.py::TestTimingEquivalence`, which predates this
  change and passes unmodified against it).

## Risks

- Low. Only relocates construction path; underlying `timing_window_fx`/`timing_fx` logic
  in `parsers.py` is untouched. No positional-optional-argument complexity (unlike #26).
- Same pre-existing `import_module`/`getattr` indirection risk as #24-#26 (deferred to
  #29): not statically typed, unchanged by this issue.
- The now-unused `timing_window_fx` import in `builders.py` was removed; confirmed no
  other branch in this module referenced it.

## Proving tests

- `tests/test_timing_dedup.py` (new): asserts `build_components` calls
  `parsers.timing_parser` with expected metrics/order for a standard window and a
  cross-year wrap-around window.
- `tests/test_parser_builder_equivalence.py::TestTimingEquivalence` (from #23,
  unmodified): asserts direct-parser and spec-path timing construction remain
  output-identical after this change.
- Full suite: 438 passed, 0 failed (no flaky test triggered this run).
- `mypy hydropattern/`: 0 errors.

## Affected code locations

- `hydropattern/parsing/builders.py` — `_build_characteristic`, TIMING branch
- `tests/test_timing_dedup.py` — new
