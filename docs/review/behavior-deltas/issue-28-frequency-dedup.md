# Issue #28: Collapse frequency + nested_frequency dup path — Behavior Delta Report

## What changed

`hydropattern/parsing/builders.py::_build_characteristic`'s FREQUENCY branch no longer
reimplements name/fx construction for the COUNT and BETWEEN forms. It now builds the
metrics list the legacy TOML-style parser expects and calls `parsers.frequency_parser`
directly. `_build_nested_frequency_characteristics` already delegated to
`parsers.nested_frequency_parser` but reached it via `importlib.import_module` + `getattr`
string lookup; it now imports `nested_frequency_parser` directly at module scope and
calls it without the reflection indirection.

The un-nested PROBABILITY form branch (`operator` set, `big_n is None`) is **kept as-is,
not delegated**. `requests.py` already calls `validate_frequency_metrics` without
`allow_probability=True` for un-nested specs, so a standalone PROBABILITY spec can never
reach `_build_characteristic` through the public parsing seam — `frequency_parser` would
reject it outright. Delegating anyway would turn genuinely dead/defensive code into a
behavior change (a hard error) for what is currently silent, safe, unreachable code. Left
untouched to guarantee zero behavior change.

## Winner determination

`parsers.frequency_parser` / `parsers.nested_frequency_parser` are the winners — same
rationale as #24-#27: older, independently covered by `tests/test_frequency_parsing.py`
(64 cases), documented, and already the thing `_build_nested_frequency_characteristics`
called (just through unnecessary reflection). The `_build_characteristic` FREQUENCY
branch's inline BETWEEN/COUNT construction was the newer, untested-in-isolation
duplicate introduced by the `parsing/` package split (commit `1c86755`).

## Before / after example

Config: `{'comp': {'magnitude': ['>', 5.0], 'frequency': ['>', 1, 3]}}` (COUNT form)

- **Before**: `builders.py` built `comp_fx = patterns.comparison_fx('>', 1)` and
  `name = f'frequency_gt1in3(event)'` inline, then called `patterns.Characteristic(...)`
  directly.
- **After**: `builders.py` builds `metrics = ['>', 1, 3]` and calls
  `parsers.frequency_parser(metrics, order=2)`, which performs the identical
  comparison/name construction internally via `_frequency_comparison_and_label`.
- **Result**: `Characteristic.name` and evaluated `fx` output are identical in both
  versions (proven by
  `tests/test_parser_builder_equivalence.py::TestFrequencyEquivalence`/`TestNestedFrequencyEquivalence`,
  unchanged, still passing).

Nested case: `_build_nested_frequency_characteristics` output is byte-identical before
and after — only the *how* of reaching `nested_frequency_parser` changed (direct import
vs. `import_module`+`getattr`), not any logic.

## Risks

- Low for BETWEEN/COUNT: only relocates construction path, verified equivalent by #23's
  characterization tests plus the pre-existing `test_frequency_parsing.py` suite (64
  tests, unchanged).
- Zero for nested: no logic touched, only import mechanism.
- The un-nested PROBABILITY branch is intentionally left un-migrated (see rationale
  above) — flagged for visibility, not a partial fix; revisit only if `requests.py`'s
  validation ever changes to permit standalone probability specs.
- Same pre-existing `import_module`/`getattr` indirection in `_build_characteristic`'s
  top-of-function bindings (timing/magnitude/duration/rate_of_change/frequency parsers)
  remains, deferred to #29 (out of scope here — only
  `_build_nested_frequency_characteristics`'s reflection was targeted per issue #28's
  acceptance criteria).

## Proving tests

- `tests/test_frequency_dedup.py` (new): 4 tests — COUNT form, BETWEEN form, non-default
  `event_bool` forwarding all delegate to `parsers.frequency_parser`; nested frequency
  delegates to `parsers.nested_frequency_parser` via direct call (patched at
  `hydropattern.parsing.builders.nested_frequency_parser`, confirming the import is now a
  direct binding, not a runtime `getattr` lookup).
- `tests/test_parser_builder_equivalence.py::TestFrequencyEquivalence` /
  `TestNestedFrequencyEquivalence` (from #23, unmodified): still pass.
- `tests/test_frequency_parsing.py` (pre-existing, unmodified): all 64 pass unchanged.
- Full suite: 441 passed, 1 pre-existing flaky `test_cli.py` tkinter/matplotlib failure
  (confirmed passes in isolation — same known environment-only flakiness class as prior
  issues, unrelated to this change).
- `mypy hydropattern/`: 0 errors.

## Affected code locations

- `hydropattern/parsing/builders.py`
  - `_build_characteristic`, FREQUENCY branch (BETWEEN/COUNT forms delegate to
    `parsers.frequency_parser`; PROBABILITY form left as defensive fallback)
  - `_build_nested_frequency_characteristics` (direct import instead of
    `import_module`/`getattr`)
- `tests/test_frequency_dedup.py` — new
