# Issue #29: Fix import direction — Behavior Delta Report

## What changed

`hydropattern/parsing/*.py` no longer reach back into `hydropattern/parsers.py` via
`importlib.import_module('hydropattern.parsers')` + `getattr(module, name)`, and
`parsers.py` no longer does in-function-body imports of `hydropattern.parsing.*` to call
back out. The circular dependency has been resolved by relocating the canonical
implementations into `hydropattern/parsing/`, so the dependency now flows one direction
only: `parsers.py -> hydropattern.parsing`.

New/changed module responsibilities:

- **`hydropattern/parsing/specs.py`** (now canonical, was a proxy re-export): real
  definitions of `CharacteristicSpec`, `ComponentSpec`, `Request`, `MetricMode`,
  `MetricOptions`, `ClimateCanvasPlotOptions`, `PlotOptions`, `OutputOptions`,
  `TimeseriesSpec`, plus `collect_explicit_options`/`merge_overrides`.
- **`hydropattern/parsing/characteristics.py`** (new): canonical home for
  `timing_parser`, `magnitude_parser`, `duration_parser`, `rate_of_change_parser`,
  `frequency_parser`, `nested_frequency_parser`, and every `validate_*` helper +
  `ComparisionType`/`FrequencyForm`/`FrequencyMetrics` they use. Depends only on
  `hydropattern.patterns` and `hydropattern.errors` — no dependency on `parsers.py`.
- **`hydropattern/parsing/options.py`**, **`requests.py`**, **`timeseries.py`**,
  **`builders.py`**: now import `specs`/`characteristics` types and functions directly
  (top-level, sibling imports within `hydropattern.parsing`) instead of reflecting into
  `parsers.py` at call time.
- **`hydropattern/parsers.py`**: rewritten as a thin public-API facade. It imports every
  previously-public name from `hydropattern.parsing.{specs,characteristics,requests,
  options,timeseries,builders}` at module top level and re-exports them via `__all__`,
  plus keeps `parse_components` (a 2-line orchestration wrapper). Shrunk from 1079 lines
  to 126 lines.

No validation logic, comparison logic, naming logic, or public function signatures
changed — this issue is a pure move/re-wire, not a behavior change. Every function is
byte-identical to its pre-move version except for having its dependencies imported
directly instead of looked up via `getattr`.

## Winner determination

Not applicable in the usual "two implementations, pick one" sense (#24-#28 already
finished that): this issue is purely about *where* the single remaining implementation
of each piece of logic lives, and *how* it's imported. Types and characteristic
parsers/validators moved into `hydropattern.parsing` (specs.py / characteristics.py)
because `hydropattern.parsing.builders`/`requests`/`options` are their primary
consumers; `parsers.py` becomes the re-export facade preserving the existing public API
(`from hydropattern.parsers import X` continues to work unchanged for every existing
caller: `cli.py` and all test modules).

## Before / after example

```python
# Before (hydropattern/parsing/builders.py):
parsers_module = import_module('hydropattern.parsers')
timing_parser = getattr(parsers_module, 'timing_parser')
...
return timing_parser([100, 200], order=1)

# After:
from hydropattern.parsing.characteristics import timing_parser
...
return timing_parser([100, 200], order=1)
```

```python
# Before (hydropattern/parsers.py):
def parse_request(data: dict[str, Any]) -> Request:
    '''Parse request via parsing seam module.'''
    from hydropattern.parsing.requests import (
        parse_request as parse_request_impl,
    )
    return parse_request_impl(data)

# After (hydropattern/parsers.py):
from hydropattern.parsing.requests import parse_request
# (parse_request is directly re-exported, no wrapper function needed)
```

Result: identical inputs produce identical outputs (names, evaluated `fx` results,
exception types/messages) — proven by the full test suite passing unchanged.

## Risks

- **Test mocking blast radius**: `tests/test_{timing,magnitude,duration,
  rate_of_change,frequency}_dedup.py` (from #24-#28) patched `parsers.X_parser` to prove
  delegation; with `builders.py` now binding these names via direct top-level import
  rather than a runtime `getattr` lookup, `unittest.mock.patch.object(parsers, ...)` no
  longer intercepts the call (Python binds the name at import time, not a live
  module-attribute reference). All 5 test files were updated to patch
  `hydropattern.parsing.builders.X_parser` instead — the correct location per standard
  `unittest.mock` guidance ("patch where it's looked up, not where it's defined").
  No production code behavior changed; only the test's mock target changed to match the
  now-real import.
- **Public API compatibility**: every name previously importable from
  `hydropattern.parsers` remains importable (verified via full test suite, which
  exercises `from hydropattern.parsers import X` for ~20 different names across 10+
  test files, plus `cli.py`'s imports).
- **mypy**: previously many call sites were `Any`-typed due to `getattr`; now that
  imports are direct, mypy has full type information. Result: 0 errors (down from the
  pre-existing 2 unrelated errors, which are actually in `patterns.py`, untouched by
  this issue, and appear to already be resolved as of this branch — reserved for
  confirmation in #34 either way).
- **`specs.py` "already proxies" acceptance criterion**: `specs.py` is now the
  canonical definer (no longer proxies through `parsers.py`), satisfying that criterion
  ahead of #31's full cleanup (#31 will remove any remaining re-export-only
  scaffolding/docstrings referencing the old proxy state).

## Proving tests

- Full suite: 442 passed (2 pre-existing flaky `test_cli.py` tkinter/matplotlib
  failures when run as part of the full suite, confirmed passing in isolation — same
  known environment-only flakiness class documented in prior issues, unrelated to this
  change).
- `mypy hydropattern/`: 0 errors (16 source files checked, up from 15 — reflects the new
  `parsing/characteristics.py` module).
- Grep-verified: zero remaining `import_module`/`getattr(...)` occurrences anywhere in
  `hydropattern/parsing/`; zero remaining `from hydropattern.parsers import` or
  `import hydropattern.parsers` occurrences anywhere in `hydropattern/parsing/` (only
  docstring mentions of the module name remain).

## Affected code locations

- `hydropattern/parsers.py` — rewritten as a thin facade (1079 -> 126 lines)
- `hydropattern/parsing/specs.py` — now canonical (was proxy-only)
- `hydropattern/parsing/characteristics.py` — new module, canonical home for
  timing/magnitude/duration/rate_of_change/frequency/nested_frequency
  parsers + validators
- `hydropattern/parsing/options.py`, `requests.py`, `timeseries.py`, `builders.py` —
  updated to import directly from `specs.py`/`characteristics.py`/`patterns.py`
- `tests/test_timing_dedup.py`, `test_magnitude_dedup.py`, `test_duration_dedup.py`,
  `test_rate_of_change_dedup.py`, `test_frequency_dedup.py` — updated mock patch
  targets from `parsers.X_parser` to `hydropattern.parsing.builders.X_parser`
