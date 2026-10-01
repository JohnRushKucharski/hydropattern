# Pattern correctness: TDD implementation plan

**Status:** approved behavior, implementation not started. **Scope:** component
calculation, configuration, validation, tests, and user-facing documentation.
This document is a handoff for a new implementation session, not a description
of current behavior. Existing code, docs, and tests often implement the opposite
rules. Do not treat passing legacy tests as evidence that the new rules work.

## Decisions and acceptance contract

### Characteristics and composition

- Remove `verbose`. Timing, magnitude, and rate-of-change diagnostic columns
  report their own truth values, irrespective of preceding characteristic
  columns. Duration remains a run-length test over the conjunction of its
  preceding source conditions. Frequency remains a window test over the
  conjunction of its preceding source conditions. Distinguish independent
  diagnostic *display* from dependency semantics; frequency and duration
  cannot ignore the conditions they measure.
- Component characteristic sequence is inferred from input sequence, never
  specified with `order`. Component options (including `success_pattern`)
  never occupy a characteristic position. The nested-frequency pair occupies
  two output columns but one configured characteristic position.
- Accept both compact component TOML keys and an ordered array of
  characteristic tables indefinitely. Warn for compact keys: TOML v1.0 does
  not guarantee order of key/value pairs within a table, although this
  project's `tomllib` and Python dict preserve encountered order. TOML arrays
  guarantee element order. Library mappings use iteration order (caller must
  preserve it); library sequences preserve sequence order. Neither form
  accepts an explicit `order` setting. Preserve the old compact config where
  feasible, except for deliberate behavior changes documented below.
- A positive component succeeds when its conditions combine successfully;
  `success_pattern=false` describes a combined *failure* condition and
  reports its logical complement, **non-failure**, not affirmative ecological
  success. For A/B diagnostics `(1,1),(1,0),(0,1),(0,0)`, output is
  `[0,1,1,1]`. An unknown/insufficient verdict must not automatically
  become success or failure; specify and test three-valued logic.
- Reject components with no characteristics and invalid dependency structure.

Portable TOML illustration (new schema, not yet implemented):

```toml
[components.pulse]
success_pattern = true

[[components.pulse.characteristics]]
type = "timing"
metrics = [305, 335]

[[components.pulse.characteristics]]
type = "magnitude"
metrics = [">", 1.0]

[[components.pulse.characteristics]]
type = "frequency"
metrics = [">=", 1, 5, true]
```

Equivalent legacy compact form stays supported, with a portability warning:

```toml
[components.pulse]
success_pattern = true
timing = [305, 335]
magnitude = [">", 1.0]
frequency = [">=", 1, 5, true]
```

### Frequency windows

`[operator, n, N, exclusive_event_window]` and
`[min_n, max_n, N, exclusive_event_window]` evaluate *forward*, event-anchored
candidate windows of up to N timesteps, not trailing counts at the current
timestep. These are retrospective diagnoses: full windows may require future
observations before their verdict is known. An incomplete window at record
end is truncated and evaluated using available observations. Document that
this is not a real-time/causal-on-the-first-day prediction.

- Count successful **source timesteps**, not maximal runs. Source for a
  frequency characteristic is the conjunction of its earlier applicable
  conditions; source values of zero do not contribute to positive counts.
- For positive-count predicates, only source-success timesteps anchor candidate
  windows. For predicates admitting zero successes (e.g. `= 0`, `< 1`), every
  timestep can anchor a candidate window, so absence can be observed.
  Comparisons decide which candidate windows qualify; qualifying windows
  mark *every* timestep they span, including source-zero timesteps.
- `exclusive_event_window=false` is the new default. Successful overlapping
  candidate windows are unioned; a new source success can extend the marked
  span. `true` starts a fixed N-timestep marked span at the first qualifying
  anchor; later anchors within that span cannot extend it. A failed candidate
  does not suppress a later candidate. At a span's end, new candidate anchors
  become eligible. Test exact boundary and truncation behavior.
- Rename `event_bool` in parser/spec/builder/factory/names/docs to
  `exclusive_event_window`; distinguish it from `Result.event_count()`.
  Frequency's terminal diagnostic already includes its source-condition
  window rule: do **not** re-AND it with source success on each output day.
- Operator-aware validation must admit valid bounds at N (e.g. `>= N in N`,
  `= N in N`, between maximum N) and reject impossible predicates. Document
  that zero-count/absence forms anchor differently; handle `!=` based on
  whether zero satisfies the predicate.

Golden arrays, zero-based timestep positions; each frequency output is also
the component output when `success_pattern=true`:

| Source diagnostic | Frequency metrics | Expected frequency and component |
|---|---|---|
| `[1,0,0,0,0,0]` | `[">=",1,5]`, either exclusive flag | `[1,1,1,1,1,0]` |
| `[0,0,0,0,1,0]` | `[">=",1,5]`, either flag | `[0,0,0,0,1,1]` |
| `[0,0,0,0,0,1,0,0,0,0,0]` | `[">=",1,5]`, either flag | `[0,0,0,0,0,1,1,1,1,1,0]` |
| `[0,1,0,0,1,0,0,0,0,0]` | `[">=",1,5]`, flag false | `[0,1,1,1,1,1,1,1,1,0]` |
| `[0,1,0,0,1,0,0,0,0,0]` | `[">=",1,5,true]` | `[0,1,1,1,1,1,0,0,0,0]` |
| `[1,0,1,0,0,0,0]` | `[">=",2,5,true]` | `[1,1,1,1,1,0,0]` |

Add independent oracle tests for zero-qualifying operators, failed then
qualifying candidates, inclusive between bounds, positive `n>1`, `N=1`,
contiguous successes, end-of-record truncation, and suppression boundary.
Do not make unit tests merely reimplement the production algorithm.

### Nested frequency

Intra-annual probability `[operator, p]` = number of eligible **timesteps**
divided by valid timesteps within a **complete** water year. It is not number
of distinct events divided by year length. Compare the probability once per
year, then broadcast the annual verdict over that water year's full output
column. Never gate the year-end verdict on whether its last timestep was
eligible. `exclusive_event_window` has no window overlap to suppress for a
one-year probability form; document its applicability/validation explicitly
before implementing that edge case rather than silently changing meaning.

Outer frequency consumes one intra-annual verdict per complete water year,
uses the same forward-window, overlap, exclusion, and record-end rules **in
units of water years**, and broadcasts verdicts back to the corresponding
rows. A full calendar/record water year is required for an annual trial;
partial N-year *outer windows* at the end can still be evaluated over the
available complete annual trials. Do not confuse those two policies.

For four-step water years and `[[">=",0.5],[">=",1,2]]`:

```text
magnitude      [1,1,1,0, 0,0,0,0, 1,0,1,0]
base fraction  [3/4,     0/4,     2/4    ]
intra-annual   [1,1,1,1, 0,0,0,0, 1,1,1,1]
interannual    [1,1,1,1, 1,1,1,1, 1,1,1,1]
component      [1,1,1,1, 1,1,1,1, 1,1,1,1]
```

With years 2 and 3 flipped, annual base verdicts `[1,1,0]`; outer default
still gives `[1,1,1]`. Outer `[">=",1,2,true]` gives `[1,1,0]`, hence
component `[1,1,1,1, 1,1,1,1, 0,0,0,0]`. Test both cases end to end.

### Other scientific contracts

- Timing thresholds specify **calendar** day-of-year, even for an October
  water-year start. Maintain the repository's stated leap-day convention and
  test February 28/29 and wrapped calendar timing ranges.
- Rate of change divides current raw or smoothed value by lagged raw or
  smoothed value. `minimum` applies to the denominator only; e.g.
  `flow=[4,1]`, `minimum=1`, comparison `<0.5` gives second-step success.
  Define tests for zero/negative denominators and smoothing separately.
- Exclude *both* leading and trailing partial water years from annual
  statistics and event-rate exposure; DOWY `== 1` alone does not prove the
  trailing year complete. Choose cadence-aware completeness from actual
  timestamps/observations, with explicit daily/monthly tests and errors for
  invalid or unsupported gaps.
- `evaluate_component(..., data_column=0)` selects a zero-based data-column
  index, excluding final DOWY column; `evaluate_components` forwards it.
  Reject negative, out-of-range, non-data positions and invalid last DOWY.
  Default retains first-data-column calculation and existing output shape.
  Proposed multi-column result: preserve unselected data columns, rename
  only selected column to `dv`, and retain `dv_name`; test output shape,
  ordering, collisions, and index preservation before implementation. CLI
  scenario splitting still passes one selected series at a time.

## Execution sequence: red -> green -> refactor per slice

Do not add all production changes at once. For each slice, write failing
acceptance/unit tests first, confirm failure for the intended reason, make
the smallest fix, rerun the targeted tests, refactor, run the relevant
integration tests, and record any intentional behavior delta. Full test suite,
type check, docs examples, and downstream smoke checks at phase boundaries.

| Phase | Test-first change | Main implementation surfaces | Gate |
|---|---|---|---|
| 0. Baseline | Capture existing results, parser/API snapshots and supplied golden arrays (expected red). | `tests/test_patterns.py`, `tests/test_stable_request_shape.py`, `tests/test_cli.py` | Baseline tests pass; new golden tests fail for known reasons. |
| 1. Schema/order | Test compact/list equivalence, option placement, warning, library iteration order, direct factory compatibility, independent base diagnostics, duration dependencies. | `hydropattern/parsing/specs.py`, `requests.py`, `builders.py`, `characteristics.py`, `hydropattern/parsers.py`, `patterns/core.py`, `patterns/characteristics.py` | No user `order` or `verbose`; valid existing compact configs still load (warning); unrelated diagnostics stable. |
| 2. Frequency | Test all six golden rows and edge cases; test component terminal dispatch. | `patterns/characteristics.py`, `patterns/core.py`, `parsing/characteristics.py`, `specs.py`, `builders.py` | Count/between modes and renamed flag match source-window contracts. |
| 3. Nested frequency | Test exact two 3-year matrices above, time fraction, full/partial water years and annual broadcasts. | `patterns/water_year.py`, `patterns/characteristics.py`, `patterns/core.py` | No lost annual success; inner/outer outputs match hand oracle. |
| 4. Component logic | Test A/B truth table and unknown propagation, including nested/unnested frequency. | `patterns/core.py` | Failure means non-failure of combined bad pattern, not all-bad-conditions-absent. |
| 5. Time handling | Test October-start timing, leap day, water-year completeness and year labeling. | `timeseries.py`, `patterns/water_year.py`, `patterns/characteristics.py`, `patterns/core.py`; evaluate formatter consumers | Timing uses calendar DOY; annual stats exclude partial years. |
| 6. Rate/validation | Test denominator-only minimum, attainable frequency bounds, empty/invalid components. | `patterns/characteristics.py`, `parsing/characteristics.py`, `parsing/requests.py`, `builders.py` | Invalid inputs produce repository-standard errors; no silently impossible configurations. |
| 7. Series selection | Test default/intermediate/invalid data-column indices across direct API and scenario seam. | `patterns/core.py`, `scenarios.py` if needed | Selected series drives every characteristic, default behavior preserved. |
| 8. Docs and release | Validate all published simple and nested examples against tests; migration and ADR review. | Files listed below | Docs accurately describe implemented output, warning, metrics and limitations. |

### Phase 8 documentation checklist (mandatory, not optional cleanup)

- `README.md`: brief CLI + Python examples with known input/output arrays;
  link complete frequency reference and show compact vs ordered TOML.
- `docs/user/reference.md`: complete frequency chapter: supported forms,
  operators, inclusive bounds, valid/invalid N/n combinations,
  timestep vs year units, source/anchor definition, forward window and
  retrospective interpretation, truncation, default/true
  `exclusive_event_window`, zero-count predicates, annual probability
  denominator, nested broadcasting, `success_pattern=false`, missing and
  partial-year treatment, data-column selection, and library API.
  Include six golden un-nested arrays and both nested 3-year matrices.
  Remove obsolete `verbose`, old trailing-history language, and incorrect
  denominator-floor claims; label examples by actual units, not assumed
  years. Document output-column shape and conditional dependencies.
- `docs/adr/`: **supersede** ADR 0002's accepted *trailing* window rule with
  a new ADR detailing forward source-anchored windows, exclusivity,
  noncausal retrospective classification, alternatives and implications.
  Link supersession from ADR 0002; retain historical text rather than
  overwriting it. Record probability, water-year completeness, output truth,
  and config ordering decisions in new ADR(s) if not covered by that ADR.
  Keep ADR 0001's inclusive-between decision consistent.
- `examples/minimal.toml`, `examples/detailed.toml`, and added focused
  frequency example config/data if useful: working compact and ordered
  forms, default/true exclusivity, nested probability. Update outdated
  comments and generated illustrations. Update `CONTEXT.md` only if it
  contains domain rules now changed.
- Release/migration notes: current `event_bool` defaults true, new
  `exclusive_event_window` defaults false; old vs new output shapes
  and statistics, config warning, removed `verbose`/`order` API,
  failure-pattern truth change, timing/calendar behavior, partial-year
  exclusions. Avoid silent defaults for unsupported old options. Mention
  that historical frequency outputs/return-period interpretations may
  shift; do not claim a Poisson recurrence probability.
- Add docs tests or reproducible golden fixtures for published examples so
  docs cannot drift from code. Revisit `docs/agents/` only when onboarding
  guidance references removed behavior.

### Verification, impact, rollback

Use `uv run pytest` on the affected tests first, then full suite; run
`uv run mypy hydropattern/` and existing lint where relevant. Add `coverage`
and `radon` to the dev group **only after a manifest update is justified**;
define and measure core branch-coverage and CRAP baselines in CI, not a
threshold inferred from outdated tests. Verify CLI `examples/minimal.toml`,
ordered and compact example parity, library direct calls, multi-scenario
results, CSV/Excel output column names, formatter portion/percentage/
return-period behavior, and plotting/color conventions. Do not change
reporting metrics as an unreviewed side effect. If existing results change,
compare to the explicit golden matrices rather than restoring legacy
behavior by accident.

The worktree may contain other uncommitted work. Before every slice, check
`git status`, keep slices isolated, avoid overwriting unrelated edits.
If a gate fails, stop that slice, diagnose it, and revert **only** its own
changes if necessary; never reset the entire worktree. Do not conflate
reporting/event-count redesign with the correctness migration.

## Deferred TODO: separate reporting plan after phases 0-8

Interview user **after** preceding behavior is implemented and documented:
does `Result.event_count()` count original hydrologic source events,
maximal runs of frequency-valid component output, or fixed exclusive
windows? How should a nested year-grain event and failure/non-failure
component count? Define denominators/exposure for `event_rate`; avoid
implied Poisson recurrence intervals from `1 / portion`.

Then create a **separate** TDD/reporting plan: fix `Result.identify_water_years`
and `frequency_table(by_water_years=True)` grouping/labeling for non-Jan-1
water years using the canonical grouping contract; add reporting tests and
review formatter metrics, table labels, zero/NA treatment, and event_count
up/downstream consumers. `Result.frequency_table` currently has CC 6,
0% branch-inclusive coverage, CRAP 42 under the reviewed full test run:
prioritize coverage and statistical interpretation. No implementation of
this deferred event-count/reporting policy before interview.

## New-session handoff

Start fresh agent session with:

> Implement `docs/plans/2026-10-01-pattern-correctness-tdd.md` one TDD
> phase at a time, starting phase 0 then phase 1. Read entire plan and
> existing dirty-worktree diff first; do not overwrite unrelated changes.
> Use red-green-refactor and report each phase's golden output and behavior
> delta. Update mandatory README, user reference, ADRs, examples, and
> migration notes by phase 8. Do not implement deferred reporting/event-count
> TODO; interview me after phases 0-8 and create a separate plan. Stop to
> clarify genuinely unresolved semantics before altering public behavior.

This is a multi-phase change. A new session can implement one phase and
use this same path as durable handoff for subsequent sessions.
