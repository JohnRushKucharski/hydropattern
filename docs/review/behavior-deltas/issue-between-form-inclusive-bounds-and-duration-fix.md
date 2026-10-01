# Between-form bounds made inclusive + duration between-form bug fix — Behavior Delta Report

## What changed

1. **Behavior change (breaking, intentional):** `between_parser`'s `inclusive=True`
   default is now actually used by every characteristic that offers a between
   ("[min, max]") form. `magnitude_parser` and `rate_of_change_parser` previously
   called `between_parser(metrics[0:2], inclusive=False)` (strict `min < n < max`);
   they now call `between_parser(metrics[0:2])` (inclusive `min <= n <= max`).
   This resolves the follow-up flagged in
   `docs/adr/0001-frequency-between-form-inclusive-bounds.md` ("A follow-up task
   should revisit changing the shared `between_parser` default to `inclusive=True`
   everywhere ... to remove the inconsistency").
2. **Bug fix:** `duration_parser`'s between-form built its comparison incorrectly:
   `patterns.comparison_fx('<', metrics[0], '>', metrics[1])` evaluates to
   `(min < n) and (n > max)`, which collapses to just `n > max` — an unbounded
   upper-only test. A `duration = [36, 60]` spec silently became "duration > 60,
   no upper limit," while true 36-60 timestep cycles were excluded entirely.
   Fixed by switching to `between_parser(metrics[0:2])`, the same construction
   magnitude/rate_of_change use — one code path for all three between-forms now.
3. `timing`'s between form was already inclusive at both bounds via its own
   direct `comparison_fx('<=', first_doy, '<=', last_doy)` construction (never
   routed through `between_parser`) — confirmed unaffected, added a regression
   test to lock this in.

## Winner determination

Not a dedup issue; this is a discovered defect (duration) plus a planned,
previously-deferred consistency fix (inclusive default). No competing
implementation — `between_parser` itself is untouched; only its callers' explicit
`inclusive=False` overrides were removed (magnitude, rate_of_change), and
duration's hand-rolled comparison was replaced with a call to the same
`between_parser` helper.

## Before / after example

Config: `duration = [36, 60]` (`[min_steps, max_steps]`, order=2, following a
magnitude characteristic at order 1)

- **Before**: comparison function was effectively `n > 60` (unbounded). A
  continuous 48-timestep qualifying run (inside the intended [36, 60] band) was
  never marked a success; a 102-timestep run (well outside/above the band) was
  incorrectly marked a success for its full length.
- **After**: comparison function is `36 <= n <= 60`. The 48-timestep run is
  marked a success for its full length; the 102-timestep run is not marked at
  all.
- Verified against `examples/longtailpoint/michiganhuron_avg.csv`
  (`low_water_cycle` component, `duration = [36, 60]`): before the fix, 8 runs
  of length 61-102 were (wrongly) counted as successes (portion ≈ 5.2%); after
  the fix, 10 genuine 36-60-timestep runs are counted (portion ≈ 3.6%).

Config: `magnitude = [0.5, 5.0]` (order=1)

- **Before**: `flow == 0.5` and `flow == 5.0` did not qualify (strict `<`).
- **After**: `flow == 0.5` and `flow == 5.0` qualify (`<=`). Same change applies
  to `rate_of_change`'s between form.

## Risks

- **Breaking, by design, for magnitude/rate_of_change**: any existing config
  relying on strict exclusion of the exact boundary value for a between-form
  magnitude or rate_of_change characteristic will now include that boundary
  value. Silent at the boundary — no error is raised, output changes only for
  timesteps exactly equal to `min`/`max`.
- **Compatibility audit performed** (see session record) across the three repos
  that could plausibly be affected:
  - `climate-canvas`: no references to `between_parser`/`comparison_fx`/any
    characteristic parser — consumes generic response-surface grids only. No risk.
  - `hydropattern-gui`: invokes hydropattern only via CLI subprocess; its
    `*_toml_integration.py` tests round-trip GUI form state -> TOML text only,
    never execute hydropattern's evaluation or assert numeric output. No risk.
  - `hydropattern` (this repo): no existing test asserted exact-boundary numeric
    equality for any between-form (magnitude/rate_of_change/duration) prior to
    this change — confirmed by full-suite pass with zero modifications needed
    to any pre-existing test.
- Low risk for duration: this is a straight bug fix; the "before" behavior was
  never a documented/intended contract (docs already said "3 to 14 timesteps" /
  "between 3 and 60 months", not "duration > 60").

## Proving tests

- `tests/test_between_inclusive_bounds.py` (new): 7 tests —
  `between_parser` default-inclusive + explicit `inclusive=False` still works,
  magnitude/rate_of_change between-form boundary inclusion, duration between-form
  boundary correctness + end-to-end (`evaluate_component`) qualifying-vs-exceeding
  run behavior, timing between-form unaffected regression.
- Full suite: 472 passed, 0 modified, 0 skipped.

## Affected code locations

- `hydropattern/parsing/characteristics.py` — `magnitude_parser`, `duration_parser`,
  `rate_of_change_parser` (BETWEEN branches) + docstrings.
- `docs/user/reference.md` — magnitude/duration/rate_of_change between-form tables
  and examples updated from "exclusive" to "inclusive".
- `docs/adr/0001-frequency-between-form-inclusive-bounds.md` — status note added;
  the deferred follow-up it names is now done.
- `tests/test_between_inclusive_bounds.py` — new.
