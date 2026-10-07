# Pattern correctness and evaluation contract

**Status:** accepted

**Partial supersession:** [ADR 0004](0004-reporting-and-unknown-outcomes.md)
records the next design's changes to annual probability denominators,
event-rate exposure, omitted leap days, and reciprocal reporting.
Its implementation is pending; the historical decisions below are retained.

**Supersedes:** [ADR 0002](0002-frequency-sliding-window.md)'s trailing
frequency-window decision. The historical ADR is retained as a record of the
previous behavior.

## Context

Pattern outcomes need to be interpretable as ecological classifications, while
remaining reproducible across component configurations and time-series shapes.
The former trailing-window frequency behavior, incomplete-year handling,
ambiguous compact configuration order, and generic result data-column name
made results difficult to interpret consistently.

## Decisions

### Frequency and probability

- Un-nested count and between frequency forms evaluate forward candidate
  windows from eligible source timesteps. The count is eligible timesteps,
  not distinct runs.
- The source is the conjunction of applicable preceding characteristics.
  Positive-count candidates anchor at source successes; predicates that admit
  zero may anchor at every timestep. A qualifying window marks all timesteps
  in its span, including source-zero positions.
- Windows at record end truncate to available observations. The result is a
  retrospective classification, not a real-time prediction at the anchor.
- `exclusive_event_window` defaults to `false`: overlapping qualifying
  windows are unioned. When true, a qualifying anchor claims a fixed-length
  span; later anchors in that span are suppressed. Failed candidates do not
  suppress later anchors.
- Nested base probability is eligible timesteps divided by valid timesteps in
  a complete water year. Compare once and broadcast the verdict across the
  year. The probability trial itself has no window overlap, so its exclusivity
  flag has no effect. The nested outer frequency operates over annual verdicts
  in units of complete water years.
- Between-form bounds are inclusive. Impossible count predicates and
  unattainable thresholds are rejected.

### Time completeness and timing

- Timing uses calendar day-of-year, including for October-start water years.
  The existing 365-day convention is retained: February 28 and 29 share a
  timing position.
- Annual probability and event-rate exposure exclude both leading and trailing
  partial water years. With datetime timestamps, supported daily or monthly
  cadence and water-year completeness are verified; gaps and unsupported
  cadence are errors for annual calculations. Without datetime timestamps,
  completeness can only be inferred from DOWY resets.

### Composition and configuration

- Timing, magnitude, and rate-of-change characteristics are independent
  diagnostics. Duration and frequency depend on their applicable preceding
  conditions. A frequency terminal includes its source condition and is not
  ANDed with that condition again.
- `success_pattern = false` treats the conjunction as a failure condition and
  reports its logical complement (non-failure). Unknown verdicts remain
  unknown unless three-valued conjunction/complement determines an outcome.
- Characteristic order is configuration sequence order; component options do
  not occupy a characteristic position. Compact TOML tables remain supported
  with a warning because TOML does not guarantee key order. Ordered
  characteristic arrays are the portable form. Explicit component `order`
  and `verbose` options are unsupported.

### Evaluation results

- `evaluate_component` and `evaluate_components` select one data column by
  zero-based position, defaulting to the first data column and excluding DOWY.
- A result contains only the selected source column under its original name,
  then DOWY, characteristic diagnostics, and the component outcome.
  `Result.dv_name` retains the selected source name. Datetime indexes are
  named `time`; other index names and types are preserved. Duplicate output
  columns are rejected.

## Consequences

Frequency outputs can differ from historical trailing-window results and
should be interpreted with the new forward-window rules. Annual statistics
may change because partial water years are excluded. The default exclusivity
change also permits overlapping windows unless explicitly enabled. A
`return_period` output is a descriptive reciprocal of portion; it does not
represent a Poisson recurrence probability or guarantee a mean recurrence
interval. See the [user reference](../user/reference.md) and
[migration notes](../user/migration.md) for examples and upgrade guidance.
