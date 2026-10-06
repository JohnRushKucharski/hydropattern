# Migration notes: pattern-correctness changes

These behavior changes affect configurations and results produced by versions
before the pattern-correctness update. Review them before comparing new results
with historical outputs.

| Area | Earlier behavior | Current behavior and action |
|---|---|---|
| Frequency flag | `event_bool` defaulted to `true`. | Use `exclusive_event_window`, now defaulting to `false` (overlapping qualifying windows are unioned). Set it to `true` to suppress later anchors within a qualifying fixed-length span. Update old configs and direct factory calls; do not rely on a silent compatibility alias. |
| Frequency windows | Trailing windows ended at each evaluated timestep. | Candidate windows extend forward from eligible source timesteps, truncate at record end, and produce retrospective classifications. Un-nested N is measured in input timesteps, not years. |
| Frequency meaning | Frequency could be interpreted as occurrence counts/events. | The predicate counts eligible source timesteps, not runs. Zero-admitting predicates can anchor at every timestep; other predicates anchor at source successes. |
| Configuration ordering | Compact characteristic keys were accepted without warning; component `order`/`verbose` settings could be used in older configurations. | Compact form emits a portability `UserWarning`. Prefer ordered `[[components.<name>.characteristics]]` tables. Unsupported `order` and `verbose` options are errors, not ignored settings. |
| Failure patterns | A false success-pattern setting did not consistently complement the combined failure condition. | `success_pattern = false` reports the logical complement of the combined failure condition (non-failure). Unknown outcomes remain unknown where the logic cannot determine a result. |
| Timing | Timing could be interpreted relative to water-year day. | Timing thresholds use calendar day-of-year, including for non-January water-year starts. The existing convention maps February 28 and 29 to the same timing position. |
| Annual calculations | A DOWY reset could make an incomplete trailing year appear complete. | Annual probability and event-rate exposure exclude leading and trailing partial water years. Datetime-backed annual calculations require supported, cadence-verified daily or monthly data without gaps. |
| Result dataframe | Results could carry all input data columns and use a generic `dv` name. | Each result contains only the evaluated data column under its original name, then DOWY, characteristic outputs, and the component output. Datetime indexes are named `time`; other indexes are preserved. |
| Statistics | Historical frequency arrays and derived statistics reflect the earlier window and completeness rules. | Recompute and review historical comparisons. `return_period` is a descriptive reciprocal of portion, not a Poisson recurrence probability or guaranteed average recurrence interval. |

For the complete frequency contract, golden arrays, configuration examples,
and Python API shape, see the [user reference](reference.md#frequency) and
[pattern-correctness ADR](../adr/0003-pattern-correctness-contract.md).
