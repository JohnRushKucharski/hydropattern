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

For frequency evaluation rules, worked examples, and Python API details, see
the [frequency reference](reference.md#frequency) and the
[pattern-correctness decision record](../adr/0003-pattern-correctness-contract.md).

## Next release: ordered characteristic tables

This change applies to the next release; its version number has not yet been
assigned. Ordered characteristic tables will use the literal TOML key
`parameters` instead of `metrics`:

```toml
[[components.pulse.characteristics]]
type = "magnitude"
metrics = [">", 1.0]
```

becomes:

```toml
[[components.pulse.characteristics]]
type = "magnitude"
parameters = [">", 1.0]
```

An ordered table containing `metrics` is rejected, even if it also contains
`parameters`. Compact characteristic-key syntax and `[output.metric]` are unchanged.
This changes configuration syntax only: characteristic order and evaluation
behavior remain the same.
