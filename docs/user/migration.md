# Migration notes: pattern-correctness changes

**Unreleased:** completed changes below are present in current source, not
PyPI v0.2.0. See [installation](getting-started/installation.md). Later
reporting changes remain pending; this page will expand as they are implemented.

These behavior changes affect configurations and results produced by versions
before the pattern-correctness update. Review them before comparing new results
with historical outputs.

| Area | Earlier behavior | Current behavior and action |
|---|---|---|
| Frequency flag | `event_bool` defaulted to `true`. | The optional trailing boolean in a count or between frequency form now defaults to `false`, so overlapping qualifying windows are unioned. Set it to `true` to suppress later anchors within a qualifying fixed-length span. |
| Frequency windows | Trailing windows ended at each evaluated timestep. | Candidate windows extend forward from eligible source timesteps, truncate at record end, and produce retrospective classifications. Un-nested N is measured in input timesteps, not years. |
| Frequency meaning | Frequency counts could be confused with counts of component events. | The frequency condition counts qualifying timesteps, not component events. A condition that accepts zero can start at every timestep; other conditions start only at qualifying timesteps. |
| Configuration ordering | Compact characteristic keys were accepted without warning; component `order`/`verbose` settings could be used in older configurations. | Compact form emits a portability `UserWarning`. Prefer ordered `[[components.<name>.characteristics]]` tables. Unsupported `order` and `verbose` options are errors, not ignored settings. |
| Failure patterns | A false success-pattern setting did not consistently complement the combined failure condition. | `success_pattern = false` reports the logical complement of the combined failure condition (non-failure). Unknown outcomes remain unknown where the logic cannot determine a result. |
| Timing | Timing could be interpreted relative to water-year day. | Timing thresholds use calendar day-of-year, including for non-January water-year starts. The existing convention maps February 28 and 29 to the same timing position. |
| Annual calculations | A day-of-water-year reset could make an incomplete trailing year appear complete. | Recompute and review affected historical annual results. |
| Result dataframe | Results could carry all input data columns and use a generic `dv` name. | Each result contains only the evaluated data column under its original name, then DOWY, characteristic outputs, and the component output. Datetime indexes are named `time`; other indexes are preserved. |
| Statistics | Historical frequency arrays and derived statistics reflect the earlier window and completeness rules. | Recompute and review historical comparisons. The next release removes `return_period`; replace it with `portion` or `percentage`. |

For frequency evaluation rules, worked examples, and Python API details, see
the [frequency reference](reference.md#frequency) and the
[pattern-correctness decision record](https://github.com/JohnRushKucharski/hydropattern/blob/docs-reporting-s3/docs/developer/adr/0003-pattern-correctness-contract.md).

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

## Next release: frequency-window API names

The next release changes names used by direct Python calls. TOML files do not
need edits: the optional boolean stays in the same position in each frequency
list, and evaluation results do not change.

```toml
frequency = [">=", 1, 5, true]
```

Replace the keyword argument `exclusive_event_window` with
`exclusive_windows` in `frequency_fx`,
`nested_frequency_intra_annual_fx`, `nested_frequency_interannual_fx`, and
`water_year_probability_ratio`. Update fields on `CharacteristicSpec`:

| Before | After |
|---|---|
| `frequency_fx(..., exclusive_event_window=True)` | `frequency_fx(..., exclusive_windows=True)` |
| `spec.exclusive_event_window` | `spec.exclusive_windows` |
| `spec.nested_exclusive_event_window` | `spec.interannual_exclusive_windows` |
| `from hydropattern.patterns import mark_events`<br>`mark_events(raw, exclusive_event_window=True)` | `from hydropattern.patterns import mark_windows`<br>`mark_windows(raw, exclusive_windows=True)` |

No compatibility aliases are provided. Evaluation results do not change.

## Next release: nested-frequency specification names

Direct Python users must update code that reads or constructs nested-frequency
specifications. Shared fields such as `operator`, `values`, `big_n`, and
`exclusive_windows` remain unchanged. Names specific to the interannual pattern
now use `interannual_`; the flag identifying that a specification has this
pattern is `has_interannual_pattern`. The generic characteristic marker is
`is_terminal`.

| Before | After |
|---|---|
| `spec.is_nested` | `spec.has_interannual_pattern` |
| `spec.nested_operator` | `spec.interannual_operator` |
| `spec.nested_values` | `spec.interannual_values` |
| `spec.nested_big_n` | `spec.interannual_big_n` |
| `Characteristic(..., is_nested=True)` | `Characteristic(..., is_terminal=True)` |
| `characteristic.is_nested` | `characteristic.is_terminal` |

There are no compatibility aliases. TOML syntax, frequency evaluation, and
generated result-column names are unchanged.
