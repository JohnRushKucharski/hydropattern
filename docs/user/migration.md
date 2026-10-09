# Migration notes for v0.3.0

**v0.3.0 is not yet released.** The changes below describe v0.3.0 and differ
from the published v0.2.0 package. See
[installation](getting-started/installation.md). Review these changes before
comparing results with v0.2.0.

The “Earlier behavior (v0.2.0)” column describes the published package;
“v0.3.0 behavior” describes the forthcoming release.

| Area | Earlier behavior (v0.2.0) | v0.3.0 behavior and action |
|---|---|---|
| Frequency flag | `event_bool` defaulted to `true`. | The optional trailing boolean in a count or between frequency form now defaults to `false`, so overlapping qualifying windows are unioned. Set it to `true` to suppress later anchors within a qualifying fixed-length span. |
| Frequency windows | Trailing windows ended at each evaluated timestep. | Candidate windows extend forward from eligible source timesteps, truncate at record end, and produce retrospective classifications. Un-nested N is measured in input timesteps, not years. |
| Frequency meaning | Frequency counts could be confused with counts of component events. | The frequency condition counts qualifying timesteps, not component events. A condition that accepts zero can start at every timestep; other conditions start only at qualifying timesteps. |
| Configuration ordering | Compact characteristic keys were accepted without warning; component `order`/`verbose` settings could be used in older configurations. | Compact form emits a portability `UserWarning`. Prefer ordered `[[components.<name>.characteristics]]` tables. Unsupported `order` and `verbose` options are errors, not ignored settings. |
| Failure patterns | A false success-pattern setting did not consistently complement the combined failure condition. | `success_pattern = false` reports the logical complement of the combined failure condition (non-failure). Unknown outcomes remain unknown where the logic cannot determine a result. |
| Timing | Timing could be interpreted relative to water-year day. | Timing thresholds use calendar day-of-year, including for non-January water-year starts. The existing convention maps February 28 and 29 to the same timing position. |
| Annual calculations | A day-of-water-year reset could make an incomplete trailing year appear complete. | Recompute and review affected historical annual results. |
| Unknown outcomes | Unavailable moving-average and rate calculations could become binary comparisons; uncertain duration/frequency inputs could be treated as non-qualifying. | Unavailable calculations and uncertain duration/frequency verdicts now remain unknown when evidence cannot settle them. Missing input values remain rejected. Review affected characteristic and component outcomes. |
| Annual probability denominator | Annual fractions could use only known qualifying trials. | Intra-annual probability now assesses all observed timesteps in each complete water year. Unknowns leave possible fractions; the verdict is known only when every possible fraction agrees. This differs deliberately from descriptive summaries, which exclude unknowns. Recompute nested-frequency results. |
| Daily leap day | Daily completeness required every date in the Gregorian water year. | An otherwise complete daily water year may omit February 29; other missing days remain invalid. A missing real February 29 is indistinguishable from deliberate synthetic omission and is accepted. Exposure denominator is 366 only when that year's recorded rows contain February 29, otherwise 365, including partial years that do not reach it. |
| Result dataframe | Results could carry all input data columns and use a generic `dv` name. | Each result contains only the evaluated data column under its original name, then DOWY, characteristic outputs, and the component output. Datetime indexes are named `time`; other indexes are preserved. |
| Statistics | Historical frequency arrays and derived statistics reflect the earlier window and completeness rules. | Recompute and review historical comparisons. `return_period` has been removed; use `portion` or `percentage` for outcome summaries. |
| Response-surface coverage | Every defined summary could appear regardless of known-outcome coverage. | Plots now require 90% known-outcome coverage by default. Set `[output.plot].minimum_coverage` or `--minimum-coverage` to a finite fraction from 0 to 1. Exactly-at-cutoff scenarios remain eligible; all-unknown summaries remain undefined even at zero cutoff. |
| Response-surface colors | Failure-pattern plots reversed the default color map. | Default coloring now uses red for lower final component-outcome fractions for both pattern types. Explicit color maps remain unchanged; colors describe configured outcomes, not ecological benefit. |

For the complete explanation of unknown representations, denominator changes,
leap-day assumptions, and worked examples, see
[unknown outcomes](concepts/unknown-outcomes.md). Keep it beside the
[output](guide/outputs.md), [data preparation](guide/preparing-data.md), and
[plotting](guide/plotting.md) guidance when updating analysis instructions.

For frequency evaluation rules, worked examples, and Python API details, see
the [frequency reference](reference/characteristics/frequency.md) and the
[pattern-correctness decision record](https://github.com/JohnRushKucharski/hydropattern/blob/v0.3.0/docs/developer/adr/0003-pattern-correctness-contract.md).

## Summary denominator and removed mode

In v0.3.0, unknown outcomes are excluded from summary denominators.
For outcomes `[1, 0, unknown, unknown]`, the `portion` is `0.5`, not `0.25`.
Each characteristic and component column uses its own known outcomes. A
group with known outcomes but no successes has portion `0`; an all-unknown
group has no defined summary. Whole-record summaries combine successes and
known counts across the record, not an unweighted average of water-year
portions. Recompute historical summaries before comparing results.

`return_period` is rejected in `[output.metric].mode`; replace it with
`portion` or `percentage`. Neither mode estimates physical event likelihood
or timing. Do not treat a reciprocal portion as a measure of event spacing;
portions do not assert independent events. No compatibility alias is provided.

## Event-count bounds and exposure in v0.3.0

Unknown final outcomes no longer count as definite separators between
component events. `Result.event_count()` and `Result.event_rate()` now raise
an error when final outcomes allow multiple answers; use
`event_count_bounds()` or `event_rate_bounds()` to inspect named lower and
upper bounds. Water-year-specific bounds are available through
`event_count_bounds_by_water_year()` and
`event_rate_bounds_by_water_year()`. Bounds use final outcomes independently,
so they may include event counts impossible under the source-characteristic
dependencies.

Event rates now divide whole-record event counts by observed exposure,
including partial water years, rather than only complete water years. For
example, 18 monthly observations provide 1.5 water years of exposure. Annual
event counts are attributed to the water year containing each run's first
successful observation. Rates require supported daily or monthly timestamps;
unsupported cadence raises an error instead of falling back to complete-year
exposure. Review historical event-rate comparisons after upgrading.

## Coverage-aware response surfaces in v0.3.0

Plots now require 90% known-outcome coverage by default. Set
`[output.plot].minimum_coverage` or pass `--minimum-coverage` as a finite
fraction from 0 through 1. Exactly-at-cutoff scenarios remain eligible;
all-unknown summaries remain undefined even at zero cutoff. Summary matrices
are unchanged. Companion coverage CSVs explain each scenario's eligibility.

Failure-pattern plots no longer reverse the default color map. Red represents
lower final component-outcome fractions for both pattern types. Explicit color
maps remain unchanged; colors describe configured outcomes, not ecological
benefit.

Plots use whole-record component summaries but omit scenarios below the
configured coverage cutoff. A companion coverage CSV records each scenario's
coordinates, raw summary, known/total counts, coverage, eligibility, and
exclusion reason. Lower the cutoff only when a less-complete surface is
appropriate for the analysis. `fillin = true` is rejected when scenarios are
withheld, because the renderer cannot preserve those protected gaps. Recheck
historical plots after upgrading.

## Unknown annual frequency in v0.3.0

Nested annual conditions now retain unknown preceding outcomes instead of
treating them as failures. An annual fraction uses all observed timesteps in
a complete water year: one success, one failure, and ten unknown monthly
trials permit fractions from `1/12` through `11/12`. The `>= 0.5` verdict is
unknown, not a known-only fraction of `1/2`.

Intra-annual count diagnostics reduce to an annual success if any timestep is
definitely successful, an annual failure if every timestep is definitely
zero, and otherwise an unknown. Interannual windows retain these unknown
annual trials and uncertain anchors, including exclusive scheduling.
Partial years remain unknown and are excluded from interannual counting.
Recompute affected nested-frequency results; see the
[worked annual examples](reference/characteristics/frequency.md#unknown-annual-outcomes).

For direct Python calls, `water_year_probability_ratio` returns `NaN` for a
complete year containing any unknown trials, because there is no unique scalar
fraction. Nested condition evaluation still returns a known verdict when all
attainable fractions agree. Annual condition denominators remain distinct from these known-outcome
summary denominators.

## Ordered characteristic tables in v0.3.0

In v0.3.0, ordered characteristic tables use the literal TOML key
`parameters` instead of the `metrics` key accepted by v0.2.0:

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

## Frequency-window API names in v0.3.0

v0.3.0 changes names used by direct Python calls from v0.2.0. TOML files do
not need edits: the optional boolean stays in the same position in each
frequency list, and evaluation results do not change.

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

## Nested-frequency specification names in v0.3.0

Direct Python users upgrading from v0.2.0 must update code that reads or
constructs nested-frequency specifications. Shared fields such as `operator`, `values`, and
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
