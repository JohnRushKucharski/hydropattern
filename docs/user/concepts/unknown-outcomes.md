# Unknown outcomes

An **unknown outcome** means available information cannot determine whether
a characteristic or component condition succeeded or failed. It is not
success, failure, or zero. Unknown outcomes appear as `NaN` in hydropattern
results and may appear as `pd.NA` in pandas-facing data. Missing flow values
in the input are rejected; an unknown result is produced only when a required
calculation or evaluation cannot be determined.

This distinction matters to duration and frequency, annual conditions,
component summaries, event counts, and response-surface plots. Unknowns must
not be silently treated as failures or successes. See the
[glossary](glossary.md#outcomes), [evaluation order](evaluation-order.md),
and the affected [duration](../reference/characteristics/duration.md),
[frequency](../reference/characteristics/frequency.md), and
[rate-of-change](../reference/characteristics/rate-of-change.md) references.

## Where unknown outcomes come from

Input flow values must be present and numeric. hydropattern does not impute
missing observations. Some calculations are unavailable even when all input
values are present:

- A moving average has no value until its full averaging interval is
  available.
- A rate-of-change ratio has no value when its earlier denominator is not
  strictly above the configured minimum. Its startup rows can also lack
  enough history.
- Duration can have an unknown verdict when an unknown preceding outcome
  could split or join a qualifying run.
- Frequency can have an unknown verdict when possible counts, anchors, or
  exclusive-window schedules disagree.
- Nested annual evaluation is unknown in partial water years; unsupported
  cadence can also make completeness undetermined.

An unavailable numeric calculation remains unknown through its comparison,
including `!=`. A known unmet condition still settles a combined
success-pattern component as failure. All known met conditions settle it as
success. Otherwise, the final component outcome is unknown. A
failure-pattern component complements known outcomes and keeps unknown
outcomes unknown; its known non-failure result is not proof of ecological
benefit.

## Duration: uncertain run boundaries

Duration assesses each whole qualifying run. It considers both possibilities
when a preceding outcome is unknown instead of treating that timestep as a
definite break.

For `duration >= 3`, the preceding outcomes
`[0, 1, 1, unknown, 1, 0]` can describe two runs of lengths 2 and 1, both
failing, or one run of length 4, which passes:

| Position | Preceding outcome | Possible run interpretation | Duration outcome |
|---:|---:|---|---:|
| 1 | 0 | Definite failure | 0 |
| 2 | 1 | Length 2 fails or length 4 passes | unknown |
| 3 | 1 | Length 2 fails or length 4 passes | unknown |
| 4 | unknown | Breaks the runs or joins them | unknown |
| 5 | 1 | Length 1 fails or length 4 passes | unknown |
| 6 | 0 | Definite failure | 0 |

Expected duration diagnostic: `[0, unknown, unknown, unknown, unknown, 0]`

Inclusive bounds can also leave an uncertain verdict. For `duration = [2, 3]`
and preceding outcomes `[1, unknown, 1]`, the unknown either separates two
single-timestep runs (both fail) or joins a three-timestep run (passes).
Every diagnostic outcome is therefore unknown. Conversely, definite failures
stay failures even next to unknowns if no possible run can change their
verdict. See the [duration reference](../reference/characteristics/duration.md)
for fully known threshold and bounded-run examples.

## Frequency: possible counts and windows

Unknown qualifying outcomes represent possible counts, not definite zeros.
For a three-timestep window containing two known qualifying timesteps and one
unknown, possible counts are 2 or 3: `>= 2` succeeds, `>= 3` is unknown, and
`>= 4` fails.

| Count condition | Possible counts | Window verdict |
|---|---|---|
| `>= 2` | 2 or 3 | success |
| `>= 3` | 2 or 3 | unknown |
| `>= 4` | 2 or 3 | failure |

Unknown timesteps can be possible anchors. For overlapping windows of length
3 with `>= 1`, preceding outcomes `[unknown, 0, 0, 0]` yield
`[unknown, unknown, unknown, 0]`; `[unknown, 1, 0, 0]` yield
`[unknown, 1, 1, 1]`. The known window in the second case settles overlapping
outcomes even though the first anchor is uncertain.

| Preceding outcomes | Overlapping-window outcomes |
|---|---|
| `[unknown, 0, 0, 0]` | `[unknown, unknown, unknown, 0]` |
| `[unknown, 1, 0, 0]` | `[unknown, 1, 1, 1]` |

Exclusive windows also account for possible schedules. For
`[unknown, 1, 0, 0]` with the same length-3 `>= 1` condition, the unknown
first timestep may or may not claim its window:

| Possible schedule | Frequency outcomes |
|---|---|
| First timestep qualifies | `[1, 1, 1, 0]` |
| First timestep does not qualify | `[0, 1, 1, 1]` |
| Verdict common to both schedules | `[unknown, 1, 1, unknown]` |

The middle outcomes stay known because both schedules mark them successful;
the first and last differ. Related windows share unknown trials, so their
possible coverage is not independent. For `[1, unknown]`, `= 1`, and a
two-timestep window, the result is `[unknown, 1]` under both overlapping and
exclusive rules. Record-end truncation remains normal behavior: a shortened
window is assessed using available observations and is not automatically
unknown. See the [frequency reference](../reference/characteristics/frequency.md)
for complete worked schedules and nested-frequency behavior.

## Annual frequency and water years

An intra-annual fraction condition uses all observed timesteps in a complete
water year, including unknown outcomes in its denominator. This is different
from a descriptive summary portion, which uses only known outcomes.

For a complete year of 12 monthly observations and condition `>= 0.5`:

| Known qualifying / known failing / unknown | Possible annual fractions | Annual verdict |
|---|---|---|
| 1 / 1 / 10 | `1/12` through `11/12` | unknown |
| 7 / 0 / 5 | `7/12` through `12/12` | success |
| 0 / 7 / 5 | `0/12` through `5/12` | failure |
| 0 / 0 / 12 | `0/12` through `12/12` | unknown |

The annual verdict is repeated across that complete water year. Partial water
years remain unknown for nested annual evaluation and are excluded from its
interannual trials. Other conditions can still have known outcomes in those
same partial years.

With an October-start boundary, observations from January 2020 through
December 2021 belong to WY2020 (partial), WY2021 (complete), and WY2022
(partial). Whole-record summaries retain every observed timestep, including
those in partial years. Event rates use observed exposure, including partial
years: 18 monthly observations represent 1.5 water years. See
[output files](../guide/outputs.md#counting-component-events) for event
attribution, bounds, and exposure.

### Daily completeness and optional February 29

An otherwise complete daily water year is accepted whether or not it
contains February 29. If present, the daily exposure denominator is 366;
if omitted, it is 365. The same rule accepts a missing real leap-day
observation as if February 29 had been deliberately omitted. This is an
intentional trade-off: hydropattern cannot distinguish those cases. Other
missing daily dates are not accepted.

For January-start water years, a complete 2019 record uses 365/365 days; a
complete 2020 record with February 29 uses 366/366; and a 2020 record with
only February 29 omitted uses 365/365. A partial January 1-February 28, 2020
record uses denominator 365 because its observed dates do not contain
February 29.

For a partial year whose observed span does not reach February 29, the
denominator is 365, even if a leap day would occur later in that water year.
The denominator is based on recorded timestamps, not normalized day-of-water-
year labels. February 28 and February 29 share a seasonal label but remain
two observed daily trials when both are present.

Use complete daily records with all dates except optional February 29, or
repair other gaps before requesting nested annual results or event rates.
Unsupported cadence, non-leap-day gaps, duplicate dates, or insufficient
timestamp information make annual completeness undetermined. Observed-outcome
counts and fractions remain available, but time-based event rates and nested
annual evaluation are rejected rather than guessed. The
[data-preparation guide](../guide/preparing-data.md#daily-calendars-and-leap-day)
and [water-year reporting](../guide/outputs.md#water-year-rows-and-completeness)
describe checks and remedies.

## Summaries, event counts, and coverage

Each outcome column has its own summary denominator. A `portion` divides
successful outcomes by known outcomes; an all-unknown interval has no defined
portion. For `[1, 0, unknown, unknown]`, the portion is `1/2`, not `1/4`.
Known-outcome coverage is known outcomes divided by recorded observations; it
is separate from the portion.

For `[1, unknown, 1]`, the component portion is 100% because both known
outcomes are successful, but coverage is only `2/3` (66.7%). The possible
number of component events is 1 or 2. These statistics answer different
questions:
the portion does not claim success through the unknown timestep, and the
event-count bounds preserve its ambiguity. MVP event bounds treat unknown
final outcomes independently and can therefore be wider than outcomes
possible under their original characteristic dependencies.

For example, an uncertain frequency calculation spanning three timesteps can produce only
`[0, 0, 0]` or `[1, 1, 1]`, so its source-dependent event count is 0 or 1.
After only `[unknown, unknown, unknown]` is retained, the MVP bounds also
allow `[1, 0, 1]`, giving conservative bounds of 0-2. These bounds are not
the exact set of event counts permitted by the original frequency rule.

At the default response-surface coverage cutoff of 90%, this scenario is not
plotted. Its defined portion remains in summary reports. The
[output guide](../guide/outputs.md) explains denominators and event bounds;
the [plotting guide](../guide/plotting.md) explains eligibility and exports.

## Response-surface eligibility and color

Plots require 90% known-outcome coverage by default. Exactly 90% qualifies.
Set `[output.plot].minimum_coverage` in TOML, use
`--minimum-coverage` on the CLI, or pass `minimum_coverage` to
`ScenarioResults.plot_response_surface`; all use fractions from 0 to 1.
Setting zero removes the coverage threshold, but it does not define an
all-unknown summary. Reports retain defined portions even when the plot
withholds them.

Withheld and undefined scenarios remain gaps in the exported plot grid and
are explained in the companion coverage CSV. `fillin = true` conflicts with
protected gaps; disable fill-in rather than allowing the renderer to obscure
them. The default color scale uses red for lower final component-outcome
fractions for both success-pattern and failure-pattern components. Colors
describe configured outcomes, not ecological benefit. The
[response-surface example](../examples/index.md#response-surface-coverage) shows an
inclusive 90% cutoff with one scenario withheld.

## Changes for existing analyses

Unknown-aware comparisons, duration/frequency evaluation, and nested annual
conditions can change results that formerly treated unavailable calculations
as binary. Annual probability now includes unknown trials in its denominator;
descriptive summaries instead exclude unknown outcomes. Optional leap-day
handling, partial-year event-rate exposure, and MVP event-count bounds also
change earlier assumptions. `return_period` is removed; use `portion` or
`percentage` for summaries. Default response-surface colors now consistently
show lower final component-outcome fractions in red.

Review the [migration guide](../migration.md) before comparing outputs across
versions. The [Python API](../api/index.md) documents unknown representations
and the corresponding result methods.
