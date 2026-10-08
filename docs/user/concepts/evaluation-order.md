# Evaluation order and interpretation

hydropattern evaluates the characteristics in the order they appear in a
component. A characteristic outcome describes that characteristic at one
timestep; the component outcome is the final classification after the
characteristics are combined.

## Independent and dependent characteristics

Timing, magnitude, and rate-of-change characteristics report their own
conditions independently of earlier characteristics. Duration and frequency
use the conjunction of the preceding characteristics to identify qualifying
timesteps. Put a duration or frequency characteristic after the conditions
that define which observations qualify. A frequency characteristic must be
last.

For example, the magnitude condition below is met on January 2, January 3,
and January 5. The duration condition assesses each entire consecutive
qualifying run and accepts runs of at least two timesteps:

| Date | Flow | Magnitude diagnostic | Duration diagnostic | Component outcome |
|---|---:|---:|---:|---:|
| 2020-01-01 | 0 | 0 | 0 | 0 |
| 2020-01-02 | 2 | 1 | 1 | 1 |
| 2020-01-03 | 3 | 1 | 1 | 1 |
| 2020-01-04 | 0 | 0 | 0 | 0 |
| 2020-01-05 | 4 | 1 | 0 | 0 |
| 2020-01-06 | 0 | 0 | 0 | 0 |

Expected magnitude diagnostic: `[0, 1, 1, 0, 1, 0]`

Expected duration diagnostic: `[0, 1, 1, 0, 0, 0]`

Expected component outcome: `[0, 1, 1, 0, 0, 0]`

The duration assessment uses the later January 3 observation to classify
January 2. It is a retrospective assessment, not a forecast available on
January 2.

If a preceding outcome is unknown, duration considers both possibilities:
the timestep may break a run or extend it. The diagnostic stays unknown when
possible run lengths disagree on whether the duration condition passes. For
example, with `duration >= 3`, preceding outcomes
`[0, 1, 1, unknown, 1, 0]` produce
`[0, unknown, unknown, unknown, unknown, 0]`: the unknown can split
length-2 and length-1 runs, or join a length-4 run. Definite failures still
settle the result where they occur.

## Combining conditions

For a success-pattern component, all configured conditions must be met for
the final outcome to be success. A known unmet condition makes the combined
outcome failure. The final outcome is not interchangeable with any one
characteristic diagnostic.

With `success_pattern = false`, the characteristics define a combined
failure condition and the component reports its logical complement:
non-failure. Non-failure does not demonstrate a beneficial ecological
response. See
[component configuration](../reference/configuration.md#component-settings).

Frequency is terminal: its output already incorporates its preceding
conditions, so the component outcome follows the frequency result rather than
requiring those conditions to be met again at the same timestep. In nested
frequency, the interannual result is terminal.

## Read the result columns

The raw result includes the selected flow column, `dowy`, a diagnostic column
for each characteristic, and the final component column. The characteristic
columns help explain how a classification was reached; use the component
column when interpreting the configured pattern as a whole. See
[result files](../guide/outputs.md) for the output layout.

Unavailable numeric calculations, such as a rate comparison without a
previous flow value, remain unknown rather than becoming a failed comparison.
Missing input flow values are still rejected. Other calculations and
reporting summaries have separate unknown-value rules. See
[unknown outcomes](unknown-outcomes.md) for their propagation, denominators,
and reporting behavior.
