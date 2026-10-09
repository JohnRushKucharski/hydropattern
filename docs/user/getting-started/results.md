# Interpreting first results

This guide explains the exact, fully known results of the
[first evaluation](first-run.md). Other analyses can contain unknown outcomes;
see the [unknown-outcomes guide](../concepts/unknown-outcomes.md) for their
evaluation and reporting behavior.

## Read the timestep CSV

Open `first-run_output/flow_sustained_flow.csv` in a text editor or spreadsheet.
It contains these columns:

| Column | Meaning |
| --- | --- |
| `time` | The observation date. |
| `flow` | The original scenario's observation, in the input units. |
| `dowy` | Day of water year; this example starts the water year on January 1. |
| `magnitude_gt1` | The magnitude characteristic outcome: 1 where flow exceeds 1, otherwise 0. |
| `duration_ge2` | The duration characteristic outcome: 1 throughout a qualifying run of at least two timesteps, otherwise 0. |
| `sustained_flow` | The final component outcome. Here it agrees with duration. |

In this example, 1 means configured success and 0 means configured failure.
These are software classifications, not proof of ecological benefit or harm.
All eight outcomes are known.

| Date | Flow | Magnitude outcome | Duration outcome | Component outcome |
| --- | ---: | ---: | ---: | ---: |
| 2020-01-01 | 0 | 0 | 0 | 0 |
| 2020-01-02 | 2 | 1 | 1 | 1 |
| 2020-01-03 | 3 | 1 | 1 | 1 |
| 2020-01-04 | 0 | 0 | 0 | 0 |
| 2020-01-05 | 4 | 1 | 0 | 0 |
| 2020-01-06 | 0 | 0 | 0 | 0 |
| 2020-01-07 | 0 | 0 | 0 | 0 |
| 2020-01-08 | 5 | 1 | 0 | 0 |

January 2 and 3 form a qualifying run of two timesteps, so both are marked
successful by duration. The isolated qualifying timesteps on January 5 and 8
fail the duration condition. This whole-run assessment is retrospective:
January 2's duration outcome uses the later observation on January 3.

## Read the summary workbook

Open `sustained_flow_summary.xlsx`. Each sheet has a `flow` column for the
scenario and rows labelled `total` and `2020`. The `total` row summarizes the
whole observed record. The `2020` row summarizes the observations in that
water year; it does not mean a full year was observed.

| Sheet | `total` and `2020` portions in this example |
| --- | ---: |
| `magnitude_gt1` | 0.50 |
| `duration_ge2` | 0.25 |
| `sustained_flow` | 0.25 |

The component has two successful timesteps among eight fully known timesteps,
giving 2/8 = 0.25, or 25%. Magnitude qualifies on four timesteps, but two of
those fail the later duration assessment. Read characteristic summaries
separately from the final component summary.

This is a deliberately fully known example. Do not extrapolate its denominator
to intervals with unknown outcomes: summaries use known outcomes only, and
all-unknown intervals have no defined portion.
For broader configuration rules, continue to the [reference](../reference.md).
