# Duration characteristic

A duration characteristic asks whether a consecutive qualifying run is
shorter than, longer than, equal to, or within configured timestep-count
bounds. It is useful when a condition must persist for a specified number of
observations.

Duration uses **timesteps**, not calendar days or months. A run of seven
monthly observations is seven timesteps; its calendar span depends on the
observation dates.

## Configuration

```toml
duration = [">=", 2]
# Alternative inclusive range:
# duration = [2, 7]
```

| Parameter | Type | Valid values | Meaning |
|---|---|---|---|
| `operator` | string | `<`, `<=`, `>`, `>=`, `=`, `!=` | Comparison with one run-length threshold. |
| `time_steps` | integer | 1 or greater | Timestep-count threshold. |
| `minimum_steps`, `maximum_steps` | integers | 1 or greater; minimum less than maximum | Inclusive acceptable range of run lengths. |

The characteristic follows the characteristics before it in the component
order. A timestep qualifies for duration only when all preceding
characteristics are met there. Duration assesses the whole qualifying run
retrospectively: if it passes, every timestep in that run is marked 1; if it
does not pass, every timestep is marked 0.

## Worked example: inclusive bounds

Suppose magnitude qualifies on the four-timestep run and the later
two-timestep run shown below. The duration bounds `[2, 3]` accept the latter
run but reject the four-timestep run because it is longer than the upper
bound.

| Position | Flow | Magnitude outcome | Qualifying run length | Duration outcome | Component outcome |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 4 | 0 | 0 |
| 2 | 1 | 1 | 4 | 0 | 0 |
| 3 | 1 | 1 | 4 | 0 | 0 |
| 4 | 1 | 1 | 4 | 0 | 0 |
| 5 | 0 | 0 | — | 0 | 0 |
| 6 | 1 | 1 | 2 | 1 | 1 |
| 7 | 1 | 1 | 2 | 1 | 1 |
| 8 | 0 | 0 | — | 0 | 0 |
| 9 | 0 | 0 | — | 0 | 0 |

Expected diagnostic: `[0, 0, 0, 0, 0, 1, 1, 0, 0]`

Expected component outcome: `[0, 0, 0, 0, 0, 1, 1, 0, 0]`

For a separate threshold example, a ten-timestep qualifying run meets
`duration >= 7`. The whole run is marked, not only the timesteps after its
seventh observation:

Expected diagnostic: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]`

Expected component outcome: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]`

This is a fully known example. Current evaluation treats an unmet preceding
condition as a run break; preservation of uncertain run boundaries is a
separate reporting change. Compare the duration diagnostic with the final
[component outcome](../../concepts/evaluation-order.md#combining-conditions).
