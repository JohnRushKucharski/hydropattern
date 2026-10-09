# Rate-of-change characteristic

A rate-of-change characteristic compares a flow value with an earlier value
using their ratio. It is useful for describing relative increases or
decreases over a chosen number of timesteps; it is not a difference per unit
of calendar time.

## Configuration

```toml
rate_of_change = [">", 2]
# Alternatives:
# rate_of_change = [">", 2, 3, 7, 0.1]
# rate_of_change = [1.2, 2.5, 3, 7, 0.1]
```

The last three parameters are optional in order. The fifth value is called
`min`; if supplying it, also include `ma_periods` and `look_back`.

| Parameter | Type | Valid values | Meaning |
|---|---|---|---|
| `operator` | string | `<`, `<=`, `>`, `>=`, `=`, `!=` | Comparison with one ratio threshold. |
| `threshold`, `lower`, `upper` | numbers | Greater than 0; for a range, lower is less than upper | Ratio threshold or inclusive ratio bounds. |
| `ma_periods` | integer | 1 or greater; default `1` | Trailing moving-average length in timesteps. |
| `look_back` | integer | 1 or greater; default `1` | Number of timesteps between the current numerator and earlier denominator. |
| `min` | number | 0 or greater; default `0` | Strict lower limit for the earlier denominator. |

With no moving average, the calculation is `flow[t] / flow[t - look_back]`.
With a moving average, both the current value and earlier value are taken
from that averaged series. The ratio has no flow unit. The earlier value is
used as the denominator only when it is strictly greater than `min`.
This setting does not replace small denominators with the threshold.
Startup rows and restricted denominators have no ratio and remain unknown,
including for `!=`. See
[unknown outcomes](../../concepts/unknown-outcomes.md#where-unknown-outcomes-come-from).

## Worked example

With `rate_of_change >= 2` and the default one-timestep look-back, a ratio of
2 or more meets the condition. The first input observation provides the
earlier value for the first displayed row; the table includes only rows with
a defined numerator and denominator.

| Flow | Earlier flow | Ratio | Rate-of-change outcome | Component outcome |
|---:|---:|---:|---:|---:|
| 2 | 1 | 2 | 1 | 1 |
| 4 | 2 | 2 | 1 | 1 |
| 8 | 4 | 2 | 1 | 1 |
| 4 | 8 | 0.5 | 0 | 0 |

Expected diagnostic from the second observation onward: `[1, 1, 1, 0]`

Expected component outcome from the second observation onward: `[1, 1, 1, 0]`

This example documents the ratio calculation only; it does not describe
startup or restricted-denominator classifications.
See [evaluation order](../../concepts/evaluation-order.md) for how the
diagnostic relates to the final component outcome.
