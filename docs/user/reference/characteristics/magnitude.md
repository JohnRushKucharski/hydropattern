# Magnitude characteristic

A magnitude characteristic compares each flow observation with a threshold
or inclusive range. It is useful for identifying observations above, below,
or between specified flow levels.

## Configuration

```toml
magnitude = [">", 1]
# Alternatives:
# magnitude = [">", 1, 7]
# magnitude = [0.5, 5.0]
# magnitude = [0.5, 5.0, 7]
```

| Parameter | Type | Valid values | Meaning |
|---|---|---|---|
| `operator` | string | `<`, `<=`, `>`, `>=`, `=`, `!=` | Comparison with one threshold. |
| `threshold` | number | 0 or greater | Flow value used in a one-threshold comparison. |
| `minimum`, `maximum` | numbers | 0 or greater; minimum less than maximum | Inclusive flow range. |
| `ma_periods` | integer | 1 or greater; default `1` | Number of timesteps used in the trailing moving average. |

When `ma_periods = k` is greater than one, the mean is calculated from the
current observation and the preceding observations in that window:

```text
average[t] = mean(flow[t-k+1], ..., flow[t])
```

Its unit is the same as the input flow unit. The calculation is unavailable
for the first `k - 1` observations because the full window is not yet
available; this statement concerns the moving-average value, not how an
outcome is classified. A value of one compares the original observation.

## Worked example

With the condition `flow >= 1`, each observation is assessed separately:

| Flow | Magnitude outcome | Component outcome |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 1 | 1 |
| 2 | 1 | 1 |
| 1 | 1 | 1 |
| 0 | 0 | 0 |

Expected diagnostic: `[0, 1, 1, 1, 0]`

Expected component outcome: `[0, 1, 1, 1, 0]`

For how this characteristic combines with duration or frequency, see
[evaluation order](../../concepts/evaluation-order.md). When a moving-average
value is unavailable during startup, its comparison outcome is unknown. This
fully known example does not illustrate those startup outcomes.
