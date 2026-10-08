# Timing characteristic

A timing characteristic sets the calendar days on which its condition is
evaluated. It is useful when a flow pattern is relevant only during a
particular season.

## Configuration

```toml
timing = [335, 60]
```

| Parameter | Type | Valid values | Meaning |
|---|---|---|---|
| `first_day` | integer | 1–366 | First inclusive calendar day-of-year. |
| `last_day` | integer | 1–366 | Last inclusive calendar day-of-year. |

The day values use the calendar day-of-year, not the day of water year. A
start greater than the end describes a window that crosses the end of the
calendar year. For example, `[335, 60]` includes December 1 through March 1
on the project's 365-day timing scale. February 28 and February 29 share one
timing position.

## Worked example

For `[335, 60]`, the inclusive seasonal window includes December 1, February
28, and March 1, but not November 30 or March 2:

| Date | Timing outcome | Component outcome |
|---|---:|---:|
| 2021-11-30 | 0 | 0 |
| 2021-12-01 | 1 | 1 |
| 2022-02-28 | 1 | 1 |
| 2022-03-01 | 1 | 1 |
| 2022-03-02 | 0 | 0 |

Expected diagnostic: `[0, 1, 1, 1, 0]`

Expected component outcome: `[0, 1, 1, 1, 0]`

The dates illustrate calendar interpretation, not a recommended ecological
season. For the meaning of this outcome in a component, see
[evaluation order](../../concepts/evaluation-order.md).
