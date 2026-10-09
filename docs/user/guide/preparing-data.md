# Preparing time-series data

Provide a CSV or Excel file with one date column followed by one or more
numeric flow columns. Each flow column is evaluated independently as a
scenario.

## Arrange the input columns

For CSV, put the date in the first column and use `time` as its heading:

```text
time,flow
2020-01-01,0
2020-01-02,2
2020-01-03,3
```

For several scenarios, add one numeric column for each:

```text
time,baseline,alternative
2020-01-01,0,1
2020-01-02,2,0
2020-01-03,3,2
```

Each observation date should correspond to one row. The dates must be
readable by the date parser; set `date_format` in the
[time-series configuration](../reference/configuration.md#input-time-series)
when automatic parsing is unsuitable. Excel input uses the first column for
dates and the selected worksheet.

## Place files and run the command

Relative data paths resolve from the terminal's current working folder, not
from the folder containing the TOML configuration. Keep the data file in the
working folder named by the relative path, or use an appropriate path in the
configuration. The [first evaluation](../getting-started/first-run.md)
demonstrates this explicitly.

## Choose characteristics for the question

Use the [scientific foundations](../concepts/scientific-foundations.md) and
[characteristic reference](../reference.md#characteristic-reference) to
choose conditions that match the analysis. Examples demonstrate syntax and
software behavior only; their thresholds are not universal ecological
criteria.

For seasonal comparisons, use the timing characteristic's calendar-day
convention. For duration and frequency, identify whether a timestep means a
day, month, or another sampling interval in your data; their configured
lengths are counts of observations, not automatically calendar lengths.

## Daily calendars and leap day

For daily records, hydropattern accepts an otherwise complete water year
whether or not February 29 is present. A missing February 29 is treated as an
optional omission, whether it was removed from a synthetic calendar or was a
missing real observation; the software cannot tell these apart. Other missing
days are not accepted. Do not use omitted leap days as a general gap-filling
rule.

Daily completeness and exposure use a denominator of 366 when the water year's
recorded dates contain February 29 and 365 when they do not. A partial year
that ends before February 29 uses 365, even when a leap day would occur later.
February 28 and 29 share a normalized day-of-water-year label, but remain
separate recorded observations. Do not infer the exposure denominator from
those labels.

Unsupported cadence, duplicate dates, other missing dates, or insufficient
timestamp information make annual completeness undetermined. Observed-outcome
counts and fractions are still available, but nested annual evaluation and
time-based event rates cannot be calculated reliably. Use supported complete
daily or monthly timestamps for those analyses; see
[unknown outcomes](../concepts/unknown-outcomes.md#daily-completeness-and-optional-february-29)
and [water-year reporting](outputs.md#water-year-rows-and-completeness).
