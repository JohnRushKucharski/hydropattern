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
