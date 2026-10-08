# Combine a seasonal window with a flow threshold

This illustrative pack checks flow above 1 on December 1 and 2. Flow units
are arbitrary; the season and threshold are not ecological recommendations.
Use current source code, not PyPI v0.2.0.

Save `flow.csv` and `seasonal-thresholds.toml` in one folder. Open a terminal
in that folder; input paths resolve from the terminal's working folder:

```console
hydropattern run seasonal-thresholds.toml --no-excel
```

Before source publication, use your local checkout from that same folder:

```powershell
uv run --no-default-groups --project C:\path\to\hydropattern hydropattern run seasonal-thresholds.toml --no-excel
```

The ordered tables assess timing and magnitude independently. Their combined
condition succeeds only when both are met:

| Date | Flow | Timing | Magnitude | Component |
| --- | ---: | ---: | ---: | ---: |
| 2021-11-29 | 2 | 0 | 1 | 0 |
| 2021-11-30 | 2 | 0 | 1 | 0 |
| 2021-12-01 | 2 | 1 | 1 | 1 |
| 2021-12-02 | 0 | 1 | 0 | 0 |
| 2021-12-03 | 3 | 0 | 1 | 0 |
| 2021-12-04 | 3 | 0 | 1 | 0 |

Expected component outcome: `[0, 0, 1, 0, 0, 0]`.
Timing's inclusive `[335, 336]` uses calendar day-of-year, not day of water
year. High flow outside that interval does not meet the component condition.

`seasonal-thresholds_output` contains `flow_early_december.csv` and
`early_december_summary.xlsx`. The workbook has timing, magnitude, and
component sheets, each with `total` and `2021` rows. Their portions are 2/6,
5/6, and 1/6 respectively. `--no-excel` selects raw CSV; it does not disable
summary workbooks. Repeating the command overwrites outputs by default.

See [timing](../../docs/user/reference/characteristics/timing.md) and
[magnitude](../../docs/user/reference/characteristics/magnitude.md).
