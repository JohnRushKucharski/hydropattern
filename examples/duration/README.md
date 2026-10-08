# Compare a duration threshold with inclusive bounds

This illustrative pack contains qualifying runs of four and two daily
timesteps. Flow units are arbitrary; thresholds demonstrate software
behavior, not ecological criteria. Use current source code, not PyPI v0.2.0.

Save `flow.csv` and `duration.toml` in one folder. Open a terminal in that
folder; relative input paths resolve from the terminal's working folder:

```console
hydropattern run duration.toml --no-excel
```

Before source publication, use your local checkout from that same folder:

```powershell
uv run --no-default-groups --project C:\path\to\hydropattern hydropattern run duration.toml --no-excel
```

Magnitude outcomes are `[1, 1, 1, 1, 0, 1, 1, 0, 0]`. Both components assess
these qualifying runs in their entirety:

| Component | Duration condition | Expected component outcome | Total portion |
| --- | --- | --- | ---: |
| `long_flow` | At least 3 qualifying timesteps | `[1, 1, 1, 1, 0, 0, 0, 0, 0]` | 4/9 |
| `bounded_flow` | Between 2 and 3, inclusive | `[0, 0, 0, 0, 0, 1, 1, 0, 0]` | 2/9 |

The four-timestep run passes `>= 3` on all four timesteps, not only from its
third timestep onward. That same run is too long for `[2, 3]`, so none of it
passes the bounded condition. The two-timestep run passes the lower bound.
Final component outcomes match their duration diagnostics.

`duration_output` contains `flow_long_flow.csv`, `flow_bounded_flow.csv`,
`long_flow_summary.xlsx`, and `bounded_flow_summary.xlsx`. Each workbook has
magnitude, duration, and component sheets, with `total` and `2021` rows.
Magnitude's portion is 6/9 in both workbooks. `--no-excel` selects raw CSV;
summary workbooks remain Excel. Outputs are overwritten by default.

See the [duration reference](../../docs/user/reference/characteristics/duration.md).
