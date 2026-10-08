# Compare overlapping and exclusive frequency windows

This illustrative pack requires at least one qualifying timestep in a
five-timestep forward window. Only January 2 and 5 qualify under magnitude.
Flow units are arbitrary; thresholds are not ecological recommendations.
Use current source code, not PyPI v0.2.0.

Save `flow.csv` and `frequency.toml` in one folder. Open a terminal there;
relative input paths resolve from that working folder:

```console
hydropattern run frequency.toml --no-excel
```

Before source publication, use your local checkout from that same folder:

```powershell
uv run --no-default-groups --project C:\path\to\hydropattern hydropattern run frequency.toml --no-excel
```

Magnitude outcomes are `[0, 1, 0, 0, 1, 0, 0, 0, 0, 0]`.

| Component | Window rule | Expected component outcome | Total portion |
| --- | --- | --- | ---: |
| `overlapping` | Default overlap union | `[0, 1, 1, 1, 1, 1, 1, 1, 1, 0]` | 8/10 |
| `exclusive` | `exclusive_windows = true` | `[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]` | 5/10 |

The January 2 window covers January 2-6. With overlap, the January 5 anchor
adds January 5-9. Exclusive windows suppress the January 5 anchor because
it lies inside the first successful window. Frequency is terminal, so its
diagnostic determines the final component outcome. Eight successful
component timesteps do not mean eight qualifying magnitude timesteps or
eight component events. Record-end windows use available observations;
the characteristic reference gives a separate truncation example.

`frequency_output` contains `flow_overlapping.csv`, `flow_exclusive.csv`,
`overlapping_summary.xlsx`, and `exclusive_summary.xlsx`. Each workbook has
magnitude, frequency, and component sheets, with `total` and `2021` rows.
Magnitude's portion is 2/10 in both workbooks. `--no-excel` selects raw CSV;
summary workbooks remain Excel. Outputs are overwritten by default.

See the [frequency reference](../../docs/user/reference/characteristics/frequency.md).
