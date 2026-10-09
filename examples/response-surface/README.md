# Response-surface plotting with uncertain outcomes

This small example creates a four-scenario grid from ten daily observations.
The flow values and ratio threshold illustrate software behavior; they are
not ecological recommendations. The rate-of-change calculation is unavailable
at the first observation and after a zero denominator, producing unknown
outcomes without missing input data.

Save `flow.csv` and `response-surface.toml` in one folder. Open a terminal
there; relative input paths resolve from that working folder:

```console
hydropattern run response-surface.toml --plot --no-excel
```

For a local checkout before source publication:

```powershell
uv run --no-default-groups --project C:\path\to\hydropattern hydropattern run response-surface.toml --plot --no-excel
```

The `[output.plot].minimum_coverage` setting is 0.9. Three scenarios have
9/10 known outcomes and are eligible; `_1_1` has 8/10 and is withheld. The
eligible scenarios are non-collinear, so a two-dimensional response region
remains.

| Scenario | Known / total | Coverage | Portion | Plot status |
|---|---:|---:|---:|---|
| `_0_0` | 9 / 10 | 90% | 5/9 | included |
| `_0_1` | 9 / 10 | 90% | 1 | included |
| `_1_0` | 9 / 10 | 90% | 4/9 | included |
| `_1_1` | 8 / 10 | 80% | 1/2 | withheld |

At exactly 90%, a scenario remains eligible. The summary workbook and raw
CSV retain all four scenarios; only the plotted grid excludes `_1_1`. The
output folder contains four raw scenario CSVs, `increasing_flow_summary.xlsx`,
`increasing_flow_grid.csv`, `increasing_flow_grid_coverage.csv`, and
`increasing_flow_plot.png`.

See the [plotting guide](../../docs/user/guide/plotting.md) for cutoff
configuration, protected gaps, and interpretation.
