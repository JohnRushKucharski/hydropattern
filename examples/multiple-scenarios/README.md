# Evaluate multiple scenarios in one input file

This illustrative pack applies the same component to two numeric columns:
`low_flow` and `high_flow`. Each is a separate scenario, not a reference or
an evaluation role. Flow units are arbitrary; thresholds are not ecological
recommendations. This pack targets v0.3.0; it does not run as documented with
PyPI v0.2.0.

Save `flow.csv` and `multiple-scenarios.toml` in one folder. Open a terminal
there; relative input paths resolve from that working folder:

```console
hydropattern run multiple-scenarios.toml --no-excel
```

Before source publication, use your local checkout from that same folder:

```powershell
uv run --no-default-groups --project C:\path\to\hydropattern hydropattern run multiple-scenarios.toml --no-excel
```

The component requires flow above 1 for at least two consecutive timesteps.
The two qualifying runs in `high_flow` each contain two timesteps:

| Scenario | Expected component outcome | Total portion |
| --- | --- | ---: |
| `low_flow` | `[0, 0, 0, 0, 0, 0]` | 0 |
| `high_flow` | `[0, 1, 1, 0, 1, 1]` | 4/6 |

For each scenario, magnitude and duration diagnostics match the component
array here. These descriptive fractions do not compare ecological benefit.
The names are not scenario-grid coordinates, so this pack does not request
a response-surface plot.

`multiple-scenarios_output` contains `low_flow_sustained_flow.csv`,
`high_flow_sustained_flow.csv`, and `sustained_flow_summary.xlsx`. Each
summary sheet has one column per scenario, with `total` and `2021` rows.
`--no-excel` selects raw CSV; summary workbooks remain Excel. Outputs are
overwritten by default.

See [preparing data](../../docs/user/guide/preparing-data.md) and
[output files](../../docs/user/guide/outputs.md).
