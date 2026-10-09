# First evaluation

This small illustrative example identifies flow above 1 for at least two
consecutive daily timesteps. The units are arbitrary; these thresholds
demonstrate the software, not ecological criteria.

Save `flow.csv` and `first-run.toml` in the same folder. Open a terminal in that
folder, then use the installed current-source application:

```console
hydropattern run first-run.toml --no-excel
```

For a local development checkout, use the same working folder and supply the
checkout location to uv, for example on Windows:

```powershell
uv run --no-default-groups --project C:\path\to\hydropattern hydropattern run first-run.toml --no-excel
```

The magnitude outcomes are `[0, 1, 1, 0, 1, 0, 0, 1]`. Whole-run duration
assessment leaves final component outcomes `[0, 1, 1, 0, 0, 0, 0, 0]`.
`first-run_output` contains `flow_sustained_flow.csv` and
`sustained_flow_summary.xlsx`; the component's `total` portion is 0.25.
The summary workbook is still written when raw results use CSV.

See [first evaluation](../../docs/user/getting-started/first-run.md) and
[interpreting first results](../../docs/user/getting-started/results.md).
