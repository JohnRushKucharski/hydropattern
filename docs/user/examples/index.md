# Runnable examples by task

**For v0.3.0:** use the [v0.3.0 source checkout](../getting-started/installation.md),
not the published PyPI v0.2.0 package. These small packs use ordered
characteristic tables with `parameters`. Their arbitrary flow units, seasons,
and thresholds illustrate software behavior, not universal ecological
criteria. The response-surface pack also demonstrates unavailable calculations
and coverage-based eligibility.

Each pack has one authoritative CSV/TOML pair under `examples` in the source
repository, plus instructions and expected results. The site links to those
files rather than keeping a second copy.

## Download and execute a pack

1. Create a separate folder for your chosen pack. Save both linked files
   there with their exact filenames; do not append `.txt`.
2. Open a terminal in that folder. Input paths resolve from the terminal's
   working folder, not from the configuration file's folder.
3. Execute the command in the table below. The named output folder contains
   raw CSV files and component summary workbooks.

Source-file links point to the `v0.3.0` tag and become available when that
release is published. Until then, copy files from your local v0.3.0 source
checkout. These packs illustrate the evaluator behavior included in v0.3.0.

With a local checkout, stay in the copied pack's folder and supply the
checkout location to uv. For example:

```powershell
uv run --no-default-groups --project C:\path\to\hydropattern hydropattern run duration.toml --no-excel
```

`--no-excel` selects CSV for raw timestep results; summary workbooks are still
Excel. Repeated commands overwrite outputs by default. To retain previous
results, choose `--output-dir` or add `--no-overwrite`.

| Task | Files | Command | Output folder |
| --- | --- | --- | --- |
| First evaluation | [CSV](https://raw.githubusercontent.com/JohnRushKucharski/hydropattern/v0.3.0/examples/first-run/flow.csv), [TOML](https://raw.githubusercontent.com/JohnRushKucharski/hydropattern/v0.3.0/examples/first-run/first-run.toml) | `hydropattern run first-run.toml --no-excel` | `first-run_output` |
| Seasonal thresholds | [CSV](https://raw.githubusercontent.com/JohnRushKucharski/hydropattern/v0.3.0/examples/seasonal-thresholds/flow.csv), [TOML](https://raw.githubusercontent.com/JohnRushKucharski/hydropattern/v0.3.0/examples/seasonal-thresholds/seasonal-thresholds.toml) | `hydropattern run seasonal-thresholds.toml --no-excel` | `seasonal-thresholds_output` |
| Duration | [CSV](https://raw.githubusercontent.com/JohnRushKucharski/hydropattern/v0.3.0/examples/duration/flow.csv), [TOML](https://raw.githubusercontent.com/JohnRushKucharski/hydropattern/v0.3.0/examples/duration/duration.toml) | `hydropattern run duration.toml --no-excel` | `duration_output` |
| Frequency | [CSV](https://raw.githubusercontent.com/JohnRushKucharski/hydropattern/v0.3.0/examples/frequency/flow.csv), [TOML](https://raw.githubusercontent.com/JohnRushKucharski/hydropattern/v0.3.0/examples/frequency/frequency.toml) | `hydropattern run frequency.toml --no-excel` | `frequency_output` |
| Multiple scenarios | [CSV](https://raw.githubusercontent.com/JohnRushKucharski/hydropattern/v0.3.0/examples/multiple-scenarios/flow.csv), [TOML](https://raw.githubusercontent.com/JohnRushKucharski/hydropattern/v0.3.0/examples/multiple-scenarios/multiple-scenarios.toml) | `hydropattern run multiple-scenarios.toml --no-excel` | `multiple-scenarios_output` |
| Response-surface coverage | [CSV](https://raw.githubusercontent.com/JohnRushKucharski/hydropattern/v0.3.0/examples/response-surface/flow.csv), [TOML](https://raw.githubusercontent.com/JohnRushKucharski/hydropattern/v0.3.0/examples/response-surface/response-surface.toml) | `hydropattern run response-surface.toml --plot --no-excel` | `response-surface_output` |

## Expected results

Arrays below follow CSV row order and describe final component outcomes:
`1` is component success and `0` is component failure. A successful component
timestep does not necessarily satisfy each preceding condition, particularly
when a terminal frequency window covers it.

| Pack | Scenario / component | Expected component outcome | Whole-record portion |
| --- | --- | --- | ---: |
| First evaluation | `flow` / `sustained_flow` | `[0, 1, 1, 0, 0, 0, 0, 0]` | 2/8 = 0.25 |
| Seasonal thresholds | `flow` / `early_december` | `[0, 0, 1, 0, 0, 0]` | 1/6 |
| Duration | `flow` / `long_flow` | `[1, 1, 1, 1, 0, 0, 0, 0, 0]` | 4/9 |
| Duration | `flow` / `bounded_flow` | `[0, 0, 0, 0, 0, 1, 1, 0, 0]` | 2/9 |
| Frequency | `flow` / `overlapping` | `[0, 1, 1, 1, 1, 1, 1, 1, 1, 0]` | 8/10 = 0.8 |
| Frequency | `flow` / `exclusive` | `[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]` | 5/10 = 0.5 |
| Multiple scenarios | `low_flow` / `sustained_flow` | `[0, 0, 0, 0, 0, 0]` | 0 |
| Multiple scenarios | `high_flow` / `sustained_flow` | `[0, 1, 1, 0, 1, 1]` | 4/6 |

The first evaluation is explained timestep by timestep in
[interpreting first results](../getting-started/results.md).

The seasonal pack uses inclusive calendar days 335-336, December 1-2.
Only December 1 also has flow above 1; high flow outside the season does
not meet the combined condition. See [timing](../reference/characteristics/timing.md)
and [magnitude](../reference/characteristics/magnitude.md).

The duration pack has qualifying runs of four and two timesteps. The
four-timestep run passes `>= 3` in its entirety but fails inclusive bounds
`[2, 3]` in its entirety. The two-timestep run passes those bounds. See
[duration](../reference/characteristics/duration.md).

The frequency pack has two qualifying magnitude timesteps. A window anchored
on January 2 covers January 2-6. With overlap, the January 5 anchor extends
successful coverage through January 9; with exclusive windows, that anchor
is suppressed. Frequency diagnostics match final component outcomes here.
Eight successful timesteps do not imply eight qualifying timesteps or eight
component events. See [frequency](../reference/characteristics/frequency.md).

The multiple-scenario pack evaluates both data columns independently.
`high_flow` has two qualifying runs of two timesteps; `low_flow` has none.
Both scenarios share a component summary workbook, with separate scenario
columns. Their names are not scenario-grid coordinates, so no plot is
requested. See [preparing data](../guide/preparing-data.md) and
[output files](../guide/outputs.md).

## Response-surface coverage

The response-surface pack has four scenario columns on a 2-by-2 grid. A
rate-of-change condition produces an unknown at startup for every scenario;
one scenario also has a zero denominator. The default 90% cutoff includes
the three scenarios with 9/10 known outcomes and withholds `_1_1`, which has
8/10. The three eligible coordinates are non-collinear and render a surface.

| Scenario | Expected component outcome | Known-outcome portion | Coverage | Plot status |
|---|---|---:|---:|---|
| `_0_0` | `[unknown, 1, 0, 1, 0, 1, 0, 1, 0, 1]` | 5/9 | 90% | included |
| `_0_1` | `[unknown, 1, 1, 1, 1, 1, 1, 1, 1, 1]` | 1 | 90% | included |
| `_1_0` | `[unknown, 0, 1, 0, 1, 0, 1, 0, 1, 0]` | 4/9 | 90% | included |
| `_1_1` | `[unknown, 1, 0, unknown, 1, 0, 1, 0, 1, 0]` | 1/2 | 80% | withheld |

Exactly 90% meets the inclusive cutoff. Withholding affects only the plot:
raw outcome files and summary workbook still report all four scenarios. The
coverage CSV records `_1_1` as `below_minimum_coverage`; its summary remains
defined. See [response-surface plotting](../guide/plotting.md) and
[unknown outcomes](../concepts/unknown-outcomes.md).

Each source folder's README lists the exact output filenames, characteristic
summaries, and local-checkout command. Browse the
[example source folders](https://github.com/JohnRushKucharski/hydropattern/tree/v0.3.0/examples)
for those details after the release tag is published.

## Comprehensive configuration

[detailed.toml](https://github.com/JohnRushKucharski/hydropattern/blob/v0.3.0/examples/detailed.toml)
demonstrates all five characteristics, a failure-pattern component, ordered
tables, and commented optional settings. It is an annotated configuration,
not a replacement for the [reference](../reference.md) or a self-contained
download pair. In a local checkout, open a terminal in the repository root:

```powershell
uv run hydropattern run examples\detailed.toml --no-excel
```

The [unknown-outcomes section](../concepts/unknown-outcomes.md) provides
worked duration, frequency, annual, reporting, and plotting examples.
