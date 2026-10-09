# First evaluation

**For v0.3.0:** use the [v0.3.0 source checkout](installation.md), not the
published PyPI v0.2.0 package. This example's ordered tables use the
`parameters` key supported in v0.3.0.

This example looks for flow above 1 for at least two consecutive daily
timesteps. The eight observations are illustrative, with arbitrary flow
units; the threshold is not an ecological recommendation.

## Obtain two files

Create a folder called `hydropattern-first-run`, then save both downloads in it:

- [flow.csv](https://raw.githubusercontent.com/JohnRushKucharski/hydropattern/v0.3.0/examples/first-run/flow.csv)
- [first-run.toml](https://raw.githubusercontent.com/JohnRushKucharski/hydropattern/v0.3.0/examples/first-run/first-run.toml)

These links become available after the v0.3.0 release tag is published. Until
then, copy the pair from `examples/first-run` in a local v0.3.0 checkout. No
full-repository clone is needed to download the pair after publication.

Use your browser's **Save as** action if a link displays text. Preserve the
exact filenames; check that your editor did not append `.txt`.
The [example source folder](https://github.com/JohnRushKucharski/hydropattern/tree/v0.3.0/examples/first-run)
also contains its purpose and expected results after publication.

## Understand the configuration

TOML is a plain-text configuration format. Sections in square brackets group
settings. The downloaded file contains:

```toml
[timeseries]
path = "flow.csv"
date_format = "%Y-%m-%d"

[components.sustained_flow]
[[components.sustained_flow.characteristics]]
type = "magnitude"
parameters = [">", 1]

[[components.sustained_flow.characteristics]]
type = "duration"
parameters = [">=", 2]

[output]
excel = false

[output.metric]
mode = "portion"
```

The ordered characteristic tables assess magnitude first, then duration.
Magnitude checks each observation against 1. Duration assesses the entire
consecutive sequence of qualifying timesteps, not only its last timestep.

## Execute the command

Open your terminal in `hydropattern-first-run`. On Windows, you can open the
folder in File Explorer, type `powershell` in its address bar, and press Enter.
On macOS or Linux, change to the folder using `cd`, supplying its location.

Confirm that **both downloaded files are in the terminal's working folder**.
`path = "flow.csv"` resolves from that working folder, not from the TOML file's
folder. Merely supplying the TOML file's full path does not change this.

With the installed current-source application:

```console
hydropattern run first-run.toml --no-excel
```

The [local-checkout route](installation.md#local-checkout-route-before-publication)
provides the equivalent uv command before remote publication.

## Locate results

The command creates `first-run_output` next to `first-run.toml`, containing:

| File | Purpose |
| --- | --- |
| `flow_sustained_flow.csv` | The observations, characteristic outcomes, and final component outcomes at each timestep. |
| `sustained_flow_summary.xlsx` | Separate summary sheets for the two characteristics and the component. |

`--no-excel` selects CSV for raw timestep results; component summaries still
use Excel. The configured `excel = false` already makes that choice, so the
flag here is explicit rather than required.

Default overwrite behavior replaces existing output files when you repeat
the command. To keep earlier results, add `--no-overwrite` or choose another
folder with `--output-dir`.

Continue with [interpreting first results](results.md).
