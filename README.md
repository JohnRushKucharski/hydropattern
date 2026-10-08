# hydropattern
Evaluate hydrologic time series against configured flow-pattern components.
Designed for hydrologists and environmental scientists, hydropattern provides
a command-line application and a secondary Python API.

## Version status

**Unreleased documentation:** these instructions describe current source and
the next release, not PyPI v0.2.0. That release has different frequency behavior
and does not support the current ordered-table syntax. A matching release
version has not yet been assigned.

The source commands and downloads below use `docs-reporting-s3`. Until that
branch is pushed, use the [local-checkout instructions](docs/user/getting-started/installation.md#local-checkout-route-before-publication).
Do not substitute the older published package.

## Documentation

Start with the [user documentation](docs/user/index.md), then follow:

- [Installation](docs/user/getting-started/installation.md)
- [First evaluation](docs/user/getting-started/first-run.md)
- [Interpreting first results](docs/user/getting-started/results.md)
- [Glossary](docs/user/concepts/glossary.md)
- [Configuration and characteristic reference](docs/user/reference.md)
- [Upgrade guidance](docs/user/migration.md)

The Material/MkDocs site is available for local preview. GitHub Pages publishing
is not enabled yet; it waits for the coordinated reporting work and release.

## Inputs and results

Provide a CSV or Excel file with a `time` column and one or more observation
columns. Each observation column is a scenario, evaluated independently.
A TOML file specifies components, ordered characteristic conditions, and
optional output settings.

For example, a small CSV can contain a date column and one flow scenario:

```csv
time,flow
2020-01-01,0
2020-01-02,2
```

Results include raw timestep outcomes and component summary workbooks.
Scenario grids can also produce response-surface plots. Characteristics cover
timing, magnitude, duration, frequency, and rate of change.
See [CLI usage](docs/user/guide/cli.md) for output choices and working folders.

## Install current source

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and Git. After the source branch is published:

```console
uv python install 3.12
uv tool install --python 3.12 "hydropattern @ git+https://github.com/JohnRushKucharski/hydropattern.git@docs-reporting-s3"
uv tool update-shell
hydropattern --help
```

Restart your terminal if uv requests it. Ordinary users do not need development
or test dependency groups. The installation guide provides platform-specific
steps, a pip/virtual-environment alternative, and separate
[Python API installation](docs/user/getting-started/installation.md#install-for-the-python-api).

After the matching release exists on PyPI, the recommended CLI command will be
`uv tool install hydropattern`; do not use it for these unreleased examples yet.

## First evaluation

Save [flow.csv](https://raw.githubusercontent.com/JohnRushKucharski/hydropattern/docs-reporting-s3/examples/first-run/flow.csv)
and [first-run.toml](https://raw.githubusercontent.com/JohnRushKucharski/hydropattern/docs-reporting-s3/examples/first-run/first-run.toml)
in one folder. The downloads become available when the source branch is pushed;
local reviewers already have them under `examples/first-run`.

Open a terminal **in that folder**, then execute:

```console
hydropattern run first-run.toml --no-excel
```

Relative input-data paths resolve from the terminal's working folder, not the
TOML file's location. No repository clone is needed for the published downloads.
The command creates `first-run_output/flow_sustained_flow.csv` and
`first-run_output/sustained_flow_summary.xlsx`.

The example marks two of eight fully known timesteps as component success:
its total portion is 0.25. It identifies flow above 1 for at least two
consecutive timesteps; isolated threshold exceedances fail duration.
Configured success does not establish ecological benefit. Read the results
guide before interpreting other configurations or unknown outcomes.

## Scientific foundations

Natural flow regimes describe hydrologic variation relevant to ecosystems;
functional flows connect that variation to environmental processes.
See [Poff et al. (1997), *The Natural Flow Regime*](https://doi.org/10.2307/1313099)
and [Yarnell et al. (2020), *A functional flows approach*](https://doi.org/10.1002/rra.3575).
Illustrative thresholds are not universal ecological criteria.

## License

hydropattern is available under the
[GNU General Public License v3 or later](LICENSE.txt).
