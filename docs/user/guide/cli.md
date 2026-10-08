# Command-line usage

The command-line interface (CLI) evaluates observations using a TOML file.
First complete [installation](../getting-started/installation.md) and the
[first evaluation](../getting-started/first-run.md).

## Choose the working folder

Open a terminal in the folder from which the input-data path should resolve.
For example, if the configuration uses `path = "flow.csv"`, the terminal's
working folder must contain `flow.csv`. Relative data paths are not resolved
from the configuration's folder.

Supply the configuration filename, quoting paths that contain spaces:

```console
hydropattern run "my configuration.toml"
```

Help is available without evaluating data:

```console
hydropattern --help
hydropattern run --help
```

## Choose raw-output format and folder

By default, raw results use Excel. Choose CSV with `--no-excel`:

```console
hydropattern run config.toml --no-excel
hydropattern run config.toml --output-dir results
hydropattern run config.toml --no-overwrite
```

Unless overridden, outputs go to `{config_stem}_output` next to the
configuration file. Component summary workbooks are written regardless of the
raw-output format. `--no-overwrite` preserves existing files by giving new
outputs a numbered suffix; the default replaces existing files.

## Combine CLI flags and TOML settings

The optional `[output]` tables configure output format, location, and plotting.
An explicitly supplied CLI flag overrides its corresponding TOML setting;
otherwise the TOML setting or its default applies.

Use `--run-toml-options` to use only the file's output settings. Combining that
flag with any explicit output-related flag raises `CLI_CONFLICTING_OPTIONS`:
remove the extra flags or use the default `--override-toml-options` behavior.

The [output reference](../reference.md#output-options) lists exact fields,
defaults, and flags. [Response-surface plotting](../reference.md#response-surface-plots-plot)
requires a scenario grid, not an arbitrary collection of scenarios.

## Existing frequency example

The existing source example demonstrates compact and ordered characteristic
syntax, overlapping and exclusive windows, and intra-annual/interannual
evaluation. With a source checkout, execute it from the repository root:

```console
uv run --no-default-groups hydropattern run examples/frequency.toml --no-excel
```

For the smaller downloadable example that does not require a checkout, use the
[first evaluation](../getting-started/first-run.md). See the
[frequency reference](../reference.md#frequency) for the existing frequency
rules and worked results.
