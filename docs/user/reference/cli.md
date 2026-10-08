# Command-line reference

The command-line interface evaluates a configuration with the `run`
command:

```console
hydropattern run config.toml
```

Use `hydropattern --help` for command groups and `hydropattern run --help`
for the options supported by the installed version.

## `run` options

| Option | Default | Description |
|---|---|---|
| `--output-dir PATH` | Derived from config filename | Write outputs to this folder. |
| `--excel` / `--no-excel` | `--excel` | Choose Excel or CSV for raw timestep results. |
| `--overwrite` / `--no-overwrite` | `--overwrite` | Replace existing files or create numbered alternatives. |
| `--plot` / `--no-plot` | `--no-plot` | Enable or disable response-surface output. |
| `--interp` / `--no-interp` | `--interp` | Enable or disable surface interpolation. |
| `--show` / `--no-show` | `--no-show` | Also display the plot interactively. |
| `--threshold FLOAT` | Midpoint of plotted range | Center the plot's diverging color scale. |
| `--color-map NAME` | `RdBu` | Choose a Matplotlib colormap. |
| `--color-map-ticks FLOAT` | Renderer-selected | Add a colorbar tick; repeat the option for multiple ticks. |
| `--fillin` / `--no-fillin` | `--no-fillin` | Enable or disable the plot renderer's fill-in option. |
| `--run-toml-options` | Off | Use the configuration's output choices without CLI output overrides. |
| `--override-toml-options` | On | Allow explicitly supplied CLI options to override matching TOML settings. |

When CLI options are used, an explicit value takes precedence over the
corresponding TOML setting. `--run-toml-options` cannot be combined with any
explicit output or plot option; doing so raises `CLI_CONFLICTING_OPTIONS`.
Remove the conflicting flags or use the default override behavior.

For exact configuration fields, see the
[configuration reference](configuration.md). For working-folder guidance,
see [command-line usage](../guide/cli.md).

## Parser and plot error codes

Configuration errors carry a stable code in a `HydropatternError` envelope.
Common parser codes are:

| Code | Meaning |
|---|---|
| `PARSER_MISSING_SECTION` | A required section, such as `[timeseries]`, is absent. |
| `PARSER_MISSING_FIELD` | A required field or characteristic parameters are missing. |
| `PARSER_INVALID_TYPE` | A setting has the wrong type. |
| `PARSER_INVALID_VALUE` | A value is outside the supported range or choices. |
| `PARSER_UNKNOWN_CHARACTERISTIC` | A characteristic name is not recognized. |
| `PARSER_UNKNOWN_COMPARISON_SYMBOL` | A comparison operator is not supported. |
| `PARSER_UNKNOWN_OPTION` | An unsupported setting appears in an options table. |
| `PARSER_FREQUENCY_NOT_LAST` | Frequency is not the last characteristic or appears more than once. |
| `PARSER_FREQUENCY_PROBABILITY_NOT_NESTED` | A probability form is used outside the intra-annual part of nested frequency. |
| `PARSER_EMPTY_COMPONENT` | A component has no characteristics. |
| `PARSER_REMOVED_OPTION` | A setting from an earlier version is no longer supported. |

`PLOT_INVALID_SCENARIO_GRID` means plotting was requested, but the scenario
names do not form a response-surface grid. The envelope source for this
error is `plot`; see [plotting requirements](../guide/plotting.md).

Python callers can inspect the error code and message; see
[structured API errors](../api/index.md#read-structured-errors).
