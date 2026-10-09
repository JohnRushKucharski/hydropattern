# User reference

This page is the entry point to detailed configuration, characteristic,
output, CLI, and Python API guidance. It replaces the former combined
reference's long-form sections while deliberately retaining its path and
important anchors for existing links.

## Start with a topic

| Topic | Reference |
|---|---|
| Input, component, and output TOML | [Configuration reference](reference/configuration.md) |
| Seasonal calendar window | [Timing](reference/characteristics/timing.md) |
| Flow thresholds and ranges | [Magnitude](reference/characteristics/magnitude.md) |
| Relative change between observations | [Rate of change](reference/characteristics/rate-of-change.md) |
| Consecutive qualifying observations | [Duration](reference/characteristics/duration.md) |
| Qualifying timesteps and water years in windows | [Frequency](reference/characteristics/frequency.md) |
| Exact `run` flags and error codes | [Command-line reference](reference/cli.md) |
| Output files | [Output guide](guide/outputs.md) |
| Scenario-grid plots | [Plotting guide](guide/plotting.md) |
| Python evaluation and result columns | [Python API](api/index.md) |

For the concepts shared across characteristics, see
[scientific foundations](concepts/scientific-foundations.md) and
[evaluation order and interpretation](concepts/evaluation-order.md).

## Configuration overview

<a id="configuration-overview"></a>

Each TOML file describes the input time series, one or more components, and
optional output settings. Begin with the
[configuration reference](reference/configuration.md) or the
[first evaluation](getting-started/first-run.md).

<a id="timeseries-options"></a>
### Time-series options

See [input time-series settings](reference/configuration.md#input-time-series).
Input columns, dates, and working-folder behavior are described in
[preparing data](guide/preparing-data.md).

<a id="characteristic-parameters"></a>
<a id="characteristic-reference"></a>
### Characteristic parameters

Each characteristic has its own page with purpose, parameter types, defaults,
units, and worked results:
[timing](reference/characteristics/timing.md),
[magnitude](reference/characteristics/magnitude.md),
[rate of change](reference/characteristics/rate-of-change.md),
[duration](reference/characteristics/duration.md), and
[frequency](reference/characteristics/frequency.md).

<a id="timing"></a>
### Timing

See the [timing reference](reference/characteristics/timing.md).

<a id="magnitude"></a>
### Magnitude

See the [magnitude reference](reference/characteristics/magnitude.md).

<a id="duration"></a>
### Duration

See the [duration reference](reference/characteristics/duration.md).

<a id="rate-of-change"></a>
### Rate of change

See the [rate-of-change reference](reference/characteristics/rate-of-change.md).

<a id="frequency"></a>
### Frequency

See the [frequency reference](reference/characteristics/frequency.md) for
un-nested, intra-annual, and interannual patterns.

<a id="component-options"></a>
### Component options

See [component settings](reference/configuration.md#component-settings) and
[evaluation order](concepts/evaluation-order.md).

<a id="python-evaluation-and-result-columns"></a>
### Python evaluation and result columns

See [Python evaluation and result columns](api/index.md#evaluate-one-scenario).

## Outputs and command line

<a id="output-options"></a>
### Output options

See [output settings](reference/configuration.md#output-settings) and
[output files](guide/outputs.md).

<a id="run-toml-options--override-toml-options"></a>
<a id="run-toml-options-override-toml-options"></a>
<a id="-run-toml-options-override-toml-options"></a>
<a id="outputmetric"></a>
<a id="output.metric"></a>
<a id="outputplot-and-outputplotclimate-canvas"></a>
<a id="output.plot-and-output.plot.climate-canvas"></a>
The former reference sections for TOML/CLI precedence and the
`[output.metric]` and `[output.plot]` tables now live in the
[configuration reference](reference/configuration.md#output-settings) and
[command-line reference](reference/cli.md#run-options).

<a id="response-surface-plots---plot"></a>
<a id="response-surface-plots-plot"></a>
### Response-surface plots

See the [plotting guide](guide/plotting.md) for scenario-grid requirements
and configuration. The `--plot` and related flags are listed in the
[command-line reference](reference/cli.md).

<a id="parser-error-codes"></a>
### Parser error codes

See [parser error codes](reference/cli.md#parser-and-plot-error-codes).

<a id="plot-error-codes"></a>
### Plot error codes

See [plot error codes](reference/cli.md#parser-and-plot-error-codes).

<a id="accessing-error-details-programmatically"></a>
The former programmatic error-details section is now in the
[Python API reference](api/index.md#read-structured-errors).
