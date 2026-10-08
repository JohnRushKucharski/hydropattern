# Output files

Each run creates an output folder. By default its name is the configuration
filename with `_output` appended. Choose another location with
`[output].directory` or `--output-dir`. By default, the folder is created
beside the configuration file.

## Timestep results

Raw results contain one row for each observation and include:

| Column | Meaning |
|---|---|
| `time` | Observation date. |
| Input flow column | The selected scenario's original observation. |
| `dowy` | Day of water year, numbered from the configured water-year start on a 365-day scale. |
| Characteristic columns | Each characteristic's diagnostic outcome. |
| Component column | The final component outcome. |

Choose a single Excel workbook with `--excel` (the default), or one CSV file
per scenario and component with `--no-excel`. The component summary workbook
is written in either case. The Excel workbook is named from the input
configuration; it contains a worksheet for each scenario and component
result. CSV filenames include the scenario and component names. See
[interpreting first results](../getting-started/results.md) for a complete
worked output.

## Component summary workbooks

hydropattern writes one `{component}_summary.xlsx` workbook per component.
It has a sheet for each characteristic diagnostic and a sheet for the final
component outcome. Rows include `total` for the whole recorded series and
water-year labels for observations assigned to each water year.

The default `portion` mode reports a fraction from 0 to 1; `percentage`
reports the same quantity from 0 to 100. For a fully known outcome column,
`portion` is the number of rows marked 1 divided by all rows in the
summarized interval. A zero outcome count therefore gives zero. The
first-run example contains only known outcomes; do not apply this
denominator description to unknown-containing data. Summary behavior for
unknown outcomes will be documented with its implementation.

The component summary is not interchangeable with any one characteristic
sheet. For example, a flow observation can meet a magnitude condition but
fail a later duration condition. See
[evaluation order](../concepts/evaluation-order.md).

## Response-surface files

When plotting is enabled for a valid scenario grid, the output also includes
a `{component}_grid.csv` file and `{component}_plot.png` for each component.
See the
[plotting guide](plotting.md) for naming requirements and options.
