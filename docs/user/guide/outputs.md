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
reports the same quantity from 0 to 100. Each outcome column has its own
denominator: only known outcomes count. For example, `[1, 0, unknown,
unknown]` has a `portion` of `0.5`, not `0.25`. An outcome column with known
failures and no successes has a `portion` of `0`; a column with no known
outcomes has no defined summary and is left blank.

The `total` row combines successes and known outcomes across the whole record;
it is not an average of water-year portions. Characteristic and component
columns can have different known-outcome coverage, so each is summarized
separately. Water-year rows use the configured water-year boundary. The
[glossary](../concepts/glossary.md#outcomes) distinguishes unknown outcomes
from failures.

The component summary is not interchangeable with any one characteristic
sheet. For example, a flow observation can meet a magnitude condition but
fail a later duration condition. See
[evaluation order](../concepts/evaluation-order.md).

## Response-surface files

When plotting is enabled for a valid scenario grid, the output also includes
a `{component}_grid.csv` file and `{component}_plot.png` for each component.
See the
[plotting guide](plotting.md) for naming requirements and options.
