# Python API

The Python API is secondary to the command-line guides. Install hydropattern
in your own Python environment using the
[API installation instructions](../getting-started/installation.md#install-for-the-python-api).
These examples require current source, not PyPI v0.2.0.

## Evaluate one scenario

This example selects the `flow` column and evaluates magnitude followed by
un-nested frequency. It uses the ordered characteristic form to retain
explicit evaluation order.

```python
import pandas as pd

from hydropattern.parsers import build_components, parse_request
from hydropattern.patterns import evaluate_component

source = [0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
data = pd.DataFrame(
    {"flow": source, "dowy": range(1, len(source) + 1)},
    index=pd.date_range("2020-01-01", periods=len(source), name="time"),
)
request = parse_request(
    {"pulse": {"characteristics": [
        {"type": "magnitude", "parameters": [">", 0]},
        {"type": "frequency", "parameters": [">=", 1, 5]},
    ]}}
)
result = evaluate_component(data, build_components(request)[0])

print(result.df["frequency_ge1in5(union)"].tolist())
# [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
# Columns: flow, dowy, magnitude_gt0, frequency_ge1in5(union), pulse
```

Frequency counts qualifying timesteps, not distinct component events. Its
successful windows mark timesteps even where the preceding magnitude
condition is not met. See the [frequency reference](../reference.md#frequency).

`evaluate_component(..., data_column=0)` selects one zero-based data-column
position. The final `dowy` column cannot be selected. The result retains the
selected scenario's name; see
[result-column details](../reference.md#python-evaluation-and-result-columns).

## Read structured errors

Parser and plot errors expose a `HydropatternError` with a code, message,
context, and source:

```python
from hydropattern.errors import HydropatternError
from hydropattern.parsers import timing_parser

try:
    timing_parser([0, 100], order=1)
except HydropatternError as exc:
    print(exc.envelope.code)     # PARSER_INVALID_VALUE
    print(exc.envelope.message)
    print(exc.envelope.context)
    print(exc.envelope.source)   # parser
```

The [reference](../reference.md#parser-error-codes) lists existing error codes.
Review [migration guidance](../migration.md) before updating Python calls.
