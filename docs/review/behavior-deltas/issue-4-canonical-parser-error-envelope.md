# Issue 4 Behavior Delta Report

## What changed

- Parser and CLI validation failures now raise a shared hydropattern error envelope instead of plain `ValueError`/`NotImplementedError` strings for the covered parser paths.
- The envelope carries stable fields: `code`, `message`, `context`, and `source`.
- Representative invalid inputs now map deterministically to stable codes such as `PARSER_MISSING_SECTION`, `PARSER_MISSING_FIELD`, `PARSER_INVALID_TYPE`, `PARSER_INVALID_VALUE`, `PARSER_UNKNOWN_CHARACTERISTIC`, and `PARSER_UNKNOWN_COMPARISON_SYMBOL`.

## Winner determination

The canonical envelope is the correct winner for this slice.

Rationale:

- It gives parser/service failures a machine-readable contract without changing the core compute model.
- CLI callers can still treat the error as a `ValueError` subtype if needed, but tests and downstream adapters can now inspect stable codes and context.
- The same envelope can be reused later by GUI adapters when they merge from hydropower-doe.

## Focused synthetic test case

- Input: configuration missing the top-level `timeseries` section.
- Expected outcome: `HydropatternError` with `code = PARSER_MISSING_SECTION` and `context.section = timeseries`.

## Regression risk

- Existing callers that assert exact exception class names or exact free-form exception strings may need to switch to envelope-based assertions.
- Parser validation now has more structured error metadata, so any new adapter should avoid parsing human-readable messages.

## Affected code locations

- [hydropattern/errors.py](../../../hydropattern/errors.py)
- [hydropattern/parsers.py](../../../hydropattern/parsers.py)
- [hydropattern/cli.py](../../../hydropattern/cli.py)
- [tests/test_cli.py](../../../tests/test_cli.py)