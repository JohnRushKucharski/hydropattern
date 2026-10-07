'''Request normalization seam extracted from hydropattern.parsers.'''

import warnings
from typing import Any

from hydropattern.errors import ParserErrorCode, raise_parser_error
from hydropattern.parsing.characteristics import (
    ComparisionType,
    is_nested_frequency_shape,
    validate_boolean,
    validate_duration_metrics,
    validate_frequency_metrics,
    validate_magnitude_metrics,
    validate_nested_frequency_metrics,
    validate_rate_of_change_metrics,
    validate_timing_metrics,
)
from hydropattern.parsing.specs import CharacteristicSpec, ComponentSpec, Request
from hydropattern.patterns import CharacteristicType

# Options removed per docs/plans/2026-10-01-pattern-correctness-tdd.md:
# `order` is always inferred from characteristic sequence; `verbose` is gone
# because timing/magnitude/rate_of_change are now unconditionally independent
# diagnostics (duration/frequency unconditionally stay gated).
_REMOVED_OPTIONS = frozenset({'order', 'verbose'})
_CHARACTERISTIC_BUILDERS = {
    'timing': lambda metrics, order: _timing_spec(metrics, order),
    'magnitude': lambda metrics, order: _magnitude_spec(metrics, order),
    'duration': lambda metrics, order: _duration_spec(metrics, order),
    'rate_of_change': lambda metrics, order: _rate_of_change_spec(metrics, order),
    'frequency': lambda metrics, order: _frequency_spec(metrics, order),
}


def _timing_spec(metrics: list[Any], order: int) -> CharacteristicSpec:
    validate_timing_metrics(metrics)
    return CharacteristicSpec(
        type=CharacteristicType.TIMING,
        operator=None,
        values=(metrics[0], metrics[1]),
        order=order,
    )


def _magnitude_spec(metrics: list[Any], order: int) -> CharacteristicSpec:
    comp_type = validate_magnitude_metrics(metrics)
    ma_periods = metrics[2] if len(metrics) == 3 else 1
    if comp_type == ComparisionType.SIMPLE:
        return CharacteristicSpec(
            type=CharacteristicType.MAGNITUDE,
            operator=metrics[0],
            values=(metrics[1],),
            ma_periods=ma_periods,
            order=order,
        )
    return CharacteristicSpec(
        type=CharacteristicType.MAGNITUDE,
        operator=None,
        values=(metrics[0], metrics[1]),
        ma_periods=ma_periods,
        order=order,
    )


def _duration_spec(metrics: list[Any], order: int) -> CharacteristicSpec:
    comp_type = validate_duration_metrics(metrics)
    if comp_type == ComparisionType.SIMPLE:
        return CharacteristicSpec(
            type=CharacteristicType.DURATION,
            operator=metrics[0],
            values=(metrics[1],),
            order=order,
        )
    return CharacteristicSpec(
        type=CharacteristicType.DURATION,
        operator=None,
        values=(metrics[0], metrics[1]),
        order=order,
    )


def _rate_of_change_spec(metrics: list[Any], order: int) -> CharacteristicSpec:
    comp_type = validate_rate_of_change_metrics(metrics)
    ma_periods = metrics[2] if len(metrics) > 2 else 1
    look_back = metrics[3] if len(metrics) > 3 else 1
    min_val = float(metrics[4]) if len(metrics) > 4 else 0.0
    if comp_type == ComparisionType.SIMPLE:
        return CharacteristicSpec(
            type=CharacteristicType.RATE_OF_CHANGE,
            operator=metrics[0],
            values=(metrics[1],),
            ma_periods=ma_periods,
            look_back=look_back,
            min_val=min_val,
            order=order,
        )
    return CharacteristicSpec(
        type=CharacteristicType.RATE_OF_CHANGE,
        operator=None,
        values=(metrics[0], metrics[1]),
        ma_periods=ma_periods,
        look_back=look_back,
        min_val=min_val,
        order=order,
    )


def _frequency_spec(metrics: list[Any], order: int) -> CharacteristicSpec:
    if is_nested_frequency_shape(metrics):
        base, nested = validate_nested_frequency_metrics(metrics)
        return CharacteristicSpec(
            type=CharacteristicType.FREQUENCY,
            operator=base.operator,
            values=base.values,
            big_n=base.big_n,
            exclusive_event_window=base.exclusive_event_window,
            is_nested=True,
            nested_operator=nested.operator,
            nested_values=nested.values,
            nested_big_n=nested.big_n,
            nested_exclusive_event_window=nested.exclusive_event_window,
            order=order,
        )
    parsed = validate_frequency_metrics(list(metrics))
    return CharacteristicSpec(
        type=CharacteristicType.FREQUENCY,
        operator=parsed.operator,
        values=parsed.values,
        big_n=parsed.big_n,
        exclusive_event_window=parsed.exclusive_event_window,
        order=order,
    )


def _parse_compact_characteristics(
        component_name: str, elements: dict[str, Any]) -> tuple[list[Any], bool]:
    '''Compact-dict form: characteristic name -> metrics, in TOML table/dict
    iteration order. Warns (portability): TOML v1.0 does not guarantee
    key/value order within a table, even though this project's tomllib and
    Python dict preserve encountered order.
    '''
    warnings.warn(
        f'''Component '{component_name}' uses the compact characteristic-key form.
        Characteristic order is inferred from key iteration order, which TOML v1.0
        does not guarantee (though tomllib/Python dict preserve it here). Prefer the
        ordered [[components.{component_name}.characteristics]] array form for
        portability.''',
        UserWarning,
        stacklevel=2,
    )
    char_specs: list[Any] = []
    success = True
    order = 1
    for name, metrics in elements.items():
        if name in _REMOVED_OPTIONS:
            raise_parser_error(
                ParserErrorCode.REMOVED_OPTION,
                f'''"{name}" is no longer a supported component option (see
                docs/plans/2026-10-01-pattern-correctness-tdd.md). Characteristic
                order is always inferred from sequence; diagnostic characteristics
                are always independent.''',
                component=component_name,
                option=name,
            )
        if name == 'success_pattern':
            validate_boolean(name, metrics)
            success = metrics
            continue
        builder = _CHARACTERISTIC_BUILDERS.get(name)
        if builder is None:
            raise_parser_error(
                ParserErrorCode.UNKNOWN_CHARACTERISTIC,
                f'Characteristic {name} not found.',
                component=component_name,
                characteristic=name,
            )
        char_specs.append(builder(metrics, order))
        order += 1
    return char_specs, success


def _parse_ordered_characteristics(
        component_name: str, elements: dict[str, Any]) -> tuple[list[Any], bool]:
    '''Ordered array-of-tables form: elements['characteristics'] is a list of
    {'type': ..., 'parameters': [...]} tables; list order is TOML-guaranteed.
    '''
    success = True
    for name, metrics in elements.items():
        if name == 'characteristics':
            continue
        if name in _REMOVED_OPTIONS:
            raise_parser_error(
                ParserErrorCode.REMOVED_OPTION,
                f'''"{name}" is no longer a supported component option (see
                docs/plans/2026-10-01-pattern-correctness-tdd.md). Characteristic
                order is always inferred from sequence; diagnostic characteristics
                are always independent.''',
                component=component_name,
                option=name,
            )
        if name == 'success_pattern':
            validate_boolean(name, metrics)
            success = metrics
            continue
        raise_parser_error(
            ParserErrorCode.UNKNOWN_OPTION,
            f'Unknown component option {name!r}.',
            component=component_name,
            option=name,
        )

    char_specs: list[Any] = []
    for order, entry in enumerate(elements['characteristics'], start=1):
        if not isinstance(entry, dict) or 'type' not in entry:
            raise_parser_error(
                ParserErrorCode.MISSING_FIELD,
                f'''Each entry in '{component_name}'.characteristics must be a table
                with a "type" field.''',
                component=component_name,
            )
        extra_keys = set(entry) - {'type', 'parameters'}
        if extra_keys & _REMOVED_OPTIONS:
            raise_parser_error(
                ParserErrorCode.REMOVED_OPTION,
                f'''Characteristic tables never accept {sorted(extra_keys & _REMOVED_OPTIONS)};
                order is always inferred from array position.''',
                component=component_name,
            )
        if 'metrics' in entry:
            if 'parameters' in entry:
                message = '''Ordered characteristic tables cannot contain both
                "metrics" and "parameters"; remove "metrics" and keep
                "parameters".'''
            else:
                message = '''Ordered characteristic tables require "parameters";
                replace "metrics" with "parameters".'''
            raise_parser_error(
                ParserErrorCode.UNKNOWN_OPTION,
                message,
                component=component_name,
            )
        if extra_keys:
            raise_parser_error(
                ParserErrorCode.UNKNOWN_OPTION,
                f'Unknown characteristic table field(s) {sorted(extra_keys)}.',
                component=component_name,
            )
        char_type = entry['type']
        builder = _CHARACTERISTIC_BUILDERS.get(char_type)
        if builder is None:
            raise_parser_error(
                ParserErrorCode.UNKNOWN_CHARACTERISTIC,
                f'Characteristic {char_type} not found.',
                component=component_name,
                characteristic=char_type,
            )
        if 'parameters' not in entry:
            raise_parser_error(
                ParserErrorCode.MISSING_FIELD,
                f'Characteristic {char_type!r} requires a "parameters" field.',
                component=component_name,
                characteristic=char_type,
            )
        parameters = entry['parameters']
        if not isinstance(parameters, list):
            raise_parser_error(
                ParserErrorCode.INVALID_TYPE,
                'Characteristic "parameters" must be an array.',
                component=component_name,
                characteristic=char_type,
            )
        if not parameters:
            raise_parser_error(
                ParserErrorCode.MISSING_FIELD,
                f'Characteristic {char_type!r} requires non-empty "parameters".',
                component=component_name,
                characteristic=char_type,
            )
        char_specs.append(builder(parameters, order))
    return char_specs, success


def parse_request(data: dict[str, Any]) -> Request:
    '''Parse component configuration data into a stable normalized Request.

    Accepts either the compact characteristic-key form (dict iteration
    order, warns) or the ordered [[components.<name>.characteristics]]
    array-of-tables form (TOML-guaranteed order, no warning). Neither form
    accepts a user-supplied `order` or the removed `verbose` option.
    '''
    component_specs: list[ComponentSpec] = []
    for component_name, elements in data.items():
        if 'characteristics' in elements:
            char_specs, success = _parse_ordered_characteristics(component_name, elements)
        else:
            char_specs, success = _parse_compact_characteristics(component_name, elements)
        if not char_specs:
            raise_parser_error(
                ParserErrorCode.EMPTY_COMPONENT,
                f"Component '{component_name}' has no characteristics.",
                component=component_name,
            )
        component_specs.append(ComponentSpec(
            name=component_name,
            characteristics=tuple(char_specs),
            is_success_pattern=success,
        ))
    return Request(components=tuple(component_specs))

__all__ = ['parse_request']
