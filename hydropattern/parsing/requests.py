'''Request normalization seam extracted from hydropattern.parsers.'''

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
    validate_verbose,
)
from hydropattern.parsing.specs import CharacteristicSpec, ComponentSpec, Request
from hydropattern.patterns import CharacteristicType


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
            event_bool=base.event_bool,
            is_nested=True,
            nested_operator=nested.operator,
            nested_values=nested.values,
            nested_big_n=nested.big_n,
            nested_event_bool=nested.event_bool,
            order=order,
        )
    parsed = validate_frequency_metrics(list(metrics))
    return CharacteristicSpec(
        type=CharacteristicType.FREQUENCY,
        operator=parsed.operator,
        values=parsed.values,
        big_n=parsed.big_n,
        event_bool=parsed.event_bool,
        order=order,
    )


def parse_request(data: dict[str, Any]) -> Request:
    '''Parse component configuration data into a stable normalized Request.'''
    component_specs: list[ComponentSpec] = []
    for component_name, elements in data.items():
        char_specs: list[Any] = []
        verbose, success, order = True, True, 1
        for name, metrics in elements.items():
            match name:
                case 'timing':
                    order = 1 if verbose else order
                    char_specs.append(_timing_spec(metrics, order))
                case 'magnitude':
                    order = 1 if verbose else order
                    char_specs.append(_magnitude_spec(metrics, order))
                case 'duration':
                    char_specs.append(_duration_spec(metrics, order))
                case 'rate_of_change':
                    order = 1 if verbose else order
                    char_specs.append(_rate_of_change_spec(metrics, order))
                case 'frequency':
                    char_specs.append(_frequency_spec(metrics, order))
                case 'verbose':
                    validate_verbose(order, metrics)
                    verbose = metrics
                case 'success_pattern':
                    validate_boolean(name, metrics)
                    success = metrics
                case _:
                    raise_parser_error(
                        ParserErrorCode.UNKNOWN_CHARACTERISTIC,
                        f'Characteristic {name} not found.',
                        component=component_name,
                        characteristic=name,
                    )
            order += 1
        component_specs.append(ComponentSpec(
            name=component_name,
            characteristics=tuple(char_specs),
            is_success_pattern=success,
            verbose=verbose,
        ))
    return Request(components=tuple(component_specs))

__all__ = ['parse_request']
