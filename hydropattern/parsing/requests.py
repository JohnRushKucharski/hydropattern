'''Request normalization seam extracted from hydropattern.parsers.'''

from importlib import import_module
from typing import Any

from hydropattern.errors import ParserErrorCode, raise_parser_error


def _timing_spec(metrics: list[Any], order: int) -> Any:
    parsers_module = import_module('hydropattern.parsers')
    validate_timing_metrics = getattr(parsers_module, 'validate_timing_metrics')
    characteristic_spec_cls = getattr(parsers_module, 'CharacteristicSpec')
    characteristic_type = getattr(parsers_module, 'CharacteristicType')
    validate_timing_metrics(metrics)
    return characteristic_spec_cls(
        type=characteristic_type.TIMING,
        operator=None,
        values=(metrics[0], metrics[1]),
        order=order,
    )


def _magnitude_spec(metrics: list[Any], order: int) -> Any:
    parsers_module = import_module('hydropattern.parsers')
    validate_magnitude_metrics = getattr(parsers_module, 'validate_magnitude_metrics')
    characteristic_spec_cls = getattr(parsers_module, 'CharacteristicSpec')
    characteristic_type = getattr(parsers_module, 'CharacteristicType')
    comparison_type = getattr(parsers_module, 'ComparisionType')
    comp_type = validate_magnitude_metrics(metrics)
    ma_periods = metrics[2] if len(metrics) == 3 else 1
    if comp_type == comparison_type.SIMPLE:
        return characteristic_spec_cls(
            type=characteristic_type.MAGNITUDE,
            operator=metrics[0],
            values=(metrics[1],),
            ma_periods=ma_periods,
            order=order,
        )
    return characteristic_spec_cls(
        type=characteristic_type.MAGNITUDE,
        operator=None,
        values=(metrics[0], metrics[1]),
        ma_periods=ma_periods,
        order=order,
    )


def _duration_spec(metrics: list[Any], order: int) -> Any:
    parsers_module = import_module('hydropattern.parsers')
    validate_duration_metrics = getattr(parsers_module, 'validate_duration_metrics')
    characteristic_spec_cls = getattr(parsers_module, 'CharacteristicSpec')
    characteristic_type = getattr(parsers_module, 'CharacteristicType')
    comparison_type = getattr(parsers_module, 'ComparisionType')
    comp_type = validate_duration_metrics(metrics)
    if comp_type == comparison_type.SIMPLE:
        return characteristic_spec_cls(
            type=characteristic_type.DURATION,
            operator=metrics[0],
            values=(metrics[1],),
            order=order,
        )
    return characteristic_spec_cls(
        type=characteristic_type.DURATION,
        operator=None,
        values=(metrics[0], metrics[1]),
        order=order,
    )


def _rate_of_change_spec(metrics: list[Any], order: int) -> Any:
    parsers_module = import_module('hydropattern.parsers')
    validate_rate_of_change_metrics = getattr(parsers_module, 'validate_rate_of_change_metrics')
    characteristic_spec_cls = getattr(parsers_module, 'CharacteristicSpec')
    characteristic_type = getattr(parsers_module, 'CharacteristicType')
    comparison_type = getattr(parsers_module, 'ComparisionType')
    comp_type = validate_rate_of_change_metrics(metrics)
    ma_periods = metrics[2] if len(metrics) > 2 else 1
    look_back = metrics[3] if len(metrics) > 3 else 1
    min_val = float(metrics[4]) if len(metrics) > 4 else 0.0
    if comp_type == comparison_type.SIMPLE:
        return characteristic_spec_cls(
            type=characteristic_type.RATE_OF_CHANGE,
            operator=metrics[0],
            values=(metrics[1],),
            ma_periods=ma_periods,
            look_back=look_back,
            min_val=min_val,
            order=order,
        )
    return characteristic_spec_cls(
        type=characteristic_type.RATE_OF_CHANGE,
        operator=None,
        values=(metrics[0], metrics[1]),
        ma_periods=ma_periods,
        look_back=look_back,
        min_val=min_val,
        order=order,
    )


def _frequency_spec(metrics: list[Any], order: int) -> Any:
    parsers_module = import_module('hydropattern.parsers')
    validate_frequency_metrics = getattr(parsers_module, 'validate_frequency_metrics')
    characteristic_spec_cls = getattr(parsers_module, 'CharacteristicSpec')
    characteristic_type = getattr(parsers_module, 'CharacteristicType')
    comparison_type = getattr(parsers_module, 'ComparisionType')
    comp_type = validate_frequency_metrics(metrics)
    ma_periods = int(metrics[2])
    if comp_type == comparison_type.SIMPLE:
        return characteristic_spec_cls(
            type=characteristic_type.FREQUENCY,
            operator=metrics[0],
            values=(metrics[1],),
            ma_periods=ma_periods,
            order=order,
        )
    return characteristic_spec_cls(
        type=characteristic_type.FREQUENCY,
        operator=None,
        values=(metrics[0], metrics[1]),
        ma_periods=ma_periods,
        order=order,
    )


def parse_request(data: dict[str, Any]) -> Any:
    '''Parse component configuration data into a stable normalized Request.'''
    parsers_module = import_module('hydropattern.parsers')
    component_spec_cls = getattr(parsers_module, 'ComponentSpec')
    request_cls = getattr(parsers_module, 'Request')
    validate_verbose = getattr(parsers_module, 'validate_verbose')
    validate_boolean = getattr(parsers_module, 'validate_boolean')
    component_specs: list[Any] = []
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
        component_specs.append(component_spec_cls(
            name=component_name,
            characteristics=tuple(char_specs),
            is_success_pattern=success,
            verbose=verbose,
        ))
    return request_cls(components=tuple(component_specs))

__all__ = ['parse_request']
