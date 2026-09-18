'''Builder seam: convert stable request specs into executable pattern components.'''

from importlib import import_module
from typing import Any

from hydropattern import patterns
from hydropattern.errors import ParserErrorCode, raise_parser_error


def _validate_frequency_position(spec: Any) -> None:
    '''A component may have at most one frequency characteristic, and it must be last.'''
    parsers_module = import_module('hydropattern.parsers')
    characteristic_type = getattr(parsers_module, 'CharacteristicType')
    freq_indices = [
        i for i, cs in enumerate(spec.characteristics)
        if cs.type == characteristic_type.FREQUENCY
    ]
    if len(freq_indices) > 1:
        raise_parser_error(
            ParserErrorCode.FREQUENCY_NOT_LAST,
            f'''Component '{spec.name}' has {len(freq_indices)} frequency characteristics;
            at most one is allowed.''',
            component=spec.name,
        )
    if freq_indices and freq_indices[0] != len(spec.characteristics) - 1:
        raise_parser_error(
            ParserErrorCode.FREQUENCY_NOT_LAST,
            f'''Component '{spec.name}' has a frequency characteristic that is not the
            last characteristic in the component.''',
            component=spec.name,
        )


def _build_characteristic(spec: Any) -> patterns.Characteristic:
    '''Convert a CharacteristicSpec to an executable Characteristic.'''
    parsers_module = import_module('hydropattern.parsers')
    characteristic_type = getattr(parsers_module, 'CharacteristicType')
    symbol_to_string = getattr(parsers_module, 'symbol_to_string')
    timing_window_fx = getattr(parsers_module, 'timing_window_fx')
    magnitude_parser = getattr(parsers_module, 'magnitude_parser')
    duration_parser = getattr(parsers_module, 'duration_parser')
    rate_of_change_parser = getattr(parsers_module, 'rate_of_change_parser')

    label = spec.type.name.lower()
    match spec.type:
        case characteristic_type.TIMING:
            first, last = int(spec.values[0]), int(spec.values[1])
            return patterns.Characteristic(
                name=f'{label}_{first}-{last}',
                fx=patterns.timing_fx(timing_window_fx(first, last), spec.order),
                type=spec.type,
            )
        case characteristic_type.MAGNITUDE:
            # Reuse parsers.magnitude_parser (single source of truth for
            # magnitude name/fx construction) instead of reimplementing it here.
            metrics: list[Any] = (
                [spec.values[0], spec.values[1]] if spec.operator is None
                else [spec.operator, spec.values[0]]
            )
            if spec.ma_periods != 1:
                metrics.append(spec.ma_periods)
            return magnitude_parser(metrics, order=spec.order)
        case characteristic_type.DURATION:
            # Reuse parsers.duration_parser (single source of truth for
            # duration name/fx construction) instead of reimplementing it here.
            metrics = (
                [spec.values[0], spec.values[1]] if spec.operator is None
                else [spec.operator, spec.values[0]]
            )
            return duration_parser(metrics, order=spec.order)
        case characteristic_type.RATE_OF_CHANGE:
            # Reuse parsers.rate_of_change_parser (single source of truth for
            # rate_of_change name/fx construction) instead of reimplementing it here.
            metrics = (
                [spec.values[0], spec.values[1]] if spec.operator is None
                else [spec.operator, spec.values[0]]
            )
            # ma_periods/look_back/min_val are positional; only trailing defaults
            # can be omitted, any non-default value requires its predecessors too.
            optional_args = [spec.ma_periods, spec.look_back, spec.min_val]
            defaults = [1, 1, 0.0]
            while optional_args and optional_args[-1] == defaults[len(optional_args) - 1]:
                optional_args.pop()
            metrics.extend(optional_args)
            return rate_of_change_parser(metrics, order=spec.order)
        case characteristic_type.FREQUENCY:
            if spec.is_nested:
                raise ValueError(
                    'Nested frequency specs must be built via _build_nested_frequency_characteristics, '
                    'not _build_characteristic (which only produces a single Characteristic).'
                )
            marker = '(event)' if spec.event_bool else '(timestep)'
            if spec.operator is None:
                # BETWEEN form: [min_n, max_n, N], inclusive bounds (see ADR 0001).
                comp_fx = patterns.comparison_fx(
                    '<=', spec.values[0], '<=', spec.values[1]
                )
                name = f'{label}_{spec.values[0]}-{spec.values[1]}in{spec.big_n}{marker}'
            elif spec.big_n is None:
                # PROBABILITY form: [operator, probability].
                comp_fx = patterns.comparison_fx(spec.operator, spec.values[0])
                name = f'{label}_{symbol_to_string(spec.operator)}{spec.values[0]}{marker}'
            else:
                # COUNT form: [operator, n, N].
                comp_fx = patterns.comparison_fx(spec.operator, spec.values[0])
                name = (
                    f'{label}_{symbol_to_string(spec.operator)}{spec.values[0]}'
                    f'in{spec.big_n}{marker}'
                )
            return patterns.Characteristic(
                name=name,
                fx=patterns.frequency_fx(comp_fx, spec.order, spec.big_n, spec.event_bool),
                type=spec.type,
            )
    raise ValueError(f'Unknown characteristic type: {spec.type}')  # unreachable


def _build_nested_frequency_characteristics(spec: Any) -> list[patterns.Characteristic]:
    '''Convert a nested-frequency CharacteristicSpec into [intra_annual, interannual]
    Characteristics via parsers.nested_frequency_parser (reuses the same
    validation/comparison-building logic the parsing-seam already ran).
    '''
    parsers_module = import_module('hydropattern.parsers')
    nested_frequency_parser = getattr(parsers_module, 'nested_frequency_parser')
    base_metrics: list[Any] = (
        [spec.operator, *spec.values]
        if spec.operator is not None else list(spec.values)
    )
    if spec.big_n is not None:
        base_metrics.append(spec.big_n)
    if not spec.event_bool:
        base_metrics.append(spec.event_bool)

    nested_metrics: list[Any] = (
        [spec.nested_operator, *spec.nested_values]
        if spec.nested_operator is not None else list(spec.nested_values)
    )
    if spec.nested_big_n is not None:
        nested_metrics.append(spec.nested_big_n)
    if not spec.nested_event_bool:
        nested_metrics.append(spec.nested_event_bool)

    return nested_frequency_parser([base_metrics, nested_metrics], spec.order)


def build_components(request: Any) -> list[patterns.Component]:
    '''Convert a Request to a list of executable Component objects.'''
    components = []
    for spec in request.components:
        _validate_frequency_position(spec)
        characteristics: list[patterns.Characteristic] = []
        for cs in spec.characteristics:
            if cs.is_nested:
                characteristics.extend(_build_nested_frequency_characteristics(cs))
            else:
                characteristics.append(_build_characteristic(cs))
        components.append(patterns.Component(
            name=spec.name,
            characteristics=characteristics,
            is_success_pattern=spec.is_success_pattern,
        ))
    return components


__all__ = ['build_components']
