'''Builder seam: convert stable request specs into executable pattern components.'''

from importlib import import_module
from typing import Any

from hydropattern import patterns


def _build_characteristic(spec: Any) -> patterns.Characteristic:
    '''Convert a CharacteristicSpec to an executable Characteristic.'''
    parsers_module = import_module('hydropattern.parsers')
    characteristic_type = getattr(parsers_module, 'CharacteristicType')
    symbol_to_string = getattr(parsers_module, 'symbol_to_string')
    timing_window_fx = getattr(parsers_module, 'timing_window_fx')

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
            if spec.operator is None:
                comp_fx = patterns.comparison_fx('<', spec.values[0], '<', spec.values[1])
                name = f'{label}_{spec.values[0]}-{spec.values[1]}'
            else:
                comp_fx = patterns.comparison_fx(spec.operator, spec.values[0])
                name = f'{label}_{symbol_to_string(spec.operator)}{spec.values[0]}'
            return patterns.Characteristic(
                name=name,
                fx=patterns.magnitude_fx(comp_fx, spec.order, spec.ma_periods),
                type=spec.type,
            )
        case characteristic_type.DURATION:
            if spec.operator is None:
                comp_fx = patterns.comparison_fx('<', spec.values[0], '>', spec.values[1])
                name = f'{label}_{spec.values[0]}-{spec.values[1]}'
            else:
                comp_fx = patterns.comparison_fx(spec.operator, spec.values[0])
                name = f'{label}_{symbol_to_string(spec.operator)}{spec.values[0]}'
            return patterns.Characteristic(
                name=name,
                fx=patterns.duration_fx(comp_fx, spec.order),
                type=spec.type,
            )
        case characteristic_type.RATE_OF_CHANGE:
            if spec.operator is None:
                comp_fx = patterns.comparison_fx('<', spec.values[0], '<', spec.values[1])
                name = f'{label}_{spec.values[0]}-{spec.values[1]}'
            else:
                comp_fx = patterns.comparison_fx(spec.operator, spec.values[0])
                name = f'{label}_{symbol_to_string(spec.operator)}{spec.values[0]}'
            return patterns.Characteristic(
                name=name,
                fx=patterns.rate_of_change_fx(
                    comp_fx, spec.order, spec.ma_periods, spec.look_back, spec.min_val
                ),
                type=spec.type,
            )
        case characteristic_type.FREQUENCY:
            if spec.operator is None:
                comp_fx = patterns.comparison_fx('<', spec.values[0], '<', spec.values[1])
                name = f'{label}_{spec.values[0]}-{spec.values[1]}in{spec.ma_periods}yrs'
            else:
                comp_fx = patterns.comparison_fx(spec.operator, spec.values[0])
                name = (
                    f'{label}_{symbol_to_string(spec.operator)}{spec.values[0]}'
                    f'in{spec.ma_periods}yrs'
                )
            return patterns.Characteristic(
                name=name,
                fx=patterns.frequency_fx(comp_fx, spec.order, spec.ma_periods),
                type=spec.type,
            )
    raise ValueError(f'Unknown characteristic type: {spec.type}')  # unreachable


def build_components(request: Any) -> list[patterns.Component]:
    '''Convert a Request to a list of executable Component objects.'''
    return [
        patterns.Component(
            name=spec.name,
            characteristics=[_build_characteristic(cs) for cs in spec.characteristics],
            is_success_pattern=spec.is_success_pattern,
        )
        for spec in request.components
    ]


__all__ = ['build_components']
