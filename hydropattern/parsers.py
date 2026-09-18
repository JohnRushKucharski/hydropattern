'''Parses data from configuration file.

This module is a thin public-API facade: canonical implementations live in
hydropattern.parsing (pure-data specs in parsing.specs, characteristic
validation/parsing in parsing.characteristics, and normalization/building seams
in parsing.requests/options/timeseries/builders). Imports flow one direction
only -- parsers.py depends on hydropattern.parsing, never the reverse -- so
there is no import_module/getattr indirection or in-function-body import
needed to dodge a circular dependency (see issue #29).
'''
from typing import Any

from hydropattern import patterns
from hydropattern.patterns import CharacteristicType

from hydropattern.parsing.specs import (
    CharacteristicSpec,
    ClimateCanvasPlotOptions,
    ComponentSpec,
    MetricMode,
    MetricOptions,
    OutputOptions,
    PlotOptions,
    Request,
    TimeseriesSpec,
    collect_explicit_options,
    merge_overrides,
)
from hydropattern.parsing.characteristics import (
    ComparisionType,
    FrequencyForm,
    FrequencyMetrics,
    between_parser,
    duration_parser,
    frequency_parser,
    is_nested_frequency_shape,
    magnitude_parser,
    nested_frequency_parser,
    normalize_operator,
    rate_of_change_parser,
    symbol_to_string,
    timing_parser,
    timing_window_fx,
    validate_between_comparision_pair,
    validate_boolean,
    validate_comparison_metrics,
    validate_duration_metrics,
    validate_frequency_metrics,
    validate_look_back,
    validate_ma_period,
    validate_magnitude_metrics,
    validate_metrics_not_empty,
    validate_nested_frequency_metrics,
    validate_rate_of_change_metrics,
    validate_simple_comparision_pair,
    validate_symbol,
    validate_timing_metrics,
    validate_verbose,
)
from hydropattern.parsing.requests import parse_request
from hydropattern.parsing.options import (
    parse_climate_canvas_plot_options,
    parse_metric_options,
    parse_output_options,
    parse_plot_options,
)
from hydropattern.parsing.timeseries import parse_timeseries_spec
from hydropattern.parsing.builders import build_components


def parse_components(data: dict[str, Any]) -> list[patterns.Component]:
    '''Build components. Delegates to parse_request + build_components.'''
    return build_components(parse_request(data))


__all__ = [
    'CharacteristicSpec',
    'CharacteristicType',
    'ClimateCanvasPlotOptions',
    'ComparisionType',
    'ComponentSpec',
    'FrequencyForm',
    'FrequencyMetrics',
    'MetricMode',
    'MetricOptions',
    'OutputOptions',
    'PlotOptions',
    'Request',
    'TimeseriesSpec',
    'between_parser',
    'build_components',
    'collect_explicit_options',
    'duration_parser',
    'frequency_parser',
    'is_nested_frequency_shape',
    'magnitude_parser',
    'merge_overrides',
    'nested_frequency_parser',
    'normalize_operator',
    'parse_climate_canvas_plot_options',
    'parse_components',
    'parse_metric_options',
    'parse_output_options',
    'parse_plot_options',
    'parse_request',
    'parse_timeseries_spec',
    'rate_of_change_parser',
    'symbol_to_string',
    'timing_parser',
    'timing_window_fx',
    'validate_between_comparision_pair',
    'validate_boolean',
    'validate_comparison_metrics',
    'validate_duration_metrics',
    'validate_frequency_metrics',
    'validate_look_back',
    'validate_ma_period',
    'validate_magnitude_metrics',
    'validate_metrics_not_empty',
    'validate_nested_frequency_metrics',
    'validate_rate_of_change_metrics',
    'validate_simple_comparision_pair',
    'validate_symbol',
    'validate_timing_metrics',
    'validate_verbose',
]
