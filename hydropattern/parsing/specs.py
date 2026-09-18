'''Shared pure-data specification types: canonical home for CharacteristicSpec,
ComponentSpec, Request, MetricMode, MetricOptions, ClimateCanvasPlotOptions,
PlotOptions, OutputOptions, and TimeseriesSpec.

Relocated from hydropattern.parsers as part of issue #29 (fix import direction):
this module has no dependency on hydropattern.parsers, so hydropattern.parsing
can be imported without a circular reference back into parsers.py.
'''
import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hydropattern.patterns import CharacteristicType

@dataclass(frozen=True)
class CharacteristicSpec:
    '''Pure-data specification for a single characteristic.

    All fields are plain values; no executable closures.
    Comparable via == for equivalent input detection.
    '''
    type: CharacteristicType
    operator: str | None          # None for between/timing form; stripped symbol e.g. ">"
    values: tuple[float | int, ...] # one value for simple; two for between/timing
    ma_periods: int = 1           # moving-average window (magnitude, rate_of_change)
    look_back: int = 1            # look-back periods (rate_of_change only)
    min_val: float = 0.0          # minimum denominator value (rate_of_change only)
    order: int = 1                # position in evaluation sequence
    big_n: int | None = None      # trial-window size N (frequency count/between forms only)
    event_bool: bool = True       # event-level (True) vs timestep-level (False) (frequency only)
    # Nested frequency (frequency = [<base>, [<nested>]]): when is_nested is True,
    # operator/values/big_n/event_bool above describe the BASE (intra-annual)
    # pattern, and nested_* below describe the NESTED (interannual) pattern.
    is_nested: bool = False
    nested_operator: str | None = None
    nested_values: tuple[float | int, ...] = ()
    nested_big_n: int | None = None
    nested_event_bool: bool = True


@dataclass(frozen=True)
class ComponentSpec:
    '''Pure-data specification for a flow regime component.'''
    name: str
    characteristics: tuple[CharacteristicSpec, ...]
    is_success_pattern: bool = True
    verbose: bool = True


@dataclass(frozen=True)
class Request:
    '''Stable normalized internal request shape produced by parser normalization.

    Use build_components to convert to executable Component objects.
    '''
    components: tuple[ComponentSpec, ...]


class MetricMode(Enum):
    '''Supported formatter summary metric modes.

    PORTION:        fraction of timesteps in [0.0, 1.0] where the condition holds.
    PERCENTAGE:      portion expressed on a 0-100 scale (portion * 100).
    RETURN_PERIOD:   1 / portion; undefined (NA) when portion is 0 or NA.
    '''
    PORTION = 'portion'
    PERCENTAGE = 'percentage'
    RETURN_PERIOD = 'return_period'


@dataclass(frozen=True)
class MetricOptions:
    '''Pure-data specification for formatter/metric behavior options.'''
    mode: MetricMode = MetricMode.PORTION


@dataclass(frozen=True)
class ClimateCanvasPlotOptions:
    '''Pure-data specification for [output.plot.climate-canvas] rendering options.

    title/zlabel default to None: when unset, callers fall back to a dynamic default
    (title -> component name; zlabel -> configured metric mode value) rather than a
    static string, since those defaults vary per component/run.
    '''
    interpolate: bool = True
    show: bool = False
    title: str | None = None
    xlabel: str = 'Precipitation Delta (%)'
    ylabel: str = 'Temperature Delta (C)'
    zlabel: str | None = None
    threshold: float | None = None
    color_map: str = 'RdBu'
    color_map_ticks: list[float] | None = None
    fillin: bool = False


@dataclass(frozen=True)
class PlotOptions:
    '''Pure-data specification for the [output.plot] section.'''
    enabled: bool = False
    climate_canvas: ClimateCanvasPlotOptions = field(default_factory=ClimateCanvasPlotOptions)


@dataclass(frozen=True)
class OutputOptions:
    '''Pure-data specification for the top-level [output] section.'''
    directory: str | None = None
    overwrite: bool = True
    excel: bool = True
    metric: MetricOptions = field(default_factory=MetricOptions)
    plot: PlotOptions = field(default_factory=PlotOptions)


def collect_explicit_options(**kwargs: Any) -> dict[str, Any]:
    '''Return only the keyword arguments whose value is not None.

    Used to identify which CLI options were explicitly passed by the user, since
    every CLI-overridable option defaults to None (meaning "not passed, defer to
    the configuration file or that option's own default").
    '''
    return {name: value for name, value in kwargs.items() if value is not None}


def merge_overrides(base: Any, **overrides: Any) -> Any:
    '''Return a copy of frozen dataclass `base` with each explicitly-set override applied.

    `overrides` keys must match `base`'s field names. A None value means "not
    explicitly passed" and is skipped, leaving that field's existing value on
    `base` untouched. Adding a new overridable field only requires passing it
    through at the call site -- no separate hand-written None-check is needed here.
    '''
    changes = collect_explicit_options(**overrides)
    return dataclasses.replace(base, **changes) if changes else base


@dataclass(frozen=True)
class TimeseriesSpec:
    '''Pure-data specification for the [timeseries] TOML section.

    path: required. File path to a *.csv or *.xlsx/*.xls timeseries.
    first_day_of_water_year: day of year (1-365) the water year starts on. Defaults to 1.
    date_format: strftime/strptime format code for the 'time' column. Defaults to ''
        (pandas infers the format automatically).
    sheet_name: Excel sheet name/index to read. Ignored for *.csv files. Defaults to 0.
    '''
    path: str
    first_day_of_water_year: int = 1
    date_format: str = ''
    sheet_name: int | str = 0

__all__ = [
    'CharacteristicSpec',
    'ClimateCanvasPlotOptions',
    'collect_explicit_options',
    'ComponentSpec',
    'merge_overrides',
    'MetricMode',
    'MetricOptions',
    'OutputOptions',
    'PlotOptions',
    'Request',
    'TimeseriesSpec',
]


