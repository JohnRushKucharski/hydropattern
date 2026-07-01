'''Parses data from configuration file.'''
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from hydropattern import patterns
from hydropattern.errors import ParserErrorCode, raise_parser_error
from hydropattern.patterns import CharacteristicType


#region Specification classes
# validated raw toml like data.
@dataclass(frozen=True)
class CharacteristicSpec:
    '''Pure-data specification for a single characteristic.

    All fields are plain values; no executable closures.
    Comparable via == for equivalent input detection.
    '''
    type: CharacteristicType
    operator: str | None          # None for between/timing form; stripped symbol e.g. ">"
    values: tuple[float | int, ...] # one value for simple; two for between/timing
    ma_periods: int = 1           # moving-average window (magnitude, rate_of_change, frequency)
    look_back: int = 1            # look-back periods (rate_of_change only)
    min_val: float = 0.0          # minimum denominator value (rate_of_change only)
    order: int = 1                # position in evaluation sequence


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


def validate_metrics_not_empty(metrics: list[Any], characteristic: str) -> None:
    '''Raise PARSER_MISSING_FIELD when metrics list is empty or absent.'''
    if not metrics:
        raise_parser_error(
            ParserErrorCode.MISSING_FIELD,
            f'{characteristic} metrics are required but missing or empty.',
            characteristic=characteristic,
        )
#endregion

def parse_components(data: dict[str, Any]) -> list[patterns.Component]:
    '''Build components. Delegates to parse_request + build_components.'''
    return build_components(parse_request(data))

ComparisionType = Enum('ComparisionType', ['SIMPLE', 'BETWEEN'])

#region: utility parsers
_VALID_SYMBOLS: frozenset[str] = frozenset({'<', '<=', '>', '>=', '=', '!='})

def normalize_operator(raw: str) -> str:
    '''Strip whitespace and validate a comparison symbol.

    Raises HydropatternError (UNKNOWN_COMPARISON_SYMBOL) for unrecognized symbols.
    Returns the stripped symbol string.
    '''
    stripped = raw.strip()
    if stripped not in _VALID_SYMBOLS:
        raise_parser_error(
            ParserErrorCode.UNKNOWN_COMPARISON_SYMBOL,
            f'Invalid comparison symbol: {raw!r}. Valid symbols: {sorted(_VALID_SYMBOLS)}.',
            symbol=raw,
        )
    return stripped

def symbol_to_string(symbol: str) -> str:
    '''Convert symbol to string name.'''
    return {
        '<': 'lt',
        '<=': 'le',
        '>': 'gt',
        '>=': 'ge',
        '=': 'eq',
        '!=': 'ne'
    }[symbol]

def between_parser(metrics: list[Any], inclusive=True) -> Callable[[float], bool]:
    '''Generates comparision function for between metrics (i.e., [minimum, maximum]).'''
    if len(metrics) != 2 or not all(isinstance(i, (int, float)) for i in metrics):
        raise_parser_error(
            ParserErrorCode.INVALID_TYPE,
            'Between metrics must have two numeric values.',
            metrics=metrics,
        )
    if metrics[0] >= metrics[1]:
        raise_parser_error(
            ParserErrorCode.INVALID_VALUE,
            'Between metrics must have values in ascending order.',
            metrics=metrics,
        )
    if inclusive:
        return patterns.comparison_fx('<=', metrics[0], '<=', metrics[1])
    return patterns.comparison_fx('<', metrics[0], '<', metrics[1])
#endregion

#region: validation utilities
def validate_symbol(symbol: str) -> str:
    '''Normalize and validate a comparison symbol. Returns the stripped symbol.'''
    return normalize_operator(symbol)

def validate_simple_comparision_pair(metrics: list[Any]) -> None:
    '''Validate comparision pair. Normalizes metrics[0] in place.'''
    metrics[0] = validate_symbol(metrics[0])
    if not isinstance(metrics[1], (int, float)):
        raise_parser_error(
            ParserErrorCode.INVALID_TYPE,
            f'''Comparision requires a symbol, threshold value pair,
                          ({metrics[0]}, {metrics[1]}) found.''',
            metrics=metrics,
        )

def validate_between_comparision_pair(metrics: list[Any]) -> None:
    '''Validate between comparision pair.'''
    if not isinstance(metrics[1], (int, float)):
        raise_parser_error(
            ParserErrorCode.INVALID_TYPE,
            f'''Between comparision requires two threshold values,
                          [{metrics[0]}, {metrics[1]}] found.''',
            metrics=metrics,
        )
    if metrics[0] >= metrics[1]:
        raise_parser_error(
            ParserErrorCode.INVALID_VALUE,
            f'''Between comparsion requires two threshold values in accending order,
                         [{metrics[0]}, {metrics[1]}] found.''',
            metrics=metrics,
        )

def validate_comparison_metrics(metrics: list[Any]) -> ComparisionType:
    '''Validate magnitude and duration comparison metrics.'''
    if isinstance(metrics[0], str):
        validate_simple_comparision_pair(metrics)
        return ComparisionType.SIMPLE
    if isinstance(metrics[0], (int, float)):
        validate_between_comparision_pair(metrics)
        return ComparisionType.BETWEEN
    raise_parser_error(
        ParserErrorCode.INVALID_TYPE,
        f'Invalid comparision metrics: {metrics}.',
        metrics=metrics,
    )

def _validate_int_param(
    metrics: list[Any], index: int, name: str, min_val: int = 1
) -> None:
    '''Validate that metrics[index] is an integer >= min_val.'''
    value = metrics[index]
    if not isinstance(value, int):
        raise_parser_error(
            ParserErrorCode.INVALID_TYPE,
            f'{name} must be an integer >= {min_val}, got {value!r}.',
            metrics=metrics,
        )
    if value < min_val:
        raise_parser_error(
            ParserErrorCode.INVALID_VALUE,
            f'{name} must be >= {min_val}, got {value}.',
            metrics=metrics,
        )


def _validate_threshold_range(
    metrics: list[Any],
    comparison_type: 'ComparisionType',
    characteristic: str,
    minimum: float,
    exclusive: bool = False,
) -> None:
    '''Validate that threshold value(s) satisfy a minimum bound.

    Parameters
    ----------
        metrics: The raw metrics list (used in error context).
        comparison_type: SIMPLE checks metrics[1]; BETWEEN checks metrics[0] and metrics[1].
        characteristic: Name used in the error message.
        minimum: Lower bound.
        exclusive: When True the bound is strict (> minimum); otherwise >= minimum.
    '''
    bound_desc = f'> {minimum}' if exclusive else f'>= {minimum}'

    def _ok(val: float) -> bool:
        return val > minimum if exclusive else val >= minimum

    if comparison_type == ComparisionType.SIMPLE:
        if not _ok(metrics[1]):
            raise_parser_error(
                ParserErrorCode.INVALID_VALUE,
                f'{characteristic} threshold must be {bound_desc}, got {metrics[1]}.',
                metrics=metrics,
            )
    else:  # BETWEEN
        if not _ok(metrics[0]) or not _ok(metrics[1]):
            raise_parser_error(
                ParserErrorCode.INVALID_VALUE,
                f'{characteristic} between values must both be {bound_desc}, '
                f'got [{metrics[0]}, {metrics[1]}].',
                metrics=metrics,
            )


def validate_ma_period(metrics: list[Any]) -> None:
    '''Validate moving average period is an integer >= 1.'''
    _validate_int_param(metrics, 2, 'ma_periods')

def validate_boolean(name: str, metrics: Any) -> None:
    '''Validate boolean.'''
    if not isinstance(metrics, bool):
        raise_parser_error(
            ParserErrorCode.INVALID_TYPE,
            f'Boolean value expected for {name}, {metrics} found.',
            name=name,
            value=metrics,
        )

def validate_verbose(order: int, metrics: Any) -> None:
    '''Validate verbose.'''
    warning_msg = f'''
                "verbose = {metrics}" appeared after {order} component characteristics.
                First {order} characteristics evaluated as "verbose = True".
                '''
    validate_boolean('verbose', metrics)
    if metrics and order != 1:
        print(warning_msg)

def validate_look_back(metrics: list[Any]) -> None:
    '''Validate look back period is an integer >= 1.'''
    _validate_int_param(metrics, 3, 'look_back')
#endregion

#region: timing parser
#region: timing validation
def validate_timing_metrics(metrics: list[Any]) -> None:
    '''Validate timing metrics.

    Parameters
    ----------
        metrics (list[int]): in the form [first_doy, last_doy]
            where first_doy and last_doy are calendar day-of-year values in [1, 366].
            Both values are inclusive.
            first_doy == last_doy is valid (single-day window).
            first_doy > last_doy is valid (cross-year wrap-around window,
            e.g. [335, 60] = 1 December through 1 March).
    Raises
    ------
        HydropatternError: PARSER_MISSING_FIELD if metrics is empty.
        HydropatternError: PARSER_INVALID_VALUE if len != 2 or any doy outside [1, 366].
        HydropatternError: PARSER_INVALID_TYPE if any value is not an integer.
    '''
    validate_metrics_not_empty(metrics, 'timing')
    if len(metrics) != 2:
        raise_parser_error(
            ParserErrorCode.INVALID_VALUE,
            f'Timing metrics must have exactly 2 values [first_doy, last_doy], '
            f'{len(metrics)} provided: {metrics}.',
            metrics=metrics,
        )
    if not all(isinstance(i, int) for i in metrics):
        raise_parser_error(
            ParserErrorCode.INVALID_TYPE,
            f'Timing day-of-year values must be integers, got: {metrics}.',
            metrics=metrics,
        )
    for doy in metrics:
        if not 1 <= doy <= 366:
            raise_parser_error(
                ParserErrorCode.INVALID_VALUE,
                f'Timing day-of-year values must be in [1, 366], got {doy} in {metrics}.',
                metrics=metrics,
            )
#endregion
def timing_window_fx(first_doy: int, last_doy: int) -> Callable[[float], bool]:
    '''Build a timing window comparison function handling cross-year wrap-around.

    Parameters
    ----------
        first_doy (int): First calendar day-of-year in the window (inclusive).
        last_doy (int): Last calendar day-of-year in the window (inclusive).

    Returns
    -------
        Callable: Returns True when the input day-of-year falls within the window.
            For first_doy <= last_doy: first_doy <= doy <= last_doy.
            For first_doy > last_doy: doy >= first_doy OR doy <= last_doy
                (cross-year wrap-around, e.g. [335, 60] = Dec through Feb).
    '''
    if first_doy <= last_doy:
        return patterns.comparison_fx('<=', first_doy, '<=', last_doy)
    lower = patterns.comparison_fx('>=', first_doy)
    upper = patterns.comparison_fx('<=', last_doy)
    def wrap_fx(doy: float) -> bool:
        return lower(doy) or upper(doy)
    return wrap_fx

def timing_parser(metrics: list[Any], order: int) -> patterns.Characteristic:
    '''Parse timing metrics.

    Parameters
    ----------
        metrics (list[int]): in the form...
            [start(int), end(int)]
            where start and end are first and last day of the water year
            over which the characteristic is evaluated.
            Note: start and end are inclusive.
        order (int): Position in which characteristic is evaluated.
    Returns
    -------
        Characteristic: characteristic name and function.
    Raises
    ------
        ValueError: if metrics are not in the correct form.
    '''
    validate_timing_metrics(metrics)
    return patterns.Characteristic(
        name=f'{patterns.CharacteristicType.TIMING.name.lower()}_{metrics[0]}-{metrics[1]}',
        fx=patterns.timing_fx(timing_window_fx(metrics[0], metrics[1]), order),
        type=patterns.CharacteristicType.TIMING
    )
#endregion

#region: magnitude parser
#region: magnitude validation
def validate_magnitude_metrics(metrics: list[Any]) -> ComparisionType:
    '''Validate magnitude metrics.

    Parameters
    ----------
        metrics (list[Any]): in the form...
            [symbol, threshold, (optional)ma_periods] or
            [minimum, maximum, (optional)ma_periods]
            where symbol is a comparison string (i.e., <, <=, etc.),
            threshold and minimum/maximum are real numbers >= 0, and
            ma_periods is an integer >= 1 (number of timesteps for moving average).
    Returns
    -------
        ComparisionType: type of comparison (Simple or Between).
    Raises
    ------
        HydropatternError: PARSER_MISSING_FIELD if metrics is empty.
        HydropatternError: PARSER_INVALID_VALUE if length or values are out of range.
        HydropatternError: PARSER_INVALID_TYPE if values are wrong type.
    '''
    validate_metrics_not_empty(metrics, 'magnitude')
    error_msg = (
        f'Magnitude metrics must be [symbol, threshold(>=0), (optional)ma_periods(>=1)] '
        f'or [minimum(>=0), maximum(>=0), (optional)ma_periods(>=1)], got: {metrics}.'
    )
    nentries = len(metrics)
    if nentries not in (2, 3):
        raise_parser_error(
            ParserErrorCode.INVALID_VALUE,
            error_msg,
            metrics=metrics,
        )
    if nentries == 3:
        validate_ma_period(metrics)
    comparison_type = validate_comparison_metrics(metrics)
    _validate_threshold_range(metrics, comparison_type, 'Magnitude', minimum=0.0)
    return comparison_type
#endregion
def magnitude_parser(metrics: list[Any], order: int) -> patterns.Characteristic:
    '''Parse magnitude metrics.

    Parameters
    ----------
        metrics (list[Any]): in the form...
            [symbol, threshold, (optional)moving_average_periods] or
            [minimum, maximum, (optional)moving_average_periods]
            where symbol is a comparision string (i.e., <, <=, etc.),
            minimum and maximum are exclusive (i.e., <, >,) boundaries for comparisons, and
            moving_average_periods is number of timesteps over which values are averaged.
        order (int): Position in which characteristic is evaluated.
    Returns
    -------
       Characteristic: characteristic name and function.
    Raises
    ------
        ValueError: if metrics are not in the correct form.
    '''
    label = patterns.CharacteristicType.MAGNITUDE.name.lower()
    comparision_type = validate_magnitude_metrics(metrics)
    ma_periods = metrics[2] if len(metrics) == 3 else 1
    match comparision_type:
        case ComparisionType.SIMPLE:
            name=f'{label}_{symbol_to_string(metrics[0])}{metrics[1]}'
            comparison_fx=patterns.comparison_fx(metrics[0], metrics[1])
        case ComparisionType.BETWEEN:
            name=f'{label}_{metrics[0]}-{metrics[1]}'
            comparison_fx=between_parser(metrics[0:2], inclusive=False)
        case _:
            raise_parser_error(
                ParserErrorCode.INVALID_VALUE,
                'Invalid comparision type.',
                metrics=metrics,
            )
    return patterns.Characteristic(
        name=name,
        fx=patterns.magnitude_fx(comparison_fx, order, ma_periods),
        type=patterns.CharacteristicType.MAGNITUDE
    )
#endregion

#region: duration parser
#region: duration validation
def validate_duration_metrics(metrics: list[Any]) -> ComparisionType:
    '''Validate duration metrics.

    Parameters
    ----------
        metrics (list[Any]): in the form [symbol, time_steps] or [min_steps, max_steps]
            where symbol is a comparison string (i.e., <, <=, etc.),
            time_steps, min_steps, and max_steps are integers >= 1.
    Returns
    -------
        ComparisionType: type of comparison (Simple or Between).
    Raises
    ------
        HydropatternError: PARSER_MISSING_FIELD if metrics is empty.
        HydropatternError: PARSER_INVALID_VALUE if length or values are out of range.
        HydropatternError: PARSER_INVALID_TYPE if time_steps values are not integers.
    '''
    validate_metrics_not_empty(metrics, 'duration')
    if len(metrics) != 2:
        raise_parser_error(
            ParserErrorCode.INVALID_VALUE,
            f'Duration metrics must have exactly 2 values, got {len(metrics)}: {metrics}.',
            metrics=metrics,
        )
    comparison_type = validate_comparison_metrics(metrics)
    if comparison_type == ComparisionType.SIMPLE:
        _validate_int_param(metrics, 1, 'time_steps')
    else:  # BETWEEN
        _validate_int_param(metrics, 0, 'min_steps')
        _validate_int_param(metrics, 1, 'max_steps')
    return comparison_type
#endregion

def duration_parser(metrics: list[Any], order: int) -> patterns.Characteristic:
    '''Parse duration metrics.

    Parameters
    ----------
        metrics (list[Any]): in the form...
            [symbol, threshold] or
            [minimum, maximum]
            where symbol is a comparision string (i.e., <, <=, etc.),
            minimum and maximum are exclusive (i.e., <, >,) boundaries for comparisons, and
        order (int): Position in which characteristic is evaluated.
    Returns
    -------
        Characteristic: characteristic name and function.
    Raises
    ------
        ValueError: if metrics are not in the correct form.
    '''
    label = patterns.CharacteristicType.DURATION.name.lower()
    comparision_type = validate_duration_metrics(metrics)
    match comparision_type:
        case ComparisionType.SIMPLE:
            name=f'{label}_{symbol_to_string(metrics[0])}{metrics[1]}'
            comparison_fx=patterns.comparison_fx(metrics[0], metrics[1])
        case ComparisionType.BETWEEN:
            name=f'{label}_{metrics[0]}-{metrics[1]}'
            comparison_fx=patterns.comparison_fx('<', metrics[0], '>', metrics[1])
        case _:
            raise_parser_error(
                ParserErrorCode.INVALID_VALUE,
                'Invalid comparision type.',
                metrics=metrics,
            )
    return patterns.Characteristic(
        name=name,
        fx=patterns.duration_fx(comparison_fx, order),
        type=patterns.CharacteristicType.DURATION
    )
#endregion

#region: rate_of_change parser
#region: rate_of_change validation
def validate_rate_of_change_metrics(metrics: list[Any]) -> ComparisionType:
    '''Validate rate of change metrics.

    Parameters
    ----------
        metrics (list[Any]): in the form...
            [symbol, value(>0), (optional)ma_periods(>=1), (optional)look_back(>=1),
            (optional)min(>=0)] or
            [lower(>0), upper(>0), (optional)ma_periods(>=1), (optional)look_back(>=1),
            (optional)min(>=0)]
            where symbol is a comparison string (i.e., <, <=, etc.),
            value/lower/upper are the threshold(s) compared against z_t = y_t / y_[t-n],
            ma_periods is integer >= 1 (moving average window),
            look_back is integer >= 1 (n in the z_t formula), and
            min is the minimum value for y_[t-n]; defaults to 0 — when min=0, y_[t-n]=0
            will cause a divide-by-zero at runtime (see docs/user/reference.md).
    Returns
    -------
        ComparisionType: type of comparison (Simple or Between).
    Raises
    ------
        HydropatternError: PARSER_MISSING_FIELD if metrics is empty.
        HydropatternError: PARSER_INVALID_VALUE if length or values are out of range.
        HydropatternError: PARSER_INVALID_TYPE if values are wrong type.
    '''
    validate_metrics_not_empty(metrics, 'rate_of_change')
    error_msg = (
        f'Rate-of-change metrics must be '
        f'[symbol, value(>0), (opt)ma_periods(>=1), (opt)look_back(>=1), (opt)min(>=0)] '
        f'or [lower(>0), upper(>0), ...], got: {metrics}.'
    )
    nentries = len(metrics)
    if nentries < 2 or nentries > 5:
        raise_parser_error(
            ParserErrorCode.INVALID_VALUE,
            error_msg,
            metrics=metrics,
        )
    if nentries > 2:
        validate_ma_period(metrics)
    if nentries > 3:
        validate_look_back(metrics)
    if nentries > 4:
        if not isinstance(metrics[4], (int, float)):
            raise_parser_error(
                ParserErrorCode.INVALID_TYPE,
                f'Rate-of-change min must be a real number >= 0, got {metrics[4]}.',
                metrics=metrics,
            )
        if metrics[4] < 0:
            raise_parser_error(
                ParserErrorCode.INVALID_VALUE,
                f'Rate-of-change min must be >= 0, got {metrics[4]}.',
                metrics=metrics,
            )
    comparison_type = validate_comparison_metrics(metrics)
    _validate_threshold_range(
        metrics, comparison_type, 'Rate-of-change', minimum=0.0, exclusive=True
    )
    return comparison_type

#endregion

def rate_of_change_parser(metrics: list[Any], order: int) -> patterns.Characteristic:
    '''Parse rate of change metrics.

    Parameters
    ----------
        metrics (list[Any]): in the form...
            [symbol, threshold, (optional)ma_periods, (optional)look_back, (optional)min] or
            [minimum, maximum, (optional)ma_periods, (optional)look_back, (optional)min]
            where symbol is a comparision string (i.e., <, <=, etc.),
            minimum and maximum are exclusive (i.e., <, >,) boundaries for comparisons, and
            ma_periods number of timesteps over which values are averaged.
                Defaults to 1. Must be 3rd parameter.
            look_back number of timesteps back from current timestep to evaluate rate of change.
                Defaults to 1. Must be 4th parameter.
            min is the minimum value hydrologic value is compared to.
                Defaults to 0. Must be 5th parameter.
        order (int): Position in which characteristic is evaluated.
    Returns
    -------
        Characteristic: characteristic name and function.
    Raises
    ------
        ValueError: if metrics are not in the correct form.
    Notes
    -----
        The order of the optional parameters is important.
            ma_period is always assumed to be the third parameter.
            look_back is always assumed to be the fourth parameter.
            min is always assumed to be the fifth parameter.
    '''
    label = patterns.CharacteristicType.RATE_OF_CHANGE.name.lower()
    comparision_type = validate_rate_of_change_metrics(metrics)
    ma_periods = metrics[2] if len(metrics) > 2 else 1
    look_back = metrics[3] if len(metrics) > 3 else 1
    min_val = metrics[4] if len(metrics) > 4 else 0
    match comparision_type:
        case ComparisionType.SIMPLE:
            name=f'{label}_{symbol_to_string(metrics[0])}{metrics[1]}'
            comparison_fx=patterns.comparison_fx(metrics[0], metrics[1])
        case ComparisionType.BETWEEN:
            name=f'{label}_{metrics[0]}-{metrics[1]}'
            comparison_fx=between_parser(metrics[0:2], inclusive=False)
        case _:
            raise_parser_error(
                ParserErrorCode.INVALID_VALUE,
                'Invalid comparision type.',
                metrics=metrics,
            )
    return patterns.Characteristic(
        name=name,
        fx=patterns.rate_of_change_fx(comparison_fx, order, ma_periods, look_back, min_val),
        type=patterns.CharacteristicType.RATE_OF_CHANGE
    )
#endregion

#region: frequency parser
#region: frequency validation
def validate_frequency_metrics(metrics: list[Any]) -> ComparisionType:
    '''Validate frequency metrics.'''
    error_msg = f'''
                Provided metrics: {metrics} must be in the form:
                [symbol(str), threshold(Real), ma_period(int)] or
                [minimum(Real), maximum(Real), ma_period(int)].
                '''
    if len(metrics) != 3:
        raise_parser_error(
            ParserErrorCode.INVALID_VALUE,
            error_msg,
            metrics=metrics,
        )
    validate_ma_period(metrics)
    return validate_comparison_metrics(metrics)
#endregion

def frequency_parser(metrics: list[Any], order: int) -> patterns.Characteristic:
    '''Parse frequency metrics.

    Parameters
    ----------
        metrics (list[Any]): in the form...
            [symbol, threshold, ma_period] or
            [minimum, maximum, ma_period]
            where symbol is a comparision string (i.e., <, <=, etc.),
            minimum and maximum are exclusive (i.e., <, >,) boundaries for comparisons, and
            ma_period specifies that the comparison condition must be met over each ma_period
            number of years.
        order (int): Position in which characteristic is evaluated.
    Returns
    -------
        Characteristic: characteristic name and function.
    Raises
    ------
        ValueError: if metrics are not in the correct form.
    '''
    label = patterns.CharacteristicType.FREQUENCY.name.lower()
    comparision_type = validate_frequency_metrics(metrics)
    match comparision_type:
        case ComparisionType.SIMPLE:
            name=f'{label}_{symbol_to_string(metrics[0])}{metrics[1]}in{metrics[2]}yrs'
            comparison_fx=patterns.comparison_fx(metrics[0], metrics[1])
        case ComparisionType.BETWEEN:
            name=f'{label}_{metrics[0]}-{metrics[1]}in{metrics[2]}yrs'
            comparison_fx=between_parser(metrics[0:2], inclusive=False)
        case _:
            raise_parser_error(
                ParserErrorCode.INVALID_VALUE,
                'Invalid comparision type.',
                metrics=metrics,
            )
    return patterns.Characteristic(
        name=name,
        fx=patterns.frequency_fx(comparison_fx, order, metrics[2]),
        type=patterns.CharacteristicType.FREQUENCY
    )
#endregion


#region parse toml file like data into "normalized" data used to build specification objects.
def parse_request(data: dict[str, Any]) -> Request:
    '''Parse component configuration data into a stable normalized Request.

    Produces pure-data ComponentSpec / CharacteristicSpec objects with no
    executable closures. Equivalent valid inputs (e.g. whitespace-padded
    operators) produce equal Request objects (comparable via ==).

    Use build_components to convert the returned Request to executable
    Component objects for evaluation.
    '''
    component_specs: list[ComponentSpec] = []
    for component_name, elements in data.items():
        char_specs: list[CharacteristicSpec] = []
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
#endregion

#region metric/formatter options
_VALID_METRIC_MODES: frozenset[str] = frozenset(m.value for m in MetricMode)

def parse_metric_options(data: dict[str, Any]) -> MetricOptions:
    '''Parse the optional top-level [metric] section into a MetricOptions spec.

    Contract
    --------
        - Section is optional. Absent [metric] -> MetricOptions(mode=MetricMode.PORTION).
        - 'mode' key is optional within [metric]. Absent -> defaults to 'portion'.
        - 'mode' must be one of: 'portion', 'percentage', 'return_period'.
        - Unrecognized keys within [metric] raise PARSER_UNKNOWN_OPTION.

    Raises
    ------
        HydropatternError: PARSER_INVALID_TYPE if [metric] is not a table (dict).
        HydropatternError: PARSER_INVALID_VALUE if 'mode' is not a recognized value.
        HydropatternError: PARSER_UNKNOWN_OPTION if an unrecognized key is present.
    '''
    if 'metric' not in data:
        return MetricOptions()

    section = data['metric']
    if not isinstance(section, dict):
        raise_parser_error(
            ParserErrorCode.INVALID_TYPE,
            f'[metric] section must be a table, got: {section!r}.',
            section='metric',
        )

    mode = MetricMode.PORTION
    for key, value in section.items():
        match key:
            case 'mode':
                if not isinstance(value, str) or value not in _VALID_METRIC_MODES:
                    raise_parser_error(
                        ParserErrorCode.INVALID_VALUE,
                        f'metric.mode must be one of {sorted(_VALID_METRIC_MODES)}, '
                        f'got: {value!r}.',
                        section='metric',
                        field='mode',
                        value=value,
                    )
                mode = MetricMode(value)
            case _:
                raise_parser_error(
                    ParserErrorCode.UNKNOWN_OPTION,
                    f'Unrecognized [metric] option: {key!r}.',
                    section='metric',
                    field=key,
                )
    return MetricOptions(mode=mode)
#endregion

#region characteristic specification object (class) building functions
# These functions build CharacteristicSpecification objects from validated metrics.

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
            operator=metrics[0],  # normalized in-place by validate_magnitude_metrics
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
            operator=metrics[0],  # normalized in-place
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
            operator=metrics[0],  # normalized in-place
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
    comp_type = validate_frequency_metrics(metrics)
    ma_periods = int(metrics[2])
    if comp_type == ComparisionType.SIMPLE:
        return CharacteristicSpec(
            type=CharacteristicType.FREQUENCY,
            operator=metrics[0],  # normalized in-place
            values=(metrics[1],),
            ma_periods=ma_periods,
            order=order,
        )
    return CharacteristicSpec(
        type=CharacteristicType.FREQUENCY,
        operator=None,
        values=(metrics[0], metrics[1]),
        ma_periods=ma_periods,
        order=order,
    )
#endregion

#region characteristic building function - from CharacteristicSpecification objects.
def _build_characteristic(spec: CharacteristicSpec) -> patterns.Characteristic:
    '''Convert a CharacteristicSpec to an executable Characteristic.'''
    label = spec.type.name.lower()
    match spec.type:
        case CharacteristicType.TIMING:
            first, last = int(spec.values[0]), int(spec.values[1])
            return patterns.Characteristic(
                name=f'{label}_{first}-{last}',
                fx=patterns.timing_fx(timing_window_fx(first, last), spec.order),
                type=spec.type,
            )
        case CharacteristicType.MAGNITUDE:
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
        case CharacteristicType.DURATION:
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
        case CharacteristicType.RATE_OF_CHANGE:
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
        case CharacteristicType.FREQUENCY:
            if spec.operator is None:
                comp_fx = patterns.comparison_fx('<', spec.values[0], '<', spec.values[1])
                name = f'{label}_{spec.values[0]}-{spec.values[1]}in{spec.ma_periods}yrs'
            else:
                comp_fx = patterns.comparison_fx(spec.operator, spec.values[0])
                name = f'{label}_{symbol_to_string(spec.operator)}{spec.values[0]}in{spec.ma_periods}yrs'
            return patterns.Characteristic(
                name=name,
                fx=patterns.frequency_fx(comp_fx, spec.order, spec.ma_periods),
                type=spec.type,
            )
    raise ValueError(f'Unknown characteristic type: {spec.type}')  # unreachable
#endregion

#region component building function - from ComponentSpecification objects.
def build_components(request: Request) -> list[patterns.Component]:
    '''Convert a Request to a list of executable Component objects.

    This is the builder step: Request (pure data) → list[Component] (executable).
    '''
    return [
        patterns.Component(
            name=spec.name,
            characteristics=[_build_characteristic(cs) for cs in spec.characteristics],
            is_success_pattern=spec.is_success_pattern,
        )
        for spec in request.components
    ]
#endregion
