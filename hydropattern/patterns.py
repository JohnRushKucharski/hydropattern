'''
Creates evaluation functions for natural flow regime characteristics.

The following characteristics are evaluated:
    - magnitude
    - duration
    - timing
    - rate of change
    - frequency
'''
from collections import namedtuple
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#region comparision functions
def lt(a: float, b: float) -> bool:
    '''Returns True if a is less than b.'''
    return a < b
def le(a: float, b: float) -> bool:
    '''Returns True if a is less than or equal to b.'''
    return a <= b
def gt(a: float, b: float) -> bool:
    '''Returns True if a is greater than b.'''
    return a > b
def ge(a: float, b: float) -> bool:
    '''Returns True if a is greater than or equal to b.'''
    return a >= b
def eq(a: float, b: float) -> bool:
    '''Returns True if a is equal to b.'''
    return a == b
def ne(a: float, b: float) -> bool:
    '''Returns True if a is not equal to b.'''
    return a != b

def comparison_fx(symbol1: str, bound1: float,
                  symbol2: str|None = None, bound2: float|None = None) -> Callable[[float], bool]:
    '''
    Returns the corresponding operator function for the given symbol.

    Examples:
    - comparison_fx('>', 1) -> lambda x: x > 1
    - comparison_fx('<', 1, '>', 0) -> lambda x: 0 < x < 1
    '''
    def closure(s: str, bound: float, is_bound_b: bool = True) -> Callable[[float], bool]:
        '''
        Returns a partially constructed comparison function
        (i.e. built-in python gt(a, b) operator function) for a single bound.

        Parameters
        ----------
            s (str): Comparison symbol (i.e., <, <=, >, >=, =, !=).
            bound (Real): Bound value (i.e., 1.0).
            is_bound_b (bool): If True, the bound is the second argument in the comparison function.
                               If False, the bound is the first argument in the comparison function.
                               Defaults to True.
        Returns
        -------
            Callable[[Real], bool]: Partially constructed comparison function.
        Raises
        -------
            KeyError: For invalid symbol.
        Examples
        -------
            [1] closure('>', 1, True) -> lambda x: 1 > x
                                      -> lt(x, 1) (same as lambda x: x < 1)
                This saves the "lt" function so a value "x" can be compared to the bound 1
                at a later time.
        '''
        symbols = {
            '<': lt,    # a < b
            '<=': le,   # a <= b
            '>': gt,    # a > b
            '>=': ge,   # a >= b
            '=': eq,    # a == b
            '!=': ne    # a != b
        }
        if is_bound_b:
            # Return a function that calls symbols[s](value, bound)
            return lambda value: symbols[s](value, bound)
        # Return a function that calls symbols[s](bound, value)
        return lambda value: symbols[s](bound, value)
    # Single bound, not a between comparison.
    if symbol2 is None and bound2 is None:
        return closure(symbol1, bound1)
    # Two bounds, a between comparison.
    if symbol2 is not None and bound2 is not None:
        # Between comparison cases:
        # - bound1 < value < bound2 (either < could be <=)
        # - bound1 > value > bound2 (either > could be >=)
        # This is provided like: [bound1, symbol1, symbol2, bound2]
        # Python comparisions (lt, gt, etc.) are: a < b, a > b, etc.
        # So it is provided like: [a(~b1), symbol1, symbol2, b(~b2)]
        fx1 = closure(symbol1, bound1, is_bound_b=False)
        fx2 = closure(symbol2, bound2, is_bound_b=True)
        def fx3(value: float) -> bool:
            return fx1(value) and fx2(value)
        return fx3
    # Every bound must have a symbol.
    raise ValueError('symbol2 must be provided if bound2 is provided.')
#endregion

#region characteristics
class CharacteristicType(StrEnum):
    '''Enumeration of characteristic types.'''
    TIMING = 'timing'
    MAGNITUDE = 'magnitude'
    RATE_OF_CHANGE = 'rate_of_change'
    DURATION = 'duration'
    FREQUENCY = 'frequency'
# CharacteristicType = Enum('CharacteristicType',
#                           ['TIMING', 'MAGNITUDE', 'RATE_OF_CHANGE', 'DURATION', 'FREQUENCY'])

type CharacteristicFx = Callable[[pd.DataFrame, None|np.ndarray], np.ndarray]

# is_nested marks the terminal (interannual) column of a nested frequency
# characteristic. evaluate_component() uses this to broadcast the interannual
# result across each qualifying water year instead of the generic row-wise AND
# used for every other characteristic (including un-nested frequency and the
# nested pattern's own intra-annual column). See
# notes/frequencyEnhancement-resolved.md.
Characteristic = namedtuple('Characteristic', ['name', 'fx', 'type', 'is_nested'],
                            defaults=[False])

#region utility functions
# def is_order_1(order: int, output: None|np.ndarray) -> bool:
#     '''Validates order and output for characteristics.
#     Returns
#     -------
#         bool: True if order is 1, False otherwise.
#     Raises
#     -------
#         ValueError: For invalid order and output combinations.
#     '''
#     if order < 1:
#         raise ValueError('Order must be greater than or equal to 1.')
#     if order > 1:
#         if output is None:
#             raise ValueError('Output must be provided for order greater than 1.')
#         # Check if output row lengths are long enough for order (i.e. ncols >= order-1),
#         # by definition numpy arrays are rectangular (no need to check len of each row).
#         if output.shape[1] < order - 1: #columns.
#             raise ValueError(
#               'Order must be less than or equal to the number of columns in output.')
#         return False
#     # order == 1
#     return True

def validate_order(order: int, output: None|np.ndarray,
                   characteristic_type: CharacteristicType) -> None:
    '''Validates order and output for characteristics.

    Returns
    -------
        bool: True if order is valid, False otherwise.
    Raises
    -------
        ValueError: For invalid order and output combinations.
    '''
    if order < 1:
        raise ValueError('Order must be greater than or equal to 1.')
    if order > 1:
        if output is None:
            raise ValueError('Output must be provided for order greater than 1.')
        # Check if output row lengths are long enough for order (i.e. ncols >= order-1),
        # by definition numpy arrays are rectangular (no need to check len of each row).
        if output.shape[1] < order - 1: #columns.
            raise ValueError('Order must be less than or equal to the number of columns in output.')
    # order == 1.
    if (order == 1 and
        characteristic_type not in {CharacteristicType.TIMING,
                                    CharacteristicType.MAGNITUDE,
                                    CharacteristicType.RATE_OF_CHANGE}):
        # valid order 1 characterstics
        raise ValueError(f'''{characteristic_type} characteristics cannot be evaluated first,
                         but was has order = {order}.''')

def is_dowy_timeseries(data: np.ndarray) -> bool:
    '''Checks if every value is integer in range [1, 365].'''
    return all(0 < i < 366 for i in data) and all(i.is_integer() for i in data)

def moving_average(data: np.ndarray,
                   period: int, min_periods: None|int = None) -> np.ndarray:
    '''
    Calculates moving average over timeseries data.

    Parameters
    ----------
    data (np.ndarray): timeseries data to average.
    period (int): window (in timesteps) over which to average.
    min_periods (None|int): minimum number of timesteps before average is computed.
    Defaults to period, i.e. average is computed after 'period' number of timesteps.

    Returns
    -------
    np.ndarray: moving average of data, with same shape as input data.
    '''
    if period < 1:
        raise ValueError(
            f'moving average period: {period} must be at least 1.')
    if min_periods and (period < min_periods or min_periods < 1):
        raise ValueError(
            f'''min_periods: {min_periods} must be at least 1 and
            less than or equal to the moving average period: {period}.''')

    # adjust to account for 0-based idx
    periods = period - 1
    min_periods = min_periods - 1 if min_periods else periods

    # convolve approach: faster but less clear...
    # ma = np.convolve(data, np.ones(period), 'valid') / period
    # return np.pad(ma, (len(data)-len(ma), 0), 'constant', constant_values=np.nan)

    ma = np.zeros(len(data))
    for t in range(len(data)):
        if t < min_periods:
            ma[t] = np.nan
        else:
            if t < periods:
                # t+1 bc max of slice is exclusive
                ma[t] = np.mean(data[:t+1])
            else:
                # t+1 bc max of slice is exclusive
                ma[t] = np.mean(data[t-periods:t+1])
    return ma

def eval_order_1_characteristic(f: Callable[[float], bool], data: np.ndarray) -> np.ndarray:
    '''Evaluates eligble order 1 characteristic, returning array of [0, 1] values.'''
    return np.array([1 if f(value) else 0 for value in data], dtype=int)

def eval_order_n_characteristic(f: Callable[[float], bool], data: np.ndarray,
                                output: np.ndarray, order: int) -> np.ndarray:
    '''Evaluates eligble order n characteristic, returning array of [0, 1] values.'''
    precedents = output[:, :order-1]
    eligible = (precedents == 1).all(axis=1)
    return np.array([
        1 if is_eligible and f(value) else 0
        for value, is_eligible in zip(data, eligible)
    ], dtype=int)
#endregion

#region timing
def timing_fx(f: Callable[[float], bool],
              order: int = 1) -> CharacteristicFx:
    '''Creates function to evaluate timing characteristics.

    Parameters
    ----------
        f (Callable[[Real], bool]): Comparision function.
        order (int): Position in which characteristic is evaluated
            within list of component characteristics.
            Defaults to 1 for timing characteristics.
    Returns
    -------
        Characteristic_fx: evaluates characteristic over timeseries.
    '''
    def closure(df: pd.DataFrame,
                output: None|np.ndarray = None) -> np.ndarray:
        # uses dowy (last) df column
        data = np.asarray(df.iloc[:, -1].values)
        if not is_dowy_timeseries(data):
            raise ValueError('''Timing characteristics must be evaluated on a
                             day of water year timeseries.''')
        validate_order(order, output, CharacteristicType.TIMING)
        return (eval_order_1_characteristic(f, data) if order == 1 else
                eval_order_n_characteristic(f, data, output, order)) # type: ignore
    return closure
    #     else: # order > 1
    #         return eval_order_n_characteristic(f, data, output, order) # type: ignore
    # return closure
        # if is_order_1(order, output):
        #     return eval_order_1_characteristic(f, data)
        # # Is valid order > 1
        # if output is not None:
        #     precedents = output[:, :order-1]
        #     eligible = (precedents == 1).all(axis=1)
        #     @np.vectorize
        #     def eligible_fx(data: float, is_eligible: bool) -> int:
        #         return 1 if is_eligible and f(data) else 0
        #     return eligible_fx(data, eligible)

        # else: # is valid order > 1
        #     if output is None:
        #         raise ValueError("Output cannot be None for order > 1")
        #     result = np.zeros(len(data))
        #     for t, row in enumerate(output):
        #         # 1st order-1 values are 1
        #         if np.all(row[-order+1:]==1):
        #             result[t] = 1 if f(data[t]) else 0
        #     return result
#endregion

#region magnitude
def magnitude_fx(f: Callable[[float], bool],
                 order: int = 1, ma_periods: int = 1) -> CharacteristicFx:
    '''
    Creates function to evaluate magnitude characteristics.

    Parameters
    ----------
        f (Callable[[Real], bool]): Comparision function.
        order (int): Position in which characteristic is evaluated
            within list of component characteristics.
            Defaults to 1 for magnitude characteristics.
    Returns
    -------
        Characteristic_fx: evaluates characteristic over timeseries.
    '''
    def closure(df: pd.DataFrame,
                output: None|np.ndarray = None) -> np.ndarray:
        # uses hydrologic data (1st) df column
        data = np.asarray(df.iloc[:, 0].values)
        data = data if ma_periods == 1 else moving_average(data, ma_periods)

        validate_order(order, output, CharacteristicType.MAGNITUDE)
        return (eval_order_1_characteristic(f, data) if order == 1 else
                eval_order_n_characteristic(f, data, output, order)) # type: ignore
    return closure

        # if is_order_1(order, output):
        #     @np.vectorize
        #     def fx(value: float) -> int:
        #         return 1 if f(value) else 0
        #     return fx(data)
        # # Is valid order > 1
        # if output is not None:
        #     precedents = output[:, :order-1]
        #     eligible = (precedents == 1).all(axis=1)
        #     @np.vectorize
        #     def eligible_fx(data: float, is_eligible: bool) -> int:
        #         return 1 if is_eligible and f(data) else 0
        #     return eligible_fx(data, eligible)
        # raise ValueError("Output cannot be None for order > 1")

        # n = len(data)
        # result = np.zeros(n)
        # # restrict t to moving average
        # # test this restriction
        # for t in range(ma_periods-1, n):
        #     if is_order_1(order, output):
        #         result[t] = 1 if f(data[t]) else 0
        #     else: # is valid order > 1
        #         if output is None:
        #             raise ValueError("Output cannot be None for order > 1")
        #         # 1st order-1 values are 1
        #         if np.all(output[t][-order+1:]==1):
        #             result[t] = 1 if f(data[t]) else 0
        #return result
    # return closure
    #     if is_order_1(order, output):
    #         @np.vectorize
    #         def fx(value: Real) -> int:
    #             return 1 if f(value) else 0
    #         out = fx(data)
    #     else: # is valid order > 1
    #         #result = np.zeros(len(data))
    #         for t, row in enumerate(output):
    #             # 1st order-1 values are 1
    #             if np.all(row[-order+1:]==1):
    #                 result[t] = 1 if f(data[t]) else 0
    #         out = result
    #     return out if ma_periods == 1 else np.pad(out, (0, n-len(out)))
    # return closure
#endregion

#region duration
def duration_fx(f: Callable[[float], bool],
                order: int) -> CharacteristicFx:
    '''
    Creates function to evaluate duration characteristics.

    Parameters
    ----------
        f (Callable[[float], bool]): Comparision function.
        order (int): Position in which characteristic is evaluated
            within list of component characteristics.
            Must be greater than 1 for duration characteristics.
    Returns
    -------
        Characteristic_fx: evaluates characteristic over timeseries.
    '''
    def closure(df: pd.DataFrame,
                output: None|np.ndarray) -> np.ndarray:
        # uses output not df to determine duration
        validate_order(order, output, CharacteristicType.DURATION)
        assert output is not None # for mypy: checked by validate_order

        n, T = 0, len(df) # pylint: disable=invalid-name
        result = np.zeros(T)
        for t, row in enumerate(output):
            # from 0th to [order - 1]
            # check if values are all 1s
            if np.all(row[:order-1]==1):
                n += 1
            # break in 1s
            else:
                # n periods of 1s
                if f(n):
                    # start at PREVIOUS period
                    # and count back n periods
                    result[t-n:t] = 1
                n = 0
            # last row
            if t == T-1:
                # n periods of 1s
                if f(n):
                    # start at CURRENT period
                    # and count back n periods
                    result[t+1-n:t+1] = 1
                n = 0
        return result
    return closure
#endregion

#region frequency
def mark_events(raw: np.ndarray, event_bool: bool = True) -> np.ndarray:
    '''
    Collapses maximal runs of consecutive successes in a raw 0/1/NaN diagnostic
    array into event-level or timestep-level markers.

    This is the shared event-marking engine used by both un-nested frequency
    (applied to a sliding-window success diagnostic) and nested frequency
    (applied to the intra-annual base-pattern diagnostic).

    Parameters
    ----------
        raw (np.ndarray): a 0/1/NaN array, e.g. a sliding-window success
            diagnostic. NaN marks insufficient history (no verdict yet).
        event_bool (bool): if True (default), each maximal run of consecutive
            1s collapses to a single 1 marked at the run's last trial
            (event-level); every other trial in the run is set to 0. If False,
            every trial in a qualifying run is marked 1 (timestep-level) and
            `raw` is returned unchanged.

    Returns
    -------
        np.ndarray: same shape as `raw`. NaNs and 0s always pass through
        unchanged; only 1s within a run may be zeroed (event_bool=True).
    '''
    result = np.array(raw, dtype=float)
    if not event_bool:
        return result

    run_start = None
    for t, value in enumerate(result):
        if np.isnan(value):
            run_start = None
            continue
        if value == 1:
            if run_start is None:
                run_start = t
        else:
            if run_start is not None and t - 1 > run_start:
                result[run_start:t - 1] = 0
            run_start = None
    # run continues to the end of the array
    if run_start is not None and len(result) - 1 > run_start:
        result[run_start:len(result) - 1] = 0
    return result

def sliding_window_count(data: np.ndarray, window: int) -> np.ndarray:
    '''
    Trailing sliding-window count of successes over a 0/1 array.

    At each trial `t`, sums the trailing window of `window` trials ending at
    `t` (inclusive). The first `window - 1` trials (insufficient history) are
    marked NaN, matching legacy moving-average/incomplete-first-year behavior
    and ADR 0002 (sliding, not fixed/non-overlapping windows).

    Parameters
    ----------
        data (np.ndarray): 0/1 success array to count over.
        window (int): trailing window size (in trials), i.e. N.

    Returns
    -------
        np.ndarray: same shape as `data`, float dtype (to allow NaN).
    '''
    if window < 1:
        raise ValueError(f'window: {window} must be at least 1.')
    result = np.full(len(data), np.nan)
    for t in range(len(data)):
        if t < window - 1:
            continue
        result[t] = np.sum(data[t - window + 1:t + 1])
    return result

def identify_full_water_years(dowy: np.ndarray) -> list[tuple[int, int]]:
    '''
    Identifies (start_idx, end_idx) index pairs (both inclusive) for each
    water year in a day-of-water-year array.

    A water year starts at any timestep where `dowy == 1` and runs to the
    timestep before the next `dowy == 1` (or the end of the series, for the
    final year). Callers are assumed to supply data trimmed to complete
    water years (as the CLI's timeseries loading does via
    `first_day_of_water_year`); a leading partial year -- timesteps before
    the first `dowy == 1` -- is excluded, since there is no way to recover
    its missing days.

    Parameters
    ----------
        dowy (np.ndarray): day-of-water-year values (1-365).

    Returns
    -------
        list[tuple[int, int]]: (start_idx, end_idx) pairs, in series order.
    '''
    starts = [i for i, day in enumerate(dowy) if day == 1]
    if not starts:
        return []
    return [
        (start, starts[i + 1] - 1 if i + 1 < len(starts) else len(dowy) - 1)
        for i, start in enumerate(starts)
    ]

def water_year_probability_ratio(eligible: np.ndarray, dowy: np.ndarray,
                                 event_bool: bool = True) -> np.ndarray:
    '''
    Computes, for each full water year (see identify_full_water_years), the
    ratio of event-marked successes to timesteps-in-year over an eligible
    (0/1) trial array -- the core statistic behind a nested frequency
    pattern's intra-annual `[operator, probability, (event_bool)]` base form.

    The ratio is placed at the water year's *last* timestep and NaN
    elsewhere (including any leading partial year before the first
    `dowy == 1`), matching the trailing-window convention used by
    sliding_window_count: a diagnostic value only becomes known once its
    full window -- here, the water year -- has elapsed.

    Parameters
    ----------
        eligible (np.ndarray): 0/1 trial outcomes (e.g. AND of preceding
            characteristic columns).
        dowy (np.ndarray): day-of-water-year values (1-365), same length as
            `eligible`.
        event_bool (bool): if True (default), a maximal run of consecutive
            successes within the year collapses to a single success
            (event-level) before computing the ratio; if False, every
            successful timestep counts toward the numerator (timestep-level).

    Returns
    -------
        np.ndarray: same shape as `eligible`; ratio at each full water year's
        last timestep, NaN elsewhere.
    '''
    if len(eligible) != len(dowy):
        raise ValueError(
            f'eligible (len={len(eligible)}) and dowy (len={len(dowy)}) must be the same length.'
        )
    marked = mark_events(eligible, event_bool)
    result = np.full(len(eligible), np.nan)
    for start, end in identify_full_water_years(dowy):
        year_length = end - start + 1
        successes = np.nansum(marked[start:end + 1])
        result[end] = successes / year_length
    return result

def windowed_count_per_water_year(eligible: np.ndarray, dowy: np.ndarray,
                                  window: int) -> np.ndarray:
    '''
    Per full water year (see identify_full_water_years), a trailing
    sliding-window count of successes -- the count/between-form counterpart
    to water_year_probability_ratio, used by a nested frequency pattern's
    intra-annual base when it is a count/between form rather than
    probability. The window resets at each water year boundary (does not
    look back into the previous year), matching the "intra-annual" framing.

    Parameters
    ----------
        eligible (np.ndarray): 0/1 trial outcomes (e.g. AND of preceding
            characteristic columns).
        dowy (np.ndarray): day-of-water-year values (1-365), same length as
            `eligible`.
        window (int): trailing window size (in timesteps), i.e. N.

    Returns
    -------
        np.ndarray: same shape as `eligible`; trailing-window count at each
        timestep within a full water year (NaN for the year's first
        `window - 1` timesteps), NaN throughout any excluded partial year.
    '''
    if len(eligible) != len(dowy):
        raise ValueError(
            f'eligible (len={len(eligible)}) and dowy (len={len(dowy)}) must be the same length.'
        )
    result = np.full(len(eligible), np.nan)
    for start, end in identify_full_water_years(dowy):
        result[start:end + 1] = sliding_window_count(eligible[start:end + 1], window)
    return result

def or_reduce_per_water_year(diag: np.ndarray, dowy: np.ndarray) -> np.ndarray:
    '''
    Per full water year (see identify_full_water_years), OR-reduces a 0/1/NaN
    diagnostic column (e.g. an intra_annual column) to a single year verdict,
    placed at the year's last timestep (NaN elsewhere, including any excluded
    partial year) -- matching the trailing-window "value known only at window
    end" convention used throughout this module.

    Verdict rule: `1` if any `1` is present among the year's non-NaN cells;
    `0` if only `0`s are present; `NaN` only if every cell in the year is NaN
    (insufficient history all year).

    Parameters
    ----------
        diag (np.ndarray): 0/1/NaN diagnostic column (e.g. intra_annual).
        dowy (np.ndarray): day-of-water-year values (1-365), same length as
            `diag`.

    Returns
    -------
        np.ndarray: same shape as `diag`; year verdict at each full water
        year's last timestep, NaN elsewhere.
    '''
    if len(diag) != len(dowy):
        raise ValueError(
            f'diag (len={len(diag)}) and dowy (len={len(dowy)}) must be the same length.'
        )
    result = np.full(len(diag), np.nan)
    for start, end in identify_full_water_years(dowy):
        year = diag[start:end + 1]
        non_nan = year[~np.isnan(year)]
        if len(non_nan) == 0:
            continue  # remains NaN: entire year is insufficient-history
        result[end] = 1.0 if np.any(non_nan == 1) else 0.0
    return result

def frequency_fx(f: Callable[[float], bool], order: int,
                 big_n: int | None = None, event_bool: bool = True) -> CharacteristicFx:
    '''
    Creates function to evaluate an un-nested frequency characteristic.

    Parameters
    ----------
        f (Callable[[float], bool]): Comparision function, applied to either a
            probability (successes/trials ratio) or a trial count, depending on form.
        order (int): Position in which characteristic is evaluated
            within list of component characteristics. Must be the last
            characteristic in the component (enforced upstream in builders.py).
        big_n (int | None): trailing trial-window size (in timesteps) for the
            [op, n, N] and [min_n, max_n, N] forms. None for the
            [op, probability] form (whole-series ratio, no windowing) -- not
            yet implemented as an un-nested form (dropped; see
            notes/frequencyEnhancement-resolved.md -- probability only exists
            as a nested base pattern, task freq-core-probability).
        event_bool (bool): if True (default), a maximal run of consecutive
            qualifying trials counts as a single success, marked at the trial
            where the run ends (event-level). If False, every trial in the run
            is marked a success (timestep-level).
    Returns
    -------
        Characteristic_fx: evaluates characteristic over timeseries.

    Note
    ----
        Windows over the AND-combined success of preceding characteristics in
        the component (same eligibility rule as duration_fx), then compares
        each trailing-window count via `f` (op vs n, or inclusive between vs
        [min_n, max_n]), then collapses to event/timestep level via mark_events.
    '''
    def closure(df: pd.DataFrame,
                output: None|np.ndarray) -> np.ndarray:
        validate_order(order, output, CharacteristicType.FREQUENCY)
        assert output is not None # for mypy: checked by validate_order

        if big_n is None:
            raise NotImplementedError(
                'un-nested [operator, probability] frequency form is not valid; '
                'probability form is only implemented as a nested base pattern '
                '(see notes/frequencyEnhancement-resolved.md).'
            )

        precedents = output[:, :order-1]
        eligible = (precedents == 1).all(axis=1).astype(int)
        counts = sliding_window_count(eligible, big_n)

        diag = np.full(len(counts), np.nan)
        has_count = ~np.isnan(counts)
        diag[has_count] = [1 if f(c) else 0 for c in counts[has_count]]

        return mark_events(diag, event_bool)
    return closure

def _intra_annual_diagnostic(eligible: np.ndarray, dowy: np.ndarray, f: Callable[[float], bool],
                             big_n: int | None) -> np.ndarray:
    '''Shared raw-diagnostic computation for the nested base (intra-annual)
    pattern: per full water year, a probability ratio (big_n is None) or a
    per-year-reset trailing window count (big_n is int), compared via `f`.
    NaN wherever the underlying statistic is undefined (excluded partial
    years, or insufficient in-year window history).
    '''
    stat = (water_year_probability_ratio(eligible, dowy, event_bool=True) if big_n is None
            else windowed_count_per_water_year(eligible, dowy, big_n))
    diag = np.full(len(stat), np.nan)
    has_stat = ~np.isnan(stat)
    diag[has_stat] = [1 if f(value) else 0 for value in stat[has_stat]]
    return diag

def nested_frequency_intra_annual_fx(f: Callable[[float], bool], order: int,
                                     big_n: int | None = None,
                                     event_bool: bool = True) -> CharacteristicFx:
    '''
    Creates function to evaluate the intra-annual (base) column of a nested
    frequency characteristic.

    Per notes/frequencyEnhancement-resolved.md: intra_annual = AND(preceding
    characteristic columns, base-pattern diagnostic), where the base pattern
    may be probability (per-water-year ratio, big_n=None), count, or between
    (windowed within each water year, big_n=N). `event_bool` is display-only:
    it is applied to the AND'd result (not the raw diagnostic), which never
    changes whether a `1` survives somewhere in a qualifying year -- only
    where, within a run, it is marked -- so the eventual year-OR-reduction
    (freq-nested-eval's or_reduce_per_water_year) is unaffected either way.

    Parameters
    ----------
        f (Callable[[float], bool]): Comparison function for the base pattern.
        order (int): Position in the component's characteristic sequence.
        big_n (int | None): trailing trial-window size for count/between base
            forms; None for the probability base form.
        event_bool (bool): base pattern's own event_bool (display-only for the
            eventual per-year OR-reduction; does not change year verdicts).
    Returns
    -------
        Characteristic_fx: evaluates the intra-annual diagnostic column.
    '''
    def closure(df: pd.DataFrame,
                output: None|np.ndarray) -> np.ndarray:
        validate_order(order, output, CharacteristicType.FREQUENCY)
        assert output is not None # for mypy: checked by validate_order

        dowy = np.asarray(df.iloc[:, -1].values, dtype=float)
        precedents = output[:, :order-1]
        eligible = (precedents == 1).all(axis=1).astype(int)

        diag = _intra_annual_diagnostic(eligible, dowy, f, big_n)

        intra_annual_raw = np.full(len(diag), np.nan)
        has_diag = ~np.isnan(diag)
        intra_annual_raw[has_diag] = [
            1 if (e == 1 and d == 1) else 0
            for e, d in zip(eligible[has_diag], diag[has_diag])
        ]
        return mark_events(intra_annual_raw, event_bool)
    return closure

def nested_frequency_interannual_fx(f: Callable[[float], bool], order: int,
                                    big_n: int | None = None,
                                    event_bool: bool = True) -> CharacteristicFx:
    '''
    Creates function to evaluate the interannual (nested) column of a nested
    frequency characteristic -- the terminal column whose result determines
    the component's final pass/fail, broadcast across each qualifying water
    year (see notes/frequencyEnhancement-resolved.md).

    Unlike the base pattern's `event_bool` (display-only), this level's
    `event_bool` changes the actual pass/fail result: a run of consecutive
    qualifying years collapses to a single event-year before broadcasting.

    Parameters
    ----------
        f (Callable[[float], bool]): Comparison function for the nested pattern.
        order (int): Position in the component's characteristic sequence.
            The intra-annual column (this pattern's input) must immediately
            precede this characteristic, at column index `order - 2`.
        big_n (int | None): trailing trial-window size (in years) for count/
            between nested forms. Probability form is not valid at this
            level (enforced upstream in validate_nested_frequency_metrics).
        event_bool (bool): nested pattern's own event_bool; changes the
            actual pass/fail result (a run of qualifying years collapses to
            a single event-year).
    Returns
    -------
        Characteristic_fx: evaluates the interannual column, already
        broadcast across each qualifying water year -- this is directly the
        component's final value when the characteristic is nested (see
        evaluate_component).
    '''
    def closure(df: pd.DataFrame,
                output: None|np.ndarray) -> np.ndarray:
        validate_order(order, output, CharacteristicType.FREQUENCY)
        assert output is not None # for mypy: checked by validate_order
        if big_n is None:
            raise NotImplementedError(
                'nested [operator, probability] interannual frequency form is not valid; '
                'probability is only valid as the intra-annual base pattern '
                '(see notes/frequencyEnhancement-resolved.md).'
            )

        dowy = np.asarray(df.iloc[:, -1].values, dtype=float)
        intra_annual = output[:, order - 2]
        year_verdicts = or_reduce_per_water_year(intra_annual, dowy)

        full_years = identify_full_water_years(dowy)
        compact_verdicts = np.array([year_verdicts[end] for _, end in full_years])
        counts = sliding_window_count(compact_verdicts, big_n)

        compact_diag = np.full(len(counts), np.nan)
        has_count = ~np.isnan(counts)
        compact_diag[has_count] = [1 if f(c) else 0 for c in counts[has_count]]
        compact_diag = mark_events(compact_diag, event_bool)

        result = np.full(len(intra_annual), np.nan)
        for (start, end), verdict in zip(full_years, compact_diag):
            result[start:end + 1] = verdict
        return result
    return closure
#endregion

#region rate_of_change
def rate_of_change_fx(f: Callable[[float], bool],
                      order: int = 1, ma_periods: int = 1,
                      look_back: int = 1, minimum: float = 0.0) -> CharacteristicFx:
    '''
    Creates function to evaluate rate of change characteristics.

    Parameters
    ----------
        f (Callable[[Real], bool]): Comparision function.
        order (int): Position in which characteristic is evaluated
            within list of component characteristics.
            Defaults to 1 for rate of change characteristics.
        ma_periods (int): window (in timesteps) over which to average data before evaluating f.
            Defaults to 1 for no moving average.
        look_back (int): number of time periods back to compare for rate of change calculation.
            Defaults to 1 for comparing to previous time period.
        minimum (float): minimum value to consider in rate of change calculations.
            Defaults to 0.0, i.e. values must be positive to avoid divide by 0s.
    '''
    if look_back < 1:
        raise ValueError(
            f'rate of change look_back must be at least 1 time period, got {look_back}.')
    if minimum < 0:
        raise ValueError(
            f'minimum must be non-negative to avoid divide by 0s, got {minimum}.')
    def closure(df: pd.DataFrame, output: None|np.ndarray=None) -> np.ndarray:
        # uses hydrologic data (1st) df column
        data = np.asarray(df.iloc[:, 0].values).astype(float)
        data = data if ma_periods == 1 else moving_average(data, ma_periods)

        validate_order(order, output, CharacteristicType.RATE_OF_CHANGE)
        if look_back > len(data):
            raise ValueError(
                f'''rate of change look_back: {look_back} must be
                less than or equal to the length of the timeseries: {len(data)}.''')
        # compute rates of change
        data[data <= minimum] = np.nan  # avoid divide by 0s, excludes values <= minimum
        data[look_back:] = data[look_back:] / data[:-look_back]
        data[:look_back] = np.nan  # not in look back window
        # send along for value comparison and eligibility checking
        return (eval_order_1_characteristic(f, data) if order == 1 else
                eval_order_n_characteristic(f, data, output, order)) # type: ignore
    return closure
        # assert output is not None or order == 1 # for mypy: checked by validate_order

        # n = len(data)
        # result = np.zeros(n)
        # # restrict t to moving average
        # for t in range(ma_periods-1, n):
        #     if order == 1:
        #         if t-look_back >= 0:
        #             if data[t-look_back] > minimum:
        #                 result[t] = 1 if f(data[t] / data[t-look_back]) else 0
        #     else: # is valid order > 1
        #         if output is None:
        #             raise ValueError("Output cannot be None for order > 1")
        #         # 1st order-1 values are 1
        #         if np.all(output[t][-order+1:]==1):
        #             if t-look_back >= 0:
        #                 if data[t-look_back] > minimum:
        #                     result[t] = 1 if f(data[t] / data[t-look_back]) else 0
    #     return result
    # return closure
#endregion
#endregion

#region components
@dataclass
class Component:
    '''Natural flow regime type component.'''
    name: str
    characteristics: list[Characteristic]
    is_success_pattern: bool

@dataclass
class Result:
    '''Result of evaluating a component on a timeseries.'''
    df: pd.DataFrame
    component: Component
    dv_name: str = field(init=False)

    def __post_init__(self):
        self.dv_name = self.df.columns[0]
        self.df = self.df.rename(columns={self.dv_name: 'dv'})

    def identify_water_years(self):
        '''Identifies water years in the timeseries.'''
        # yr = np.nan
        data = self.df['dowy']
        wy = np.full(len(data), np.nan)
        for i in range(len(data)):
            # if data.iat[i] == 1:
            #     yr = data.index[i].year
            wy[i] = data.index[i].year
        df = self.df.copy()
        df['water_year'] = wy
        return df
        # for i, row in self.df.iterrows():
        #     if row['dowy'] == 1:
        #         self.df.at[i, 'water_year'] = self.df.index[self.df.index.get_loc(i)-1].year
        # return df

    def frequency_table(self, by_water_years: bool = False) -> pd.DataFrame:
        '''Returns a frequency table of the component success.'''
        T = len(self.df) # pylint: disable=invalid-name
        data = {'T': [T]}
        for _, characteristic in enumerate(self.component.characteristics):
            n = self.df[characteristic.name].sum()
            data[characteristic.name] = [n]
            data[f'{characteristic.name}(%)'] = [(n / T) * 100]
        n = self.df[self.component.name].sum()
        data[self.component.name] = [n]
        data[f'{self.component.name}(%)'] = [(n / T) * 100]
        if by_water_years:
            df = self.identify_water_years().dropna(subset=['water_year'])
            wys = df['water_year'].dropna().unique()
            for _, wy in enumerate(wys):
                df_wy = df[df['water_year'] == wy]
                T = len(df_wy) # pylint: disable=invalid-name
                data['T'].append(T)
                for _, characteristic in enumerate(self.component.characteristics):
                    n_wy = df_wy[characteristic.name].sum()
                    data[characteristic.name].append(n_wy)
                    data[f'{characteristic.name}(%)'].append((n_wy / T) * 100)
                n_wy = df_wy[self.component.name].sum()
                data[self.component.name].append(n_wy)
                data[f'{self.component.name}(%)'].append((n_wy / T) * 100)
            indexs = ['total'] + [str(int(wy)) for wy in wys]
            return pd.DataFrame(data, index=indexs)
        return pd.DataFrame(data)

    def plot_success(self,
                     ylimits: None|tuple[float, float] = None,
                     full_timeseries: bool = True) -> None:
        '''Plot the component success over time.'''
        df = self.df
        if not full_timeseries:
            df = self.df[self.df[self.component.characteristics[0].name] == 1]
        _, ax = plt.subplots(figsize=(15, 5))
        df['success'] = self.df[self.component.name] * self.df.dv
        df['possible'] = self.df[self.component.characteristics[0].name] * self.df.dv
        df.possible.replace({0: np.nan}).plot(
            color='yellow', linewidth=10, label=self.component.characteristics[0].name, ax=ax)
        df.dv.plot(
            color='grey', linewidth=0.5, label=self.dv_name, ax=ax)
        df.success.replace({0: np.nan}).plot(
            color='black', linewidth=1, label=self.component.name, ax=ax)
        # widths = np.arange(
        #     start=1.0 + len(self.component.characteristics) * 0.5,
        #     stop= 1.0,step=-0.5)
        # colors = mpl.colormaps['summer_r'](np.linspace(0, 1, len(self.component.characteristics)))
        # for i, c in enumerate(self.component.characteristics):
        #     df[f'{c.name}_dv'] = df[c.name] * df.dv
        #     df[f'{c.name}_dv'].replace({0: None}).plot(
        #         color='black' if i == 0 else colors[i],
        #         linewidth=widths[i], label=self.component.characteristics[i].name, ax=ax)
        plt.xlabel('Time')
        plt.ylabel(self.dv_name)
        if ylimits:
            plt.ylim(ylimits)
        plt.title(f'Component: {self.component.name}')
        plt.legend()
        plt.show()

def evaluate_component(df: pd.DataFrame, component: Component) -> Result:
    '''Evaluates a single component on a single timeseries.

    Args:
        df (pd.DataFrame): assumes a dataframe in the form:
            | idx  | flows | dowy |
            |------|-------|------|
            | ...  | ...   | ...  |
        component (Component): a component to evaluate.

    Returns:
        pd.DataFrame: in the form:
            | idx  | flows | dowy | char_1 | char_2 | ... | component_name |
            |------|-------|------|--------|--------|-----|----------------|
            | ...  | ...   | ...  | 0/1    | 0/1    | ... | 0/1            |
    '''
    # This function expects one flow column + trailing dowy column.
    validate_timeseries(df)
    # length of timeseries, one row per characteristics
    # float dtype (not int) preserves NaN emitted by frequency's sliding-window
    # diagnostic (insufficient trailing history) instead of silently casting it.
    output = np.zeros((len(df), len(component.characteristics)), dtype=float)
    for i, characteristic in enumerate(component.characteristics):
        output[:, i] = characteristic.fx(df, output)
    # evaluate component
    success_value = 1 if component.is_success_pattern else 0
    # Nested frequency's terminal (interannual) column is already the fully
    # broadcast per-water-year verdict (see nested_frequency_interannual_fx);
    # it replaces the generic AND-of-all-columns rule used everywhere else,
    # since frequency here operates at the water-year grain, not the
    # per-timestep grain of magnitude/duration (see
    # notes/frequencyEnhancement-resolved.md, "Nested: final component").
    if component.characteristics and component.characteristics[-1].is_nested:
        success = (output[:, -1] == success_value).astype(int).reshape(-1, 1)
    else:
        # (output==success_value).all(axis=1) converts to booleans, row-wise if true operation
        # .reshape(-1, 1) makes it column vector and concatenation as final column
        success = (output==success_value).all(axis=1).astype(int).reshape(-1, 1)
    results = np.concatenate((output, success), axis=1)
    # add 2D array to dataframe
    cols = [j.name for j in component.characteristics] + [component.name]
    df = pd.concat([df.reset_index(), pd.DataFrame(results, columns=cols)], axis=1
                   ).set_index('time')
    return Result(df, component)

def evaluate_components(df: pd.DataFrame, components: list[Component]) -> list[Result]:
    ''''Evaluates a list of components on a single timeseries.'''
    return [evaluate_component(df, component) for component in components]

#     Parameters
#     ----------
#     timeseries (pd.DataFrame): Timeseries data. Created by Timeseries class using:
#     components (list[Component]): List of components to evaluate.
#
#     Returns
#     -------
#     list[pd.DataFrame]
#         Input timeseries data appended with characteristic and component evaluation columns.
#         Each column of hydrologic data in the input timeseries is output as a separate dataframe.
#     '''
#     dfs = []
#     validate_timeseries(timeseries)
#     # all the columns except dowy column
#     for col in range(len(timeseries.columns)-1):
#         # single timeseries of hydrologic data and dowy
#         df = timeseries.iloc[:, [col, -1]]
#         comp_outcomes = np.zeros((len(df), len(components) + 1), dtype=int)
#         for c, component in enumerate(components):
#             rows, cols = len(df), len(component.characteristics) + 1
#             char_outcomes = np.zeros((rows, cols), dtype=int)
#             for i, characteristic in enumerate(component.characteristics):
#                 char_outcomes[:, i] = characteristic.fx(df, char_outcomes)
#             # evaluate component
#             for row in range(char_outcomes.shape[0]):
#                 char_outcomes[row, -1] = 1 if np.all(char_outcomes[row,:-1]==1) else 0
#             # invert outcomes if not a success pattern
#             if not component.is_success_pattern:
#                 char_outcomes[:, -1] = np.where(char_outcomes[:, -1]==1, 0, 1)
#             # somethingtodo if is not success pattern invert outcomes
#             comp_outcomes[:, c] = char_outcomes[:, -1]
#             # add outcomes to df
#             if c == 0:
#                 df_out = df.copy()
#             df_out[[j.name for j in component.characteristics] + [component.name]] = char_outcomes
#         # evaluate patterns
#         for row in range(comp_outcomes.shape[0]):
#             comp_outcomes[row, -1] = 1 if np.all(comp_outcomes[row,:-1]==1) else 0
#         df_out['all_components'] = comp_outcomes[:, -1]
#         dfs.append(df_out)
#     return dfs

def validate_timeseries(timeseries: pd.DataFrame) -> None:
    '''Validates the timeseries data.'''
    # Keep validation close to evaluate_component; callers rely on this contract.
    df = timeseries.apply(pd.to_numeric, errors='coerce')
    if df.isnull().values.any():
        raise ValueError('''Timeseries must contain only
                         numeric non-null values.''')
    if len(df.columns) < 2:
        raise ValueError('''Timeseries must contain at a minimum one hydrologic data column
                         and one day of water year column.''')
    if not is_dowy_timeseries(np.asarray(timeseries.iloc[:, -1].values)):
        raise ValueError('''Timeseries must contain
                         day of water year column in last position.''')
#endregion
