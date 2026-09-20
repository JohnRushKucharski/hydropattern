'''
Characteristic-fx factories: creates the evaluation functions used by
Component.characteristics (timing, magnitude, duration, rate of change,
frequency) plus the shared comparison-fx helpers used to build their
comparison predicates.

Relocated from hydropattern.patterns.core as part of issue #32 (continuing
the patterns.py decomposition started in #30).
'''
from typing import Callable

import numpy as np
import pandas as pd

from hydropattern.patterns.core import (
    CharacteristicFx,
    CharacteristicType,
    eval_order_1_characteristic,
    eval_order_n_characteristic,
    is_dowy_timeseries,
    mark_events,
    moving_average,
    sliding_window_count,
    validate_order,
)
from hydropattern.patterns.water_year import (
    identify_full_water_years,
    or_reduce_per_water_year,
    water_year_probability_ratio,
    windowed_count_per_water_year,
)


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
#endregion
