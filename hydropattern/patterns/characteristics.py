'''
Characteristic-fx factories: creates evaluation functions used by
Component.characteristics (timing, magnitude, duration, rate of change,
frequency) and shared comparison-fx helpers used to build comparison predicates.

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
    find_runs,
    is_dowy_timeseries,
    moving_average,
    validate_order,
)
from hydropattern.patterns.water_year import (
    identify_full_water_years,
    or_reduce_per_water_year,
    water_year_probability_ratio,
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
        dowy = np.asarray(df.iloc[:, -1].values)
        if not is_dowy_timeseries(dowy):
            raise ValueError('''Timing characteristics must be evaluated on a
                             day of water year timeseries.''')
        validate_order(order, output, CharacteristicType.TIMING)
        # Timing is an independent diagnostic (see docs/plans/2026-10-01-
        # pattern-correctness-tdd.md): it reports its own truth value
        # regardless of position/preceding characteristics, never gated by
        # `output`'s earlier columns.
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError('Timing characteristics require a datetime index.')
        calendar_doy = df.index.dayofyear.to_numpy()
        leap_day = df.index.is_leap_year & (calendar_doy > 59)
        calendar_doy = calendar_doy - leap_day.astype(int)
        return eval_order_1_characteristic(f, calendar_doy)
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
        # Magnitude is an independent diagnostic: own truth value regardless
        # of preceding characteristics (see note in timing_fx above).
        return eval_order_1_characteristic(f, data)
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

        # from 0th to [order - 1]; eligible wherever preceding
        # characteristics are all 1 (find_runs treats anything else, incl.
        # NaN, as a run-breaker -- same as the prior hand-rolled loop, which
        # only ever compared against 1).
        precedents = (output[:, :order - 1] == 1).all(axis=1).astype(float)
        result = np.zeros(len(df))
        for start, end in find_runs(precedents):
            if f(end - start + 1):
                result[start:end + 1] = 1
        return result
    return closure
#endregion

#region frequency
def _forward_frequency_window(eligible: np.ndarray, f: Callable[[float], bool],
                             big_n: int, exclusive_event_window: bool) -> np.ndarray:
    result = np.zeros(len(eligible))
    zero_admitting = f(0)
    span_end = -1
    for t in range(len(eligible)):
        if not zero_admitting and eligible[t] != 1:
            continue
        if exclusive_event_window and t <= span_end:
            continue
        end = min(t + big_n - 1, len(eligible) - 1)
        if f(int(eligible[t:end + 1].sum())):
            result[t:end + 1] = 1
            if exclusive_event_window:
                span_end = end
    return result


def frequency_fx(f: Callable[[float], bool], order: int,
                 big_n: int | None = None,
                 exclusive_event_window: bool = False) -> CharacteristicFx:
    '''
    Creates function to evaluate an un-nested frequency characteristic.

    Parameters
    ----------
        f (Callable[[float], bool]): Comparison function, applied to
            probability (successes/trials ratio) or a trial count, depending on form.
        order (int): Position in which characteristic is evaluated within component.
            Must last component charactersistic (enforced upstream in builders.py).
        big_n (int | None): forward-looking trial-window size (in timesteps) for
            [op, n, N] and [min_n, max_n, N] forms. None for the [op, probability]
            form, which is only implemented as a nested base pattern.
        exclusive_event_window (bool): False by default. Defines if overlapping windows are allowed.
            Each windows anchored at time, t has a fixed N-length spans.
            If true, each forward-looking window is exclusive, i.e., all timesteps within that span
            belong to only the fixed length window. If false, timesteps can belong to multiple 
            overlapping windows.
    Returns
    -------
        Characteristic_fx: evaluates characteristic over timeseries.

    Note
    ----
        Windows are anchored at timesteps where preceding characteristics in the component
        are eligible (met), when the trial count, n in `f(n)` is not zero. If exclusive_event_window
        is true, only the first eligible timestep in a N-length span anchors the window.
        When the trial count is zero, i.e., `f(n=0)` every timestep is eligible, so every timestep
        anchors a window. Since windows are forward-looking no warm-up/NaN period is needed.
        Windows at the end of the record count the observations available within the truncated window.
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
        return _forward_frequency_window(eligible, f, big_n, exclusive_event_window)
    return closure

def _intra_annual_diagnostic(
    eligible: np.ndarray,
    dowy: np.ndarray,
    timestamps: pd.DatetimeIndex | None,
    f: Callable[[float], bool],
    big_n: int | None,
    exclusive_event_window: bool,
) -> np.ndarray:
    '''Shared raw-diagnostic computation for the nested base (intra-annual)
    pattern. Probability verdicts are compared once per year and broadcast;
    count/between forms use forward candidate windows within each water year.
    '''
    diag = np.full(len(eligible), np.nan)
    full_years = identify_full_water_years(dowy, timestamps)
    if big_n is None:
        ratios = water_year_probability_ratio(eligible, dowy, timestamps=timestamps)
        for start, end in full_years:
            if not np.isnan(ratios[end]):
                diag[start:end + 1] = 1 if f(ratios[end]) else 0
        return diag

    for start, end in full_years:
        diag[start:end + 1] = _forward_frequency_window(
            eligible[start:end + 1], f, big_n, exclusive_event_window
        )
    return diag

def nested_frequency_intra_annual_fx(f: Callable[[float], bool], order: int,
                                     big_n: int | None = None,
                                     exclusive_event_window: bool = False) -> CharacteristicFx:
    '''
    Creates function to evaluate the intra-annual (base) column of a nested
    frequency characteristic.

    Probability base verdicts use eligible source timesteps and are broadcast
    across each water year. Count/between base forms evaluate forward windows
    within each year; their terminal diagnostic includes its source conditions.
    `exclusive_event_window` applies to those windows, not annual probability.

    Parameters
    ----------
        f (Callable[[float], bool]): Comparison function for the base pattern.
        order (int): Position in the component's characteristic sequence.
        big_n (int | None): forward trial-window size for count/between base
            forms; None for the probability base form.
        exclusive_event_window (bool): base window suppression mode; ignored
            for the single annual probability trial.
    Returns
    -------
        Characteristic_fx: evaluates the intra-annual diagnostic column.
    '''
    def closure(df: pd.DataFrame,
                output: None|np.ndarray) -> np.ndarray:
        validate_order(order, output, CharacteristicType.FREQUENCY)
        assert output is not None # for mypy: checked by validate_order

        dowy = np.asarray(df.iloc[:, -1].values, dtype=float)
        timestamps = df.index if isinstance(df.index, pd.DatetimeIndex) else None
        precedents = output[:, :order-1]
        eligible = (precedents == 1).all(axis=1).astype(int)

        return _intra_annual_diagnostic(
            eligible, dowy, timestamps, f, big_n, exclusive_event_window
        )
    return closure

def nested_frequency_interannual_fx(f: Callable[[float], bool], order: int,
                                    big_n: int | None = None,
                                    exclusive_event_window: bool = False) -> CharacteristicFx:
    '''
    Creates function to evaluate the interannual (nested) column of a nested
    frequency characteristic -- the terminal column whose result determines
    the component's final pass/fail, broadcast across each qualifying water
    year (see notes/frequencyEnhancement-resolved.md).

    The base probability form has no event window; count/between base forms
    use their own window flag. This outer flag controls forward windows over
    annual verdicts and therefore changes the broadcast pass/fail result.

    Parameters
    ----------
        f (Callable[[float], bool]): Comparison function for the nested pattern.
        order (int): Position in the component's characteristic sequence.
            The intra-annual column (this pattern's input) must immediately
            precede this characteristic, at column index `order - 2`.
        big_n (int | None): forward trial-window size (in years) for count/
            between nested forms. Probability form is not valid at this
            level (enforced upstream in validate_nested_frequency_metrics).
        exclusive_event_window (bool): nested pattern's own exclusive_event_window; changes the
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
        timestamps = df.index if isinstance(df.index, pd.DatetimeIndex) else None
        intra_annual = output[:, order - 2]
        year_verdicts = or_reduce_per_water_year(intra_annual, dowy, timestamps)

        full_years = identify_full_water_years(dowy, timestamps)
        compact_verdicts = np.array([year_verdicts[end] for _, end in full_years])
        compact_diag = _forward_frequency_window(
            np.nan_to_num(compact_verdicts, nan=0).astype(int),
            f, big_n, exclusive_event_window
        )

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
        # The minimum constrains only the lagged denominator. Mask it before
        # division so invalid denominators do not emit divide-by-zero warnings.
        denominators = data[:-look_back]
        numerators = data[look_back:]
        rates = np.full(data.shape, np.nan)
        rates[look_back:] = np.divide(
            numerators,
            denominators,
            out=np.full(denominators.shape, np.nan),
            where=denominators > minimum,
        )
        # Rate-of-change is an independent diagnostic: own truth value
        # regardless of preceding characteristics (see note in timing_fx).
        return eval_order_1_characteristic(f, rates)
    return closure
#endregion
