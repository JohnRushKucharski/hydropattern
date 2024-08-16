'''
Creates evaluation functions for natural flow regime characteristics.

The following characteristics are evaluated:
    - magnitude
    - duration
    - timing
    - rate of change
    - frequency
'''
from enum import Enum
from numbers import Real
from functools import partial
from typing import Callable
from dataclasses import dataclass
from collections import namedtuple

import numpy as np
import pandas as pd

#region comparision functions
def lt(a: Real, b: Real) -> bool:
    '''Returns True if a is less than b.'''
    return a < b
def le(a: Real, b: Real) -> bool:
    '''Returns True if a is less than or equal to b.'''
    return a <= b
def gt(a: Real, b: Real) -> bool:
    '''Returns True if a is greater than b.'''
    return a > b
def ge(a: Real, b: Real) -> bool:
    '''Returns True if a is greater than or equal to b.'''
    return a >= b
def eq(a: Real, b: Real) -> bool:
    '''Returns True if a is equal to b.'''
    return a == b
def ne(a: Real, b: Real) -> bool:
    '''Returns True if a is not equal to b.'''
    return a != b

def comparison_fx(symbol1: str, bound1: Real,
                  symbol2: str|None = None, bound2: Real|None = None) -> Callable[[Real], bool]:
    '''Returns the corresponding operator function for the given symbol.'''
    def closure(s: str, bound: Real, is_bound_b: bool = True) -> Callable[[Real], bool]:
        symbols = {
            '<': lt,    # a < b
            '<=': le,   # a <= b
            '>': gt,    # a > b
            '>=': ge,   # a >= b
            '=': eq,   # a == b
            '!=': ne    # a != b
        }
        if is_bound_b:
            return partial(symbols[s], b=bound)
        return partial(symbols[s], a=bound)
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
        def fx3(value: Real) -> bool:
            return fx1(b=value) and fx2(a=value)
        return fx3
    # Every bound must have a symbol.
    raise ValueError('symbol2 must be provided if bound2 is provided.')
#endregion

#region characteristics
CharacteristicType = Enum('CharacteristicType',
                          ['TIMING', 'MAGNITUDE', 'DURATION', 'RATE_OF_CHANGE', 'FREQUENCY'])

type Characteristic_fx = Callable[[pd.DataFrame, int, None|np.ndarray], np.ndarray]

Characteristic = namedtuple('Characteristic', ['name', 'fx', 'type'])

#region utility functions
def is_order_1(order: int, output: None|np.ndarray) -> bool:
    '''Validates order and output for characteristics.
    
    Returns
    -------
        bool: True if order is 1, False otherwise.
    Raises
    -------
        ValueError: For invalid order and output combinations.
    '''
    if order < 1:
        raise ValueError('Order must be greater than or equal to 1.')
    if order > 1:
        if output is None:
            raise ValueError('Output must be provided for order greater than 1.')
        if order > len(output):
            raise ValueError('Order must be less than or equal to the length of the output.')
        return False
    # order == 1
    return True

def is_dowy_timeseries(data: np.ndarray) -> bool:
    '''Checks if every value is integer in range [1, 365].'''
    return all(0 < i < 366 for i in data) and all(i.is_integer() for i in data)

def moving_average(data: np.ndarray,
                   period: int, min_periods: None|int = None) -> np.ndarray:
    '''Calculates moving average over timeseries data.'''
    if min_periods:
        if period < min_periods or min_periods < 1:
            raise ValueError(f'''min_periods: {min_periods} must be greater than or equal to 1
                             and less than or equal to the moving average period: {period}.''')
    if period < 1:
        raise ValueError(f'The moving average period: {period} must be greater than or equal to 1.')
    # adjust values to account for 0-based index
    periods = period - 1
    min_periods = min_periods - 1 if min_periods else periods
    # convolve does  this faster but is less clear and harder to debug
    ma = np.zeros(len(data))
    for t in range(len(data)):
        if t < min_periods:
            ma[t] = np.nan
        else:
            if t < periods:
                ma[t] = np.mean(data[:t+1])
            else:
                ma[t] = np.mean(data[t-periods:t+1])
            # average over min_periods or period depending on t
            # t+1 because max of range is exclusive
            #ma[t] = np.mean(data[:t+1]) if t < period else np.mean(data[t-period-1:t+1])
    return ma
    # ma = np.convolve(data, np.ones(period), 'valid') / period
    # return np.pad(ma, (len(data)-len(ma), 0), 'constant', constant_values=np.nan)
#endregion

#region timing
def timing_fx(f: Callable[[Real], bool],
              order: int = 1) -> Characteristic_fx:
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
        data = df.iloc[:, -1].values
        if not is_dowy_timeseries(data):
            raise ValueError('''Timing characteristics must be evaluated on a
                             day of water year timeseries.''')
        if is_order_1(order, output):
            @np.vectorize
            def fx(value: Real) -> int:
                return 1 if f(value) else 0
            return fx(data)
        else: # is valid order > 1
            result = np.zeros(len(data))
            for t, row in enumerate(output):
                # 1st order-1 values are 1
                if np.all(row[-order+1:]==1):
                    result[t] = 1 if f(data[t]) else 0
            return result
    return closure
#endregion

#region magnitude
def magnitude_fx(f: Callable[[Real], bool],
                 order: int = 1, ma_periods: int = 1) -> Characteristic_fx:
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
        data = df.iloc[:, 0].values
        data = data if ma_periods == 1 else moving_average(data, ma_periods)

        n = len(data)
        result = np.zeros(n)
        # restrict t to moving average
        for t in range(ma_periods-1, n):
            if is_order_1(order, output):
                result[t] = 1 if f(data[t]) else 0
            else: # is valid order > 1
                # 1st order-1 values are 1
                if np.all(output[t][-order+1:]==1):
                    result[t] = 1 if f(data[t]) else 0
        return result
    return closure
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
def duration_fx(f: Callable[[Real], bool],
                order: int) -> Characteristic_fx:
    '''
    Creates function to evaluate duration characteristics.

    Parameters
    ----------
        f (Callable[[Real], bool]): Comparision function.
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
        if is_order_1(order, output):
            raise ValueError('Order must be greater than 1 for duration characteristics.')
        n, T = 0, len(df) # pylint: disable=invalid-name
        result = np.zeros(T)
        for t, row in enumerate(output):
            # 1st order-1 values are 1
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
def frequency_fx(f: Callable[[Real], bool],
                 order: int, ma_period: int) -> Characteristic_fx:
    '''
    Creates function to evaluate frequency characteristics.

    Parameters
    ----------
        f (Callable[[Real], bool]): Comparision function.
        order (int): Position in which characteristic is evaluated
            within list of component characteristics.
        ma_period (int): window (in years) over which f is evaluated. 
    Returns
    -------
        Characteristic_fx: evaluates characteristic over timeseries.
    '''
    def closure(df: pd.DataFrame,
                output: None|np.ndarray) -> np.ndarray:
        # uses dowy (last) df column
        data = df.iloc[:, -1].values
        if is_order_1(order, output):
            raise ValueError('Order must be greater than 1 for frequency characteristics.')
        if not is_dowy_timeseries(data):
            raise ValueError('''Frequency characteristics must be evaluated on a
                             day of water year timeseries.''')
        nyr = []
        is_true = False
        n, T = 0, len(data) # pylint: disable=invalid-name
        for t in range(T):
            # last day of water year
            if data[t] == 365:
                # first yr is full yr
                if not nyr and data[0] == 1:
                    nyr.append(n)
                # not first year
                if nyr:
                    nyr.append(n)
                # # excludes first year if it
                # # starts on a day other than 1
                # if t > 0:
                #     # number of times condition
                #     # was met in previous water year
                #     print(f'COUNTING: {n} at {t}')
                #     in_yr_count.append(n)
                # n = 0
            if np.all(output[t][-order+1:]==1):
                # count up if mets condition
                # did not meet condition in t-1
                if not is_true:
                    n += 1
                    is_true = True
            else:
                # in t-1 met condition but
                # in t does not meet condition
                if is_true:
                    is_true = False

        result = np.zeros(T)
        yr = 0 if data[0] == 1 else -1
        mayr = moving_average(np.array(nyr), ma_period, 0)
        # 2nd loop to fill in result values
        # probably a faster way to do this but this is clear.
        for t in range(T):
            # starts on less than full yr.
            if yr == -1:
                # this could cause problems
                result[t] = np.nan
            # starts on full yr
            # or past first year
            else:
                result[t] = f(mayr[yr]) and np.all(output[t][-order+1:]==1)
            if data[t] == 365:
                yr += 1
            # # won't start until
            # # 1st full water year
            # if data[t] == 1:
            #     yr += 1
            #     is_true = f(btw_yr_count[yr])
            # # criteria met for water year
            # if is_true and np.all(output[t][-order+1:]==1):
            #     result[t] = 1
        return result
    return closure
#endregion

#region rate_of_change
def rate_of_change_fx(f: Callable[[Real], bool],
                      order: int = 1, ma_periods: int = 1,
                      look_back: int = 1, minimum: float = 0.0) -> Characteristic_fx:
    '''
    Creates function to evaluate rate of change characteristics.

    Parameters
    ----------
        f (Callable[[Real], bool]): Comparision function.
        order (int): Position in which characteristic is evaluated
            within list of component characteristics.
    Returns
    -------
        Characteristic_fx: evaluates characteristic over timeseries.
    '''
    def closure(df: pd.DataFrame,
                output: None|np.ndarray) -> np.ndarray:
        # uses hydrologic data (1st) df column
        data = df.iloc[:, 0].values
        data = data if ma_periods == 1 else moving_average(data, ma_periods)

        n = len(data)
        result = np.zeros(n)
        # restrict t to moving average
        for t in range(ma_periods-1, n):
            if is_order_1(order, output):
                if t-look_back >= 0:
                    if data[t-look_back] > minimum:
                        result[t] = 1 if f(data[t] / data[t-look_back]) else 0
            else: # is valid order > 1
                # 1st order-1 values are 1
                if np.all(output[t][-order+1:]==1):
                    if t-look_back >= 0:
                        if data[t-look_back] > minimum:
                            result[t] = 1 if f(data[t] / data[t-look_back]) else 0
        return result
    return closure
#endregion
#endregion

#region components
@dataclass
class Component:
    '''Natural flow regime type component.'''
    name: str
    characteristics: list[Characteristic]
    is_success_pattern: bool

def evaluate_patterns(timeseries: pd.DataFrame, components: list[Component]) -> pd.DataFrame:
    '''
    Evaluate the components.

    Parameters
    ----------
    timeseries (pd.DataFrame): Timeseries data. Created by Timeseries class using:
    components (list[Component]): List of components to evaluate.
        
    Returns
    -------
    list[pd.DataFrame]
        Input timeseries data appended with characteristic and component evaluation columns.
        Each column of hydrologic data in the input timeseries is output as a separate dataframe.  
    '''
    dfs = []
    validate_timeseries(timeseries)
    # all the columns except dowy column
    for col in range(len(timeseries.columns)-1):
        # single timeseries of hydrologic data and dowy
        df = timeseries.iloc[:, [col, -1]]
        comp_outcomes = np.zeros((len(df), len(components) + 1), dtype=int)
        for c, component in enumerate(components):
            rows, cols = len(df), len(component.characteristics) + 1
            char_outcomes = np.zeros((rows, cols), dtype=int)
            for i, characteristic in enumerate(component.characteristics):
                char_outcomes[:, i] = characteristic.fx(df, char_outcomes)
            # evaluate component
            for row in range(char_outcomes.shape[0]):
                char_outcomes[row, -1] = 1 if np.all(char_outcomes[row,:-1]==1) else 0
            # invert outcomes if not a success pattern
            if not component.is_success_pattern:
                char_outcomes[:, -1] = np.where(char_outcomes[:, -1]==1, 0, 1)
            # todo: if is not success pattern invert outcomes
            comp_outcomes[:, c] = char_outcomes[:, -1]
            # add outcomes to df
            if c == 0:
                df_out = df.copy()
            df_out[[j.name for j in component.characteristics] + [component.name]] = char_outcomes
        # evaluate patterns
        for row in range(comp_outcomes.shape[0]):
            comp_outcomes[row, -1] = 1 if np.all(comp_outcomes[row,:-1]==1) else 0
        df_out['all_components'] = comp_outcomes[:, -1]
        dfs.append(df_out)
    return dfs

def validate_timeseries(timeseries: pd.DataFrame) -> None:
    '''Validates the timeseries data.'''
    # todo: this seems misplaced, but is near evaluate patterns where it is used.
    df = timeseries.apply(pd.to_numeric, errors='coerce')
    if df.isnull().values.any():
        raise ValueError('''Timeseries must contain only
                         numeric non-null values.''')
    if len(df.columns) < 2:
        raise ValueError('''Timeseries must contain at a minimum one hydrologic data column
                         and one day of water year column.''')
    if not is_dowy_timeseries(timeseries.iloc[:, -1].values):
        raise ValueError('''Timeseries must contain
                         day of water year column in last position.''')
#endregion
