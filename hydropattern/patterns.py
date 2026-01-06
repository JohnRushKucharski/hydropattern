'''
Creates evaluation functions for natural flow regime characteristics.

The following characteristics are evaluated:
    - magnitude
    - duration
    - timing
    - rate of change
    - frequency
'''
from enum import StrEnum
from typing import Callable
from dataclasses import dataclass
from collections import namedtuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        else:
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

Characteristic = namedtuple('Characteristic', ['name', 'fx', 'type'])

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

def eval_order_1_characteristic(f: Callable[[float], bool], data: np.ndarray) -> np.ndarray:
    '''Evaluates eligble order 1 characteristic, returning array of [0, 1] values.'''
    @np.vectorize
    def fx(value: float) -> int:
        return 1 if f(value) else 0
    return fx(data)

def eval_order_n_characteristic(f: Callable[[float], bool], data: np.ndarray,
                                output: np.ndarray, order: int) -> np.ndarray:
    '''Evaluates eligble order n characteristic, returning array of [0, 1] values.'''
    precedents = output[:, :order-1]
    eligible = (precedents == 1).all(axis=1)
    @np.vectorize
    def eligible_fx(data: float, is_eligible: bool) -> int:
        return 1 if is_eligible and f(data) else 0
    return eligible_fx(data, eligible)
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
        # # todo: test this restriction
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
def frequency_fx(f: Callable[[float], bool],
                 order: int, ma_period: int) -> CharacteristicFx:
    '''
    Creates function to evaluate frequency characteristics.

    Parameters
    ----------
        f (Callable[[float], bool]): Comparision function.
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
        data = np.asarray(df.iloc[:, -1].values)

        validate_order(order, output, CharacteristicType.FREQUENCY)
        assert output is not None # for mypy: checked by validate_order
        if not is_dowy_timeseries(data):
            raise ValueError('''Frequency characteristics must be evaluated on a
                             day of water year timeseries.''')
        nyr: list[int] = []
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
def rate_of_change_fx(f: Callable[[float], bool],
                      order: int = 1, ma_periods: int = 1,
                      look_back: int = 1, minimum: float = 0.0) -> CharacteristicFx:
    '''
    Creates function to evaluate rate of change characteristics.

    Parameters
    ----------
        f (Callable[[float], bool]): Comparision function.
        order (int): Position in which characteristic is evaluated
            within list of component characteristics.
    Returns
    -------
        Characteristic_fx: evaluates characteristic over timeseries.
    '''
    def closure(df: pd.DataFrame,
                output: None|np.ndarray) -> np.ndarray:
        # uses hydrologic data (1st) df column
        data = np.asarray(df.iloc[:, 0].values)
        data = data if ma_periods == 1 else moving_average(data, ma_periods)
        validate_order(order, output, CharacteristicType.RATE_OF_CHANGE)
        assert output is not None or order == 1 # for mypy: checked by validate_order

        n = len(data)
        result = np.zeros(n)
        # restrict t to moving average
        for t in range(ma_periods-1, n):
            if order == 1:
                if t-look_back >= 0:
                    if data[t-look_back] > minimum:
                        result[t] = 1 if f(data[t] / data[t-look_back]) else 0
            else: # is valid order > 1
                if output is None:
                    raise ValueError("Output cannot be None for order > 1")
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

@dataclass
class Result:
    '''Result of evaluating a component on a timeseries.'''
    df: pd.DataFrame
    component: Component

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
        # todo: add option to plot only eligible periods (based on 1st or n-th characteristic)
        # df if full_timeseries else self.df[self.df[self.component.characteristics[0].name] == 1
        df = self.df
        _, ax = plt.subplots(figsize=(15, 5))
        df['success'] = self.df[self.component.name] * self.df.dv
        df['possible'] = self.df[self.component.characteristics[0].name] * self.df.dv
        df.possible.replace({0: None}).plot(
            color='yellow', linewidth=10, label=self.component.characteristics[0].name, ax=ax)
        df.dv.plot(
            color='grey', linewidth=0.5, label=self.dv_name, ax=ax)
        df.success.replace({0: None}).plot(
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
    # todo: split ts with multiple columns into separate evaluations.
    # todo: check flow column name is useful pandas column name.
    validate_timeseries(df)
    # length of timeseries, one row per characteristics
    output = np.zeros((len(df), len(component.characteristics)), dtype=int)
    for i, characteristic in enumerate(component.characteristics):
        output[:, i] = characteristic.fx(df, output)
    # evaluate component
    success_value = 1 if component.is_success_pattern else 0
    # (output==1).all(axis=1) converts to booleans, row-wise if true operation
    # .reshape(-1, 1) makes it column vector and concatenation as final column
    success = (output==success_value).all(axis=1).astype(int).reshape(-1, 1)
    results = np.concatenate((output, success), axis=1)
    # add 2D array to dataframe
    cols = [j.name for j in component.characteristics] + [component.name]
    df = pd.concat([df.reset_index(), pd.DataFrame(results, columns=cols)], axis=1
                   ).set_index('time')
    return Result(df, component)

# def evaluate_patterns(timeseries: pd.DataFrame, components: list[Component]) -> pd.DataFrame:
#     '''
#     Evaluate the components.

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
    # todo: this seems misplaced, but is near evaluate patterns where it is used.
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
