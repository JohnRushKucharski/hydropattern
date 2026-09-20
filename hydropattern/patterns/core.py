'''
Component/Result orchestration and shared characteristic-evaluation utilities.

Characteristic-fx factories (timing, magnitude, duration, rate of change,
frequency) live in hydropattern.patterns.characteristics; water-year utilities
live in hydropattern.patterns.water_year (see issue #32, continuing the
patterns.py decomposition started in #30).
'''
from collections import namedtuple
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

#region event/window helpers
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
