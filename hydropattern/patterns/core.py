'''
Component/Result orchestration and shared characteristic-evaluation utilities.

Characteristic-fx factories (timing, magnitude, duration, rate of change,
frequency) live in hydropattern.patterns.characteristics; water-year utilities
live in hydropattern.patterns.water_year (see issue #32, continuing the
patterns.py decomposition started in #30).
'''
from collections import namedtuple
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, NamedTuple

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

# is_terminal identifies a characteristic that completes its component's
# evaluation. Nested frequency's interannual pattern is currently the only
# characteristic that sets this marker.
Characteristic = namedtuple('Characteristic', ['name', 'fx', 'type', 'is_terminal'],
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
    '''Evaluate an independent characteristic, preserving unavailable values as NaN.'''
    return np.array([
        np.nan if np.isnan(value) else int(f(value))
        for value in data
    ], dtype=float)

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
def mark_windows(raw: np.ndarray, exclusive_windows: bool = True) -> np.ndarray:
    '''
    Optionally collapses each maximal run of consecutive successes in a raw
    0/1/NaN diagnostic array to one marker at the run's last timestep.

    Parameters
    ----------
        raw (np.ndarray): a 0/1/NaN diagnostic array. NaN marks an unknown
            outcome and breaks a run.
        exclusive_windows (bool): if True (default), each maximal run of
            consecutive 1s collapses to a single 1 at its last timestep.
            Earlier 1s in that run become 0. If False, all values are
            returned unchanged.

    Returns
    -------
        np.ndarray: same shape as `raw`. NaNs and 0s are unchanged; only
        earlier 1s in a run may become 0 when `exclusive_windows=True`.
    '''
    result = np.array(raw, dtype=float)
    if not exclusive_windows:
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

class EventCountBounds(NamedTuple):
    '''Inclusive lower and upper bounds for an observed component event count.'''
    lower: int
    upper: int


class EventRateBounds(NamedTuple):
    '''Inclusive lower and upper bounds for an observed component event rate.'''
    lower: float
    upper: float


def _validate_outcomes(success: np.ndarray) -> np.ndarray:
    raw_outcomes = np.asarray(success)
    if raw_outcomes.ndim != 1 or raw_outcomes.dtype.kind not in 'biuf':
        raise ValueError('outcomes must contain only 0, 1, or NaN.')
    outcomes = raw_outcomes.astype(float, copy=False)
    if np.any(
        ~np.isnan(outcomes) & (outcomes != 0) & (outcomes != 1)
    ):
        raise ValueError('outcomes must contain only 0, 1, or NaN.')
    return outcomes


def _event_count_bounds(
    success: np.ndarray, included: np.ndarray | None = None
) -> EventCountBounds:
    outcomes = _validate_outcomes(success)
    if included is None:
        included = np.ones(len(outcomes), dtype=bool)
    if len(included) != len(outcomes):
        raise ValueError('included must have the same length as outcomes.')

    bounds_by_previous = {0: (0, 0)}
    for index, outcome in enumerate(outcomes):
        choices = (0, 1) if np.isnan(outcome) else (int(outcome),)
        next_bounds: dict[int, tuple[int, int]] = {}
        for previous, (minimum, maximum) in bounds_by_previous.items():
            for current in choices:
                starts_event = int(bool(included[index]) and current == 1 and previous == 0)
                candidate = (minimum + starts_event, maximum + starts_event)
                existing = next_bounds.get(current)
                if existing is None:
                    next_bounds[current] = candidate
                else:
                    next_bounds[current] = (
                        min(existing[0], candidate[0]),
                        max(existing[1], candidate[1]),
                    )
        bounds_by_previous = next_bounds

    return EventCountBounds(
        min(bounds[0] for bounds in bounds_by_previous.values()),
        max(bounds[1] for bounds in bounds_by_previous.values()),
    )


def count_event_bounds(success: np.ndarray) -> EventCountBounds:
    '''Return conservative event-count bounds for final 0/1/unknown outcomes.

    Unknown timesteps are considered independently; bounds may therefore be
    wider than counts permitted by dependencies in the source characteristics.
    '''
    return _event_count_bounds(success)


def count_events(success: np.ndarray) -> int:
    '''
    Counts distinct component events when final outcomes determine one count.

    Works uniformly for any component's success column, regardless of
    whether it is a raw per-timestep grain (magnitude, duration, un-nested
    frequency) or a nested-frequency terminal column broadcast across full
    water years: nested_frequency_interannual_fx assigns one scalar verdict
    per full water year (result[start:end+1] = verdict), so a maximal run's
    boundaries at timestep grain always coincide exactly with water-year
    boundaries -- no separate per-water-year dedup step is needed before
    collapsing (verified in tests/test_event_count.py).

    Parameters
    ----------
        success (np.ndarray): 0/1(/NaN) success array (e.g. a component's
            final success column).

    Returns
    -------
    int: number of distinct component events.

    Raises
    ------
    ValueError: If unknown outcomes permit more than one count.
    '''
    bounds = count_event_bounds(success)
    if bounds.lower != bounds.upper:
        raise ValueError(
            'Event count is ambiguous; use count_event_bounds() to inspect bounds.'
        )
    return bounds.lower

def find_runs(eligible: np.ndarray) -> list[tuple[int, int]]:
    '''
    Finds every maximal run of consecutive successes in a 0/1(/NaN) array.

    Generic run-detection primitive shared by duration_fx (which tests each
    run's length against a duration bound) and find_exceeding_events (which
    reports runs whose length falls outside a [min, max] bound) -- one
    run-detection implementation, not two.

    NaN breaks a run (does not count as, or extend, a run of successes),
    matching mark_windows' existing NaN semantics.

    Parameters
    ----------
        eligible (np.ndarray): 0/1(/NaN) array.

    Returns
    -------
        list[tuple[int, int]]: (start, end) inclusive index pairs, in series
        order, for each maximal run of 1s.
    '''
    runs = []
    start = None
    for t, value in enumerate(eligible):
        if not np.isnan(value) and value == 1:
            if start is None:
                start = t
        else:
            if start is not None:
                runs.append((start, t - 1))
            start = None
    if start is not None:
        runs.append((start, len(eligible) - 1))
    return runs

def find_exceeding_events(eligible: np.ndarray, min_duration: int | None = None,
                          max_duration: int | None = None) -> list[tuple[int, int, int]]:
    '''
    Finds every maximal run whose length falls outside a [min_duration,
    max_duration] bound -- e.g. the runs a duration_parser's between-form
    would exclude, whether for being too short or too long.

    Deliberately standalone: NOT wired into Result, event_count(),
    event_rate(), frequency_table(), or any CLI/report output. Exists so a
    future "what's driving success/failure" feature can identify and inspect
    disqualified runs (e.g. a low-water spell that lasted 68 months against
    a 36-60 month bound); that wiring is a separate, future change.

    Reuses find_runs() -- no new run-detection logic.

    Parameters
    ----------
        eligible (np.ndarray): 0/1(/NaN) array (e.g. a magnitude
            characteristic's success column from a Result).
        min_duration (int | None): runs shorter than this are reported. None
            (default) disables the lower-bound check.
        max_duration (int | None): runs longer than this are reported. None
            (default) disables the upper-bound check.

    Returns
    -------
        list[tuple[int, int, int]]: (start, end, length) for each
        out-of-bounds run, in series order.
    '''
    if min_duration is None and max_duration is None:
        raise ValueError('at least one of min_duration/max_duration must be given.')
    exceeding = []
    for start, end in find_runs(eligible):
        length = end - start + 1
        if (min_duration is not None and length < min_duration) or \
           (max_duration is not None and length > max_duration):
            exceeding.append((start, end, length))
    return exceeding

def event_rate(events: int, years: float) -> float:
    '''
    Plain descriptive event rate: events / years.

    This is a descriptive statistic only -- it does NOT assume a Poisson
    process, does NOT support recurrence-interval-style probability claims,
    and does NOT assume independence between events. Great Lakes water-level
    records show documented multi-decadal persistence/clustering, so
    between-event independence should not be assumed when interpreting this
    rate (see docs/developer/adr and prior review notes on this component's stats).

    Parameters
    ----------
        events (int): number of qualifying events (e.g. from count_events()).
        years (float): record length in years (e.g. from
            record_length_years()); must be > 0.

    Returns
    -------
        float: events / years.
    '''
    if years <= 0:
        raise ValueError(f'years: {years} must be greater than 0.')
    return events / years
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
    '''Result of evaluating a component on a timeseries.

    `dv_name` is the original name of the evaluated data column in `df`.
    '''
    df: pd.DataFrame
    component: Component
    dv_name: str = ''
    first_day_of_water_year: int | None = None

    def __post_init__(self):
        if not self.dv_name:
            self.dv_name = self.df.columns[0]
        if self.first_day_of_water_year is not None:
            from hydropattern.patterns.water_year import validate_water_year_boundary
            self.first_day_of_water_year = validate_water_year_boundary(
                self.first_day_of_water_year
            )

    def event_count(self) -> int:
        '''Return component event count; reject ambiguous counts.'''
        bounds = self.event_count_bounds()
        if bounds.lower != bounds.upper:
            raise ValueError(
                'Event count is ambiguous; use event_count_bounds() to inspect bounds.'
            )
        return bounds.lower

    def event_count_bounds(self) -> EventCountBounds:
        '''Return conservative whole-record event-count bounds.

        Unknown final outcomes are treated independently, so bounds may be
        wider than counts permitted by dependencies in the source evaluation.
        '''
        return count_event_bounds(self.df[self.component.name].to_numpy())

    def event_count_bounds_by_water_year(self) -> dict[int, EventCountBounds]:
        '''Return event-count bounds attributed to each event's start water year.'''
        labels = self._water_year_labels()
        outcomes = self.df[self.component.name].to_numpy()
        return {
            int(year): _event_count_bounds(outcomes, labels == year)
            for year in np.unique(labels)
        }

    def event_rate(self) -> float:
        '''Return whole-record event rate; reject ambiguous counts or exposure.'''
        bounds = self.event_rate_bounds()
        if bounds.lower != bounds.upper:
            raise ValueError(
                'Event rate is ambiguous; use event_rate_bounds() to inspect bounds.'
            )
        return bounds.lower

    def event_rate_bounds(self) -> EventRateBounds:
        '''Return whole-record event-rate bounds per observed water year.'''
        from hydropattern.patterns.water_year import water_year_exposure
        timestamps = self._timestamps()
        boundary = self._water_year_boundary(timestamps)
        exposure = water_year_exposure(timestamps, boundary)
        counts = self.event_count_bounds()
        if exposure <= 0:
            raise ValueError('Observed water-year exposure must be greater than 0.')
        return EventRateBounds(counts.lower / exposure, counts.upper / exposure)

    def event_rate_bounds_by_water_year(self) -> dict[int, EventRateBounds]:
        '''Return annual event-rate bounds using each year's observed exposure.'''
        from hydropattern.patterns.water_year import water_year_exposure_by_year
        timestamps = self._timestamps()
        boundary = self._water_year_boundary(timestamps)
        exposures = water_year_exposure_by_year(timestamps, boundary)
        counts = self.event_count_bounds_by_water_year()
        return {
            year: EventRateBounds(bounds.lower / exposures[year],
                                  bounds.upper / exposures[year])
            for year, bounds in counts.items()
        }

    def _timestamps(self) -> pd.DatetimeIndex:
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError(
                'DatetimeIndex is required to determine observed water-year exposure.'
            )
        return self.df.index

    def _water_year_boundary(self, timestamps: pd.DatetimeIndex) -> int:
        from hydropattern.patterns.water_year import infer_first_day_of_water_year
        inferred = infer_first_day_of_water_year(
            self.df['dowy'].to_numpy(), timestamps
        )
        if (
            self.first_day_of_water_year is not None
            and self.first_day_of_water_year != inferred
        ):
            raise ValueError(
                'first_day_of_water_year conflicts with timestamps and dowy.'
            )
        return self.first_day_of_water_year or inferred

    def _water_year_labels(self) -> np.ndarray:
        from hydropattern.patterns.water_year import water_year_label
        timestamps = self._timestamps()
        boundary = self._water_year_boundary(timestamps)
        return np.array([water_year_label(date, boundary) for date in timestamps])

    def identify_water_years(
        self, first_day_of_water_year: int | None = None
    ) -> pd.DataFrame:
        '''Identifies water years in the timeseries.'''
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError('DatetimeIndex is required to identify water years.')
        if (
            first_day_of_water_year is not None
            and self.first_day_of_water_year is not None
            and first_day_of_water_year != self.first_day_of_water_year
        ):
            raise ValueError(
                'first_day_of_water_year conflicts with Result boundary metadata.'
            )
        boundary = first_day_of_water_year
        if boundary is None:
            boundary = self.first_day_of_water_year
        if boundary is None:
            raise ValueError(
                'first_day_of_water_year is required to identify water years.'
            )
        from hydropattern.patterns.water_year import (
            validate_water_year_boundary,
            water_year_label,
        )
        boundary = validate_water_year_boundary(boundary)
        wy = [
            water_year_label(timestamp, boundary)
            for timestamp in self.df.index
        ]
        df = self.df.copy()
        df['water_year'] = wy
        return df
        # for i, row in self.df.iterrows():
        #     if row['dowy'] == 1:
        #         self.df.at[i, 'water_year'] = self.df.index[self.df.index.get_loc(i)-1].year
        # return df

    def frequency_table(self, by_water_years: bool = False) -> pd.DataFrame:
        '''Returns success counts and known-outcome percentages per column.'''
        def summarize(group: pd.DataFrame) -> dict[str, int | float]:
            row: dict[str, int | float] = {'T': len(group)}
            for column in [
                *(characteristic.name for characteristic in self.component.characteristics),
                self.component.name,
            ]:
                known = int(group[column].notna().sum())
                successes = int(group[column].sum())
                row[column] = successes
                row[f'{column}(%)'] = (
                    successes / known * 100 if known else float('nan')
                )
            return row

        rows = [summarize(self.df)]
        indexes: list[str | int] = ['total']
        if by_water_years:
            df = self.identify_water_years().dropna(subset=['water_year'])
            wys = df['water_year'].dropna().unique()
            for wy in wys:
                df_wy = df[df['water_year'] == wy]
                rows.append(summarize(df_wy))
                indexes.append(int(wy))
        return pd.DataFrame(rows, index=indexes)

    def plot_success(self,
                     ylimits: None|tuple[float, float] = None,
                     full_timeseries: bool = True) -> None:
        '''Plot the component success over time.'''
        df = self.df
        if not full_timeseries:
            df = self.df[self.df[self.component.characteristics[0].name] == 1]
        _, ax = plt.subplots(figsize=(15, 5))
        data = self.df[self.dv_name]
        df['success'] = self.df[self.component.name] * data
        df['possible'] = self.df[self.component.characteristics[0].name] * data
        df.possible.replace({0: np.nan}).plot(
            color='yellow', linewidth=10, label=self.component.characteristics[0].name, ax=ax)
        data.plot(
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

def evaluate_component(
    df: pd.DataFrame,
    component: Component,
    data_column: int = 0,
    first_day_of_water_year: int | None = None,
) -> Result:
    '''Evaluates a single component on a single timeseries.

    Args:
        df (pd.DataFrame): assumes a dataframe in the form:
            | idx  | flows | dowy |
            |------|-------|------|
            | ...  | ...   | ...  |
        component (Component): a component to evaluate.
        data_column (int): zero-based position of the data column to evaluate;
            the final DOWY column is not eligible. Defaults to 0.

    Returns:
        Result: `df` contains the selected data column (under its original
        name), DOWY, characteristic outputs, and component output. Its index
        is preserved, with DatetimeIndex names normalized to "time".
        `dv_name` is the original selected data-column name.
    '''
    validate_timeseries(df)
    if isinstance(data_column, bool) or not isinstance(data_column, int):
        raise ValueError('data_column must be a zero-based integer data-column index.')
    if data_column < 0 or data_column >= len(df.columns) - 1:
        raise ValueError(
            f'data_column {data_column} is outside the data-column range '
            f'[0, {len(df.columns) - 2}].'
        )
    selected_name = df.columns[data_column]

    # Characteristic factories use the first column for flow data and the last
    # column for DOWY; arrange a view with selected data first for evaluation.
    evaluation_positions = [
        data_column,
        *(i for i in range(len(df.columns) - 1) if i != data_column),
        len(df.columns) - 1,
    ]
    evaluation_df = df.iloc[:, evaluation_positions]
    # length of timeseries, one row per characteristics
    # float dtype (not int) preserves NaN emitted by frequency's sliding-window
    # diagnostic (insufficient trailing history) instead of silently casting it.
    output = np.zeros((len(df), len(component.characteristics)), dtype=float)
    for i, characteristic in enumerate(component.characteristics):
        output[:, i] = characteristic.fx(evaluation_df, output)
    # A terminal frequency diagnostic already incorporates its source
    # conditions and must not be ANDed with them again at the current timestep.
    is_terminal_frequency = (
        component.characteristics[-1].type == CharacteristicType.FREQUENCY
    )
    conditions = output[:, -1:] if is_terminal_frequency else output

    has_failure = np.any(conditions == 0, axis=1)
    all_success = np.all(conditions == 1, axis=1)
    combined = np.full(len(df), np.nan)
    combined[has_failure] = 0
    combined[all_success] = 1
    if not component.is_success_pattern:
        known = ~np.isnan(combined)
        combined[known] = 1 - combined[known]
    success = combined.reshape(-1, 1)
    results = np.concatenate((output, success), axis=1)
    # add 2D array to dataframe
    cols = [j.name for j in component.characteristics] + [component.name]
    result_columns = [selected_name, df.columns[-1], *cols]
    if len(set(result_columns)) != len(result_columns):
        raise ValueError('Result column names must be unique.')
    df = pd.concat(
        [
            df.iloc[:, [data_column, len(df.columns) - 1]],
            pd.DataFrame(results, index=df.index, columns=cols),
        ],
        axis=1,
    )
    if isinstance(df.index, pd.DatetimeIndex):
        df.index.name = 'time'
    return Result(df, component, str(selected_name), first_day_of_water_year)

def evaluate_components(
    df: pd.DataFrame,
    components: list[Component],
    data_column: int = 0,
    first_day_of_water_year: int | None = None,
) -> list[Result]:
    ''''Evaluates a list of components on a single timeseries.'''
    return [
        evaluate_component(
            df,
            component,
            data_column=data_column,
            first_day_of_water_year=first_day_of_water_year,
        )
        for component in components
    ]

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
