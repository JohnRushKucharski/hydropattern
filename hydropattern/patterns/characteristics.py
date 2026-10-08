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
        # Timing is an independent diagnostic (see docs/developer/plans/2026-10-01-
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
def _duration_run_length_bounds(
    eligible: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    '''Return shortest and longest possible qualifying run at each timestep.'''
    length = len(eligible)
    indices = np.arange(length)
    known_zero = eligible == 0
    segment_start = np.maximum.accumulate(
        np.where(known_zero, indices + 1, 0)
    )
    segment_end = np.minimum.accumulate(
        np.where(known_zero, indices - 1, length - 1)[::-1]
    )[::-1]
    preceding_ones = np.zeros(length, dtype=int)
    following_ones = np.zeros(length, dtype=int)
    for index in range(1, length):
        if eligible[index - 1] == 1:
            preceding_ones[index] = preceding_ones[index - 1] + 1
    for index in range(length - 2, -1, -1):
        if eligible[index + 1] == 1:
            following_ones[index] = following_ones[index + 1] + 1

    minimum = preceding_ones + following_ones + 1
    maximum = segment_end - segment_start + 1
    maximum[known_zero] = 0
    return minimum, maximum


def _possible_duration_runs(
    eligible: np.ndarray,
    run_lengths: np.ndarray,
) -> np.ndarray:
    '''Mark timesteps contained in a possible run of any supplied length.'''
    length = len(eligible)
    coverage = np.zeros(length + 1, dtype=int)
    zero_counts = np.concatenate(([0], np.cumsum(eligible == 0)))
    all_starts = np.arange(length)
    for run_length in run_lengths:
        if run_length > length:
            continue
        starts = all_starts[:length - run_length + 1]
        ends = starts + run_length - 1
        valid = zero_counts[ends + 1] == zero_counts[starts]
        # A neighboring known success must belong to the same maximal run.
        valid &= (starts == 0) | (eligible[np.maximum(starts - 1, 0)] != 1)
        valid &= (ends == length - 1) | (
            eligible[np.minimum(ends + 1, length - 1)] != 1
        )
        starts = starts[valid]
        ends = ends[valid]
        coverage[starts] += 1
        coverage[ends + 1] -= 1
    return np.cumsum(coverage[:-1]) > 0


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

        precedents = output[:, :order - 1]
        eligible = np.full(len(df), 1.0)
        known_failure = np.any(
            (precedents != 1) & ~np.isnan(precedents), axis=1
        )
        unknown = np.any(np.isnan(precedents), axis=1) & ~known_failure
        eligible[known_failure] = 0
        eligible[unknown] = np.nan

        length = len(eligible)
        if not np.isnan(eligible).any():
            result = np.zeros(length)
            for start, end in find_runs(eligible):
                if f(end - start + 1):
                    result[start:end + 1] = 1
            return result

        matches = np.fromiter(
            (bool(f(run_length)) for run_length in range(1, length + 1)),
            dtype=bool,
            count=length,
        )
        true_runs = find_runs(matches.astype(float))
        increasing = np.all(matches[:-1] <= matches[1:])
        decreasing = np.all(matches[:-1] >= matches[1:])
        if increasing or decreasing:
            minimum_length, maximum_length = _duration_run_length_bounds(eligible)
            possible_minimum = matches[minimum_length - 1]
            possible_maximum = np.zeros(length, dtype=bool)
            known_zero = eligible == 0
            possible_maximum[~known_zero] = matches[
                maximum_length[~known_zero] - 1
            ]
            same_verdict = possible_minimum == possible_maximum
            known_success = eligible == 1
            unknown_eligible = np.isnan(eligible)
            result = np.zeros(length)
            result[
                known_success & same_verdict & possible_minimum
            ] = 1
            result[
                (known_success & ~same_verdict)
                | (
                    unknown_eligible
                    & (possible_minimum | possible_maximum)
                )
            ] = np.nan
            return result

        if len(true_runs) == 1:
            minimum_length, maximum_length = _duration_run_length_bounds(eligible)
            possible_minimum = matches[minimum_length - 1]
            possible_maximum = np.zeros(length, dtype=bool)
            known_zero = eligible == 0
            possible_maximum[~known_zero] = matches[
                maximum_length[~known_zero] - 1
            ]
            can_succeed = possible_minimum | possible_maximum
            scan_lengths = np.arange(true_runs[0][0] + 1, true_runs[0][1] + 2)
            can_succeed |= _possible_duration_runs(eligible, scan_lengths)
            known_success = eligible == 1
            unknown_eligible = np.isnan(eligible)
            can_fail = ~possible_minimum | ~possible_maximum
            result = np.zeros(length)
            result[known_success & can_succeed & ~can_fail] = 1
            result[
                (known_success & can_succeed & can_fail)
                | (unknown_eligible & can_succeed)
            ] = np.nan
            return result

        if (
            len(true_runs) == 2
            and true_runs[1][0] - true_runs[0][1] == 2
        ):
            minimum_length, maximum_length = _duration_run_length_bounds(eligible)
            possible_minimum = matches[minimum_length - 1]
            possible_maximum = np.zeros(length, dtype=bool)
            known_zero = eligible == 0
            possible_maximum[~known_zero] = matches[
                maximum_length[~known_zero] - 1
            ]
            rejected_length = true_runs[0][1] + 2
            can_fail = ~possible_minimum | ~possible_maximum
            can_fail |= _possible_duration_runs(
                eligible, np.array([rejected_length])
            )
            can_succeed = possible_minimum | possible_maximum
            known_success = eligible == 1
            unknown_eligible = np.isnan(eligible)
            result = np.zeros(length)
            result[known_success & can_succeed & ~can_fail] = 1
            result[
                (known_success & can_succeed & can_fail)
                | (unknown_eligible & can_succeed)
            ] = np.nan
            return result

        run_starts = np.arange(length)
        zero_counts = np.concatenate(
            ([0], np.cumsum(eligible == 0))
        )
        possible_success = np.zeros(length + 1, dtype=int)
        possible_failure = np.zeros(length + 1, dtype=int)
        for run_length in range(1, length + 1):
            starts = run_starts[:length - run_length + 1]
            ends = starts + run_length - 1
            valid = zero_counts[ends + 1] == zero_counts[starts]
            valid &= (starts == 0) | (eligible[np.maximum(starts - 1, 0)] != 1)
            valid &= (ends == length - 1) | (
                eligible[np.minimum(ends + 1, length - 1)] != 1
            )
            starts = starts[valid]
            ends = ends[valid]
            target = possible_success if matches[run_length - 1] else possible_failure
            target[starts] += 1
            target[ends + 1] -= 1

        can_succeed = np.cumsum(possible_success[:-1]) > 0
        can_fail_as_run = np.cumsum(possible_failure[:-1]) > 0
        known_success = eligible == 1
        unknown_eligible = np.isnan(eligible)
        result = np.zeros(length)
        result[known_success & can_succeed & ~can_fail_as_run] = 1
        result[
            (known_success & can_succeed & can_fail_as_run)
            | (unknown_eligible & can_succeed)
        ] = np.nan
        return result
    return closure
#endregion

#region frequency
def _forward_frequency_window(eligible: np.ndarray, f: Callable[[float], bool],
                             big_n: int, exclusive_windows: bool) -> np.ndarray:
    '''Evaluate forward count windows while preserving unknown trials.'''
    length = len(eligible)
    result = np.zeros(length)
    if length == 0:
        return result

    count_matches = np.fromiter(
        (bool(f(count)) for count in range(min(big_n, length) + 1)),
        dtype=bool,
        count=min(big_n, length) + 1,
    )
    matching_counts = np.concatenate(
        ([0], np.cumsum(count_matches, dtype=int))
    )
    zero_admitting = count_matches[0]
    known_ones = np.concatenate(([0], np.cumsum(eligible == 1)))
    unknowns = np.concatenate(([0], np.cumsum(np.isnan(eligible))))
    possible_success = np.zeros(length, dtype=bool)
    definite_success = np.zeros(length, dtype=bool)
    successful_anchor_possible = np.zeros(length, dtype=bool)
    definite_anchor = np.zeros(length, dtype=bool)
    possible_fail = np.zeros(length, dtype=bool)

    for start in range(length):
        end = min(start + big_n, length)
        ones = int(known_ones[end] - known_ones[start])
        unknown_count = int(unknowns[end] - unknowns[start])
        anchor_unknown = not zero_admitting and np.isnan(eligible[start])
        if not zero_admitting and eligible[start] == 0:
            continue

        minimum = ones + int(anchor_unknown)
        maximum = ones + unknown_count
        matching_count = (
            matching_counts[maximum + 1] - matching_counts[minimum]
        )
        possible_count = maximum - minimum + 1
        can_pass = matching_count > 0
        can_fail = matching_count < possible_count
        if not can_pass:
            continue

        successful_anchor_possible[start] = True
        definite_anchor[start] = zero_admitting or eligible[start] == 1
        possible_fail[start] = can_fail
        possible_success[start:end] = True
        if definite_anchor[start] and not can_fail:
            definite_success[start:end] = True

    if not exclusive_windows:
        result[definite_success] = 1
        result[possible_success & ~definite_success] = np.nan
        return result

    schedules = {-1}
    for timestep in range(length):
        next_schedules: set[int] = set()
        has_covered_schedule = False
        has_uncovered_schedule = False
        for claimed_until in schedules:
            if claimed_until >= timestep:
                next_schedules.add(claimed_until)
                has_covered_schedule = True
                continue

            anchor_possible = successful_anchor_possible[timestep]
            anchor_definite = definite_anchor[timestep]
            if not anchor_possible or not anchor_definite:
                next_schedules.add(-1)
                has_uncovered_schedule = True
            if anchor_possible:
                end = min(timestep + big_n - 1, length - 1)
                next_schedules.add(end)
                has_covered_schedule = True
                if possible_fail[timestep]:
                    next_schedules.add(-1)
                    has_uncovered_schedule = True

        result[timestep] = (
            1 if has_covered_schedule and not has_uncovered_schedule
            else np.nan if has_covered_schedule
            else 0
        )
        schedules = next_schedules
    return result


def frequency_fx(f: Callable[[float], bool], order: int,
                 big_n: int | None = None,
                 exclusive_windows: bool = False) -> CharacteristicFx:
    '''
    Creates function to evaluate an un-nested frequency characteristic.

    Parameters
    ----------
        f (Callable[[float], bool]): Comparison function applied to the
            timestep count in the forward frequency window.
        order (int): Position in which characteristic is evaluated within component.
            Must last component charactersistic (enforced upstream in builders.py).
        big_n (int | None): maximum forward frequency-window length in
            timesteps for [op, n, N] and [min_n, max_n, N] forms. None is
            invalid here; the annual fraction form is only implemented as the
            intra-annual part of a two-part frequency characteristic.
        exclusive_windows (bool): False by default. If true, a qualifying
            frequency window prevents later anchors within its fixed N-timestep
            span, so windows do not overlap. If false, overlapping windows are
            combined.
    Returns
    -------
        Characteristic_fx: evaluates characteristic over timeseries.

    Note
    ----
        When the count condition does not accept zero, windows are anchored at
        qualifying timesteps, where all preceding component conditions are met.
        If the condition accepts zero, every timestep can anchor a window.
        With exclusive windows, a qualifying window suppresses later anchors
        within its span; a window that does not qualify does not suppress them.
        Windows at the end of the record are evaluated using the available
        timesteps in the shortened window.
    '''
    def closure(df: pd.DataFrame,
                output: None|np.ndarray) -> np.ndarray:
        validate_order(order, output, CharacteristicType.FREQUENCY)
        assert output is not None # for mypy: checked by validate_order

        if big_n is None:
            raise NotImplementedError(
                'un-nested [operator, probability] frequency form is not valid; '
                'probability form is only implemented as the intra-annual pattern '
                '(see notes/frequencyEnhancement-resolved.md).'
            )

        precedents = output[:, :order-1]
        known_failure = np.any(
            (precedents != 1) & ~np.isnan(precedents), axis=1
        )
        unknown = np.any(np.isnan(precedents), axis=1) & ~known_failure
        eligible = np.ones(len(df))
        eligible[known_failure] = 0
        eligible[unknown] = np.nan
        return _forward_frequency_window(eligible, f, big_n, exclusive_windows)
    return closure

def _intra_annual_diagnostic(
    eligible: np.ndarray,
    dowy: np.ndarray,
    timestamps: pd.DatetimeIndex | None,
    f: Callable[[float], bool],
    big_n: int | None,
    exclusive_windows: bool,
) -> np.ndarray:
    '''Shared raw-diagnostic computation for the intra-annual pattern in a
    two-part frequency characteristic. Annual fractions are compared once per
    complete water year and broadcast; count/range forms use forward frequency
    windows within each water year.
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
            eligible[start:end + 1], f, big_n, exclusive_windows
        )
    return diag

def nested_frequency_intra_annual_fx(f: Callable[[float], bool], order: int,
                                     big_n: int | None = None,
                                     exclusive_windows: bool = False) -> CharacteristicFx:
    '''
    Creates the function that evaluates the intra-annual column of a two-part
    frequency characteristic.

    The annual qualifying fraction uses qualifying timesteps and is compared
    once per complete water year, with its verdict broadcast across that year.
    Count/range forms evaluate forward frequency windows within each water
    year. `exclusive_windows` applies to those windows, not to the annual
    fraction.

    Parameters
    ----------
        f (Callable[[float], bool]): Comparison function for the intra-annual pattern.
        order (int): Position in the component's characteristic sequence.
        big_n (int | None): maximum forward frequency-window length in
            timesteps for intra-annual count/range forms; None for the annual
            fraction form.
        exclusive_windows (bool): Controls overlap among intra-annual
            frequency windows; it has no effect on the annual fraction form.
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
            eligible, dowy, timestamps, f, big_n, exclusive_windows
        )
    return closure

def nested_frequency_interannual_fx(f: Callable[[float], bool], order: int,
                                    big_n: int | None = None,
                                    exclusive_windows: bool = False) -> CharacteristicFx:
    '''
    Creates the function that evaluates the interannual column of a two-part
    frequency characteristic. Its result determines the final component
    outcome and is broadcast across complete water years.

    The intra-annual fraction form has no frequency window; its count and
    range forms use their own window setting. The interannual setting
    controls overlap among forward windows measured in complete water years.
    A window counts the water years that qualify under the intra-annual
    condition.

    Parameters
    ----------
        f (Callable[[float], bool]): Comparison function for the interannual pattern.
        order (int): Position in the component's characteristic sequence.
            The intra-annual column (this pattern's input) must immediately
            precede this characteristic, at column index `order - 2`.
        big_n (int | None): maximum forward frequency-window length in water
            years for interannual count/range forms. The annual fraction form
            is not valid at this level (enforced upstream in
            validate_nested_frequency_metrics).
        exclusive_windows (bool): Controls overlap among interannual
            frequency windows by suppressing later water-year anchors within
            a successful window.
    Returns
    -------
        Characteristic_fx: Evaluates the interannual column, broadcast across
        complete water years. This is the component's final value when
        frequency is the terminal characteristic (see evaluate_component).
    '''
    def closure(df: pd.DataFrame,
                output: None|np.ndarray) -> np.ndarray:
        validate_order(order, output, CharacteristicType.FREQUENCY)
        assert output is not None # for mypy: checked by validate_order
        if big_n is None:
            raise NotImplementedError(
                'nested [operator, probability] interannual frequency form is not valid; '
                'probability is only valid as the intra-annual pattern '
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
            f, big_n, exclusive_windows
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
