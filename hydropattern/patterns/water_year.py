'''
Water-year utility functions.

Pure helpers for locating and reducing statistics over full water years
(day-of-water-year runs starting where dowy == 1), shared by the frequency
characteristic-fx factories in hydropattern.patterns.core.
'''
import numpy as np
import pandas as pd

from hydropattern.patterns.core import sliding_window_count

def identify_full_water_years(
    dowy: np.ndarray,
    timestamps: pd.DatetimeIndex | None = None,
) -> list[tuple[int, int]]:
    '''
    Identifies (start_idx, end_idx) index pairs (both inclusive) for each
    water year in a day-of-water-year array.

    A water year starts where DOWY resets (or equals 1). If timestamps are
    provided, only complete daily or monthly water years are returned, and
    unsupported or gapped observation schedules raise ValueError. Without
    timestamps, retain legacy boundary-only behavior.

    Parameters
    ----------
        dowy (np.ndarray): day-of-water-year values (1-365).
        timestamps (pd.DatetimeIndex | None): observation dates aligned to
            `dowy`; enables cadence-aware water-year completeness.

    Returns
    -------
        list[tuple[int, int]]: (start_idx, end_idx) pairs, in series order.
    '''
    dates = None
    cadence = None
    if timestamps is not None:
        if len(timestamps) != len(dowy):
            raise ValueError('timestamps and dowy must have the same length.')
        dates = pd.DatetimeIndex(timestamps)
        if dates.hasnans or not dates.is_monotonic_increasing or dates.has_duplicates:
            raise ValueError('timestamps must be valid, unique, and increasing.')
        cadence = _water_year_cadence(dates)

    starts = [
        i for i, day in enumerate(dowy)
        if day == 1 or (i > 0 and day < dowy[i - 1])
    ]
    if not starts:
        return []
    years = [
        (start, starts[i + 1] - 1 if i + 1 < len(starts) else len(dowy) - 1)
        for i, start in enumerate(starts)
    ]
    if timestamps is None:
        return years
    assert dates is not None and cadence is not None
    return [
        (start, end)
        for start, end in years
        if _is_complete_water_year(start, end, dates, cadence)
    ]


def _water_year_cadence(dates: pd.DatetimeIndex) -> str:
    if len(dates) < 2:
        raise ValueError('Cannot determine water-year cadence from fewer than two timestamps.')
    day_numbers = np.array([date.date().toordinal() for date in dates])
    if np.all(np.diff(day_numbers) == 1):
        return 'daily'
    month_numbers = dates.year * 12 + dates.month
    regular_month_day = (
        np.all(dates.day == dates[0].day)
        and dates[0].day <= 28
    )
    if (
        np.all(np.diff(month_numbers) == 1)
        and (np.all(dates.day == 1) or np.all(dates.is_month_end) or regular_month_day)
    ):
        if np.all(dates.day == 1):
            return 'month_start'
        if np.all(dates.is_month_end):
            return 'month_end'
        return 'month_day'
    raise ValueError(
        'Water-year completeness requires contiguous daily or monthly timestamps; '
        'unsupported cadence or data gap found.'
    )


def _is_complete_water_year(
    start: int,
    end: int,
    dates: pd.DatetimeIndex,
    cadence: str,
) -> bool:
    start_date = dates[start]
    next_boundary = start_date + pd.DateOffset(years=1)
    if cadence == 'daily':
        expected = pd.date_range(start_date, next_boundary, freq='D', inclusive='left')
    elif cadence == 'month_start':
        expected = pd.date_range(
            start_date, next_boundary, freq=pd.offsets.MonthBegin(), inclusive='left'
        )
    elif cadence == 'month_end':
        expected = pd.date_range(
            start_date, next_boundary, freq=pd.offsets.MonthEnd(), inclusive='left'
        )
    else:
        expected = pd.date_range(
            start_date, next_boundary, freq=pd.DateOffset(months=1), inclusive='left'
        )

    observed = dates[start:end + 1]
    return observed.equals(expected)

def record_length_years(
    dowy: np.ndarray,
    timestamps: pd.DatetimeIndex | None = None,
) -> float:
    '''
    Record length, in (water) years, of a day-of-water-year array.

    Reuses identify_full_water_years() directly, so years begin where
    DOWY resets, not on January 1.

    Counts only cadence-verified complete water years. Timestamps are
    required because DOWY resets alone cannot establish year completeness.

    Parameters
    ----------
        dowy (np.ndarray): day-of-water-year values (1-365), one per
            timestep.
        timestamps (pd.DatetimeIndex | None): observation dates aligned to
            `dowy`; when provided, only cadence-verified complete years count.

    Returns
    -------
        float: number of full water years in `dowy`.
    '''
    if len(dowy) == 0:
        raise ValueError('dowy must not be empty.')
    if timestamps is None:
        raise ValueError('timestamps are required to determine complete water years.')
    full_years = identify_full_water_years(dowy, timestamps)
    if not full_years:
        raise ValueError('dowy contains no full water years (no dowy == 1 found).')
    return float(len(full_years))

def water_year_probability_ratio(eligible: np.ndarray, dowy: np.ndarray,
                                 exclusive_windows: bool = False,
                                 timestamps: pd.DatetimeIndex | None = None) -> np.ndarray:
    '''
    Computes, for each full water year (see identify_full_water_years), the
    ratio of eligible timesteps to valid timesteps in a (0/1) trial array.
    This is the statistic behind a nested frequency pattern's intra-annual
    `[operator, probability]` base form.

    The ratio is placed at the water year's last timestep and NaN elsewhere;
    the nested characteristic broadcasts the annual verdict after comparing
    this ratio.

    Parameters
    ----------
        eligible (np.ndarray): 0/1 trial outcomes (e.g. AND of preceding
            characteristic columns); NaN entries are excluded from numerator
            and denominator.
        dowy (np.ndarray): day-of-water-year values (1-365), same length as
            `eligible`.
        exclusive_windows (bool): has no effect because a single annual
            probability has no overlapping windows.

    Returns
    -------
        np.ndarray: same shape as `eligible`; ratio at each full water year's
        last timestep, NaN elsewhere.
    '''
    if len(eligible) != len(dowy):
        raise ValueError(
            f'eligible (len={len(eligible)}) and dowy (len={len(dowy)}) must be the same length.'
        )
    result = np.full(len(eligible), np.nan)
    for start, end in identify_full_water_years(dowy, timestamps):
        year = eligible[start:end + 1]
        valid = ~np.isnan(year)
        valid_count = int(valid.sum())
        if valid_count:
            successes = int(np.count_nonzero(year[valid] == 1))
            result[end] = successes / valid_count
    return result

def windowed_count_per_water_year(eligible: np.ndarray, dowy: np.ndarray,
                                  window: int,
                                  timestamps: pd.DatetimeIndex | None = None) -> np.ndarray:
    '''
    Per full water year (see identify_full_water_years), a sliding-window
    count of successes -- the count/between-form counterpart
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
        window (int): window size (in timesteps), i.e. N.

    Returns
    -------
        np.ndarray: same shape as `eligible`; window count at each
        timestep within a full water year (NaN for the year's first
        `window - 1` timesteps), NaN throughout any excluded partial year.
    '''
    if len(eligible) != len(dowy):
        raise ValueError(
            f'eligible (len={len(eligible)}) and dowy (len={len(dowy)}) must be the same length.'
        )
    result = np.full(len(eligible), np.nan)
    for start, end in identify_full_water_years(dowy, timestamps):
        result[start:end + 1] = sliding_window_count(eligible[start:end + 1], window)
    return result

def or_reduce_per_water_year(
    diag: np.ndarray,
    dowy: np.ndarray,
    timestamps: pd.DatetimeIndex | None = None,
) -> np.ndarray:
    '''
    Per full water year (see identify_full_water_years), OR-reduces a 0/1/NaN
    diagnostic column (e.g. an intra_annual column) to a single year verdict,
    placed at the year's last timestep (NaN elsewhere, including excluded
    partial years).

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
    for start, end in identify_full_water_years(dowy, timestamps):
        year = diag[start:end + 1]
        non_nan = year[~np.isnan(year)]
        if len(non_nan) == 0:
            continue  # remains NaN: entire year is insufficient-history
        result[end] = 1.0 if np.any(non_nan == 1) else 0.0
    return result
