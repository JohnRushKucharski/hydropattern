'''
Water-year utility functions.

Pure helpers for locating and reducing statistics over full water years
(day-of-water-year runs starting where dowy == 1), shared by the frequency
characteristic-fx factories in hydropattern.patterns.core.
'''
import numpy as np
import pandas as pd

from hydropattern.patterns.core import sliding_window_count


def validate_water_year_boundary(first_day_of_water_year: int) -> int:
    '''Validate and return a one-based non-leap-year water-year start day.'''
    if (
        isinstance(first_day_of_water_year, bool)
        or not isinstance(first_day_of_water_year, (int, np.integer))
        or not 1 <= first_day_of_water_year <= 365
    ):
        raise ValueError('first_day_of_water_year must be an integer between 1 and 365.')
    return int(first_day_of_water_year)


def water_year_label(timestamp: pd.Timestamp, first_day_of_water_year: int) -> int:
    '''Return ending-year water-year label for a timestamp and boundary day.'''
    boundary = validate_water_year_boundary(first_day_of_water_year)
    if boundary == 1:
        return timestamp.year
    day_of_year = timestamp.dayofyear
    if timestamp.is_leap_year and day_of_year > 59:
        day_of_year -= 1
    return timestamp.year + (day_of_year >= boundary)


def identify_full_water_years(
    dowy: np.ndarray,
    timestamps: pd.DatetimeIndex | None = None,
    first_day_of_water_year: int | None = None,
) -> list[tuple[int, int]]:
    '''
    Identifies (start_idx, end_idx) index pairs (both inclusive) for each
    water year in a day-of-water-year array.

    With timestamps, calendar labels and the configured boundary determine
    year segments; only complete daily or monthly segments are returned.
    Without timestamps, retain legacy DOWY-boundary behavior.

    Parameters
    ----------
        dowy (np.ndarray): day-of-water-year values (1-365).
        timestamps (pd.DatetimeIndex | None): observation dates aligned to
            `dowy`; enables cadence-aware water-year completeness.
        first_day_of_water_year (int | None): normalized boundary day. When
            omitted, infer it from the aligned timestamps and DOWY values.

    Returns
    -------
        list[tuple[int, int]]: (start_idx, end_idx) pairs, in series order.
    '''
    values = np.asarray(dowy)
    if (
        values.ndim != 1
        or np.any(~np.isfinite(values))
        or np.any(values != np.floor(values))
        or np.any((values < 1) | (values > 365))
    ):
        raise ValueError('dowy must contain integer values between 1 and 365.')
    dates = None
    cadence = None
    if timestamps is not None:
        if len(timestamps) != len(dowy):
            raise ValueError('timestamps and dowy must have the same length.')
        dates = pd.DatetimeIndex(timestamps)
        if dates.hasnans or not dates.is_monotonic_increasing or dates.has_duplicates:
            raise ValueError('timestamps must be valid, unique, and increasing.')
        cadence = _water_year_cadence(dates)
        inferred_boundary = _infer_first_day_of_water_year(values, dates)
        boundary = first_day_of_water_year
        if boundary is None:
            boundary = inferred_boundary
        elif validate_water_year_boundary(boundary) != inferred_boundary:
            raise ValueError(
                'first_day_of_water_year conflicts with timestamps and dowy.'
            )
        water_year_label(dates[0], boundary)
        labels = np.array([water_year_label(date, boundary) for date in dates])

    if timestamps is None:
        starts = [
            i for i, day in enumerate(dowy)
            if day == 1 or (i > 0 and day < dowy[i - 1])
        ]
    else:
        starts = [0] + [
            i for i in range(1, len(labels)) if labels[i] != labels[i - 1]
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


def water_year_exposure(
    timestamps: pd.DatetimeIndex,
    first_day_of_water_year: int,
) -> float:
    '''Return observed exposure in water years across supported daily/monthly dates.

    Daily water years use 366 days only when February 29 is present; otherwise
    they use 365. Monthly exposure is the number of observations divided by 12.
    '''
    dates = pd.DatetimeIndex(timestamps)
    if dates.hasnans or not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError('timestamps must be valid, unique, and increasing.')
    cadence = _water_year_cadence(dates)
    boundary = validate_water_year_boundary(first_day_of_water_year)
    labels = np.array([water_year_label(date, boundary) for date in dates])
    exposure = 0.0
    for label in np.unique(labels):
        year_dates = dates[labels == label]
        if cadence == 'daily':
            denominator = 366 if ((year_dates.month == 2) & (year_dates.day == 29)).any() else 365
        else:
            denominator = 12
        exposure += len(year_dates) / denominator
    return exposure


def _water_year_cadence(dates: pd.DatetimeIndex) -> str:
    if len(dates) < 2:
        raise ValueError('Cannot determine water-year cadence from fewer than two timestamps.')
    day_numbers = np.array([date.date().toordinal() for date in dates])
    differences = np.diff(day_numbers)
    leap_day_omission = np.array([
        left.month == 2 and left.day == 28 and left.is_leap_year
        and right.month == 3 and right.day == 1
        for left, right in zip(dates[:-1], dates[1:])
    ])
    if np.all((differences == 1) | ((differences == 2) & leap_day_omission)):
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
        observed = dates[start:end + 1]
        if not observed.isin(expected).all():
            return False
        missing = expected.difference(observed)
        if any(not (date.month == 2 and date.day == 29) for date in missing):
            return False
        return True
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


def _infer_first_day_of_water_year(
    dowy: np.ndarray, dates: pd.DatetimeIndex
) -> int:
    normalized_days = np.array([
        date.dayofyear - int(date.is_leap_year and date.dayofyear > 59)
        for date in dates
    ])
    candidates = (normalized_days - np.asarray(dowy, dtype=int)) % 365 + 1
    unique_candidates = np.unique(candidates)
    if len(unique_candidates) != 1:
        raise ValueError(
            'Cannot establish one water-year boundary from timestamps and dowy.'
        )
    return int(unique_candidates[0])

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
    ratio of qualifying timesteps to all observed timesteps in a (0/1) array.
    This is the statistic behind a nested frequency pattern's intra-annual
    `[operator, probability]` base form.

    The ratio is placed at the water year's last timestep and NaN elsewhere;
    the nested characteristic broadcasts the annual verdict after comparing
    this ratio.

    Parameters
    ----------
        eligible (np.ndarray): 0/1/NaN trial outcomes (e.g. AND of preceding
            characteristic columns). An unknown trial makes the scalar annual
            ratio undefined (NaN); nested frequency evaluates all possible
            fractions instead.
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
        if not np.isnan(year).any():
            result[end] = np.count_nonzero(year == 1) / len(year)
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
    `0` if every cell is `0`; `NaN` when there is no definite success and
    at least one unknown outcome.

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
        if np.any(year == 1):
            result[end] = 1
        elif np.all(year == 0):
            result[end] = 0
    return result
