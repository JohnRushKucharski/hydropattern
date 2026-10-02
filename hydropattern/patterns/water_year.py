'''
Water-year utility functions.

Pure helpers for locating and reducing statistics over full water years
(day-of-water-year runs starting where dowy == 1), shared by the frequency
characteristic-fx factories in hydropattern.patterns.core.
'''
import numpy as np

from hydropattern.patterns.core import sliding_window_count

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

def record_length_years(dowy: np.ndarray) -> float:
    '''
    Record length, in (water) years, of a day-of-water-year array.

    Reuses identify_full_water_years() directly -- the same dowy-based
    full-water-year detection already used by
    nested_frequency_interannual_fx/water_year_probability_ratio/
    windowed_count_per_water_year -- so "a year" means the same thing here as
    it does everywhere else in this package: it starts wherever dowy==1
    does, not Jan 1. No new run-detection/year-counting logic is introduced.

    Inherits identify_full_water_years' existing (asymmetric) convention: a
    leading partial water year (before the first dowy==1) is excluded, but a
    trailing partial water year still counts as one full year (the final
    entry always runs to the end of the array). That convention is mirrored
    here, not re-litigated.

    Parameters
    ----------
        dowy (np.ndarray): day-of-water-year values (1-365), one per
            timestep.

    Returns
    -------
        float: number of full water years in `dowy`.
    '''
    if len(dowy) == 0:
        raise ValueError('dowy must not be empty.')
    full_years = identify_full_water_years(dowy)
    if not full_years:
        raise ValueError('dowy contains no full water years (no dowy == 1 found).')
    return float(len(full_years))

def water_year_probability_ratio(eligible: np.ndarray, dowy: np.ndarray,
                                 exclusive_event_window: bool = False) -> np.ndarray:
    '''
    Computes, for each full water year (see identify_full_water_years), the
    ratio of eligible timesteps to valid timesteps in a (0/1) trial array.
    This is the statistic behind a nested frequency pattern's intra-annual
    `[operator, probability]` base form.

    The ratio is placed at the water year's *last* timestep and NaN
    elsewhere (including any leading partial year before the first
    `dowy == 1`), matching the trailing-window convention used by
    sliding_window_count: a diagnostic value only becomes known once its
    full window -- here, the water year -- has elapsed.

    Parameters
    ----------
        eligible (np.ndarray): 0/1 trial outcomes (e.g. AND of preceding
            characteristic columns); NaN entries are excluded from numerator
            and denominator.
        dowy (np.ndarray): day-of-water-year values (1-365), same length as
            `eligible`.
        exclusive_event_window (bool): retained for API compatibility; has no
            effect because a single annual probability has no overlapping windows.

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
    for start, end in identify_full_water_years(dowy):
        year = eligible[start:end + 1]
        valid = ~np.isnan(year)
        valid_count = int(valid.sum())
        if valid_count:
            successes = int(np.count_nonzero(year[valid] == 1))
            result[end] = successes / valid_count
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
