'''Phase 5 acceptance tests for calendar timing and complete water years.'''
import numpy as np
import pandas as pd
import pytest

from hydropattern.patterns import (
    comparison_fx,
    identify_full_water_years,
    nested_frequency_intra_annual_fx,
)
from hydropattern.patterns.characteristics import timing_fx
from hydropattern.parsers import timing_parser
from hydropattern.timeseries import Timeseries


def test_timing_uses_calendar_day_of_year_with_october_water_year():
    dates = pd.to_datetime(['2020-09-30', '2020-10-01', '2020-10-02'])
    data = pd.DataFrame(
        {'flow': [1, 1, 1], 'dowy': [365, 1, 2]},
        index=pd.DatetimeIndex(dates, name='time'),
    )

    result = timing_fx(comparison_fx('=', 274))(data)

    np.testing.assert_array_equal(result, [0, 1, 0])


def test_timing_uses_repository_leap_day_convention():
    dates = pd.to_datetime(['2020-02-28', '2020-02-29', '2020-03-01'])
    data = pd.DataFrame(
        {'flow': [1, 1, 1], 'dowy': [150, 151, 152]},
        index=pd.DatetimeIndex(dates, name='time'),
    )

    result = timing_fx(comparison_fx('=', 59))(data)

    np.testing.assert_array_equal(result, [1, 1, 0])


def test_timing_calendar_range_wraps_across_new_year():
    dates = pd.to_datetime(['2020-12-01', '2021-03-01', '2021-06-29'])
    data = pd.DataFrame(
        {'flow': [1, 1, 1], 'dowy': [91, 92, 93]},
        index=pd.DatetimeIndex(dates, name='time'),
    )

    result = timing_parser([335, 60], order=1).fx(data)

    np.testing.assert_array_equal(result, [1, 1, 0])


def test_daily_water_years_exclude_leading_and_trailing_partial_years():
    dates = pd.date_range('2018-11-01', '2020-09-29', freq='D')
    data = pd.DataFrame({'flow': np.ones(len(dates))}, index=dates)
    data.index.name = 'time'
    dowy = Timeseries.from_dataframe(data, first_dowy=274).data['dowy'].to_numpy()

    assert identify_full_water_years(dowy, dates) == []


def test_monthly_water_years_are_complete_with_one_observation_per_month():
    dates = pd.date_range('2019-10-01', '2021-09-01', freq='MS')
    data = pd.DataFrame({'flow': np.ones(len(dates))}, index=dates)
    data.index.name = 'time'
    dowy = Timeseries.from_dataframe(data, first_dowy=274).data['dowy'].to_numpy()

    assert identify_full_water_years(dowy, dates) == [(0, 11), (12, 23)]


def test_monthly_water_years_support_consistent_midmonth_observations():
    dates = pd.date_range('2019-10-15', '2021-09-15', freq=pd.DateOffset(months=1))
    data = pd.DataFrame({'flow': np.ones(len(dates))}, index=dates)
    data.index.name = 'time'
    dowy = Timeseries.from_dataframe(data, first_dowy=288).data['dowy'].to_numpy()

    assert identify_full_water_years(dowy, dates) == [(0, 11), (12, 23)]


def test_month_end_observations_find_water_years_at_dowy_reset():
    dates = pd.date_range('2019-09-30', '2021-10-31', freq=pd.offsets.MonthEnd())
    data = pd.DataFrame({'flow': np.ones(len(dates))}, index=dates)
    data.index.name = 'time'
    dowy = Timeseries.from_dataframe(data, first_dowy=274).data['dowy'].to_numpy()

    assert identify_full_water_years(dowy, dates) == [(1, 12), (13, 24)]


def test_water_year_completeness_rejects_gaps_and_unsupported_cadence():
    daily = pd.date_range('2019-10-01', '2020-09-30', freq='D').delete(100)
    unsupported = pd.to_datetime(['2019-10-01', '2019-10-03', '2019-10-06'])

    with pytest.raises(ValueError, match='gap|cadence'):
        identify_full_water_years(np.ones(len(daily)), daily)
    with pytest.raises(ValueError, match='unsupported|cadence'):
        identify_full_water_years(np.ones(len(unsupported)), unsupported)


def test_daily_full_water_year_survives_adjacent_partial_years():
    dates = pd.date_range('2018-11-01', '2020-10-01', freq='D')
    data = pd.DataFrame({'flow': np.ones(len(dates))}, index=dates)
    data.index.name = 'time'
    dowy = Timeseries.from_dataframe(data, first_dowy=274).data['dowy'].to_numpy()

    assert identify_full_water_years(dowy, dates) == [(334, 699)]


def test_nested_annual_verdict_uses_complete_calendar_years_only():
    dates = pd.date_range('2019-10-01', '2021-10-01', freq='D', name='time')
    data = pd.DataFrame({'flow': np.ones(len(dates))}, index=dates)
    data.iloc[366:731, 0] = 0
    data.index.name = 'time'
    timeseries = Timeseries.from_dataframe(data, first_dowy=274).data
    output = timeseries['flow'].to_numpy().reshape(-1, 1)
    fx = nested_frequency_intra_annual_fx(
        comparison_fx('>=', 0.5), order=2, big_n=None
    )

    result = fx(timeseries, output)

    np.testing.assert_array_equal(result[:366], np.ones(366))
    np.testing.assert_array_equal(result[366:731], np.zeros(365))
    assert np.isnan(result[731])


def test_monthly_nested_verdict_uses_month_end_dowy_resets():
    dates = pd.date_range(
        '2019-09-30', '2021-10-31', freq=pd.offsets.MonthEnd(), name='time'
    )
    data = pd.DataFrame({'flow': np.ones(len(dates))}, index=dates)
    timeseries = Timeseries.from_dataframe(data, first_dowy=274).data
    output = np.ones((len(timeseries), 1))
    fx = nested_frequency_intra_annual_fx(
        comparison_fx('>=', 0.5), order=2, big_n=None
    )

    result = fx(timeseries, output)

    assert np.isnan(result[0])
    np.testing.assert_array_equal(result[1:25], np.ones(24))
    assert np.isnan(result[25])
