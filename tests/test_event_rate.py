'''TDD for Phase 3: event_rate.

event_rate(events, years) = events / years -- a plain descriptive rate, not a
recurrence-interval/Poisson-probability claim (Great Lakes water-level
records show multi-decadal persistence/clustering, so between-event
independence should not be assumed; see docs/developer/adr and prior review notes).

record_length_years(dowy) computes the record length in *water* years, not
calendar years -- reusing identify_full_water_years() (the same dowy-based
full-water-year detection already used by nested_frequency_interannual_fx/
water_year_probability_ratio/windowed_count_per_water_year), so "a year"
means the same thing here as it does everywhere else in this package: it
starts wherever dowy==1 does, not Jan 1. No new run-detection/year-counting
logic is introduced. This also inherits that helper's existing (asymmetric)
convention: a leading partial water year (before the first dowy==1) is
excluded, but a trailing partial water year still counts as one full year --
this mirrors, not re-litigates, the convention already established for
frequency characteristics.

Result.event_rate() = event_count() / record_length_years(self.df['dowy']),
again with no terminal-marker special-casing: T is a property of the record's own
water-year structure (already present via the dowy column validated by
validate_timeseries), not of any one component's success-column grain
(already established grain-agnostic in Phase 2 -- see test_event_count.py).
'''
# pylint: disable=missing-class-docstring,missing-function-docstring
import unittest

import numpy as np
import pandas as pd

from hydropattern.patterns import (
    Component,
    EventCountBounds,
    EventRateBounds,
    Result,
    count_event_bounds,
    evaluate_component,
    event_rate,
    record_length_years,
    water_year_exposure,
    water_year_exposure_by_year,
)
from hydropattern.parsers import duration_parser, magnitude_parser
from hydropattern.timeseries import Timeseries


class TestEventRatePureFunction(unittest.TestCase):
    '''event_rate: plain division, no statistical claims baked in.'''

    def test_events_per_year(self):
        self.assertAlmostEqual(event_rate(10, 1030.0), 10 / 1030.0)

    def test_zero_events_is_zero_rate(self):
        self.assertEqual(event_rate(0, 50.0), 0.0)

    def test_raises_on_non_positive_years(self):
        with self.assertRaises(ValueError):
            event_rate(1, 0.0)
        with self.assertRaises(ValueError):
            event_rate(1, -5.0)


class TestRecordLengthYears(unittest.TestCase):
    '''record_length_years: water-year length via identify_full_water_years,
    same convention as nested_frequency_interannual_fx -- not calendar years.
    '''

    @staticmethod
    def _daily_water_year(start, end):
        dates = pd.date_range(start, end, freq='D', name='time')
        data = pd.DataFrame({'flow': np.ones(len(dates))}, index=dates)
        return Timeseries.from_dataframe(data, first_dowy=274).data['dowy'].to_numpy(), dates

    def test_counts_full_water_years(self):
        dowy, dates = self._daily_water_year('2018-10-01', '2021-10-01')
        self.assertEqual(record_length_years(dowy, dates), 3.0)

    def test_leading_partial_water_year_is_excluded(self):
        dowy, dates = self._daily_water_year('2018-11-01', '2020-10-01')
        self.assertEqual(record_length_years(dowy, dates), 1.0)

    def test_trailing_partial_water_year_is_excluded(self):
        dowy, dates = self._daily_water_year('2019-10-01', '2021-09-29')
        self.assertEqual(record_length_years(dowy, dates), 1.0)

    def test_raises_on_no_full_water_years(self):
        dowy, dates = self._daily_water_year('2019-11-01', '2020-02-01')
        with self.assertRaises(ValueError):
            record_length_years(dowy, dates)

    def test_raises_on_empty_dowy(self):
        with self.assertRaises(ValueError):
            record_length_years(np.array([]), pd.DatetimeIndex([]))

    def test_timestamps_are_required(self):
        with self.assertRaisesRegex(ValueError, 'timestamps are required'):
            record_length_years(np.array([1.0, 2.0]))


class TestResultEventRate(unittest.TestCase):
    '''Result.event_rate() combines event_count() with record_length_years().'''

    def test_matches_manual_division(self):
        dates = pd.date_range('1970-01-01', '1977-01-01', freq='D', name='time')
        data = pd.DataFrame({'flow': 100.0}, index=dates)
        data.iloc[:48, 0] = 1.0
        data.iloc[48 + 5:48 + 5 + 40, 0] = 1.0
        dowy = Timeseries.from_dataframe(data).data['dowy'].to_numpy()
        df = data.assign(dowy=dowy)
        component = Component(
            name='low_water_cycle',
            characteristics=[
                magnitude_parser(['<', 2.0], order=1),
                duration_parser([36, 60], order=2),
            ],
            is_success_pattern=True,
        )
        result = evaluate_component(df, component)
        expected = result.event_count() / water_year_exposure(dates, 1)
        self.assertAlmostEqual(result.event_rate(), expected)
        self.assertEqual(result.event_count(), 2)
        self.assertEqual(record_length_years(dowy, dates), 7.0)

    def test_unknown_outcomes_return_conservative_count_and_rate_bounds(self):
        dates = pd.date_range('2020-01-01', periods=3, freq='D', name='time')
        result = self._result([1, np.nan, 1], dates, 1)

        self.assertEqual(result.event_count_bounds(), EventCountBounds(1, 2))
        rates = result.event_rate_bounds()
        self.assertAlmostEqual(rates.lower, 365 / 3)
        self.assertAlmostEqual(rates.upper, 2 * 365 / 3)
        with self.assertRaisesRegex(ValueError, 'event_count_bounds'):
            result.event_count()
        with self.assertRaisesRegex(ValueError, 'event_rate_bounds'):
            result.event_rate()

    def test_whole_record_rate_uses_partial_year_exposure(self):
        dates = pd.date_range('2020-01-01', periods=18, freq='MS', name='time')
        outcomes = np.zeros(18)
        outcomes[[0, 2, 14]] = 1
        result = self._result(outcomes, dates, 1)

        self.assertEqual(water_year_exposure_by_year(dates, 1), {2020: 1.0, 2021: 0.5})
        self.assertEqual(result.event_count_bounds(), EventCountBounds(3, 3))
        self.assertEqual(result.event_rate_bounds(), EventRateBounds(2.0, 2.0))
        self.assertEqual(result.event_rate(), 2.0)

    def test_result_infers_boundary_from_timestamps_and_dowy(self):
        dates = pd.date_range('2020-10-01', periods=18, freq='MS', name='time')
        result = self._result(np.zeros(len(dates)), dates, 274, include_metadata=False)

        self.assertEqual(result.event_rate_bounds(), EventRateBounds(0.0, 0.0))

    def test_annual_event_attribution_uses_start_year_across_boundary(self):
        dates = pd.date_range('2020-09-29', '2020-10-03', freq='D', name='time')
        result = self._result(np.ones(len(dates)), dates, 274)

        self.assertEqual(
            result.event_count_bounds_by_water_year(),
            {2020: EventCountBounds(1, 1), 2021: EventCountBounds(0, 0)},
        )
        rates = result.event_rate_bounds_by_water_year()
        self.assertEqual(rates[2020], EventRateBounds(365 / 2, 365 / 2))
        self.assertEqual(rates[2021], EventRateBounds(0.0, 0.0))

    def test_unknown_bounds_allow_independent_mvp_completions(self):
        self.assertEqual(count_event_bounds(np.full(3, np.nan)), EventCountBounds(0, 2))

    def test_invalid_final_outcome_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'only 0, 1, or NaN'):
            count_event_bounds(np.array([0, 2]))

    def test_unsupported_timestamps_reject_event_rate_bounds(self):
        dates = pd.DatetimeIndex(
            ['2020-01-01', '2020-01-03', '2020-01-04'], name='time'
        )
        result = self._result([0, 1, 0], dates, 1)

        with self.assertRaisesRegex(ValueError, 'unsupported cadence or data gap'):
            result.event_rate_bounds()

    @staticmethod
    def _result(outcomes, dates, boundary, include_metadata=True):
        dowy = Timeseries.from_dataframe(
            pd.DataFrame({'flow': np.ones(len(dates))}, index=dates),
            first_dowy=boundary,
        ).data['dowy'].to_numpy()
        frame = pd.DataFrame(
            {'flow': np.ones(len(dates)), 'dowy': dowy, 'component': outcomes},
            index=dates,
        )
        boundary_metadata = boundary if include_metadata else None
        return Result(frame, Component('component', [], True),
                      first_day_of_water_year=boundary_metadata)


if __name__ == '__main__':
    unittest.main()
