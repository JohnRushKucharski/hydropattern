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
    evaluate_component,
    event_rate,
    record_length_years,
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
        expected = result.event_count() / record_length_years(dowy, dates)
        self.assertAlmostEqual(result.event_rate(), expected)
        self.assertEqual(result.event_count(), 2)
        self.assertEqual(record_length_years(dowy, dates), 7.0)


if __name__ == '__main__':
    unittest.main()
