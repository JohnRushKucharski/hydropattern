'''TDD for Phase 3: event_rate.

event_rate(events, years) = events / years -- a plain descriptive rate, not a
recurrence-interval/Poisson-probability claim (Great Lakes water-level
records show multi-decadal persistence/clustering, so between-event
independence should not be assumed; see docs/adr and prior review notes).

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
again with NO is_nested special-casing: T is a property of the record's own
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

    def test_counts_full_water_years(self):
        # 3 full "water years" of 5 rows each (dowy resets to 1 at each start).
        dowy = np.tile(np.arange(1, 6), 3).astype(float)
        self.assertEqual(record_length_years(dowy), 3.0)

    def test_leading_partial_water_year_is_excluded(self):
        # Matches identify_full_water_years' own documented convention: data
        # before the first dowy==1 (here, a stub "tail end" of a prior,
        # unobserved water year) does not count as a year.
        leading_partial = np.array([4.0, 5.0])  # before first dowy==1
        full_years = np.tile(np.arange(1, 6), 2).astype(float)  # 2 full years
        dowy = np.concatenate([leading_partial, full_years])
        self.assertEqual(record_length_years(dowy), 2.0)

    def test_trailing_partial_water_year_still_counts_as_one(self):
        # Also matches the existing (asymmetric) convention: a trailing
        # partial year -- data after the last full dowy==1 restart -- still
        # counts as a full year (identify_full_water_years always runs the
        # final entry to the end of the array). Not re-litigated here, only
        # mirrored.
        full_years = np.tile(np.arange(1, 6), 2).astype(float)  # 2 full years
        trailing_partial = np.array([1.0, 2.0])  # 3rd year, incomplete
        dowy = np.concatenate([full_years, trailing_partial])
        self.assertEqual(record_length_years(dowy), 3.0)

    def test_raises_on_no_full_water_years(self):
        # dowy never hits 1 -> identify_full_water_years finds no years.
        with self.assertRaises(ValueError):
            record_length_years(np.array([4.0, 5.0, 6.0]))

    def test_raises_on_empty_dowy(self):
        with self.assertRaises(ValueError):
            record_length_years(np.array([]))


class TestResultEventRate(unittest.TestCase):
    '''Result.event_rate() combines event_count() with record_length_years().'''

    def test_matches_manual_division(self):
        run_a, gap, run_b, tail = 48, 5, 40, 5
        total = run_a + gap + run_b + tail  # 98 == 14 * 7 water years below
        flow = np.concatenate([
            np.full(run_a, 1.0),
            np.full(gap, 100.0),
            np.full(run_b, 1.0),
            np.full(tail, 100.0),
        ])
        dowy = np.tile(np.arange(1, 15), 7).astype(float)  # 7 full water years
        index = pd.date_range('1970-01-01', periods=total, freq='MS', name='time')
        df = pd.DataFrame({'flow': flow, 'dowy': dowy}, index=index)
        component = Component(
            name='low_water_cycle',
            characteristics=[
                magnitude_parser(['<', 2.0], order=1),
                duration_parser([36, 60], order=2),
            ],
            is_success_pattern=True,
        )
        result = evaluate_component(df, component)
        expected = result.event_count() / record_length_years(dowy)
        self.assertAlmostEqual(result.event_rate(), expected)
        self.assertEqual(result.event_count(), 2)
        self.assertEqual(record_length_years(dowy), 7.0)


if __name__ == '__main__':
    unittest.main()
