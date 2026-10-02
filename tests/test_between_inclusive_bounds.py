'''TDD for making between-form comparisons inclusive at both bounds
(min <= n <= max), consistently across magnitude, rate_of_change, and
duration characteristics, and fixing the duration between-form bug
(currently mis-built as `comparison_fx('<', min, '>', max)`, which
collapses to `n > max` instead of `min <= n <= max`).

See docs/agents session plan: Phase 1 (between-bounds inclusivity +
duration bug fix).
'''
# pylint: disable=missing-class-docstring,missing-function-docstring
import unittest

import numpy as np
import pandas as pd

from hydropattern.parsers import (
    between_parser,
    duration_parser,
    magnitude_parser,
    rate_of_change_parser,
    timing_parser,
)
from hydropattern.patterns.core import evaluate_component
from hydropattern.patterns import Component


class TestBetweenParserDefaultsInclusive(unittest.TestCase):
    '''between_parser() called with no explicit `inclusive` override.'''

    def test_boundaries_are_included_by_default(self):
        f = between_parser([1, 5])
        self.assertTrue(f(1))
        self.assertTrue(f(5))
        self.assertTrue(f(3))
        self.assertFalse(f(0))
        self.assertFalse(f(6))

    def test_explicit_inclusive_false_still_excludes_boundaries(self):
        '''The exclusive path (inclusive=False) is still reachable/functional;
        no caller uses it after this change, but the parameter itself must
        keep working for any direct/advanced use of between_parser.
        '''
        f = between_parser([1, 5], inclusive=False)
        self.assertFalse(f(1))
        self.assertFalse(f(5))
        self.assertTrue(f(3))
        self.assertFalse(f(0))
        self.assertFalse(f(6))


class TestMagnitudeParserBetweenIsInclusive(unittest.TestCase):
    '''magnitude_parser between-form must include both boundary values.'''

    def test_boundaries_included(self):
        char = magnitude_parser([1.0, 5.0], order=1)
        df = pd.DataFrame({'flow': [1.0, 5.0, 0.5, 5.5, 3.0]})
        result = char.fx(df, None)
        self.assertTrue(np.all(result == np.array([1, 1, 0, 0, 1])))


class TestRateOfChangeParserBetweenIsInclusive(unittest.TestCase):
    '''rate_of_change_parser between-form must include both boundary values.'''

    def test_boundaries_included(self):
        # rate of change z_t = y_t / y_(t-1); between [1.0, 2.0] inclusive.
        char = rate_of_change_parser([1.0, 2.0], order=1)
        df = pd.DataFrame({'flow': [1.0, 1.0, 2.0, 3.0, 1.5]})
        # z = [nan, 1.0, 2.0, 1.5, 0.5]
        result = char.fx(df)
        self.assertTrue(np.all(result == np.array([0, 1, 1, 1, 0])))


class TestDurationParserBetweenIsFixedAndInclusive(unittest.TestCase):
    '''duration_parser between-form: fixes the `n > max`-only bug and
    makes both bounds inclusive (min <= n <= max).
    '''

    def test_comparison_function_boundaries(self):
        char = duration_parser([36, 60], order=2)
        # patterns.Characteristic doesn't expose the raw comparison fx
        # directly; exercise it end-to-end through evaluate_component
        # instead (see test_duration_between_end_to_end below) plus a
        # direct check here using the same construction magnitude/rate_of_change
        # use, to pin the fx boundaries precisely.
        f = between_parser([36, 60])
        for n, expected in [(35, False), (36, True), (48, True),
                             (60, True), (61, False), (102, False)]:
            with self.subTest(n=n):
                self.assertEqual(f(n), expected)
        self.assertEqual(char.name, 'duration_36-60')

    def test_duration_between_end_to_end(self):
        '''A continuous magnitude-success run of exactly 48 months (inside
        [36, 60]) must be marked a success for its full length; a run of
        102 months (outside/above 60) must not qualify at all.
        '''
        n_qualifying = 48
        n_exceeding = 102
        gap = 5
        total = n_qualifying + gap + n_exceeding + gap
        flow = np.concatenate([
            np.full(n_qualifying, 1.0),   # qualifying run: 48 months
            np.full(gap, 100.0),          # break
            np.full(n_exceeding, 1.0),    # exceeding run: 102 months
            np.full(gap, 100.0),          # break
        ])
        # Build a valid day-of-water-year column (1..365 repeating) matching validate_timeseries.
        dowy = (np.arange(total) % 365) + 1
        index = pd.RangeIndex(total, name='time')
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
        success = result.df['low_water_cycle'].values

        # qualifying run (rows 0:48) -> all 1
        self.assertTrue(np.all(success[0:n_qualifying] == 1))
        # exceeding run -> all 0 (does not qualify)
        start = n_qualifying + gap
        end = start + n_exceeding
        self.assertTrue(np.all(success[start:end] == 0))


class TestTimingBetweenWasAlreadyInclusiveUnaffected(unittest.TestCase):
    '''timing's between form builds its own comparison_fx('<=', ..., '<=', ...)
    directly (never routes through between_parser) and was already inclusive
    at both bounds; confirms this change didn't need to (and didn't) touch it.
    '''

    def test_boundaries_included(self):
        char = timing_parser([100, 110], order=1)
        df = pd.DataFrame({'flow': [1.0] * 5,
                            'dowy': [99, 100, 105, 110, 111]},
                          index=pd.DatetimeIndex(
                              ['2021-04-09', '2021-04-10', '2021-04-15',
                               '2021-04-20', '2021-04-21'],
                              name='time',
                          ))
        result = char.fx(df, None)
        self.assertTrue(np.all(result == np.array([0, 1, 1, 1, 0])))


if __name__ == '__main__':
    unittest.main()
