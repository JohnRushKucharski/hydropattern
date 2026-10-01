'''TDD for Phase 4: exceeding-events identification utility.

This is a standalone, un-wired utility -- deliberately NOT plugged into
Result, event_count(), event_rate(), frequency_table(), or any CLI/report
output. It exists so a future "what's driving success/failure" feature can
identify which qualifying-but-out-of-bounds runs (e.g. a low-water spell
that lasted 68 months against a 36-60 month duration bound) got excluded,
and why. That future wiring is explicitly out of scope for this change.

find_runs() extracts duration_fx's own run-detection loop into a shared,
reusable primitive (same NaN-breaks-a-run semantics as mark_events) --
duration_fx itself is refactored to call it, so there is exactly one
run-detection implementation in the codebase, not two. Regression-verified
against duration_fx's own existing test suite (tests/test_patterns.py),
unchanged.

find_exceeding_events() reuses find_runs() to report every maximal run
whose length falls outside a [min_duration, max_duration] bound -- i.e. the
runs a duration_parser's between-form would have excluded, whether for
being too short or too long.
'''
# pylint: disable=missing-class-docstring,missing-function-docstring
import unittest

import numpy as np

from hydropattern.patterns import find_exceeding_events, find_runs


class TestFindRuns(unittest.TestCase):
    '''find_runs: generic maximal-run (start, end) extraction.'''

    def test_no_runs(self):
        self.assertEqual(find_runs(np.array([0, 0, 0])), [])

    def test_single_run_mid_array(self):
        self.assertEqual(find_runs(np.array([0, 1, 1, 1, 0])), [(1, 3)])

    def test_run_reaching_end_of_array(self):
        self.assertEqual(find_runs(np.array([0, 1, 1])), [(1, 2)])

    def test_run_from_start_of_array(self):
        self.assertEqual(find_runs(np.array([1, 1, 0])), [(0, 1)])

    def test_all_ones_is_one_run(self):
        self.assertEqual(find_runs(np.array([1, 1, 1])), [(0, 2)])

    def test_two_separated_runs(self):
        self.assertEqual(find_runs(np.array([1, 1, 0, 0, 1, 1, 1, 0])), [(0, 1), (4, 6)])

    def test_nan_breaks_a_run(self):
        self.assertEqual(find_runs(np.array([1, 1, np.nan, 1])), [(0, 1), (3, 3)])


class TestFindExceedingEvents(unittest.TestCase):
    '''find_exceeding_events: runs outside [min_duration, max_duration].'''

    def test_run_exceeding_max_is_reported(self):
        # 5-month run against max_duration=3.
        eligible = np.array([0, 1, 1, 1, 1, 1, 0])
        result = find_exceeding_events(eligible, max_duration=3)
        self.assertEqual(result, [(1, 5, 5)])

    def test_run_within_bounds_is_not_reported(self):
        eligible = np.array([0, 1, 1, 1, 0])
        self.assertEqual(find_exceeding_events(eligible, min_duration=2, max_duration=5), [])

    def test_run_below_min_is_reported(self):
        eligible = np.array([0, 1, 0, 1, 1, 1, 0])
        result = find_exceeding_events(eligible, min_duration=2)
        self.assertEqual(result, [(1, 1, 1)])

    def test_both_bounds_report_both_directions(self):
        # run1: length 1 (too short, min=2); run2: length 5 (too long, max=3).
        eligible = np.array([1, 0, 1, 1, 1, 1, 1])
        result = find_exceeding_events(eligible, min_duration=2, max_duration=3)
        self.assertEqual(result, [(0, 0, 1), (2, 6, 5)])

    def test_raises_when_no_bounds_given(self):
        with self.assertRaises(ValueError):
            find_exceeding_events(np.array([1, 1, 1]))

    def test_nan_breaks_a_run_before_length_check(self):
        eligible = np.array([1, 1, np.nan, 1, 1, 1, 1])
        result = find_exceeding_events(eligible, max_duration=2)
        self.assertEqual(result, [(3, 6, 4)])


if __name__ == '__main__':
    unittest.main()
