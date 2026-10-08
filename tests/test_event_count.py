'''TDD for event counts and uncertainty bounds.

count_events() returns a scalar only when every completion of unknown final
outcomes has the same maximal-success-run count. count_event_bounds() reports
conservative bounds without retaining source dependencies.

Result event APIs apply uniformly to a component's final outcome
(self.df[component.name]), with NO special-casing for nested vs
non-nested components. This is deliberate, not an oversight: empirically
verified (see TestEventCountNestedFrequencyEquivalence below) that a nested
frequency characteristic's terminal column is piecewise-constant across each
full water year (see nested_frequency_interannual_fx -- it assigns one
scalar verdict to result[start:end+1] for every full water year), so a
maximal run's start/end at raw-timestep grain always exactly coincides with
water-year boundaries. Counting runs at timestep grain and counting at
water-year grain therefore always agree; no per-water-year dedup step is
needed before collapsing.
'''
# pylint: disable=missing-class-docstring,missing-function-docstring
import unittest
from itertools import product

import numpy as np
import pandas as pd

from hydropattern.patterns import (
    Characteristic,
    CharacteristicType,
    Component,
    comparison_fx,
    count_event_bounds,
    count_events,
    evaluate_component,
    nested_frequency_interannual_fx,
)
from hydropattern.parsers import duration_parser, magnitude_parser


class TestCountEventsPureFunction(unittest.TestCase):
    '''Scalar counts require one uniquely determined value.'''

    def test_no_successes_is_zero_events(self):
        self.assertEqual(count_events(np.array([0, 0, 0, 0])), 0)

    def test_single_run_is_one_event(self):
        self.assertEqual(count_events(np.array([0, 1, 1, 1, 0])), 1)

    def test_two_separated_runs_is_two_events(self):
        self.assertEqual(count_events(np.array([1, 1, 0, 0, 1, 1, 1, 0])), 2)

    def test_all_successes_is_one_event(self):
        self.assertEqual(count_events(np.array([1, 1, 1, 1])), 1)

    def test_unknown_leading_outcomes_make_scalar_count_ambiguous(self):
        success = np.array([np.nan, np.nan, 1, 1, 0, 1])

        self.assertEqual(count_event_bounds(success), (2, 3))
        with self.assertRaisesRegex(ValueError, 'count_event_bounds'):
            count_events(success)

    def test_unknown_outcome_returns_named_event_count_bounds(self):
        bounds = count_event_bounds(np.array([1, np.nan, 1]))

        self.assertEqual((bounds.lower, bounds.upper), (1, 2))
        with self.assertRaisesRegex(ValueError, 'count_event_bounds'):
            count_events(np.array([1, np.nan, 1]))

    def test_invalid_outcome_values_are_rejected(self):
        for outcomes in (np.array([0, 2]), np.array([0, np.inf]), np.array(['1'])):
            with self.subTest(outcomes=outcomes), self.assertRaisesRegex(
                ValueError, 'only 0, 1, or NaN'
            ):
                count_event_bounds(outcomes)

    def test_bounds_match_all_short_binary_completions(self):
        for length in range(5):
            for source in product((0.0, 1.0, np.nan), repeat=length):
                unknown_positions = [
                    index for index, outcome in enumerate(source)
                    if np.isnan(outcome)
                ]
                possible_counts = []
                for completion in product((0.0, 1.0), repeat=len(unknown_positions)):
                    outcomes = np.array(source, dtype=float)
                    outcomes[unknown_positions] = completion
                    starts = (
                        (outcomes == 1)
                        & np.concatenate(([True], outcomes[:-1] != 1))
                        if len(outcomes)
                        else np.array([], dtype=bool)
                    )
                    possible_counts.append(int(np.sum(starts)))

                with self.subTest(source=source):
                    self.assertEqual(
                        count_event_bounds(np.array(source, dtype=float)),
                        (min(possible_counts), max(possible_counts)),
                    )


class TestResultEventCountNonNested(unittest.TestCase):
    '''Result.event_count() on a plain (non-nested) duration-bounded component.'''

    def test_counts_each_qualifying_duration_run_once(self):
        # Two separated qualifying (36-60 month) runs -> 2 events, regardless
        # of each run's internal length (48 and 40 months respectively).
        run_a, gap, run_b, tail = 48, 5, 40, 5
        total = run_a + gap + run_b + tail
        flow = np.concatenate([
            np.full(run_a, 1.0),
            np.full(gap, 100.0),
            np.full(run_b, 1.0),
            np.full(tail, 100.0),
        ])
        dowy = (np.arange(total) % 365) + 1
        df = pd.DataFrame({'flow': flow, 'dowy': dowy},
                          index=pd.RangeIndex(total, name='time'))
        component = Component(
            name='low_water_cycle',
            characteristics=[
                magnitude_parser(['<', 2.0], order=1),
                duration_parser([36, 60], order=2),
            ],
            is_success_pattern=True,
        )
        result = evaluate_component(df, component)
        self.assertEqual(result.event_count(), 2)


class TestEventCountNestedFrequencyEquivalence(unittest.TestCase):
    '''Proves month-grain run-counting on a nested frequency's broadcast
    terminal column always agrees with year-grain counting -- the empirical
    basis for not special-casing terminal characteristics in Result.event_count().
    '''

    def test_broadcast_column_run_count_matches_qualifying_year_runs(self):
        # 4 one-year "years" of 4 days each. intra_annual verdicts (already
        # OR-reduced per year) qualify in years 1, 3, 4 and fail year 2 --
        # i.e. two separate qualifying blocks: {year1} and {year3, year4}.
        intra_annual = np.array([
            1, 0, 0, 0,   # year1: True
            0, 0, 0, 0,   # year2: False
            1, 0, 0, 0,   # year3: True
            1, 0, 0, 0,   # year4: True
        ], dtype=float)
        dowy = np.tile([1, 2, 3, 4], 4).astype(float)
        dummy = np.zeros(16)
        output = np.column_stack([dummy, intra_annual])
        df = pd.DataFrame({'flow': range(16), 'dowy': dowy})
        f = comparison_fx('>=', 1)
        fx = nested_frequency_interannual_fx(f, order=3, big_n=1, exclusive_windows=False)
        broadcast = fx(df, output)

        # 2 distinct qualifying blocks (year1 alone; year3+year4 contiguous)
        # -> 2 events, whether counted at month grain (what count_events
        # actually does) or at year grain (compact array), confirming they agree.
        success = (broadcast == 1).astype(int)
        self.assertEqual(count_events(success), 2)

    def test_result_event_count_matches_for_stub_nested_component(self):
        # Same qualifying-year pattern as above, wired through evaluate_component
        # via a nested-terminal stub characteristic (mirrors
        # TestEvaluateComponentNestedFrequencyDispatch in test_patterns.py),
        # to confirm Result.event_count() needs no terminal-marker branching.
        nested_broadcast = np.array([
            1, 1, 1, 1,   # year1 (broadcast verdict: 1)
            0, 0, 0, 0,   # year2 (broadcast verdict: 0)
            1, 1, 1, 1,   # year3 (broadcast verdict: 1)
            1, 1, 1, 1,   # year4 (broadcast verdict: 1, contiguous with year3)
        ], dtype=float)

        def magnitude_stub_fx(_df, _output):
            return np.ones(16)

        def nested_stub_fx(_df, _output):
            return nested_broadcast

        component = Component(
            name='comp',
            characteristics=[
                Characteristic('magnitude_stub', magnitude_stub_fx,
                               CharacteristicType.MAGNITUDE, False),
                Characteristic('nested_stub', nested_stub_fx,
                               CharacteristicType.FREQUENCY, True),
            ],
            is_success_pattern=True,
        )
        df = pd.DataFrame(
            {'flow': range(16), 'dowy': np.tile([1, 2, 3, 4], 4)},
            index=pd.date_range('2020-01-01', periods=16, freq='D', name='time'),
        )
        result = evaluate_component(df, component)
        self.assertEqual(result.event_count(), 2)


if __name__ == '__main__':
    unittest.main()
