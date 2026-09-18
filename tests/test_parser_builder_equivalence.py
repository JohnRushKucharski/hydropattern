'''Characterization tests for issue #23.

Safety net for the upcoming parser/builder dedup work (#24-#28): proves the
direct-parser path (`parsers.py`'s `*_parser` functions, called with raw
metrics) and the spec-based path (`parse_request` -> `build_components`, via
`hydropattern.parsing`) produce behaviorally identical `Characteristic`
objects -- same `.name`, same evaluated fx output -- for one representative
component per characteristic type: timing, magnitude, rate_of_change,
duration, frequency, and nested_frequency.

No production code is modified by this issue.
'''
# pylint: disable=missing-class-docstring,missing-function-docstring
import unittest

import numpy as np
import pandas as pd

from hydropattern.parsers import (
    build_components,
    duration_parser,
    frequency_parser,
    magnitude_parser,
    nested_frequency_parser,
    parse_request,
    rate_of_change_parser,
    timing_parser,
)
from hydropattern.patterns import Component, evaluate_component


def _sample_timeseries() -> pd.DataFrame:
    '''12-row sample: 3 cycles of a 4-day 'water year' (dowy 1-4), with a
    flow pattern that produces non-trivial magnitude/duration/frequency
    diagnostics (bursts of highs separated by lows).
    '''
    flow = [10.0, 10.0, 2.0, 2.0, 10.0, 2.0, 2.0, 2.0, 10.0, 10.0, 10.0, 2.0]
    dowy = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    df = pd.DataFrame(
        {'flow': flow, 'dowy': dowy},
        index=pd.date_range('2020-01-01', periods=len(flow)),
    )
    df.index.name = 'time'
    return df


class ParserBuilderEquivalenceTestCase(unittest.TestCase):
    '''Shared assertion helper: direct-parser Component vs spec-path Component
    must produce identical characteristic names and identical evaluated
    output (including NaN placement) on the same sample timeseries.
    '''

    def _assert_equivalent(self, direct_characteristics, config):
        df = _sample_timeseries()
        direct_component = Component(
            name='comp', characteristics=direct_characteristics, is_success_pattern=True,
        )
        spec_component = build_components(parse_request(config))[0]

        direct_result = evaluate_component(df.copy(), direct_component)
        spec_result = evaluate_component(df.copy(), spec_component)

        self.assertEqual(
            [c.name for c in direct_component.characteristics],
            [c.name for c in spec_component.characteristics],
        )
        for direct_char, spec_char in zip(
            direct_component.characteristics, spec_component.characteristics
        ):
            np.testing.assert_array_equal(
                direct_result.df[direct_char.name].to_numpy(),
                spec_result.df[spec_char.name].to_numpy(),
                err_msg=f'fx output diverged for characteristic {direct_char.name!r}',
            )
        np.testing.assert_array_equal(
            direct_result.df['comp'].to_numpy(),
            spec_result.df['comp'].to_numpy(),
            err_msg='component-level success column diverged',
        )


class TestTimingEquivalence(ParserBuilderEquivalenceTestCase):

    def test_timing_direct_vs_spec_path(self):
        self._assert_equivalent(
            direct_characteristics=[timing_parser([2, 3], order=1)],
            config={'comp': {'timing': [2, 3]}},
        )


class TestMagnitudeEquivalence(ParserBuilderEquivalenceTestCase):

    def test_magnitude_direct_vs_spec_path(self):
        self._assert_equivalent(
            direct_characteristics=[magnitude_parser(['>', 5.0], order=1)],
            config={'comp': {'magnitude': ['>', 5.0]}},
        )


class TestRateOfChangeEquivalence(ParserBuilderEquivalenceTestCase):

    def test_rate_of_change_direct_vs_spec_path(self):
        self._assert_equivalent(
            direct_characteristics=[rate_of_change_parser(['>', 0.1], order=1)],
            config={'comp': {'rate_of_change': ['>', 0.1]}},
        )


class TestDurationEquivalence(ParserBuilderEquivalenceTestCase):

    def test_duration_direct_vs_spec_path(self):
        # duration must follow a preceding order-1 characteristic (magnitude here).
        self._assert_equivalent(
            direct_characteristics=[
                magnitude_parser(['>', 5.0], order=1),
                duration_parser(['>', 1], order=2),
            ],
            config={'comp': {'magnitude': ['>', 5.0], 'duration': ['>', 1]}},
        )


class TestFrequencyEquivalence(ParserBuilderEquivalenceTestCase):

    def test_frequency_direct_vs_spec_path(self):
        # frequency must be last and cannot be order 1; precede it with magnitude.
        self._assert_equivalent(
            direct_characteristics=[
                magnitude_parser(['>', 5.0], order=1),
                frequency_parser(['>', 1, 3], order=2),
            ],
            config={'comp': {'magnitude': ['>', 5.0], 'frequency': ['>', 1, 3]}},
        )


class TestNestedFrequencyEquivalence(ParserBuilderEquivalenceTestCase):

    def test_nested_frequency_direct_vs_spec_path(self):
        # nested frequency (base + nested pattern) expands to two characteristics.
        nested_direct = nested_frequency_parser([['>', 0.5], ['>', 1, 2]], order=2)
        self._assert_equivalent(
            direct_characteristics=[magnitude_parser(['>', 5.0], order=1), *nested_direct],
            config={'comp': {'magnitude': ['>', 5.0], 'frequency': [['>', 0.5], ['>', 1, 2]]}},
        )


if __name__ == '__main__':
    unittest.main()
