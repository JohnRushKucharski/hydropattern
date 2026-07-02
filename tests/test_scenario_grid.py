'''Tests for the scenario_grid module.'''
import unittest

import numpy as np

from hydropattern.errors import HydropatternError, PlotErrorCode
from hydropattern.scenario_grid import (
    build_grid,
    is_scenario_grid,
    parse_scenario_name,
    require_scenario_grid,
)


class TestParseScenarioName(unittest.TestCase):
    '''Tests for parse_scenario_name.'''

    def test_parses_precip_and_temp_delta(self):
        '''`_0_1.5` -> precip_delta=0.0, temp_delta=1.5.'''
        self.assertEqual(parse_scenario_name('_0_1.5'), (0.0, 1.5))

    def test_returns_none_for_non_matching_name(self):
        '''A name that doesn't match the `_x_y` convention returns None.'''
        self.assertIsNone(parse_scenario_name('scenario_a'))
        self.assertIsNone(parse_scenario_name('flow'))
        self.assertIsNone(parse_scenario_name('_only_one_part_missing'))


class TestIsScenarioGrid(unittest.TestCase):
    '''Tests for is_scenario_grid.'''

    def test_true_when_names_form_a_grid(self):
        '''Multiple distinct precip/temp deltas across matching names -> a real grid.'''
        names = ['_0_0', '_0_1.5', '_5_0', '_5_1.5']
        self.assertTrue(is_scenario_grid(names))

    def test_false_when_a_name_does_not_match(self):
        '''Any non-matching name disqualifies the set as a scenario grid.'''
        names = ['_0_0', '_0_1.5', 'flow']
        self.assertFalse(is_scenario_grid(names))

    def test_false_for_single_scenario(self):
        '''A single scenario is not a grid, even if its name matches the convention.'''
        self.assertFalse(is_scenario_grid(['_0_0']))


class TestBuildGrid(unittest.TestCase):
    '''Tests for build_grid.'''

    def test_builds_rectangular_grid_from_full_combos(self):
        '''All precip/temp combos present -> a fully populated rectangular grid.

        xs = sorted unique precip deltas, ys = sorted unique temp deltas,
        zs[row=temp, col=precip] = metric value for that scenario.
        '''
        scenario_names = ['_0_0', '_0_1.5', '_5_0', '_5_1.5']
        metric_values = {'_0_0': 1.0, '_0_1.5': 2.0, '_5_0': 3.0, '_5_1.5': 4.0}

        xs, ys, zs = build_grid(scenario_names, metric_values)

        np.testing.assert_array_equal(xs, [0.0, 5.0])
        np.testing.assert_array_equal(ys, [0.0, 1.5])
        np.testing.assert_array_equal(zs, [[1.0, 3.0], [2.0, 4.0]])

    def test_missing_combo_is_nan(self):
        '''A precip/temp combo with no scenario -> NaN in that grid cell.'''
        scenario_names = ['_0_0', '_0_1.5', '_5_1.5']  # missing _5_0
        metric_values = {'_0_0': 1.0, '_0_1.5': 2.0, '_5_1.5': 4.0}

        xs, ys, zs = build_grid(scenario_names, metric_values)

        np.testing.assert_array_equal(xs, [0.0, 5.0])
        np.testing.assert_array_equal(ys, [0.0, 1.5])
        self.assertTrue(np.isnan(zs[0, 1]))  # temp=0, precip=5 -> missing
        self.assertEqual(zs[0, 0], 1.0)
        self.assertEqual(zs[1, 0], 2.0)
        self.assertEqual(zs[1, 1], 4.0)


class TestRequireScenarioGrid(unittest.TestCase):
    '''Tests for require_scenario_grid.'''

    def test_raises_plot_error_for_non_grid_names(self):
        '''Non-grid scenario names raise HydropatternError with PLOT_INVALID_SCENARIO_GRID.'''
        with self.assertRaises(HydropatternError) as context:
            require_scenario_grid(['flow'])

        self.assertEqual(context.exception.envelope.code, PlotErrorCode.INVALID_SCENARIO_GRID)
        self.assertEqual(context.exception.envelope.source, 'plot')

    def test_does_not_raise_for_valid_grid_names(self):
        '''A valid scenario grid does not raise.'''
        require_scenario_grid(['_0_0', '_0_1.5', '_5_0', '_5_1.5'])
