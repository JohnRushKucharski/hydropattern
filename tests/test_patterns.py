'''Tests for the patterns module.'''
# test coverage backlog: is_order_1, frequency_fx, evaluate_patterns
# pylint: disable=too-many-public-methods
import unittest

import numpy as np
import pandas as pd

from hydropattern.patterns import (
    Characteristic,
    CharacteristicType,
    Component,
    comparison_fx,
    duration_fx,
    evaluate_component,
    frequency_fx,
    identify_full_water_years,
    is_dowy_timeseries,
    magnitude_fx,
    mark_events,
    moving_average,
    nested_frequency_interannual_fx,
    nested_frequency_intra_annual_fx,
    or_reduce_per_water_year,
    rate_of_change_fx,
    sliding_window_count,
    timing_fx,
    water_year_probability_ratio,
    windowed_count_per_water_year,
)

# used in some simple characteristic function tests.
df = pd.DataFrame({'col1': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                   'col2': [1, 2, 3, 4, 5, 6]})
# df =
#     col1  col2
# 0   10.0     1
# 1   20.0     2
# 2   30.0     3
# 3   40.0     4
# 4   50.0     5
# 5   60.0     6


class TestPatterns(unittest.TestCase):
    '''Tests for the patterns module.'''
    #region: comparison_fx tests
    #region: single symbol
    def test_comparison_fx_lt(self):
        '''Test comparison_fx function.'''
        fx = comparison_fx('<', 5, None, None)
        self.assertTrue(fx(4))
        self.assertFalse(fx(5))
        self.assertFalse(fx(6))

    def test_comparison_fx_le(self):
        '''Test comparison_fx function.'''
        fx = comparison_fx('<=', 5, None, None)
        self.assertTrue(fx(4))
        self.assertTrue(fx(5))
        self.assertFalse(fx(6))

    def test_comparison_fx_gt(self):
        '''Test comparison_fx function.'''
        fx = comparison_fx('>', 5, None, None)
        self.assertFalse(fx(4))
        self.assertFalse(fx(5))
        self.assertTrue(fx(6))

    def test_comparison_fx_ge(self):
        '''Test comparison_fx function.'''
        fx = comparison_fx('>=', 5, None, None)
        self.assertFalse(fx(4))
        self.assertTrue(fx(5))
        self.assertTrue(fx(6))

    def test_comparison_fx_eq(self):
        '''Test comparison_fx function.'''
        fx = comparison_fx('=', 5, None, None)
        self.assertFalse(fx(4))
        self.assertTrue(fx(5))

    def test_comparison_fx_ne(self):
        '''Test comparison_fx function.'''
        fx = comparison_fx('!=', 5, None, None)
        self.assertTrue(fx(4))
        self.assertFalse(fx(5))
    #endregion

    #region: two bounds
    def test_comparison_fx_btwn(self):
        '''Test comparison_fx function.'''
        fx = comparison_fx('<', 3, '<', 5)
        self.assertTrue(fx(4))
        self.assertFalse(fx(5))
        self.assertFalse(fx(6))

    def test_comparison_fx_btwneq(self):
        '''Test comparison_fx function.'''
        fx = comparison_fx('<=', 3, '<=', 5)
        self.assertTrue(fx(4))
        self.assertTrue(fx(5))
        self.assertFalse(fx(6))

    def test_comparison_fx_btwnop(self):
        '''Test comparison_fx function.'''
        fx = comparison_fx('>', 5, '>', 3)
        self.assertTrue(fx(4))
        self.assertFalse(fx(5))
        self.assertFalse(fx(6))

    def test_comparison_fx_btwnopeq(self):
        '''Test comparison_fx function.'''
        fx = comparison_fx('>=', 5, '>=', 3)
        self.assertTrue(fx(4))
        self.assertTrue(fx(5))
        self.assertFalse(fx(6))
    #endregion
    #endregion

    #region: characteristics
    #region: moving_average tests
    def test_moving_average(self):
        '''Test moving_average function.'''
        i = np.array([1, 2, 3, 4, 5, 6])
        self.assertTrue(np.array_equal(
            moving_average(i, 3),
            np.array([np.nan, np.nan, 2., 3., 4., 5.]), equal_nan=True))

    def test_moving_average_period1_returns_input(self):
        '''Test moving_average function.'''
        i = np.array([1, 2, 3, 4, 5, 6])
        self.assertTrue(np.array_equal(
            moving_average(i, 1),
            i, equal_nan=True))

    def test_moving_average_min_periods(self):
        '''Test moving_average function.'''
        i = np.array([1, 2, 3, 4, 5, 6])
        self.assertTrue(np.array_equal(
            moving_average(i, 3, min_periods=1),
            np.array([1, 1.5, 2., 3., 4., 5.]), equal_nan=True))

    def test_moving_average_min_periods2(self):
        '''Test moving_average function.'''
        i = np.array([1, 2, 3, 4, 5, 6])
        self.assertTrue(np.array_equal(
            moving_average(i, 3, min_periods=2),
            np.array([np.nan, 1.5, 2., 3., 4., 5.]), equal_nan=True))
    #endregion

    #region: is_dowy_timeseries tests
    def test_is_dowy_timeseries(self):
        '''Test is_dowy_timeseries function.'''
        self.assertTrue(is_dowy_timeseries([1, 2, 3.0, 4]))

    def test_is_dowy_timeseries_false_for_nonint(self):
        '''Test is_dowy_timeseries function.'''
        self.assertFalse(is_dowy_timeseries([1, 2, 3.5, 4]))
    #endregion

    #region: timing_fx tests
    def test_timing_fx(self):
        '''Test timing_fx function.'''
        fx = timing_fx(comparison_fx('<', 3, '<', 6))
        self.assertTrue(np.all(fx(df) == np.array([0, 0, 0, 1, 1, 0])))
    #endregion

    #region: magnitude_fx tests
    def test_magnitude_fx(self):
        '''Test magnitude_fx function.'''
        fx = magnitude_fx(comparison_fx('>', 50.0, None, None))
        self.assertTrue(np.all(fx(df) == np.array([0, 0, 0, 0, 0, 1])))
    #endregion

    #region: duration_fx tests
    def test_duration_fx_whole_order3(self):
        '''Test duration_fx function.'''
        order = 3
        o = np.ones(shape=(len(df), order-1))
        fx = duration_fx(comparison_fx('>', 5, None, None), order)
        self.assertTrue(np.all(fx(df, o) == np.ones(len(df))))
    def test_duration_fx_end_order3(self):
        '''Test duration_fx function.'''
        order = 3
        o = np.ones(shape=(len(df), order-1))
        o[2,:] = 0 # breaks up the streak of 1s
        fx = duration_fx(comparison_fx('>=', 3, None, None), order)
        self.assertTrue(np.all(fx(df, o) == np.array([0, 0, 0, 1, 1, 1])))
    def test_duration_fx_mid_order3(self):
        '''Test duration_fx function.'''
        order = 3
        o = np.zeros(shape=(len(df), order-1))
        o[2:5,:] = 1 # breaks up the streak of 1s
        fx = duration_fx(comparison_fx('>=', 3, None, None), order)
        self.assertTrue(np.all(fx(df, o) == np.array([0, 0, 1, 1, 1, 0])))
    def test_duration_fx_startstop_order3(self):
        '''Test duration_fx function.'''
        order = 3
        o = np.ones(shape=(len(df), order-1))
        o[0,:] = 0 # breaks up the streak of 1s
        o[3,:] = 0 # breaks up the streak of 1s
        fx = duration_fx(comparison_fx('>', 1, None, None), order)
        self.assertTrue(np.all(fx(df, o) == np.array([0, 1, 1, 0, 1, 1])))
    def test_duration_fx_start_ordermismatch(self):
        '''Test duration_fx function.'''
        order = 3
        # array with 6 rows and 4 columns.
        o = np.zeros(shape=(len(df), 4))
        # 1s in first 5 rows last 2 columns.
        o[0:5,2:4] = 1 # add 1s in columns that matter
        # o = [
        #     [0, 0, 1, 1],
        #     [0, 0, 1, 1],
        #     [0, 0, 1, 1],
        #     [0, 0, 1, 1],
        #     [0, 0, 1, 1],
        #     [0, 0, 0, 0]
        # ]
        # duration_fx(gt(x, 1)~f(x>1), order=3) -> fx
        fx = duration_fx(comparison_fx('>', 1, None, None), order)
        # fx(dataframe, output_array) -> 1D array (this case with 6 rows).
        self.assertTrue(np.all(fx(df, o) == np.array([0, 0, 0, 0, 0, 0])))

    def test_duration_fx_start_order_not_mismatched(self):
        '''Test duration check performed on start of output row arrays.'''
        order = 3
        o = np.zeros(shape=(len(df), 4))
        o[0:5,0:3] = 1 # add 1s in columns that matter
        # o = [
        #     [1, 1, 0, 0],
        #     [1, 1, 0, 0],
        #     [1, 1, 0, 0],
        #     [1, 1, 0, 0],
        #     [1, 1, 0, 0],
        #     [0, 0, 0, 0]
        # ]
        # duration_fx(gt(x, 1)~f(x>1), order=3) -> fx
        fx = duration_fx(comparison_fx('>', 1, None, None), order)
        self.assertTrue(np.all(fx(df, o) == np.array([1, 1, 1, 1, 1, 0])))
    #endregion

    #region: rate_of_change_fx tests
    def test_rate_of_change_fx_defaults(self):
        '''Test rate_of_change_fx function.'''
        # order=1, ma_periods=1, look_back=1, minimum=0.0

        # GT increasing rate of change
        fx = rate_of_change_fx(comparison_fx('>', 1, None, None))
        df_ = pd.DataFrame({'col1': [0, 1, 2, 1] ,'col2': [1.0, 2.0, 3.0, 4.0]})
        # rate of change is [nan, 1/0, 2/1, 1/2] -> [nan, nan, 2.0, 0.5]
        self.assertTrue(np.all(fx(df_) == np.array([0, 0, 1, 0])))

        # LT decreasing rate of change
        fx = rate_of_change_fx(comparison_fx('<', 1, None, None))
        self.assertTrue(np.all(fx(df_) == np.array([0, 0, 0, 1])))

        # BETWEEN decreasign rate of change
        fx = rate_of_change_fx(comparison_fx('<', 0.25, '<', 0.75))
        self.assertTrue(np.all(fx(df_) == np.array([0, 0, 0, 1])))

    def test_rate_of_change_fx_with_ma_periods(self):
        '''Test rate_of_change_fx with moving average periods.'''
        # ma_periods=2, look_back=1, minimum=0.0
        fx = rate_of_change_fx(comparison_fx('>', 1.5, None, None), ma_periods=2)
        df_ = pd.DataFrame({'col1': [1.0, 3.0, 5.0, 7.0, 3.0]})
        # moving average (2): [nan, 2.0, 4.0, 6.0, 5.0]
        # rate of change: [nan, nan, 4.0/2.0, 6.0/4.0, 5.0/6.0] = [nan, nan, 2.0, 1.5, 0.833...]
        # comparison > 1.5: [0, 0, 1, 0, 0]
        self.assertTrue(np.all(fx(df_) == np.array([0, 0, 1, 0, 0])))

    def test_rate_of_change_fx_with_look_back(self):
        '''Test rate_of_change_fx with non-default look_back period.'''
        # ma_periods=1, look_back=2, minimum=0.0
        fx = rate_of_change_fx(comparison_fx('>', 2.0, None, None), look_back=2)
        df_ = pd.DataFrame({'col1': [1.0, 2.0, 4.0, 10.0, 12.0, 15.0]})
        # rate of change look_back=2: [nan, nan, 4.0/1.0, 10.0/2.0, 12.0/4.0, 15.0/10.0]
        #                            = [nan, nan, 4.0, 5.0, 3.0, 1.5]
        # comparison > 2.0: [0, 0, 1, 1, 1, 0]
        self.assertTrue(np.all(fx(df_) == np.array([0, 0, 1, 1, 1, 0])))

    def test_rate_of_change_fx_with_minimum(self):
        '''Test rate_of_change_fx with non-default minimum threshold.'''
        # ma_periods=1, look_back=1, minimum=1.0
        fx = rate_of_change_fx(comparison_fx('>', 1.5, None, None), minimum=1.0)
        df_ = pd.DataFrame({'col1': [0.5, 2.0, 4.0, 6.0]})
        # Values <= 1.0 become nan: [nan, 2.0, 4.0, 6.0]
        # rate of change: [nan, nan, 4.0/2.0, 6.0/4.0] = [nan, nan, 2.0, 1.5]
        # comparison > 1.5: [0, 0, 1, 0]
        self.assertTrue(np.all(fx(df_) == np.array([0, 0, 1, 0])))

    def test_rate_of_change_fx_with_ma_and_lookback(self):
        '''Test rate_of_change_fx with both ma_periods and look_back non-default.'''
        # ma_periods=2, look_back=2, minimum=0.0
        fx = rate_of_change_fx(comparison_fx('>', 2.0, None, None), ma_periods=2, look_back=2)
        df_ = pd.DataFrame({'col1': [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]})
        # moving average (2): [nan, 3.0, 5.0, 7.0, 9.0, 11.0]
        # rate of change look_back=2: [nan, nan, nan, 7.0/3.0, 9.0/5.0, 11.0/7.0]
        #                            = [nan, nan, nan, 2.333..., 1.8, 1.571...]
        # comparison > 2.0: [0, 0, 0, 1, 0, 0]
        self.assertTrue(np.all(fx(df_) == np.array([0, 0, 0, 1, 0, 0])))

    def test_rate_of_change_fx_order2(self):
        '''Test rate_of_change_fx with order=2 (second characteristic in sequence).'''
        order = 2
        fx = rate_of_change_fx(comparison_fx('>', 1.5, None, None), order=order)
        df_ = pd.DataFrame({'col1': [1.0, 2.0, 4.0, 6.0, 3.0, 2.0]})
        # rate of change: [nan, 2.0/1.0, 4.0/2.0, 6.0/4.0, 3.0/6.0, 2.0/3.0]
        #               = [nan, 2.0, 2.0, 1.5, 0.5, 0.667]
        # comparison > 1.5: [0, 1, 1, 0, 0, 0] (without order check)

        # output array with all 1s (previous characteristic passed for all rows)
        o = np.ones(shape=(len(df_), order-1))
        result = fx(df_, o)
        # With order check, only rows where output[:, 0:order-1] are all 1s
        self.assertTrue(np.all(result == np.array([0, 1, 1, 0, 0, 0])))

    def test_rate_of_change_fx_order3_partial_eligibility(self):
        '''Test rate_of_change_fx with order=3 and partial eligibility.'''
        order = 3
        fx = rate_of_change_fx(comparison_fx('<', 1.0, None, None), order=order, look_back=1)
        df_ = pd.DataFrame({'col1': [4.0, 3.0, 2.0, 6.0, 5.0, 4.0]})
        # rate of change: [nan, 3.0/4.0, 2.0/3.0, 6.0/2.0, 5.0/6.0, 4.0/5.0]
        #               = [nan, 0.75, 0.667, 3.0, 0.833, 0.8]
        # comparison < 1.0: [0, 1, 1, 0, 1, 1] (without order check)

        # output array where only some rows pass previous characteristics
        o = np.zeros(shape=(len(df_), order-1))
        o[1:4, :] = 1  # rows 1, 2, 3 pass previous characteristics
        result = fx(df_, o)
        # Only rows where output columns 0:order-1 are all 1s AND comparison passes
        # Rows 1, 2, 3 have all 1s in output; comparison passes for rows 1, 2
        self.assertTrue(np.all(result == np.array([0, 1, 1, 0, 0, 0])))
    #endregion
    #endregion

class TestMarkEvents(unittest.TestCase):
    '''Tests for the mark_events frequency event-detection/marking engine.'''

    def test_event_bool_false_returns_raw_unchanged(self):
        '''event_bool=False: timestep-level, every trial in a run stays marked.'''
        raw = np.array([np.nan, np.nan, 1, 1, 0, 1])
        result = mark_events(raw, event_bool=False)
        np.testing.assert_array_equal(result, raw)

    def test_single_run_collapses_to_last_trial(self):
        '''A single maximal run of 1s collapses to a 1 at its last trial.'''
        raw = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        result = mark_events(raw, event_bool=True)
        np.testing.assert_array_equal(result, np.array([0, 0, 0, 1, 0]))

    def test_run_ending_at_end_of_array(self):
        '''A run that continues through the last trial marks the final trial.'''
        raw = np.array([0.0, 1.0, 1.0, 1.0])
        result = mark_events(raw, event_bool=True)
        np.testing.assert_array_equal(result, np.array([0, 0, 0, 1]))

    def test_single_trial_run_stays_marked(self):
        '''A run of length 1 is already correctly marked at its own trial.'''
        raw = np.array([0.0, 1.0, 0.0])
        result = mark_events(raw, event_bool=True)
        np.testing.assert_array_equal(result, np.array([0, 1, 0]))

    def test_multiple_separate_runs_each_collapse(self):
        '''Each maximal run collapses independently to its own last trial.'''
        raw = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        result = mark_events(raw, event_bool=True)
        np.testing.assert_array_equal(result, np.array([0, 1, 0, 0, 0, 1, 0]))

    def test_leading_nan_preserved_and_does_not_bridge_runs(self):
        '''NaN (insufficient history) is preserved and starts a fresh run boundary.'''
        raw = np.array([np.nan, np.nan, 1.0, 1.0, 0.0, 0.0])
        result = mark_events(raw, event_bool=True)
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))
        np.testing.assert_array_equal(result[2:], np.array([0, 1, 0, 0]))

    def test_all_zeros_unchanged(self):
        '''No runs present: array of all 0s is unchanged.'''
        raw = np.array([0.0, 0.0, 0.0])
        result = mark_events(raw, event_bool=True)
        np.testing.assert_array_equal(result, raw)

    def test_all_ones_collapses_to_last_trial_only(self):
        '''A run spanning the whole array collapses to a single 1 at the end.'''
        raw = np.array([1.0, 1.0, 1.0, 1.0])
        result = mark_events(raw, event_bool=True)
        np.testing.assert_array_equal(result, np.array([0, 0, 0, 1]))

    def test_does_not_mutate_input_array(self):
        '''mark_events must not mutate the caller's raw array in place.'''
        raw = np.array([0.0, 1.0, 1.0, 0.0])
        original = raw.copy()
        mark_events(raw, event_bool=True)
        np.testing.assert_array_equal(raw, original)


class TestSlidingWindowCount(unittest.TestCase):
    '''Tests for the sliding_window_count trailing-window count engine.'''

    def test_first_window_minus_1_trials_are_nan(self):
        data = np.array([1, 1, 1, 0, 1])
        result = sliding_window_count(data, window=3)
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))

    def test_counts_trailing_window_inclusive_of_current_trial(self):
        data = np.array([1, 1, 1, 0, 1])
        result = sliding_window_count(data, window=3)
        # windows: [1,1,1]=3, [1,1,0]=2, [1,0,1]=2
        np.testing.assert_array_equal(result[2:], np.array([3, 2, 2]))

    def test_window_of_1_equals_input(self):
        data = np.array([1, 0, 1, 1])
        result = sliding_window_count(data, window=1)
        np.testing.assert_array_equal(result, np.array([1, 0, 1, 1]))

    def test_invalid_window_raises(self):
        with self.assertRaises(ValueError):
            sliding_window_count(np.array([1, 0]), window=0)


class TestIdentifyFullWaterYears(unittest.TestCase):
    '''Tests for identify_full_water_years water-year boundary detection.'''

    def test_two_full_years(self):
        # 6-day years: year1 = idx 0-5, year2 = idx 6-11 (trailing year included)
        dowy = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])
        self.assertEqual(identify_full_water_years(dowy), [(0, 5), (6, 11)])

    def test_three_starts_gives_three_years(self):
        dowy = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        self.assertEqual(identify_full_water_years(dowy), [(0, 2), (3, 5), (6, 8)])

    def test_leading_partial_year_excluded(self):
        # series starts mid-year (dowy=4), first full year starts at idx 2
        dowy = np.array([4, 5, 1, 2, 3, 1, 2, 3])
        self.assertEqual(identify_full_water_years(dowy), [(2, 4), (5, 7)])

    def test_trailing_year_included_even_without_subsequent_start(self):
        # last (short) year has no subsequent dowy==1 but is still included --
        # callers are assumed to supply data trimmed to complete water years.
        dowy = np.array([1, 2, 3, 1, 2])
        self.assertEqual(identify_full_water_years(dowy), [(0, 2), (3, 4)])

    def test_no_starts_returns_empty(self):
        dowy = np.array([2, 3, 4, 5])
        self.assertEqual(identify_full_water_years(dowy), [])

    def test_single_start_covers_whole_series(self):
        dowy = np.array([1, 2, 3, 4, 5])
        self.assertEqual(identify_full_water_years(dowy), [(0, 4)])


class TestWaterYearProbabilityRatio(unittest.TestCase):
    '''Tests for water_year_probability_ratio (nested frequency base-probability engine).'''

    def test_ratio_placed_at_last_day_of_year_event_level(self):
        # 6-day years; eligible run of 3 consecutive successes -> 1 event.
        dowy = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])
        eligible = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        result = water_year_probability_ratio(eligible, dowy, event_bool=True)
        self.assertAlmostEqual(result[5], 1 / 6)
        self.assertAlmostEqual(result[11], 0.0)  # second year: no successes

    def test_event_bool_false_counts_every_success(self):
        dowy = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])
        eligible = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
        result = water_year_probability_ratio(eligible, dowy, event_bool=False)
        self.assertAlmostEqual(result[5], 3 / 6)

    def test_event_bool_true_collapses_run_to_single_success(self):
        dowy = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])
        eligible = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
        result = water_year_probability_ratio(eligible, dowy, event_bool=True)
        self.assertAlmostEqual(result[5], 1 / 6)

    def test_non_last_timesteps_are_nan(self):
        dowy = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])
        eligible = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
        result = water_year_probability_ratio(eligible, dowy, event_bool=True)
        for t in [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]:
            self.assertTrue(np.isnan(result[t]))

    def test_leading_partial_year_is_nan(self):
        dowy = np.array([4, 5, 1, 2, 3, 1, 2])
        eligible = np.array([1, 1, 1, 1, 1, 1, 1])
        result = water_year_probability_ratio(eligible, dowy, event_bool=True)
        # idx 0-1 (leading partial) excluded; idx 2-4 and 5-6 are full years
        self.assertFalse(np.isnan(result[4]))
        self.assertFalse(np.isnan(result[6]))
        for t in [0, 1, 2, 3, 5]:
            self.assertTrue(np.isnan(result[t]))

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            water_year_probability_ratio(np.array([1, 0]), np.array([1, 2, 3]))


class TestFrequencyFxUnNestedCountForm(unittest.TestCase):
    '''
    frequency_fx count/between forms, per notes/frequencyEnhancement-resolved.md
    Examples 1 and 2 (component = AND(magnitude, frequency), same-grain rule).
    '''

    def test_example_1_count_form_event_level(self):
        # flow: t0=4 t1=6 t2=6 t3=6 t4=4 t5=6; magnitude_gt5 = [0,1,1,1,0,1]
        magnitude = np.array([0, 1, 1, 1, 0, 1])
        f = comparison_fx('>=', 1)  # [op, n, N] with n=1, N=2
        fx = frequency_fx(f, order=2, big_n=2, event_bool=True)
        output = magnitude.reshape(-1, 1)
        result = fx(pd.DataFrame({'x': range(6)}), output)
        # windows (N=2): nan, [0,1]=1, [1,1]=2, [1,1]=2, [1,0]=1, [0,1]=1
        # diag (count>=1): nan, 1, 1, 1, 1, 1
        # event_bool=True collapses the run [t1..t5] to a single 1 at t5
        np.testing.assert_array_equal(
            result, np.array([np.nan, 0, 0, 0, 0, 1])
        )

    def test_example_2_count_form_timestep_level(self):
        magnitude = np.array([0, 1, 1, 1, 0, 1])
        f = comparison_fx('>=', 1)
        fx = frequency_fx(f, order=2, big_n=2, event_bool=False)
        output = magnitude.reshape(-1, 1)
        result = fx(pd.DataFrame({'x': range(6)}), output)
        np.testing.assert_array_equal(
            result, np.array([np.nan, 1, 1, 1, 1, 1])
        )

    def test_windows_over_and_of_preceding_characteristics(self):
        # eligibility = AND(magnitude, duration) -- both must be 1 for a trial
        # to count as a frequency-window success.
        magnitude = np.array([1, 1, 1, 1])
        duration = np.array([0, 1, 1, 1])
        output = np.column_stack([magnitude, duration])
        f = comparison_fx('>=', 2)  # n=2, N=3
        fx = frequency_fx(f, order=3, big_n=3, event_bool=False)
        result = fx(pd.DataFrame({'x': range(4)}), output)
        # eligible = AND(magnitude, duration) = [0,1,1,1]
        # window(N=3): nan, nan, [0,1,1]=2, [1,1,1]=3
        # diag (count>=2): nan, nan, 1, 1
        np.testing.assert_array_equal(result, np.array([np.nan, np.nan, 1, 1]))

    def test_between_form(self):
        magnitude = np.array([1, 1, 0, 1, 1])
        f = comparison_fx('<=', 1, '<=', 2)  # between [1, 2] inclusive
        fx = frequency_fx(f, order=2, big_n=2, event_bool=False)
        output = magnitude.reshape(-1, 1)
        result = fx(pd.DataFrame({'x': range(5)}), output)
        # windows (N=2): nan, [1,1]=2, [1,0]=1, [0,1]=1, [1,1]=2
        # between [1,2] inclusive: nan, 1, 1, 1, 1
        np.testing.assert_array_equal(result, np.array([np.nan, 1, 1, 1, 1]))

    def test_probability_form_not_yet_implemented(self):
        f = comparison_fx('>', 0.5)
        fx = frequency_fx(f, order=2, big_n=None, event_bool=True)
        output = np.array([[1], [0], [1]])
        with self.assertRaises(NotImplementedError):
            fx(pd.DataFrame({'x': range(3)}), output)


class TestWindowedCountPerWaterYear(unittest.TestCase):
    '''windowed_count_per_water_year: intra-annual count/between engine, resets per year.'''

    def test_window_resets_at_year_boundary(self):
        dowy = np.array([1, 2, 3, 1, 2, 3])
        eligible = np.array([1, 1, 0, 0, 1, 1])
        result = windowed_count_per_water_year(eligible, dowy, window=2)
        # year1: nan, [1,1]=2, [1,0]=1 ; year2: nan, [0,1]=1, [1,1]=2
        np.testing.assert_array_equal(result, np.array([np.nan, 2, 1, np.nan, 1, 2]))

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            windowed_count_per_water_year(np.array([1, 0]), np.array([1, 2, 3]), window=1)


class TestOrReducePerWaterYear(unittest.TestCase):
    '''or_reduce_per_water_year: per-year OR-reduction of an intra_annual diagnostic column.'''

    def test_year_with_a_one_is_true(self):
        dowy = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])
        diag = np.array([np.nan, np.nan, 1, 0, 0, 0, np.nan, np.nan, 0, 0, 0, 0])
        result = or_reduce_per_water_year(diag, dowy)
        self.assertEqual(result[5], 1.0)
        self.assertEqual(result[11], 0.0)

    def test_all_nan_year_stays_nan(self):
        dowy = np.array([1, 2, 3])
        diag = np.array([np.nan, np.nan, np.nan])
        result = or_reduce_per_water_year(diag, dowy)
        self.assertTrue(np.isnan(result[2]))

    def test_non_last_timesteps_are_nan(self):
        dowy = np.array([1, 2, 3, 4, 5, 6])
        diag = np.array([np.nan, np.nan, 1, 0, 0, 0])
        result = or_reduce_per_water_year(diag, dowy)
        for t in [0, 1, 2, 3, 4]:
            self.assertTrue(np.isnan(result[t]))

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            or_reduce_per_water_year(np.array([1, 0]), np.array([1, 2, 3]))


class TestNestedFrequencyIntraAnnualFx(unittest.TestCase):
    '''
    nested_frequency_intra_annual_fx, reproducing notes/frequencyEnhancement-resolved.md
    Example 3's intra_annual column (base pattern narrower than a full year).
    '''

    def test_example_3_intra_annual_column(self):
        # Water year = 6 timesteps. magnitude_gt5 = [1,1,1,0,0,1].
        # Base (intra-annual) pattern [>=,2,3], event_bool=False for illustration.
        magnitude = np.array([1, 1, 1, 0, 0, 1])
        output = magnitude.reshape(-1, 1).astype(float)
        dowy = np.array([1, 2, 3, 4, 5, 6])
        df = pd.DataFrame({'flow': range(6), 'dowy': dowy})
        f = comparison_fx('>=', 2)  # [op, n, N] with n=2, N=3
        fx = nested_frequency_intra_annual_fx(f, order=2, big_n=3, event_bool=False)
        result = fx(df, output)
        np.testing.assert_array_equal(
            result, np.array([np.nan, np.nan, 1, 0, 0, 0])
        )

    def test_example_3_event_bool_does_not_change_which_ones_survive(self):
        # Per the resolved doc: event_bool is display-only for the year verdict --
        # it never removes the only 1 an OR-reduction is looking for.
        magnitude = np.array([1, 1, 1, 0, 0, 1])
        output = magnitude.reshape(-1, 1).astype(float)
        dowy = np.array([1, 2, 3, 4, 5, 6])
        df = pd.DataFrame({'flow': range(6), 'dowy': dowy})
        f = comparison_fx('>=', 2)
        fx_event = nested_frequency_intra_annual_fx(f, order=2, big_n=3, event_bool=True)
        fx_timestep = nested_frequency_intra_annual_fx(f, order=2, big_n=3, event_bool=False)
        result_event = fx_event(df, output)
        result_timestep = fx_timestep(df, output)
        self.assertEqual(
            np.nansum(result_event == 1), np.nansum(result_timestep == 1)
        )
        self.assertTrue(np.any(result_event[~np.isnan(result_event)] == 1))

    def test_probability_base_form(self):
        # eligible with a run of 3 plus an isolated success at year's last day
        # (so magnitude also holds where the ratio-based diag is placed).
        magnitude = np.array([1, 1, 1, 0, 0, 1])
        output = magnitude.reshape(-1, 1).astype(float)
        dowy = np.array([1, 2, 3, 4, 5, 6])
        df = pd.DataFrame({'flow': range(6), 'dowy': dowy})
        f = comparison_fx('>', 0.1)  # ratio > 0.1; 2 events / 6 ~= 0.333 -> True
        fx = nested_frequency_intra_annual_fx(f, order=2, big_n=None, event_bool=True)
        result = fx(df, output)
        self.assertEqual(result[5], 1)
        for t in range(5):
            self.assertTrue(np.isnan(result[t]))


class TestNestedFrequencyInterannualFx(unittest.TestCase):
    '''
    nested_frequency_interannual_fx: sliding N-year window over per-year verdicts,
    broadcast across each qualifying water year.
    '''

    def test_broadcasts_verdict_across_qualifying_year(self):
        # Two 6-day years. intra_annual (already computed) year1 OR-reduces to
        # True (a 1 at idx5); year2 has no 1s -> False.
        intra_annual = np.array(
            [np.nan, np.nan, 1, 0, 0, 0, np.nan, np.nan, 0, 0, 0, 0]
        )
        dummy = np.zeros(12)
        output = np.column_stack([dummy, intra_annual])
        dowy = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])
        df = pd.DataFrame({'flow': range(12), 'dowy': dowy})
        f = comparison_fx('>=', 1)  # nested [op, n, N] with n=1, N=2 (years)
        fx = nested_frequency_interannual_fx(f, order=3, big_n=2, event_bool=True)
        result = fx(df, output)
        # year1 (idx0-5): insufficient interannual history (only 1 year seen) -> NaN
        for t in range(6):
            self.assertTrue(np.isnan(result[t]))
        # year2 (idx6-11): 2-year window count = 1 (only year1's True) -> >=1 -> 1
        for t in range(6, 12):
            self.assertEqual(result[t], 1)

    def test_probability_form_not_valid_at_interannual_level(self):
        intra_annual = np.array([np.nan, np.nan, 1, 0, 0, 0])
        output = np.column_stack([np.zeros(6), intra_annual])
        df = pd.DataFrame({'flow': range(6), 'dowy': [1, 2, 3, 4, 5, 6]})
        f = comparison_fx('>', 0.5)
        fx = nested_frequency_interannual_fx(f, order=3, big_n=None, event_bool=True)
        with self.assertRaises(NotImplementedError):
            fx(df, output)


class TestEvaluateComponentNestedFrequencyDispatch(unittest.TestCase):
    '''evaluate_component: nested frequency's terminal column replaces AND-of-all-columns.'''

    def test_nested_terminal_column_broadcasts_without_and(self):
        # magnitude column would fail AND at some timesteps, but since the
        # last characteristic is_nested, component == the nested column value.
        magnitude_values = np.array([0, 1, 0, 1])

        def magnitude_stub_fx(df, output):
            return magnitude_values

        def nested_stub_fx(df, output):
            # nested (interannual) column: 1 everywhere, unrelated to magnitude
            return np.array([1, 1, 1, 1], dtype=float)

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
            {'flow': [1.0, 2.0, 3.0, 4.0], 'dowy': [1.0, 2.0, 3.0, 4.0]},
            index=pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04']),
        )
        df.index.name = 'time'
        result = evaluate_component(df, component)
        # component should equal the nested column (all 1s), NOT AND(magnitude, nested)
        np.testing.assert_array_equal(result.df['comp'].values, np.array([1, 1, 1, 1]))

    def test_nan_in_nested_column_is_not_a_success(self):
        def magnitude_stub_fx(df, output):
            return np.array([1, 1], dtype=float)

        def nested_stub_fx(df, output):
            return np.array([np.nan, 1], dtype=float)

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
            {'flow': [1.0, 2.0], 'dowy': [1.0, 2.0]},
            index=pd.to_datetime(['2020-01-01', '2020-01-02']),
        )
        df.index.name = 'time'
        result = evaluate_component(df, component)
        np.testing.assert_array_equal(result.df['comp'].values, np.array([0, 1]))

