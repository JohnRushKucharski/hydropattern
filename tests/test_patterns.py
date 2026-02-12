'''Tests for the patterns module.'''
# todo: test is_order_1, frequency_fx, evaluate_pattens
import unittest

import numpy as np
import pandas as pd

from hydropattern.patterns import (comparison_fx,
                                   moving_average, is_dowy_timeseries,
                                   timing_fx, magnitude_fx, duration_fx, rate_of_change_fx)

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
