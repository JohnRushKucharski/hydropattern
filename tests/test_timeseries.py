'''Tests for the timeseries module.'''
import os
import tempfile
import unittest

import pandas as pd

from hydropattern.timeseries import Timeseries, first_day_of_water_year, to_day_of_water_year


class TestTimeseries(unittest.TestCase):
    '''Tests for the Timeseries module.'''
    #region: first_day_of_water_year tests
    def test_first_day_of_water_year(self):
        '''Tests first_day_of_water_year.'''
        self.assertEqual(first_day_of_water_year(day=29, month=2), 59)
        self.assertEqual(first_day_of_water_year(day=1, month=10), 274)

    def test_first_day_of_water_year_raises_error(self):
        '''Tests first_day_of_water_year raises error.'''
        with self.assertRaises(ValueError):
            first_day_of_water_year(1, 0)
        with self.assertRaises(ValueError):
            first_day_of_water_year(1, 32)
    #endregion

    #region: to_day_of_water_year tests
    def test_to_day_of_water_year_at_start_of_wy(self):
        '''Tests to_day_of_water_year.'''
        self.assertEqual(to_day_of_water_year(pd.Timestamp('1900-10-01'), 274), 1)
        self.assertEqual(to_day_of_water_year(pd.Timestamp('1904-10-01'), 274), 1)
    def test_to_day_of_water_year_at_end_of_wy(self):
        '''Tests to_day_of_water_year before start.'''
        self.assertEqual(to_day_of_water_year(pd.Timestamp('1900-09-30'), 274), 365)
        self.assertEqual(to_day_of_water_year(pd.Timestamp('1904-09-30'), 274), 365)
    def test_to_day_of_water_year_at_end_of_yr(self):
        '''Tests to_day_of_water_year at start.'''
        self.assertEqual(to_day_of_water_year(pd.Timestamp('1900-12-31'), 274), 92)
        self.assertEqual(to_day_of_water_year(pd.Timestamp('1904-12-31'), 274), 92)
    def test_to_day_of_water_year_after_end_of_yr(self):
        '''Tests to_day_of_water_year at start.'''
        self.assertEqual(to_day_of_water_year(pd.Timestamp('1900-2-28'), 274), 151)
        self.assertEqual(to_day_of_water_year(pd.Timestamp('1904-2-29'), 274), 151)
    def test_to_day_of_water_year_raises_error(self):
        '''Tests to_day_of_water_year raises error.'''
        with self.assertRaises(ValueError):
            to_day_of_water_year(pd.Timestamp('1900-10-01'), 0)
        with self.assertRaises(ValueError):
            to_day_of_water_year(pd.Timestamp('1904-10-01'), 366)
    #endregion

    #region: Timeseries class tests
    #region: from_dataframe tests
    def test_from_dataframe_good_data(self):
        '''Tests from_dataframe with good data.'''
        df = pd.DataFrame({'time': pd.date_range(start='1/1/1900', periods=10),
                           'value': range(10)}).set_index('time')
        # a second column containing dowy is added to the dataframe
        self.assertEqual(Timeseries.from_dataframe(df).data.shape, (10, 2))
    #endregion

    #region from_csv tests
    def test_from_csv_iso_dates_beyond_nanosecond_range(self):
        '''Tests from_csv with ISO dates spanning years that overflow datetime64[ns].'''
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'data.csv')
            pd.DataFrame({'time': ['1970-01-01', '2999-01-01'], 'value': [1.0, 2.0]}
                        ).to_csv(path, index=False)
            ts = Timeseries.from_csv(path)
            self.assertEqual(list(ts.data.index.year), [1970, 2999])

    def test_from_csv_custom_date_format_beyond_nanosecond_range(self):
        '''Tests from_csv with a custom date_format spanning years beyond datetime64[ns].'''
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'data.csv')
            pd.DataFrame({'time': ['01-01-1970', '01-01-2999'], 'value': [1.0, 2.0]}
                        ).to_csv(path, index=False)
            ts = Timeseries.from_csv(path, date_format='%d-%m-%Y')
            self.assertEqual(list(ts.data.index.year), [1970, 2999])
    #endregion

    #region from_excel tests
    def test_from_excel_string_dates_beyond_nanosecond_range(self):
        '''Tests from_excel with string-formatted date cells beyond datetime64[ns] range.'''
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'data.xlsx')
            pd.DataFrame({'time': ['1970-01-01', '2999-01-01'], 'value': [1.0, 2.0]}
                        ).to_excel(path, index=False)
            ts = Timeseries.from_excel(path)
            self.assertEqual(list(ts.data.index.year), [1970, 2999])

    def test_from_excel_string_dates_custom_format_beyond_nanosecond_range(self):
        '''Tests from_excel with custom-format string date cells beyond datetime64[ns].'''
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'data.xlsx')
            pd.DataFrame({'time': ['01-01-1970', '01-01-2999'], 'value': [1.0, 2.0]}
                        ).to_excel(path, index=False)
            ts = Timeseries.from_excel(path, date_format='%d-%m-%Y')
            self.assertEqual(list(ts.data.index.year), [1970, 2999])
    #endregion

    #region day_of_water_year_to_date tests
    def test_day_of_water_year_to_date_no_leap_yr(self):
        '''Tests day_of_water_year_to_datetime.'''
        df = pd.DataFrame({'time': pd.date_range(start='1/1/1900', periods=365),
                           'value': range(365)}).set_index('time')
        ts = Timeseries.from_dataframe(df, first_dowy=274)
        self.assertEqual(ts.day_of_water_year_to_date(dowy=1, year=1900),
                         pd.Timestamp('1900-10-01'))
        self.assertEqual(ts.day_of_water_year_to_date(dowy=92, year=1900),
                         pd.Timestamp('1900-12-31'))
        self.assertEqual(ts.day_of_water_year_to_date(dowy=93, year=1900),
                         pd.Timestamp('1900-01-01'))
        self.assertEqual(ts.day_of_water_year_to_date(dowy=365, year=1900),
                         pd.Timestamp('1900-09-30'))
    #endregion

    #region: dowy tests
    def test_dowy(self):
        '''Looks for dowy series.'''
        df = pd.DataFrame({'time': pd.date_range(start='1/1/1900', periods=365),
                           'value': range(365)}).set_index('time')
        ts = Timeseries.from_dataframe(df, first_dowy=274)
        self.assertTrue(pd.Series.equals(ts.data.dowy,
                                         pd.Series(ts.data.dowy, index=ts.data.index)))
    #endregion

    #region: plot_timeseries tests
    def test_plot_timeseries_with_broken_axis_ranges(self, show_plot: bool = False):
        '''Tests plot_timeseries with custom broken_axis_ranges.

        Args:
            show_plot (bool): If True, displays the plot window. Default False.
        '''
        #pylint: disable-all
        import os
        import tempfile

        import matplotlib
        import matplotlib.pyplot as plt

        # Use non-interactive backend to prevent window from spawning (unless requested)
        original_backend = matplotlib.get_backend()
        if not show_plot:
            matplotlib.use('Agg')

        try:
            # Create test data
            df = pd.DataFrame({
                'time': pd.date_range(start='1/1/1900', periods=10, freq='D'),
                'value': [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000],
            }).set_index('time')
            ts = Timeseries.from_dataframe(df)

            # Test with custom broken axis ranges (should trigger _parse_divisions)
            # Using even number of ranges: [min1, max1, min2, max2]
            broken_axis_ranges = [0.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

            # Create temporary file for output
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_path = tmp.name

            ts.plot_timeseries(
                data_columns=[0],
                output_path=temp_path,
                broken_axis=True,
                broken_axis_ranges=broken_axis_ranges
            )

            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

            # Close any open figures to prevent memory leaks
            plt.close('all')

        except Exception as e:
            self.fail(f"plot_timeseries with broken_axis_ranges raised an exception: {e}")
        finally:
            # Restore original backend
            matplotlib.use(original_backend)

    def test_plot_timeseries_without_broken_axis(self, show_plot: bool = False):
        '''Tests plot_timeseries with broken_axis=False.

        Args:
            show_plot (bool): If True, displays the plot window. Default False.
        '''
        #pylint: disable-all
        import os
        import tempfile

        import matplotlib
        import matplotlib.pyplot as plt

        # Use non-interactive backend to prevent window from spawning (unless requested)
        original_backend = matplotlib.get_backend()
        if not show_plot:
            matplotlib.use('Agg')

        try:
            # Create test data with a range of values
            df = pd.DataFrame({
                'time': pd.date_range(start='1/1/1900', periods=15, freq='D'),
                'value': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
            }).set_index('time')
            ts = Timeseries.from_dataframe(df)

            # Create temporary file for output
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_path = tmp.name

            # Test with broken_axis=False (should trigger _global_min_max wrapped in list)
            ts.plot_timeseries(
                data_columns=[0],
                output_path=temp_path,
                broken_axis=False  # This is the key part we're testing
            )

            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

            # Close any open figures to prevent memory leaks
            plt.close('all')

        except Exception as e:
            self.fail(f"plot_timeseries with broken_axis=False raised an exception: {e}")
        finally:
            # Restore original backend
            matplotlib.use(original_backend)

    def test_plot_timeseries_with_year_rows_and_comparison_series(self, show_plot: bool = False):
        '''Tests plot_timeseries over multiple year rows with a comparison series.'''
        #pylint: disable-all
        import os
        import tempfile

        import matplotlib
        import matplotlib.pyplot as plt

        original_backend = matplotlib.get_backend()
        if not show_plot:
            matplotlib.use('Agg')

        try:
            dates = pd.date_range(start='1900-01-01', periods=800, freq='D')
            values = [float(i % 120) for i in range(len(dates))]
            comparison = pd.Series([v * 0.8 for v in values], index=dates, name='comparison')
            df = pd.DataFrame({'time': dates, 'value': values}).set_index('time')
            ts = Timeseries.from_dataframe(df)

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_path = tmp.name

            ts.plot_timeseries(
                data_columns=['value'],
                output_path=temp_path,
                comparision_series=comparison,
                yrs_per_row=1,
                broken_axis=False,
            )

            self.assertTrue(os.path.exists(temp_path))
            self.assertGreater(os.path.getsize(temp_path), 0)

            if os.path.exists(temp_path):
                os.unlink(temp_path)
            plt.close('all')
        except Exception as e:
            self.fail(f'plot_timeseries with years/comparison raised an exception: {e}')
        finally:
            matplotlib.use(original_backend)
    #endregion
    #endregion
