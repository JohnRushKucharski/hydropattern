'''Tests for formatter metric computation.'''
# pylint: disable=import-outside-toplevel

import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from hydropattern.errors import HydropatternError
from hydropattern.formatters import (
    build_summary_sheet,
    compute_metric_series,
    compute_portion_series,
    plot_components,
    resolve_color_map,
    write_results,
    write_summary,
)
from hydropattern.parsers import ClimateCanvasPlotOptions, MetricMode, MetricOptions
from hydropattern.patterns import Characteristic, CharacteristicType, Component, Result
from hydropattern.timeseries import to_day_of_water_year


def _make_result(values: list[int], years: list[int],
                 char_name: str = 'magnitude',
                 comp_name: str = 'comp') -> Result:
    '''Build a minimal Result using Jan 1 timestamps (WY = CY).'''
    index = pd.DatetimeIndex(
        [pd.Timestamp(f'{y}-01-01') for y in years], name='time'
    )
    return _make_result_with_dates(values, index, char_name=char_name, comp_name=comp_name)


def _make_result_with_dates(values: list[int], index: pd.DatetimeIndex,
                             char_name: str = 'magnitude',
                             comp_name: str = 'comp') -> Result:
    '''Build a minimal Result with arbitrary timestamps.'''
    dowy = list(range(1, len(values) + 1))
    df = pd.DataFrame({
        'dv': [1.0] * len(values),
        'dowy': dowy,
        char_name: values,
        comp_name: values,
    }, index=index)
    char = Characteristic(
        name=char_name,
        fx=lambda df, out: np.array(values),
        type=CharacteristicType.MAGNITUDE,
    )
    component = Component(name=comp_name, characteristics=[char], is_success_pattern=True)
    result = object.__new__(Result)
    result.df = df
    result.dv_name = 'dv'
    result.component = component
    return result


class TestComputePortionSeries(unittest.TestCase):
    '''Tests for compute_portion_series().'''

    def test_unknown_outcomes_are_excluded_from_each_column_denominator(self):
        result = _make_result([1, 0, np.nan, np.nan], [2000, 2000, 2001, 2001])
        result.df['comp'] = [1, 1, np.nan, np.nan]

        magnitude = compute_portion_series(result, 'magnitude')
        component = compute_portion_series(result, 'comp')

        self.assertAlmostEqual(magnitude['total'], 0.5)
        self.assertAlmostEqual(component['total'], 1.0)
        self.assertTrue(pd.isna(magnitude[2001]))
        self.assertTrue(pd.isna(component[2001]))

    def test_all_unknown_outcomes_have_undefined_summary(self):
        result = _make_result([np.nan, np.nan], [2000, 2000])

        portion = compute_portion_series(result, 'magnitude')

        self.assertTrue(pd.isna(portion['total']))
        self.assertTrue(pd.isna(portion[2000]))

    def test_empty_record_has_undefined_total_summary(self):
        result = _make_result([], [])

        portion = compute_portion_series(result, 'magnitude')

        self.assertTrue(pd.isna(portion['total']))

    def test_frequency_table_uses_known_outcomes_and_configured_water_years(self):
        result = _make_result([1, 0, np.nan, np.nan], [2000, 2000, 2000, 2000])
        result.df.index = pd.DatetimeIndex(
            ['2000-09-30', '2000-10-01', '2001-09-30', '2001-10-01'],
            name='time',
        )
        result.first_day_of_water_year = 274
        result.df['comp'] = [1, 1, np.nan, np.nan]

        summary = result.frequency_table(by_water_years=True)

        self.assertEqual(list(summary.index), ['total', 2000, 2001, 2002])
        self.assertEqual(summary.loc['total', 'T'], 4)
        self.assertEqual(summary.loc['total', 'magnitude'], 1)
        self.assertAlmostEqual(summary.loc['total', 'magnitude(%)'], 50.0)
        self.assertAlmostEqual(summary.loc['total', 'comp(%)'], 100.0)
        self.assertTrue(pd.isna(summary.loc[2002, 'magnitude(%)']))

    def test_all_success_total_is_one(self):
        '''All successes -> total portion = 1.0.'''
        result = _make_result([1, 1, 1, 1], [2000, 2000, 2001, 2001])
        s = compute_portion_series(result, 'magnitude')
        self.assertAlmostEqual(s['total'], 1.0)

    def test_no_success_total_is_zero(self):
        '''Zero successes -> total portion = 0.0 (not blank).'''
        result = _make_result([0, 0, 0, 0], [2000, 2000, 2001, 2001])
        s = compute_portion_series(result, 'magnitude')
        self.assertAlmostEqual(s['total'], 0.0)

    def test_partial_success_total_portion(self):
        '''2 of 4 successes -> total portion = 0.5.'''
        result = _make_result([1, 0, 1, 0], [2000, 2000, 2001, 2001])
        s = compute_portion_series(result, 'magnitude')
        self.assertAlmostEqual(s['total'], 0.5)

    def test_per_water_year_rows(self):
        '''Series index contains total + one entry per distinct water year.'''
        result = _make_result([1, 1, 0, 0], [2000, 2000, 2001, 2001])
        s = compute_portion_series(result, 'magnitude')
        self.assertIn('total', s.index)
        self.assertIn(2000, s.index)
        self.assertIn(2001, s.index)

    def test_per_water_year_values(self):
        '''Per-year portions computed correctly.'''
        result = _make_result([1, 1, 0, 0], [2000, 2000, 2001, 2001])
        s = compute_portion_series(result, 'magnitude')
        self.assertAlmostEqual(s[2000], 1.0)
        self.assertAlmostEqual(s[2001], 0.0)

    def test_zero_year_portion_is_zero_not_na(self):
        '''Year with zero successes -> 0.0, not NaN/None.'''
        result = _make_result([1, 1, 0, 0], [2000, 2000, 2001, 2001])
        s = compute_portion_series(result, 'magnitude')
        self.assertFalse(pd.isna(s[2001]))
        self.assertAlmostEqual(s[2001], 0.0)

    def test_component_column(self):
        '''Works for the component column, not just a characteristic.'''
        result = _make_result([1, 0, 1, 0], [2000, 2000, 2001, 2001], comp_name='mycomp')
        s = compute_portion_series(result, 'mycomp')
        self.assertAlmostEqual(s['total'], 0.5)

    def test_single_year(self):
        '''Single water year -> index is [total, year].'''
        result = _make_result([1, 0, 1], [2005, 2005, 2005])
        s = compute_portion_series(result, 'magnitude')
        self.assertEqual(list(s.index), ['total', 2005])
        self.assertAlmostEqual(s['total'], 2 / 3)
        self.assertAlmostEqual(s[2005], 2 / 3)


class TestComputePortionSeriesWaterYear(unittest.TestCase):
    '''Tests for compute_portion_series() with non-calendar water years.'''

    # first_day_of_wy=274 = Oct 1 (US water year).
    # WY label = ending calendar year (US convention).
    # Oct 1 1970 (doy 274) -> WY 1971; Jan 1 1971 (doy 1) -> WY 1971.

    def _oct_jan_result(self) -> Result:
        '''Two records: Oct 1 1970 (success) and Jan 1 1971 (failure).'''
        index = pd.DatetimeIndex([
            pd.Timestamp('1970-10-01'),  # doy 274 -> WY 1971
            pd.Timestamp('1971-01-01'),  # doy   1 -> WY 1971
        ], name='time')
        return _make_result_with_dates([1, 0], index)

    def test_oct_jan_grouped_into_same_wy(self):
        '''Oct 1970 and Jan 1971 both fall in WY 1971 with first_day_of_wy=274.'''
        result = self._oct_jan_result()
        s = compute_portion_series(result, 'magnitude', first_day_of_wy=274)
        self.assertIn(1971, s.index)
        self.assertNotIn(1970, s.index)

    def test_oct_jan_wy_portion(self):
        '''1 success + 1 failure in WY 1971 -> portion 0.5.'''
        result = self._oct_jan_result()
        s = compute_portion_series(result, 'magnitude', first_day_of_wy=274)
        self.assertAlmostEqual(s[1971], 0.5)

    def test_sep_and_oct_in_different_wys(self):
        '''Sep 30 1970 (doy 273) -> WY 1970; Oct 1 1970 (doy 274) -> WY 1971.'''
        index = pd.DatetimeIndex([
            pd.Timestamp('1970-09-30'),  # doy 273 < 274 -> WY 1970
            pd.Timestamp('1970-10-01'),  # doy 274 >= 274 -> WY 1971
        ], name='time')
        result = _make_result_with_dates([1, 1], index)
        s = compute_portion_series(result, 'magnitude', first_day_of_wy=274)
        self.assertIn(1970, s.index)
        self.assertIn(1971, s.index)

    def test_default_first_day_of_wy_is_calendar_year(self):
        '''Default first_day_of_wy=1 -> WY label = calendar year.'''
        result = _make_result([1, 0], [2000, 2001])
        s = compute_portion_series(result, 'magnitude')
        self.assertIn(2000, s.index)
        self.assertIn(2001, s.index)


class TestComputeMetricSeries(unittest.TestCase):
    '''Tests for compute_metric_series() — metric-mode transform on top of portion.'''

    def test_portion_mode_matches_compute_portion_series(self):
        '''PORTION mode is a passthrough of compute_portion_series.'''
        result = _make_result([1, 0, 1, 0], [2000, 2000, 2001, 2001])
        expected = compute_portion_series(result, 'magnitude')
        actual = compute_metric_series(result, 'magnitude', MetricMode.PORTION)
        pd.testing.assert_series_equal(actual, expected)

    def test_default_mode_is_portion(self):
        '''Default mode argument behaves like PORTION.'''
        result = _make_result([1, 1, 0, 0], [2000, 2000, 2001, 2001])
        s = compute_metric_series(result, 'magnitude')
        self.assertAlmostEqual(s['total'], 0.5)

    def test_percentage_mode_scales_by_100(self):
        '''PERCENTAGE mode = portion * 100.'''
        result = _make_result([1, 0, 1, 0], [2000, 2000, 2001, 2001])
        s = compute_metric_series(result, 'magnitude', MetricMode.PERCENTAGE)
        self.assertAlmostEqual(s['total'], 50.0)

    def test_percentage_mode_zero_success_is_zero(self):
        '''PERCENTAGE mode: zero successes -> 0.0, not NA.'''
        result = _make_result([0, 0, 0, 0], [2000, 2000, 2001, 2001])
        s = compute_metric_series(result, 'magnitude', MetricMode.PERCENTAGE)
        self.assertAlmostEqual(s['total'], 0.0)

class TestBuildSummarySheet(unittest.TestCase):
    '''Tests for build_summary_sheet().

    build_summary_sheet(scenario_results, component_name, column, first_day_of_wy=1)
    -> DataFrame: index=['total', wy...], columns=scenario_names.
    '''

    def _two_scenario_results(self) -> dict[str, list[Result]]:
        '''Two scenarios, same years, different success patterns.'''
        r_a = _make_result([1, 1, 0, 0], [2000, 2000, 2001, 2001])
        r_b = _make_result([0, 0, 1, 1], [2000, 2000, 2001, 2001])
        return {'scenario_a': [r_a], 'scenario_b': [r_b]}

    def test_columns_are_scenario_names(self):
        '''DataFrame columns = scenario names in insertion order.'''
        sr = self._two_scenario_results()
        df = build_summary_sheet(sr, 'comp', 'magnitude')
        self.assertEqual(list(df.columns), ['scenario_a', 'scenario_b'])

    def test_index_starts_with_total(self):
        '''First row index is "total".'''
        sr = self._two_scenario_results()
        df = build_summary_sheet(sr, 'comp', 'magnitude')
        self.assertEqual(df.index[0], 'total')

    def test_index_contains_water_years(self):
        '''Index contains all water years present in the data.'''
        sr = self._two_scenario_results()
        df = build_summary_sheet(sr, 'comp', 'magnitude')
        self.assertIn(2000, df.index)
        self.assertIn(2001, df.index)

    def test_total_row_values(self):
        '''Total row = portion across full dataset per scenario.'''
        sr = self._two_scenario_results()
        df = build_summary_sheet(sr, 'comp', 'magnitude')
        self.assertAlmostEqual(df.loc['total', 'scenario_a'], 0.5)
        self.assertAlmostEqual(df.loc['total', 'scenario_b'], 0.5)

    def test_total_portion_uses_aggregate_known_counts_not_year_average(self):
        result = _make_result(
            [1, np.nan, np.nan, 0, 0, 0],
            [2000, 2000, 2001, 2001, 2001, 2001],
        )

        summary = build_summary_sheet({'scenario': [result]}, 'comp', 'magnitude')

        self.assertAlmostEqual(summary.loc[2000, 'scenario'], 1.0)
        self.assertAlmostEqual(summary.loc[2001, 'scenario'], 0.0)
        self.assertAlmostEqual(summary.loc['total', 'scenario'], 0.25)

    def test_per_year_values(self):
        '''Per-year values match compute_portion_series.'''
        sr = self._two_scenario_results()
        df = build_summary_sheet(sr, 'comp', 'magnitude')
        self.assertAlmostEqual(df.loc[2000, 'scenario_a'], 1.0)
        self.assertAlmostEqual(df.loc[2001, 'scenario_a'], 0.0)
        self.assertAlmostEqual(df.loc[2000, 'scenario_b'], 0.0)
        self.assertAlmostEqual(df.loc[2001, 'scenario_b'], 1.0)

    def test_single_scenario(self):
        '''Single scenario -> single column DataFrame.'''
        r = _make_result([1, 0, 1, 0], [2000, 2000, 2001, 2001])
        df = build_summary_sheet({'only': [r]}, 'comp', 'magnitude')
        self.assertEqual(list(df.columns), ['only'])
        self.assertAlmostEqual(df.loc['total', 'only'], 0.5)

    def test_component_column(self):
        '''Works for the component column, not just a characteristic.'''
        r = _make_result([1, 0, 1, 0], [2000, 2000, 2001, 2001])
        df = build_summary_sheet({'s': [r]}, 'comp', 'comp')
        self.assertAlmostEqual(df.loc['total', 's'], 0.5)

    def test_metric_mode_is_threaded_through(self):
        '''mode parameter controls the metric computed for the whole sheet.'''
        r = _make_result([1, 0, 1, 0], [2000, 2000, 2001, 2001])
        df = build_summary_sheet({'s': [r]}, 'comp', 'magnitude', mode=MetricMode.PERCENTAGE)
        self.assertAlmostEqual(df.loc['total', 's'], 50.0)

    def test_non_default_water_year(self):
        '''first_day_of_wy is passed through to compute_portion_series.'''
        index = pd.DatetimeIndex([
            pd.Timestamp('1970-10-01'),  # WY 1971 with first_day_of_wy=274
            pd.Timestamp('1971-01-01'),  # WY 1971
        ], name='time')
        r = _make_result_with_dates([1, 0], index)
        df = build_summary_sheet({'s': [r]}, 'comp', 'magnitude', first_day_of_wy=274)
        self.assertIn(1971, df.index)
        self.assertNotIn(1970, df.index)
        self.assertAlmostEqual(df.loc[1971, 's'], 0.5)


class TestWriteSummary(unittest.TestCase):
    '''Tests for write_summary() — file output of per-component summary Excel files.'''

    def _scenario_results(self) -> dict[str, list[Result]]:
        r_a = _make_result([1, 1, 0, 0], [2000, 2000, 2001, 2001])
        r_b = _make_result([0, 0, 1, 1], [2000, 2000, 2001, 2001])
        return {'scenario_a': [r_a], 'scenario_b': [r_b]}

    def _monthly_result(
        self,
        characteristic_name: str = 'magnitude',
        characteristic_values: list[float] | None = None,
        component_values: list[float] | None = None,
    ) -> Result:
        dates = pd.date_range('2020-01-01', periods=24, freq='MS', name='time')
        characteristic_values = characteristic_values or [1, np.nan] + [0] * 22
        component_values = component_values or [1, 0, np.nan] + [0] * 21
        first_day_of_wy = 274
        component = Component(
            name='comp',
            characteristics=[
                Characteristic(
                    name=characteristic_name,
                    fx=lambda df, out: np.array(characteristic_values),
                    type=CharacteristicType.MAGNITUDE,
                )
            ],
            is_success_pattern=True,
        )
        df = pd.DataFrame({
            'flow': np.ones(len(dates)),
            'dowy': [to_day_of_water_year(date, first_day_of_wy) for date in dates],
            characteristic_name: characteristic_values,
            'comp': component_values,
        }, index=dates)
        return Result(df=df, component=component, first_day_of_water_year=first_day_of_wy)

    def _monthly_scenario_results(self, characteristic_name: str = 'magnitude'):
        return {
            'scenario_a': [self._monthly_result(characteristic_name)],
            'scenario_b': [self._monthly_result(
                characteristic_name, [0] * 24, [1] * 24
            )],
        }

    def test_creates_summary_file_per_component(self):
        '''One {component}_summary.xlsx file per component in output dir.'''
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp)
            write_summary(self._scenario_results(), output_path)
            self.assertTrue((output_path / 'comp_summary.xlsx').exists())

    def test_summary_has_sheet_per_characteristic_and_component(self):
        '''Summary xlsx has one sheet per characteristic + one for the component.'''
        import tempfile

        from openpyxl import load_workbook
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp)
            write_summary(self._scenario_results(), output_path)
            wb = load_workbook(output_path / 'comp_summary.xlsx', read_only=True)
            sheetnames = wb.sheetnames
            wb.close()
            # component 'comp' has one characteristic 'magnitude' + component itself
            self.assertIn('magnitude', sheetnames)
            self.assertIn('comp', sheetnames)

    def test_reporting_details_align_counts_coverage_and_partial_years(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp)
            write_summary(
                self._monthly_scenario_results(), output_path, first_day_of_wy=274
            )
            details = pd.read_excel(
                output_path / 'comp_summary.xlsx',
                sheet_name='reporting_details',
            )

        characteristic_total = details[
            (details['scenario'] == 'scenario_a')
            & (details['outcome_column'] == 'magnitude')
            & (details['interval'] == 'total')
        ].iloc[0]
        self.assertEqual(characteristic_total['successful_timesteps'], 1)
        self.assertEqual(characteristic_total['known_timesteps'], 23)
        self.assertEqual(characteristic_total['total_timesteps'], 24)
        self.assertAlmostEqual(characteristic_total['known_outcome_coverage'], 23 / 24)
        self.assertEqual(characteristic_total['completeness'], 'whole_record')
        self.assertTrue(pd.isna(characteristic_total['event_count_lower']))

        component_total = details[
            (details['scenario'] == 'scenario_a')
            & (details['outcome_column'] == 'comp')
            & (details['interval'] == 'total')
        ].iloc[0]
        self.assertEqual(component_total['known_timesteps'], 23)
        self.assertEqual(component_total['successful_timesteps'], 1)
        self.assertEqual(component_total['observed_exposure_water_years'], 2.0)
        self.assertEqual(component_total['availability_status'], 'available')
        self.assertEqual(component_total['event_count_lower'], 1)
        self.assertEqual(component_total['event_count_upper'], 2)

        annual = details[
            (details['scenario'] == 'scenario_a')
            & (details['outcome_column'] == 'comp')
            & (details['interval'] != 'total')
        ].set_index('interval')
        self.assertEqual(annual.loc[2020, 'completeness'], 'partial')
        self.assertEqual(annual.loc[2021, 'completeness'], 'complete')
        self.assertEqual(annual.loc[2022, 'completeness'], 'partial')
        self.assertEqual(annual.loc[2021, 'total_timesteps'], 12)
        self.assertEqual(annual.loc[2020, 'observed_exposure_water_years'], 0.75)
        self.assertEqual(
            details[
                (details['scenario'] == 'scenario_b')
                & (details['outcome_column'] == 'comp')
                & (details['interval'] == 'total')
            ].iloc[0]['successful_timesteps'],
            24,
        )

    def test_reporting_details_sheet_name_avoids_characteristic_collision(self):
        import tempfile

        from openpyxl import load_workbook

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp)
            write_summary(
                self._monthly_scenario_results('reporting_details'),
                output_path,
                first_day_of_wy=274,
            )
            workbook = load_workbook(
                output_path / 'comp_summary.xlsx', read_only=True
            )
            sheetnames = workbook.sheetnames
            workbook.close()

        self.assertIn('reporting_details', sheetnames)
        self.assertIn('reporting_details_1', sheetnames)

    def test_unsupported_cadence_keeps_counts_and_marks_event_rates_unavailable(self):
        import tempfile

        result = self._monthly_result()
        result.df = result.df.drop(result.df.index[4])

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp)
            write_summary({'scenario': [result]}, output_path, first_day_of_wy=274)
            details = pd.read_excel(
                output_path / 'comp_summary.xlsx',
                sheet_name='reporting_details',
            )

        component_total = details[
            (details['outcome_column'] == 'comp')
            & (details['interval'] == 'total')
        ].iloc[0]
        self.assertEqual(component_total['known_timesteps'], 22)
        self.assertEqual(component_total['event_count_lower'], 1)
        self.assertEqual(
            component_total['availability_status'], 'event_rate_unavailable'
        )
        self.assertTrue(pd.isna(component_total['observed_exposure_water_years']))
        self.assertIn('unsupported cadence', component_total['availability_reason'])
        annual = details[details['interval'] != 'total']
        self.assertTrue((annual['completeness'] == 'undetermined').all())

    def test_csv_and_excel_exports_keep_raw_layout_and_add_details_summary(self):
        import tempfile

        scenario_results = self._monthly_scenario_results()
        with tempfile.TemporaryDirectory() as tmp:
            csv_directory = Path(tmp) / 'csv'
            excel_directory = Path(tmp) / 'excel'
            write_results(
                scenario_results, 'input.toml', str(csv_directory), False,
                first_day_of_wy=274,
            )
            write_results(
                scenario_results, 'input.toml', str(excel_directory), True,
                first_day_of_wy=274,
            )

            raw_csv = pd.read_csv(csv_directory / 'scenario_a_comp.csv')
            raw_excel = pd.read_excel(excel_directory / 'input_output.xlsx')
            self.assertEqual(
                list(raw_csv.columns),
                ['time', 'flow', 'dowy', 'magnitude', 'comp'],
            )
            self.assertEqual(
                list(raw_excel.columns),
                ['time', 'flow', 'dowy', 'magnitude', 'comp'],
            )
            csv_details = pd.read_excel(
                csv_directory / 'comp_summary.xlsx',
                sheet_name='reporting_details',
            )
            excel_details = pd.read_excel(
                excel_directory / 'comp_summary.xlsx',
                sheet_name='reporting_details',
            )

        pd.testing.assert_frame_equal(csv_details, excel_details)

    def test_summary_sheet_columns_are_scenarios(self):
        '''Each summary sheet has scenario names as columns.'''
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp)
            write_summary(self._scenario_results(), output_path)
            df = pd.read_excel(output_path / 'comp_summary.xlsx',
                               sheet_name='magnitude', index_col=0)
            self.assertIn('scenario_a', df.columns)
            self.assertIn('scenario_b', df.columns)

    def test_summary_total_row_values(self):
        '''Total row in summary sheet matches expected portion.'''
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp)
            write_summary(self._scenario_results(), output_path)
            df = pd.read_excel(output_path / 'comp_summary.xlsx',
                               sheet_name='magnitude', index_col=0)
            self.assertAlmostEqual(df.loc['total', 'scenario_a'], 0.5)
            self.assertAlmostEqual(df.loc['total', 'scenario_b'], 0.5)

    def test_write_summary_overwrite_replaces_file(self):
        '''Default overwrite=True replaces existing summary file.'''
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp)
            write_summary(self._scenario_results(), output_path)
            write_summary(self._scenario_results(), output_path)
            files = list(output_path.glob('comp_summary*.xlsx'))
            self.assertEqual(len(files), 1)

    def test_write_summary_no_overwrite_appends_suffix(self):
        '''overwrite=False appends __1 suffix instead of replacing.'''
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp)
            write_summary(self._scenario_results(), output_path, overwrite=False)
            write_summary(self._scenario_results(), output_path, overwrite=False)
            self.assertTrue((output_path / 'comp_summary.xlsx').exists())
            self.assertTrue((output_path / 'comp_summary__1.xlsx').exists())


class TestPlotComponents(unittest.TestCase):
    '''Tests for plot_components.'''

    def _grid_result(self, component_name: str, success: bool,
                      is_success_pattern: bool = True) -> Result:
        '''Minimal Result whose component column is all-success or all-failure.'''
        component = Component(name=component_name, characteristics=[],
                               is_success_pattern=is_success_pattern)
        index = pd.DatetimeIndex([
            pd.Timestamp('2000-01-01'), pd.Timestamp('2000-02-01'),
        ], name='time')
        value = 1 if success else 0
        df = pd.DataFrame({
            'flow': [1.0, 2.0],
            component_name: [value, value],
        }, index=index)
        return Result(df=df, component=component)

    def test_plot_components_writes_grid_csv_and_png_per_component(self):
        '''One {component}_grid.csv and {component}_plot.png written per component.'''
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            scenario_results = {
                '_0_0': [self._grid_result('single_characteristic', True)],
                '_0_1.5': [self._grid_result('single_characteristic', False)],
                '_5_0': [self._grid_result('single_characteristic', True)],
                '_5_1.5': [self._grid_result('single_characteristic', True)],
            }

            plot_components(scenario_results, output_path, MetricOptions(), 1,
                            ClimateCanvasPlotOptions(interpolate=False, show=False))

            self.assertTrue((output_path / 'single_characteristic_grid.csv').exists())
            self.assertTrue((output_path / 'single_characteristic_plot.png').exists())

    def test_plot_components_raises_for_non_grid_scenarios(self):
        '''Non-grid scenario names raise HydropatternError (PLOT_INVALID_SCENARIO_GRID).'''
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            scenario_results = {
                'flow_a': [self._grid_result('single_characteristic', True)],
                'flow_b': [self._grid_result('single_characteristic', False)],
            }

            with self.assertRaises(HydropatternError):
                plot_components(scenario_results, output_path, MetricOptions(), 1,
                                ClimateCanvasPlotOptions(interpolate=False, show=False))

    def test_plot_components_defaults_title_to_component_name_and_zlabel_to_metric_mode(self):
        '''title/zlabel default to component name and mode value when unset.'''
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch('hydropattern.formatters.plot_response_surface') as mocked:
            output_path = Path(temp_dir)
            scenario_results = {
                '_0_0': [self._grid_result('single_characteristic', True)],
                '_0_1.5': [self._grid_result('single_characteristic', False)],
                '_5_0': [self._grid_result('single_characteristic', True)],
                '_5_1.5': [self._grid_result('single_characteristic', True)],
            }

            plot_components(scenario_results, output_path,
                            MetricOptions(mode=MetricMode.PERCENTAGE), 1,
                            ClimateCanvasPlotOptions())

            _, kwargs = mocked.call_args
            self.assertEqual(kwargs['title'], 'single_characteristic')
            self.assertEqual(
                kwargs['labels'],
                ('Precipitation Delta (%)', 'Temperature Delta (C)', 'percentage'),
            )

    def test_plot_components_uses_configured_title_and_labels_when_set(self):
        '''Explicit title/xlabel/ylabel/zlabel override the dynamic defaults.'''
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch('hydropattern.formatters.plot_response_surface') as mocked:
            output_path = Path(temp_dir)
            scenario_results = {
                '_0_0': [self._grid_result('single_characteristic', True)],
                '_0_1.5': [self._grid_result('single_characteristic', False)],
                '_5_0': [self._grid_result('single_characteristic', True)],
                '_5_1.5': [self._grid_result('single_characteristic', True)],
            }

            plot_components(scenario_results, output_path, MetricOptions(), 1,
                            ClimateCanvasPlotOptions(
                                title='Custom Title', xlabel='X', ylabel='Y', zlabel='Z',
                            ))

            _, kwargs = mocked.call_args
            self.assertEqual(kwargs['title'], 'Custom Title')
            self.assertEqual(kwargs['labels'], ('X', 'Y', 'Z'))

    def test_plot_components_forwards_threshold_color_map_and_ticks(self):
        '''Configured climate-canvas tuning options are forwarded to plotting.'''
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch('hydropattern.formatters.plot_response_surface') as mocked:
            output_path = Path(temp_dir)
            scenario_results = {
                '_0_0': [self._grid_result('single_characteristic', True)],
                '_0_1.5': [self._grid_result('single_characteristic', False)],
                '_5_0': [self._grid_result('single_characteristic', True)],
                '_5_1.5': [self._grid_result('single_characteristic', True)],
            }

            plot_components(
                scenario_results, output_path, MetricOptions(), 1,
                ClimateCanvasPlotOptions(
                    threshold=1.5, color_map='viridis', color_map_ticks=[-1.0, 0.0, 1.0],
                ),
            )

            _, kwargs = mocked.call_args
            self.assertEqual(kwargs['threshold'], 1.5)
            self.assertEqual(kwargs['color_map'], 'viridis')
            self.assertEqual(kwargs['color_map_ticks'], [-1.0, 0.0, 1.0])

    def test_plot_components_forwards_fillin(self):
        '''Configured climate-canvas fillin option is forwarded to plotting.'''
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch('hydropattern.formatters.plot_response_surface') as mocked:
            output_path = Path(temp_dir)
            scenario_results = {
                '_0_0': [self._grid_result('single_characteristic', True)],
                '_0_1.5': [self._grid_result('single_characteristic', False)],
                '_5_0': [self._grid_result('single_characteristic', True)],
                '_5_1.5': [self._grid_result('single_characteristic', True)],
            }

            plot_components(scenario_results, output_path, MetricOptions(), 1,
                            ClimateCanvasPlotOptions(fillin=True))

            _, kwargs = mocked.call_args
            self.assertTrue(kwargs['fillin'])

    def test_plot_components_handles_degenerate_all_zero_result(self):
        '''All-scenario 0.0 portion (pattern never succeeds) must not crash.

        climate_canvas's default TwoSlopeNorm requires vmin < vcenter < vmax; an
        all-equal z-grid makes vmin == vmax, which used to raise ValueError. See
        formatters._degenerate_range_norm.
        '''
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            scenario_results = {
                '_0_0': [self._grid_result('single_characteristic', False)],
                '_0_1.5': [self._grid_result('single_characteristic', False)],
                '_5_0': [self._grid_result('single_characteristic', False)],
                '_5_1.5': [self._grid_result('single_characteristic', False)],
            }

            plot_components(scenario_results, output_path, MetricOptions(), 1,
                            ClimateCanvasPlotOptions(interpolate=False, show=False))

            self.assertTrue((output_path / 'single_characteristic_plot.png').exists())

    def test_plot_components_forwards_degenerate_norm_and_levels(self):
        '''Degenerate (all-equal) z-grids get a caller-supplied norm/levels/widths.'''
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch('hydropattern.formatters.plot_response_surface') as mocked:
            output_path = Path(temp_dir)
            scenario_results = {
                '_0_0': [self._grid_result('single_characteristic', False)],
                '_0_1.5': [self._grid_result('single_characteristic', False)],
                '_5_0': [self._grid_result('single_characteristic', False)],
                '_5_1.5': [self._grid_result('single_characteristic', False)],
            }

            plot_components(scenario_results, output_path, MetricOptions(), 1,
                            ClimateCanvasPlotOptions())

            _, kwargs = mocked.call_args
            self.assertIsNotNone(kwargs['norm'])
            self.assertEqual(kwargs['levels'], (0.0,))
            self.assertEqual(kwargs['widths'], (1.0,))

    def test_plot_components_reverses_default_color_map_for_failure_pattern(self):
        '''success_pattern=False reverses the default RdBu colormap to RdBu_r.'''
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch('hydropattern.formatters.plot_response_surface') as mocked:
            output_path = Path(temp_dir)
            scenario_results = {
                '_0_0': [self._grid_result('single_characteristic', True,
                                            is_success_pattern=False)],
                '_0_1.5': [self._grid_result('single_characteristic', False,
                                              is_success_pattern=False)],
                '_5_0': [self._grid_result('single_characteristic', True,
                                            is_success_pattern=False)],
                '_5_1.5': [self._grid_result('single_characteristic', True,
                                              is_success_pattern=False)],
            }

            plot_components(scenario_results, output_path, MetricOptions(), 1,
                            ClimateCanvasPlotOptions())

            _, kwargs = mocked.call_args
            self.assertEqual(kwargs['color_map'], 'RdBu_r')

    def test_plot_components_keeps_explicit_color_map_for_failure_pattern(self):
        '''An explicitly-configured color_map is never auto-reversed.'''
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch('hydropattern.formatters.plot_response_surface') as mocked:
            output_path = Path(temp_dir)
            scenario_results = {
                '_0_0': [self._grid_result('single_characteristic', True,
                                            is_success_pattern=False)],
                '_0_1.5': [self._grid_result('single_characteristic', False,
                                              is_success_pattern=False)],
                '_5_0': [self._grid_result('single_characteristic', True,
                                            is_success_pattern=False)],
                '_5_1.5': [self._grid_result('single_characteristic', True,
                                              is_success_pattern=False)],
            }

            plot_components(scenario_results, output_path, MetricOptions(), 1,
                            ClimateCanvasPlotOptions(color_map='viridis'))

            _, kwargs = mocked.call_args
            self.assertEqual(kwargs['color_map'], 'viridis')


class TestResolveColorMap(unittest.TestCase):
    '''Tests for resolve_color_map.'''

    def test_default_map_portion_success_pattern_stays_rdbu(self):
        self.assertEqual(
            resolve_color_map('RdBu', is_success_pattern=True),
            'RdBu',
        )

    def test_default_map_portion_failure_pattern_reverses(self):
        self.assertEqual(
            resolve_color_map('RdBu', is_success_pattern=False),
            'RdBu_r',
        )

    def test_default_map_percentage_mode_behaves_like_portion(self):
        self.assertEqual(
            resolve_color_map('RdBu', is_success_pattern=False),
            'RdBu_r',
        )

    def test_explicit_color_map_never_reversed(self):
        for is_success_pattern in (True, False):
            self.assertEqual(
                resolve_color_map('viridis', is_success_pattern=is_success_pattern),
                'viridis',
            )
