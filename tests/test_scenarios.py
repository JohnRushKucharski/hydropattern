'''Tests for the hydropattern.scenarios module (library entrypoint for scenario-based
evaluation: split_scenarios, evaluate_scenarios, and ScenarioResults).'''
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from hydropattern.errors import HydropatternError
from hydropattern.formatters import build_summary_sheet
from hydropattern.parsers import ClimateCanvasPlotOptions, MetricMode, MetricOptions
from hydropattern.parsing.builders import build_components
from hydropattern.parsing.specs import CharacteristicSpec, ComponentSpec, Request
from hydropattern.patterns import CharacteristicType, Component, evaluate_components
from hydropattern.scenarios import evaluate_scenarios, split_scenarios
from hydropattern.timeseries import Timeseries


def _above_threshold_component(threshold: float = 2.0,
                               name: str = 'AboveThreshold') -> list[Component]:
    '''Builds a single-characteristic magnitude component for test fixtures.'''
    spec = ComponentSpec(
        name=name,
        characteristics=(
            CharacteristicSpec(type=CharacteristicType.MAGNITUDE, operator='>=',
                               values=(threshold,), order=1),
        ),
    )
    return build_components(Request(components=(spec,)))


class TestSplitScenarios(unittest.TestCase):
    '''Tests for split_scenarios.'''

    def test_splits_multi_column_timeseries_into_one_df_per_scenario(self):
        '''Each non-dowy column becomes its own [scenario_col, dowy] DataFrame.'''
        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0],
            'b': [4.0, 5.0, 6.0],
            'dowy': [1, 2, 3],
        })

        scenarios = split_scenarios(df)

        self.assertEqual(set(scenarios.keys()), {'a', 'b'})
        self.assertEqual(list(scenarios['a'].columns), ['a', 'dowy'])
        self.assertEqual(list(scenarios['b'].columns), ['b', 'dowy'])
        pd.testing.assert_series_equal(scenarios['a']['a'], df['a'])
        pd.testing.assert_series_equal(scenarios['a']['dowy'], df['dowy'])
        pd.testing.assert_series_equal(scenarios['b']['b'], df['b'])

    def test_single_column_timeseries_returns_one_scenario(self):
        '''A single data column + dowy is the degenerate one-scenario case.'''
        df = pd.DataFrame({'value': [10.0, 20.0], 'dowy': [1, 2]})

        scenarios = split_scenarios(df)

        self.assertEqual(list(scenarios.keys()), ['value'])
        self.assertEqual(list(scenarios['value'].columns), ['value', 'dowy'])


class TestEvaluateScenarios(unittest.TestCase):
    '''Tests for evaluate_scenarios.'''

    def test_matches_manual_evaluate_components_per_scenario(self):
        '''evaluate_scenarios must produce the same per-scenario Results as manually
        calling split_scenarios + evaluate_components (the CLI's current inline logic).'''
        dates = pd.date_range('1900-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'time': dates,
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [5.0, 4.0, 3.0, 2.0, 1.0],
        }).set_index('time')
        ts = Timeseries.from_dataframe(df)
        components = _above_threshold_component()

        result = evaluate_scenarios(ts, components)

        expected_scenarios = split_scenarios(ts.data)
        expected = {
            name: evaluate_components(scenario_df, components)
            for name, scenario_df in expected_scenarios.items()
        }

        self.assertEqual(set(result.scenario_results.keys()), set(expected.keys()))
        for name, expected_results in expected.items():
            actual_results = result.scenario_results[name]
            self.assertEqual(len(actual_results), len(expected_results))
            for actual, exp in zip(actual_results, expected_results):
                pd.testing.assert_frame_equal(actual.df, exp.df)
                self.assertEqual(actual.component.name, exp.component.name)

    def test_multiple_components_produce_one_result_each_per_scenario(self):
        '''Each scenario's Result list must contain one Result per component, in order.'''
        dates = pd.date_range('1900-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'time': dates, 'value': [1.0, 2.0, 3.0, 4.0, 5.0],
        }).set_index('time')
        ts = Timeseries.from_dataframe(df)
        components = _above_threshold_component(2.0) + _above_threshold_component(4.0)

        result = evaluate_scenarios(ts, components)

        results = result.scenario_results['value']
        self.assertEqual(len(results), 2)
        self.assertEqual([r.component.name for r in results],
                         [c.name for c in components])

    def test_captures_first_day_of_water_year_from_timeseries(self):
        '''ScenarioResults must retain the source Timeseries's first_day_of_water_year.'''
        df = pd.DataFrame({
            'time': pd.date_range('1900-01-01', periods=3, freq='D'),
            'value': [1.0, 2.0, 3.0],
        }).set_index('time')
        ts = Timeseries.from_dataframe(df, first_dowy=274)
        components = _above_threshold_component()

        result = evaluate_scenarios(ts, components)

        self.assertEqual(result.first_day_of_water_year, 274)
        self.assertEqual(
            result.scenario_results['value'][0].first_day_of_water_year, 274
        )

    def test_result_water_year_labels_use_configured_boundary(self):
        dates = pd.to_datetime(['2020-09-30', '2020-10-01', '2020-10-02'])
        data = pd.DataFrame({'value': [1.0, 2.0, 3.0]}, index=dates)
        data.index.name = 'time'
        ts = Timeseries.from_dataframe(data, first_dowy=274)
        evaluated = evaluate_scenarios(ts, _above_threshold_component())
        result = evaluated.scenario_results['value'][0]

        labels = result.identify_water_years()['water_year'].tolist()

        self.assertEqual(labels, [2020, 2021, 2021])

    def test_result_water_year_labels_require_boundary_for_direct_callers(self):
        dates = pd.date_range('2020-09-30', periods=3, freq='D')
        data = pd.DataFrame({'value': [1.0, 2.0, 3.0]}, index=dates)
        data.index.name = 'time'
        result = evaluate_scenarios(
            Timeseries.from_dataframe(data), _above_threshold_component()
        ).scenario_results['value'][0]
        result.first_day_of_water_year = None

        with self.assertRaisesRegex(ValueError, 'first_day_of_water_year is required'):
            result.identify_water_years()

        labels = result.identify_water_years(first_day_of_water_year=274)
        self.assertEqual(labels['water_year'].tolist(), [2020, 2021, 2021])

    def test_result_rejects_conflicting_water_year_boundary(self):
        dates = pd.date_range('2020-09-30', periods=3, freq='D')
        data = pd.DataFrame({'value': [1.0, 2.0, 3.0]}, index=dates)
        data.index.name = 'time'
        result = evaluate_scenarios(
            Timeseries.from_dataframe(data, first_dowy=274),
            _above_threshold_component(),
        ).scenario_results['value'][0]

        with self.assertRaisesRegex(ValueError, 'conflicts with Result'):
            result.identify_water_years(first_day_of_water_year=1)

class TestByComponent(unittest.TestCase):
    '''Tests for ScenarioResults.by_component.'''

    def test_filters_to_single_component_result_per_scenario(self):
        '''by_component must return {scenario_name: Result} for the named component,
        keeping the scenario keys but dropping the other components' Results.'''
        dates = pd.date_range('1900-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'time': dates,
            'a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [5.0, 4.0, 3.0, 2.0, 1.0],
        }).set_index('time')
        ts = Timeseries.from_dataframe(df)
        low = _above_threshold_component(2.0, name='Low')
        high = _above_threshold_component(4.0, name='High')
        low, high = low[0], high[0]
        result = evaluate_scenarios(ts, [low, high])

        filtered = result.by_component(high.name)

        self.assertEqual(set(filtered.keys()), {'a', 'b'})
        for scenario_name, res in filtered.items():
            self.assertEqual(res.component.name, high.name)
            # matches the manually-selected Result from the unfiltered list
            expected = next(
                r for r in result.scenario_results[scenario_name]
                if r.component.name == high.name
            )
            pd.testing.assert_frame_equal(res.df, expected.df)

    def test_raises_for_unknown_component_name(self):
        '''An unknown component name must raise, not silently return an empty dict.'''
        df = pd.DataFrame({
            'time': pd.date_range('1900-01-01', periods=3, freq='D'),
            'value': [1.0, 2.0, 3.0],
        }).set_index('time')
        ts = Timeseries.from_dataframe(df)
        components = _above_threshold_component()
        result = evaluate_scenarios(ts, components)

        with self.assertRaises(ValueError):
            result.by_component('NotAComponent')


class TestSummary(unittest.TestCase):
    '''Tests for ScenarioResults.summary.'''

    def _evaluate(self):
        dates = pd.date_range('1900-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'time': dates,
            'a': [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            'b': [5.0, 4.0, 3.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        }).set_index('time')
        ts = Timeseries.from_dataframe(df)
        low = _above_threshold_component(2.0, name='Low')[0]
        high = _above_threshold_component(4.0, name='High')[0]
        return evaluate_scenarios(ts, [low, high]), low, high

    def test_single_component_name_returns_dataframe_matching_build_summary_sheet(self):
        '''summary(name) must match calling build_summary_sheet directly (no new logic).'''
        result, low, _ = self._evaluate()

        summary = result.summary(low.name)

        expected = build_summary_sheet(
            result.scenario_results, low.name, low.name, result.first_day_of_water_year,
            MetricMode.PORTION,
        )
        self.assertIsInstance(summary, pd.DataFrame)
        pd.testing.assert_frame_equal(summary, expected)

    def test_respects_metric_mode(self):
        '''Passing a non-default mode must be forwarded to build_summary_sheet.'''
        result, low, _ = self._evaluate()

        summary = result.summary(low.name, mode=MetricMode.PERCENTAGE)

        expected = build_summary_sheet(
            result.scenario_results, low.name, low.name, result.first_day_of_water_year,
            MetricMode.PERCENTAGE,
        )
        pd.testing.assert_frame_equal(summary, expected)

    def test_no_component_name_returns_dict_of_all_components(self):
        '''Omitting component_name must return {component_name: DataFrame} for every
        component present in the scenario results.'''
        result, low, high = self._evaluate()

        summaries = result.summary()

        self.assertIsInstance(summaries, dict)
        self.assertEqual(set(summaries.keys()), {low.name, high.name})
        for name in (low.name, high.name):
            expected = build_summary_sheet(
                result.scenario_results, name, name, result.first_day_of_water_year,
                MetricMode.PORTION,
            )
            pd.testing.assert_frame_equal(summaries[name], expected)


class TestToExcelAndToCsv(unittest.TestCase):
    '''Tests for ScenarioResults.to_excel / to_csv.'''

    def _evaluate_from_csv(self, temp_dir: str) -> tuple:
        '''Builds a ScenarioResults from a Timeseries loaded from an actual csv file,
        so timeseries.file_path (and thus ScenarioResults.source_path) is set.'''
        csv_path = Path(temp_dir) / 'flows.csv'
        dates = pd.date_range('1900-01-01', periods=10, freq='D')
        pd.DataFrame({
            'time': dates.strftime('%Y-%m-%d'),
            'a': [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }).to_csv(csv_path, index=False)
        ts = Timeseries.from_csv(str(csv_path))
        component = _above_threshold_component(2.0, name='Low')[0]
        return evaluate_scenarios(ts, [component]), component

    def test_to_excel_default_output_dir_uses_timeseries_file_path_stem(self):
        '''With no output_directory, to_excel must derive <csv_stem>_output/<csv_stem>_output.xlsx
        from the source Timeseries's file_path -- same convention write_results/CLI use.'''
        with tempfile.TemporaryDirectory() as temp_dir:
            result, _ = self._evaluate_from_csv(temp_dir)

            output_path = result.to_excel()
            try:
                expected_dir = Path(temp_dir) / 'flows_output'
                self.assertEqual(output_path, expected_dir)
                self.assertTrue((expected_dir / 'flows_output.xlsx').exists())
                self.assertTrue((expected_dir / 'low_summary.xlsx').exists())
            finally:
                for f in expected_dir.glob('*'):
                    os.remove(f)
                expected_dir.rmdir()

    def test_to_excel_custom_output_directory(self):
        '''An explicit output_directory must be used instead of the default derivation.'''
        with tempfile.TemporaryDirectory() as temp_dir:
            result, _ = self._evaluate_from_csv(temp_dir)
            custom_dir = Path(temp_dir) / 'custom_out'

            output_path = result.to_excel(output_directory=str(custom_dir))

            self.assertEqual(output_path, custom_dir)
            self.assertTrue((custom_dir / 'flows_output.xlsx').exists())
            self.assertTrue((custom_dir / 'low_summary.xlsx').exists())

    def test_to_csv_writes_per_scenario_component_csv_plus_summary(self):
        '''write_to_excel=False path: one csv per scenario/component pair, plus the
        per-component summary xlsx (always written, matching CLI parity).'''
        with tempfile.TemporaryDirectory() as temp_dir:
            result, _ = self._evaluate_from_csv(temp_dir)
            custom_dir = Path(temp_dir) / 'csv_out'

            output_path = result.to_csv(output_directory=str(custom_dir))

            self.assertEqual(output_path, custom_dir)
            csv_files = list(custom_dir.glob('*.csv'))
            self.assertEqual(len(csv_files), 1)
            self.assertTrue((custom_dir / 'low_summary.xlsx').exists())

    def test_to_excel_without_source_file_falls_back_to_hydropattern_output_name(self):
        '''An in-memory Timeseries (no file_path) must still produce a sensible default
        name: 'hydropattern_output', not an error or a blank/None-derived path.'''
        dates = pd.date_range('1900-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'time': dates, 'value': [1.0, 2.0, 3.0, 4.0, 5.0],
        }).set_index('time')
        ts = Timeseries.from_dataframe(df)
        component = _above_threshold_component(2.0, name='Low')[0]
        result = evaluate_scenarios(ts, [component])

        with tempfile.TemporaryDirectory() as temp_dir:
            custom_dir = Path(temp_dir) / 'out'
            output_path = result.to_excel(output_directory=str(custom_dir))

            self.assertEqual(output_path, custom_dir)
            self.assertTrue((custom_dir / 'hydropattern_output.xlsx').exists())


class TestPlotResponseSurface(unittest.TestCase):
    '''Tests for ScenarioResults.plot_response_surface.'''

    def _grid_evaluate(self):
        '''Builds a ScenarioResults over 4 scenario columns forming a valid
        precip/temp scenario grid (_0_0, _0_1.5, _5_0, _5_1.5).'''
        dates = pd.date_range('1900-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'time': dates,
            '_0_0': [1.0, 2.0, 3.0, 4.0, 5.0],
            '_0_1.5': [1.0, 1.0, 1.0, 1.0, 1.0],
            '_5_0': [3.0, 3.0, 3.0, 3.0, 3.0],
            '_5_1.5': [5.0, 5.0, 5.0, 5.0, 5.0],
        }).set_index('time')
        ts = Timeseries.from_dataframe(df)
        component = _above_threshold_component(2.0, name='AboveThreshold')[0]
        return evaluate_scenarios(ts, [component]), component

    def test_raises_for_unknown_component_name(self):
        result, _ = self._grid_evaluate()
        with self.assertRaises(ValueError):
            result.plot_response_surface('NotAComponent')

    def test_raises_for_non_grid_scenarios(self):
        '''Non-grid scenario names raise HydropatternError (PLOT_INVALID_SCENARIO_GRID).'''
        dates = pd.date_range('1900-01-01', periods=3, freq='D')
        df = pd.DataFrame({
            'time': dates, 'flow_a': [1.0, 2.0, 3.0], 'flow_b': [3.0, 2.0, 1.0],
        }).set_index('time')
        ts = Timeseries.from_dataframe(df)
        component = _above_threshold_component(2.0, name='AboveThreshold')[0]
        result = evaluate_scenarios(ts, [component])

        with self.assertRaises(HydropatternError):
            result.plot_response_surface(component.name)

    def test_output_path_none_shows_and_writes_nothing(self):
        '''output_path=None must skip both file writes and force show=True.'''
        result, component = self._grid_evaluate()

        with mock.patch('hydropattern.formatters.plot_response_surface') as mocked:
            result.plot_response_surface(component.name)

        _, kwargs = mocked.call_args
        self.assertIsNone(kwargs['save_path'])
        self.assertTrue(kwargs['show'])

    def test_output_path_given_writes_grid_csv_and_png(self):
        '''output_path given must write both {component}_grid.csv and
        {component}_plot.png into it, following the CLI file layout.'''
        result, component = self._grid_evaluate()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'plots'
            result.plot_response_surface(component.name, output_path=output_path)

            self.assertTrue((output_path / f'{component.name}_grid.csv').exists())
            self.assertTrue((output_path / f'{component.name}_plot.png').exists())

    def test_output_path_given_show_follows_climate_canvas_option(self):
        '''When output_path is given, show must follow climate_canvas.show (default
        False), not be force-shown like the output_path=None case.'''
        result, component = self._grid_evaluate()

        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch('hydropattern.formatters.plot_response_surface') as mocked:
            result.plot_response_surface(
                component.name, output_path=Path(temp_dir),
                climate_canvas=ClimateCanvasPlotOptions(show=False),
            )

        _, kwargs = mocked.call_args
        self.assertIsNotNone(kwargs['save_path'])
        self.assertFalse(kwargs['show'])

    def test_forwards_metric_options_and_climate_canvas_to_shared_plotting_logic(self):
        '''title/labels must reflect metric_options.mode and climate_canvas overrides,
        matching plot_components's behavior for the same inputs.'''
        result, component = self._grid_evaluate()

        with mock.patch('hydropattern.formatters.plot_response_surface') as mocked:
            result.plot_response_surface(
                component.name,
                metric_options=MetricOptions(mode=MetricMode.PERCENTAGE),
            )

        _, kwargs = mocked.call_args
        self.assertTrue(kwargs['title'].startswith(component.name + '\n'))
        self.assertEqual(
            kwargs['labels'],
            (
                'Precipitation Delta (%)', 'Temperature Delta (C)',
                'Percentage of known outcomes (%)',
            ),
        )

    def test_forwards_minimum_coverage_to_shared_plotting_logic(self):
        result, component = self._grid_evaluate()
        with mock.patch('hydropattern.formatters.plot_response_surface') as mocked:
            result.plot_response_surface(component.name, minimum_coverage=0.65)

        _, kwargs = mocked.call_args
        self.assertIn('Minimum coverage: 65%', kwargs['title'])

    def test_rejects_invalid_python_minimum_coverage(self):
        result, component = self._grid_evaluate()
        for value in (True, float('nan'), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                result.plot_response_surface(component.name, minimum_coverage=value)


if __name__ == '__main__':
    unittest.main()
