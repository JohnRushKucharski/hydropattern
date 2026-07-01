'''Tests for CLI module functionality.'''

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from hydropattern.cli import (
    app,
    load_components,
    load_config_file,
    load_metric_options,
    load_timeseries,
    write_output,
)
from hydropattern.errors import HydropatternError, ParserErrorCode
from hydropattern.parsers import MetricMode, MetricOptions
from hydropattern.patterns import Component, Result

RUNNER = CliRunner()


class TestCLI(unittest.TestCase):
    '''Tests for CLI functions.'''

    def setUp(self):
        '''Set up test fixtures.'''
        self.test_files_dir = Path(__file__).parent / "test_files"
        self.test_toml_path = self.test_files_dir / "key_order_test.toml"
        self.reverse_order_path = self.test_files_dir / "reverse_order_test.toml"
        self.cli_smoke_config_path = self.test_files_dir / "cli_smoke_config.toml"

    def test_toml_key_order_preservation_top_level(self):
        '''Test that key order is preserved in top-level TOML tables.'''
        config = load_config_file(str(self.test_toml_path))

        # Check timeseries section key order
        timeseries_keys = list(config['timeseries'].keys())
        expected_timeseries_order = [
            'first_key', 'second_key', 'third_key',
            'path', 'date_format', 'first_day_of_water_year'
        ]

        self.assertEqual(timeseries_keys, expected_timeseries_order,
                        f"Expected timeseries keys in order {expected_timeseries_order}, "
                        f"but got {timeseries_keys}")

    def test_toml_key_order_preservation_components_section(self):
        '''Test that key order is preserved in components section.'''
        config = load_config_file(str(self.test_toml_path))

        # Check components section key order (excluding nested tables)
        components_keys = []
        for key in config['components'].keys():
            if not isinstance(config['components'][key], dict):
                components_keys.append(key)

        expected_components_order = [
            'alpha_component', 'zebra_component', 'middle_component'
        ]

        self.assertEqual(components_keys, expected_components_order,
                        f"Expected components keys in order {expected_components_order}, "
                        f"but got {components_keys}")

    def test_toml_key_order_preservation_nested_table(self):
        '''Test that key order is preserved in nested TOML tables.'''
        config = load_config_file(str(self.test_toml_path))

        # Check nested test_section key order
        test_section_keys = list(config['components']['test_section'].keys())
        expected_test_section_order = ['z_key', 'a_key', 'm_key', 'b_key']

        self.assertEqual(test_section_keys, expected_test_section_order,
                        f"Expected test_section keys in order {expected_test_section_order}, "
                        f"but got {test_section_keys}")

    def test_toml_key_order_preservation_deeply_nested(self):
        '''Test that key order is preserved in deeply nested TOML tables.'''
        config = load_config_file(str(self.test_toml_path))

        # Check nested_table top level
        nested_table_keys = []
        for key in config['nested_table'].keys():
            if not isinstance(config['nested_table'][key], dict):
                nested_table_keys.append(key)

        expected_nested_order = ['level1_first', 'level1_last']
        self.assertEqual(nested_table_keys, expected_nested_order,
                        f"Expected nested_table keys in order {expected_nested_order}, "
                        f"but got {nested_table_keys}")

        # Check deeply nested keys
        deeper_keys = list(config['nested_table']['deeper'].keys())
        expected_deeper_order = ['deep_first', 'deep_middle', 'deep_last']

        self.assertEqual(deeper_keys, expected_deeper_order,
                        f"Expected deeper keys in order {expected_deeper_order}, "
                        f"but got {deeper_keys}")

    def test_load_config_file_returns_dict(self):
        '''Test that load_config_file returns a dictionary.'''
        config = load_config_file(str(self.test_toml_path))

        self.assertIsInstance(config, dict,
                             "load_config_file should return a dictionary")

        # Verify expected top-level sections exist
        expected_sections = ['timeseries', 'components', 'nested_table']
        for section in expected_sections:
            self.assertIn(section, config,
                         f"Expected section '{section}' not found in config")

    def test_load_config_file_preserves_data_types(self):
        '''Test that load_config_file preserves correct data types.'''
        config = load_config_file(str(self.test_toml_path))

        # Check string values
        self.assertIsInstance(config['timeseries']['first_key'], str)
        self.assertEqual(config['timeseries']['first_key'], "first_value")

        # Check integer values
        self.assertIsInstance(config['timeseries']['first_day_of_water_year'], int)
        self.assertEqual(config['timeseries']['first_day_of_water_year'], 274)

        # Check nested integer values
        self.assertIsInstance(config['components']['test_section']['z_key'], int)
        self.assertEqual(config['components']['test_section']['z_key'], 1)

        # Check nested table structure
        self.assertIsInstance(config['nested_table']['deeper'], dict)

    def test_toml_key_order_comprehensive_check(self):
        '''Test comprehensive key order preservation across the entire structure.'''
        config = load_config_file(str(self.test_toml_path))

        # Get all top-level keys
        top_level_keys = list(config.keys())
        expected_top_level_order = ['timeseries', 'components', 'nested_table']

        self.assertEqual(top_level_keys, expected_top_level_order,
                        f"Expected top-level keys in order {expected_top_level_order}, "
                        f"but got {top_level_keys}")

        # Verify the order is not alphabetical (which would indicate sorting)
        alphabetical_order = sorted(expected_top_level_order)
        self.assertNotEqual(top_level_keys, alphabetical_order,
                           "Keys appear to be alphabetically sorted, not in original order")

    def test_reverse_alphabetical_order_preservation(self):
        '''Test that reverse alphabetical order is preserved (proving no auto-sorting).'''
        config = load_config_file(str(self.reverse_order_path))

        # Test z_section keys (zebra, apple, banana)
        z_section_keys = list(config['z_section'].keys())
        expected_z_order = ['zebra', 'apple', 'banana']

        self.assertEqual(z_section_keys, expected_z_order,
                        f"Expected z_section keys in order {expected_z_order}, "
                        f"but got {z_section_keys}")

        # Verify this is NOT alphabetical order
        alphabetical_z = sorted(expected_z_order)
        self.assertNotEqual(z_section_keys, alphabetical_z,
                           "Keys should NOT be in alphabetical order")

    def test_numeric_key_order_preservation(self):
        '''Test that numeric key order is preserved (not sorted numerically).'''
        config = load_config_file(str(self.reverse_order_path))

        # Test numeric_keys section (key_100, key_1, key_50, key_10)
        numeric_keys = list(config['numeric_keys'].keys())
        expected_numeric_order = ['key_100', 'key_1', 'key_50', 'key_10']

        self.assertEqual(numeric_keys, expected_numeric_order,
                        f"Expected numeric_keys in order {expected_numeric_order}, "
                        f"but got {numeric_keys}")

        # Verify this is NOT numerically sorted
        numeric_sorted = ['key_1', 'key_10', 'key_50', 'key_100']
        self.assertNotEqual(numeric_keys, numeric_sorted,
                           "Keys should NOT be in numeric order")

    def test_section_order_preservation(self):
        '''Test that section order is preserved at the top level.'''
        config = load_config_file(str(self.reverse_order_path))

        # Get all top-level section keys
        section_keys = list(config.keys())
        expected_section_order = ['z_section', 'a_section',
                                  'middle_section', 'order_test', 'numeric_keys']

        self.assertEqual(section_keys, expected_section_order,
                        f"Expected sections in order {expected_section_order}, "
                        f"but got {section_keys}")

        # Verify NOT alphabetical
        alphabetical_sections = sorted(expected_section_order)
        self.assertNotEqual(section_keys, alphabetical_sections,
                           "Sections should NOT be in alphabetical order")

    def test_mixed_value_key_order(self):
        '''Test key order preservation with mixed value types.'''
        config = load_config_file(str(self.reverse_order_path))

        # Test order_test nested section with mixed ordering
        nested_keys = list(config['order_test']['nested'].keys())
        expected_nested_order = ['nine', 'one', 'five', 'three']

        self.assertEqual(nested_keys, expected_nested_order,
                        f"Expected nested keys in order {expected_nested_order}, "
                        f"but got {nested_keys}")

        # Verify values are correct and in expected positions
        self.assertEqual(config['order_test']['nested']['nine'], 9)
        self.assertEqual(config['order_test']['nested']['one'], 1)
        self.assertEqual(config['order_test']['nested']['five'], 5)
        self.assertEqual(config['order_test']['nested']['three'], 3)


class TestCLICommand(unittest.TestCase):
    '''Command-level tests for Typer CLI behavior.'''

    def setUp(self):
        '''Set up command-level fixture paths.'''
        self.test_files_dir = Path(__file__).parent / "test_files"
        self.cli_smoke_config_path = self.test_files_dir / "cli_smoke_config.toml"

    def test_cli_without_args_shows_help(self):
        '''Invoking the app with no args should show help text.'''
        result = RUNNER.invoke(app, [])

        self.assertEqual(result.exit_code, 0)
        self.assertIn('Usage', result.stdout)
        self.assertIn('run', result.stdout)

    def test_run_command_smoke_with_temp_files(self):
        '''Default run (Excel) succeeds and writes an xlsx file.'''
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'out'

            result = RUNNER.invoke(
                app,
                ['run', str(self.cli_smoke_config_path), '--output-dir', str(output_dir)],
            )

            self.assertEqual(result.exit_code, 0, msg=result.stdout)
            self.assertIn('Output written to:', result.stdout)
            self.assertTrue(output_dir.exists())
            self.assertTrue(any(output_dir.glob('*.xlsx')))

    def test_run_command_no_excel_writes_csv(self):
        '''--no-excel flag produces per-scenario csv files instead of xlsx.'''
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'out'

            result = RUNNER.invoke(
                app,
                ['run', str(self.cli_smoke_config_path),
                 '--output-dir', str(output_dir), '--no-excel'],
            )

            self.assertEqual(result.exit_code, 0, msg=result.stdout)
            self.assertTrue(any(output_dir.glob('*.csv')))


class TestCLIValidationErrors(unittest.TestCase):
    '''Validation error tests for CLI configuration parsing.'''

    def test_load_timeseries_raises_when_missing_timeseries_section(self):
        '''Missing top-level timeseries section should raise ValueError.'''
        with self.assertRaises(HydropatternError) as context:
            load_timeseries({'components': {'single_characteristic': {'magnitude': ['>', 1.0]}}})

        self.assertEqual(context.exception.envelope.code, ParserErrorCode.MISSING_SECTION)
        self.assertEqual(context.exception.envelope.context['section'], 'timeseries')
        self.assertEqual(context.exception.envelope.source, 'parser')
        self.assertEqual(
            context.exception.envelope.message,
            'No timeseries data in configuration file.',
        )

    def test_load_timeseries_raises_when_missing_path(self):
        '''Missing timeseries path should raise ValueError.'''
        with self.assertRaises(HydropatternError) as context:
            load_timeseries({'timeseries': {'date_format': '%Y-%m-%d'}})

        self.assertEqual(context.exception.envelope.code, ParserErrorCode.MISSING_FIELD)
        self.assertEqual(context.exception.envelope.context['section'], 'timeseries')
        self.assertEqual(context.exception.envelope.context['field'], 'path')

    def test_load_components_raises_when_missing_components_section(self):
        '''Missing top-level components section should raise ValueError.'''
        with self.assertRaises(HydropatternError) as context:
            load_components({'timeseries': {'path': 'tests/test_files/cli_smoke_input.csv'}})

        self.assertEqual(context.exception.envelope.code, ParserErrorCode.MISSING_SECTION)
        self.assertEqual(context.exception.envelope.context['section'], 'components')

    def test_load_components_raises_error_for_unknown_characteristic(self):
        '''Unknown characteristic names should use the shared error envelope.'''
        with self.assertRaises(HydropatternError) as context:
            load_components({
                'timeseries': {'path': 'tests/test_files/cli_smoke_input.csv'},
                'components': {
                    'single_characteristic': {
                        'unsupported_characteristic': [1, 2]
                    }
                },
            })

        self.assertEqual(context.exception.envelope.code, ParserErrorCode.UNKNOWN_CHARACTERISTIC)
        self.assertEqual(context.exception.envelope.context['component'], 'single_characteristic')
        self.assertEqual(
            context.exception.envelope.context['characteristic'],
            'unsupported_characteristic',
        )

    def test_load_metric_options_defaults_to_portion(self):
        '''Missing [metric] section should default to MetricOptions(mode=PORTION).'''
        options = load_metric_options({'timeseries': {'path': 'x.csv'}})
        self.assertEqual(options, MetricOptions(mode=MetricMode.PORTION))

    def test_load_metric_options_reads_configured_mode(self):
        '''Configured [metric].mode should be reflected in the returned options.'''
        options = load_metric_options({'metric': {'mode': 'percentage'}})
        self.assertEqual(options.mode, MetricMode.PERCENTAGE)

    def test_load_metric_options_raises_for_invalid_mode(self):
        '''Invalid [metric].mode should raise the shared parser error envelope.'''
        with self.assertRaises(HydropatternError) as context:
            load_metric_options({'metric': {'mode': 'invalid'}})

        self.assertEqual(context.exception.envelope.code, ParserErrorCode.INVALID_VALUE)


class TestCLIOutputModes(unittest.TestCase):
    '''Output mode tests for csv and excel behavior.'''

    def _sample_result(self, component_name: str = 'sample_component') -> Result:
        '''Create a minimal Result object for output writing tests.

        Uses a DatetimeIndex and includes the component column so that
        write_summary can compute portions correctly.
        '''
        component = Component(name=component_name, characteristics=[], is_success_pattern=True)
        index = pd.DatetimeIndex([
            pd.Timestamp('2000-01-01'),
            pd.Timestamp('2000-02-01'),
            pd.Timestamp('2000-03-01'),
        ], name='time')
        df = pd.DataFrame({
            'flow': [1.0, 2.0, 3.0],
            component_name: [1, 0, 1],
        }, index=index)
        return Result(df=df, component=component)

    def test_write_output_default_csv_creates_stem_output_dir(self):
        '''Default csv mode should create <input_stem>_output and write a csv file.'''
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / 'config.toml'
            input_path.write_text('', encoding='utf-8')

            write_output({'flow': [self._sample_result()]}, str(input_path), None, False)

            expected_output_dir = temp_path / 'config_output'
            self.assertTrue(expected_output_dir.exists())
            self.assertTrue((expected_output_dir / 'flow_sample_component.csv').exists())

    def test_write_output_default_excel_creates_single_file(self):
        '''Default excel mode should write a single xlsx in the input directory.'''
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / 'config.toml'
            input_path.write_text('', encoding='utf-8')

            write_output({'flow': [self._sample_result()]}, str(input_path), None, True)

            self.assertTrue((temp_path / 'config_output.xlsx').exists())
            self.assertFalse((temp_path / 'config_output').exists())

    def test_write_output_custom_directory_overrides_default(self):
        '''Provided output directory should be used instead of any default path logic.'''
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / 'config.toml'
            custom_output_dir = temp_path / 'custom_out'
            input_path.write_text('', encoding='utf-8')

            write_output({'flow': [self._sample_result()]}, str(input_path), str(custom_output_dir), False)

            self.assertTrue(custom_output_dir.exists())
            self.assertTrue((custom_output_dir / 'flow_sample_component.csv').exists())
            self.assertFalse((temp_path / 'config_output').exists())

    def test_write_output_csv_filenames_are_deterministic_for_multiple_results(self):
        '''CSV filenames should be stable and include scenario and component context.'''
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / 'config.toml'
            input_path.write_text('', encoding='utf-8')

            scenario_results = {
                'flow': [
                    self._sample_result('alpha component'),
                    self._sample_result('beta_component'),
                ]
            }
            write_output(scenario_results, str(input_path), None, False)

            expected_output_dir = temp_path / 'config_output'
            filenames = sorted(path.name for path in expected_output_dir.glob('*.csv'))
            self.assertEqual(
                filenames,
                [
                    'flow_alpha_component.csv',
                    'flow_beta_component.csv',
                ],
            )

    def test_write_output_multi_scenario_produces_per_scenario_files(self):
        '''Multi-scenario runs produce one file per scenario×component, labeled by scenario.'''
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / 'config.toml'
            input_path.write_text('', encoding='utf-8')

            scenario_results = {
                'scenario_a': [self._sample_result('comp1')],
                'scenario_b': [self._sample_result('comp1')],
            }
            write_output(scenario_results, str(input_path), None, False)

            expected_output_dir = temp_path / 'config_output'
            filenames = sorted(path.name for path in expected_output_dir.glob('*.csv'))
            self.assertEqual(
                filenames,
                [
                    'scenario_a_comp1.csv',
                    'scenario_b_comp1.csv',
                ],
            )

    def test_write_output_overwrite_default_replaces_existing_file(self):
        '''Default overwrite=True should silently replace existing output files.'''
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / 'config.toml'
            input_path.write_text('', encoding='utf-8')

            result = self._sample_result('alpha component')
            write_output({'flow': [result]}, str(input_path), None, False)
            write_output({'flow': [result]}, str(input_path), None, False)

            expected_output_dir = temp_path / 'config_output'
            csv_files = list(expected_output_dir.glob('*.csv'))
            self.assertEqual(len(csv_files), 1)
            self.assertEqual(csv_files[0].name, 'flow_alpha_component.csv')

    def test_write_output_no_overwrite_appends_suffix(self):
        '''no-overwrite mode should append __1 suffix instead of replacing existing files.'''
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / 'config.toml'
            input_path.write_text('', encoding='utf-8')

            result = self._sample_result('alpha component')
            write_output({'flow': [result]}, str(input_path), None, False, overwrite=False)
            write_output({'flow': [result]}, str(input_path), None, False, overwrite=False)

            expected_output_dir = temp_path / 'config_output'
            self.assertTrue((expected_output_dir / 'flow_alpha_component.csv').exists())
            self.assertTrue((expected_output_dir / 'flow_alpha_component__1.csv').exists())

    def test_write_output_metric_options_wired_to_summary(self):
        '''metric_options.mode should control the computed summary values.'''
        import pandas as pd  # noqa: PLC0415 - local import matches existing test style

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / 'config.toml'
            input_path.write_text('', encoding='utf-8')

            result = self._sample_result('sample_component')
            write_output(
                {'flow': [result]}, str(input_path), None, False,
                metric_options=MetricOptions(mode=MetricMode.PERCENTAGE),
            )

            expected_output_dir = temp_path / 'config_output'
            summary_df = pd.read_excel(
                expected_output_dir / 'sample_component_summary.xlsx',
                sheet_name='sample_component', index_col=0,
            )
            # 2 of 3 timesteps succeed -> portion 2/3 -> percentage ~66.67.
            self.assertAlmostEqual(summary_df.loc['total', 'flow'], 200 / 3)
