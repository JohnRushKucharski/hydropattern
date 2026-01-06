'''Tests for CLI module functionality.'''

import unittest
from pathlib import Path

from hydropattern.cli import load_config_file


class TestCLI(unittest.TestCase):
    '''Tests for CLI functions.'''

    def setUp(self):
        '''Set up test fixtures.'''
        self.test_files_dir = Path(__file__).parent / "test_files"
        self.test_toml_path = self.test_files_dir / "key_order_test.toml"
        self.reverse_order_path = self.test_files_dir / "reverse_order_test.toml"

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
