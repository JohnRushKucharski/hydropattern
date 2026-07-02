'''Tests for resolve_output_options: CLI-flag-vs-toml precedence for [output].

CLI flags default to None (not explicitly passed). When None, the toml [output]
section's value applies (or its own default when toml is silent too). When a CLI
flag is explicitly given (not None), it always wins over the toml value.
'''

import unittest

from hydropattern.cli import resolve_output_options
from hydropattern.parsers import (
    ClimateCanvasPlotOptions,
    OutputOptions,
    PlotOptions,
)


class TestResolveOutputOptionsAllCliOmitted(unittest.TestCase):
    '''All CLI flags None (omitted) -> toml (or its defaults) applies untouched.'''

    def test_empty_config_and_no_cli_flags_returns_defaults(self):
        opts = resolve_output_options({}, plot=None, output_directory=None,
                                      write_to_excel=None, overwrite=None,
                                      interp=None, show=None)
        self.assertEqual(opts, OutputOptions())

    def test_toml_values_used_when_cli_omitted(self):
        data = {'output': {
            'directory': 'out/', 'overwrite': False, 'excel': False,
            'plot': {'enabled': True, 'climate-canvas': {'interpolate': False, 'show': True}},
        }}
        opts = resolve_output_options(data, plot=None, output_directory=None,
                                      write_to_excel=None, overwrite=None,
                                      interp=None, show=None)
        self.assertEqual(opts.directory, 'out/')
        self.assertFalse(opts.overwrite)
        self.assertFalse(opts.excel)
        self.assertTrue(opts.plot.enabled)
        self.assertFalse(opts.plot.climate_canvas.interpolate)
        self.assertTrue(opts.plot.climate_canvas.show)


class TestResolveOutputOptionsCliOverridesWin(unittest.TestCase):
    '''Explicit CLI flags override toml values, regardless of toml content.'''

    def test_cli_overrides_toml_directory_overwrite_excel(self):
        data = {'output': {'directory': 'toml_dir/', 'overwrite': False, 'excel': False}}
        opts = resolve_output_options(data, plot=None, output_directory='cli_dir/',
                                      write_to_excel=True, overwrite=True,
                                      interp=None, show=None)
        self.assertEqual(opts.directory, 'cli_dir/')
        self.assertTrue(opts.overwrite)
        self.assertTrue(opts.excel)

    def test_cli_plot_false_overrides_toml_enabled_true(self):
        data = {'output': {'plot': {'enabled': True}}}
        opts = resolve_output_options(data, plot=False, output_directory=None,
                                      write_to_excel=None, overwrite=None,
                                      interp=None, show=None)
        self.assertFalse(opts.plot.enabled)

    def test_cli_plot_true_overrides_toml_enabled_false(self):
        opts = resolve_output_options({}, plot=True, output_directory=None,
                                      write_to_excel=None, overwrite=None,
                                      interp=None, show=None)
        self.assertTrue(opts.plot.enabled)

    def test_cli_interp_and_show_override_toml_climate_canvas(self):
        data = {'output': {'plot': {'climate-canvas': {'interpolate': True, 'show': False}}}}
        opts = resolve_output_options(data, plot=None, output_directory=None,
                                      write_to_excel=None, overwrite=None,
                                      interp=False, show=True)
        self.assertFalse(opts.plot.climate_canvas.interpolate)
        self.assertTrue(opts.plot.climate_canvas.show)

    def test_title_xlabel_ylabel_zlabel_unaffected_by_cli(self):
        '''No CLI flags exist yet for these -- toml values (or defaults) pass through untouched.'''
        data = {'output': {'plot': {'climate-canvas': {'title': 'T', 'xlabel': 'X'}}}}
        opts = resolve_output_options(data, plot=None, output_directory=None,
                                      write_to_excel=None, overwrite=None,
                                      interp=None, show=None)
        self.assertEqual(opts.plot.climate_canvas.title, 'T')
        self.assertEqual(opts.plot.climate_canvas.xlabel, 'X')
        self.assertEqual(opts.plot.climate_canvas.ylabel, 'Temperature Delta (C)')


if __name__ == '__main__':
    unittest.main()
