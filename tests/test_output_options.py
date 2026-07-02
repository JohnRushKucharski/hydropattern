'''Tests for the [output] TOML section parser contract (directory/overwrite/excel,
[output.metric], [output.plot], [output.plot.climate-canvas]).
'''

import unittest

from hydropattern.errors import HydropatternError, ParserErrorCode
from hydropattern.parsers import (
    ClimateCanvasPlotOptions,
    MetricMode,
    MetricOptions,
    OutputOptions,
    PlotOptions,
    parse_output_options,
)


class TestParseOutputOptionsDefaults(unittest.TestCase):
    '''Defaults when [output] (and all nested sections) are absent.'''

    def test_absent_section_returns_all_defaults(self):
        opts = parse_output_options({})
        self.assertEqual(opts, OutputOptions(
            directory=None,
            overwrite=True,
            excel=True,
            metric=MetricOptions(mode=MetricMode.PORTION),
            plot=PlotOptions(
                enabled=False,
                climate_canvas=ClimateCanvasPlotOptions(
                    interpolate=True,
                    show=False,
                    title=None,
                    xlabel='Precipitation Delta (%)',
                    ylabel='Temperature Delta (C)',
                    zlabel=None,
                ),
            ),
        ))

    def test_empty_output_section_returns_all_defaults(self):
        opts = parse_output_options({'output': {}})
        self.assertEqual(opts, OutputOptions())


class TestParseOutputOptionsTopLevelOverrides(unittest.TestCase):
    '''[output] directory/overwrite/excel overrides.'''

    def test_directory_override(self):
        opts = parse_output_options({'output': {'directory': 'out/'}})
        self.assertEqual(opts.directory, 'out/')

    def test_overwrite_override(self):
        opts = parse_output_options({'output': {'overwrite': False}})
        self.assertFalse(opts.overwrite)

    def test_excel_override(self):
        opts = parse_output_options({'output': {'excel': False}})
        self.assertFalse(opts.excel)


class TestParseOutputOptionsMetric(unittest.TestCase):
    '''[output.metric] is delegated to parse_metric_options.'''

    def test_metric_mode_override(self):
        opts = parse_output_options({'output': {'metric': {'mode': 'percentage'}}})
        self.assertEqual(opts.metric.mode, MetricMode.PERCENTAGE)


class TestParseOutputOptionsPlot(unittest.TestCase):
    '''[output.plot] enabled + nested [output.plot.climate-canvas].'''

    def test_plot_enabled_override(self):
        opts = parse_output_options({'output': {'plot': {'enabled': True}}})
        self.assertTrue(opts.plot.enabled)

    def test_climate_canvas_all_overrides(self):
        opts = parse_output_options({'output': {'plot': {'climate-canvas': {
            'interpolate': False,
            'show': True,
            'title': 'My Title',
            'xlabel': 'X',
            'ylabel': 'Y',
            'zlabel': 'Z',
        }}}})
        cc = opts.plot.climate_canvas
        self.assertEqual(cc, ClimateCanvasPlotOptions(
            interpolate=False, show=True, title='My Title',
            xlabel='X', ylabel='Y', zlabel='Z',
        ))


class TestParseOutputOptionsErrors(unittest.TestCase):
    '''Unrecognized keys / bad types raise deterministic, machine-readable errors.'''

    def test_unknown_output_key_raises_unknown_option(self):
        with self.assertRaises(HydropatternError) as ctx:
            parse_output_options({'output': {'bogus': 1}})
        self.assertEqual(ctx.exception.envelope.code, ParserErrorCode.UNKNOWN_OPTION)
        self.assertEqual(ctx.exception.envelope.context.get('section'), 'output')

    def test_non_table_output_section_raises_invalid_type(self):
        with self.assertRaises(HydropatternError) as ctx:
            parse_output_options({'output': 'nope'})
        self.assertEqual(ctx.exception.envelope.code, ParserErrorCode.INVALID_TYPE)

    def test_non_bool_overwrite_raises_invalid_type(self):
        with self.assertRaises(HydropatternError) as ctx:
            parse_output_options({'output': {'overwrite': 'yes'}})
        self.assertEqual(ctx.exception.envelope.code, ParserErrorCode.INVALID_TYPE)
        self.assertEqual(ctx.exception.envelope.context.get('field'), 'overwrite')

    def test_non_bool_excel_raises_invalid_type(self):
        with self.assertRaises(HydropatternError) as ctx:
            parse_output_options({'output': {'excel': 'yes'}})
        self.assertEqual(ctx.exception.envelope.code, ParserErrorCode.INVALID_TYPE)

    def test_non_str_directory_raises_invalid_type(self):
        with self.assertRaises(HydropatternError) as ctx:
            parse_output_options({'output': {'directory': 1}})
        self.assertEqual(ctx.exception.envelope.code, ParserErrorCode.INVALID_TYPE)

    def test_unknown_plot_key_raises_unknown_option(self):
        with self.assertRaises(HydropatternError) as ctx:
            parse_output_options({'output': {'plot': {'bogus': 1}}})
        self.assertEqual(ctx.exception.envelope.code, ParserErrorCode.UNKNOWN_OPTION)
        self.assertEqual(ctx.exception.envelope.context.get('section'), 'output.plot')

    def test_non_bool_plot_enabled_raises_invalid_type(self):
        with self.assertRaises(HydropatternError) as ctx:
            parse_output_options({'output': {'plot': {'enabled': 'yes'}}})
        self.assertEqual(ctx.exception.envelope.code, ParserErrorCode.INVALID_TYPE)

    def test_unknown_climate_canvas_key_raises_unknown_option(self):
        with self.assertRaises(HydropatternError) as ctx:
            parse_output_options({'output': {'plot': {'climate-canvas': {'bogus': 1}}}})
        self.assertEqual(ctx.exception.envelope.code, ParserErrorCode.UNKNOWN_OPTION)
        self.assertEqual(
            ctx.exception.envelope.context.get('section'), 'output.plot.climate-canvas'
        )

    def test_non_bool_interpolate_raises_invalid_type(self):
        with self.assertRaises(HydropatternError) as ctx:
            parse_output_options({'output': {'plot': {'climate-canvas': {'interpolate': 1}}}})
        self.assertEqual(ctx.exception.envelope.code, ParserErrorCode.INVALID_TYPE)

    def test_non_str_title_raises_invalid_type(self):
        with self.assertRaises(HydropatternError) as ctx:
            parse_output_options({'output': {'plot': {'climate-canvas': {'title': 1}}}})
        self.assertEqual(ctx.exception.envelope.code, ParserErrorCode.INVALID_TYPE)


if __name__ == '__main__':
    unittest.main()
