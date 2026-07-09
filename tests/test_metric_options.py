'''Tests for the parse_metric_options section parser (used by [output.metric]).'''
# pylint: disable=missing-function-docstring,line-too-long

import unittest

from hydropattern.errors import HydropatternError, ParserErrorCode
from hydropattern.parsers import MetricMode, MetricOptions, parse_metric_options


class TestParseMetricOptionsDefaults(unittest.TestCase):
    '''Defaults when the section is absent (None) or empty.'''

    def test_none_section_defaults_to_portion(self):
        '''section=None (section absent from config) -> MetricOptions(mode=PORTION).'''
        opts = parse_metric_options(None)
        self.assertEqual(opts, MetricOptions(mode=MetricMode.PORTION))

    def test_empty_section_defaults_to_portion(self):
        '''section present but empty -> mode defaults to portion.'''
        opts = parse_metric_options({})
        self.assertEqual(opts.mode, MetricMode.PORTION)


class TestParseMetricOptionsValidModes(unittest.TestCase):
    '''Each documented mode string maps deterministically to a MetricMode.'''

    def test_portion_mode(self):
        opts = parse_metric_options({'mode': 'portion'})
        self.assertEqual(opts.mode, MetricMode.PORTION)

    def test_percentage_mode(self):
        opts = parse_metric_options({'mode': 'percentage'})
        self.assertEqual(opts.mode, MetricMode.PERCENTAGE)

    def test_return_period_mode(self):
        opts = parse_metric_options({'mode': 'return_period'})
        self.assertEqual(opts.mode, MetricMode.RETURN_PERIOD)


class TestParseMetricOptionsInvalid(unittest.TestCase):
    '''Invalid configuration inputs raise deterministic, machine-readable errors.'''

    def test_unknown_mode_value_raises_invalid_value(self):
        '''Unrecognized mode string -> PARSER_INVALID_VALUE.'''
        with self.assertRaises(HydropatternError) as ctx:
            parse_metric_options({'mode': 'average'})
        self.assertEqual(ctx.exception.envelope.code, ParserErrorCode.INVALID_VALUE)

    def test_non_string_mode_raises_invalid_value(self):
        '''Non-string mode value -> PARSER_INVALID_VALUE.'''
        with self.assertRaises(HydropatternError) as ctx:
            parse_metric_options({'mode': 1})
        self.assertEqual(ctx.exception.envelope.code, ParserErrorCode.INVALID_VALUE)

    def test_non_table_section_raises_invalid_type(self):
        '''section not a table -> PARSER_INVALID_TYPE.'''
        with self.assertRaises(HydropatternError) as ctx:
            parse_metric_options('portion')
        self.assertEqual(ctx.exception.envelope.code, ParserErrorCode.INVALID_TYPE)

    def test_unknown_key_raises_unknown_option(self):
        '''Unrecognized key within the section -> PARSER_UNKNOWN_OPTION.'''
        with self.assertRaises(HydropatternError) as ctx:
            parse_metric_options({'modee': 'portion'})
        self.assertEqual(ctx.exception.envelope.code, ParserErrorCode.UNKNOWN_OPTION)

    def test_error_context_includes_section_and_field(self):
        '''Error envelope context identifies the offending section/field.'''
        with self.assertRaises(HydropatternError) as ctx:
            parse_metric_options({'mode': 'bogus'})
        self.assertEqual(ctx.exception.envelope.context.get('section'), 'metric')
        self.assertEqual(ctx.exception.envelope.context.get('field'), 'mode')

    def test_custom_section_name_used_in_error_context(self):
        '''section_name overrides the default 'metric' label in error context (e.g. output.metric).'''
        with self.assertRaises(HydropatternError) as ctx:
            parse_metric_options({'mode': 'bogus'}, section_name='output.metric')
        self.assertEqual(ctx.exception.envelope.context.get('section'), 'output.metric')


class TestMetricOptionsEquivalence(unittest.TestCase):
    '''Equivalent valid configs produce equal, comparable MetricOptions.'''

    def test_equal_configs_produce_equal_options(self):
        '''Two independently-parsed configs with the same mode are equal (==).'''
        a = parse_metric_options({'mode': 'percentage'})
        b = parse_metric_options({'mode': 'percentage'})
        self.assertEqual(a, b)

    def test_default_equals_explicit_portion(self):
        '''Omitting the section is equivalent to explicitly requesting portion.'''
        default = parse_metric_options(None)
        explicit = parse_metric_options({'mode': 'portion'})
        self.assertEqual(default, explicit)


if __name__ == '__main__':
    unittest.main()
