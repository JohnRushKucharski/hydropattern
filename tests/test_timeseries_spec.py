'''Tests for the [timeseries] TOML section parser contract.'''

import unittest

from hydropattern.errors import HydropatternError, ParserErrorCode
from hydropattern.parsers import TimeseriesSpec, parse_timeseries_spec


class TestParseTimeseriesSpecDefaults(unittest.TestCase):
    '''Defaults when optional [timeseries] keys are omitted.'''

    def test_only_path_given_uses_defaults(self):
        '''path is the only required key; everything else defaults.'''
        spec = parse_timeseries_spec({'timeseries': {'path': 'data/flow.csv'}})
        self.assertEqual(
            spec,
            TimeseriesSpec(
                path='data/flow.csv',
                first_day_of_water_year=1,
                date_format='',
                sheet_name=0,
            ),
        )


class TestParseTimeseriesSpecOverrides(unittest.TestCase):
    '''Explicit values override defaults.'''

    def test_all_fields_overridden(self):
        spec = parse_timeseries_spec({'timeseries': {
            'path': 'data/flow.xlsx',
            'first_day_of_water_year': 274,
            'date_format': '%Y-%m-%d',
            'sheet_name': 'Sheet2',
        }})
        self.assertEqual(spec, TimeseriesSpec(
            path='data/flow.xlsx',
            first_day_of_water_year=274,
            date_format='%Y-%m-%d',
            sheet_name='Sheet2',
        ))


class TestParseTimeseriesSpecErrors(unittest.TestCase):
    '''Missing required data raises deterministic, machine-readable errors.'''

    def test_missing_section_raises(self):
        with self.assertRaises(HydropatternError) as context:
            parse_timeseries_spec({'components': {}})

        self.assertEqual(context.exception.envelope.code, ParserErrorCode.MISSING_SECTION)
        self.assertEqual(context.exception.envelope.context['section'], 'timeseries')
        self.assertEqual(context.exception.envelope.source, 'parser')

    def test_missing_path_raises(self):
        with self.assertRaises(HydropatternError) as context:
            parse_timeseries_spec({'timeseries': {'date_format': '%Y-%m-%d'}})

        self.assertEqual(context.exception.envelope.code, ParserErrorCode.MISSING_FIELD)
        self.assertEqual(context.exception.envelope.context['section'], 'timeseries')
        self.assertEqual(context.exception.envelope.context['field'], 'path')


if __name__ == '__main__':
    unittest.main()
