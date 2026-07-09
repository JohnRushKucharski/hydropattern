'''Compatibility tests for new hydropattern.parsing facade modules.'''
# pylint: disable=missing-function-docstring

import unittest

from hydropattern import parsers
from hydropattern.parsing import builders, options, requests, specs, timeseries


class TestParsingFacade(unittest.TestCase):
    '''New parsing package re-exports existing parser interfaces unchanged.'''

    def test_specs_module_reexports_existing_types(self):
        self.assertIs(specs.Request, parsers.Request)
        self.assertIs(specs.ComponentSpec, parsers.ComponentSpec)
        self.assertIs(specs.CharacteristicSpec, parsers.CharacteristicSpec)
        self.assertIs(specs.MetricMode, parsers.MetricMode)
        self.assertIs(specs.OutputOptions, parsers.OutputOptions)
        self.assertIs(specs.TimeseriesSpec, parsers.TimeseriesSpec)

    def test_options_module_reexports_existing_option_parsers(self):
        metric = options.parse_metric_options({'mode': 'percentage'})
        self.assertEqual(metric.mode, parsers.MetricMode.PERCENTAGE)
        output = options.parse_output_options({'output': {'excel': False}})
        self.assertFalse(output.excel)

    def test_timeseries_module_reexports_existing_timeseries_parser(self):
        spec = timeseries.parse_timeseries_spec({'timeseries': {'path': 'x.csv'}})
        self.assertEqual(spec.path, 'x.csv')
        self.assertEqual(spec.first_day_of_water_year, 1)
        self.assertEqual(spec.date_format, '')

    def test_requests_module_reexports_existing_request_parser(self):
        self.assertEqual(requests.parse_request.__module__, 'hydropattern.parsing.requests')
        data = {'component': {'magnitude': [' > ', 1.0]}}
        self.assertEqual(requests.parse_request(data), parsers.parse_request(data))

    def test_builders_module_reexports_existing_component_builder(self):
        self.assertEqual(builders.build_components.__module__, 'hydropattern.parsing.builders')
        request = parsers.parse_request({'comp': {'magnitude': ['>', 1.0]}})
        built = builders.build_components(request)
        facade_built = parsers.build_components(request)
        self.assertEqual([c.name for c in built], [c.name for c in facade_built])
        self.assertEqual(
            [[ch.name for ch in c.characteristics] for c in built],
            [[ch.name for ch in c.characteristics] for c in facade_built],
        )


if __name__ == '__main__':
    unittest.main()
