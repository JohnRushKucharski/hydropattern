'''Tests for issue #24: build_components delegates MAGNITUDE construction to
parsers.magnitude_parser instead of reimplementing name/fx construction.
'''
# pylint: disable=missing-class-docstring,missing-function-docstring
import unittest
from unittest.mock import patch

from hydropattern import parsers
from hydropattern.parsing.builders import build_components


class TestMagnitudeDelegatesToMagnitudeParser(unittest.TestCase):

    def test_simple_form_delegates_to_magnitude_parser(self):
        request = parsers.parse_request({'comp': {'magnitude': ['>', 5.0]}})
        sentinel = parsers.magnitude_parser(['>', 5.0], order=1)
        with patch.object(parsers, 'magnitude_parser', return_value=sentinel) as mock_parser:
            components = build_components(request)
        mock_parser.assert_called_once_with(['>', 5.0], order=1)
        self.assertIs(components[0].characteristics[0], sentinel)

    def test_between_form_delegates_to_magnitude_parser(self):
        request = parsers.parse_request({'comp': {'magnitude': [1.0, 5.0]}})
        sentinel = parsers.magnitude_parser([1.0, 5.0], order=1)
        with patch.object(parsers, 'magnitude_parser', return_value=sentinel) as mock_parser:
            components = build_components(request)
        mock_parser.assert_called_once_with([1.0, 5.0], order=1)
        self.assertIs(components[0].characteristics[0], sentinel)

    def test_ma_periods_forwarded_when_not_default(self):
        request = parsers.parse_request({'comp': {'magnitude': ['>', 5.0, 3]}})
        sentinel = parsers.magnitude_parser(['>', 5.0, 3], order=1)
        with patch.object(parsers, 'magnitude_parser', return_value=sentinel) as mock_parser:
            build_components(request)
        mock_parser.assert_called_once_with(['>', 5.0, 3], order=1)


if __name__ == '__main__':
    unittest.main()
