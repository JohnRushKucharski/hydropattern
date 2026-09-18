'''Tests for issue #26: build_components delegates RATE_OF_CHANGE construction to
parsers.rate_of_change_parser instead of reimplementing name/fx construction.
'''
# pylint: disable=missing-class-docstring,missing-function-docstring
import unittest
from unittest.mock import patch

from hydropattern import parsers
from hydropattern.parsing import builders
from hydropattern.parsing.builders import build_components


class TestRateOfChangeDelegatesToRateOfChangeParser(unittest.TestCase):

    def test_simple_form_delegates_to_rate_of_change_parser(self):
        request = parsers.parse_request({'comp': {'rate_of_change': ['>', 0.5]}})
        sentinel = parsers.rate_of_change_parser(['>', 0.5], order=1)
        with patch.object(
            builders, 'rate_of_change_parser', return_value=sentinel
        ) as mock_parser:
            components = build_components(request)
        mock_parser.assert_called_once_with(['>', 0.5], order=1)
        self.assertIs(components[0].characteristics[0], sentinel)

    def test_between_form_delegates_to_rate_of_change_parser(self):
        request = parsers.parse_request({'comp': {'rate_of_change': [0.5, 2.0]}})
        sentinel = parsers.rate_of_change_parser([0.5, 2.0], order=1)
        with patch.object(
            builders, 'rate_of_change_parser', return_value=sentinel
        ) as mock_parser:
            components = build_components(request)
        mock_parser.assert_called_once_with([0.5, 2.0], order=1)
        self.assertIs(components[0].characteristics[0], sentinel)

    def test_trailing_optional_args_trimmed_when_default(self):
        request = parsers.parse_request({'comp': {'rate_of_change': ['>', 0.5, 3]}})
        sentinel = parsers.rate_of_change_parser(['>', 0.5, 3], order=1)
        with patch.object(
            builders, 'rate_of_change_parser', return_value=sentinel
        ) as mock_parser:
            build_components(request)
        mock_parser.assert_called_once_with(['>', 0.5, 3], order=1)

    def test_all_optional_args_forwarded_when_non_default(self):
        request = parsers.parse_request({'comp': {'rate_of_change': ['>', 0.5, 1, 1, 0.2]}})
        sentinel = parsers.rate_of_change_parser(['>', 0.5, 1, 1, 0.2], order=1)
        with patch.object(
            builders, 'rate_of_change_parser', return_value=sentinel
        ) as mock_parser:
            build_components(request)
        mock_parser.assert_called_once_with(['>', 0.5, 1, 1, 0.2], order=1)


if __name__ == '__main__':
    unittest.main()


