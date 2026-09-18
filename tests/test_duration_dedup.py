'''Tests for issue #25: build_components delegates DURATION construction to
parsers.duration_parser instead of reimplementing name/fx construction.
'''
# pylint: disable=missing-class-docstring,missing-function-docstring
import unittest
from unittest.mock import patch

from hydropattern import parsers
from hydropattern.parsing.builders import build_components


class TestDurationDelegatesToDurationParser(unittest.TestCase):

    def test_simple_form_delegates_to_duration_parser(self):
        request = parsers.parse_request(
            {'comp': {'magnitude': ['>', 5.0], 'duration': ['>', 1]}}
        )
        sentinel = parsers.duration_parser(['>', 1], order=2)
        with patch.object(parsers, 'duration_parser', return_value=sentinel) as mock_parser:
            components = build_components(request)
        mock_parser.assert_called_once_with(['>', 1], order=2)
        self.assertIs(components[0].characteristics[1], sentinel)

    def test_between_form_delegates_to_duration_parser(self):
        request = parsers.parse_request(
            {'comp': {'magnitude': ['>', 5.0], 'duration': [1, 5]}}
        )
        sentinel = parsers.duration_parser([1, 5], order=2)
        with patch.object(parsers, 'duration_parser', return_value=sentinel) as mock_parser:
            components = build_components(request)
        mock_parser.assert_called_once_with([1, 5], order=2)
        self.assertIs(components[0].characteristics[1], sentinel)


if __name__ == '__main__':
    unittest.main()
