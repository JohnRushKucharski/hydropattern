'''Tests for issue #27: build_components delegates TIMING construction to
parsers.timing_parser instead of reimplementing name/fx construction.
'''
# pylint: disable=missing-class-docstring,missing-function-docstring
import unittest
from unittest.mock import patch

from hydropattern import parsers
from hydropattern.parsing.builders import build_components


class TestTimingDelegatesToTimingParser(unittest.TestCase):

    def test_standard_window_delegates_to_timing_parser(self):
        request = parsers.parse_request({'comp': {'timing': [100, 200]}})
        sentinel = parsers.timing_parser([100, 200], order=1)
        with patch.object(parsers, 'timing_parser', return_value=sentinel) as mock_parser:
            components = build_components(request)
        mock_parser.assert_called_once_with([100, 200], order=1)
        self.assertIs(components[0].characteristics[0], sentinel)

    def test_wrap_around_window_delegates_to_timing_parser(self):
        request = parsers.parse_request({'comp': {'timing': [335, 60]}})
        sentinel = parsers.timing_parser([335, 60], order=1)
        with patch.object(parsers, 'timing_parser', return_value=sentinel) as mock_parser:
            components = build_components(request)
        mock_parser.assert_called_once_with([335, 60], order=1)
        self.assertIs(components[0].characteristics[0], sentinel)


if __name__ == '__main__':
    unittest.main()
