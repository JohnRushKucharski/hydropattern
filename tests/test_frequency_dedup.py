'''RED->GREEN characterization for issue #28: builders.py's FREQUENCY branch and
_build_nested_frequency_characteristics must delegate to parsers.frequency_parser /
parsers.nested_frequency_parser instead of reimplementing name/fx construction.
'''
# pylint: disable=missing-class-docstring,missing-function-docstring
import unittest
from unittest.mock import patch

from hydropattern import parsers, patterns
from hydropattern.parsing import builders
from hydropattern.parsing.builders import build_components
from hydropattern.parsers import parse_request


class TestFrequencyDelegatesToFrequencyParser(unittest.TestCase):

    def test_count_form_delegates_to_frequency_parser(self):
        sentinel = patterns.Characteristic(
            name='sentinel', fx=lambda df, output: df, type=patterns.CharacteristicType.FREQUENCY
        )
        with patch.object(
            parsers, 'frequency_parser', return_value=sentinel
        ) as mock_parser:
            request = parse_request({'comp': {'magnitude': ['>', 5.0], 'frequency': ['>', 1, 3]}})
            components = build_components(request)

        mock_parser.assert_called_once_with(['>', 1, 3], order=2)
        self.assertIs(components[0].characteristics[1], sentinel)

    def test_between_form_delegates_to_frequency_parser(self):
        sentinel = patterns.Characteristic(
            name='sentinel', fx=lambda df, output: df, type=patterns.CharacteristicType.FREQUENCY
        )
        with patch.object(
            parsers, 'frequency_parser', return_value=sentinel
        ) as mock_parser:
            request = parse_request({'comp': {'magnitude': ['>', 5.0], 'frequency': [1, 3, 5]}})
            components = build_components(request)

        mock_parser.assert_called_once_with([1, 3, 5], order=2)
        self.assertIs(components[0].characteristics[1], sentinel)

    def test_event_bool_forwarded_when_not_default(self):
        sentinel = patterns.Characteristic(
            name='sentinel', fx=lambda df, output: df, type=patterns.CharacteristicType.FREQUENCY
        )
        with patch.object(
            parsers, 'frequency_parser', return_value=sentinel
        ) as mock_parser:
            request = parse_request(
                {'comp': {'magnitude': ['>', 5.0], 'frequency': ['>', 1, 3, False]}}
            )
            components = build_components(request)

        mock_parser.assert_called_once_with(['>', 1, 3, False], order=2)
        self.assertIs(components[0].characteristics[1], sentinel)


class TestNestedFrequencyDelegatesToNestedFrequencyParser(unittest.TestCase):

    def test_nested_frequency_delegates_to_nested_frequency_parser(self):
        base = patterns.Characteristic(
            name='base', fx=lambda df, output: df, type=patterns.CharacteristicType.FREQUENCY
        )
        nested = patterns.Characteristic(
            name='nested', fx=lambda df, output: df, type=patterns.CharacteristicType.FREQUENCY
        )
        with patch.object(
            builders, 'nested_frequency_parser', return_value=[base, nested]
        ) as mock_parser:
            request = parse_request(
                {'comp': {'magnitude': ['>', 5.0], 'frequency': [['>', 0.5], ['>', 1, 2]]}}
            )
            components = build_components(request)

        mock_parser.assert_called_once_with([['>', 0.5], ['>', 1, 2]], 2)
        self.assertIs(components[0].characteristics[1], base)
        self.assertIs(components[0].characteristics[2], nested)


if __name__ == '__main__':
    unittest.main()
