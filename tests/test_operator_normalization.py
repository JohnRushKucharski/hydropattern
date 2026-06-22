'''Tests for operator/value format normalization (issue #5).

Behavioral delta: whitespace-padded symbols (e.g. " > ") previously raised
UNKNOWN_COMPARISON_SYMBOL; they now normalize to the stripped symbol and succeed.
Symbols that are not in the valid set still raise UNKNOWN_COMPARISON_SYMBOL.
'''
import unittest

from hydropattern.errors import HydropatternError, ParserErrorCode
from hydropattern.parsers import (
    duration_parser,
    frequency_parser,
    magnitude_parser,
    normalize_operator,
    rate_of_change_parser,
    validate_symbol,
)


class TestNormalizeOperator(unittest.TestCase):
    '''Unit tests for normalize_operator.'''

    def test_exact_symbols_pass_through(self):
        '''Exact valid symbols return unchanged.'''
        for sym in ('<', '<=', '>', '>=', '=', '!='):
            with self.subTest(sym=sym):
                self.assertEqual(normalize_operator(sym), sym)

    def test_leading_whitespace_stripped(self):
        self.assertEqual(normalize_operator(' >'), '>')
        self.assertEqual(normalize_operator('  <='), '<=')
        self.assertEqual(normalize_operator('\t!='), '!=')

    def test_trailing_whitespace_stripped(self):
        self.assertEqual(normalize_operator('> '), '>')
        self.assertEqual(normalize_operator('>= '), '>=')

    def test_surrounding_whitespace_stripped(self):
        for sym in ('<', '<=', '>', '>=', '=', '!='):
            with self.subTest(sym=sym):
                self.assertEqual(normalize_operator(f'  {sym}  '), sym)

    def test_invalid_symbol_raises_canonical_error(self):
        '''Unrecognized symbols raise HydropatternError with UNKNOWN_COMPARISON_SYMBOL.'''
        for bad in ('gt', 'gte', 'lt', 'lte', 'eq', 'ne', '==', '<>', 'invalid', ''):
            with self.subTest(bad=bad):
                with self.assertRaises(HydropatternError) as ctx:
                    normalize_operator(bad)
                self.assertEqual(
                    ctx.exception.envelope.code,
                    str(ParserErrorCode.UNKNOWN_COMPARISON_SYMBOL),
                )

    def test_whitespace_only_string_raises(self):
        with self.assertRaises(HydropatternError) as ctx:
            normalize_operator('   ')
        self.assertEqual(
            ctx.exception.envelope.code,
            str(ParserErrorCode.UNKNOWN_COMPARISON_SYMBOL),
        )


class TestValidateSymbol(unittest.TestCase):
    '''validate_symbol delegates to normalize_operator.'''

    def test_returns_stripped_symbol(self):
        self.assertEqual(validate_symbol(' >= '), '>=')

    def test_invalid_raises(self):
        with self.assertRaises(HydropatternError):
            validate_symbol('gt')


class TestParserNormalizationIntegration(unittest.TestCase):
    '''Whitespace-padded operators accepted end-to-end through characteristic parsers.'''

    def test_magnitude_parser_accepts_padded_operator(self):
        '''magnitude_parser normalizes " > " to ">".'''
        char = magnitude_parser([' > ', 10.0], order=1)
        self.assertIsNotNone(char)
        self.assertIn('gt', char.name)

    def test_magnitude_parser_accepts_all_padded_symbols(self):
        for sym, expected_name_part in (
            (' < ', 'lt'),
            (' <= ', 'le'),
            (' > ', 'gt'),
            (' >= ', 'ge'),
            (' = ', 'eq'),
            (' != ', 'ne'),
        ):
            with self.subTest(sym=sym):
                char = magnitude_parser([sym, 5.0], order=1)
                self.assertIn(expected_name_part, char.name)

    def test_duration_parser_accepts_padded_operator(self):
        char = duration_parser([' >= ', 3], order=2)
        self.assertIsNotNone(char)
        self.assertIn('ge', char.name)

    def test_rate_of_change_parser_accepts_padded_operator(self):
        char = rate_of_change_parser([' <= ', 2.0], order=1)
        self.assertIsNotNone(char)
        self.assertIn('le', char.name)

    def test_frequency_parser_accepts_padded_operator(self):
        char = frequency_parser([' >= ', 0.5, 5], order=2)
        self.assertIsNotNone(char)
        self.assertIn('ge', char.name)

    def test_invalid_operator_still_raises(self):
        '''Parser rejects non-symbol strings even after strip.'''
        with self.assertRaises(HydropatternError) as ctx:
            magnitude_parser(['gt', 10.0], order=1)
        self.assertEqual(
            ctx.exception.envelope.code,
            str(ParserErrorCode.UNKNOWN_COMPARISON_SYMBOL),
        )

    def test_regression_exact_symbols_unchanged(self):
        '''Baseline: exact symbols still work as before.'''
        for sym in ('<', '<=', '>', '>=', '=', '!='):
            with self.subTest(sym=sym):
                char = magnitude_parser([sym, 1.0], order=1)
                self.assertIsNotNone(char)


if __name__ == '__main__':
    unittest.main()
