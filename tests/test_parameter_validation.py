'''Tests for issue #6: deterministic parameter validation (presence/type/range).

Covers:
  - Empty/missing metrics (PARSER_MISSING_FIELD) for every characteristic.
  - Type errors (PARSER_INVALID_TYPE) per characteristic.
  - Range errors (PARSER_INVALID_VALUE) per characteristic.
  - Positive-path boundary values that must pass unchanged.
  - Parity: load_components (CLI entry path) produces the same error codes as
    calling parsers directly (direct Python API path).
'''
import unittest

from hydropattern.cli import load_components
from hydropattern.errors import HydropatternError, ParserErrorCode
from hydropattern.parsers import (
    duration_parser,
    magnitude_parser,
    rate_of_change_parser,
    timing_parser,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _components(characteristic: str, metrics: list) -> dict:
    '''Build a minimal load_components-compatible dict for parity tests.'''
    return {'components': {'test_component': {characteristic: metrics}}}


# ===========================================================================
# 1. Empty metrics raise PARSER_MISSING_FIELD
# ===========================================================================

class TestEmptyMetricsRaiseMissingField(unittest.TestCase):
    '''Empty metrics list raises PARSER_MISSING_FIELD for every characteristic.'''

    def _assert_missing_field(self, fn):
        with self.assertRaises(HydropatternError) as ctx:
            fn()
        self.assertEqual(
            ctx.exception.envelope.code,
            str(ParserErrorCode.MISSING_FIELD),
        )

    def test_timing_empty(self):
        self._assert_missing_field(lambda: timing_parser([], order=1))

    def test_magnitude_empty(self):
        self._assert_missing_field(lambda: magnitude_parser([], order=1))

    def test_duration_empty(self):
        self._assert_missing_field(lambda: duration_parser([], order=2))

    def test_rate_of_change_empty(self):
        self._assert_missing_field(lambda: rate_of_change_parser([], order=1))


# ===========================================================================
# 2. Timing validation
# ===========================================================================

class TestTimingTypeErrors(unittest.TestCase):

    def test_float_doy_raises_invalid_type(self):
        with self.assertRaises(HydropatternError) as ctx:
            timing_parser([1.0, 100], order=1)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_TYPE))

    def test_string_doy_raises_invalid_type(self):
        with self.assertRaises(HydropatternError) as ctx:
            timing_parser(['jan', 100], order=1)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_TYPE))


class TestTimingRangeErrors(unittest.TestCase):

    def test_zero_first_doy_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            timing_parser([0, 100], order=1)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_367_last_doy_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            timing_parser([1, 367], order=1)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_negative_doy_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            timing_parser([-1, 100], order=1)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_wrong_length_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            timing_parser([1, 100, 200], order=1)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))


class TestTimingPositivePaths(unittest.TestCase):

    def test_standard_window(self):
        char = timing_parser([305, 335], order=1)
        self.assertIsNotNone(char)
        self.assertIn('305', char.name)

    def test_full_year(self):
        char = timing_parser([1, 366], order=1)
        self.assertIsNotNone(char)

    def test_single_day_window(self):
        '''first_doy == last_doy evaluates exactly 1 day per year.'''
        char = timing_parser([180, 180], order=1)
        self.assertIsNotNone(char)
        self.assertIn('180', char.name)

    def test_wrap_around_window(self):
        '''first_doy > last_doy represents a cross-year window.'''
        char = timing_parser([335, 60], order=1)
        self.assertIsNotNone(char)
        self.assertIn('335', char.name)
        self.assertIn('60', char.name)

    def test_wrap_around_includes_boundary_days(self):
        '''Cross-year window includes both boundary days.'''
        char = timing_parser([335, 60], order=1)
        import pandas as pd
        # Build a minimal dataframe with a dowy column
        # Rows: doy 335, 60, 180 — first two should be in-window, last should not
        df = pd.DataFrame({'dowy': [335, 60, 180]})
        result = char.fx(df)
        self.assertEqual(result[0], 1, 'doy 335 should be in wrap-around window')
        self.assertEqual(result[1], 1, 'doy 60 should be in wrap-around window')
        self.assertEqual(result[2], 0, 'doy 180 should NOT be in wrap-around window')


# ===========================================================================
# 3. Magnitude validation
# ===========================================================================

class TestMagnitudeRangeErrors(unittest.TestCase):

    def test_negative_simple_value_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            magnitude_parser(['>', -0.1], order=1)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_negative_between_min_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            magnitude_parser([-1.0, 5.0], order=1)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_ma_periods_zero_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            magnitude_parser(['>', 1.0, 0], order=1)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_ma_periods_negative_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            magnitude_parser(['>', 1.0, -1], order=1)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))


class TestMagnitudePositivePaths(unittest.TestCase):

    def test_simple_value_zero_is_valid(self):
        char = magnitude_parser(['>', 0.0], order=1)
        self.assertIsNotNone(char)

    def test_between_zero_to_positive_is_valid(self):
        char = magnitude_parser([0.0, 5.0], order=1)
        self.assertIsNotNone(char)

    def test_ma_periods_one_is_valid(self):
        char = magnitude_parser(['>', 1.0, 1], order=1)
        self.assertIsNotNone(char)


# ===========================================================================
# 4. Duration validation
# ===========================================================================

class TestDurationRangeErrors(unittest.TestCase):

    def test_zero_timesteps_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            duration_parser(['>', 0], order=2)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_negative_timesteps_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            duration_parser(['>', -1], order=2)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_float_timesteps_raises_invalid_type(self):
        with self.assertRaises(HydropatternError) as ctx:
            duration_parser(['>', 1.5], order=2)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_TYPE))

    def test_between_float_values_raise_invalid_type(self):
        with self.assertRaises(HydropatternError) as ctx:
            duration_parser([1.5, 5.5], order=2)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_TYPE))

    def test_between_zero_min_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            duration_parser([0, 5], order=2)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))


class TestDurationPositivePaths(unittest.TestCase):

    def test_simple_one_timestep_is_valid(self):
        char = duration_parser(['>', 1], order=2)
        self.assertIsNotNone(char)

    def test_between_one_to_five_is_valid(self):
        char = duration_parser([1, 5], order=2)
        self.assertIsNotNone(char)


# ===========================================================================
# 5. Rate of change validation
# ===========================================================================

class TestRateOfChangeRangeErrors(unittest.TestCase):

    def test_zero_value_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            rate_of_change_parser(['>', 0.0], order=1)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_negative_value_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            rate_of_change_parser(['>', -1.0], order=1)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_ma_periods_zero_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            rate_of_change_parser(['>', 1.0, 0], order=1)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_look_back_zero_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            rate_of_change_parser(['>', 1.0, 1, 0], order=1)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_min_negative_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            rate_of_change_parser(['>', 1.0, 1, 1, -0.1], order=1)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_between_zero_lower_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            rate_of_change_parser([0.0, 2.0], order=1)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))


class TestRateOfChangePositivePaths(unittest.TestCase):

    def test_small_positive_value_is_valid(self):
        char = rate_of_change_parser(['>', 0.001], order=1)
        self.assertIsNotNone(char)

    def test_min_zero_is_valid(self):
        char = rate_of_change_parser(['>', 1.0, 1, 1, 0.0], order=1)
        self.assertIsNotNone(char)

    def test_all_optional_params_at_min_boundary(self):
        char = rate_of_change_parser(['>', 1.0, 1, 1], order=1)
        self.assertIsNotNone(char)

    def test_between_form_positive_values_valid(self):
        char = rate_of_change_parser([0.5, 2.0], order=1)
        self.assertIsNotNone(char)


# ===========================================================================
# 6. Parity tests: load_components (CLI path) == direct parser API path
# ===========================================================================

class TestParityCliPathVsDirectApi(unittest.TestCase):
    '''Prove no forked validation exists in cli.py.

    For each invalid input, both load_components (the CLI's component-parsing
    entry point) and the direct parser function must produce the same
    HydropatternError code.
    '''

    def _direct_code(self, fn) -> str:
        with self.assertRaises(HydropatternError) as ctx:
            fn()
        return ctx.exception.envelope.code

    def _cli_code(self, characteristic: str, metrics: list) -> str:
        with self.assertRaises(HydropatternError) as ctx:
            load_components(_components(characteristic, metrics))
        return ctx.exception.envelope.code

    def test_timing_empty_parity(self):
        direct = self._direct_code(lambda: timing_parser([], order=1))
        cli = self._cli_code('timing', [])
        self.assertEqual(direct, cli)

    def test_timing_range_error_parity(self):
        direct = self._direct_code(lambda: timing_parser([0, 100], order=1))
        cli = self._cli_code('timing', [0, 100])
        self.assertEqual(direct, cli)

    def test_magnitude_empty_parity(self):
        direct = self._direct_code(lambda: magnitude_parser([], order=1))
        cli = self._cli_code('magnitude', [])
        self.assertEqual(direct, cli)

    def test_magnitude_negative_value_parity(self):
        direct = self._direct_code(lambda: magnitude_parser(['>', -1.0], order=1))
        cli = self._cli_code('magnitude', ['>', -1.0])
        self.assertEqual(direct, cli)

    def test_duration_empty_parity(self):
        direct = self._direct_code(lambda: duration_parser([], order=2))
        cli = self._cli_code('duration', [])
        self.assertEqual(direct, cli)

    def test_duration_zero_timesteps_parity(self):
        direct = self._direct_code(lambda: duration_parser(['>', 0], order=2))
        cli = self._cli_code('duration', ['>', 0])
        self.assertEqual(direct, cli)

    def test_rate_of_change_empty_parity(self):
        direct = self._direct_code(lambda: rate_of_change_parser([], order=1))
        cli = self._cli_code('rate_of_change', [])
        self.assertEqual(direct, cli)

    def test_rate_of_change_non_positive_value_parity(self):
        direct = self._direct_code(lambda: rate_of_change_parser(['>', 0.0], order=1))
        cli = self._cli_code('rate_of_change', ['>', 0.0])
        self.assertEqual(direct, cli)


if __name__ == '__main__':
    unittest.main()
