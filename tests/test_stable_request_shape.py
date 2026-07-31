'''Tests for issue #7: stable normalized internal request shape.

Covers:
  - CharacteristicSpec, ComponentSpec, Request dataclass equality.
  - parse_request golden/snapshot tests for each characteristic type.
  - Equivalence: whitespace-padded operators produce equal Request objects.
  - Negative: shape-incompatible inputs raise HydropatternError with canonical codes.
  - build_components: Request -> list[Component] conversion.
  - Round-trip: Request -> build_components -> evaluate_components produces correct results.
'''
# pylint: disable=missing-function-docstring,line-too-long
import unittest

import numpy as np
import pandas as pd

from hydropattern.errors import HydropatternError, ParserErrorCode
from hydropattern.parsers import (
    CharacteristicSpec,
    ComponentSpec,
    Request,
    build_components,
    parse_request,
)
from hydropattern.patterns import CharacteristicType, Component, evaluate_components

# ---------------------------------------------------------------------------
# Shared test DataFrame fixture
# ---------------------------------------------------------------------------

def _make_df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            'flow': [1.0, 1.2, 1.4, 0.8, 1.5],
            'dowy': [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        index=pd.to_datetime(
            ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05']
        ),
    )
    df.index.name = 'time'
    return df


# ===========================================================================
# 1. Dataclass structural equality
# ===========================================================================

class TestCharacteristicSpecEquality(unittest.TestCase):
    '''CharacteristicSpec equality uses structural field comparison (no closures).'''

    def _mag_spec(
            self,
            operator: str = '>',
            values: tuple = (1.0,),
            order: int = 1,
    ) -> CharacteristicSpec:
        return CharacteristicSpec(
            type=CharacteristicType.MAGNITUDE,
            operator=operator,
            values=values,
            order=order,
        )

    def test_identical_specs_are_equal(self):
        self.assertEqual(self._mag_spec(), self._mag_spec())

    def test_different_operator_not_equal(self):
        self.assertNotEqual(self._mag_spec(operator='>'), self._mag_spec(operator='>='))

    def test_different_values_not_equal(self):
        self.assertNotEqual(self._mag_spec(values=(1.0,)), self._mag_spec(values=(2.0,)))

    def test_different_order_not_equal(self):
        self.assertNotEqual(self._mag_spec(order=1), self._mag_spec(order=2))

    def test_between_spec_with_none_operator_equal(self):
        a = CharacteristicSpec(
            type=CharacteristicType.MAGNITUDE, operator=None, values=(0.5, 2.0), order=1
        )
        b = CharacteristicSpec(
            type=CharacteristicType.MAGNITUDE, operator=None, values=(0.5, 2.0), order=1
        )
        self.assertEqual(a, b)

    def test_different_types_not_equal(self):
        a = CharacteristicSpec(
            type=CharacteristicType.MAGNITUDE, operator='>', values=(1.0,), order=1
        )
        b = CharacteristicSpec(
            type=CharacteristicType.TIMING, operator=None, values=(1, 90), order=1
        )
        self.assertNotEqual(a, b)

    def test_different_ma_periods_not_equal(self):
        a = CharacteristicSpec(
            type=CharacteristicType.MAGNITUDE, operator='>', values=(1.0,),
            ma_periods=1, order=1,
        )
        b = CharacteristicSpec(
            type=CharacteristicType.MAGNITUDE, operator='>', values=(1.0,),
            ma_periods=3, order=1,
        )
        self.assertNotEqual(a, b)


class TestRequestEquality(unittest.TestCase):
    '''Request equality is determined by field-by-field structural comparison.'''

    def _simple_request(self) -> Request:
        return Request(components=(
            ComponentSpec(
                name='comp',
                characteristics=(
                    CharacteristicSpec(
                        type=CharacteristicType.MAGNITUDE, operator='>', values=(1.0,), order=1
                    ),
                ),
            ),
        ))

    def test_identical_requests_are_equal(self):
        self.assertEqual(self._simple_request(), self._simple_request())

    def test_different_component_name_not_equal(self):
        r1 = Request(components=(ComponentSpec(name='a', characteristics=()),))
        r2 = Request(components=(ComponentSpec(name='b', characteristics=()),))
        self.assertNotEqual(r1, r2)


# ===========================================================================
# 2. Golden / snapshot tests
# ===========================================================================

class TestParseRequestGolden(unittest.TestCase):
    '''parse_request produces the expected Request for representative valid configs.'''

    def test_magnitude_simple_golden(self):
        result = parse_request({'comp_a': {'magnitude': ['>', 1.0]}})
        expected = Request(components=(
            ComponentSpec(
                name='comp_a',
                characteristics=(
                    CharacteristicSpec(
                        type=CharacteristicType.MAGNITUDE,
                        operator='>',
                        values=(1.0,),
                        ma_periods=1,
                        look_back=1,
                        min_val=0.0,
                        order=1,
                    ),
                ),
                is_success_pattern=True,
                verbose=True,
            ),
        ))
        self.assertEqual(result, expected)

    def test_timing_golden(self):
        result = parse_request({'comp_a': {'timing': [1, 90]}})
        expected = Request(components=(
            ComponentSpec(
                name='comp_a',
                characteristics=(
                    CharacteristicSpec(
                        type=CharacteristicType.TIMING,
                        operator=None,
                        values=(1, 90),
                        order=1,
                    ),
                ),
                is_success_pattern=True,
                verbose=True,
            ),
        ))
        self.assertEqual(result, expected)

    def test_magnitude_between_golden(self):
        result = parse_request({'comp_a': {'magnitude': [0.5, 2.0]}})
        expected = Request(components=(
            ComponentSpec(
                name='comp_a',
                characteristics=(
                    CharacteristicSpec(
                        type=CharacteristicType.MAGNITUDE,
                        operator=None,
                        values=(0.5, 2.0),
                        ma_periods=1,
                        order=1,
                    ),
                ),
            ),
        ))
        self.assertEqual(result, expected)

    def test_magnitude_with_ma_periods_golden(self):
        result = parse_request({'comp_a': {'magnitude': ['>', 1.0, 3]}})
        spec = result.components[0].characteristics[0]
        self.assertEqual(spec.ma_periods, 3)
        self.assertEqual(spec.operator, '>')
        self.assertEqual(spec.values, (1.0,))

    def test_rate_of_change_with_all_optional_params_golden(self):
        result = parse_request({'comp_a': {'rate_of_change': ['>', 0.5, 2, 3, 1.0]}})
        spec = result.components[0].characteristics[0]
        self.assertEqual(spec.type, CharacteristicType.RATE_OF_CHANGE)
        self.assertEqual(spec.operator, '>')
        self.assertEqual(spec.values, (0.5,))
        self.assertEqual(spec.ma_periods, 2)
        self.assertEqual(spec.look_back, 3)
        self.assertEqual(spec.min_val, 1.0)

    def test_duration_simple_golden(self):
        result = parse_request({'comp_a': {'timing': [1, 90], 'duration': ['>', 1]}})
        dur_spec = result.components[0].characteristics[1]
        self.assertEqual(dur_spec.type, CharacteristicType.DURATION)
        self.assertEqual(dur_spec.operator, '>')
        self.assertEqual(dur_spec.values, (1,))
        self.assertEqual(dur_spec.order, 2)

    def test_multi_component_golden(self):
        data = {
            'comp_a': {'timing': [1, 90]},
            'comp_b': {'magnitude': ['>=', 2.0]},
        }
        result = parse_request(data)
        self.assertEqual(len(result.components), 2)
        self.assertEqual(result.components[0].name, 'comp_a')
        self.assertEqual(result.components[1].name, 'comp_b')

    def test_success_pattern_false_preserved(self):
        result = parse_request({'comp_a': {'magnitude': ['>', 1.0], 'success_pattern': False}})
        self.assertFalse(result.components[0].is_success_pattern)

    def test_verbose_false_preserved(self):
        result = parse_request({'comp_a': {'magnitude': ['>', 1.0], 'verbose': False}})
        self.assertFalse(result.components[0].verbose)

    def test_rate_of_change_defaults_applied(self):
        result = parse_request({'comp_a': {'rate_of_change': ['>', 0.5]}})
        spec = result.components[0].characteristics[0]
        self.assertEqual(spec.ma_periods, 1)
        self.assertEqual(spec.look_back, 1)
        self.assertEqual(spec.min_val, 0.0)


# ===========================================================================
# 3. Equivalence tests
# ===========================================================================

class TestParseRequestEquivalence(unittest.TestCase):
    '''Equivalent valid input variants produce equal Request objects.'''

    def test_whitespace_padded_operator_equals_exact(self):
        exact = parse_request({'comp': {'magnitude': ['>', 1.0]}})
        padded = parse_request({'comp': {'magnitude': [' > ', 1.0]}})
        self.assertEqual(exact, padded)

    def test_all_symbols_normalize_via_whitespace_stripping(self):
        for sym in ('<', '<=', '>', '>=', '=', '!='):
            with self.subTest(sym=sym):
                exact = parse_request({'comp': {'magnitude': [sym, 1.0]}})
                padded = parse_request({'comp': {'magnitude': [f'  {sym}  ', 1.0]}})
                self.assertEqual(exact, padded)

    def test_two_identical_configs_produce_equal_requests(self):
        data = {'comp_a': {'timing': [91, 180], 'magnitude': ['>=', 5.0]}}
        self.assertEqual(parse_request(data), parse_request(data))

    def test_duration_configs_equal_independent_of_construction_time(self):
        r1 = parse_request({'comp': {'timing': [1, 90], 'duration': ['>', 3]}})
        r2 = parse_request({'comp': {'timing': [1, 90], 'duration': ['>', 3]}})
        self.assertEqual(r1, r2)


# ===========================================================================
# 4. Negative / shape-incompatible input tests
# ===========================================================================

class TestParseRequestNegative(unittest.TestCase):
    '''Shape-incompatible inputs fail with canonical error codes.'''

    def _assert_code(self, data: dict, expected_code: ParserErrorCode) -> None:
        with self.assertRaises(HydropatternError) as ctx:
            parse_request(data)
        self.assertEqual(ctx.exception.envelope.code, str(expected_code))

    def test_unknown_characteristic_raises_unknown_characteristic(self):
        self._assert_code(
            {'comp': {'unsupported_char': [1, 2]}},
            ParserErrorCode.UNKNOWN_CHARACTERISTIC,
        )

    def test_empty_magnitude_metrics_raises_missing_field(self):
        self._assert_code({'comp': {'magnitude': []}}, ParserErrorCode.MISSING_FIELD)

    def test_invalid_operator_raises_unknown_comparison_symbol(self):
        self._assert_code(
            {'comp': {'magnitude': ['invalid', 1.0]}},
            ParserErrorCode.UNKNOWN_COMPARISON_SYMBOL,
        )

    def test_timing_wrong_length_raises_invalid_value(self):
        self._assert_code({'comp': {'timing': [1, 2, 3]}}, ParserErrorCode.INVALID_VALUE)

    def test_timing_out_of_range_raises_invalid_value(self):
        self._assert_code({'comp': {'timing': [0, 90]}}, ParserErrorCode.INVALID_VALUE)

    def test_duration_float_threshold_raises_invalid_type(self):
        self._assert_code(
            {'comp': {'timing': [1, 90], 'duration': ['>', 1.5]}},
            ParserErrorCode.INVALID_TYPE,
        )

    def test_magnitude_negative_threshold_raises_invalid_value(self):
        self._assert_code({'comp': {'magnitude': ['>', -1.0]}}, ParserErrorCode.INVALID_VALUE)

    def test_rate_of_change_zero_threshold_raises_invalid_value(self):
        self._assert_code(
            {'comp': {'rate_of_change': ['>', 0.0]}},
            ParserErrorCode.INVALID_VALUE,
        )


# ===========================================================================
# 5. build_components tests
# ===========================================================================

class TestBuildComponents(unittest.TestCase):
    '''build_components converts a Request to a list[Component] with callable fxs.'''

    def test_build_from_magnitude_simple_returns_component(self):
        request = parse_request({'comp_a': {'magnitude': ['>', 1.0]}})
        components = build_components(request)
        self.assertIsInstance(components, list)
        self.assertEqual(len(components), 1)
        self.assertIsInstance(components[0], Component)

    def test_build_preserves_component_name(self):
        request = parse_request({'my_flow_component': {'magnitude': ['>', 1.0]}})
        components = build_components(request)
        self.assertEqual(components[0].name, 'my_flow_component')

    def test_build_preserves_is_success_pattern_false(self):
        request = parse_request({'comp': {'magnitude': ['>', 1.0], 'success_pattern': False}})
        components = build_components(request)
        self.assertFalse(components[0].is_success_pattern)

    def test_build_characteristic_count_matches_spec(self):
        request = parse_request({'comp': {'timing': [1, 90], 'magnitude': ['>', 1.0]}})
        components = build_components(request)
        self.assertEqual(len(components[0].characteristics), 2)

    def test_build_multiple_components(self):
        request = parse_request({
            'comp_a': {'magnitude': ['>', 1.0]},
            'comp_b': {'timing': [91, 180]},
        })
        components = build_components(request)
        self.assertEqual(len(components), 2)
        self.assertEqual(components[0].name, 'comp_a')
        self.assertEqual(components[1].name, 'comp_b')

    def test_built_characteristic_has_callable_fx(self):
        request = parse_request({'comp': {'magnitude': ['>', 1.0]}})
        components = build_components(request)
        char = components[0].characteristics[0]
        self.assertTrue(callable(char.fx))

    def test_built_characteristic_name_for_magnitude_simple(self):
        request = parse_request({'comp': {'magnitude': ['>', 1.0]}})
        components = build_components(request)
        self.assertEqual(components[0].characteristics[0].name, 'magnitude_gt1.0')

    def test_built_characteristic_name_for_timing(self):
        request = parse_request({'comp': {'timing': [1, 90]}})
        components = build_components(request)
        self.assertEqual(components[0].characteristics[0].name, 'timing_1-90')

    def test_build_frequency_as_last_characteristic_succeeds(self):
        request = parse_request({
            'comp': {'magnitude': ['>', 1.0], 'frequency': ['>', 1, 2]},
        })
        components = build_components(request)
        self.assertEqual(len(components[0].characteristics), 2)

    def test_build_frequency_not_last_raises_frequency_not_last(self):
        request = parse_request({
            'comp': {
                'magnitude': ['>', 1.0],
                'frequency': ['>', 1, 2],
                'duration': ['>', 1],
            },
        })
        with self.assertRaises(HydropatternError) as ctx:
            build_components(request)
        self.assertEqual(ctx.exception.envelope.code, ParserErrorCode.FREQUENCY_NOT_LAST)

    def test_build_multiple_frequency_characteristics_raises_frequency_not_last(self):
        freq_spec = CharacteristicSpec(
            type=CharacteristicType.FREQUENCY,
            operator='>',
            values=(0.5,),
            ma_periods=2,
            order=2,
        )
        request = Request(components=(
            ComponentSpec(
                name='comp',
                characteristics=(
                    CharacteristicSpec(
                        type=CharacteristicType.MAGNITUDE,
                        operator='>',
                        values=(1.0,),
                        order=1,
                    ),
                    freq_spec,
                    freq_spec,
                ),
            ),
        ))
        with self.assertRaises(HydropatternError) as ctx:
            build_components(request)
        self.assertEqual(ctx.exception.envelope.code, ParserErrorCode.FREQUENCY_NOT_LAST)


# ===========================================================================
# 6. Round-trip tests
# ===========================================================================

class TestRoundTrip(unittest.TestCase):
    '''Request -> build_components -> evaluate_components produces correct results.'''

    def setUp(self):
        self.df = _make_df()

    def test_magnitude_simple_produces_correct_output(self):
        '''flow=[1.0,1.2,1.4,0.8,1.5]; threshold > 1.0 → [0,1,1,0,1].'''
        request = parse_request({'comp': {'magnitude': ['>', 1.0]}})
        components = build_components(request)
        results = evaluate_components(self.df, components)

        self.assertEqual(len(results), 1)
        expected = np.array([0, 1, 1, 0, 1])
        np.testing.assert_array_equal(results[0].df['comp'].values, expected)

    def test_timing_simple_produces_correct_output(self):
        '''dowy=[1,2,3,4,5]; timing window [1,3] → [1,1,1,0,0].'''
        request = parse_request({'comp': {'timing': [1, 3]}})
        components = build_components(request)
        results = evaluate_components(self.df, components)

        expected = np.array([1, 1, 1, 0, 0])
        np.testing.assert_array_equal(results[0].df['comp'].values, expected)

    def test_timing_then_magnitude_produces_correct_output(self):
        '''timing [1,5] → all days eligible; magnitude > 1.0 → [0,1,1,0,1].'''
        request = parse_request({'comp': {'timing': [1, 5], 'magnitude': ['>', 1.0]}})
        components = build_components(request)
        results = evaluate_components(self.df, components)

        expected_timing = np.array([1, 1, 1, 1, 1])
        expected_mag = np.array([0, 1, 1, 0, 1])
        expected_comp = np.array([0, 1, 1, 0, 1])
        np.testing.assert_array_equal(results[0].df['timing_1-5'].values, expected_timing)
        np.testing.assert_array_equal(results[0].df['magnitude_gt1.0'].values, expected_mag)
        np.testing.assert_array_equal(results[0].df['comp'].values, expected_comp)

    def test_whitespace_padded_operator_produces_same_result(self):
        '''Padded and exact operators produce identical evaluation results.'''
        exact = parse_request({'comp': {'magnitude': ['>', 1.0]}})
        padded = parse_request({'comp': {'magnitude': [' > ', 1.0]}})

        exact_result = evaluate_components(self.df, build_components(exact))
        padded_result = evaluate_components(self.df, build_components(padded))

        np.testing.assert_array_equal(
            exact_result[0].df['comp'].values,
            padded_result[0].df['comp'].values,
        )


def _make_frequency_example_df() -> pd.DataFrame:
    '''6-row fixture matching notes/frequencyEnhancement-resolved.md's worked examples.'''
    df = pd.DataFrame(
        {
            'flow': [4.0, 6.0, 6.0, 6.0, 4.0, 6.0],
            'dowy': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        },
        index=pd.to_datetime(
            ['2020-01-01', '2020-01-02', '2020-01-03',
             '2020-01-04', '2020-01-05', '2020-01-06']
        ),
    )
    df.index.name = 'time'
    return df


def _make_nested_frequency_example_4_df() -> pd.DataFrame:
    '''3-year, 2-timestep-per-year fixture matching resolved-doc Example 4:
    magnitude_gt5 per year = [0,1], [1,1], [1,0] (2010, 2011, 2012).'''
    df = pd.DataFrame(
        {
            'flow': [4.0, 6.0, 6.0, 6.0, 6.0, 4.0],
            'dowy': [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        },
        index=pd.to_datetime(
            ['2010-01-01', '2010-06-01', '2011-01-01',
             '2011-06-01', '2012-01-01', '2012-06-01']
        ),
    )
    df.index.name = 'time'
    return df


class TestNestedFrequencyRoundTrip(unittest.TestCase):
    '''Nested frequency: Request -> build_components -> evaluate_components,
    reproducing notes/frequencyEnhancement-resolved.md Example 4.'''

    def test_example_4_full_year_base_pattern(self):
        request = parse_request(
            {'comp': {'magnitude': ['>', 5], 'frequency': [['>', 1, 2], ['>', 1, 2]]}}
        )
        components = build_components(request)
        results = evaluate_components(self.df, components)

        df = results[0].df
        expected_magnitude = np.array([0, 1, 1, 1, 1, 0])
        # year-verdicts (per resolved doc): 2010=0, 2011=1, 2012=0
        expected_intra_annual = np.array([np.nan, 0, np.nan, 1, np.nan, 0])
        # interannual: 2010 insufficient history -> component=0 (NaN never
        # counts as a success, matching every other characteristic's convention);
        # 2011 and 2012 both fail the interannual pattern -> component=0.
        expected_comp = np.array([0, 0, 0, 0, 0, 0])

        magnitude_col = [c for c in df.columns if c.startswith('magnitude')][0]
        intra_col = 'frequency_gt1in2(event)'
        interannual_col = 'frequency_gt1in2(interannual_event)'
        np.testing.assert_array_equal(df[magnitude_col].values, expected_magnitude)
        np.testing.assert_array_equal(df[intra_col].values, expected_intra_annual)
        self.assertIn(interannual_col, df.columns)
        np.testing.assert_array_equal(df['comp'].values, expected_comp)

    def setUp(self):
        self.df = _make_nested_frequency_example_4_df()


def _make_nested_frequency_example_3_df() -> pd.DataFrame:
    '''Two 6-day water years matching resolved-doc Example 3's magnitude_gt5
    = [1,1,1,0,0,1] for year 1; year 2 has no successes at all.'''
    df = pd.DataFrame(
        {
            'flow': [6.0, 6.0, 6.0, 4.0, 4.0, 6.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
            'dowy': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        },
        index=pd.date_range('2020-01-01', periods=12, freq='D'),
    )
    df.index.name = 'time'
    return df


class TestNestedFrequencyBroadcastRoundTrip(unittest.TestCase):
    '''Nested frequency: reproduces resolved-doc Example 3's intra_annual column
    and demonstrates the interannual broadcast rule end-to-end.'''

    def setUp(self):
        self.df = _make_nested_frequency_example_3_df()

    def test_example_3_intra_annual_and_broadcast(self):
        request = parse_request(
            {'comp': {'magnitude': ['>', 5],
                      'frequency': [['>=', 2, 3, False], ['>=', 1, 2]]}}
        )
        components = build_components(request)
        results = evaluate_components(self.df, components)
        df = results[0].df

        intra_col = 'frequency_ge2in3(timestep)'
        expected_intra_annual = np.array([
            np.nan, np.nan, 1, 0, 0, 0,  # year 1 (matches Example 3's table)
            np.nan, np.nan, 0, 0, 0, 0,  # year 2: no successes
        ])
        np.testing.assert_array_equal(df[intra_col].values, expected_intra_annual)

        # interannual: year 1 alone is insufficient history for a 2-year window
        # (component=0, NaN never counts as a success); year 2's window covers
        # [year1=True, year2=False] -> count=1, satisfies [>=,1,2] ->
        # component broadcasts 1 across all of year 2.
        for t in range(6):
            self.assertEqual(df['comp'].values[t], 0)
        for t in range(6, 12):
            self.assertEqual(df['comp'].values[t], 1)


class TestFrequencyRoundTrip(unittest.TestCase):
    '''Un-nested frequency: Request -> build_components -> evaluate_components,
    reproducing notes/frequencyEnhancement-resolved.md Examples 1 and 2.'''

    def setUp(self):
        self.df = _make_frequency_example_df()

    def test_example_1_count_form_event_level(self):
        request = parse_request(
            {'comp': {'magnitude': ['>', 5], 'frequency': ['>=', 1, 2]}}
        )
        components = build_components(request)
        results = evaluate_components(self.df, components)

        expected_magnitude = np.array([0, 1, 1, 1, 0, 1])
        expected_frequency = np.array([np.nan, 0, 0, 0, 0, 1])
        expected_comp = np.array([0, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(
            results[0].df['magnitude_gt5'].values, expected_magnitude
        )
        np.testing.assert_array_equal(
            results[0].df['frequency_ge1in2(event)'].values, expected_frequency
        )
        np.testing.assert_array_equal(results[0].df['comp'].values, expected_comp)

    def test_example_2_count_form_timestep_level(self):
        request = parse_request(
            {'comp': {'magnitude': ['>', 5], 'frequency': ['>=', 1, 2, False]}}
        )
        components = build_components(request)
        results = evaluate_components(self.df, components)

        expected_frequency = np.array([np.nan, 1, 1, 1, 1, 1])
        expected_comp = np.array([0, 1, 1, 1, 0, 1])
        np.testing.assert_array_equal(
            results[0].df['frequency_ge1in2(timestep)'].values, expected_frequency
        )
        np.testing.assert_array_equal(results[0].df['comp'].values, expected_comp)
