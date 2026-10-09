'''Tests for the frequency characteristic enhancement (un-nested forms).

Covers notes/frequencyEnhancement-resolved.md's un-nested config parsing:
  - [operator, n, N, (exclusive_windows)]         -> FrequencyForm.COUNT        (task: freq-parse-count)
  - [min_n, max_n, N, (exclusive_windows)]        -> FrequencyForm.BETWEEN      (task: freq-parse-between)

Standalone [operator, probability, (exclusive_windows)] is INVALID un-nested (see
notes/frequencyEnhancement-resolved.md): it survives only as the intra-annual
pattern of a nested frequency spec (freq-nested-parse). validate_frequency_metrics
still knows how to parse/validate the probability shape via an
allow_probability=True escape hatch, reserved for the nested parser's reuse.

These tests validate parsing/classification/naming only (CharacteristicSpec shape,
Characteristic.name). Evaluation-engine correctness (frequency_fx's actual output)
is covered separately once the engine lands (freq-core-* tasks) -- frequency_fx
currently raises NotImplementedError when invoked.
'''
# pylint: disable=missing-function-docstring
import unittest

from hydropattern.errors import HydropatternError, ParserErrorCode
from hydropattern.parsers import (
    FrequencyForm,
    build_components,
    frequency_parser,
    is_nested_frequency_shape,
    nested_frequency_parser,
    parse_request,
    validate_frequency_metrics,
    validate_nested_frequency_metrics,
)


class TestValidateFrequencyMetricsProbabilityFormRejectedUnNested(unittest.TestCase):
    '''Standalone [operator, probability, (exclusive_windows)] must be rejected un-nested.'''

    def test_bare_probability_form_raises(self):
        with self.assertRaises(HydropatternError) as ctx:
            validate_frequency_metrics(['>', 0.5])
        self.assertEqual(
            ctx.exception.envelope.code, str(ParserErrorCode.FREQUENCY_PROBABILITY_NOT_NESTED)
        )

    def test_probability_with_explicit_exclusive_windows_raises(self):
        with self.assertRaises(HydropatternError) as ctx:
            validate_frequency_metrics(['>', 0.5, False])
        self.assertEqual(
            ctx.exception.envelope.code, str(ParserErrorCode.FREQUENCY_PROBABILITY_NOT_NESTED)
        )

    def test_probability_boundary_values_still_rejected(self):
        with self.assertRaises(HydropatternError):
            validate_frequency_metrics(['>=', 0.0])
        with self.assertRaises(HydropatternError):
            validate_frequency_metrics(['<=', 1.0])


class TestValidateFrequencyMetricsProbabilityFormAllowedNested(unittest.TestCase):
    '''allow_probability=True (reserved for the nested parser) still parses correctly.'''

    def test_bare_probability_form(self):
        parsed = validate_frequency_metrics(['>', 0.5], allow_probability=True)
        self.assertEqual(parsed.form, FrequencyForm.PROBABILITY)
        self.assertEqual(parsed.operator, '>')
        self.assertEqual(parsed.values, (0.5,))
        self.assertIsNone(parsed.big_n)
        self.assertFalse(parsed.exclusive_windows)  # defaults to False (union)

    def test_probability_with_explicit_exclusive_windows_false(self):
        parsed = validate_frequency_metrics(['>', 0.5, False], allow_probability=True)
        self.assertEqual(parsed.form, FrequencyForm.PROBABILITY)
        self.assertFalse(parsed.exclusive_windows)

    def test_probability_with_explicit_exclusive_windows_true(self):
        parsed = validate_frequency_metrics(['>', 0.1, True], allow_probability=True)
        self.assertEqual(parsed.form, FrequencyForm.PROBABILITY)
        self.assertTrue(parsed.exclusive_windows)

    def test_probability_boundary_zero_and_one_accepted(self):
        self.assertEqual(
            validate_frequency_metrics(['>=', 0.0], allow_probability=True).values, (0.0,)
        )
        self.assertEqual(
            validate_frequency_metrics(['<=', 1.0], allow_probability=True).values, (1.0,)
        )

    def test_probability_below_zero_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            validate_frequency_metrics(['>', -0.1], allow_probability=True)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_probability_above_one_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            validate_frequency_metrics(['>', 1.1], allow_probability=True)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_probability_non_numeric_raises_invalid_type(self):
        with self.assertRaises(HydropatternError) as ctx:
            validate_frequency_metrics(['>', 'oops'], allow_probability=True)
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_TYPE))

    def test_invalid_operator_raises_unknown_comparison_symbol(self):
        with self.assertRaises(HydropatternError) as ctx:
            validate_frequency_metrics(['gt', 0.5], allow_probability=True)
        self.assertEqual(
            ctx.exception.envelope.code, str(ParserErrorCode.UNKNOWN_COMPARISON_SYMBOL)
        )

    def test_padded_operator_normalized(self):
        parsed = validate_frequency_metrics([' >= ', 0.5], allow_probability=True)
        self.assertEqual(parsed.operator, '>=')


class TestValidateFrequencyMetricsCountForm(unittest.TestCase):
    '''[operator, n, N, (exclusive_windows)] -> FrequencyForm.COUNT.'''

    def test_bare_count_form(self):
        parsed = validate_frequency_metrics(['>', 1, 2])
        self.assertEqual(parsed.form, FrequencyForm.COUNT)
        self.assertEqual(parsed.operator, '>')
        self.assertEqual(parsed.values, (1,))
        self.assertEqual(parsed.big_n, 2)
        self.assertFalse(parsed.exclusive_windows)

    def test_count_with_explicit_exclusive_windows_true(self):
        parsed = validate_frequency_metrics(['>', 1, 2, True])
        self.assertEqual(parsed.form, FrequencyForm.COUNT)
        self.assertTrue(parsed.exclusive_windows)

    def test_zero_count_threshold_is_valid_when_predicate_is_attainable(self):
        parsed = validate_frequency_metrics(['=', 0, 5])
        self.assertEqual(parsed.values, (0,))

    def test_n_non_integer_raises_invalid_type(self):
        with self.assertRaises(HydropatternError) as ctx:
            validate_frequency_metrics(['>', 1.5, 5])
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_TYPE))

    def test_predicate_above_maximum_count_is_invalid(self):
        with self.assertRaises(HydropatternError) as ctx:
            validate_frequency_metrics(['>', 6, 5])
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_big_n_less_than_n_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            validate_frequency_metrics(['>', 5, 3])
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))


class TestValidateFrequencyMetricsBetweenForm(unittest.TestCase):
    '''[min_n, max_n, N, (exclusive_windows)] -> FrequencyForm.BETWEEN (inclusive bounds, ADR 0001).'''

    def test_bare_between_form(self):
        parsed = validate_frequency_metrics([1, 3, 5])
        self.assertEqual(parsed.form, FrequencyForm.BETWEEN)
        self.assertIsNone(parsed.operator)
        self.assertEqual(parsed.values, (1, 3))
        self.assertEqual(parsed.big_n, 5)
        self.assertFalse(parsed.exclusive_windows)

    def test_between_with_explicit_exclusive_windows_true(self):
        parsed = validate_frequency_metrics([1, 3, 5, True])
        self.assertTrue(parsed.exclusive_windows)

    def test_min_n_must_be_less_than_max_n(self):
        with self.assertRaises(HydropatternError) as ctx:
            validate_frequency_metrics([3, 3, 5])
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_max_n_may_equal_big_n(self):
        parsed = validate_frequency_metrics([1, 5, 5])
        self.assertEqual(parsed.values, (1, 5))
        self.assertEqual(parsed.big_n, 5)

    def test_non_integer_values_raise_invalid_type(self):
        with self.assertRaises(HydropatternError) as ctx:
            validate_frequency_metrics([1.5, 3, 5])
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_TYPE))


class TestValidateFrequencyMetricsShape(unittest.TestCase):
    '''General shape validation shared across all forms.'''

    def test_too_few_elements_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            validate_frequency_metrics(['>'])
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_too_many_elements_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            validate_frequency_metrics(['>', 1, 2, 3, True])
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))


class TestFrequencyParserNaming(unittest.TestCase):
    '''frequency_parser produces names matching notes/frequencyEnhancement-resolved.md examples.'''

    def test_probability_form_rejected_un_nested(self):
        with self.assertRaises(HydropatternError) as ctx:
            frequency_parser(['>', 0.5], order=2)
        self.assertEqual(
            ctx.exception.envelope.code, str(ParserErrorCode.FREQUENCY_PROBABILITY_NOT_NESTED)
        )

    def test_count_union_level_name(self):
        char = frequency_parser(['>', 1, 2], order=2)
        self.assertEqual(char.name, 'frequency_gt1in2(union)')

    def test_between_union_level_name(self):
        char = frequency_parser([1, 3, 5], order=2)
        self.assertEqual(char.name, 'frequency_1-3in5(union)')


class TestFrequencyBuildComponentsIntegration(unittest.TestCase):
    '''Un-nested frequency forms flow through parse_request -> build_components.'''

    def test_probability_form_via_toml_shape_rejected(self):
        with self.assertRaises(HydropatternError) as ctx:
            parse_request({'comp': {'magnitude': ['>', 1.0], 'frequency': ['>', 0.5]}})
        self.assertEqual(
            ctx.exception.envelope.code, str(ParserErrorCode.FREQUENCY_PROBABILITY_NOT_NESTED)
        )

    def test_count_form_via_toml_shape(self):
        request = parse_request({'comp': {'magnitude': ['>', 1.0], 'frequency': ['>', 1, 2]}})
        components = build_components(request)
        self.assertEqual(components[0].characteristics[-1].name, 'frequency_gt1in2(union)')

    def test_between_form_via_toml_shape(self):
        request = parse_request({'comp': {'magnitude': ['>', 1.0], 'frequency': [1, 3, 5]}})
        components = build_components(request)
        self.assertEqual(components[0].characteristics[-1].name, 'frequency_1-3in5(union)')

    def test_exclusive_windows_true_via_toml_shape(self):
        request = parse_request(
            {'comp': {'magnitude': ['>', 1.0], 'frequency': ['>', 1, 2, True]}}
        )
        components = build_components(request)
        self.assertEqual(components[0].characteristics[-1].name, 'frequency_gt1in2(exclusive)')


class TestIsNestedFrequencyShape(unittest.TestCase):
    '''Detects two-part frequency shape [<intra-annual>, <interannual>] forms.'''

    def test_nested_shape_detected(self):
        self.assertTrue(is_nested_frequency_shape([['>', 0.5], ['>', 1, 2]]))

    def test_count_form_not_nested(self):
        self.assertFalse(is_nested_frequency_shape(['>', 1, 2]))

    def test_between_form_not_nested(self):
        self.assertFalse(is_nested_frequency_shape([1, 3, 5]))

    def test_probability_form_not_nested(self):
        self.assertFalse(is_nested_frequency_shape(['>', 0.5]))

    def test_two_element_non_list_values_not_nested(self):
        self.assertFalse(is_nested_frequency_shape(['>', 0.5]))

    def test_single_nested_list_not_nested(self):
        self.assertFalse(is_nested_frequency_shape([['>', 0.5]]))

    def test_three_nested_lists_not_nested(self):
        self.assertFalse(is_nested_frequency_shape([['>', 0.5], ['>', 1, 2], ['>', 1, 2]]))


class TestValidateNestedFrequencyMetrics(unittest.TestCase):
    '''validate_nested_frequency_metrics: intra-annual allows probability, interannual does not.'''

    def test_probability_intra_annual_with_count_interannual(self):
        intra_annual, interannual = validate_nested_frequency_metrics(
            [['>', 0.5], ['>', 1, 2]]
        )
        self.assertEqual(intra_annual.form, FrequencyForm.PROBABILITY)
        self.assertEqual(intra_annual.values, (0.5,))
        self.assertEqual(interannual.form, FrequencyForm.COUNT)
        self.assertEqual(interannual.values, (1,))
        self.assertEqual(interannual.big_n, 2)

    def test_count_intra_annual_with_count_interannual(self):
        intra_annual, interannual = validate_nested_frequency_metrics(
            [['>', 1, 3], ['>', 1, 2]]
        )
        self.assertEqual(intra_annual.form, FrequencyForm.COUNT)
        self.assertEqual(interannual.form, FrequencyForm.COUNT)

    def test_between_intra_annual_with_between_interannual(self):
        intra_annual, interannual = validate_nested_frequency_metrics(
            [[1, 2, 3], [1, 2, 3]]
        )
        self.assertEqual(intra_annual.form, FrequencyForm.BETWEEN)
        self.assertEqual(interannual.form, FrequencyForm.BETWEEN)

    def test_probability_interannual_pattern_rejected(self):
        with self.assertRaises(HydropatternError) as ctx:
            validate_nested_frequency_metrics([['>', 1, 2], ['>', 0.5]])
        self.assertEqual(
            ctx.exception.envelope.code, str(ParserErrorCode.FREQUENCY_PROBABILITY_NOT_NESTED)
        )

    def test_un_nested_shape_raises_invalid_value(self):
        with self.assertRaises(HydropatternError) as ctx:
            validate_nested_frequency_metrics(['>', 1, 2])
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_invalid_intra_annual_pattern_propagates_error(self):
        with self.assertRaises(HydropatternError) as ctx:
            validate_nested_frequency_metrics([['>', -0.1], ['>', 1, 2]])
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))

    def test_invalid_interannual_pattern_propagates_error(self):
        with self.assertRaises(HydropatternError) as ctx:
            validate_nested_frequency_metrics([['>', 0.5], ['>', 2, 2]])
        self.assertEqual(ctx.exception.envelope.code, str(ParserErrorCode.INVALID_VALUE))


class TestNestedFrequencyParserNaming(unittest.TestCase):
    '''nested_frequency_parser produces intra-annual and interannual Characteristics.'''

    def test_only_terminal_characteristic_has_terminal_marker(self):
        chars = nested_frequency_parser([['>', 0.5], ['>', 1, 2]], order=2)
        self.assertFalse(chars[0].is_terminal)
        self.assertTrue(chars[1].is_terminal)

    def test_returns_two_characteristics(self):
        chars = nested_frequency_parser([['>', 0.5], ['>', 1, 2]], order=2)
        self.assertEqual(len(chars), 2)

    def test_intra_annual_is_not_terminal(self):
        chars = nested_frequency_parser([['>', 0.5], ['>', 1, 2]], order=2)
        self.assertFalse(chars[0].is_terminal)

    def test_interannual_is_terminal(self):
        chars = nested_frequency_parser([['>', 0.5], ['>', 1, 2]], order=2)
        self.assertTrue(chars[1].is_terminal)

    def test_names_use_union_and_interannual_union_markers(self):
        # Intra-annual uses its own marker; interannual uses its prefix
        # to distinguish the two columns when both share the same value label
        # (see notes/frequencyEnhancement.md's nested examples).
        chars = nested_frequency_parser([['>', 0.5], ['>', 1, 2]], order=2)
        self.assertEqual(chars[0].name, 'frequency_gt0.5(union)')
        self.assertEqual(chars[1].name, 'frequency_gt1in2(interannual_union)')

    def test_exclusive_level_markers(self):
        chars = nested_frequency_parser([['>', 0.5, True], ['>', 1, 2, True]], order=2)
        self.assertEqual(chars[0].name, 'frequency_gt0.5(exclusive)')
        self.assertEqual(chars[1].name, 'frequency_gt1in2(interannual_exclusive)')

    def test_fx_calls_are_callable_evaluation_functions(self):
        # freq-nested-eval implemented these; just confirm they're wired up,
        # not raw NotImplementedError stubs. Full evaluation correctness is
        # covered by TestNestedFrequencyEvaluation in test_patterns.py.
        chars = nested_frequency_parser([['>', 0.5], ['>', 1, 2]], order=2)
        self.assertTrue(callable(chars[0].fx))
        self.assertTrue(callable(chars[1].fx))


class TestNestedFrequencyBuildComponentsIntegration(unittest.TestCase):
    '''Nested frequency flows through parse_request -> build_components as 2 characteristics.'''

    def test_nested_spec_uses_interannual_specific_field_names(self):
        request = parse_request(
            {'comp': {'frequency': [['>', 0.5], ['>=', 1, 2, True]]}}
        )
        spec = request.components[0].characteristics[0]

        self.assertTrue(spec.has_interannual_pattern)
        self.assertEqual(spec.interannual_operator, '>=')
        self.assertEqual(spec.interannual_values, (1,))
        self.assertEqual(spec.interannual_big_n, 2)
        self.assertTrue(spec.interannual_exclusive_windows)

    def test_nested_spec_names_interannual_exclusivity(self):
        request = parse_request(
            {
                'comp': {
                    'magnitude': ['>', 1.0],
                    'frequency': [['>', 0.5, True], ['>', 1, 2, True]],
                }
            }
        )
        spec = request.components[0].characteristics[-1]
        self.assertTrue(spec.exclusive_windows)
        self.assertTrue(spec.interannual_exclusive_windows)

    def test_nested_shape_via_toml_produces_two_characteristics(self):
        request = parse_request(
            {'comp': {'magnitude': ['>', 1.0], 'frequency': [['>', 0.5], ['>', 1, 2]]}}
        )
        components = build_components(request)
        self.assertEqual(len(components[0].characteristics), 3)  # magnitude + 2 frequency cols
        self.assertFalse(components[0].characteristics[1].is_terminal)
        self.assertTrue(components[0].characteristics[2].is_terminal)

    def test_nested_shape_via_toml_names(self):
        request = parse_request(
            {'comp': {'magnitude': ['>', 1.0], 'frequency': [['>', 0.5], ['>', 1, 2]]}}
        )
        components = build_components(request)
        self.assertEqual(components[0].characteristics[1].name, 'frequency_gt0.5(union)')
        self.assertEqual(
            components[0].characteristics[2].name, 'frequency_gt1in2(interannual_union)'
        )


if __name__ == '__main__':
    unittest.main()
