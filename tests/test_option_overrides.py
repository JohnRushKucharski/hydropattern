'''Tests for the shared CLI-option-override helpers in hydropattern.parsers.

These helpers centralize the "collect explicitly-passed (non-None) values" and
"apply them onto a frozen dataclass" logic shared by resolve_output_options and
require_no_conflicting_cli_options, so adding a new option field only requires
passing it through at the call site -- no separate hand-written None-check.
'''
# pylint: disable=missing-function-docstring

import unittest
from dataclasses import dataclass

from hydropattern.parsers import collect_explicit_options, merge_overrides


@dataclass(frozen=True)
class _Widget:
    '''Minimal frozen dataclass fixture for exercising merge_overrides.'''
    a: int = 1
    b: str = 'default'
    c: bool = False


class TestCollectExplicitOptions(unittest.TestCase):
    '''collect_explicit_options filters out None-valued (not explicitly passed) kwargs.'''

    def test_all_none_returns_empty_dict(self):
        self.assertEqual(collect_explicit_options(a=None, b=None), {})

    def test_mixed_none_and_values_keeps_only_non_none(self):
        result = collect_explicit_options(a=None, b=2, c=False, d='x')
        self.assertEqual(result, {'b': 2, 'c': False, 'd': 'x'})

    def test_no_kwargs_returns_empty_dict(self):
        self.assertEqual(collect_explicit_options(), {})


class TestMergeOverrides(unittest.TestCase):
    '''merge_overrides applies only explicitly-set (non-None) overrides onto a dataclass copy.'''

    def test_no_overrides_returns_equal_but_unchanged_instance(self):
        base = _Widget()
        result = merge_overrides(base, a=None, b=None, c=None)
        self.assertEqual(result, base)

    def test_single_override_applied(self):
        base = _Widget()
        result = merge_overrides(base, a=5)
        self.assertEqual(result, _Widget(a=5))

    def test_multiple_overrides_applied_others_untouched(self):
        base = _Widget(a=1, b='x', c=False)
        result = merge_overrides(base, b='y', c=None)
        self.assertEqual(result, _Widget(a=1, b='y', c=False))

    def test_false_and_zero_are_explicit_values_not_skipped(self):
        base = _Widget(a=9, c=True)
        result = merge_overrides(base, a=0, c=False)
        self.assertEqual(result, _Widget(a=0, b='default', c=False))

    def test_unknown_field_name_raises(self):
        with self.assertRaises(TypeError):
            merge_overrides(_Widget(), nonexistent=1)


if __name__ == '__main__':
    unittest.main()
