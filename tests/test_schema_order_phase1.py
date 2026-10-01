'''Phase 1 (schema/order) tests, per docs/plans/2026-10-01-pattern-correctness-tdd.md.

Covers:
  - `verbose` removed from ComponentSpec and rejected as a config key.
  - Component options (success_pattern) never occupy a characteristic
    position / never perturb order, regardless of where they appear.
  - Compact-dict and ordered-array-of-tables component forms are equivalent
    and never accept a user-supplied `order`.
  - Compact form warns (portability); ordered form does not.
  - Timing/magnitude/rate_of_change are independent diagnostics: they report
    their own truth value regardless of a preceding characteristic's result.
    Duration/frequency are unaffected (still gated on preceding conjunction).
'''
import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from hydropattern.errors import HydropatternError
from hydropattern.parsers import ComponentSpec, build_components, parse_request
from hydropattern.patterns import (
    comparison_fx,
    duration_fx,
    evaluate_component,
    magnitude_fx,
    rate_of_change_fx,
    timing_fx,
)


def _make_df(flow, dowy=None):
    n = len(flow)
    dowy = dowy or list(range(1, n + 1))
    df = pd.DataFrame(
        {'flow': flow, 'dowy': dowy},
        index=pd.date_range('2020-01-01', periods=n, freq='D'),
    )
    df.index.name = 'time'
    return df


# ---------------------------------------------------------------------------
# verbose removed
# ---------------------------------------------------------------------------

def test_component_spec_has_no_verbose_field():
    field_names = {f.name for f in dataclasses.fields(ComponentSpec)}
    assert 'verbose' not in field_names


def test_verbose_key_in_config_raises():
    data = {
        'pulse': {
            'verbose': True,
            'magnitude': ['>', 1.0],
        }
    }
    with pytest.raises(HydropatternError):
        parse_request(data)


# ---------------------------------------------------------------------------
# order inferred purely from characteristic sequence; options never count.
# ---------------------------------------------------------------------------

def test_success_pattern_before_characteristics_does_not_perturb_order():
    # success_pattern listed first must not bump the first real
    # characteristic's order past 1.
    data_first = {
        'pulse': {
            'success_pattern': False,
            'magnitude': ['>', 1.0],
            'duration': ['>=', 2],
        }
    }
    data_last = {
        'pulse': {
            'magnitude': ['>', 1.0],
            'duration': ['>=', 2],
            'success_pattern': False,
        }
    }
    req_first = parse_request(data_first)
    req_last = parse_request(data_last)
    orders_first = [c.order for c in req_first.components[0].characteristics]
    orders_last = [c.order for c in req_last.components[0].characteristics]
    assert orders_first == [1, 2]
    assert orders_last == [1, 2]


# ---------------------------------------------------------------------------
# compact vs ordered-array equivalence + warning behavior
# ---------------------------------------------------------------------------

def test_compact_form_warns_ordered_form_does_not():
    compact = {'pulse': {'magnitude': ['>', 1.0], 'duration': ['>=', 2]}}
    ordered = {
        'pulse': {
            'characteristics': [
                {'type': 'magnitude', 'metrics': ['>', 1.0]},
                {'type': 'duration', 'metrics': ['>=', 2]},
            ]
        }
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        parse_request(compact)
        assert any('order' in str(w.message).lower() for w in caught)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        parse_request(ordered)
        assert not caught


def test_compact_and_ordered_forms_produce_equivalent_requests():
    compact = {'pulse': {'magnitude': ['>', 1.0], 'duration': ['>=', 2]}}
    ordered = {
        'pulse': {
            'characteristics': [
                {'type': 'magnitude', 'metrics': ['>', 1.0]},
                {'type': 'duration', 'metrics': ['>=', 2]},
            ]
        }
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        req_compact = parse_request(compact)
    req_ordered = parse_request(ordered)
    assert req_compact == req_ordered


def test_ordered_form_rejects_explicit_order_key():
    data = {
        'pulse': {
            'characteristics': [
                {'type': 'magnitude', 'metrics': ['>', 1.0], 'order': 1},
            ]
        }
    }
    with pytest.raises(HydropatternError):
        parse_request(data)


def test_empty_component_is_rejected():
    data = {'pulse': {'success_pattern': True}}
    with pytest.raises(HydropatternError):
        parse_request(data)


# ---------------------------------------------------------------------------
# independent diagnostics: timing/magnitude/rate_of_change ignore precedents.
# ---------------------------------------------------------------------------

def test_magnitude_after_failing_characteristic_reports_own_truth():
    # order=2 magnitude; precedent (output col 0) is all failing (0), but the
    # magnitude's own truth (always True here) must still show through.
    df = _make_df([5.0, 5.0, 5.0])
    output = np.zeros((3, 1))  # precedent column: all 0 (failed)
    fx = magnitude_fx(comparison_fx('>', 1.0), order=2)
    result = fx(df, output)
    assert result.tolist() == [1, 1, 1]


def test_timing_after_failing_characteristic_reports_own_truth():
    df = _make_df([5.0, 5.0, 5.0], dowy=[1, 2, 3])
    output = np.zeros((3, 1))  # precedent column: all 0 (failed)
    fx = timing_fx(comparison_fx('<=', 3), order=2)
    result = fx(df, output)
    assert result.tolist() == [1, 1, 1]


def test_rate_of_change_after_failing_characteristic_reports_own_truth():
    df = _make_df([1.0, 4.0, 16.0])
    output = np.zeros((3, 1))  # precedent column: all 0 (failed)
    fx = rate_of_change_fx(comparison_fx('>', 1.0), order=2)
    result = fx(df, output)
    # look_back=1: ratios are [nan, 4.0, 4.0] -> comparison [0,1,1]; own truth,
    # not gated to 0 by the failing precedent column.
    assert result.tolist() == [0, 1, 1]


def test_duration_still_gated_on_preceding_conjunction():
    # Regression: duration must remain dependent on preceding characteristics
    # (unlike timing/magnitude/rate_of_change).
    df = _make_df([5.0, 5.0, 5.0])
    output = np.zeros((3, 1))  # precedent column: all 0 (failed) -> no runs
    fx = duration_fx(comparison_fx('>=', 1), order=2)
    result = fx(df, output)
    assert result.tolist() == [0, 0, 0]


def test_end_to_end_component_independent_then_dependent():
    # magnitude always fails; timing always true (independent, order 2);
    # duration (order 3) stays gated on [magnitude AND timing] = always 0.
    data = {
        'pulse': {
            'magnitude': ['>', 100.0],
            'timing': [1, 3],
            'duration': ['>=', 1],
        }
    }
    req = parse_request(data)
    components = build_components(req)
    df = _make_df([5.0, 5.0, 5.0], dowy=[1, 2, 3])
    result = evaluate_component(df, components[0])
    magnitude_col = result.df[components[0].characteristics[0].name].tolist()
    timing_col = result.df[components[0].characteristics[1].name].tolist()
    duration_col = result.df[components[0].characteristics[2].name].tolist()
    assert magnitude_col == [0, 0, 0]
    assert timing_col == [1, 1, 1]  # independent: own truth despite failing magnitude
    assert duration_col == [0, 0, 0]  # still gated: magnitude AND timing never both 1
