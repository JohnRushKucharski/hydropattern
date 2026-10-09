'''Phase 3 acceptance tests for nested-frequency windows and annual verdicts.'''
import numpy as np
import pandas as pd

from hydropattern.parsers import build_components, parse_request
from hydropattern.patterns import (
    Characteristic,
    CharacteristicType,
    Component,
    comparison_fx,
    evaluate_components,
    evaluate_component,
    frequency_fx,
    nested_frequency_intra_annual_fx,
    nested_frequency_interannual_fx,
)


def _evaluate_nested_frequency(
    flow: list[int],
    intra_metrics: list,
    interannual_metrics: list,
) -> pd.DataFrame:
    year_count = (len(flow) + 3) // 4
    data = pd.DataFrame(
        {
            'flow': flow,
            'dowy': np.tile(np.arange(1, 5), year_count)[:len(flow)],
        },
        index=pd.RangeIndex(len(flow), name='time'),
    )
    request = parse_request(
        {
            'component': {
                'magnitude': ['>=', 1],
                'frequency': [intra_metrics, interannual_metrics],
            }
        }
    )
    return evaluate_components(data, build_components(request))[0].df


def test_nested_frequency_annual_probability_and_forward_windows():
    result = _evaluate_nested_frequency(
        [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        ['>=', 0.5],
        ['>=', 1, 2],
    )

    np.testing.assert_array_equal(
        result['frequency_ge0.5(union)'].to_numpy(),
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    )
    np.testing.assert_array_equal(
        result['frequency_ge1in2(interannual_union)'].to_numpy(),
        [1] * 12,
    )
    np.testing.assert_array_equal(result['component'].to_numpy(), [1] * 12)


def test_nested_frequency_exclusive_outer_windows_use_annual_trials():
    result = _evaluate_nested_frequency(
        [1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        ['>=', 0.5],
        ['>=', 1, 2, True],
    )

    np.testing.assert_array_equal(
        result['frequency_ge0.5(union)'].to_numpy(),
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    )
    np.testing.assert_array_equal(
        result['frequency_ge1in2(interannual_exclusive)'].to_numpy(),
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    )
    np.testing.assert_array_equal(
        result['component'].to_numpy(),
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    )


def test_outer_window_truncates_at_last_complete_water_year():
    result = _evaluate_nested_frequency(
        [1, 1, 1, 0, 0, 0, 0, 0],
        ['>=', 0.5],
        ['>=', 1, 3],
    )

    np.testing.assert_array_equal(
        result['frequency_ge0.5(union)'].to_numpy(),
        [1, 1, 1, 1, 0, 0, 0, 0],
    )
    np.testing.assert_array_equal(
        result['frequency_ge1in3(interannual_union)'].to_numpy(),
        [1] * 8,
    )


def test_leading_partial_water_year_has_no_annual_probability_verdict():
    data = pd.DataFrame({'flow': range(6), 'dowy': [3, 4, 1, 2, 3, 4]})
    output = np.ones((6, 1))
    fx = nested_frequency_intra_annual_fx(
        comparison_fx('>=', 0.5), order=2, big_n=None
    )

    np.testing.assert_array_equal(
        fx(data, output),
        [np.nan, np.nan, 1, 1, 1, 1],
    )


def test_antecedent_frequency_exclusivity_changes_nested_probability():
    data = pd.DataFrame(
        {
            'flow': [1, 0, 1, 0, 1, 0],
            'dowy': [1, 2, 3, 4, 5, 6],
        },
        index=pd.RangeIndex(6, name='time'),
    )
    results = {}
    for exclusive in (False, True):
        characteristics = [
            Characteristic(
                'source',
                lambda _df, _output: np.array([1, 0, 1, 0, 1, 0]),
                CharacteristicType.MAGNITUDE,
            ),
            Characteristic(
                'antecedent_frequency',
                frequency_fx(
                    comparison_fx('>=', 2),
                    order=2,
                    big_n=3,
                    exclusive_windows=exclusive,
                ),
                CharacteristicType.FREQUENCY,
            ),
            Characteristic(
                'annual_probability',
                nested_frequency_intra_annual_fx(
                    comparison_fx('>=', 0.5),
                    order=3,
                    big_n=None,
                ),
                CharacteristicType.FREQUENCY,
            ),
            Characteristic(
                'outer_frequency',
                nested_frequency_interannual_fx(
                    comparison_fx('>=', 1),
                    order=4,
                    big_n=1,
                ),
                CharacteristicType.FREQUENCY,
                True,
            ),
        ]
        component = Component('component', characteristics, True)
        results[exclusive] = evaluate_component(data, component).df

    np.testing.assert_array_equal(
        results[False]['antecedent_frequency'].to_numpy(),
        [1, 1, 1, 1, 1, 0],
    )
    np.testing.assert_array_equal(
        results[True]['antecedent_frequency'].to_numpy(),
        [1, 1, 1, 0, 0, 0],
    )
    np.testing.assert_array_equal(
        results[False]['annual_probability'].to_numpy(), [1, 1, 1, 1, 1, 1]
    )
    np.testing.assert_array_equal(
        results[True]['annual_probability'].to_numpy(), [0, 0, 0, 0, 0, 0]
    )
    np.testing.assert_array_equal(results[False]['component'], [1] * 6)
    np.testing.assert_array_equal(results[True]['component'], [0] * 6)
