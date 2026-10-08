'''Phase 4 acceptance tests for component truth and unknown propagation.'''
import numpy as np
import pandas as pd
import pytest

from hydropattern.patterns import (
    Characteristic,
    CharacteristicType,
    Component,
    evaluate_component,
)


def _evaluate(
    values: list[list[float]],
    *,
    success_pattern: bool = True,
    terminal_type: CharacteristicType = CharacteristicType.MAGNITUDE,
    mark_terminal: bool = False,
) -> np.ndarray:
    data = pd.DataFrame(
        {'flow': np.arange(len(values)), 'dowy': np.arange(1, len(values) + 1)},
        index=pd.date_range('2020-01-01', periods=len(values), name='time'),
    )
    characteristics = []
    for index, column in enumerate(np.asarray(values, dtype=float).T):
        is_terminal = index == len(values[0]) - 1
        char_type = terminal_type if is_terminal else CharacteristicType.MAGNITUDE
        characteristics.append(
            Characteristic(
                name=f'char_{index}',
                fx=lambda _df, _output, column=column: column,
                type=char_type,
                is_terminal=mark_terminal and is_terminal,
            )
        )
    component = Component(
        name='component',
        characteristics=characteristics,
        is_success_pattern=success_pattern,
    )
    return evaluate_component(data, component).df['component'].to_numpy()


def test_failure_pattern_reports_complement_of_combined_failure_conditions():
    assert _evaluate([[1, 1], [1, 0], [0, 1], [0, 0]], success_pattern=False).tolist() == [
        0, 1, 1, 1
    ]


@pytest.mark.parametrize(
    'values, expected',
    [
        ([[np.nan, 1], [1, np.nan], [0, np.nan], [np.nan, np.nan]], [np.nan, np.nan, 0, np.nan]),
    ],
)
def test_positive_component_uses_three_valued_and(values, expected):
    np.testing.assert_equal(_evaluate(values), expected)


def test_failure_pattern_preserves_unknown_unless_failure_is_decisive():
    result = _evaluate(
        [[np.nan, 1], [1, np.nan], [0, np.nan], [np.nan, np.nan]],
        success_pattern=False,
    )

    np.testing.assert_equal(result, [np.nan, np.nan, 1, np.nan])


def test_nested_frequency_terminal_is_inverted_without_collapsing_unknown():
    result = _evaluate(
        [[1], [0], [np.nan]],
        success_pattern=False,
        terminal_type=CharacteristicType.FREQUENCY,
        mark_terminal=True,
    )

    np.testing.assert_equal(result, [0, 1, np.nan])


def test_unnested_frequency_terminal_is_not_re_and_with_source_condition():
    result = _evaluate(
        [[1, 1], [0, 1], [1, 0]],
        terminal_type=CharacteristicType.FREQUENCY,
    )

    np.testing.assert_array_equal(result, [1, 1, 0])
