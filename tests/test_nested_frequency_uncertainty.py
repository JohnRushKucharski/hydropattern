'''Unknown-aware intra-annual and interannual frequency acceptance tests.'''
import itertools

import numpy as np
import pandas as pd
import pytest

from hydropattern.patterns import (
    Characteristic,
    CharacteristicType,
    Component,
    comparison_fx,
    evaluate_component,
    nested_frequency_interannual_fx,
    nested_frequency_intra_annual_fx,
    or_reduce_per_water_year,
    water_year_probability_ratio,
)


def monthly_frame(years: int = 1) -> pd.DataFrame:
    index = pd.date_range('2020-01-01', periods=12 * years, freq='MS')
    dowy = index.dayofyear - (index.is_leap_year & (index.month > 2)).astype(int)
    return pd.DataFrame(
        {'flow': np.ones(len(index)), 'dowy': dowy},
        index=index,
    )


@pytest.mark.parametrize(
    'source, operator, threshold, expected',
    [
        ([1, 0] + [np.nan] * 10, '>=', 0.5, np.nan),
        ([1] * 7 + [np.nan] * 5, '>=', 0.5, 1),
        ([0] * 7 + [np.nan] * 5, '>=', 0.5, 0),
        ([np.nan] * 12, '>=', 0.5, np.nan),
        ([np.nan] * 12, '>=', 0, 1),
        ([1, 0] + [np.nan] * 10, '=', 0.5, np.nan),
        ([1, 0] + [np.nan] * 10, '!=', 0.5, np.nan),
        ([1] * 6 + [0] * 6, '>=', 0.5, 1),
    ],
)
def test_annual_fraction_assesses_all_observed_trials(
    source, operator, threshold, expected,
):
    fx = nested_frequency_intra_annual_fx(
        comparison_fx(operator, threshold), order=2
    )

    actual = fx(monthly_frame(), np.asarray(source).reshape(-1, 1))

    np.testing.assert_equal(actual, np.full(12, expected))


def test_scalar_annual_ratio_is_undefined_when_trials_are_unknown():
    frame = monthly_frame()
    source = np.array([1, 0] + [np.nan] * 10)

    actual = water_year_probability_ratio(
        source, frame['dowy'].to_numpy(), timestamps=frame.index
    )

    np.testing.assert_equal(actual, np.full(12, np.nan))


@pytest.mark.parametrize(
    'source, expected',
    [
        ([0] * 11 + [np.nan], np.nan),
        ([np.nan] * 12, np.nan),
        ([0] * 12, 0),
        ([0] * 10 + [np.nan, 1], 1),
    ],
)
def test_annual_reduction_requires_definite_failure_or_success(source, expected):
    frame = monthly_frame()

    actual = or_reduce_per_water_year(
        np.asarray(source), frame['dowy'].to_numpy(), frame.index
    )

    np.testing.assert_equal(actual[-1], expected)
    assert np.isnan(actual[:-1]).all()


@pytest.mark.parametrize('exclusive', [False, True])
@pytest.mark.parametrize(
    'predicate',
    [comparison_fx('>=', 2), comparison_fx('<=', 2, '<=', 3)],
)
def test_intra_annual_count_windows_keep_unknown_counts(predicate, exclusive):
    frame = monthly_frame()
    source = np.array([1, np.nan] + [0] * 10).reshape(-1, 1)
    fx = nested_frequency_intra_annual_fx(
        predicate, order=2, big_n=3, exclusive_windows=exclusive
    )

    actual = fx(frame, source)

    np.testing.assert_equal(actual, [np.nan] * 3 + [0] * 9)


@pytest.mark.parametrize(
    'exclusive, expected',
    [
        (False, [np.nan, 1, 1, 1]),
        (True, [np.nan, 1, 1, np.nan]),
    ],
)
def test_unknown_annual_anchor_preserves_interannual_windows(exclusive, expected):
    frame = monthly_frame(4)
    annual_source = np.repeat([np.nan, 1, 0, 0], 12)
    output = np.column_stack([np.ones(48), annual_source])
    fx = nested_frequency_interannual_fx(
        comparison_fx('>=', 1), order=3, big_n=3,
        exclusive_windows=exclusive,
    )

    actual = fx(frame, output)

    np.testing.assert_equal(actual, np.repeat(expected, 12))

@pytest.mark.parametrize('exclusive', [False, True])
def test_correlated_annual_equality_windows_settle_final_year(exclusive):
    frame = monthly_frame(2)
    output = np.column_stack([np.ones(24), np.repeat([1, np.nan], 12)])
    fx = nested_frequency_interannual_fx(
        comparison_fx('=', 1), order=3, big_n=2,
        exclusive_windows=exclusive,
    )

    actual = fx(frame, output)

    np.testing.assert_equal(actual, [np.nan] * 12 + [1] * 12)


@pytest.mark.parametrize(
    'predicate, expected',
    [
        (comparison_fx('=', 0), [np.nan, 1, 1]),
        (comparison_fx('<=', 1, '<=', 2), [np.nan, 1, 1]),
    ],
)
@pytest.mark.parametrize('exclusive', [False, True])
def test_interannual_zero_and_between_predicates(predicate, expected, exclusive):
    frame = monthly_frame(3)
    source = (
        [np.nan, 0, 0] if predicate(0) else [np.nan, 1, 1]
    )
    output = np.column_stack([np.ones(36), np.repeat(source, 12)])
    fx = nested_frequency_interannual_fx(
        predicate, order=3, big_n=2, exclusive_windows=exclusive,
    )

    actual = fx(frame, output)

    np.testing.assert_equal(actual, np.repeat(expected, 12))


def test_annual_fraction_conjunction_preserves_settled_failures():
    source = np.column_stack([np.full(12, np.nan), np.zeros(12)])
    fx = nested_frequency_intra_annual_fx(
        comparison_fx('>=', 0.5), order=3,
    )

    actual = fx(monthly_frame(), source)

    np.testing.assert_array_equal(actual, np.zeros(12))


def test_nested_unknowns_broadcast_and_exclude_both_partial_years():
    frame = monthly_frame(3).iloc[6:-6]
    source = np.ones((len(frame), 1))
    source[6:18] = np.nan
    intra_fx = nested_frequency_intra_annual_fx(
        comparison_fx('>=', 0.5), order=2
    )
    intra = intra_fx(frame, source)
    inter_fx = nested_frequency_interannual_fx(
        comparison_fx('>=', 1), order=3, big_n=1
    )

    actual = inter_fx(frame, np.column_stack([source, intra]))

    assert np.isnan(intra).all()
    assert np.isnan(actual).all()


def test_nested_fraction_component_retains_terminal_unknown_verdict():
    frame = monthly_frame(2)
    source = np.array([1, 0] + [np.nan] * 10 + [1] * 12)
    component = Component('component', [
        Characteristic(
            'source', lambda _df, _output: source,
            CharacteristicType.MAGNITUDE,
        ),
        Characteristic(
            'intra',
            nested_frequency_intra_annual_fx(
                comparison_fx('>=', 0.5), order=2,
            ),
            CharacteristicType.FREQUENCY,
        ),
        Characteristic(
            'interannual',
            nested_frequency_interannual_fx(
                comparison_fx('>=', 1), order=3, big_n=1,
            ),
            CharacteristicType.FREQUENCY, True,
        ),
    ], True)

    actual = evaluate_component(
        frame, component, first_day_of_water_year=1
    ).df

    np.testing.assert_equal(
        actual['component'].to_numpy(), [np.nan] * 12 + [1] * 12
    )


@pytest.mark.parametrize('operator', ['<', '<=', '>', '>=', '=', '!='])
def test_annual_fraction_matches_binary_completion_oracle(operator):
    frame = monthly_frame()
    predicate = comparison_fx(operator, 0.5)
    fx = nested_frequency_intra_annual_fx(predicate, order=2)
    for prefix in itertools.product((0, 1, np.nan), repeat=3):
        source = np.array([*prefix, *([1] * 4), *([0] * 5)], dtype=float)
        unknown_indices = np.flatnonzero(np.isnan(source))
        possible = []
        for assignment in itertools.product((0, 1), repeat=len(unknown_indices)):
            completed = source.copy()
            completed[unknown_indices] = assignment
            possible.append(predicate(completed.sum() / 12))
        expected = (
            float(possible[0]) if all(value == possible[0] for value in possible)
            else np.nan
        )

        actual = fx(frame, source.reshape(-1, 1))

        np.testing.assert_equal(actual, np.full(12, expected))
