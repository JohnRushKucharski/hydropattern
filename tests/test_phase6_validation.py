'''Phase 6 rate-of-change and frequency validation acceptance tests.'''

import numpy as np
import pandas as pd
import pytest

from hydropattern.errors import HydropatternError, ParserErrorCode
from hydropattern.parsers import (
    ComponentSpec,
    Request,
    build_components,
    frequency_parser,
    parse_request,
    rate_of_change_parser,
    validate_frequency_metrics,
)
from hydropattern.patterns import comparison_fx, rate_of_change_fx


def test_rate_of_change_minimum_applies_to_denominator_not_current_flow():
    df = pd.DataFrame({'flow': [4.0, 1.0]})
    fx = rate_of_change_fx(comparison_fx('<', 0.5), minimum=1.0)

    assert np.array_equal(fx(df), [0, 1])


def test_rate_of_change_minimum_is_forwarded_from_component_configuration():
    request = parse_request(
        {
            'pulse': {
                'characteristics': [
                    {'type': 'rate_of_change', 'parameters': ['<', 0.5, 1, 1, 1.0]},
                ]
            }
        }
    )
    component = build_components(request)[0]

    assert component.characteristics[0].fx(pd.DataFrame({'flow': [4.0, 1.0]}), None).tolist() == [
        0,
        1,
    ]


def test_rate_of_change_rejects_zero_and_negative_denominators():
    df = pd.DataFrame({'flow': [0.0, 4.0, -2.0, 4.0]})
    fx = rate_of_change_fx(comparison_fx('<', 0.5))

    with np.errstate(divide='raise', invalid='raise'):
        result = fx(df)

    assert result.tolist() == [0, 0, 1, 0]


def test_rate_of_change_minimum_applies_to_smoothed_denominator_only():
    df = pd.DataFrame({'flow': [2.0, 4.0, 1.0, 1.0]})
    fx = rate_of_change_fx(comparison_fx('<', 1.0), ma_periods=2, minimum=2.0)

    assert fx(df).tolist() == [0, 0, 1, 1]


def test_rate_of_change_minimum_rejects_boolean_as_non_numeric_configuration():
    with pytest.raises(HydropatternError) as exc_info:
        rate_of_change_parser(['>', 1.0, 1, 1, True], order=1)

    assert exc_info.value.envelope.code == str(ParserErrorCode.INVALID_TYPE)


@pytest.mark.parametrize(
    'metrics',
    [
        ['>=', 3, 3],
        ['=', 3, 3],
        [1, 3, 3],
        ['=', 0, 3],
        ['!=', 0, 3],
        ['!=', 3, 3],
    ],
)
def test_frequency_validation_accepts_attainable_count_endpoints(metrics):
    parsed = validate_frequency_metrics(metrics)

    assert parsed.big_n == metrics[-1]


@pytest.mark.parametrize(
    'metrics',
    [
        ['<', 0, 3],
        ['>', 3, 3],
    ],
)
def test_frequency_validation_rejects_impossible_count_predicates(metrics):
    with pytest.raises(HydropatternError) as exc_info:
        validate_frequency_metrics(metrics)

    assert exc_info.value.envelope.code == str(ParserErrorCode.INVALID_VALUE)


@pytest.mark.parametrize(
    'metrics',
    [
        ['>=', True, 2],
        [0, True, 2],
    ],
)
def test_frequency_validation_rejects_boolean_count_parameters(metrics):
    with pytest.raises(HydropatternError) as exc_info:
        validate_frequency_metrics(metrics)

    assert exc_info.value.envelope.code == str(ParserErrorCode.INVALID_TYPE)


@pytest.mark.parametrize(
    'metrics,expected',
    [
        (['!=', 0, 2], [1, 1, 0, 0]),
        (['!=', 1, 2], [0, 1, 1, 1]),
    ],
)
def test_not_equal_anchor_behavior_follows_zero_count_truth(metrics, expected):
    source = [1, 0, 0, 0]
    df = pd.DataFrame({'flow': source, 'dowy': [1, 2, 3, 4]})
    output = np.asarray(source, dtype=float).reshape(-1, 1)
    characteristic = frequency_parser(metrics, order=2)

    assert characteristic.fx(df, output).tolist() == expected


def test_empty_component_uses_repository_parser_error():
    with pytest.raises(HydropatternError) as exc_info:
        parse_request({'pulse': {'characteristics': []}})

    assert exc_info.value.envelope.code == str(ParserErrorCode.EMPTY_COMPONENT)


def test_direct_empty_component_request_is_rejected_by_builder():
    request = Request(
        components=(ComponentSpec(name='pulse', characteristics=()),)
    )

    with pytest.raises(HydropatternError) as exc_info:
        build_components(request)

    assert exc_info.value.envelope.code == str(ParserErrorCode.EMPTY_COMPONENT)


def test_frequency_before_another_characteristic_is_rejected():
    request = parse_request(
        {
            'pulse': {
                'characteristics': [
                    {'type': 'frequency', 'parameters': ['>=', 1, 2]},
                    {'type': 'magnitude', 'parameters': ['>', 1.0]},
                ]
            }
        }
    )

    with pytest.raises(HydropatternError) as exc_info:
        build_components(request)

    assert exc_info.value.envelope.code == str(ParserErrorCode.FREQUENCY_NOT_LAST)
