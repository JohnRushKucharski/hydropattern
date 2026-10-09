'''Phase 7 acceptance tests for selecting a data column for component evaluation.'''

import numpy as np
import pandas as pd
import pytest

from hydropattern.parsers import build_components, parse_request
from hydropattern.patterns import evaluate_component, evaluate_components
from hydropattern.scenarios import evaluate_scenarios
from hydropattern.timeseries import Timeseries


def _dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'low_flow': [1.0, 2.0, 3.0],
            'target_flow': [10.0, 1.0, 10.0],
            'dowy': [1, 2, 3],
        },
        index=pd.date_range('2020-01-01', periods=3, name='time'),
    )


def test_component_evaluation_returns_only_selected_data_column_under_its_name():
    component = build_components(
        parse_request({'pulse': {'characteristics': [
            {'type': 'magnitude', 'parameters': ['>', 5.0]},
        ]}})
    )[0]

    result = evaluate_component(_dataframe(), component, data_column=1)

    assert result.df.columns.tolist() == [
        'target_flow', 'dowy', 'magnitude_gt5.0', 'pulse'
    ]
    assert result.dv_name == 'target_flow'
    assert result.df['target_flow'].tolist() == [10.0, 1.0, 10.0]
    np.testing.assert_array_equal(result.df['magnitude_gt5.0'], [1, 0, 1])
    np.testing.assert_array_equal(result.df['pulse'], [1, 0, 1])
    pd.testing.assert_index_equal(result.df.index, _dataframe().index)


def test_default_component_evaluation_returns_first_data_column_under_original_name():
    component = build_components(
        parse_request({'pulse': {'characteristics': [
            {'type': 'magnitude', 'parameters': ['>', 2.0]},
        ]}})
    )[0]

    result = evaluate_component(_dataframe(), component)

    assert result.df.columns.tolist() == [
        'low_flow', 'dowy', 'magnitude_gt2.0', 'pulse'
    ]
    assert result.dv_name == 'low_flow'
    assert result.df['low_flow'].tolist() == [1.0, 2.0, 3.0]
    np.testing.assert_array_equal(result.df['pulse'], [0, 0, 1])


def test_component_list_evaluation_forwards_selected_data_column():
    components = build_components(
        parse_request({
            'above': {'characteristics': [
                {'type': 'magnitude', 'parameters': ['>', 5.0]},
            ]},
            'below': {'characteristics': [
                {'type': 'magnitude', 'parameters': ['<', 5.0]},
            ]},
        })
    )

    results = evaluate_components(_dataframe(), components, data_column=1)

    np.testing.assert_array_equal(results[0].df['above'], [1, 0, 1])
    np.testing.assert_array_equal(results[1].df['below'], [0, 1, 0])
    assert [result.dv_name for result in results] == ['target_flow', 'target_flow']


def test_scenario_evaluation_result_contains_only_its_evaluated_flow_column():
    source = _dataframe().drop(columns='dowy')
    timeseries = Timeseries.from_dataframe(source)
    component = build_components(
        parse_request({'pulse': {'characteristics': [
            {'type': 'magnitude', 'parameters': ['>', 5.0]},
        ]}})
    )[0]

    result = evaluate_scenarios(timeseries, [component]).scenario_results['target_flow'][0]

    assert result.df.columns.tolist() == [
        'target_flow', 'dowy', 'magnitude_gt5.0', 'pulse'
    ]
    assert result.dv_name == 'target_flow'


def test_selected_flow_drives_magnitude_and_frequency_source():
    component = build_components(
        parse_request({'pulse': {'characteristics': [
            {'type': 'magnitude', 'parameters': ['>', 5.0]},
            {'type': 'frequency', 'parameters': ['>=', 2, 3]},
        ]}})
    )[0]

    result = evaluate_component(_dataframe(), component, data_column=1)

    np.testing.assert_array_equal(result.df['magnitude_gt5.0'], [1, 0, 1])
    np.testing.assert_array_equal(result.df['frequency_ge2in3(union)'], [1, 1, 1])
    np.testing.assert_array_equal(result.df['pulse'], [1, 1, 1])


def test_selected_flow_drives_rate_of_change():
    component = build_components(
        parse_request({'pulse': {'characteristics': [
            {'type': 'rate_of_change', 'parameters': ['>', 2.0]},
        ]}})
    )[0]

    result = evaluate_component(_dataframe(), component, data_column=1)

    np.testing.assert_array_equal(result.df['rate_of_change_gt2.0'], [np.nan, 0, 1])


@pytest.mark.parametrize('data_column', [-1, 2, 3, True, 1.0])
def test_component_evaluation_rejects_invalid_data_column(data_column):
    component = build_components(
        parse_request({'pulse': {'characteristics': [
            {'type': 'magnitude', 'parameters': ['>', 5.0]},
        ]}})
    )[0]

    with pytest.raises(ValueError, match='data_column'):
        evaluate_component(_dataframe(), component, data_column=data_column)


def test_component_evaluation_allows_flow_column_named_dv():
    data = _dataframe().rename(columns={'low_flow': 'dv'})
    component = build_components(
        parse_request({'pulse': {'characteristics': [
            {'type': 'magnitude', 'parameters': ['>', 5.0]},
        ]}})
    )[0]

    result = evaluate_component(data, component)

    assert result.dv_name == 'dv'
    assert result.df.columns.tolist() == ['dv', 'dowy', 'magnitude_gt5.0', 'pulse']


def test_component_evaluation_preserves_dv_as_original_column_name():
    data = _dataframe().rename(columns={'target_flow': 'dv'})
    component = build_components(
        parse_request({'pulse': {'characteristics': [
            {'type': 'magnitude', 'parameters': ['>', 5.0]},
        ]}})
    )[0]

    result = evaluate_component(data, component)

    assert result.dv_name == 'low_flow'
    assert result.df.columns.tolist() == ['low_flow', 'dowy', 'magnitude_gt5.0', 'pulse']


def test_component_evaluation_rejects_result_column_name_collisions():
    data = _dataframe().rename(columns={'low_flow': 'magnitude_gt5.0'})
    component = build_components(
        parse_request({'pulse': {'characteristics': [
            {'type': 'magnitude', 'parameters': ['>', 5.0]},
        ]}})
    )[0]

    with pytest.raises(ValueError, match='column names must be unique'):
        evaluate_component(data, component)


def test_component_evaluation_rejects_invalid_final_dowy_column():
    data = _dataframe()
    data.loc[data.index[0], 'dowy'] = 0
    component = build_components(
        parse_request({'pulse': {'characteristics': [
            {'type': 'magnitude', 'parameters': ['>', 5.0]},
        ]}})
    )[0]

    with pytest.raises(ValueError, match='day of water year column in last position'):
        evaluate_component(data, component, data_column=1)


def test_component_evaluation_names_datetime_index_time():
    data = _dataframe()
    data.index.name = 'observation'
    component = build_components(
        parse_request({'pulse': {'characteristics': [
            {'type': 'magnitude', 'parameters': ['>', 5.0]},
        ]}})
    )[0]

    result = evaluate_component(data, component, data_column=1)

    assert result.df.index.name == 'time'
    pd.testing.assert_index_equal(result.df.index, data.index.rename('time'))
