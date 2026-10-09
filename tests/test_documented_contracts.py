'''Executable checks for examples and golden outputs published in the docs.'''

import re
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from hydropattern.parsers import build_components, parse_request
from hydropattern.patterns import evaluate_component
from hydropattern.timeseries import Timeseries

ROOT = Path(__file__).resolve().parents[1]


def test_api_frequency_example_produces_published_golden_array(capsys):
    page = ROOT / 'docs' / 'user' / 'api' / 'index.md'
    blocks = re.findall(r'```python\n(.*?)```', page.read_text(encoding='utf-8'), re.DOTALL)
    namespace = {}
    exec(compile(blocks[0], str(page), 'exec'), namespace)
    result = namespace['result']
    assert capsys.readouterr().out.strip() == (
        '[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]'
    )

    np.testing.assert_array_equal(
        result.df['frequency_ge1in5(union)'],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    )
    np.testing.assert_array_equal(
        result.df['pulse'],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    )
    assert result.df.columns.tolist() == [
        'flow',
        'dowy',
        'magnitude_gt0',
        'frequency_ge1in5(union)',
        'pulse',
    ]


def test_frequency_example_toml_is_valid_and_uses_all_supported_schemas():
    config_path = ROOT / 'examples' / 'frequency.toml'
    config = tomllib.loads(config_path.read_text(encoding='utf-8'))
    timeseries_path = ROOT / config['timeseries']['path']
    assert timeseries_path.is_file()

    request = parse_request(config['components'])
    components = build_components(request)
    assert len(components) == 3


def test_frequency_example_runs_and_preserves_named_scenario_data_column():
    config_path = ROOT / 'examples' / 'frequency.toml'
    config = tomllib.loads(config_path.read_text(encoding='utf-8'))
    timeseries_path = ROOT / config['timeseries']['path']
    timeseries = Timeseries.from_csv(
        str(timeseries_path),
        first_dowy=config['timeseries'].get('first_day_of_water_year', 1),
        date_format=config['timeseries']['date_format'],
    )
    components = build_components(parse_request(config['components']))
    results = [
        evaluate_component(scenario, components[0])
        for scenario in (
            timeseries.data.loc[:, [column, timeseries.data.columns[-1]]]
            for column in timeseries.data.columns[:-1]
        )
    ]

    assert results
    assert results[0].df.columns[0] == '0-0'
    assert results[0].dv_name == '0-0'


def test_reference_nested_frequency_matrices_match_published_golden_arrays():
    cases = [
        (
            [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            ['>=', 1, 2],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        (
            [1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            ['>=', 1, 2, True],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        ),
    ]

    for source, outer_metrics, expected in cases:
        data = pd.DataFrame(
            {'flow': source, 'dowy': np.tile(range(1, 5), 3)}
        )
        request = parse_request(
            {
                'nested': {
                    'magnitude': ['>', 0],
                    'frequency': [['>=', 0.5], outer_metrics],
                }
            }
        )
        result = evaluate_component(data, build_components(request)[0])

        np.testing.assert_array_equal(result.df['nested'], expected)
