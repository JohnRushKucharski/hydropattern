'''Entry point for the hydropattern command line interface.'''

import tomllib
from typing import Any

import typer
import numpy as np
from climate_canvas.plots_utilities import plot_response_surface

from hydropattern.timeseries import Timeseries
from hydropattern.parsers import parse_components
from hydropattern.patterns import Component, evaluate_patterns

app = typer.Typer(no_args_is_help=True)

@app.callback()
def callback():
    '''hydropattern command line interface.'''

@app.command()
def run(path: str = typer.Argument(..., help='''
                                   Path to *.toml configuration file.
                                   ''')):
    '''Run the hydropattern command line interface.'''
    data = load_config_file(path)
    timeseries = load_timeseries(data)
    components = load_components(data)
    output = evaluate_patterns(timeseries.data, components)
    typer.echo(output)
    xs, ys, zs = np.array([0, 0.5, 1]), np.array([0, 1]), np.array([[2, 1.9, 1], [5, 4.5, 4]])
    plot_response_surface(xs, ys, zs, interpolate=True)

def load_config_file(path: str) -> dict[str, Any]:
    '''Load a configuration file.'''
    with open(path, 'rb') as file:
        data = tomllib.load(file)
    return data

def load_timeseries(data: dict[str, Any]) -> Timeseries:
    '''Parse a timeseries from the configuration file.'''
    # todo: test inputs improve error handling
    if 'timeseries' not in data:
        raise ValueError('No timeseries data in configuration file.')
    ts_data = data['timeseries']
    if 'path' not in ts_data:
        raise ValueError('No path in timeseries data.')
    path = ts_data['path']
    date_format = ts_data['date_format'] if 'date_format' in ts_data else ''
    first_day_of_water_year = ts_data['first_day_of_water_year'] if 'first_day_of_water_year' in ts_data else 1 # pylint: disable=line-too-long
    return Timeseries.from_csv(path, first_day_of_water_year, date_format)

def load_components(data: dict[str, Any]) -> list[Component]:
    '''Parse components from the configuration file.'''
    if 'components' not in data:
        raise ValueError('No components data in configuration file.')
    return parse_components(data['components'])
