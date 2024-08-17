'''Entry point for the hydropattern command line interface.'''

import tomllib
from typing import Any
from pathlib import Path

import typer
import numpy as np
import pandas as pd
from climate_canvas.plots_utilities import plot_response_surface

from hydropattern.timeseries import Timeseries
from hydropattern.parsers import parse_components
from hydropattern.patterns import Component, evaluate_patterns

app = typer.Typer(no_args_is_help=True)

@app.callback()
def callback():
    '''hydropattern command line interface.'''

@app.command()
def run(path: str = typer.Argument(...,
                                   help='Path to *.toml configuration file.'),
        plot: bool = typer.Option(False, "--plot",
                                  help='Plot response surface.'),
        output_directory: str = typer.Option(None, "--output-dir",
                                             help='''Directory for output .csvs.
                                             If write_to_excel is False, 
                                             by default
                                            '_output' is appended to the path file name,
                                             and a directory with that name is created in 
                                             the path directory.
                                             If write_to_excel is True,
                                             by default the output file is written
                                             to the path directory 
                                             and no output directory is created.'''),
        write_to_excel: bool = typer.Option(False, "--excel",
                                            help='''If true, all outputs are written
                                            to a single Excel file.
                                            If false, each time series output is written
                                            to a separate .csv file.''')):
    '''Run the hydropattern command line interface.'''
    data = load_config_file(path)
    timeseries = load_timeseries(data)
    components = load_components(data)
    output = evaluate_patterns(timeseries.data, components)
    write_output(output, path, output_directory, write_to_excel)

    if plot:
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

def write_output(dfs: list[pd.DataFrame],
                 input_path: str, output_directory: str, write_to_excel: bool):
    '''Write output to .csv files or an Excel file.'''
    if output_directory:
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(input_path).parent
        if not write_to_excel:
            output_path = Path(input_path).parent/(Path(input_path).stem + '_output')
            output_path.mkdir(parents=True, exist_ok=True)
    if write_to_excel:
        output_filename = Path(input_path).stem + '_output.xlsx'
        writer = pd.ExcelWriter(output_path/output_filename)
        for _, df in enumerate(dfs):
            df.to_excel(writer, sheet_name=df.columns[0])
        writer.close()
        typer.echo(f'Output written to: {output_path}{chr(92)}{output_filename}.')
    else:
        for _, df in enumerate(dfs):
            output_filename = f'{df.columns[0]}.csv'
            df.to_csv(output_path/output_filename)
        typer.echo(f'Output written to: {output_path}.')
