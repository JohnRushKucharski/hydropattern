'''Entry point for the hydropattern command line interface.'''

import tomllib
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from climate_canvas.plots_utilities import plot_response_surface  # type: ignore[import-untyped]

from hydropattern.errors import ParserErrorCode, raise_parser_error
from hydropattern.formatters import build_summary_sheet, write_results
from hydropattern.parsers import (
    MetricOptions,
    build_components,
    parse_metric_options,
    parse_request,
)
from hydropattern.patterns import Component, Result, evaluate_components
from hydropattern.scenario_grid import build_grid, require_scenario_grid
from hydropattern.timeseries import Timeseries

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
                                             help='''Directory for output files.
                                             By default, '_output' is appended to the path
                                             file name, and a directory with that name is
                                             created in the path directory (used for both
                                             Excel and csv output).'''),
        write_to_excel: bool = typer.Option(True, "--excel/--no-excel",
                                            help='''If true (default), all outputs are written
                                            to Excel files.
                                            Use --no-excel to write per-scenario csv files
                                            instead.'''),
        overwrite: bool = typer.Option(True, "--overwrite/--no-overwrite",
                                       help='''If true (default), existing output files
                                       are replaced on each run.
                                       If false, a numeric suffix is appended to avoid
                                       overwriting existing files.'''),
        interp: bool = typer.Option(True, "--interp/--no-interp",
                                    help='''If true (default), interpolate the response
                                    surface plot to a finer grid. Only affects --plot.'''),
        show: bool = typer.Option(False, "--show",
                                  help='''Also open an interactive window for each
                                  response surface plot. Only affects --plot.''')):
    '''Run the hydropattern command line interface.'''
    data = load_config_file(path)
    timeseries = load_timeseries(data)
    components = load_components(data)
    metric_options = load_metric_options(data)
    scenarios = split_scenarios(timeseries.data)
    scenario_results = {name: evaluate_components(df, components)
                        for name, df in scenarios.items()}
    output_path = write_output(scenario_results, path, output_directory, write_to_excel,
                               overwrite, timeseries.first_day_of_water_year, metric_options)
    if plot:
        plot_components(scenario_results, output_path, metric_options,
                        timeseries.first_day_of_water_year, interp, show)

def load_config_file(path: str) -> dict[str, Any]:
    '''Load a configuration file.'''
    with open(path, 'rb') as file:
        data = tomllib.load(file)
    return data

def load_timeseries(data: dict[str, Any]) -> Timeseries:
    '''Parse a timeseries from the configuration file.'''
    if 'timeseries' not in data:
        raise_parser_error(
            ParserErrorCode.MISSING_SECTION,
            'No timeseries data in configuration file.',
            section='timeseries',
        )
    ts_data = data['timeseries']
    if 'path' not in ts_data:
        raise_parser_error(
            ParserErrorCode.MISSING_FIELD,
            'No path in timeseries data.',
            section='timeseries',
            field='path',
        )
    path = ts_data['path']
    first_day_of_water_year = (
        ts_data['first_day_of_water_year'] if 'first_day_of_water_year' in ts_data else 1
    )
    ext = Path(path).suffix.lower()
    if ext in ('.xlsx', '.xls'):
        sheet_name = ts_data.get('sheet_name', 0)
        date_format = ts_data.get('date_format', '')
        return Timeseries.from_excel(path, first_day_of_water_year, date_format, sheet_name)
    date_format = ts_data.get('date_format', '')
    return Timeseries.from_csv(path, first_day_of_water_year, date_format)

def split_scenarios(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    '''Split a multi-column timeseries into one DataFrame per scenario.

    The last column is always 'dowy' and is included in every scenario slice.
    Each returned DataFrame has exactly two columns: the scenario data column
    and 'dowy', matching the shape expected by evaluate_component.

    A single-column timeseries (one data column + dowy) returns a dict with
    one entry — the degenerate single-scenario case.
    '''
    dowy_col = data.columns[-1]
    return {col: data[[col, dowy_col]] for col in data.columns[:-1]}

def load_components(data: dict[str, Any]) -> list[Component]:
    '''Parse components from the configuration file.'''
    if 'components' not in data:
        raise_parser_error(
            ParserErrorCode.MISSING_SECTION,
            'No components data in configuration file.',
            section='components',
        )
    return build_components(parse_request(data['components']))

def load_metric_options(data: dict[str, Any]) -> MetricOptions:
    '''Parse the optional [metric] section from the configuration file.

    Absent [metric] section -> MetricOptions() (default mode: portion).
    '''
    return parse_metric_options(data)

def write_output(scenario_results: dict[str, list[Result]],
                 input_path: str, output_directory: str | None,
                 write_to_excel: bool, overwrite: bool = True,
                 first_day_of_wy: int = 1,
                 metric_options: MetricOptions = MetricOptions()) -> Path:
    '''Write output using the formatter entrypoint. Returns the resolved output path.'''
    output_path = write_results(scenario_results, input_path, output_directory,
                                write_to_excel, overwrite, first_day_of_wy,
                                metric_options.mode)
    if write_to_excel:
        output_file = output_path / (Path(input_path).stem + '_output.xlsx')
        typer.echo(f'Output written to: {output_file}.')
        return output_path
    typer.echo(f'Output written to: {output_path}.')
    return output_path

def plot_components(scenario_results: dict[str, list[Result]],
                    output_path: Path, metric_options: MetricOptions,
                    first_day_of_wy: int, interpolate: bool, show: bool) -> None:
    '''Save one response-surface grid csv + plot png per component to output_path.

    Requires scenario names to form a valid precip/temp scenario grid (see
    hydropattern.scenario_grid). Raises HydropatternError otherwise.
    '''
    first_scenario_results = next(iter(scenario_results.values()))
    scenario_names = list(scenario_results.keys())
    require_scenario_grid(scenario_names)
    for result in first_scenario_results:
        component = result.component
        summary = build_summary_sheet(scenario_results, component.name, component.name,
                                      first_day_of_wy, metric_options.mode)
        metric_values = summary.loc['total'].to_dict()
        xs, ys, zs = build_grid(scenario_names, metric_values)
        write_grid_csv(xs, ys, zs, output_path / f'{component.name}_grid.csv')
        plot_response_surface(
            xs, ys, zs, interpolate=interpolate,
            labels=('Precipitation Delta (%)', 'Temperature Delta (C)', metric_options.mode.value),
            title=component.name,
            save_path=output_path / f'{component.name}_plot.png',
            show=show,
        )

def write_grid_csv(xs, ys, zs, path: Path) -> None:
    '''Write a (precip_delta x temp_delta) grid to csv: rows=temp deltas, columns=precip deltas.'''
    pd.DataFrame(zs, index=ys, columns=xs).to_csv(path, index_label='temp_delta\\precip_delta')
