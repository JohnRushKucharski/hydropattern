'''Entry point for the hydropattern command line interface.'''

import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

import typer

from hydropattern.errors import CliErrorCode, ParserErrorCode, raise_cli_error, raise_parser_error
from hydropattern.formatters import plot_components, write_results
from hydropattern.parsers import (
    MetricOptions,
    OutputOptions,
    build_components,
    collect_explicit_options,
    merge_overrides,
    parse_metric_options,
    parse_output_options,
    parse_request,
    parse_timeseries_spec,
)
from hydropattern.parsing.specs import is_valid_minimum_coverage
from hydropattern.patterns import Component, Result
from hydropattern.scenarios import evaluate_scenarios
from hydropattern.timeseries import Timeseries

app = typer.Typer(no_args_is_help=True)

@app.callback()
def callback():
    '''hydropattern command line interface.'''

@app.command()
# Typer command signature intentionally mirrors public CLI flags.
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def run(path: str = typer.Argument(...,
                                   help='Path to *.toml configuration file.'),
        plot: bool = typer.Option(None, "--plot/--no-plot",
                                  rich_help_panel="Output",
                                  help='''Plot response surface. Defaults to the
                                  configuration file's [output.plot].enabled
                                  (false if unset).'''),
        output_directory: str = typer.Option(None, "--output-dir",
                                             rich_help_panel="Output",
                                             help='''Directory for output files.
                                             Defaults to the configuration file's
                                             [output].directory. If neither is given,
                                             '_output' is appended to the path file name,
                                             and a directory with that name is created
                                             in the path directory (used for both Excel
                                             and csv output).'''),
        write_to_excel: bool = typer.Option(None, "--excel/--no-excel",
                                            rich_help_panel="Output",
                                            help='''If true, all outputs are written
                                            to Excel files. Use --no-excel to write
                                            per-scenario csv files instead. Defaults to
                                            the configuration file's [output].excel
                                            (true if unset).'''),
        overwrite: bool = typer.Option(None, "--overwrite/--no-overwrite",
                                       rich_help_panel="Output",
                                       help='''If true, existing output files are
                                       replaced on each run. If false, a numeric suffix
                                       is appended to avoid overwriting existing files.
                                       Defaults to the configuration file's
                                       [output].overwrite (true if unset).'''),
        interp: bool = typer.Option(None, "--interp/--no-interp",
                                    rich_help_panel="Climate Canvas Plot",
                                    help='''If true, interpolate the response surface
                                    plot to a finer grid. Only affects --plot. Defaults
                                    to the configuration file's
                                    [output.plot.climate-canvas].interpolate
                                    (true if unset).'''),
        show: bool = typer.Option(None, "--show/--no-show",
                                  rich_help_panel="Climate Canvas Plot",
                                  help='''Also open an interactive window for each
                                  response surface plot. Only affects --plot. Defaults
                                  to the configuration file's
                                  [output.plot.climate-canvas].show (false if unset).'''),
        threshold: float = typer.Option(None, "--threshold",
                                        rich_help_panel="Climate Canvas Plot",
                                        help='''Z-value centered on the colormap for
                                        each response surface plot. Only affects --plot.
                                        Defaults to the configuration file's
                                        [output.plot.climate-canvas].threshold (the
                                        midpoint of the z-value range if unset).'''),
        color_map: str = typer.Option(None, "--color-map",
                                      rich_help_panel="Climate Canvas Plot",
                                      help='''Matplotlib colormap name for the response
                                      surface plot. Only affects --plot. Defaults to
                                      the configuration file's
                                      [output.plot.climate-canvas].color_map
                                      ('RdBu' if unset).'''),
        color_map_ticks: list[float] = typer.Option(None, "--color-map-ticks",
                                                    rich_help_panel="Climate Canvas Plot",
                                                    help='''Explicit colorbar tick
                                                    value. Repeat for multiple ticks.
                                                    Only affects --plot. Defaults to
                                                    the configuration file's
                                                    [output.plot.climate-canvas].color_map_ticks
                                                    (unset).'''),
        fillin: bool = typer.Option(None, "--fillin/--no-fillin",
                                    rich_help_panel="Climate Canvas Plot",
                                    help='''If true, estimate missing (NaN) grid cells
                                    in the response surface via Delaunay triangulation.
                                    Only affects --plot. Defaults to the configuration
                                    file's [output.plot.climate-canvas].fillin
                                    (false if unset).'''),
        minimum_coverage: float = typer.Option(
            None, "--minimum-coverage",
            rich_help_panel="Climate Canvas Plot",
            help='''Minimum known-outcome coverage required to include a scenario
            in a response-surface plot, as a fraction from 0 to 1. Defaults to
            [output.plot].minimum_coverage (0.9 if unset).''',
        ),
        run_toml_options: bool = typer.Option(False, "--run-toml-options/--override-toml-options",
                                              rich_help_panel="Output",
                                              help='''If true, run exactly as specified in
                                              the configuration file's [output] section;
                                              no other output-related CLI option
                                              (--plot/--no-plot, --output-dir, --excel/--no-excel,
                                              --overwrite/--no-overwrite, --interp/--no-interp,
                                              --show/--no-show, --threshold, --color-map,
                                              --color-map-ticks, --fillin/--no-fillin,
                                              --minimum-coverage)
                                              may also be passed explicitly,
                                              or a CLI_CONFLICTING_OPTIONS error is raised.
                                              If false (default), any explicit CLI option
                                              overrules a conflicting configuration file
                                              option.''')):
    '''Run the hydropattern command line interface.'''
    if run_toml_options:
        require_no_conflicting_cli_options(
            plot=plot, output_directory=output_directory, write_to_excel=write_to_excel,
            overwrite=overwrite, interp=interp, show=show, threshold=threshold,
            color_map=color_map, color_map_ticks=color_map_ticks, fillin=fillin,
            minimum_coverage=minimum_coverage,
        )
    if minimum_coverage is not None and not is_valid_minimum_coverage(minimum_coverage):
        raise typer.BadParameter(
            'must be a finite fraction between 0 and 1.',
            param_hint='--minimum-coverage',
        )
    data = load_config_file(path)
    timeseries = load_timeseries(data)
    components = load_components(data)
    output_options = resolve_output_options(data, plot, output_directory, write_to_excel,
                                            overwrite, interp, show, threshold,
                                            color_map, color_map_ticks, fillin,
                                            minimum_coverage)
    scenario_results = evaluate_scenarios(timeseries, components).scenario_results
    output_path = write_output(scenario_results, path, output_options.directory,
                               output_options.excel, output_options.overwrite,
                               timeseries.first_day_of_water_year, output_options.metric)
    if output_options.plot.enabled:
        plot_components(scenario_results, output_path, output_options.metric,
                        timeseries.first_day_of_water_year, output_options.plot.climate_canvas,
                        output_options.plot.minimum_coverage)

# Signature mirrors explicit CLI override surface.
# pylint: disable=too-many-arguments,too-many-positional-arguments
def require_no_conflicting_cli_options(
        plot: bool | None,
        output_directory: str | None,
        write_to_excel: bool | None,
        overwrite: bool | None,
        interp: bool | None,
        show: bool | None,
        threshold: float | None = None,
        color_map: str | None = None,
        color_map_ticks: list[float] | None = None,
        fillin: bool | None = None,
        minimum_coverage: float | None = None,
) -> None:
    '''Raise if any explicit output-related CLI option was passed alongside --run-toml-options.

    --run-toml-options means "run exactly as specified in the configuration file", so no
    other output-related CLI flag may be explicitly passed at the same time.
    '''
    conflicts = collect_explicit_options(
        plot=plot,
        output_directory=output_directory,
        write_to_excel=write_to_excel,
        overwrite=overwrite,
        interp=interp,
        show=show,
        threshold=threshold,
        color_map=color_map,
        color_map_ticks=color_map_ticks,
        fillin=fillin,
        minimum_coverage=minimum_coverage,
    )
    if conflicts:
        raise_cli_error(
            CliErrorCode.CONFLICTING_OPTIONS,
            '--run-toml-options was passed with conflicting CLI option(s): '
            f'{", ".join(conflicts)}. Remove these options or use --override-toml-options.',
            options=list(conflicts),
        )

# Signature mirrors explicit CLI override surface.
# pylint: disable=too-many-arguments,too-many-positional-arguments
def resolve_output_options(data: dict[str, Any],
                           plot: bool | None,
                           output_directory: str | None,
                           write_to_excel: bool | None,
                           overwrite: bool | None,
                           interp: bool | None,
                           show: bool | None,
                           threshold: float | None = None,
                           color_map: str | None = None,
                           color_map_ticks: list[float] | None = None,
                           fillin: bool | None = None,
                           minimum_coverage: float | None = None) -> OutputOptions:
    '''Merge explicit CLI flags with the configuration file's [output] section.

    CLI flags default to None (not explicitly passed by the user). An explicit
    (non-None) CLI value always wins; otherwise the toml value applies (or that
    value's own default when the toml is silent too).
    '''
    toml_options = parse_output_options(data)
    climate_canvas = merge_overrides(
        toml_options.plot.climate_canvas,
        interpolate=interp, show=show, threshold=threshold,
        color_map=color_map, color_map_ticks=color_map_ticks, fillin=fillin,
    )
    plot_options = merge_overrides(
        replace(toml_options.plot, climate_canvas=climate_canvas),
        enabled=plot, minimum_coverage=minimum_coverage,
    )
    return merge_overrides(
        toml_options,
        directory=output_directory, overwrite=overwrite, excel=write_to_excel, plot=plot_options,
    )

def load_config_file(path: str) -> dict[str, Any]:
    '''Load a configuration file.'''
    with open(path, 'rb') as file:
        data = tomllib.load(file)
    return data

def load_timeseries(data: dict[str, Any]) -> Timeseries:
    '''Parse a timeseries from the configuration file.'''
    spec = parse_timeseries_spec(data)
    ext = Path(spec.path).suffix.lower()
    if ext in ('.xlsx', '.xls'):
        return Timeseries.from_excel(
            spec.path, spec.first_day_of_water_year, spec.date_format, spec.sheet_name
        )
    return Timeseries.from_csv(spec.path, spec.first_day_of_water_year, spec.date_format)

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
    '''Parse the optional [output.metric] section from the configuration file.

    Absent [output] or [output.metric] section -> MetricOptions() (default mode: portion).
    '''
    output_section = data.get('output', {})
    return parse_metric_options(output_section.get('metric'), section_name='output.metric')

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
