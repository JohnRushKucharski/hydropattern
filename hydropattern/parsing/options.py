'''Option parsing seam extracted from hydropattern.parsers.'''

from dataclasses import replace
from typing import Any

from hydropattern.errors import ParserErrorCode, raise_parser_error
from hydropattern.parsing.specs import (
    ClimateCanvasPlotOptions,
    MetricMode,
    MetricOptions,
    OutputOptions,
    PlotOptions,
)


def parse_metric_options(section: Any = None, section_name: str = 'metric') -> MetricOptions:
    '''Parse an already-extracted metric options section into MetricOptions.'''
    valid_metric_modes: frozenset[str] = frozenset(m.value for m in MetricMode)
    if section is None:
        return MetricOptions()
    if not isinstance(section, dict):
        raise_parser_error(
            ParserErrorCode.INVALID_TYPE,
            f'[{section_name}] section must be a table, got: {section!r}.',
            section=section_name,
        )

    mode = MetricMode.PORTION
    for key, value in section.items():
        match key:
            case 'mode':
                if not isinstance(value, str) or value not in valid_metric_modes:
                    raise_parser_error(
                        ParserErrorCode.INVALID_VALUE,
                        f'{section_name}.mode must be one of {sorted(valid_metric_modes)}, '
                        f'got: {value!r}.',
                        section=section_name,
                        field='mode',
                        value=value,
                    )
                mode = MetricMode(value)
            case _:
                raise_parser_error(
                    ParserErrorCode.UNKNOWN_OPTION,
                    f'Unrecognized [{section_name}] option: {key!r}.',
                    section=section_name,
                    field=key,
                )
    return MetricOptions(mode=mode)


def _require_type(value: Any, expected: type | tuple[type, ...], section: str, field_name: str,
                  ) -> None:
    '''Raise PARSER_INVALID_TYPE if value is not an instance of expected.'''
    if not isinstance(value, expected):
        raise_parser_error(
            ParserErrorCode.INVALID_TYPE,
            f'{section}.{field_name} must be {expected}, got: {value!r}.',
            section=section,
            field=field_name,
            value=value,
        )


def parse_climate_canvas_plot_options(section: Any = None) -> ClimateCanvasPlotOptions:
    '''Parse optional [output.plot.climate-canvas] section into ClimateCanvasPlotOptions.'''
    name = 'output.plot.climate-canvas'
    if section is None:
        return ClimateCanvasPlotOptions()
    if not isinstance(section, dict):
        raise_parser_error(
            ParserErrorCode.INVALID_TYPE,
            f'[{name}] section must be a table, got: {section!r}.',
            section=name,
        )

    opts = ClimateCanvasPlotOptions()
    for key, value in section.items():
        match key:
            case 'interpolate':
                _require_type(value, bool, name, 'interpolate')
                opts = replace(opts, interpolate=value)
            case 'show':
                _require_type(value, bool, name, 'show')
                opts = replace(opts, show=value)
            case 'title':
                _require_type(value, str, name, 'title')
                opts = replace(opts, title=value)
            case 'xlabel':
                _require_type(value, str, name, 'xlabel')
                opts = replace(opts, xlabel=value)
            case 'ylabel':
                _require_type(value, str, name, 'ylabel')
                opts = replace(opts, ylabel=value)
            case 'zlabel':
                _require_type(value, str, name, 'zlabel')
                opts = replace(opts, zlabel=value)
            case 'threshold':
                _require_type(value, (int, float), name, 'threshold')
                opts = replace(opts, threshold=float(value))
            case 'color_map':
                _require_type(value, str, name, 'color_map')
                opts = replace(opts, color_map=value)
            case 'color_map_ticks':
                _require_type(value, list, name, 'color_map_ticks')
                for tick in value:
                    _require_type(tick, (int, float), name, 'color_map_ticks')
                opts = replace(opts, color_map_ticks=[float(t) for t in value])
            case 'fillin':
                _require_type(value, bool, name, 'fillin')
                opts = replace(opts, fillin=value)
            case _:
                raise_parser_error(
                    ParserErrorCode.UNKNOWN_OPTION,
                    f'Unrecognized [{name}] option: {key!r}.',
                    section=name,
                    field=key,
                )
    return opts


def parse_plot_options(section: Any = None) -> PlotOptions:
    '''Parse optional [output.plot] section into PlotOptions.'''
    name = 'output.plot'
    if section is None:
        return PlotOptions()
    if not isinstance(section, dict):
        raise_parser_error(
            ParserErrorCode.INVALID_TYPE,
            f'[{name}] section must be a table, got: {section!r}.',
            section=name,
        )

    enabled = False
    climate_canvas = ClimateCanvasPlotOptions()
    for key, value in section.items():
        match key:
            case 'enabled':
                _require_type(value, bool, name, 'enabled')
                enabled = value
            case 'climate-canvas':
                climate_canvas = parse_climate_canvas_plot_options(value)
            case _:
                raise_parser_error(
                    ParserErrorCode.UNKNOWN_OPTION,
                    f'Unrecognized [{name}] option: {key!r}.',
                    section=name,
                    field=key,
                )
    return PlotOptions(enabled=enabled, climate_canvas=climate_canvas)


def parse_output_options(data: dict[str, Any]) -> OutputOptions:
    '''Parse optional top-level [output] section into OutputOptions.'''
    name = 'output'
    if name not in data:
        return OutputOptions()
    section = data[name]
    if not isinstance(section, dict):
        raise_parser_error(
            ParserErrorCode.INVALID_TYPE,
            f'[{name}] section must be a table, got: {section!r}.',
            section=name,
        )

    directory: str | None = None
    overwrite = True
    excel = True
    metric = MetricOptions()
    plot = PlotOptions()
    for key, value in section.items():
        match key:
            case 'directory':
                _require_type(value, str, name, 'directory')
                directory = value
            case 'overwrite':
                _require_type(value, bool, name, 'overwrite')
                overwrite = value
            case 'excel':
                _require_type(value, bool, name, 'excel')
                excel = value
            case 'metric':
                metric = parse_metric_options(value, section_name='output.metric')
            case 'plot':
                plot = parse_plot_options(value)
            case _:
                raise_parser_error(
                    ParserErrorCode.UNKNOWN_OPTION,
                    f'Unrecognized [{name}] option: {key!r}.',
                    section=name,
                    field=key,
                )
    return OutputOptions(
        directory=directory, overwrite=overwrite, excel=excel, metric=metric, plot=plot,
    )

__all__ = [
    'parse_climate_canvas_plot_options',
    'parse_metric_options',
    'parse_output_options',
    'parse_plot_options',
]
