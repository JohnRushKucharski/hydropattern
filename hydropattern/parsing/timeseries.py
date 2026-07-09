'''Timeseries section parsing seam extracted from hydropattern.parsers.'''

from importlib import import_module
from typing import Any

from hydropattern.errors import ParserErrorCode, raise_parser_error


def parse_timeseries_spec(data: dict[str, Any]) -> Any:
    '''Parse required top-level [timeseries] section into a TimeseriesSpec.'''
    if 'timeseries' not in data:
        raise_parser_error(
            ParserErrorCode.MISSING_SECTION,
            'No timeseries data in configuration file.',
            section='timeseries',
        )
    section = data['timeseries']
    if 'path' not in section:
        raise_parser_error(
            ParserErrorCode.MISSING_FIELD,
            'No path in timeseries data.',
            section='timeseries',
            field='path',
        )
    timeseries_spec_cls = getattr(import_module('hydropattern.parsers'), 'TimeseriesSpec')
    return timeseries_spec_cls(
        path=section['path'],
        first_day_of_water_year=section.get('first_day_of_water_year', 1),
        date_format=section.get('date_format', ''),
        sheet_name=section.get('sheet_name', 0),
    )


__all__ = ['parse_timeseries_spec']
