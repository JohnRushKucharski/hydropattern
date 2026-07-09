'''
Input data structure for analysis.

Expects:
    - *.csv file
    - first column: 'time'
    - parse_dates = True will successfully parse 'time' column.
    - second to N column: values for each time series.

Example:
    time,value
    1900-01-01,11
    1900-01-02,13
    ...
'''
from calendar import month_abbr
from dataclasses import dataclass
import datetime as datetime_mod
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class _PlotContext:
    '''Resolved plotting context for Timeseries.plot_timeseries rendering.'''
    dt: pd.Timedelta
    min_plot_date: pd.Timestamp
    max_plot_date: pd.Timestamp
    nrows: int
    yrs_per_row: int | None
    broken_axis: bool
    column_names: list[str]
    dfs: list[pd.DataFrame]
    divisions: list[tuple[float, float]]
    comparison_series: pd.Series | None


@dataclass(frozen=True)
class _AxisStyleSpec:
    '''Per-axis styling inputs for broken/continuous axis rendering.'''
    row_index: int
    axis_index: int
    broken_axis: bool
    divisions: list[tuple[float, float]]
    break_kwargs: dict[str, object]


@dataclass(frozen=True)
class _PlotRequest:
    '''Raw request options for Timeseries.plot_timeseries.'''
    data_columns: list[int] | list[str] | None
    comparison_series: pd.Series | None
    yrs_per_row: int | None
    broken_axis: bool
    broken_axis_ranges: list[float] | None


def first_day_of_water_year(day: int, month: int, yr: int = 1900) -> int:
    '''
    Returns the day of the year that is the first day of the water year.

    Note:
        Feb 29 recoded as Feb 28 for non-leap years.
        Days after Feb 28 in leap years recorded as previous day.
        Default year 1900 was not a leap year.
    '''
    if month == 2 and day == 29:
        day = 28
    date = pd.Timestamp(f'{yr}-{month}-{day}')
    return date.dayofyear -1 if date.is_leap_year and date.dayofyear > 59 else date.dayofyear

def to_day_of_water_year(date: pd.Timestamp, first_day_of_wy: int = 1):
    '''
    Returns the day of the water year for the date.

    Note: water year only has 365 days, even in leap years.
    '''
    if first_day_of_wy < 1 or first_day_of_wy > 365:
        raise ValueError('first_day_of_water_year must be between 1 and 365.')
    start, end = first_day_of_wy, 365
    # if leap year, subtract 1 for dates after Feb 28
    doy = date.dayofyear - 1 if date.is_leap_year and date.dayofyear > 59 else date.dayofyear
    return doy + (end - start) + 1 if doy < start else doy - (start - 1)

def to_doy_from_dowy(dowy: int, first_day_of_wy: int = 1, yr: int = 1900) -> int:
    '''
    Converts day of water year to day of year.

    Args:
         dowy (int): day of water year (WY).
        first_day_of_wy (int): first day of water year (1-365).
        yr (int): year. Default 1900.
    '''
    if first_day_of_wy < 1 or first_day_of_wy > 365:
        raise ValueError('first_day_of_water_year must be between 1 and 365.')
    days_to_new_year = 365 - first_day_of_wy + 1
    # dowy is after start of new CY, so substract WY days in previous CY.
    if dowy > days_to_new_year:
        doy = dowy - days_to_new_year
    # dowy is before start of new CY, so add CY days before the WY started.
    else:
        doy = dowy + first_day_of_wy - 1
    # check for leap year effects
    if pd.to_datetime(f'{yr}-01-01').is_leap_year and doy > 59:
        # remove leap year effect if after Feb 28.
        doy -= 1
    return doy

@dataclass
class Timeseries:
    '''Class for holding time series of hydrology data.'''
    data: pd.DataFrame
    file_path: None|str = None
    first_day_of_water_year: int = 1

    def __post_init__(self):
        self.data = self.data.sort_index()
        self.validate_dataframe(self.data)
        self.data['dowy'] = self.data.index.map(
            lambda x: to_day_of_water_year(x, self.first_day_of_water_year))
        if self.first_day_of_water_year < 1 or self.first_day_of_water_year > 365:
            raise ValueError('first_day_of_water_year must be between 1 and 365.')
        if self.file_path:
            # When using .from_csv this can sneak in as a Path.
            # This causes issues later, for ex: when saving plots.
            if isinstance(self.file_path, Path):
                self.file_path = str(self.file_path)
            # If a file path is provided, check that it exists.
            if not Path(self.file_path).exists():
                raise ValueError('File path does not exist.')

    @staticmethod
    def validate_dataframe(data: pd.DataFrame) -> None:
        '''
        Validates the data frame.

        Expects:
            - columns: ['time', ...]
            - 'time' is datetime index.
            - second to N column: values for each time series.
        '''
        if data.index.name != 'time':
            raise ValueError('Data frame must have a time index.')
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError('Data frame must have a datetime index.')
        if len(data.columns) < 0:
            raise ValueError('Data frame must have at least one column.')

    @staticmethod
    def from_dataframe(data: pd.DataFrame,
                       first_dowy: int = 1, path: str|None = None) -> 'Timeseries':
        '''
        Returns a Timeseries object from a pandas DataFrame.

        Expects:
            - columns: ['time', ...]
            - time column is datetime index.
            - second to N column: values for each time series.
        '''
        return Timeseries(file_path=path, data=data,
                          first_day_of_water_year=first_dowy)

    @staticmethod
    def from_csv(path: str, first_dowy: int = 1,  date_format: str = '') -> 'Timeseries':
        '''
        Returns a Timeseries object, with file_path.

        Expects:
            - *.csv file
            - columns: ['time', ...]
            - parse_dates = True will successfully parse 'time' column.
        '''
        if date_format:
            df = pd.read_csv(path, header=0, index_col=0, parse_dates=[0],
                             date_format=date_format,
                             ).rename_axis('time', axis=0)
            pd.to_datetime(df.index, format=date_format, errors='raise')
        else:
            df = pd.read_csv(path, header=0, index_col=0, parse_dates=[0],
                             ).rename_axis('time', axis=0)
        return Timeseries(file_path=path, data=df.apply(pd.to_numeric, errors='raise').sort_index(),
                          first_day_of_water_year=first_dowy)

    @staticmethod
    def from_excel(path: str, first_dowy: int = 1, date_format: str = '',
                   sheet_name: int | str = 0) -> 'Timeseries':
        '''
        Returns a Timeseries object from an Excel file.

        Expects:
            - *.xlsx or *.xls file
            - first column: datetime values (index)
            - second to N columns: numeric time series values

        Note:
            Dates are normalised to datetime64[s] resolution so that years
            beyond 2262 (which overflow nanosecond precision) are supported.
            date_format is used when index cells are strings rather than
            native Excel datetime values.
        '''
        df = pd.read_excel(path, header=0, index_col=0,
                           sheet_name=sheet_name).rename_axis('time', axis=0)
        # Index cells may be Python datetime objects (native Excel dates) or
        # strings. Parse strings with date_format when provided.
        if df.index.dtype == object and not isinstance(df.index[0], datetime_mod.datetime):
            fmt = date_format or None
            idx_parsed = pd.to_datetime(df.index, format=fmt)
        else:
            # datetime objects: convert via ISO strings → datetime64[D] so
            # years > 2262 (nanosecond overflow) are handled correctly.
            idx_parsed = pd.DatetimeIndex(
                np.array([x.strftime('%Y-%m-%d') for x in df.index], dtype='datetime64[D]')
            )
        df.index = idx_parsed
        df.index.name = 'time'
        return Timeseries(file_path=path, data=df.apply(pd.to_numeric, errors='raise').sort_index(),
                          first_day_of_water_year=first_dowy)

    def date_to_day_of_water_year(self, date: pd.Timestamp) -> int:
        '''
        Returns the day of the water year for the date.

        Note: water year only has 365 days, even in leap years.
        '''
        days_to_new_year = 365 - self.first_day_of_water_year
        # if leap year, subtract 1 for dates after Feb 28
        doy = date.dayofyear - 1 if date.is_leap_year and date.dayofyear > 59 else date.dayofyear
        # doy is before start of new WY, so add WY's number of days to new year (for previous CY).
        if doy < self.first_day_of_water_year:
            return doy + days_to_new_year + 1
        # doy is after start of WY (but before new CY), so subtract days in CY before WY started.
        return doy - (self.first_day_of_water_year - 1)

    def month_day_year_to_day_of_water_year(self, month: int, day: int, year: int = 1900) -> int:
        '''
        Converts month, day, year to day of water year.
        '''
        date = pd.Timestamp(f'{year}-{month}-{day}')
        return self.date_to_day_of_water_year(date)

    def day_of_water_year_to_date(self, dowy: int, year: int = 1900) -> pd.Timestamp:
        '''
        Converts day of water year to date.

        Args:
             dowy (int): day of water year (WY).
            yr (int): year. Default 1900.
        '''
        days_to_new_year = 365 - self.first_day_of_water_year + 1
        # dowy is after start of new CY, so substract WY days in previous CY.
        if dowy > days_to_new_year:
            doy = dowy - days_to_new_year
        # dowy is before start of new CY, so add CY days before the WY started.
        else:
            doy = dowy + self.first_day_of_water_year - 1
        # check for leap year effects
        if pd.to_datetime(f'{year}-01-01').is_leap_year and doy > 59:
            # remove leap year effect if after Feb 28.
            doy -= 1
        return pd.to_datetime(f'{year}-{doy}', format='%Y-%j')

    def day_of_water_year_to_day_month(self, dowy: int) -> tuple[int, int]:
        '''
        Converts day of water year to day and month.
        '''
        return (self.day_of_water_year_to_date(dowy=dowy).day,
                self.day_of_water_year_to_date(dowy=dowy).month)

    def day_of_water_year_to_day_of_year(self, dowy: int) -> int:
        '''
        Converts day of water year to day of year.
        '''
        return to_doy_from_dowy(dowy=dowy, first_day_of_wy=self.first_day_of_water_year)

    def _min_plot_date(self, date: pd.Timestamp) -> pd.Timestamp:
        '''
        Returns the minimum date for plotting.
        '''
        dowy = self.date_to_day_of_water_year(date)
        # dowy is before start of new CY.
        if dowy < (365 - self.first_day_of_water_year):
            return self.day_of_water_year_to_date(dowy=1, year=date.year)
        # dowy is after start of new CY, first dowy is in last CY.
        return self.day_of_water_year_to_date(dowy=1, year=date.year - 1)

    def _max_plot_date(self, date: pd.Timestamp) -> pd.Timestamp:
        '''
        Returns the maximum date for plotting.
        '''
        return self.day_of_water_year_to_date(dowy=365, year=date.year)

    @staticmethod
    def _fillnan(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        '''
        Expands data to specfied start and end time window, filling missing values with NaN.
        '''
        datetime_index = (
            df.index if isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex(df.index)
        )
        return df.apply(lambda x: x.reindex(pd.date_range(start, end, freq=datetime_index.freq,
                                                          name='time'), fill_value=np.nan))

    @staticmethod
    def _global_min_max(dfs: list[pd.DataFrame]) -> tuple[float, float]:
        '''
        Returns the global min and max values for plotting.
        '''
        for i, df in enumerate(dfs):
            if i == 0:
                x = df.to_numpy().reshape(-1)
                min_, max_ = np.nanmin(x), np.nanmax(x)
            else:
                x = df.to_numpy().reshape(-1)
                min_, max_ = min(min_, np.nanmin(x)), max(max_, np.nanmax(x))
        return min_, max_

    @staticmethod
    def _order_of_magnitude_divisions(dfs: list[pd.DataFrame]) -> list[tuple[float, float]]:
        '''
        Returns the order of magnitude divisions for broken axis.
        '''
        divisions = []
        # find global min and max
        min_, max_ = Timeseries._global_min_max(dfs)
        # number of orders of magnitude (oom)s
        min_oom = int(np.log10(min_)) if min_ > 1 else 0
        max_oom = int(np.log10(max_)) if max_ > 1 else 0
        bins = max_oom - min_oom + 1
        # find divisions
        local_min = min_
        for _ in range(bins):
            digits = len(str(int(local_min)))
            local_max = min(max_, float('9' * digits))
            divisions.append((local_min, local_max))
            local_min = local_max + 1
        return divisions

    def _parse_divisions(self, divisions: list[float]) -> list[tuple[float, float]]:
        '''
        Returns the order of magnitude divisions for broken axis.
        '''
        if len(divisions) % 2 != 0:
            raise ValueError('Even number of divisions required.')
        return [(divisions[i], divisions[i + 1]) for i in range(0, len(divisions), 2)]

    def _resolve_plot_columns(self, data_columns: None | list[int] | list[str]) -> list[str]:
        '''Resolve plot columns to concrete column names.'''
        selected = [0] if data_columns is None else data_columns
        column_names: list[str] = []
        for col in selected:
            if isinstance(col, str):
                column_names.append(col)
            else:
                column_names.append(self.data.columns[col])
        return column_names

    def _resolve_x_axis_layout(self, yrs_per_row: None | int
                               ) -> tuple[pd.Timedelta, pd.Timestamp, pd.Timestamp, int]:
        '''Resolve x-axis timestep, padded min/max bounds, and number of rows.'''
        dt = self.data.index[1] - self.data.index[0]
        min_plot_date = self._min_plot_date(self.data.index.min())
        max_plot_date = self._max_plot_date(self.data.index.max())
        if yrs_per_row is None:
            return dt, min_plot_date, max_plot_date, 1
        nrows = int(np.ceil((max_plot_date.year - min_plot_date.year) / yrs_per_row))
        max_plot_date = min_plot_date + relativedelta(years=yrs_per_row * nrows) - dt
        return dt, min_plot_date, max_plot_date, nrows

    def _resolve_y_axis_divisions(self,
                                  dfs: list[pd.DataFrame],
                                  broken_axis: bool,
                                  broken_axis_ranges: None | list[float]
                                  ) -> list[tuple[float, float]]:
        '''Resolve y-axis divisions for broken or continuous plotting mode.'''
        if not broken_axis:
            return [self._global_min_max(dfs)]
        if broken_axis_ranges is None:
            return self._order_of_magnitude_divisions(dfs)[::-1]
        return self._parse_divisions(broken_axis_ranges)[::-1]

    @staticmethod
    def _initial_row_bounds(min_plot_date: pd.Timestamp,
                            max_plot_date: pd.Timestamp,
                            yrs_per_row: None | int,
                            dt: pd.Timedelta) -> tuple[pd.Timestamp, pd.Timestamp]:
        '''Resolve first row min/max plotting dates.'''
        if yrs_per_row is None:
            return min_plot_date, max_plot_date
        return min_plot_date, min_plot_date + relativedelta(years=yrs_per_row) - dt

    @staticmethod
    def _next_row_bounds(max_row_date: pd.Timestamp,
                         yrs_per_row: None | int,
                         dt: pd.Timedelta) -> tuple[pd.Timestamp, pd.Timestamp]:
        '''Resolve min/max plotting dates for the next row.'''
        next_min = max_row_date + dt
        next_max = next_min + relativedelta(years=yrs_per_row or 1) - dt
        return next_min, next_max

    @staticmethod
    def _broken_axis_kwargs() -> dict[str, object]:
        '''Build marker styling kwargs used at broken-axis boundaries.'''
        d = 0.5
        return {
            'marker': [(-1, -d), (1, d)],
            'markersize': 5,
            'linestyle': 'none',
            'color': 'k',
            'mec': 'k',
            'mew': 1,
            'clip_on': False,
        }

    def _plot_row_data(self,
                       ax: plt.Axes,
                       dfs_period: list[pd.DataFrame],
                       column_names: list[str],
                       comparision_series: None | pd.Series) -> None:
        '''Plot selected primary columns and optional comparison series for one row/axis.'''
        for col_name in column_names:
            ax.plot(dfs_period[0].index, dfs_period[0][col_name], label=col_name)
        if comparision_series is not None:
            ax.plot(dfs_period[1].index, dfs_period[1].iloc[:, 0], label=comparision_series.name)

    def _style_axis(self,
                    ax: plt.Axes,
                    spec: _AxisStyleSpec) -> None:
        '''Apply axis limits and styling for broken/continuous y-axis layouts.'''
        if spec.broken_axis:
            ax.set_ylim(spec.divisions[spec.axis_index][0], spec.divisions[spec.axis_index][1])
        else:
            ax.set_ylim(spec.divisions[0][0], spec.divisions[0][1])

        if spec.broken_axis and spec.axis_index == 0:
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.tick_top()
            ax.xaxis.set_minor_locator(mdates.YearLocator())
            ax.tick_params(labeltop=False)
            ax.plot([0, 1], [0, 0], transform=ax.transAxes,
                    **spec.break_kwargs)  # type: ignore[arg-type]
            if spec.row_index == 0:
                ax.legend(frameon=False)
        elif spec.broken_axis and spec.axis_index == len(spec.divisions) - 1:
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[3, 5, 7, 9, 11]))
            ax.xaxis.tick_bottom()
            ax.tick_params(axis='x', which='both', top=False, labeltop=False)
            ax.plot([0, 1], [1, 1], transform=ax.transAxes,
                    **spec.break_kwargs)  # type: ignore[arg-type]
        elif spec.broken_axis:
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both',
                           bottom=False, labelbottom=False, top=False, labeltop=False)
            ax.plot([0, 0], [0, 1], transform=ax.transAxes,
                    **spec.break_kwargs)  # type: ignore[arg-type]
            ax.plot([1, 1], [1, 0], transform=ax.transAxes,
                    **spec.break_kwargs)  # type: ignore[arg-type]
        else:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
            if spec.row_index == 0:
                ax.legend(frameon=False)

    def _build_plot_context(self, request: _PlotRequest) -> _PlotContext:
        '''Resolve and package all plotting inputs for plot_timeseries rendering.'''
        dt, min_plot_date, max_plot_date, nrows = self._resolve_x_axis_layout(request.yrs_per_row)
        column_names = self._resolve_plot_columns(request.data_columns)
        all_series = [self.data]
        if request.comparison_series is not None:
            all_series.append(request.comparison_series.to_frame())
        divisions = self._resolve_y_axis_divisions(
            all_series, request.broken_axis, request.broken_axis_ranges
        )
        dfs = [self._fillnan(df, min_plot_date, max_plot_date) for df in all_series]
        return _PlotContext(
            dt=dt,
            min_plot_date=min_plot_date,
            max_plot_date=max_plot_date,
            nrows=nrows,
            yrs_per_row=request.yrs_per_row,
            broken_axis=request.broken_axis,
            column_names=column_names,
            dfs=dfs,
            divisions=divisions,
            comparison_series=request.comparison_series,
        )

    def _render_plot_context(self, context: _PlotContext) -> None:
        '''Render a timeseries plot from a precomputed plotting context.'''
        fig = plt.figure(figsize=(15, 5 * context.nrows))
        outer = gridspec.GridSpec(nrows=context.nrows, ncols=1, wspace=0.5, hspace=0.1)
        min_row_date, max_row_date = self._initial_row_bounds(
            context.min_plot_date, context.max_plot_date, context.yrs_per_row, context.dt
        )
        for row_index in range(context.nrows):
            dfs_period = [df.loc[min_row_date:max_row_date] for df in context.dfs]
            if context.broken_axis:
                inner = gridspec.GridSpecFromSubplotSpec(
                    nrows=len(context.divisions), ncols=1, subplot_spec=outer[row_index],
                    wspace=0.1, hspace=0.1,
                )
                axes = [fig.add_subplot(inner[j]) for j in range(len(context.divisions))]
            else:
                axes = [fig.add_subplot(outer[row_index])]
            break_kwargs = self._broken_axis_kwargs()
            for axis_index, ax in enumerate(axes):
                self._plot_row_data(ax, dfs_period, context.column_names, context.comparison_series)
                ax.set_xlim(mdates.date2num(min_row_date), mdates.date2num(max_row_date))
                style = _AxisStyleSpec(
                    row_index=row_index,
                    axis_index=axis_index,
                    broken_axis=context.broken_axis,
                    divisions=context.divisions,
                    break_kwargs=break_kwargs,
                )
                self._style_axis(ax, style)
            if row_index + 1 < context.nrows:
                min_row_date, max_row_date = self._next_row_bounds(
                    max_row_date, context.yrs_per_row, context.dt
                )

    # Public plotting API: argument count reflects exposed user options.
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def plot_timeseries(self,
                        data_columns: None|list[int]|list[str] = None,
                        output_path: None|str = None,
                        comparision_series: None|pd.Series = None,
                        yrs_per_row: None|int = None,
                        broken_axis: bool = True,
                        broken_axis_ranges: None|list[float] = None) -> None:
        '''
        Plots the data, if specified:
        breaks x-axis into subplot rows based on number of years per row, and
        breaks y-axis based on broken axis data ranges. Saves plot to output path.

        Args:
            data_columns (list[str]): column names to plot.
                Default None plots the first column (i.e. [0]).
            output_path (str): path to save plot.
                Default None saves plot in same name and directory as timeseries file path.
            comparison_series (pd.Series): additional time series to plot.
                Default None.
            yrs_per_row (int): The plot can be broken along the x-axis
                into a number of equal length rows. Use this option to select
                the number of years to plot in each row of the plot.
                Default None does not break x-axis.
            broken_axis (bool): whether to break the y axis.
                ex: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
                Default True.
            broken_axis_ranges (list[float]): even number list of float values that
                specify min, max value for division of values on broken axis.
                Default None breaks y-axis with order of magnitude divisions.

        Returns:
            None
            Places plot in output_path if specified.
        '''
        request = _PlotRequest(
            data_columns=data_columns,
            comparison_series=comparision_series,
            yrs_per_row=yrs_per_row,
            broken_axis=broken_axis,
            broken_axis_ranges=broken_axis_ranges,
        )
        context = self._build_plot_context(request)
        self._render_plot_context(context)
        if output_path:
            plt.savefig(output_path)
        else:
            output_path = self.file_path.replace('.csv', '.png') if self.file_path else 'output.png'
            plt.savefig(output_path)
        # Avoid warnings in non-interactive environments (e.g., test runners using Agg backend).
        if 'agg' not in plt.get_backend().lower():
            plt.show()

    # Public plotting API: argument count/local values reflect exposed user options.
    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def plot_hydrograph_quantiles(self, col: int|str = 0,
                                  rolling_periods: int = 1, min_periods: int = 1,
                                  quantiles: None|list[float] = None,
                                  output_path: None|str = None) -> None:
        '''
        Plots the hydrograph quantiles for a specified column.

        Args:
            col (int|str): column index or name to plot.
                Default 1 (second column).
            rolling_periods (int): number of periods for rolling mean.
                Default 1 (no rolling mean).
            min_periods (int): minimum number of periods for rolling mean.
                Default 1.
            quantiles (list[float]): list of quantiles to plot.
                Default [0.25, 0.50, 0.75].
            output_path (str): path to save plot.
                Default None saves plot in same name and directory as timeseries file path,
                with '_quantiles.png' suffix.

        Returns:
            None
            Places plot in output_path if specified.
        '''
        def q(quantile: float):
            '''
            Builds quantile closure functions for pandas groupby.agg.
            '''
            def closure(x):
                return x.quantile(quantile)
            closure.__name__ = f'q{int(quantile*100)}'
            return closure
        if quantiles is None:
            quantiles = [0.25, 0.50, 0.75]
        quantiles = sorted(quantiles)
        col = col if isinstance(col, str) else self.data.columns[col]
        roll = self.data[col].rolling(rolling_periods, min_periods=min_periods, center=True).mean()
        df = pd.DataFrame({
            'dowy': self.data.dowy.values,
            col: roll.values
        }).groupby('dowy').agg({col: [q(qt) for qt in quantiles]})

        n = len(quantiles)
        pairs, is_odd = n // 2, n % 2 == 1
        _, ax = plt.subplots(figsize=(15, 7))
        for i in range(pairs):
            low = df.columns[i]
            high = df.columns[-i-1]
            label = f'{low[0]} {low[1][-2:]}-{high[1][-2:]}th percentile'
            ax.fill_between(  # type: ignore[misc]
                df.index, df[low], df[high], alpha=0.3, label=label
            )
        if is_odd:
            median = df[df.columns[pairs]]
            ax.plot(df.index, median, color='k', label='Median')

        def date_format(day_month: tuple[int, int]) -> str:
            return f'{day_month[0]:02d}-{month_abbr[day_month[1]]}'
        formatter = FuncFormatter(
            lambda x, pos: date_format(self.day_of_water_year_to_day_month(int(x))))
        ax.xaxis.set_major_formatter(formatter)

        ax.legend(frameon=False)
        if output_path:
            plt.savefig(output_path)
        else:
            output_path = self.file_path.replace(
                '.csv', '_quantiles.png') if self.file_path else 'output_quantiles.png'
            plt.savefig(output_path)
        plt.show()
# Quantile hydrograph support pending port from functional flows.
