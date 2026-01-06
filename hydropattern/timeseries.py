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
from pathlib import Path
from calendar import month_abbr
from dataclasses import dataclass
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib import gridspec

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
        dowy = self.date_to_day_of_water_year(date)
        # dowy is in CY that WY started in, last dowy is in next CY.
        if dowy < (365 - self.first_day_of_water_year):
            # TODO: removed -1 here, check this is correct, remove IF if so.
            return self.day_of_water_year_to_date(dowy=365, year=date.year)
        # dowy is in last CY of WY.
        return self.day_of_water_year_to_date(dowy=365, year=date.year)

    @staticmethod
    def _fillnan(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        '''
        Expands data to specfied start and end time window, filling missing values with NaN.
        '''
        datetime_index = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.DatetimeIndex(df.index) # pylint: disable=line-too-long
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
        # compute data range.
        dt = self.data.index[1] - self.data.index[0]
        min_plot_date = self._min_plot_date(self.data.index.min())
        max_plot_date = self._max_plot_date(self.data.index.max())
        if data_columns is None:
            data_columns = [0]

        # x-axis divisions
        if yrs_per_row is None:
            nrows = 1
        else:
            # divisions by number of years per row, fill out empy periods.
            nrows = int(np.ceil((max_plot_date.year - min_plot_date.year) / yrs_per_row))
            max_plot_date = min_plot_date + relativedelta(years=yrs_per_row * nrows) - dt

        # all data
        _dfs = [self.data]
        if comparision_series is not None:
            _dfs.append(comparision_series.to_frame())

        # y-axis break by order of magnitude divisions.
        if broken_axis:
            if broken_axis_ranges is None:
                divisions = self._order_of_magnitude_divisions(_dfs)[::-1]
            else:
                divisions = self._parse_divisions(broken_axis_ranges)[::-1]
        else:
            divisions = [self._global_min_max(_dfs)]

        # plot
        # https://stackoverflow.com/questions/34933905/adding-subplots-to-a-subplot
        fig = plt.figure(figsize=(15, 5 * nrows))
        # fill data gaps with nans
        dfs = [self._fillnan(df, min_plot_date, max_plot_date) for df in _dfs]
        outer = gridspec.GridSpec(nrows=nrows, ncols=1, wspace=0.5, hspace=0.1)
        min_row_date = min_plot_date
        if yrs_per_row is None:
            max_row_date = max_plot_date
        else:
            max_row_date = min_plot_date + relativedelta(years=yrs_per_row) - dt
        for i in range(nrows):
            dfs_period = [df.loc[min_row_date:max_row_date] for df in dfs]
            if broken_axis:
                # Use broken axis layout with multiple divisions
                inner = gridspec.GridSpecFromSubplotSpec(nrows=len(divisions), ncols=1,
                                                         subplot_spec=outer[i],
                                                         wspace=0.1, hspace=0.1)
                axs = [fig.add_subplot(inner[j]) for j in range(len(divisions))]
            else:
                # Use single continuous axis
                axs = [fig.add_subplot(outer[i])]
            # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
            for j, ax in enumerate(axs):
                d = 0.5
                break_kwargs: dict = dict(
                    marker=[(-1, -d), (1, d)], markersize=5,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)

                for col in data_columns:
                    if isinstance(col, str):
                        idx = self.data.columns.get_loc(col)
                        col_name = col
                    else:
                        idx = col
                        col_name = self.data.columns[idx]
                    # Use column-based indexing instead of iloc to help mypy
                    series = dfs_period[0][col_name]
                    ax.plot(dfs_period[0].index, series, label=col_name)
                if comparision_series is not None:
                    # extra = self._fillnan(extra_timeseries.to_df(), min_row_date, max_row_date
                    #                       ).loc[min_row_date:max_row_date]
                    # ax.plot(extra.index, extra.iloc[:,0], label=extra_timeseries.name)
                    ax.plot(dfs_period[1].index, dfs_period[1].iloc[:,0],
                            label=comparision_series.name)
                ax.set_xlim(mdates.date2num(min_row_date), mdates.date2num(max_row_date))

                if broken_axis:
                    # Set y-limits for broken axis divisions
                    ax.set_ylim(divisions[j][0], divisions[j][1])
                else:
                    # Set y-limits for continuous axis using the min/max values
                    ax.set_ylim(divisions[0][0], divisions[0][1])

                if broken_axis and j == 0: # top division of broken axis
                    ax.spines['bottom'].set_visible(False)
                    ax.xaxis.tick_top()
                    ax.xaxis.set_minor_locator(mdates.YearLocator())
                    ax.tick_params(labeltop=False)
                    ax.plot([0, 1], [0, 0], transform=ax.transAxes,
                            **break_kwargs)  # type: ignore[misc]
                    if i == 0:
                        ax.legend(frameon=False)
                elif broken_axis and j == len(divisions) - 1: # bottom division of broken axis
                    ax.spines['top'].set_visible(False)
                    ax.xaxis.set_major_locator(mdates.YearLocator())
                    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[3, 5, 7, 9, 11]))
                    ax.xaxis.tick_bottom()
                    ax.tick_params(axis='x', which='both', top=False, labeltop=False)
                    ax.plot([0, 1], [1, 1], transform=ax.transAxes,
                            **break_kwargs)  # type: ignore[misc]
                elif broken_axis: # middle division of broken axis
                    ax.spines['top'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.tick_params(axis='x', which='both',
                                   bottom=False, labelbottom=False, top=False, labeltop=False)
                    ax.plot([0, 0], [0, 1], transform=ax.transAxes,
                            **break_kwargs)  # type: ignore[misc]
                    ax.plot([1, 1], [1, 0], transform=ax.transAxes,
                            **break_kwargs)  # type: ignore[misc]
                else:
                    # Normal continuous axis - no special formatting needed
                    ax.xaxis.set_major_locator(mdates.YearLocator())
                    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
                    if i == 0:
                        ax.legend(frameon=False)

            if i + 1 < nrows:
                min_row_date = max_row_date + dt
                max_row_date = min_row_date + relativedelta(years=yrs_per_row or 1) - dt
        if output_path:
            plt.savefig(output_path)
        else:
            output_path = self.file_path.replace('.csv', '.png') if self.file_path else 'output.png'
            plt.savefig(output_path)
        plt.show()

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
            ax.fill_between(df.index, df[low], df[high], alpha=0.3, label=f'{low[0]} {low[1][-2:]}-{high[1][-2:]}th percentile') # type: ignore[misc] # pylint: disable=line-too-long
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
#todo: quantile hydrograph support (port from functional flows)
