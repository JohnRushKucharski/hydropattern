# hydropattern
Finds natural flow regimes type patterns in time series data.

## Background
Natural flow regimes are widely used in water resources management. Learn more about natural flow regimes:
> Poff, N. L., Allan, J. D., Bain, M. B., Karr, J. R., Prestegaard, K. L., Richter, B. D., Sparks, R. E., & Stromberg, J. C. (1997). The Natural Flow Regime. BioScience, 47(11), 769–784. https://doi.org/10.2307/1313099

The repository tends to use functional flows terminology. Functional flows are natural flow regimes linked to specific environmental processes. Learn more about functional flows:
> Yarnell, S. M., Stein, E. D., Webb, J. A., Grantham, T., Lusardi, R. A., Zimmerman, J., Peek, R. A., Lane, B. A., Howard, J., & Sandoval-Solis, S. (2020). A functional flows approach to selecting ecologically relevant flow metrics for environmental flow applications. River Research and Applications, 36(2), 318-324. https://doi.org/10.1002/rra.3575

> Note: Figure 2 and Table 2 are particularly helpful for understanding the natural flow regimes this program tracks.

Natural flow regimes can be adapted to classify hydrologic regimes in non-riverine environments, like static water levels in lakes. They can be used to evaluate the alteration of natural hydrologic patterns. This program imagines their usage in climate impact studies.

## Basic Terminology
To define a natural flow regime the following hierarchical labels must be defined:

**Component:** Natural flow regimes consist of one or more *components*.

**Characteristic:** Each component consists of one or more of the following *characteristics*.

- Timing: when the hydrologic pattern occurs (i.e., wet season).
- Magnitude: the size hydrologic pattern (i.e., flow, stage, etc.).
- Duration: how long the hydrologic pattern persists (i.e., 7 days).
- Frequency: how often the pattern occurs (i.e. in 1 out of every 5 years).
- Rate of Change: change in the size of the hydrologic pattern (i.e., doubling of the previous day's flow).

**Metric:** A metric defines the truth value for each characteristic. For example, the magnitude of flow > 100.

Examples are provided below.

## Getting Started
The program can be used as either: (a) python package, imported from the project GitHub repository: https://github.com/JohnRushKucharski/hydropattern or the PiPl python package index. (b) a command line application.

### Installation Instructions

hydropattern requires python 3.12+. It aims to be multi-platform and has been run on Windows 11 and MacOS 14 and 15.

#### Clone or Fork climate-canvas from GitHub
The hydropattern source code can be found here: https://github.com/JohnRushKucharski/hydropattern is available under the GNU Version 3 General Public License.

It can be cloned or forked by following the normal cloning or forking instructions, which are available here: https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository and here: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo.


#### Installation with Poetry

hydropattern is developed with Poetry, which can be used to simply the installation process.

To install Poetry, follow the instructions here: https://python-poetry.org/docs/.

Once Poetry is installed, use your favorite shell to go to the location of the local hydropattern repository, e.g.

``
cd <PATH_TO_LOCAL>\hydropattern
``

Next run:

``
poetry install
``

This will create a python virtual environment containing all the required hydropattern dependencies, without affecting your system's global python environment.

The hydropattern program should now be ready for use as either a python package or command line utility. To test the command line interface (CLI) type the following command into your shell:

``
poetry run python hydropattern
``

This should return help instructions for the hydropattern CLI.

### Inputs
The program requires two primary inputs:

1. A .toml configuration file. This file must contain the following sections:

    a. **[timeseries]**: in this section the *path* variable provides the location of the .csv timeseries input file, described below. The optional *date_format* variable is used to provide the timeseries datetime format code, see: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior. By default pandas will, with a warning message and possible error, attempt to guess format of this string, if not date format is provided. The optional *first_day_of_water_year* is used to distinguish between water and calendar years, see: https://en.wikipedia.org/wiki/Water_year. By default, the water and calendar year are assumed be the same (i.e., first_day_of_water_year = 1). The optional *columns* parameter tells the program which columns in the timeseries .csv contain hydrologic data. By default only the first time, and first column are read (i.e., columns = [1]). 

    b. **[components]**: in this section components, characteristics, and metrics are provided.
    
The toml configuration file follows basic toml file syntax (see: https://toml.io/en/). A minimal example can be found in the project GitHub repository at .\examples\minimal.toml. A more complete example file with extensive instructions and comments can be found at .\examples\detailed.toml.

2. One or more hydrologic time series provided in a .csv file. This file must have the following format:

time    | column_0      | column_1  | ... | column_n-1  | column_n      |
---     | ---           | ---       | --- | ---         | ---           | 
t_0     | value_0,0     | value_1,0 | ... | value_n-1,0 | value_n,0     |
t_1     | value_0,1     | ...       | ... | ...         | value_n,1     |
...     | ...           | ...       | ... | ...         | ...           |         
t_m-1   | value_0,m-1   | ...       | ... | ...         | value_n,m-1   |
t_m     | value_0,m     | value_1,m | ... | value_n-1,m | value_n,m     |

where the 'time' column contains a datetimestring that can be parsed as a pandas datetime index. By default pandas will, with a warning message and possible error, attempt to guess format of this string. However, the format of this string can be specified in the toml file, described above. Example time series are provided in the .\examples directory on the project's GiHub repository.

## CLI Basic Usage
The program can be run on a timeseries of input data, given a valid .toml file configuration (see inputs section above). Using the **run** command. For example, using poetry in a terminal the run command has the following syntax:

``
poetry python hydropattern run "path_to_toml_file"
``

where "path_to_toml_file" is replaced with a valid path to the input .toml file. The following **optional arguments**, can be appended to the end of the run command above:

``
--output-dir "path_to_output_csv_or_xlsx_files"
``

> Output file are placed in the same directory as the timeseries specified in the .toml file by default. Providing a valid path to this optinal argument will store the outputs in a different location.

``
--plot
``

> This plots a response surface of with the output data.

``
--excel
``

> This writes the outputs to a single excel file. Otherwise each timeseries in the input timeseries is written to a seperate .csv file.
