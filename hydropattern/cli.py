'''Entry point for the hydropattern command line interface.'''

import tomllib
from typing import Any

import typer

app = typer.Typer(no_args_is_help=True)

@app.callback()
def callback():
    '''hydropattern command line interface.'''

@app.command()
def run(path: str = typer.Argument(..., help='''
                                   Path to *.toml configuration file.
                                   ''')):
    '''Run the hydropattern command line interface.'''
    typer.echo('Hello, world!')
    data = load_config(path)
    typer.echo(data)

def load_config(path: str) -> dict[str, Any]:
    '''Load a configuration file.'''
    with open(path, 'rb') as file:
        data = tomllib.load(file)
    return data

def load_timeseries(data: dict[str, Any]):
    '''Load a timeseries.'''
