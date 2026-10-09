# Installation

## Install uv and Python

hydropattern requires **Python 3.12 or newer**. uv manages Python environments
and installs command-line tools without changing your system packages.
Follow the [official uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
for your operating system. Open a new terminal after installation.

On **Windows**, open PowerShell from the Start menu. On **macOS**, open Terminal.
On **Linux**, open your terminal application. In any of these terminals, check
uv and install a suitable Python version:

```console
uv --version
uv python install 3.12
```

Source installation also requires [Git](https://git-scm.com/downloads).
Install Git for your operating system, reopen the terminal, and check it:

```console
git --version
```

## Install the command-line application

Install the CLI on Windows, macOS, or Linux:

```console
uv tool install --python 3.12 hydropattern==0.3.1
uv tool update-shell
```

uv installs the application and its dependencies in an isolated environment.
If instructed by uv, restart your terminal so the command is on your PATH.
Confirm the application responds:

```console
hydropattern --help
```

You do not need to clone the repository or install development or test
dependencies. Continue with the [first evaluation](first-run.md).

## Alternative: pip in a virtual environment

Use this option if you already manage Python yourself. Check that your Python
is at least 3.12. A virtual environment keeps dependencies separate.

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install hydropattern==0.3.1
.\.venv\Scripts\hydropattern.exe --help
```

On macOS or Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install hydropattern==0.3.1
.venv/bin/hydropattern --help
```

You can use the full path to the environment's command without activating the
environment. For the first evaluation, either keep the example files in this
working folder or activate the environment before changing folders.

## Source checkout

For development or to run directly from a clone, open a terminal in the
repository root:

```console
uv sync --no-default-groups
uv run --no-default-groups hydropattern --help
```

This installs the application, not the development or test groups. When you
change to the example folder, provide the repository location with `--project`.
For example, on Windows:

```powershell
cd examples\first-run
uv run --no-default-groups --project ..\.. hydropattern run first-run.toml --no-excel
```

On macOS or Linux:

```sh
cd examples/first-run
uv run --no-default-groups --project ../.. hydropattern run first-run.toml --no-excel
```

The working folder still determines the relative input-data path.
See [interpreting first results](results.md) for the expected files.

## Install for the Python API

A uv tool environment is intended for command-line use; it does not make
hydropattern importable in your own Python project. In a Python 3.12+ project,
install it as a project dependency:

```console
uv add "hydropattern==0.3.1"
```

Alternatively, use the pip virtual-environment instructions above and execute
your script with that environment's Python. Local-checkout users can execute
scripts using `uv run --no-default-groups python script.py`.
Continue with the [Python API example](../api/index.md).
