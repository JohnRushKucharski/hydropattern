# Installation

**Unreleased:** use current source for these examples. PyPI v0.2.0 and the
latest published GitHub release do not match this documentation. The version
number for the matching release has not yet been assigned.

The commands below use the `docs-reporting-s3` source branch, including its
matching example files. **Until that branch is pushed to GitHub, remote
installation and downloads are unavailable.** If you are reviewing a local
checkout, use the local-checkout instructions below instead. Do not silently
substitute the older published package or another source branch.

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

## Install the current-source command-line application

After the source branch is available on GitHub, use this command on Windows,
macOS, or Linux:

```console
uv tool install --python 3.12 "hydropattern @ git+https://github.com/JohnRushKucharski/hydropattern.git@docs-reporting-s3"
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

**After the matching release is on PyPI**, the recommended command will be
`uv tool install hydropattern`. PyPI supplies the package; uv installs it.
Do not use that unqualified command for the unreleased examples yet.

## Alternative: pip in a virtual environment

Use this option if you already manage Python yourself. Check that your Python
is at least 3.12. A virtual environment keeps dependencies separate.

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "hydropattern @ git+https://github.com/JohnRushKucharski/hydropattern.git@docs-reporting-s3"
.\.venv\Scripts\hydropattern.exe --help
```

On macOS or Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install "hydropattern @ git+https://github.com/JohnRushKucharski/hydropattern.git@docs-reporting-s3"
.venv/bin/hydropattern --help
```

These source commands also require the branch to be published. You can use the
full path to the environment's command without activating the environment.
For the first evaluation, either keep the example files in this working folder
or activate the environment before changing folders.

## Local-checkout route before publication

If you already have the S3 checkout, open a terminal in the repository root.
Check that it contains the current code and `examples/first-run`:

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
after the source branch is published, install it as a project dependency:

```console
uv add "hydropattern @ git+https://github.com/JohnRushKucharski/hydropattern.git@docs-reporting-s3"
```

Alternatively, use the pip virtual-environment instructions above and execute
your script with that environment's Python. Local-checkout users can execute
scripts using `uv run --no-default-groups python script.py`.
Continue with the [Python API example](../api/index.md).
