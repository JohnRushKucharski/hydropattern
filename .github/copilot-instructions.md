# Copilot Instructions for hydropattern

This document provides comprehensive onboarding information for coding agents working on the hydropattern repository. Following these guidelines will minimize exploration time, reduce build failures, and accelerate task completion.

## Repository Overview

**hydropattern** is a Python library for detecting and evaluating natural flow regime patterns in streamflow time series data. It processes time series data through pattern detection algorithms and generates visualizations and reports.

### Core Purpose
- Analyze streamflow time series data for natural flow regime patterns
- Evaluate ecological flow characteristics against reference conditions  
- Generate reports and visualizations for hydrological analysis
- Support both single and multi-timeseries processing workflows

### Key Technologies
- **Python 3.12+** (strict requirement)
- **uv** for dependency and environment management
- **typer** for CLI framework
- **pandas/matplotlib** for data processing and visualization
- **TOML** configuration files with preserved key ordering

## Quick Start - Essential Commands

### Environment Setup
```bash
# Verify uv is installed
uv --version

# Install dependencies (creates virtual environment automatically)
uv sync --group test --group dev
```

### Build and Test
```bash
# Run all tests (54 tests total)
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_patterns.py

# Run CLI help
uv run python -m hydropattern --help

# Test CLI with example
uv run python -m hydropattern run examples/minimal.toml
```

### Code Quality
```bash
# Type checking (has 2 known errors, see Known Issues)
uv run mypy hydropattern/

# No other linters (ruff, pylint, flake8) currently installed
```

## Project Structure

```
hydropattern/
├── hydropattern/           # Main source code
│   ├── __main__.py        # Entry point for python -m hydropattern
│   ├── cli.py             # CLI implementation (typer-based)
│   ├── patterns.py        # Core pattern evaluation logic
│   ├── timeseries.py      # Time series processing and plotting
│   └── parsers.py         # Configuration file parsing
├── tests/                 # Test suite (54 tests)
│   ├── test_cli.py        # CLI functionality tests
│   ├── test_patterns.py   # Pattern evaluation tests
│   └── test_timeseries.py # Time series processing tests
├── examples/              # Sample configurations and data
│   ├── minimal.toml       # Simple single-timeseries config
│   ├── multi_timeseries.toml  # Multi-timeseries config
│   ├── detailed.toml      # Comprehensive config example
│   └── *.csv              # Sample time series data
├── notebooks/             # Jupyter demonstrations
└── docs/                  # Documentation and papers
```

## Key Files to Understand

### Entry Points
- **`hydropattern/__main__.py`**: 6-line entry point, imports and runs CLI
- **`hydropattern/cli.py`**: Main CLI logic using typer framework

### Core Logic
- **`hydropattern/patterns.py`**: Pattern evaluation algorithms, Component/Characteristic classes
- **`hydropattern/timeseries.py`**: Time series processing, plotting with broken-axis support
- **`hydropattern/parsers.py`**: TOML configuration parsing

### Configuration
- **`pyproject.toml`**: Project metadata and dependencies, Python 3.12+ requirement
- **`examples/*.toml`**: Working example configurations

## CLI Interface

The main interface is through the CLI:

```bash
# Basic usage
uv run python -m hydropattern run <config.toml>

# With plotting
uv run python -m hydropattern run <config.toml> --plot

# With custom output directory  
uv run python -m hydropattern run <config.toml> --output-dir <path>

# Generate Excel output
uv run python -m hydropattern run <config.toml> --excel
```

### Working Examples
- `examples/minimal.toml` - Validated working example
- `examples/multi_timeseries.toml` - Multi-series processing
- `examples/detailed.toml` - Comprehensive configuration

## Dependencies and Environment

### Python Requirements
- **Python 3.12+** (strict requirement in pyproject.toml)
- Managed via uv virtual environment (`.venv`)

### Key Dependencies
```toml
python = "^3.12"
typer = "*"           # CLI framework
pandas = "*"          # Data processing
matplotlib = "*"      # Plotting
tomli-w = "*"        # TOML writing
climate-canvas = "*"  # Visualization (external dependency)
```

### Development Dependencies
```toml
pytest = "*"
mypy = "*"
types-* = "*"        # Type stubs for mypy
jupyter = "*"
```

## Known Issues and Workarounds

### mypy Type Errors (2 current)
1. **`patterns.py:643`** - Dict entry contains `None` where the inferred type does not allow it
2. **`patterns.py:647`** - Dict entry contains `None` where the inferred type does not allow it

**Workaround**: These are known issues, PRs should not break builds due to existing mypy errors.

### Missing Linters
- **ruff**: Not installed (command not found)
- **pylint**: Not installed 
- **flake8**: Not installed

**Implication**: Only mypy type checking is available. Consider adding ruff for comprehensive linting.

### VS Code Configuration
If working in VS Code, ensure Python interpreter points to the project's `.venv` virtual environment:
```
.vscode/settings.json should reference `.venv\\Scripts\\python.exe` on Windows
```

## Configuration File Format

The system uses TOML configuration files with specific structure:

```toml
# Key ordering is preserved (tested and verified)
[config]
reference_timeseries = "path/to/reference.csv"
evaluation_timeseries = ["path/to/eval1.csv", "path/to/eval2.csv"]

[output]
directory = "output/"
prefix = "results"

# Component and characteristic definitions follow
```

**Key Insight**: TOML key ordering is preserved during parsing - important for reproducible outputs.

## Testing Strategy

### Test Coverage
- **54 tests total** across 3 test files
- Comprehensive coverage of CLI, patterns, and timeseries modules
- All tests currently passing

### Running Tests
```bash
# Full test suite
uv run pytest

# Specific test categories
uv run pytest tests/test_cli.py      # CLI functionality
uv run pytest tests/test_patterns.py # Pattern evaluation
uv run pytest tests/test_timeseries.py # Time series processing

# Verbose output for debugging
uv run pytest -v -s
```

## Development Workflow

### Standard Development Cycle
1. **Environment**: `uv sync --group test --group dev` (if dependencies changed)
2. **Code Changes**: Edit source files in `hydropattern/`
3. **Testing**: `uv run pytest` (ensure all pass)
4. **Type Check**: `uv run mypy hydropattern/` (aware of known errors)
5. **Manual Testing**: `uv run python -m hydropattern run examples/minimal.toml`

### Adding New Features
- Update relevant module in `hydropattern/`
- Add tests in corresponding `tests/test_*.py` file  
- Update examples if CLI interface changes
- Consider notebook demonstrations for complex features

### Configuration Changes
- Modify `pyproject.toml` for dependencies
- Update example TOML files in `examples/`
- Test with `uv run python -m hydropattern run examples/minimal.toml`

## Common Patterns

### Error Handling
- CLI uses typer's built-in error handling
- Configuration validation in `parsers.py`
- Time series validation in `timeseries.py`

### Data Flow
1. **Input**: TOML config → `parsers.py`
2. **Processing**: Time series data → `timeseries.py` 
3. **Analysis**: Pattern evaluation → `patterns.py`
4. **Output**: Results → CLI output handling

### Visualization
- Plotting functionality in `timeseries.py`
- Broken-axis support for large data ranges
- Integration with climate-canvas for advanced plots

## Debugging Tips

### Build Failures
- Check Python version: `python --version` (must be 3.12+)
- Verify uv environment: `uv python list`
- Reinstall dependencies: `uv sync --reinstall --group test --group dev`

### Runtime Issues  
- Test with known-good config: `examples/minimal.toml`
- Check file paths in TOML configuration
- Verify time series data format (CSV expected)

### Type Checking
- Known mypy errors are acceptable (2 current)
- New type errors should be addressed
- Use type stubs for external dependencies

## Performance Considerations

- Single time series: Fast processing (< 1 second)
- Multi-time series: Scales linearly with number of series
- Plotting: Can be slow for large datasets, use `--plot` selectively
- Memory usage: Proportional to time series data size

## Security and Dependencies

- No known security issues
- External dependencies managed via `uv.lock`
- climate-canvas is external dependency (not on PyPI standard channels)

---

**Last Updated**: Based on repository state with uv-managed dependencies, Python 3.12.4, 54 passing tests, and 2 known mypy errors.

**For Issues**: Check existing tests and examples first. Most functionality is well-tested and examples are verified working.