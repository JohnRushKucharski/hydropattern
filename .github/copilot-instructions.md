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
- **Poetry 1.8.3** for dependency management
- **typer** for CLI framework
- **pandas/matplotlib** for data processing and visualization
- **TOML** configuration files with preserved key ordering

## Quick Start - Essential Commands

### Environment Setup
```bash
# Verify Poetry is installed
poetry --version  # Should show 1.8.3+

# Install dependencies (creates virtual environment automatically)
poetry install

# Activate shell (optional, poetry run handles this)
poetry shell
```

### Build and Test
```bash
# Run all tests (47 tests total)
poetry run pytest

# Run tests with verbose output
poetry run pytest -v

# Run specific test file
poetry run pytest tests/test_patterns.py

# Run CLI help
poetry run python -m hydropattern --help

# Test CLI with example
poetry run python -m hydropattern run examples/minimal.toml
```

### Code Quality
```bash
# Type checking (has 3 known errors, see Known Issues)
poetry run mypy hydropattern/

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
├── tests/                 # Test suite (47 tests)
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
- **`pyproject.toml`**: Poetry configuration, Python 3.12+ requirement
- **`examples/*.toml`**: Working example configurations

## CLI Interface

The main interface is through the CLI:

```bash
# Basic usage
poetry run python -m hydropattern run <config.toml>

# With plotting
poetry run python -m hydropattern run <config.toml> --plot

# With custom output directory  
poetry run python -m hydropattern run <config.toml> --output-dir <path>

# Generate Excel output
poetry run python -m hydropattern run <config.toml> --excel
```

### Working Examples
- `examples/minimal.toml` - Validated working example
- `examples/multi_timeseries.toml` - Multi-series processing
- `examples/detailed.toml` - Comprehensive configuration

## Dependencies and Environment

### Python Requirements
- **Python 3.12+** (strict requirement in pyproject.toml)
- Managed via Poetry virtual environment

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

### mypy Type Errors (3 current)
1. **`patterns.py:394`** - Return type mismatch in `evaluate_patterns()`
2. **`cli.py:28`** - Import error with `climate_canvas`  
3. **`cli.py:145`** - Type mismatch in return statement

**Workaround**: These are known issues, PRs should not break builds due to existing mypy errors.

### Missing Linters
- **ruff**: Not installed (command not found)
- **pylint**: Not installed 
- **flake8**: Not installed

**Implication**: Only mypy type checking is available. Consider adding ruff for comprehensive linting.

### VS Code Configuration
If working in VS Code, ensure Python interpreter points to Poetry virtual environment:
```
.vscode/settings.json should reference Poetry's Python path
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
- **47 tests total** across 3 test files
- Comprehensive coverage of CLI, patterns, and timeseries modules
- All tests currently passing

### Running Tests
```bash
# Full test suite
poetry run pytest

# Specific test categories
poetry run pytest tests/test_cli.py      # CLI functionality
poetry run pytest tests/test_patterns.py # Pattern evaluation  
poetry run pytest tests/test_timeseries.py # Time series processing

# Verbose output for debugging
poetry run pytest -v -s
```

## Development Workflow

### Standard Development Cycle
1. **Environment**: `poetry install` (if dependencies changed)
2. **Code Changes**: Edit source files in `hydropattern/`
3. **Testing**: `poetry run pytest` (ensure all pass)
4. **Type Check**: `poetry run mypy hydropattern/` (aware of known errors)
5. **Manual Testing**: `poetry run python -m hydropattern run examples/minimal.toml`

### Adding New Features
- Update relevant module in `hydropattern/`
- Add tests in corresponding `tests/test_*.py` file  
- Update examples if CLI interface changes
- Consider notebook demonstrations for complex features

### Configuration Changes
- Modify `pyproject.toml` for dependencies
- Update example TOML files in `examples/`
- Test with `poetry run python -m hydropattern run examples/minimal.toml`

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
- Verify Poetry environment: `poetry env info`
- Reinstall dependencies: `poetry install --verbose`

### Runtime Issues  
- Test with known-good config: `examples/minimal.toml`
- Check file paths in TOML configuration
- Verify time series data format (CSV expected)

### Type Checking
- Known mypy errors are acceptable (3 current)
- New type errors should be addressed
- Use type stubs for external dependencies

## Performance Considerations

- Single time series: Fast processing (< 1 second)
- Multi-time series: Scales linearly with number of series
- Plotting: Can be slow for large datasets, use `--plot` selectively
- Memory usage: Proportional to time series data size

## Security and Dependencies

- No known security issues
- External dependencies managed via Poetry lock file
- climate-canvas is external dependency (not on PyPI standard channels)

---

**Last Updated**: Based on repository state with Poetry 1.8.3, Python 3.12.4, 47 passing tests, and 3 known mypy errors.

**For Issues**: Check existing tests and examples first. Most functionality is well-tested and examples are verified working.