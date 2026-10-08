# Evaluate flow patterns with hydropattern

**Unreleased documentation.** These pages describe current source code and the
next release, not PyPI v0.2.0. That release has different frequency behavior and
does not support the current ordered-table syntax. Follow the
[current-source installation instructions](getting-started/installation.md).
The matching package version has not yet been assigned.

hydropattern evaluates hydrologic time series against conditions you configure.
You supply observations in CSV or Excel and describe components in a TOML text
file. The application produces timestep outcomes and component summary
workbooks; scenario grids can also produce response-surface plots.

Start with [installation](getting-started/installation.md), then complete the
[first evaluation](getting-started/first-run.md) using eight daily observations.
The [results guide](getting-started/results.md) explains each output column and
the example's summary. You do not need to write Python.

## Find an explanation

- The [glossary](concepts/glossary.md) defines components, characteristics,
  qualifying timesteps, frequency windows, and scenarios.
- [Command-line usage](guide/cli.md) explains commands, working folders, and
  output choices.
- The [reference](reference.md) lists configuration fields and existing
  characteristic rules.
- [Upgrade guidance](migration.md) identifies completed breaking changes.
- The [Python API](api/index.md) is available for programmatic evaluation.

The existing combined reference is being reorganized. Reporting and
unknown-outcome improvements are not yet implemented; later pages will describe
each change only after it exists in the application. The glossary introduces
terms without promising those future capabilities.

## Scientific context

The natural flow regime describes hydrologic variation relevant to ecosystems.
Functional flows connect that variation to environmental processes. See
[Poff et al. (1997)](https://doi.org/10.2307/1313099) and
[Yarnell et al. (2020)](https://doi.org/10.1002/rra.3575) for the original
scientific foundations.

The examples illustrate software behavior. Their thresholds are not universal
ecological criteria, and a configured component success is not evidence of
ecological benefit.

## Website status

This site is available for local preview and pull-request validation. GitHub
Pages publishing is not enabled yet; it follows completion of the reporting
work and a matching release. The unreleased notice remains until that package
release is available on PyPI.
