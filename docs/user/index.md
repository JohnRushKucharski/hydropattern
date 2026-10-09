# Evaluate flow patterns with hydropattern

hydropattern evaluates hydrologic time series against conditions you configure.
You supply observations in CSV or Excel and describe components in a TOML text
file. The application produces timestep outcomes and component summary
workbooks; scenario grids can also produce response-surface plots.

Start with [installation](getting-started/installation.md), then complete the
[first evaluation](getting-started/first-run.md) using eight daily observations.
The [results guide](getting-started/results.md) explains each output column and
the example's summary. You do not need to write Python.

## Find an explanation

- [Scientific foundations](concepts/scientific-foundations.md) introduce the
  flow-regime dimensions, and [evaluation order](concepts/evaluation-order.md)
  explains how characteristic outcomes form a component outcome. See
  [unknown outcomes](concepts/unknown-outcomes.md) for uncertainty,
  denominators, event bounds, and plot eligibility.
- The [glossary](concepts/glossary.md) defines components, characteristics,
  qualifying timesteps, frequency windows, and scenarios.
- [Preparing data](guide/preparing-data.md), [output files](guide/outputs.md),
  and [plotting](guide/plotting.md) cover common analysis tasks.
- [Command-line usage](guide/cli.md) explains commands, working folders, and
  output choices.
- The [reference](reference.md) lists configuration fields and existing
  characteristic rules.
- [Runnable examples](examples/index.md) cover seasonal thresholds, duration,
  frequency windows, multiple scenarios, and response-surface coverage.
- [Upgrade guidance](migration.md) identifies completed breaking changes.
- The [Python API](api/index.md) is available for programmatic evaluation.

## Scientific context

The natural flow regime describes hydrologic variation relevant to ecosystems.
Functional flows connect that variation to environmental processes. See
[Poff et al. (1997)](https://doi.org/10.2307/1313099) and
[Yarnell et al. (2020)](https://doi.org/10.1002/rra.3575) for the original
scientific foundations.

The examples illustrate software behavior. Their thresholds are not universal
ecological criteria, and a configured component success is not evidence of
ecological benefit.

The [hydropattern documentation site](https://johnrushkucharski.github.io/hydropattern/)
is published on GitHub Pages.
