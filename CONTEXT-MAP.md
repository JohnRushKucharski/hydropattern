# Context Map

## Contexts

- [hydropattern core](./CONTEXT.md) — the library/CLI: evaluates hydrologic timeseries
  against configured flow-pattern components.
- [Great Lakes example](./examples/great_lakes/CONTEXT.md) — example tooling for batch-
  generating and running hydropattern `.toml` configs against Great Lakes lake-level data.

## Relationships

- **Great Lakes example → hydropattern core**: the Great Lakes example generates
  `.toml` configuration files consumed unmodified by hydropattern core's `run` command;
  it does not change core parsing/evaluation behavior.
