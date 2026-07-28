# Design Pattern Review: hydropattern CLI option surface

**Date**: 2026-07-28
**Mode**: Code review (no design doc found for the CLI)
**Scope**: `hydropattern/cli.py`, `hydropattern/parsers.py` (`OutputOptions`/`PlotOptions`/`ClimateCanvasPlotOptions`), and the two consumers of `hydropattern.cli.run`: the `hydropattern` package itself and the sibling `hydropattern-gui` repo (`runner_service.py`, `config_model.py`). `climate-canvas` was inspected only for its `plot_response_surface` signature (new `fillin: bool = False` parameter).
**Trigger**: user wants to add `--fillin` (a new climate-canvas plotting option) to the CLI, and is concerned the `run` command's option surface is already too large/flat.

## Summary

The concern is well-founded, but the underlying data model is already good — `OutputOptions -> PlotOptions -> ClimateCanvasPlotOptions` is a clean, layered **Parameter Object** hierarchy in `hydropattern/parsers.py`. The problem is that this hierarchy gets **flattened back into ~10 individual flags on one Typer command**, and — more importantly — that same flattening is **independently re-implemented a second and third time** in `hydropattern-gui` (its own `RunOptions` dataclass, its own `build_command()` flag mapper, and its own from-scratch `ClimateCanvasPlotOptions`-equivalent parser/serializer in `config_model.py`). Adding one new climate-canvas option today means touching **5 places across 2 repos** in lockstep. That's the real "bloat" — not the Typer command itself, which is comparatively a symptom.

## Current state

### `hydropattern/cli.py::run`
- One Typer command, ~14 parameters: `path`, `plot`, `output_directory`, `write_to_excel`, `overwrite`, `interp`, `show`, `threshold`, `color_map`, `color_map_ticks`, `run_toml_options` — mixing four distinct concerns in one flat namespace:
  1. run-level config (`path`)
  2. `[output]` (`output_directory`, `write_to_excel`, `overwrite`)
  3. `[output.plot]` (`plot`)
  4. `[output.plot.climate-canvas]` (`interp`, `show`, `threshold`, `color_map`, `color_map_ticks`)
  - plus one cross-cutting escape hatch (`run_toml_options`) that has to know about every option above to check for conflicts (`require_no_conflicting_cli_options`).
- All CLI flags default to `None` ("not explicitly passed") and are merged over TOML-file values in `resolve_output_options` — this **CLI-overrides-file** pattern is good practice and should be preserved.
- `_resolve_color_map` (auto-reversing colormap direction) is a nice small **pure function** — no complaints there.

### `hydropattern/parsers.py`
- `OutputOptions(directory, overwrite, excel, metric, plot: PlotOptions)`
- `PlotOptions(enabled, climate_canvas: ClimateCanvasPlotOptions)`
- `ClimateCanvasPlotOptions(interpolate, show, title, xlabel, ylabel, zlabel, threshold, color_map, color_map_ticks)`
- This is already a textbook nested **Parameter Object** / value-object hierarchy — frozen dataclasses, one section of the TOML schema per class. This is exactly what a CLI option surface should be built *from*, not re-derived independently.

### `hydropattern-gui`
- `runner_service.py::RunOptions` is a **flat dataclass duplicating the CLI's flat flags one-for-one** (not the nested parser hierarchy), plus a `build_command()` that hand-maps each field to a `--flag` string, plus an `InProcessHydropatternRunner` that unpacks the same flat dataclass a third time into keyword arguments calling `hydropattern.cli.run` directly.
- `config_model.py` **separately** defines its own `ClimateCanvasPlotOptions`-equivalent class and its own hand-written TOML parse/serialize functions (`_parse_climate_canvas`, the inline serializer around line 170-197) — this is a **second, independent reimplementation** of the exact same schema that already lives in `hydropattern.parsers.ClimateCanvasPlotOptions`. It is not importing/reusing the hydropattern package's parser types at all.

### Net effect
Three independent copies of "what fields does `[output.plot.climate-canvas]` have":
1. `hydropattern/parsers.py::ClimateCanvasPlotOptions` (+ its TOML parser)
2. `hydropattern/cli.py::run`'s flat Typer parameters (+ `require_no_conflicting_cli_options`'s hardcoded field list)
3. `hydropattern-gui/config_model.py`'s own dataclass + parser/serializer
4. `hydropattern-gui/runner_service.py::RunOptions` + `build_command()`'s flag mapping

Adding `--fillin` requires editing all four in sync, with no compiler/test enforcement that they stay in sync (nothing currently fails if `hydropattern-gui` forgets a field — it would just silently drop it when building the CLI command).

## Code smells identified

1. **Long Parameter List** (`run`, `RunOptions`, `require_no_conflicting_cli_options`, `resolve_output_options`, `InProcessHydropatternRunner.run`) — canonical trigger for a **Parameter Object**, which... already exists (`OutputOptions`/`PlotOptions`/`ClimateCanvasPlotOptions`) but isn't being used at the CLI boundary or in the GUI.
2. **Duplicated Knowledge / Shotgun Surgery** — the climate-canvas option schema is defined independently in (at least) 3 places across 2 repos. A single new upstream option requires a coordinated multi-repo, multi-file change with no shared source of truth.
3. **Feature Envy / cross-cutting validation coupled to a flat list** — `require_no_conflicting_cli_options` (hydropattern) and `_has_explicit_output_options` (hydropattern-gui) both hand-enumerate the exact same flag list just to answer "was anything explicitly set?" — this is naturally expressed by iterating a dataclass's fields, not by name.
4. **Single "God Command"** — one Typer command does config loading, timeseries loading, component building, output writing, *and* plotting. Not unreasonable for a small CLI, but it is the reason every new climate-canvas knob shows up as another top-level flag instead of a sub-option.

## Applicable patterns

- **Parameter Object (already half-applied)** — the fix is to *use* the existing `ClimateCanvasPlotOptions`/`PlotOptions` dataclasses as the CLI's own vocabulary (e.g. via Typer's `Annotated[...]` + a small "collect overrides into a partial `ClimateCanvasPlotOptions`" helper, or a Click/Typer parameter-group convention), instead of re-flattening them into positional-like `None`-defaulted args every time. The same dataclasses could then be *reused verbatim* by `hydropattern-gui` instead of being redefined — turning 3-4 copies into 1.
- **rich_help_panel grouping (Typer/Click built-in, no upgrade needed)** — `typer.Option(..., rich_help_panel="Climate Canvas Plot")` already exists in the currently pinned `typer==0.12.3` (confirmed via `inspect.signature`) and `rich` is already an installed transitive dependency. This lets `--help` visually group the plot/climate-canvas flags under their own heading with **zero architecture change** — cheapest possible win if the goal is just decluttering `--help` output.
- **Facade** — `run()` is already acting as a facade over load/build/write/plot; that's appropriate and shouldn't be split apart just to reduce flag count (splitting the *command* doesn't reduce the *option schema* duplication, which is the actual pain point).
- **Anti-corruption / shared kernel** — since `hydropattern-gui` is a separate repo, the long-term fix for the *architectural* duplication is for it to import `hydropattern.parsers.ClimateCanvasPlotOptions`/`PlotOptions`/`OutputOptions` (it already depends on the `hydropattern` package at runtime, per `InProcessHydropatternRunner`) rather than re-declaring an equivalent shape. This is a "shared kernel" between the two repos, not two independently-versioned schemas that must be kept in sync by hand.

## What's already good (no changes needed)

- The CLI-overrides-TOML merge pattern (`resolve_output_options`, all-`None`-default flags) is correct and idiomatic Typer/Click design — keep it regardless of what else changes.
- `errors.py`'s `CliErrorCode`/`ParserErrorCode` stable-code error envelope is a solid, consistent validation pattern already in place.
- `_resolve_color_map` and `plot_components` are appropriately small, single-purpose functions.
- Typer itself does not need to be upgraded to solve this problem — `rich_help_panel` and `Annotated`-based options are already available in the pinned `0.12.3`. An upgrade could still be worth doing on its own general-hygiene merits (current pin is from mid-2024; latest supports Python 3.14, likely has bug fixes/docs improvements), but it is **not a blocker** for anything discussed here.

## Non-goals / things this review is not recommending by default

- Splitting `run` into separate `run` / `plot` subcommands is *possible* but is a bigger, more disruptive change (breaks the single `hydropattern run x.toml --plot` muscle memory, and the GUI/tests would need parallel updates) for a problem (flat flag duplication across repos) that a shared Parameter Object would solve more directly. This is flagged as a question below rather than a recommendation.

---

## Questions for you (please answer one at a time, I won't start implementing until we agree on a plan)

## Decisions reached (2026-07-28 interview)

1. **Keep a single `run` command.** No `run`/`plot` subcommand split — the flat-flag duplication across repos was the real problem, not the single command.
2. **`hydropattern-gui` will import `hydropattern.parsers`'s dataclasses directly** (`ClimateCanvasPlotOptions`, `PlotOptions`, `OutputOptions`) as the shared vocabulary between the two repos, replacing its own independently-defined equivalents in `config_model.py`. This is an explicit shared-kernel relationship (the GUI already imports `hydropattern.cli.run` directly via `InProcessHydropatternRunner`, so this isn't a new coupling, just a more complete one).
3. **Named `--flag` per climate-canvas option** (not a generic `--cc-opt key=value` passthrough) — preserves discoverability, type validation at the CLI boundary, and shell completion; a passthrough would undermine the CLI/GUI/library consistency goal (an opaque key/value bag doesn't translate into a GUI form the way typed dataclass fields do).
4. **DRY-ness boundary**: individual `typer.Option(...)` flag declarations stay hand-written (for good, flag-specific `--help` text) but the repetitive "collect the non-`None` overrides and merge them onto the base dataclass" logic (currently duplicated per-field in `resolve_output_options` and `require_no_conflicting_cli_options`) gets centralized into one small helper that works off `dataclasses.fields()`, so a future new field only needs one flag declaration, not a matching new `if X is not None: ...` line in two/three different functions across two repos.
5. **Scope: both repos, this session.** `hydropattern` (parsers/cli) and `hydropattern-gui` (`config_model.py`, `runner_service.py`) are updated together, not sequenced across sessions.
6. **Also apply `rich_help_panel` grouping** to `run`'s `--help` output (e.g. "Output", "Climate Canvas Plot" panels) — zero-risk, available in the currently pinned Typer.
7. **Upgrade Typer 0.12.3 → latest 0.27.x, drop the now-redundant direct `click` pin.** Typer 0.26+ vendors Click internally, so the separate `click>=8.1.7,<8.2.0` dependency becomes unnecessary. No code in this repo imports `click` directly, and existing tests use only the stable `typer.testing.CliRunner`/`result.exit_code`/`result.stdout` surface, so risk is low. Typer 0.27.0 has one relevant breaking change (`--help` metavar formatting) to sanity-check after upgrading — no snapshot/golden-file tests of `--help` output exist today, so this is a manual spot-check, not a text-diff risk.
8. **`--fillin` semantics** (confirmed from `climate_canvas.plots_utilities.plot_response_surface`/`data_utilities.evenly_space`): a simple boolean flag (climate-canvas's own CLI models it as `typer.Option(False, "--fillin", ...)`, no `--no-fillin` pair), independent of (not mutually exclusive with) `--interp`/`interpolate` — both are passed to `evenly_space` together. `fillin=True` estimates missing (NaN) grid cells from a global Delaunay triangulation rather than leaving them blank.

