# hydropattern-gui Agent Handoff (Copy-Ready)

Purpose: single file with all context needed to build `hydropattern-gui` well, in vertical TDD slices, without re-discovery.

---

## 1) Product goal

Build small, organized GUI for hydropattern.

GUI must:
- build valid TOML config
- run hydropattern through CLI using that TOML
- preserve reproducibility and parity with CLI/library behavior
- ship as Windows executable

---

## 2) Decisions already made (locked)

1. **Repo boundary**
   - GUI lives in separate repo: `hydropattern-gui`.
   - Rationale: avoid brittleness/regression risk in core hydropattern CLI/library code.

2. **UI stack**
   - Tkinter + ttk.

3. **Execution path**
   - Always generate TOML.
   - Always execute subprocess CLI: `hydropattern run <toml>`.
   - Do **not** use direct hydropattern internal imports as primary run path.

4. **TOML output policy**
   - Two modes:
     - Complete explicit TOML (default, reproducibility-first)
     - Minimal TOML (optional, readability-first)

5. **Run flow**
   - Quick test run first, then normal run.

6. **Packaging and distribution**
   - Windows-first executable via PyInstaller.
   - Bundle pinned `hydropattern` + compatible `climate-canvas`.
   - Strict pin policy: each GUI release pins one core stack set.

7. **Must-have v1 UX**
   - live log pane
   - cancel button (terminates subprocess)
   - final status + open output-folder action
   - About panel with versions (gui/hydropattern/climate-canvas)

8. **Import/edit behavior**
   - GUI must open existing TOML and hydrate form for edit/re-run.

9. **Testing strategy**
   - Unit tests + smoke integration tests.
   - No full GUI automation required in v1.

---

## 3) Domain language to keep consistent

From hydropattern `CONTEXT.md`:
- **Scenario** = one data column evaluated independently.
- **Scenario grid** = columns named with `_x_y` pattern.
- **Precipitation delta** = first numeric value in scenario-grid name (x-axis).
- **Temperature delta** = second numeric value (y-axis).
- **Metric** = scalar per scenario used for z-axis (`portion` | `percentage` | `return_period`).

Use these terms in GUI labels/docs/tests.

---

## 4) Non-negotiable reproducibility/parity rules

1. Generated TOML used for run must be saved with outputs.
2. Run artifacts must be persistable:
   - TOML used
   - full command string
   - stdout/stderr logs
   - version metadata
3. Validation should prefer real CLI behavior over custom drift-prone GUI logic.
4. GUI defaults must not silently diverge from hydropattern parser defaults.

---

## 5) Suggested architecture (target shape)

Use clear boundaries:

- `src/hydropattern_gui/domain/`
  - dataclasses/state models for GUI config
  - normalization-independent value objects

- `src/hydropattern_gui/toml_io/`
  - serializer/deserializer
  - stable key ordering
  - complete vs minimal mode

- `src/hydropattern_gui/runner/`
  - subprocess command builder
  - process lifecycle, streaming logs, cancel/terminate
  - run result object/state machine

- `src/hydropattern_gui/ui/`
  - Tkinter frames/widgets/controllers
  - keep thin; call domain/toml_io/runner

- `src/hydropattern_gui/versioning/` (optional)
  - about/version metadata
  - pinned dependency visibility

- `tests/unit/`
  - toml mapping + mode behavior
  - runner behavior

- `tests/integration/`
  - real `hydropattern run` smoke tests with fixture inputs

- `packaging/`
  - PyInstaller spec/build scripts

---

## 6) Explicit anti-patterns (do not do)

1. Do not execute hydropattern internals directly as primary path.
2. Do not hide generated TOML from user.
3. Do not skip cancel/termination behavior.
4. Do not ship unpinned hydropattern/climate-canvas in exe.
5. Do not make GUI-only config semantics that drift from parser behavior.

---

## 7) Backlog: 5 vertical AFK slices (approved)

### Slice 1 — Tracer bullet: TOML round-trip core

**Type:** AFK  
**Blocked by:** None

**What to build**
- Domain config model + TOML writer/reader.
- Round-trip path: GUI state -> TOML -> GUI state.

**Acceptance criteria**
- [ ] Complete mode writes explicit full TOML with stable key order.
- [ ] Minimal mode omits defaults correctly.
- [ ] Importing generated TOML restores same config state.
- [ ] Unit tests cover round-trip + mode differences.

---

### Slice 2 — Runner service: CLI execution contract

**Type:** AFK  
**Blocked by:** Slice 1

**What to build**
- Subprocess runner for `hydropattern run <toml>`.
- Live log streaming.
- Cancel support.
- Deterministic run result object.

**Acceptance criteria**
- [ ] Deterministic command construction from config.
- [ ] Incremental stdout/stderr streaming callback.
- [ ] Cancel terminates process tree cleanly.
- [ ] Quick test run then normal run flow implemented.
- [ ] Integration smoke test runs fixture TOML.

---

### Slice 3 — Core UI shell (timeseries/output/metric)

**Type:** AFK  
**Blocked by:** Slices 1, 2

**What to build**
- Tkinter UI for core sections.
- Load/save TOML.
- Preview TOML.
- Run wiring to runner service.

**Acceptance criteria**
- [ ] User can author timeseries/output/metric config without manual TOML editing.
- [ ] Open existing TOML hydrates form correctly.
- [ ] Preview TOML matches current form + selected mode (complete/minimal).
- [ ] Run button executes via runner service and shows status/log/output link.

---

### Slice 4 — Component editor + climate-canvas advanced panel

**Type:** AFK  
**Blocked by:** Slice 3

**What to build**
- Structured component/characteristic editor.
- Advanced climate-canvas controls.

**Acceptance criteria**
- [ ] Supports characteristic rows:
  - timing
  - magnitude
  - duration
  - rate_of_change
  - frequency
- [ ] Supports ordering and per-component `verbose` + `success_pattern`.
- [ ] Supports climate-canvas keys:
  - `interpolate`
  - `show`
  - `title`
  - `xlabel`
  - `ylabel`
  - `zlabel`
  - `threshold`
  - `color_map`
  - `color_map_ticks`
- [ ] Validation errors shown pre-run with clear field mapping.

---

### Slice 5 — Windows executable packaging + release docs

**Type:** AFK  
**Blocked by:** Slice 4

**What to build**
- PyInstaller packaging flow.
- Release docs and compatibility table.

**Acceptance criteria**
- [ ] Single Windows executable built reproducibly.
- [ ] Bundles pinned `hydropattern` + `climate-canvas`.
- [ ] About panel shows GUI/core dependency versions.
- [ ] Release docs include install, run, known limits, compatibility table.

---

## 8) TDD process contract per slice

For each slice:
1. RED: add one behavior test.
2. GREEN: minimal implementation to pass.
3. REFACTOR: improve structure, keep green.
4. Verify with:
   - unit tests
   - relevant integration smoke tests
   - `ruff`, `mypy`

Prefer thin, demoable vertical increments.

---

## 9) Minimal implementation notes for first agent pass

1. Start with parser-safe TOML mapping before complex UI.
2. Keep run engine independent from UI to simplify testing.
3. For cancel on Windows, terminate process tree (not parent only).
4. Keep fixture data small for smoke tests.
5. Add simple run-state enum (idle/running/succeeded/failed/cancelled).

---

## 10) Setup baseline already prepared

Local repo created at:
- `C:\Users\kucharsk\dev\hydropattern-gui`

Using `uv` with:
- deps: `tomli-w`
- dev deps: `pytest`, `ruff`, `mypy`, `pyinstaller`

Baseline checks previously passed:
- `uv run pytest`
- `uv run ruff check .`
- `uv run mypy src`

---

## 11) Definition of done for v1

v1 done when:
- all 5 slices accepted
- full CLI-based run path stable
- TOML import/export round-trip stable
- Windows exe produced and manually smoke-tested
- docs sufficient for user install and run

