# Next-session handoff

v0.3.0 release/site and corrective v0.3.1 PyPI description release are
complete. PR #50 merged version-synchronized README/install guidance at
`988fd09`; PyPI now lists 0.3.1 as latest and its project page description has
no stale unreleased banner. The v0.3.0-specific release metadata remains
historical. PyPI publisher run `37918580505` and Pages deployment run
`37918727579` succeeded. Pattern-correctness documentation and reporting
R0–R10 are complete. Optional R11 remains out of scope and needs separate
approval.

Before further work:

1. Run `git branch --show-current`, `git status --short`, and
   `git --no-pager log -5 --oneline`. Preserve any changes made since this
   handoff; do not assume the worktree is clean.
2. Read `CONTEXT.md`, `docs/developer/plans/user-documentation.md`,
   `docs/developer/plans/2026-10-06-reporting-metrics-tdd.md`, and
   `docs/developer/plans/2026-10-07-documentation-and-reporting-sequence.md`.
3. Confirm `main` and `origin/main` contain `988fd09`; retain existing
   documentation and reporting branches for provenance.
4. Review R1–R10 completion notes, including the R4 correction, in the reporting plan.
5. S10/D7 for v0.3.0 and the corrective v0.3.1 PyPI-description release are
   complete. S9 cleanup retained reviewed files; no deletion was authorized.
   Do not begin optional R11 without approval.

## Release completion record

S9 cleanup review is complete. The six generated-output candidates and
historical developer records were reviewed and retained. No deletion was
performed or authorized.

**S10 / D7 completed 2026-10-09.** GitHub Release `v0.3.0` targets commit
`9acb8cf`; publisher workflow `37915725339` succeeded. PyPI serves version
`0.3.0` wheel and sdist, and a clean Python 3.12 install passed
`hydropattern --help`. PR #48 merged post-release user guidance at `67e964b`.
Pages uses the GitHub Actions source; deployment workflow `37916615564`
succeeded. The public homepage and installation page returned HTTP 200 with
current content; both first-run example downloads returned HTTP 200.

**PyPI description correction completed 2026-10-09.** The `v0.3.0` tag's
README still contained its unreleased notice, which was embedded in its
immutable PyPI metadata. PR #50 prepared and merged v0.3.1 with corrected
README content and install/download links. Publisher workflow `37918580505`
and Pages deployment `37918727579` succeeded. Current PyPI project page and
live install docs were verified; the old v0.3.0-specific metadata remains
unchanged.

## Completed implementation and validation

R1 preserves unavailable order-1 comparison inputs as `NaN`. R2 propagates
configured water-year boundaries to `Result`, uses shared ending-year labels,
supports optional February 29 in daily completeness, and exposes cadence-
validated daily/monthly observed exposure. R7 now uses observed exposure for
event rates, including partial water years. R3 evaluates all possible duration run
boundaries: known verdicts remain known only when every possible run agrees;
definite preceding failures still settle failure. R4 propagates unknown
qualifying trials through un-nested forward frequency windows and considers
possible exclusive-window schedules. R4 correction `e53612a` restores exact
shared-trial correlations, including `[1, unknown]`, `= 1`, N=2 producing
`[unknown, 1]`; the full-output oracle now exhausts ternary inputs of lengths
1–4, all six operators, inclusive bounds, and both window modes.
R5 uses all observed annual timesteps as the annual condition denominator,
evaluates every attainable fraction, reduces yearly count diagnostics with
three-valued OR, and retains unknown annual anchors in interannual windows.
Complete-year verdicts broadcast; partial years remain unknown and excluded.
R6 excludes unknowns from each column's summary denominator, aggregates
whole-record portions from total successes and known outcomes, aligns
`Result.frequency_table()` water-year labels and percentages with formatter
summaries, and rejects removed `return_period` mode with migration guidance.

R7 adds named event-count/rate bounds, scalar ambiguity errors, annual
first-success attribution, and whole-record observed exposure, including
partial water years. Bounds treat unknown final outcomes independently and
may include counts impossible under source-characteristic dependencies.
Implementation commit: `27112d2`.

R8 adds `reporting_details` sheet to each component summary workbook while
preserving existing metric sheets and raw timestep files. It reports
successful/known/total counts, known-outcome coverage, completeness, component
event-count/rate bounds, observed exposure, and availability reasons. It
handles partial years, sheet-name collisions, CSV/Excel exports, and Python/CLI
parity. `docs/user/guide/outputs.md` documents the sheet. R8 is committed; full suite
passed (776 tests), mypy, strict MkDocs, Ruff, and `git diff --check`.

R9 adds coverage-aware response surfaces with `minimum_coverage = 0.9` in
`[output.plot]`, CLI override `--minimum-coverage`, and a Python API argument.
Under-covered and undefined component summaries are excluded from plotted
grids, with `{component}_grid_coverage.csv` explaining coordinates, raw
summary, counts, coverage, eligibility, and reason. Plot titles and warnings
show cutoff and excluded scenarios. `fillin` conflicts with withheld
scenarios; fewer than three non-collinear eligible scenarios raises an
explicit no-surface error after grid exports. Actual renderer tests cover
protected gaps with interpolation on/off. Plotting, configuration, CLI, API,
and migration pages are updated. R9 is committed; see its completion record.
Full `uv run pytest -q` passed (791 tests), `uv run mypy
hydropattern/`, strict MkDocs build, Ruff on changed Python files, and
`git diff --check` passed.

R10 adds a dedicated unknown-outcomes page linked from affected concepts,
references, reporting, plotting, migration, and API guidance. The
`examples/response-surface` pack has four scenarios, 90% inclusive eligibility,
one withheld scenario, and verified outcome arrays, retained summaries, grid
exports, coverage reasons, and rendered output. The unknown-outcomes page
covers calculation uncertainty, duration/frequency and annual examples,
partial water years, leap-day assumptions, unsupported schedules, event
bounds, summary denominators, plot cutoff/fillin/color behavior, and links to
worked pages. Migration and plan/handoff states are current. R10 is committed
on `reporting-metrics`; required reporting MVP work is complete.

Validation: full `uv run pytest -q` passed (801 tests), `uv run mypy
hydropattern/`, strict MkDocs build, Ruff on the new acceptance test, and
`git diff --check` passed on 2026-10-09. Reporting-branch changes have not
been merged to `main` or pushed.
Do not delete reviewed artifacts or publish/enable deployment without explicit
user authorization.

Dense-unknown exact count windows cost more than the superseded R4
approximation. A 3,653-step, 95%-unknown record with N=30 took approximately
15 ms / 1.1 s for overlapping / exclusive `>= 2`, and 9.3 s / 1.5 s for
`= 2`. Details and limitations are recorded in the reporting plan.
