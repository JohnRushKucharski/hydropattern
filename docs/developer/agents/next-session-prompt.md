# Next-session handoff

Continue hydropattern documentation work from local branch `reporting-metrics`.
R0–R10 and S8 are complete and committed on this branch, based on local
`main` at `b05b326`. R9 is `b657ba4`; the following R10 commit is titled
`docs: complete R10 and prepare documentation handoff`.
Do not repeat reporting implementation. Do not merge to `main` or push
without separate user authorization.

Before further work:

1. Run `git branch --show-current`, `git status --short`, and
   `git --no-pager log -5 --oneline`. Expect `reporting-metrics` and a clean
   worktree at this handoff; preserve any changes made since then.
2. Read `CONTEXT.md`, `docs/developer/plans/user-documentation.md`,
   `docs/developer/plans/2026-10-06-reporting-metrics-tdd.md`, and
   `docs/developer/plans/2026-10-07-documentation-and-reporting-sequence.md`.
3. Confirm commit history. `main`,
   `docs-reporting-s5`, and `docs-reporting-r0` intentionally remain at
   `b05b326`; previous documentation branches remain unchanged. Local `main`
   is already ten commits ahead of `origin/main` from earlier approved work;
   do not push it.
4. Review R1–R10 completion notes, including the R4 correction, in the reporting plan.
5. S8 / R10 is complete and committed. S9 / D6 cleanup is separate and
   non-blocking; obtain explicit user permission before each deletion.
   S10 / D7 publishing remains gated on a matching package release. Do not
   begin optional R11.

## Next work and authorization gates

The next sequence step is **S9 / D6: review cleanup candidates**, not
automatic deletion. Inspect each named output candidate in the
user-documentation plan, check tracked status and fixture/documentation
references, and determine whether it is reproducible or deliberately curated.
Review developer records individually for unique decisions and provenance.
Present specific candidates with evidence and any preserved-content
destination; obtain explicit permission before each deletion. Approval to
commit R10 does not authorize cleanup. Preserve protected case studies,
research data/notebooks, and both scientific PDFs.

S9 is non-blocking: retain files if deletion is declined. **S10 / D7 remains
pending**, including developer authoring/release instructions and deployment
configuration. The existing `.github\workflows\docs.yml` only builds pull
requests; it does not deploy. Before publication, obtain release/integration
authorization, confirm a matching package is available on PyPI, and align
installation and example-download refs with that release. Current refs
(`docs-reporting-s3`, `docs-reporting-s5`, `reporting-metrics`) remain local
and their new download links are unavailable until published. Keep the
unreleased notice and deployment disabled until the release gate is met.
Deployment must use a version-bump commit tagged `v*`, not every main update;
verify the first tagged deployment before marking S10 complete.

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
Obtain user approval before deleting files during D6 or enabling publication
during D7.

Dense-unknown exact count windows cost more than the superseded R4
approximation. A 3,653-step, 95%-unknown record with N=30 took approximately
15 ms / 1.1 s for overlapping / exclusive `>= 2`, and 9.3 s / 1.5 s for
`= 2`. Details and limitations are recorded in the reporting plan.
