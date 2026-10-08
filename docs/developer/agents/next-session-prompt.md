# Next-session handoff

Continue hydropattern reporting work from branch `reporting-metrics`.
R1–R7 are complete and committed. This branch is based on local `main` at
`b05b326`. The working tree should be clean. Do not merge to `main` or push.

Before further implementation:

1. Read `CONTEXT.md`, `docs/developer/plans/user-documentation.md`,
   `docs/developer/plans/2026-10-06-reporting-metrics-tdd.md`, and
   `docs/developer/plans/2026-10-07-documentation-and-reporting-sequence.md`.
2. Confirm current branch, clean status, and commit history. `main`,
   `docs-reporting-s5`, and `docs-reporting-r0` intentionally remain at
   `b05b326`; previous documentation branches remain unchanged. Local `main`
   is already ten commits ahead of `origin/main` from earlier approved work;
   do not push it.
3. Review R1–R7 completion notes, including the R4 correction, in the reporting plan.
4. Report understanding and identify R8 as next. Wait for user approval before
   starting R8. Do not begin R9–R10 or optional R11.

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

R8 adds a reporting-details sheet while preserving existing metric sheets and
raw timestep files. Capture successful/known/total counts, known-outcome
coverage, completeness, component event-count/rate bounds, observed exposure,
and availability reasons. Test scenario/column alignment, partial-year rows,
Excel sheet-name collisions, CSV/Excel behavior, and Python/CLI consistency.
Sequence plan identifies `docs/user/guide/outputs.md` as its user-doc target.

Last verified before handoff: `uv run pytest -q` (771 passed),
`uv run mypy hydropattern/`, `uv run mkdocs build --strict`, and
`git diff --check` all passed. R7 commit is local and not pushed.

Dense-unknown exact count windows cost more than the superseded R4
approximation. A 3,653-step, 95%-unknown record with N=30 took approximately
15 ms / 1.1 s for overlapping / exclusive `>= 2`, and 9.3 s / 1.5 s for
`= 2`. Details and limitations are recorded in the reporting plan.
