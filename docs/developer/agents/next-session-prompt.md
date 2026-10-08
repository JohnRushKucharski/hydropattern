# Next-session handoff

Continue hydropattern reporting work from branch `reporting-metrics`.
R1–R5 are complete and committed. This branch is based on local `main` at
`b05b326`. The working tree should be clean. Do not merge to `main` or push.

Before further implementation:

1. Read `CONTEXT.md`, `docs/developer/plans/2026-10-06-reporting-metrics-tdd.md`,
   and `docs/developer/plans/2026-10-07-documentation-and-reporting-sequence.md`.
2. Confirm current branch, clean status, and commit history. `main`,
   `docs-reporting-s5`, and `docs-reporting-r0` intentionally remain at
   `b05b326`; previous documentation branches remain unchanged. Local `main`
   is already ten commits ahead of `origin/main` from earlier approved work;
   do not push it.
3. Review R1–R5 completion notes, including the R4 correction, in the reporting plan.
4. Report understanding and identify R6 as next. Wait for user approval before
   starting R6. Do not begin R7–R10 or optional R11.

R1 preserves unavailable order-1 comparison inputs as `NaN`. R2 propagates
configured water-year boundaries to `Result`, uses shared ending-year labels,
supports optional February 29 in daily completeness, and exposes cadence-
validated daily/monthly observed exposure. `Result.event_rate()` retains its
old exposure behavior until R7. R3 evaluates all possible duration run
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
Summary denominators and reciprocal-mode removal are R6 work, not implemented.

Last verified before handoff: `uv run pytest -q` (760 passed),
`uv run mypy hydropattern/`, `uv run mkdocs build --strict`, and
`git diff --check` all passed. Changes are local and not pushed.

Dense-unknown exact count windows cost more than the superseded R4
approximation. A 3,653-step, 95%-unknown record with N=30 took approximately
15 ms / 1.1 s for overlapping / exclusive `>= 2`, and 9.3 s / 1.5 s for
`= 2`. Details and limitations are recorded in the reporting plan.
