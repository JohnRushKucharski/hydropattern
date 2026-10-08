# Next-session handoff

Continue hydropattern reporting work from branch `docs-reporting-r1-r2`.
R1 and R2 are complete and committed; this branch is based on local `main`
at `b05b326`. The working tree should be clean. Do not merge to `main` or push.

Before any implementation:

1. Read `CONTEXT.md`, `docs/developer/plans/2026-10-06-reporting-metrics-tdd.md`,
   and `docs/developer/plans/2026-10-07-documentation-and-reporting-sequence.md`.
2. Confirm current branch, clean status, and commit history. `main`,
   `docs-reporting-s5`, and `docs-reporting-r0` intentionally remain at
   `b05b326`; previous documentation branches remain unchanged.
3. Review R1/R2 completion notes in the reporting plan and run focused tests if
   state differs from this handoff.
4. Report understanding and identify R3 as next. Wait for user approval before
   starting R3. Do not begin R4–R10 or optional R11.

R1 preserves unavailable order-1 comparison inputs as `NaN`. R2 propagates
configured water-year boundaries to `Result`, uses shared ending-year labels,
supports optional February 29 in daily completeness, and exposes cadence-
validated daily/monthly observed exposure. `Result.event_rate()` retains its
old exposure behavior until R7.

Last verified before handoff: `uv run pytest -q` (706 passed),
`uv run mypy hydropattern/`, `uv run mkdocs build --strict`, and
`git diff --check` all passed. Changes were not pushed.
