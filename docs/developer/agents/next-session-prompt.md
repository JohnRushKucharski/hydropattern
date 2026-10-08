# Next-session handoff

Continue hydropattern reporting work from branch `docs-reporting-r1-r2`.
R1 and R2 are committed; R3 implementation and documentation are complete in
the current uncommitted worktree. This branch is based on local `main` at
`b05b326`. Expect a dirty worktree; do not discard changes, merge to `main`,
or push.

Before further implementation:

1. Read `CONTEXT.md`, `docs/developer/plans/2026-10-06-reporting-metrics-tdd.md`,
   and `docs/developer/plans/2026-10-07-documentation-and-reporting-sequence.md`.
2. Confirm current branch, status, and commit history. `main`,
   `docs-reporting-s5`, and `docs-reporting-r0` intentionally remain at
   `b05b326`; previous documentation branches remain unchanged. Local `main`
   is already ten commits ahead of `origin/main` from earlier approved work;
   do not push it.
3. Review uncommitted R3 changes and completion notes. R3 passed full pytest,
   mypy, strict MkDocs build, and `git diff --check`.
4. Report understanding and identify R4 as next. Wait for user approval before
   starting R4. Do not begin R5–R10 or optional R11.

R1 preserves unavailable order-1 comparison inputs as `NaN`. R2 propagates
configured water-year boundaries to `Result`, uses shared ending-year labels,
supports optional February 29 in daily completeness, and exposes cadence-
validated daily/monthly observed exposure. `Result.event_rate()` retains its
old exposure behavior until R7. R3 evaluates all possible duration run
boundaries: known verdicts remain known only when every possible run agrees;
definite preceding failures still settle failure.

Last verified before handoff: `uv run pytest -q` (713 passed),
`uv run mypy hydropattern/`, `uv run mkdocs build --strict`, and
`git diff --check` all passed. R3 changes are uncommitted and not pushed.
