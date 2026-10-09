# Documentation and reporting work sequence

**Status:** S1a–S8 and R0–R10 are complete and merged to remote `main` by
PR #46 at `d82d2d3`. Their phase records below preserve the original branch
and handoff state. The v0.3.0 source-install/download refs point to tag
`v0.3.0` and remain unavailable until that tag is published.
S9 review is complete; reviewed artifacts were retained and no deletion was
authorized. S10 release preparation is in progress; package release,
deployment, and optional R11 have not been authorized.
See the [next-session handoff](../agents/next-session-prompt.md) for the pickup
checklist.

This plan defines the order in which agents pick up work from two plans:

- [User documentation redesign](user-documentation.md)
  (phases **D1–D7**)
- [Reporting metrics and unknown outcomes](2026-10-06-reporting-metrics-tdd.md)
  (slices **R0–R10**, optional R11)

It adds no scientific decisions. Those plans and [`CONTEXT.md`](../../../CONTEXT.md)
remain authoritative for their content; `CONTEXT.md` is the source of truth
for terminology. If a step here conflicts with either plan, stop and ask.

## Why interleave

Reporting slices change user-visible behavior: summary denominators,
`return_period` removal, plot colors and coverage cutoff, event-rate exposure,
annual probability, leap-day handling, and unknown outcomes. Documenting those
topics before the change would publish wrong behavior; documenting them only
afterwards would leave every R-slice without a page to update. Therefore the
site structure and unaffected content come first, each R-slice then updates
named pages, and remaining content is completed with R10.

## Order

Pick the first step whose prerequisites are done. Do not start a later step
early unless marked parallel.

| Step | Work | Prerequisites | Method and gate |
| --- | --- | --- | --- |
| S0 | Plan amendments (D0): update both plans and `CONTEXT.md` with agreed terms and this sequence | None | Editorial; user review |
| S1a | D1: ordered tables accept `parameters`, reject `metrics` | S0 | TDD; pytest + mypy; migration note; complete (`b2d612e`) |
| S1b | D1: rename `exclusive_event_window` → `exclusive_windows` everywhere (spec fields, function arguments, docstrings, tests, docs); interannual spec field `interannual_exclusive_windows`; rename `mark_events` to `mark_windows`; no aliases | S0 | TDD; targeted pytest + mypy; migration note; complete (`f5c93b7`) |
| S1c | D1: keep shared fields neutral; use `interannual_*` and `has_interannual_pattern` for the interannual part, and `is_terminal` for the characteristic marker. Preserve overall helper/result-column names and evaluation behavior. See the [migration guide](../../user/migration.md) for API mappings. **Complete.** | S1b | TDD; targeted pytest + mypy; migration note; no aliases |
| S2 | D2: organize retained engineering records under `docs\developer\` (ADRs, agent guidance, plans, reviews, handoff, and scientific PDFs); update repository-wide links and file references. **Complete.** | S1a–S1c | `git mv`; stale-reference search; pytest; protected case studies unchanged |
| S3 | D3: `mkdocs.yml`, `docs` dependency group, PR strict-build workflow (**no deploy**), version notice, README, installation, first run, glossary page, avoided-term pytest scan. **Complete (`079d381`), integrated into local `main`; not pushed.** See the [D3 completion record](user-documentation.md#d3-completion-record). | S2 | Strict build, pytest, local links/anchors, generated-site exclusions, and protected-path checks pass; documentation-editor review complete |
| S4 | D4 (unaffected part): practical foundations, evaluation order, characteristic references, configuration/CLI/API organization, and reference entry point. **Complete on `docs-reporting-s4`; integrated into local `main`, not pushed.** See the [S4 completion record](user-documentation.md#d4-s4-completion-record). | S3 | Executable fixture per worked example, written before prose; all user-site TOML blocks parsed; terminology/link/navigation checks, strict site build, and documentation-editor review pass |
| S5 | D5 (unaffected part): `detailed.toml` curation; packs for first run, seasonal thresholds, duration, frequency, multiple scenarios. **Complete and integrated into local `main` at `b05b326`; not pushed.** See the [S5 completion record](user-documentation.md#d5-s5-completion-record). | S4 (parallel with S4 allowed after S3) | Expected results first; parametrized CLI test per pack |
| S6 | R0 baseline; inventory documentation consumers against the S3–S5 pages. **Complete; captured while working on `docs-reporting-r0`, with evidence now committed in the reporting plan on `reporting-metrics`.** | S3 | Per reporting plan |
| S7 | R1–R9 in reporting-plan order (complete and committed); each slice updates its target pages below | S6 | Per reporting plan; strict docs build passes after each slice |
| S8 | R10 with remaining D4/D5: unknown-outcome section, deferred reference topics, response-surface pack, migration guide. **Complete and committed on `reporting-metrics`.** | S7 | Executable fixtures for worked examples; strict build passed. See the R10 completion record in the reporting plan. |
| S9 | D6 cleanup review: **complete**; per-file review recommended retaining all six generated-output candidates and historical records. No deletion was authorized or performed. | S2 (not blocking) | Review evidence reported to user; retaining files is acceptable |
| S10 | D7 publish: **in progress**, for package v0.3.0; validated GitHub Release triggers PyPI publish; manual GitHub Pages deployment follows package verification and release-notice update | S8; v0.3.0 package release on PyPI before Pages enablement/deployment | Matching install/download refs; successful package and site verification; unreleased notice retained until PyPI release exists |

R11 is optional and is not part of this sequence.

S5 example-pack curation is complete in its separately approved scope; the
first-run pack remains unchanged. S6 reporting baseline was captured while
working on `docs-reporting-r0`; its evidence is committed in the reporting
plan. R1–R9 implementation and documentation are complete. R10 documentation
is complete and was committed in `docs: complete R10 and prepare documentation
handoff`. All required work is integrated into `main` through PR #46.

## Status after S9 review (2026-10-09)

S9 review is complete. The six generated-output candidates and historical
developer records were reviewed; artifacts were retained, no deletion was
performed, and cleanup does not block S10.

Continue on local `reporting-metrics`; verify branch, worktree, and latest
commit before editing, preserving any intervening changes. The documentation
workflow strictly builds PRs and supports a manual deploy after PyPI confirms
the matching package and release notices have been removed. Installation and
download references target v0.3.0, but remote downloads remain unavailable
until its tag is published. Keep Pages disabled and the unreleased notice
until the package is available. The user authorized commit, merge, and push on
2026-10-09; package publication and Pages enablement remain gated. R10 release
readiness does not mean package or site publication has happened.

The package workflow validates the tag and runs tests before publishing.
PyPI's `hydropattern/0.3.0` endpoint returned HTTP 404 during preparation, so
the package has not yet been released. After publishing and verifying the
package, remove pre-release notices, merge that update, enable Pages, and run
the docs workflow with `release_tag=v0.3.0`. The workflow verifies the GitHub
Release, PyPI distributions, source version, and release notices before
deployment.

## R10 topics completed in S8

S8 completes remaining user-facing topics after R1–R9:

- Optional leap day, water-year completeness, and unsupported cadence.
- Dedicated unknown-outcome section and uncertainty examples.
- Response-surface example pack and worked plotting examples.

S4 may describe fully known behavior of duration and frequency (whole-run
assessment, overlapping and exclusive windows, record-end truncation) because
R3–R5 preserve it.

## Target pages for R-slices

Indicative paths under `docs\user\`; follow names chosen in S3.

| Slice | Pages to update |
| --- | --- |
| R1 | `concepts/unknown-outcomes`, `reference/characteristics/magnitude`, `.../rate-of-change`, component-combination concept |
| R2 | `concepts/water-years`, `guide/preparing-data` (leap day, cadence) |
| R3 | `reference/characteristics/duration`, `concepts/unknown-outcomes` |
| R4 | `reference/characteristics/frequency` (un-nested), `concepts/unknown-outcomes` |
| R5 | `reference/characteristics/frequency` (intra-annual/interannual) |
| R6 | `guide/outputs`, `reference/configuration` (`[output.metric]` modes), migration |
| R7 | `guide/outputs` (event statistics), `api/`, migration |
| R8 | `guide/outputs` (reporting-details sheet) |
| R9 | `guide/plotting`, `reference/configuration`, `reference/cli`, `api/`, migration |
| R10 | All above; completes unknown-outcome section and migration guide |

## Agent checklist per step

1. Read `CONTEXT.md` and the relevant plan section; use its terms.
2. Confirm prerequisites are done; mark the step in progress.
3. Follow the method column (TDD for code, fixtures for worked examples).
4. Run the smallest relevant checks, then the strict docs build once S3 exists.
5. Record completion in this table's source plan, not only here.
