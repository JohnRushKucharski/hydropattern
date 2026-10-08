# Documentation and reporting work sequence

**Status:** S1a–S3 complete and integrated into local `main` (not pushed).
S3 commit `079d381` was fast-forwarded from `docs-reporting-s3`; keep that branch
available because current source-install and download links reference it.
S4's unaffected D4 portion is complete on `docs-reporting-s4` and integrated
into local `main`, without pushing. Do not repeat S1a–S4. S1a (`b2d612e`), S1b
(`f5c93b7`), and S1c are complete; do not repeat them.
S5's unaffected D5 portion is complete and integrated into local `main` at
`b05b326`, without pushing. S6 / R0 baseline was captured while working on
`docs-reporting-r0`; its evidence is committed in the reporting plan on
`reporting-metrics`. `docs-reporting-r0` remains at `b05b326` as the clean
pre-R0 starting point. R0–R10 are complete and committed on `reporting-metrics`,
based on `b05b326`; S8 and all D4/D5 topics are complete.
Reporting-branch changes have not been merged to `main` or pushed.
Next is S9 review; S10 remains release-gated. No cleanup deletion,
deployment, or optional R11 work has been authorized.
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
| S9 | D6 cleanup: **pending review**, per-file evidence and explicit permission before any deletion | S2 (not blocking) | User approval for each deletion; retaining files is acceptable |
| S10 | D7 publish: **pending**, GitHub Pages deploy triggered by a version-bump commit tagged `v*`; developer authoring and release checklist | S8; matching package release on PyPI before enabling deployment | Separate integration/publication authorization; matching install/download refs; first tagged deploy succeeds; unreleased notice kept until matching package release exists |

R11 is optional and is not part of this sequence.

S5 example-pack curation is complete in its separately approved scope; the
first-run pack remains unchanged. S6 reporting baseline was captured while
working on `docs-reporting-r0`; its evidence is committed on
`reporting-metrics`. R1–R9 implementation and documentation are complete and
committed there. R10 documentation is complete and committed in
`docs: complete R10 and prepare documentation handoff`; reporting changes
have not been merged to `main` or pushed.

## Pickup after S8 (2026-10-09)

Continue on local `reporting-metrics`; verify branch, clean handoff worktree,
and latest commit before editing. Review the named cleanup candidates and
historical records under D6, then request per-file deletion approval. Do not
infer deletion permission from approval to implement or commit R10.
S9 may be deferred or declined without blocking S10.

The current documentation workflow builds PRs only. S10 must include authoring
and release instructions, matching-version installation/download refs, and
verification of the first tagged deployment. Existing source/download refs
remain local and unpublished; do not promise remote downloads work yet.
Keep deployment disabled and the unreleased notice until a matching package
release exists. Do not merge, push, select/publish a release, or enable Pages
without separate user authorization. R10 release readiness does not mean a
package or documentation release has happened.

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
