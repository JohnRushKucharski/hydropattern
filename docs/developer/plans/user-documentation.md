# User documentation redesign

Status: D1–D3 complete (S1a–S3 in the sequence); D4–D7 remain pending.
S3 is implemented on local `docs-reporting-s3`, based on local `main` at
`7227686`; it has not been merged or pushed.

Related prerequisite: the
[reporting and unknown-outcome TDD plan](2026-10-06-reporting-metrics-tdd.md)
now defines additional scientific and reporting changes for the next release.
Keep those changes separate from editorial implementation, but coordinate the
user-facing explanations and worked examples with this redesign. In particular,
do not publish the former reciprocal mode, denominator policies, or failure-pattern
color reversal as the new behavior; document the implementation actually released.
The reporting plan requires a dedicated unknown-outcome section and characteristic,
event-count, water-year, and plotting examples, not merely developer records.

Work order across both plans is defined in the
[documentation and reporting sequence](2026-10-07-documentation-and-reporting-sequence.md).
Phases below are labelled D1–D7 there.

## Goal and audience

Provide clear, searchable documentation for hydrologists and environmental
scientists using the CLI, with little assumed terminal or TOML experience.
Keep Python API documentation available but secondary.

Use complete, professional sentences. Retain necessary scientific terminology,
define it before use, and replace engineering language such as "golden arrays",
"contracts", and "adapter layer" with explanations relevant to users.
Documentation must explain actual behavior, not merely improve existing prose.

## Non-negotiable boundaries

- Preserve existing modified and untracked work. The pattern-correctness plan
  records documentation changes awaiting review; integrate them, do not reset
  or silently replace them.
- Do not edit, move, or delete these case-study directories:
  `examples\Data_comparing`, `examples\frio`, `examples\longtailpoint`,
  `examples\Luvuvhu`. Do not include them in the documentation site.
- Retain research datasets and notebooks outside the published site.
- Keep both scientific PDFs tracked in GitHub; exclude them from the generated
  site. Cite the original papers through DOI links.
- Prefer retaining and reorganizing files over deleting them.
- Before every deletion, review the specific file and obtain explicit user
  permission. Consolidating its contents elsewhere is not deletion permission.
  Treat Git renames as reorganization, preserving contents and local edits.
- Do not change scientific algorithms as part of editorial work. Discuss any
  suspected scientific issue with the user before planning a behavior change.
  The ordered-table field rename below is an explicitly agreed schema change.

## Publishing and maintenance

Use MkDocs with the open-source Material theme and built-in search. Publish on
GitHub Pages using the default `github.io` address and standard GitHub-hosted
Actions runners. The public-repository setup requires no paid service or theme
features; normal GitHub Pages usage limits still apply.

Keep the authoring system close to ordinary Markdown:

- One `mkdocs.yml`, explicit navigation, one documentation workflow.
- A separate `docs` dependency group, with `mkdocs-material` as the only new
  directly requested package; its dependencies provide MkDocs and extensions.
- No additional plugins, external search service, Docker requirement, custom
  frontend, generated API documentation, or multi-version tooling.
- Local preview: `uv run --group docs mkdocs serve`.
- Pull requests build the site with strict validation. Deployment through
  GitHub Pages is enabled only after reporting slice R10 and is triggered by a
  version-bump commit tagged `v*`, not by every main-branch update.
- Set the documentation source directory to `docs\user`. Developer records,
  PDFs, and protected case studies must not be copied into the site.

Authors edit Markdown and update navigation when adding or moving pages.
Put short authoring and release instructions in developer documentation, not
the user navigation.

## Version and installation policy

Document current code and the next release. Display a prominent **unreleased**
notice until a matching package release exists.

At design time, PyPI and the latest GitHub release provide v0.2.0, whose frequency
behavior differs from current code. Do not pair current examples with an
unqualified instruction to install that release.

After the matching release, lead CLI installation with `uv tool install
hydropattern`. Explain Python 3.12+ requirements and provide platform-specific
setup steps. PyPI is the package source; uv is the installer, so publishing to
PyPI does not change this recommendation.

During the unreleased transition, provide an explicit current-source installation
route. Keep pip/virtual-environment installation as an alternative, and explain
Python API installation separately. Ordinary users should not install test or
development dependency groups.

The first-run guide provides a small downloadable CSV/TOML pair, a command, and
an explanation of the resulting files. No full-repository clone is needed just
to obtain these example files. Relative data paths currently resolve from the
terminal's working directory, not the TOML file's directory; instructions must
make the working folder explicit.

## Information architecture

Separate user documentation from engineering records:

```text
README.md
CONTEXT.md
mkdocs.yml
docs\
  user\                 # Website source; user material only
    index.md
    getting-started\
    guide\
    concepts\
    reference.md        # Reference entry point
    reference\
      characteristics\
    examples\
    troubleshooting\
    api\
  developer\            # Not published in user site
    adr\
    agents\
    plans\
    review\
    references\         # Retained scientific PDFs
examples\               # Runnable packs and protected case studies
```

Exact topic filenames can follow existing conventions; avoid empty placeholder
pages. Move existing engineering documents, including the GUI handoff, into
the developer area. Update repository instructions, links, relevant source
comments/error guidance, and repository skill references that use old paths.
Preserve links to retained decisions and historical rationale.

Website navigation:

| Section | Contents |
| --- | --- |
| Start here | Installation, first run, interpreting first results |
| User guide | Preparing data, configuring components, CLI usage, outputs, plotting |
| Scientific concepts | Practical foundations, plain-language glossary, unknown outcomes, evaluation order and interpretation |
| Reference | Exact TOML fields, characteristic pages, CLI options, technical evaluation rules |
| Examples | Task-labelled runnable examples with expected results |
| Troubleshooting | Common problems, corrective actions, upgrade guidance |
| Python API | Secondary usage and result interpretation |

Keep scientific foundations concise rather than writing a literature review.
Use Poff and Yarnell citations for deeper theory. Explain that example thresholds
illustrate software behavior, not universal ecological criteria.

## README and reference roles

Make README a practical front door, roughly 80-120 lines if content permits:
purpose, inputs/results, version status, documentation link, installation,
first-run downloads and command, short result interpretation, brief scientific
context/citations, and license.

Move exhaustive CLI options, detailed parameter tables, Python examples, long
theory, and migration discussion to the appropriate pages. Avoid maintaining
two copies of the same manual.

Split the current combined reference by topic. Preserve its existing entry path
as a useful reference index where practical; update incoming links and anchors
deliberately rather than leaving broken links.

Provide one page per characteristic:

1. Plain-language purpose and concrete hydrologic example.
2. Worked timestep table showing preceding conditions, characteristic outcome,
   and final component outcome.
3. TOML syntax, parameters, defaults, and units.
4. Evaluation rules and edge cases.
5. Equations and technical detail when useful, later on the same page.

The concepts section owns the shared model; characteristic pages own their
detailed semantics. Frequency has separate subsections for un-nested frequency
and nested intra-annual/interannual evaluation. Link directly to sections instead of
duplicating their explanations.

Troubleshooting should lead with symptoms and remedies. Retain searchable error
codes, but put Python error-envelope access and API internals in the API section.

## Agreed language and scientific interpretation

[`CONTEXT.md`](../../../CONTEXT.md) is the single source of truth for domain
terms; do not maintain a separate term table here. Use its preferred terms and
respect every _Avoid_ entry. Resolve any new terminology conflict by updating
`CONTEXT.md` with the user before writing documentation. The user site has a
plain-language glossary page that follows `CONTEXT.md`; `CONTEXT.md` itself is
not published.

Documentation-specific rules:

- A period denotes an interval, not a timestep count. Do not call sampling
  "timestep frequency" or use "duration" indiscriminately for run lengths.
- Name concrete units ("qualifying timesteps", "qualifying water years");
  use "trial" only when generalizing over both, without implying independence.
- Use intra-annual pattern, interannual pattern, and un-nested frequency;
  not base/nested or inner/outer.
- Calendar lengths require explicit units and boundary assumptions; do not
  infer equal day counts from monthly cadence. Add no calendar-length outputs
  other than the water-year exposure and event rates agreed in the reporting plan.
- Replace "classified coverage" with **fraction of known outcomes marked as
  component success** for `portion` (or the **number of successful timesteps**
  for a count), always alongside **known-outcome coverage**. Successful means
  marked as component success; do not imply ecological benefit. Summary
  denominators and unknown-outcome policies are defined by the reporting plan;
  document them in the slice that implements them.

### Frequency example and interpretation

Lead with an explanation along these lines before presenting syntax:

> The preceding characteristic conditions are met on Monday; for example, flow
> exceeds the magnitude threshold. This starts a five-timestep frequency window,
> Monday-Friday. The preceding conditions are not met Tuesday-Friday, but the
> window contains one qualifying timestep, meeting the frequency condition
> "at least one qualifying timestep". hydropattern therefore marks the component
> as successful throughout Monday-Friday. Five successful timesteps do not mean
> that the preceding conditions were met on five timesteps.

State the assumptions of this example and show its configuration/results.
Explain overlapping windows, `exclusive_windows`, zero-admitting comparisons, record-end
truncation, intra-annual/interannual evaluation, and water-year completeness in later sections.
Frequency counts qualifying timesteps or qualifying water years, not distinct
component events. A qualifying window and a component success period can differ
because overlapping windows can merge into a longer component-success run.

### Duration interpretation

Whole-run assessment is intended. A qualifying run of 10 timesteps passes a
`>= 7` duration threshold, so all 10 are marked successful. The same run fails
duration bounds `[3, 7]`, so none are marked successful. Entire qualifying runs
of length 3, 4, 5, 6, or 7 pass those bounds.

Do not describe this as testing only previous timesteps or accumulating success
from the seventh timestep onward. Qualifying run length and final
component-success run length can differ when further characteristics or
failure-pattern settings affect the component outcome.

### Other interpretation safeguards

Explain retrospective assessment, independent characteristic conditions versus
dependent duration/frequency assessment, and the meaning of `success_pattern`.
For failure-pattern configurations, distinguish non-failure from demonstrated
ecological success. `return_period` is removed in the next release; document
only its removal and migration, and never present `1 / portion` as a hydrologic
recurrence interval or guarantee of event independence.

Check timing dates/leap-year conventions, moving-average startup behavior,
rate-of-change denominator restrictions, missing values, annual completeness,
summary units, and result-column meaning against code before rewriting claims.

## Ordered-table schema change

Teach ordered characteristic lists as the default; retain compact syntax as a
supported secondary form and explain its order warning plainly.

Before converting documentation, make this agreed breaking change for the next
release:

- Ordered characteristic tables accept `type` and **`parameters`**, not `metrics`.
- Reject `metrics` with explicit replacement guidance; no compatibility alias.
- Reject tables containing both keys.
- Keep compact characteristic syntax and `[output.metric]` unchanged.
- Keep evaluation behavior unchanged; avoid unrelated internal/API renaming.
- Update migration notes with before/after examples and the affected version.

Use existing parser error conventions. Test every characteristic's ordered
form, missing/invalid parameters, legacy-key rejection, conflicting keys,
ordering, and equivalent evaluation outcomes. Preserve tests for compact syntax.

The S1b frequency-window API changes are complete: use `exclusive_windows` for
Python arguments and the un-nested or intra-annual specification field,
`interannual_exclusive_windows` for the interannual specification field, and
`mark_windows` for the run-marking helper. There are no compatibility aliases;
migration guidance is in `docs\user\migration.md`.

S1c is complete in its own TDD slice with no compatibility aliases. Fields
shared with un-nested frequency keep neutral names; fields specific to the
interannual part use `interannual_*`. The generic characteristic marker is
`is_terminal`. Helper names for the overall nested-frequency construct,
result-column names, and evaluation behavior are unchanged. See the
[migration guide](../../user/migration.md) for the Python API name changes.
This explicitly overrides "avoid unrelated internal/API renaming" for these
identifiers only.

## Examples and reviewed cleanup

Keep `examples\detailed.toml` comprehensive. Improve comments, terminology,
section organization, supported syntax, and links without turning it into a
replacement for the reference.

Add focused runnable packs for first run, seasonal thresholds, duration,
frequency, multiple scenarios, and response-surface plotting. Each pack includes
input data, configuration, purpose, command/working-folder instructions, and
expected results. Use small illustrative data rather than repurposing protected
case studies.

Keep one authoritative copy of runnable files under `examples`; user pages link
to matching-version GitHub downloads/source instead of duplicating files or
adding a build-time copying plugin.

Potential reproducible-output removals, subject to individual review and
explicit permission:

- `examples\minimal_output.xlsx`
- `examples\multi_timeseries_output.xlsx`
- `examples\multi_timeseries_output\0-0.csv`
- `examples\multi_timeseries_output\0-1.csv`
- `examples\multi_timeseries_output\1-0.csv`
- `examples\multi_timeseries_output\1-1.csv`

Check whether any file is a fixture or deliberately curated example before
proposing deletion. Prevent newly generated outputs from being accidentally
tracked without broadly ignoring case-study assets.

Review plans, reviews, behavior-change records, and handoffs individually.
Retain unique decisions, scientific rationale, and provenance before proposing
any consolidation/deletion. The default outcome is reorganization.

## Implementation sequence and completion criteria

Execute in the order given by the
[documentation and reporting sequence](2026-10-07-documentation-and-reporting-sequence.md);
phases 4 and 5 are split around reporting slices R0–R10 there.

1. **Schema prerequisite (D1):** add failing tests, implement `parameters`-only
   ordered tables and the `exclusive_windows` and intra-annual/interannual
   renames, retain evaluation behavior, and prepare migration guidance.
2. **Separate audiences:** move retained engineering records/PDFs, update
   references, preserve local changes, and leave protected case studies intact.
3. **Build the reading path:** add minimal MkDocs setup, navigation, version
   notice, revised README, installation and first-run/result guides.
4. **Rewrite scientific/reference content:** establish shared concepts, split
   characteristic/configuration/output/CLI/API topics, and verify every
   substantive behavior claim.
5. **Curate examples:** improve comprehensive `detailed.toml`, convert affected
   ordered examples, add focused packs and their expected-result explanations.
6. **Review cleanup:** present specific candidates and preserved-content
   destinations; ask permission before each deletion. Cleanup is not a
   prerequisite for publishing if approval is withheld.
7. **Publish and document maintenance:** strict PR builds, tag-triggered
   deployment after R10, developer authoring instructions, and matching-release checklist.

Validate with the smallest relevant existing pytest selectors, executable
documentation checks (a fixture for every worked example and a parse test for
every TOML block in `docs\user`), a pytest scan of `docs\user` for `CONTEXT.md`
_Avoid_ terms, runnable example checks, type checks for parser changes,
and strict MkDocs builds. Add no additional validation tools unless existing
checks prove insufficient.

Completion means:

- A new user can install the documented version, obtain the first-run pair,
  run it from the stated folder, locate results, and interpret them.
- Search/navigation finds each characteristic and major user task without
  reading unrelated topics.
- Exact examples and expected outcomes agree with current code; schema changes
  are tested without changing scientific calculations.
- Existing links are updated or deliberately retained as entry points.
- Built site contains only intended user material, not developer records,
  PDFs, or protected case-study assets.
- Protected directories have no changes; no unapproved deletions occurred.
- Release status is accurate; the unreleased notice is removed only after
  publishing a package that supports the documented behavior and syntax.

## D3 completion record

The user-site foundation now has explicit Material/MkDocs navigation, built-in
search, a `docs` dependency group, and a read-only pull-request strict-build
workflow. README is the shorter front door; installation, first evaluation,
result interpretation, glossary, CLI guidance, and the secondary API example
live under `docs\user`. The existing reference and migration pages remain
available, with their reference reorganization deferred to D4 and reporting
updates deferred to the implementing R-slices.

The single authoritative first-run pair is under `examples\first-run`.
Acceptance tests execute the pair from a separate working folder, compare the
published timestep table and summary rows with actual output, check its TOML
block against the download, execute the relocated Python example, and scan
user prose for unambiguous `CONTEXT.md` avoided terms. Context-dependent terms
still require editorial review; code identifiers and historical migration
syntax are excluded from the prose scan. A separate documentation editor
reviewed the draft, and its changes were inspected.

Completion checks passed: strict MkDocs build, all 642 pytest tests, linting
of the changed tests, built-site local links and anchors, site exclusions, and
preservation of scientific code, protected directories, and both tracked PDFs.

Local preview is `uv run --group docs mkdocs serve`; strict validation is
`uv run --group docs mkdocs build --strict`. Authors edit Markdown in
`docs\user` and update explicit navigation in `mkdocs.yml`. Developer records,
scientific PDFs, protected case studies, and research files stay outside the
site source.

Source-install and first-run download links currently share the
`docs-reporting-s3` ref. They are explicitly unavailable remotely until the
branch is pushed; local-checkout instructions work before publication. Keep
that branch available while those links use it, and update installation and
download references together when integrating subsequent work or preparing
the matching release. GitHub Pages deployment is not enabled; S10 will add it
after R10. No cleanup deletions or scientific changes were made in D3.
