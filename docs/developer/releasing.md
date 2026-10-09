# Release and documentation maintenance

This guide covers the CLI/package release and matching user-site publication.
The package release workflow publishes to PyPI from a published GitHub Release.
The documentation deployment is a separate, manually gated step so the site
cannot claim a release exists before PyPI confirms it.

## Authoring and local checks

Edit user pages under `docs/user`; update explicit navigation in `mkdocs.yml`
when adding or moving pages. Keep developer records outside the site source.
Preview locally and build strictly with:

```console
uv run --locked --group docs mkdocs serve
uv run --locked --group docs mkdocs build --strict
```

Before a release, run the suite and build checks from the repository root:

```console
uv sync --locked --group test --group dev --group docs
uv run --locked pytest -q
uv run --locked mypy hydropattern/
uv run --locked --group docs mkdocs build --strict
uv build --out-dir dist
```

Run Ruff on changed Python files; unrelated existing Ruff findings elsewhere
in the repository are not part of this release-prep change.

Check `git status` and inspect the complete diff before committing. Keep
`pyproject.toml`, `uv.lock`, the release tag, installation commands, and
download refs on the same version. The corrective package version is
`0.3.1`; update project versions with `uv version <version>` so the lockfile
stays synchronized.

PyPI's project description comes from the README embedded in uploaded
distribution metadata. Existing release files cannot be replaced to change
that description. If a published description is stale, update the README and
publish a new patch version. The README/version regression test and inspection
of both built distribution metadata catch version drift before upload.

## Publishing a package

1. Finish work on the release branch, run the checks above, and open a pull
   request to `main`. Merge only after CI and strict documentation builds pass.
2. Push the merged `main` commit. Confirm its `pyproject.toml` version is
   `0.3.1`, its lockfile is current, and its tested source includes all
   documented behavior.
3. Create and publish GitHub Release `v0.3.1` from that commit. The
   `Publish to PyPI` workflow verifies that tag matches the project version,
   reruns pytest, builds distributions, and publishes through the configured
   PyPI trusted publisher. A mismatch or failed test blocks publication.
4. Confirm the GitHub release uses tag `v0.3.1`, points to the intended
   `main` commit, and its `Publish to PyPI` workflow run succeeded.
5. Verify PyPI reports version `0.3.1` and provides distributions:
   <https://pypi.org/pypi/hydropattern/0.3.1/json>. Install that exact version
   in a clean Python 3.12 environment and smoke-test `hydropattern --help`
   before changing any unreleased notices.

Do not treat a GitHub Release or a successful workflow dispatch alone as proof
that package publication completed. The PyPI version endpoint and a clean
installation are the release checks.

## Publishing documentation

Only proceed after the exact package version is visible on PyPI and its clean
installation smoke test passes.

1. Remove the unreleased notices and pre-publication instructions from
   `README.md` and affected `docs/user` pages. Keep the installation and
   example-download versions pinned to `0.3.1`.
2. Run the checks above, especially pytest's executable documentation/link
   checks and the strict MkDocs build. Commit and merge this release-status
   update to `main`.
3. In GitHub repository settings, set **Pages → Build and deployment → Source**
   to **GitHub Actions**. Do this only after PyPI verification.
4. Run **Actions → Documentation → Run workflow** on `main`, entering
   `v0.3.1` as `release_tag`. The workflow checks that the tag exists as a
   published GitHub Release, the tag is included in current `main`, the source
   version matches the tag, PyPI has the matching distributions, and
   unreleased notices are gone. It then builds strictly and deploys.
5. Confirm the deployment workflow succeeds and the published
   `https://johnrushkucharski.github.io/hydropattern/` site loads. Check
   installation guidance, first-run downloads, and example links against the
   release tag. Resolve and redeploy if any check fails.

The workflow has no `push` or tag-push deployment trigger: publishing a
version tag happens before PyPI is available, and the tagged documentation
source must still carry its unreleased notice at that point. Manual dispatch
after package and documentation status verification avoids publishing stale
release messaging or an unverified package.
