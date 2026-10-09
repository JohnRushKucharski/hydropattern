"""Checks for the user-site foundation and its first-run example."""

import re
import shutil
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pandas as pd
import pytest
from typer.testing import CliRunner

from hydropattern.cli import app

ROOT = Path(__file__).resolve().parents[1]
USER_DOCS = ROOT / "docs" / "user"

# These words have valid uses outside the senses prohibited by CONTEXT.md.
# Qualified terms are removed before scanning; scientific prose still needs review.
CONTEXT_DEPENDENT_TERMS = {
    "Run", "x", "y", "metric", "value",
    "failure", "zero", "component outcome", "summary metric", "success fraction",
    "complete year", "event", "period", "calendar year", "frequency window",
    "duration", "run length", "success period", "intra-annual",
}


def _discouraged_terms() -> list[str]:
    context = (ROOT / "CONTEXT.md").read_text(encoding="utf-8")
    avoided = []
    for entry in re.findall(r"_Avoid_: (.*?)(?:\.\s*$)", context, re.MULTILINE):
        entry = re.sub(r"\([^)]*\)", "", entry)
        avoided.extend(part.strip() for part in re.split(r"[,;]", entry) if part.strip())
    return avoided


def _prose(markdown: str) -> str:
    text = re.sub(r"^```.*?^```\s*$", "", markdown, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    return text.replace("\n", " ")


def _avoided_terms(markdown: str) -> list[str]:
    avoided = _discouraged_terms()
    text = _prose(markdown)
    return [
        term for term in avoided
        if term not in CONTEXT_DEPENDENT_TERMS
        and re.search(rf"\b{re.escape(term)}s?\b", text, flags=re.IGNORECASE)
    ]


@pytest.mark.parametrize("path", sorted(USER_DOCS.rglob("*.md")), ids=lambda p: p.name)
def test_user_prose_avoids_unambiguous_discouraged_domain_terms(path):
    assert not _avoided_terms(path.read_text(encoding="utf-8")), str(path.relative_to(ROOT))


@pytest.mark.parametrize(
    "phrase",
    ["timestep frequency", "base pattern", "outer pattern", "event window",
     "successful year", "qualifying run duration", "elapsed-duration tracking"],
)
def test_terminology_scan_detects_discouraged_phrases(phrase):
    assert phrase in _avoided_terms(f"This explanation uses {phrase}.")


def test_terminology_scan_allows_code_and_canonical_qualified_terms():
    assert not _avoided_terms(
        "A component event occupies a component success period.\n"
        "`exclusive_event_window` is a former Python argument.\n"
        "```python\nspec.exclusive_event_window\n```\n"
    )


def test_first_run_documented_toml_matches_download():
    markdown = (USER_DOCS / "getting-started" / "first-run.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```toml\n(.*?)```", markdown, re.DOTALL)
    assert len(blocks) == 1
    download = ROOT / "examples" / "first-run" / "first-run.toml"
    assert tomllib.loads(blocks[0]) == tomllib.loads(download.read_text(encoding="utf-8"))


def test_first_run_pair_works_from_download_folder(tmp_path, monkeypatch):
    example = ROOT / "examples" / "first-run"
    for name in ("flow.csv", "first-run.toml"):
        shutil.copyfile(example / name, tmp_path / name)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["run", "first-run.toml", "--no-excel"])

    assert result.exit_code == 0, result.output
    output = tmp_path / "first-run_output"
    assert {path.name for path in output.iterdir()} == {
        "flow_sustained_flow.csv", "sustained_flow_summary.xlsx",
    }
    outcomes = pd.read_csv(output / "flow_sustained_flow.csv")
    assert outcomes.columns.tolist() == [
        "time", "flow", "dowy", "magnitude_gt1", "duration_ge2", "sustained_flow",
    ]
    assert outcomes["magnitude_gt1"].tolist() == [0, 1, 1, 0, 1, 0, 0, 1]
    assert outcomes["duration_ge2"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    assert outcomes["sustained_flow"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    page = (USER_DOCS / "getting-started" / "results.md").read_text(encoding="utf-8")
    rows = re.findall(
        r"^\| (\d{4}-\d{2}-\d{2}) \| (\d+) \| ([01]) \| ([01]) \| ([01]) \|$",
        page, re.MULTILINE,
    )
    published = pd.DataFrame(rows, columns=[
        "time", "flow", "magnitude_gt1", "duration_ge2", "sustained_flow",
    ])
    published = published.astype({column: int for column in published if column != "time"})
    pd.testing.assert_frame_equal(
        published, outcomes.drop(columns="dowy"), check_dtype=False,
    )
    summaries = pd.read_excel(output / "sustained_flow_summary.xlsx", sheet_name=None, index_col=0)
    assert set(summaries) == {
        "magnitude_gt1", "duration_ge2", "sustained_flow", "reporting_details",
    }
    for name, expected in {"magnitude_gt1": 0.5, "duration_ge2": 0.25,
                           "sustained_flow": 0.25}.items():
        assert summaries[name].index.tolist() == ["total", 2020]
        assert summaries[name]["flow"].tolist() == [expected, expected]


def test_site_foundation_uses_only_user_sources_and_material_search():
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert re.search(r"^docs_dir: docs/user$", config, re.MULTILINE)
    assert re.search(r"^  name: material$", config, re.MULTILINE)
    assert re.search(r"^plugins:\n  - search$", config, re.MULTILINE)
    pages = re.findall(r": ([\w/-]+\.md)$", config, re.MULTILINE)
    assert pages
    assert set(pages) == {
        path.relative_to(USER_DOCS).as_posix() for path in USER_DOCS.rglob("*.md")
    }
    assert not any(path.suffix.lower() in {".pdf", ".csv", ".xlsx", ".ipynb"}
                   for path in USER_DOCS.rglob("*"))
    groups = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert len(groups["dependency-groups"]["docs"]) == 1
    assert groups["dependency-groups"]["docs"][0].startswith("mkdocs-material")


def test_release_notices_are_cleared_from_published_user_content():
    paths = [
        ROOT / "README.md",
        *sorted(USER_DOCS.rglob("*.md")),
    ]
    notice = re.compile(r"not yet released|until v0\.3\.0|become available after", re.I)

    assert not [
        str(path.relative_to(ROOT))
        for path in paths
        if notice.search(path.read_text(encoding="utf-8"))
    ]


def test_pypi_readme_matches_the_current_package_release():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = project["project"]["version"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    installation = (USER_DOCS / "getting-started" / "installation.md").read_text(
        encoding="utf-8"
    )

    assert "not yet released" not in readme.lower()
    assert f"## Install v{version}" in readme
    assert f"hydropattern=={version}" in readme
    assert f"/v{version}/examples/first-run/" in readme
    assert f"hydropattern=={version}" in installation


def test_docs_workflow_keeps_pull_request_build_strict_and_read_only():
    workflow = (ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
    assert "pull_request:" in workflow
    assert "uv run --locked --group docs mkdocs build --strict" in workflow
    assert "contents: read" in workflow
    assert "push:" not in workflow


def test_docs_deployment_requires_manual_release_and_pypi_checks():
    workflow = (ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "release_tag:" in workflow
    assert "if: github.event_name == 'workflow_dispatch'" in workflow
    assert "git merge-base --is-ancestor" in workflow
    assert "gh release view" in workflow
    assert "https://pypi.org/pypi/hydropattern/" in workflow
    assert "release notice is cleared" in workflow
    assert "pages: write" in workflow
    assert "id-token: write" in workflow
    assert "enablement: false" in workflow
    assert "actions/deploy-pages@v4" in workflow
    assert "push:" not in workflow


def test_pypi_publisher_validates_version_and_tests_before_building():
    workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8"
    )
    assert "types: [published]" in workflow
    assert 'test "$RELEASE_TAG" = "v$(uv version --short)"' in workflow
    assert "uv sync --locked --group test" in workflow
    assert "uv run --locked pytest -q" in workflow
    assert "uv build --out-dir dist" in workflow
    assert "id-token: write" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "workflow_dispatch" not in workflow


@pytest.mark.parametrize("path", sorted(USER_DOCS.rglob("*.md")), ids=lambda p: p.name)
def test_user_markdown_file_links_resolve_without_copying_developer_records(path):
    markdown = path.read_text(encoding="utf-8")
    for link in re.findall(r"\[[^]]*\]\(([^)\s]+)\)", markdown):
        parsed = urlsplit(link)
        if parsed.scheme or not parsed.path:
            continue
        target = (path.parent / unquote(parsed.path)).resolve()
        assert target.is_relative_to(USER_DOCS), f"{path.name}: {link}"
        assert target.is_file(), f"{path.name}: {link}"
