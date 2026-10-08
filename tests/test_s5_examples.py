"""CLI acceptance checks for the focused S5 example packs."""

import shutil
import tomllib
from fractions import Fraction
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from hydropattern.cli import app
from hydropattern.parsers import parse_request

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SHEETS = {
    "timing_335-336": "timing_335_336",
    "duration_2-3": "duration_2_3",
    "frequency_ge1in5(union)": "frequency_ge1in5_union",
    "frequency_ge1in5(exclusive)": "frequency_ge1in5_exclusive",
}
PACK_OUTCOMES = {
    "first-run": {
        ("flow", "sustained_flow"): {
            "magnitude_gt1": [0, 1, 1, 0, 1, 0, 0, 1],
            "duration_ge2": [0, 1, 1, 0, 0, 0, 0, 0],
            "sustained_flow": [0, 1, 1, 0, 0, 0, 0, 0],
        },
    },
    "seasonal-thresholds": {
        ("flow", "early_december"): {
            "timing_335-336": [0, 0, 1, 1, 0, 0],
            "magnitude_gt1": [1, 1, 1, 0, 1, 1],
            "early_december": [0, 0, 1, 0, 0, 0],
        },
    },
    "duration": {
        ("flow", "long_flow"): {
            "magnitude_gt0": [1, 1, 1, 1, 0, 1, 1, 0, 0],
            "duration_ge3": [1, 1, 1, 1, 0, 0, 0, 0, 0],
            "long_flow": [1, 1, 1, 1, 0, 0, 0, 0, 0],
        },
        ("flow", "bounded_flow"): {
            "magnitude_gt0": [1, 1, 1, 1, 0, 1, 1, 0, 0],
            "duration_2-3": [0, 0, 0, 0, 0, 1, 1, 0, 0],
            "bounded_flow": [0, 0, 0, 0, 0, 1, 1, 0, 0],
        },
    },
    "frequency": {
        ("flow", "overlapping"): {
            "magnitude_gt0": [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            "frequency_ge1in5(union)": [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            "overlapping": [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        },
        ("flow", "exclusive"): {
            "magnitude_gt0": [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            "frequency_ge1in5(exclusive)": [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            "exclusive": [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        },
    },
    "multiple-scenarios": {
        ("low_flow", "sustained_flow"): {
            "magnitude_gt1": [0, 0, 0, 0, 0, 0],
            "duration_ge2": [0, 0, 0, 0, 0, 0],
            "sustained_flow": [0, 0, 0, 0, 0, 0],
        },
        ("high_flow", "sustained_flow"): {
            "magnitude_gt1": [0, 1, 1, 0, 1, 1],
            "duration_ge2": [0, 1, 1, 0, 1, 1],
            "sustained_flow": [0, 1, 1, 0, 1, 1],
        },
    },
}


@pytest.mark.parametrize("pack", PACK_OUTCOMES)
def test_example_pack_cli_matches_published_outcomes(pack, tmp_path, monkeypatch):
    source = ROOT / "examples" / pack
    for name in ("flow.csv", f"{pack}.toml"):
        shutil.copyfile(source / name, tmp_path / name)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["run", f"{pack}.toml", "--no-excel"])

    assert result.exit_code == 0, result.output
    output = tmp_path / f"{pack}_output"
    expected_files = set()
    instructions = (source / "README.md").read_text(encoding="utf-8")
    page = (ROOT / "docs" / "user" / "examples" / "index.md").read_text(encoding="utf-8")
    assert f"hydropattern run {pack}.toml --no-excel" in instructions
    for (scenario, component), columns in PACK_OUTCOMES[pack].items():
        raw_name = f"{scenario}_{component}.csv"
        summary_name = f"{component}_summary.xlsx"
        expected_files.update((raw_name, summary_name))
        outcomes = pd.read_csv(output / raw_name)
        assert outcomes.columns.tolist() == ["time", scenario, "dowy", *columns]
        summaries = pd.read_excel(output / summary_name, sheet_name=None, index_col=0)
        assert set(summaries) == {SUMMARY_SHEETS.get(column, column) for column in columns}
        for column, expected in columns.items():
            assert outcomes[column].tolist() == expected
            portion = sum(expected) / len(expected)
            summary = summaries[SUMMARY_SHEETS.get(column, column)]
            assert summary.index.tolist() == [
                "total", int(outcomes["time"].iloc[0][:4]),
            ]
            assert summary[scenario].tolist() == pytest.approx([portion, portion])
        assert f"`{columns[component]}`" in instructions
        row = next(
            line for line in page.splitlines()
            if f"`{scenario}` / `{component}` | `{columns[component]}` |" in line
        )
        published_portion = row.split("|")[-2].strip().split("=")[0].strip()
        assert float(Fraction(published_portion)) == pytest.approx(
            sum(columns[component]) / len(columns[component])
        )
    assert {path.name for path in output.iterdir()} == expected_files


def test_detailed_ordered_configuration_preserves_component_specs():
    config = tomllib.loads((ROOT / "examples" / "detailed.toml").read_text(encoding="utf-8"))
    components = config["components"]
    assert all("characteristics" in component for component in components.values())
    original = {
        "november_pulse_flow": {
            "timing": [305, 335],
            "magnitude": [">", 1.0],
            "rate_of_change": [">", 2.0, 1],
            "success_pattern": True,
        },
        "dry_season_baseflow": {
            "timing": [152, 305],
            "magnitude": ["<", 1.0],
            "duration": [">", 7],
            "success_pattern": False,
        },
        "bankfull_flow": {"magnitude": [">", 2.0], "frequency": [">=", 1, 5]},
    }
    with pytest.warns(UserWarning, match="compact characteristic-key form") as warnings:
        original_request = parse_request(original)
    assert len(warnings) == 3
    assert parse_request(components) == original_request
    assert config["timeseries"] == {
        "path": "examples/single_timeseries.csv", "date_format": "%Y-%m-%d",
    }


def test_detailed_configuration_cli_works_from_repository_root(tmp_path, monkeypatch):
    examples = tmp_path / "examples"
    examples.mkdir()
    for name in ("detailed.toml", "single_timeseries.csv"):
        shutil.copyfile(ROOT / "examples" / name, examples / name)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["run", str(Path("examples") / "detailed.toml"), "--no-excel"])

    assert result.exit_code == 0, result.output
    assert {path.name for path in (examples / "detailed_output").iterdir()} == {
        f"{prefix}{component}{suffix}"
        for component in ("november_pulse_flow", "dry_season_baseflow", "bankfull_flow")
        for prefix, suffix in (("flow_", ".csv"), ("", "_summary.xlsx"))
    }


@pytest.mark.parametrize("pack", PACK_OUTCOMES)
def test_pack_download_links_identify_authoritative_files(pack):
    page = (ROOT / "docs" / "user" / "examples" / "index.md").read_text(encoding="utf-8")
    ref = "docs-reporting-s3" if pack == "first-run" else "docs-reporting-s5"
    for name in ("flow.csv", f"{pack}.toml"):
        url = (
            "https://raw.githubusercontent.com/JohnRushKucharski/hydropattern/"
            f"{ref}/examples/{pack}/{name}"
        )
        assert f"]({url})" in page
        assert (ROOT / "examples" / pack / name).is_file()
