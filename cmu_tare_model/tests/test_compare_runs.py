"""Tests for scripts/compare_runs.py, the tool that checks two runs match.

Sessions that add a switch rely on this tool to show that, with the switch
off, a run equals the baseline. These tests build tiny made-up runs in a
temporary folder, so no model output is needed.
"""

import importlib.util
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "compare_runs.py"

# scripts/ is not a package, so the script is loaded from its file.
_spec = importlib.util.spec_from_file_location("compare_runs", SCRIPT_PATH)
compare_runs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_runs)

BEFORE = "2026-01-01_00-00"
AFTER = "2026-01-02_00-00"


def write_run(output_root: Path, stamp: str, tables: Dict[str, pd.DataFrame]) -> None:
    """Writes one made-up run: each table as <name>_<stamp>.csv in a subfolder.

    Args:
        output_root: The temporary output_results folder.
        stamp: The run stamp to put in each file name.
        tables: File name (without stamp) mapped to its table.
    """
    folder = output_root / "results"
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(folder / f"{name}_{stamp}.csv", index=False)


def run_main(output_root: Path) -> int:
    """Runs the tool on the two made-up runs and returns its exit code."""
    return compare_runs.main(
        ["--before", BEFORE, "--after", AFTER, "--output-root", str(output_root)])


@pytest.fixture
def df_homes() -> pd.DataFrame:
    """A small home table with a number, a blank and a text column."""
    return pd.DataFrame({
        "bldg_id": [1, 2, 3],
        "npv": [1.0, 1_000_000.0, np.nan],
        "fuel": ["Natural Gas", "Electricity", None],
    })


@pytest.fixture
def df_county() -> pd.DataFrame:
    """A small county table; two counties share the same values."""
    return pd.DataFrame({
        "county": ["G0800130", "G0800310", "G0800690"],
        "adoption_rate_pct": [0.0, 0.0, 12.5],
    })


def test_default_output_root_is_in_this_repo():
    assert compare_runs.OUTPUT_ROOT == REPO_ROOT / "cmu_tare_model" / "output_results"


def test_identical_runs_return_zero(tmp_path, df_homes):
    write_run(tmp_path, BEFORE, {"homes": df_homes})
    write_run(tmp_path, AFTER, {"homes": df_homes})
    assert run_main(tmp_path) == 0


def test_tiny_gap_counts_as_equal(tmp_path, df_homes):
    # A gap far below the tolerance, both near 1 and near a million.
    df_after = df_homes.copy()
    df_after["npv"] = df_after["npv"] * (1 + 1e-13)
    write_run(tmp_path, BEFORE, {"homes": df_homes})
    write_run(tmp_path, AFTER, {"homes": df_after})
    assert run_main(tmp_path) == 0


def test_real_gap_is_reported(tmp_path, df_homes):
    df_after = df_homes.copy()
    df_after.loc[0, "npv"] += 1e-6
    write_run(tmp_path, BEFORE, {"homes": df_homes})
    write_run(tmp_path, AFTER, {"homes": df_after})
    differences = compare_runs.compare_tables(
        tmp_path / "results" / f"homes_{BEFORE}.csv",
        tmp_path / "results" / f"homes_{AFTER}.csv")
    assert len(differences) == 1
    assert differences[0].startswith("column npv: 1 of 3 cells differ")
    assert run_main(tmp_path) == 1


def test_blank_equals_blank_and_blank_differs_from_value(tmp_path, df_homes):
    df_after = df_homes.copy()
    df_after.loc[0, "npv"] = np.nan
    write_run(tmp_path, BEFORE, {"homes": df_homes})
    write_run(tmp_path, AFTER, {"homes": df_after})
    differences = compare_runs.compare_tables(
        tmp_path / "results" / f"homes_{BEFORE}.csv",
        tmp_path / "results" / f"homes_{AFTER}.csv")
    # Row 3 is blank in both runs and is not reported; row 1 is.
    assert len(differences) == 1
    assert "1 of 3 cells differ" in differences[0]


def test_swapped_county_rows_count_as_equal(tmp_path, df_county):
    df_swapped = df_county.iloc[[1, 0, 2]].reset_index(drop=True)
    write_run(tmp_path, BEFORE, {"tepper_county_mp3_CO": df_county})
    write_run(tmp_path, AFTER, {"tepper_county_mp3_CO": df_swapped})
    assert run_main(tmp_path) == 0


def test_swapped_rows_in_other_files_are_reported(tmp_path, df_county):
    # Only the county Tepper file is matched by county; elsewhere row order counts.
    df_swapped = df_county.iloc[[2, 0, 1]].reset_index(drop=True)
    write_run(tmp_path, BEFORE, {"county_summary": df_county})
    write_run(tmp_path, AFTER, {"county_summary": df_swapped})
    assert run_main(tmp_path) == 1


def test_county_in_one_run_only_is_reported(tmp_path, df_county):
    write_run(tmp_path, BEFORE, {"tepper_county_mp3_CO": df_county})
    write_run(tmp_path, AFTER, {"tepper_county_mp3_CO": df_county.iloc[:2]})
    differences = compare_runs.compare_tables(
        tmp_path / "results" / f"tepper_county_mp3_CO_{BEFORE}.csv",
        tmp_path / "results" / f"tepper_county_mp3_CO_{AFTER}.csv",
        key="county")
    assert any("county only before (1): ['G0800690']" in line for line in differences)
    assert run_main(tmp_path) == 1


def test_file_in_one_run_only_returns_one(tmp_path, df_homes):
    write_run(tmp_path, BEFORE, {"homes": df_homes, "extra": df_homes})
    write_run(tmp_path, AFTER, {"homes": df_homes})
    assert run_main(tmp_path) == 1
