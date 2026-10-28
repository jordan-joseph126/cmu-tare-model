"""Compares two TARE runs' output files and logs.

Step 2.5.2 of the port protocol. Each output CSV of the "before" run is paired
with the "after" run's file of the same name (only the run stamp differs). The
copies in tepper_export/source_data/ carry no stamp and are not compared.

A pair is first compared byte for byte. Only if the bytes differ are the two
tables loaded and compared cell by cell, with blank equal to blank and numbers
equal within TOLERANCE. The county Tepper file is matched county by county.

Usage:
    python scripts/compare_runs.py --before 2026-10-05_21-33 --after 2026-10-06_20-15
    python scripts/compare_runs.py --before STAMP --after STAMP \
        --before-log logs/a.log --after-log logs/b.log
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# The runs' output folder in this repo, found from this script's own location
# (as in run_tare_notebooks.py), so it works on any computer.
OUTPUT_ROOT = (
    Path(__file__).resolve().parent.parent / "cmu_tare_model" / "output_results")
STAMP_PATTERN = re.compile(r"20\d{2}-\d{2}-\d{2}_\d{2}-\d{2}")
# Two numbers this close count as equal. Repeating a run on the same inputs can
# change the last digits of a float (sums added in another order), about 1e-16
# of the value, so 1e-11 leaves a wide margin and still catches a real change.
# Relative as well as absolute: dollar and GWh totals are too large for an
# absolute 1e-11 alone.
TOLERANCE = 1e-11
# The county Tepper file has one row per county, but counties with equal
# values can be written in either order, so it is compared county by county.
COUNTY_FILE_PREFIX = "tepper_county_"
COUNTY_KEY = "county"


def find_run_files(output_root: Path, stamp: str) -> Dict[str, Path]:
    """Finds one run's output files.

    Args:
        output_root: The output_results folder.
        stamp: The run stamp in the file names.

    Returns:
        Each file's path below output_root with the stamp replaced by
        '<stamp>', mapped to the file itself.
    """
    run_files = {}
    for path in sorted(output_root.rglob(f"*{stamp}*")):
        if not path.is_file() or "source_data" in path.parts:
            continue
        relative = path.relative_to(output_root).as_posix()
        run_files[relative.replace(stamp, "<stamp>")] = path
    return run_files


def file_sha256(path: Path) -> str:
    """Returns the SHA-256 of a file, read in pieces."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for piece in iter(lambda: handle.read(1 << 22), b""):
            digest.update(piece)
    return digest.hexdigest()


def compare_tables(
    before_path: Path, after_path: Path, key: Optional[str] = None
) -> List[str]:
    """Compares two CSV files cell by cell.

    Args:
        before_path: The "before" run's file.
        after_path: The "after" run's file.
        key: A column naming each row (such as 'county'). When given, rows are
            matched by this column instead of by position.

    Returns:
        One line per difference found; empty when rows, columns and values
        are all the same (blank equal to blank, numbers within TOLERANCE).
    """
    before = pd.read_csv(before_path, low_memory=False)
    after = pd.read_csv(after_path, low_memory=False)
    differences = []
    if before.shape != after.shape:
        differences.append(f"shape {before.shape} --> {after.shape}")
    if key is not None:
        for label, table in (("before", before), ("after", after)):
            if key not in table.columns:
                return differences + [f"no '{key}' column in the {label} file"]
        only_before_keys = sorted(set(before[key]) - set(after[key]))
        only_after_keys = sorted(set(after[key]) - set(before[key]))
        if only_before_keys:
            differences.append(
                f"{key} only before ({len(only_before_keys)}): {only_before_keys[:8]}")
        if only_after_keys:
            differences.append(
                f"{key} only after ({len(only_after_keys)}): {only_after_keys[:8]}")
        # Keep the rows both runs have, in the same order, so the cell
        # comparison below matches each county with itself.
        shared_keys = set(before[key]) & set(after[key])
        before = (before[before[key].isin(shared_keys)]
                  .sort_values(key, kind="mergesort").reset_index(drop=True))
        after = (after[after[key].isin(shared_keys)]
                 .sort_values(key, kind="mergesort").reset_index(drop=True))
    only_before = [column for column in before.columns if column not in after.columns]
    only_after = [column for column in after.columns if column not in before.columns]
    if only_before:
        differences.append(f"columns only before ({len(only_before)}): {only_before[:8]}")
    if only_after:
        differences.append(f"columns only after ({len(only_after)}): {only_after[:8]}")
    shared = [column for column in before.columns if column in after.columns]
    if list(before.columns) != list(after.columns) and not (only_before or only_after):
        differences.append("same columns in a different order")
    if len(before) != len(after):
        return differences

    for column in shared:
        before_values = before[column]
        after_values = after[column]
        numeric = (pd.api.types.is_numeric_dtype(before_values)
                   and pd.api.types.is_numeric_dtype(after_values))
        if numeric:
            close = np.isclose(
                before_values.astype(float), after_values.astype(float),
                rtol=TOLERANCE, atol=TOLERANCE, equal_nan=True)
            differs = pd.Series(~close, index=before_values.index)
        else:
            both_blank = before_values.isna() & after_values.isna()
            differs = ~((before_values == after_values) | both_blank)
        n_differs = int(differs.sum())
        if n_differs == 0:
            continue
        detail = f"column {column}: {n_differs:,} of {len(before):,} cells differ"
        if numeric:
            gap = (after_values[differs].astype(float)
                   - before_values[differs].astype(float)).abs()
            detail += f", largest gap {np.nanmax(gap.to_numpy()):.6g}"
        first_row = int(np.flatnonzero(differs.to_numpy())[0])
        detail += (f"; first at row {first_row}: {before_values.iloc[first_row]!r}"
                   f" --> {after_values.iloc[first_row]!r}")
        differences.append(detail)
    return differences


def marked_lines(log_path: Path) -> List[str]:
    """The [OK], [PASS], [FAIL] and [WARN] lines of a log, with stamps masked.

    Runner timing lines are left out; they differ on every run.
    """
    kept = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("[runner]"):
            continue
        if any(mark in line for mark in ("[OK]", "[PASS]", "[FAIL]", "[WARN]")):
            kept.append(STAMP_PATTERN.sub("<stamp>", line.rstrip()))
    return kept


def main(argv: Optional[List[str]] = None) -> int:
    """Compares the two runs and prints a report. Returns 0 when identical."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True, help="Run stamp of the first run")
    parser.add_argument("--after", required=True, help="Run stamp of the second run")
    parser.add_argument("--before-log", default=None)
    parser.add_argument("--after-log", default=None)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    options = parser.parse_args(argv)

    output_root = Path(options.output_root)
    before_files = find_run_files(output_root, options.before)
    after_files = find_run_files(output_root, options.after)
    print(f"before {options.before}: {len(before_files)} files")
    print(f"after  {options.after}: {len(after_files)} files")

    n_problems = 0
    for name in sorted(set(before_files) - set(after_files)):
        print(f"  [ONLY BEFORE] {name}")
        n_problems += 1
    for name in sorted(set(after_files) - set(before_files)):
        print(f"  [ONLY AFTER]  {name}")
        n_problems += 1

    for name in sorted(set(before_files) & set(after_files)):
        before_path = before_files[name]
        after_path = after_files[name]
        if file_sha256(before_path) == file_sha256(after_path):
            size_mb = before_path.stat().st_size / 1e6
            print(f"  [SAME BYTES]  {name} ({size_mb:,.1f} MB)")
            continue
        if before_path.suffix.lower() != ".csv":
            print(f"  [DIFFERENT]   {name}: bytes differ (not a CSV, not opened)")
            n_problems += 1
            continue
        key = COUNTY_KEY if before_path.name.startswith(COUNTY_FILE_PREFIX) else None
        differences = compare_tables(before_path, after_path, key)
        if not differences:
            print(f"  [SAME VALUES] {name}: bytes differ, every cell equal "
                  f"(numbers within {TOLERANCE:g})")
            continue
        n_problems += 1
        print(f"  [DIFFERENT]   {name}: {len(differences)} difference(s)")
        for difference in differences[:25]:
            print(f"      {difference}")
        if len(differences) > 25:
            print(f"      ... and {len(differences) - 25} more")

    if options.before_log and options.after_log:
        before_lines = marked_lines(Path(options.before_log))
        after_lines = marked_lines(Path(options.after_log))
        print(f"log lines marked [OK]/[PASS]/[FAIL]/[WARN]: "
              f"{len(before_lines)} before, {len(after_lines)} after")
        if before_lines == after_lines:
            print("  [SAME] the marked lines are equal, in order")
        else:
            n_problems += 1
            import difflib
            for line in difflib.unified_diff(
                    before_lines, after_lines, "before", "after", lineterm="", n=0):
                print(f"      {line}")

    print(f"RESULT: {'identical' if n_problems == 0 else f'{n_problems} difference(s)'}")
    return 0 if n_problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
