"""Column-inventory diagnostics for the TARE pipeline.

Purpose
-------
Answer two questions before we touch the export module:
  1. What columns exist in the *initial loaded* frames (raw EUSS + baseline)?
  2. What columns exist in the *final exported/loaded* frames the analysis uses?

and, as a bonus for scoping the two Tepper CSVs, show which columns survive
the ``df_enduse_compare`` rename/whitelist step and roughly what theme each
column belongs to.

This module deliberately imports nothing from ``cmu_tare_model`` so it can be
pasted into the notebook (to inspect live in-memory frames) or run standalone
against exported CSVs on disk. Pure pandas, ASCII only.

Typical notebook use
--------------------
    from inventory_tare_columns import inventory_many, diff_stages, write_report

    frames = {
        # initial loaded (raw ResStock CSV, indexed by bldg_id, after filters)
        "raw_euss_mpX":        df_euss_am_mpX,
        # after df_enduse_refactored (baseline home, renamed/whitelisted)
        "baseline_home":       df_euss_am_baseline_home,
        # after df_enduse_compare (post-retrofit home frame, renamed)
        "mpX_home_renamed":    df_euss_am_mpX_home,
        # final loaded-for-analysis household frame
        "loaded_baseline":     df_outputs_baseline_home,
        "loaded_mp3_fixed":    DATAFRAMES_BY_MP[3]["fixed_base"],
        "loaded_mp4_fixed":    DATAFRAMES_BY_MP[4]["fixed_base"],
    }
    inv = inventory_many(frames)                 # tidy long table
    diff_stages(inv, "raw_euss_mpX", "mpX_home_renamed")   # what the rename drops
    write_report(inv, "tare_column_inventory")   # -> .csv + .txt to paste back

Disk-only use (no notebook needed)
----------------------------------
    from inventory_tare_columns import inventory_csv_headers
    inv = inventory_csv_headers("/path/to/output_folder")
"""

from __future__ import annotations

import os
import glob
from typing import Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Heuristic theme tagging
# ---------------------------------------------------------------------------
# These substrings are matched against the CURRENT run's naming (2025 Reference
# Case, six-case NPV scheme). Order matters: the first match wins, so the more
# specific families (npv/adopter) are checked before the generic ones. This is
# a convenience for scoping the Tepper subset, not an authoritative schema.
_THEME_RULES: List[tuple[str, tuple[str, ...]]] = [
    ("identifier",        ("bldg_id", "county_fips", "county_and_puma", "puma",
                           "county", "state", "gisjoin", "weight")),
    ("geo_climate",       ("census_region", "census_division", "climate_zone",
                           "gea_region", "reeds_balancing", "latitude",
                           "longitude", "weather_file", "city", "urbanicity")),
    ("building",          ("square_footage", "vintage", "building_type",
                           "occupancy", "occupants", "tenure", "vacancy",
                           "income", "federal_poverty")),
    ("equipment",         ("_type", "_efficiency", "heating_fuel", "cooling_fuel",
                           "waterheating_fuel", "hvac_", "size_", "seer", "hspf",
                           "has_ducts", "insulation", "capacity")),
    # adopter and npv must come before the broad "fuel_cost"/"consumption" tags
    ("adopter_flag",      ("econ_adopter",)),
    ("npv_case",          ("private_npv", "climate_npv", "total_npv",
                           "heatinglcc", "coolinglcc", "heatingsavings",
                           "coolingsavings")),
    ("rebate_policy",     ("rebate",)),
    ("capital_cost",      ("capital_cost", "installed_cost", "replacement",
                           "installation_premium", "net_capital", "total_capital")),
    ("fuel_cost",         ("lifetime_fuel_cost", "savings_fuel_cost",
                           "fuelcost", "fuel_cost", "operating")),
    ("consumption",       ("consumption",)),
    ("emissions_damages", ("mt_co2e", "damages_climate", "damages_health",
                           "avoided", "lrmer", "srmer")),
]


def categorize_column(col: str) -> str:
    """Assign a heuristic theme to a single column name.

    The match is case-insensitive and substring-based. Themes exist only to
    make the inventory scannable when scoping the Tepper subset; a column that
    lands in "other" is a signal to inspect it by hand, not an error.

    Args:
        col: The column name to classify.

    Returns:
        The name of the first matching theme, or "other" if nothing matches.
    """
    lowered = col.lower()
    for theme, needles in _THEME_RULES:
        if any(needle in lowered for needle in needles):
            return theme
    return "other"


# ---------------------------------------------------------------------------
# Core inventory
# ---------------------------------------------------------------------------
def inventory_columns(
    df: pd.DataFrame,
    stage_label: str,
    include_index: bool = True,
    print_summary: bool = True,
) -> pd.DataFrame:
    """Inventory one DataFrame's columns into a tidy long table.

    Records the index separately from the columns because the TARE frames are
    indexed by ``bldg_id`` -- the household join key -- and it is easy to forget
    that it is not a regular column. Any later merge for the Tepper export keys
    on this index, so it must be visible in the inventory.

    Args:
        df: The DataFrame to inspect.
        stage_label: A short name for this pipeline stage (e.g. "raw_euss_mpX").
        include_index: If True, add one row describing the index as role="index".
        print_summary: If True, print a compact shape/index/theme summary.

    Returns:
        A tidy DataFrame with one row per column, holding: stage, role, column,
        theme, dtype, n_non_null, pct_non_null, n_rows.

    Raises:
        TypeError: If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"'{stage_label}' is not a DataFrame (got {type(df)!r})")

    n_rows = len(df)
    records: List[Dict[str, object]] = []

    if include_index:
        # A named index (bldg_id) is the join key; record it explicitly.
        idx_name = df.index.name if df.index.name is not None else "<unnamed_index>"
        records.append({
            "stage": stage_label,
            "role": "index",
            "column": idx_name,
            "theme": categorize_column(idx_name),
            "dtype": str(df.index.dtype),
            "n_non_null": int(df.index.notna().sum()),
            "pct_non_null": round(100.0 * df.index.notna().mean(), 1) if n_rows else 0.0,
            "n_rows": n_rows,
        })

    for col in df.columns:
        non_null = int(df[col].notna().sum())
        records.append({
            "stage": stage_label,
            "role": "column",
            "column": col,
            "theme": categorize_column(col),
            "dtype": str(df[col].dtype),
            "n_non_null": non_null,
            "pct_non_null": round(100.0 * non_null / n_rows, 1) if n_rows else 0.0,
            "n_rows": n_rows,
        })

    inv = pd.DataFrame.from_records(records)

    if print_summary:
        theme_counts = (
            inv.loc[inv["role"] == "column", "theme"]
            .value_counts()
            .to_dict()
        )
        print(
            f"[{stage_label}] rows={n_rows:,} | "
            f"columns={df.shape[1]:,} | index='{df.index.name}'"
        )
        print(f"    themes: {theme_counts}")

    return inv


def inventory_many(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inventory several stages and stack the results.

    Args:
        frames: Mapping of stage_label -> DataFrame. Insertion order is
            preserved so upstream stages appear before downstream ones.

    Returns:
        A single tidy DataFrame concatenating every stage's inventory.
    """
    parts = [inventory_columns(df, label) for label, df in frames.items()]
    return pd.concat(parts, ignore_index=True)


def inventory_csv_headers(
    directory: str,
    pattern: str = "**/*.csv",
    index_col_name: str = "bldg_id",
) -> pd.DataFrame:
    """Inventory exported CSV headers from disk without loading full files.

    Reads only the header row (``nrows=0``) of each CSV, so it is cheap even
    for the full national export. Use this when you would rather not re-run the
    notebook to inspect the final exported schema.

    Args:
        directory: Root folder to search (e.g. the model's output_folder_path).
        pattern: Recursive glob relative to directory. Defaults to all CSVs.
        index_col_name: Name of the id column written as the CSV index; recorded
            as role="index" when found in the header.

    Returns:
        A tidy DataFrame with one row per (file, column): stage (the file's
        path relative to directory), role, column, theme. Dtypes are omitted
        because only the header is read.
    """
    root = os.path.abspath(directory)
    files = sorted(glob.glob(os.path.join(root, pattern), recursive=True))
    if not files:
        print(f"WARNING: no CSVs matched {pattern!r} under {root}")
        return pd.DataFrame(
            columns=["stage", "role", "column", "theme"]
        )

    records: List[Dict[str, object]] = []
    for path in files:
        rel = os.path.relpath(path, root)
        header = pd.read_csv(path, nrows=0)
        cols = list(header.columns)
        # A CSV written with a named index shows that name as the first column.
        for pos, col in enumerate(cols):
            role = "index" if (pos == 0 and col == index_col_name) else "column"
            records.append({
                "stage": rel,
                "role": role,
                "column": col,
                "theme": categorize_column(col),
            })
        print(f"[{rel}] columns={len(cols):,}")

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Stage-to-stage diff -- "did the column carry through?"
# ---------------------------------------------------------------------------
def diff_stages(
    inv: pd.DataFrame,
    stage_a: str,
    stage_b: str,
    print_result: bool = True,
) -> Dict[str, List[str]]:
    """Compare two stages' column sets to see what carried, dropped, or was added.

    This is the direct answer to "are the columns carried from the start of the
    pipeline?" Point it at an upstream stage (stage_a) and a downstream stage
    (stage_b): "dropped" is what the rename/whitelist step discarded, "added"
    is what that step introduced, "carried" survived unchanged by name.

    Note this compares column *names*. A variable that was renamed (not dropped)
    shows up as one entry in "dropped" (old name) and one in "added" (new name);
    that is exactly the case the rename-map audit needs to reconcile by hand.

    Args:
        inv: A tidy inventory (from inventory_many) containing both stages.
        stage_a: Upstream stage label.
        stage_b: Downstream stage label.
        print_result: If True, print counts and previews.

    Returns:
        Dict with keys "carried", "dropped", "added", each a sorted name list.

    Raises:
        ValueError: If either stage label is absent from inv.
    """
    present = set(inv["stage"].unique())
    for stage in (stage_a, stage_b):
        if stage not in present:
            raise ValueError(f"stage {stage!r} not in inventory (have: {sorted(present)})")

    cols_a = set(inv.loc[inv["stage"] == stage_a, "column"])
    cols_b = set(inv.loc[inv["stage"] == stage_b, "column"])

    result = {
        "carried": sorted(cols_a & cols_b),
        "dropped": sorted(cols_a - cols_b),
        "added": sorted(cols_b - cols_a),
    }

    if print_result:
        print(f"\n{stage_a} -> {stage_b}")
        print(f"  carried: {len(result['carried']):,}")
        print(f"  dropped: {len(result['dropped']):,}  (renamed-away or discarded)")
        print(f"  added:   {len(result['added']):,}  (renamed-to or computed)")
        for key in ("dropped", "added"):
            preview = result[key][:12]
            if preview:
                print(f"    {key} sample: {preview}")

    return result


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------
def write_report(inv: pd.DataFrame, out_stem: str) -> Dict[str, str]:
    """Write the inventory as a machine-readable CSV and a readable TXT.

    The CSV is for us to diff programmatically later; the TXT is a clean,
    paste-friendly listing grouped by stage and theme.

    Args:
        inv: A tidy inventory (from inventory_many or inventory_csv_headers).
        out_stem: Path stem without extension; ".csv" and ".txt" are appended.

    Returns:
        Dict with the two written paths, keyed "csv" and "txt".
    """
    csv_path = f"{out_stem}.csv"
    txt_path = f"{out_stem}.txt"
    inv.to_csv(csv_path, index=False)

    has_dtype = "dtype" in inv.columns
    lines: List[str] = []
    for stage in inv["stage"].drop_duplicates():
        block = inv[inv["stage"] == stage]
        lines.append("=" * 90)
        lines.append(f"STAGE: {stage}  ({len(block):,} entries)")
        lines.append("=" * 90)
        for theme in sorted(block["theme"].unique()):
            names = block.loc[block["theme"] == theme]
            lines.append(f"  [{theme}] ({len(names)})")
            for _, row in names.iterrows():
                suffix = f"  <{row['dtype']}>" if has_dtype else ""
                marker = "  (index)" if row["role"] == "index" else ""
                lines.append(f"      {row['column']}{suffix}{marker}")
        lines.append("")

    with open(txt_path, "w", encoding="ascii", errors="replace") as fh:
        fh.write("\n".join(lines))

    print(f"Wrote {csv_path} and {txt_path}")
    return {"csv": csv_path, "txt": txt_path}


if __name__ == "__main__":
    # Standalone smoke test on a tiny synthetic frame so the module can be run
    # directly (python inventory_tare_columns.py) without the notebook.
    demo = pd.DataFrame(
        {
            "county": ["G4200030"],
            "square_footage": [1800],
            "ref2025_mp3_heating_lifetime_fuel_cost": [1234.5],
            "ref2025_mp3_heatingLCC_coolingLCC_sub_private_npv_fixed_base": [-500.0],
            "ref2025_mp3_heatingLCC_coolingLCC_sub_econ_adopter_fixed_base": [0.0],
        }
    )
    demo.index.name = "bldg_id"
    inv_demo = inventory_columns(demo, "demo")
    write_report(inv_demo, "inventory_demo")
