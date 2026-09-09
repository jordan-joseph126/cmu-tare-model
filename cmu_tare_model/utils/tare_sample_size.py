"""Report how many homes survive each TARE sample filter, from raw EUSS data.

The TARE model starts from the ResStock/EUSS national sample and narrows it in
steps: occupied homes, then single-family homes, then homes whose heating fuel
the study prices, then homes whose heating technology appears in the cost
database. This module reads the raw EUSS baseline CSV and reports the home
count, the share of the stock, and the weighted number of real dwellings after
each step.

It exists so the counts quoted in documentation can be reproduced rather than
pasted, and so the same cascade can be run for any geography -- a county, a
state, or the whole country.

Reads only. Nothing here writes to the model output tree or changes any model
result.

Location: cmu_tare_model/utils/tare_sample_size.py
"""

import os
from typing import Optional, Tuple

import pandas as pd

from config import PROJECT_ROOT
from cmu_tare_model.constants import (
    ALLOWED_HOUSING_TYPES,
    ALLOWED_TECHNOLOGIES,
    FUEL_MAPPING,
    VERBOSE,
)

# Same location and vintage the rest of the model reads from. Mirrors
# EUSS_DATA_DIR in adoption_kpis/data_loading.py; defined again here so this
# module can be imported without pulling in the mapping dependencies.
EUSS_DATA_DIR: str = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "euss_data",
    "resstock_amy2018_release_1.1", "national", "csv"
)

BASELINE_FILENAME: str = "baseline_metadata_and_annual_results.csv"

# Only these columns are read. The baseline CSV is about 1.7 GB, so reading the
# whole thing to count rows would be slow and would not fit comfortably in
# memory.
REQUIRED_EUSS_COLUMNS = [
    "bldg_id",
    "in.county",
    "in.state",
    "in.vacancy_status",
    "in.geometry_building_type_recs",
    "in.heating_fuel",
    "in.hvac_heating_type_and_fuel",
    "in.hvac_cooling_type",
    "in.hvac_has_ducts",
    "weight",
]

# Rows read at a time. Large enough to be fast, small enough that one chunk of
# the baseline file stays well under a gigabyte.
DEFAULT_CHUNK_SIZE: int = 200_000


def load_raw_euss_baseline(
    scope_column: Optional[str] = None,
    scope_value: Optional[object] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    verbose: bool = VERBOSE,
) -> pd.DataFrame:
    """Read the raw EUSS baseline, optionally narrowed to one geography.

    No TARE filter is applied here. The frame returned is every sampled home
    in the requested geography, which is step 0 of the cascade.

    Args:
        scope_column: Raw EUSS column to filter on, either 'in.county' (a
            Census GISJOIN code such as 'G4200030') or 'in.state' (a two-letter
            abbreviation). None reads the whole national sample.
        scope_value: The value that column must equal. Ignored when
            scope_column is None.
        chunk_size: Rows read at a time.
        verbose: Whether to print progress.

    Returns:
        DataFrame indexed by bldg_id, holding only the columns the cascade
        needs.

    Raises:
        FileNotFoundError: If the baseline CSV is not where it is expected.
        ValueError: If scope_column is given but is not a supported column, or
            if the filter matches no homes.
    """
    filepath = os.path.join(EUSS_DATA_DIR, BASELINE_FILENAME)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"EUSS baseline file not found at {filepath}. This module reads "
            "the raw ResStock download, not the model's exported results.")

    if scope_column is not None and scope_column not in ("in.county", "in.state"):
        raise ValueError(
            f"scope_column must be 'in.county' or 'in.state', got "
            f"{scope_column!r}. Those are the two geography columns present in "
            "the raw EUSS baseline file.")

    if verbose:
        scope_label = "national" if scope_column is None else (
            f"{scope_column} = {scope_value!r}")
        print(f"Reading raw EUSS baseline ({scope_label}) from {filepath}")

    matching_chunks = []
    for chunk in pd.read_csv(filepath, usecols=REQUIRED_EUSS_COLUMNS,
                             chunksize=chunk_size, low_memory=False):
        if scope_column is None:
            matching_chunks.append(chunk)
        else:
            matching_chunks.append(chunk[chunk[scope_column] == scope_value])

    df_raw = pd.concat(matching_chunks).set_index("bldg_id")

    if len(df_raw) == 0:
        raise ValueError(
            f"No homes matched {scope_column} = {scope_value!r} in the raw "
            "EUSS baseline.")

    if verbose:
        print(f"  {len(df_raw):,} sampled homes before any TARE filter")

    return df_raw


def compute_sample_size_cascade(
    scope_column: Optional[str] = None,
    scope_value: Optional[object] = None,
    df_raw: Optional[pd.DataFrame] = None,
    verbose: bool = VERBOSE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Count homes after each TARE sample filter, in the order they apply.

    The four filters, in order:

    1. Occupied dwellings only.
    2. Single-family only (ALLOWED_HOUSING_TYPES).
    3. Heating fuel the study prices (the four fuels in FUEL_MAPPING).
    4. Heating technology present in the cost database
       (ALLOWED_TECHNOLOGIES['heating']).

    Steps 3 and 4 together are what the model records as include_heating, so
    the final row of the cascade is the number of homes the model evaluates.

    Cooling is deliberately not a step. It is reported separately because a
    home with no air conditioning still gets a heating result and is never
    removed from the sample.

    Args:
        scope_column: 'in.county', 'in.state', or None for national. Ignored
            when df_raw is supplied.
        scope_value: Value that column must equal. Ignored when df_raw is
            supplied.
        df_raw: An already-loaded raw frame from load_raw_euss_baseline. Pass
            this to run several cascades without re-reading the 1.7 GB file.
        verbose: Whether to print the tables as they are built.

    Returns:
        A tuple of two DataFrames:
        - cascade: one row per step, with columns 'step', 'homes',
          'pct_of_stock', 'weighted_homes', and 'removed'.
        - removed_detail: one row per reason a home was removed, with columns
          'step', 'reason', 'homes'.

    Raises:
        ValueError: If neither df_raw nor a readable scope is available.
    """
    if df_raw is None:
        df_raw = load_raw_euss_baseline(
            scope_column=scope_column, scope_value=scope_value, verbose=verbose)

    total_sampled = len(df_raw)
    cascade_rows = []
    removed_rows = []

    def add_cascade_row(step_label, df_step, removed_count):
        """Append one row of the cascade table."""
        cascade_rows.append({
            "step": step_label,
            "homes": len(df_step),
            "pct_of_stock": len(df_step) / total_sampled * 100.0,
            "weighted_homes": df_step["weight"].sum(),
            "removed": removed_count,
        })

    def add_removed_detail(step_label, removed_values):
        """Record why homes were removed at one step, counted by value."""
        counts = removed_values.value_counts(dropna=False)
        for reason, count in counts.items():
            removed_rows.append({
                "step": step_label,
                "reason": "(not recorded)" if pd.isna(reason) else reason,
                "homes": int(count),
            })

    # Step 0 -- everything in scope, before any TARE filter.
    add_cascade_row("0. Sampled homes, no filters", df_raw, 0)

    # Step 1 -- occupied dwellings only.
    is_occupied = df_raw["in.vacancy_status"] == "Occupied"
    df_occupied = df_raw[is_occupied]
    add_cascade_row("1. Occupied only", df_occupied,
                    len(df_raw) - len(df_occupied))
    add_removed_detail("1. Occupied only",
                       df_raw[~is_occupied]["in.vacancy_status"])

    # Step 2 -- single-family only.
    is_single_family = df_occupied["in.geometry_building_type_recs"].isin(
        ALLOWED_HOUSING_TYPES)
    df_single_family = df_occupied[is_single_family]
    add_cascade_row("2. Single-family only", df_single_family,
                    len(df_occupied) - len(df_single_family))
    add_removed_detail(
        "2. Single-family only",
        df_occupied[~is_single_family]["in.geometry_building_type_recs"])

    # Step 3 -- heating fuel the study prices.
    modeled_fuels = list(FUEL_MAPPING.keys())
    has_modeled_fuel = df_single_family["in.heating_fuel"].isin(modeled_fuels)
    df_fuel = df_single_family[has_modeled_fuel]
    add_cascade_row("3. Heating fuel in scope", df_fuel,
                    len(df_single_family) - len(df_fuel))
    add_removed_detail("3. Heating fuel in scope",
                       df_single_family[~has_modeled_fuel]["in.heating_fuel"])

    # Step 4 -- heating technology present in the cost database.
    has_modeled_tech = df_fuel["in.hvac_heating_type_and_fuel"].isin(
        ALLOWED_TECHNOLOGIES["heating"])
    df_heating = df_fuel[has_modeled_tech]
    add_cascade_row("4. Heating technology in scope (include_heating)",
                    df_heating, len(df_fuel) - len(df_heating))
    add_removed_detail(
        "4. Heating technology in scope (include_heating)",
        df_fuel[~has_modeled_tech]["in.hvac_heating_type_and_fuel"])

    df_cascade = pd.DataFrame(cascade_rows)
    df_removed = pd.DataFrame(removed_rows)

    if verbose:
        print_sample_size_cascade(df_cascade, df_removed)

    return df_cascade, df_removed


def summarize_cooling_scope(
    df_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Count cooling systems on the single-family frame.

    Cooling never removes a home from the sample. A home with no air
    conditioning still receives a heating result; its cooling savings are zero
    and its cooling columns are blank. This is reported so documentation can
    say how many homes have cooling in scope without implying it is a filter.

    Args:
        df_raw: Raw frame from load_raw_euss_baseline.

    Returns:
        DataFrame with columns 'cooling_system', 'homes', and 'in_scope'.
    """
    is_occupied = df_raw["in.vacancy_status"] == "Occupied"
    is_single_family = df_raw["in.geometry_building_type_recs"].isin(
        ALLOWED_HOUSING_TYPES)
    df_single_family = df_raw[is_occupied & is_single_family]

    counts = df_single_family["in.hvac_cooling_type"].value_counts(dropna=False)

    rows = []
    for cooling_system, count in counts.items():
        if pd.isna(cooling_system):
            label = "(no cooling recorded)"
            in_scope = False
        else:
            label = cooling_system
            in_scope = cooling_system in ALLOWED_TECHNOLOGIES["cooling"]
        rows.append({
            "cooling_system": label,
            "homes": int(count),
            "in_scope": in_scope,
        })

    return pd.DataFrame(rows)


def summarize_duct_scope(
    df_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Count ducted and non-ducted homes among those the model evaluates.

    TARE places no duct requirement on a home. This count exists for comparison
    with studies that do -- ResStock 2025 requires ducts for its dual-fuel
    package -- so documentation can say what a duct requirement would remove.

    Args:
        df_raw: Raw frame from load_raw_euss_baseline.

    Returns:
        DataFrame with columns 'has_ducts', 'homes', and 'pct_of_included'.
    """
    is_occupied = df_raw["in.vacancy_status"] == "Occupied"
    is_single_family = df_raw["in.geometry_building_type_recs"].isin(
        ALLOWED_HOUSING_TYPES)
    df_step = df_raw[is_occupied & is_single_family]

    has_modeled_fuel = df_step["in.heating_fuel"].isin(list(FUEL_MAPPING.keys()))
    df_step = df_step[has_modeled_fuel]

    has_modeled_tech = df_step["in.hvac_heating_type_and_fuel"].isin(
        ALLOWED_TECHNOLOGIES["heating"])
    df_included = df_step[has_modeled_tech]

    counts = df_included["in.hvac_has_ducts"].value_counts(dropna=False)

    rows = []
    for has_ducts, count in counts.items():
        rows.append({
            "has_ducts": "(not recorded)" if pd.isna(has_ducts) else has_ducts,
            "homes": int(count),
            "pct_of_included": int(count) / len(df_included) * 100.0,
        })

    return pd.DataFrame(rows)


def print_sample_size_cascade(
    df_cascade: pd.DataFrame,
    df_removed: Optional[pd.DataFrame] = None,
) -> None:
    """Print the cascade, and optionally the reasons homes were removed.

    Args:
        df_cascade: First frame returned by compute_sample_size_cascade.
        df_removed: Second frame returned by compute_sample_size_cascade.
            Omit it to print the cascade alone.
    """
    print("\n" + "=" * 88)
    print("TARE SAMPLE SIZE CASCADE")
    print("=" * 88)
    print(f"{'Step':<48}{'Homes':>10}{'% stock':>10}{'Weighted':>16}")
    print("-" * 88)
    for _, row in df_cascade.iterrows():
        print(f"{row['step']:<48}{row['homes']:>10,}"
              f"{row['pct_of_stock']:>9.2f}%{row['weighted_homes']:>16,.0f}")
    print("=" * 88)

    if df_removed is None or len(df_removed) == 0:
        return

    for step_label in df_removed["step"].unique():
        step_rows = df_removed[df_removed["step"] == step_label]
        total_removed = step_rows["homes"].sum()
        if total_removed == 0:
            continue
        print(f"\nRemoved at {step_label} ({total_removed:,} homes):")
        for _, row in step_rows.iterrows():
            print(f"    {row['reason']:<44}{row['homes']:>10,}")
