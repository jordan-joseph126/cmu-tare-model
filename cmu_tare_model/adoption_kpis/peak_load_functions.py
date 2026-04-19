"""Reusable functions for post-TARE peak load analysis.

Extracted from calculate_postTARE_ts_aws_peak_demand.ipynb (Phase 2 BSQ refactor).
Used by the notebook and by the national loop (Step 9).

Author: Jordan M. Joseph, PhD — Carnegie Mellon University
"""

from typing import Any

import numpy as np
import pandas as pd


# Column name constants (must match BSQ output + notebook conventions)
BLDG_ID_COL: str = "bldg_id"
BSQ_ELEC_COL: str = "electricity.total.energy_consumption"


def gisjoin_to_fips(gisjoin: str) -> str:
    """Convert a GISJOIN county identifier to a 5-digit FIPS code.

    GISJOIN format: G + 2-digit state FIPS + 0 + 3-digit county FIPS.
    Example: 'G4200030' → '42003'.

    Args:
        gisjoin: GISJOIN string from the EUSS ``in.county`` column.

    Returns:
        5-digit county FIPS code as a string.

    Raises:
        ValueError: If *gisjoin* is shorter than 7 characters.
    """
    if len(gisjoin) < 7:
        raise ValueError(
            f"GISJOIN string too short ({len(gisjoin)} chars): '{gisjoin}'. "
            f"Expected format 'G##0###' (≥7 chars)."
        )
    return gisjoin[1:3] + gisjoin[4:7]


def find_adoption_column(
    df: pd.DataFrame,
    mp: int,
    cost_scenario: str,
    discount_rate_key: str = "fixed_base",
    rcm_model_key: str = "inmap",
) -> str:
    """Locate the adoption-tier column in a TARE output DataFrame.

    Builds the expected column name using ``create_adoption_col`` for the
    IRA-reference scenario with central SCC, InMAP, ACS, and the specified
    discount rate.  Falls back to a fuzzy search if the exact column is absent.

    Args:
        df: TARE output DataFrame (one row per building).
        mp: Measure-package number (e.g. 3 or 4).
        cost_scenario: REMDB cost scenario key (e.g. ``'v4MID'``).
        discount_rate_key: Discount rate variant key.
        rcm_model_key: Reduced-complexity model key.

    Returns:
        The matched column name string.

    Raises:
        KeyError: If no matching adoption column is found.
    """
    from cmu_tare_model.utils.column_names import create_adoption_col

    expected = create_adoption_col(
        scenario_prefix=f"iraRef_mp{mp}_",
        category="heating",
        column_type="adoption",
        cost_scenario=cost_scenario,
        method_suffix=f"_{discount_rate_key}",
        scc_assumption="central",
        rcm_model=rcm_model_key,
        cr_function="acs",
    )
    if expected in df.columns:
        return expected

    # Fuzzy fallback
    adoption_candidates = [c for c in df.columns if "adoption" in c.lower()]
    if adoption_candidates:
        raise KeyError(
            f"Expected adoption column '{expected}' not found.\n"
            f"  Candidates containing 'adoption' ({len(adoption_candidates)}):\n"
            + "\n".join(f"    • {c}" for c in adoption_candidates)
        )
    raise KeyError(
        f"Expected adoption column '{expected}' not found, "
        f"and no columns containing 'adoption' exist."
    )


def extract_adopter_ids(
    df_tare: pd.DataFrame,
    adoption_col: str,
    tier_1_value: str = "Tier 1: Feasible",
    tier_2_value: str = "Tier 2: Feasible vs. Alternative",
) -> dict[str, dict[str, list[int]]]:
    """Build per-county adopter ID dictionary from a TARE output DataFrame.

    For each county (identified via the ``county`` GISJOIN column or
    ``in.county`` column), extracts building IDs for Tier 1, Tier 2,
    constrained (T1 + T2), and all filtered buildings.

    Args:
        df_tare: TARE output DataFrame with ``bldg_id`` as index.
        adoption_col: Name of the adoption-tier column.
        tier_1_value: String label for Tier 1 in the adoption column.
        tier_2_value: String label for Tier 2 in the adoption column.

    Returns:
        Nested dict keyed by 5-digit FIPS string → sub-dict with keys
        ``'tier1'``, ``'tier2'``, ``'constrained'``, ``'all_filtered'``.
    """
    # Detect county column
    if "county" in df_tare.columns:
        county_col_name = "county"
    elif "in.county" in df_tare.columns:
        county_col_name = "in.county"
    else:
        raise KeyError(
            f"Neither 'county' nor 'in.county' found in TARE DataFrame.\n"
            f"  Available columns containing 'county': "
            f"{[c for c in df_tare.columns if 'county' in c.lower()]}"
        )

    df_work = df_tare[[county_col_name, adoption_col]].copy()
    df_work["county_fips"] = df_work[county_col_name].apply(gisjoin_to_fips)

    result: dict[str, dict[str, list[int]]] = {}
    for fips, grp in df_work.groupby("county_fips"):
        bldg_ids = grp.index.tolist()
        tier_vals = grp[adoption_col]
        tier1_ids = grp.index[tier_vals == tier_1_value].tolist()
        tier2_ids = grp.index[tier_vals == tier_2_value].tolist()
        result[str(fips)] = {
            "tier1": tier1_ids,
            "tier2": tier2_ids,
            "constrained": tier1_ids + tier2_ids,
            "all_filtered": bldg_ids,
        }
    return result


def compute_county_scenario_profile(
    df_baseline: pd.DataFrame,
    df_upgrade: pd.DataFrame,
    adopter_bldg_ids: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute hourly baseline and scenario demand profiles for one county.

    Args:
        df_baseline: Columns [bldg_id, hour, baseline_kwh]. 8,760 rows per building.
            Values are weight-applied (kWh × BSQ weight) from aggregate_timeseries.
        df_upgrade: Columns [bldg_id, hour, retrofit_kwh]. May be a subset of baseline.
            Values are weight-applied (kWh × BSQ weight) from aggregate_timeseries.
        adopter_bldg_ids: Buildings that adopt the retrofit.

    Returns:
        (df_profile, peak_dict) where df_profile has
        [hour, baseline_mw, scenario_mw, delta_mw].
    """
    adopter_set: set[int] = set(adopter_bldg_ids)
    all_baseline_bldgs: set[int] = set(df_baseline[BLDG_ID_COL].unique())
    upgrade_bldgs: set[int] = set(df_upgrade[BLDG_ID_COL].unique())

    adopters_missing_upgrade = adopter_set - upgrade_bldgs
    if adopters_missing_upgrade:
        print(
            f"  {len(adopters_missing_upgrade):,d} adopter bldg_ids "
            f"have no upgrade data — using baseline."
        )
    effective_adopters: set[int] = adopter_set & upgrade_bldgs

    # Left-join baseline ← upgrade
    df_merged: pd.DataFrame = df_baseline.merge(
        df_upgrade[[BLDG_ID_COL, "hour", "retrofit_kwh"]],
        on=[BLDG_ID_COL, "hour"],
        how="left",
    )

    # Vectorized adopter mask
    is_effective_adopter = df_merged[BLDG_ID_COL].isin(effective_adopters)
    retrofit_filled = df_merged["retrofit_kwh"].fillna(df_merged["baseline_kwh"])
    df_merged["scenario_kwh"] = np.where(
        is_effective_adopter, retrofit_filled, df_merged["baseline_kwh"]
    )

    # Aggregate across buildings → hourly county profile (MW)
    # BSQ values are already weight-applied, so just ÷ 1000 → MW
    df_profile: pd.DataFrame = (
        df_merged.groupby("hour", as_index=False)
        .agg(
            baseline_kwh=("baseline_kwh", "sum"),
            scenario_kwh=("scenario_kwh", "sum"),
        )
    )
    df_profile["baseline_mw"] = df_profile["baseline_kwh"] / 1000.0
    df_profile["scenario_mw"] = df_profile["scenario_kwh"] / 1000.0
    df_profile["delta_mw"] = df_profile["scenario_mw"] - df_profile["baseline_mw"]
    df_profile = df_profile[["hour", "baseline_mw", "scenario_mw", "delta_mw"]]

    if len(df_profile) != 8760:
        raise ValueError(
            f"Expected 8,760 hourly rows, got {len(df_profile):,d}. "
            f"Hour range: {df_profile['hour'].min()}..{df_profile['hour'].max()}"
        )

    peak_dict: dict[str, Any] = {
        "peak_hour_baseline": int(
            df_profile.loc[df_profile["baseline_mw"].idxmax(), "hour"]
        ),
        "peak_hour_scenario": int(
            df_profile.loc[df_profile["scenario_mw"].idxmax(), "hour"]
        ),
        "baseline_peak_mw": float(df_profile["baseline_mw"].max()),
        "scenario_peak_mw": float(df_profile["scenario_mw"].max()),
        "delta_mw": float(
            df_profile["scenario_mw"].max() - df_profile["baseline_mw"].max()
        ),
        "n_adopters": len(effective_adopters),
        "n_total_buildings": len(all_baseline_bldgs),
    }

    return df_profile, peak_dict
