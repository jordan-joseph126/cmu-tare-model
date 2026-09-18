"""Reusable functions for post-TARE peak load analysis.

Extracted from calculate_postTARE_ts_aws_peak_demand.ipynb (Phase 2 BSQ refactor).
Used by the notebook and by the national loop (Step 9).

Author: Jordan M. Joseph, PhD — Carnegie Mellon University
"""

import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from cmu_tare_model.constants import (
    BLDG_ID_COL,
    BSQ_ELEC_COL,
    FIGURE_DPI,
    TIMESTAMP_COL,
)
from cmu_tare_model.utils.column_names import BASE_CASE_NPV_CASE


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
    npv_case: str = BASE_CASE_NPV_CASE,
) -> str:
    """Locate the economic-adopter column in a TARE output DataFrame.

    Builds the expected column name using ``create_adoption_col`` for the
    2025 Reference Case with the given NPV case and discount rate, then
    checks whether that column exists in the DataFrame.  Falls back to
    listing candidates if the exact column is absent.

    Args:
        df: TARE output DataFrame (one row per building).
        mp: Measure-package number (e.g. 3 or 4).
        cost_scenario: Retained for caller compatibility; not used to build
            the column name (the cost-scenario token was removed from
            output column names in the July 2026 refactor).
        discount_rate_key: Discount rate variant key (e.g. ``'fixed_base'``).
        npv_case: One of the NPV cases in NPV_CASE_CATEGORIES. Defaults to
            ``BASE_CASE_NPV_CASE`` (``'heatingLCC_coolingLCC_unsub'``) from
            column_names.py -- the study base case: unsubsidized, with both the
            heating and cooling replacement costs credited in the NPV. Pass this
            argument by keyword at every call site so a positional slip cannot
            silently substitute another case.

    Returns:
        The matched column name string.

    Raises:
        KeyError: If no matching adoption column is found.
    """
    from cmu_tare_model.utils.column_names import create_adoption_col

    expected = create_adoption_col(
        scenario_prefix=f"ref2025_mp{mp}_",
        npv_case=npv_case,
        method_suffix=f"_{discount_rate_key}",
    )
    if expected in df.columns:
        return expected

    # Fallback: list any economic-adopter columns to aid debugging.
    adopter_candidates = [c for c in df.columns if "econ_adopter" in c.lower()]
    if adopter_candidates:
        raise KeyError(
            f"Expected adoption column '{expected}' not found.\n"
            f"  Candidates containing 'econ_adopter' ({len(adopter_candidates)}):\n"
            + "\n".join(f"    - {c}" for c in adopter_candidates)
        )
    raise KeyError(
        f"Expected adoption column '{expected}' not found, "
        f"and no columns containing 'econ_adopter' exist."
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


def print_heating_fuel_distribution_table(
    df_by_mp: Dict[int, pd.DataFrame],
    adopter_ids_by_mp: Dict[int, Dict[str, Dict[str, list]]],
    selected_mps: list,
    case_study_fips: str,
    *,
    fuel_col: str = "base_heating_fuel",
) -> None:
    """Print baseline heating-fuel counts and shares for one county.

    One column pair per measure package -- economic adopters (constrained)
    and every filtered building (100% adoption) -- so the fuel mix behind
    each grid-impact scenario is visible at a glance.

    Args:
        df_by_mp: {mp: df} -- one TARE output dataframe per measure
            package, indexed by bldg_id.
        adopter_ids_by_mp: {mp: {fips: {"all_filtered": [...],
            "constrained": [...]}}}, built once nationally.
        selected_mps: Measure-package numbers to show as column pairs.
        case_study_fips: County FIPS this table reports on.
        fuel_col: Column holding each building's baseline heating fuel.

    Raises:
        KeyError: If case_study_fips is not found for a measure package.
    """
    # Count each fuel type for the constrained and 100% adoption sets, per MP.
    fuel_results = {}
    for mp in selected_mps:
        df_tare = df_by_mp[mp]
        if case_study_fips not in adopter_ids_by_mp[mp]:
            raise KeyError(
                f"MP{mp}: county FIPS {case_study_fips} not in "
                "adopter_ids_by_mp."
            )
        for scenario, key in [("constrained", "constrained"),
                               ("100pct", "all_filtered")]:
            bldg_ids = set(adopter_ids_by_mp[mp][case_study_fips][key])
            counts = (
                df_tare.loc[df_tare.index.isin(bldg_ids), fuel_col]
                .value_counts()
                .sort_index()
            )
            fuel_results[(mp, scenario)] = {
                "n": len(bldg_ids),
                "counts": counts,
                "pcts": counts / counts.sum() * 100,
            }

    all_fuels = sorted(
        set().union(*(r["counts"].index for r in fuel_results.values()))
    )

    # Column widths generalized to any number of MPs (the original version
    # hardcoded 4 columns, assuming exactly two).
    fuel_col_width = 20
    data_col_width = 26
    n_columns = len(selected_mps) * 2
    divider_width = fuel_col_width + data_col_width * n_columns + 3

    print(f"County FIPS {case_study_fips} -- Baseline Heating Fuel Distribution")
    print("=" * divider_width)

    # Header row -- one Constrained and one 100% column per measure package.
    headers = []
    for mp in selected_mps:
        n_con = fuel_results[(mp, "constrained")]["n"]
        n_all = fuel_results[(mp, "100pct")]["n"]
        headers.append(f"MP{mp} Constrained (n={n_con:,})")
        headers.append(f"MP{mp} 100% (n={n_all:,})")

    print(f"{'Fuel':<{fuel_col_width}}", end="")
    for header in headers:
        print(f"  {header:>{data_col_width - 2}}", end="")
    print()

    print(f"{'-' * fuel_col_width}", end="")
    for _ in headers:
        print(f"  {'-' * (data_col_width - 2)}", end="")
    print()

    # One row per fuel type, one cell per (MP, scenario) column.
    for fuel in all_fuels:
        print(f"{fuel:<{fuel_col_width}}", end="")
        for mp in selected_mps:
            for scenario in ("constrained", "100pct"):
                result = fuel_results[(mp, scenario)]
                count = result["counts"].get(fuel, 0)
                pct = result["pcts"].get(fuel, 0.0)
                cell_text = f"{count:,} ({pct:.1f}%)"
                print(f"  {cell_text:>{data_col_width - 2}}", end="")
        print()

    print(f"{'-' * fuel_col_width}", end="")
    for _ in headers:
        print(f"  {'-' * (data_col_width - 2)}", end="")
    print()

    print(f"{'TOTAL':<{fuel_col_width}}", end="")
    for mp in selected_mps:
        for scenario in ("constrained", "100pct"):
            n = fuel_results[(mp, scenario)]["n"]
            print(f"  {f'{n:,} (100.0%)':>{data_col_width - 2}}", end="")
    print()

    print(f"\n[OK] County FIPS {case_study_fips} heating fuel table complete")


def compute_county_scenario_profile(
    df_baseline: pd.DataFrame,
    df_upgrade: pd.DataFrame,
    adopter_bldg_ids: list[int],
    *,
    custom_weighting: bool = False,
    weight_dict: Optional[Dict[int, float]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute hourly baseline and scenario demand profiles for one county.

    Args:
        df_baseline: Columns [bldg_id, hour, baseline_kwh]. 8,760 rows per
            building. kWh is weight-applied when custom_weighting is False,
            raw (unweighted) when custom_weighting is True.
        df_upgrade: Columns [bldg_id, hour, retrofit_kwh]. May be a subset of
            baseline. Same weighting convention as df_baseline.
        adopter_bldg_ids: Buildings that adopt the retrofit.
        custom_weighting: If True, apply the per-building weights in
            weight_dict instead of trusting BSQ's own weighting. See the
            comments below for what changes, step by step, in each mode.
        weight_dict: Required when custom_weighting is True. Maps bldg_id to
            its custom weight (e.g. the count of tax parcels matched to that
            representative building). Ignored when custom_weighting is
            False.

    Returns:
        (df_profile, peak_dict) where df_profile has
        [hour, baseline_mw, scenario_mw, delta_mw].

    Raises:
        ValueError: If custom_weighting is True and weight_dict is None or
            empty, or if the resulting profile does not have exactly 8,760
            hourly rows.
    """
    # custom_weighting=True requires a weight to apply.
    if custom_weighting and not weight_dict:
        raise ValueError(
            "weight_dict is required and must be non-empty when "
            "custom_weighting=True."
        )

    adopter_set: set[int] = set(adopter_bldg_ids)

    if custom_weighting:
        # Unmatched buildings have no valid weight -- excluded, never
        # defaulted to the uniform weight or to zero.
        weighted_bldg_ids = set(weight_dict.keys())

        # Drop unmatched baseline buildings; report how many were dropped.
        n_baseline_before = df_baseline[BLDG_ID_COL].nunique()
        df_baseline = df_baseline[
            df_baseline[BLDG_ID_COL].isin(weighted_bldg_ids)
        ]
        n_baseline_dropped = (
            n_baseline_before - df_baseline[BLDG_ID_COL].nunique()
        )
        if n_baseline_dropped:
            print(
                f"  custom_weighting: {n_baseline_dropped:,d} baseline "
                f"building(s) have no entry in weight_dict -- excluded."
            )

        # Same for upgrade buildings.
        df_upgrade = df_upgrade[df_upgrade[BLDG_ID_COL].isin(weighted_bldg_ids)]

        # Same for the adopter list -- an unmatched adopter is removed too.
        n_adopters_before = len(adopter_set)
        adopter_set = adopter_set & weighted_bldg_ids
        n_adopters_dropped = n_adopters_before - len(adopter_set)
        if n_adopters_dropped:
            print(
                f"  custom_weighting: {n_adopters_dropped:,d} adopter "
                f"building(s) have no entry in weight_dict -- excluded."
            )

    all_baseline_bldgs: set[int] = set(df_baseline[BLDG_ID_COL].unique())
    upgrade_bldgs: set[int] = set(df_upgrade[BLDG_ID_COL].unique())

    adopters_missing_upgrade = adopter_set - upgrade_bldgs
    if adopters_missing_upgrade:
        print(
            f"  {len(adopters_missing_upgrade):,d} adopter bldg_ids "
            f"have no upgrade data -- using baseline."
        )
    effective_adopters: set[int] = adopter_set & upgrade_bldgs

    # Left-join baseline <- upgrade
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

    if custom_weighting:
        # Every remaining row is guaranteed to be in weight_dict by the
        # filtering above, so this lookup cannot produce a NaN weight.
        weight_multiplier = (
            df_merged[BLDG_ID_COL].map(weight_dict).astype("float64")
        )
    else:
        # BSQ already applied its weight -- 1.0 is a no-op multiply, so this
        # matches the pre-custom-weighting behavior exactly.
        weight_multiplier = 1.0
    df_merged["weighted_baseline_kwh"] = (
        df_merged["baseline_kwh"] * weight_multiplier
    )
    df_merged["weighted_scenario_kwh"] = (
        df_merged["scenario_kwh"] * weight_multiplier
    )

    # Aggregate across buildings -> hourly county profile (MW)
    df_profile: pd.DataFrame = (
        df_merged.groupby("hour", as_index=False)
        .agg(
            baseline_kwh=("weighted_baseline_kwh", "sum"),
            scenario_kwh=("weighted_scenario_kwh", "sum"),
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


def prepare_bsq_timeseries(
    df_ts_raw: pd.DataFrame,
    kwh_col_name: str,
    *,
    bsq_col: str = BSQ_ELEC_COL,
    bldg_id_col: str = BLDG_ID_COL,
    timestamp_col: str = TIMESTAMP_COL,
) -> pd.DataFrame:
    """Clean up one raw BuildStockQuery hourly timeseries result.

    The baseline (upgrade_id="0") query and each measure package's upgrade
    query return the same raw shape, and both need the same three fixes
    before compute_county_scenario_profile can use them. Factored out so
    both call sites share one implementation instead of repeating it.

    Args:
        df_ts_raw: Raw aggregate_timeseries() result -- one row per
            building-hour, with a bsq_col column and a timestamp_col column.
        kwh_col_name: Name to give the renamed kWh column (e.g.
            "baseline_kwh" or "retrofit_kwh").
        bsq_col: BSQ's own column name for the electricity total (BSQ
            strips the "out." prefix off the enduse name it was queried
            with).
        bldg_id_col: Building id column name.
        timestamp_col: Timestamp column name BSQ returns.

    Returns:
        A copy of df_ts_raw, sorted by [bldg_id_col, timestamp_col], with
        the kWh column renamed and downcast to float32, plus a new "hour"
        column numbered 1..N per building.
    """
    # Rename to a name this notebook controls; downcast to float32 to halve
    # memory use with no meaningful precision loss for kWh at this scale.
    df_prepared = df_ts_raw.rename(columns={bsq_col: kwh_col_name})
    df_prepared[kwh_col_name] = df_prepared[kwh_col_name].astype(np.float32)

    # Sort into time order so the hour numbering below comes out correct.
    df_prepared = df_prepared.sort_values(
        [bldg_id_col, timestamp_col]
    ).reset_index(drop=True)

    # BSQ returns a timestamp, not an hour index -- number each building's
    # rows 1..N.
    df_prepared["hour"] = df_prepared.groupby(bldg_id_col).cumcount() + 1

    return df_prepared


def summarize_hourly_timeseries(
    df_ts: pd.DataFrame,
    kwh_col: str,
    label: str,
    query_time_s: float,
    *,
    bldg_id_col: str = BLDG_ID_COL,
    expected_hours_per_bldg: int = 8760,
) -> None:
    """Print a summary of one prepared BSQ timeseries and check its coverage.

    Used for both the baseline query and each measure package's upgrade
    query, so the two summaries always report the same statistics and are
    checked against the same full-year requirement.

    Args:
        df_ts: A timeseries dataframe already processed by
            prepare_bsq_timeseries (must have an "hour" column).
        kwh_col: Name of the kWh column to summarize (e.g. "baseline_kwh").
        label: Short label printed in the summary header (e.g.
            "df_ts_baseline_case_study" or "df_ts_upgrade_case_study (MP3)").
        query_time_s: How long the query took, for the printed summary.
        bldg_id_col: Building id column name.
        expected_hours_per_bldg: Required row count per building --
            compute_county_scenario_profile assumes a full 8,760-hour year.

    Raises:
        ValueError: If any building does not have exactly
            expected_hours_per_bldg rows. Raised rather than asserted, so
            this check cannot be silently skipped (Python's -O flag strips
            assert statements).
    """
    n_bldgs = df_ts[bldg_id_col].nunique()
    n_hours_per_bldg = df_ts.groupby(bldg_id_col).size()

    print(f"\n========== {label} summary ==========")
    print(f"  Rows       : {len(df_ts):,d}")
    print(f"  Buildings  : {n_bldgs:,d}")
    print(
        f"  Hours/bldg : {n_hours_per_bldg.min():,d} - "
        f"{n_hours_per_bldg.max():,d}"
    )
    print(
        f"  kWh range  : {df_ts[kwh_col].min():.3f} to "
        f"{df_ts[kwh_col].max():.3f}"
    )
    print(f"  Query time : {query_time_s:.2f} s")

    if (
        n_hours_per_bldg.min() != expected_hours_per_bldg
        or n_hours_per_bldg.max() != expected_hours_per_bldg
    ):
        raise ValueError(
            f"{label}: expected exactly {expected_hours_per_bldg:,d} hours "
            f"per building, got a range of {n_hours_per_bldg.min():,d} - "
            f"{n_hours_per_bldg.max():,d}."
        )


def check_upgrade_building_coverage(
    baseline_bldg_ids: set,
    upgrade_bldg_ids: set,
    mp: int,
) -> set:
    """Compare which buildings a baseline and an upgrade query cover.

    Both queries are built from the same restrict=[("bldg_id", ...)] list,
    so they should cover the same buildings.

    Args:
        baseline_bldg_ids: Building ids present in the baseline query.
        upgrade_bldg_ids: Building ids present in this MP's upgrade query.
        mp: Measure-package number, used only in the printed note and the
            error message.

    Returns:
        The set of buildings present in baseline but missing from the
        upgrade query. This is expected occasionally -- ResStock does not
        model every upgrade for every building -- and
        compute_county_scenario_profile already falls back to that
        building's baseline value for these.

    Raises:
        ValueError: If any building appears in the upgrade query but not in
            the baseline query. This should never happen with a shared
            restrict list, so it is treated as an error rather than a note.
    """
    only_in_baseline = baseline_bldg_ids - upgrade_bldg_ids
    only_in_upgrade = upgrade_bldg_ids - baseline_bldg_ids

    if only_in_baseline:
        print(
            f"  Note: {len(only_in_baseline):,d} buildings have no MP{mp} "
            "upgrade data and will use baseline."
        )
    if only_in_upgrade:
        raise ValueError(
            f"MP{mp}: {len(only_in_upgrade):,d} upgrade buildings "
            "missing baseline."
        )

    return only_in_baseline


def plot_demand_panel(
    ax: Any,
    df_profile: pd.DataFrame,
    peak_result: dict[str, Any],
    mp: int,
    scenario_label: str,
    county_name: str = "Allegheny County, PA",
) -> None:
    """Plot baseline and scenario demand timeseries on a single axes panel.

    Args:
        ax: Matplotlib Axes to draw on.
        df_profile: DataFrame output of ``compute_county_scenario_profile``.
        peak_result: Dict output of ``compute_county_scenario_profile``.
        mp: Measure-package number.
        scenario_label: Human-readable scenario name.
        county_name: County name shown in the title.
    """
    ax.plot(df_profile["hour"], df_profile["baseline_mw"],
            color="tab:red", linewidth=0.8, alpha=0.5)
    ax.plot(df_profile["hour"], df_profile["scenario_mw"],
            color="tab:blue", linewidth=0.8, alpha=0.5)

    # Horizontal dashed lines mark each series' peak MW so panels sharing a
    # y-axis can be compared by eye. What solid vs. dashed and red vs. blue
    # mean is explained once in the shared figure legend built by the
    # notebook cell that calls this function, not repeated as per-panel text
    # (the old text-box annotations overlapped and were hard to read).
    ax.axhline(y=peak_result["baseline_peak_mw"], color="tab:red",
               linestyle="--", linewidth=2.0, alpha=0.85)
    ax.axhline(y=peak_result["scenario_peak_mw"], color="tab:blue",
               linestyle="--", linewidth=2.0, alpha=0.85)

    # Black open circle marks the exact (hour, MW) point where each series
    # peaks -- the dashed line shows the peak height across the whole panel,
    # this marker pins down exactly when it happens.
    ax.plot(peak_result["peak_hour_baseline"], peak_result["baseline_peak_mw"],
            marker="o", markerfacecolor="none", markeredgecolor="black",
            markersize=11, markeredgewidth=2.0, linestyle="none", zorder=5)
    ax.plot(peak_result["peak_hour_scenario"], peak_result["scenario_peak_mw"],
            marker="o", markerfacecolor="none", markeredgecolor="black",
            markersize=11, markeredgewidth=2.0, linestyle="none", zorder=5)

    ax.set_xlabel("Hour of Year", fontsize=17)
    ax.set_ylabel("Demand (MW)", fontsize=17)
    ax.tick_params(labelsize=15)


def plot_county_demand_grid(
    df_profiles_by_mp: Dict[int, Dict[str, pd.DataFrame]],
    peak_results_by_mp: Dict[int, Dict[str, Dict[str, Any]]],
    selected_mps: list,
    *,
    mp_labels: Optional[Dict[int, str]] = None,
    county_display_name: str = "Allegheny County, PA",
    save_figure: bool = False,
    output_dir: Optional[str] = None,
    figure_dpi: int = FIGURE_DPI,
) -> plt.Figure:
    """Draw the MP x scenario demand-profile grid with a shared legend.

    Rows are measure packages (in ``selected_mps`` order); columns are the two
    adoption scenarios, economic adopters left and 100% adoption right. All
    panels share a y-axis so peak MW is directly comparable across both
    scenarios and both measure packages.

    This consolidates two duplicate 2x2 notebook blocks that differed only in
    row/column axis ordering, font sizes, and a shared legend -- consolidated
    2 Sep 2026 during the notebook/codebase cleanup session onto the second,
    later block (row=MP, column=scenario, with the shared legend), which was
    the more complete of the two. The superseded first block (row=scenario,
    column=MP, no shared legend, smaller fonts) was not kept.

    Args:
        df_profiles_by_mp: ``{mp: {'100pct': df_profile, 'constrained':
            df_profile}}`` -- outputs of ``compute_county_scenario_profile``,
            one pair per measure package.
        peak_results_by_mp: ``{mp: {'100pct': peak_dict, 'constrained':
            peak_dict}}`` -- the matching peak dicts from the same calls.
        selected_mps: Measure-package numbers to render as rows, in order.
        mp_labels: Row label per MP (e.g. ``{3: 'Minimum-Efficiency Heat
            Pump'}``). Defaults to the MP3/MP4 labels this notebook uses.
        county_display_name: County name shown in the legend box title.
        save_figure: If True and ``output_dir`` is set, save the figure.
        output_dir: Directory the figure is saved under (the file goes in
            ``output_dir/outputs/``); required when ``save_figure`` is True.
        figure_dpi: Resolution used when saving.

    Returns:
        The matplotlib Figure.

    Raises:
        ValueError: If save_figure is True but output_dir is None.
    """
    if save_figure and output_dir is None:
        raise ValueError("output_dir is required when save_figure=True.")

    if mp_labels is None:
        mp_labels = {
            3: "Minimum-efficiency heat pump",
            4: "High-efficiency heat pump",
        }

    scenarios = ["constrained", "100pct"]
    scenario_labels = ["Only Economic Adopters", "100% Adoption"]
    subplot_title_fontsize = 18
    tick_label_fontsize = 16

    # Month x-axis: hour is hour-of-year. Ticks land at the first hour of each
    # month (non-leap year, matching the 8,760-row profile). Hours are
    # cumulative, so this is built once from the days-per-month table rather
    # than hardcoding twelve hour offsets.
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    _starts_days, _running = [], 0
    for _d in days_in_month:
        _starts_days.append(_running)
        _running += _d
    month_start_hours = [s * 24 for s in _starts_days]

    # Scope the white style to THIS figure only via sns.axes_style, independent
    # of any global sns.set_theme(style=...) the caller has set, so this
    # figure's styling stays fixed even if that global changes later.
    with sns.axes_style("white"):
        # sharey='all' -> all panels share a single y-scale, so peak MW is
        # directly comparable across both scenarios and both measure packages.
        fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey='all')
        fig.patch.set_facecolor("white")

        for row_idx, mp in enumerate(selected_mps):
            for col_idx, (scenario, scenario_label) in enumerate(
                    zip(scenarios, scenario_labels)):
                ax = axes[row_idx, col_idx]
                ax.set_facecolor("white")
                df_profile = df_profiles_by_mp[mp][scenario]
                peak_result = peak_results_by_mp[mp][scenario]
                plot_demand_panel(ax, df_profile, peak_result, mp, scenario_label)
                ax.set_title(
                    f"{mp_labels.get(mp, f'MP{mp}')} ({scenario_label})",
                    fontsize=subplot_title_fontsize,
                    fontweight="bold",
                )

                # --- Override x-axis to months + enlarge tick labels ---
                h0 = df_profile["hour"].min()
                ax.set_xticks([h0 + m for m in month_start_hours])
                ax.set_xticklabels(month_labels)
                ax.set_xlim(h0, h0 + 8760)
                ax.set_xlabel("Month", fontsize=17)
                ax.tick_params(labelsize=tick_label_fontsize)

        # --- Shared legend, bottom center, drawn as a fancy box ---
        # Proxy handles only (no data) -- the real lines/markers are drawn per
        # panel by plot_demand_panel. The black peak-X marker is left out of
        # the legend on purpose (self-evident on the panels, and a fifth
        # entry would add a row and compress the figure) -- only solid vs.
        # dashed and red vs. blue are explained here.
        # Order is [solid_red, dashed_red, solid_blue, dashed_blue] so that
        # matplotlib's column-major legend fill (with ncol=2) lays them out as
        # two rows -- row 1 solid red/blue, row 2 dashed red/blue -- matching
        # "Solid Red | Solid Blue" then "Dashed Red | Dashed Blue".
        legend_handles = [
            Line2D([0], [0], color="tab:red", linewidth=2.5, linestyle="-"),
            Line2D([0], [0], color="tab:red", linewidth=2.5, linestyle="--"),
            Line2D([0], [0], color="tab:blue", linewidth=2.5, linestyle="-"),
            Line2D([0], [0], color="tab:blue", linewidth=2.5, linestyle="--"),
        ]
        legend_labels = [
            "Existing HVAC",
            "Peak Existing HVAC",
            "Post-Retrofit",
            "Peak Post-Retrofit",
        ]
        fig_legend = fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=2,
            fontsize=16,
            title=f"Residential Electricity Load (MW) for {county_display_name}",
            title_fontsize=17,
            frameon=True,
            fancybox=True,
            shadow=True,
            facecolor="white",
            edgecolor="0.3",
            framealpha=0.95,
            borderpad=1.1,
            labelspacing=0.9,
            handlelength=2.5,
        )
        fig_legend.get_title().set_fontweight("bold")

        # Extra bottom margin so the legend box has room below the panels.
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        if save_figure:
            out_path = os.path.join(
                output_dir,
                "outputs",
                f"allegheny_demand_profiles_MP"
                f"{'_'.join(str(m) for m in selected_mps)}.png",
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fig.savefig(out_path, dpi=figure_dpi, bbox_inches="tight")
            print(f"[OK] Figure saved: {out_path}")
        plt.show()

    return fig


# =============================================================================
# The non-time-aligned peak-load summary path (compute_peak_load_summary,
# build_adopter_ids_for_scope, prompt_peak_load_scope, print_peak_load_summary,
# and the private _print_seasonal_block helper) was moved out of this module
# on 2 Sep 2026, during the notebook/codebase cleanup session -- the live
# notebook's grid-impact cells had already migrated to the time-aligned
# BuildStockQuery hourly-profile approach above (compute_county_scenario_profile
# + plot_demand_panel), and a full-repo grep found zero importers of the moved
# functions outside this file. They are kept, not deleted, at
# cmu_tare_model/grid_impact/archived_files/peak_load_functions_legacy.py.
# =============================================================================
