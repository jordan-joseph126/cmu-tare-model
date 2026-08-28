"""Reusable functions for post-TARE peak load analysis.

Extracted from calculate_postTARE_ts_aws_peak_demand.ipynb (Phase 2 BSQ refactor).
Used by the notebook and by the national loop (Step 9).

Author: Jordan M. Joseph, PhD — Carnegie Mellon University
"""

from typing import Any, Dict, Iterable, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cmu_tare_model.constants import BLDG_ID_COL, BSQ_ELEC_COL
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


# =============================================================================
# NON-TIME-ALIGNED peak-load summary (per-home annual maxima; NO peak hour).
# Separate from compute_county_scenario_profile above, which is the coincident,
# hourly-profile approach. These two answer different questions and must not be
# mixed: the profile finds a single peak hour across homes; the summary below
# sums each home's own annual peak, so it has no hour and is a non-coincident
# upper bound on the true feeder peak.
# =============================================================================
def compute_peak_load_summary(
    df: pd.DataFrame,
    mp: int,
    adopter_bldg_ids: Iterable[int],
    *,
    weight_col: str = "weight",
    already_weighted: bool = False,
    county_name: str = "County",
) -> Dict[str, Any]:
    """Summarize residential electric peak demand WITHOUT time alignment.

    Unlike ``compute_county_scenario_profile`` (which sums 8,760 hourly
    profiles and finds one coincident peak hour), this sums each home's OWN
    annual peak. The homes are not aligned to a common hour, so the result has
    no peak hour and is a non-coincident upper bound on the true feeder peak --
    it overstates the coincident peak because it assumes every home peaks at
    once.

    Heating and cooling are reported as two independent seasonal quantities so
    the seasonal peak shift stays visible: a home's electric peak is usually in
    summer (cooling) before electrification and can move to winter (heating)
    after a heat-pump retrofit. The four seasonal quantities are summed over the
    adopter homes only. A secondary whole-home total takes each home's
    max(heating, cooling) and, for the scenario, gives adopters the retrofit
    value while non-adopters keep the baseline value -- the same per-home
    adopter logic ``compute_county_scenario_profile`` applies per hour.

    Weighting has two modes so one code path serves the standard sampled frame
    and a row-duplicated (per-parcel) frame:
      - default (``already_weighted=False``): multiply each home by its EUSS
        sample weight from ``weight_col``, then divide by 1,000 -> MW.
      - ``already_weighted=True``: the frame is already weighted by row
        duplication, so sum the raw per-home values, then divide by 1,000 -> MW,
        applying no sample weight (this avoids double-weighting).

    A frame with duplicate bldg_id rows (e.g. Tamar's row-duplicated per-parcel
    frame) combined with ``already_weighted=False`` raises ``ValueError``
    instead of silently double-weighting -- the caller must pass
    ``already_weighted=True`` for a row-duplicated frame.

    Args:
        df: TARE household frame for the scope of interest (e.g. one county),
            indexed by bldg_id. Must carry the base and mp{mp} peak-electricity
            columns and, unless ``already_weighted``, the ``weight_col`` column.
        mp: Measure-package number (3 or 4); selects the ``mp{mp}_`` columns.
        adopter_bldg_ids: Building IDs that adopt the retrofit. For the 100%
            scenario pass every applicable home; for the economically
            constrained scenario pass the NPV >= 0 homes.
        weight_col: Column holding the per-home EUSS sample weight. Ignored
            when ``already_weighted`` is True.
        already_weighted: True if the frame is pre-weighted by row duplication
            (do not apply the sample weight; sum rows directly).
        county_name: Display label for the scope (printing only).

    Returns:
        Dict with the four seasonal peaks (``baseline_heating_peak_mw``,
        ``baseline_cooling_peak_mw``, ``retrofit_heating_peak_mw``,
        ``retrofit_cooling_peak_mw``), the derived ``baseline_peak_season`` /
        ``retrofit_peak_season`` ('heating' or 'cooling'), the whole-home totals
        (``wholehome_baseline_total_mw``, ``wholehome_scenario_total_mw``,
        ``wholehome_delta_mw``), ``n_adopters``, ``n_total_buildings``, the
        ``mp`` and ``county_name`` echoed back, and ``not_time_aligned=True``.
        There are deliberately NO peak-hour keys.

    Raises:
        TypeError: If df is not a DataFrame.
        ValueError: If mp is not a positive integer, the frame is empty, or
            df has duplicate bldg_id rows with already_weighted=False (the
            double-weighting trap -- see already_weighted above).
        KeyError: If a required peak or weight column is absent.
    """
    # Step 1 -- validate inputs before any computation.
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a DataFrame, got {type(df)!r}")
    if not isinstance(mp, (int, np.integer)) or int(mp) <= 0:
        raise ValueError(f"mp must be a positive integer, got {mp!r}")
    if len(df) == 0:
        raise ValueError("df is empty; nothing to summarize.")
    mp = int(mp)

    base_heating = "base_peak_electricity_heating_kw"
    base_cooling = "base_peak_electricity_cooling_kw"
    mp_heating = f"mp{mp}_peak_electricity_heating_kw"
    mp_cooling = f"mp{mp}_peak_electricity_cooling_kw"

    required = [base_heating, base_cooling, mp_heating, mp_cooling]
    if not already_weighted:
        required.append(weight_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"MP{mp} peak summary is missing required column(s): {missing}"
        )

    # Guard: the double-weighting trap. If bldg_id has duplicate rows (a row-
    # duplicated per-parcel frame such as Tamar's feeder match) and
    # already_weighted=False, applying the sample weight on top of the row
    # duplication would double-count those homes. Refuse the combination
    # instead of silently computing a ~242x-inflated answer.
    if not already_weighted:
        n_dup_rows = int(df.index.duplicated().sum())
        if n_dup_rows > 0:
            raise ValueError(
                f"df has {n_dup_rows:,d} row(s) sharing a bldg_id with "
                f"another row, but already_weighted=False. This looks like a "
                f"row-duplicated per-parcel frame (e.g. Tamar's feeder "
                f"match) -- applying the '{weight_col}' sample weight on top "
                f"of the duplication would double-count those homes. Pass "
                f"already_weighted=True instead, which sums the duplicated "
                f"rows directly with no sample weight."
            )

    # Step 2 -- per-home weights. A pre-weighted frame counts each row once.
    if already_weighted:
        weights_all = pd.Series(1.0, index=df.index)
        # A pre-weighted frame's own 'weight' column, if present, is expected
        # to be informational only (e.g. the constant EUSS weight carried
        # through unchanged) -- it is never applied. Warn, don't raise.
        if weight_col in df.columns and not (df[weight_col] == 1.0).all():
            print(
                f"  NOTE: already_weighted=True but '{weight_col}' is "
                f"present and not all 1.0 -- it is being ignored by design; "
                f"row duplication is treated as the weighting, not this "
                f"column."
            )
    else:
        weights_all = df[weight_col].astype("float64")

    # Step 3 -- adopter mask over this frame (aligned to df row order). Kept
    # as a NumPy boolean array throughout -- never converted to a label list
    # and re-looked-up with .loc, which explodes into a cartesian product
    # when bldg_id has duplicates (confirmed empirically in the Task 1
    # audit). Boolean-array .loc indexing is always positional, so it stays
    # correct whether or not the index has duplicates.
    adopter_set: Set[int] = set(adopter_bldg_ids)
    is_adopter = df.index.isin(adopter_set)
    w_adopt = weights_all.loc[is_adopter]

    def weighted_mw(values: pd.Series, w: pd.Series) -> float:
        """Weighted kW sum -> MW. Skips NaN peaks (does not fill with zero)."""
        return float((values * w).sum() / 1000.0)

    # Step 4 -- four seasonal peaks, summed over the adopter homes only.
    baseline_heating_peak_mw = weighted_mw(
        df.loc[is_adopter, base_heating], w_adopt)
    baseline_cooling_peak_mw = weighted_mw(
        df.loc[is_adopter, base_cooling], w_adopt)
    retrofit_heating_peak_mw = weighted_mw(
        df.loc[is_adopter, mp_heating], w_adopt)
    retrofit_cooling_peak_mw = weighted_mw(
        df.loc[is_adopter, mp_cooling], w_adopt)

    def peak_season(heating_mw: float, cooling_mw: float) -> str:
        """Season carrying the larger summed peak (ties resolve to heating)."""
        return "heating" if heating_mw >= cooling_mw else "cooling"

    # Step 5 -- secondary whole-home total over ALL homes in the scope: each
    # home's own season max, with adopters switched to the retrofit and
    # non-adopters left on baseline. This is the non-coincident counterpart of
    # the profile's baseline_mw / scenario_mw, computed per home instead of per
    # hour.
    baseline_max = df[[base_heating, base_cooling]].max(axis=1)
    retrofit_max = df[[mp_heating, mp_cooling]].max(axis=1)
    scenario_max = pd.Series(
        np.where(is_adopter, retrofit_max, baseline_max), index=df.index
    )

    wholehome_baseline_total_mw = weighted_mw(baseline_max, weights_all)
    wholehome_scenario_total_mw = weighted_mw(scenario_max, weights_all)

    return {
        "county_name": county_name,
        "mp": mp,
        "not_time_aligned": True,
        "n_adopters": int(is_adopter.sum()),
        "n_total_buildings": int(len(df)),
        "baseline_heating_peak_mw": baseline_heating_peak_mw,
        "baseline_cooling_peak_mw": baseline_cooling_peak_mw,
        "retrofit_heating_peak_mw": retrofit_heating_peak_mw,
        "retrofit_cooling_peak_mw": retrofit_cooling_peak_mw,
        "baseline_peak_season": peak_season(
            baseline_heating_peak_mw, baseline_cooling_peak_mw),
        "retrofit_peak_season": peak_season(
            retrofit_heating_peak_mw, retrofit_cooling_peak_mw),
        "wholehome_baseline_total_mw": wholehome_baseline_total_mw,
        "wholehome_scenario_total_mw": wholehome_scenario_total_mw,
        "wholehome_delta_mw": (
            wholehome_scenario_total_mw - wholehome_baseline_total_mw),
    }


def build_adopter_ids_for_scope(
    df_tare: pd.DataFrame,
    adoption_col: str,
    *,
    county_fips: Optional[str] = None,
    custom_bldg_ids: Optional[Iterable[int]] = None,
) -> Dict[str, np.ndarray]:
    """Build the 100% and constrained adopter ID sets for one scope.

    Replaces hardcoding a county filter: the scope is chosen at the call site.
    Exactly one selector must be given:
      - county_fips: a 5-digit county FIPS string; keeps the homes whose GISJOIN
        'county' code maps to that FIPS (via gisjoin_to_fips).
      - custom_bldg_ids: an explicit set of building IDs (e.g. a feeder or
        matched-parcel set); keeps those homes present in the frame.

    Within the scope, two sets are returned, matching the study's economic
    adoption definition (NOT the deprecated Tier 1+2 split):
      - all_filtered: every building in the scope (the 100% adoption bound).
      - constrained: economic adopters only (adoption_col == 1.0). A NaN adopter
        value (excluded home) is not equal to 1.0 and is left out.

    Args:
        df_tare: TARE household frame indexed by bldg_id, carrying 'county'
            (GISJOIN, needed only for county scope) and adoption_col.
        adoption_col: The 0/1 economic-adopter column from find_adoption_column.
        county_fips: 5-digit county FIPS string, or None.
        custom_bldg_ids: Iterable of building IDs, or None.

    Returns:
        Dict with 'all_filtered' and 'constrained', each a NumPy boolean
        array the same length as df_tare, aligned by row position (True =
        home is in that set). NOT a list of building IDs: bldg_id is not
        guaranteed unique on every frame this is used on (e.g. Tamar's row-
        duplicated per-parcel frame), and a label list would silently
        multiply rows if re-selected with df.loc[label_list] on a duplicate
        index (confirmed empirically in the Task 1 audit). Select rows with
        df.loc[mask]; select bldg_id values with df.index[mask].

    Raises:
        ValueError: If not exactly one selector is given, county_fips is not 5
            digits, or the scope matches no homes.
        KeyError: If adoption_col or (for county scope) 'county' is absent.
    """
    # Step 1 -- exactly one scope selector.
    if (county_fips is None) == (custom_bldg_ids is None):
        raise ValueError(
            "Provide exactly one scope: county_fips OR custom_bldg_ids "
            "(not both, not neither)."
        )
    if adoption_col not in df_tare.columns:
        raise KeyError(f"adoption_col '{adoption_col}' not in the frame.")

    # Step 2 -- resolve the scope to buildings present in the frame.
    if county_fips is not None:
        county_fips = str(county_fips).strip()
        if len(county_fips) != 5 or not county_fips.isdigit():
            raise ValueError(
                f"county_fips must be a 5-digit string, got {county_fips!r}."
            )
        if "county" not in df_tare.columns:
            raise KeyError(
                "county-scope needs a 'county' (GISJOIN) column; not found."
            )
        fips_series = df_tare["county"].apply(gisjoin_to_fips)
        in_scope = (fips_series == county_fips).to_numpy()
        scope_desc = f"county FIPS {county_fips}"
    else:
        requested: Set[int] = set(custom_bldg_ids)
        in_scope = df_tare.index.isin(requested)
        found = int(in_scope.sum())
        if found < len(requested):
            print(
                f"  NOTE: {len(requested) - found:,d} of {len(requested):,d} "
                f"requested building IDs are not in this frame; using the "
                f"{found:,d} that are."
            )
        scope_desc = f"custom set of {len(requested):,d} building IDs"

    if not in_scope.any():
        raise ValueError(f"No homes match the requested scope ({scope_desc}).")

    # Step 3 -- split the scope into the 100% and economic-adopter boolean
    # masks. Both are full-length NumPy boolean arrays aligned to df_tare's
    # row order (True = in that set) -- NOT label lists. bldg_id is not
    # guaranteed unique on every frame this is used on (e.g. Tamar's row-
    # duplicated per-parcel frame), and re-selecting via a label list with
    # .loc explodes into a cartesian product for any duplicated id (confirmed
    # in the Task 1 audit). Boolean masks stay positionally correct either
    # way.
    # 100% (all_filtered): every building in scope, regardless of its economic
    # decision -- i.e. assume every sample home adopts the retrofit.
    # constrained: only the economic adopters (NPV >= 0) within that scope.
    is_adopter_full = df_tare[adoption_col].to_numpy() == 1.0
    all_filtered_mask = in_scope
    constrained_mask = in_scope & is_adopter_full

    n_frame = len(df_tare)
    n_scope = int(all_filtered_mask.sum())
    n_con = int(constrained_mask.sum())
    # Diagnostics: make the state-vs-scope building counts explicit so a small
    # in-scope count (e.g. one county) is not mistaken for the whole frame.
    print(
        f"  scope resolved: {scope_desc}\n"
        f"    buildings in frame (e.g. whole state) : {n_frame:,d}\n"
        f"    buildings in scope (100% adoption set): {n_scope:,d}\n"
        f"    economic adopters in scope (NPV >= 0) : {n_con:,d}"
    )
    return {
        "all_filtered": all_filtered_mask,
        "constrained": constrained_mask,
    }


def prompt_peak_load_scope(
    preset_county_fips: Optional[str] = None,
    preset_custom_bldg_ids: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    """Resolve the peak-load scope from presets or an interactive prompt.

    Batch-aware, mirroring the notebook's measure-package selector: if a preset
    is passed (a county FIPS or a custom building-ID set), it is used with no
    prompt, so a non-interactive run -- or code injecting a feeder set -- never
    blocks on input(). If neither preset is given, the user is asked whether to
    scope by county (then for a 5-digit FIPS) or to use a custom set.

    A custom building-ID set is never typed at the prompt (it is a variable such
    as a feeder list); choosing 'custom' interactively without a preset raises,
    directing the caller to pass preset_custom_bldg_ids.

    Args:
        preset_county_fips: Pre-set 5-digit county FIPS, or None.
        preset_custom_bldg_ids: Pre-set iterable of building IDs, or None.

    Returns:
        Dict with 'mode' ('county' or 'custom'), 'county_fips' (or None), and
        'custom_bldg_ids' (a set or None), ready to unpack into
        build_adopter_ids_for_scope.

    Raises:
        ValueError: If both presets are given, or an interactive choice or FIPS
            entry is invalid.
    """
    # Batch mode: a preset wins and skips the prompt. Only one may be set.
    if preset_county_fips is not None and preset_custom_bldg_ids is not None:
        raise ValueError(
            "Set only one preset: preset_county_fips OR preset_custom_bldg_ids."
        )
    if preset_county_fips is not None:
        fips = str(preset_county_fips).strip()
        print(f"[BATCH] Peak-load scope: county FIPS {fips}.")
        return {"mode": "county", "county_fips": fips, "custom_bldg_ids": None}
    if preset_custom_bldg_ids is not None:
        ids: Set[int] = set(preset_custom_bldg_ids)
        print(f"[BATCH] Peak-load scope: custom set of {len(ids):,d} IDs.")
        return {"mode": "custom", "county_fips": None, "custom_bldg_ids": ids}

    # Interactive mode. Type the full word so a bare 'c' can't mean either.
    choice = input(
        "Scope the peak-load summary by 'county' (a FIPS code) or 'custom' "
        "(a building-ID set)? Type county or custom: "
    ).strip().lower()
    # Echo the entry so the run log records exactly what was typed.
    print(f"[INPUT] scope choice entered: {choice!r}")
    if choice == "county":
        fips = input("Enter the 5-digit county FIPS code: ").strip()
        print(f"[INPUT] county FIPS entered: {fips!r}")
        if len(fips) != 5 or not fips.isdigit():
            raise ValueError(f"Expected a 5-digit FIPS, got {fips!r}.")
        return {"mode": "county", "county_fips": fips, "custom_bldg_ids": None}
    if choice == "custom":
        raise ValueError(
            "Custom-set scope selected: pass the building IDs via "
            "preset_custom_bldg_ids (e.g. a feeder set), not the prompt."
        )
    raise ValueError(
        f"Invalid choice {choice!r}; type 'county' or 'custom'."
    )


def _print_seasonal_block(r: Dict[str, Any]) -> None:
    """Print the four seasonal peaks and the peak-season line for one result."""
    print(f"  baseline  heating peak : {r['baseline_heating_peak_mw']:.2f} MW")
    print(f"  baseline  cooling peak : {r['baseline_cooling_peak_mw']:.2f} MW")
    print(f"  retrofit  heating peak : {r['retrofit_heating_peak_mw']:.2f} MW")
    print(f"  retrofit  cooling peak : {r['retrofit_cooling_peak_mw']:.2f} MW")
    print(f"  -> peak season shift: baseline={r['baseline_peak_season']}, "
          f"retrofit={r['retrofit_peak_season']}")


def print_peak_load_summary(
    result_100pct: Dict[str, Any],
    result_constrained: Optional[Dict[str, Any]] = None,
) -> None:
    """Print the non-time-aligned peak-load diagnostic block.

    Two adoption scenarios are reported, kept distinct:
      - 100% adoption: assume EVERY home in scope adopts the retrofit,
        regardless of its economic decision. Its seasonal peaks are summed over
        all homes in scope.
      - Economic adopters only: the subset of homes that are economic adopters
        (NPV >= 0, recovering the incremental cost through operating savings).
        Its seasonal peaks are summed over that subset only.

    Each scenario reports heating and cooling electric peaks (baseline vs
    retrofit, in MW) with the derived peak season, so the summer-to-winter shift
    is legible. All values are per-home annual maxima summed WITHOUT time
    alignment across buildings -- there is no peak hour, and any total is a
    non-coincident upper bound. A final whole-home block compares the two
    scenarios' totals against the shared baseline.

    Number formatting matches the time-aligned output: ',d' for counts, '.2f'
    for MW, and a leading '+' on deltas.

    Args:
        result_100pct: compute_peak_load_summary output for the 100% adoption
            scenario (adopter set = every building in scope).
        result_constrained: Optional compute_peak_load_summary output for the
            economic-adopters-only scenario (adopter set = NPV >= 0 homes).
            When present, its seasonal block and the whole-home comparison print.
    """
    r = result_100pct
    mp = r["mp"]
    n_scope = r["n_total_buildings"]

    print(f"{r['county_name']} peak results (MP{mp})   "
          f"[not time-aligned across bldg_ids]")
    if result_constrained is None:
        print(f"  buildings in scope: {n_scope:,d}")
    else:
        print(f"  buildings in scope: {n_scope:,d}   |   "
              f"economic adopters (NPV >= 0): "
              f"{result_constrained['n_adopters']:,d}")

    # Scenario 1 -- 100% adoption: every home in scope adopts the retrofit.
    print()
    print(f"[100% adoption -- all {n_scope:,d} homes assume the retrofit]")
    _print_seasonal_block(r)

    # Scenario 2 -- economic adopters only: the NPV >= 0 subset.
    if result_constrained is not None:
        c = result_constrained
        print()
        print(f"[Economic adopters only -- {c['n_adopters']:,d} of "
              f"{n_scope:,d} homes (NPV >= 0)]")
        _print_seasonal_block(c)

        # Whole-home max(heating, cooling) total: one shared baseline, one
        # scenario total per adoption assumption. A non-coincident upper bound.
        print()
        print("[whole-home max(heating, cooling) total -- "
              "non-coincident upper bound]")
        print(f"  baseline total                    : "
              f"{r['wholehome_baseline_total_mw']:.2f} MW")
        print(f"  scenario [100% adoption]          : "
              f"{r['wholehome_scenario_total_mw']:.2f} MW  "
              f"(delta {r['wholehome_delta_mw']:+.2f} MW)")
        print(f"  scenario [economic adopters only] : "
              f"{c['wholehome_scenario_total_mw']:.2f} MW  "
              f"(delta {c['wholehome_delta_mw']:+.2f} MW)")
