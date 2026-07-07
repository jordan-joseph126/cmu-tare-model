"""
Adoption rate computation functions for the TARE model KPIs.

Computes the weighted adoption rate -- the share of homes (by EUSS sampling
weight) that are economic adopters -- aggregated to county or state level.

Adoption rate definition:
    adoption_rate = sum(w x is_adopter) / sum(w) x 100  (percent)

    is_adopter = True  if the economic-adopter column equals 1.0
    is_adopter = False otherwise (0.0 non-adopter; NaN excluded homes)

For legacy tiered-adoption columns (string values), is_adopter is True when
the tier is in adopter_tiers (Tier 1 or Tier 2 by default).

Location: cmu_tare_model/adoption_kpis/compute_adoption_rate.py
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from cmu_tare_model.adoption_kpis.data_loading import COUNTY_COL
from cmu_tare_model.constants import MIN_HOME_COUNT


# ============================================================================
# MODULE-LEVEL CONSTANTS
# ============================================================================

DEFAULT_ADOPTER_TIERS: List[str] = [
    "Tier 1: Feasible",
    "Tier 2: Feasible vs. Alternative",
]
"""Default tier string values counted as adopters (Tier 1 + Tier 2)."""

ROUND_RATE_DECIMALS: int = 2
"""Decimal places for rounding adoption_rate_pct."""


# ---------------------------------------------------------------------------
# NOTE ON SAMPLING WEIGHTS
# ---------------------------------------------------------------------------
# ResStock assigns a uniform sampling weight (~242) to every building.
# Because the weight is constant, it cancels in any ratio, rate, or
# percentage computation.  Simple counts and sums are used for these
# metrics.  The weight IS applied when computing absolute population
# totals (e.g., home_count in millions) to scale from sample to national.
# ---------------------------------------------------------------------------


# ============================================================================
# ADOPTION RATE AGGREGATION
# ============================================================================

def compute_adoption_rate(
    df: pd.DataFrame,
    adoption_col: str,
    adopter_tiers: Optional[List[str]] = None,
    geo_level: str = "county",
    min_home_count: int = MIN_HOME_COUNT,
    weight_col: str = "weight",
    county_col: str = "in.county",
    state_col: str = "in.state",
    df_euss: Optional[pd.DataFrame] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute weighted adoption rate aggregated to county- or state-level.

    Adoption rate is the population-weighted share of homes whose adoption
    tier falls in ``adopter_tiers`` (default: Tier 1 + Tier 2).

    Formula::

        adoption_rate_pct = Σ(w × is_adopter) / Σ(w) × 100

    where ``is_adopter = 1`` if ``adoption_col ∈ adopter_tiers``, else ``0``.

    The ``min_home_count`` threshold is applied to the **sample** count
    (number of rows), not the weighted population, to ensure statistical
    reliability of the estimate.  Counties below the threshold have
    ``adoption_rate_pct`` set to ``NaN`` and are excluded from choropleth maps.

    Args:
        df: TARE output DataFrame (one row per building, indexed by bldg_id).
            Must contain ``county_col`` (GISJOIN format), ``weight_col``, and
            ``adoption_col``.  If ``state_col`` is present it is included in
            the output; otherwise the ``state`` column is omitted.
        adoption_col: Economic-adopter column name (numeric 1.0/0.0/NaN), e.g.
            ``'ref2025_mp3_heatingLCC_coolingLCC_econ_adopter_fixed_base'``.
            A legacy tier-string column is also accepted.
        adopter_tiers: Tier string values counted as adopters for legacy
            tier-string columns.  Defaults to
            ``['Tier 1: Feasible', 'Tier 2: Feasible vs. Alternative']``.
        geo_level: ``'county'`` (default) or ``'state'``.
        min_home_count: Minimum **sample** buildings per county/state.
            Counties below this threshold have ``adoption_rate_pct = NaN``.
            Only applied when ``geo_level='county'``.
        weight_col: Column name containing EUSS sampling weights.
        county_col: Column name containing GISJOIN county codes
            (e.g. ``'G4200030'``).  Used when ``geo_level='county'``.
        state_col: Column name containing state abbreviations.  Included in
            output only when present in ``df``.
        verbose: If ``True``, print national weighted adoption rate and
            county/state count.

    Returns:
        DataFrame sorted by ``adoption_rate_pct`` (descending) with columns:
        ``county`` (GISJOIN, when ``geo_level='county'``),
        ``state`` (when available), ``home_count`` (weighted population),
        ``adoption_rate_pct``.

    Raises:
        ValueError: If ``geo_level`` is not ``'county'`` or ``'state'``.
        KeyError: If ``adoption_col``, ``weight_col``, or ``county_col``
            (when ``geo_level='county'``) are missing from ``df`` and
            cannot be resolved via aliases or ``df_euss``.
    """
    if geo_level not in ("county", "state"):
        raise ValueError(
            f"geo_level must be 'county' or 'state', got {geo_level!r}"
        )

    if adopter_tiers is None:
        adopter_tiers = DEFAULT_ADOPTER_TIERS

    # --- Resolve column name aliases ---
    # TARE summary CSVs may drop the 'in.' prefix (e.g. 'county' not 'in.county').
    # Fall back to the unprefixed form when the prefixed form is absent.
    _county_col = county_col if county_col in df.columns else (
        "county" if "county" in df.columns else county_col
    )
    _state_col = state_col if state_col in df.columns else (
        "state" if "state" in df.columns else state_col
    )

    # --- Resolve weight column ---
    # weight may be absent from older TARE summary CSVs; merge from df_euss.
    if weight_col not in df.columns:
        if df_euss is None:
            raise ValueError(
                f"Column '{weight_col}' not found in df and df_euss was not provided. "
                "Pass the EUSS baseline DataFrame as df_euss to supply sampling weights."
            )
        if weight_col not in df_euss.columns:
            raise KeyError(
                f"Column '{weight_col}' not found in df_euss. "
                f"Available columns (first 20): {list(df_euss.columns[:20])}"
            )
        df = df.join(df_euss[[weight_col]], how="inner")

    # --- Validate remaining required columns ---
    required = [adoption_col, weight_col]
    geo_key = _county_col if geo_level == "county" else _state_col
    required.append(geo_key)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Required column(s) not found in df: {missing}.\n"
            f"  Available columns (first 20): {list(df.columns[:20])}"
        )

    # --- Build working DataFrame ---
    group_col = _county_col if geo_level == "county" else _state_col
    has_state = _state_col in df.columns

    keep_cols = [group_col, weight_col, adoption_col]
    if has_state and group_col != _state_col:
        keep_cols.append(_state_col)

    df_work = df[keep_cols].copy()

    # --- Adopter flag ---
    # ResStock uses uniform sampling weight (~242) for all buildings.
    # Rates and percentages use simple counts (weight cancels in ratios).
    # home_count uses the sum of weights for scaling to national population totals.
    #
    # The economic-adopter columns are numeric (1.0 = adopter, 0.0 = non-adopter,
    # NaN = excluded home). Legacy tiered-adoption columns instead store tier
    # strings, so match those against adopter_tiers when the column is not
    # numeric.
    adoption_values = df_work[adoption_col]
    if pd.api.types.is_numeric_dtype(adoption_values):
        df_work["_is_adopter"] = (adoption_values == 1.0).astype(int)
    else:
        df_work["_is_adopter"] = adoption_values.isin(adopter_tiers).astype(int)

    # --- Aggregate ---
    grouped = df_work.groupby(group_col).agg(
        _sample_count=(weight_col, "size"),
        home_count=(weight_col, "sum"),
        _n_adopters=("_is_adopter", "sum"),
    ).reset_index()

    grouped["adoption_rate_pct"] = np.where(
        grouped["_sample_count"] > 0,
        grouped["_n_adopters"] / grouped["_sample_count"] * 100,
        np.nan,
    )

    # --- State column ---
    if geo_level == "county":
        if has_state:
            state_lookup = df_work.groupby(group_col)[_state_col].first()
            grouped["state"] = grouped[group_col].map(state_lookup)
        grouped = grouped.rename(columns={group_col: "county"})
    else:
        grouped = grouped.rename(columns={group_col: "state"})

    # --- min_home_count masking (sample count, county level only) ---
    _metric_cols = ["adoption_rate_pct"]
    if geo_level == "county":
        below = grouped["_sample_count"] < min_home_count
        grouped.loc[below, _metric_cols] = np.nan

    grouped = grouped.drop(columns=["_sample_count", "_n_adopters"])

    grouped["adoption_rate_pct"] = grouped["adoption_rate_pct"].round(ROUND_RATE_DECIMALS)

    if verbose:
        n_valid = grouped["adoption_rate_pct"].notna().sum()
        nat_rate = grouped["adoption_rate_pct"].mean()
        level_label = "counties" if geo_level == "county" else "states"
        print(
            f"{level_label.title()} with data: {n_valid} / {len(grouped)}\n"
            f"Mean {level_label} adoption rate: {nat_rate:.1f}%"
        )

    # --- Column ordering ---
    if geo_level == "county":
        base_cols = ["county"] + (["state"] if has_state else []) + ["home_count"]
    else:
        base_cols = ["state", "home_count"]
    result_cols = base_cols + _metric_cols
    result = grouped[result_cols].copy()

    return result.sort_values("adoption_rate_pct", ascending=False).reset_index(drop=True)
