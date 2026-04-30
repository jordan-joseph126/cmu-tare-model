"""
Bill savings ratio computation functions for the TARE model adoption KPIs.

Provides per-building and state-level aggregation of heating bill savings
ratios using actual TARE model lifetime fuel cost outputs. Unlike the
retired analytical formula (bill_impact_ratio = spark_gap × AFUE / COP),
these functions use the per-building lifetime fuel costs already computed
by the TARE model, which incorporate building-specific consumption, state
fuel prices, and system efficiency.

Bill savings ratio definition:
    ratio = retrofit_lifetime_fuel_cost / baseline_lifetime_fuel_cost

    ratio < 1  → heat pump saves money over the equipment lifetime
    ratio > 1  → heat pump costs more over the equipment lifetime
    ratio = 1  → break-even

Location: cmu_tare_model/adoption_kpis/bill_savings.py
"""

from typing import Optional

import numpy as np
import pandas as pd

from cmu_tare_model.adoption_kpis.data_loading import COUNTY_COL


# ============================================================================
# MODULE-LEVEL CONSTANTS
# ============================================================================

BASELINE_COST_COL: str = "baseline_heating_lifetime_fuel_cost"
"""TARE output column for baseline furnace lifetime fuel cost (shared across MPs)."""

POLICY_SCENARIOS: tuple[str, ...] = ("iraRef", "preIRA")
"""Valid policy scenario identifiers for fuel cost column lookup."""

ROUND_RATIO_DECIMALS: int = 3
"""Decimal places for rounding bill savings ratio columns."""

ROUND_COST_DECIMALS: int = 2
"""Decimal places for rounding lifetime cost columns."""

# Mapping from raw EUSS column names → processed df_enduse column names.
# The processed DataFrame (from df_enduse_refactored) renames these columns.
# Used to resolve metadata columns when df_euss is the processed version.
_EUSS_COL_ALIASES: dict[str, str] = {
    "in.state": "state",
    "in.heating_fuel": "base_heating_fuel",
}
"""Aliases for EUSS metadata columns renamed by df_enduse_refactored."""


# ============================================================================
# PER-BUILDING BILL SAVINGS RATIO
# ============================================================================

def compute_bill_savings_ratio(
    df_tare: pd.DataFrame,
    mp: int,
    policy_scenario: str = "iraRef",
    fuel_filter: Optional[str] = "Natural Gas",
    verbose: bool = False,
    df_euss: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute per-building bill savings ratio from TARE lifetime fuel costs.

    ``ratio = retrofit_lifetime_fuel_cost / baseline_lifetime_fuel_cost``

    Ratio < 1 = HP saves money; ratio > 1 = HP costs more.

    Unlike the retired analytical ``bill_impact_ratio``, this uses actual
    per-building lifetime fuel costs from the TARE model that already
    incorporate building-specific consumption, state fuel prices, and
    system efficiency.

    The TARE summary DataFrame does not include ALL EUSS metadata. Pass ``df_euss`` (the
    EUSS baseline DataFrame indexed by ``bldg_id``) to supply ``in.state``, ``in.heating_fuel``, and ``weight`` via index join.

    Args:
        df_tare: TARE model output DataFrame (from
            ``DATAFRAMES_BY_MP[mp]['fixed_base']['inmap']``).
            Must contain ``baseline_heating_lifetime_fuel_cost`` and
            ``{policy_scenario}_mp{mp}_heating_lifetime_fuel_cost``.
            May also contain ``in.state``, ``in.heating_fuel``, ``weight``
            directly; if absent, ``df_euss`` must be provided.
        mp: Measure package number (e.g., 3 or 4).
        policy_scenario: ``'iraRef'`` for the IRA reference case or
            ``'preIRA'`` for the no-IRA scenario. Determines which
            retrofit fuel cost column is used.
        fuel_filter: Baseline heating fuel to filter by
            (e.g., ``'Natural Gas'``). Set to ``None`` to include all fuels.
        verbose: If ``True``, print record count, median ratio, mean ratio.
        df_euss: EUSS baseline DataFrame (indexed by ``bldg_id``) used to
            supply ``in.state``, ``in.heating_fuel``, and ``weight`` when
            those columns are absent from ``df_tare``. Required when
            ``df_tare`` is a TARE model summary output.

    Returns:
        DataFrame indexed by bldg_id with columns: ``in.state``,
        ``in.heating_fuel``, ``weight``,
        ``baseline_lifetime_fuel_cost``,
        ``retrofit_lifetime_fuel_cost``, ``bill_savings_ratio``.

    Raises:
        ValueError: If ``policy_scenario`` is not ``'iraRef'`` or ``'preIRA'``.
        KeyError: If the required fuel cost columns are missing from ``df_tare``.
        ValueError: If metadata columns are absent from ``df_tare`` and
            ``df_euss`` is not provided.
    """
    if policy_scenario not in POLICY_SCENARIOS:
        raise ValueError(
            f"policy_scenario must be one of {POLICY_SCENARIOS!r}, "
            f"got {policy_scenario!r}"
        )

    baseline_col = BASELINE_COST_COL
    retrofit_col = f"{policy_scenario}_mp{mp}_heating_lifetime_fuel_cost"

    missing = [c for c in (baseline_col, retrofit_col) if c not in df_tare.columns]
    if missing:
        available = [c for c in df_tare.columns if "fuel_cost" in c]
        raise KeyError(
            f"Required column(s) not found in df_tare: {missing}. "
            f"Available fuel cost columns: {available}"
        )

    # Determine source of metadata columns (may not be in the TARE summary CSV)
    _META_COLS = ["in.state", "in.heating_fuel", "weight"]
    _has_meta = all(c in df_tare.columns for c in _META_COLS)

    if _has_meta:
        df_out = df_tare[_META_COLS].copy()
    else:
        if df_euss is None:
            raise ValueError(
                f"Columns {_META_COLS} not found in df_tare and df_euss was not provided. "
                "Pass the EUSS baseline DataFrame as df_euss to supply these columns."
            )
        # Resolve column names: try raw EUSS name first, then processed alias
        col_map: dict[str, str] = {}  # canonical name → actual name in df_euss
        for canonical in _META_COLS:
            if canonical in df_euss.columns:
                col_map[canonical] = canonical
            elif canonical in _EUSS_COL_ALIASES and _EUSS_COL_ALIASES[canonical] in df_euss.columns:
                col_map[canonical] = _EUSS_COL_ALIASES[canonical]
        missing_meta = [c for c in _META_COLS if c not in col_map]
        if missing_meta:
            available = list(df_euss.columns)
            raise KeyError(
                f"df_euss is missing required columns: {missing_meta}. "
                f"Looked for aliases {[_EUSS_COL_ALIASES.get(c) for c in missing_meta]}. "
                f"Available columns (first 20): {available[:20]}"
            )
        # Extract and rename to canonical names
        df_meta = df_euss[[col_map[c] for c in _META_COLS]].copy()
        df_meta.columns = _META_COLS
        # Align on shared index (inner join)
        df_out = df_meta.join(df_tare[[baseline_col, retrofit_col]], how="inner")
        df_out = df_out.rename(columns={
            baseline_col: "_baseline_tmp",
            retrofit_col: "_retrofit_tmp",
        })

    if _has_meta:
        df_out["baseline_lifetime_fuel_cost"] = df_tare[baseline_col]
        df_out["retrofit_lifetime_fuel_cost"] = df_tare[retrofit_col]
    else:
        df_out["baseline_lifetime_fuel_cost"] = df_out.pop("_baseline_tmp")
        df_out["retrofit_lifetime_fuel_cost"] = df_out.pop("_retrofit_tmp")

    if fuel_filter is not None:
        n_before = len(df_out)
        df_out = df_out[df_out["in.heating_fuel"] == fuel_filter]
        if verbose:
            print(
                f"Filtered to '{fuel_filter}': "
                f"{len(df_out):,} / {n_before:,} homes"
            )

    df_out["bill_savings_ratio"] = np.where(
        df_out["baseline_lifetime_fuel_cost"] > 0,
        df_out["retrofit_lifetime_fuel_cost"] / df_out["baseline_lifetime_fuel_cost"],
        np.nan,
    )

    if verbose:
        n_valid = df_out["bill_savings_ratio"].notna().sum()
        print(f"Per-building records (non-null ratio): {n_valid:,}")
        print(f"Median bill savings ratio: {df_out['bill_savings_ratio'].median():.3f}")
        print(f"Mean bill savings ratio:   {df_out['bill_savings_ratio'].mean():.3f}")

    # Attach county for downstream county-level aggregation
    if COUNTY_COL not in df_out.columns:
        if df_euss is not None and COUNTY_COL in df_euss.columns:
            df_out = df_out.join(df_euss[[COUNTY_COL]], how='left')
        elif COUNTY_COL in df_tare.columns:
            df_out[COUNTY_COL] = df_tare[COUNTY_COL]

    return df_out


# ============================================================================
# STATE-LEVEL AGGREGATION
# ============================================================================

# ============================================================================
# AGGREGATION (STATE OR COUNTY)
# ============================================================================

def aggregate_bill_savings(
    df_ratio: pd.DataFrame,
    geo_level: str = 'state',
    min_home_count: int = 30,
    verbose: bool = False,
) -> pd.DataFrame:
    """Aggregate per-building bill savings ratios to state- or county-level summary.

    Uses weighted MEDIAN as the primary statistic — robust to outlier buildings
    with extreme consumption patterns. Mean and total_cost_ratio are
    included as cross-checks only.

    Args:
        df_ratio: Per-building DataFrame from
            ``compute_bill_savings_ratio()``. Must contain ``in.state``,
            ``weight``, ``baseline_lifetime_fuel_cost``,
            ``retrofit_lifetime_fuel_cost``, ``bill_savings_ratio``.
            For county-level, must also contain ``in.county``
            (automatically attached by ``compute_bill_savings_ratio``
            when ``df_euss`` is provided).
        geo_level: ``'state'`` (default) or ``'county'``. When ``'county'``,
            groups by ``in.county`` and includes ``in.state`` in the output.
        min_home_count: Minimum sample buildings per county for metric values.
            Counties below this threshold have metric columns set to ``NaN``.
            Threshold is applied to sample count (not weighted population)
            to ensure statistical reliability of the estimate.
            Only applied when ``geo_level='county'``.
        verbose: If ``True``, print count of states/counties where median < 1
            and the national median ratio.

    Returns:
        DataFrame sorted by ``median_bill_savings_ratio`` (ascending) with
        columns: ``state`` (or ``county`` + ``state``), ``home_count``
        (weighted population), ``median_bill_savings_ratio``,
        ``mean_bill_savings_ratio``, ``total_baseline_cost``,
        ``total_retrofit_cost``, ``total_cost_ratio``.

    Raises:
        ValueError: If ``geo_level`` is not ``'state'`` or ``'county'``.
        KeyError: If ``in.county`` is missing when ``geo_level='county'``.
    """
    if geo_level not in ('state', 'county'):
        raise ValueError(f"geo_level must be 'state' or 'county', got {geo_level!r}")

    group_col = COUNTY_COL if geo_level == 'county' else 'in.state'

    if geo_level == 'county' and COUNTY_COL not in df_ratio.columns:
        raise KeyError(
            f"Column '{COUNTY_COL}' not found in df_ratio. "
            "Pass df_euss to compute_bill_savings_ratio() to attach county info."
        )

    # Sample count (for quality threshold) + weighted home_count (for display)
    grouped = df_ratio.groupby(group_col).agg(
        _sample_count=("bill_savings_ratio", "size"),
        home_count=("weight", "sum"),
        mean_bill_savings_ratio=("bill_savings_ratio", "mean"),
    ).reset_index()

    # Weighted median: sorted cumulative weights, no external dependencies
    def _wmedian(grp):
        mask = grp['bill_savings_ratio'].notna()
        vals = grp.loc[mask, 'bill_savings_ratio'].to_numpy(dtype=float)
        wts = grp.loc[mask, 'weight'].to_numpy(dtype=float)
        if len(vals) == 0:
            return np.nan
        order = np.argsort(vals)
        cumwt = np.cumsum(wts[order])
        return float(vals[order][np.searchsorted(cumwt, cumwt[-1] / 2.0)])

    wmed = (
        df_ratio.groupby(group_col)
        .apply(_wmedian)
        .rename('median_bill_savings_ratio')
        .reset_index()
    )
    grouped = grouped.merge(wmed, on=group_col, how='left')

    if geo_level == 'county':
        state_lookup = df_ratio.groupby(COUNTY_COL)['in.state'].first()
        grouped['state'] = grouped[COUNTY_COL].map(state_lookup)
        grouped = grouped.rename(columns={COUNTY_COL: 'county'})
    else:
        grouped = grouped.rename(columns={'in.state': 'state'})

    # Weighted cost totals for total_cost_ratio cross-check
    weighted_agg = df_ratio.copy()
    weighted_agg["_weighted_baseline"] = (
        weighted_agg["baseline_lifetime_fuel_cost"] * weighted_agg["weight"]
    )
    weighted_agg["_weighted_retrofit"] = (
        weighted_agg["retrofit_lifetime_fuel_cost"] * weighted_agg["weight"]
    )
    totals = weighted_agg.groupby(group_col).agg(
        total_baseline_cost=("_weighted_baseline", "sum"),
        total_retrofit_cost=("_weighted_retrofit", "sum"),
    ).reset_index()
    if geo_level == 'county':
        totals = totals.rename(columns={COUNTY_COL: 'county'})
    else:
        totals = totals.rename(columns={'in.state': 'state'})

    merge_col = 'county' if geo_level == 'county' else 'state'
    grouped = grouped.merge(totals, on=merge_col, how="left")

    # total_cost_ratio = Σ(w×retrofit_cost) / Σ(w×baseline_cost)
    # Aggregate cost ratio — NOT the weighted average of per-building ratios
    grouped["total_cost_ratio"] = np.where(
        grouped["total_baseline_cost"] > 0,
        grouped["total_retrofit_cost"] / grouped["total_baseline_cost"],
        np.nan,
    )

    _metric_cols = ["median_bill_savings_ratio", "mean_bill_savings_ratio",
                    "total_cost_ratio", "total_baseline_cost", "total_retrofit_cost"]

    if geo_level == 'county':
        below = grouped['_sample_count'] < min_home_count
        grouped.loc[below, _metric_cols] = np.nan

    grouped = grouped.drop(columns=['_sample_count'])

    for col in ("median_bill_savings_ratio", "mean_bill_savings_ratio", "total_cost_ratio"):
        grouped[col] = grouped[col].round(ROUND_RATIO_DECIMALS)
    for col in ("total_baseline_cost", "total_retrofit_cost"):
        grouped[col] = grouped[col].round(ROUND_COST_DECIMALS)

    if verbose:
        n_savings = (grouped["median_bill_savings_ratio"] < 1.0).sum()
        level_label = 'counties' if geo_level == 'county' else 'states'
        national_median = grouped["median_bill_savings_ratio"].median()
        print(f"{level_label.title()} where median home saves money (ratio < 1): {n_savings} / {len(grouped)}")
        print(f"National median of {level_label} medians: {national_median:.3f}")

    return grouped.sort_values("median_bill_savings_ratio", ascending=True).reset_index(drop=True)


# Backward-compatible alias
aggregate_bill_savings_by_state = aggregate_bill_savings
