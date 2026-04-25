"""
Heating demand change computation functions for the TARE model adoption KPIs.

Provides per-building and state-level aggregation of electricity demand
change and site energy change under 100% heat pump adoption scenarios.
These metrics capture the grid impact (electricity demand change) and
efficiency benefit (site energy change) of widespread electrification.

Location: cmu_tare_model/adoption_kpis/demand.py
"""

from typing import Optional

import numpy as np
import pandas as pd

from cmu_tare_model.adoption_kpis.data_loading import (
    ELEC_TOTAL_COL,
    COUNTY_COL,
)


# ============================================================================
# MODULE-LEVEL CONSTANTS
# ============================================================================

KWH_TO_GWH: float = 1e6
"""Divisor to convert kWh to GWh."""


# ============================================================================
# PER-BUILDING DEMAND CHANGE
# ============================================================================

def compute_scenario_demand(
    df_baseline: pd.DataFrame,
    df_upgrade: pd.DataFrame,
    fuel_filter: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute per-building electricity demand change under 100% adoption scenario.

    Uses total residential electricity consumption (all end uses) rather than 
    heating-only columns, so the percent change reflects the true grid impact
    of whole-home electrification. For EUSS MP3 and MP4, this change is specific to heating electrification. 

    - ``elec_demand_change_kwh``: total electricity change (retrofit − baseline).
      Positive = more grid electricity needed after electrification.
    - ``site_energy_change_kwh``: alias for ``elec_demand_change_kwh`` (with total
      electricity, the two concepts converge; retained for backward compatibility).

    Args:
        df_baseline: EUSS baseline DataFrame (indexed by bldg_id).
            Must contain ``in.state``, ``in.county``, ``in.heating_fuel``,
            ``weight``, and ``ELEC_TOTAL_COL``.
        df_upgrade: EUSS upgrade DataFrame (indexed by bldg_id,
            already filtered to ``applicability == True``).
            Must contain ``ELEC_TOTAL_COL``.
        fuel_filter: Filter to this baseline heating fuel string
            (e.g., ``'Natural Gas'``). ``None`` includes all fuel types.
        verbose: If ``True``, print diagnostic summary.

    Returns:
        DataFrame indexed by bldg_id with columns: ``in.state``,
        ``in.county``, ``in.heating_fuel``, ``weight``,
        ``baseline_electric_kwh``, ``retrofit_electric_kwh``,
        ``elec_demand_change_kwh``, ``site_energy_change_kwh``, and
        ``weighted_*`` variants.

    Raises:
        KeyError: If required columns are missing from either DataFrame.
    """
    baseline_total_elec = df_baseline[ELEC_TOTAL_COL].fillna(0)
    retrofit_total_elec = df_upgrade[ELEC_TOTAL_COL].fillna(0)

    df_demand = pd.DataFrame({
        'in.state': df_baseline['in.state'],
        'in.county': df_baseline[COUNTY_COL],
        'in.heating_fuel': df_baseline['in.heating_fuel'],
        'weight': df_baseline['weight'],
        'baseline_electric_kwh': baseline_total_elec,
    }).join(
        retrofit_total_elec.rename('retrofit_electric_kwh'),
        how='inner',
    )

    if fuel_filter is not None:
        n_before = len(df_demand)
        df_demand = df_demand[df_demand['in.heating_fuel'] == fuel_filter]
        if verbose:
            print(f"Filtered to '{fuel_filter}': {len(df_demand):,} / {n_before:,} homes")

    df_demand['elec_demand_change_kwh'] = (
        df_demand['retrofit_electric_kwh'] - df_demand['baseline_electric_kwh']
    )
    # site_energy_change_kwh == elec_demand_change_kwh when using total electricity
    df_demand['site_energy_change_kwh'] = df_demand['elec_demand_change_kwh']

    for col in [
        'baseline_electric_kwh', 'retrofit_electric_kwh',
        'elec_demand_change_kwh', 'site_energy_change_kwh',
    ]:
        df_demand[f'weighted_{col}'] = df_demand[col] * df_demand['weight']

    if verbose:
        fuel_label = fuel_filter if fuel_filter else 'all fuels'
        print(f"\n--- Demand Scenario Summary (100% adoption, {fuel_label}) ---")
        print(f"Total homes: {len(df_demand):,}")
        elec_gwh = df_demand['weighted_elec_demand_change_kwh'].sum() / KWH_TO_GWH
        site_gwh = df_demand['weighted_site_energy_change_kwh'].sum() / KWH_TO_GWH
        print(f"Weighted electricity demand change:  {elec_gwh:+,.1f} GWh (grid impact)")
        print(f"Weighted total site energy change:   {site_gwh:+,.1f} GWh (efficiency)")

    return df_demand


# ============================================================================
# AGGREGATION (STATE OR COUNTY)
# ============================================================================

def aggregate_demand(
    df_demand: pd.DataFrame,
    geo_level: str = 'state',
    min_home_count: int = 30,
    verbose: bool = False,
) -> pd.DataFrame:
    """Aggregate per-building demand results to state- or county-level GWh totals.

    Uses EUSS sampling weights to produce population-representative totals.
    Percentage changes are computed at the aggregate level:
    ``(Σ weighted_retrofit - Σ weighted_baseline) / Σ weighted_baseline × 100``.

    Args:
        df_demand: Per-building demand DataFrame from
            ``compute_scenario_demand()``. Must contain ``in.state``,
            ``in.county``, ``weight``, and the ``weighted_*`` columns.
        geo_level: ``'state'`` (default) or ``'county'``. When ``'county'``,
            groups by ``in.county`` and includes ``in.state`` in the output.
        min_home_count: Minimum number of homes required for a county to
            receive metric values. Counties below this threshold have metric
            columns set to ``NaN``. Only applied when ``geo_level='county'``.
        verbose: If ``True``, print state/county-level summary.

    Returns:
        DataFrame sorted by ``elec_change_gwh`` (descending). State-level
        columns: ``state``, ``home_count``, ``baseline_elec_gwh``,
        ``retrofit_elec_gwh``, ``elec_change_gwh``, ``pct_elec_demand_change``,
        ``site_energy_change_gwh``, ``pct_site_energy_change``.
        County-level also includes ``county`` and ``state``.

    Raises:
        ValueError: If ``geo_level`` is not ``'state'`` or ``'county'``.
        KeyError: If expected weighted columns are missing from ``df_demand``.
    """
    if geo_level not in ('state', 'county'):
        raise ValueError(f"geo_level must be 'state' or 'county', got {geo_level!r}")

    group_col = 'in.county' if geo_level == 'county' else 'in.state'

    grouped = df_demand.groupby(group_col).agg(
        home_count=('weight', 'size'),
        weighted_baseline_elec=('weighted_baseline_electric_kwh', 'sum'),
        weighted_retrofit_elec=('weighted_retrofit_electric_kwh', 'sum'),
        weighted_elec_change=('weighted_elec_demand_change_kwh', 'sum'),
        weighted_site_change=('weighted_site_energy_change_kwh', 'sum'),
    ).reset_index()

    if geo_level == 'county':
        state_lookup = df_demand.groupby('in.county')['in.state'].first()
        grouped['state'] = grouped['in.county'].map(state_lookup)
        grouped = grouped.rename(columns={'in.county': 'county'})
    else:
        grouped = grouped.rename(columns={'in.state': 'state'})

    grouped['baseline_elec_gwh'] = grouped['weighted_baseline_elec'] / KWH_TO_GWH
    grouped['retrofit_elec_gwh'] = grouped['weighted_retrofit_elec'] / KWH_TO_GWH
    grouped['elec_change_gwh'] = grouped['weighted_elec_change'] / KWH_TO_GWH
    grouped['site_energy_change_gwh'] = grouped['weighted_site_change'] / KWH_TO_GWH

    grouped['pct_elec_demand_change'] = np.where(
        grouped['weighted_baseline_elec'] != 0,
        grouped['weighted_elec_change'] / grouped['weighted_baseline_elec'] * 100,
        np.nan,
    )
    grouped['pct_site_energy_change'] = grouped['pct_elec_demand_change']

    _metric_cols = [
        'baseline_elec_gwh', 'retrofit_elec_gwh', 'elec_change_gwh',
        'site_energy_change_gwh', 'pct_elec_demand_change', 'pct_site_energy_change',
    ]

    if geo_level == 'county':
        below = grouped['home_count'] < min_home_count
        grouped.loc[below, _metric_cols] = np.nan

    # Demand accounting check (state level only, totals still hold)
    total_sum = df_demand['weighted_elec_demand_change_kwh'].sum()
    total_agg = grouped['weighted_elec_change'].sum()
    if not np.isclose(total_sum, total_agg, rtol=1e-6):
        print(f"⚠ DEMAND ACCOUNTING MISMATCH: sum={total_sum:.0f}, agg={total_agg:.0f}")
    elif verbose:
        print("✓ Demand accounting check passed")

    for col in ['baseline_elec_gwh', 'retrofit_elec_gwh', 'elec_change_gwh', 'site_energy_change_gwh']:
        grouped[col] = grouped[col].round(2)
    grouped['pct_elec_demand_change'] = grouped['pct_elec_demand_change'].round(2)
    grouped['pct_site_energy_change'] = grouped['pct_site_energy_change'].round(2)

    if geo_level == 'county':
        result_cols = ['county', 'state', 'home_count'] + _metric_cols
    else:
        result_cols = ['state', 'home_count'] + _metric_cols

    result = grouped[result_cols].copy()

    if verbose:
        level_label = 'counties' if geo_level == 'county' else 'states'
        print(f"\n--- {geo_level.title()}-Level Demand Summary ---")
        print(f"{level_label.title()}: {len(result)}")
        print(f"Total elec demand change:    {result['elec_change_gwh'].sum():+.1f} GWh (grid impact)")

    return result.sort_values('elec_change_gwh', ascending=False).reset_index(drop=True)


# Backward-compatible alias
aggregate_demand_by_state = aggregate_demand
