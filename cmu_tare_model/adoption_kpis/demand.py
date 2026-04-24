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
    HEATING_FUEL_COLS,
    HP_BACKUP_ELEC_COL,
    HP_FANS_PUMPS_COL,
)


# ============================================================================
# MODULE-LEVEL CONSTANTS
# ============================================================================

ELEC_HEATING_COL: str = "out.electricity.heating.energy_consumption.kwh"
"""EUSS column for electric heating energy consumption (kWh)."""

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
    """Compute per-building heating demand change under 100% adoption scenario.

    Produces two change metrics per home:

    - ``elec_demand_change_kwh``: grid impact (electricity-only change).
      Positive = more grid electricity needed after electrification.
    - ``site_energy_change_kwh``: efficiency (total site energy change).
      Negative = net energy reduction from HP efficiency gains.

    Fan/pump electricity is always included in the retrofit electricity
    total because it is part of the heating system's grid draw.

    Args:
        df_baseline: EUSS baseline DataFrame (indexed by bldg_id).
            Must contain ``in.state``, ``in.heating_fuel``, ``weight``,
            and all columns in ``HEATING_FUEL_COLS`` plus
            ``ELEC_HEATING_COL``.
        df_upgrade: EUSS upgrade DataFrame (indexed by bldg_id,
            already filtered to ``applicability == True``).
            Must contain ``ELEC_HEATING_COL``, ``HP_BACKUP_ELEC_COL``,
            ``HP_FANS_PUMPS_COL``.
        fuel_filter: Filter to this baseline heating fuel string
            (e.g., ``'Natural Gas'``). ``None`` includes all fuel types.
        verbose: If ``True``, print diagnostic summary.

    Returns:
        DataFrame indexed by bldg_id with columns: ``in.state``,
        ``in.heating_fuel``, ``weight``, ``baseline_electric_kwh``,
        ``baseline_heating_total_kwh``, ``retrofit_electric_kwh``,
        ``elec_demand_change_kwh``, ``site_energy_change_kwh``, and
        weighted variants of each energy column.

    Raises:
        KeyError: If required columns are missing from either DataFrame.
    """
    baseline_total = df_baseline[HEATING_FUEL_COLS].sum(axis=1)
    retrofit_total_elec = (
        df_upgrade[ELEC_HEATING_COL].fillna(0)
        + df_upgrade[HP_BACKUP_ELEC_COL].fillna(0)
        + df_upgrade[HP_FANS_PUMPS_COL].fillna(0)
    )

    df_demand = pd.DataFrame({
        'in.state': df_baseline['in.state'],
        'in.heating_fuel': df_baseline['in.heating_fuel'],
        'weight': df_baseline['weight'],
        'baseline_electric_kwh': df_baseline[ELEC_HEATING_COL],
        'baseline_heating_total_kwh': baseline_total,
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
    df_demand['site_energy_change_kwh'] = (
        df_demand['retrofit_electric_kwh'] - df_demand['baseline_heating_total_kwh']
    )

    for col in [
        'baseline_electric_kwh', 'baseline_heating_total_kwh',
        'retrofit_electric_kwh', 'elec_demand_change_kwh',
        'site_energy_change_kwh',
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
# STATE-LEVEL AGGREGATION
# ============================================================================

def aggregate_demand_by_state(
    df_demand: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """Aggregate per-building demand results to state-level totals in GWh.

    Uses EUSS sampling weights to produce population-representative totals.
    Percentage changes are computed relative to the weighted baseline.

    Args:
        df_demand: Per-building demand DataFrame from
            ``compute_scenario_demand()``. Must contain ``in.state``,
            ``weight``, and the ``weighted_*`` columns produced by that
            function.
        verbose: If ``True``, print state-level summary.

    Returns:
        DataFrame sorted by ``elec_change_gwh`` (descending) with columns:
        ``state``, ``home_count``, ``baseline_elec_gwh``,
        ``baseline_total_gwh``, ``retrofit_elec_gwh``,
        ``elec_change_gwh``, ``pct_elec_demand_change``,
        ``site_energy_change_gwh``, ``pct_site_energy_change``.

    Raises:
        KeyError: If expected weighted columns are missing from ``df_demand``.
    """
    grouped = df_demand.groupby('in.state').agg(
        home_count=('weight', 'size'),
        weighted_baseline_elec=('weighted_baseline_electric_kwh', 'sum'),
        weighted_baseline_total=('weighted_baseline_heating_total_kwh', 'sum'),
        weighted_retrofit_elec=('weighted_retrofit_electric_kwh', 'sum'),
        weighted_elec_change=('weighted_elec_demand_change_kwh', 'sum'),
        weighted_site_change=('weighted_site_energy_change_kwh', 'sum'),
    ).reset_index().rename(columns={'in.state': 'state'})

    grouped['baseline_elec_gwh'] = grouped['weighted_baseline_elec'] / KWH_TO_GWH
    grouped['baseline_total_gwh'] = grouped['weighted_baseline_total'] / KWH_TO_GWH
    grouped['retrofit_elec_gwh'] = grouped['weighted_retrofit_elec'] / KWH_TO_GWH
    grouped['elec_change_gwh'] = grouped['weighted_elec_change'] / KWH_TO_GWH
    grouped['site_energy_change_gwh'] = grouped['weighted_site_change'] / KWH_TO_GWH

    grouped['pct_elec_demand_change'] = np.where(
        grouped['weighted_baseline_elec'] != 0,
        grouped['weighted_elec_change'] / grouped['weighted_baseline_elec'] * 100,
        np.nan,
    )
    grouped['pct_site_energy_change'] = np.where(
        grouped['weighted_baseline_total'] != 0,
        grouped['weighted_site_change'] / grouped['weighted_baseline_total'] * 100,
        np.nan,
    )

    # Demand accounting check
    total_sum = df_demand['weighted_elec_demand_change_kwh'].sum()
    total_agg = grouped['weighted_elec_change'].sum()
    if not np.isclose(total_sum, total_agg, rtol=1e-6):
        print(f"⚠ DEMAND ACCOUNTING MISMATCH: sum={total_sum:.0f}, agg={total_agg:.0f}")
    elif verbose:
        print("✓ Demand accounting check passed")

    result = grouped[[
        'state', 'home_count',
        'baseline_elec_gwh', 'baseline_total_gwh', 'retrofit_elec_gwh',
        'elec_change_gwh', 'pct_elec_demand_change',
        'site_energy_change_gwh', 'pct_site_energy_change',
    ]].copy()

    for col in [
        'baseline_elec_gwh', 'baseline_total_gwh', 'retrofit_elec_gwh',
        'elec_change_gwh', 'site_energy_change_gwh',
    ]:
        result[col] = result[col].round(2)
    result['pct_elec_demand_change'] = result['pct_elec_demand_change'].round(2)
    result['pct_site_energy_change'] = result['pct_site_energy_change'].round(2)

    if verbose:
        print(f"\n--- State-Level Demand Summary ---")
        print(f"States: {len(result)}")
        print(f"Total elec demand change:    {result['elec_change_gwh'].sum():+.1f} GWh (grid impact)")
        print(f"Total site energy change:    {result['site_energy_change_gwh'].sum():+.1f} GWh (efficiency)")

    return result.sort_values('elec_change_gwh', ascending=False).reset_index(drop=True)
