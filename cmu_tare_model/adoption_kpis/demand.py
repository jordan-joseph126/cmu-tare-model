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
    DWELLING_UNIT_WEIGHT,
    ELEC_TOTAL_COL,
    ELEC_TOTAL_SAVINGS_COL,
    SITE_ENERGY_TOTAL_COL,
    SITE_ENERGY_TOTAL_SAVINGS_COL,
    HEATING_ELEC_SAVINGS_COLS,
    HEATING_FUEL_COL,
    COOLING_ELEC_SAVINGS_COLS,
    HOT_WATER_ELEC_SAVINGS_COL,
    COUNTY_COL,
    STATE_COL,
)
from cmu_tare_model.constants import MIN_HOME_COUNT


# ============================================================================
# MODULE-LEVEL CONSTANTS
# ============================================================================

KWH_TO_GWH: float = 1e6
"""Divisor to convert kWh to GWh."""

DEMAND_CHANGE_TOLERANCE_KWH: float = 1.0
"""Largest gap allowed in one home between ResStock's whole-home savings column
and retrofit minus baseline (kWh). Far above what number storage can explain."""


# ============================================================================
# PER-BUILDING DEMAND CHANGE
# ============================================================================

def compute_scenario_demand(
    df_baseline: pd.DataFrame,
    df_upgrade: pd.DataFrame,
    sample_bldg_ids: pd.Index,
    fuel_filter: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute per-building electricity demand change under 100% adoption scenario.

    Uses total residential electricity consumption (all end uses) rather than 
    heating-only columns, so the percent change reflects the true grid impact
    of whole-home electrification. For EUSS MP3 and MP4, this change is specific to heating electrification. 

    Every change is read from ResStock's own savings column in the upgrade
    file (savings = baseline minus upgrade, so change = minus savings):

    - ``elec_demand_change_kwh``: total electricity change (retrofit - baseline).
      Positive = more grid electricity needed after electrification.
    - ``heating_elec_change_kwh``, ``cooling_elec_change_kwh``,
      ``hot_water_elec_change_kwh``: the same change for one end use (heating
      and cooling include their fans, pumps, and backup heat).
      ``other_elec_change_kwh`` is whatever is left of the total.
    - ``site_energy_change_kwh``: total site energy change, all fuels
      (retrofit - baseline). Negative = less energy used overall after
      electrification, because the fossil fuel is no longer burned.

    Args:
        df_baseline: EUSS baseline DataFrame (indexed by bldg_id).
            Must contain ``STATE_COL``, ``COUNTY_COL``, ``HEATING_FUEL_COL``,
            ``DWELLING_UNIT_WEIGHT``, ``ELEC_TOTAL_COL``, and
            ``SITE_ENERGY_TOTAL_COL`` (this release's names; data_loading.py).
        df_upgrade: EUSS upgrade DataFrame (indexed by bldg_id,
            already filtered to ``applicability == True``).
            Must contain ``ELEC_TOTAL_COL``, ``SITE_ENERGY_TOTAL_COL``, and
            the savings columns named in data_loading.py.
        sample_bldg_ids: The study sample, TARE_SAMPLE_IDS['all'] (or one
            county's list from TARE_SAMPLE_IDS['by_county']). Only these homes
            are counted, so the demand maps describe the same homes as the
            adoption results.
        fuel_filter: Filter to this baseline heating fuel string
            (e.g., ``'Natural Gas'``). ``None`` includes all fuel types.
        verbose: If ``True``, print diagnostic summary.

    Returns:
        DataFrame indexed by bldg_id with columns: ``in.state``,
        ``in.county``, ``in.heating_fuel``, ``weight``,
        ``baseline_electric_kwh``, ``retrofit_electric_kwh``,
        ``elec_demand_change_kwh``, ``heating_elec_change_kwh``,
        ``cooling_elec_change_kwh``, ``hot_water_elec_change_kwh``,
        ``other_elec_change_kwh``, ``baseline_site_energy_kwh``,
        ``retrofit_site_energy_kwh``, ``site_energy_change_kwh``, and
        ``weighted_*`` variants.

    Raises:
        KeyError: If required columns are missing from either DataFrame.
        TypeError: If sample_bldg_ids is not a pd.Index.
        ValueError: If sample_bldg_ids is empty; a sample home is missing
            from the baseline or upgrade frame, or has a blank energy value;
            or ResStock's whole-home savings disagree with retrofit minus
            baseline by more than DEMAND_CHANGE_TOLERANCE_KWH.
    """
    if not isinstance(sample_bldg_ids, pd.Index):
        raise TypeError(
            f"sample_bldg_ids must be a pd.Index, got {type(sample_bldg_ids)}")
    if sample_bldg_ids.empty:
        raise ValueError("sample_bldg_ids is empty")

    # Step 1 -- retrofit totals, and each change read from ResStock's own
    # savings columns. Savings are baseline minus upgrade, so the change is
    # 0 - savings (written that way so a zero does not become -0.0).
    df_changes = pd.DataFrame({
        'retrofit_electric_kwh': df_upgrade[ELEC_TOTAL_COL],
        'retrofit_site_energy_kwh': df_upgrade[SITE_ENERGY_TOTAL_COL],
        'elec_demand_change_kwh': 0.0 - df_upgrade[ELEC_TOTAL_SAVINGS_COL],
        'heating_elec_change_kwh': 0.0 - df_upgrade[
            HEATING_ELEC_SAVINGS_COLS].sum(axis=1, skipna=False),
        'cooling_elec_change_kwh': 0.0 - df_upgrade[
            COOLING_ELEC_SAVINGS_COLS].sum(axis=1, skipna=False),
        'hot_water_elec_change_kwh': 0.0 - df_upgrade[HOT_WATER_ELEC_SAVINGS_COL],
        # Site energy counts every fuel (gas/oil/propane in kWh-equivalent), so
        # it is NOT the electricity change: electricity rises when a fossil
        # system is replaced, while site energy falls because the fuel is no
        # longer burned.
        'site_energy_change_kwh': 0.0 - df_upgrade[SITE_ENERGY_TOTAL_SAVINGS_COL],
    })
    # The ResStock columns are read under this release's names (data_loading.py)
    # and kept under fixed labels, so the code below and the county tables are
    # the same for both releases.
    df_demand = pd.DataFrame({
        'in.state': df_baseline[STATE_COL],
        'in.county': df_baseline[COUNTY_COL],
        'in.heating_fuel': df_baseline[HEATING_FUEL_COL],
        'weight': df_baseline[DWELLING_UNIT_WEIGHT],
        'baseline_electric_kwh': df_baseline[ELEC_TOTAL_COL],
        'baseline_site_energy_kwh': df_baseline[SITE_ENERGY_TOTAL_COL],
    }).join(df_changes, how='inner')

    # Step 2 -- keep only the study sample. A sample home missing here means
    # the ids and the ResStock files disagree (e.g., a different release), so
    # stop.
    missing = sample_bldg_ids.difference(df_demand.index)
    if not missing.empty:
        raise ValueError(
            f"{len(missing):,} sample homes are missing from the baseline or "
            f"upgrade frame (first few: {list(missing[:5])})")
    df_demand = df_demand.loc[sample_bldg_ids]

    # Step 3 -- stop on a blank, which would otherwise count as zero use
    level_cols = [
        'baseline_electric_kwh', 'retrofit_electric_kwh',
        'baseline_site_energy_kwh', 'retrofit_site_energy_kwh']
    change_cols = [
        'elec_demand_change_kwh', 'heating_elec_change_kwh',
        'cooling_elec_change_kwh', 'hot_water_elec_change_kwh',
        'site_energy_change_kwh']
    blank_counts = df_demand[level_cols + change_cols].isna().sum()
    if blank_counts.any():
        raise ValueError(
            "Sample homes have blank energy values (column: rdu): "
            f"{blank_counts[blank_counts > 0].to_dict()}")

    # Step 4 -- ResStock's whole-home savings must agree with retrofit minus
    # baseline, the way the change used to be computed
    for change_col, quantity in (('elec_demand_change_kwh', 'electric'),
                                 ('site_energy_change_kwh', 'site_energy')):
        retrofit_minus_baseline = (
            df_demand[f'retrofit_{quantity}_kwh']
            - df_demand[f'baseline_{quantity}_kwh'])
        largest_gap = (df_demand[change_col] - retrofit_minus_baseline).abs().max()
        if largest_gap > DEMAND_CHANGE_TOLERANCE_KWH:
            raise ValueError(
                f"{change_col}: ResStock's savings column disagrees with "
                f"retrofit minus baseline by up to {largest_gap:,.3f} kWh in a home")

    # Step 5 -- everything outside heating, cooling, and hot water
    df_demand['other_elec_change_kwh'] = (
        df_demand['elec_demand_change_kwh']
        - df_demand['heating_elec_change_kwh']
        - df_demand['cooling_elec_change_kwh']
        - df_demand['hot_water_elec_change_kwh'])

    if fuel_filter is not None:
        n_before = len(df_demand)
        df_demand = df_demand[df_demand['in.heating_fuel'] == fuel_filter]
        if verbose:
            print(f"Filtered to '{fuel_filter}': {len(df_demand):,} / {n_before:,} rdu")

    for col in level_cols + change_cols + ['other_elec_change_kwh']:
        df_demand[f'weighted_{col}'] = df_demand[col] * df_demand['weight']

    if verbose:
        fuel_label = fuel_filter if fuel_filter else 'all fuels'
        print(f"\n--- Demand Scenario Summary (100% adoption, {fuel_label}) ---")
        print(f"Study-sample rdu: {len(df_demand):,}")
        print(build_demand_breakdown(df_demand).to_string(
            float_format='{:,.3f}'.format))

    return df_demand


# ============================================================================
# DEMAND BREAKDOWN (ONE PACKAGE)
# ============================================================================

def build_demand_breakdown(df_demand: pd.DataFrame) -> pd.Series:
    """Summarize one package's demand change at 100% adoption, in GWh.

    Rows, top to bottom: baseline electricity; the net change in heating,
    cooling, hot water, and all other end uses; post-retrofit electricity and
    the net change; the heating change split by the fuel each home heated
    with before the retrofit; then all-fuel site energy (baseline, change,
    and percent change). Put two packages side by side with
    ``pd.DataFrame({'MP3': ..., 'MP4': ...})``.

    Args:
        df_demand: Per-home frame from ``compute_scenario_demand()``.

    Returns:
        Weighted totals in GWh, indexed by row label. The last row is a
        percent, not GWh.

    Raises:
        KeyError: If df_demand lacks a column ``compute_scenario_demand()``
            writes.
    """
    def weighted_gwh(col: str) -> float:
        return df_demand[f'weighted_{col}'].sum() / KWH_TO_GWH

    breakdown = {
        'Baseline electricity': weighted_gwh('baseline_electric_kwh'),
        'Net heating': weighted_gwh('heating_elec_change_kwh'),
        'Net cooling': weighted_gwh('cooling_elec_change_kwh'),
        'Net hot water': weighted_gwh('hot_water_elec_change_kwh'),
        'Other end uses': weighted_gwh('other_elec_change_kwh'),
        'Post-retrofit electricity': weighted_gwh('retrofit_electric_kwh'),
        'Net change': weighted_gwh('elec_demand_change_kwh'),
    }

    # Heating electricity by the fuel the home heated with before the retrofit:
    # fossil homes add a heat pump's load, electric-resistance homes shed load.
    heating_gwh_by_fuel = df_demand.groupby('in.heating_fuel')[
        'weighted_heating_elec_change_kwh'].sum() / KWH_TO_GWH
    for heating_fuel, heating_gwh in heating_gwh_by_fuel.items():
        breakdown[f'Net heating, {heating_fuel} homes'] = heating_gwh

    baseline_site_gwh = weighted_gwh('baseline_site_energy_kwh')
    site_change_gwh = weighted_gwh('site_energy_change_kwh')
    breakdown['Baseline site energy (all fuels)'] = baseline_site_gwh
    breakdown['Site energy change (all fuels)'] = site_change_gwh
    breakdown['Site energy change (%)'] = site_change_gwh / baseline_site_gwh * 100
    return pd.Series(breakdown, name='GWh')


# ============================================================================
# AGGREGATION (STATE OR COUNTY)
# ============================================================================

def aggregate_demand(
    df_demand: pd.DataFrame,
    geo_level: str = 'state',
    min_home_count: int = MIN_HOME_COUNT,
    verbose: bool = False,
) -> pd.DataFrame:
    """Aggregate per-building demand results to state- or county-level GWh totals.

    Uses EUSS sampling weights to produce population-representative totals.
    Percentage changes are computed at the aggregate level:
    ``(sum of weighted_retrofit - sum of weighted_baseline)
    / sum of weighted_baseline x 100``.
    The electricity percent is taken against baseline electricity and the
    site-energy percent against baseline site energy (all fuels).

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
        ValueError: If ``geo_level`` is not ``'state'`` or ``'county'``, or
            the grouped totals do not add up to the per-home total.
        KeyError: If expected weighted columns are missing from ``df_demand``.
    """
    if geo_level not in ('state', 'county'):
        raise ValueError(f"geo_level must be 'state' or 'county', got {geo_level!r}")

    # ---------------------------------------------------------------------------
    # NOTE ON SAMPLING WEIGHTS
    # ---------------------------------------------------------------------------
    # ResStock assigns a uniform sampling weight (~242) to every building.
    # Because the weight is constant, it cancels in any ratio, rate, or
    # percentage computation.  Simple counts and sums are used for these
    # metrics.  The weight IS applied when computing absolute population
    # totals (e.g., home_count in millions) to scale from sample to national.
    # ---------------------------------------------------------------------------

    group_col = 'in.county' if geo_level == 'county' else 'in.state'

    grouped = df_demand.groupby(group_col).agg(
        _sample_count=('weight', 'size'),
        home_count=('weight', 'sum'),
        weighted_baseline_elec=('weighted_baseline_electric_kwh', 'sum'),
        weighted_retrofit_elec=('weighted_retrofit_electric_kwh', 'sum'),
        weighted_elec_change=('weighted_elec_demand_change_kwh', 'sum'),
        weighted_baseline_site=('weighted_baseline_site_energy_kwh', 'sum'),
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

    # ResStock uses uniform sampling weight (~242) for all buildings.
    # Percentage changes derive from already-computed GWh totals -- with
    # uniform weights, sum(w x change) / sum(w x baseline) equals
    # change_gwh / baseline_gwh.
    grouped['pct_elec_demand_change'] = np.where(
        grouped['baseline_elec_gwh'] != 0,
        grouped['elec_change_gwh'] / grouped['baseline_elec_gwh'] * 100,
        np.nan,
    )
    # Site energy is its own all-fuel quantity, so its percent is taken against
    # baseline site energy, not baseline electricity. Blank only if a county's
    # baseline site energy is zero; zero-change homes still count.
    grouped['pct_site_energy_change'] = np.where(
        grouped['weighted_baseline_site'] != 0,
        grouped['weighted_site_change'] / grouped['weighted_baseline_site'] * 100,
        np.nan,
    )

    _metric_cols = [
        'baseline_elec_gwh', 'retrofit_elec_gwh', 'elec_change_gwh',
        'site_energy_change_gwh', 'pct_elec_demand_change', 'pct_site_energy_change',
    ]

    if geo_level == 'county':
        below = grouped['_sample_count'] < min_home_count
        grouped.loc[below, _metric_cols] = np.nan

    # Demand accounting check (state level only, totals still hold)
    total_sum = df_demand['weighted_elec_demand_change_kwh'].sum()
    total_agg = grouped['weighted_elec_change'].sum()
    if not np.isclose(total_sum, total_agg, rtol=1e-6):
        raise ValueError(
            "Demand accounting mismatch: per-home total "
            f"{total_sum:.0f} kWh vs grouped total {total_agg:.0f} kWh")
    # Printed even when quiet; a failed check stops the run just above.
    print("[OK] Demand accounting check passed")

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
