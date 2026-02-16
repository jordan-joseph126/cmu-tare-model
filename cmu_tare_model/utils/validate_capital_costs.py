"""
Capital Cost Validation: Equipment-Level Disaggregation
========================================================
Analyzes installed capital costs from the TARE model DataFrames,
disaggregated by equipment type, capacity, efficiency rating, and fuel type.

Equipment types validated:
  - ASHP (Air Source Heat Pump, centrally ducted)
  - Central AC
  - Gas Furnace
  - Propane Furnace

For each equipment configuration, reports the 10th, 50th, and 90th percentile
of installed costs across homes in the DataFrame, for each cost scenario
(v3, v4MID, or whichever scenarios are active).

Binning approach:
  - Capacity (tons): round to nearest integer — [N-0.5, N+0.5) maps to bin N.
    Bins from 2 to 10 tons.  Homes < 1.5 or >= 10.5 tons reported as outliers.
  - Capacity (kBTU/h): round to nearest 10 — [N-5, N+5) maps to bin N.
    Bins from 40 to 200 kBTU/h.  Outliers below/above reported separately.
  - SEER: round to nearest integer — [N-0.5, N+0.5) maps to bin N.
    Bins from 13 to 25.  Outliers reported separately.
  - AFUE: round to nearest integer — [N-0.5, N+0.5) maps to bin N.
    Bins from 78 to 98.  Outliers reported separately.

Usage (from notebook):
    from cmu_tare_model.utils.validate_capital_costs import run_capital_cost_validation
    run_capital_cost_validation(
        df=df_euss_am_mpX_home,
        capital_costs_mpx=CAPITAL_COSTS_MPX,
        menu_mp=menu_mp,
    )
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from cmu_tare_model.constants import REMDB_COST_SCENARIO_KEYS
from cmu_tare_model.utils.column_names import create_cost_col


# ─────────────────────────────────────────────────────────────────────────────
# Binning configuration
# ─────────────────────────────────────────────────────────────────────────────

# Capacity bins (tons) — 2 through 10, using floor rounding (1.5–2.4 → 2, etc.)
CAPACITY_BINS_TONS = list(range(2, 11))       # [2, 3, 4, 5, 6, 7, 8, 9, 10]
CAPACITY_TONS_LOW = 1.5                        # below this = outlier
CAPACITY_TONS_HIGH = 10.5                      # at or above this = outlier

# Capacity bins (kBTU/h) — 40 through 200 in steps of 10
CAPACITY_BINS_KBTUH = list(range(40, 201, 10))
CAPACITY_KBTUH_LOW = 35.0
CAPACITY_KBTUH_HIGH = 205.0

# SEER bins — 13 through 25
SEER_BINS = list(range(13, 26))
SEER_LOW = 12.5
SEER_HIGH = 25.5

# AFUE bins — 78 through 98
AFUE_BINS = list(range(78, 99))
AFUE_LOW = 77.5
AFUE_HIGH = 98.5


# ─────────────────────────────────────────────────────────────────────────────
# Binning helpers
# ─────────────────────────────────────────────────────────────────────────────

def _capacity_tons(kbtuh: pd.Series) -> pd.Series:
    """Convert kBTU/h to tons (÷12)."""
    return kbtuh / 12.0


def _round_to_bin(values: pd.Series, bins: List[int],
                  lo: float, hi: float) -> pd.Series:
    """
    Assign each value to the nearest integer bin using rounding.

    Values in [bin - 0.5, bin + 0.5) map to that bin.
    Values < lo or >= hi are set to NaN (outliers).
    """
    result = values.round(0)
    # Mark outliers as NaN
    result = result.where(result.between(bins[0], bins[-1]), other=np.nan)
    return result


def _round_to_bin_step(values: pd.Series, bins: List[int],
                       step: int, lo: float, hi: float) -> pd.Series:
    """
    Assign each value to the nearest bin with a given step size.

    For step=10:  [bin-5, bin+5) maps to that bin.
    Values < lo or >= hi are set to NaN (outliers).
    """
    result = (values / step).round(0) * step
    result = result.where(result.between(bins[0], bins[-1]), other=np.nan)
    return result


def _count_outliers(values: pd.Series, lo: float, hi: float) -> Tuple[int, int]:
    """Count values below lo and at/above hi."""
    valid = values.dropna()
    n_below = (valid < lo).sum()
    n_above = (valid >= hi).sum()
    return int(n_below), int(n_above)


def _extract_seer(efficiency_str: pd.Series) -> pd.Series:
    """Extract SEER value from efficiency strings like 'SEER 15, 8.5 HSPF'."""
    return efficiency_str.str.extract(r'SEER (\d+\.?\d*)', expand=False).astype(float)


def _extract_afue(efficiency_str: pd.Series) -> pd.Series:
    """Extract AFUE value from efficiency strings like '80% AFUE'."""
    afue = efficiency_str.str.extract(r'(\d+\.?\d*)% AFUE', expand=False).astype(float)
    # Also handle 'XX% Efficiency' pattern (e.g., Electric Baseboard)
    mask = afue.isna()
    eff = efficiency_str.loc[mask].str.extract(r'(\d+\.?\d*)% Efficiency', expand=False).astype(float)
    afue.loc[mask] = eff
    return afue


def _compute_percentiles(series: pd.Series) -> Dict:
    """Compute 10th, 50th, 90th percentiles and count for a cost series."""
    s = series.dropna()
    if len(s) == 0:
        return {'N': 0, 'P10': np.nan, 'P50': np.nan, 'P90': np.nan}
    return {
        'N': len(s),
        'P10': s.quantile(0.10),
        'P50': s.quantile(0.50),
        'P90': s.quantile(0.90),
    }


def _format_dollar(val) -> str:
    """Format a numeric value as a dollar string."""
    if pd.isna(val):
        return 'N/A'
    return f'${val:,.0f}'


def _print_outlier_counts(outliers: Dict,
                          cap_unit: str = 'tons',
                          eff_label: str = 'SEER') -> None:
    """Print outlier counts for capacity and efficiency from outliers dict."""
    cap_below = outliers.get('cap_below', 0)
    cap_above = outliers.get('cap_above', 0)
    eff_below = outliers.get('eff_below', 0)
    eff_above = outliers.get('eff_above', 0)

    if cap_unit == 'tons':
        cap_lo_label = f'{CAPACITY_TONS_LOW} {cap_unit}'
        cap_hi_label = f'{CAPACITY_TONS_HIGH} {cap_unit}'
    else:
        cap_lo_label = f'{CAPACITY_KBTUH_LOW} {cap_unit}'
        cap_hi_label = f'{CAPACITY_KBTUH_HIGH} {cap_unit}'

    if eff_label == 'SEER':
        eff_lo_label = f'SEER {SEER_LOW}'
        eff_hi_label = f'SEER {SEER_HIGH}'
    else:
        eff_lo_label = f'{AFUE_LOW}% AFUE'
        eff_hi_label = f'{AFUE_HIGH}% AFUE'

    print(f"  Outliers excluded from bins:")
    print(f"    Capacity  < {cap_lo_label}: {cap_below:,} homes")
    print(f"    Capacity >= {cap_hi_label}: {cap_above:,} homes")
    print(f"    {eff_label}  < {eff_lo_label}: {eff_below:,} homes")
    print(f"    {eff_label} >= {eff_hi_label}: {eff_above:,} homes")


# ─────────────────────────────────────────────────────────────────────────────
# Analysis functions for each equipment type
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_ashp(df: pd.DataFrame,
                  menu_mp: int,
                  cost_scenarios: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze ASHP (heating replacement) installed costs by SEER × capacity.

    Filters to homes with Electricity ASHP heating type (centrally ducted),
    then groups by baseline SEER and heating capacity in tons.

    Returns:
        Tuple of (results DataFrame, outlier info dict).
    """
    outliers = {}

    # Filter: homes currently using ASHP heating
    mask = pd.Series(True, index=df.index)

    if 'heating_type' in df.columns:
        mask &= (df['heating_type'] == 'Electricity ASHP')
    elif 'hvac_heating_type_and_fuel' in df.columns:
        mask &= (df['hvac_heating_type_and_fuel'] == 'Electricity ASHP')
    else:
        return pd.DataFrame(), outliers

    # Require ducts (centrally ducted ASHP)
    if 'hvac_has_ducts' in df.columns:
        mask &= (df['hvac_has_ducts'] == 'Yes')

    df_filtered = df.loc[mask].copy()
    outliers['total_filtered'] = len(df_filtered)
    if len(df_filtered) == 0:
        return pd.DataFrame(), outliers

    # Extract baseline SEER (the SEER of the existing ASHP being replaced)
    if 'baseline_SEER' in df_filtered.columns:
        df_filtered['_seer'] = df_filtered['baseline_SEER']
    elif 'hvac_heating_efficiency' in df_filtered.columns:
        df_filtered['_seer'] = _extract_seer(df_filtered['hvac_heating_efficiency'])
    else:
        return pd.DataFrame(), outliers

    # Capacity in tons (heating system, converted from kBTU/h)
    if 'size_heating_system_primary_k_btu_h' not in df_filtered.columns:
        return pd.DataFrame(), outliers
    df_filtered['_cap_tons'] = _capacity_tons(df_filtered['size_heating_system_primary_k_btu_h'])

    # Count outliers before binning
    cap_below, cap_above = _count_outliers(df_filtered['_cap_tons'],
                                           CAPACITY_TONS_LOW, CAPACITY_TONS_HIGH)
    seer_below, seer_above = _count_outliers(df_filtered['_seer'],
                                              SEER_LOW, SEER_HIGH)
    outliers.update({
        'cap_below': cap_below, 'cap_above': cap_above,
        'eff_below': seer_below, 'eff_above': seer_above,
    })

    # Assign bins using rounding
    df_filtered['_seer_bin'] = _round_to_bin(df_filtered['_seer'], SEER_BINS,
                                              SEER_LOW, SEER_HIGH)
    df_filtered['_cap_bin'] = _round_to_bin(df_filtered['_cap_tons'], CAPACITY_BINS_TONS,
                                             CAPACITY_TONS_LOW, CAPACITY_TONS_HIGH)

    # Drop rows that don't match any bin
    df_filtered = df_filtered.dropna(subset=['_seer_bin', '_cap_bin'])
    if len(df_filtered) == 0:
        return pd.DataFrame(), outliers

    # Determine which SEER and capacity bins actually have data
    active_seer = sorted(df_filtered['_seer_bin'].dropna().unique())
    active_caps = sorted(df_filtered['_cap_bin'].dropna().unique())

    # Build results
    rows = []
    for cap in active_caps:
        for seer in active_seer:
            bin_mask = (df_filtered['_cap_bin'] == cap) & (df_filtered['_seer_bin'] == seer)
            row = {
                'Capacity (tons)': int(cap),
                'SEER': int(seer),
            }
            for scenario in cost_scenarios:
                col = create_cost_col(menu_mp=menu_mp, category='heating',
                                      cost_type='replacement', cost_scenario=scenario)
                if col in df_filtered.columns:
                    stats = _compute_percentiles(df_filtered.loc[bin_mask, col])
                else:
                    stats = {'N': 0, 'P10': np.nan, 'P50': np.nan, 'P90': np.nan}

                row[f'{scenario} N'] = stats['N']
                row[f'{scenario} P10'] = stats['P10']
                row[f'{scenario} P50'] = stats['P50']
                row[f'{scenario} P90'] = stats['P90']
            rows.append(row)

    return pd.DataFrame(rows), outliers


def _analyze_central_ac(df: pd.DataFrame,
                         menu_mp: int,
                         cost_scenarios: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze Central AC (cooling replacement) installed costs by SEER × capacity.

    Filters to homes with Central AC cooling type, then groups by
    cooling SEER and cooling capacity in tons.

    Returns:
        Tuple of (results DataFrame, outlier info dict).
    """
    outliers = {}

    # Filter: Central AC homes
    mask = pd.Series(True, index=df.index)
    if 'hvac_cooling_type' in df.columns:
        mask &= (df['hvac_cooling_type'] == 'Central AC')
    else:
        return pd.DataFrame(), outliers

    df_filtered = df.loc[mask].copy()
    outliers['total_filtered'] = len(df_filtered)
    if len(df_filtered) == 0:
        return pd.DataFrame(), outliers

    # Extract cooling SEER
    if 'hvac_cooling_efficiency' in df_filtered.columns:
        df_filtered['_seer'] = _extract_seer(df_filtered['hvac_cooling_efficiency'])
    else:
        return pd.DataFrame(), outliers

    # Capacity in tons
    if 'size_cooling_system_primary_k_btu_h' not in df_filtered.columns:
        return pd.DataFrame(), outliers
    df_filtered['_cap_tons'] = _capacity_tons(df_filtered['size_cooling_system_primary_k_btu_h'])

    # Count outliers before binning
    cap_below, cap_above = _count_outliers(df_filtered['_cap_tons'],
                                           CAPACITY_TONS_LOW, CAPACITY_TONS_HIGH)
    seer_below, seer_above = _count_outliers(df_filtered['_seer'],
                                              SEER_LOW, SEER_HIGH)
    outliers.update({
        'cap_below': cap_below, 'cap_above': cap_above,
        'eff_below': seer_below, 'eff_above': seer_above,
    })

    # Assign bins using rounding
    df_filtered['_seer_bin'] = _round_to_bin(df_filtered['_seer'], SEER_BINS,
                                              SEER_LOW, SEER_HIGH)
    df_filtered['_cap_bin'] = _round_to_bin(df_filtered['_cap_tons'], CAPACITY_BINS_TONS,
                                             CAPACITY_TONS_LOW, CAPACITY_TONS_HIGH)

    # Drop rows that don't match any bin
    df_filtered = df_filtered.dropna(subset=['_seer_bin', '_cap_bin'])
    if len(df_filtered) == 0:
        return pd.DataFrame(), outliers

    # Central AC has no v3 data — filter to v4 scenarios only
    cool_scenarios = [s for s in cost_scenarios if s != 'v3']

    # Determine which bins actually have data
    active_seer = sorted(df_filtered['_seer_bin'].dropna().unique())
    active_caps = sorted(df_filtered['_cap_bin'].dropna().unique())

    rows = []
    for cap in active_caps:
        for seer in active_seer:
            bin_mask = (df_filtered['_cap_bin'] == cap) & (df_filtered['_seer_bin'] == seer)
            row = {
                'Capacity (tons)': int(cap),
                'SEER': int(seer),
            }

            # v3 = N/A for cooling
            if 'v3' in cost_scenarios:
                row['v3 N'] = 0
                row['v3 P10'] = np.nan
                row['v3 P50'] = np.nan
                row['v3 P90'] = np.nan

            for scenario in cool_scenarios:
                col = create_cost_col(menu_mp=menu_mp, category='cooling',
                                      cost_type='replacement', cost_scenario=scenario)
                if col in df_filtered.columns:
                    stats = _compute_percentiles(df_filtered.loc[bin_mask, col])
                else:
                    stats = {'N': 0, 'P10': np.nan, 'P50': np.nan, 'P90': np.nan}

                row[f'{scenario} N'] = stats['N']
                row[f'{scenario} P10'] = stats['P10']
                row[f'{scenario} P50'] = stats['P50']
                row[f'{scenario} P90'] = stats['P90']
            rows.append(row)

    return pd.DataFrame(rows), outliers


def _analyze_furnace(df: pd.DataFrame,
                      menu_mp: int,
                      cost_scenarios: List[str],
                      fuel_type: str = 'Natural Gas') -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze Furnace (heating replacement) installed costs by AFUE × capacity.

    Filters to homes with the specified fuel type and furnace heating type,
    then groups by baseline AFUE and heating capacity in kBTU/h.

    Args:
        fuel_type: 'Natural Gas' or 'Propane'

    Returns:
        Tuple of (results DataFrame, outlier info dict).
    """
    outliers = {}

    # Filter: Furnace homes of the specified fuel type
    mask = pd.Series(True, index=df.index)

    if 'base_heating_fuel' in df.columns:
        mask &= (df['base_heating_fuel'] == fuel_type)

    # Filter to furnace-type heating systems
    if 'heating_type' in df.columns:
        mask &= df['heating_type'].str.contains('Furnace', case=False, na=False)
    elif 'hvac_heating_type_and_fuel' in df.columns:
        mask &= df['hvac_heating_type_and_fuel'].str.contains('Furnace', case=False, na=False)

    df_filtered = df.loc[mask].copy()
    outliers['total_filtered'] = len(df_filtered)
    if len(df_filtered) == 0:
        return pd.DataFrame(), outliers

    # Extract AFUE from heating efficiency
    if 'hvac_heating_efficiency' in df_filtered.columns:
        df_filtered['_afue'] = _extract_afue(df_filtered['hvac_heating_efficiency'])
    elif 'baseline_AFUE' in df_filtered.columns:
        df_filtered['_afue'] = df_filtered['baseline_AFUE']
    else:
        return pd.DataFrame(), outliers

    # Capacity in kBTU/h
    if 'size_heating_system_primary_k_btu_h' not in df_filtered.columns:
        return pd.DataFrame(), outliers
    df_filtered['_cap_kbtuh'] = df_filtered['size_heating_system_primary_k_btu_h']

    # Count outliers before binning
    cap_below, cap_above = _count_outliers(df_filtered['_cap_kbtuh'],
                                           CAPACITY_KBTUH_LOW, CAPACITY_KBTUH_HIGH)
    afue_below, afue_above = _count_outliers(df_filtered['_afue'],
                                              AFUE_LOW, AFUE_HIGH)
    outliers.update({
        'cap_below': cap_below, 'cap_above': cap_above,
        'eff_below': afue_below, 'eff_above': afue_above,
    })

    # Assign bins using rounding
    df_filtered['_afue_bin'] = _round_to_bin(df_filtered['_afue'], AFUE_BINS,
                                              AFUE_LOW, AFUE_HIGH)
    df_filtered['_cap_bin'] = _round_to_bin_step(df_filtered['_cap_kbtuh'],
                                                  CAPACITY_BINS_KBTUH, 10,
                                                  CAPACITY_KBTUH_LOW, CAPACITY_KBTUH_HIGH)

    # Drop rows that don't match any bin
    df_filtered = df_filtered.dropna(subset=['_afue_bin', '_cap_bin'])
    if len(df_filtered) == 0:
        return pd.DataFrame(), outliers

    # Determine which bins actually have data
    active_afue = sorted(df_filtered['_afue_bin'].dropna().unique())
    active_caps = sorted(df_filtered['_cap_bin'].dropna().unique())

    rows = []
    for cap in active_caps:
        for afue in active_afue:
            bin_mask = (df_filtered['_cap_bin'] == cap) & (df_filtered['_afue_bin'] == afue)
            row = {
                'Capacity (kBTU/h)': int(cap),
                'AFUE': int(afue),
            }
            for scenario in cost_scenarios:
                col = create_cost_col(menu_mp=menu_mp, category='heating',
                                      cost_type='replacement', cost_scenario=scenario)
                if col in df_filtered.columns:
                    stats = _compute_percentiles(df_filtered.loc[bin_mask, col])
                else:
                    stats = {'N': 0, 'P10': np.nan, 'P50': np.nan, 'P90': np.nan}

                row[f'{scenario} N'] = stats['N']
                row[f'{scenario} P10'] = stats['P10']
                row[f'{scenario} P50'] = stats['P50']
                row[f'{scenario} P90'] = stats['P90']
            rows.append(row)

    return pd.DataFrame(rows), outliers


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_table(title: str,
                 df_result: pd.DataFrame,
                 cost_scenarios: List[str],
                 id_cols: List[str],
                 notes: Optional[List[str]] = None,
                 outliers: Optional[Dict] = None,
                 cap_unit: str = 'tons',
                 eff_label: str = 'SEER') -> None:
    """Format and print a cost disaggregation table with outlier info."""
    print(f"\n{'=' * 110}")
    print(f"  {title}")
    print(f"{'=' * 110}")

    if notes:
        for note in notes:
            print(f"  {note}")
        print()

    # Print outlier counts
    if outliers:
        total = outliers.get('total_filtered', 0)
        print(f"  Total homes matching filter: {total:,}")
        _print_outlier_counts(outliers, cap_unit=cap_unit, eff_label=eff_label)
        print()

    if df_result.empty:
        print("  No matching homes found in the DataFrame for this equipment type.")
        print(f"{'=' * 110}")
        return

    # Filter to rows with data (N > 0 for at least one scenario)
    n_cols = [f'{s} N' for s in cost_scenarios if f'{s} N' in df_result.columns]
    if n_cols:
        has_data = df_result[n_cols].sum(axis=1) > 0
        df_result = df_result.loc[has_data]

    if df_result.empty:
        print("  All bins are empty after filtering.")
        print(f"{'=' * 110}")
        return

    # Build display DataFrame
    display_rows = []
    for _, row in df_result.iterrows():
        disp = {}
        for c in id_cols:
            disp[c] = row[c]

        for scenario in cost_scenarios:
            n = row.get(f'{scenario} N', 0)
            if n == 0:
                disp[f'{scenario} N'] = '—'
                disp[f'{scenario} P10'] = 'N/A'
                disp[f'{scenario} P50'] = 'N/A'
                disp[f'{scenario} P90'] = 'N/A'
            else:
                disp[f'{scenario} N'] = f'{n:,}'
                disp[f'{scenario} P10'] = _format_dollar(row[f'{scenario} P10'])
                disp[f'{scenario} P50'] = _format_dollar(row[f'{scenario} P50'])
                disp[f'{scenario} P90'] = _format_dollar(row[f'{scenario} P90'])
        display_rows.append(disp)

    df_display = pd.DataFrame(display_rows)
    print(df_display.to_string(index=False))
    print(f"{'=' * 110}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_capital_cost_validation(
    df: pd.DataFrame,
    menu_mp: int,
    capital_costs_mpx: Optional[Dict] = None,
    cost_scenarios: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run capital cost validation disaggregated by equipment type and capacity.

    Analyzes installed costs already computed in the DataFrame, filtering and
    grouping by equipment SEER/AFUE ratings and capacity (tons or kBTU/h).

    Args:
        df: The main home-level DataFrame (df_euss_am_mpX_home) with cost
            columns already computed.
        menu_mp: Measure package number (e.g. 3, 4, 8).
        capital_costs_mpx: Optional CAPITAL_COSTS_MPX dict. If provided and
            the active scenarios include v4MID, the v4MID DataFrame is used
            for scenarios that may have additional columns.
        cost_scenarios: List of cost scenario keys (default: REMDB_COST_SCENARIO_KEYS).

    Returns:
        Dict mapping equipment type label to its results DataFrame.
    """
    if cost_scenarios is None:
        cost_scenarios = list(REMDB_COST_SCENARIO_KEYS)

    print("\n" + "#" * 110)
    print(f"#  CAPITAL COST VALIDATION: Equipment-Level Disaggregation (MP{menu_mp})")
    print(f"#  Active cost scenarios: {cost_scenarios}")
    print(f"#  Total homes in DataFrame: {len(df):,}")
    print("#" * 110)

    # ── Determine which DataFrame to use for each scenario ──
    # If CAPITAL_COSTS_MPX is provided, v4 scenario columns may only exist
    # in those DataFrames (not yet merged into the main df). Build a merged
    # view that includes all cost columns.
    df_work = df.copy()

    if capital_costs_mpx is not None:
        for cost_type in ['replacement', 'upgrade']:
            for scenario_key in cost_scenarios:
                if scenario_key == 'v3':
                    continue  # v3 already in df
                scenario_df = capital_costs_mpx.get('heating', {}).get(cost_type, {}).get(scenario_key)
                if scenario_df is not None:
                    col = create_cost_col(menu_mp=menu_mp, category='heating',
                                          cost_type=cost_type, cost_scenario=scenario_key)
                    if col not in df_work.columns and col in scenario_df.columns:
                        df_work[col] = scenario_df[col].values

                # Also pull cooling replacement from the scenario DF
                if cost_type == 'replacement':
                    cool_scenario_df = scenario_df  # cooling cols are on the same DF
                    if cool_scenario_df is not None:
                        cool_col = create_cost_col(menu_mp=menu_mp, category='cooling',
                                                    cost_type='replacement', cost_scenario=scenario_key)
                        if cool_col not in df_work.columns and cool_col in cool_scenario_df.columns:
                            df_work[cool_col] = cool_scenario_df[cool_col].values

    results = {}

    # ── 1. ASHP (Heating Replacement) ──
    df_ashp, ashp_outliers = _analyze_ashp(df_work, menu_mp, cost_scenarios)
    results['ASHP (Heating Replacement)'] = df_ashp
    _print_table(
        title='ASHP — Air Source Heat Pump (Heating Replacement, Centrally Ducted)',
        df_result=df_ashp,
        cost_scenarios=cost_scenarios,
        id_cols=['Capacity (tons)', 'SEER'],
        notes=[
            'Cost type: heating replacement (like-for-like ASHP)',
            'Filter: heating_type = Electricity ASHP, hvac_has_ducts = Yes',
            f'Capacity bins: {CAPACITY_BINS_TONS[0]}–{CAPACITY_BINS_TONS[-1]} tons (floor rounding, e.g. 1.5–2.4 → 2)',
            f'SEER bins: {SEER_BINS[0]}–{SEER_BINS[-1]} (floor rounding, e.g. 12.5–13.4 → 13)',
            'v3 note: v3 uses a fixed efficiency key — same cost across SEER bins for a given capacity',
        ],
        outliers=ashp_outliers,
        cap_unit='tons',
        eff_label='SEER',
    )

    # ── 2. Central AC (Cooling Replacement) ──
    df_cac, cac_outliers = _analyze_central_ac(df_work, menu_mp, cost_scenarios)
    results['Central AC (Cooling Replacement)'] = df_cac
    _print_table(
        title='Central AC — Centrally Ducted (Cooling Replacement)',
        df_result=df_cac,
        cost_scenarios=cost_scenarios,
        id_cols=['Capacity (tons)', 'SEER'],
        notes=[
            'Cost type: cooling replacement (like-for-like Central AC)',
            'Filter: hvac_cooling_type = Central AC',
            f'Capacity bins: {CAPACITY_BINS_TONS[0]}–{CAPACITY_BINS_TONS[-1]} tons (floor rounding)',
            f'SEER bins: {SEER_BINS[0]}–{SEER_BINS[-1]} (floor rounding)',
            'v3 note: No v3 data exists for cooling replacement — v3 columns show N/A',
        ],
        outliers=cac_outliers,
        cap_unit='tons',
        eff_label='SEER',
    )

    # ── 3. Gas Furnace (Heating Replacement) ──
    df_gas, gas_outliers = _analyze_furnace(df_work, menu_mp, cost_scenarios, fuel_type='Natural Gas')
    results['Gas Furnace (Heating Replacement)'] = df_gas
    _print_table(
        title='Gas Furnace — Natural Gas (Heating Replacement)',
        df_result=df_gas,
        cost_scenarios=cost_scenarios,
        id_cols=['Capacity (kBTU/h)', 'AFUE'],
        notes=[
            'Cost type: heating replacement (like-for-like furnace)',
            'Filter: base_heating_fuel = Natural Gas, heating_type contains Furnace',
            f'Capacity bins: {CAPACITY_BINS_KBTUH[0]}–{CAPACITY_BINS_KBTUH[-1]} kBTU/h (step=10, floor rounding)',
            f'AFUE bins: {AFUE_BINS[0]}–{AFUE_BINS[-1]}% (floor rounding)',
            'v3 note: v3 uses a fixed efficiency key — same cost across AFUE bins for a given capacity',
        ],
        outliers=gas_outliers,
        cap_unit='kBTU/h',
        eff_label='AFUE',
    )

    # ── 4. Propane Furnace (Heating Replacement) ──
    df_propane, propane_outliers = _analyze_furnace(df_work, menu_mp, cost_scenarios, fuel_type='Propane')
    results['Propane Furnace (Heating Replacement)'] = df_propane
    _print_table(
        title='Propane Furnace (Heating Replacement)',
        df_result=df_propane,
        cost_scenarios=cost_scenarios,
        id_cols=['Capacity (kBTU/h)', 'AFUE'],
        notes=[
            'Cost type: heating replacement (like-for-like furnace)',
            'Filter: base_heating_fuel = Propane, heating_type contains Furnace',
            f'Capacity bins: {CAPACITY_BINS_KBTUH[0]}–{CAPACITY_BINS_KBTUH[-1]} kBTU/h (step=10, floor rounding)',
            f'AFUE bins: {AFUE_BINS[0]}–{AFUE_BINS[-1]}% (floor rounding)',
            'v3 note: v3 uses a fixed efficiency key — same cost across AFUE bins for a given capacity',
        ],
        outliers=propane_outliers,
        cap_unit='kBTU/h',
        eff_label='AFUE',
    )

    # ── Summary ──
    print(f"\n{'#' * 110}")
    print(f"#  VALIDATION SUMMARY")
    print(f"{'#' * 110}")
    for label, df_r in results.items():
        if df_r.empty:
            print(f"  {label:<45}  No matching homes")
        else:
            # Count total homes across all bins
            n_col = f'{cost_scenarios[0]} N' if cost_scenarios else None
            total_n = df_r[n_col].sum() if n_col and n_col in df_r.columns else 0
            n_bins_with_data = (df_r[n_col] > 0).sum() if n_col and n_col in df_r.columns else 0
            total_bins = len(df_r)
            print(f"  {label:<45}  {n_bins_with_data}/{total_bins} bins with data  |  "
                  f"{int(total_n):,} total homes matched")

    print(f"{'#' * 110}\n")

    return results
