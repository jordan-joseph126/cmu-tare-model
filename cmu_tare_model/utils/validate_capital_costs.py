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
from matplotlib.figure import Figure
from openpyxl import Workbook
from openpyxl.styles import Font
from typing import Dict, List, Optional, Tuple

from cmu_tare_model.constants import REMDB_COST_SCENARIO_KEYS
from cmu_tare_model.utils.column_names import create_cost_col
from cmu_tare_model.utils.data_visualization_histograms import create_subplot_grid_histogram


# ─────────────────────────────────────────────────────────────────────────────
# Binning configuration
# ─────────────────────────────────────────────────────────────────────────────

# Capacity bins (tons) — 2 through 10, using floor rounding (1.5–2.4 → 2, etc.)
CAPACITY_BINS_TONS = list(range(2, 11))       # [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Capacity bins (kBTU/h) — 40 through 200 in steps of 10
CAPACITY_BINS_KBTUH = list(range(40, 201, 10))

# SEER bins — 13 through 30
SEER_BINS = list(range(13, 31))

# AFUE bins — 78 through 98
AFUE_BINS = list(range(78, 99))


# ─────────────────────────────────────────────────────────────────────────────
# Binning helpers
# ─────────────────────────────────────────────────────────────────────────────

def _capacity_tons(kbtuh: pd.Series) -> pd.Series:
    """Convert kBTU/h to tons (÷12)."""
    return kbtuh / 12.0


def _round_to_bin(values: pd.Series, bins: List[int],
                  step: int = 1) -> pd.Series:
    """Assign each value to the nearest bin with a given step size.

    For step=1: values round to nearest integer bin.
    For step=10: values in [bin-5, bin+5) map to that bin.
    Values outside [bins[0], bins[-1]] are set to NaN (outliers).
    """
    result = (values / step).round(0) * step
    return result.where(result.between(bins[0], bins[-1]), other=np.nan)


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


def _format_dollar(val) -> str:
    """Format a numeric value as a dollar string."""
    if pd.isna(val):
        return 'N/A'
    return f'${val:,.0f}'


def _print_outlier_counts(outliers: Dict,
                          cap_unit: str = 'tons',
                          eff_label: str = 'SEER') -> None:
    """Print outlier counts for capacity and efficiency from outliers dict."""
    cap_lo = outliers.get('cap_lo', '?')
    cap_hi = outliers.get('cap_hi', '?')
    eff_lo = outliers.get('eff_lo', '?')
    eff_hi = outliers.get('eff_hi', '?')

    print(f"  Outliers excluded from bins:")
    print(f"    Capacity  < {cap_lo} {cap_unit}: {outliers.get('cap_below', 0):,} homes")
    print(f"    Capacity >= {cap_hi} {cap_unit}: {outliers.get('cap_above', 0):,} homes")
    print(f"    {eff_label}  < {eff_lo}: {outliers.get('eff_below', 0):,} homes")
    print(f"    {eff_label} >= {eff_hi}: {outliers.get('eff_above', 0):,} homes")
    cap_nan = outliers.get('cap_nan', 0)
    eff_nan = outliers.get('eff_nan', 0)
    if cap_nan > 0 or eff_nan > 0:
        print(f"    Capacity NaN (missing data): {cap_nan:,} homes")
        print(f"    {eff_label} NaN (missing/unparseable): {eff_nan:,} homes")


def _build_clamping_summary(
    df_f: pd.DataFrame,
    pm2_col: str,
    pm2_original_col: str,
    eff_label: str,
    display_scale: float = 1.0,
) -> Optional[List[str]]:
    """Build efficiency floor clamping impact summary.

    Compares floored and original efficiency values to show how homes
    migrated from each sub-floor efficiency level into the floor bin.
    Only produces output for replacement metrics where clamping occurred.

    Args:
        df_f: Filtered DataFrame for this equipment type.
        pm2_col: Column with floored efficiency (used by cost regression).
        pm2_original_col: Column with original pre-floor EUSS efficiency.
        eff_label: Display label ('SEER' or 'AFUE').
        display_scale: Multiplier for display (1.0 for SEER, 100.0 for
            AFUE decimal→percentage).

    Returns:
        List of formatted summary strings, or None if no clamping occurred.
    """
    if pm2_original_col not in df_f.columns or pm2_col not in df_f.columns:
        return None

    floored = df_f[pm2_col]
    original = df_f[pm2_original_col]

    # Identify homes where the floor changed the efficiency value
    clamped_mask = (floored != original) & floored.notna() & original.notna()
    if not clamped_mask.any():
        return None

    total_filtered = len(df_f)
    floor_value = floored[clamped_mask].mode().iloc[0]
    floor_display = floor_value * display_scale

    # Total homes now in the floor bin (originally at floor + clamped up)
    in_floor_bin = int((floored == floor_value).sum())

    lines = [
        f"  Efficiency floor impact on {eff_label} {floor_display:.0f} bin composition "
        f"({in_floor_bin:,} homes in bin):"
    ]

    # Group clamped homes by their original efficiency for the migration summary
    orig_display = (original[clamped_mask] * display_scale).round(1)
    for orig_val, count in orig_display.value_counts().sort_index().items():
        pct_of_total = count / total_filtered * 100
        pct_of_bin = count / in_floor_bin * 100
        lines.append(
            f"    {orig_val:.0f} {eff_label} ({count:,} / {total_filtered:,} homes, "
            f"{pct_of_total:.1f}%) --> {floor_display:.0f} {eff_label} "
            f"({count:,} / {in_floor_bin:,} in bin, {pct_of_bin:.1f}%)"
        )

    # Show homes that were already at the floor (completes the bin composition)
    originally_at_floor = int(
        ((original * display_scale).round(1) == round(floor_display, 1)).sum()
    )
    if originally_at_floor > 0:
        pct_of_bin = originally_at_floor / in_floor_bin * 100
        lines.append(
            f"    {floor_display:.0f} {eff_label} (original) "
            f"({originally_at_floor:,} / {in_floor_bin:,} in bin, {pct_of_bin:.1f}%)"
        )

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Shared analysis engine
# ─────────────────────────────────────────────────────────────────────────────

def _bin_group_summarize(
    df_filtered: pd.DataFrame,
    eff_values: pd.Series,
    cap_values: pd.Series,
    eff_bins: List[int],
    cap_bins: List[int],
    eff_label: str,
    cap_label: str,
    menu_mp: int,
    cost_category: str,
    cost_type: str,
    cost_scenarios: List[str],
    cap_bin_step: int = 1,
    exclude_v3: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """Bin by efficiency × capacity and compute cost percentiles per bin.

    Shared engine for all equipment-type analyses: counts outliers, assigns
    bins, groups by (capacity, efficiency), and computes P10/P50/P90 for
    each cost scenario.

    Args:
        df_filtered: Pre-filtered DataFrame (only matching homes).
        eff_values: Efficiency values (SEER or AFUE %) aligned to df_filtered.
        cap_values: Capacity values in bin units, aligned to df_filtered.
        eff_bins: Efficiency bin centers (e.g. SEER_BINS).
        cap_bins: Capacity bin centers (e.g. CAPACITY_BINS_TONS).
        eff_label: Display label — 'SEER' or 'AFUE'.
        cap_label: Display label — 'Capacity (tons)' or 'Capacity (kBTU/h)'.
        menu_mp: Measure package number.
        cost_category: 'heating' or 'cooling' (for cost column name).
        cost_type: 'replacement' or 'upgrade'.
        cost_scenarios: List of cost scenario keys.
        cap_bin_step: Bin step size (1 for tons/SEER, 10 for kBTU/h).
        exclude_v3: If True, set v3 columns to N/A (for cooling).

    Returns:
        Tuple of (results DataFrame, outlier info dict).
    """
    # Derive outlier thresholds from bin range ± half-step
    cap_lo = cap_bins[0] - cap_bin_step / 2
    cap_hi = cap_bins[-1] + cap_bin_step / 2
    eff_lo = eff_bins[0] - 0.5
    eff_hi = eff_bins[-1] + 0.5

    cap_below, cap_above = _count_outliers(cap_values, cap_lo, cap_hi)
    eff_below, eff_above = _count_outliers(eff_values, eff_lo, eff_hi)
    cap_nan = int(cap_values.isna().sum())
    eff_nan = int(eff_values.isna().sum())
    outliers = {
        'total_filtered': len(df_filtered),
        'cap_below': cap_below, 'cap_above': cap_above,
        'eff_below': eff_below, 'eff_above': eff_above,
        'cap_nan': cap_nan, 'eff_nan': eff_nan,
        'cap_lo': cap_lo, 'cap_hi': cap_hi,
        'eff_lo': eff_lo, 'eff_hi': eff_hi,
    }

    # Bin values — build a slim DataFrame with only the columns we need
    # to avoid copying the entire wide source DataFrame (memory-intensive).
    compute_scenarios = [s for s in cost_scenarios if not (exclude_v3 and s == 'v3')]

    cost_cols = []
    for scenario in compute_scenarios:
        col = create_cost_col(menu_mp=menu_mp, category=cost_category,
                              cost_type=cost_type, cost_scenario=scenario)
        if col in df_filtered.columns:
            cost_cols.append(col)

    df_work = df_filtered[cost_cols].copy()
    df_work['_eff_bin'] = _round_to_bin(eff_values, eff_bins)
    df_work['_cap_bin'] = _round_to_bin(cap_values, cap_bins, step=cap_bin_step)
    df_work = df_work.dropna(subset=['_eff_bin', '_cap_bin'])
    if len(df_work) == 0:
        return pd.DataFrame(), outliers

    # Build results via groupby — replaces manual cap × eff nested loop
    grouped = df_work.groupby(['_cap_bin', '_eff_bin'])
    stats_parts: Dict[str, pd.Series] = {}
    for scenario in compute_scenarios:
        col = create_cost_col(menu_mp=menu_mp, category=cost_category,
                              cost_type=cost_type, cost_scenario=scenario)
        if col in df_work.columns:
            g = grouped[col]
            stats_parts[f'{scenario} N'] = g.count()
            stats_parts[f'{scenario} P10'] = g.quantile(0.10)
            stats_parts[f'{scenario} P50'] = g.quantile(0.50)
            stats_parts[f'{scenario} P90'] = g.quantile(0.90)
        else:
            stats_parts[f'{scenario} N'] = 0
            for stat in ['P10', 'P50', 'P90']:
                stats_parts[f'{scenario} {stat}'] = np.nan

    results = pd.DataFrame(stats_parts).reset_index()
    results = results.rename(columns={'_cap_bin': cap_label, '_eff_bin': eff_label})
    results[cap_label] = results[cap_label].astype(int)
    results[eff_label] = results[eff_label].astype(int)

    # v3 = N/A for cooling
    if exclude_v3 and 'v3' in cost_scenarios:
        results['v3 N'] = 0
        for stat in ['P10', 'P50', 'P90']:
            results[f'v3 {stat}'] = np.nan

    return results, outliers


# ─────────────────────────────────────────────────────────────────────────────
# Equipment-specific filter + extraction (thin wrappers)
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_ashp(df: pd.DataFrame, menu_mp: int, cost_scenarios: List[str],
                  cost_type: str = 'replacement') -> Tuple[pd.DataFrame, Dict]:
    """Analyze ASHP heating costs by SEER × capacity (tons)."""
    mask = pd.Series(True, index=df.index)
    if cost_type == 'replacement':
        if 'heating_type' in df.columns:
            mask &= (df['heating_type'] == 'Electricity ASHP')
        elif 'hvac_heating_type_and_fuel' in df.columns:
            mask &= (df['hvac_heating_type_and_fuel'] == 'Electricity ASHP')
        else:
            return pd.DataFrame(), {}
        if 'hvac_has_ducts' in df.columns:
            mask &= (df['hvac_has_ducts'] == 'Yes')
    else:
        pm2_col = f'heating_{cost_type}_pm2_euss'
        if pm2_col in df.columns:
            mask &= df[pm2_col].notna() & (df[pm2_col] > 0)
        else:
            return pd.DataFrame(), {}

    df_f = df.loc[mask]
    if len(df_f) == 0:
        return pd.DataFrame(), {'total_filtered': 0}

    # Efficiency for binning: use FLOORED pm2 so bins reflect what the cost
    # regression actually computed.  Clamping summary (below) shows where
    # homes in the floor bin came from.
    pm2_col = f'heating_{cost_type}_pm2_euss'
    pm2_original_col = f'heating_{cost_type}_pm2_euss_original'

    if pm2_col in df_f.columns:
        eff = df_f[pm2_col]
    elif 'hvac_heating_efficiency' in df_f.columns:
        eff = _extract_seer(df_f['hvac_heating_efficiency'])
    else:
        return pd.DataFrame(), {}
        
    # Capacity source depends on cost_type: replacement costs are now priced
    # off the OLD system's own size (see calculate_equipment_replacement_costs.py,
    # 20 Aug 2026 fix); upgrade costs are priced off the new heat pump's size.
    capacity_col = ('base_size_heating_system_primary_k_btu_h' if cost_type == 'replacement'
                    else 'size_heating_system_primary_k_btu_h')
    if capacity_col not in df_f.columns:
        return pd.DataFrame(), {}
    cap = _capacity_tons(df_f[capacity_col])

    results, outliers = _bin_group_summarize(
        df_f, eff, cap,
        eff_bins=SEER_BINS, cap_bins=CAPACITY_BINS_TONS,
        eff_label='SEER', cap_label='Capacity (tons)',
        menu_mp=menu_mp, cost_category='heating',
        cost_type=cost_type, cost_scenarios=cost_scenarios,
    )

    # Attach clamping summary for replacement metrics
    outliers['clamping_lines'] = _build_clamping_summary(
        df_f, pm2_col, pm2_original_col, eff_label='SEER', display_scale=1.0
    )

    return results, outliers


def _analyze_central_ac(df: pd.DataFrame, menu_mp: int, cost_scenarios: List[str],
                         cost_type: str = 'replacement') -> Tuple[pd.DataFrame, Dict]:
    """Analyze Central AC cooling costs by SEER × capacity (tons)."""
    mask = pd.Series(True, index=df.index)
    if cost_type == 'replacement':
        if 'hvac_cooling_type' in df.columns:
            mask &= (df['hvac_cooling_type'] == 'Central AC')
        else:
            return pd.DataFrame(), {}
    else:
        pm2_col = f'cooling_{cost_type}_pm2_euss'
        if pm2_col in df.columns:
            mask &= df[pm2_col].notna() & (df[pm2_col] > 0)
        else:
            return pd.DataFrame(), {}

    df_f = df.loc[mask]
    if len(df_f) == 0:
        return pd.DataFrame(), {'total_filtered': 0}

    # Efficiency for binning: use FLOORED pm2 so bins reflect what the cost
    # regression actually computed.
    pm2_col = f'cooling_{cost_type}_pm2_euss'
    pm2_original_col = f'cooling_{cost_type}_pm2_euss_original'

    if pm2_col in df_f.columns:
        eff = df_f[pm2_col]
    elif 'hvac_cooling_efficiency' in df_f.columns:
        eff = _extract_seer(df_f['hvac_cooling_efficiency'])
    else:
        return pd.DataFrame(), {}

    # Capacity source depends on cost_type: replacement costs are now priced
    # off the OLD system's own size (see calculate_equipment_replacement_costs.py,
    # 20 Aug 2026 fix); upgrade costs are priced off the new heat pump's size.
    capacity_col = ('base_size_cooling_system_primary_k_btu_h' if cost_type == 'replacement'
                    else 'size_cooling_system_primary_k_btu_h')
    if capacity_col not in df_f.columns:
        return pd.DataFrame(), {}
    cap = _capacity_tons(df_f[capacity_col])

    results, outliers = _bin_group_summarize(
        df_f, eff, cap,
        eff_bins=SEER_BINS, cap_bins=CAPACITY_BINS_TONS,
        eff_label='SEER', cap_label='Capacity (tons)',
        menu_mp=menu_mp, cost_category='cooling',
        cost_type=cost_type, cost_scenarios=cost_scenarios,
        exclude_v3=True,
    )

    outliers['clamping_lines'] = _build_clamping_summary(
        df_f, pm2_col, pm2_original_col, eff_label='SEER', display_scale=1.0
    )

    return results, outliers


def _analyze_furnace(df: pd.DataFrame, menu_mp: int, cost_scenarios: List[str],
                      fuel_type: str = 'Natural Gas',
                      cost_type: str = 'replacement') -> Tuple[pd.DataFrame, Dict]:
    """Analyze Furnace heating costs by AFUE × capacity (kBTU/h)."""
    mask = pd.Series(True, index=df.index)
    if cost_type == 'replacement':
        if 'base_heating_fuel' in df.columns:
            mask &= (df['base_heating_fuel'] == fuel_type)
        if 'heating_type' in df.columns:
            mask &= df['heating_type'].str.contains('Furnace', case=False, na=False)
        elif 'hvac_heating_type_and_fuel' in df.columns:
            mask &= df['hvac_heating_type_and_fuel'].str.contains('Furnace', case=False, na=False)
    else:
        pm2_col = f'heating_{cost_type}_pm2_euss'
        if pm2_col in df.columns:
            mask &= df[pm2_col].notna() & (df[pm2_col] > 0)
        else:
            return pd.DataFrame(), {}

    df_f = df.loc[mask]
    if len(df_f) == 0:
        return pd.DataFrame(), {'total_filtered': 0}

    # AFUE: pm2 stores as decimal (0.80) → multiply by 100 for % binning.
    # _extract_afue() already returns percentage-scale values, so only
    # the pm2 column paths need the ×100 conversion.
    # Efficiency for binning: use FLOORED pm2 so bins reflect what the
    # cost regression actually computed.
    pm2_col = f'heating_{cost_type}_pm2_euss'
    pm2_original_col = f'heating_{cost_type}_pm2_euss_original'

    if pm2_col in df_f.columns:
        eff = df_f[pm2_col] * 100
    elif 'hvac_heating_efficiency' in df_f.columns:
        eff = _extract_afue(df_f['hvac_heating_efficiency'])
    else:
        return pd.DataFrame(), {}

    # Capacity source depends on cost_type: replacement costs are now priced
    # off the OLD system's own size (see calculate_equipment_replacement_costs.py,
    # 20 Aug 2026 fix); upgrade costs are priced off the new heat pump's size.
    capacity_col = ('base_size_heating_system_primary_k_btu_h' if cost_type == 'replacement'
                    else 'size_heating_system_primary_k_btu_h')
    if capacity_col not in df_f.columns:
        return pd.DataFrame(), {}
    cap = df_f[capacity_col]

    results, outliers = _bin_group_summarize(
        df_f, eff, cap,
        eff_bins=AFUE_BINS, cap_bins=CAPACITY_BINS_KBTUH,
        eff_label='AFUE', cap_label='Capacity (kBTU/h)',
        menu_mp=menu_mp, cost_category='heating',
        cost_type=cost_type, cost_scenarios=cost_scenarios,
        cap_bin_step=10,
    )

    # display_scale=100 converts decimal AFUE (0.60) to percentage (60) for summary
    outliers['clamping_lines'] = _build_clamping_summary(
        df_f, pm2_col, pm2_original_col, eff_label='AFUE', display_scale=100.0
    )

    return results, outliers


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

        # Print clamping impact summary (replacement metrics only)
        clamping_lines = outliers.get('clamping_lines')
        if clamping_lines:
            print()
            for line in clamping_lines:
                print(line)

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

    # Build display DataFrame — vectorized formatting replaces iterrows()
    df_display = df_result[id_cols].copy()
    for scenario in cost_scenarios:
        n_col = f'{scenario} N'
        if n_col not in df_result.columns:
            continue
        n_vals = df_result[n_col].fillna(0).astype(int)
        has_data = n_vals > 0
        df_display[n_col] = np.where(
            has_data, n_vals.apply(lambda x: f'{x:,}'), '—'
        )
        for stat in ['P10', 'P50', 'P90']:
            stat_col = f'{scenario} {stat}'
            df_display[stat_col] = np.where(
                has_data, df_result[stat_col].apply(_format_dollar), 'N/A'
            )
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
        Dict structured as results[category][technology][cost_type] = DataFrame:
          - category: 'heating', 'cooling'
          - technology: 'ashp', 'central_ac', 'gas_furnace', 'propane_furnace'
          - cost_type: 'replacement', 'upgrade'
        Each DataFrame contains numeric columns: Capacity, Efficiency rating,
        and per-scenario N/P10/P50/P90 values (e.g. 'v4MID N', 'v4MID P50').
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
    # in those DataFrames (not yet merged into the main df). Merge only the
    # missing columns to avoid copying the entire wide DataFrame.
    extra_cols = {}

    if capital_costs_mpx is not None:
        for cost_type in ['replacement', 'upgrade']:
            for scenario_key in cost_scenarios:
                if scenario_key == 'v3':
                    continue  # v3 already in df
                scenario_df = capital_costs_mpx.get('heating', {}).get(cost_type, {}).get(scenario_key)
                if scenario_df is not None:
                    # Pull heating/cooling cost + pm2 columns
                    cols_to_pull = [
                        create_cost_col(menu_mp=menu_mp, category='heating',
                                        cost_type=cost_type, cost_scenario=scenario_key),
                        create_cost_col(menu_mp=menu_mp, category='cooling',
                                        cost_type=cost_type, cost_scenario=scenario_key),
                        f'heating_{cost_type}_pm2_euss',
                        f'heating_{cost_type}_pm2_euss_original',
                        f'cooling_{cost_type}_pm2_euss',
                        f'cooling_{cost_type}_pm2_euss_original',
                    ]

                    for col in cols_to_pull:
                        if col not in df.columns and col in scenario_df.columns:
                            extra_cols[col] = scenario_df[col].values

    # Build a lightweight view: original df + any extra columns from capital_costs_mpx
    if extra_cols:
        df_work = df.assign(**extra_cols)
    else:
        df_work = df

    total_homes = len(df_work)
    results = {}
    outlier_info = {}

    # ── Analysis specifications ──
    # Each entry defines one equipment-type analysis to run.
    # 'category', 'technology', and 'cost_type' define the structured return dict keys:
    #   results_structured[category][technology][cost_type] = DataFrame
    analyses = [
        {
            'label': 'ASHP (Heating Replacement)',
            'category': 'heating',
            'technology': 'ashp',
            'cost_type': 'replacement',
            'title': 'ASHP — Air Source Heat Pump (Heating Replacement, Centrally Ducted)',
            'fn': _analyze_ashp,
            'fn_kwargs': {'cost_type': 'replacement'},
            'id_cols': ['Capacity (tons)', 'SEER'],
            'cap_unit': 'tons', 'eff_label': 'SEER',
            'notes': [
                'Cost type: heating replacement (like-for-like ASHP)',
                'Filter: heating_type = Electricity ASHP, hvac_has_ducts = Yes',
                'SEER bins: heating_replacement_pm2_euss (floored efficiency, see clamping summary)',
                'Costs: computed using floored efficiency (see Section 5 of protocol)',
            ],
        },
        {
            'label': 'Central AC (Cooling Replacement)',
            'category': 'cooling',
            'technology': 'central_ac',
            'cost_type': 'replacement',
            'title': 'Central AC — Centrally Ducted (Cooling Replacement)',
            'fn': _analyze_central_ac,
            'fn_kwargs': {'cost_type': 'replacement'},
            'id_cols': ['Capacity (tons)', 'SEER'],
            'cap_unit': 'tons', 'eff_label': 'SEER',
            'notes': [
                'Cost type: cooling replacement (like-for-like Central AC)',
                'Filter: hvac_cooling_type = Central AC',
                'SEER bins: cooling_replacement_pm2_euss (floored efficiency, see clamping summary)',
                'Costs: computed using floored efficiency (see Section 5 of protocol)',
                'v3 note: No v3 data exists for cooling replacement — v3 columns show N/A',
            ],
        },
        {
            'label': 'Gas Furnace (Heating Replacement)',
            'category': 'heating',
            'technology': 'gas_furnace',
            'cost_type': 'replacement',
            'title': 'Gas Furnace — Natural Gas (Heating Replacement)',
            'fn': _analyze_furnace,
            'fn_kwargs': {'fuel_type': 'Natural Gas', 'cost_type': 'replacement'},
            'id_cols': ['Capacity (kBTU/h)', 'AFUE'],
            'cap_unit': 'kBTU/h', 'eff_label': 'AFUE',
            'notes': [
                'Cost type: heating replacement (like-for-like furnace)',
                'Filter: base_heating_fuel = Natural Gas, heating_type contains Furnace',
                'AFUE bins: heating_replacement_pm2_euss (floored efficiency, see clamping summary)',
                'Costs: computed using floored efficiency (see Section 5 of protocol)',
            ],
        },
        {
            'label': 'Propane Furnace (Heating Replacement)',
            'category': 'heating',
            'technology': 'propane_furnace',
            'cost_type': 'replacement',
            'title': 'Propane Furnace (Heating Replacement)',
            'fn': _analyze_furnace,
            'fn_kwargs': {'fuel_type': 'Propane', 'cost_type': 'replacement'},
            'id_cols': ['Capacity (kBTU/h)', 'AFUE'],
            'cap_unit': 'kBTU/h', 'eff_label': 'AFUE',
            'notes': [
                'Cost type: heating replacement (like-for-like furnace)',
                'Filter: base_heating_fuel = Propane, heating_type contains Furnace',
                'AFUE bins: heating_replacement_pm2_euss (floored efficiency, see clamping summary)',
                'Costs: computed using floored efficiency (see Section 5 of protocol)',
            ],
        },
        {
            'label': 'ASHP (Heating Upgrade)',
            'category': 'heating',
            'technology': 'ashp',
            'cost_type': 'upgrade',
            'title': 'ASHP — Air Source Heat Pump (Heating Upgrade)',
            'fn': _analyze_ashp,
            'fn_kwargs': {'cost_type': 'upgrade'},
            'id_cols': ['Capacity (tons)', 'SEER'],
            'cap_unit': 'tons', 'eff_label': 'SEER',
            'notes': [
                'Cost type: heating upgrade (new ASHP installation)',
                'Filter: all homes with valid heating_upgrade_pm2_euss',
                'SEER source: heating_upgrade_pm2_euss (MP-defined upgrade efficiency)',
                'Costs: computed using floored efficiency (see Section 5 of protocol)',
            ],
        },
        {
            'label': 'Central AC (Cooling Upgrade)',
            'category': 'cooling',
            'technology': 'central_ac',
            'cost_type': 'upgrade',
            'title': 'Central AC (Cooling Upgrade)',
            'fn': _analyze_central_ac,
            'fn_kwargs': {'cost_type': 'upgrade'},
            'id_cols': ['Capacity (tons)', 'SEER'],
            'cap_unit': 'tons', 'eff_label': 'SEER',
            'notes': [
                'Cost type: cooling upgrade',
                'Filter: all homes with valid cooling_upgrade_pm2_euss',
                'SEER source: cooling_upgrade_pm2_euss (MP-defined upgrade efficiency)',
                'Costs: computed using floored efficiency (see Section 5 of protocol)',
                'v3 note: No v3 data exists for cooling upgrade — v3 columns show N/A',
            ],
        },
    ]

    # Structured results dict: results_structured[category][technology][cost_type] = DataFrame
    results_structured = {}

    # ── Run all analyses ──
    for spec in analyses:
        df_result, oi = spec['fn'](
            df_work, menu_mp, cost_scenarios, **spec['fn_kwargs']
        )
        results[spec['label']] = df_result
        outlier_info[spec['label']] = oi

        # Populate structured dict
        cat = spec['category']
        tech = spec['technology']
        ct = spec['cost_type']
        results_structured.setdefault(cat, {}).setdefault(tech, {})[ct] = df_result

        _print_table(
            title=spec['title'],
            df_result=df_result,
            cost_scenarios=cost_scenarios,
            id_cols=spec['id_cols'],
            notes=spec['notes'],
            outliers=oi,
            cap_unit=spec['cap_unit'],
            eff_label=spec['eff_label'],
        )

    # ── Summary ──
    print(f"\n{'#' * 110}")
    print(f"#  VALIDATION SUMMARY")
    print(f"#  Total homes in DataFrame: {total_homes:,}")
    print(f"{'#' * 110}")
    for label, df_r in results.items():
        if df_r.empty:
            print(f"  {label:<45}  No matching homes")
        else:
            # Use the max N across ALL scenarios (not just the first) to avoid
            # the v3 issue where Central AC has no v3 data.
            n_cols = [f'{s} N' for s in cost_scenarios if f'{s} N' in df_r.columns]
            if n_cols:
                # For each row, take the max N across scenarios
                max_n_per_row = df_r[n_cols].max(axis=1)
                total_n = int(max_n_per_row.sum())
                n_bins_with_data = int((max_n_per_row > 0).sum())
            else:
                total_n = 0
                n_bins_with_data = 0

            total_bins = len(df_r)

            # Percentage of total homes in DataFrame
            pct_of_total = (total_n / total_homes * 100) if total_homes > 0 else 0.0

            # Percentage of appliance-filtered homes (from outlier info)
            oi = outlier_info.get(label, {})
            appliance_filtered = oi.get('total_filtered', total_n)
            pct_of_appliance = (total_n / appliance_filtered * 100) if appliance_filtered > 0 else 0.0

            print(f"  {label:<45}  {n_bins_with_data}/{total_bins} bins with data  |  "
                  f"{total_n:,} homes matched  |  "
                  f"{pct_of_total:.1f}% of all homes  |  "
                  f"{pct_of_appliance:.1f}% of {appliance_filtered:,} filtered")

    print(f"{'#' * 110}\n")

    return results_structured


# ─────────────────────────────────────────────────────────────────────────────
# Population filters (shared by the distribution figure and the disaggregation
# tables above, so both describe the same homes for a given equipment type)
# ─────────────────────────────────────────────────────────────────────────────

def get_ashp_upgrade_homes(df: pd.DataFrame) -> pd.DataFrame:
    """Homes in the ASHP heating-upgrade population.

    heating_upgrade_pm2_euss is populated for nearly every row regardless of
    adoption eligibility -- the new heat pump's spec doesn't depend on the
    home's baseline -- so include_heating is applied explicitly here.
    _analyze_ashp's own N counts stay correct without this gate because they
    group on the cost column (NaN for ineligible homes), not on this
    population directly.

    Args:
        df: Home-level DataFrame with heating_upgrade_pm2_euss and
            include_heating columns.

    Returns:
        The subset of df in the ASHP upgrade population.
    """
    pm2_col = 'heating_upgrade_pm2_euss'
    mask = df[pm2_col].notna() & (df[pm2_col] > 0) & df['include_heating']
    return df.loc[mask]


def get_central_ac_replacement_homes(df: pd.DataFrame) -> pd.DataFrame:
    """Homes in the baseline Central AC replacement population.

    Args:
        df: Home-level DataFrame with hvac_cooling_type and include_cooling
            columns.

    Returns:
        The subset of df with a baseline Central AC.
    """
    return df.loc[(df['hvac_cooling_type'] == 'Central AC') & df['include_cooling']]


def get_natural_gas_furnace_replacement_homes(df: pd.DataFrame) -> pd.DataFrame:
    """Homes in the baseline natural gas furnace replacement population.

    A plain 'Furnace' substring match on heating_type also catches Wall/Floor
    Furnace homes, which are excluded from the modeled population (invalid
    heating tech -- include_heating = False), so that gate is applied here.

    Args:
        df: Home-level DataFrame with base_heating_fuel, heating_type, and
            include_heating columns.

    Returns:
        The subset of df with a baseline natural gas furnace.
    """
    mask = (
        (df['base_heating_fuel'] == 'Natural Gas')
        & df['heating_type'].str.contains('Furnace', case=False, na=False)
        & df['include_heating']
    )
    return df.loc[mask]


# ─────────────────────────────────────────────────────────────────────────────
# Distribution figure
# ─────────────────────────────────────────────────────────────────────────────

def build_capital_cost_distribution_figure(
    df_mp3: pd.DataFrame,
    df_mp4: pd.DataFrame,
    figure_size: Tuple[int, int] = (28, 16),
    bin_number: str = 'auto',
    lower_percentile: float = 2.5,
    upper_percentile: float = 97.5,
) -> Figure:
    """Build the ASHP / Central AC / NG Furnace consumption-and-size grid.

    One figure, 3 equipment rows (ASHP upgrade, Central AC replacement,
    natural gas Furnace replacement) x 4 columns (MP3 consumption, MP3 size,
    MP4 consumption, MP4 size), color-coded by base_heating_fuel. Each row
    uses the same population as the matching table from
    run_capital_cost_validation, via get_ashp_upgrade_homes,
    get_central_ac_replacement_homes, and
    get_natural_gas_furnace_replacement_homes.

    Each panel's display range is trimmed to [lower_percentile,
    upper_percentile] of that panel's own column -- the full min-max range
    compresses every panel near zero because of a long right tail of
    outliers, so a default 95% CI view (2.5-97.5) keeps the visible bins
    informative. This only affects what's plotted; the underlying
    populations and cost tables are unchanged.

    Args:
        df_mp3: MP3 home-level DataFrame (e.g. DATAFRAMES_BY_MP[3]['fixed_base']).
        df_mp4: MP4 home-level DataFrame (e.g. DATAFRAMES_BY_MP[4]['fixed_base']).
        figure_size: Figure (width, height) in inches.
        bin_number: Passed through to create_subplot_grid_histogram.
        lower_percentile: Lower bound (0-100) of each panel's display range.
        upper_percentile: Upper bound (0-100) of each panel's display range.

    Returns:
        The matplotlib Figure.
    """
    df_ashp_mp3 = get_ashp_upgrade_homes(df_mp3)
    df_ashp_mp4 = get_ashp_upgrade_homes(df_mp4)
    df_cac_mp3 = get_central_ac_replacement_homes(df_mp3)
    df_cac_mp4 = get_central_ac_replacement_homes(df_mp4)
    df_furnace_mp3 = get_natural_gas_furnace_replacement_homes(df_mp3)
    df_furnace_mp4 = get_natural_gas_furnace_replacement_homes(df_mp4)

    print(f"ASHP upgrade population:      MP3 {len(df_ashp_mp3):,} | MP4 {len(df_ashp_mp4):,}")
    print(f"Central AC replacement pop.:  MP3 {len(df_cac_mp3):,} | MP4 {len(df_cac_mp4):,}")
    print(f"NG Furnace replacement pop.:  MP3 {len(df_furnace_mp3):,} | MP4 {len(df_furnace_mp4):,}")

    dataframes = [
        df_ashp_mp3, df_ashp_mp4,
        df_cac_mp3, df_cac_mp4,
        df_furnace_mp3, df_furnace_mp4,
    ]
    subplot_positions = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2), (2, 3),
    ]
    dataframe_indices = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    x_cols = [
        'mp3_heating_consumption', 'size_heating_system_primary_k_btu_h',
        'mp4_heating_consumption', 'size_heating_system_primary_k_btu_h',
        'base_electricity_cooling_consumption', 'base_size_cooling_system_primary_k_btu_h',
        'base_electricity_cooling_consumption', 'base_size_cooling_system_primary_k_btu_h',
        'base_naturalGas_heating_consumption', 'base_size_heating_system_primary_k_btu_h',
        'base_naturalGas_heating_consumption', 'base_size_heating_system_primary_k_btu_h',
    ]
    subplot_titles = [
        'ASHP Heating Consumption (MP3)', 'ASHP Heating Size (MP3)',
        'ASHP Heating Consumption (MP4)', 'ASHP Heating Size (MP4)',
        'Central AC Cooling Consumption (MP3)', 'Central AC Cooling Size (MP3)',
        'Central AC Cooling Consumption (MP4)', 'Central AC Cooling Size (MP4)',
        'NG Furnace Heating Consumption (MP3)', 'NG Furnace Heating Size (MP3)',
        'NG Furnace Heating Consumption (MP4)', 'NG Furnace Heating Size (MP4)',
    ]
    x_labels = [
        'Heating Consumption (kWh)', 'Heating Size (kBTU/h)',
        'Heating Consumption (kWh)', 'Heating Size (kBTU/h)',
        'Cooling Consumption (kWh)', 'Cooling Size (kBTU/h)',
        'Cooling Consumption (kWh)', 'Cooling Size (kBTU/h)',
        'Heating Consumption (therms)', 'Heating Size (kBTU/h)',
        'Heating Consumption (therms)', 'Heating Size (kBTU/h)',
    ]

    suptitle = (
        'ASHP / Central AC / Furnace (NG) -- Consumption and Size Distributions '
        f'(each panel shown at its {lower_percentile:g}-{upper_percentile:g} percentile range)'
    )
    return create_subplot_grid_histogram(
        dataframes=dataframes,
        dataframe_indices=dataframe_indices,
        subplot_positions=subplot_positions,
        x_cols=x_cols,
        x_labels=x_labels,
        subplot_titles=subplot_titles,
        suptitle=suptitle,
        figure_size=figure_size,
        color_code='base_heating_fuel',
        bin_number=bin_number,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        show_legend=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Disaggregation workbook (matches the Equipment_Installed_TARE reference
# workbook's sheet names and block layout, minus its v3 columns)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CAPITAL_COST_MP_LABELS: Dict[int, str] = {
    3: 'MP3 - Min Efficiency',
    4: 'MP4 - High Efficiency',
}


def _write_capital_cost_block(
    ws,
    start_row: int,
    df_result: pd.DataFrame,
    menu_mp: int,
    technology: str,
    cost_type_label: str,
    id_cols: List[str],
    row_label: str,
    mp_labels: Dict[int, str],
) -> int:
    """Write one MP block (title, header, data rows) to an Excel worksheet.

    Mirrors the Equipment_Installed_TARE reference workbook's block layout:
    a bold title row naming the row count, a blank row, a bold header row,
    then one row per capacity x efficiency bin.

    Args:
        ws: openpyxl worksheet to write into.
        start_row: 1-indexed row to start this block at.
        df_result: One equipment/cost-type table from
            run_capital_cost_validation's structured results.
        menu_mp: Measure package number, used to look up mp_labels.
        technology: Display string for the 'technology' column.
        cost_type_label: Display string for the 'cost type' column.
        id_cols: The capacity/efficiency identifier columns, in display order.
        row_label: Prefix for the block's title row (e.g. 'MP3, ashp, upgrade').
        mp_labels: Menu package number -> EUSS Measure Package display label.

    Returns:
        The next free row after this block (for chaining the next block).
    """
    n_rows = len(df_result)
    bold = Font(bold=True)
    ws.cell(row=start_row, column=1, value=f'{row_label}: {n_rows} rows').font = bold

    header_row = start_row + 2
    columns = ['index', 'EUSS Measure Package', 'technology', 'cost type'] + id_cols + [
        'v4MID N', 'v4MID P10', 'v4MID P50', 'v4MID P90',
    ]
    for col_idx, col_name in enumerate(columns, start=1):
        ws.cell(row=header_row, column=col_idx, value=col_name).font = bold

    df_sorted = df_result.sort_values(id_cols).reset_index(drop=True)
    data_start_row = header_row + 1
    for i, row in df_sorted.iterrows():
        r = data_start_row + i
        ws.cell(row=r, column=1, value=i)
        ws.cell(row=r, column=2, value=mp_labels[menu_mp])
        ws.cell(row=r, column=3, value=technology)
        ws.cell(row=r, column=4, value=cost_type_label)
        for j, id_col in enumerate(id_cols):
            ws.cell(row=r, column=5 + j, value=row[id_col])
        base_col = 5 + len(id_cols)
        ws.cell(row=r, column=base_col + 0, value=row['v4MID N'])
        ws.cell(row=r, column=base_col + 1, value=row['v4MID P10'])
        ws.cell(row=r, column=base_col + 2, value=row['v4MID P50'])
        ws.cell(row=r, column=base_col + 3, value=row['v4MID P90'])

    return data_start_row + n_rows + 2  # blank row + next block's title row


def build_capital_cost_disaggregation_workbook(
    results_mp3: Dict,
    results_mp4: Dict,
    mp_labels: Optional[Dict[int, str]] = None,
) -> Workbook:
    """Build the 3-sheet REMDB v4MID capital-cost disaggregation workbook.

    Combines the ASHP upgrade, Central AC replacement, and natural gas
    Furnace replacement tables (from run_capital_cost_validation, MP3 and
    MP4) into one workbook whose sheet names, block layout, and column order
    match the Equipment_Installed_TARE reference workbook, minus its v3
    columns.

    Args:
        results_mp3: Structured results dict from
            run_capital_cost_validation(df=..., menu_mp=3, cost_scenarios=['v4MID']).
        results_mp4: Same, for menu_mp=4.
        mp_labels: Menu package number -> EUSS Measure Package display label.
            Defaults to DEFAULT_CAPITAL_COST_MP_LABELS.

    Returns:
        An openpyxl Workbook with sheets TARE_Disag_Capex_ASHP,
        TARE_Disag_Capex_CAC, TARE_Disag_Capex_Furnace.
    """
    if mp_labels is None:
        mp_labels = DEFAULT_CAPITAL_COST_MP_LABELS

    ashp_mp3 = results_mp3['heating']['ashp']['upgrade']
    ashp_mp4 = results_mp4['heating']['ashp']['upgrade']
    cac_mp3 = results_mp3['cooling']['central_ac']['replacement']
    cac_mp4 = results_mp4['cooling']['central_ac']['replacement']
    fur_mp3 = results_mp3['heating']['gas_furnace']['replacement']
    fur_mp4 = results_mp4['heating']['gas_furnace']['replacement']

    wb = Workbook()
    wb.remove(wb.active)

    ws_ashp = wb.create_sheet('TARE_Disag_Capex_ASHP')
    row = _write_capital_cost_block(
        ws_ashp, 1, ashp_mp3, menu_mp=3, technology='ASHP',
        cost_type_label='Upgrade (MP3)', id_cols=['Capacity (tons)', 'SEER'],
        row_label='MP3, ashp, upgrade', mp_labels=mp_labels,
    )
    _write_capital_cost_block(
        ws_ashp, row, ashp_mp4, menu_mp=4, technology='ASHP',
        cost_type_label='Upgrade (MP4)', id_cols=['Capacity (tons)', 'SEER'],
        row_label='MP4, ashp, upgrade', mp_labels=mp_labels,
    )

    ws_cac = wb.create_sheet('TARE_Disag_Capex_CAC')
    row = _write_capital_cost_block(
        ws_cac, 1, cac_mp3, menu_mp=3, technology='Central AC',
        cost_type_label='Replacement (MP0)', id_cols=['Capacity (tons)', 'SEER'],
        row_label='MP3, central ac, replacement', mp_labels=mp_labels,
    )
    _write_capital_cost_block(
        ws_cac, row, cac_mp4, menu_mp=4, technology='Central AC',
        cost_type_label='Replacement (MP0)', id_cols=['Capacity (tons)', 'SEER'],
        row_label='MP4, central ac, replacement', mp_labels=mp_labels,
    )

    ws_furnace = wb.create_sheet('TARE_Disag_Capex_Furnace')
    row = _write_capital_cost_block(
        ws_furnace, 1, fur_mp3, menu_mp=3, technology='Furnace (Natural Gas)',
        cost_type_label='Replacement (MP0)', id_cols=['Capacity (kBTU/h)', 'AFUE'],
        row_label='MP3, gas_furnace, replacement', mp_labels=mp_labels,
    )
    _write_capital_cost_block(
        ws_furnace, row, fur_mp4, menu_mp=4, technology='Furnace (Natural Gas)',
        cost_type_label='Replacement (MP0)', id_cols=['Capacity (kBTU/h)', 'AFUE'],
        row_label='MP4, gas_furnace, replacement', mp_labels=mp_labels,
    )

    return wb
