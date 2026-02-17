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

# Capacity bins (kBTU/h) — 40 through 200 in steps of 10
CAPACITY_BINS_KBTUH = list(range(40, 201, 10))

# SEER bins — 13 through 25
SEER_BINS = list(range(13, 26))

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
            f"{pct_of_total:.1f}%) ──→ {floor_display:.0f} {eff_label} "
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

    # Bin values
    df_work = df_filtered.copy()
    df_work['_eff_bin'] = _round_to_bin(eff_values, eff_bins)
    df_work['_cap_bin'] = _round_to_bin(cap_values, cap_bins, step=cap_bin_step)
    df_work = df_work.dropna(subset=['_eff_bin', '_cap_bin'])
    if len(df_work) == 0:
        return pd.DataFrame(), outliers

    # Determine which scenarios to compute
    compute_scenarios = [s for s in cost_scenarios if not (exclude_v3 and s == 'v3')]

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
        
    if 'size_heating_system_primary_k_btu_h' not in df_f.columns:
        return pd.DataFrame(), {}
    cap = _capacity_tons(df_f['size_heating_system_primary_k_btu_h'])

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

    if 'size_cooling_system_primary_k_btu_h' not in df_f.columns:
        return pd.DataFrame(), {}
    cap = _capacity_tons(df_f['size_cooling_system_primary_k_btu_h'])

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

    if 'size_heating_system_primary_k_btu_h' not in df_f.columns:
        return pd.DataFrame(), {}
    cap = df_f['size_heating_system_primary_k_btu_h']

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
                        if col not in df_work.columns and col in scenario_df.columns:
                            df_work[col] = scenario_df[col].values

    total_homes = len(df_work)
    results = {}
    outlier_info = {}

    # ── Analysis specifications ──
    # Each entry defines one equipment-type analysis to run.
    analyses = [
        {
            'label': 'ASHP (Heating Replacement)',
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

    # ── Run all analyses ──
    for spec in analyses:
        df_result, oi = spec['fn'](
            df_work, menu_mp, cost_scenarios, **spec['fn_kwargs']
        )
        results[spec['label']] = df_result
        outlier_info[spec['label']] = oi
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

    return results
