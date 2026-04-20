"""
Adoption KPI computation functions for the TARE model.

This module contains all reusable computation functions for calculating
spark gap, thermal COP, break-even COP, bill savings, and demand
change metrics from EUSS data. Both the preTARE and postTARE notebooks
import from this single source.

Location: cmu_tare_model/adoption_kpis/kpi_functions.py

See README_adoption_kpis.md for methodology notes and design decisions.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config import PROJECT_ROOT
from cmu_tare_model.constants import ALLOWED_HOUSING_TYPES


# ============================================================================
# CONVERSION CONSTANTS (Source: EIA)
# ============================================================================

BTU_PER_CF_NATURAL_GAS = 1038   # BTU per cubic foot of natural gas (EIA average)
BTU_PER_KWH = 3412              # BTU per kilowatt-hour (by definition)

# Derived: natural gas $/1000cf to $/kWh
NG_CONVERSION_FACTOR = BTU_PER_KWH / (1000 * BTU_PER_CF_NATURAL_GAS)

KWH_PER_MMBTU = 293.07107      # 1 MMBTU = 293.07107 kWh
KBTU_PER_KWH = 3.412           # 1 kWh = 3.412 kBtu

STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'DC': 'District of Columbia', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii',
    'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine',
    'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
    'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska',
    'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico',
    'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island',
    'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas',
    'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington',
    'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
}


# ============================================================================
# EUSS DATA PATHS AND COLUMN DEFINITIONS
# ============================================================================

EUSS_DATA_DIR = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "euss_data",
    "resstock_amy2018_release_1.1", "national", "csv"
)

FUEL_PRICES_PATH = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "fuel_prices", "fuel_prices_nominal_2015_2024.csv"
)

SHAPEFILE_PATH = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "electricity_ng_price_ratio",
    "nhgis0011_shapefile_tl2015_us_state_2015", "US_state_2015.shp"
)

# Original EUSS heating energy columns (all in kWh)
HEATING_FUEL_COLS = [
    'out.electricity.heating.energy_consumption.kwh',
    'out.natural_gas.heating.energy_consumption.kwh',
    'out.fuel_oil.heating.energy_consumption.kwh',
    'out.propane.heating.energy_consumption.kwh',
]

HEATING_LOAD_COL = 'out.load.heating.energy_delivered.kbtu'
HP_BACKUP_ELEC_COL = 'out.electricity.heating_hp_bkup.energy_consumption.kwh'
HP_FANS_PUMPS_COL = 'out.electricity.heating_fans_pumps.energy_consumption.kwh'

# ASSUMPTION: Column names match EUSS ResStock output schema.
CLIMATE_ZONE_COL = 'in.ashrae_iecc_climate_zone_2004'
COUNTY_COL = 'in.county'

BASELINE_USECOLS = [
    'bldg_id', 'in.state', 'in.vacancy_status',
    'in.geometry_building_type_recs',
    'in.heating_fuel', 'in.hvac_heating_type_and_fuel',
    'in.hvac_heating_efficiency',
    CLIMATE_ZONE_COL, COUNTY_COL,
    'weight',
] + HEATING_FUEL_COLS + [HEATING_LOAD_COL, HP_BACKUP_ELEC_COL, HP_FANS_PUMPS_COL]

UPGRADE_USECOLS = BASELINE_USECOLS + ['applicability']


# ============================================================================
# HELPER
# ============================================================================

def mp_to_upgrade(mp_num: int) -> str:
    """Convert MP number to EUSS upgrade string (e.g., 4 -> 'upgrade04')."""
    return f"upgrade{mp_num:02d}"


# ASSUMPTION: Benchmark ranges are literature-derived estimates for validation.
# mp3 = ASHP (SEER 15 / HSPF 9), mp4 = ASHP (SEER 24 / HSPF 13).
COP_BENCHMARK_RANGES: dict[str, dict] = {
    '1-3': {'mp3': (2.4, 3.2), 'mp4': (3.0, 4.2), 'label': 'Warm (CZ 1-3)'},
    '4-5': {'mp3': (2.0, 2.8), 'mp4': (2.5, 3.5), 'label': 'Mixed (CZ 4-5)'},
    '6-7': {'mp3': (1.6, 2.4), 'mp4': (2.0, 3.0), 'label': 'Cold (CZ 6-7)'},
}


def iecc_to_cz_group(iecc_zone: str) -> str:
    """Map IECC climate zone string to benchmark group.

    Args:
        iecc_zone: IECC zone string like '4A', '5B', '7A'.

    Returns:
        Benchmark group string: '1-3', '4-5', or '6-7'.

    Raises:
        ValueError: If the numeric prefix is outside 1-7.
    """
    if iecc_zone is None or (isinstance(iecc_zone, float) and np.isnan(iecc_zone)):
        return 'unknown'
    try:
        zone_num = int(str(iecc_zone)[0])
    except (ValueError, IndexError):
        return 'unknown'
    if 1 <= zone_num <= 3:
        return '1-3'
    elif 4 <= zone_num <= 5:
        return '4-5'
    elif 6 <= zone_num <= 7:
        return '6-7'
    else:
        raise ValueError(f"IECC zone numeric prefix {zone_num} outside 1-7: '{iecc_zone}'")


# ============================================================================
# STEP 1b: LOAD EUSS DATA
# ============================================================================

def load_euss_baseline(
    filename: str = 'baseline_metadata_and_annual_results.csv',
) -> pd.DataFrame:
    """Load EUSS baseline CSV, filter to occupied single-family homes.

    Args:
        filename: Baseline CSV filename within EUSS_DATA_DIR.

    Returns:
        DataFrame indexed by bldg_id, filtered to occupied SF homes.
    """
    filepath = os.path.join(EUSS_DATA_DIR, filename)
    print(f"Loading baseline from: {filepath}")
    # df = pd.read_csv(filepath, usecols=BASELINE_USECOLS, index_col='bldg_id')
    df = pd.read_csv(filepath, index_col='bldg_id')

    n_total = len(df)
    df = df[df['in.vacancy_status'] == 'Occupied']
    print(f"  After occupancy filter: {len(df):,} / {n_total:,}")

    df = df[df['in.geometry_building_type_recs'].isin(ALLOWED_HOUSING_TYPES)]
    print(f"  After housing type filter ({ALLOWED_HOUSING_TYPES}): {len(df):,}")

    return df


def load_euss_upgrade(upgrade_name: str) -> pd.DataFrame:
    """Load EUSS upgrade CSV, filter to occupied SF homes where measure was applicable.

    Args:
        upgrade_name: Upgrade identifier (e.g., 'upgrade04').

    Returns:
        DataFrame indexed by bldg_id, filtered to applicable occupied SF homes.
    """
    filename = f'{upgrade_name}_metadata_and_annual_results.csv'
    filepath = os.path.join(EUSS_DATA_DIR, filename)
    print(f"Loading {upgrade_name} from: {filepath}")
    # df = pd.read_csv(filepath, usecols=UPGRADE_USECOLS, index_col='bldg_id')
    df = pd.read_csv(filepath, index_col='bldg_id')

    n_total = len(df)
    df = df[df['in.vacancy_status'] == 'Occupied']
    print(f"  After occupancy filter: {len(df):,} / {n_total:,}")

    df = df[df['in.geometry_building_type_recs'].isin(ALLOWED_HOUSING_TYPES)]
    print(f"  After housing type filter: {len(df):,}")

    df = df[df['applicability'] == True]
    print(f"  After applicability filter: {len(df):,}")

    return df


# ============================================================================
# STEP 2: FUEL PRICE RATIOS
# ============================================================================

def calculate_price_ratios(
    filepath: str,
    year: Union[int, List[int]] = 2024,
) -> pd.DataFrame:
    """Load fuel price data and calculate electricity-to-gas price ratios by state.

    Reads nominal EIA prices from CSV, converts to $/kWh and $/MMBTU,
    and computes the spark gap (electricity/gas price ratio).

    Args:
        filepath: Path to fuel_prices_nominal.csv file.
        year: Single year or list of years to average (default: 2024).

    Returns:
        DataFrame with state-level price comparisons, $/MMBTU values, and spark gap.
    """
    df = pd.read_csv(filepath)

    years = [year] if isinstance(year, int) else list(year)
    price_cols = [f'{y}_nominal_unit_price' for y in years]
    missing = [c for c in price_cols if c not in df.columns]
    if missing:
        available = [c for c in df.columns if 'nominal_unit_price' in c]
        raise KeyError(
            f"Column(s) {missing} not found. "
            f"Available year columns: {available}"
        )

    # Average across years if multiple
    df['_avg_price'] = df[price_cols].mean(axis=1)

    # Extract natural gas prices (state-level only)
    df_ng = df[
        (df['fuel_type'] == 'naturalGas') &
        (df['state_region'].str.len() == 2) &
        (df['state_region'] != 'National')
    ][['state_region', '_avg_price']].copy()
    df_ng.columns = ['state', 'ng_price_per_1000cf']

    # Extract electricity prices (state-level only)
    df_elec = df[
        (df['fuel_type'] == 'electricity') &
        (df['state_region'].str.len() == 2) &
        (df['state_region'] != 'National')
    ][['state_region', '_avg_price']].copy()
    df_elec.columns = ['state', 'elec_price_cents_kwh']

    df_merged = df_elec.merge(df_ng, on='state', how='inner')

    # Unit conversions
    df_merged['elec_price_kwh'] = df_merged['elec_price_cents_kwh'] / 100
    df_merged['gas_price_kwh'] = df_merged['ng_price_per_1000cf'] * NG_CONVERSION_FACTOR
    df_merged['elec_price_mmbtu'] = df_merged['elec_price_kwh'] * KWH_PER_MMBTU
    df_merged['gas_price_mmbtu'] = df_merged['gas_price_kwh'] * KWH_PER_MMBTU
    df_merged['spark_gap'] = df_merged['elec_price_mmbtu'] / df_merged['gas_price_mmbtu']
    df_merged['state_name'] = df_merged['state'].map(STATE_NAMES)

    result = df_merged[[
        'state', 'state_name',
        'elec_price_kwh', 'gas_price_kwh',
        'elec_price_mmbtu', 'gas_price_mmbtu',
        'spark_gap',
    ]].copy()

    for col in ['elec_price_kwh', 'gas_price_kwh']:
        result[col] = result[col].round(4)
    for col in ['elec_price_mmbtu', 'gas_price_mmbtu', 'spark_gap']:
        result[col] = result[col].round(2)

    return result.sort_values('spark_gap', ascending=False).reset_index(drop=True)


# ============================================================================
# STEP 3: THERMAL COP AND BASELINE AFUE
# ============================================================================

def compute_thermal_cop(
    df_baseline: pd.DataFrame,
    df_upgrade: pd.DataFrame,
    group_cols: list[str] = ['state'],
    fuel_filter: str = 'Natural Gas',
    require_baseline_heating: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute grouped thermal COP and baseline furnace AFUE from EUSS data.

    COP = Σ(Q_delivered) / Σ(E_hp + E_backup + E_fans_pumps) per group.
    AFUE = Σ(Q_delivered) / Σ(F_gas) per group.

    The denominator includes fan/pump electricity because the numerator
    (heating load) includes fan motor heat delivered to the space.

    Args:
        df_baseline: EUSS baseline DataFrame (indexed by bldg_id).
        df_upgrade: EUSS upgrade DataFrame (indexed by bldg_id,
            already filtered to applicability == True).
        group_cols: Columns to aggregate by. Accepts any combination of
            'state', 'cz_group', 'county'. Defaults to ['state'].
        fuel_filter: Value of 'in.heating_fuel' to filter to.
            Set to None to include all fuel types.
        require_baseline_heating: If True (default), exclude homes with
            zero baseline heating load before aggregation. Set to False
            to reproduce the old (unfixed) behavior.
        verbose: Print diagnostic info.

    Returns:
        DataFrame with ``group_cols`` as leading columns followed by:
        thermal_cop, baseline_afue, Q_upgrade_total_kbtu,
        hp_total_elec_kbtu, Q_baseline_total_kbtu,
        gas_consumed_total_kbtu, fans_pumps_pct, home_count.
    """
    elec_col = 'out.electricity.heating.energy_consumption.kwh'
    bkup_col = HP_BACKUP_ELEC_COL
    fans_col = HP_FANS_PUMPS_COL
    load_col = HEATING_LOAD_COL
    gas_col = 'out.natural_gas.heating.energy_consumption.kwh'

    # Validate required columns
    for col, name in [(load_col, 'baseline'), (gas_col, 'baseline')]:
        if col not in df_baseline.columns:
            raise KeyError(f"Missing column '{col}' in {name} DataFrame")
    for col, name in [(load_col, 'upgrade'), (elec_col, 'upgrade'),
                       (bkup_col, 'upgrade'), (fans_col, 'upgrade')]:
        if col not in df_upgrade.columns:
            raise KeyError(f"Missing column '{col}' in {name} DataFrame")

    # Build merged frame with all potential grouping columns
    merge_dict: dict[str, pd.Series] = {
        'state': df_baseline['in.state'],
        'heating_fuel': df_baseline['in.heating_fuel'],
        'Q_baseline_kbtu': df_baseline[load_col].fillna(0),
        'gas_consumed_kwh': df_baseline[gas_col].fillna(0),
    }
    # ASSUMPTION: Column names match EUSS ResStock output schema.
    if COUNTY_COL in df_baseline.columns:
        merge_dict['county'] = df_baseline[COUNTY_COL]
    if CLIMATE_ZONE_COL in df_baseline.columns:
        merge_dict['cz_group'] = df_baseline[CLIMATE_ZONE_COL].map(iecc_to_cz_group)

    df_merged = pd.DataFrame(merge_dict).join(
        pd.DataFrame({
            'Q_upgrade_kbtu': df_upgrade[load_col].fillna(0),
            'hp_elec_kwh': df_upgrade[elec_col].fillna(0),
            'hp_bkup_elec_kwh': df_upgrade[bkup_col].fillna(0),
            'hp_fans_pumps_kwh': df_upgrade[fans_col].fillna(0),
        }),
        how='inner',
    )

    if verbose:
        print(f"Matched homes (baseline ∩ upgrade): {len(df_merged):,}")

    if fuel_filter is not None:
        n_before = len(df_merged)
        df_merged = df_merged[df_merged['heating_fuel'] == fuel_filter]
        if verbose:
            print(f"Filtered to '{fuel_filter}': {len(df_merged):,} / {n_before:,} homes "
                  f"({100 * len(df_merged) / n_before:.1f}%)")

    # Exclude homes with zero or missing baseline heating load.
    # These are homes without an active heating system (e.g., no prior furnace),
    # which inflate COP when included in the aggregation.
    if require_baseline_heating:
        n_before_heating = len(df_merged)
        df_merged = df_merged[df_merged['Q_baseline_kbtu'] > 0]
        n_excluded = n_before_heating - len(df_merged)
        if verbose and n_excluded > 0:
            print(f"Excluded {n_excluded:,} homes with zero baseline heating "
                  f"({len(df_merged):,} remaining)")

    # Total retrofit heating electricity
    df_merged['hp_total_elec_kwh'] = (
        df_merged['hp_elec_kwh']
        + df_merged['hp_bkup_elec_kwh']
        + df_merged['hp_fans_pumps_kwh']
    )

    # Convert to kBtu for consistent units with heating load
    df_merged['hp_total_elec_kbtu'] = df_merged['hp_total_elec_kwh'] * KBTU_PER_KWH
    df_merged['hp_fans_pumps_kbtu'] = df_merged['hp_fans_pumps_kwh'] * KBTU_PER_KWH
    df_merged['gas_consumed_kbtu'] = df_merged['gas_consumed_kwh'] * KBTU_PER_KWH

    grouped = df_merged.groupby(group_cols).agg(
        Q_upgrade_total_kbtu=('Q_upgrade_kbtu', 'sum'),
        hp_total_elec_kbtu=('hp_total_elec_kbtu', 'sum'),
        hp_fans_pumps_total_kbtu=('hp_fans_pumps_kbtu', 'sum'),
        Q_baseline_total_kbtu=('Q_baseline_kbtu', 'sum'),
        gas_consumed_total_kbtu=('gas_consumed_kbtu', 'sum'),
        home_count=('Q_baseline_kbtu', 'size'),
    ).reset_index()

    grouped['thermal_cop'] = np.where(
        grouped['hp_total_elec_kbtu'] > 0,
        grouped['Q_upgrade_total_kbtu'] / grouped['hp_total_elec_kbtu'],
        np.nan,
    )

    grouped['baseline_afue'] = np.where(
        grouped['gas_consumed_total_kbtu'] > 0,
        grouped['Q_baseline_total_kbtu'] / grouped['gas_consumed_total_kbtu'],
        np.nan,
    )

    grouped['fans_pumps_pct'] = np.where(
        grouped['hp_total_elec_kbtu'] > 0,
        grouped['hp_fans_pumps_total_kbtu'] / grouped['hp_total_elec_kbtu'] * 100,
        0,
    )

    if verbose:
        fuel_label = fuel_filter if fuel_filter else 'all fuels'
        group_label = ' × '.join(group_cols)
        print(f"\n--- Thermal COP Summary ({fuel_label}, by {group_label}) ---")
        print(f"Groups: {len(grouped)}")
        print(f"Mean:   {grouped['thermal_cop'].mean():.2f}")
        print(f"Median: {grouped['thermal_cop'].median():.2f}")
        print(f"Range:  {grouped['thermal_cop'].min():.2f} - {grouped['thermal_cop'].max():.2f}")

        suspect_cop = grouped[
            (grouped['thermal_cop'] < 1.5) | (grouped['thermal_cop'] > 5.0)
        ]
        if len(suspect_cop) > 0:
            print(f"⚠ {len(suspect_cop)} groups with suspicious COP (<1.5 or >5.0):")
            for _, row in suspect_cop.iterrows():
                label = ', '.join(str(row[c]) for c in group_cols)
                print(f"    {label}: {row['thermal_cop']:.2f}")
        else:
            print("✓ All groups within expected COP range (1.5–5.0)")

        print(f"\n--- Baseline AFUE Summary ({fuel_label}) ---")
        print(f"Mean:   {grouped['baseline_afue'].mean():.2f}")
        print(f"Median: {grouped['baseline_afue'].median():.2f}")
        print(f"Range:  {grouped['baseline_afue'].min():.2f} - {grouped['baseline_afue'].max():.2f}")

        suspect_afue = grouped[
            (grouped['baseline_afue'] < 0.50) | (grouped['baseline_afue'] > 1.0)
        ]
        if len(suspect_afue) > 0:
            print(f"⚠ {len(suspect_afue)} groups with suspicious AFUE (<0.50 or >1.0):")
            for _, row in suspect_afue.iterrows():
                label = ', '.join(str(row[c]) for c in group_cols)
                print(f"    {label}: {row['baseline_afue']:.2f}")

        print(f"\n--- Fan/Pump Energy as % of Total HP Electricity ---")
        print(f"Mean:   {grouped['fans_pumps_pct'].mean():.1f}%")
        print(f"Range:  {grouped['fans_pumps_pct'].min():.1f}% - {grouped['fans_pumps_pct'].max():.1f}%")

    return grouped.sort_values(group_cols[0]).reset_index(drop=True)


def compute_thermal_cop_by_state(
    df_baseline: pd.DataFrame,
    df_upgrade: pd.DataFrame,
    fuel_filter: str = 'Natural Gas',
    verbose: bool = False,
) -> pd.DataFrame:
    """Backward-compatible wrapper. Use compute_thermal_cop() for new code."""
    return compute_thermal_cop(
        df_baseline, df_upgrade,
        group_cols=['state'],
        fuel_filter=fuel_filter,
        verbose=verbose,
    )


# ============================================================================
# STEP 4: BREAK-EVEN COP
# ============================================================================

def compute_breakeven_cop(
    df_prices: pd.DataFrame,
    afue_scenarios: List[float] = [0.80, 0.90, 0.95, 1.00],
) -> pd.DataFrame:
    """Compute break-even COP for each state and AFUE scenario.

    The break-even COP is the heat pump efficiency at which operating
    costs equal those of a gas furnace at a given AFUE. Derived from:

        bill_impact_ratio = spark_gap × (AFUE / COP)

    Setting bill_impact_ratio = 1 and solving for COP:

        COP_breakeven = spark_gap × AFUE

    where spark_gap = P_elec / P_gas on a $/MMBTU (or $/kWh) basis.

    # ASSUMPTION: State-level EIA residential prices used as proxy.
    # County-level fuel prices are not available in this dataset.

    Args:
        df_prices: Output of calculate_price_ratios(). Must contain
            columns 'state', 'spark_gap'. Optionally 'state_name'.
        afue_scenarios: Furnace efficiencies to evaluate.

    Returns:
        DataFrame with columns: state, spark_gap, plus one
        'breakeven_cop_{int(afue*100)}' column per scenario.
    """
    result = df_prices[['state', 'spark_gap']].copy()
    if 'state_name' in df_prices.columns:
        result.insert(1, 'state_name', df_prices['state_name'])

    for afue in afue_scenarios:
        col = f'breakeven_cop_{int(afue * 100)}'
        result[col] = (result['spark_gap'] * afue).round(2)

    return result.reset_index(drop=True)


def compute_spark_gap_metrics(
    df_prices: pd.DataFrame,
    df_cop: pd.DataFrame,
    df_breakeven: Optional[pd.DataFrame] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Merge price ratios, thermal COP/AFUE, and break-even COP into one table.

    Produces the primary state-level comparison DataFrame used for maps
    and the final results display in the preTARE notebook.

    Args:
        df_prices: Output of calculate_price_ratios().
        df_cop: Output of compute_thermal_cop_by_state().
        df_breakeven: Output of compute_breakeven_cop(). If None,
            break-even columns are computed inline from df_prices.
        verbose: Print diagnostic info.

    Returns:
        DataFrame with columns: state, state_name, spark_gap,
        elec_price_kwh, gas_price_kwh, thermal_cop, baseline_afue,
        breakeven_cop_* columns, fans_pumps_pct, home_count.
    """
    # Start from price data
    df = df_prices[[
        'state', 'state_name', 'elec_price_kwh', 'gas_price_kwh', 'spark_gap',
    ]].copy()

    # Merge thermal COP and AFUE
    cop_cols = ['state', 'thermal_cop', 'baseline_afue', 'fans_pumps_pct', 'home_count']
    df = df.merge(df_cop[cop_cols], on='state', how='inner')

    # Merge break-even COP columns
    if df_breakeven is not None:
        be_cols = ['state'] + [c for c in df_breakeven.columns if c.startswith('breakeven_cop_')]
        df = df.merge(df_breakeven[be_cols], on='state', how='left')
    else:
        # Compute inline from spark_gap × AFUE for default scenarios
        for afue in [0.80, 0.90, 0.95, 1.00]:
            col = f'breakeven_cop_{int(afue * 100)}'
            df[col] = (df['spark_gap'] * afue).round(2)

    if verbose:
        n = len(df)
        print(f"\n--- Spark Gap Metrics Summary ({n} states) ---")
        if 'breakeven_cop_90' in df.columns:
            be90 = df['breakeven_cop_90']
            print(f"Break-even COP @90% AFUE — "
                  f"Mean: {be90.mean():.2f}, "
                  f"Range: {be90.min():.2f}–{be90.max():.2f}")
        cop_col = 'thermal_cop'
        if cop_col in df.columns:
            beats = (df[cop_col] < df.get('breakeven_cop_90', np.inf)).sum()
            print(f"States where effective COP < break-even @90%: {beats}/{n}")

    return df


# ============================================================================
# STEP 5: DEMAND CHANGE
# ============================================================================

def compute_scenario_demand(
    df_baseline: pd.DataFrame,
    df_upgrade: pd.DataFrame,
    fuel_filter: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute per-building heating demand change under 100% adoption scenario.

    Produces two change metrics per home:
    - elec_demand_change_kwh: grid impact (electricity-only change)
    - site_energy_change_kwh: efficiency (total site energy change)

    Args:
        df_baseline: EUSS baseline DataFrame (indexed by bldg_id).
        df_upgrade: EUSS upgrade DataFrame (indexed by bldg_id,
            filtered to applicability == True).
        fuel_filter: Filter to this heating fuel. None for all.
        verbose: Print diagnostic info.

    Returns:
        DataFrame (indexed by bldg_id) with baseline, retrofit, and
        change columns plus weighted versions.
    """
    elec_col = 'out.electricity.heating.energy_consumption.kwh'

    baseline_total = df_baseline[HEATING_FUEL_COLS].sum(axis=1)
    retrofit_total_elec = (
        df_upgrade[elec_col].fillna(0)
        + df_upgrade[HP_BACKUP_ELEC_COL].fillna(0)
        + df_upgrade[HP_FANS_PUMPS_COL].fillna(0)
    )

    df_demand = pd.DataFrame({
        'in.state': df_baseline['in.state'],
        'in.heating_fuel': df_baseline['in.heating_fuel'],
        'weight': df_baseline['weight'],
        'baseline_electric_kwh': df_baseline[elec_col],
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

    for col in ['baseline_electric_kwh', 'baseline_heating_total_kwh',
                'retrofit_electric_kwh', 'elec_demand_change_kwh',
                'site_energy_change_kwh']:
        df_demand[f'weighted_{col}'] = df_demand[col] * df_demand['weight']

    if verbose:
        fuel_label = fuel_filter if fuel_filter else 'all fuels'
        print(f"\n--- Demand Scenario Summary (100% adoption, {fuel_label}) ---")
        print(f"Total homes: {len(df_demand):,}")
        elec_gwh = df_demand['weighted_elec_demand_change_kwh'].sum() / 1e6
        site_gwh = df_demand['weighted_site_energy_change_kwh'].sum() / 1e6
        print(f"Weighted electricity demand change:  {elec_gwh:+,.1f} GWh (grid impact)")
        print(f"Weighted total site energy change:   {site_gwh:+,.1f} GWh (efficiency)")

    return df_demand


def aggregate_demand_by_state(
    df_demand: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """Aggregate per-building demand results to state-level totals in GWh.

    Args:
        df_demand: Per-building demand DataFrame from compute_scenario_demand().
        verbose: Print diagnostic info.

    Returns:
        DataFrame with columns: state, home_count, baseline_elec_gwh,
        baseline_total_gwh, retrofit_elec_gwh, elec_change_gwh,
        pct_elec_demand_change, site_energy_change_gwh, pct_site_energy_change.
    """
    grouped = df_demand.groupby('in.state').agg(
        home_count=('weight', 'size'),
        weighted_baseline_elec=('weighted_baseline_electric_kwh', 'sum'),
        weighted_baseline_total=('weighted_baseline_heating_total_kwh', 'sum'),
        weighted_retrofit_elec=('weighted_retrofit_electric_kwh', 'sum'),
        weighted_elec_change=('weighted_elec_demand_change_kwh', 'sum'),
        weighted_site_change=('weighted_site_energy_change_kwh', 'sum'),
    ).reset_index().rename(columns={'in.state': 'state'})

    grouped['baseline_elec_gwh'] = grouped['weighted_baseline_elec'] / 1e6
    grouped['baseline_total_gwh'] = grouped['weighted_baseline_total'] / 1e6
    grouped['retrofit_elec_gwh'] = grouped['weighted_retrofit_elec'] / 1e6
    grouped['elec_change_gwh'] = grouped['weighted_elec_change'] / 1e6
    grouped['site_energy_change_gwh'] = grouped['weighted_site_change'] / 1e6

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

    for col in ['baseline_elec_gwh', 'baseline_total_gwh', 'retrofit_elec_gwh',
                'elec_change_gwh', 'site_energy_change_gwh']:
        result[col] = result[col].round(2)
    result['pct_elec_demand_change'] = result['pct_elec_demand_change'].round(2)
    result['pct_site_energy_change'] = result['pct_site_energy_change'].round(2)

    if verbose:
        print(f"\n--- State-Level Demand Summary ---")
        print(f"States: {len(result)}")
        print(f"Total elec demand change:    {result['elec_change_gwh'].sum():+.1f} GWh (grid impact)")
        print(f"Total site energy change:    {result['site_energy_change_gwh'].sum():+.1f} GWh (efficiency)")

    return result.sort_values('elec_change_gwh', ascending=False).reset_index(drop=True)
