"""
Shared data loading functions and constants for the adoption KPI modules.

Provides EUSS baseline/upgrade loading, unit conversion constants,
column name constants, file path constants, and state name lookups
used across spark_gap.py and thermal_cop.py.

Location: cmu_tare_model/adoption_kpis/data_loading.py
"""

import os
from typing import Optional

import pandas as pd

from config import PROJECT_ROOT
from cmu_tare_model.constants import ALLOWED_HOUSING_TYPES


# ============================================================================
# UNIT CONVERSION CONSTANTS (Source: EIA)
# ============================================================================

BTU_PER_CF_NATURAL_GAS: int = 1036
"""
BTU per cubic foot of natural gas (EIA US 2025 average).
https://www.eia.gov/dnav/ng/ng_cons_heat_a_epg0_vgth_btucf_a.htm
"""

BTU_PER_KWH: int = 3412
"""BTU per kilowatt-hour (by definition)."""

KWH_PER_MMBTU: float = 293.07107
"""1 MMBTU = 293.07107 kWh."""

KBTU_PER_KWH: float = 3.412
"""1 kWh = 3.412 kBtu."""

# Derived: natural gas $/1000cf → $/kWh
NG_CONVERSION_FACTOR: float = BTU_PER_KWH / (1000 * BTU_PER_CF_NATURAL_GAS)
"""Multiply ng_price_per_1000cf by this to get $/kWh."""


# ============================================================================
# FILE PATHS
# ============================================================================

EUSS_DATA_DIR: str = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "euss_data",
    "resstock_amy2018_release_1.1", "national", "csv"
)

FUEL_PRICES_PATH: str = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "fuel_prices",
    "fuel_prices_nominal_2015_2024.csv"
)

SHAPEFILE_PATH: str = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "electricity_ng_price_ratio",
    "nhgis0011_shapefile_tl2015_us_state_2015", "US_state_2015.shp"
)

COUNTY_SHAPEFILE_PATH: str = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "electricity_ng_price_ratio",
    "tl_2025_us_county", "tl_2025_us_county.shp"
)
# Source: US Census Bureau TIGER/Line 2025 county boundaries.
# Download: https://www.census.gov/cgi-bin/geo/shapefiles/index.php
# Required columns: GEOID (5-digit FIPS), STATEFP (2-digit state FIPS).


# ============================================================================
# COLUMN NAME CONSTANTS
# ============================================================================

DWELLING_UNIT_WEIGHT: str = "weight"
"""EUSS survey weight column (applies to all dwelling-unit counts)."""

GAS_FUEL_COL: str = "out.natural_gas.heating.energy_consumption.kwh"
"""EUSS column for natural gas heating energy consumption."""

HEATING_LOAD_COL: str = "out.load.heating.energy_delivered.kbtu"
"""EUSS column for heating load delivered to the space (kBtu)."""

HP_BACKUP_ELEC_COL: str = "out.electricity.heating_hp_bkup.energy_consumption.kwh"
"""EUSS column for heat-pump backup (resistance) electricity."""

HP_FANS_PUMPS_COL: str = "out.electricity.heating_fans_pumps.energy_consumption.kwh"
"""EUSS column for fan and pump electricity. Always included in COP denominator."""

ELEC_TOTAL_COL: str = "out.electricity.total.energy_consumption.kwh"
"""EUSS column for total residential electricity (kWh). Includes all end uses.
Use this for demand change calculations — do NOT use the heating-only column."""

CLIMATE_ZONE_COL: str = "in.ashrae_iecc_climate_zone_2004"
"""EUSS column for ASHRAE/IECC 2004 climate zone."""

COUNTY_COL: str = "in.county"
"""EUSS column for county GISJOIN code (e.g., 'G4200030')."""

HEATING_FUEL_COLS: list[str] = [
    "out.electricity.heating.energy_consumption.kwh",
    "out.natural_gas.heating.energy_consumption.kwh",
    "out.fuel_oil.heating.energy_consumption.kwh",
    "out.propane.heating.energy_consumption.kwh",
]
"""All EUSS heating energy consumption columns (kWh)."""

FUEL_PRICE_MAP: dict[str, str] = {
    "electricity": "elec_price_kwh",
    "naturalGas": "gas_price_kwh",
}
"""Mapping from EIA fuel-type string to price column name."""

# Column subsets for CSV loading
BASELINE_USECOLS: list[str] = [
    "bldg_id", "in.state", "in.vacancy_status",
    "in.geometry_building_type_recs",
    "in.heating_fuel", "in.hvac_heating_type_and_fuel",
    "in.hvac_heating_efficiency",
    CLIMATE_ZONE_COL, COUNTY_COL,
    DWELLING_UNIT_WEIGHT,
] + HEATING_FUEL_COLS + [HEATING_LOAD_COL, HP_BACKUP_ELEC_COL, HP_FANS_PUMPS_COL, ELEC_TOTAL_COL]

UPGRADE_USECOLS: list[str] = BASELINE_USECOLS + ["applicability"]


# ============================================================================
# STATE NAME LOOKUP
# ============================================================================

STATE_NAMES: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}
"""Mapping from 2-letter state abbreviation to full state name."""


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def mp_to_upgrade(mp_num: int) -> str:
    """Convert a measure package number to its EUSS upgrade identifier string.

    Args:
        mp_num: Measure package number (e.g., 3 or 4).

    Returns:
        EUSS upgrade identifier string (e.g., ``'upgrade03'``, ``'upgrade04'``).
    """
    return f"upgrade{mp_num:02d}"


def load_euss_baseline(
    filename: str = "baseline_metadata_and_annual_results.csv",
) -> pd.DataFrame:
    """Load the EUSS baseline CSV and filter to occupied single-family homes.

    Applies two filters in sequence:
    1. Vacancy status == 'Occupied'
    2. Building type in ``ALLOWED_HOUSING_TYPES`` (from cmu_tare_model.constants)

    Args:
        filename: Baseline CSV filename within ``EUSS_DATA_DIR``.
            Defaults to ``'baseline_metadata_and_annual_results.csv'``.

    Returns:
        DataFrame indexed by ``bldg_id``, restricted to occupied SF homes.
        All EUSS columns are preserved (no column subsetting applied).

    Raises:
        FileNotFoundError: If the CSV does not exist at the resolved path.
    """
    filepath = os.path.join(EUSS_DATA_DIR, filename)
    print(f"Loading baseline from: {filepath}")
    df = pd.read_csv(filepath, index_col="bldg_id")

    n_total = len(df)
    df = df[df["in.vacancy_status"] == "Occupied"]
    print(f"  After occupancy filter: {len(df):,} / {n_total:,}")

    df = df[df["in.geometry_building_type_recs"].isin(ALLOWED_HOUSING_TYPES)]
    print(f"  After housing type filter ({ALLOWED_HOUSING_TYPES}): {len(df):,}")

    return df


def load_euss_upgrade(upgrade_name: str) -> pd.DataFrame:
    """Load an EUSS upgrade CSV and filter to applicable occupied SF homes.

    Applies three filters in sequence:
    1. Vacancy status == 'Occupied'
    2. Building type in ``ALLOWED_HOUSING_TYPES``
    3. ``applicability == True``

    Args:
        upgrade_name: EUSS upgrade identifier (e.g., ``'upgrade04'``).
            Use :func:`mp_to_upgrade` to convert a measure package number.

    Returns:
        DataFrame indexed by ``bldg_id``, restricted to applicable occupied
        SF homes. All EUSS columns are preserved.

    Raises:
        FileNotFoundError: If the CSV does not exist at the resolved path.
    """
    filename = f"{upgrade_name}_metadata_and_annual_results.csv"
    filepath = os.path.join(EUSS_DATA_DIR, filename)
    print(f"Loading {upgrade_name} from: {filepath}")
    df = pd.read_csv(filepath, index_col="bldg_id")

    n_total = len(df)
    df = df[df["in.vacancy_status"] == "Occupied"]
    print(f"  After occupancy filter: {len(df):,} / {n_total:,}")

    df = df[df["in.geometry_building_type_recs"].isin(ALLOWED_HOUSING_TYPES)]
    print(f"  After housing type filter: {len(df):,}")

    df = df[df["applicability"] == True]  # noqa: E712
    print(f"  After applicability filter: {len(df):,}")

    return df
