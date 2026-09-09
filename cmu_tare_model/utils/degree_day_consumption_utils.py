"""
HDD and CDD-adjusted consumption calculation utilities.

This module provides on-demand calculation of energy consumption with degree day adjustments:
- HDD (Heating Degree Days) for heating equipment
- CDD (Cooling Degree Days) for cooling equipment

This eliminates the need for project_future_energy_consumption.py and its 180+ columns.

Design Principles:
- Fail-fast: Immediate input validation with clear exceptions
- DRY: Minimal, focused functions that handle related tasks
- Memory efficient: Zero storage of intermediary results
- Uses actual AEO HDD and CDD projection data from Excel file
"""


import os
from typing import Dict, Tuple
import pandas as pd

from config import PROJECT_ROOT
from cmu_tare_model.constants import (
    ANCHOR_YEAR,
    EQUIPMENT_SPECS,
    FUEL_MAPPING,
    PROJECTION_END_YEAR,
)

# Load degree-day factors from the new AEO2026 CSV artifact.
# Year column headers arrive as strings from pd.read_csv -- cast to int so the
# per-year lookups below find integer keys.
_DD_PATH = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "projections",
    "aeo2026_degree_day_factors_2025_2050.csv"
)

_df_dd = pd.read_csv(_DD_PATH)
_df_dd.columns = [int(c) if isinstance(c, str) and c.isdigit() else c
                  for c in _df_dd.columns]

# Heating Degree Day (HDD) factors lookup
lookup_hdd_factor = (
    _df_dd[_df_dd["dd_type"] == "hdd"]
    .drop(columns="dd_type")
    .set_index("census_division")
    .to_dict("index")
)

# Cooling Degree Day (CDD) factors lookup
lookup_cdd_factor = (
    _df_dd[_df_dd["dd_type"] == "cdd"]
    .drop(columns="dd_type")
    .set_index("census_division")
    .to_dict("index")
)


def _validate_degree_day_factors() -> None:
    """Check the degree-day file covers the expected years and is anchored at 1.0.

    Mirrors the check on the fuel-price projection file. Two things must hold,
    and if either fails the model would still run while quietly mis-scaling
    heating and cooling energy, so both are checked when this module is
    imported:

    1. Each census division's year keys are exactly ANCHOR_YEAR through
       PROJECTION_END_YEAR, with no gaps and no year earlier than ANCHOR_YEAR.
    2. Every ANCHOR_YEAR factor is exactly 1.0, for both heating and cooling
       and every census division. That is what makes ANCHOR_YEAR the anchor:
       energy use in that year is the ResStock value, unscaled.

    Raises:
        ValueError: If either lookup is empty, the year coverage is wrong, or
            any ANCHOR_YEAR factor is not exactly 1.0.
    """
    expected_years = list(range(ANCHOR_YEAR, PROJECTION_END_YEAR + 1))

    for dd_type, lookup in (("hdd", lookup_hdd_factor),
                            ("cdd", lookup_cdd_factor)):
        if not lookup:
            raise ValueError(
                f"No '{dd_type}' rows found in the degree-day file. "
                f"File: {_DD_PATH}")

        for division, year_factors in lookup.items():
            years = sorted(y for y in year_factors if isinstance(y, int))
            if years != expected_years:
                missing = sorted(set(expected_years) - set(years))
                unexpected = sorted(set(years) - set(expected_years))
                raise ValueError(
                    f"Degree-day years for '{division}' ({dd_type}) must be "
                    f"exactly {ANCHOR_YEAR}-{PROJECTION_END_YEAR} with no "
                    f"gaps. Missing: {missing}. Unexpected: {unexpected}. "
                    f"File: {_DD_PATH}")

            if year_factors[ANCHOR_YEAR] != 1.0:
                raise ValueError(
                    f"Every {ANCHOR_YEAR} degree-day factor must be exactly "
                    f"1.0 ({ANCHOR_YEAR} is the anchor year, so its energy "
                    f"use is unscaled). Got "
                    f"{year_factors[ANCHOR_YEAR]} for '{division}' "
                    f"({dd_type}). File: {_DD_PATH}")


_validate_degree_day_factors()


def get_hdd_factor_for_year(
        df: pd.DataFrame,
        year_label: int) -> pd.Series:
    """
    Calculate HDD adjustment factors for a specific year using AEO projection data.
    
    Replicates exact logic from precompute_hdd_factors.py for a single year.
    
    Args:
        df: DataFrame containing census_division column.
        year_label: Year for calculation (e.g., 2025, 2026).

    Returns:
        Series of HDD adjustment factors.

    Raises:
        KeyError: If census_division column missing, or a home's census
            division has no heating factor for year_label.
        ValueError: If year_label is not an integer in the projection range.
    """
    # Fail-fast validation
    if 'census_division' not in df.columns:
        raise KeyError("Required column 'census_division' not found in DataFrame")

    if (not isinstance(year_label, int)
            or year_label < ANCHOR_YEAR or year_label > PROJECTION_END_YEAR):
        raise ValueError(
            f"Invalid year_label: {year_label}. Must be an integer between "
            f"{ANCHOR_YEAR} and {PROJECTION_END_YEAR}")

    def get_factor_for_division(division):
        """Get the heating factor for one census division, or fail loudly.

        Falls back to the National row for a division the file does not list,
        which is a deliberate allowance for an unfamiliar region. A missing
        YEAR is not allowed to fall back: a factor of 1.0 would read as a real
        answer meaning "no degree-day adjustment" and would quietly leave that
        year's heating energy unscaled.
        """
        division_data = lookup_hdd_factor.get(division)
        if division_data is None:
            division_data = lookup_hdd_factor.get('National')
        if division_data is None:
            raise KeyError(
                f"No heating degree-day factors for census division "
                f"'{division}', and no 'National' row to fall back on. "
                f"File: {_DD_PATH}")
        if year_label not in division_data:
            raise KeyError(
                f"No heating degree-day factor for year {year_label} in "
                f"census division '{division}'. The projection file covers "
                f"{ANCHOR_YEAR}-{PROJECTION_END_YEAR}. File: {_DD_PATH}")
        return division_data[year_label]

    return df['census_division'].map(get_factor_for_division)


def get_cdd_factor_for_year(
        df: pd.DataFrame,
        year_label: int) -> pd.Series:
    """
    Calculate CDD (Cooling Degree Days) adjustment factors for a specific year using AEO projection data.
    
    Mirrors the logic from get_hdd_factor_for_year() but for cooling calculations.
    
    Args:
        df: DataFrame containing census_division column.
        year_label: Year for calculation (e.g., 2025, 2026).

    Returns:
        Series of CDD adjustment factors.

    Raises:
        KeyError: If census_division column missing, or a home's census
            division has no cooling factor for year_label.
        ValueError: If year_label is not an integer in the projection range.
    """
    # Fail-fast validation
    if 'census_division' not in df.columns:
        raise KeyError("Required column 'census_division' not found in DataFrame")

    if (not isinstance(year_label, int)
            or year_label < ANCHOR_YEAR or year_label > PROJECTION_END_YEAR):
        raise ValueError(
            f"Invalid year_label: {year_label}. Must be an integer between "
            f"{ANCHOR_YEAR} and {PROJECTION_END_YEAR}")

    def get_factor_for_division(division):
        """Get the cooling factor for one census division, or fail loudly.

        Same rule as the heating version: an unfamiliar census division may
        fall back to the National row, but a missing YEAR may not, because a
        factor of 1.0 would read as a real answer meaning "no degree-day
        adjustment" and would quietly leave that year's cooling energy
        unscaled.
        """
        division_data = lookup_cdd_factor.get(division)
        if division_data is None:
            division_data = lookup_cdd_factor.get('National')
        if division_data is None:
            raise KeyError(
                f"No cooling degree-day factors for census division "
                f"'{division}', and no 'National' row to fall back on. "
                f"File: {_DD_PATH}")
        if year_label not in division_data:
            raise KeyError(
                f"No cooling degree-day factor for year {year_label} in "
                f"census division '{division}'. The projection file covers "
                f"{ANCHOR_YEAR}-{PROJECTION_END_YEAR}. File: {_DD_PATH}")
        return division_data[year_label]

    return df['census_division'].map(get_factor_for_division)


# def apply_hdd_adjustment(
#         consumption: pd.Series,
#         category: str,
#         hdd_factor: pd.Series) -> pd.Series:
#     """
#     Apply HDD adjustment to consumption based on category-specific rules.
    
#     Critical implementation detail: Only 'heating' category gets HDD adjustment.
    
#     Args:
#         consumption: Base consumption values.
#         category: Equipment category to determine if HDD applies.
#         hdd_factor: HDD adjustment factors.
        
#     Returns:
#         Series with HDD adjustment applied if applicable.
#     """
#     if category == 'heating':
#         return consumption * hdd_factor
#     else:
#         # For all other categories, return consumption unchanged
#         return consumption

# Updated function to handle both HDD and CDD adjustments
def apply_degree_day_adjustment(
        consumption: pd.Series,
        category: str,
        hdd_factor: pd.Series = None,
        cdd_factor: pd.Series = None) -> pd.Series:
    """
    Apply degree day adjustment to consumption based on category-specific rules.
    
    Critical implementation details:
    - 'heating' category gets HDD adjustment
    - 'cooling' category gets CDD adjustment
    - Other categories return consumption unchanged
    
    Args:
        consumption: Base consumption values.
        category: Equipment category to determine which adjustment applies.
        hdd_factor: HDD adjustment factors (for heating).
        cdd_factor: CDD adjustment factors (for cooling).
        
    Returns:
        Series with degree day adjustment applied if applicable.
    """
    if category == 'heating' and hdd_factor is not None:
        return consumption * hdd_factor
    elif category == 'cooling' and cdd_factor is not None:
        return consumption * cdd_factor
    else:
        # For all other categories, return consumption unchanged
        return consumption


def get_electricity_consumption_for_year(
        df: pd.DataFrame, 
        category: str, 
        year_label: int, 
        menu_mp: int) -> pd.Series:
    """
    Get electricity consumption with HDD adjustment for emissions calculations.
    
    Primary function replacing year-labeled consumption columns for climate/health modules.
    
    Args:
        df: DataFrame containing consumption data.
        category: Equipment category ('heating', 'waterHeating', etc.).
        year_label: Year for calculation.
        menu_mp: Measure package (0 for baseline, >0 for retrofits).
        
    Returns:
        Series of electricity consumption with HDD adjustment applied.
        
    Raises:
        ValueError: If parameters invalid or columns missing.
    """
    # Fail-fast validation
    if category not in EQUIPMENT_SPECS:
        raise ValueError(f"Invalid category: {category}. Must be one of {list(EQUIPMENT_SPECS.keys())}")
    
    # Determine consumption column
    if menu_mp == 0:
        consumption_col = f'base_electricity_{category}_consumption'
    else:
        consumption_col = f'mp{menu_mp}_{category}_consumption'
    
    if consumption_col not in df.columns:
        raise ValueError(f"Required column '{consumption_col}' not found in DataFrame")
    
    consumption = df[consumption_col]
    
    # Updated logic to handle both HDD and CDD adjustments
    # Apply degree day adjustments based on category
    if category == 'heating':
        hdd_factor = get_hdd_factor_for_year(df, year_label)
        consumption = apply_degree_day_adjustment(consumption, category, hdd_factor=hdd_factor)
    elif category == 'cooling':
        cdd_factor = get_cdd_factor_for_year(df, year_label)
        consumption = apply_degree_day_adjustment(consumption, category, cdd_factor=cdd_factor)

    return consumption


def get_hdd_adjusted_consumption(
        df: pd.DataFrame, 
        category: str, 
        year_label: int, 
        menu_mp: int) -> pd.Series:
    """
    Calculate total consumption for any category/year, replacing ALL pre-computed columns.
    
    Master function that eliminates project_future_energy_consumption.py entirely.
    
    Args:
        df: DataFrame containing base consumption data.
        category: Equipment category.
        year_label: Year for calculation.
        menu_mp: Measure package (0 for baseline, >0 for retrofits).
        
    Returns:
        Series of HDD-adjusted consumption values.
        
    Raises:
        ValueError: If parameters invalid or required data missing.
    
    Note:
        - For menu_mp = 0: Returns total baseline consumption across all fuels
        - For menu_mp > 0: Returns retrofit consumption (electricity only)
        - HDD adjustment applied according to category rules
    """
    # Fail-fast validation
    if category not in EQUIPMENT_SPECS:
        raise ValueError(f"Invalid category: {category}. Must be one of {list(EQUIPMENT_SPECS.keys())}")
    
    if menu_mp == 0:
        # Baseline: sum all applicable fuel types
        return get_total_baseline_consumption(df, category, year_label)
    else:
        # Retrofit: use electricity consumption from measure package
        return get_electricity_consumption_for_year(df, category, year_label, menu_mp)


def get_total_baseline_consumption(
        df: pd.DataFrame,
        category: str,
        year_label: int) -> pd.Series:
    """
    Sum baseline consumption across all fuel types for a category.
    
    Public function for accessing baseline consumption calculation logic.
    
    Args:
        df: DataFrame containing base consumption data.
        category: Equipment category.
        year_label: Year for calculation.
        
    Returns:
        Total baseline consumption across all applicable fuel types.
        
    Raises:
        ValueError: If category is invalid.
    """
    # Define fuel types by category - FIXED: Include electricity for cooking
    if category in ['heating', 'waterHeating']:
        fuel_types = ['electricity', 'naturalGas', 'propane', 'fuelOil']
    # Updated to handle cooling category
    elif category == 'cooling':
        fuel_types = ['electricity']
    elif category == 'clothesDrying':
        fuel_types = ['electricity', 'naturalGas', 'propane']
    elif category == 'cooking':
        # FIXED: Include electricity for cooking baseline
        # The data validation framework handles filtering out invalid fuel/tech combinations
        fuel_types = ['electricity', 'naturalGas', 'propane'] 
    else:
        raise ValueError(f"Unknown fuel pattern for category: {category}")
    
    # UPDATED: Handle both HDD and CDD adjustments (newly added cooling category)
    # Get degree day factors based on category
    hdd_factor = None
    cdd_factor = None
    if category == 'heating':
        hdd_factor = get_hdd_factor_for_year(df, year_label)
    elif category == 'cooling':
        cdd_factor = get_cdd_factor_for_year(df, year_label)
    
    # Sum consumption across fuel types
    total_consumption = pd.Series(0.0, index=df.index)
    
    for fuel_type in fuel_types:
        consumption_col = f'base_{fuel_type}_{category}_consumption'
        if consumption_col in df.columns:
            fuel_consumption = df[consumption_col].fillna(0)
            
            # Apply degree day adjustments
            fuel_consumption = apply_degree_day_adjustment(
                fuel_consumption, 
                category, 
                hdd_factor=hdd_factor, 
                cdd_factor=cdd_factor
            )
                
            total_consumption += fuel_consumption
    
    return total_consumption
