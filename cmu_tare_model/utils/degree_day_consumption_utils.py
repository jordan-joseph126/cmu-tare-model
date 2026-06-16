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
from cmu_tare_model.constants import EQUIPMENT_SPECS, FUEL_MAPPING

# Load degree-day factors from the new AEO2026 CSV artifact.
# Year column headers arrive as strings from pd.read_csv — cast to int so that
# division_data.get(year_label, 1.0) finds integer keys (Constraint 4).
_DD_PATH = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "projections",
    "aeo2026_degree_day_factors_2025_2050.csv"
)

try:
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
except Exception as e:
    print(f"Warning: Could not load degree-day factors from {_DD_PATH}: {e}")
    lookup_hdd_factor = {}
    lookup_cdd_factor = {}


def get_hdd_factor_for_year(
        df: pd.DataFrame,
        year_label: int) -> pd.Series:
    """
    Calculate HDD adjustment factors for a specific year using AEO projection data.
    
    Replicates exact logic from precompute_hdd_factors.py for a single year.
    
    Args:
        df: DataFrame containing census_division column.
        year_label: Year for calculation (e.g., 2024, 2025).
        
    Returns:
        Series of HDD adjustment factors.
        
    Raises:
        KeyError: If census_division column missing.
        ValueError: If year_label invalid.
    """
    # Fail-fast validation
    if 'census_division' not in df.columns:
        raise KeyError("Required column 'census_division' not found in DataFrame")
    
    if not isinstance(year_label, int) or year_label < 2024 or year_label > 2050:
        raise ValueError(f"Invalid year_label: {year_label}. Must be integer between 2024-2050")
    
    # Apply exact logic from precompute_hdd_factors.py
    def get_factor_for_division(division):
        """Get HDD factor for census division with exact fallback logic."""
        # Try specific division first
        division_data = lookup_hdd_factor.get(division)
        if division_data is None:
            # Fallback to National
            division_data = lookup_hdd_factor.get('National', {})
        # Get year factor, default to 1.0
        return division_data.get(year_label, 1.0)
    
    return df['census_division'].map(get_factor_for_division).fillna(1.0)


def get_cdd_factor_for_year(
        df: pd.DataFrame,
        year_label: int) -> pd.Series:
    """
    Calculate CDD (Cooling Degree Days) adjustment factors for a specific year using AEO projection data.
    
    Mirrors the logic from get_hdd_factor_for_year() but for cooling calculations.
    
    Args:
        df: DataFrame containing census_division column.
        year_label: Year for calculation (e.g., 2024, 2025).
        
    Returns:
        Series of CDD adjustment factors.
        
    Raises:
        KeyError: If census_division column missing.
        ValueError: If year_label invalid.
    """
    # Fail-fast validation
    if 'census_division' not in df.columns:
        raise KeyError("Required column 'census_division' not found in DataFrame")
    
    if not isinstance(year_label, int) or year_label < 2024 or year_label > 2050:
        raise ValueError(f"Invalid year_label: {year_label}. Must be integer between 2024-2050")
    
    # Apply exact logic from get_hdd_factor_for_year() but for CDD
    def get_factor_for_division(division):
        """Get CDD factor for census division with exact fallback logic."""
        # Try specific division first
        division_data = lookup_cdd_factor.get(division)
        if division_data is None:
            # Fallback to National
            division_data = lookup_cdd_factor.get('National', {})
        # Get year factor, default to 1.0
        return division_data.get(year_label, 1.0)
    
    return df['census_division'].map(get_factor_for_division).fillna(1.0)


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
