"""
HDD-adjusted consumption calculation utilities.

This module provides on-demand calculation of energy consumption with HDD adjustments,
eliminating the need for project_future_energy_consumption.py and its 180+ columns.

Design Principles:
- Fail-fast: Immediate input validation with clear exceptions
- DRY: Minimal, focused functions that handle related tasks
- Memory efficient: Zero storage of intermediary results
- Uses actual AEO HDD projection data from Excel file
"""

import os
from typing import Dict, Tuple
import pandas as pd

from config import PROJECT_ROOT
from cmu_tare_model.constants import EQUIPMENT_SPECS, FUEL_MAPPING

# Load HDD factors data (same source as precompute_hdd_factors.py)
filename = 'aeo_projections_2022_2050.xlsx'
relative_path = os.path.join("cmu_tare_model", "data", "projections", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)

try:
    df_hdd_projection_factors = pd.read_excel(io=file_path, sheet_name='hdd_factors_2022_2050')
    lookup_hdd_factor = df_hdd_projection_factors.set_index(['census_division']).to_dict('index')
except Exception as e:
    print(f"Warning: Could not load HDD factors from {file_path}: {e}")
    lookup_hdd_factor = {}


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
    
    if not isinstance(year_label, int) or year_label < 2020 or year_label > 2060:
        raise ValueError(f"Invalid year_label: {year_label}. Must be integer between 2020-2060")
    
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


def apply_hdd_adjustment(
        consumption: pd.Series,
        category: str,
        hdd_factor: pd.Series) -> pd.Series:
    """
    Apply HDD adjustment to consumption based on category-specific rules.
    
    Critical implementation detail: Only 'heating' category gets HDD adjustment.
    
    Args:
        consumption: Base consumption values.
        category: Equipment category to determine if HDD applies.
        hdd_factor: HDD adjustment factors.
        
    Returns:
        Series with HDD adjustment applied if applicable.
    """
    if category == 'heating':
        return consumption * hdd_factor
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
    
    # Apply HDD adjustment ONLY to heating category
    if category == 'heating':
        hdd_factor = get_hdd_factor_for_year(df, year_label)
        consumption = apply_hdd_adjustment(consumption, category, hdd_factor)

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
    elif category == 'clothesDrying':
        fuel_types = ['electricity', 'naturalGas', 'propane']
    elif category == 'cooking':
        # FIXED: Include electricity for cooking baseline
        # The data validation framework handles filtering out invalid fuel/tech combinations
        fuel_types = ['electricity', 'naturalGas', 'propane'] 
    else:
        raise ValueError(f"Unknown fuel pattern for category: {category}")
    
    # Get HDD factor (only applied to HEATING category)
    hdd_factor = get_hdd_factor_for_year(df, year_label) if category == 'heating' else None
    
    # Sum consumption across fuel types
    total_consumption = pd.Series(0.0, index=df.index)
    
    for fuel_type in fuel_types:
        consumption_col = f'base_{fuel_type}_{category}_consumption'
        if consumption_col in df.columns:
            fuel_consumption = df[consumption_col].fillna(0)
            
            # Apply HDD adjustment only to HEATING category
            if category == 'heating' and hdd_factor is not None:
                fuel_consumption = apply_hdd_adjustment(fuel_consumption, category, hdd_factor)
                
            total_consumption += fuel_consumption
    
    return total_consumption
