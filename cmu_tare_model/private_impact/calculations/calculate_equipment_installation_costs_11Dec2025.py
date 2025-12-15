import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Literal

from cmu_tare_model.utils.validation_framework import (
    apply_new_columns_to_dataframe,
    apply_final_masking,
    initialize_validation_tracking,
    create_retrofit_only_series
    )
from cmu_tare_model.utils.remdb_v4_installed_cost_utils import (
    map_remdb_cost_parameters,
    remdb_cost_regression_formula
    )


"""
========================================================================================================================================================================
OVERVIEW: CALCULATE UPGRADE INSTALLED COSTS FOR VARIOUS END USES (REMDB V4 METHODOLOGY)
========================================================================================================================================================================
This module calculates upgrade installed costs for equipment retrofits using REMDB v4 regression methodology.
It replaces the probabilistic sampling approach (REMDB v3) with deterministic regression equations.

Key changes from REMDB v3 to v4:
- Regression-based calculation: Material_Price = (pm1_coef × Metric1) + (pm2_coef × Metric2) + Intercept
- Costs already in 2023$ (no CPI adjustment needed)
- Upgrade installed costs via multipliers OR adders (component-specific)
- Added cooling as new end-use category
- Dynamic row_id mapping replaces hardcoded technology-efficiency pairs

Key Features:
- **Regression-based calculation**: Material_Price = (pm1_coef × Metric1) + (pm2_coef × Metric2) + Intercept
- **Installed cost formula**: Installed_Cost = (Material_Price × multiplier) + adder
- **All costs in 2023$**: No CPI adjustment needed
- **Data validation framework**: Ensures only valid homes receive cost calculations
- **Dynamic row_id mapping**: Replaces hardcoded (technology, efficiency) tuples

End-Use Categories:
- heating: Heat pumps (ducted and ductless variants)
- cooling: Air conditioning upgrades (new in v4)
- waterHeating: Heat pump water heaters
- clothesDrying: Heat pump and standard electric dryers
- cooking: Induction and standard electric ranges

# UPDATED MARCH 24, 2025 @ 4:30 PM - REMOVED RSMEANS CCI ADJUSTMENTS
# UPDATED APRIL 9, 2025 @ 7:30 PM - IMPROVED DOCUMENTATION
# UPDATED APRIL 21, 2025 @ 11:45 PM - COST UTILITY FUNCTION REPLACED REDUNDANT CODE (SEE UTILS FOLDER)
# UPDATED DECEMBER 2, 2025 @ 5:00 PM - UPDATED TO REMDB V4 METHODOLOGY
# UPDATED DECEMBER 10, 2025 @ 2:00 PM - INTEGRATED VALIDATION FRAMEWORK AND UPDATED COLUMN NAMING
"""

# ========================================================================================================================================================================
# FUNCTIONS: HELPER FUNCTIONS FOR PERFORMANCE METRIC EXTRACTION
# ========================================================================================================================================================================

# ========== Step 1/4 Extract performance metrics for REMDB v4 cost estimation. Then add cols to main df ==========
def add_remdb_upgrade_metrics(
    df: pd.DataFrame,
    end_use: Literal['heating', 'cooling', 'waterHeating', 'clothesDrying', 'cooking']
) -> pd.DataFrame:
    """
    Performance metrics for upgrade installed cost for REMDB v4 cost calculations.
    
    Extracts or calculates performance metrics from upgrade/retrofit specifications.
    These metrics feed into the REMDB v4 regression formula:
    Material_Price = (pm1_coef × Metric1) + (pm2_coef × Metric2) + Intercept
    
    Args:
        df: DataFrame with equipment upgrade specifications
        end_use: Equipment category
        
    Returns:
        DataFrame with {end_use}_{replace_or_upgrade}_metric1 and {end_use}_{replace_or_upgrade}_metric2 columns
        
    Note:
        Uses upgrade columns (not baseline) since these are for installation cost calculations.
        For replacement costs, a separate function would extract from baseline columns.
    """
    # This function is for retrofit upgrade installed costs
    replace_or_upgrade = 'upgrade'

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df)}")
    
    valid = ['heating', 'cooling', 'waterHeating', 'clothesDrying', 'cooking']
    if end_use not in valid:
        raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of {valid}")
    
    df = df.copy()
    
    # ========== SPACE HEATING PERFORMANCE METRICS  ==========
    if end_use == 'heating':
        # Metric1: Heating capacity in tons (standard HVAC unit)
        # Calculate total load by summing primary, backup, and secondary systems
        if 'total_heating_load_kBtuh' not in df.columns:
            load_cols = ['size_heating_system_primary_k_btu_h', 
                        'size_heat_pump_backup_primary_k_btu_h',
                        'size_heating_system_secondary_k_btu_h']
            available = [c for c in load_cols if c in df.columns]
            if not available:
                raise KeyError(f"Missing heating load columns. Need one of: {load_cols}")
            # Sum available columns, treating NaN as zero (homes may not have all systems)
            df['total_heating_load_kBtuh'] = sum(df[c].fillna(0) for c in available)
        
        # Convert kBtuh to tons (1 ton = 12,000 Btuh - industry standard)
        df[f'{end_use}_{replace_or_upgrade}_metric1'] = df['total_heating_load_kBtuh'] / 12.0
        
        # Metric2: SEER efficiency rating from upgrade specification
        if 'upgrade_hvac_heating_efficiency' not in df.columns:
            raise KeyError("Missing 'upgrade_hvac_heating_efficiency'")
        # Extract numeric SEER value from strings like "SEER 18, 9.3 HSPF"
        df[f'{end_use}_{replace_or_upgrade}_metric2'] = df['upgrade_hvac_heating_efficiency'].str.extract(
            r'SEER (\d+\.?\d*)', expand=False).astype(float)
    
    # ========== COOLING PERFORMANCE METRICS  ==========
    elif end_use == 'cooling':
        # Metric1: Cooling capacity in tons
        if 'total_cooling_load_kBtuh' not in df.columns:
            if 'size_cooling_system_primary_k_btu_h' in df.columns:
                df['total_cooling_load_kBtuh'] = df['size_cooling_system_primary_k_btu_h'].fillna(0)
            elif 'total_heating_load_kBtuh' in df.columns:
                # For heat pumps, cooling load often equals heating load
                df['total_cooling_load_kBtuh'] = df['total_heating_load_kBtuh']
            else:
                raise KeyError("Missing cooling load columns")
        
        df[f'{end_use}_{replace_or_upgrade}_metric1'] = df['total_cooling_load_kBtuh'] / 12.0
        
        # Metric2: SEER efficiency rating
        if 'upgrade_hvac_cooling_efficiency' in df.columns:
            df[f'{end_use}_{replace_or_upgrade}_metric2'] = df['upgrade_hvac_cooling_efficiency'].str.extract(
                r'SEER (\d+\.?\d*)', expand=False).astype(float)
        elif 'upgrade_SEER' in df.columns:
            # Fallback to heating SEER for heat pumps (same equipment serves both)
            df[f'{end_use}_{replace_or_upgrade}_metric2'] = df['upgrade_SEER'].astype(float)
        else:
            raise KeyError("Missing upgrade_hvac_cooling_efficiency or upgrade_SEER")
    
    elif end_use == 'waterHeating':
        # Metric1: UEF (Unified Energy Factor) - DOE efficiency standard
        if 'upgrade_water_heater_efficiency' not in df.columns:
            raise KeyError("Missing 'upgrade_water_heater_efficiency'")
        # Extract UEF from strings like "Electric Heat Pump, 50 gal, 3.45 UEF"
        df[f'{end_use}_{replace_or_upgrade}_metric1'] = df['upgrade_water_heater_efficiency'].str.extract(
            r'(\d+\.?\d*)\s*UEF', expand=False).astype(float)
        
        # Metric2: Tank capacity in gallons (drives material cost)
        if 'size_water_heater_gal' not in df.columns:
            raise KeyError("Missing 'size_water_heater_gal'")
        df[f'{end_use}_{replace_or_upgrade}_metric2'] = df['size_water_heater_gal'].astype(float)
    
    elif end_use == 'clothesDrying':
        # Metric1: Drum volume (constant across residential dryers - ~7 cu ft)
        # pm1_lower_bound + pm1_upper_bound / 2
        df[f'{end_use}_{replace_or_upgrade}_metric1'] = 7.0
        
        # Metric2: CEF (Combined Energy Factor) - varies by technology
        if 'upgrade_clothes_dryer' not in df.columns:
            raise KeyError("Missing 'upgrade_clothes_dryer'")
        
        # Heat pump dryers are much more efficient (5.2 vs 2.7 lbs/kWh)
        # The clothes dryer upgrade is either a heat pump or standard electric dryer
        is_hp = df['upgrade_clothes_dryer'].str.contains('Heat Pump|HP', case=False, na=False)
        df[f'{end_use}_{replace_or_upgrade}_metric2'] = is_hp.map({True: 5.2, False: 2.7})
    
    elif end_use == 'cooking':
        # Metric1: Oven volume (constant - ~5 cu ft for residential ranges)
        # DO NOT HARDCODE. USE LOWER+UPPER / 2 TO AVOID HARDCODING
        df[f'{end_use}_{replace_or_upgrade}_metric1'] = 5.0
        
        # Metric2: Not used for cooking (all ranges have similar configurations)
        df[f'{end_use}_{replace_or_upgrade}_metric2'] = np.nan
    
    return df


def add_remdb_upgrade_row_ids(
    df: pd.DataFrame,
    end_use: str,
    menu_mp: int
) -> pd.DataFrame:
    """
    Assign unique row_id for matching to REMDB v4 data for upgrade installed cost calculations.


    Maps equipment specifications to REMDB v4 row identifiers based on
    technology type and measure package.
    
    Args:
        df: DataFrame with equipment specifications
        end_use: Equipment category
        menu_mp: Measure package (7, 8, 9, 10)
        
    Returns:
        DataFrame with row_id_{end_use}_{replace_or_upgrade} column
    """
    # This function is for retrofit upgrade installed costs
    replace_or_upgrade = 'upgrade'

    df = df.copy()
    
    if end_use == 'heating':
        if 'hvac_has_ducts' not in df.columns:
            raise ValueError("Missing 'hvac_has_ducts' column for heating row_id assignment")
        
        # Menu package 7: Standard heat pumps (SEER 18)
        # Menu packages 8-10: High-efficiency heat pumps (SEER 29.3)
        # Ducted vs non-ducted determines ASHP vs MSHP technology
        conditions = [
            (df['hvac_has_ducts'] == 'Yes') & (menu_mp == 7),
            (df['hvac_has_ducts'] == 'No') & (menu_mp == 7),
            (df['hvac_has_ducts'] == 'Yes') & (menu_mp != 7),
            (df['hvac_has_ducts'] == 'No') & (menu_mp != 7)
        ]
        
        choices = [
            'air_source_heat_pump_centrally_ducted',
            'air_source_heat_pump_non_ducted_single_zone',
            'air_source_heat_pump_centrally_ducted_with_new_circuit',
            'air_source_heat_pump_non_ducted_multi_zone'
        ]
        
        df[f'row_id_{end_use}_{replace_or_upgrade}'] = np.select(conditions, choices, default='unknown')
        
    elif end_use == 'cooling':
        # Similar logic to heating - cooling upgrades mirror heat pump installations
        if 'hvac_has_ducts' not in df.columns:
            raise ValueError("Missing 'hvac_has_ducts' column for cooling row_id assignment")
        
        conditions = [
            (df['hvac_has_ducts'] == 'Yes') & (menu_mp == 7),
            (df['hvac_has_ducts'] == 'No') & (menu_mp == 7),
            (df['hvac_has_ducts'] == 'Yes') & (menu_mp != 7),
            (df['hvac_has_ducts'] == 'No') & (menu_mp != 7)
        ]
        
        choices = [
            'air_source_heat_pump_centrally_ducted',
            'air_source_heat_pump_non_ducted_single_zone', 
            'air_source_heat_pump_centrally_ducted_with_new_circuit',
            'air_source_heat_pump_non_ducted_multi_zone'
        ]
        
        df[f'row_id_{end_use}_{replace_or_upgrade}'] = np.select(conditions, choices, default='unknown')
        
    elif end_use == 'waterHeating':
        # All water heating upgrades use heat pump water heaters
        df[f'row_id_{end_use}_{replace_or_upgrade}'] = 'water_heater_hp_tank'
        
    elif end_use == 'clothesDrying':
        if 'upgrade_clothes_dryer' not in df.columns:
            raise ValueError("Missing 'upgrade_clothes_dryer' column")
        
        # Heat pump dryers vs standard electric dryers
        is_heat_pump = df['upgrade_clothes_dryer'].str.contains('Heat Pump', na=False)
        df[f'row_id_{end_use}_{replace_or_upgrade}'] = np.where(
            is_heat_pump,
            'clothes_dryer_heat_pump',
            'clothes_dryer_electric'
        )
        
    elif end_use == 'cooking':
        if 'upgrade_cooking_range' not in df.columns:
            raise ValueError("Missing 'upgrade_cooking_range' column")
        
        # Induction ranges vs standard electric ranges
        is_induction = df['upgrade_cooking_range'].str.contains('Induction', na=False)
        df[f'row_id_{end_use}_{replace_or_upgrade}'] = np.where(
            is_induction,
            'cooking_range_induction',
            'cooking_range_electric'
        )
        
    else:
        raise ValueError(f"Invalid end_use: {end_use}")
    
    return df


# ========== Main function to call all steps in sequence ==========
def calculate_upgrade_installed_cost(
    df: pd.DataFrame,
    remdb_v4_costs: pd.DataFrame,
    menu_mp: int,
    end_use: str,
    replace_or_upgrade: str = 'upgrade',
    percentile: str = 'mid'
) -> pd.DataFrame:
    """
    Calculate Upgrade installed costs using REMDB v4 methodology with validation framework.
    
    This function orchestrates the four-step process:
    1. Extract metrics (capacity, efficiency)
    2. Assign row_id (technology mapping)
    3. Map REMDB parameters (coefficients, multiplier)
    4. Calculate costs (regression formula)
    
    Uses the validation framework to ensure only valid homes receive cost calculations.
    
    Args:
        df: DataFrame with equipment specifications
        remdb_v4_costs: REMDB v4 cost data (indexed by row_id)
        menu_mp: Measure package (7, 8, 9, 10)
        end_use: Equipment category
        replace_or_upgrade: 'replace' or 'upgrade'
        percentile: Cost percentile ('low', 'mid', 'high')
        
    Returns:
        DataFrame with calculated costs and intermediate columns
        
    Notes:
        This function implements the validation framework:
        1. Uses initialize_validation_tracking() to determine valid homes
        2. Creates retrofit-only series with NaN for invalid homes
        3. Calculates values only for valid homes with identifiable technology
        4. Applies final verification masking
    """
    # This function is for retrofit upgrade installed costs
    replace_or_upgrade = 'upgrade'

    print(f"\nStarting {end_use} {replace_or_upgrade} installed cost calculation (REMDB v4)")
    
    if menu_mp not in [7, 8, 9, 10]:
        raise ValueError(f"Invalid menu_mp: {menu_mp}. Must be 7, 8, 9, or 10")
    
    # ===== STEP 1: Initialize validation tracking =====
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        df, end_use, menu_mp, verbose=True)
    
    print(f"Found {valid_mask.sum()} valid homes out of {len(df_copy)} for {end_use} installation")
    
    # Step 1/4 Extract performance metrics for REMDB v4 cost estimation. Then add cols to main df
    print(f"  Step 1/4: Extracting performance metrics for REMDB v4 cost estimation...")
    df_copy = add_remdb_upgrade_metrics(df_copy, end_use)
    
    # Step 2: Assign REMDB row_id based on technology
    print(f"  Step 2/4: Assign unique row_id based on technology to map relevant REMDB v4 data cols...")
    df_copy = add_remdb_upgrade_row_ids(df_copy, end_use, menu_mp)
    
    # Step 3: Map cost parameters from REMDB v4 database using unique row_id
    print(f"  Step 3/4: Mapping cost parameters from REMDB v4 database using unique row_id...")
    df_copy = map_remdb_cost_parameters(df_copy, remdb_v4_costs, end_use, replace_or_upgrade, percentile)
    
    # ===== STEP 2: Initialize result series with template =====
    # Use create_retrofit_only_series to properly initialize with zeros for valid homes, NaN for others
    result_series = create_retrofit_only_series(df_copy, valid_mask)
    
    # ===== STEP 3 & 4: Valid-Only Calculation =====
    # Step 4/4: Calculate installed cost of measure package retrofit upgrades using regression formula
    print(f"  Step 4/4: Calculating installed cost of measure package retrofit upgrades using regression formula...")
    
    # UPDATED: Column name changed from 'installationCost' to 'upgrade_installed_cost'
    cost_col = f'mp{menu_mp}_{end_use}_upgrade_installed_cost'
    
    # Calculate costs - the regression formula applies validation mask internally
    calculated_costs = remdb_cost_regression_formula(df_copy, replace_or_upgrade, end_use, percentile)
    
    # Update result series with calculated values (only for valid homes due to internal masking)
    result_series.loc[valid_mask] = calculated_costs.loc[valid_mask]
    
    # Create DataFrame with new column
    df_new_columns = pd.DataFrame({cost_col: result_series})
    
    # Apply new columns to DataFrame with proper tracking
    df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
        df_copy, df_new_columns, end_use, category_columns_to_mask, all_columns_to_mask)
    
    # ===== STEP 5: Apply final verification masking for consistency =====
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=True)
    
    # Report summary
    valid_count = df_copy[cost_col].notna().sum()
    mean_cost = df_copy[cost_col].mean()
    print(f"  Calculated costs for {valid_count:,} homes (mean: ${mean_cost:,.2f})\n")
    
    return df_copy
