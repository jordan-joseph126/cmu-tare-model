import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal

from cmu_tare_model.utils.validation_framework import (
    apply_new_columns_to_dataframe,
    apply_final_masking,
    initialize_validation_tracking,
    create_retrofit_only_series
    )
from cmu_tare_model.utils.remdb_v4_installed_cost_utils import (
    map_remdb_cost_parameters,
    remdb_cost_regression_formula,
    calculate_metric_from_remdb_bounds
    )


"""
========================================================================================================================================================================
OVERVIEW: CALCULATE UPGRADE INSTALLED COSTS FOR VARIOUS END USES (REMDB V4 METHODOLOGY)
========================================================================================================================================================================
This module calculates UPGRADE installed costs for equipment retrofits using REMDB v4 regression methodology.
It replaces the probabilistic sampling approach (REMDB v3) with deterministic regression equations.

Key changes from REMDB v3 to v4:
- Regression-based calculation: Material_Price = (pm1_coef × Metric1) + (pm2_coef × Metric2) + Intercept
- Costs already in 2023$ (no CPI adjustment needed)
- Upgrade installed costs via multipliers OR adders (component-specific)
- Added cooling as new end-use category
- Dynamic row_id mapping replaces hardcoded technology-efficiency pairs

PREREQUISITE: Metrics must be extracted FIRST using add_remdb_upgrade_metrics() from remdb_v4_installed_cost_utils.py

# UPDATED DECEMBER 11, 2025 - REMDB V4 METHODOLOGY, SIMPLIFIED ARCHITECTURE

# UPDATED MARCH 24, 2025 @ 4:30 PM - REMOVED RSMEANS CCI ADJUSTMENTS
# UPDATED APRIL 9, 2025 @ 7:30 PM - IMPROVED DOCUMENTATION
# UPDATED APRIL 21, 2025 @ 11:45 PM - COST UTILITY FUNCTION REPLACED REDUNDANT CODE (SEE UTILS FOLDER)
# UPDATED DECEMBER 2, 2025 @ 5:00 PM - UPDATED TO REMDB V4 METHODOLOGY
# UPDATED DECEMBER 10, 2025 @ 2:00 PM - INTEGRATED VALIDATION FRAMEWORK AND UPDATED COLUMN NAMING
"""

# ========== Extract performance metrics for REMDB v4 cost estimation. Then add cols to main df ==========
# add_remdb_upgrade_metrics was moved to calulation_utils.py for modularity

# ========== Assign REMDB row_id based on technology ==========
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
        end_use: Equipment category ('heating', 'cooling', 'waterHeating', 'clothesDrying', 'cooking').
        menu_mp: Measure package (7, 8, 9, 10)
        
    Returns:
        DataFrame with row_id_{end_use}_upgrade column
    """
    df_copy = df.copy()

    # This function is for retrofit upgrade installed costs
    replace_or_upgrade = 'upgrade'
    
    # ========== HVAC OPTIONS: HEATING & COOLING ---> HEAT PUMP ==========
    # MP7 Standard heat pumps (SEER 18) | MP8-10: High-efficiency heat pumps
    # However, the efficiency level does not impact row_id mapping in REMDB v4 but instead pm1/pm2 in the regression formula
    # Generally we use multi-zone non-ducted for homes without ducts, but may update to single-zone in the future for smaller homes 
    # New circuit will be addressed in future versions, but excluded here for simplicity.
    if end_use == 'heating':
        if 'hvac_has_ducts' not in df_copy.columns:
            raise ValueError("Missing 'hvac_has_ducts' column for heating row_id assignment")
        
        conditions = [
            (df_copy['hvac_has_ducts'] == 'Yes'),
            (df_copy['hvac_has_ducts'] == 'No'),
            # (df_copy['hvac_has_ducts'] == 'No') & (df_copy['square_footage'] < 1200)
        ]
        
        choices = [
            'air_source_heat_pump_centrally_ducted',
            'air_source_heat_pump_non_ducted_multi_zone',
            # 'air_source_heat_pump_non_ducted_single_zone',
        ]
        
        df_copy[f'row_id_{end_use}_{replace_or_upgrade}'] = np.select(conditions, choices, default='unknown')
        
    elif end_use == 'cooling':
        # Similar logic to heating - cooling upgrades mirror heat pump installations
        if 'hvac_has_ducts' not in df_copy.columns:
            raise ValueError("Missing 'hvac_has_ducts' column for cooling row_id assignment")
        
        conditions = [
            (df_copy['hvac_has_ducts'] == 'Yes'),
            (df_copy['hvac_has_ducts'] == 'No'),
            # (df_copy['hvac_has_ducts'] == 'No') & (df_copy['square_footage'] < 1200)
        ]
        
        choices = [
            'air_source_heat_pump_centrally_ducted',
            'air_source_heat_pump_non_ducted_multi_zone',
            # 'air_source_heat_pump_non_ducted_single_zone',
        ]
                
        df_copy[f'row_id_{end_use}_{replace_or_upgrade}'] = np.select(conditions, choices, default='unknown')
        
    # ========== NON-HVAC SYSTEMS: WATER HEATING, CLOTHES DRYING, COOKING ==========
    elif end_use == 'waterHeating':
        # All water heating upgrades use heat pump water heaters
        df_copy[f'row_id_{end_use}_{replace_or_upgrade}'] = 'water_heater_hp_tank'
        
    elif end_use == 'clothesDrying':
        if 'upgrade_clothes_dryer' not in df_copy.columns:
            raise ValueError("Missing 'upgrade_clothes_dryer' column")
        
        # Heat pump dryers vs standard electric dryers
        is_heat_pump = df_copy['upgrade_clothes_dryer'].str.contains('Heat Pump', na=False)
        df_copy[f'row_id_{end_use}_{replace_or_upgrade}'] = np.where(
            is_heat_pump,
            'clothes_dryer_heat_pump',
            'clothes_dryer_electric'
        )
        
    elif end_use == 'cooking':
        if 'upgrade_cooking_range' not in df_copy.columns:
            raise ValueError("Missing 'upgrade_cooking_range' column")
        
        # Induction ranges vs standard electric ranges
        is_induction = df_copy['upgrade_cooking_range'].str.contains('Induction', na=False)
        df_copy[f'row_id_{end_use}_{replace_or_upgrade}'] = np.where(
            is_induction,
            'cooking_range_induction',
            'cooking_range_electric'
        )
        
    else:
        raise ValueError(f"Invalid end_use: {end_use}")
    
    return df_copy

# ========== Map cost parameters from REMDB v4 database using unique row_id ==========
# map_remdb_cost_parameters was moved to calulation_utils.py for modularity

# ========== Calculate installed cost of measure package retrofit upgrades using regression formula ==========
def calculate_upgrade_installed_cost(
    df: pd.DataFrame,
    remdb_v4_costs: pd.DataFrame,
    menu_mp: int,
    end_use: str,
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
    
    # PREREQUISITE: Metrics must be extracted FIRST using add_remdb_replacement_metrics().

    # ===== Assign row_ids =====
    df_copy = add_remdb_upgrade_row_ids(df_copy, end_use, menu_mp)
    
    # ===== Map REMDB parameters =====
    df_copy = map_remdb_cost_parameters(df_copy, remdb_v4_costs, end_use, replace_or_upgrade, percentile)
    
    # ===== Missing Performance Metrics Handling =====
    # Calculate missing metrics from REMDB bounds
    # This handles any end-use where physical dimensions aren't in home metadata
    # Currently used for: clothes drying (drum volume), cooking (oven volume)
    metric1_col = f'{end_use}_{replace_or_upgrade}_metric1'
    metric2_col = f'{end_use}_{replace_or_upgrade}_metric2'

    # Identify rows with missing metrics
    metric1_missing_mask = df_copy[metric1_col].isna()
    metric2_missing_mask = df_copy[metric2_col].isna()

    # Calculate metric1 from bounds where missing
    if metric1_missing_mask.any():
        # Check if bounds exist in REMDB (not all end-uses may have bounds)
        if 'pm1_lower_bound' in remdb_v4_costs.columns and 'pm1_upper_bound' in remdb_v4_costs.columns:
            df_copy.loc[metric1_missing_mask, metric1_col] = calculate_metric_from_remdb_bounds(
                df=df_copy[metric1_missing_mask],  # Pass only rows needing calculation
                remdb_v4_costs=remdb_v4_costs,
                end_use=end_use,
                replace_or_upgrade=replace_or_upgrade,
                lower_bound_col='pm1_lower_bound',
                upper_bound_col='pm1_upper_bound'
            )
            print(f"  Calculated {metric1_missing_mask.sum():,} missing {metric1_col} values from REMDB bounds")

    # Calculate metric2 from bounds where missing
    if metric2_missing_mask.any():
        # Check if bounds exist in REMDB (metric2 bounds may not exist for all end-uses)
        if 'pm2_lower_bound' in remdb_v4_costs.columns and 'pm2_upper_bound' in remdb_v4_costs.columns:
            df_copy.loc[metric2_missing_mask, metric2_col] = calculate_metric_from_remdb_bounds(
                df=df_copy[metric2_missing_mask],  # Pass only rows needing calculation
                remdb_v4_costs=remdb_v4_costs,
                end_use=end_use,
                replace_or_upgrade=replace_or_upgrade,
                lower_bound_col='pm2_lower_bound',
                upper_bound_col='pm2_upper_bound'
            )
            print(f"  Calculated {metric2_missing_mask.sum():,} missing {metric2_col} values from REMDB bounds")

    # ===== STEP 2: Initialize result series with template =====
    # Use create_retrofit_only_series to properly initialize with zeros for valid homes, NaN for others
    result_series = create_retrofit_only_series(df_copy, valid_mask)
    
    # ===== STEP 3 & 4: Valid-Only Calculation =====
    
    # UPDATED: Column name changed from 'installationCost' to 'upgrade_installed_cost'
    cost_col = f'mp{menu_mp}_{end_use}_upgrade_installed_cost_{percentile}'
    
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
