import pandas as pd
import numpy as np
from typing import Literal

from cmu_tare_model.utils.validation_framework import (
    apply_new_columns_to_dataframe,
    apply_final_masking,
    initialize_validation_tracking,
    create_retrofit_only_series
)
from cmu_tare_model.utils.remdb_v4_installed_cost_utils import (
    map_remdb_cost_parameters,
    calculate_metric_from_remdb_bounds,
    remdb_cost_regression_formula
)

"""
========================================================================================================================================================================
OVERVIEW: CALCULATE REPLACEMENT INSTALLED COSTS FOR VARIOUS END USES (REMDB V4 METHODOLOGY)
========================================================================================================================================================================
This module calculates REPLACEMENT installed costs for equipment retrofits using REMDB v4 regression methodology.
It replaces the probabilistic sampling approach (REMDB v3) with deterministic regression equations.

CRITICAL DISTINCTION: REPLACEMENT vs UPGRADE
- **Replacement costs**: Cost to replace existing equipment with LIKE-FOR-LIKE technology (counterfactual)
- **Upgrade costs**: Cost to retrofit to improved technology (e.g., gas furnace → heat pump)

PREREQUISITE: Metrics must be extracted FIRST using add_remdb_replacement_metrics() from remdb_v4_installed_cost_utils.py

Key changes from REMDB v3 to v4:
- Regression-based calculation: Material_Price = (pm1_coef × Metric1) + (pm2_coef × Metric2) + Intercept
- Costs already in 2023$ (no CPI adjustment needed)
- Replacement installed costs via multipliers OR adders (component-specific)
- Added cooling as new end-use category
- Dynamic row_id mapping replaces hardcoded technology-efficiency pairs

# UPDATED DECEMBER 11, 2025 - REMDB V4 METHODOLOGY, SIMPLIFIED ARCHITECTURE

# UPDATED MARCH 24, 2025 @ 4:30 PM - REMOVED RSMEANS CCI ADJUSTMENTS
# UPDATED APRIL 9, 2025 @ 7:30 PM - IMPROVED DOCUMENTATION
# UPDATED APRIL 21, 2025 @ 11:45 PM - COST UTILITY FUNCTION REPLACED REDUNDANT CODE (SEE UTILS FOLDER)
# UPDATED DECEMBER 2, 2025 @ 5:00 PM - UPDATED TO REMDB V4 METHODOLOGY
# UPDATED DECEMBER 10, 2025 @ 2:00 PM - INTEGRATED VALIDATION FRAMEWORK AND UPDATED COLUMN NAMING
"""

# ========== Extract performance metrics for REMDB v4 cost estimation. Then add cols to main df ==========
# add_remdb_replacement_metrics was moved to calulation_utils.py for modularity

# ========== Assign REMDB row_id based on technology ==========
def add_remdb_replacement_row_ids(
    df: pd.DataFrame,
    end_use: str
) -> pd.DataFrame:
    """Assign REMDB v4 row_id for baseline equipment cost lookups.
    
    Maps baseline equipment technologies to REMDB v4 row identifiers.
    
    Args:
        df: DataFrame with baseline fuel type columns.
        end_use: Equipment category ('heating', 'cooling').
        
    Returns:
        DataFrame with row_id_{end_use}_replace column added.

    FUTURE: After successful testing, expand to non-HVAC end-uses (waterHeating, clothesDrying, cooking).
        
    """
    df_copy = df.copy()

    replace_or_upgrade = 'replace'
    
    # The validation/include flags resolve options where there is None

    # ========== HVAC OPTIONS: HEATING & COOLING ==========
    # The efficiency level does not impact row_id mapping in REMDB v4 but instead pm1/pm2 in the regression formula
    # Generally we use multi-zone non-ducted for homes without ducts, but may update to single-zone in the future for smaller homes 
    # New circuit will be addressed in future versions, but excluded here for simplicity.
    if end_use == 'heating':
        if 'base_heating_fuel' not in df_copy.columns:
            raise ValueError("Missing 'base_heating_fuel' column")
        
        conditions = [
            (df_copy['base_heating_fuel'] == 'Propane'),
            (df_copy['base_heating_fuel'] == 'Fuel Oil'),
            (df_copy['base_heating_fuel'] == 'Natural Gas'),
            (df_copy['base_heating_fuel'] == 'Electricity') & (df_copy['heating_type'] != 'Electricity ASHP'),
            (df_copy['heating_type'] == 'Electricity ASHP') & (df_copy['hvac_has_ducts'] == 'Yes'),
            (df_copy['heating_type'] == 'Electricity ASHP') & (df_copy['hvac_has_ducts'] == 'No')
            ]

        choices = [
            'furnaces_gas_furnace',  # Proxy for propane
            'furnaces_gas_furnace',  # Proxy for fuel oil
            'furnaces_gas_furnace',
            'electric_baseboard_default',
            'air_source_heat_pump_centrally_ducted',
            'air_source_heat_pump_non_ducted_multi_zone'
            ]
        
        df_copy[f'row_id_{end_use}_{replace_or_upgrade}'] = np.select(conditions, choices, default='unknown')
    
    elif end_use == 'cooling':
        if 'hvac_has_ducts' not in df_copy.columns:
            raise ValueError("Missing 'hvac_has_ducts' column")
        
        conditions = [
            (df_copy['hvac_cooling_type'] == 'Room AC'),
            (df_copy['hvac_cooling_type'] == 'Central AC'),
            (df_copy['hvac_cooling_type'] == 'Heat Pump') & (df_copy['hvac_has_ducts'] == 'Yes'),
            (df_copy['hvac_cooling_type'] == 'Heat Pump') & (df_copy['hvac_has_ducts'] == 'No')
        ]

        choices = [
            'air_conditioner_room_ac_window_or_through_wall',
            'air_conditioner_centrally_ducted',
            'air_source_heat_pump_centrally_ducted',
            'air_source_heat_pump_non_ducted_multi_zone'
        ]

        df_copy[f'row_id_{end_use}_{replace_or_upgrade}'] = np.select(conditions, choices, default='unknown')
        
    # =========================================
    # DELETE FOR NOW - NON-HVAC END USES
    # =========================================

    else:
        # raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of: 'heating', 'cooling', 'waterHeating', 'clothesDrying', 'cooking'")
        raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of: 'heating', 'cooling'")

    return df_copy

# ========== Map cost parameters from REMDB v4 database using unique row_id ==========
# map_remdb_cost_parameters was moved to calulation_utils.py for modularity

# ========== Calculate installed cost of replacing existing equipment using regression formula ==========
def calculate_replacement_installed_cost(
    df: pd.DataFrame,
    remdb_v4_costs: pd.DataFrame,
    end_use: str,
    percentile: str = 'mid'
) -> pd.DataFrame:
    """Calculate REPLACEMENT installed costs using REMDB v4 methodology.
    
    PREREQUISITE: Metrics must be extracted FIRST using add_remdb_replacement_metrics().
    
    Args:
        df: DataFrame with baseline metrics already extracted.
        remdb_v4_costs: REMDB v4 cost database (indexed by row_id).
        end_use: Equipment category.
        percentile: Cost percentile ('low', 'mid', 'high'). Default 'mid'.
        
    Returns:
        DataFrame with baseline_{end_use}_replacement_installed_cost column added.
    """
    replace_or_upgrade = 'replace'
    
    if percentile not in ['low', 'mid', 'high']:
        raise ValueError(f"Invalid percentile: '{percentile}'. Must be 'low', 'mid', or 'high'")
    
    print(f"\nStarting {end_use} replacement cost calculation (REMDB v4)")
    
    # ===== STEP 1: Initialize validation tracking =====
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        df, end_use, menu_mp=0, verbose=True)
    
    # ===== Assign row_ids =====
    df_copy = add_remdb_replacement_row_ids(df_copy, end_use)
    
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
    
    # UPDATED: Column name changed from 'replacementCost' to 'replacement_installed_cost'
    cost_col = f'baseline_{end_use}_replacement_installed_cost_{percentile}'

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
