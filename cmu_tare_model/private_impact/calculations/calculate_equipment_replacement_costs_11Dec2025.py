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
    remdb_cost_regression_formula
)

"""
========================================================================================================================================================================
OVERVIEW: CALCULATE REPLACEMENT INSTALLED COSTS FOR VARIOUS END USES (REMDB V4 METHODOLOGY)
========================================================================================================================================================================
This module calculates REPLACEMENT installed costs for equipment retrofits using REMDB v4 regression methodology.
It replaces the probabilistic sampling approach (REMDB v3) with deterministic regression equations.

This module calculates REPLACEMENT installed costs for baseline equipment using REMDB v4 regression methodology.

CRITICAL DISTINCTION: REPLACEMENT vs UPGRADE
- **Replacement costs**: Cost to replace existing equipment with LIKE-FOR-LIKE technology (counterfactual)
- **Upgrade costs**: Cost to retrofit to improved technology (e.g., gas furnace → heat pump)

PREREQUISITE: Metrics must be extracted FIRST using add_remdb_replacement_metrics() from extract_equipment_metrics.py

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
        end_use: Equipment category ('heating', 'cooling', 'waterHeating', 
                 'clothesDrying', 'cooking').
        
    Returns:
        DataFrame with row_id_{end_use}_replace column added.
    """
    replace_or_upgrade = 'replace'
    df = df.copy()
    
    if end_use == 'heating':
        if 'base_heating_fuel' not in df.columns:
            raise ValueError("Missing 'base_heating_fuel' column")
        
        has_heating_type = 'heating_type' in df.columns
        
        if has_heating_type:
            conditions = [
                (df['base_heating_fuel'] == 'Propane'),
                (df['base_heating_fuel'] == 'Fuel Oil'),
                (df['base_heating_fuel'] == 'Natural Gas'),
                (df['base_heating_fuel'] == 'Electricity') & (df['heating_type'] == 'Electricity ASHP'),
                (df['base_heating_fuel'] == 'Electricity')
            ]
            choices = [
                'furnaces_gas_furnace',  # Proxy for propane
                'furnaces_gas_furnace',  # Proxy for fuel oil
                'furnaces_gas_furnace',
                'air_source_heat_pump_centrally_ducted',
                'electric_baseboard_default'
            ]
        else:
            conditions = [
                (df['base_heating_fuel'] == 'Propane'),
                (df['base_heating_fuel'] == 'Fuel Oil'),
                (df['base_heating_fuel'] == 'Natural Gas'),
                (df['base_heating_fuel'] == 'Electricity')
            ]
            choices = [
                'furnaces_gas_furnace',
                'furnaces_gas_furnace',
                'furnaces_gas_furnace',
                'electric_baseboard_default'
            ]
        
        df[f'row_id_{end_use}_{replace_or_upgrade}'] = np.select(conditions, choices, default='unknown')
    
    elif end_use == 'cooling':
        if 'hvac_has_ducts' not in df.columns:
            raise ValueError("Missing 'hvac_has_ducts' column")
        
        conditions = [
            (df['hvac_has_ducts'] == 'Yes'),
            (df['hvac_has_ducts'] == 'No')
        ]
        choices = [
            'air_conditioner_centrally_ducted',
            'air_conditioner_room_ac_window_or_through_wall'
        ]
        df[f'row_id_{end_use}_{replace_or_upgrade}'] = np.select(conditions, choices, default='unknown')
    
    elif end_use == 'waterHeating':
        if 'base_waterHeating_fuel' not in df.columns:
            raise ValueError("Missing 'base_waterHeating_fuel' column")
        
        has_efficiency = 'water_heater_efficiency' in df.columns
        
        if has_efficiency:
            is_hp = df['water_heater_efficiency'].str.contains('Heat Pump', case=False, na=False)
            conditions = [
                (df['base_waterHeating_fuel'] == 'Fuel Oil'),
                (df['base_waterHeating_fuel'] == 'Natural Gas'),
                (df['base_waterHeating_fuel'] == 'Propane'),
                (df['base_waterHeating_fuel'] == 'Electricity') & ~is_hp,
                (df['base_waterHeating_fuel'] == 'Electricity') & is_hp
            ]
            choices = [
                'water_heater_gas_storage',
                'water_heater_gas_storage',
                'water_heater_gas_storage',
                'water_heater_electric_storage',
                'water_heater_hp_tank'
            ]
        else:
            conditions = [
                (df['base_waterHeating_fuel'] == 'Fuel Oil'),
                (df['base_waterHeating_fuel'] == 'Natural Gas'),
                (df['base_waterHeating_fuel'] == 'Propane'),
                (df['base_waterHeating_fuel'] == 'Electricity')
            ]
            choices = [
                'water_heater_gas_storage',
                'water_heater_gas_storage',
                'water_heater_gas_storage',
                'water_heater_electric_storage'
            ]
        
        df[f'row_id_{end_use}_{replace_or_upgrade}'] = np.select(conditions, choices, default='unknown')
    
    elif end_use == 'clothesDrying':
        if 'base_clothesDrying_fuel' not in df.columns:
            raise ValueError("Missing 'base_clothesDrying_fuel' column")
        
        conditions = [
            (df['base_clothesDrying_fuel'] == 'Electricity'),
            (df['base_clothesDrying_fuel'] == 'Natural Gas'),
            (df['base_clothesDrying_fuel'] == 'Propane')
        ]
        choices = [
            'clothes_dryer_electric',
            'clothes_dryer_gas',
            'clothes_dryer_gas'
        ]
        df[f'row_id_{end_use}_{replace_or_upgrade}'] = np.select(conditions, choices, default='unknown')
    
    elif end_use == 'cooking':
        if 'base_cooking_fuel' not in df.columns:
            raise ValueError("Missing 'base_cooking_fuel' column")
        
        conditions = [
            (df['base_cooking_fuel'] == 'Electricity'),
            (df['base_cooking_fuel'] == 'Natural Gas'),
            (df['base_cooking_fuel'] == 'Propane')
        ]
        choices = [
            'cooking_range_electric',
            'cooking_range_gas',
            'cooking_range_gas'
        ]
        df[f'row_id_{end_use}_{replace_or_upgrade}'] = np.select(conditions, choices, default='unknown')
    
    else:
        raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of: 'heating', 'cooling', 'waterHeating', 'clothesDrying', 'cooking'")
    
    return df

# ========== Map cost parameters from REMDB v4 database using unique row_id ==========
# map_remdb_cost_parameters was moved to calulation_utils.py for modularity

# ========== Calculate installed cost of measure package retrofit upgrades using regression formula ==========
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
    
    # For clothes drying and cooking: calculate metric1 from REMDB bounds
    if end_use in ['clothesDrying', 'cooking']:
        row_id_col = f'row_id_{end_use}_replace'
        metric1_col = f'{end_use}_replace_metric1'
        
        if row_id_col in df_copy.columns and 'pm1_lower_bound' in remdb_v4_costs.columns:
            pm1_lower = df_copy[row_id_col].map(remdb_v4_costs['pm1_lower_bound'])
            pm1_upper = df_copy[row_id_col].map(remdb_v4_costs['pm1_upper_bound'])
            df_copy[metric1_col] = (pm1_lower + pm1_upper) / 2.0
    
    # ===== STEP 2: Initialize result series =====
    result_series = create_retrofit_only_series(df_copy, valid_mask)
    
    # ===== STEP 3 & 4: Calculate costs =====
    cost_col = f'baseline_{end_use}_replacement_installed_cost'
    calculated_costs = remdb_cost_regression_formula(df_copy, replace_or_upgrade, end_use, percentile)
    result_series.loc[valid_mask] = calculated_costs.loc[valid_mask]
    
    df_new_columns = pd.DataFrame({cost_col: result_series})
    
    # ===== Track and apply columns =====
    df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
        df_copy, df_new_columns, end_use, category_columns_to_mask, all_columns_to_mask)
    
    # ===== STEP 5: Final masking =====
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=True)
    
    # Report
    valid_count = df_copy[cost_col].notna().sum()
    if valid_count > 0:
        mean_cost = df_copy[cost_col].mean()
        print(f"Calculated costs for {valid_count:,} homes (mean: ${mean_cost:,.2f})\n")
    
    return df_copy
