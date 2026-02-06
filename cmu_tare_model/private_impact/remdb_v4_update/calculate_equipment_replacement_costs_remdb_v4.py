"""
========================================================================================================================================================================
CALCULATE REPLACEMENT INSTALLED COSTS FOR VARIOUS END USES (REMDB V4 METHODOLOGY)
========================================================================================================================================================================
This module calculates REPLACEMENT installed costs for equipment retrofits using REMDB v4 regression methodology.
It replaces the probabilistic sampling approach (REMDB v3) with deterministic regression equations.

CRITICAL DISTINCTION: REPLACEMENT vs UPGRADE
- **Replacement costs**: Cost to replace existing equipment with LIKE-FOR-LIKE technology (counterfactual)
- **Upgrade costs**: Cost to retrofit to improved technology (e.g., gas furnace → heat pump)

Key changes from REMDB v3 to v4:
- Regression-based calculation: Material_Price = (pm1_coef × Metric1) + (pm2_coef × Metric2) + Intercept
- Costs already in 2023$ (no CPI adjustment needed)
- Replacement installed costs via multipliers OR adders (component-specific)
- Added cooling as new end-use category
- Dynamic row_id mapping replaces hardcoded technology-efficiency pairs

PREREQUISITE: Call add_remdb_replacement_metrics() first to prepare pm1/pm2 columns.

The cost calculation is simple:
    Material_Price = (pm1 × pm1_coef) + (pm2 × pm2_coef) + intercept
    Installed_Cost = (Material_Price × multiplier) + adder

# UPDATED DECEMBER 15, 2025 - REMDB V4 METHODOLOGY, SIMPLIFIED ARCHITECTURE

# UPDATED MARCH 24, 2025 @ 4:30 PM - REMOVED RSMEANS CCI ADJUSTMENTS
# UPDATED APRIL 9, 2025 @ 7:30 PM - IMPROVED DOCUMENTATION
# UPDATED APRIL 21, 2025 @ 11:45 PM - COST UTILITY FUNCTION REPLACED REDUNDANT CODE (SEE UTILS FOLDER)
# UPDATED DECEMBER 2, 2025 @ 5:00 PM - UPDATED TO REMDB V4 METHODOLOGY
# UPDATED DECEMBER 10, 2025 @ 2:00 PM - INTEGRATED VALIDATION FRAMEWORK AND UPDATED COLUMN NAMING
"""

import pandas as pd
import numpy as np
from typing import Tuple

from cmu_tare_model.constants import VALID_MENU_MPS, EQUIPMENT_SPECS, VALID_CATEGORIES

from cmu_tare_model.utils.validation_framework import (
    apply_new_columns_to_dataframe,
    apply_final_masking,
    initialize_validation_tracking,
    create_retrofit_only_series
)

def calculate_replacement_installed_cost(
    df: pd.DataFrame,
    df_detailed: pd.DataFrame,
    menu_mp: int,
    end_use: str,
    percentile: str = 'mid'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate REPLACEMENT installed costs using REMDB v4 regression formula.
    
    PREREQUISITE: Call add_remdb_replacement_metrics() first to do the following:
    - Extract metrics (capacity, efficiency)
    - Assign row_id (technology mapping)
    - Map REMDB parameters (coefficients, multiplier)

    Then use this function to calculate installed costs:
    - Calculate costs (regression formula)
    - Uses the validation framework to ensure only valid homes receive cost calculations.

    SPECIAL CASE - COOLING: Creates temporary include_cooling flag for validation,
    then removes it to prevent downstream interference.

    Args:
        df: DataFrame with prepared metrics.
        df_detailed: Detailed DataFrame with regression parameters.
        menu_mp: Measure package number.
        end_use: Equipment category.
        percentile: Cost percentile.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - df_copy: Updated DataFrame with new cost column.
            - df_detailed_out: Updated detailed DataFrame with new cost column.
    """
    # This function is for retrofit REPLACEMENT installed costs
    replacement_or_upgrade = 'replacement'
    
    # Validate menu_mp 
    if menu_mp not in VALID_MENU_MPS:
        raise ValueError(f"Please enter a valid measure package number for menu_mp. Should be one of {VALID_MENU_MPS}.")
    
    if 'cooling' not in EQUIPMENT_SPECS and end_use == 'cooling':
        VALID_CATEGORIES.append('cooling')

    if end_use not in VALID_CATEGORIES:
        raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of {VALID_CATEGORIES}")

    if percentile not in ['low', 'mid', 'high']:
        raise ValueError(f"Invalid percentile: '{percentile}'")
    
    prefix = f'{end_use}_{replacement_or_upgrade}_'

    # ===== Verify prerequisite columns exist =====
    required_cols = [
        f'{prefix}pm1_euss',
        f'{prefix}pm2_euss',
        f'{prefix}pm1_coef_{percentile}',
        f'{prefix}pm2_coef_{percentile}',
        f'{prefix}intercept_{percentile}',
        f'{prefix}multiplier_retrofit',
        f'{prefix}adder_retrofit',
    ]
    missing = [c for c in required_cols if c not in df_detailed.columns]

    if missing:
        raise KeyError(
            f"Missing columns: {missing}\n"
            f"Call add_remdb_replacement_metrics() first to prepare these columns."
        )
    
    print(f"\nCalculating {end_use} replacement costs (REMDB v4)")

    # ===== SPECIAL HANDLING FOR COOLING (METADATA-ONLY) =====
    # Cooling is not in EQUIPMENT_SPECS so doesn't have include_cooling flag.
    # Create temporary flag for validation framework, then clean up.

    created_temp_flag = False
    if end_use == 'cooling' and f'include_{end_use}' not in df.columns:
        # Create temporary inclusion flag: any home with a cooling system
        # (Central AC, Room AC, or Heat Pump in cooling mode)
        df_copy = df.copy()
        df_copy['include_cooling'] = (
            df_copy['hvac_cooling_type'].notna() & 
            (df_copy['hvac_cooling_type'] != 'None')
        )
        created_temp_flag = True
        print("  Created temporary include_cooling flag for metadata calculation")
    else:
        df_copy = df.copy()

    # ===== STEP 1: Initialize validation tracking =====
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        df_copy, end_use, menu_mp=menu_mp, verbose=True)
    
    # ===== COOLING METADATA FIX: Initialize tracking dictionary key =====
    # Since cooling isn't in EQUIPMENT_SPECS, initialize_validation_tracking()
    # doesn't create all_columns_to_mask['cooling']. We need it for Step 4.
    if end_use == 'cooling' and end_use not in all_columns_to_mask:
        all_columns_to_mask['cooling'] = []
        print("  Initialized all_columns_to_mask['cooling'] for metadata tracking")
    
    # ===== STEP 2: Initialize result series with template =====
    # Use create_retrofit_only_series to properly initialize with zeros for valid homes, NaN for others
    result_series = create_retrofit_only_series(df_copy, valid_mask)

    # ===== STEP 3 & 4: Valid-Only Calculation =====
    # Calculate cost using REMDB v4 regression formula
    # Read calculation columns from df_detailed
    pm1 = df_detailed[f'{prefix}pm1_euss']
    pm2 = df_detailed[f'{prefix}pm2_euss']
    pm1_coef = df_detailed[f'{prefix}pm1_coef_{percentile}']
    pm2_coef = df_detailed[f'{prefix}pm2_coef_{percentile}']
    intercept = df_detailed[f'{prefix}intercept_{percentile}']
    multiplier = df_detailed[f'{prefix}multiplier_retrofit']
    adder = df_detailed[f'{prefix}adder_retrofit']
    
    # REMDB v4 regression formula
    material_price = (pm1 * pm1_coef) + (pm2 * pm2_coef) + intercept
    installed_cost = (material_price * multiplier) + adder
    
    # ===== UPDATE: Ensure costs never go negative =====
    # Safety net: Ensure costs never go negative (defense against extreme extrapolation)
    installed_cost = installed_cost.clip(lower=0)

    # ===== Apply validation mask =====
    # Update result series with calculated values (only for valid homes due to internal masking)
    result_series.loc[valid_mask] = installed_cost.loc[valid_mask].round(2)

    # UPDATED: Column name changed from 'replacementCost' to 'replacement_installed_cost'
    cost_col = f'mp{menu_mp}_{end_use}_replacement_installed_cost_{percentile}'

    # Create DataFrame with new column
    df_new_columns = pd.DataFrame({cost_col: result_series})
    
    # Apply new columns to DataFrame with proper tracking
    df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
        df_copy, df_new_columns, end_use, category_columns_to_mask, all_columns_to_mask)
    
    # ===== STEP 5: Apply final verification masking for consistency =====
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=True)
    
    # ===== Add cost column to detailed DataFrame =====
    df_detailed_out = df_detailed.copy()
    df_detailed_out[cost_col] = df_copy[cost_col]
    
    # ===== CLEANUP: Remove temporary cooling flag if created =====
    if created_temp_flag:
        df_copy = df_copy.drop(columns=['include_cooling'])
        print("  Removed temporary include_cooling flag")
    
    # Report summary
    valid_count = df_copy[cost_col].notna().sum()
    mean_cost = df_copy[cost_col].mean()
    print(f"  Calculated costs for {valid_count:,} homes (mean: ${mean_cost:,.2f})\n")
    
    return df_copy, df_detailed_out
