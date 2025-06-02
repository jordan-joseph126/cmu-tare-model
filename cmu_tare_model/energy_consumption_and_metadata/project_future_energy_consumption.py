import pandas as pd
import numpy as np
from typing import Tuple

from cmu_tare_model.constants import EQUIPMENT_SPECS
from cmu_tare_model.utils.precompute_hdd_factors import precompute_hdd_factors
from cmu_tare_model.utils.validation_framework import (
    mask_category_specific_data,
    get_valid_calculation_mask,
)

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
PROJECT FUTURE ENERGY CONSUMPTION
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
- HDD factors for different census divisions and years
- Functions to project future energy consumption based on HDD projections
"""


def project_future_consumption(
    df: pd.DataFrame, 
    menu_mp: int,
    base_year: int = 2024
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Projects future energy consumption based on baseline or upgraded equipment specifications.

    This function calculates projected energy consumption for each equipment category (heating,
    water heating, clothes drying, and cooking) for future years based on their expected lifetime.
    
    UPDATED: It applies climate adjustment factors for SPACE HEATING ONLY. (previously also applied to water heating)

    Args:
        df: The input DataFrame containing baseline consumption data.
        menu_mp: Indicates the measure package to apply. 0 for baseline, or a positive integer
            (e.g., 8, 9, 10) for retrofit scenarios.
        base_year: The starting year for projections. Defaults to 2024.

    Returns:
        A tuple containing two DataFrames:
            - df_copy: The updated DataFrame with all columns including the projected consumption data.
            - df_consumption: The DataFrame containing only the projected consumption data.

    Raises:
        KeyError: If the 'census_division' column is missing from the DataFrame.
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Check if the 'census_division' column exists in the DataFrame
    if 'census_division' not in df_copy.columns:
        raise KeyError("'census_division' column is missing from the DataFrame")

    # Prepare a dictionary to hold new columns for projected consumption
    new_columns = {}
    
    # Dictionary to track all columns that need masking, organized by category
    all_columns_to_mask = {category: [] for category in EQUIPMENT_SPECS}

    # Precompute HDD adjustment factors by region and year
    hdd_factors_per_year = precompute_hdd_factors(df_copy)

    # Get valid calculation masks for each category
    print("\nDetermining valid homes for calculations:")
    category_masks = {}
    for category in EQUIPMENT_SPECS:
        category_masks[category] = get_valid_calculation_mask(df_copy, category, menu_mp, verbose=True)
    
    # Process each equipment category
    for category, lifetime in EQUIPMENT_SPECS.items():
        print(f"\n===== Processing {category.upper()} =====")
        
        # Add baseline consumption column to masking list
        baseline_cons_col = f'baseline_{category}_consumption'
        all_columns_to_mask[category].append(baseline_cons_col)
        
        # Get the valid homes mask for this category
        valid_mask = category_masks[category]
        valid_count = valid_mask.sum()
        total_count = len(valid_mask)
        
        # Skip if no valid homes for this category
        if valid_count == 0:
            print(f"WARNING: No valid homes for {category}. Skipping projections.")
            continue
            
        print(f"Projecting consumption for {valid_count} out of {total_count} homes")
        
        # =================================================================================
        # BASELINE SCENARIO (menu_mp == 0)
        # =================================================================================
        if menu_mp == 0:
            print(f"Projecting Future Energy Consumption (Baseline Equipment): {category}")
            for year in range(1, lifetime + 1):
                year_label = year + (base_year - 1)

                # Create column name for projected baseline consumption
                projected_baseline_cons_col = f'baseline_{year_label}_{category}_consumption'
                all_columns_to_mask[category].append(projected_baseline_cons_col)

                # Check that an HDD factor exists for the current year
                if year_label not in hdd_factors_per_year.columns:
                    raise KeyError(f"HDD factor for year {year_label} not found.")

                # Retrieve HDD factor for the current year; use factor only for HEATING categories
                hdd_factor = hdd_factors_per_year[year_label]
                adjusted_hdd_factor = hdd_factor if category in ['heating'] else pd.Series(1.0, index=df_copy.index)

                # Initialize with NaN for all homes
                new_columns[projected_baseline_cons_col] = pd.Series(np.nan, index=df_copy.index)
                
                # Calculate only for valid homes
                valid_indices = valid_mask[valid_mask].index
                if len(valid_indices) > 0:
                    new_columns[projected_baseline_cons_col].loc[valid_indices] = (
                        df_copy.loc[valid_indices, baseline_cons_col] * 
                        adjusted_hdd_factor.loc[valid_indices]
                    ).round(2)

        # =================================================================================
        # POST-RETROFIT SCENARIO (menu_mp != 0)
        # =================================================================================
        else:
            print(f"Projecting Future Energy Consumption (Upgraded Equipment): {category}")
            # Get post-retrofit consumption column and add to masking list
            mp_cons_col = f'mp{menu_mp}_{category}_consumption'
            all_columns_to_mask[category].append(mp_cons_col)

            for year in range(1, lifetime + 1):
                year_label = year + (base_year - 1)

                # Create column names for projected values
                projected_baseline_cons_col = f'baseline_{year_label}_{category}_consumption'
                projected_mp_cons_col = f'mp{menu_mp}_{year_label}_{category}_consumption'
                reduction_cons_col = f'mp{menu_mp}_{year_label}_{category}_reduction_consumption'
                
                # Add all projected columns to masking list
                all_columns_to_mask[category].extend([
                    projected_baseline_cons_col, 
                    projected_mp_cons_col, 
                    reduction_cons_col
                ])

                # Check that an HDD factor exists for the current year
                if year_label not in hdd_factors_per_year.columns:
                    raise KeyError(f"HDD factor for year {year_label} not found.")

                # Retrieve HDD factor for the current year; use factor only for HEATING categories
                hdd_factor = hdd_factors_per_year[year_label]
                adjusted_hdd_factor = hdd_factor if category in ['heating'] else pd.Series(1.0, index=df_copy.index)

                # Initialize with NaN for all homes
                valid_indices = valid_mask[valid_mask].index
                
                # Calculate baseline projections if not already calculated
                if projected_baseline_cons_col not in new_columns and projected_baseline_cons_col not in df_copy.columns:
                    new_columns[projected_baseline_cons_col] = pd.Series(np.nan, index=df_copy.index)
                    if len(valid_indices) > 0:
                        new_columns[projected_baseline_cons_col].loc[valid_indices] = (
                            df_copy.loc[valid_indices, baseline_cons_col] * 
                            adjusted_hdd_factor.loc[valid_indices]
                        ).round(2)
                
                # Calculate post-retrofit projections
                new_columns[projected_mp_cons_col] = pd.Series(np.nan, index=df_copy.index)
                if len(valid_indices) > 0:
                    new_columns[projected_mp_cons_col].loc[valid_indices] = (
                        df_copy.loc[valid_indices, mp_cons_col] * 
                        adjusted_hdd_factor.loc[valid_indices]
                    ).round(2)
                
                # Calculate consumption reduction
                new_columns[reduction_cons_col] = pd.Series(np.nan, index=df_copy.index)
                if len(valid_indices) > 0:
                    # Get baseline projection from new_columns or df_copy
                    if projected_baseline_cons_col in new_columns:
                        baseline_proj = new_columns[projected_baseline_cons_col]
                    else:
                        baseline_proj = df_copy[projected_baseline_cons_col]
                    
                    # Calculate reduction only for valid homes
                    new_columns[reduction_cons_col].loc[valid_indices] = (
                        baseline_proj.loc[valid_indices] - 
                        new_columns[projected_mp_cons_col].loc[valid_indices]
                    ).round(2)

    # Calculate the new columns based on policy scenario and create dataframe based on df_copy index
    df_new_columns = pd.DataFrame(new_columns, index=df_copy.index)

    # Identify overlapping columns between the new and existing DataFrame
    overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)

    # Remove any columns that overlap with the newly generated columns from df_copy
    if not overlapping_columns.empty:
        df_copy.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into df_copy, aligning rows by index
    df_copy = df_copy.join(df_new_columns, how='left')

    # Double-check masking for all columns to ensure consistency
    print("\nVerifying masking for all projected columns:")
    for category, cols_to_mask in all_columns_to_mask.items():
        # Filter out columns that don't exist in df_copy
        existing_cols = [col for col in cols_to_mask if col in df_copy.columns]
        if existing_cols:
            df_copy = mask_category_specific_data(df_copy, existing_cols, category, verbose=True)

    # Create a copy for the consumption-only data
    df_consumption = df_copy.copy()

    return df_copy, df_consumption
