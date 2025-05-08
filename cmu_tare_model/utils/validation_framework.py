"""
validation_framework.py

Core utilities for implementing the five-step data validation framework:
1. Mask Initialization: Determine which homes have valid data
2. Series Initialization: Initialize result series with zeros for valid homes, NaN for others
3. Valid-Only Calculation: Perform calculations only for valid homes
4. Valid-Only Updates: Update only valid homes with calculated values
5. Final Masking: Apply consistent masking to all result columns

This module consolidates and standardizes validation utilities from:
- cost_calculation_utils.py
- data_quality_utils.py
- retrofit_status_utils.py
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Set, Any

from cmu_tare_model.constants import EQUIPMENT_SPECS, UPGRADE_COLUMNS

# ====================================================================================================
# STEP 1: MASK INITIALIZATION
# ====================================================================================================

def initialize_validation_tracking(
    df: pd.DataFrame, 
    category: str, 
    menu_mp: Union[int, str], 
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]], List[str]]:
    """
    Initialize validation tracking for cost calculations.
    
    Sets up common validation elements needed for all calculation functions:
    - Creates a copy of the input DataFrame
    - Determines which homes have valid data
    - Initializes column tracking dictionaries
    
    Args:
        df: DataFrame containing the data to validate
        category: Equipment category (e.g., 'heating', 'waterHeating')
        menu_mp: Measure package identifier (0 for baseline, nonzero for measure packages)
        verbose: Whether to print validation information
        
    Returns:
        Tuple containing:
        - df_copy: Copy of input DataFrame
        - valid_mask: Boolean Series indicating valid homes
        - all_columns_to_mask: Dictionary to track columns by category
        - category_columns_to_mask: List to track columns for this category
    """
    # Create a copy of the original DataFrame to avoid modifying it directly
    df_copy = df.copy()
    
    # Initialize dictionary to track columns for masking verification
    all_columns_to_mask = {cat: [] for cat in EQUIPMENT_SPECS}
    category_columns_to_mask = []
    
    # Determine which homes have valid data for this category
    valid_mask = get_valid_calculation_mask(df_copy, category, menu_mp, verbose=verbose)
    
    return df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask

def get_valid_fuel_types(category: str) -> List[str]:
    """
    Returns the list of valid fuel types for a category.
    
    Args:
        category: Equipment category name.
        
    Returns:
        List of valid fuel type strings for the specified category.
        
    Raises:
        ValueError: If an invalid category is provided.
    """
    # Tech filters handle excluding heat pump technologies for heating and water heating
    # So we can keep electricity as a valid fuel type.
    if category in ['heating', 'waterHeating']:
        return ['Electricity', 'Natural Gas', 'Propane', 'Fuel Oil']
    
    # Heat pump clothes dryers are different from existing electric resistance dryers in EUSS.
    # So we can keep electricity as a valid fuel type for clothes drying.
    elif category == 'clothesDrying':
        return ['Electricity', 'Natural Gas', 'Propane']
    
    # We exclude electricity for cooking because the electric upgrade in MP7 is the same technology.
    elif category == 'cooking':
        return ['Natural Gas', 'Propane']
    
    else:
        raise ValueError(f"Invalid category. Must be one of the following: {EQUIPMENT_SPECS.keys()}")

def get_valid_calculation_mask(
    df: pd.DataFrame, 
    category: str, 
    menu_mp: Union[int, str] = 0,
    verbose: bool = True
) -> pd.Series:
    """
    Combines data validation and retrofit status for comprehensive masking.
    
    This function addresses a key integration issue between the data validation
    system and the retrofit status tracking system. It ensures:
    - For baseline scenarios: Only homes with valid data are processed
    - For measure packages: Only homes with both valid data AND scheduled for retrofits are processed
    
    Args:
        df: DataFrame containing the validation flags and retrofit information.
        category: Equipment category (e.g., 'heating', 'waterHeating').
        menu_mp: Measure package identifier (0 for baseline, nonzero for measure packages).
        verbose: Whether to print information about valid homes.
        
    Returns:
        Series of boolean values indicating which homes should be included in calculations.
        
    Raises:
        ValueError: If the inclusion flag for the given category doesn't exist in the DataFrame.
    """
    # Standardize menu_mp to facilitate comparisons
    menu_mp_str = str(menu_mp)
    is_baseline = menu_mp_str == "0"
    
    # Check if inclusion flag exists
    include_col = f'include_{category}'
    if include_col not in df.columns:
        raise ValueError(f"Inclusion flag '{include_col}' not found in DataFrame. "
                         f"Ensure identify_valid_homes() has been called.")
    
    # Get data validation mask
    data_valid_mask = df[include_col]
    
    # For baseline scenarios, only use data validation
    if is_baseline:
        if verbose:
            valid_count = data_valid_mask.sum()
            invalid_count = (~data_valid_mask).sum()
            print(f"Baseline calculation for {category}:")
            print(f"  - {valid_count} homes have valid data")
            print(f"  - {invalid_count} homes have invalid data (values will be NaN)")
        
        return data_valid_mask
    
    # For measure packages, combine with retrofit status
    else:
        retrofit_mask = get_retrofit_homes_mask(df, category, menu_mp, verbose=False)
        combined_mask = data_valid_mask & retrofit_mask
        
        if verbose:
            valid_data_count = data_valid_mask.sum()
            retrofit_count = retrofit_mask.sum()
            final_count = combined_mask.sum()
            
            print(f"Measure package calculation for {category}:")
            print(f"  - {valid_data_count} homes have valid baseline data")
            print(f"  - {retrofit_count} homes will receive retrofits")
            print(f"  - {final_count} homes have both valid data AND will receive retrofits")
            print(f"  - {len(df) - final_count} homes excluded (values will be NaN)")
        
        # Check if all homes are excluded
        if combined_mask.sum() == 0:
            print(f"WARNING: All homes excluded for {category}. Check data quality and retrofit criteria.")
        
        return combined_mask

def get_retrofit_homes_mask(
    df: pd.DataFrame, 
    category: str, 
    menu_mp: Union[int, str], 
    verbose: bool = True
) -> pd.Series:
    """
    Determine which homes will receive retrofits for a given category.
    
    Args:
        df: DataFrame containing the upgrade columns.
        category: Equipment category (e.g., 'heating', 'waterHeating').
        menu_mp: Measure package identifier (0 for baseline, nonzero for measure packages).
        verbose: Whether to print information about retrofitted homes.
        
    Returns:
        Series of boolean values indicating which homes get retrofits (True = retrofit, False = no retrofit).
    """
    # Standardize menu_mp to facilitate comparisons
    menu_mp_str = str(menu_mp)
    is_baseline = menu_mp_str == "0"
    
    # Get the upgrade column for this category
    upgrade_col = UPGRADE_COLUMNS.get(category)
    
    # For baseline calculations, process all homes
    if is_baseline:
        retrofit_mask = pd.Series(True, index=df.index)
        return retrofit_mask
        
    # For measure packages, check if the upgrade column exists
    if upgrade_col and upgrade_col in df.columns:
        retrofit_mask = df[upgrade_col].notna()
        
        if verbose:
            num_retrofits = retrofit_mask.sum()
            num_no_retrofits = (~retrofit_mask).sum()
            print(f"  Using '{upgrade_col}' column:")
            print(f"  - {num_retrofits} homes will receive {category} retrofits.")
            print(f"  - {num_no_retrofits} homes will NOT receive retrofits (values will be NaN).")
    else:
        # If no upgrade column exists, assume all homes get retrofits
        retrofit_mask = pd.Series(True, index=df.index)
        
        if verbose:
            print(f"  WARNING: No upgrade column found for '{category}'.")
            print(f"  Assuming all homes receive retrofits for this category.")
            
    return retrofit_mask

# ====================================================================================================
# STEP 2: SERIES INITIALIZATION
# ====================================================================================================

def create_retrofit_only_series(
    df: pd.DataFrame,
    retrofit_mask: Optional[pd.Series] = None,
    category: Optional[str] = None,
    menu_mp: Optional[Union[int, str]] = None,
    verbose: bool = False
) -> pd.Series:
    """
    Initialize a Series with zeros for homes getting retrofits, NaN for others.
    
    This function can work in two ways:
    1. Pass a pre-computed retrofit_mask
    2. Pass category and menu_mp to have it determine the retrofit mask
    
    Args:
        df: DataFrame containing the upgrade columns and index.
        retrofit_mask: Optional pre-computed retrofit mask. If provided, category and menu_mp are ignored.
        category: Equipment category (e.g., 'heating', 'waterHeating'). Required if retrofit_mask is not provided.
        menu_mp: Measure package identifier. Required if retrofit_mask is not provided.
        verbose: Whether to print information about retrofitted homes.
        
    Returns:
        Series initialized with zeros for homes getting retrofits and NaN for others.
        
    Raises:
        ValueError: If retrofit_mask is not provided and either category or menu_mp is None.
    """
    # If no mask is provided, determine it using the helper function
    if retrofit_mask is None:
        if category is None or menu_mp is None:
            raise ValueError("Either retrofit_mask must be provided or both category and menu_mp")
        retrofit_mask = get_retrofit_homes_mask(df, category, menu_mp, verbose)
    
    # Initialize series with NaN for all homes
    result = pd.Series(np.nan, index=df.index)
    
    # Set 0.0 for homes that will be retrofitted
    result.loc[retrofit_mask] = 0.0
    
    return result

# ====================================================================================================
# STEP 5: FINAL MASKING
# ====================================================================================================

def apply_final_masking(
    df: pd.DataFrame, 
    all_columns_to_mask: Dict[str, List[str]], 
    verbose: bool = True
) -> pd.DataFrame:
    """
    Apply final verification masking for all tracked columns.
    
    Ensures data consistency by applying masking to all tracked columns
    based on respective category inclusion flags.
    
    Args:
        df: DataFrame containing the calculated data
        all_columns_to_mask: Dictionary mapping categories to lists of column names
        verbose: Whether to print masking information
        
    Returns:
        DataFrame with masking applied to invalid data
    """
    print("\nVerifying masking for all calculated columns:")
    for category, cols_to_mask in all_columns_to_mask.items():
        # Filter out columns that don't exist in df
        cols_to_mask = [col for col in cols_to_mask if col in df.columns]
        
        if cols_to_mask:
            df = mask_category_specific_data(df, cols_to_mask, category, verbose=verbose)
    
    return df

def mask_category_specific_data(
        df: pd.DataFrame, 
        columns: List[str], 
        category: str,
        verbose: bool = False) -> pd.DataFrame:
    """
    Applies NaN masking to specified columns based on a category's inclusion flag.
    
    This utility function applies NaN masking to all provided columns based 
    on the inclusion flag for the specified category. It can be used anywhere
    in the codebase after calculations to ensure data quality.
    
    Args:
        df: DataFrame with inclusion flags already created.
        columns: List of column names to apply masking to.
        category: The equipment category that determines which inclusion flag to use.
        verbose: Whether to print details about masking operations.
        
    Returns:
        DataFrame with specified columns masked based on the category's inclusion flag.
        
    Raises:
        ValueError: If the category's inclusion flag is not found in the DataFrame.
    """
    include_col = f'include_{category}'
    
    if include_col not in df.columns:
        raise ValueError(f"Inclusion flag '{include_col}' not found in DataFrame")
        
    # Filter out columns that don't exist in the DataFrame
    valid_columns = [col for col in columns if col in df.columns]
    
    if not valid_columns:
        if verbose:
            print(f"No valid columns to mask for category '{category}'")
        return df
        
    if verbose:
        print(f"Masking {len(valid_columns)} columns for category '{category}'")
        
    masked_count = 0
    df_copy = df.copy()
    
    for col in valid_columns:
        # Count non-NaN values before masking
        non_nan_before = df_copy[col].notna().sum()
        
        # Apply masking
        df_copy.loc[~df_copy[include_col], col] = np.nan
        
        # Count non-NaN values after masking
        non_nan_after = df_copy[col].notna().sum()
        
        # Handle different types safely
        try:
            # Try direct conversion for scalar values
            masked_this_col = int(non_nan_before - non_nan_after)
        except TypeError:
            # If we got a Series, take its sum
            if isinstance(non_nan_before - non_nan_after, pd.Series):
                masked_this_col = int((non_nan_before - non_nan_after).sum())
            else:
                # For other types, try a more robust approach
                masked_this_col = int(float(non_nan_before - non_nan_after))
        
        # Now masked_this_col is guaranteed to be a scalar
        if masked_this_col > 0 and verbose:
            print(f"    {col}: Masked {masked_this_col} values")
            masked_count += masked_this_col
            
    if verbose and masked_count > 0:
        print(f"  Total: Masked {masked_count} values across {len(valid_columns)} columns")
        
    return df_copy

# ====================================================================================================
# HELPER FUNCTIONS
# ====================================================================================================

def apply_new_columns_to_dataframe(
    df_original: pd.DataFrame,
    df_new_columns: pd.DataFrame,
    category: str,
    category_columns_to_mask: List[str],
    all_columns_to_mask: Dict[str, List[str]]
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Apply new columns to a DataFrame with proper tracking for validation.
    
    This utility function handles common tasks when adding new calculated columns:
    - Tracks columns for validation
    - Handles overlapping columns
    - Joins new columns to the original DataFrame
    
    Args:
        df_original: Original DataFrame to update
        df_new_columns: DataFrame containing new columns to add
        category: Category for tracking purposes
        category_columns_to_mask: List of columns to track for this category
        all_columns_to_mask: Master dictionary of columns to track by category
        
    Returns:
        Tuple containing:
        - Updated DataFrame with new columns
        - Updated all_columns_to_mask dictionary
    """
    # Track columns for masking
    category_columns_to_mask.extend(df_new_columns.columns.tolist())
    all_columns_to_mask[category].extend(category_columns_to_mask)
    
    # Identify overlapping columns to avoid duplicates
    overlapping_columns = df_new_columns.columns.intersection(df_original.columns)
    
    # Drop overlapping columns from the original DataFrame
    if not overlapping_columns.empty:
        df_original.drop(columns=overlapping_columns, inplace=True)
    
    # Merge new columns into the original DataFrame using join
    df_updated = df_original.join(df_new_columns, how='left')
    
    return df_updated, all_columns_to_mask

def replace_small_values_with_nan(
    series_or_dict: Union[pd.Series, pd.DataFrame, Dict[Any, pd.Series]], 
    threshold: float = 1e-10
) -> Union[pd.Series, pd.DataFrame, Dict[Any, pd.Series]]:
    """
    Replace values close to zero with NaN to avoid numerical artifacts.
    
    Args:
        series_or_dict: A pandas Series, DataFrame, or dictionary of Series.
        threshold: Values with absolute value below this threshold will be replaced with NaN.
        
    Returns:
        The input with small values replaced by NaN.
        
    Raises:
        TypeError: If input is not a pandas Series, DataFrame, or dictionary of Series.
    """
    if isinstance(series_or_dict, pd.Series):
        return series_or_dict.where(abs(series_or_dict) > threshold, np.nan)
    elif isinstance(series_or_dict, pd.DataFrame):
        return series_or_dict.where(abs(series_or_dict) > threshold, np.nan)
    elif isinstance(series_or_dict, dict):
        return {k: replace_small_values_with_nan(v, threshold) for k, v in series_or_dict.items()}
    else:
        raise TypeError("Input must be a pandas Series, DataFrame, or dictionary of Series")

def calculate_avoided_values(
    baseline_values: pd.Series,
    measure_values: pd.Series,
    retrofit_mask: pd.Series
) -> pd.Series:
    """
    Calculate avoided values (baseline - measure) only for retrofitted homes.
    
    Args:
        baseline_values: Series of baseline values.
        measure_values: Series of measure package values.
        retrofit_mask: Boolean Series indicating which homes get retrofits.
        
    Returns:
        Series with avoided values for retrofitted homes and NaN for others.
    """
    # Initialize with NaN
    avoided_values = pd.Series(np.nan, index=baseline_values.index)
    
    # Calculate only for homes with retrofits
    if retrofit_mask.any():
        avoided_values.loc[retrofit_mask] = (
            baseline_values.loc[retrofit_mask] - measure_values.loc[retrofit_mask]
        )
        
    return avoided_values
