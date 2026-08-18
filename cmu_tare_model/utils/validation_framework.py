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

from cmu_tare_model.constants import EQUIPMENT_SPECS, UPGRADE_COLUMNS, VERBOSE

# ====================================================================================================
# STEP 1: MASK INITIALIZATION
# ====================================================================================================

def initialize_validation_tracking(
    df: pd.DataFrame,
    category: str,
    menu_mp: Union[int, str],
    verbose: bool = VERBOSE,
    copy: bool = False
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]], List[str]]:
    """
    Initialize validation tracking for cost calculations.

    Sets up common validation elements needed for all calculation functions:
    - Optionally creates a copy of the input DataFrame (disabled by default for memory efficiency)
    - Determines which homes have valid data
    - Initializes column tracking dictionaries

    MEMORY OPTIMIZATION: By default, this function no longer creates a copy of the DataFrame.
    The caller is responsible for creating a copy if needed before calling this function.
    Set copy=True only when you need to modify the DataFrame and want to preserve the original.

    Args:
        df: DataFrame containing the data to validate
        category: Equipment category (e.g., 'heating', 'waterHeating')
        menu_mp: Measure package identifier (0 for baseline, nonzero for measure packages)
        verbose: Whether to print validation information
        copy: Whether to create a copy of the DataFrame (default: False for memory efficiency)

    Returns:
        Tuple containing:
        - df_ref: Reference to input DataFrame (or copy if copy=True)
        - valid_mask: Boolean Series indicating valid homes
        - all_columns_to_mask: Dictionary to track columns by category
        - category_columns_to_mask: List to track columns for this category
    """
    # Only create a copy if explicitly requested (memory optimization)
    df_ref = df.copy() if copy else df

    # Initialize dictionary to track columns for masking verification
    all_columns_to_mask = {cat: [] for cat in EQUIPMENT_SPECS}
    category_columns_to_mask = []

    # Determine which homes have valid data for this category
    valid_mask = get_valid_calculation_mask(df_ref, category, menu_mp, verbose=verbose)

    return df_ref, valid_mask, all_columns_to_mask, category_columns_to_mask


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
    
    # UPDATED (Expanded to include cooling): Include electricity for cooling
    elif category == 'cooling':
        return ['Electricity']

    else:
        raise ValueError(f"Invalid category. Must be one of the following: {EQUIPMENT_SPECS.keys()}")


def get_valid_calculation_mask(
    df: pd.DataFrame, 
    category: str, 
    menu_mp: Union[int, str] = 0,
    verbose: bool = VERBOSE
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
    menu_mp_str = str(menu_mp).lower()
    is_baseline = menu_mp_str == "0" or menu_mp_str == "baseline"
        
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
        retrofit_mask = get_retrofit_homes_mask(df, category, menu_mp, verbose=verbose)
        combined_mask = data_valid_mask & retrofit_mask
        
        if verbose:
            valid_data_count = data_valid_mask.sum()
            retrofit_count = retrofit_mask.sum()
            final_count = combined_mask.sum()
            
            print(f"Measure package calculation for {category}:")
            print(f"  - {valid_data_count} homes have valid data")
            print(f"  - {retrofit_count} homes will receive retrofits")
            print(f"  - {final_count} homes have both valid data AND will receive retrofits")
            print(f"  - {len(df) - final_count} homes excluded (values will be NaN)")
        
        # Check if all homes are excluded. Keep this one - critical information
        if combined_mask.sum() == 0:
            raise ValueError(f"WARNING: All homes excluded for {category}. Check data quality and retrofit criteria.")
        
        return combined_mask
    

def get_retrofit_homes_mask(
    df: pd.DataFrame, 
    category: str, 
    menu_mp: Union[int, str], 
    verbose: bool = VERBOSE
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
            if category != 'cooling':  # Suppress for cooling as per discussion
                raise ValueError(f"WARNING: No upgrade column found for '{category}'. \
                                Assuming all homes receive retrofits for this category.")
    return retrofit_mask

# ====================================================================================================
# STEP 2: SERIES INITIALIZATION
# ====================================================================================================

def create_retrofit_only_series(
    df: pd.DataFrame,
    retrofit_mask: Optional[pd.Series] = None,
    category: Optional[str] = None,
    menu_mp: Optional[Union[int, str]] = None,
    verbose: bool = VERBOSE
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
    
    # # Initialize series with NaN for all homes
    # result = pd.Series(np.nan, index=df.index)
    
    # # Set 0.0 for homes that will be retrofitted
    # result.loc[retrofit_mask] = 0.0

    # OPTIMIZED: Use np.where for fully vectorized initialization
    result = pd.Series(
        np.where(retrofit_mask, 0.0, np.nan),
        index=df.index
    )

    return result

# ====================================================================================================
# STEP 5: FINAL MASKING
# ====================================================================================================

def apply_final_masking(
    df: pd.DataFrame, 
    all_columns_to_mask: Dict[str, List[str]], 
    verbose: bool = VERBOSE
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
    if verbose:
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
        verbose: bool = VERBOSE,
        inplace: bool = True) -> pd.DataFrame:
    """
    Applies NaN masking to specified columns based on a category's inclusion flag.

    This utility function applies NaN masking to all provided columns based
    on the inclusion flag for the specified category. It can be used anywhere
    in the codebase after calculations to ensure data quality.

    MEMORY OPTIMIZATION: By default, this function modifies the DataFrame in-place
    to avoid creating unnecessary copies of large DataFrames.

    Args:
        df: DataFrame with inclusion flags already created.
        columns: List of column names to apply masking to.
        category: The equipment category that determines which inclusion flag to use.
        verbose: Whether to print details about masking operations.
        inplace: Whether to modify the DataFrame in-place (default: True for memory efficiency).
                 Set to False only when you need to preserve the original DataFrame.

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
    # Only create a copy if not operating in-place (memory optimization)
    df_result = df if inplace else df.copy()

    # Pre-compute the mask once for efficiency
    invalid_mask = ~df_result[include_col]

    for col in valid_columns:
        # Count non-NaN values before masking
        non_nan_before = df_result[col].notna().sum()

        # Apply masking using vectorized operation
        df_result.loc[invalid_mask, col] = np.nan

        # Count non-NaN values after masking
        non_nan_after = df_result[col].notna().sum()

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

    return df_result

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
    Replace tiny nonzero values with NaN to avoid numerical artifacts.

    A value is treated as an artifact only if it is nonzero and its magnitude
    is at or below the threshold -- the signature of a rounding remainder that
    should have cancelled to zero. An exact 0.0 is kept as 0.0, because it is a
    real answer: two quantities that genuinely match produce a difference of
    zero, which is different from "no result". Turning those into NaN removed
    real homes from downstream results (see the 12 Aug 2026 changelog).

    Args:
        series_or_dict: A pandas Series, DataFrame, or dictionary of Series.
        threshold: Nonzero values with absolute value at or below this
            threshold will be replaced with NaN.

    Returns:
        The input with tiny nonzero values replaced by NaN, exact zeros kept.

    Raises:
        TypeError: If input is not a pandas Series, DataFrame, or dictionary of Series.
    """
    if isinstance(series_or_dict, pd.Series):
        keep = (series_or_dict == 0) | (abs(series_or_dict) > threshold)
        return series_or_dict.where(keep, np.nan)
    elif isinstance(series_or_dict, pd.DataFrame):
        # Process each column individually and return a new DataFrame
        result_df = pd.DataFrame(index=series_or_dict.index)
        for col in series_or_dict.columns:
            # Apply column-wise replacement directly with the same threshold
            keep_col = (
                (series_or_dict[col] == 0)
                | (abs(series_or_dict[col]) > threshold))
            result_df[col] = series_or_dict[col].where(keep_col, np.nan)
        return result_df
    elif isinstance(series_or_dict, dict):
        return {k: replace_small_values_with_nan(v, threshold) for k, v in series_or_dict.items()}
    else:
        raise TypeError("Input must be a pandas Series, DataFrame, or dictionary of Series")


# UPDATED: NOW HANDLES NONE VALUES FOR RETROFIT_MASK
def calculate_avoided_values(
    baseline_values: pd.Series,
    measure_values: pd.Series,
    retrofit_mask: Optional[pd.Series] = None
) -> pd.Series:
    """
    Calculate avoided values (baseline - measure) only for retrofitted homes.
    
    OPTIMIZED: Uses np.where() for fully vectorized operations instead of 
    .loc[] assignment which is slower due to index alignment overhead.

    Args:
        baseline_values: Series of baseline values.
        measure_values: Series of measure package values.
        retrofit_mask: Boolean Series indicating which homes get retrofits.
                      If None, calculates for all homes.

    Returns:
        Series with avoided values for retrofitted homes and NaN for others.
    """
    # Ensure all series have the same index by aligning to measure_values index
    # This handles cases where baseline data and current data have different indices
    if not baseline_values.index.equals(measure_values.index):
        baseline_values = baseline_values.reindex(measure_values.index)

    # For baseline scenarios (when retrofit_mask is None), calculate for all homes
    if retrofit_mask is None:
        return baseline_values - measure_values
        
    # Align retrofit_mask to measure_values index if needed
    if not retrofit_mask.index.equals(measure_values.index):
        retrofit_mask = retrofit_mask.reindex(measure_values.index, fill_value=False)
    
    # OPTIMIZED: Use np.where for fully vectorized calculation
    # This is faster than creating a NaN series and using .loc[] assignment
    # np.where operates on the underlying numpy arrays directly
    
    # Create combined mask: retrofit homes with valid data in both baseline and measure
    valid_data_mask = baseline_values.notna() & measure_values.notna()
    combined_mask = retrofit_mask & valid_data_mask
    
    # Calculate avoided values using np.where (fully vectorized)
    avoided_values = pd.Series(
        np.where(
            combined_mask,
            baseline_values.values - measure_values.values,
            np.nan
        ),
        index=measure_values.index
    )
    
    return avoided_values

# =====================================================================================================
# OPTIMIZED HELPER: Apply validation mask using np.where (fully vectorized)
# ====================================================================================================
def apply_validation_mask_vectorized(
    values: pd.Series,
    valid_mask: pd.Series,
    menu_mp: int
) -> pd.Series:
    """
    Apply validation mask to a Series using fully vectorized operations.
    
    This helper function replaces the common pattern of:
        values_copy = values.copy()
        if menu_mp != 0:
            values_copy.loc[~valid_mask] = np.nan
    
    With a faster vectorized approach using np.where().
    
    Args:
        values: Series of values to mask.
        valid_mask: Boolean Series indicating valid homes.
        menu_mp: Measure package ID (0=baseline, >0=retrofit).
        
    Returns:
        New Series with NaN for invalid homes (if menu_mp != 0), 
        otherwise a copy of the original values.
    """
    if menu_mp == 0:
        # For baseline, return a copy (matches original .copy() behavior)
        return values.copy()
    
    # For measure packages, apply mask using np.where (fully vectorized)
    # np.where creates a new array, so no explicit copy needed
    return pd.Series(
        np.where(valid_mask, values.values, np.nan),
        index=values.index
    )
