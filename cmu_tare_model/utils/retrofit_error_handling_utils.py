import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any

from cmu_tare_model.constants import UPGRADE_COLUMNS

def determine_retrofit_status(
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
    # Get the upgrade column for this category
    upgrade_col = UPGRADE_COLUMNS.get(category)
    
    # For baseline calculations (menu_mp = 0 or "0"), process all homes
    if menu_mp == 0 or menu_mp == "0":
        retrofit_mask = pd.Series(True, index=df.index)
        return retrofit_mask
        
    # For measure packages, check if the upgrade column exists
    if upgrade_col and upgrade_col in df.columns:
        retrofit_mask = df[upgrade_col].astype(bool)
        
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


def initialize_npv_series(
    df: pd.DataFrame,
    retrofit_mask: Optional[pd.Series] = None,
    category: Optional[str] = None,
    menu_mp: Optional[Union[int, str]] = None,
    verbose: bool = False
) -> pd.Series:
    """
    Initialize an NPV Series with zeros for homes getting retrofits, NaN for others.
    
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
        retrofit_mask = determine_retrofit_status(df, category, menu_mp, verbose)
    
    # Initialize series with NaN for all homes
    result = pd.Series(np.nan, index=df.index)
    
    # Set 0.0 for homes that will be retrofitted
    result.loc[retrofit_mask] = 0.0
    
    return result


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


def update_values_for_retrofits(
    target_series: pd.Series,
    update_values: pd.Series,
    retrofit_mask: pd.Series,
    menu_mp: Union[int, str]
) -> pd.Series:
    """
    Update values only for homes with retrofits if this is a measure package.
    
    Args:
        target_series: Series to update.
        update_values: Values to add.
        retrofit_mask: Boolean Series indicating which homes get retrofits.
        menu_mp: Measure package identifier (0 for baseline).
        
    Returns:
        Updated Series.
    """
    # Convert menu_mp to appropriate type for comparison
    mp_value = 0 if menu_mp == "0" else menu_mp
    
    if mp_value != 0:
        # Only update retrofitted homes
        target_series.loc[retrofit_mask] += update_values.loc[retrofit_mask].fillna(0)
    else:
        # Update all homes for baseline calculations
        target_series += update_values.fillna(0)
        
    return target_series
