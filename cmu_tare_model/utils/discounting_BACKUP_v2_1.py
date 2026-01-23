"""
Discount rate utilities for net present value calculations.

Supports three discounting methods:
- public: Fixed 2% social discount rate
- private_fixed: Fixed 7% private discount rate  
- private_variable: Household-specific rate based on AMI (2% to 12%)
"""
import numpy as np
import pandas as pd

from cmu_tare_model.constants import PUBLIC_DISCOUNT_RATE, PRIVATE_FIXED_RATE, VARIABLE_RATE_MIN, VARIABLE_RATE_MAX, AMI_THRESHOLD

def prepare_discount_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare all discount rate columns in a single operation.
    
    Creates three columns:
    - 'public_discount_rate': 2% constant (for social/public NPV)
    - 'private_fixed_discount_rate': 7% constant (for private NPV)
    - 'private_variable_discount_rate': AMI-based 2%-12% (for equity-adjusted private NPV)
    
    Args:
        df: DataFrame containing 'percent_AMI' column for variable rate calculation.
    
    Returns:
        DataFrame with three additional discount rate columns.
        
    Raises:
        ValueError: If 'percent_AMI' column is missing.
    """    
    if 'percent_AMI' not in df.columns:
        raise ValueError(
            "DataFrame must contain 'percent_AMI' column for variable rate calculation. "
            f"Available columns: {list(df.columns)}"
        )

    df_copy = df.copy()
    
    # Fixed rates: simple broadcasting
    df_copy['public_discount_rate'] = PUBLIC_DISCOUNT_RATE  # 0.02 (2%)
    df_copy['private_fixed_discount_rate'] = PRIVATE_FIXED_RATE  # 0.07 (7%)
    
    # Variable rate: AMI-based calculation
    percent_ami = df_copy['percent_AMI']

    # Define the interpolation bounds
    ami_x_bounds = [0, AMI_THRESHOLD]                       # x-coordinates: 0% to 150% AMI
    rate_y_bounds = [VARIABLE_RATE_MAX, VARIABLE_RATE_MIN]  # y-coordinates: 12% down to 2%

    # LINEAR INTERPOLATION: AMI → Discount Rate (INVERSE relationship)
    # Values below 0% AMI get 12%, above 150% AMI get 2%.
    # Clamping behavior is built into np.interp
    variable_rate = np.interp(x=percent_ami, xp=ami_x_bounds, fp=rate_y_bounds)
    df_copy['private_variable_discount_rate'] = variable_rate
    
    return df_copy


def calculate_discount_factors(
    df: pd.DataFrame,
    base_year: int,
    target_year: int,
    discounting_method: str
) -> pd.Series:
    """
    Calculate discount factors for all households using pre-prepared rates.
    
    ALWAYS returns Series (never scalar) for consistency.
    
    Args:
        df: DataFrame with discount rate columns already prepared.
        base_year: Reference year to discount to (e.g., 2024).
        target_year: Future year to discount from (e.g., 2030).
        discounting_method: 'public', 'private_fixed', or 'private_variable'.
        
    Returns:
        Series of discount factors (one per household).
        
    Raises:
        ValueError: If discounting_method is invalid or rate column missing.
    """
    # Map method to column name
    rate_column_map = {
        'public': 'public_discount_rate',
        'private_fixed': 'private_fixed_discount_rate',
        'private_variable': 'private_variable_discount_rate'
    }
    
    if discounting_method not in rate_column_map:
        valid_methods = list(rate_column_map.keys())
        raise ValueError(
            f"Invalid discounting method: '{discounting_method}'. "
            f"Must be one of {valid_methods}"
        )
    
    rate_col = rate_column_map[discounting_method]
    
    if rate_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{rate_col}' column. "
            f"Call prepare_discount_rates(df) first to create it."
        )
    
    # Get discount rates (ALWAYS a Series)
    discount_rate = df[rate_col]
    
    # Calculate time difference
    years_difference = max(0, target_year - base_year)
    
    # Apply discount formula element-wise: PV = FV / (1+r)^t
    discount_factors = 1 / ((1 + discount_rate) ** years_difference)
    
    return discount_factors
