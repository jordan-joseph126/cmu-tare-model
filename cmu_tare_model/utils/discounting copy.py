"""
Discount rate utilities for net present value calculations.

Supports three discounting methods:
- public: Fixed 2% social discount rate
- private_fixed: Fixed 7% private discount rate  
- private_variable: Household-specific rate based on AMI (2% to 12%)
"""
import numpy as np
import pandas as pd
from typing import Union

from cmu_tare_model.constants import PUBLIC_DISCOUNT_RATE, PRIVATE_FIXED_RATE, VARIABLE_RATE_MIN, VARIABLE_RATE_MAX, AMI_THRESHOLD

def calculate_discount_factor(
    base_year: int, 
    target_year: int, 
    discounting_method: str
) -> float:
    """
    Calculate a scalar discount factor for fixed-rate discounting methods.
    Formula: PV = FV / (1+r)^t

    Args:
        base_year (int): The reference year to discount to (e.g., 2024).
        target_year (int): The future year to discount from (e.g., 2030).
        discounting_method (str): Either 'public' (2%, 0.02) or 'private_fixed' (7%, 0.07).
        
    Returns:
        discount_factor (float): Scalar discount factor to multiply with future values.
        
    Raises:
        ValueError: If discounting_method is invalid or 'private_variable'
                   (which requires the household-specific function).
    
    Example:
        >>> calculate_discount_factor(2024, 2030, 'public')
        0.8879...  # 1 / (1.02)^6
    """
    if discounting_method == 'public':
        discount_rate = PUBLIC_DISCOUNT_RATE
    elif discounting_method == 'private_fixed':
        discount_rate = PRIVATE_FIXED_RATE
    elif discounting_method == 'private_variable':
        raise ValueError(
            "For 'private_variable' discounting, use calculate_variable_discount_factors() "
            "which returns household-specific discount factors as a Series."
        )
    else:
        raise ValueError(
            f"Invalid discounting method: '{discounting_method}'. "
            "Must be 'public', 'private_fixed', or 'private_variable'."
        )
    
    # Cannot have negative years (future must be >= base year)
    years_difference = max(0, target_year - base_year)

    # Formula: PV = FV / (1+r)^t
    discount_factor = 1 / ((1 + discount_rate) ** years_difference)
    
    # Return scalar discount factor
    return discount_factor


def calculate_variable_discount_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate household-specific discount rates based on AMI percentage.
    
    Variable discount rate is inversely proportional to AMI:
    - Values below 0% AMI get 12%, above 150% AMI get 2%.
    - At 0% AMI: 12% discount rate (highest, reflecting financial constraints)
    - At 150%+ AMI: 2% discount rate (lowest, same as public rate)
    - Linear interpolation between these bounds. Clamping behavior is built into np.interp.

    Args:
        df (pd.DataFrame): DataFrame containing 'percent_AMI' column with household AMI percentages.
    
    Returns:
        pd.DataFrame: DataFrame with additional column 'household_variable_discount_rate'.
        
    Raises:
        ValueError: If 'percent_AMI' column is missing from DataFrame.
        
    Example:
        >>> df = pd.DataFrame({'percent_AMI': [0, 75, 150, 200]})
        >>> result = calculate_variable_discount_rate(df)
        >>> result['household_variable_discount_rate'].tolist()
        [0.12, 0.07, 0.02, 0.02]  # Linear interpolation, capped at 150% AMI
    """
    if 'percent_AMI' not in df.columns:
        raise ValueError(
            "DataFrame must contain 'percent_AMI' column. "
            f"Available columns: {list(df.columns)}"
        )

    df_copy = df.copy()
    percent_ami = df_copy['percent_AMI']
    
    # Define the interpolation bounds
    ami_x_bounds = [0, AMI_THRESHOLD]                       # x-coordinates: 0% to 150% AMI
    rate_y_bounds = [VARIABLE_RATE_MAX, VARIABLE_RATE_MIN]  # y-coordinates: 12% down to 2%
    
    # LINEAR INTERPOLATION: AMI → Discount Rate (INVERSE relationship)
    # Values below 0% AMI get 12%, above 150% AMI get 2%.
    # Clamping behavior is built into np.interp
    discount_rate = np.interp(x=percent_ami, xp=ami_x_bounds, fp=rate_y_bounds)
    
    df_copy['household_variable_discount_rate'] = discount_rate
    
    return df_copy


def calculate_variable_discount_factors(
    df: pd.DataFrame,
    base_year: int,
    target_year: int
) -> pd.Series:
    """
    Calculate household-specific discount factors for variable-rate discounting.
    
    This function applies each household's individual discount rate to compute
    their discount factor for a specific year. Must be called after 
    calculate_variable_discount_rate() has added the rate column.
    
    Args:
        df: DataFrame with 'household_variable_discount_rate' column.
        base_year: The reference year to discount to (e.g., 2024).
        target_year: The future year to discount from.
        
    Returns:
        Series of discount factors, one per household (same index as input df).
        
    Raises:
        ValueError: If 'household_variable_discount_rate' column is missing.
        
    Example:
        >>> df = pd.DataFrame({
        ...     'percent_AMI': [0, 150],
        ...     'household_variable_discount_rate': [0.12, 0.02]
        ... })
        >>> factors = calculate_variable_discount_factors(df, 2024, 2030)
        >>> factors.round(4).tolist()
        [0.5066, 0.8880]  # Low-AMI household discounted more heavily
    """
    rate_col = 'household_variable_discount_rate'
    
    if rate_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{rate_col}' column. "
            "Call calculate_variable_discount_rate(df) first to create it."
        )
    
    discount_rate = df[rate_col]
    years_difference = max(0, target_year - base_year)
    
    # Same formula as scalar version, but applied element-wise to Series
    discount_factors = 1 / ((1 + discount_rate) ** years_difference)
    
    # Return Series of discount factors because household uses a different rate
    return discount_factors
