"""
Discount rate utilities for net present value calculations.

Supports three discounting methods:
- public: Fixed 2% social discount rate
- private_fixed: Fixed 7% private discount rate  
- private_variable: Household-specific rate based on AMI (2% to 12%)
"""
import numpy as np
import pandas as pd

from cmu_tare_model.constants import (
    PUBLIC_DISCOUNT_RATE, PRIVATE_FIXED_RATE_LOW, PRIVATE_FIXED_RATE_BASE, PRIVATE_FIXED_RATE_HIGH,
    VARIABLE_RATE_MIN, VARIABLE_RATE_MAX, AMI_THRESHOLD, VERBOSE
)

# Method suffixes for fixed and variable discounting
PRIVATE_DISCOUNTING_METHOD_SUFFIXES = {
    'private_discount_rate_fixed_low': '_fixed_low',
    'private_discount_rate_fixed_base': '_fixed_base',
    'private_discount_rate_fixed_high': '_fixed_high',
    'private_discount_rate_variable': '_variable'
    }

PRIVATE_DISCOUNT_RATE_COLS = [
    'private_discount_rate_fixed_low',
    'private_discount_rate_fixed_base',
    'private_discount_rate_fixed_high',
    'private_discount_rate_variable'
]

PUBLIC_DISCOUNTING_METHOD_SUFFIXES = {
    'public_discount_rate': ''
}

def prepare_discount_rates(
    df: pd.DataFrame,
    verbose: bool = VERBOSE
) -> pd.DataFrame:
    """
    Prepare all discount rate columns in a single operation.
    
    Creates five columns:
    - 'public_discount_rate': 2% constant (for social/public NPV)
    - 'private_discount_rate_fixed_low': 2% constant (for private NPV)
    - 'private_discount_rate_fixed_base': 7% constant (for private NPV)
    - 'private_discount_rate_fixed_high': 12% constant (for private NPV)
    - 'private_discount_rate_variable': AMI-based 2%-12% (for equity-adjusted private NPV)
    
    Args:
        df: DataFrame containing 'percent_AMI' column for variable rate calculation.
        verbose: Enable detailed output.
    
    Returns:
        DataFrame with five additional discount rate columns.
        
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
    df_copy['private_discount_rate_fixed_low'] = PRIVATE_FIXED_RATE_LOW  # 0.02 (2%)
    df_copy['private_discount_rate_fixed_base'] = PRIVATE_FIXED_RATE_BASE  # 0.07 (7%)
    df_copy['private_discount_rate_fixed_high'] = PRIVATE_FIXED_RATE_HIGH  # 0.12 (12%)

    # Variable rate: AMI-based calculation
    percent_ami = df_copy['percent_AMI']

    # Define the interpolation bounds
    ami_x_bounds = [0, AMI_THRESHOLD]                       # x-coordinates: 0% to 150% AMI
    rate_y_bounds = [VARIABLE_RATE_MAX, VARIABLE_RATE_MIN]  # y-coordinates: 12% down to 2%

    # LINEAR INTERPOLATION: AMI → Discount Rate (INVERSE relationship)
    # Values below 0% AMI get 12%, above 150% AMI get 2%.
    # Clamping behavior is built into np.interp
    variable_rate = np.interp(x=percent_ami, xp=ami_x_bounds, fp=rate_y_bounds)
    df_copy['private_discount_rate_variable'] = variable_rate
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"DISCOUNT RATE DIAGNOSTIC: Calculating for BOTH private discount methods")
        print(f"{'='*80}")
        
        # ===== PUBLIC Discount Rate Diagnostic =====
        public_rate_fixed = df_copy['public_discount_rate'].iloc[0]
        print(f"Public Rate: {public_rate_fixed:.1%} (constant across all households)")

        # ===== PRIVATE Discount Rate Diagnostic =====
        # Private fixed rate diagnostic
        private_rate_fixed_low = df_copy['private_discount_rate_fixed_low'].iloc[0]
        print(f"Private Fixed Rate (Low): {private_rate_fixed_low:.1%} (constant across all households)")
        
        private_rate_fixed_base = df_copy['private_discount_rate_fixed_base'].iloc[0]
        print(f"Private Fixed Rate (Base): {private_rate_fixed_base:.1%} (constant across all households)")

        private_rate_fixed_high = df_copy['private_discount_rate_fixed_high'].iloc[0]
        print(f"Private Fixed Rate (High): {private_rate_fixed_high:.1%} (constant across all households)")
        
        # Private variable rate diagnostic  
        private_rate_variable = df_copy['private_discount_rate_variable']
        print(f"Private Variable Rate (AMI-based):")
        print(f"  Minimum: {private_rate_variable.min():.1%} (highest AMI)")
        print(f"  Median:  {private_rate_variable.median():.1%}")
        print(f"  Maximum: {private_rate_variable.max():.1%} (lowest AMI)")
        
        print(f"{'='*80}\n")

    return df_copy


def calculate_discount_factors(
    df: pd.DataFrame,
    base_year: int,
    target_year: int,
    discount_rate_col: str
) -> pd.Series:
    """
    Calculate discount factors for all households using pre-prepared rates.
    
    ALWAYS returns Series (never scalar) for consistency.
    
    Args:
        df: DataFrame with discount rate columns already prepared.
        base_year: Reference year to discount to (e.g., 2024).
        target_year: Future year to discount from (e.g., 2030).
        discount_rate_col: Column name in df containing the discount rates to use.
        
    Returns:
        Series of discount factors (one per household).
        
    Raises:
        ValueError: If discount_rate_col is missing in the DataFrame.
    """
    if discount_rate_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain '{discount_rate_col}' column. "
            f"Call prepare_discount_rates(df) first to create it."
        )
    
    # Get discount rates (ALWAYS a Series)
    discount_rate = df[discount_rate_col]
    
    # Calculate time difference
    years_difference = max(0, target_year - base_year)
    
    # Apply discount formula element-wise: PV = FV / (1+r)^t
    discount_factors = 1 / ((1 + discount_rate) ** years_difference)

    return discount_factors
