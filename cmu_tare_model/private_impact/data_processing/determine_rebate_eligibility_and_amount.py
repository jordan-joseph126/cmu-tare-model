"""
========================================================================================================
CALCULATE REBATE ELIGIBILITY AND AMOUNTS (REMDB V4 METHODOLOGY)
========================================================================================================
This module calculates IRA rebate eligibility and amounts for residential equipment upgrades.

Key rebate structure:
- Low-Income (≤80% AMI): 100% coverage up to maximum rebate per equipment
- Moderate-Income (>80% to ≤150% AMI): 50% coverage up to maximum rebate
- Middle-to-Upper Income (>150% AMI): No rebate eligibility

Rebate amounts defined in REBATE_MAPPING (constants.py):
- Space conditioning (heating/cooling): $8,000 max (applies once for both)
- Water heating: $1,750 max
- Clothes dryer: $840 max
- Cooking range: $840 max
- Weatherization: $1,600 max (MPs 9-10 only)

PREREQUISITE: Call calculate_upgrade_installed_cost() first to create cost columns.

# UPDATED DECEMBER 16, 2025 - REMDB V4 METHODOLOGY COMPATIBILITY
# UPDATED APRIL 22, 2025 - IMPROVED DOCUMENTATION, MODULARITY, ERROR HANDLING
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple, Union, Callable

from cmu_tare_model.constants import REBATE_MAPPING, VALID_MENU_MPS, EQUIPMENT_SPECS
from cmu_tare_model.utils.inflation_adjustment import cpi_ratio_2023_2022
from cmu_tare_model.utils.validation_framework import (
    create_retrofit_only_series,
    initialize_validation_tracking,
    apply_new_columns_to_dataframe,
    apply_final_masking,
)

from cmu_tare_model.private_impact.data_processing.process_income_data_for_rebates import (
    df_puma_medianIncome,
    df_county_medianIncome, 
    df_state_medianIncome,
    cpi_ratio_2023_2022
)


# ================================================================================================================
# FUNCTIONS: AMI AND INCOME GROUP DESIGNATION FOR REBATE ELIGIBILITY
# ================================================================================================================

def generate_household_medianIncome_2023(row: pd.Series) -> float:
    """
    Generate a household median income value for 2023 using a probabilistic approach.
    
    Samples from a normal distribution based on income range bounds, then
    ensures the final value remains within the valid income range.
    
    Args:
        row: DataFrame row containing income_low, income_high, and income values
        
    Returns:
        float: Generated median income value in 2023 dollars
    """
    # Inflate the income bins to USD 2023 first
    low = row['income_low'] * cpi_ratio_2023_2022
    high = row['income_high'] * cpi_ratio_2023_2022
    mean = row['income'] * cpi_ratio_2023_2022
    
    # Calculate std assuming 10th and 90th percentiles
    std = (high - low) / (norm.ppf(0.90) - norm.ppf(0.10))
    
    # Sample from the normal distribution
    ami_2023 = np.random.normal(loc=mean, scale=std)
    
    # Ensure the generated income is within the bounds
    ami_2023 = max(low, min(high, ami_2023))
    return ami_2023


def fill_na_with_hierarchy(
        df: pd.DataFrame, 
        df_puma: pd.DataFrame, 
        df_county: pd.DataFrame, 
        df_state: pd.DataFrame) -> pd.DataFrame:
    """
    Fills NaN values in 'census_area_medianIncome' using a hierarchical lookup:
    first using the Puma level, then county, and finally state level median incomes.

    Args:
        df: The main DataFrame with NaNs to fill
        df_puma: DataFrame with median incomes at the Puma level
        df_county: DataFrame with median incomes at the county level
        df_state: DataFrame with median incomes at the state level
    
    Returns:
        DataFrame: Modified DataFrame with NaNs filled in 'census_area_medianIncome'
    """
    # First, attempt to fill using Puma-level median incomes
    df['census_area_medianIncome'] = df['puma'].map(
        df_puma.set_index('gis_joinID_puma')['median_income_USD2023']
    )

    # Find the rows where 'census_area_medianIncome' is NaN
    nan_mask = df['census_area_medianIncome'].isna()

    # Attempt to fill NaNs using county-level median incomes
    df.loc[nan_mask, 'census_area_medianIncome'] = df.loc[nan_mask, 'county'].map(
        df_county.set_index('gis_joinID_county')['median_income_USD2023']
    )

    # Update the NaN mask after attempting to fill with county-level data
    nan_mask = df['census_area_medianIncome'].isna()

    # Attempt to fill remaining NaNs using state-level median incomes
    df.loc[nan_mask, 'census_area_medianIncome'] = df.loc[nan_mask, 'state'].map(
        df_state.set_index('state_abbrev')['median_income_USD2023']
    )
    
    return df


def calculate_percent_AMI(df_results_IRA: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the percentage of Area Median Income (AMI) and assigns income level designations.

    This function processes household income data, calculates the percentage relative to 
    Area Median Income, and creates two categorical classifications:
    1. Detailed income level categories (Low, Moderate, Middle-to-Upper Income)
    2. Binary Low-to-Moderate Income (LMI) or Middle-to-Upper Income (MUI) classification for policy analysis

    Args:
        df_results_IRA: Input DataFrame containing income information with columns:
                       - 'income': Income data (ranges or values)
                       - Other demographic/geographic columns for median income lookup

    Returns:
        DataFrame: Modified DataFrame with additional columns:
                  - 'household_income': Calculated household income (float)
                  - 'census_area_medianIncome': Area median income (float)
                  - 'percent_AMI': Percentage of AMI (float)
                  - 'income_level': Detailed income category (str)
                  - 'lmi_or_mui': Binary Low-to-Moderate Income (LMI) or Middle-to-Upper Income (MUI) (str)
        
    Raises:
        ValueError: If an unexpected income format is encountered during processing
    """
    # Create a mapping for special income ranges
    income_map = {
        '<10000': (9999.0, 9999.0),
        '200000+': (200000.0, 200000.0)
    }

    def split_income_range(income):
        """
        Processes income data which may be ranges, special values, or direct floats.
        
        Args:
            income: Income value (str, float, or special format)
            
        Returns:
            tuple: (low_income, high_income) for range calculation
            
        Raises:
            ValueError: If income format cannot be parsed
        """
        if isinstance(income, float):  # Handle float income directly
            return income, income
        if income in income_map:
            return income_map[income]
        try:
            # Parse income ranges like "50000-75000"
            low, high = map(float, income.split('-'))
            return low, high
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Unexpected income format: {income}. Expected format: 'low-high', '<10000', '200000+', or numeric value.") from e

    # Apply the income range split
    income_ranges = df_results_IRA['income'].apply(split_income_range)
    df_results_IRA['income_low'], df_results_IRA['income_high'] = zip(*income_ranges)
    df_results_IRA['income'] = (df_results_IRA['income_low'] + df_results_IRA['income_high']) / 2
    
    # Apply the generate_household_medianIncome_2023 function
    df_results_IRA['household_income'] = df_results_IRA.apply(generate_household_medianIncome_2023, axis=1)

    # Drop the intermediate columns
    df_results_IRA.drop(['income_low', 'income_high'], axis=1, inplace=True)

    # Fill NaNs in 'census_area_medianIncome' with the hierarchical lookup
    # Attempt to match median income for puma, then county, then state
    df_results_IRA = fill_na_with_hierarchy(
        df_results_IRA, 
        df_puma=df_puma_medianIncome, 
        df_county=df_county_medianIncome, 
        df_state=df_state_medianIncome
    )

    # Ensure income and census_area_medianIncome columns are float
    df_results_IRA['household_income'] = df_results_IRA['household_income'].astype(float).round(2)
    df_results_IRA['census_area_medianIncome'] = df_results_IRA['census_area_medianIncome'].astype(float).round(2)

    # Calculate percent_AMI
    df_results_IRA['percent_AMI'] = ((df_results_IRA['household_income'] / df_results_IRA['census_area_medianIncome']) * 100).round(2)

    # Assign income level designation based on percent_AMI thresholds
    income_conditions = [
        df_results_IRA['percent_AMI'] <= 80.0,
        (df_results_IRA['percent_AMI'] > 80.0) & (df_results_IRA['percent_AMI'] <= 150.0)
    ]
    income_choices = ['Low-Income', 'Moderate-Income']

    # Default to 'Middle-to-Upper-Income' for values above 150% AMI (ineligible for rebates)
    df_results_IRA['income_level'] = np.select(
        income_conditions, 
        income_choices, 
        default='Middle-to-Upper-Income'
    )

    # Create binary LMI/MUI classification
    df_results_IRA['lmi_or_mui'] = df_results_IRA['income_level'].apply(
        lambda x: 'LMI' if x in ['Low-Income', 'Moderate-Income'] else 'MUI'
    )

    return df_results_IRA


# ================================================================================================================
# FUNCTIONS: CALCULATE REBATE AMOUNTS
# ================================================================================================================

def get_max_rebate_amount(
        row: pd.Series,
        category: str) -> Tuple[float, float]:
    """
    Determine the maximum rebate amounts based on the category and row data.
    
    Looks up rebate eligibility based on predefined mapping in REBATE_MAPPING.
    
    Args:
        row: DataFrame row containing upgrade information
        category: Equipment category (e.g., 'heating', 'waterHeating')
        
    Returns:
        Tuple containing:
            - max_rebate_amount: Maximum rebate amount for the equipment
            - max_weatherization_rebate_amount: Maximum rebate amount for weatherization
    """
    if category in REBATE_MAPPING:
        column, conditions, rebate_amount = REBATE_MAPPING[category]
        max_rebate_amount = rebate_amount if any(cond in str(row[column]) for cond in conditions) else 0.00
    else:
        max_rebate_amount = 0.00

    max_weatherization_rebate_amount = 1600.00
    return max_rebate_amount, max_weatherization_rebate_amount


def calculate_rebateIRA(
        df: pd.DataFrame, 
        category: str, 
        menu_mp: int,
        percentile: str = 'mid'
        ) -> pd.DataFrame:
    """
    Calculate IRA rebate amounts based on income level and equipment costs.
    
    PREREQUISITE: Call calculate_upgrade_installed_cost() first to create the required
    mp{menu_mp}_{category}_upgrade_installed_cost_{percentile} columns.
    
    Applies income-based rebate rates:
    - Low-Income (≤80% AMI): 100% of cost up to max rebate
    - Moderate-Income (>80-150% AMI): 50% of cost up to max rebate  
    - Middle-to-Upper Income (>150% AMI): $0 rebate
    
    Uses the 5-step validation framework to ensure only valid homes receive rebates.
    
    Args:
        df: DataFrame with income data and upgrade costs.
        category: Equipment category (e.g., 'heating', 'cooling', 'waterHeating').
        menu_mp: Measure package number.
        percentile: Cost percentile ('low', 'mid', 'high').
        
    Returns:
        DataFrame with mp{menu_mp}_{category}_rebate_amount_{percentile} column.
        
    Raises:
        ValueError: If invalid menu_mp, category, or percentile provided.
        KeyError: If required columns are missing.
    """
    # Validate inputs
    if menu_mp not in VALID_MENU_MPS:
        raise ValueError(f"Invalid menu_mp: {menu_mp}. Must be one of {VALID_MENU_MPS}")
    
    # Valid categories are defined in EQUIPMENT_SPECS
    valid_categories = list(EQUIPMENT_SPECS.keys())
    if category not in valid_categories:
        raise ValueError(f"Invalid category: '{category}'. Must be one of {valid_categories}")
    
    if percentile not in ['low', 'mid', 'high']:
        raise ValueError(f"Invalid percentile: '{percentile}'. Must be 'low', 'mid', or 'high'")
    
    # Verify prerequisite columns exist
    cost_col = f'mp{menu_mp}_{category}_upgrade_installed_cost_{percentile}'
    if cost_col not in df.columns:
        raise KeyError(
            f"Missing required column: {cost_col}\n"
            f"Call calculate_upgrade_installed_cost() first to create this column."
        )
    
    # Verify income data exists
    required_income_cols = ['income_level']
    missing_income = [c for c in required_income_cols if c not in df.columns]
    if missing_income:
        raise KeyError(
            f"Missing income columns: {missing_income}\n"
            f"Call calculate_percent_AMI() first to create these columns."
        )
    
    print(f"\nCalculating {category} rebates (Menu MP{menu_mp}, {percentile} percentile)")
    
    # ===== STEP 1: Initialize validation tracking =====
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = \
        initialize_validation_tracking(df, category, menu_mp=menu_mp, verbose=True)
    
    # Define column names
    rebate_col = f'mp{menu_mp}_{category}_rebate_amount_{percentile}'
    weatherization_rebate_col = f'mp{menu_mp}_weatherization_rebate_amount_{percentile}'
    
    # ===== STEP 2: Initialize result series with template =====
    # Use create_retrofit_only_series to properly initialize with zeros for valid homes, NaN for others
    rebate_series = create_retrofit_only_series(df_copy, valid_mask)
    
    # Initialize weatherization series if applicable
    weatherization_series = None
    if menu_mp in [9, 10]:
        weatherization_series = create_retrofit_only_series(df_copy, valid_mask)
    
    # ===== STEP 3 & 4: Calculate for valid homes only =====
    # Extract cost and income data for valid homes
    install_cost = df_copy[cost_col]
    income_level = df_copy['income_level']
    
    # Get upgrade column name from REBATE_MAPPING for eligibility check
    upgrade_column = None
    max_rebate_base = 0.0
    if category in REBATE_MAPPING:
        upgrade_column, conditions, max_rebate_base = REBATE_MAPPING[category]
        # Check eligibility: home must have qualifying upgrade technology
        if upgrade_column in df_copy.columns:
            eligibility_mask = df_copy[upgrade_column].apply(
                lambda x: any(cond in str(x) for cond in conditions) if pd.notna(x) else False
            )
        else:
            eligibility_mask = pd.Series(False, index=df_copy.index)
    else:
        eligibility_mask = pd.Series(False, index=df_copy.index)
    
    # Calculate rebates for valid, eligible homes
    # Low-Income: 100% coverage up to max rebate
    low_income_mask = valid_mask & eligibility_mask & (income_level == 'Low-Income')
    if low_income_mask.any():
        project_coverage = (install_cost * 1.0).round(2)
        rebate_series.loc[low_income_mask] = np.minimum(project_coverage, max_rebate_base).loc[low_income_mask]
    
    # Moderate-Income: 50% coverage up to max rebate
    moderate_income_mask = valid_mask & eligibility_mask & (income_level == 'Moderate-Income')
    if moderate_income_mask.any():
        project_coverage = (install_cost * 0.5).round(2)
        rebate_series.loc[moderate_income_mask] = np.minimum(project_coverage, max_rebate_base).loc[moderate_income_mask]
    
    # Middle-to-Upper Income: No rebate (already initialized to 0)
    
    # Calculate weatherization rebates if applicable (MPs 9-10 only)
    if menu_mp in [9, 10] and weatherization_series is not None:
        enclosure_col = f'mp{menu_mp}_enclosure_upgrade_installed_cost_{percentile}'
        
        if enclosure_col in df_copy.columns:
            enclosure_cost = df_copy[enclosure_col]
            max_weatherization = 1600.0
            
            # Low-Income: 100% coverage up to max
            if low_income_mask.any():
                weatherization_coverage = (enclosure_cost * 1.0).round(2)
                weatherization_series.loc[low_income_mask] = np.minimum(
                    weatherization_coverage, max_weatherization
                ).loc[low_income_mask]
            
            # Moderate-Income: 50% coverage up to max
            if moderate_income_mask.any():
                weatherization_coverage = (enclosure_cost * 0.5).round(2)
                weatherization_series.loc[moderate_income_mask] = np.minimum(
                    weatherization_coverage, max_weatherization
                ).loc[moderate_income_mask]
    
    # Create DataFrame with new columns
    df_new_columns = pd.DataFrame({rebate_col: rebate_series})
    if weatherization_series is not None:
        df_new_columns[weatherization_rebate_col] = weatherization_series
    
    # Apply new columns to DataFrame with proper tracking
    df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
        df_copy, df_new_columns, category, category_columns_to_mask, all_columns_to_mask)
    
    # ===== STEP 5: Apply final verification masking for consistency =====
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=True)
    
    # Report summary
    valid_rebates = df_copy[rebate_col].notna().sum()
    mean_rebate = df_copy[rebate_col].mean()
    total_rebates = df_copy[rebate_col].sum()
    
    print(f"  Calculated rebates for {valid_rebates:,} homes")
    print(f"  Mean rebate: ${mean_rebate:,.2f}")
    print(f"  Total rebates: ${total_rebates:,.2f}\n")
    
    if menu_mp in [9, 10] and weatherization_rebate_col in df_copy.columns:
        valid_weatherization = df_copy[weatherization_rebate_col].notna().sum()
        mean_weatherization = df_copy[weatherization_rebate_col].mean()
        print(f"  Weatherization rebates: {valid_weatherization:,} homes (mean: ${mean_weatherization:,.2f})\n")
    
    return df_copy
