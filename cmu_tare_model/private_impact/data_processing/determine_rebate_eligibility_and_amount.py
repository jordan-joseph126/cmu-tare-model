import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple, Union, Callable

from cmu_tare_model.constants import REBATE_MAPPING
from cmu_tare_model.utils.inflation_adjustment import cpi_ratio_2023_2022
from cmu_tare_model.utils.validation_framework import (
    create_retrofit_only_series,
    initialize_validation_tracking,
    apply_final_masking,
)

from cmu_tare_model.private_impact.data_processing.process_income_data_for_rebates import (
    df_puma_medianIncome,
    df_county_medianIncome, 
    df_state_medianIncome,
    cpi_ratio_2023_2022
)

"""
================================================================================================================================================================================
FUNCTIONS: AMI AND INCOME GROUP DESIGNATION FOR REBATE ELIGIBILITY
================================================================================================================================================================================
- UPDATED APRIL 22, 2025 WITH IMPROVED DOCUMENTATION, MODULARITY, ERROR HANDLING
"""


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
    Calculates the percentage of Area Median Income (AMI) and assigns a designation based on the income level.

    Args:
        df_results_IRA: Input DataFrame containing income information

    Returns:
        DataFrame: Modified DataFrame with additional columns for income calculations and designation
        
    Raises:
        ValueError: If an unexpected income format is encountered
    """
    # Create a mapping for income ranges
    income_map = {
        '<10000': (9999.0, 9999.0),
        '200000+': (200000.0, 200000.0)
    }

    # Split the income ranges and map values
    def split_income_range(income):
        if isinstance(income, float):  # Handle float income directly
            return income, income
        if income in income_map:
            return income_map[income]
        try:
            low, high = map(float, income.split('-'))
            return low, high
        except Exception as e:
            raise ValueError(f"Unexpected income format: {income}") from e

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
    df_results_IRA = fill_na_with_hierarchy(df_results_IRA, df_puma=df_puma_medianIncome, df_county=df_county_medianIncome, df_state=df_state_medianIncome)

    # Ensure income and census_area_medianIncome columns are float
    df_results_IRA['household_income'] = df_results_IRA['household_income'].astype(float).round(2)
    df_results_IRA['census_area_medianIncome'] = df_results_IRA['census_area_medianIncome'].astype(float).round(2)

    # Calculate percent_AMI
    df_results_IRA['percent_AMI'] = ((df_results_IRA['household_income'] / df_results_IRA['census_area_medianIncome']) * 100).round(2)

    # Categorize the income level based on percent_AMI
    conditions_lmi = [
        df_results_IRA['percent_AMI'] <= 80.0,
        (df_results_IRA['percent_AMI'] > 80.0) & (df_results_IRA['percent_AMI'] <= 150.0)
    ]
    choices_lmi = ['Low-Income', 'Moderate-Income']

    df_results_IRA['lowModerateIncome_designation'] = np.select(
        conditions_lmi, choices_lmi, default='Middle-to-Upper-Income'
    )

    # Output the modified DataFrame
    return df_results_IRA


"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FUNCTIONS: CALCULATE REBATE AMOUNTS
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

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


def calculate_rebate(
        df_results_IRA: pd.DataFrame, 
        row: pd.Series,
        category: str, 
        menu_mp: int, 
        coverage_rate: float) -> None:
    """
    Calculate and assign the rebate amounts for a specific row.
    
    Args:
        df_results_IRA: DataFrame to update with rebate amounts
        row: Row containing installation cost data
        category: Equipment category
        menu_mp: Measure package identifier
        coverage_rate: Rebate coverage rate (1.0 for low-income, 0.5 for moderate-income)
        
    Raises:
        ValueError: If an invalid category is provided
        KeyError: If required columns are missing
    """
    try:
        max_rebate_amount, max_weatherization_rebate_amount = get_max_rebate_amount(row, category)
        
        # Calculate equipment rebate
        install_cost_col = f'mp{menu_mp}_{category}_installationCost'
        rebate_col = f'mp{menu_mp}_{category}_rebate_amount'
        
        if install_cost_col in row and not pd.isna(row[install_cost_col]):
            project_coverage = round(row[install_cost_col] * coverage_rate, 2)
            df_results_IRA.at[row.name, rebate_col] = min(project_coverage, max_rebate_amount)
        else:
            df_results_IRA.at[row.name, rebate_col] = 0.00
            if coverage_rate > 0 and max_rebate_amount > 0:
                print(f"Warning: Installation cost data missing for row {row.name}, category {category}. Setting rebate to 0.")
        
        # Calculate weatherization rebate if applicable
        if f'mp{menu_mp}_enclosure_upgradeCost' in df_results_IRA.columns and menu_mp in [9, 10]:
            if f'mp{menu_mp}_enclosure_upgradeCost' in row and not pd.isna(row[f'mp{menu_mp}_enclosure_upgradeCost']):
                weatherization_project_coverage = round(row[f'mp{menu_mp}_enclosure_upgradeCost'] * coverage_rate, 2)
                df_results_IRA.at[row.name, 'weatherization_rebate_amount'] = min(weatherization_project_coverage, max_weatherization_rebate_amount)
            else:
                df_results_IRA.at[row.name, 'weatherization_rebate_amount'] = 0.00
                if coverage_rate > 0 and menu_mp in [9, 10]:
                    print(f"Warning: Enclosure cost data missing for row {row.name}. Setting weatherization rebate to 0.")
    
    except Exception as e:
        print(f"Error calculating rebate for row {row.name}, category {category}: {str(e)}")
        # Set default values to prevent calculations from breaking
        df_results_IRA.at[row.name, rebate_col] = 0.00
        if menu_mp in [9, 10] and 'weatherization_rebate_amount' in df_results_IRA.columns:
            df_results_IRA.at[row.name, 'weatherization_rebate_amount'] = 0.00


def calculate_rebateIRA(
        df_results_IRA: pd.DataFrame, 
        category: str, 
        menu_mp: int) -> pd.DataFrame:
    """
    Calculates rebate amounts for different end-uses based on income designation.
    
    This function applies the appropriate rebate percentage based on income designation
    and applies data validation to ensure only valid homes are considered for rebates.
    Rebates are calculated at different rates:
    - 100% coverage rate for low-income homes (up to maximum rebate amount)
    - 50% coverage rate for moderate-income homes (up to maximum rebate amount)
    - 0% coverage rate for middle-to-upper-income homes
    
    Args:
        df_results_IRA: DataFrame containing income designations and cost data
        category: Equipment category (e.g., 'heating', 'waterHeating')
        menu_mp: Measure package identifier
        
    Returns:
        Updated DataFrame with calculated rebate amounts
        
    Notes:
        This function implements the validation framework:
        1. Uses initialize_validation_tracking() to determine valid homes
        2. Creates retrofit-only series with NaN for invalid homes
        3. Calculates rebates only for valid homes
        4. Applies final verification masking
    """

    # Initialize validation tracking
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        df_results_IRA, category, menu_mp, verbose=True)
    
    # Create rebate columns
    rebate_col = f'mp{menu_mp}_{category}_rebate_amount'

    df_copy[rebate_col] = create_retrofit_only_series(df_copy, valid_mask)
    
    # Track the rebate column
    category_columns_to_mask.append(rebate_col)
    
    # Also track and create weatherization rebate column if relevant
    if menu_mp in [9, 10]:
        weatherization_col = 'weatherization_rebate_amount'
        df_copy[weatherization_col] = 0.0
        
        # Track weatherization column under the category
        category_columns_to_mask.append(weatherization_col)
    
    # Apply rebates based on income designation
    def apply_rebate(row):
        # Skip invalid homes
        if not valid_mask.loc[row.name]:
            return
            
        income_designation = row['lowModerateIncome_designation']
        if income_designation == 'Low-Income':
            calculate_rebate(df_copy, row, category, menu_mp, 1.00)
        elif income_designation == 'Moderate-Income':
            calculate_rebate(df_copy, row, category, menu_mp, 0.50)
        else:
            df_copy.at[row.name, rebate_col] = 0.00
            if menu_mp in [9, 10] and 'weatherization_rebate_amount' in df_copy.columns:
                df_copy.at[row.name, 'weatherization_rebate_amount'] = 0.00

    df_copy.apply(apply_rebate, axis=1)
    
    # Apply final verification masking for consistency
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=True)
    
    return df_copy
