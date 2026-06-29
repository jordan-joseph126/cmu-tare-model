import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple, Union, Callable

from cmu_tare_model.constants import REBATE_MAPPING, REBATE_ELIGIBLE_HEATING_MPS, VERBOSE
from cmu_tare_model.utils.inflation_adjustment import cpi_ratio_2025_2018
from cmu_tare_model.utils.column_names import (
    create_cost_col,
    create_rebate_col,
    create_enclosure_cost_col,
    create_weatherization_rebate_col
)
from cmu_tare_model.utils.validation_framework import (
    create_retrofit_only_series,
    initialize_validation_tracking,
    apply_final_masking,
)

from cmu_tare_model.private_impact.data_processing.process_income_data_for_rebates import (
    df_county_medianIncome,
    df_state_medianIncome,
)

"""
================================================================================================================================================================================
FUNCTIONS: AMI AND INCOME GROUP DESIGNATION FOR REBATE ELIGIBILITY
================================================================================================================================================================================
- UPDATED APRIL 22, 2025 WITH IMPROVED DOCUMENTATION, MODULARITY, ERROR HANDLING
"""


def generate_household_medianIncome_2025(row: pd.Series) -> float:
    """
    Generate a household median income value in USD2025 using a probabilistic
    approach.

    Samples from a normal distribution based on income range bounds, then
    ensures the final value remains within the valid income range.

    Args:
        row: DataFrame row containing income_low, income_high, and income values

    Returns:
        float: Generated median income value in 2025 dollars
    """
    # The ResStock household-income bins are reported in USD2018, so inflate
    # them to the model reference year (USD2025) before sampling.
    low = row['income_low'] * cpi_ratio_2025_2018
    high = row['income_high'] * cpi_ratio_2025_2018
    mean = row['income'] * cpi_ratio_2025_2018

    # Calculate std assuming 10th and 90th percentiles
    std = (high - low) / (norm.ppf(0.90) - norm.ppf(0.10))

    # Sample from the normal distribution
    ami_2025 = np.random.normal(loc=mean, scale=std)

    # Ensure the generated income is within the bounds
    ami_2025 = max(low, min(high, ami_2025))
    return ami_2025


def fill_na_with_hierarchy(
        df: pd.DataFrame,
        df_county: pd.DataFrame,
        df_state: pd.DataFrame) -> pd.DataFrame:
    """
    Fills 'census_area_medianIncome' using a two-level lookup: county-level
    median income first, then state-level for any county that does not match.

    Connecticut is the main reason a county can miss the county-level join. The
    Census switched Connecticut from its legacy counties (FIPS 09001-09015) to
    nine planning regions (09110-09190), but ResStock still uses the legacy
    county codes. Those codes do not exist in the current ACS file, so
    Connecticut homes fall through to the state-level value. This is expected,
    not a data error.

    Args:
        df: The main DataFrame with area median income to fill
        df_county: DataFrame with median incomes at the county level
        df_state: DataFrame with median incomes at the state level

    Returns:
        DataFrame: Modified DataFrame with 'census_area_medianIncome' filled
    """
    # Fill using county-level median incomes first.
    df['census_area_medianIncome'] = df['county'].map(
        df_county.set_index('gis_joinID_county')['median_income_USD2025']
    )

    # Any county that did not match (notably Connecticut, see above) falls
    # back to the state-level median income.
    nan_mask = df['census_area_medianIncome'].isna()
    df.loc[nan_mask, 'census_area_medianIncome'] = df.loc[nan_mask, 'state'].map(
        df_state.set_index('state_abbrev')['median_income_USD2025']
    )

    return df


def calculate_percent_AMI(df_results_IRA: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
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
        random_seed: Random seed for reproducible income sampling. Ensures consistent
                    income classifications across different measure package runs (e.g.,
                    MP4 and MP8 produce identical rebate eligibility). Default: 42.

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
    
    # Set random seed for reproducible income sampling across MP runs.
    # This ensures identical income classifications (and thus identical rebate
    # eligibility) for the same homes regardless of which measure package is
    # being processed — critical for MP4 vs MP8 result consistency.
    np.random.seed(random_seed)
    
    # Apply the generate_household_medianIncome_2025 function
    df_results_IRA['household_income'] = df_results_IRA.apply(generate_household_medianIncome_2025, axis=1)

    # Drop the intermediate columns
    df_results_IRA.drop(['income_low', 'income_high'], axis=1, inplace=True)

    # Fill 'census_area_medianIncome' with the hierarchical lookup:
    # match county-level median income first, then state-level.
    df_results_IRA = fill_na_with_hierarchy(
        df_results_IRA,
        df_county=df_county_medianIncome,
        df_state=df_state_medianIncome
    )

    # Ensure income and census_area_medianIncome columns are float
    df_results_IRA['household_income'] = df_results_IRA['household_income'].astype(float).round(2)
    df_results_IRA['census_area_medianIncome'] = df_results_IRA['census_area_medianIncome'].astype(float).round(2)

    # Calculate percent_AMI
    df_results_IRA['percent_AMI'] = ((df_results_IRA['household_income'] / df_results_IRA['census_area_medianIncome']) * 100).round(2)

    # Create detailed income level categories
    income_conditions = [
        df_results_IRA['percent_AMI'] <= 80.0,
        (df_results_IRA['percent_AMI'] > 80.0) & (df_results_IRA['percent_AMI'] <= 150.0)
    ]
    income_choices = ['Low-Income', 'Moderate-Income']

    df_results_IRA['income_level'] = np.select(
        income_conditions, 
        income_choices, 
        default='Middle-to-Upper-Income'
    )

    # Create binary LMI/MUI classification
    # Method 1: Using the income_level column we just created
    df_results_IRA['lmi_or_mui'] = df_results_IRA['income_level'].apply(
        lambda x: 'LMI' if x in ['Low-Income', 'Moderate-Income'] else 'MUI'
    )
    
    # Alternative Method 2: Direct threshold-based approach (more efficient for large datasets)
    # df_results_IRA['lmi_or_mui'] = np.where(
    #     df_results_IRA['percent_AMI'] <= 150.0, 'LMI', 'MUI'
    # )

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
        coverage_rate: float,
        cost_scenario: str) -> None:
    """
    Calculate and assign the rebate amounts for a specific row.
    
    Args:
        df_results_IRA: DataFrame to update with rebate amounts
        row: Row containing installation cost data
        category: Equipment category
        menu_mp: Measure package identifier
        coverage_rate: Rebate coverage rate (1.0 for low-income, 0.5 for moderate-income)
        cost_scenario: Cost methodology key ('v3' or 'v4LOW/MID/HIGH').
        
    Raises:
        ValueError: If an invalid category is provided
        KeyError: If required columns are missing
    """
    try:
        max_rebate_amount, max_weatherization_rebate_amount = get_max_rebate_amount(row, category)
        
        # Calculate equipment rebate
        install_cost_col_name = create_cost_col(menu_mp=menu_mp, category=category, cost_type='upgrade', cost_scenario=cost_scenario)
        rebate_col_name = create_rebate_col(menu_mp=menu_mp, category=category, cost_scenario=cost_scenario)
        
        if install_cost_col_name in row and not pd.isna(row[install_cost_col_name]):
            project_coverage = round(row[install_cost_col_name] * coverage_rate, 2)
            df_results_IRA.at[row.name, rebate_col_name] = min(project_coverage, max_rebate_amount)
        else:
            df_results_IRA.at[row.name, rebate_col_name] = 0.00
            if coverage_rate > 0 and max_rebate_amount > 0:
                raise ValueError(f"Warning: Installation cost data missing for row {row.name}, category {category}. Setting rebate to 0.")
        
        # Calculate weatherization rebate if applicable
        enclosure_cost_col_name = create_enclosure_cost_col(menu_mp=menu_mp, cost_scenario=cost_scenario)
        weatherization_rebate_col_name = create_weatherization_rebate_col(cost_scenario=cost_scenario)
        if enclosure_cost_col_name in df_results_IRA.columns and menu_mp in [9, 10]:
            if enclosure_cost_col_name in row and not pd.isna(row[enclosure_cost_col_name]):
                weatherization_project_coverage = round(row[enclosure_cost_col_name] * coverage_rate, 2)
                df_results_IRA.at[row.name, weatherization_rebate_col_name] = min(weatherization_project_coverage, max_weatherization_rebate_amount)
            else:
                df_results_IRA.at[row.name, weatherization_rebate_col_name] = 0.00
                if coverage_rate > 0 and menu_mp in [9, 10]:
                    raise ValueError(f"Warning: Enclosure cost data missing for row {row.name}. Setting weatherization rebate to 0.")
    
    except Exception as e:
        print(f"Error calculating rebate for row {row.name}, category {category}: {str(e)}")
        
        # Set default values to prevent calculations from breaking
        df_results_IRA.at[row.name, rebate_col_name] = 0.00
        weatherization_rebate_col_name = create_weatherization_rebate_col(cost_scenario=cost_scenario)
        if menu_mp in [9, 10] and weatherization_rebate_col_name in df_results_IRA.columns:
            df_results_IRA.at[row.name, weatherization_rebate_col_name] = 0.00


def calculate_rebateIRA(
    df_results_IRA: pd.DataFrame, 
    category: str, 
    menu_mp: int,
    cost_scenario: str,
    verbose: bool = VERBOSE
) -> pd.DataFrame:
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
        cost_scenario: Cost methodology key ('v3' or 'v4LOW/MID/HIGH').
        verbose: Flag to enable verbose logging
        
    Returns:
        Updated DataFrame with calculated rebate amounts
        
    Notes:
        This function implements the validation framework:
        1. Uses initialize_validation_tracking() to determine valid homes
        2. Creates retrofit-only series with NaN for invalid homes
        3. Calculates rebates only for valid homes
        4. Applies final verification masking
    """

    # Cooling rebates are not modeled separately -- the heat-pump rebate covers
    # both heating and cooling, so cooling is a no-op here and the DataFrame
    # passes through unchanged.
    if category == 'cooling':
        if verbose:
            print("Skipping rebate for 'cooling' (covered by the heating heat-pump rebate).")
        return df_results_IRA

    # Validate category has rebate mapping
    if category not in REBATE_MAPPING:
        raise ValueError(
            f"Category '{category}' is not supported for rebate calculations. "
            f"Valid categories with rebates: {list(REBATE_MAPPING.keys())}. "
            f"Note: Cooling rebates are not modeled separately - heat pump rebates cover both heating and cooling."
        )

    # Initialize validation tracking
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        df_results_IRA, category, menu_mp, verbose=verbose)
    
    # Create rebate columns
    rebate_col_name = create_rebate_col(menu_mp=menu_mp, category=category, cost_scenario=cost_scenario)

    df_copy[rebate_col_name] = create_retrofit_only_series(df_copy, valid_mask)
    
    # Track the rebate column
    category_columns_to_mask.append(rebate_col_name)
    
    # Also track and create weatherization rebate column if relevant
    if menu_mp in [9, 10]:
        weatherization_rebate_col_name = create_weatherization_rebate_col(cost_scenario=cost_scenario)
        df_copy[weatherization_rebate_col_name] = 0.0
        
        # Track weatherization column under the category
        category_columns_to_mask.append(weatherization_rebate_col_name)
    
    # ===== REBATE ELIGIBILITY CHECK =====
    # Only high-efficiency MPs qualify for IRA rebates.
    # Standard-efficiency MPs (e.g., MP3) get zero rebates.
    if category == 'heating' and menu_mp not in REBATE_ELIGIBLE_HEATING_MPS:
        if verbose:
            print(f"  MP{menu_mp} is NOT eligible for heating rebates (standard efficiency). "
                  f"Setting all rebate amounts to 0.")
        df_copy[rebate_col_name] = 0.0
        # Apply valid_mask: NaN for invalid homes, 0 for valid homes
        df_copy.loc[~valid_mask, rebate_col_name] = np.nan
        
        # Apply final verification masking for consistency
        df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=verbose)
        return df_copy
    
    # Apply rebates based on income designation
    def apply_rebate(row):
        # Skip invalid homes
        if not valid_mask.loc[row.name]:
            return
            
        income_designation = row['income_level']
        if income_designation == 'Low-Income':
            calculate_rebate(df_copy, row, category, menu_mp, 1.00, cost_scenario=cost_scenario)
        elif income_designation == 'Moderate-Income':
            calculate_rebate(df_copy, row, category, menu_mp, 0.50, cost_scenario=cost_scenario)
        else:
            df_copy.at[row.name, rebate_col_name] = 0.00
            if menu_mp in [9, 10] and weatherization_rebate_col_name in df_copy.columns:
                df_copy.at[row.name, weatherization_rebate_col_name] = 0.00

    df_copy.apply(apply_rebate, axis=1)
    
    # Apply final verification masking for consistency
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=verbose)
    
    return df_copy
