import numpy as np
from scipy.stats import norm

from config import PROJECT_ROOT
print(f"Project root directory: {PROJECT_ROOT}")

from cmu_tare_model.utils.inflation_adjustment import cpi_ratio_2023_2022
from cmu_tare_model.utils.process_income_data_for_rebates import *

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FUNCTIONS: AMI AND INCOME GROUP DESIGNATION FOR REBATE ELIGIBILITY
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# UPDATED SEPTEMBER 6, 2024 @ 12:10 AM
def generate_household_medianIncome_2023(row):
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

def fill_na_with_hierarchy(df, df_puma, df_county, df_state):
    """
    Fills NaN values in 'census_area_medianIncome' using a hierarchical lookup:
    first using the Puma level, then county, and finally state level median incomes.

    Parameters:
        df (DataFrame): The main DataFrame with NaNs to fill.
        df_puma (DataFrame): DataFrame with median incomes at the Puma level.
        df_county (DataFrame): DataFrame with median incomes at the county level.
        df_state (DataFrame): DataFrame with median incomes at the state level.
    
    Returns:
        DataFrame: Modified DataFrame with NaNs filled in 'census_area_medianIncome'.
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

def calculate_percent_AMI(df_results_IRA):
    """
    Calculates the percentage of Area Median Income (AMI) and assigns a designation based on the income level.

    Parameters:
        df_results_IRA (DataFrame): Input DataFrame containing income information.

    Returns:
        DataFrame: Modified DataFrame with additional columns for income calculations and designation.
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

# UPDATED AUGUST 20, 2024 @ 3:08 AM
# Mapping for categories and their corresponding conditions
rebate_mapping = {
    'heating': ('upgrade_hvac_heating_efficiency', ['ASHP', 'MSHP'], 8000.00),
    'waterHeating': ('upgrade_water_heater_efficiency', ['Electric Heat Pump'], 1750.00),
    'clothesDrying': ('upgrade_clothes_dryer', ['Electric, Premium, Heat Pump, Ventless'], 840.00),
    'cooking': ('upgrade_cooking_range', ['Electric, '], 840.00)
}

def get_max_rebate_amount(row, category):
    """
    Determine the maximum rebate amounts based on the category and row data.
    """
    if category in rebate_mapping:
        column, conditions, rebate_amount = rebate_mapping[category]
        max_rebate_amount = rebate_amount if any(cond in str(row[column]) for cond in conditions) else 0.00
    else:
        max_rebate_amount = 0.00

    max_weatherization_rebate_amount = 1600.00
    return max_rebate_amount, max_weatherization_rebate_amount

def calculate_rebate(df_results_IRA, row, category, menu_mp, coverage_rate):
    """
    Calculate and assign the rebate amounts.
    """
    max_rebate_amount, max_weatherization_rebate_amount = get_max_rebate_amount(row, category)
    
    project_coverage = round(row[f'mp{menu_mp}_{category}_installationCost'] * coverage_rate, 2)
    df_results_IRA.at[row.name, f'mp{menu_mp}_{category}_rebate_amount'] = min(project_coverage, max_rebate_amount)
    
    if f'mp{menu_mp}_enclosure_upgradeCost' in df_results_IRA.columns:
        weatherization_project_coverage = round(row[f'mp{menu_mp}_enclosure_upgradeCost'] * coverage_rate, 2)
        df_results_IRA.at[row.name, 'weatherization_rebate_amount'] = min(weatherization_project_coverage, max_weatherization_rebate_amount)

def calculate_rebateIRA(df_results_IRA, category, menu_mp):
    """
    Calculates rebate amounts for different end-uses based on income designation.
    """
    def apply_rebate(row):
        income_designation = row['lowModerateIncome_designation']
        if income_designation == 'Low-Income':
            calculate_rebate(df_results_IRA, row, category, menu_mp, 1.00)
        elif income_designation == 'Moderate-Income':
            calculate_rebate(df_results_IRA, row, category, menu_mp, 0.50)
        else:
            df_results_IRA.at[row.name, f'mp{menu_mp}_{category}_rebate_amount'] = 0.00
            if menu_mp in [9, 10]:
                df_results_IRA.at[row.name, 'weatherization_rebate_amount'] = 0.00

    df_results_IRA.apply(apply_rebate, axis=1)
    return df_results_IRA