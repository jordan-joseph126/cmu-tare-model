import pandas as pd

# Constants
from cmu_tare_model.constants import EQUIPMENT_SPECS, POLLUTANTS, TD_LOSSES_MULTIPLIER, CR_FUNCTIONS, RCM_MODELS

# Imports for lookup dictionaries, functions, and calculations
from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import calculate_fossil_fuel_emissions
from cmu_tare_model.public_impact.emissions_scenario_settings import define_scenario_settings
from cmu_tare_model.public_impact.calculations.precompute_hdd_factors import precompute_hdd_factors
from cmu_tare_model.public_impact.data_processing.create_lookup_msc_health_county import (
    lookup_health_fossil_fuel_acs,
    lookup_health_fossil_fuel_h6c,
    lookup_health_electricity_acs,
    lookup_health_electricity_h6c,
)

"""
======================================================================================================================
# Project VSL Estimates for Future Years
Future years are estimated using the following formula:
VSL_future = VSL_base * (1.01)^(year - base_year)

Where: 
VSL_base and VSL_future are both in constant 2023 dollars. (VSL_base is 11.0M in 2022 dollars, adjusted to 2023 dollars)
VSL_base = 11.0M * (CPI_2023 / CPI_2022) = ___ in 2023 dollars.
We assume a 1% annual growth rate for real earnings, consistent with the HHS VSL Guidance.

Additional details are given below
======================================================================================================================
From Federal Register, Notice by the Consumer Product Safety Commission (CPSC) on 2024-04-18:
Notice of Availability of Final Guidance for Estimating Value per Statistical Life
https://www.federalregister.gov/documents/2024/04/18/2024-08300/notice-of-availability-of-final-guidance-for-estimating-value-per-statistical-life

We start with the EPA VSL of $11.0M in USD2022 because much air quality regulation work is based on this value.
We then adjust it to 2023 dollars using the CPI-U (Consumer Price Index for All Urban Consumers) inflation rate.

CPSC suggests: 
"When estimating VSL for future years, CPSC will increase the VSL by the expected growth in real earnings
and discount the resulting benefit values to reflect the time value of money, consistent with its approach 
for all cost and benefits estimates."
    - Inflation: Inflate to year where full annual data is available for changes in prices (inflation). Use data and formula in HHS VSL guidance
    - Income Elasticity: Using value from HHS VSL Guidance
======================================================================================================================

HHS VSL Guidance: HHS Standard Values for Regulatory Analysis, 2024
https://aspe.hhs.gov/sites/default/files/documents/cd2a1348ea0777b1aa918089e4965b8c/standard-ria-values.pdf

The VSL estimates reported in the literature review correspond to a 2013 base year.(6) We update these values
to a 2023 base year by adjusting for inflation(7) and changes in real income.(8) These adjustments increase the VSL
estimates in nominal terms by about 44% compared to 2013. From the 2023 base-year VSL estimates, we
report estimates for 2024 and future years. These estimates increase over time in real terms, consistent with a
long-term annual growth rate for real earnings of 1.0%(9) and an assumption that the VSL income elasticity is
1.0. For mortality risk changes occurring in 2024, we adopt $6.1 million, $13.1 million, and $19.9 million for the
low, central, and high estimates of VSL, respectively. For impacts in other years, including the base year, please
refer to Table 1 or the unrounded estimates available in a supplemental table to this Data Point.

    7 U.S. Bureau of Labor Statistics. CPI for all Urban Consumers (CPI-U), Not Seasonally Adjusted,
    https://data.bls.gov/timeseries/CUUR0000SA0. Annual figures for 2013 to 2023. Accessed January 11, 2024.

    8 U.S. Bureau of Labor Statistics. Weekly and hourly earnings data from the Current Population Survey, Not Seasonally Adjusted.
    https://data.bls.gov/timeseries/LEU0252881600. Annual figures for 2013 to 2023. Accessed January 18, 2024.

    9 Congressional Budget Office. June 2023. “The 2023 Long-Term Budget Outlook.” Table C-1. Average Annual Values for Additional
    Economic Variables That Underlie CBO’s Extended Baseline Projections: Growth of Real Earnings per Worker, Overall, 2023-2053.
    https://www.cbo.gov/publication/59014.
"""

def calculate_health_impacts(df, menu_mp, policy_scenario, df_baseline_damages=None):
    """
    Calculate lifetime health impacts for each equipment category over all (rcm, cr) combinations.
    
    This function returns a tuple of two dataframes:
      - The main dataframe (df_main) is the input dataframe updated with lifetime health damages.
      - The detailed dataframe (df_detailed) contains annual results along with lifetime breakdowns.
    
    Args:
        df (pd.DataFrame): Input data.
        menu_mp (int): Measure package identifier.
        policy_scenario (str): Policy scenario name that determines model inputs ('No Inflation Reduction Act' or 'AEO2023 Reference Case').
        df_baseline_damages (pd.DataFrame, optional): Baseline health damages data.
    
    Returns:
        tuple: (df_main, df_detailed)
            - df_main contains the aggregated lifetime health impacts: lifetime damages and avoided damages (if applicable).
            - df_detailed contains the detailed annual health damages results.
    """
    # Create a copy of the input df
    # Then initialize the detailed dataframe (df_copy will become df_main)
    df_copy = df.copy()
    df_detailed = pd.DataFrame(index=df_copy.index)

    # Initialize a dictionary to hold lifetime health impacts columns for each category
    lifetime_columns_data = {}

    # Retrieve scenario-specific settings for electricity/fossil-fuel emissions
    # Ignored (underscored) the climate lookup and cambium scenario as it's not used in this function
    scenario_prefix, _, lookup_emissions_fossil_fuel, _, lookup_emissions_electricity_health = define_scenario_settings(menu_mp, policy_scenario)

    # Precompute HDD adjustment factors by region and year
    hdd_factors_per_year = precompute_hdd_factors(df_copy)

    # Loop over each equipment category and its lifetime
    for category, lifetime in EQUIPMENT_SPECS.items():
        # Reinitialize lifetime health impacts for this category for each (rcm, cr) pair
        lifetime_health_damages = {(rcm, cr): pd.Series(0.0, index=df_copy.index)
                                   for rcm in RCM_MODELS for cr in CR_FUNCTIONS}
        
        # Loop over each year in the equipment's lifetime
        for year in range(1, lifetime + 1):
            year_label = year + 2023
            # Retrieve HDD factor for the current year; use default factor if missing
            # The adjusted HDD factor only applies to heating/waterHeating categories
            hdd_factor = hdd_factors_per_year.get(year_label, pd.Series(1.0, index=df_copy.index))
            adjusted_hdd_factor = hdd_factor if category in ['heating', 'waterHeating'] else pd.Series(1.0, index=df_copy.index)
            
            # Calculate fossil fuel emissions for the current category and year
            total_fossil_fuel_emissions = calculate_fossil_fuel_emissions(
                df_copy, category, adjusted_hdd_factor, lookup_emissions_fossil_fuel, menu_mp
            )
            
            # For each (rcm, cr) combination, compute health damages for this year and accumulate lifetime values
            for rcm in RCM_MODELS:
                for cr in CR_FUNCTIONS:
                    health_results_pair = calculate_health_damages_for_pair(
                        df=df_copy,
                        category=category,
                        year_label=year_label,
                        adjusted_hdd_factor=adjusted_hdd_factor,
                        lookup_emissions_electricity_health=lookup_emissions_electricity_health,
                        scenario_prefix=scenario_prefix,
                        total_fossil_fuel_emissions=total_fossil_fuel_emissions,
                        menu_mp=menu_mp,
                        rcm=rcm,
                        cr=cr
                    )
                    overall_col = f'{scenario_prefix}{year_label}_{category}_damages_health_{cr}_{rcm}'
                    lifetime_health_damages[(rcm, cr)] += health_results_pair[overall_col]
                    
                    # Append annual detailed results
                    df_detailed = pd.concat([df_detailed, pd.DataFrame(health_results_pair, index=df_copy.index)], axis=1)
        
        # After processing all years for this category, prepare lifetime columns
        lifetime_dict = {}
        if menu_mp != 0 and df_baseline_damages is not None:
            for rcm in RCM_MODELS:
                for cr in CR_FUNCTIONS:
                    baseline_health_col = f'baseline_{category}_lifetime_damages_health_{cr}_{rcm}'
                    if baseline_health_col in df_baseline_damages.columns:
                        avoided_health_damages_col = f'{scenario_prefix}{category}_avoided_damages_health_{cr}_{rcm}'
                        lifetime_dict[avoided_health_damages_col] = df_baseline_damages[baseline_health_col] - lifetime_health_damages[(rcm, cr)]
                    else:
                        print(f"Warning: Missing baseline for {category} ({cr}, {rcm}). Avoided health values skipped.")
        for rcm in RCM_MODELS:
            for cr in CR_FUNCTIONS:
                overall_health_col = f'{scenario_prefix}{category}_lifetime_damages_health_{cr}_{rcm}'
                lifetime_dict[overall_health_col] = lifetime_health_damages[(rcm, cr)]
        
        # Add this category's lifetime results to the global lifetime dictionary
        lifetime_columns_data.update(lifetime_dict)
        # Also add these lifetime columns to the detailed dataframe for completeness
        df_detailed = pd.concat([df_detailed, pd.DataFrame(lifetime_dict, index=df_copy.index)], axis=1)
    
    # Create a dataframe for lifetime results and merge with the main dataframe
    df_lifetime = pd.DataFrame(lifetime_columns_data, index=df_copy.index)
    df_main = df_copy.join(df_lifetime, how='left')
    
    # Apply final rounding
    df_main = df_main.round(2)
    df_detailed = df_detailed.round(2)
    
    return df_main, df_detailed

def calculate_health_damages_for_pair(df, category, year_label, adjusted_hdd_factor,
                                     lookup_emissions_electricity_health, scenario_prefix,
                                     total_fossil_fuel_emissions, menu_mp, rcm, cr):
    """
    Calculate health-related damages for a single (rcm, cr) pair for a given category and year.
    
    Returns a dictionary mapping column names to computed Series.
    """
    def get_emissions_factor(row, pollutant, lookup_dict):
        key = (row['year'], row['gea_region'])
        if not pollutant.startswith("delta_egrid_"):
            pollutant = "delta_egrid_" + pollutant.lower()
        return lookup_dict.get(key, {}).get(pollutant, 0.0)
    
    def get_msc_value(row, pollutant, lookup_dict, rcm):
        return lookup_dict.get((row['county_fips'], row['state']), {}).get(rcm, {}).get(pollutant.upper(), 0.0)
    
    # Select lookup dictionaries based on the concentration-response function
    if cr == 'acs':
        lookup_msc_fossil_fuel = lookup_health_fossil_fuel_acs
        lookup_msc_electricity = lookup_health_electricity_acs
    elif cr == 'h6c':
        lookup_msc_fossil_fuel = lookup_health_fossil_fuel_h6c
        lookup_msc_electricity = lookup_health_electricity_h6c
    else:
        raise ValueError(f"Invalid C-R function: {cr}")
    
    annual_damages = pd.Series(0.0, index=df.index)
    health_results = {}
    
    for pollutant in POLLUTANTS:
        if pollutant == 'co2e':
            continue
        # Fossil fuel damages
        fossil_fuel_emissions = total_fossil_fuel_emissions.get(pollutant, pd.Series(0.0, index=df.index))
        fossil_fuel_msc = df.apply(lambda row: get_msc_value(row, pollutant, lookup_msc_fossil_fuel, rcm), axis=1)
        fossil_fuel_damages = fossil_fuel_emissions * fossil_fuel_msc
        
        # Electricity damages
        electricity_emissions_factor = df.apply(lambda row: get_emissions_factor(row, pollutant, lookup_emissions_electricity_health), axis=1)
        if menu_mp == 0:
            electricity_consumption = df.get(f'base_electricity_{category}_consumption', pd.Series(0.0, index=df.index)).fillna(0)
            electricity_consumption *= adjusted_hdd_factor
        else:
            electricity_consumption = df.get(f'mp{menu_mp}_{year_label}_{category}_consumption', pd.Series(0.0, index=df.index)).fillna(0)
        electricity_msc = df.apply(lambda row: get_msc_value(row, pollutant, lookup_msc_electricity, rcm), axis=1)
        electricity_emissions = electricity_consumption * TD_LOSSES_MULTIPLIER * electricity_emissions_factor
        electricity_damages = electricity_emissions * electricity_msc
        
        total_damages = fossil_fuel_damages + electricity_damages
        col_name = f'{scenario_prefix}{year_label}_{category}_damages_{pollutant}_{cr}_{rcm}'
        health_results[col_name] = total_damages  # No rounding here; deferred to final stage
        annual_damages += total_damages
    
    overall_col = f'{scenario_prefix}{year_label}_{category}_damages_health_{cr}_{rcm}'
    health_results[overall_col] = annual_damages  # No rounding here
    return health_results
