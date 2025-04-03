import pandas as pd

# Constants
from cmu_tare_model.constants import EQUIPMENT_SPECS, POLLUTANTS, TD_LOSSES_MULTIPLIER, CR_FUNCTIONS, RCM_MODELS

# Imports for lookup dictionaries, functions, and calculations
from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import calculate_fossil_fuel_emissions
from cmu_tare_model.public_impact.emissions_scenario_settings import define_scenario_settings
from cmu_tare_model.public_impact.calculations.precompute_hdd_factors import precompute_hdd_factors
from cmu_tare_model.public_impact.data_processing.create_lookup_health_impact_county import (
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

See technical documentation for additional details
======================================================================================================================
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
