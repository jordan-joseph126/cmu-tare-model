import pandas as pd
import numpy as np
from typing import Optional, Tuple  # Added for type hints

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
from cmu_tare_model.public_impact.data_processing.create_lookup_health_vsl_adjustment import lookup_health_vsl_adjustment

def calculate_lifetime_health_impacts(
    df: pd.DataFrame,
    menu_mp: int,
    policy_scenario: str,
    df_baseline_damages: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate lifetime health impacts for each equipment category over all (rcm, cr) combinations.

    This function returns a tuple of two dataframes:
      - The main dataframe (df_main) is the input dataframe updated with lifetime health damages.
      - The detailed dataframe (df_detailed) contains annual results along with lifetime breakdowns.

    Args:
        df (pd.DataFrame): Input data.
        menu_mp (int): Measure package identifier.
        policy_scenario (str): Policy scenario name that determines model inputs
            ('No Inflation Reduction Act' or 'AEO2023 Reference Case').
        df_baseline_damages (pd.DataFrame, optional): Baseline health damages data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            A tuple containing:
            - df_main (pd.DataFrame): The aggregated lifetime health impacts.
            - df_detailed (pd.DataFrame): The detailed annual health damages results.

    Raises:
        RuntimeError: If processing fails at the category or year level due to missing data or calculation errors.
        KeyError: If an HDD factor is not found for a required year.
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
        try:
            print(f"Calculating Health Emissions and Damages from 2024 to {2024 + lifetime} for {category}")

            # Reinitialize lifetime health impacts for this category for each (rcm, cr) pair
            lifetime_health_damages = {
                (rcm, cr): pd.Series(0.0, index=df_copy.index)
                for rcm in RCM_MODELS
                for cr in CR_FUNCTIONS
            }
            
            # Loop over each year in the equipment's lifetime
            for year in range(1, lifetime + 1):
                try:
                    year_label = year + 2023
                    
                    # Validate that we have an HDD factor
                    if year_label not in hdd_factors_per_year:
                        raise KeyError(f"HDD factor for year {year_label} not found.")

                    # Retrieve HDD factor for the current year; the adjusted HDD factor only applies to heating/waterHeating
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
                            
                            # Concatenate annual results to the detailed DataFrame
                            df_detailed = pd.concat([df_detailed, pd.DataFrame(health_results_pair, index=df_copy.index)], axis=1)

                except Exception as e:
                    raise RuntimeError(f"Error processing year {year_label} for category '{category}': {e}")

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

        except Exception as e:
            raise RuntimeError(f"Error processing category '{category}': {e}")

    # Create a dataframe for lifetime results and merge with the main dataframe
    df_lifetime = pd.DataFrame(lifetime_columns_data, index=df_copy.index)
    df_main = df_copy.join(df_lifetime, how='left')
    
    # Apply final rounding
    df_main = df_main.round(2)
    df_detailed = df_detailed.round(2)
    
    return df_main, df_detailed

def calculate_health_damages_for_pair(
    df: pd.DataFrame,
    category: str,
    year_label: int,
    adjusted_hdd_factor: pd.Series,
    lookup_emissions_electricity_health: dict,
    scenario_prefix: str,
    total_fossil_fuel_emissions: dict,
    menu_mp: int,
    rcm: str,
    cr: str
) -> dict:
    """
    Calculate health-related damages for a single (rcm, cr) pair for a given category and year.

    Returns a dictionary mapping column names to computed Series.
    """
    # Get the VSL adjustment factor for the current year
    if year_label not in lookup_health_vsl_adjustment:
        raise KeyError(f"VSL adjustment factor not found for year {year_label}")
    vsl_adjustment_factor = lookup_health_vsl_adjustment[year_label]
    
    # Select lookup dictionaries based on the concentration-response function
    if cr == 'acs':
        lookup_msc_fossil_fuel = lookup_health_fossil_fuel_acs
        lookup_msc_electricity = lookup_health_electricity_acs
    elif cr == 'h6c':
        lookup_msc_fossil_fuel = lookup_health_fossil_fuel_h6c
        lookup_msc_electricity = lookup_health_electricity_h6c
    else:
        raise ValueError(f"Invalid C-R function: {cr}")
    
    # Ensure a county key exists for vectorized lookups.
    if 'county_key' not in df.columns:
        df['county_key'] = list(zip(df['county_fips'], df['state']))
    
    annual_damages = pd.Series(0.0, index=df.index)
    health_results = {}
    
    for pollutant in POLLUTANTS:
        if pollutant == 'co2e':
            continue
        
        # Validate fossil fuel emissions exist for the pollutant
        if pollutant not in total_fossil_fuel_emissions:
            raise KeyError(f"Fossil fuel emissions not found for pollutant '{pollutant}'")
        fossil_fuel_emissions = total_fossil_fuel_emissions[pollutant]
        
        # --- FOSSIL FUEL MSC Lookup (Vectorized) ---
        unique_counties = df['county_key'].unique()
        msc_lookup_fossil = {
            key: lookup_msc_fossil_fuel.get(key, {}).get(rcm, {}).get(pollutant.lower(), np.nan)
            for key in unique_counties
        }
        fossil_fuel_msc = df['county_key'].map(msc_lookup_fossil)
        fossil_fuel_msc_adjusted = fossil_fuel_msc * vsl_adjustment_factor
        fossil_fuel_damages = fossil_fuel_emissions * fossil_fuel_msc_adjusted
        
        # --- ELECTRICITY Emissions Factor Lookup (Vectorized) ---
        region_mapping = {}
        for (yr, region), data in lookup_emissions_electricity_health.items():
            if yr == year_label:
                key_name = f"delta_egrid_{pollutant.lower()}"
                region_mapping[region] = data.get(key_name, np.nan)
        electricity_emissions_factor = df['gea_region'].map(region_mapping)
        
        # Retrieve electricity consumption based on menu_mp
        if menu_mp == 0:
            col_name = f'base_electricity_{category}_consumption'
            if col_name not in df.columns:
                raise KeyError(f"Required column '{col_name}' is missing from the DataFrame.")
            electricity_consumption = df[col_name].fillna(0) * adjusted_hdd_factor
        else:
            col_name = f'mp{menu_mp}_{year_label}_{category}_consumption'
            if col_name not in df.columns:
                raise KeyError(f"Required column '{col_name}' is missing from the DataFrame.")
            electricity_consumption = df[col_name].fillna(0)
        electricity_emissions = electricity_consumption * TD_LOSSES_MULTIPLIER * electricity_emissions_factor
        
        # --- ELECTRICITY MSC Lookup (Vectorized) ---
        msc_lookup_electricity = {
            key: lookup_msc_electricity.get(key, {}).get(rcm, {}).get(pollutant.lower(), np.nan)
            for key in unique_counties
        }
        electricity_msc = df['county_key'].map(msc_lookup_electricity)
        electricity_msc_adjusted = electricity_msc * vsl_adjustment_factor
        electricity_damages = electricity_emissions * electricity_msc_adjusted
        
        # Combine fossil fuel and electricity damages for this pollutant
        total_damages = fossil_fuel_damages + electricity_damages
        col_name = f'{scenario_prefix}{year_label}_{category}_damages_{pollutant}_{cr}_{rcm}'
        health_results[col_name] = total_damages
        
        # Accumulate annual damages across pollutants
        annual_damages += total_damages

    overall_col = f'{scenario_prefix}{year_label}_{category}_damages_health_{cr}_{rcm}'
    health_results[overall_col] = annual_damages
    return health_results


# def calculate_health_damages_for_pair(
#     df: pd.DataFrame,
#     category: str,
#     year_label: int,
#     adjusted_hdd_factor: pd.Series,
#     lookup_emissions_electricity_health: dict,
#     scenario_prefix: str,
#     total_fossil_fuel_emissions: dict,
#     menu_mp: int,
#     rcm: str,
#     cr: str
# ) -> dict:
#     """
#     Calculate health-related damages for a single (rcm, cr) pair for a given category and year.

#     Returns a dictionary mapping column names to computed Series.

#     Args:
#         df (pd.DataFrame): Main DataFrame containing region and consumption data.
#         category (str): The equipment category, e.g. 'heating' or 'waterHeating'.
#         year_label (int): The year for which damages are being calculated (e.g. 2024).
#         adjusted_hdd_factor (pd.Series): The HDD adjustment factor for heating/waterHeating.
#         lookup_emissions_electricity_health (dict): Lookup for electricity-based emissions factors.
#         scenario_prefix (str): Prefix for naming output columns (e.g. 'baseline_' or 'preIRA_mp1_').
#         total_fossil_fuel_emissions (dict): Mapping pollutant -> pd.Series of emissions values.
#         menu_mp (int): Measure package identifier.
#         rcm (str): Regional climate model name (e.g. 'rcm1').
#         cr (str): Concentration-response function identifier (e.g. 'acs' or 'h6c').

#     Returns:
#         dict: A dictionary whose keys are column names (str) and values are pd.Series of calculated damages.

#     Raises:
#         KeyError: If required data is missing from the lookups or DataFrame (e.g., emissions factor, VSL factor).
#         ValueError: If an invalid C-R function is provided.
#     """
#     def get_emissions_factor(row, pollutant, lookup_dict, year_value):
#         # Use the provided year_value instead of row['year']
#         key = (year_value, row['gea_region'])
#         if not pollutant.startswith("delta_egrid_"):
#             pollutant = "delta_egrid_" + pollutant.lower()
#         if key not in lookup_dict:
#             raise KeyError(f"No emissions factor info for region/year key {key} in lookup.")
#         if pollutant not in lookup_dict[key]:
#             raise KeyError(f"No emissions factor for pollutant '{pollutant}' in key {key}.")
#         return lookup_dict[key][pollutant]
    
#     def get_msc_value(row, pollutant, lookup_dict, rcm):
#         """
#         Safely retrieve the marginal social cost for a given county, RCM, and pollutant.

#         If missing, return np.nan and print a warning (or silently do so).
#         """
#         county_key = (row['county_fips'], row['state'])

#         # Convert everything to a consistent case. Let's assume lowercase:
#         pol_lower = pollutant.lower()

#         try:
#             return lookup_dict[county_key][rcm][pol_lower]
#         except KeyError:
#             # Optional: Log or print a warning. For example:
#             # print(f"Warning: Missing MSC data for (rcm={rcm}, pollutant={pol_lower}, county={county_key}). Using NaN.")
#             # Return a default
#             return np.nan

#     def get_vsl_adjustment_factor(lookup_dict, year):
#         if year not in lookup_dict:
#             raise KeyError(f"VSL adjustment factor not found for year {year}")
#         return lookup_dict[year]
        
#     # Get the VSL adjustment factor for the current year
#     vsl_adjustment_factor = get_vsl_adjustment_factor(lookup_health_vsl_adjustment, year_label)
    
#     # Select lookup dictionaries based on the concentration-response function
#     if cr == 'acs':
#         lookup_msc_fossil_fuel = lookup_health_fossil_fuel_acs
#         lookup_msc_electricity = lookup_health_electricity_acs
#     elif cr == 'h6c':
#         lookup_msc_fossil_fuel = lookup_health_fossil_fuel_h6c
#         lookup_msc_electricity = lookup_health_electricity_h6c
#     else:
#         raise ValueError(f"Invalid C-R function: {cr}")
    
#     annual_damages = pd.Series(0.0, index=df.index)
#     health_results = {}
    
#     for pollutant in POLLUTANTS:
#         if pollutant == 'co2e':
#             continue
        
#         # ==================================================================
#         # FOSSIL FUEL
#         # ==================================================================
#         if pollutant not in total_fossil_fuel_emissions:
#             raise KeyError(f"Fossil fuel emissions not found for pollutant '{pollutant}'")
#         fossil_fuel_emissions = total_fossil_fuel_emissions[pollutant]

#         # Obtain the MSC (GROUND) and multiply by VSL factor
#         fossil_fuel_msc = df.apply(lambda row: get_msc_value(row, pollutant, lookup_msc_fossil_fuel, rcm), axis=1)
#         fossil_fuel_msc_adjusted = fossil_fuel_msc * vsl_adjustment_factor
#         fossil_fuel_damages = fossil_fuel_emissions * fossil_fuel_msc_adjusted
        
#         # ==================================================================
#         # ELECTRICITY
#         # ==================================================================
#         electricity_emissions_factor = df.apply(lambda row: get_emissions_factor(row, pollutant, lookup_emissions_electricity_health, year_label), axis=1)
#         if menu_mp == 0:
#             col_name = f'base_electricity_{category}_consumption'
#             if col_name not in df.columns:
#                 raise KeyError(f"Required column '{col_name}' is missing from the DataFrame.")
#             electricity_consumption = df[col_name].fillna(0)
#             # The baseline_{year_label}_{category}_consumption columns are the projected consumption values
#             electricity_consumption *= adjusted_hdd_factor
#         else:
#             col_name = f'mp{menu_mp}_{year_label}_{category}_consumption'
#             if col_name not in df.columns:
#                 raise KeyError(f"Required column '{col_name}' is missing from the DataFrame.")
#             electricity_consumption = df[col_name].fillna(0)
#         electricity_emissions = electricity_consumption * TD_LOSSES_MULTIPLIER * electricity_emissions_factor
        
#         # Obtain the MSC (ELEVATED) and multiply by VSL factor
#         electricity_msc = df.apply(lambda row: get_msc_value(row, pollutant, lookup_msc_electricity, rcm), axis=1)
#         electricity_msc_adjusted = electricity_msc * vsl_adjustment_factor
#         electricity_damages = electricity_emissions * electricity_msc_adjusted
        
#         # Combine fossil fuel + electricity damages for this pollutant
#         total_damages = fossil_fuel_damages + electricity_damages
#         col_name = f'{scenario_prefix}{year_label}_{category}_damages_{pollutant}_{cr}_{rcm}'
#         health_results[col_name] = total_damages
        
#         # Accumulate the annual damages for all pollutants
#         annual_damages += total_damages
    
#     overall_col = f'{scenario_prefix}{year_label}_{category}_damages_health_{cr}_{rcm}'
#     health_results[overall_col] = annual_damages
#     return health_results
