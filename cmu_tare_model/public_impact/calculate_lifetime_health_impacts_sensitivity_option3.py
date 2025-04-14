import pandas as pd
import numpy as np
from typing import Optional, Tuple  # Added for type hints

# Constants
from cmu_tare_model.constants import EQUIPMENT_SPECS, POLLUTANTS, TD_LOSSES_MULTIPLIER, CR_FUNCTIONS, RCM_MODELS

# Imports for lookup dictionaries, functions, and calculations
from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import calculate_fossil_fuel_emissions
from cmu_tare_model.utils.modeling_params import define_scenario_params
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
    """Calculate lifetime health impacts for each equipment category over all (rcm, cr) combinations.

    This function calculates lifetime health damages for each equipment category by iterating over each year of the
    equipment's lifetime and each (rcm, cr) combination. It returns a tuple of two DataFrames:
      - The main DataFrame updated with lifetime health damages.
      - A detailed DataFrame containing annual results with lifetime breakdowns.

    Args:
        df (pd.DataFrame): Input data.
        menu_mp (int): Measure package identifier.
        policy_scenario (str): Policy scenario name that determines model inputs
            ('No Inflation Reduction Act' or 'AEO2023 Reference Case').
        df_baseline_damages (Optional[pd.DataFrame]): Baseline health damages data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - df_main (pd.DataFrame): The aggregated lifetime health impacts.
            - df_detailed (pd.DataFrame): The detailed annual health damages results.

    Raises:
        RuntimeError: If processing fails at the category or year level due to missing data or calculation errors.
        KeyError: If an HDD factor is not found for a required year.
    """
    # Create a copy of the input DataFrame to avoid modifying the original data
    df_copy = df.copy()
    # Initialize the detailed DataFrame with the same index as df_copy
    df_detailed = pd.DataFrame(index=df_copy.index)

    # Dictionary to hold lifetime health impacts columns for each category
    lifetime_columns_data = {}

    # Retrieve scenario-specific params for electricity/fossil-fuel emissions.
    # Note: The climate lookup and cambium scenario are ignored as they are not used in this function.
    scenario_prefix, _, lookup_emissions_fossil_fuel, _, lookup_emissions_electricity_health, _ = define_scenario_params(menu_mp, policy_scenario)

    # Precompute HDD adjustment factors by region and year
    hdd_factors_per_year = precompute_hdd_factors(df_copy)

    # Loop over each equipment category and its lifetime from EQUIPMENT_SPECS
    for category, lifetime in EQUIPMENT_SPECS.items():
        try:
            print(f"Calculating Health Emissions and Damages from 2024 to {2024 + lifetime} for {category}")

            # Initialize lifetime health damages for each (rcm, cr) pair with zeros
            lifetime_health_damages = {
                (rcm, cr): pd.Series(0.0, index=df_copy.index)
                for rcm in RCM_MODELS
                for cr in CR_FUNCTIONS
            }
            
            # Loop over each year in the equipment's lifetime
            for year in range(1, lifetime + 1):
                try:
                    # Calculate the calendar year label (e.g., 2024, 2025, etc.)
                    year_label = year + 2023
                    
                    # Check that an HDD factor exists for the current year; raise KeyError if missing
                    if year_label not in hdd_factors_per_year:
                        raise KeyError(f"HDD factor for year {year_label} not found.")

                    # Retrieve HDD factor for the current year; use factor only for heating/waterHeating categories
                    hdd_factor = hdd_factors_per_year.get(year_label, pd.Series(1.0, index=df_copy.index))
                    adjusted_hdd_factor = hdd_factor if category in ['heating', 'waterHeating'] else pd.Series(1.0, index=df_copy.index)
                    
                    # Calculate fossil fuel emissions for the current category and year
                    total_fossil_fuel_emissions = calculate_fossil_fuel_emissions(
                        df_copy, category, adjusted_hdd_factor, lookup_emissions_fossil_fuel, menu_mp
                    )
                    
                    # For each (rcm, cr) combination, compute health damages for this year and accumulate lifetime values
                    for rcm in RCM_MODELS:
                        for cr in CR_FUNCTIONS:
                            # Compute health damages for the current (rcm, cr) pair using the helper function
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
                            # Construct the overall column name for the annual damages for this pair
                            overall_col = f'{scenario_prefix}{year_label}_{category}_damages_health_{rcm}_{cr}'
                            # Accumulate the damages for the current year into the lifetime total
                            lifetime_health_damages[(rcm, cr)] += health_results_pair[overall_col]
                            
                            # Append the annual results to the detailed DataFrame
                            df_detailed = pd.concat([df_detailed, pd.DataFrame(health_results_pair, index=df_copy.index)], axis=1)

                except Exception as e:
                    # Raise a RuntimeError with context if an error occurs during processing for a year
                    raise RuntimeError(f"Error processing year {year_label} for category '{category}': {e}")

            # After processing all years, prepare lifetime results for the category
            lifetime_dict = {}
            # Compute avoided damages if baseline data is provided and menu_mp is non-zero
            if menu_mp != 0 and df_baseline_damages is not None:
                for rcm in RCM_MODELS:
                    for cr in CR_FUNCTIONS:
                        baseline_health_col = f'baseline_{category}_lifetime_damages_health_{rcm}_{cr}'
                        if baseline_health_col in df_baseline_damages.columns:
                            avoided_health_damages_col = f'{scenario_prefix}{category}_avoided_damages_health_{rcm}_{cr}'
                            # Calculate avoided damages by subtracting computed damages from baseline values
                            lifetime_dict[avoided_health_damages_col] = df_baseline_damages[baseline_health_col] - lifetime_health_damages[(rcm, cr)]
                        else:
                            print(f"Warning: Missing baseline for {category} ({cr}, {rcm}). Avoided health values skipped.")
            # Record overall lifetime damages for each (rcm, cr) pair
            for rcm in RCM_MODELS:
                for cr in CR_FUNCTIONS:
                    overall_health_col = f'{scenario_prefix}{category}_lifetime_damages_health_{rcm}_{cr}'
                    lifetime_dict[overall_health_col] = lifetime_health_damages[(rcm, cr)]
            
            # Add the lifetime results for the category to the global dictionary
            lifetime_columns_data.update(lifetime_dict)
            # Append the lifetime columns to the detailed DataFrame for completeness
            df_detailed = pd.concat([df_detailed, pd.DataFrame(lifetime_dict, index=df_copy.index)], axis=1)

        except Exception as e:
            # Raise a RuntimeError with context if an error occurs for a category
            raise RuntimeError(f"Error processing category '{category}': {e}")

    # UPDATED CODE TO FIX FRAGMENTATION ISSUES
    df_lifetime = pd.DataFrame(lifetime_columns_data, index=df_copy.index)

    if df_baseline_damages is not None:
        # Start with baseline DataFrame - this is correct for your workflow
        df_main = df_baseline_damages.copy()
        
        # Add all lifetime columns at once instead of one-by-one (fixes fragmentation)
        df_main = pd.concat([df_main, df_lifetime], axis=1, copy=False)
        
        # Get any essential columns from df_copy that aren't in df_main
        # This preserves your existing functionality but does it efficiently
        missing_cols = set(df_copy.columns) - set(df_main.columns)
        if missing_cols:
            df_main = pd.concat([df_main, df_copy[list(missing_cols)]], axis=1, copy=False)
    else:
        # Original behavior for baseline case
        df_main = df_copy.join(df_lifetime, how='left') 

    # Round final results
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
    """Calculate health-related damages for a single (rcm, cr) pair for a given category and year.

    This function computes the health damages by performing vectorized lookups for fossil fuel and electricity-related 
    marginal social costs (MSC) and emissions factors. It returns a dictionary mapping column names to computed 
    damage Series.

    Args:
        df (pd.DataFrame): Main DataFrame containing region and consumption data.
        category (str): Equipment category (e.g., 'heating' or 'waterHeating').
        year_label (int): Year for which damages are being calculated (e.g., 2024).
        adjusted_hdd_factor (pd.Series): HDD adjustment factor for heating/waterHeating.
        lookup_emissions_electricity_health (dict): Lookup for electricity-based emissions factors.
        scenario_prefix (str): Prefix for naming output columns (e.g., 'baseline_' or 'preIRA_mp1_').
        total_fossil_fuel_emissions (dict): Mapping of pollutant to Series of fossil fuel emissions.
        menu_mp (int): Measure package identifier.
        rcm (str): Regional climate model identifier (e.g., 'rcm1').
        cr (str): Concentration-response function identifier (e.g., 'acs' or 'h6c').

    Returns:
        dict: A dictionary where keys are column names (str) and values are pd.Series representing calculated damages.

    Raises:
        KeyError: If required lookup data (e.g., emissions factor or VSL adjustment factor) is missing.
        ValueError: If an invalid concentration-response function is provided.
    """
    # Retrieve the VSL adjustment factor for the current year; raise KeyError if not found
    if year_label not in lookup_health_vsl_adjustment:
        raise KeyError(f"VSL adjustment factor not found for year {year_label}")
    vsl_adjustment_factor = lookup_health_vsl_adjustment[year_label]
    
    # Select lookup dictionaries based on the provided concentration-response function
    if cr == 'acs':
        lookup_msc_fossil_fuel = lookup_health_fossil_fuel_acs
        lookup_msc_electricity = lookup_health_electricity_acs
    elif cr == 'h6c':
        lookup_msc_fossil_fuel = lookup_health_fossil_fuel_h6c
        lookup_msc_electricity = lookup_health_electricity_h6c
    else:
        raise ValueError(f"Invalid C-R function: {cr}")
    
    # Ensure a county key exists for vectorized lookups; create one if missing
    if 'county_key' not in df.columns:
        df['county_key'] = list(zip(df['county_fips'], df['state']))
    
    # Initialize a Series to accumulate annual damages and a dictionary for storing results
    annual_damages = pd.Series(0.0, index=df.index)
    health_results = {}
    
    # Loop through each pollutant defined in POLLUTANTS
    for pollutant in POLLUTANTS:
        if pollutant == 'co2e':
            continue
        
        # Ensure fossil fuel emissions exist for the pollutant; raise KeyError if missing
        if pollutant not in total_fossil_fuel_emissions:
            raise KeyError(f"Fossil fuel emissions not found for pollutant '{pollutant}'")
        fossil_fuel_emissions = total_fossil_fuel_emissions[pollutant]
        
        # --- FOSSIL FUEL MSC Lookup (Vectorized) ---
        # Get unique county keys for an efficient vectorized lookup
        unique_counties = df['county_key'].unique()
        msc_lookup_fossil = {
            key: lookup_msc_fossil_fuel.get(key, {}).get(rcm, {}).get(pollutant.lower(), np.nan)
            for key in unique_counties
        }
        # Map the lookup dictionary to the county_key column to retrieve MSC values
        fossil_fuel_msc = df['county_key'].map(msc_lookup_fossil)
        # Adjust the MSC values using the VSL adjustment factor
        fossil_fuel_msc_adjusted = fossil_fuel_msc * vsl_adjustment_factor
        # Calculate fossil fuel damages by multiplying emissions with adjusted MSC values
        fossil_fuel_damages = fossil_fuel_emissions * fossil_fuel_msc_adjusted
        
        # --- ELECTRICITY Emissions Factor Lookup (Vectorized) ---
        # Build a mapping from region to its emissions factor for the current pollutant and year
        region_mapping = {}
        for (yr, region), data in lookup_emissions_electricity_health.items():
            if yr == year_label:
                key_name = f"delta_egrid_{pollutant.lower()}"
                region_mapping[region] = data.get(key_name, np.nan)
        # Map the region mapping to the 'gea_region' column to retrieve emissions factors
        electricity_emissions_factor = df['gea_region'].map(region_mapping)
        
        # Retrieve electricity consumption based on the menu_mp value
        if menu_mp == 0:
            col_name = f'base_electricity_{category}_consumption'
            if col_name not in df.columns:
                raise KeyError(f"Required column '{col_name}' is missing from the DataFrame.")
            # Fill missing values with 0 and apply the adjusted HDD factor
            electricity_consumption = df[col_name].fillna(0) * adjusted_hdd_factor
        else:
            col_name = f'mp{menu_mp}_{year_label}_{category}_consumption'
            if col_name not in df.columns:
                raise KeyError(f"Required column '{col_name}' is missing from the DataFrame.")
            electricity_consumption = df[col_name].fillna(0)
        # Calculate electricity emissions by accounting for TD losses and emissions factors
        electricity_emissions = electricity_consumption * TD_LOSSES_MULTIPLIER * electricity_emissions_factor
        
        # --- ELECTRICITY MSC Lookup (Vectorized) ---
        # Build a lookup dictionary for electricity MSC values
        msc_lookup_electricity = {
            key: lookup_msc_electricity.get(key, {}).get(rcm, {}).get(pollutant.lower(), np.nan)
            for key in unique_counties
        }
        # Map the lookup dictionary to retrieve electricity MSC values
        electricity_msc = df['county_key'].map(msc_lookup_electricity)
        # Adjust the MSC values using the VSL adjustment factor
        electricity_msc_adjusted = electricity_msc * vsl_adjustment_factor
        # Calculate electricity damages by multiplying emissions with adjusted MSC values
        electricity_damages = electricity_emissions * electricity_msc_adjusted
        
        # Combine fossil fuel and electricity damages for the current pollutant
        total_damages = fossil_fuel_damages + electricity_damages
        # Construct the column name for the pollutant-specific damages
        col_name = f'{scenario_prefix}{year_label}_{category}_damages_{pollutant}_{rcm}_{cr}'
        health_results[col_name] = total_damages
        
        # Accumulate the annual damages across pollutants
        annual_damages += total_damages

    # Store the overall annual damages under a summary column name
    overall_col = f'{scenario_prefix}{year_label}_{category}_damages_health_{rcm}_{cr}'
    health_results[overall_col] = annual_damages
    return health_results
