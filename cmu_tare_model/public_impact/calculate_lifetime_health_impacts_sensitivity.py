import pandas as pd
import numpy as np
from typing import Optional, Tuple  # Added for type hints

# Constants
from cmu_tare_model.constants import EQUIPMENT_SPECS, POLLUTANTS, TD_LOSSES_MULTIPLIER, CR_FUNCTIONS, RCM_MODELS

# Imports for lookup dictionaries, functions, and calculations
from cmu_tare_model.utils.modeling_params import define_scenario_params
from cmu_tare_model.utils.precompute_hdd_factors import precompute_hdd_factors
from cmu_tare_model.utils.data_validation.retrofit_status_utils import (
    create_retrofit_only_series,
    calculate_avoided_values,
    update_values_for_retrofits
)
from cmu_tare_model.utils.data_validation.data_quality_utils import (
    mask_category_specific_data,
    get_valid_calculation_mask
)
from cmu_tare_model.public_impact.data_processing.create_lookup_health_vsl_adjustment import lookup_health_vsl_adjustment
from cmu_tare_model.public_impact.data_processing.create_lookup_health_impact_county import (
    lookup_health_fossil_fuel_acs,
    lookup_health_fossil_fuel_h6c,
    lookup_health_electricity_acs,
    lookup_health_electricity_h6c,
    get_health_impact_with_fallback,
    analyze_health_impact_coverage
)
from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import calculate_fossil_fuel_emissions

def calculate_lifetime_health_impacts(
    df: pd.DataFrame,
    menu_mp: int,
    policy_scenario: str,
    base_year: int = 2024,
    df_baseline_damages: Optional[pd.DataFrame] = None,
    debug: bool = False,
    verbose: bool = False
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
        base_year (int): The base year for calculations (default is 2024).
        debug (bool): If True, enables debug mode for additional output.
        verbose (bool, optional): Whether to print detailed progress messages. Defaults to False.

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

    # Copy inclusion flags and validation columns from df_copy to df_detailed
    validation_prefixes = ["include_", "valid_tech_", "valid_fuel_"]
    validation_cols = []
    for prefix in validation_prefixes:
        validation_cols.extend([col for col in df_copy.columns if col.startswith(prefix)])
        
    for col in validation_cols:
        df_detailed[col] = df_copy[col]

    if debug and verbose:
        analyze_health_impact_coverage(
            df=df_copy,
            lookup_health_fossil_fuel_acs=lookup_health_fossil_fuel_acs,
            lookup_health_fossil_fuel_h6c=lookup_health_fossil_fuel_h6c,
            lookup_health_electricity_acs=lookup_health_electricity_acs,
            lookup_health_electricity_h6c=lookup_health_electricity_h6c,
            rcm_models=RCM_MODELS,
            cr_functions=CR_FUNCTIONS,
            verbose=True
        )
    
    # Dictionary to hold lifetime health impacts columns for each category
    lifetime_columns_data = {}

    # Add this line
    all_columns_to_mask = {category: [] for category in EQUIPMENT_SPECS}

    # Retrieve scenario-specific params for electricity/fossil-fuel emissions.
    # Note: The climate lookup and cambium scenario are ignored as they are not used in this function.
    scenario_prefix, _, lookup_emissions_fossil_fuel, _, lookup_emissions_electricity_health, _ = define_scenario_params(menu_mp, policy_scenario)

    # Precompute HDD adjustment factors by region and year
    hdd_factors_per_year = precompute_hdd_factors(df_copy)

    # Loop over each equipment category and its lifetime from EQUIPMENT_SPECS
    for category, lifetime in EQUIPMENT_SPECS.items():
        try:
            if verbose:
                print(f"Calculating Health Emissions and Damages from 2024 to {2024 + lifetime} for {category}")
            
            # UPDATED: Use new validation approach to mask data for this category
            # Determine which homes have valid data and get retrofits for this category
            valid_mask = get_valid_calculation_mask(df_copy, category, menu_mp, verbose=verbose)

            # Add a tracking array for this category
            category_columns_to_mask = []

            # Initialize with zeros for valid homes, NaN for others
            lifetime_health_damages = {
                (rcm, cr): create_retrofit_only_series(df_copy, valid_mask)
                for rcm in RCM_MODELS
                for cr in CR_FUNCTIONS
            }

            # Loop over each year in the equipment's lifetime
            for year in range(1, lifetime + 1):
                try:
                    # Calculate the calendar year label (e.g., 2024, 2025, etc.)
                    year_label = year + (base_year - 1)
                    
                    # Check that an HDD factor exists for the current year; raise KeyError if missing
                    if year_label not in hdd_factors_per_year:
                        raise KeyError(f"HDD factor for year {year_label} not found.")

                    # Retrieve HDD factor for the current year; use factor only for heating/waterHeating categories
                    hdd_factor = hdd_factors_per_year.get(year_label, pd.Series(1.0, index=df_copy.index))
                    adjusted_hdd_factor = hdd_factor if category in ['heating', 'waterHeating'] else pd.Series(1.0, index=df_copy.index)
                    
                    # Calculate fossil fuel emissions for the current category and year
                    # In calculate_lifetime_climate_impacts and calculate_lifetime_health_impacts:
                    total_fossil_fuel_emissions = calculate_fossil_fuel_emissions(
                        df=df_copy,
                        category=category,
                        adjusted_hdd_factor=adjusted_hdd_factor,
                        lookup_emissions_fossil_fuel=lookup_emissions_fossil_fuel,
                        menu_mp=menu_mp,
                        retrofit_mask=valid_mask,  # Pass the valid mask instead
                        verbose=verbose
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
                            category_columns_to_mask.append(overall_col)

                            # Accumulate the damages for the current year into the lifetime total, only for homes with retrofits
                            lifetime_health_damages[(rcm, cr)] = update_values_for_retrofits(
                                target_series=lifetime_health_damages[(rcm, cr)],
                                update_values=health_results_pair[overall_col],
                                retrofit_mask=valid_mask,
                                menu_mp=menu_mp
                            )

                            # Append the annual results to the detailed DataFrame
                            if health_results_pair:
                                df_detailed = pd.concat([df_detailed, pd.DataFrame(health_results_pair, index=df_copy.index)], axis=1)
                                # Add the column name to the category-specific tracking array
                                category_columns_to_mask.extend(health_results_pair.keys())

                except Exception as e:
                    # Raise a RuntimeError with context if an error occurs during processing for a year
                    raise RuntimeError(f"Error processing year {year_label} for category '{category}': {e}")

            # After processing all years, prepare lifetime results for the category
            lifetime_dict = {}

            # Compute avoided damages if baseline data is provided and menu_mp is non-zero
            if menu_mp != 0 and df_baseline_damages is not None:
                for rcm in RCM_MODELS:
                    for cr in CR_FUNCTIONS:
                        # Construct the column name for the lifetime damages for this (rcm, cr) pair                        
                        baseline_health_col = f'baseline_{category}_lifetime_damages_health_{rcm}_{cr}'
                        if baseline_health_col in df_baseline_damages.columns:
                            avoided_health_damages_col = f'{scenario_prefix}{category}_avoided_damages_health_{rcm}_{cr}'
                            # Calculate avoided damages only for homes with retrofits
                            lifetime_dict[avoided_health_damages_col] = calculate_avoided_values(
                                baseline_values=df_baseline_damages[baseline_health_col],
                                measure_values=lifetime_health_damages[(rcm, cr)],
                                retrofit_mask=valid_mask
                            )
                            # Append the column name to the category-specific tracking array
                            category_columns_to_mask.append(avoided_health_damages_col)

                        else:
                            if verbose:
                                print(f"Warning: Missing baseline for {category} ({cr}, {rcm}). Avoided health values skipped.")

            # Record overall lifetime damages for each (rcm, cr) pair
            for rcm in RCM_MODELS:
                for cr in CR_FUNCTIONS:
                    overall_health_col = f'{scenario_prefix}{category}_lifetime_damages_health_{rcm}_{cr}'
                    lifetime_dict[overall_health_col] = lifetime_health_damages[(rcm, cr)]
                    category_columns_to_mask.append(overall_health_col)

            # Add the lifetime results for the category to the global dictionary
            lifetime_columns_data.update(lifetime_dict)

            # Append the lifetime columns to the detailed DataFrame for completeness
            df_detailed = pd.concat([df_detailed, pd.DataFrame(lifetime_dict, index=df_copy.index)], axis=1)

            # Add all columns for this category to the masking dictionary
            all_columns_to_mask[category].extend(category_columns_to_mask)

        except Exception as e:
            # Raise a RuntimeError with context if an error occurs for a category
            raise RuntimeError(f"Error processing category '{category}': {e}")

    # Create a DataFrame from the lifetime results and merge it with the main DataFrame
    df_lifetime = pd.DataFrame(lifetime_columns_data, index=df_copy.index)
    df_main = df_copy.join(df_lifetime, how='left')
    
    # Apply final verification masking for consistency
    print("\nVerifying masking for all calculated columns:")
    for category, cols_to_mask in all_columns_to_mask.items():
        # Filter out columns that don't exist in df_main or df_detailed
        main_cols = [col for col in cols_to_mask if col in df_main.columns]
        detailed_cols = [col for col in cols_to_mask if col in df_detailed.columns]
        
        if main_cols:
            df_main = mask_category_specific_data(df_main, main_cols, category, verbose=verbose)
        if detailed_cols:
            df_detailed = mask_category_specific_data(df_detailed, detailed_cols, category, verbose=verbose)

    # Apply rounding to final results
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
            key: get_health_impact_with_fallback(lookup_msc_fossil_fuel, key, rcm, pollutant.lower())
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
            key: get_health_impact_with_fallback(lookup_msc_electricity, key, rcm, pollutant.lower())
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
