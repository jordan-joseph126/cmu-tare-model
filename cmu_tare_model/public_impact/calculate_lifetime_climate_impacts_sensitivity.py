import pandas as pd
from typing import Optional, Tuple, Dict

from cmu_tare_model.constants import EQUIPMENT_SPECS, TD_LOSSES_MULTIPLIER, MER_TYPES, SCC_ASSUMPTIONS
from cmu_tare_model.public_impact.data_processing.create_lookup_climate_impact_scc import lookup_climate_impact_scc
from cmu_tare_model.public_impact.calculations.precompute_hdd_factors import precompute_hdd_factors
from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import calculate_fossil_fuel_emissions
from cmu_tare_model.utils.modeling_params import define_scenario_params
from cmu_tare_model.utils.retrofit_error_handling_utils import (
    determine_retrofit_status,
    initialize_npv_series,
    calculate_avoided_values,
    update_values_for_retrofits
)

def calculate_lifetime_climate_impacts(
    df: pd.DataFrame,
    menu_mp: int,
    policy_scenario: str,
    base_year: int = 2024,
    df_baseline_damages: Optional[pd.DataFrame] = None,
    verbose: bool = False    
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate lifetime climate impacts (CO2e emissions and climate damages) for each
    equipment category across all (mer_type, scc_value) combinations.

    This function processes each equipment category over its lifetime, computing annual
    and lifetime climate emissions/damages. Results are combined into two DataFrames:
    a main summary (df_main) and a detailed annual breakdown (df_detailed).

    Args:
        df (pd.DataFrame): Input DataFrame containing equipment consumption data, region info, etc.
        menu_mp (int): Measure package identifier (0 for baseline, nonzero for different scenarios).
        policy_scenario (str): Determines emissions scenario inputs (e.g., 'No Inflation Reduction Act' or 'AEO2023 Reference Case').
        df_baseline_damages (pd.DataFrame, optional): Baseline damages for computing avoided emissions/damages.
        verbose (bool, optional): Whether to print detailed progress messages. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - df_main: Main summary of lifetime climate impacts (rounded to 2 decimals).
            - df_detailed: Detailed annual and lifetime results (rounded to 2 decimals).

    Raises:
        RuntimeError: If processing fails at the category or year level (e.g., missing data or key lookups).
    """
    # Create a copy of the input df
    # Then initialize the detailed dataframe (df_copy will become df_main)
    df_copy = df.copy()
    df_detailed = pd.DataFrame(index=df_copy.index)
    
    # Initialize a dictionary to store lifetime climate impacts columns
    lifetime_columns_data = {}

    # Retrieve scenario-specific params for electricity/fossil-fuel emissions
    # Ignored (underscored) the health lookup as it's not used in this function
    scenario_prefix, cambium_scenario, lookup_emissions_fossil_fuel, lookup_emissions_electricity_climate, _, _ = define_scenario_params(menu_mp, policy_scenario)

    # Precompute HDD adjustment factors by region and year
    hdd_factors_per_year = precompute_hdd_factors(df_copy)

    # Loop over each equipment category and its lifetime
    for category, lifetime in EQUIPMENT_SPECS.items():
        # Use try-except to wrap the entire category’s processing
        # so that we can raise an error message that includes the category info if something fails.
        try:
            if verbose:
                print(f"Calculating Climate Emissions and Damages from 2024 to {2024 + lifetime} for {category}")                    
            
            # Determine which homes get retrofits for this category - do this once per category
            retrofit_mask = determine_retrofit_status(df_copy, category, menu_mp, verbose=verbose)

            # Initialize with zeros for homes with retrofits, NaN for others
            lifetime_climate_emissions = {
                mer_type: initialize_npv_series(df_copy, retrofit_mask)
                for mer_type in MER_TYPES
            }
            lifetime_climate_damages = {
                (mer_type, scc_value): initialize_npv_series(df_copy, retrofit_mask)
                for mer_type in MER_TYPES
                for scc_value in SCC_ASSUMPTIONS
            }

            # Loop over each year in the equipment's lifetime
            for year in range(1, lifetime + 1):
                # Try-except here helps isolate issues specific to a given year.
                try:
                    # Calculate the calendar year label (e.g., 2024, 2025, etc.)
                    year_label = year + (base_year - 1)
                    
                    # Retrieve HDD factor for the current year; raise exception if missing
                    if year_label not in hdd_factors_per_year:
                        raise KeyError(f"HDD factor for year {year_label} not found.")
                    hdd_factor = hdd_factors_per_year[year_label]

                    # The adjusted HDD factor only applies to heating/waterHeating categories
                    # For other categories, use a default value of 1.0
                    adjusted_hdd_factor = hdd_factor if category in ['heating', 'waterHeating'] else pd.Series(1.0, index=df_copy.index)
                    
                    # Calculate fossil fuel emissions for the current category and year
                    # In calculate_lifetime_climate_impacts and calculate_lifetime_health_impacts:
                    total_fossil_fuel_emissions = calculate_fossil_fuel_emissions(
                        df=df_copy,
                        category=category,
                        adjusted_hdd_factor=adjusted_hdd_factor,
                        lookup_emissions_fossil_fuel=lookup_emissions_fossil_fuel,
                        menu_mp=menu_mp,
                        retrofit_mask=retrofit_mask,  # Pass the pre-computed mask
                        verbose=verbose
                    ) 

                    # Compute climate emissions and damages with scc_value sensitivities
                    climate_results, annual_emissions, annual_damages = calculate_climate_emissions_and_damages(
                        df=df_copy,
                        category=category,
                        year_label=year_label,
                        adjusted_hdd_factor=adjusted_hdd_factor,
                        lookup_emissions_electricity_climate=lookup_emissions_electricity_climate,
                        cambium_scenario=cambium_scenario,
                        total_fossil_fuel_emissions=total_fossil_fuel_emissions,
                        scenario_prefix=scenario_prefix,
                        menu_mp=menu_mp
                    )

                    # Accumulate annual emissions (scc_value-independent)
                    for mer_type in MER_TYPES:
                        lifetime_climate_emissions[mer_type] = update_values_for_retrofits(
                            lifetime_climate_emissions[mer_type],
                            annual_emissions.get(mer_type, 0.0),
                            retrofit_mask,
                            menu_mp
                        )

                    # Accumulate annual damages for each scc_value assumption
                    for key, value in annual_damages.items():
                        lifetime_climate_damages[key] = update_values_for_retrofits(
                            lifetime_climate_damages[key],
                            value,
                            retrofit_mask,
                            menu_mp
                        )

                    # If there are results, attach them to the detailed DataFrame
                    if climate_results:
                        df_detailed = pd.concat([df_detailed, pd.DataFrame(climate_results, index=df_copy.index)], axis=1)
                except Exception as e:
                    # Convert any exception into a RuntimeError with additional context
                    raise RuntimeError(f"Error processing year {year_label} for category '{category}': {e}")

            # Prepare lifetime columns
            lifetime_dict = {}
            for mer_type in MER_TYPES:
                emissions_col = f'{scenario_prefix}{category}_lifetime_mt_co2e_{mer_type}'
                lifetime_dict[emissions_col] = lifetime_climate_emissions[mer_type]
                
                for scc_assumption in SCC_ASSUMPTIONS:
                    damages_col = f'{scenario_prefix}{category}_lifetime_damages_climate_{mer_type}_{scc_assumption}'
                    lifetime_dict[damages_col] = lifetime_climate_damages[(mer_type, scc_assumption)]
                                   
                    # Calculate avoided damages if baseline data is provided
                    if menu_mp != 0 and df_baseline_damages is not None:
                        baseline_damages_col = f'baseline_{category}_lifetime_damages_climate_{mer_type}_{scc_assumption}'
                        avoided_damages_col = f'{scenario_prefix}{category}_avoided_damages_climate_{mer_type}_{scc_assumption}'
                        
                        # Calculate avoided damages only for homes with retrofits
                        lifetime_dict[avoided_damages_col] = calculate_avoided_values(
                            df_baseline_damages[baseline_damages_col],
                            lifetime_dict[damages_col],
                            retrofit_mask
                        )

                # Calculate avoided emissions if baseline data is provided
                if menu_mp != 0 and df_baseline_damages is not None:
                    baseline_emissions_col = f'baseline_{category}_lifetime_mt_co2e_{mer_type}'
                    avoided_emissions_col = f'{scenario_prefix}{category}_avoided_mt_co2e_{mer_type}'
                    
                    # Calculate avoided emissions only for homes with retrofits
                    lifetime_dict[avoided_emissions_col] = calculate_avoided_values(
                        df_baseline_damages[baseline_emissions_col],
                        lifetime_dict[emissions_col],
                        retrofit_mask
                    )

            # Store in global lifetime dictionary
            lifetime_columns_data.update(lifetime_dict)
            # Append these columns to df_detailed for completeness
            df_detailed = pd.concat([df_detailed, pd.DataFrame(lifetime_dict, index=df_copy.index)], axis=1)

        except Exception as e:
            # Convert any exception into a RuntimeError with additional context
            raise RuntimeError(f"Error processing category '{category}': {e}")

    # Create a dataframe for lifetime results and merge with the main dataframe
    df_lifetime = pd.DataFrame(lifetime_columns_data, index=df_copy.index)
    df_main = df_copy.join(df_lifetime, how='left')
    
    # Round final results
    df_main = df_main.round(2)
    df_detailed = df_detailed.round(2)
    
    return df_main, df_detailed

def calculate_climate_emissions_and_damages(
    df: pd.DataFrame,
    category: str,
    year_label: int,
    adjusted_hdd_factor: pd.Series,
    lookup_emissions_electricity_climate: dict,
    cambium_scenario: str,
    total_fossil_fuel_emissions: dict,
    scenario_prefix: str,
    menu_mp: int
) -> Tuple[dict, dict, dict]:
    """
    Calculate climate-related emissions (CO2e) and damages for a given category/year.

    This function looks up electricity emission factors (LRMER/SRMER) for the specified
    region and year, calculates annual CO2e from electricity plus fossil fuel usage,
    and multiplies by an SCC (Social Cost of Carbon) value to estimate damages.

    Args:
        df (pd.DataFrame): DataFrame containing consumption data and region info.
        category (str): Equipment category (e.g., 'heating', 'waterHeating').
        year_label (int): The calendar year (e.g., 2024).
        adjusted_hdd_factor (pd.Series): Heating degree-day factors for the current category/year.
        lookup_emissions_electricity_climate (dict): Nested dict with CO2e factors for electricity usage.
        cambium_scenario (str): Label identifying the emissions scenario.
        total_fossil_fuel_emissions (dict): Fossil fuel CO2e amounts (keyed by pollutant).
        scenario_prefix (str): Prefix for output column naming.
        menu_mp (int): Measure package identifier (0 for baseline, nonzero for a measure scenario).

    Returns:
        Tuple[dict, dict, dict]:
            - dict: Annual columns of climate emissions/damages, keyed by output column names.
            - dict: Annual climate emissions by mer_type type (for aggregation).
            - dict: Annual climate damages by (mer_type type, SCC assumption).

    Raises:
        KeyError: If emission factors for a specific region/year are missing.
        ValueError: If the required consumption column does not exist in the DataFrame.
    """
    # Results dictionaries (no rounding here)
    climate_results = {}
    annual_climate_emissions = {}
    annual_climate_damages = {}
    
    # Extract total fossil fuel CO2e from the input dictionary
    total_fossil_fuel_emissions_co2e = total_fossil_fuel_emissions['co2e']
    
    def get_emission_factor_lrmer(region: str) -> float:
        # Retrieve LRMER factor based on the region and year. If not found, raise an exception.
        region_data = lookup_emissions_electricity_climate.get((cambium_scenario, region))
        if not region_data or year_label not in region_data or 'lrmer_mt_per_kWh_co2e' not in region_data[year_label]:
            raise KeyError(f"Emission factor for LRMER not found for region '{region}' in year {year_label}.")
        
        # Otherwise, return the LRMER factor for the specified region and year
        return region_data[year_label]['lrmer_mt_per_kWh_co2e']

    def get_emission_factor_srmer(region: str) -> float:
        # Retrieve SRMER factor based on the region and year. Raise an exception if not found.
        region_data = lookup_emissions_electricity_climate.get((cambium_scenario, region))
        if not region_data or year_label not in region_data or 'srmer_mt_per_kWh_co2e' not in region_data[year_label]:
            raise KeyError(f"Emission factor for SRMER not found for region '{region}' in year {year_label}.")
        
        # Otherwise, return the SRMER factor for the specified region and year
        return region_data[year_label]['srmer_mt_per_kWh_co2e']

    def get_scc_value(year_label: int, scc_assumption: str, lookup_climate_impact_scc: dict) -> float:
        """
        Retrieve the SCC value for the given year and assumption ('lower', 'central', 'upper').

        Args:
            year_label (int): The year for which we need the SCC value.
            scc_assumption (str): 'lower', 'central', or 'upper'.
            lookup_climate_impact_scc (dict): Nested dict with structure:
                lookup_climate_impact_scc[scc_assumption][year_label] -> float SCC value.

        Returns:
            float: The SCC value for the specified year and assumption.

        Raises:
            KeyError: If the specified year is not in the lookup for that assumption.
        
        Notes: 
            In the older code, it clamps to the maximum available year if the exact year is not found in the lookup.

        """
        # Check if the year exists; if not, raise an exception
        if year_label not in lookup_climate_impact_scc[scc_assumption]:
            raise KeyError(f"SCC value for year {year_label} with assumption '{scc_assumption}' not found.")
        
        # Otherwise, return the SCC value for the specified year and assumption
        return lookup_climate_impact_scc[scc_assumption][year_label]

    # Map each row's region to the appropriate LRMER or SRMER factor
    mer_factors = {
        'lrmer': df['gea_region'].map(get_emission_factor_lrmer),
        'srmer': df['gea_region'].map(get_emission_factor_srmer)
    }
    
    # Determine electricity consumption column based on menu_mp and apply the HDD adjustment factor (for relevant categories).
    if menu_mp == 0:
        consumption_col = f'base_electricity_{category}_consumption'
    else:
        consumption_col = f'mp{menu_mp}_{year_label}_{category}_consumption'
    
    # Check if the column exists in the DataFrame. If not, raise an exception.
    if consumption_col not in df.columns:
        raise ValueError(f"Required column '{consumption_col}' not found in the input DataFrame.")

    # Adjust electricity consumption by the HDD factor for heating/waterHeating
    electricity_consumption = df[consumption_col] * adjusted_hdd_factor

    # Calculate annual emissions for each mer_type type
    for mer_type in MER_TYPES:
        # Multiply by transmission/distribution losses
        annual_emissions_electricity = electricity_consumption * TD_LOSSES_MULTIPLIER * mer_factors[mer_type]
        # Combine fossil fuel and electricity emissions
        total_annual_climate_emissions = total_fossil_fuel_emissions_co2e + annual_emissions_electricity
        
        # Store emissions in the results dictionary
        emissions_col = f'{scenario_prefix}{year_label}_{category}_mt_co2e_{mer_type}'
        climate_results[emissions_col] = total_annual_climate_emissions
        annual_climate_emissions[mer_type] = total_annual_climate_emissions
        
        # Now calculate damages using lower, central, and upper SCC assumptions
        for scc_assumption in ["lower", "central", "upper"]:
            # Multiply total annual CO2e by the year-specific SCC value
            scc_value = get_scc_value(year_label, scc_assumption, lookup_climate_impact_scc)                        
            total_annual_climate_damages = annual_climate_emissions[mer_type] * scc_value

            # Store each sensitivity result keyed by (mer_type, scc_assumption)
            damages_col = f'{scenario_prefix}{year_label}_{category}_damages_climate_{mer_type}_{scc_assumption}'
            climate_results[damages_col] = total_annual_climate_damages
            annual_climate_damages[(mer_type, scc_assumption)] = total_annual_climate_damages
    
    return climate_results, annual_climate_emissions, annual_climate_damages
