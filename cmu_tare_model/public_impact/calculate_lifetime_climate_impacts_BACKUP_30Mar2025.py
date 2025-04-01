import pandas as pd

from cmu_tare_model.constants import MER_TYPES, EQUIPMENT_SPECS, TD_LOSSES_MULTIPLIER, EPA_SCC_USD2023_PER_MT_LOW, EPA_SCC_USD2023_PER_MT_BASE, EPA_SCC_USD2023_PER_MT_HIGH
from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import calculate_fossil_fuel_emissions
from cmu_tare_model.public_impact.emissions_scenario_settings import define_scenario_settings
from cmu_tare_model.public_impact.calculations.precompute_hdd_factors import precompute_hdd_factors

"""
Consider updating so that the future scc_value values considered rather than a fixed set of 3 based on 2020 emission year.
- Spreadsheet with year, low, base, high values

scc_value Low and Base: https://www.energy.gov/sites/default/files/2023-04/57.%20Social%20Cost%20of%20Carbon%202021.pdf
scc_value High: https://www.epa.gov/system/files/documents/2023-12/epa_scghg_2023_report_final.pdf
"""

def calculate_climate_impacts(df, menu_mp, policy_scenario, df_baseline_damages=None):
    """
    Calculate lifetime climate impacts (emissions and damages) for each equipment category all (mer, scc_value) combinations.
    
    This function returns a tuple of two dataframes (rounded to 2 decimals):
      - The main dataframe (df_main) is the input dataframe updated with lifetime climate impacts: emissions, damages, and avoided emissions/damages (if applicable).
      - The detailed dataframe (df_detailed) contains annual results along with lifetime breakdowns.

    Args:
        df (pd.DataFrame): Input data.
        menu_mp (int): Measure package identifier.
        policy_scenario (str): Policy scenario name that determines model inputs ('No Inflation Reduction Act' or 'AEO2023 Reference Case').
        df_baseline_damages (pd.DataFrame, optional): Baseline climate emissions and damages data.
    
    Returns:
        tuple: (df_main, df_detailed)
            - df_main (rounded to 2 decimals) contains the aggregated lifetime climate impacts: emissions, damages, and avoided emissions/damages (if applicable).
            - df_detailed (rounded to 2 decimals) contains the detailed annual climate impacts.
    """
    # Create a copy of the input df
    # Then initialize the detailed dataframe (df_copy will become df_main)
    df_copy = df.copy()
    df_detailed = pd.DataFrame(index=df_copy.index)
    
    # Initialize a dictionary to store lifetime climate impacts columns
    lifetime_columns_data = {}

    # Retrieve scenario-specific settings for electricity/fossil-fuel emissions
    # Ignored (underscored) the health lookup as it's not used in this function
    scenario_prefix, cambium_scenario, lookup_emissions_fossil_fuel, lookup_emissions_electricity_climate, _ = define_scenario_settings(menu_mp, policy_scenario)

    # Precompute HDD adjustment factors by region and year
    hdd_factors_per_year = precompute_hdd_factors(df_copy)

    # Loop over each equipment category and its lifetime
    for category, lifetime in EQUIPMENT_SPECS.items():        
        # Reinitialize lifetime climate impacts for this category for each (mer, scc_assumption) pair
        lifetime_climate_emissions = {mer: pd.Series(0.0, index=df_copy.index) for mer in MER_TYPES}
        
        # For damages, store separate accumulation for each (mer, scc_assumption)
        lifetime_climate_damages = {(mer, scc_value): pd.Series(0.0, index=df_copy.index)
                                    for mer in MER_TYPES for scc_value in ['low', 'base', 'high'] }

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
            for mer in MER_TYPES:
                lifetime_climate_emissions[mer] += annual_emissions.get(mer, 0.0)
            # Accumulate annual damages for each scc_value assumption
            for key, value in annual_damages.items():
                lifetime_climate_damages[key] += value

            if climate_results:
                df_detailed = pd.concat([df_detailed, pd.DataFrame(climate_results, index=df_copy.index)], axis=1)
        
        # Create lifetime columns for climate impacts
        lifetime_dict = {}
        for mer in MER_TYPES:
            emissions_col = f'{scenario_prefix}{category}_lifetime_mt_co2e_{mer}'
            lifetime_dict[emissions_col] = lifetime_climate_emissions[mer]
            
            for scc_assumption in ['low', 'base', 'high']:
                damages_col = f'{scenario_prefix}{category}_lifetime_damages_climate_{mer}_{scc_assumption}'
                lifetime_dict[damages_col] = lifetime_climate_damages[(mer, scc_assumption)]
                # Calculate avoided damages if baseline data is provided
                if menu_mp != 0 and df_baseline_damages is not None:
                    baseline_damages_col = f'baseline_{category}_lifetime_damages_climate_{mer}_{scc_assumption}'
                    avoided_damages_col = f'{scenario_prefix}{category}_avoided_damages_climate_{mer}_{scc_assumption}'
                    lifetime_dict[avoided_damages_col] = df_baseline_damages[baseline_damages_col] - lifetime_dict[damages_col]
            # Calculate avoided emissions if baseline data is provided
            if menu_mp != 0 and df_baseline_damages is not None:
                baseline_emissions_col = f'baseline_{category}_lifetime_mt_co2e_{mer}'
                avoided_emissions_col = f'{scenario_prefix}{category}_avoided_mt_co2e_{mer}'
                lifetime_dict[avoided_emissions_col] = df_baseline_damages[baseline_emissions_col] - lifetime_dict[emissions_col]

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

def calculate_climate_emissions_and_damages(
    df,
    category,
    year_label,
    adjusted_hdd_factor,
    lookup_emissions_electricity_climate,
    cambium_scenario,
    total_fossil_fuel_emissions,
    scenario_prefix,
    menu_mp
):
    """Calculate climate-related emissions and damages for a given category and year.

    Args:
        df (pd.DataFrame): DataFrame containing base or measure-package consumption data and region info.
        category (str): Equipment category (e.g., 'heating', 'waterHeating').
        year_label (int): The year offset from 2023 (e.g., 2024 = 1 + 2023).
        adjusted_hdd_factor (pd.Series): Heating degree-day adjustment factors for the specified category.
        lookup_emissions_electricity_climate (dict): Lookup for electricity-based CO2e emissions factors.
        cambium_scenario (str): Label identifying the Cambium scenario for emissions lookups.
        total_fossil_fuel_emissions (dict): Mapping of pollutant -> pd.Series, from fossil fuel usage.
        scenario_prefix (str): Prefix for output column naming.
        menu_mp (int): Measure package identifier.

    Returns:
        tuple:
            - dict: Annual climate emissions and damages columns (by year).
            - dict: Annual climate emissions by MER type for aggregation.
            - dict: Annual climate damages by MER type and scc_value assumption for aggregation.
              (Keys are tuples of the form (mer_type, scc_assumption).)
    """
    # Results dictionaries (without rounding)
    climate_results = {}
    annual_climate_emissions = {}
    annual_climate_damages = {}
    
    total_fossil_fuel_emissions_co2e = total_fossil_fuel_emissions['co2e']
    
    # Define vectorized helper functions for LRMER/SRMER emission factor lookups
    def get_emission_factor_lrmer(region):
        return lookup_emissions_electricity_climate.get(
            (cambium_scenario, region), {}
        ).get(year_label, {}).get('lrmer_mt_per_kWh_co2e', 0)
    
    def get_emission_factor_srmer(region):
        return lookup_emissions_electricity_climate.get(
            (cambium_scenario, region), {}
        ).get(year_label, {}).get('srmer_mt_per_kWh_co2e', 0)
    
    mer_factors = {
        'lrmer': df['gea_region'].map(get_emission_factor_lrmer),
        'srmer': df['gea_region'].map(get_emission_factor_srmer)
    }
    
    # Determine electricity consumption column based on menu_mp and apply the HDD adjustment factor
    if menu_mp == 0:
        electricity_consumption = df.get(f'base_electricity_{category}_consumption', pd.Series(0.0, index=df.index)).fillna(0)
        electricity_consumption *= adjusted_hdd_factor
    else:
        consumption_col = f'mp{menu_mp}_{year_label}_{category}_consumption'
        electricity_consumption = df.get(consumption_col, pd.Series(0.0, index=df.index)).fillna(0)
    
    # Calculate annual emissions and damages for each MER type
    for mer_type in MER_TYPES:
        annual_emissions_electricity = electricity_consumption * TD_LOSSES_MULTIPLIER * mer_factors[mer_type]
        
        # Sum fossil fuel CO2e and electricity CO2e
        total_annual_climate_emissions = total_fossil_fuel_emissions_co2e + annual_emissions_electricity
        
        # Store emissions results
        emissions_col = f'{scenario_prefix}{year_label}_{category}_mt_co2e_{mer_type}'
        climate_results[emissions_col] = total_annual_climate_emissions
        annual_climate_emissions[mer_type] = total_annual_climate_emissions
        
        # scc_value Sensitivity Analysis: store each sensitivity result separately.
        for scc_assumption in ['low', 'base', 'high']:
            if scc_assumption == 'low':
                scc_value = EPA_SCC_USD2023_PER_MT_LOW
            elif scc_assumption == 'base':
                scc_value = EPA_SCC_USD2023_PER_MT_BASE
            elif scc_assumption == 'high':
                scc_value = EPA_SCC_USD2023_PER_MT_HIGH
            else:
                raise ValueError(f"Invalid scc_value assumption: {scc_assumption}")
            
            damages_col = f'{scenario_prefix}{year_label}_{category}_damages_climate_{mer_type}_{scc_assumption}'
            total_annual_climate_damages = total_annual_climate_emissions * scc_value
            climate_results[damages_col] = total_annual_climate_damages
            # Store each sensitivity result keyed by (mer_type, scc_assumption)
            annual_climate_damages[(mer_type, scc_assumption)] = total_annual_climate_damages
    
    return climate_results, annual_climate_emissions, annual_climate_damages
