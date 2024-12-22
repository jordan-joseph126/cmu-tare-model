import pandas as pd
import os

# import functions.tare_setup as tare_setup
from cmu_tare_model.functions.inflation_adjustment import cpi_ratio_2023_2020
from cmu_tare_model.functions.project_future_energy_consumption import lookup_hdd_factor
from cmu_tare_model.functions.create_lookup_emissions_fossil_fuel import lookup_emis_fossil_fuel
from cmu_tare_model.functions.create_lookup_climate_damages_electricity import lookup_co2e_emis_electricity_preIRA, lookup_co2e_emis_electricity_IRA
from cmu_tare_model.functions.create_lookup_health_damages_electricity import lookup_health_damages_electricity_preIRA, lookup_health_damages_electricity_iraRef

# Constants (Assuming these are defined elsewhere in your code)
TD_LOSSES = 0.06
TD_LOSSES_MULTIPLIER = 1 / (1 - TD_LOSSES)
EQUIPMENT_SPECS = {'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15}
POLLUTANTS = ['so2', 'nox', 'pm25', 'co2e']
EPA_SCC_USD2023_PER_TON = 190 * cpi_ratio_2023_2020 # For co2e adjust SCC

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ANNUAL EMISSIONS AND DAMAGES
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# UPDATED DEC 9, 2024 @ 3:00 PM
def calculate_marginal_damages(df, menu_mp, policy_scenario, df_baseline_damages=None, df_detailed_damages=None):
    """
    Calculate marginal damages of pollutants based on equipment usage, emissions, and policy scenarios.
    
    Parameters:
        df (DataFrame): Input data with emissions and consumption data.
        menu_mp (int): Measure package identifier.
        policy_scenario (str): Specifies the policy scenario ('No Inflation Reduction Act' or 'AEO2023 Reference Case').
        df_baseline_damages (DataFrame): Precomputed baseline damages. This dataframe is only used if menu_mp != 0.
        df_detailed_damages (DataFrame, optional): Summary DataFrame to store aggregated results.
    
    Returns:
        Tuple[DataFrame, DataFrame]: 
            - Updated `df_copy` with lifetime damages.
            - Updated `df_detailed_damages` with detailed annual data.
    """
    
    # Create a copy of the input DataFrame to prevent modifying the original
    df_copy = df.copy()
    
    # Only copy df_baseline_damages if menu_mp is not 0
    if menu_mp != 0 and df_baseline_damages is not None:
        df_baseline_damages_copy = df_baseline_damages.copy()
    else:
        df_baseline_damages_copy = None  # Indicate that baseline damages are not used
    
    # Initialize df_detailed_damages if not provided
    if df_detailed_damages is None:
        df_detailed_damages = pd.DataFrame(index=df_copy.index)
    
    # Define policy-specific settings
    scenario_prefix, cambium_scenario, lookup_emis_fossil_fuel, lookup_co2e_emis_electricity, lookup_health_damages_electricity = define_scenario_settings(menu_mp, policy_scenario)
    
    # Precompute HDD adjustment factors by region and year
    hdd_factors_per_year = precompute_hdd_factors(df_copy)
    
    # Define equipment lifetimes (if not defined elsewhere)
    equipment_specs = {
        'heating': 15,
        'waterHeating': 12,
        'clothesDrying': 13,
        'cooking': 15
    }
    
    # Calculate damages using the updated calculate_damages_grid_scenario
    df_new_columns, df_detailed_damages = calculate_damages_grid_scenario(
        df_copy=df_copy,
        df_baseline_damages_copy=df_baseline_damages_copy,
        df_detailed_damages=df_detailed_damages,
        menu_mp=menu_mp,
        td_losses_multiplier=TD_LOSSES_MULTIPLIER,
        lookup_co2e_emis_electricity=lookup_co2e_emis_electricity,
        cambium_scenario=cambium_scenario,
        scenario_prefix=scenario_prefix,
        hdd_factors_df=hdd_factors_per_year,
        lookup_emis_fossil_fuel=lookup_emis_fossil_fuel,
        lookup_health_damages_electricity=lookup_health_damages_electricity,
        EPA_SCC_USD2023_PER_TON=EPA_SCC_USD2023_PER_TON,
        equipment_specs=equipment_specs
    )
    
    # Handle overlapping columns to prevent duplication
    overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)
    if not overlapping_columns.empty:
        df_copy.drop(columns=overlapping_columns, inplace=True)
    
    # Merge newly calculated lifetime damages into df_copy
    df_copy = df_copy.join(df_new_columns, how='left')
    
    return df_copy, df_detailed_damages

def define_scenario_settings(menu_mp, policy_scenario):
    """
    Define scenario-specific settings based on menu and policy inputs.

    Parameters:
        menu_mp (int): Measure package identifier.
        policy_scenario (str): Policy scenario.

    Returns:
        Tuple: (scenario_prefix, cambium_scenario, lookup_emis_fossil_fuel, lookup_co2e_emis_electricity, lookup_health_damages_electricity)
    """
        
    if menu_mp == 0:
        print(f"""-- Scenario: Baseline -- 
              scenario_prefix: 'baseline_', cambium_scenario: 'MidCase', lookup_emis_fossil_fuel: 'lookup_emis_fossil_fuel', 
              lookup_co2e_emis_electricity: 'emis_preIRA_co2e_cambium21_lookup', lookup_health_damages_electricity: 'damages_preIRA_health_damages_lookup'
              """)
        return "baseline_", "MidCase", lookup_emis_fossil_fuel, lookup_co2e_emis_electricity_preIRA, lookup_health_damages_electricity_preIRA

    if policy_scenario == 'No Inflation Reduction Act':
        print(f"""-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp{menu_mp}_', cambium_scenario: 'MidCase', lookup_emis_fossil_fuel: 'lookup_emis_fossil_fuel', 
              lookup_co2e_emis_electricity: 'emis_preIRA_co2e_cambium21_lookup', lookup_health_damages_electricity: 'damages_preIRA_health_damages_lookup'
              """)
        return f"preIRA_mp{menu_mp}_", "MidCase", lookup_emis_fossil_fuel, lookup_co2e_emis_electricity_preIRA, lookup_health_damages_electricity_preIRA

    if policy_scenario == 'AEO2023 Reference Case':
        print(f"""-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp{menu_mp}_', cambium_scenario: 'MidCase', lookup_emis_fossil_fuel: 'lookup_emis_fossil_fuel', 
              lookup_co2e_emis_electricity: 'emis_IRA_co2e_cambium22_lookup', lookup_health_damages_electricity: 'damages_iraRef_health_damages_lookup'
              """)
        return f"iraRef_mp{menu_mp}_", "MidCase", lookup_emis_fossil_fuel, lookup_co2e_emis_electricity_IRA, lookup_health_damages_electricity_iraRef

    raise ValueError("Invalid Policy Scenario! Choose 'No Inflation Reduction Act' or 'AEO2023 Reference Case'.")
    # Return the appropriate variables (assuming these lookups are defined elsewhere)

def precompute_hdd_factors(df):
    """
    Precompute heating degree day (HDD) factors for each region and year.

    Parameters:
        df (DataFrame): Input data.

    Returns:
        dict: HDD factors mapped by year and region.
    """
    
    max_lifetime = max(EQUIPMENT_SPECS.values())
    years = range(2024, 2024 + max_lifetime + 1)
    hdd_factors_df = pd.DataFrame(index=df.index, columns=years)

    for year_label in years:
        hdd_factors_df[year_label] = df['census_division'].map(
            lambda x: lookup_hdd_factor.get(x, lookup_hdd_factor['National']).get(year_label, 1.0)
        )

    return hdd_factors_df

def calculate_damages_grid_scenario(df_copy, df_baseline_damages_copy, df_detailed_damages, menu_mp, td_losses_multiplier, lookup_co2e_emis_electricity, cambium_scenario, scenario_prefix, 
                                    hdd_factors_df, lookup_emis_fossil_fuel, lookup_health_damages_electricity, EPA_SCC_USD2023_PER_TON, equipment_specs):
    """
    Calculate damages for the specified electricity grid scenario using helper functions.

    This version avoids repeated DataFrame insertions by collecting annual and lifetime results in dictionaries,
    then concatenating them to df_detailed_damages in bulk at the end of each iteration.
    """

    new_columns_data = {}  # Will hold lifetime aggregated results

    print("Available columns in df_copy:", df_copy.columns.tolist())

    for category, lifetime in equipment_specs.items():
        print(f"Calculating marginal emissions and marginal damages for {category}")

        # Initialize lifetime accumulators
        lifetime_climate_emissions = {'lrmer': pd.Series(0.0, index=df_copy.index),
                                      'srmer': pd.Series(0.0, index=df_copy.index)}
        lifetime_climate_damages = {'lrmer': pd.Series(0.0, index=df_copy.index),
                                    'srmer': pd.Series(0.0, index=df_copy.index)}
        lifetime_health_damages = pd.Series(0.0, index=df_copy.index)

        for year in range(1, lifetime + 1):
            year_label = year + 2023

            # Get HDD factors for the year
            hdd_factor = hdd_factors_df.get(year_label, pd.Series(1.0, index=df_copy.index))
            adjusted_hdd_factor = hdd_factor if category in ['heating', 'waterHeating'] else pd.Series(1.0, index=df_copy.index)

            # Calculate fossil fuel emissions
            total_fossil_emissions = calculate_fossil_fuel_emissions(
                df_copy, category, adjusted_hdd_factor, lookup_emis_fossil_fuel, menu_mp
            )

            # Calculate climate data (annual)
            climate_results, annual_climate_emissions, annual_climate_damages = calculate_climate_emissions_and_damages(
                df=df_copy, category=category, year_label=year_label, adjusted_hdd_factor=adjusted_hdd_factor, td_losses_multiplier=td_losses_multiplier,
                lookup_co2e_emis_electricity=lookup_co2e_emis_electricity, cambium_scenario=cambium_scenario, EPA_SCC_USD2023_PER_TON=EPA_SCC_USD2023_PER_TON,
                total_fossil_emissions=total_fossil_emissions, scenario_prefix=scenario_prefix, menu_mp=menu_mp
            )

            # Calculate health data (annual)
            health_results, annual_health_damages = calculate_health_damages(
                df=df_copy, category=category, year_label=year_label, adjusted_hdd_factor=adjusted_hdd_factor, td_losses_multiplier=td_losses_multiplier,
                lookup_health_damages_electricity=lookup_health_damages_electricity, cambium_scenario=cambium_scenario, 
                total_fossil_emissions=total_fossil_emissions, scenario_prefix=scenario_prefix, POLLUTANTS=POLLUTANTS, menu_mp=menu_mp
            )

            # Update lifetime accumulators
            for mer_type in ['lrmer', 'srmer']:
                lifetime_climate_emissions[mer_type] += annual_climate_emissions.get(mer_type, 0.0)
                lifetime_climate_damages[mer_type] += annual_climate_damages.get(mer_type, 0.0)

            lifetime_health_damages += annual_health_damages

            # Concatenate annual results once for this year
            annual_data_all = {**climate_results, **health_results}
            if annual_data_all:
                df_detailed_damages = pd.concat([df_detailed_damages, pd.DataFrame(annual_data_all, index=df_copy.index)], axis=1)

        # After computing all years for this category, store lifetime values
        lifetime_dict = {}
        for mer_type in ['lrmer', 'srmer']:
            # Columns for Lifetime (Current Scenario Equipment) and Avoided Emissions and Damages
            lifetime_emissions_col = f'{scenario_prefix}{category}_lifetime_tons_co2e_{mer_type}'
            lifetime_damages_col = f'{scenario_prefix}{category}_lifetime_damages_climate_{mer_type}'

            # Lifetime Emissions and Damages
            lifetime_dict[lifetime_emissions_col] = lifetime_climate_emissions[mer_type].round(2)
            lifetime_dict[lifetime_damages_col] = lifetime_climate_damages[mer_type].round(2)

            # Avoided Emissions and Damages (only if menu_mp != 0 and baseline data is provided)
            if menu_mp != 0 and df_baseline_damages_copy is not None:
                avoided_emissions_col = f'{scenario_prefix}{category}_avoided_tons_co2e_{mer_type}'
                avoided_damages_col = f'{scenario_prefix}{category}_avoided_damages_climate_{mer_type}'

                baseline_emissions_col = f'baseline_{category}_lifetime_tons_co2e_{mer_type}'
                baseline_damages_col = f'baseline_{category}_lifetime_damages_climate_{mer_type}'

                if baseline_emissions_col in df_baseline_damages_copy.columns and baseline_damages_col in df_baseline_damages_copy.columns:
                    lifetime_dict[avoided_emissions_col] = np.round(
                        df_baseline_damages_copy[baseline_emissions_col] - lifetime_dict[lifetime_emissions_col], 2
                    )
                    lifetime_dict[avoided_damages_col] = np.round(
                        df_baseline_damages_copy[baseline_damages_col] - lifetime_dict[lifetime_damages_col], 2
                    )

                    new_columns_data[avoided_emissions_col] = lifetime_dict[avoided_emissions_col]
                    new_columns_data[avoided_damages_col] = lifetime_dict[avoided_damages_col]
                else:
                    print(f"Warning: Missing baseline columns for {category}, {mer_type}. Avoided values skipped.")

        # Store lifetime health damages
        lifetime_health_damages_col = f'{scenario_prefix}{category}_lifetime_damages_health'
        lifetime_dict[lifetime_health_damages_col] = lifetime_health_damages.round(2)

        # Avoided Health Damages (only if menu_mp != 0 and baseline data is provided)
        if menu_mp != 0 and df_baseline_damages_copy is not None:
            avoided_health_damages_col = f'{scenario_prefix}{category}_avoided_damages_health'
            baseline_health_col = f'baseline_{category}_lifetime_damages_health'

            if baseline_health_col in df_baseline_damages_copy.columns:
                lifetime_dict[avoided_health_damages_col] = np.round(
                    df_baseline_damages_copy[baseline_health_col] - lifetime_dict[lifetime_health_damages_col], 2
                )
                new_columns_data[avoided_health_damages_col] = lifetime_dict[avoided_health_damages_col]
            else:
                print(f"Warning: Missing baseline health column for {category}. Avoided health damages skipped.")

        new_columns_data[lifetime_health_damages_col] = lifetime_health_damages.round(2)

        # Concatenate lifetime results for this category to df_detailed_damages in one go
        df_detailed_damages = pd.concat([df_detailed_damages, pd.DataFrame(lifetime_dict, index=df_copy.index)], axis=1)

    # Finalize the DataFrame for lifetime columns
    df_new_columns = pd.DataFrame(new_columns_data, index=df_copy.index)

    # Return both the new lifetime columns and the updated df_detailed_damages
    return df_new_columns, df_detailed_damages

def calculate_fossil_fuel_emissions(df, category, adjusted_hdd_factor, lookup_emis_fossil_fuel, menu_mp):
    """
    Calculate fossil fuel emissions for a given row and category.
    """

    total_fossil_emissions = {pollutant: pd.Series(0.0, index=df.index) for pollutant in POLLUTANTS}

    if menu_mp == 0:
        fuels = ['naturalGas', 'propane']
        if category not in ['cooking', 'clothesDrying']:
            fuels.append('fuelOil')

        for fuel in fuels:
            consumption_col = f'base_{fuel}_{category}_consumption'
            fuel_consumption = df.get(consumption_col, pd.Series(0.0, index=df.index)).fillna(0) * adjusted_hdd_factor

            for pollutant in total_fossil_emissions.keys():
                emis_factor = lookup_emis_fossil_fuel.get((fuel, pollutant), 0)
                total_fossil_emissions[pollutant] += fuel_consumption * emis_factor

    return total_fossil_emissions

def calculate_climate_emissions_and_damages(df, category, year_label, adjusted_hdd_factor, td_losses_multiplier, lookup_co2e_emis_electricity, cambium_scenario, 
                                            EPA_SCC_USD2023_PER_TON, total_fossil_emissions, scenario_prefix, menu_mp):
    """
    Calculate climate-related emissions and damages for a given row, category, and year.
    Returns dicts of results and series for annual emissions/damages aggregation.
    """

    climate_results = {}
    annual_climate_emissions = {}
    annual_climate_damages = {}

    total_fossil_emissions_co2e = total_fossil_emissions['co2e']

    # Define functions for vectorized lookup of emission factors
    def get_emission_factor_lrmer(region):
        return lookup_co2e_emis_electricity.get(
            (cambium_scenario, region), {}
        ).get(year_label, {}).get('lrmer_ton_per_kWh_co2e', 0)

    def get_emission_factor_srmer(region):
        return lookup_co2e_emis_electricity.get(
            (cambium_scenario, region), {}
        ).get(year_label, {}).get('srmer_ton_per_kWh_co2e', 0)

    mer_factors = {
        'lrmer': df['gea_region'].map(get_emission_factor_lrmer),
        'srmer': df['gea_region'].map(get_emission_factor_srmer)
    }

    # Electricity consumption depends on the scenario
    if menu_mp == 0:
        elec_consumption = df.get(f'base_electricity_{category}_consumption', pd.Series(0.0, index=df.index)).fillna(0) * adjusted_hdd_factor
    else:
        consumption_col = f'mp{menu_mp}_{year_label}_{category}_consumption'
        elec_consumption = df.get(consumption_col, pd.Series(0.0, index=df.index)).fillna(0)

    for mer_type in ['lrmer', 'srmer']:
        annual_emis_electricity = elec_consumption * td_losses_multiplier * mer_factors[mer_type]

        total_annual_climate_emissions = total_fossil_emissions_co2e + annual_emis_electricity
        total_annual_climate_damages = total_annual_climate_emissions * EPA_SCC_USD2023_PER_TON

        # Store annual CO2e emissions and climate damages in the dictionary
        emis_col = f'{scenario_prefix}{year_label}_{category}_tons_co2e_{mer_type}'
        damage_col = f'{scenario_prefix}{year_label}_{category}_damages_climate_{mer_type}'
        climate_results[emis_col] = total_annual_climate_emissions.round(2)
        climate_results[damage_col] = total_annual_climate_damages.round(2)

        annual_climate_emissions[mer_type] = total_annual_climate_emissions
        annual_climate_damages[mer_type] = total_annual_climate_damages

    return climate_results, annual_climate_emissions, annual_climate_damages


def calculate_health_damages(df, category, year_label, adjusted_hdd_factor, td_losses_multiplier, lookup_health_damages_electricity, cambium_scenario, 
                             scenario_prefix, POLLUTANTS, total_fossil_emissions, menu_mp):
    """
    Calculate health-related damages for a given row, category, and year.
    Returns a dict of annual pollutant damage results and a series of aggregated health damages.
    """
    
    health_results = {}
    annual_health_damages = pd.Series(0.0, index=df.index)

    for pollutant in POLLUTANTS:
        if pollutant != 'co2e':
            # Fossil fuel damages
            fossil_emissions = total_fossil_emissions.get(pollutant, pd.Series(0.0, index=df.index))
            marginal_damage_col = f'marginal_damages_{pollutant}'
            if marginal_damage_col in df.columns:
                marginal_damages = df[marginal_damage_col]
            else:
                marginal_damages = pd.Series(0.0, index=df.index)
            fossil_fuel_damage = fossil_emissions * marginal_damages

            # Define a function for vectorized lookup
            def get_electricity_damage_factor(region):
                pollutant_damage_key = f'{pollutant}_dollarPerkWh_adjustVSL'
                return lookup_health_damages_electricity.get(
                    (cambium_scenario, region), {}
                ).get(year_label, {}).get(pollutant_damage_key, 0)

            elec_damage_factor = df['gea_region'].map(get_electricity_damage_factor)

            # Electricity consumption depends on the scenario
            if menu_mp == 0:
                elec_consumption = df.get(f'base_electricity_{category}_consumption', pd.Series(0.0, index=df.index)).fillna(0) * adjusted_hdd_factor
            else:
                consumption_col = f'mp{menu_mp}_{year_label}_{category}_consumption'
                elec_consumption = df.get(consumption_col, pd.Series(0.0, index=df.index)).fillna(0)

            electricity_damages = elec_consumption * td_losses_multiplier * elec_damage_factor

            total_pollutant_damage = fossil_fuel_damage + electricity_damages

            # Store the annual health damages in the dictionary
            damage_col = f'{scenario_prefix}{year_label}_{category}_damages_{pollutant}'
            health_results[damage_col] = total_pollutant_damage.round(2)

            # Accumulate annual health-related damages
            annual_health_damages += total_pollutant_damage

    # Calculate total health damages for the category/year
    health_damage_col = f'{scenario_prefix}{year_label}_{category}_damages_health'
    # Ensure required columns exist in health_results before summation
    so2_col = f'{scenario_prefix}{year_label}_{category}_damages_so2'
    nox_col = f'{scenario_prefix}{year_label}_{category}_damages_nox'
    pm25_col = f'{scenario_prefix}{year_label}_{category}_damages_pm25'

    if menu_mp == 0:
        if all(col in health_results for col in [so2_col, nox_col, pm25_col]):
            health_results[health_damage_col] = round(health_results[so2_col] +
                                                      health_results[nox_col] +
                                                      health_results[pm25_col], 2)
        else:
            health_results[health_damage_col] = annual_health_damages.round(2)
    else:
        if all(col in health_results for col in [so2_col, nox_col, pm25_col]):
            health_results[health_damage_col] = round((health_results[so2_col] +
                                                       health_results[nox_col] +
                                                       health_results[pm25_col]), 2)
        else:
            health_results[health_damage_col] = annual_health_damages.round(2)
                
    return health_results, annual_health_damages

# # calculate_marginal_damages(df, menu_mp, policy_scenario)
# df_euss_am_baseline_home, df_baseline_scenario_damages = calculate_marginal_damages(df=df_euss_am_baseline_home,
#                                                                                     menu_mp=menu_mp,
#                                                                                     policy_scenario='No Inflation Reduction Act',
#                                                                                     df_detailed_damages=df_baseline_scenario_damages
#                                                                                     )
# # df_euss_am_baseline_home