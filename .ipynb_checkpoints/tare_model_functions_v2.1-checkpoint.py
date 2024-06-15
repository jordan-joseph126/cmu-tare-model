#!/usr/bin/env python
# coding: utf-8

# # Baseline

# In[3]:


import pandas as pd
import numpy as np
import re

def standardize_fuel_name(fuel_desc):
    # Ensure that the input is a string
    if pd.isna(fuel_desc):
        return 'Other'  # Return 'Other' for NaN values
    elif isinstance(fuel_desc, str):
        if 'Electric' in fuel_desc:
            return 'Electricity'
        elif 'Gas' in fuel_desc:
            return 'Natural Gas'
        elif 'Propane' in fuel_desc:
            return 'Propane'
        elif 'Oil' in fuel_desc:
            return 'Fuel Oil'
        else:
            return 'Other'  # For any unexpected types, categorize as 'Other'
    else:
        return 'Other'  # Non-string, non-NaN values are categorized as 'Other'

def preprocess_fuel_data(df, column_name):
    """Applies standardization to a specified column in the DataFrame."""
    print(f"Processing column: {column_name}")
    print(f"Initial data types: {df[column_name].dtype}")
    
    # Updated this portion of the code to prevent the setting with copy warning
    df.loc[:, column_name] = df[column_name].apply(standardize_fuel_name)
    
    print(f"Data types after processing: {df[column_name].dtype}")
    return df

def apply_fuel_filter(df, category, enable):
    if enable == 'Yes':
        fuel_list = ['Natural Gas', 'Electricity', 'Propane', 'Fuel Oil']
        df_filtered = df[df[f'base_{category}_fuel'].isin(fuel_list)]
        print(f"Filtered for the following fuels: {fuel_list}")
        return df_filtered
    return df

def apply_technology_filter(df, category, enable):
    """
    Applies technology filters to the dataframe based on the category and whether filtering is enabled.
    
    Parameters:
    - df: The DataFrame to filter.
    - category: The category of consumption (e.g., 'heating', 'waterHeating').
    - enable: String flag ('Yes' or 'No') indicating whether to apply the filter.
    """
    if enable == 'Yes':
        if category == 'heating':
            tech_list = [
                'Electricity ASHP', 'Electricity Baseboard', 'Electricity Electric Boiler', 'Electricity Electric Furnace',
                'Fuel Oil Fuel Boiler', 'Fuel Oil Fuel Furnace', 'Natural Gas Fuel Boiler', 'Natural Gas Fuel Furnace',
                'Propane Fuel Boiler', 'Propane Fuel Furnace'
            ]
            df_filtered = df[df['heating_type'].isin(tech_list)]
            print(f"Filtered for the following Heating technologies: {tech_list}")    
            return df_filtered
        
        elif category == 'waterHeating':
            tech_list = [
                'Electric Heat Pump, 80 gal', 'Electric Premium', 'Electric Standard',
                'Fuel Oil Premium', 'Fuel Oil Standard', 'Natural Gas Premium', 'Natural Gas Standard',
                'Propane Premium', 'Propane Standard'
            ]
            df_filtered = df[df['waterHeating_type'].isin(tech_list)]
            print(f"Filtered for the following Water Heating technologies: {tech_list}")
            return df_filtered
    
    return df

def debug_filters(df, filter_name):
    if df.empty:
        print(f"No rows left after applying {filter_name}")
    else:
        print(f"{len(df)} rows remain after applying {filter_name}")

# Function to extract city name
def extract_city_name(row):
    match = re.match(r'^[A-Z]{2}, (.+)$', row)
    return match.group(1) if match else row
        
def df_enduse_refactored(df_baseline, fuel_filter='Yes', tech_filter='Yes'):
    # Initial check
    if df_baseline.empty:
        print("Warning: Input DataFrame is empty")
        return df_baseline

    # Standardize fuel names in the base columns before creating the df_enduse
    df_baseline = preprocess_fuel_data(df_baseline, 'in.clothes_dryer')
    df_baseline = preprocess_fuel_data(df_baseline, 'in.cooking_range')

    # Map standardized names to new columns
    df_baseline['base_clothesDrying_fuel'] = df_baseline['in.clothes_dryer']
    df_baseline['base_cooking_fuel'] = df_baseline['in.cooking_range']
    
    # Initialize df_enduse from df_baseline with all required columns
    # (assuming columns are correctly listed here)
    # Create a new DataFrame named df_enduse
    # using pd.DataFrame constructor and initialize it with columns from df_baseline
    df_enduse = pd.DataFrame({
        'bldg_id': df_baseline['bldg_id'],
        'square_footage': df_baseline['in.sqft'],
        'census_region': df_baseline['in.census_region'],
        'building_america_climate_zone': df_baseline['in.building_america_climate_zone'],
        'cambium_GEA_region': df_baseline['in.generation_and_emissions_assessment_region'],
        'state': df_baseline['in.state'],
        'city': df_baseline['in.city'].apply(extract_city_name),
        'county': df_baseline['in.county'],
        'puma': df_baseline['in.puma'],
        'county_and_puma': df_baseline['in.county_and_puma'],
        'weather_file_city': df_baseline['in.weather_file_city'],
        'Longitude': df_baseline['in.weather_file_longitude'],
        'Latitude': df_baseline['in.weather_file_latitude'],
        'building_type': df_baseline['in.geometry_building_type_recs'],
        'income': df_baseline['in.income'],
        'federal_poverty_level': df_baseline['in.federal_poverty_level'],
        'occupancy': df_baseline['in.occupants'],
        'tenure': df_baseline['in.tenure'],
        'vacancy_status': df_baseline['in.vacancy_status'],
        'base_heating_fuel': df_baseline['in.heating_fuel'],
        'heating_type': df_baseline['in.hvac_heating_type_and_fuel'],
        'hvac_cooling_type': df_baseline['in.hvac_cooling_type'],
        'vintage': df_baseline['in.vintage'],
        'base_heating_efficiency': df_baseline['in.hvac_heating_efficiency'],
        'base_electricity_heating_consumption': df_baseline['out.electricity.heating.energy_consumption.kwh'],
        'base_fuelOil_heating_consumption': df_baseline['out.fuel_oil.heating.energy_consumption.kwh'],
        'base_naturalGas_heating_consumption': df_baseline['out.natural_gas.heating.energy_consumption.kwh'],
        'base_propane_heating_consumption': df_baseline['out.propane.heating.energy_consumption.kwh'],
        'base_waterHeating_fuel': df_baseline['in.water_heater_fuel'],
        'waterHeating_type': df_baseline['in.water_heater_efficiency'],
        'base_electricity_waterHeating_consumption': df_baseline['out.electricity.hot_water.energy_consumption.kwh'],
        'base_fuelOil_waterHeating_consumption': df_baseline['out.fuel_oil.hot_water.energy_consumption.kwh'],
        'base_naturalGas_waterHeating_consumption': df_baseline['out.natural_gas.hot_water.energy_consumption.kwh'],
        'base_propane_waterHeating_consumption': df_baseline['out.propane.hot_water.energy_consumption.kwh'],
        'base_clothesDrying_fuel': df_baseline['in.clothes_dryer'],
        'base_electricity_clothesDrying_consumption': df_baseline['out.electricity.clothes_dryer.energy_consumption.kwh'],
        'base_naturalGas_clothesDrying_consumption': df_baseline['out.natural_gas.clothes_dryer.energy_consumption.kwh'],
        'base_propane_clothesDrying_consumption': df_baseline['out.propane.clothes_dryer.energy_consumption.kwh'],
        'base_cooking_fuel': df_baseline['in.cooking_range'],
        'base_electricity_cooking_consumption': df_baseline['out.electricity.range_oven.energy_consumption.kwh'],
        'base_naturalGas_cooking_consumption': df_baseline['out.natural_gas.range_oven.energy_consumption.kwh'],
        'base_propane_cooking_consumption': df_baseline['out.propane.range_oven.energy_consumption.kwh']
    })
    
    categories = ['heating', 'waterHeating', 'clothesDrying', 'cooking']
    for category in categories:
        if category == 'heating' or category == 'waterHeating':
            fuel_types = ['electricity', 'fuelOil', 'naturalGas', 'propane']
            # Calculate and update total consumption
            total_consumption = sum(df_enduse.get(f'base_{fuel}_{category}_consumption', pd.Series([], dtype=float)).fillna(0) for fuel in fuel_types)
            df_enduse[f'baseline_{category}_consumption'] = total_consumption.replace(0, np.nan)

            debug_filters(df_enduse, f"total {category} consumption calculation")

            # Apply filters
            df_enduse = apply_fuel_filter(df_enduse, category, fuel_filter)
            debug_filters(df_enduse, f"{category} fuel filter")

            df_enduse = apply_technology_filter(df_enduse, category, tech_filter)
            debug_filters(df_enduse, f"{category} technology filter")

        else:
            fuel_types = ['electricity', 'naturalGas', 'propane']
            # Calculate and update total consumption
            total_consumption = sum(df_enduse.get(f'base_{fuel}_{category}_consumption', pd.Series([], dtype=float)).fillna(0) for fuel in fuel_types)
            df_enduse[f'baseline_{category}_consumption'] = total_consumption.replace(0, np.nan)

            debug_filters(df_enduse, f"total {category} consumption calculation")

            # Apply filters
            df_enduse = apply_fuel_filter(df_enduse, category, fuel_filter)
            debug_filters(df_enduse, f"{category} fuel filter")
            
    return df_enduse


# In[4]:


# Calculate emissions factors for fossil fuels
# This is before adjusting for natural gas leakage
# Note: We use electricity marginal damages directly instead of multiplying
# CEDM emissions factors by the EASIUR marginal damages. 
def calculate_fossilFuel_emission_factor(fuel_type, so2_factor, nox_factor, pm25_factor, co2_factor, fuelConversion_factor1, fuelConversion_factor2):
    """
    Calculate Emissions Factors: FOSSIL FUELS
    Fossil Fuels (Natural Gas, Fuel Oil, Propane):
    - NOx, SO2, CO2: 
        - RESNET Table 7.1.2 Emissions Factors for Household Combustion Fuels
        - Source: https://www.resnet.us/wp-content/uploads/ANSIRESNETICC301-2022_resnetpblshd.pdf
        - All factors are in units of lb/Mbtu so energy consumption in kWh need to be converted to kWh 
        - (1 lb / Mbtu) * (1 Mbtu / 1x10^6 Btu) * (3412 Btu / 1 kWh)
    - PM2.5: 
        - A National Methodology and Emission Inventory for Residential Fuel Combustion
        - Source: https://www3.epa.gov/ttnchie1/conference/ei12/area/haneke.pdf
    """
    
    # Create an empty dictionary called margEmis_factors to store the values
    margEmis_factors = {}

    # SO2, NOx, CO2: (1 lb / Mbtu) * (1 Mbtu / 1x10^6 Btu) * (3412 Btu / 1 kWh)
    # PM2.5 - FUEL OIL: 0.83 lb/thousand gallons * (1 thousand gallons / 1000 gallons) * (1 gallon heating oil/138,500 BTU) * (3412 BTU/1 kWh)
    # PM2.5 - NATURAL GAS: 1.9 lb/million cf * (million cf/1000000 cf) * (1 cf natural gas/1039 BTU) * (3412 BTU/1 kWh)
    # PM2.5 - PROPANE: 0.17 lb/thousand gallons * (1 thousand gallons / 1000 gallons) * (1 gallon propane/91,452 BTU) * (3412 BTU/1 kWh)
    margEmis_factors[f"{fuel_type}_so2"] = so2_factor * (1 / 1000000) * (3412 / 1)
    margEmis_factors[f"{fuel_type}_nox"] = nox_factor * (1 / 1000000) * (3412 / 1)
    margEmis_factors[f"{fuel_type}_pm25"] = pm25_factor * (1 / fuelConversion_factor1) * (1 / fuelConversion_factor2) * (3412 / 1)
    margEmis_factors[f"{fuel_type}_co2"] = co2_factor * (1 / 1000000) * (3412 / 1)

    return margEmis_factors


# In[5]:


# Working as of May 6, 2024
# Added documentation
# Improved efficiency and prevent dataframe fragmentation warnings
gea_eGRID_mapping = {
    'AZNMc': 'AZNM',
    'CAMXc': 'CAMX',
    'ERCTc': 'ERCT',
    'FRCCc': 'FRCC',
    'MROEc': 'MROE',
    'MROWc': 'MROW',
    'NEWEc': 'NEWE',   
    'NWPPc': 'NWPP',
    'NYSTc': 'NYUP',   # NYSTc contains 'NYUP', 'NYCW', 'NYLI'
    'RFCMc': 'RFCM',
    'RFCWc': 'RFCW',
    'RFCEc': 'RFCE',
    'RMPAc': 'RMPA',
    'SRSOc': 'SRSO',
    'SRTVc': 'SRTV',
    'SRMVc': 'SRMV',
    'SRMWc': 'SRMW',
    'SRVCc': 'SRVC',
    'SPNOc': 'SPNO',
    'SPSOc': 'SPSO'
}

def calculate_marginal_damages(df, grid_decarb=False):
    """
    Calculate the marginal damages of different pollutants based on various conditions and mappings.
    
    Parameters:
    - df (DataFrame): The primary data frame containing pollutant emissions data and other relevant attributes.
    - grid_decarb (bool): Flag to determine if grid decarbonization calculations are to be applied.
    
    Returns:
    - DataFrame: The updated data frame with calculated marginal damages and potentially new columns.
    
    This function processes a given DataFrame 'df' to:
    - Copy the DataFrame to avoid modification of the original data.
    - Map regional identifiers to a subregion grid.
    - Calculate the natural gas leakage factor based on state.
    - Create and calculate damage factor columns if they do not exist.
    - Depending on the flag 'grid_decarb', apply different damage calculation methods.
    - Manage and merge newly created columns to avoid duplicates and ensure data integrity.
    """
    # Create a copy of the DataFrame to work on
    df_copy = df.copy()
    
    # Define lists of pollutants and categories for calculations
    pollutants = ['so2', 'nox', 'pm25', 'co2']
    categories = ['heating', 'waterHeating', 'clothesDrying', 'cooking']

    # Map 'cambium_GEA_region' to 'subregion_eGRID' for regional breakdown
    df_copy['subregion_eGRID'] = df_copy['cambium_GEA_region'].map(gea_eGRID_mapping)

    # Map 'state' to 'naturalGas_leakage_factor'
    state_to_factor = dict(zip(df_margEmis_factors['state'], df_margEmis_factors['naturalGas_leakage_factor']))
    df_copy['naturalGas_leakage_factor'] = df_copy['state'].map(state_to_factor)

    # Process each pollutant for social costs and damage factors
    for pollutant in pollutants:
        # Check if damage factor columns already exist; if not, create them
        if f'margSocialCosts_{pollutant}' not in df_copy.columns:
            df_copy[f'margSocialCosts_{pollutant}'] = df_copy.apply(lambda row: damages_fossilFuel_lookup[f'margSocialCosts_{pollutant}'][(row['Longitude'], row['Latitude'])], axis=1)
        
        if f'margDamage_factor_{pollutant}' not in df_copy.columns:
            df_copy[f'margDamage_factor_{pollutant}'] = df_copy['subregion_eGRID'].apply(lambda x: damages_CEDM_lookup.get((pollutant, x), None))

    # Set constant for transmission and distribution losses
    td_losses = 0.06
   
    # Placeholder DataFrame for new or modified columns
    new_columns_df = pd.DataFrame(index=df_copy.index)  # DataFrame to hold new or modified columns

    # Depending on the grid decarbonization status, calculate damages accordingly
    if grid_decarb:
        new_columns_df = calculate_damages_decarb_grid(df_copy, menu_mp, categories, years, td_losses, dict_margDamages_gridDecarb)
    else:
        new_columns_df = calculate_damages_current_grid(df_copy, menu_mp, categories, pollutants, td_losses, national_lookup, electricity_lookup, damages_CEDM_lookup)

    # Exclude columns that already exist in df_copy to avoid duplicates
    columns_to_add = new_columns_df.columns.difference(df_copy.columns)

    # Concatenate new columns avoiding duplicates and reducing fragmentation
    df_copy = pd.concat([df_copy, new_columns_df[columns_to_add]], axis=1)

    return df_copy

# def calculate_damages_current_grid(df_copy, menu_mp, categories, pollutants, td_losses, national_lookup, electricity_lookup, damages_CEDM_lookup):
#     """
#     Calculate damages for the current electricity grid scenario.

#     Parameters:
#         df_copy (DataFrame): The DataFrame containing consumption data.
#         menu_mp (int): The menu number for the measure package.
#         categories (list): List of end-use categories.
#         pollutants (list): List of pollutants.
#         td_losses (float): Transmission and distribution losses.
#         national_lookup (dict): Lookup table for national emissions factors.
#         electricity_lookup (dict): Lookup table for electricity emissions.
#         damages_CEDM_lookup (dict): Lookup table for damages from CEDM.

#     Returns:
#         DataFrame: The DataFrame with calculated damages.
#     """
#     if menu_mp == 0:
#         # Handling based on 'menu_mp' (Measure Packages)
#         for category in categories:
#             print(f"End-use category: {category}")
#             for pollutant in pollutants:
#                 # Calculate emissions for each fuel type
#                 emis_naturalGas = df_copy[f'base_naturalGas_{category}_consumption'] * national_lookup.get(('naturalGas', pollutant), np.nan)
#                 emis_propane = df_copy[f'base_propane_{category}_consumption'] * national_lookup.get(('propane', pollutant), np.nan)
#                 emis_electricity = df_copy[f'base_electricity_{category}_consumption'] * (1 / (1 - td_losses)) * electricity_lookup.get((pollutant, tuple(df_copy['state'])), np.nan)

#                 if 'cooking' in category or 'clothesDrying' in category:
#                     # Total emissions for categories without fuel oil usage
#                     total_emissions = emis_electricity.fillna(0) + emis_naturalGas.fillna(0) + emis_propane.fillna(0)
#                 else:
#                     emis_fuelOil = df_copy[f'base_fuelOil_{category}_consumption'] * national_lookup.get(('fuelOil', pollutant), np.nan)
#                     # Total emissions for categories with fuel oil usage
#                     total_emissions = emis_electricity.fillna(0) + emis_naturalGas.fillna(0) + emis_propane.fillna(0) + emis_fuelOil.fillna(0)

#                 # Calculate and store damages
#                 df_copy[f'baseline_{category}_damages_{pollutant}'] = round(total_emissions * df_copy[f'margSocialCosts_{pollutant}'], 2)

#             # Calculate total health and climate damages for the category
#             df_copy[f'baseline_{category}_damages_health'] = round(df_copy[f'baseline_{category}_damages_so2'] + df_copy[f'baseline_{category}_damages_nox'] + df_copy[f'baseline_{category}_damages_pm25'], 2)
#             df_copy[f'baseline_{category}_damages_climate'] = round(df_copy[f'baseline_{category}_damages_co2'], 2)

#     else:
#         # MEASURE PACKAGES scenario (not the baseline, but specific interventions)
#         for category in categories:
#             print(f"End-use category: {category}")
#             for pollutant in pollutants:                
#                 df_copy[f'mp{menu_mp}_{category}_damages_{pollutant}'] = df_copy.apply(lambda row: row[f'mp{menu_mp}_{category}_consumption'] * (1/(1-td_losses)) * damages_CEDM_lookup.get((pollutant, row['subregion_eGRID']), np.nan), axis=1).fillna(0).round(2)

#                 df_copy[f'mp{menu_mp}_{category}_reduction_damages_{pollutant}'] = (df_copy[f'baseline_{category}_damages_{pollutant}'] - df_copy[f'mp{menu_mp}_{category}_damages_{pollutant}']).round(2)

#             df_copy[f'mp{menu_mp}_{category}_damages_health'] = (df_copy[f'mp{menu_mp}_{category}_damages_so2'] + df_copy[f'mp{menu_mp}_{category}_damages_nox'] + df_copy[f'mp{menu_mp}_{category}_damages_pm25']).round(2)
#             df_copy[f'mp{menu_mp}_{category}_damages_climate'] = df_copy[f'mp{menu_mp}_{category}_damages_co2'].round(2)

#             df_copy[f'mp{menu_mp}_{category}_reduction_damages_health'] = (df_copy[f'baseline_{category}_damages_health'] - df_copy[f'mp{menu_mp}_{category}_damages_health']).round(2)
#             df_copy[f'mp{menu_mp}_{category}_reduction_damages_climate'] = (df_copy[f'baseline_{category}_damages_climate'] - df_copy[f'mp{menu_mp}_{category}_damages_climate']).round(2)

#     return df_copy

def calculate_damages_current_grid(df_copy, menu_mp, categories, pollutants, td_losses, national_lookup, electricity_lookup, damages_CEDM_lookup):
    """
    Calculate damages for the current electricity grid scenario.

    Parameters:
        df_copy (DataFrame): The DataFrame containing consumption data.
        menu_mp (int): The menu number for the measure package.
        categories (list): List of end-use categories.
        pollutants (list): List of pollutants.
        td_losses (float): Transmission and distribution losses.
        national_lookup (dict): Lookup table for national emissions factors.
        electricity_lookup (dict): Lookup table for electricity emissions.
        damages_CEDM_lookup (dict): Lookup table for damages from CEDM.

    Returns:
        DataFrame: The DataFrame with calculated damages.
    """
    if menu_mp == 0:
        # Handling based on 'menu_mp' (Measure Packages)
        for category in categories:
            print(f"End-use category: {category}")
            for pollutant in pollutants:
                # Calculate emissions for each fuel type
                emis_naturalGas = df_copy[f'base_naturalGas_{category}_consumption'] * national_lookup.get(('naturalGas', pollutant), np.nan)
                emis_propane = df_copy[f'base_propane_{category}_consumption'] * national_lookup.get(('propane', pollutant), np.nan)
                
                # Calculate electricity emissions using the lookup
                df_copy['electricity_lookup_values'] = df_copy.apply(lambda row: electricity_lookup.get((pollutant, row['state']), np.nan), axis=1)
                emis_electricity = df_copy[f'base_electricity_{category}_consumption'] * (1 / (1 - td_losses)) * df_copy['electricity_lookup_values']

                if 'cooking' in category or 'clothesDrying' in category:
                    # Total emissions for categories without fuel oil usage
                    total_emissions = emis_electricity.fillna(0) + emis_naturalGas.fillna(0) + emis_propane.fillna(0)
                else:
                    emis_fuelOil = df_copy[f'base_fuelOil_{category}_consumption'] * national_lookup.get(('fuelOil', pollutant), np.nan)
                    # Total emissions for categories with fuel oil usage
                    total_emissions = emis_electricity.fillna(0) + emis_naturalGas.fillna(0) + emis_propane.fillna(0) + emis_fuelOil.fillna(0)

                # Calculate and store damages
                df_copy[f'baseline_{category}_damages_{pollutant}'] = round(total_emissions * df_copy[f'margSocialCosts_{pollutant}'], 2)
                
                # Debug: Check baseline damages
                print(f"Baseline Damages for {category} - {pollutant}:")
                print(df_copy[f'baseline_{category}_damages_{pollutant}'].head())

            # Calculate total health and climate damages for the category
            df_copy[f'baseline_{category}_damages_health'] = round(df_copy[f'baseline_{category}_damages_so2'] + df_copy[f'baseline_{category}_damages_nox'] + df_copy[f'baseline_{category}_damages_pm25'], 2)
            df_copy[f'baseline_{category}_damages_climate'] = round(df_copy[f'baseline_{category}_damages_co2'], 2)
            
            # Debug: Check health and climate damages
            print(f"Baseline Health Damages for {category}:")
            print(df_copy[f'baseline_{category}_damages_health'].head())
            print(f"Baseline Climate Damages for {category}:")
            print(df_copy[f'baseline_{category}_damages_climate'].head())

    else:
        # MEASURE PACKAGES scenario (not the baseline, but specific interventions)
        for category in categories:
            print(f"End-use category: {category}")
            for pollutant in pollutants:                
                df_copy[f'mp{menu_mp}_{category}_damages_{pollutant}'] = df_copy.apply(lambda row: row[f'mp{menu_mp}_{category}_consumption'] * (1/(1-td_losses)) * damages_CEDM_lookup.get((pollutant, row['subregion_eGRID']), np.nan), axis=1).fillna(0).round(2)

                df_copy[f'mp{menu_mp}_{category}_reduction_damages_{pollutant}'] = (df_copy[f'baseline_{category}_damages_{pollutant}'] - df_copy[f'mp{menu_mp}_{category}_damages_{pollutant}']).round(2)

            df_copy[f'mp{menu_mp}_{category}_damages_health'] = (df_copy[f'mp{menu_mp}_{category}_damages_so2'] + df_copy[f'mp{menu_mp}_{category}_damages_nox'] + df_copy[f'mp{menu_mp}_{category}_damages_pm25']).round(2)
            df_copy[f'mp{menu_mp}_{category}_damages_climate'] = df_copy[f'mp{menu_mp}_{category}_damages_co2'].round(2)

            df_copy[f'mp{menu_mp}_{category}_reduction_damages_health'] = (df_copy[f'baseline_{category}_damages_health'] - df_copy[f'mp{menu_mp}_{category}_damages_health']).round(2)
            df_copy[f'mp{menu_mp}_{category}_reduction_damages_climate'] = (df_copy[f'baseline_{category}_damages_climate'] - df_copy[f'mp{menu_mp}_{category}_damages_climate']).round(2)

    return df_copy


def calculate_damages_decarb_grid(df_copy, menu_mp, categories, years, td_losses, dict_margDamages_gridDecarb):
    """
    Calculates the damages due to decarbonization of the grid across multiple categories
    and pollutants, taking transmission and distribution losses into account.

    Parameters:
    - df_copy (DataFrame): A DataFrame containing the data on which calculations will be performed.
    - menu_mp (str): A modifier representing a specific menu policy or scenario.
    - categories (list): A list of equipment categories to calculate damages for.
    - years (list): A list of years for which damages will be calculated.
    - td_losses (float): Transmission and distribution loss factor to adjust consumption data.
    - dict_margDamages_gridDecarb (dict): A nested dictionary where keys are years and values are 
      sub-dictionaries mapping (subregion, pollutant) pairs to marginal damage values.

    Returns:
    - DataFrame: The original DataFrame with new columns added for calculated damages.

    This function iterates over specified equipment categories and their respective lifetimes,
    calculating damages for each pollutant in each year based on grid consumption data adjusted
    for transmission and distribution losses. The damages are calculated separately for health 
    impacts (from SO2, NOx, PM2.5) and climate impacts (from CO2). The results are added as new
    columns to the input DataFrame.
    """
    # Specifications for equipment lifetimes in years
    equipment_specs = {
        'heating': 15,
        'waterHeating': 12,
        'clothesDrying': 13,
        'cooking': 15
    }
    
    # List of considered pollutants
    pollutants = ['so2', 'nox', 'pm25', 'co2']

    # Dictionary to store new data columns
    new_columns_data = {}

    for category, lifetime in equipment_specs.items():
        print(f"End-use category: {category}")  
        for year in range(1, lifetime + 1):
            year_label = year + 2018
            for pollutant in pollutants:
                # Determine column names based on health or climate impact
                damage_col = f'{year_label}_mp{menu_mp}_{category}_damages' + ('_health' if pollutant in ['so2', 'nox', 'pm25'] else '_climate')
                
                # Adjust consumption data by transmission and distribution losses
                td_losses_multiplier = (1 / (1 - td_losses))

                # Calculate damages
                damage_data = df_copy[f'mp{menu_mp}_{category}_consumption'] * td_losses_multiplier
                damage_data *= df_copy['subregion_eGRID'].map(lambda x: dict_margDamages_gridDecarb[year_label].get((x, pollutant), np.nan))
                damage_data *= df_copy['naturalGas_leakage_factor']
                new_columns_data[damage_col] = damage_data.round(2)

    # Creating a new DataFrame from the dictionary of new columns
    new_columns_df = pd.DataFrame(new_columns_data, index=df_copy.index)
    
    # Concatenate this DataFrame with the original DataFrame
    df_copy = pd.concat([df_copy, new_columns_df], axis=1)

    return df_copy


# In[6]:


def calculate_annual_fuelCost(df, state_region, df_fuelPrices_perkWh, cpi_ratio):
    """
    -------------------------------------------------------------------------------------------------------
    Step 2: Calculate Annual Operating (Fuel) Costs
    -------------------------------------------------------------------------------------------------------
    - Create a mapping dictionary for fuel types
    - Create new merge cifolumns to ensure a proper match.
    - Merge df_copy with df_fuel_prices to get fuel prices for electricity, natural gas, propane, and fuel oil
    - Calculate the per kWh fuel costs for each fuel type and region
    - Calculate the baseline fuel cost 
    -------------------------------------------------------------------------------------------------------
    """
    df_copy = df.copy()

    # For Baseline Consumption (Measure Package 0)
    if menu_mp == 0:    
        # Fuel and region mappings remain unchanged
        region_mapping = {
            'South': 'South',
            'Midwest': 'Midwest',
            'West': 'Midwest',
            'Northeast': 'Northeast'
        }
        df_copy['region_merge'] = df_copy['census_region'].map(region_mapping)

        # Standardize the fuel types
        fuel_mapping = {
            'Electricity': 'electricity',
            'Natural Gas': 'naturalGas',
            'Fuel Oil': 'fuelOil',
            'Propane': 'propane'
        }
        # Apply mapping for each category of fuel usage
        categories = ['heating', 'waterHeating', 'clothesDrying', 'cooking']
        for category in categories:
            df_copy[f'fuel_type_{category}'] = df_copy[f'base_{category}_fuel'].map(fuel_mapping)

        # Calculate the per kWh fuel costs for each fuel type and region
        df_fuel_prices_lookup = df_fuelPrices_perkWh.set_index(['fuel_type', 'state_region'])['cost_per_kWh'].to_dict()

        # Step 4: Calculate the per kWh fuel costs for each fuel type and region
        df_copy['fuelPrice_electricity_perkWh'] = df_copy.apply(lambda row: df_fuel_prices_lookup.get(('electricity', row['state']), np.nan), axis=1) * cpi_ratio
        df_copy['fuelPrice_naturalGas_perkWh'] = df_copy.apply(lambda row: df_fuel_prices_lookup.get(('naturalGas', row['state']), np.nan), axis=1) * cpi_ratio
        df_copy['fuelPrice_propane_perkWh'] = df_copy.apply(lambda row: df_fuel_prices_lookup.get(('propane', row['region_merge']), np.nan), axis=1) * cpi_ratio
        df_copy['fuelPrice_fuelOil_perkWh'] = df_copy.apply(lambda row: df_fuel_prices_lookup.get(('fuelOil', 'National'), np.nan), axis=1) * cpi_ratio

        # Calculate the baseline fuel cost for each category based on the respective fuel type
        for category in categories:
            fuel_type_col = f'fuel_type_{category}'
            df_copy[f'baseline_{category}_fuelCost'] = df_copy.apply(
                lambda row: row[f'baseline_{category}_consumption'] * row[f'fuelPrice_{row[fuel_type_col]}_perkWh'], axis=1
            ).round(2)

        return df_copy
        return df_fuel_prices_lookup
    
    # For Measure Packages 7, 8, 9, and 10
    else:
        # Apply mapping for each category of fuel usage
        categories = ['heating', 'waterHeating', 'clothesDrying', 'cooking']
        for category in categories:
            # Use the calculated per kWh electricity price
            df_copy[f'mp{menu_mp}_{category}_fuelCost'] = (df_copy[f'mp{menu_mp}_{category}_consumption'] * df_copy['fuelPrice_electricity_perkWh']).round(2)
            df_copy[f'mp{menu_mp}_{category}_savings_fuelCost'] = (df_copy[f'baseline_{category}_fuelCost'].sub(df_copy[f'mp{menu_mp}_{category}_fuelCost'], axis=0, fill_value=0)).round(2)
            df_copy[f'mp{menu_mp}_{category}_delta_fuelCost'] = (df_copy[f'mp{menu_mp}_{category}_fuelCost'].sub(df_copy[f'baseline_{category}_fuelCost'], axis=0, fill_value=0)).round(2)
            df_copy[f'mp{menu_mp}_{category}_percentChange_fuelCost'] = (((df_copy[f'mp{menu_mp}_{category}_fuelCost'].sub(df_copy[f'baseline_{category}_fuelCost'], axis=0, fill_value=0)) / df_copy[f'baseline_{category}_fuelCost']) * 100).round(2)
    
        return df_copy


# # Retrofit Packages

# ## Basic Retrofit

# In[9]:


def calculate_consumption_reduction(df, category):
    categories = ['heating', 'waterHeating', 'clothesDrying', 'cooking']
    for category in categories:
        if category == 'heating':
            df[f'mp{menu_mp}_heating_reduction_consumption'] = (df[f'baseline_heating_consumption'].sub(df[f'mp{menu_mp}_heating_consumption'], axis=0, fill_value=0)).round(2) 
            df[f'mp{menu_mp}_heating_change_consumption'] = (df[f'mp{menu_mp}_heating_consumption'].sub(df[f'baseline_heating_consumption'], axis=0, fill_value=0)).round(2)
            df[f'mp{menu_mp}_heating_percentChange_consumption'] = (((df[f'mp{menu_mp}_heating_consumption'].sub(df[f'baseline_heating_consumption'], axis=0, fill_value=0)) / df[f'baseline_heating_consumption']) * 100).round(2)
        else:
            df[f'mp{menu_mp}_{category}_reduction_consumption'] = (df[f'baseline_{category}_consumption'].sub(df[f'mp{menu_mp}_{category}_consumption'], axis=0, fill_value=0)).round(2) 
            df[f'mp{menu_mp}_{category}_change_consumption'] = (df[f'mp{menu_mp}_{category}_consumption'].sub(df[f'baseline_{category}_consumption'], axis=0, fill_value=0)).round(2)
            df[f'mp{menu_mp}_{category}_percentChange_consumption'] = (((df[f'mp{menu_mp}_{category}_consumption'].sub(df[f'baseline_{category}_consumption'], axis=0, fill_value=0)) / df[f'baseline_{category}_consumption']) * 100).round(2)
    return df


# In[10]:


def df_enduse_compare(df_mp, menu_mp, df_baseline):
    # Create a new DataFrame named df_compare
    # using pd.DataFrame constructor and initialize it with columns from df_mp
    df_compare = pd.DataFrame({
        'bldg_id':df_mp['bldg_id'],
        'hvac_has_ducts': df_mp['in.hvac_has_ducts'],
        'baseline_heating_type': df_mp['in.hvac_heating_type_and_fuel'],
        'hvac_heating_efficiency': df_mp['in.hvac_heating_efficiency'],
        'hvac_heating_type_and_fuel': df_mp['in.hvac_heating_type_and_fuel'],
        'size_heat_pump_backup_primary_k_btu_h': df_mp['out.params.size_heat_pump_backup_primary_k_btu_h'],
        'size_heating_system_primary_k_btu_h': df_mp['out.params.size_heating_system_primary_k_btu_h'],
        'size_heating_system_secondary_k_btu_h': df_mp['out.params.size_heating_system_secondary_k_btu_h'],
        'upgrade_hvac_heating_efficiency': df_mp['upgrade.hvac_heating_efficiency'],
        'water_heater_efficiency': df_mp['in.water_heater_efficiency'],
        'water_heater_fuel': df_mp['in.water_heater_fuel'],
        'water_heater_in_unit': df_mp['in.water_heater_in_unit'],
        'size_water_heater_gal': df_mp['out.params.size_water_heater_gal'],
        'upgrade_water_heater_efficiency': df_mp['upgrade.water_heater_efficiency'],
        'clothes_dryer_in_unit': df_mp['in.clothes_dryer'],
        'upgrade_clothes_dryer': df_mp['upgrade.clothes_dryer'],
        'cooking_range_in_unit': df_euss_am_mp7['in.cooking_range'],
        'upgrade_cooking_range': df_euss_am_mp7['upgrade.cooking_range']
    })
    
    categories = ['heating', 'waterHeating', 'clothesDrying', 'cooking']
    for category in categories:
        if category == 'heating':
            # Heating Dataframe
            # MP9 = MP8 (Electrification, High Efficiency) + MP1 (Basic Enclosure)
            if input_mp == 'upgrade09':
                menu_mp = 9
                df_compare[f'mp{menu_mp}_heating_consumption'] = df_mp['out.electricity.heating.energy_consumption.kwh'].round(2)

                # Measure Package 1: Basic Enclosure Package
                # Attic floor insulation (upgrade.insulation_ceiling)
                df_compare['base_insulation_atticFloor'] = df_mp['in.insulation_ceiling']
                df_compare['upgrade_insulation_atticFloor'] = df_mp['upgrade.insulation_ceiling']
                df_compare['out_params_floor_area_attic_ft_2'] = df_mp['out.params.floor_area_attic_ft_2']

                # Air leakage reduction (upgrade.infiltration_reduction == '30%')
                df_compare['upgrade_infiltration_reduction'] = df_mp['upgrade.infiltration_reduction']

                # Duct sealing (upgrade.ducts == '10% Leakage, R-8')            
                df_compare['base_ducts'] = df_mp['in.ducts']
                df_compare['upgrade_duct_sealing'] = df_mp['upgrade.ducts']
                df_compare['out_params_duct_unconditioned_surface_area_ft_2'] = df_mp['out.params.duct_unconditioned_surface_area_ft_2']

                # Drill-and-fill wall insulation (upgrade.insulation_wall == 'Wood Stud, R-13')
                df_compare['base_insulation_wall'] = df_mp['in.insulation_wall']
                df_compare['upgrade_insulation_wall'] = df_mp['upgrade.insulation_wall']
                df_compare['out_params_wall_area_above_grade_exterior_ft_2'] = df_mp['out.params.wall_area_above_grade_exterior_ft_2']

            # MP8 = MP8 (Electrification, High Efficiency) + MP2 (Enhanced Enclosure)
            elif input_mp == 'upgrade10':
                menu_mp = 10
                df_compare[f'mp{menu_mp}_heating_consumption'] = df_mp['out.electricity.heating.energy_consumption.kwh'].round(2)

                # Measure Package 1: Basic Enclosure Package
                # Attic floor insulation (upgrade.insulation_ceiling)
                df_compare['base_insulation_atticFloor'] = df_mp['in.insulation_ceiling']
                df_compare['upgrade_insulation_atticFloor'] = df_mp['upgrade.insulation_ceiling']
                df_compare['out_params_floor_area_attic_ft_2'] = df_mp['out.params.floor_area_attic_ft_2']

                # Air leakage reduction (upgrade.infiltration_reduction == '30%')
                df_compare['upgrade_infiltration_reduction'] = df_mp['upgrade.infiltration_reduction']

                # Duct sealing (upgrade.ducts == '10% Leakage, R-8')                        
                df_compare['base_ducts'] = df_mp['in.ducts']
                df_compare['upgrade_duct_sealing'] = df_mp['upgrade.ducts']
                df_compare['out_params_duct_unconditioned_surface_area_ft_2'] = df_mp['out.params.duct_unconditioned_surface_area_ft_2']

                # Drill-and-fill wall insulation (upgrade.insulation_wall == 'Wood Stud, R-13')
                df_compare['base_insulation_wall'] = df_mp['in.insulation_wall']
                df_compare['upgrade_insulation_wall'] = df_mp['upgrade.insulation_wall']
                df_compare['out_params_wall_area_above_grade_exterior_ft_2'] = df_mp['out.params.wall_area_above_grade_exterior_ft_2']

                # Measure Package 2: Enhanced Enclosure Package
                # Foundation wall insulation and rim joist insulation
                df_compare['base_foundation_type'] = df_mp['in.geometry_foundation_type']
                df_compare['base_insulation_foundation_wall'] = df_mp['in.insulation_foundation_wall']
                df_compare['base_insulation_rim_joist'] = df_mp['in.insulation_rim_joist']

                # Only upgrade column for foundation wall insulation, but we will assume technical documentation and modeling consistent
                df_compare['upgrade_insulation_foundation_wall'] = df_mp['upgrade.insulation_foundation_wall']
                df_compare['out_params_floor_area_foundation_ft_2'] = df_mp['out.params.floor_area_foundation_ft_2']
                df_compare['out_params_rim_joist_area_above_grade_exterior_ft_2'] = df_mp['out.params.rim_joist_area_above_grade_exterior_ft_2']                        

                # Seal Vented Crawl Space
                df_compare['upgrade_seal_crawlspace'] = df_mp['upgrade.geometry_foundation_type']

                # Insulate finished attics and cathedral ceilings
                df_compare['base_insulation_roof'] = df_mp['in.insulation_roof']
                df_compare['upgrade_insulation_roof'] = df_mp['upgrade.insulation_roof']
                df_compare['out_params_roof_area_ft_2'] = df_mp['out.params.roof_area_ft_2']
            
            else:
                df_compare[f'mp{menu_mp}_heating_consumption'] = df_mp['out.electricity.heating.energy_consumption.kwh'].round(2)
        # Water Heating Dataframe    
        elif category == 'waterHeating':
            df_compare[f'mp{menu_mp}_waterHeating_consumption'] = df_mp['out.electricity.hot_water.energy_consumption.kwh'].round(2)

        # Clothes Drying Dataframe
        elif category == 'clothesDrying':
            df_compare[f'mp{menu_mp}_clothesDrying_consumption'] = df_mp['out.electricity.clothes_dryer.energy_consumption.kwh'].round(2)

        # Cooking Dataframe
        elif category == 'cooking':
            df_compare[f'mp{menu_mp}_cooking_consumption'] = df_euss_am_mp7['out.electricity.range_oven.energy_consumption.kwh'].round(2)
            
    # Merge dataframes on bldg id column so everything is lined up
    df_compare = pd.merge(df_baseline, df_compare, how='inner', on = 'bldg_id')
    calculate_consumption_reduction(df_compare, category)    
        
    return df_compare


# In[11]:


def summarize_stats_table(df, data_columns, column_name_mapping, number_formatting, include_zero=True):
    """
    Generate a formatted summary statistics table for specified columns in a DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame from which to compute statistics.
    - data_columns (list of str): The columns to include in the summary statistics.
    - column_name_mapping (dict): A dictionary to rename the columns in the summary statistics output.
    - number_formatting (str): The format string to use for numeric values in the output.
    - include_zero (bool, optional): Whether to include zero values in the statistics. Defaults to True.
      If False, zeros are replaced with NaN, which are then ignored in the computations.

    Returns:
    - DataFrame: A DataFrame containing the summary statistics, with formatted numeric values
      and renamed columns according to the input specifications.
    """

    # Create a copy of the DataFrame to avoid modifying the original data
    df_copy = df.copy()

    # Replace 0 values with NaN in the selected columns if include_zero is set to False
    if not include_zero:
        df_copy[data_columns] = df_copy[data_columns].replace(0, np.nan)

    # Compute summary statistics for the selected columns
    # The 'describe' function returns summary statistics including count, mean, std, min, 25%, 50%, 75%, max
    # Apply formatting to each number in these statistics according to the given format
    summary_stats = df_copy[data_columns].describe().apply(lambda col: col.map(lambda x: f"{x:{number_formatting}}"))

    # Rename the columns in the summary statistics DataFrame according to the provided mapping
    summary_stats.rename(columns=column_name_mapping, inplace=True)

    return summary_stats


# In[12]:


# Dictionary for storing the lifetimes of different equipment categories
equipment_specs = {
    'heating': 15,
    'waterHeating': 12,
    'clothesDrying': 13,
    'cooking': 15
}

def calculate_public_npv(df, interest_rate, grid_decarb=False):
    """
    Calculate the public Net Present Value (NPV) for a specific category of damages,
    taking into account changes in emissions-related damages over the lifetime of equipment,
    under scenarios with and without grid decarbonization.

    The function computes the NPV from a public perspective by assessing the health and climate damages
    over the lifetime of a project (e.g., a heat pump installation). It supports two scenarios:
    - Non-Decarbonized Grid: Assumes consistent damage reductions over the equipment's lifetime.
    - Decarbonized Grid: Accounts for annual changes in damage reductions due to potential grid decarbonization.

    Parameters:
    - df (DataFrame): A pandas DataFrame containing the relevant data.
    - interest_rate (float): The discount rate used in the NPV calculation.
    - grid_decarb (bool, optional): Flag indicating whether grid decarbonization is considered. Defaults to False.

    Returns:
    - DataFrame: The input DataFrame with an additional column containing the calculated public NPV for the specified category.
    """
    df_copy = df.copy()
    
    # Check and manage existing DataFrame with new columns
    new_columns_df = pd.DataFrame(index=df_copy.index)  # DataFrame to hold new or modified columns

    # Depending on the grid decarbonization status, calculate damages accordingly
    if not grid_decarb:
        new_columns_df = calculate_lifetime_damages(df_copy, menu_mp, equipment_specs, interest_rate)
    else:
        new_columns_df = calculate_lifetime_damages_gridDecarb(df_copy, menu_mp, equipment_specs, interest_rate)

    # Merge new columns ensuring no duplicates
    for col in new_columns_df.columns:
        if col not in df_copy.columns:
            df_copy[col] = new_columns_df[col]

    return df_copy

def calculate_lifetime_damages(df_copy, menu_mp, equipment_specs, interest_rate):
    for category, lifetime in equipment_specs.items():      
        base_climate = df_copy[f'baseline_{category}_damages_climate']
        base_health = df_copy[f'baseline_{category}_damages_health']
                
        retrofit_climate = df_copy[f'mp{menu_mp}_{category}_damages_climate']
        retrofit_health = df_copy[f'mp{menu_mp}_{category}_damages_health']                
        
        base_damages = base_climate + base_health
        retrofit_damages = retrofit_climate + retrofit_health
        
        # Compute and round the NPV, then store it in a new DataFrame column
        df_copy[f'mp{menu_mp}_{category}_climate_npv'] = round(((base_climate - retrofit_climate)) * ((1 - ((1 + interest_rate) ** (-1 * lifetime))) / interest_rate), 2)
        df_copy[f'mp{menu_mp}_{category}_health_npv'] = round(((base_health - retrofit_health)) * ((1 - ((1 + interest_rate) ** (-1 * lifetime))) / interest_rate), 2)
        df_copy[f'mp{menu_mp}_{category}_public_npv'] = round(((base_damages - retrofit_damages)) * ((1 - ((1 + interest_rate) ** (-1 * lifetime))) / interest_rate), 2)
        
    return df_copy

def calculate_lifetime_damages_gridDecarb(df_copy, menu_mp, equipment_specs, interest_rate):
    # Decarbonized Grid Scenario:
    # Process each category of damages
    for category, lifetime in equipment_specs.items():
        df_copy[f'gridDecarb_mp{menu_mp}_{category}_climate_npv'] = 0
        df_copy[f'gridDecarb_mp{menu_mp}_{category}_health_npv'] = 0
        df_copy[f'gridDecarb_mp{menu_mp}_{category}_public_npv'] = 0
            
        for year in range(1, lifetime + 1):
            base_climate = df_copy[f'baseline_{category}_damages_climate']
            base_health = df_copy[f'baseline_{category}_damages_health']
                
            gridDecarb_retrofit_climate = df_copy[f'{year + 2018}_mp{menu_mp}_{category}_damages_climate']
            gridDecarb_retrofit_health = df_copy[f'{year + 2018}_mp{menu_mp}_{category}_damages_health']
                
            base_damages = base_climate + base_health
            gridDecarb_retrofit_damages = gridDecarb_retrofit_climate + gridDecarb_retrofit_health
                
            discount_factor = ( 1 / ((1 + interest_rate) ** year))
                
            gridDecarb_climate_npv = (base_climate - gridDecarb_retrofit_climate) * discount_factor
            df_copy[f'gridDecarb_mp{menu_mp}_{category}_climate_npv'] += gridDecarb_climate_npv.round(2)
                
            gridDecarb_health_npv = (base_health - gridDecarb_retrofit_health) * discount_factor
            df_copy[f'gridDecarb_mp{menu_mp}_{category}_health_npv'] += gridDecarb_health_npv.round(2)
                
            gridDecarb_public_npv = (base_damages - gridDecarb_retrofit_damages) * discount_factor
            df_copy[f'gridDecarb_mp{menu_mp}_{category}_public_npv'] += gridDecarb_public_npv.round(2)
    
    return df_copy


# In[13]:


# Use CCI to adjust for cost differences when compared to the national average
# Function to map city to its average cost
def map_average_cost(city):
    if city in average_cost_map:
        return average_cost_map[city]
    elif city == 'Not in a census Place' or city == 'In another census Place':
        return average_cost_map.get('+30 City Average')
    else:
        return average_cost_map.get('+30 City Average')


# In[14]:


def obtain_heating_system_specs(df):
    # Check if necessary columns are in the DataFrame
    necessary_columns = ['size_heating_system_primary_k_btu_h', 'size_heat_pump_backup_primary_k_btu_h',
                         'size_heating_system_secondary_k_btu_h', 'baseline_heating_type']
    if not all(column in df.columns for column in necessary_columns):
        raise ValueError("DataFrame does not contain all necessary columns.")

    # Total heating load in kBtuh
    df['total_heating_load_kBtuh'] = df['size_heating_system_primary_k_btu_h'] + df['size_heat_pump_backup_primary_k_btu_h'] + df['size_heating_system_secondary_k_btu_h']
    
#     # Total heating load in kW
#     df['total_heating_load_kW'] = df['total_heating_load_kBtuh'] * 1000 / 3412.142
   
    # Use regex to remove the fuel and leave only the heating type:
    df['baseline_heating_type'] = df['baseline_heating_type'].str.extract(r'^(?:\d+\s+)?(?:Natural Gas|Electricity|Propane|Fuel Oil|Fuel)\s+(?:Fuel\s+)?(?:Electric\s+)?(.+)$')
    
    # AFUE extraction for existing, baseline equipment (Replacement Costs)
    df['baseline_AFUE'] = df['hvac_heating_efficiency'].str.extract(r'([\d.]+)%').astype(float)
    
    # SEER extraction for existing, baseline equipment (Replacement Costs)
    df['baseline_SEER'] = df['hvac_heating_efficiency'].str.extract(r'SEER ([\d.]+)').astype(float)
    
    # HSPF extraction for existing, baseline equipment (Replacement Costs)
    df['baseline_HSPF'] = df['hvac_heating_efficiency'].str.extract(r'([\d.]+) HSPF').astype(float)

    # HSPF extraction for upgraded equipment (New Install Costs)
    df['ugrade_newInstall_HSPF'] = df['upgrade_hvac_heating_efficiency'].str.extract(r'(\d+\.\d+)')
    
    return df


# In[15]:


def calculate_heating_replacementCost(df, cost_dict, rsMeans_national_avg):
    # Create conditions for different scenarios
    conditions = [
        (df['base_heating_fuel'] == 'Propane'),
        (df['base_heating_fuel'] == 'Fuel Oil'),
        (df['base_heating_fuel'] == 'Natural Gas'),
        (df['base_heating_fuel'] == 'Electricity') & (df['heating_type'] == 'Electricity ASHP'),
        (df['base_heating_fuel'] == 'Electricity')
    ]

    # Corresponding (technology, efficiency) pairs
    tech_eff_pairs = [
        ('Propane Furnace', '94 AFUE'),
        ('Fuel Oil Furnace', '95 AFUE'),
        ('Natural Gas Furnace', '95 AFUE'),
        ('Electric ASHP', 'SEER 18, 9.3 HSPF'),
        ('Electric Furnace', '100 AFUE')
    ]

    # Map each condition to its tech and efficiency
    tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
    eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default='unknown')

    # Prepare for calculation
    for cost_type in ['progressive', 'reference', 'conservative']:
        # Compute costs vectorized
        unit_cost = np.array([cost_dict.get((t, e), {}).get(f'unitCost_{cost_type}', 0) for t, e in zip(tech, eff)])
        other_cost = np.array([cost_dict.get((t, e), {}).get(f'otherCost_{cost_type}', 0) for t, e in zip(tech, eff)])
        cost_per_kBtuh = np.array([cost_dict.get((t, e), {}).get(f'cost_per_kBtuh_{cost_type}', 0) for t, e in zip(tech, eff)])
        
        # Calculate installed cost
        replacement_cost = (unit_cost + other_cost + df['total_heating_load_kBtuh'] * cost_per_kBtuh) * (df['rsMeans_CCI_avg'] / rsMeans_national_avg)
        df[f'mp{menu_mp}_heating_replacementCost_{cost_type}'] = np.round(replacement_cost, 2)

    return df

# Assume `df` is your DataFrame, `cost_dict` is your preloaded cost dictionary, and `rsMeans_national_avg` is the national average cost index.


# In[16]:


# def calculate_heating_installationCost(df, cost_dict, rsMeans_national_avg):
#     # Create conditions for different scenarios
#     conditions = [
#         (df['hvac_has_ducts'] == 'Yes') & (menu_mp == 7),
#         (df['hvac_has_ducts'] == 'No') & (menu_mp == 7),
#         (df['hvac_has_ducts'] == 'Yes') & (menu_mp != 7),
#         (df['hvac_has_ducts'] == 'No') & (menu_mp != 7)
#     ]

#     # Corresponding (technology, efficiency) pairs
#     tech_eff_pairs = [
#         ('Electric ASHP', 'SEER 18, 9.3 HSPF'),
#         ('Electric MSHP', 'SEER 18, 9.6 HSPF'),
#         ('Electric MSHP - Ducted', 'SEER 15.5, 10 HSPF'),
#         ('Electric MSHP', 'SEER 29.3, 14 HSPF')
#     ]

#     # Map each condition to its tech and efficiency
#     tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
#     eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default='unknown')

#     # Prepare for calculation
#     for cost_type in ['progressive', 'reference', 'conservative']:
#         # Compute costs vectorized
#         unit_cost = np.array([cost_dict.get((t, e), {}).get(f'unitCost_{cost_type}', 0) for t, e in zip(tech, eff)])
#         other_cost = np.array([cost_dict.get((t, e), {}).get(f'otherCost_{cost_type}', 0) for t, e in zip(tech, eff)])
#         cost_per_kBtuh = np.array([cost_dict.get((t, e), {}).get(f'cost_per_kBtuh_{cost_type}', 0) for t, e in zip(tech, eff)])
        
#         # Calculate installed cost
#         installed_cost = (unit_cost + other_cost + df['total_heating_load_kBtuh'] * cost_per_kBtuh) * (df['rsMeans_CCI_avg'] / rsMeans_national_avg)
#         df[f'mp{menu_mp}_heating_installationCost_{cost_type}'] = np.round(installed_cost, 2)

#     return df


# In[17]:


import unittest
import pandas as pd
import numpy as np
from scipy.stats import norm

# Assume the calculate_heating_installationCost function is already imported

def calculate_heating_installationCost(df, cost_dict, rsMeans_national_avg, menu_mp):
    """
    Calculate the heating installation cost based on HVAC configurations, costs, and efficiency.

    Parameters:
    df (pd.DataFrame): DataFrame containing HVAC data for different scenarios.
    cost_dict (dict): Dictionary with cost information for different technology and efficiency combinations.
    rsMeans_national_avg (float): National average value for cost adjustment.
    menu_mp (int): Menu option identifier.

    Returns:
    pd.DataFrame: Updated DataFrame with calculated heating installation costs.
    """
    
    # Validate menu_mp
    valid_menu_mps = [7, 8, 9, 10]
    if menu_mp not in valid_menu_mps:
        raise ValueError("Please enter a valid measure package number for menu_mp. Should be 7, 8, 9, or 10.")
    
    # Define conditions based on HVAC duct presence and menu option
    conditions = [
        (df['hvac_has_ducts'] == 'Yes') & (menu_mp == 7),
        (df['hvac_has_ducts'] == 'No') & (menu_mp == 7),
        (df['hvac_has_ducts'] == 'Yes') & (menu_mp != 7),
        (df['hvac_has_ducts'] == 'No') & (menu_mp != 7)
    ]

    # Define corresponding technology and efficiency pairs for each condition
    tech_eff_pairs = [
        ('Electric ASHP', 'SEER 18, 9.3 HSPF'),
        ('Electric MSHP', 'SEER 18, 9.6 HSPF'),
        ('Electric MSHP - Ducted', 'SEER 15.5, 10 HSPF'),
        ('Electric MSHP', 'SEER 29.3, 14 HSPF')
    ]

    # Map conditions to their respective technology and efficiency
    tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
    eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default='unknown')

    # Initialize dictionaries to store sampled costs
    sampled_costs_dict = {}

    # Calculate costs for each component (unit cost, other cost, cost per kBtuh)
    for cost_component in ['unitCost', 'otherCost', 'cost_per_kBtuh']:
        # Extract progressive, reference, and conservative costs from the cost dictionary
        progressive_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_progressive', np.nan) for t, e in zip(tech, eff)])
        reference_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_reference', np.nan) for t, e in zip(tech, eff)])
        conservative_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_conservative', np.nan) for t, e in zip(tech, eff)])

        # Handle missing cost data
        if np.isnan(progressive_costs).any() or np.isnan(reference_costs).any() or np.isnan(conservative_costs).any():
            raise ValueError(f"Missing cost data for some technology and efficiency combinations in cost_component {cost_component}")

        # Calculate mean and standard deviation assuming the costs represent the 10th, 50th, and 90th percentiles of a normal distribution
        mean_costs = reference_costs
        std_costs = (conservative_costs - progressive_costs) / (norm.ppf(0.90) - norm.ppf(0.10))

        # Sample from the normal distribution for each row
        sampled_costs = np.random.normal(loc=mean_costs, scale=std_costs)
        sampled_costs_dict[cost_component] = sampled_costs

    # Calculate the installed cost for each row
    installed_cost = (
        sampled_costs_dict['unitCost'] +
        sampled_costs_dict['otherCost'] +
        (df['total_heating_load_kBtuh'] * sampled_costs_dict['cost_per_kBtuh'])
    ) * (df['rsMeans_CCI_avg'] / rsMeans_national_avg)

    # Add the calculated costs to the DataFrame, rounded to 2 decimal places
    df[f'mp{menu_mp}_heating_installationCost'] = np.round(installed_cost, 2)

    return df

# class TestCalculateHeatingInstallationCost(unittest.TestCase):
#     def setUp(self):
#         # Example DataFrame setup
#         self.df = pd.DataFrame({
#             'hvac_has_ducts': ['Yes', 'No', 'Yes', 'No'],
#             'total_heating_load_kBtuh': [50, 70, 80, 60],
#             'rsMeans_CCI_avg': [1.0, 1.1, 0.9, 1.05]
#         })
#         self.cost_dict = {
#             ('Electric ASHP', 'SEER 18, 9.3 HSPF'): {
#                 'unitCost_progressive': 1000,
#                 'unitCost_reference': 1200,
#                 'unitCost_conservative': 1400,
#                 'otherCost_progressive': 200,
#                 'otherCost_reference': 250,
#                 'otherCost_conservative': 300,
#                 'cost_per_kBtuh_progressive': 10,
#                 'cost_per_kBtuh_reference': 12,
#                 'cost_per_kBtuh_conservative': 14,
#             },
#             ('Electric MSHP', 'SEER 18, 9.6 HSPF'): {
#                 'unitCost_progressive': 1100,
#                 'unitCost_reference': 1300,
#                 'unitCost_conservative': 1500,
#                 'otherCost_progressive': 210,
#                 'otherCost_reference': 260,
#                 'otherCost_conservative': 310,
#                 'cost_per_kBtuh_progressive': 11,
#                 'cost_per_kBtuh_reference': 13,
#                 'cost_per_kBtuh_conservative': 15,
#             },
#             ('Electric MSHP - Ducted', 'SEER 15.5, 10 HSPF'): {
#                 'unitCost_progressive': 1200,
#                 'unitCost_reference': 1400,
#                 'unitCost_conservative': 1600,
#                 'otherCost_progressive': 220,
#                 'otherCost_reference': 270,
#                 'otherCost_conservative': 320,
#                 'cost_per_kBtuh_progressive': 12,
#                 'cost_per_kBtuh_reference': 14,
#                 'cost_per_kBtuh_conservative': 16,
#             },
#             ('Electric MSHP', 'SEER 29.3, 14 HSPF'): {
#                 'unitCost_progressive': 1300,
#                 'unitCost_reference': 1500,
#                 'unitCost_conservative': 1700,
#                 'otherCost_progressive': 230,
#                 'otherCost_reference': 280,
#                 'otherCost_conservative': 330,
#                 'cost_per_kBtuh_progressive': 13,
#                 'cost_per_kBtuh_reference': 15,
#                 'cost_per_kBtuh_conservative': 17,
#             }
#         }
#         self.rsMeans_national_avg = 1.0

#     def test_basic_valid_input_mp7(self):
#         menu_mp = 7
#         df_result = calculate_heating_installationCost(self.df.copy(), self.cost_dict, self.rsMeans_national_avg, menu_mp)
#         self.assertIn(f'mp{menu_mp}_heating_installationCost', df_result.columns)
#         self.assertFalse(df_result[f'mp{menu_mp}_heating_installationCost'].isnull().any())

#     def test_basic_valid_input_not_mp7(self):
#         menu_mp = 8
#         df_result = calculate_heating_installationCost(self.df.copy(), self.cost_dict, self.rsMeans_national_avg, menu_mp)
#         self.assertIn(f'mp{menu_mp}_heating_installationCost', df_result.columns)
#         self.assertFalse(df_result[f'mp{menu_mp}_heating_installationCost'].isnull().any())

#     def test_missing_cost_data(self):
#         menu_mp = 7
#         cost_dict_incomplete = self.cost_dict.copy()
#         del cost_dict_incomplete[('Electric MSHP', 'SEER 18, 9.6 HSPF')]
#         with self.assertRaises(ValueError):
#             calculate_heating_installationCost(self.df.copy(), cost_dict_incomplete, self.rsMeans_national_avg, menu_mp)

#     def test_invalid_menu_option(self):
#         menu_mp = 99  # Invalid menu_mp value
#         with self.assertRaises(ValueError):
#             calculate_heating_installationCost(self.df.copy(), self.cost_dict, self.rsMeans_national_avg, menu_mp)

#     def test_extreme_values(self):
#         menu_mp = 7
#         df_extreme = self.df.copy()
#         df_extreme['total_heating_load_kBtuh'] = [10000, 20000, 30000, 40000]  # Extreme values
#         df_result = calculate_heating_installationCost(df_extreme, self.cost_dict, self.rsMeans_national_avg, menu_mp)
#         self.assertIn(f'mp{menu_mp}_heating_installationCost', df_result.columns)
#         self.assertFalse(df_result[f'mp{menu_mp}_heating_installationCost'].isnull().any())

#     def test_empty_dataframe(self):
#         menu_mp = 7
#         df_empty = pd.DataFrame(columns=['hvac_has_ducts', 'total_heating_load_kBtuh', 'rsMeans_CCI_avg'])
#         df_result = calculate_heating_installationCost(df_empty, self.cost_dict, self.rsMeans_national_avg, menu_mp)
#         self.assertIn(f'mp{menu_mp}_heating_installationCost', df_result.columns)
#         self.assertTrue(df_result.empty)

# # Run the tests
# unittest.main(argv=[''], exit=False)

# # Print Statement to ensure functions are performing as expected
# print("Ran unit tests for function: calculate_heating_installationCost")


# In[18]:


def calculate_heating_installation_premium(df, rsMeans_national_avg, cpi_ratio_2021_2013):
    necessary_columns = ['hvac_cooling_type', 'heating_type', 'rsMeans_CCI_avg']
    if not all(column in df.columns for column in necessary_columns):
        raise ValueError("DataFrame does not contain all necessary columns.")
    
    for index, row in df.iterrows():
        # Initialization to zero
        premium_cost = 0
        
        # Installation cost for homes with existing AC
        # Deetjen: Replace SEER 15, 8.5 HSPF ASHP with SEER 15, 8.5 HSPF ASHP: NREL REMDB 50th Percentile Cost is $3300 USD-2013        
        if row['hvac_cooling_type'] != 'None':
            premium_cost = 0
        
        # Installation cost for homes without central AC, but an existing furnace or baseboard
        # Deetjen: Install SEER 15, 8.5 HSPF ASHP: NREL REMDB 50th Percentile Cost is $3700 USD-2013        
        elif 'Furnace' in row['heating_type'] or 'Baseboard' in row['heating_type']:
            premium_cost = 400 * cpi_ratio_2021_2013
        
        # Installation cost for homes without central AC and an existing boiler as heating system
        # Deetjen: Install SEER 15, 8.5 HSPF ASHP: NREL REMDB High Cost is $4800 USD-2013        
        elif 'Boiler' in row['heating_type']:
            premium_cost = 1500 * cpi_ratio_2021_2013
        
        # Apply CPI adjustment above and regional cost index adjustment below
        adjusted_cost = round(premium_cost * (row['rsMeans_CCI_avg'] / rsMeans_national_avg), 2)
        df.at[index, f'mp{menu_mp}_heating_installation_premium'] = adjusted_cost
        
    return df


# In[19]:


# def calculate_waterHeating_replacementCost(df, cost_dict, rsMeans_national_avg):
#     # Create conditions for different scenarios
#     conditions = [
#         (df['base_waterHeating_fuel'] == 'Fuel Oil'),
#         (df['base_waterHeating_fuel'] == 'Natural Gas'),
#         (df['base_waterHeating_fuel'] == 'Propane'),
#         (df['water_heater_efficiency'].isin(['Electric Standard', 'Electric Premium'])),
#         (df['water_heater_efficiency'] == 'Electric Heat Pump, 80 gal')
#     ]

#     # Corresponding (technology, efficiency) pairs with efficiencies as floats
#     tech_eff_pairs = [
#         ('Fuel Oil Water Heater', 0.68),
#         ('Natural Gas Water Heater', 0.67),
#         ('Propane Water Heater', 0.67),
#         ('Electric Water Heater', 0.95),
#         ('Electric Heat Pump Water Heater, 80 gal', 2.35),
#     ]

#     # Map each condition to its tech and efficiency
#     tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
#     eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default='unknown')

#     # Prepare for calculation
#     for cost_type in ['progressive', 'reference', 'conservative']:
#         # Compute costs vectorized, ensure efficiency is used as float in lookups
#         unit_cost = np.array([cost_dict.get((t, float(e) if e != 'unknown' else e), {}).get(f'unitCost_{cost_type}', 0) for t, e in zip(tech, eff)])
#         cost_per_gallon = np.array([cost_dict.get((t, float(e) if e != 'unknown' else e), {}).get(f'cost_per_gallon_{cost_type}', 0) for t, e in zip(tech, eff)])    
    
#         # Calculate installed cost
#         replacement_cost = (unit_cost + (cost_per_gallon * df['size_water_heater_gal'])) * (df['rsMeans_CCI_avg'] / rsMeans_national_avg)
#         df[f'mp{menu_mp}_waterHeating_replacementCost_{cost_type}'] = np.round(replacement_cost, 2)

#     return df


# In[20]:


import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_waterHeating_replacementCost(df, cost_dict, rsMeans_national_avg, menu_mp):
    """
    Calculate the water heating replacement cost based on existing heating fuel types, costs, and efficiency.

    Parameters:
    df (pd.DataFrame): DataFrame containing water heating data for different scenarios.
    cost_dict (dict): Dictionary with cost information for different technology and efficiency combinations.
    rsMeans_national_avg (float): National average value for cost adjustment.
    menu_mp (int): Measure package identifier.

    Returns:
    pd.DataFrame: Updated DataFrame with calculated water heating replacement costs.
    """

    # Create conditions for different scenarios
    conditions = [
        (df['base_waterHeating_fuel'] == 'Fuel Oil'),
        (df['base_waterHeating_fuel'] == 'Natural Gas'),
        (df['base_waterHeating_fuel'] == 'Propane'),
        (df['water_heater_efficiency'].isin(['Electric Standard', 'Electric Premium'])),
        (df['water_heater_efficiency'] == 'Electric Heat Pump, 80 gal')
    ]

    # Corresponding (technology, efficiency) pairs with efficiencies as floats
    tech_eff_pairs = [
        ('Fuel Oil Water Heater', 0.68),
        ('Natural Gas Water Heater', 0.67),
        ('Propane Water Heater', 0.67),
        ('Electric Water Heater', 0.95),
        ('Electric Heat Pump Water Heater, 80 gal', 2.35),
    ]

    # Map each condition to its tech and efficiency
    tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
    eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default='unknown')

    # Initialize dictionaries to store sampled costs
    sampled_costs_dict = {}

    # Calculate costs for each component (unit cost, cost per gallon)
    for cost_component in ['unitCost', 'cost_per_gallon']:
        # Extract progressive, reference, and conservative costs from the cost dictionary
        progressive_costs = np.array([cost_dict.get((t, float(e) if e != 'unknown' else e), {}).get(f'{cost_component}_progressive', np.nan) for t, e in zip(tech, eff)])
        reference_costs = np.array([cost_dict.get((t, float(e) if e != 'unknown' else e), {}).get(f'{cost_component}_reference', np.nan) for t, e in zip(tech, eff)])
        conservative_costs = np.array([cost_dict.get((t, float(e) if e != 'unknown' else e), {}).get(f'{cost_component}_conservative', np.nan) for t, e in zip(tech, eff)])

        # Handle missing cost data
        if np.isnan(progressive_costs).any() or np.isnan(reference_costs).any() or np.isnan(conservative_costs).any():
            raise ValueError(f"Missing cost data for some technology and efficiency combinations in cost_component {cost_component}")

        # Calculate mean and standard deviation assuming the costs represent the 10th, 50th, and 90th percentiles of a normal distribution
        mean_costs = reference_costs
        std_costs = (conservative_costs - progressive_costs) / (norm.ppf(0.90) - norm.ppf(0.10))

        # Sample from the normal distribution for each row
        sampled_costs = np.random.normal(loc=mean_costs, scale=std_costs)
        sampled_costs_dict[cost_component] = sampled_costs

    # Calculate the replacement cost for each row
    replacement_cost = (
        sampled_costs_dict['unitCost'] +
        (sampled_costs_dict['cost_per_gallon'] * df['size_water_heater_gal'])
    ) * (df['rsMeans_CCI_avg'] / rsMeans_national_avg)

    # Add the calculated costs to the DataFrame, rounded to 2 decimal places
    df['waterHeating_replacementCost'] = np.round(replacement_cost, 2)

    return df


# In[21]:


def calculate_waterHeating_installationCost(df, cost_dict, rsMeans_national_avg, menu_mp):
    """
    Calculate the water heating installation cost based on configurations, costs, and efficiency.

    Parameters:
    df (pd.DataFrame): DataFrame containing water heater data for different scenarios.
    cost_dict (dict): Dictionary with cost information for different technology and efficiency combinations.
    rsMeans_national_avg (float): National average value for cost adjustment.
    menu_mp (int): Measure package identifier.

    Returns:
    pd.DataFrame: Updated DataFrame with calculated water heating installation costs.
    """
    
    # Create conditions for different scenarios
    conditions = [
        (df['upgrade_water_heater_efficiency'] == 'Electric Heat Pump, 50 gal, 3.45 UEF'),
        (df['upgrade_water_heater_efficiency'] == 'Electric Heat Pump, 66 gal, 3.35 UEF'),
        (df['upgrade_water_heater_efficiency'] == 'Electric Heat Pump, 80 gal, 3.45 UEF')
    ]

    # Corresponding (technology, efficiency) pairs with efficiencies as floats
    tech_eff_pairs = [
        ('Electric Heat Pump Water Heater, 50 gal', 3.45),
        ('Electric Heat Pump Water Heater, 66 gal', 3.35),
        ('Electric Heat Pump Water Heater, 80 gal', 3.45),
    ]

    # Map each condition to its tech and efficiency
    tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
    eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default='unknown')

    # Initialize dictionaries to store sampled costs
    sampled_costs_dict = {}

    # Calculate costs for each component (unit cost and cost per gallon)
    for cost_component in ['unitCost', 'cost_per_gallon']:
        # Extract progressive, reference, and conservative costs from the cost dictionary
        progressive_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_progressive', np.nan) for t, e in zip(tech, eff)])
        reference_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_reference', np.nan) for t, e in zip(tech, eff)])
        conservative_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_conservative', np.nan) for t, e in zip(tech, eff)])

        # Handle missing cost data
        if np.isnan(progressive_costs).any() or np.isnan(reference_costs).any() or np.isnan(conservative_costs).any():
            raise ValueError(f"Missing cost data for some technology and efficiency combinations in cost_component {cost_component}")

        # Calculate mean and standard deviation assuming the costs represent the 10th, 50th, and 90th percentiles of a normal distribution
        mean_costs = reference_costs
        std_costs = (conservative_costs - progressive_costs) / (norm.ppf(0.90) - norm.ppf(0.10))

        # Sample from the normal distribution for each row
        sampled_costs = np.random.normal(loc=mean_costs, scale=std_costs)
        sampled_costs_dict[cost_component] = sampled_costs

    # Calculate the installed cost for each row
    installed_cost = (
        sampled_costs_dict['unitCost'] +
        df['size_water_heater_gal'] * sampled_costs_dict['cost_per_gallon']
    ) * (df['rsMeans_CCI_avg'] / rsMeans_national_avg)

    # Add the calculated costs to the DataFrame, rounded to 2 decimal places
    df[f'mp{menu_mp}_waterHeating_installationCost'] = np.round(installed_cost, 2)

    return df


# In[22]:


# def calculate_clothesDrying_replacementCost(df, cost_dict, rsMeans_national_avg):
#     # Create conditions for different scenarios
#     conditions = [
#         (df['base_clothesDrying_fuel'] == 'Electricity'),
#         (df['base_clothesDrying_fuel'] == 'Natural Gas'),
#         (df['base_clothesDrying_fuel'] == 'Propane'),
#     ]

#     # Corresponding (technology, efficiency) pairs with efficiencies as floats
#     tech_eff_pairs = [
#         ('Electric Clothes Dryer', 3.1),
#         ('Natural Gas Clothes Dryer', 2.75),
#         ('Propane Clothes Dryer', 2.75),
#     ]

#     # Map each condition to its tech and efficiency
#     tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
#     eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default='unknown')

#     # Prepare for calculation
#     for cost_type in ['progressive', 'reference', 'conservative']:
#         # Compute costs vectorized, ensure efficiency is used as float in lookups
#         unit_cost = np.array([cost_dict.get((t, float(e) if e != 'unknown' else e), {}).get(f'unitCost_{cost_type}', 0) for t, e in zip(tech, eff)])
    
#         # Calculate installed cost
#         replacement_cost = unit_cost * (df['rsMeans_CCI_avg'] / rsMeans_national_avg)
#         df[f'mp{menu_mp}_clothesDrying_replacementCost_{cost_type}'] = np.round(replacement_cost, 2)

#     return df


# In[23]:


import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_clothesDrying_replacementCost(df, cost_dict, rsMeans_national_avg, menu_mp):
    """
    Calculate the clothes drying replacement cost based on existing fuel types, costs, and efficiency.

    Parameters:
    df (pd.DataFrame): DataFrame containing clothes drying data for different scenarios.
    cost_dict (dict): Dictionary with cost information for different technology and efficiency combinations.
    rsMeans_national_avg (float): National average value for cost adjustment.
    menu_mp (int): Measure package identifier.

    Returns:
    pd.DataFrame: Updated DataFrame with calculated clothes drying replacement costs.
    """

    # Create conditions for different scenarios
    conditions = [
        (df['base_clothesDrying_fuel'] == 'Electricity'),
        (df['base_clothesDrying_fuel'] == 'Natural Gas'),
        (df['base_clothesDrying_fuel'] == 'Propane'),
    ]

    # Corresponding (technology, efficiency) pairs with efficiencies as floats
    tech_eff_pairs = [
        ('Electric Clothes Dryer', 3.1),
        ('Natural Gas Clothes Dryer', 2.75),
        ('Propane Clothes Dryer', 2.75),
    ]

    # Map each condition to its tech and efficiency
    tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
    eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default='unknown')

    # Initialize dictionary to store sampled costs
    sampled_costs_dict = {}

    # Calculate costs for each component (unit cost)
    cost_component = 'unitCost'
    # Extract progressive, reference, and conservative costs from the cost dictionary
    progressive_costs = np.array([cost_dict.get((t, float(e) if e != 'unknown' else e), {}).get(f'{cost_component}_progressive', np.nan) for t, e in zip(tech, eff)])
    reference_costs = np.array([cost_dict.get((t, float(e) if e != 'unknown' else e), {}).get(f'{cost_component}_reference', np.nan) for t, e in zip(tech, eff)])
    conservative_costs = np.array([cost_dict.get((t, float(e) if e != 'unknown' else e), {}).get(f'{cost_component}_conservative', np.nan) for t, e in zip(tech, eff)])

    # Handle missing cost data
    if np.isnan(progressive_costs).any() or np.isnan(reference_costs).any() or np.isnan(conservative_costs).any():
        raise ValueError(f"Missing cost data for some technology and efficiency combinations in cost_component {cost_component}")

    # Calculate mean and standard deviation assuming the costs represent the 10th, 50th, and 90th percentiles of a normal distribution
    mean_costs = reference_costs
    std_costs = (conservative_costs - progressive_costs) / (norm.ppf(0.90) - norm.ppf(0.10))

    # Sample from the normal distribution for each row
    sampled_costs = np.random.normal(loc=mean_costs, scale=std_costs)
    sampled_costs_dict[cost_component] = sampled_costs

    # Calculate the replacement cost for each row
    replacement_cost = sampled_costs_dict['unitCost'] * (df['rsMeans_CCI_avg'] / rsMeans_national_avg)

    # Add the calculated costs to the DataFrame, rounded to 2 decimal places
    df['clothesDrying_replacementCost'] = np.round(replacement_cost, 2)

    return df


# In[24]:


import numpy as np
from scipy.stats import norm

def calculate_clothesDrying_installationCost(df, cost_dict, rsMeans_national_avg, menu_mp):
    """
    Calculate the clothes drying installation cost based on configurations, costs, and efficiency.

    Parameters:
    df (pd.DataFrame): DataFrame containing clothes dryer data for different scenarios.
    cost_dict (dict): Dictionary with cost information for different technology and efficiency combinations.
    rsMeans_national_avg (float): National average value for cost adjustment.
    menu_mp (int): Measure package identifier.

    Returns:
    pd.DataFrame: Updated DataFrame with calculated clothes drying installation costs.
    """
    
    # Create conditions for different scenarios
    conditions = [
        df['upgrade_clothes_dryer'].str.contains('Electric, Premium, Heat Pump, Ventless', na=False),
        ~df['upgrade_clothes_dryer'].str.contains('Electric, Premium, Heat Pump, Ventless', na=False),
    ]
    
    tech_eff_pairs = [
        ('Electric HP Clothes Dryer', 5.2),
        ('Electric Clothes Dryer', 3.1),
    ]
    
    # Map each condition to its tech and efficiency
    tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
    eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default='unknown')

    # Initialize dictionary to store sampled costs
    sampled_costs_dict = {}

    # Calculate costs for the unit cost component
    cost_component = 'unitCost'
    
    # Extract progressive, reference, and conservative costs from the cost dictionary
    progressive_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_progressive', np.nan) for t, e in zip(tech, eff)])
    reference_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_reference', np.nan) for t, e in zip(tech, eff)])
    conservative_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_conservative', np.nan) for t, e in zip(tech, eff)])

    # Handle missing cost data
    if np.isnan(progressive_costs).any() or np.isnan(reference_costs).any() or np.isnan(conservative_costs).any():
        raise ValueError(f"Missing cost data for some technology and efficiency combinations in cost_component {cost_component}")

    # Calculate mean and standard deviation assuming the costs represent the 10th, 50th, and 90th percentiles of a normal distribution
    mean_costs = reference_costs
    std_costs = (conservative_costs - progressive_costs) / (norm.ppf(0.90) - norm.ppf(0.10))

    # Sample from the normal distribution for each row
    sampled_costs = np.random.normal(loc=mean_costs, scale=std_costs)
    sampled_costs_dict[cost_component] = sampled_costs

    # Calculate the installed cost for each row
    installed_cost = sampled_costs_dict['unitCost'] * (df['rsMeans_CCI_avg'] / rsMeans_national_avg)

    # Add the calculated costs to the DataFrame, rounded to 2 decimal places
    df[f'mp{menu_mp}_clothesDrying_installationCost'] = np.round(installed_cost, 2)

    return df


# In[25]:


# def calculate_cooking_replacementCost(df, cost_dict, rsMeans_national_avg):
#     # Create conditions for different scenarios
#     conditions = [
#         (df['base_cooking_fuel'] == 'Electricity'),
#         (df['base_cooking_fuel'] == 'Natural Gas'),
#         (df['base_cooking_fuel'] == 'Propane'),
#     ]

#     # Corresponding (technology, efficiency) pairs with efficiencies as floats
#     tech_eff_pairs = [
#         ('Electric Range', 0.74),
#         ('Natural Gas Range', 0.4),
#         ('Propane Range', 0.4)
#     ]

#     # Map each condition to its tech and efficiency
#     tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
#     eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default='unknown')

#     # Prepare for calculation
#     for cost_type in ['progressive', 'reference', 'conservative']:
#         # Compute costs vectorized, ensure efficiency is used as float in lookups
#         unit_cost = np.array([cost_dict.get((t, float(e) if e != 'unknown' else e), {}).get(f'unitCost_{cost_type}', 0) for t, e in zip(tech, eff)])
    
#         # Calculate installed cost
#         replacement_cost = unit_cost  * (df['rsMeans_CCI_avg'] / rsMeans_national_avg)
#         df[f'mp{menu_mp}_cooking_replacementCost_{cost_type}'] = np.round(replacement_cost, 2)

#     return df


# In[26]:


def calculate_cooking_replacementCost(df, cost_dict, rsMeans_national_avg, menu_mp):
    """
    Calculate the cooking replacement cost based on existing fuel types, costs, and efficiency.

    Parameters:
    df (pd.DataFrame): DataFrame containing cooking data for different scenarios.
    cost_dict (dict): Dictionary with cost information for different technology and efficiency combinations.
    rsMeans_national_avg (float): National average value for cost adjustment.
    menu_mp (int): Measure package identifier.

    Returns:
    pd.DataFrame: Updated DataFrame with calculated cooking replacement costs.
    """

    # Create conditions for different scenarios
    conditions = [
        (df['base_cooking_fuel'] == 'Electricity'),
        (df['base_cooking_fuel'] == 'Natural Gas'),
        (df['base_cooking_fuel'] == 'Propane'),
    ]

    # Corresponding (technology, efficiency) pairs with efficiencies as floats
    tech_eff_pairs = [
        ('Electric Range', 0.74),
        ('Natural Gas Range', 0.4),
        ('Propane Range', 0.4)
    ]

    # Map each condition to its tech and efficiency
    tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
    eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default='unknown')

    # Initialize dictionary to store sampled costs
    sampled_costs_dict = {}

    # Calculate costs for each component (unit cost)
    cost_component = 'unitCost'
    # Extract progressive, reference, and conservative costs from the cost dictionary
    progressive_costs = np.array([cost_dict.get((t, float(e) if e != 'unknown' else e), {}).get(f'{cost_component}_progressive', np.nan) for t, e in zip(tech, eff)])
    reference_costs = np.array([cost_dict.get((t, float(e) if e != 'unknown' else e), {}).get(f'{cost_component}_reference', np.nan) for t, e in zip(tech, eff)])
    conservative_costs = np.array([cost_dict.get((t, float(e) if e != 'unknown' else e), {}).get(f'{cost_component}_conservative', np.nan) for t, e in zip(tech, eff)])

    # Handle missing cost data
    if np.isnan(progressive_costs).any() or np.isnan(reference_costs).any() or np.isnan(conservative_costs).any():
        raise ValueError(f"Missing cost data for some technology and efficiency combinations in cost_component {cost_component}")

    # Calculate mean and standard deviation assuming the costs represent the 10th, 50th, and 90th percentiles of a normal distribution
    mean_costs = reference_costs
    std_costs = (conservative_costs - progressive_costs) / (norm.ppf(0.90) - norm.ppf(0.10))

    # Sample from the normal distribution for each row
    sampled_costs = np.random.normal(loc=mean_costs, scale=std_costs)
    sampled_costs_dict[cost_component] = sampled_costs

    # Calculate the replacement cost for each row
    replacement_cost = sampled_costs_dict['unitCost'] * (df['rsMeans_CCI_avg'] / rsMeans_national_avg)

    # Add the calculated costs to the DataFrame, rounded to 2 decimal places
    df['cooking_replacementCost'] = np.round(replacement_cost, 2)

    return df


# In[27]:


def gather_enduse_installation_data(category):
    if category == 'heating':
        # Define conditions based on HVAC duct presence and menu option
        conditions = [
            (df['hvac_has_ducts'] == 'Yes') & (menu_mp == 7),
            (df['hvac_has_ducts'] == 'No') & (menu_mp == 7),
            (df['hvac_has_ducts'] == 'Yes') & (menu_mp != 7),
            (df['hvac_has_ducts'] == 'No') & (menu_mp != 7)
        ]
    
        # Define corresponding technology and efficiency pairs for each condition
        tech_eff_pairs = [
            ('Electric ASHP', 'SEER 18, 9.3 HSPF'),
            ('Electric MSHP', 'SEER 18, 9.6 HSPF'),
            ('Electric MSHP - Ducted', 'SEER 15.5, 10 HSPF'),
            ('Electric MSHP', 'SEER 29.3, 14 HSPF')
        ]

        cost_component_list = ['unitCost', 'otherCost', 'cost_per_kBtuh']
    
    elif category == 'waterHeating':
        # Create conditions for different scenarios
        conditions = [
            (df['upgrade_water_heater_efficiency'] == 'Electric Heat Pump, 50 gal, 3.45 UEF'),
            (df['upgrade_water_heater_efficiency'] == 'Electric Heat Pump, 66 gal, 3.35 UEF'),
            (df['upgrade_water_heater_efficiency'] == 'Electric Heat Pump, 80 gal, 3.45 UEF')
        ]
    
        # Corresponding (technology, efficiency) pairs with efficiencies as floats
        tech_eff_pairs = [
            ('Electric Heat Pump Water Heater, 50 gal', 3.45),
            ('Electric Heat Pump Water Heater, 66 gal', 3.35),
            ('Electric Heat Pump Water Heater, 80 gal', 3.45),
        ]    

        cost_component_list = ['unitCost', 'cost_per_gallon']

    elif category == 'clothesDrying':
        conditions = [
            df['upgrade_clothes_dryer'].str.contains('Electric, Premium, Heat Pump, Ventless', na=False),
            ~df['upgrade_clothes_dryer'].str.contains('Electric, Premium, Heat Pump, Ventless', na=False),
        ]
        
        tech_eff_pairs = [
            ('Electric HP Clothes Dryer', 5.2),
            ('Electric Clothes Dryer', 3.1),
        ]

        cost_component_list = ['unitCost']
    
    elif category == 'cooking':
        # Create conditions for different scenarios
        conditions = [
            df['upgrade_cooking_range'].str.contains('Electric, Induction', na=False),
            ~df['upgrade_cooking_range'].str.contains('Electric, Induction', na=False),
        ]
        
        tech_eff_pairs = [
            ('Electric Induction Range', 0.84),
            ('Electric Range, Modern', 0.74),
        ]

        cost_component_list = ['unitCost']
        
    return conditions, tech_eff_pairs, cost_component_list


# In[28]:


import numpy as np
from scipy.stats import norm

def calculate_cooking_installationCost(df, cost_dict, rsMeans_national_avg, menu_mp):
    """
    Calculate the cooking installation cost based on configurations, costs, and efficiency.

    Parameters:
    df (pd.DataFrame): DataFrame containing cooking range data for different scenarios.
    cost_dict (dict): Dictionary with cost information for different technology and efficiency combinations.
    rsMeans_national_avg (float): National average value for cost adjustment.
    menu_mp (int): Menu option identifier.

    Returns:
    pd.DataFrame: Updated DataFrame with calculated cooking installation costs.
    """
    
    # Create conditions for different scenarios
    conditions = [
        df['upgrade_cooking_range'].str.contains('Electric, Induction', na=False),
        ~df['upgrade_cooking_range'].str.contains('Electric, Induction', na=False),
    ]
    
    tech_eff_pairs = [
        ('Electric Induction Range', 0.84),
        ('Electric Range, Modern', 0.74),
    ]
    
    # Map each condition to its tech and efficiency
    tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
    eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default='unknown')

    # Initialize dictionary to store sampled costs
    sampled_costs_dict = {}

    # Calculate costs for the unit cost component
    cost_component = 'unitCost'
    
    # Extract progressive, reference, and conservative costs from the cost dictionary
    progressive_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_progressive', np.nan) for t, e in zip(tech, eff)])
    reference_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_reference', np.nan) for t, e in zip(tech, eff)])
    conservative_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_conservative', np.nan) for t, e in zip(tech, eff)])

    # Handle missing cost data
    if np.isnan(progressive_costs).any() or np.isnan(reference_costs).any() or np.isnan(conservative_costs).any():
        raise ValueError(f"Missing cost data for some technology and efficiency combinations in cost_component {cost_component}")

    # Calculate mean and standard deviation assuming the costs represent the 10th, 50th, and 90th percentiles of a normal distribution
    mean_costs = reference_costs
    std_costs = (conservative_costs - progressive_costs) / (norm.ppf(0.90) - norm.ppf(0.10))

    # Sample from the normal distribution for each row
    sampled_costs = np.random.normal(loc=mean_costs, scale=std_costs)
    sampled_costs_dict[cost_component] = sampled_costs

    # Calculate the installed cost for each row
    installed_cost = sampled_costs_dict['unitCost'] * (df['rsMeans_CCI_avg'] / rsMeans_national_avg)

    # Add the calculated costs to the DataFrame, rounded to 2 decimal places
    df[f'mp{menu_mp}_cooking_installationCost'] = np.round(installed_cost, 2)

    return df


# In[29]:


import pandas as pd

def calculate_private_NPV(df, interest_rate, ira_rebates=False):
    """
    Calculate the private net present value (NPV) for various equipment categories,
    considering different cost assumptions and potential IRA rebates. The function adjusts
    equipment costs for inflation and regional cost differences, and calculates NPV based
    on cost savings between baseline and retrofit scenarios.

    Parameters:
        df (DataFrame): Input DataFrame with installation costs, fuel savings, and potential rebates.
        interest_rate (float): Annual discount rate used for NPV calculation.
        ira_rebates (bool): Flag to consider IRA rebates in calculations.

    Returns:
        DataFrame: The input DataFrame updated with calculated private NPV and adjusted equipment costs.
    """
    # Define the lifetimes of different equipment categories    
    equipment_specs = {
        'heating': 15,
        'waterHeating': 12,
        'clothesDrying': 13,
        'cooking': 15
    }
    
    # Define the cost assumption scenarios
    cost_assumptions = ['progressive', 'reference', 'conservative']
    new_columns_df = pd.DataFrame(index=df.index)  # To hold new columns
    
    # Iterate over each equipment category and cost type to calculate financial metrics
    for category, lifetime in equipment_specs.items():
        for cost_type in cost_assumptions:
            total_capital_cost, net_capital_cost = calculate_costs(df, category, cost_type, ira_rebates)
            calculate_and_update_npv(new_columns_df, df, category, cost_type, interest_rate, lifetime, total_capital_cost, net_capital_cost, ira_rebates)
    
    # Add all columns from new_columns_df at once with .concatenate to avoid dataframe fragmentation
    df = pd.concat([df, new_columns_df], axis=1)
    return df

def calculate_costs(df, category, cost_type, ira_rebates):
    """
    Calculate total and net capital costs based on the equipment category and cost assumptions.

    Parameters:
        df (DataFrame): DataFrame containing cost data.
        category (str): Equipment category.
        cost_type (str): Type of cost assumption.
        ira_rebates (bool): Flag indicating whether IRA rebates are applied.

    Returns:
        tuple: Total and net capital costs.
    """
    if not ira_rebates:
        if category == 'heating':
            # Include specific weatherization upgrade costs based on scenarios and calculate private NPV
            if input_mp == 'upgrade09':            
                weatherization_cost = df[f'mp9_enclosure_upgradeCost_{cost_type}']
            elif input_mp == 'upgrade10':
                weatherization_cost = df[f'mp10_enclosure_upgradeCost_{cost_type}']
            else:
                weatherization_cost = 0.0
            
            # Add together
            total_capital_cost = df[f'mp{menu_mp}_{category}_installationCost_{cost_type}'] + weatherization_cost + df[f'mp{menu_mp}_heating_installation_premium']
            net_capital_cost = total_capital_cost - df[f'mp{menu_mp}_{category}_replacementCost_{cost_type}']
            
        else:
            total_capital_cost = df[f'mp{menu_mp}_{category}_installationCost_{cost_type}']
            net_capital_cost = total_capital_cost - df[f'mp{menu_mp}_{category}_replacementCost_{cost_type}']
    else:
        if category == 'heating':
            # Include specific weatherization upgrade costs based on scenarios and calculate private NPV
            if input_mp == 'upgrade09':            
                weatherization_cost = df[f'mp9_enclosure_upgradeCost_{cost_type}'] - df[f'weatherization_rebate_amount_{cost_type}']
            elif input_mp == 'upgrade10':
                weatherization_cost = df[f'mp10_enclosure_upgradeCost_{cost_type}'] - df[f'weatherization_rebate_amount_{cost_type}']
            else:
                weatherization_cost = 0.0       
            
            installation_cost = df[f'mp{menu_mp}_{category}_installationCost_{cost_type}'] + weatherization_cost + df[f'mp{menu_mp}_{category}_installation_premium']
            rebate_amount = df[f'mp{menu_mp}_{category}_rebate_amount_{cost_type}']
            total_capital_cost = installation_cost - rebate_amount
            net_capital_cost = total_capital_cost - df[f'mp{menu_mp}_{category}_replacementCost_{cost_type}']
        
        else:
            installation_cost = df[f'mp{menu_mp}_{category}_installationCost_{cost_type}']
            rebate_amount = df[f'mp{menu_mp}_{category}_rebate_amount_{cost_type}']
            total_capital_cost = installation_cost - rebate_amount
            net_capital_cost = total_capital_cost - df[f'mp{menu_mp}_{category}_replacementCost_{cost_type}']

    return total_capital_cost, net_capital_cost

def calculate_and_update_npv(new_columns_df, df, category, cost_type, interest_rate, lifetime, total_capital_cost, net_capital_cost, ira_rebates):
    """
    Calculate and update the NPV values in the DataFrame based on provided capital costs.

    Parameters:
        df (DataFrame): DataFrame to update.
        category (str): Equipment category.
        cost_type (str): Type of cost assumption.
        interest_rate (float): Discount rate for NPV calculation.
        lifetime (int): Expected lifetime of the equipment.
        total_capital_cost (float): Total capital cost of the equipment.
        net_capital_cost (float): Net capital cost after considering replacements.
    """
    savings = df[f'mp{menu_mp}_{category}_savings_fuelCost']
    discount_factor = (1 - ((1 + interest_rate) ** (-lifetime))) / interest_rate
    
    # Store total and net capital costs and calculate retrofit lifecycle costs (private NPV)
    if not ira_rebates:
        # Assigning NPV calculations to new_columns_df
        new_columns_df[f'mp{menu_mp}_{category}_total_capitalCost_{cost_type}'] = total_capital_cost
        new_columns_df[f'mp{menu_mp}_{category}_net_capitalCost_{cost_type}'] = net_capital_cost
        
        new_columns_df[f'mp{menu_mp}_{category}_private_npv_total_{cost_type}'] = round(savings * discount_factor - total_capital_cost, 2)
        new_columns_df[f'mp{menu_mp}_{category}_private_npv_{cost_type}'] = round(savings * discount_factor - net_capital_cost, 2)
    else:
        # Assigning NPV calculations to new_columns_df with IRA prefixes
        new_columns_df[f'ira_mp{menu_mp}_{category}_total_capitalCost_{cost_type}'] = total_capital_cost
        new_columns_df[f'ira_mp{menu_mp}_{category}_net_capitalCost_{cost_type}'] = net_capital_cost
        
        new_columns_df[f'ira_mp{menu_mp}_{category}_private_npv_total_{cost_type}'] = round(savings * discount_factor - total_capital_cost, 2)
        new_columns_df[f'ira_mp{menu_mp}_{category}_private_npv_{cost_type}'] = round(savings * discount_factor - net_capital_cost, 2)


# In[30]:


import pandas as pd
import numpy as np

def adoption_decision(df, ira_rebates=False, grid_decarb=False):
    """
    Updates the provided DataFrame with new columns that reflect decisions about equipment adoption
    and public impacts based on net present values (NPV). The function handles different scenarios
    based on input flags for incentives and grid decarbonization.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing home equipment data.
        ira_rebates (bool, optional): Flag to include IRA rebates in calculations. Defaults to False.
        grid_decarb (bool, optional): Flag to include grid decarbonization impacts. Defaults to False.

    Returns:
        pandas.DataFrame: The modified DataFrame with additional columns for decisions and impacts.

    Notes:
        - The function handles multiple cost assumptions ('progressive', 'reference', 'conservative').
        - It adds columns for both individual and public economic evaluations.
        - Adoption decisions and public impacts are dynamically calculated based on the input parameters.
    """
    # Define the lifetimes of different equipment categories
    upgrade_columns = {
        'heating': 'upgrade_hvac_heating_efficiency',
        'waterHeating': 'upgrade_water_heater_efficiency',
        'clothesDrying': 'upgrade_clothes_dryer',
        'cooking': 'upgrade_cooking_range'
    }
    
    # Define different cost scenarios
    cost_assumptions = ['progressive', 'reference', 'conservative']
    new_columns_df = pd.DataFrame(index=df.index)  # DataFrame to hold new or modified columns

    # Iterate over each equipment category and its respective upgrade column
    for category, upgrade_column in upgrade_columns.items():
        for cost_type in cost_assumptions:
            # Determine prefix and suffix based on IRA rebates and grid decarbonization
            prefix = f'ira_' if ira_rebates else ''
            grid_suffix = 'gridDecarb_' if grid_decarb and ira_rebates else ''

            # Column names for net NPV, private NPV, and public NPV
            net_npv_col_name = f'{prefix}{grid_suffix}mp{menu_mp}_{category}_net_npv_{cost_type}'
            private_npv_col = f'{prefix}mp{menu_mp}_{category}_private_npv_{cost_type}'
            public_npv_col = f'{grid_suffix}mp{menu_mp}_{category}_public_npv'

            # Calculate net NPV by summing private and public NPVs
            new_columns_df[net_npv_col_name] = df[private_npv_col] + df[public_npv_col]

            # Initialize columns for adoption decisions and public impact
            adoption_col_name = f'{prefix}{grid_suffix}mp{menu_mp}_{category}_adoption_{cost_type}'
            retrofit_col_name = f'{grid_suffix}mp{menu_mp}_{category}_retrofit_publicImpact'
            new_columns_df[adoption_col_name] = 'Averse to Adoption'  # Default value for all rows
            new_columns_df[retrofit_col_name] = 'No Retrofit'  # Default public impact

            # Conditions for determining adoption decisions
            conditions = [
                (df[f'mp{menu_mp}_{category}_reduction_consumption'] == 0) | df[upgrade_column].isna(),
                df[private_npv_col] > 0,
                (df[private_npv_col] <= 0) & (new_columns_df[net_npv_col_name] > 0)
            ]
            choices = ['Existing Equipment', 'Adoption', 'Potential Adoption with Subsidy']
            new_columns_df[adoption_col_name] = np.select(conditions, choices, default='Averse to Adoption')

            # Conditions and choices for public impacts
            public_conditions = [
                df[public_npv_col] > 0,
                df[public_npv_col] < 0
            ]
            public_choices = ['Public Benefit', 'Public Detriment']
            new_columns_df[retrofit_col_name] = np.select(public_conditions, public_choices, default='No Retrofit')

    # Concatenate the new columns DataFrame to the original DataFrame once, outside the loop
    df = pd.concat([df, new_columns_df], axis=1)
    return df


# In[31]:


def check_adoption_consistency(df, category, upgrade_column):
    df_copy = df.copy()
    
    cols_to_display = ['bldg_id',
                       f'base_{category}_fuel',
                       f'{upgrade_column}',
                       f'baseline_{category}_consumption',
                       f'mp{menu_mp}_{category}_consumption',
                       f'mp{menu_mp}_{category}_reduction_consumption',
                       f'baseline_{category}_fuelCost',
                       f'mp{menu_mp}_{category}_fuelCost',        
                       f'mp{menu_mp}_{category}_savings_fuelCost',
                       f'baseline_{category}_damages_health',
                       f'baseline_{category}_damages_climate',
                       f'mp{menu_mp}_{category}_damages_health',
                       f'mp{menu_mp}_{category}_damages_climate',
                       f'mp{menu_mp}_{category}_reduction_damages_health',
                       f'mp{menu_mp}_{category}_reduction_damages_climate',
                       f'mp{menu_mp}_{category}_public_npv',
                       f'mp{menu_mp}_{category}_retrofit_publicImpact']    
    
    cost_assumptions = ['progressive', 'reference', 'conservative']    
    for cost_type in cost_assumptions:
        # Specific to current cost_type
        cost_type_cols = [
            f'mp{menu_mp}_{category}_net_capitalCost_{cost_type}',
            f'mp{menu_mp}_{category}_private_npv_{cost_type}',
            f'mp{menu_mp}_{category}_net_npv_{cost_type}',
            f'mp{menu_mp}_{category}_adoption_{cost_type}',  
        ]
        
        cols_to_display.extend(cost_type_cols)  # Use extend to flatten the list
        
    # Filter the dataframe to show only the columns relevant for the current cost_type
    df_filtered = df_copy[cols_to_display]
    
    return df_filtered


# In[32]:


def summarize_results(df_compare, category, upgrade_column):
    """
    Summarizes results from a DataFrame based on the given category.

    Parameters:
        df_compare (DataFrame): The DataFrame containing the results to be summarized.
        category (str): The category for which the results should be summarized.

    Returns:
        DataFrame: The summarized results DataFrame.

    Raises:
        None
    """
    # Check if the category is related to heating or water heating
    # These are the only end-uses with Fuel Oil
    if 'heating' in category or 'waterHeating' in category:
        fuels = ['electricity', 'fuelOil', 'naturalGas', 'propane']
    # Cooking and Clothes Drying do not use Fuel Oil versions
    else:
        fuels = ['electricity', 'naturalGas', 'propane']
    
    years = [
        '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025',
        '2026', '2027', '2028', '2029', '2030', '2031', '2032', '2033',
        '2034', '2035', '2036', '2037', '2038', '2039', '2040', '2041',
        '2042', '2043', '2044', '2045', '2046', '2047', '2048', '2049', '2050'
    ]
    
    # Define the columns to be displayed in the summarized results
    cols_to_display = ['bldg_id',
                          'state',
                          'county',
                          'puma',
                          'square_footage',
                          'income',
                          'federal_poverty_level',
                          'occupancy',
                          'tenure',
                          f'base_{category}_fuel'] + [f'base_{fuel}_{category}_consumption' for fuel in fuels] + [
                          f'baseline_{category}_consumption',
                          f'{upgrade_column}',
                          f'mp{menu_mp}_{category}_consumption', 
                          f'mp{menu_mp}_{category}_reduction_consumption',
                          f'baseline_{category}_fuelCost',
                          f'mp{menu_mp}_{category}_fuelCost',        
                          f'mp{menu_mp}_{category}_savings_fuelCost',
                          f'mp{menu_mp}_{category}_delta_fuelCost',
                          f'baseline_{category}_damages_health',
                          f'baseline_{category}_damages_climate',
                          f'mp{menu_mp}_{category}_damages_health',
                          f'mp{menu_mp}_{category}_damages_climate',
                          f'mp{menu_mp}_{category}_reduction_damages_health',
                          f'mp{menu_mp}_{category}_reduction_damages_climate',    
                          f'mp{menu_mp}_{category}_public_npv',
                          f'mp{menu_mp}_{category}_retrofit_publicImpact']
    
    cost_assumptions = ['progressive', 'reference', 'conservative']    
    for cost_type in cost_assumptions:
        # Add enclosure upgrades for MP9 and MP10 heating
        if category == 'heating':
            if menu_mp == 9 or menu_mp == 10:
                enclosure_columns = [f'mp{menu_mp}_enclosure_upgradeCost_{cost_type}']
                cols_to_display.extend(enclosure_columns)

        # The individual year calculation (2018) indicates that it is the gridDecarb scenario
        if f'2018_mp{menu_mp}_{category}_damages_health' in df_compare.columns or f'2018_mp{menu_mp}_{category}_damages_climate' in df_compare.columns:
            # Add columns for gradually decarbonizing grid
            for year in years:
                damages_gridDecarb_columns = [f'{year}_mp{menu_mp}_{category}_damages_health', f'{year}_mp{menu_mp}_{category}_damages_climate']
                cols_to_display += damages_gridDecarb_columns

        # Specific to current cost_type
        cost_type_cols = [
            f'mp{menu_mp}_{category}_installationCost_{cost_type}',
            f'mp{menu_mp}_{category}_replacementCost_{cost_type}', 
            f'mp{menu_mp}_{category}_net_capitalCost_{cost_type}',
            f'mp{menu_mp}_{category}_private_npv_{cost_type}',
            f'mp{menu_mp}_{category}_net_npv_{cost_type}',
            f'mp{menu_mp}_{category}_adoption_{cost_type}',  
        ]
        
        cols_to_display.extend(cost_type_cols)  # Use extend to flatten the list
        
    df_results = df_compare[cols_to_display]
    return df_results


# In[33]:


def calculate_percent_AMI(df_results_IRA):
    """
    Calculates the percentage of Area Median Income (AMI) and assigns a designation based on the income level.

    Parameters:
        df_results_IRA (DataFrame): Input DataFrame containing income information.
            - This is the dataframe returned from the summarize_results function
    Returns:
        DataFrame: Modified DataFrame with additional columns for income calculations and designation.
    """

    # Simplify the conditions for income categorization
    conditions = [
        df_results_IRA['income'] == '<10000',
        df_results_IRA['income'] == '200000+',
        ~df_results_IRA['income'].str.contains(r'\d+-\d+')
    ]

    # Define the choices for income categorization
    choices_med = [9999.0, 200000.0, np.nan]

    # Extract the lower and higher bounds of income ranges
    df_results_IRA['income_low'] = df_results_IRA['income'].str.extract(r'(\d+)-\d+').astype(float)
    df_results_IRA['income_high'] = df_results_IRA['income'].str.extract(r'\d+-([\d+]+)').astype(float)

    # Fill missing values in income_low and income_high with defaults
    df_results_IRA['income_low'].fillna(9999.0, inplace=True)
    df_results_IRA['income_high'].fillna(200000.0, inplace=True)

    # Calculate the median income and assign to income_med column
    df_results_IRA['income_med'] = np.select(
        conditions,
        choices_med,
        default=(df_results_IRA['income_low'] + df_results_IRA['income_high']) / 2
    )

    # Drop the intermediate columns income_low and income_high
    df_results_IRA.drop(['income_low', 'income_high'], axis=1, inplace=True)

    # Perform additional tasks on the DataFrame
    df_results_IRA['census_county_medianIncome'] = df_results_IRA['puma'].map(df_county_medianIncome.set_index('gis_joinID_puma')['median_income_USD2018'])
    df_results_IRA['income_med'] = df_results_IRA['income_med'].astype(float)
    df_results_IRA['census_county_medianIncome'] = df_results_IRA['census_county_medianIncome'].astype(float)
    df_results_IRA['percent_AMI'] = ((df_results_IRA['income_med'] / df_results_IRA['census_county_medianIncome']) * 100).round(2)

    # Categorize the income level based on percent_AMI
    conditions_lmi = [
        df_results_IRA['percent_AMI'] <= 80.0,
        (df_results_IRA['percent_AMI'] > 80.0) & (df_results_IRA['percent_AMI'] <= 150.0)
    ]
    
    choices_lmi = ['Low-Income', 'Moderate-Income']

    df_results_IRA['lowModerateIncome_designation'] = np.select(conditions_lmi, choices_lmi, default='Middle-to-Upper-Income')

    # Output the modified DataFrame
    return df_results_IRA


# In[34]:


def calculate_rebateIRA(df_results_IRA, category):
    """
    Calculates rebate amounts for different end-uses based on income designation.
    - Low-income was assigned to household incomes below 80% of area-median-income (AMI)
    - Moderate-Income was assigned to household incomes 80% to 150% AMI.
    - The Middle-to-Upper-Income label is assigned to all household incomes above 150% AMI
    
    Rebates in amounts up to 100% project coverage for low-income and 50% for moderate-income
        - In all cases below, the rebates cannot exceed the total project cost.

    Parameters:
    - df_results_IRA (pandas.DataFrame): The DataFrame containing the data.
        - From the summarize_results and calculate_percentAMI function
    - category (str): The end-use category for which the rebate is being calculated.
    - menu_mp (int): The menu_mp variable to determine weatherization rebates.

    Returns:
    - df_results_IRA (pandas.DataFrame): The DataFrame with the added f'mp{menu_mp}_{category}_rebate_amount_{cost_type}' and f'weatherization_rebate_amount_{cost_type}' columns.
    """
    
    # Measure Package 7 only replaces non-electric end-uses
    # Measure Package 8 is also an efficiency measure and replaces electric resistance
    for index, row in df_results_IRA.iterrows():
        if category == 'heating':                
            # Only rebates for air source heat pumps or mini-split heat pumps
            if 'ASHP' in str(row['upgrade_hvac_heating_efficiency']) or 'MSHP' in str(row['upgrade_hvac_heating_efficiency']):
                max_rebate_amount = 8000.0  # Maximum rebate amount for heating category
            else:
                max_rebate_amount = 0.0  # No rebate amount for electric resistance
                        
        elif category == 'waterHeating':
            # Only rebates for heat pump water heaters
            if 'Electric Heat Pump' in str(row['upgrade_water_heater_efficiency']):
                max_rebate_amount = 1750.0  # Maximum rebate amount for water heating category
            else:
                max_rebate_amount = 0.0  # No rebate amount for electric resistance

        elif category == 'clothesDrying':
            # Only rebates for ventless heat pump dryers
            if 'Electric, Premium, Heat Pump, Ventless' in str(row['upgrade_clothes_dryer']):
                max_rebate_amount = 840.0  # Maximum rebate amount for clothes drying category
            else:
                max_rebate_amount = 0.0  # No rebate amount for electric resistance

        elif category == 'cooking':
#             # Only rebates for induction
#             if 'Electric, Induction' in str(row['upgrade_cooking_range']):
            # Rebates for induction AND electric resistance
            if 'Electric, ' in str(row['upgrade_cooking_range']):
                max_rebate_amount = 840.0  # Maximum rebate amount for cooking category
            else:
                max_rebate_amount = 0.0  # No rebate amount for electric resistance

        # Initialize weatherization rebate amount to 1600
        max_weatherization_rebate_amount = 1600

        cost_assumptions = ['progressive', 'reference', 'conservative']    
        for cost_type in cost_assumptions:
            # Calculate rebate amounts for different end-uses and weatherization
            if row['lowModerateIncome_designation'] == 'Low-Income':
                project_coverage = row[f'mp{menu_mp}_{category}_installationCost_{cost_type}'] * 1.0  # Full project coverage for low-income
                if project_coverage <= max_rebate_amount:
                    df_results_IRA.at[index, f'mp{menu_mp}_{category}_rebate_amount_{cost_type}'] = project_coverage
                else:
                    df_results_IRA.at[index, f'mp{menu_mp}_{category}_rebate_amount_{cost_type}'] = max_rebate_amount

                if f'mp{menu_mp}_enclosure_upgradeCost_{cost_type}' in df_results_IRA.columns:
                    weatherization_project_coverage = row[f'mp{menu_mp}_enclosure_upgradeCost_{cost_type}'] * 1.0  # Full project coverage for low-income
                    if weatherization_project_coverage <= max_weatherization_rebate_amount:
                        df_results_IRA.at[index, f'weatherization_rebate_amount_{cost_type}'] = weatherization_project_coverage
                    else:
                        df_results_IRA.at[index, f'weatherization_rebate_amount_{cost_type}'] = max_weatherization_rebate_amount

            elif row['lowModerateIncome_designation'] == 'Moderate-Income':
                project_coverage = row[f'mp{menu_mp}_{category}_installationCost_{cost_type}'] * 0.50  # Half project coverage for moderate-income
                if project_coverage <= max_rebate_amount:
                    df_results_IRA.at[index, f'mp{menu_mp}_{category}_rebate_amount_{cost_type}'] = project_coverage
                else:
                    df_results_IRA.at[index, f'mp{menu_mp}_{category}_rebate_amount_{cost_type}'] = max_rebate_amount

                if f'mp{menu_mp}_enclosure_upgradeCost_{cost_type}' in df_results_IRA.columns:
                    weatherization_project_coverage = row[f'mp{menu_mp}_enclosure_upgradeCost_{cost_type}'] * 0.50  # Half project coverage for moderate-income
                    if weatherization_project_coverage <= max_weatherization_rebate_amount:
                        df_results_IRA.at[index, f'weatherization_rebate_amount_{cost_type}'] = weatherization_project_coverage
                    else:
                        df_results_IRA.at[index, f'weatherization_rebate_amount_{cost_type}'] = max_weatherization_rebate_amount
            else:
                df_results_IRA.at[index, f'mp{menu_mp}_{category}_rebate_amount_{cost_type}'] = 0.0  # No rebate amount for other income designations

                if menu_mp == 9 or menu_mp == 10:
                    df_results_IRA.at[index, f'weatherization_rebate_amount_{cost_type}'] = 0.0  # No rebate amount for other income designations

    return df_results_IRA


# In[35]:


def check_ira_adoption_consistency(df, category, upgrade_column):
    df_copy = df.copy()
    
    cols_to_display = ['bldg_id',
                       f'base_{category}_fuel',
                       f'{upgrade_column}',
                       f'baseline_{category}_consumption',
                       f'mp{menu_mp}_{category}_consumption',
                       f'mp{menu_mp}_{category}_reduction_consumption',
                       f'baseline_{category}_fuelCost',
                       f'mp{menu_mp}_{category}_fuelCost',        
                       f'mp{menu_mp}_{category}_savings_fuelCost',
                       f'baseline_{category}_damages_health',
                       f'baseline_{category}_damages_climate',
                       f'mp{menu_mp}_{category}_damages_health',
                       f'mp{menu_mp}_{category}_damages_climate',
                       f'mp{menu_mp}_{category}_reduction_damages_health',
                       f'mp{menu_mp}_{category}_reduction_damages_climate',
                       f'mp{menu_mp}_{category}_public_npv',
                       f'mp{menu_mp}_{category}_retrofit_publicImpact']    
    
    cost_assumptions = ['progressive', 'reference', 'conservative']    
    for cost_type in cost_assumptions:
        # Specific to current cost_type
        cost_type_cols = [
            f'ira_mp{menu_mp}_{category}_net_capitalCost_{cost_type}',
            f'ira_mp{menu_mp}_{category}_private_npv_{cost_type}',
            f'ira_mp{menu_mp}_{category}_net_npv_{cost_type}',
            f'ira_mp{menu_mp}_{category}_adoption_{cost_type}',  
        ]
        
        cols_to_display.extend(cost_type_cols)  # Use extend to flatten the list
        
    # Filter the dataframe to show only the columns relevant for the current cost_type
    df_filtered = df_copy[cols_to_display]
    
    return df_filtered


# ## Moderate Retrofit (MP9): MP8 + Basic Enclosure

# ## Advanced Retrofit (MP10): MP8 + Enhanced Enclosure
# **Notes**
# - There are some inconsistencies for variable names and syntax for calculations
# - The calculations should still end up the same regardless because of order of operations
# - Plan to update for consistency to avoid user confusion.

# In[38]:


def calculate_enclosure_retrofit_upgradeCosts(df, cost_dict, retrofit_col, params_col, rsMeans_national_avg):
    # ATTIC FLOOR INSULATION
    if retrofit_col == 'insulation_atticFloor_upgradeCost':
        # Create conditions for different scenarios
        conditions = [
            (df['upgrade_insulation_atticFloor'] == 'R-30') & (df['base_insulation_atticFloor'] == 'R-13'),
            (df['upgrade_insulation_atticFloor'] == 'R-30') & (df['base_insulation_atticFloor'] == 'R-7'),
            (df['upgrade_insulation_atticFloor'] == 'R-30') & (df['base_insulation_atticFloor'] == 'Uninsulated'),
            (df['upgrade_insulation_atticFloor'] == 'R-49') & (df['base_insulation_atticFloor'] == 'R-30'),
            (df['upgrade_insulation_atticFloor'] == 'R-49') & (df['base_insulation_atticFloor'] == 'R-19'),
            (df['upgrade_insulation_atticFloor'] == 'R-49') & (df['base_insulation_atticFloor'] == 'R-13'),
            (df['upgrade_insulation_atticFloor'] == 'R-49') & (df['base_insulation_atticFloor'] == 'R-7'),
            (df['upgrade_insulation_atticFloor'] == 'R-49') & (df['base_insulation_atticFloor'] == 'Uninsulated'),
            (df['upgrade_insulation_atticFloor'] == 'R-60') & (df['base_insulation_atticFloor'] == 'R-38'),
            (df['upgrade_insulation_atticFloor'] == 'R-60') & (df['base_insulation_atticFloor'] == 'R-30'),
            (df['upgrade_insulation_atticFloor'] == 'R-60') & (df['base_insulation_atticFloor'] == 'R-19'),
            (df['upgrade_insulation_atticFloor'] == 'R-60') & (df['base_insulation_atticFloor'] == 'R-13'),
            (df['upgrade_insulation_atticFloor'] == 'R-60') & (df['base_insulation_atticFloor'] == 'R-7'),
            (df['upgrade_insulation_atticFloor'] == 'R-60') & (df['base_insulation_atticFloor'] == 'Uninsulated'),
        ]

        # Corresponding (technology, efficiency) pairs
        tech_eff_pairs = [
            ('Attic Floor Insulation: R-30', 'R-13'),
            ('Attic Floor Insulation: R-30', 'R-7'),
            ('Attic Floor Insulation: R-30', 'Uninsulated'),
            ('Attic Floor Insulation: R-49', 'R-30'),
            ('Attic Floor Insulation: R-49', 'R-19'),
            ('Attic Floor Insulation: R-49', 'R-13'),
            ('Attic Floor Insulation: R-49', 'R-7'),
            ('Attic Floor Insulation: R-49', 'Uninsulated'),
            ('Attic Floor Insulation: R-60', 'R-38'),
            ('Attic Floor Insulation: R-60', 'R-30'),
            ('Attic Floor Insulation: R-60', 'R-19'),
            ('Attic Floor Insulation: R-60', 'R-13'),
            ('Attic Floor Insulation: R-60', 'R-7'),
            ('Attic Floor Insulation: R-60', 'Uninsulated'),
        ]
    
    # INFILTRATION REDUCTION
    elif retrofit_col == 'infiltration_reduction_upgradeCost':
        # Create conditions for different scenarios
        conditions = [
            (df['upgrade_infiltration_reduction'] == '30%')
        ]

        # Corresponding (technology, efficiency) pairs
        tech_eff_pairs = [
            ('Air Leakage Reduction: 30% Reduction', 'Varies'),
        ]

    # DUCT SEALING
    elif retrofit_col == 'duct_sealing_upgradeCost':
        # Create conditions for different scenarios
        conditions = [
            (df['upgrade_duct_sealing'] == '10% Leakage, R-8') & (df['base_ducts'].isin(['10%'])) & (df['base_ducts'] != '10% Leakage, R-8'),
            (df['upgrade_duct_sealing'] == '10% Leakage, R-8') & (df['base_ducts'].isin(['20%'])),
            (df['upgrade_duct_sealing'] == '10% Leakage, R-8') & (df['base_ducts'].isin(['30%'])),
        ]

        # Corresponding (technology, efficiency) pairs
        tech_eff_pairs = [
            ('Duct Sealing: 10% Leakage, R-8', '10%'),
            ('Duct Sealing: 10% Leakage, R-8', '20%'),
            ('Duct Sealing: 10% Leakage, R-8', '30%'),
        ]

    # DRILL AND FILL WALL INSULATION
    elif retrofit_col == 'insulation_wall_upgradeCost':
        # Create conditions for different scenarios
        conditions = [
            (df['upgrade_insulation_wall'] == 'Wood Stud, R-13')
        ]

        # Corresponding (technology, efficiency) pairs
        tech_eff_pairs = [
            ('Drill-and-fill Wall Insulation: Wood Stud, R-13', 'Wood Stud, Uninsulated'),
        ]
    
    # FOUNDATION WALL INSULATION
    elif retrofit_col == 'insulation_foundation_wall_upgradeCost':
        # Create conditions for different scenarios
        conditions = [
            (df['upgrade_insulation_foundation_wall'] == 'Wall R-10, Interior')
        ]

        # Corresponding (technology, efficiency) pairs
        tech_eff_pairs = [
            ('Foundation Wall Insulation: Wall R-10, Interior', 'Uninsulated'),
        ]
    
    # RIM JOIST INSULATION
    elif retrofit_col == 'insulation_rim_joist_upgradeCost':
        # Create conditions for different scenarios
        conditions = [
            (df['base_insulation_foundation_wall'] == 'Uninsulated') & (df['base_foundation_type'].isin(['Unvented Crawlspace', 'Vented Crawlspace', 'Heated Basement']))
        ]

        # Corresponding (technology, efficiency) pairs
        tech_eff_pairs = [
            ('Rim Joist Insulation: Wall R-10, Exterior', 'Uninsulated'),
        ]
    
    # SEAL VENTED CRAWLSPACE
    elif retrofit_col == 'seal_crawlspace_upgradeCost':
        # Create conditions for different scenarios
        conditions = [
            (df['upgrade_seal_crawlspace'] == 'Unvented Crawlspace')
        ]

        # Corresponding (technology, efficiency) pairs
        tech_eff_pairs = [
            ('Seal Vented Crawlspace: Unvented Crawlspace', 'Vented Crawlspace'),
        ]
    
    # INSULATE FINISHED ATTICS AND CATHEDRAL CEILINGS
    elif retrofit_col == 'insulation_roof_upgradeCost':
        # Create conditions for different scenarios
        conditions = [
            (df['upgrade_insulation_roof'] == 'Finished, R-30')
        ]

        # Corresponding (technology, efficiency) pairs
        tech_eff_pairs = [
            ('Insulate Finished Attics and Cathedral Ceilings: Finished, R-30', 'R-30'),
        ]
    
    # INVALID RETROFIT UPGRADE
    else:
        print("Invalid Enclosure Upgrade Selection.")

    # Map each condition to its tech and efficiency
    tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
    eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default='unknown')

    # Prepare for calculation
    for cost_type in ['progressive', 'reference', 'conservative']:
        # Compute costs vectorized
        normalized_cost = np.array([cost_dict.get((t, e), {}).get(f'normalized_cost_{cost_type}', 0) for t, e in zip(tech, eff)])
        
        # Calculate installed cost
        installed_cost = (normalized_cost * df[f'{params_col}']) * (df['rsMeans_CCI_avg'] / rsMeans_national_avg)
        df[f'{retrofit_col}_{cost_type}'] = np.round(installed_cost, 2)

    return df

# Assume `df` is your DataFrame, `cost_dict` is your preloaded cost dictionary, and `rsMeans_national_avg` is the national average cost index.


# In[ ]:





# # Storing Output Results and Data Visualization

# ## Save Results: Merge DFs and Export to CSV

# In[41]:


def clean_df_merge(df_compare, df_results_IRA, df_results_IRA_gridDecarb):
    # Identify common columns (excluding 'bldg_id' which is the merging key)
    common_columns_IRA = set(df_compare.columns) & set(df_results_IRA.columns)
    common_columns_IRA.discard('bldg_id')
        
    # Drop duplicate columns in df_results_IRA and merge
    df_results_IRA = df_results_IRA.drop(columns=common_columns_IRA)
    print(f"""Dropped the following duplicate columns before merge: 
    {common_columns_IRA}
    """)
    merged_df = pd.merge(df_compare, df_results_IRA, on='bldg_id', how='inner')

    # Repeat the steps above for the merged_df and df_results_IRA_gridDecarb
    common_columnsIRA_gridDecarb = set(merged_df.columns) & set(df_results_IRA_gridDecarb.columns)
    common_columnsIRA_gridDecarb.discard('bldg_id')
    df_results_IRA_gridDecarb = df_results_IRA_gridDecarb.drop(columns=common_columnsIRA_gridDecarb)
    print(f"""Dropped the following duplicate columns before merge: 
    {common_columnsIRA_gridDecarb}
    """)
        
    # Create cleaned, merged results df with no duplicate columns
    df_results_export = pd.merge(merged_df, df_results_IRA_gridDecarb, on='bldg_id', how='inner')
    print("Dataframes have been cleaned of duplicate columns and merged successfully. Ready to export!")
    return df_results_export


# In[42]:


def export_model_run_output(df_results_export):
    print("-------------------------------------------------------------------------------------------------------")
    # Baseline model run results
    if menu_mp == '0' or menu_mp==0:
        results_filename = f"baseline_wholeHome_results_{location_id}_{results_export_formatted_date}.csv"
        print(f"BASELINE RESULTS:")
        print(f"Dataframe results will be saved in this csv file: {results_filename}")

        # Change the directory to the upload folder and export the file
        results_change_directory = "baseline"

    # Measure Package model run results
    else:
        if menu_mp == '8' or menu_mp==8:
            print(f"MEASURE PACKAGE {menu_mp} (MP{menu_mp}) RESULTS:")
            results_filename = f"mp{menu_mp}_scenarios_results_wholeHome_{location_id}_{results_export_formatted_date}.csv"
            print(f"Dataframe results will be saved in this csv file: {results_filename}")

            # Change the directory to the upload folder and export the file
            results_change_directory = "retrofit_basic"

        elif menu_mp == '9' or menu_mp==9:
            results_filename = f"mp{menu_mp}_scenarios_results_wholeHome_{location_id}_{results_export_formatted_date}.csv"
            print(f"MEASURE PACKAGE {menu_mp} (MP{menu_mp}) RESULTS:")
            print(f"Dataframe results will be saved in this csv file: {results_filename}")

            # Change the directory to the upload folder and export the file
            results_change_directory = "retrofit_moderate"

        elif menu_mp == '10' or menu_mp==10:
            results_filename = f"mp{menu_mp}_scenarios_results_wholeHome_{location_id}_{results_export_formatted_date}.csv"
            print(f"MEASURE PACKAGE {menu_mp} (MP{menu_mp}) RESULTS:")
            print(f"Dataframe results will be saved in this csv file: {results_filename}")

            # Change the directory to the upload folder and export the file
            results_change_directory = "retrofit_advanced"

        else:
            print("No matching scenarios for this Measure Package (MP)")

    # Export dataframe results as a csv to the specified filepath
    results_export_filepath = str(results_export_directory) + "\\" + str(results_change_directory) + "\\" + str(results_filename) 
#     df_results_export.to_csv(results_export_filepath, header=False, index=False)
    df_results_export.to_csv(results_export_filepath)
    print(f"Dataframe for MP{menu_mp} WHOLE-HOME results were exported here: {results_export_filepath}")
    print("-------------------------------------------------------------------------------------------------------", "\n")


# ## Convert Results Output CSVs to Dataframes

# In[44]:


def load_scenario_data(end_use, output_folder_path, scenario_string, model_run_date_time, columns_to_string):
    # Construct the output folder path with the scenario of interest
    scenario_folder_path = os.path.join(output_folder_path, scenario_string)
    print(f"Output Results Folder Path: {scenario_folder_path}")

    # List all files in the specified folder with the specified date in the filename
    files = [f for f in os.listdir(scenario_folder_path) if os.path.isfile(os.path.join(scenario_folder_path, f)) and model_run_date_time in f]

    # Initialize dataframe as None
    df_outputs = None

    # Assume there is one main file per scenario that includes all necessary data
    if files:
        file_path = os.path.join(scenario_folder_path, files[0])  # Assumes the first file is the correct one

        if os.path.exists(file_path):
            df_outputs = pd.read_csv(file_path, index_col=0, dtype=columns_to_string)
            print(f"Loaded {end_use} data for Scenario '{scenario_string}'", "\n")
        else:
            print("File not found for the specified scenario", "\n")

    if df_outputs is None:
        print(f"No {end_use} data found for Scenario '{scenario_string}'")

    return df_outputs


# ## Visuals for Public and Private Perspective

# In[46]:


# Added base fuel color-coded legend
# Possibly update colors to make more color blind accessible
color_map_fuel = {
    'Electricity': 'seagreen',
    'Natural Gas': 'steelblue',
    'Propane': 'orange',
    'Fuel Oil': 'firebrick',
}

# Define a function to plot the histogram and percentile subplot
def create_subplot_histogram(ax, df, x_col, bin_number, x_label=None, y_label=None, lower_percentile=2.5, upper_percentile=97.5, color_code='base_fuel', statistic='count', include_zero=False, show_legend=False):
    df_copy = df.copy()
    
    if not include_zero:
        df_copy[x_col] = df_copy[x_col].replace(0, np.nan)

    lower_limit = df_copy[x_col].quantile(lower_percentile / 100)
    upper_limit = df_copy[x_col].quantile(upper_percentile / 100)

    valid_data = df_copy[x_col][(df_copy[x_col] >= lower_limit) & (df_copy[x_col] <= upper_limit)]

    # Get the corresponding color for each fuel category
    colors = [color_map_fuel.get(fuel, 'gray') for fuel in df_copy[color_code].unique()]

    # Set the hue_order to match the unique fuel categories and their corresponding colors
    hue_order = [fuel for fuel in df_copy[color_code].unique() if fuel in color_map_fuel]

    ax = sns.histplot(data=df_copy, x=valid_data, kde=False, bins=bin_number, hue=color_code, hue_order=hue_order, stat=statistic, multiple="stack", palette=colors, ax=ax, legend=show_legend)

    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=22)  # Set font size for x-axis label

    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=22)  # Set font size for y-axis label

    ax.set_xlim(left=lower_limit, right=upper_limit)

    # Set font size for tick labels
    ax.tick_params(axis='both', labelsize=22)

    sns.despine()

def create_subplot_grid_histogram(df, subplot_positions, x_cols, x_labels, y_label=None, bin_number=20, lower_percentile=2.5, upper_percentile=97.5, statistic='count', color_code='base_fuel', include_zero=False, suptitle=None, sharex=False, sharey=False, column_titles=None, show_legend=True, figure_size=(12, 10), export_filename=None, export_format='png', dpi=300):
    num_subplots = len(subplot_positions)
    num_cols = max(pos[1] for pos in subplot_positions) + 1
    num_rows = max(pos[0] for pos in subplot_positions) + 1

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figure_size, sharex=sharex, sharey=sharey)

    # Create a dictionary to map subplot positions to their respective axes
    subplot_axes = {(pos[0], pos[1]): axes[pos[0], pos[1]] for pos in subplot_positions}

    # Define the parameters for each histogram subplot
    plot_params = [{'ax': subplot_axes[pos], 'x_col': col, 'x_label': label, 'y_label': y_label, 'bin_number': bin_number, 'lower_percentile': lower_percentile, 'upper_percentile': upper_percentile, 'statistic': statistic, 'color_code': color_code, 'include_zero': include_zero, 'show_legend': show_legend}
                   for pos, col, label in zip(subplot_positions, x_cols, x_labels)]

    # Plot each histogram subplot using the defined parameters
    for params in plot_params:
        create_subplot_histogram(df=df, **params)

    # Add a super title to the entire figure if suptitle is provided
    if suptitle:
        plt.suptitle(suptitle, fontweight='bold')

    # Add titles over the columns
    if column_titles:
        for col_index, title in enumerate(column_titles):
            axes[0, col_index].set_title(title, fontsize=22, fontweight='bold')
    
    # If sharey is True, remove y-axis labels on all subplots except the leftmost ones in each row
    if sharey:
        for row_index in range(num_rows):
            for col_index in range(num_cols):
                if col_index > 0:
                    axes[row_index, col_index].set_yticklabels([])

    # Add a legend for the color mapping at the bottom of the entire figure
    legend_labels = list(color_map_fuel.keys())
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_map_fuel[label]) for label in legend_labels]
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=len(legend_labels), prop={'size': 22}, labelspacing=0.5, bbox_to_anchor=(0.5, -0.05))             
    
    # Adjust the layout
    plt.tight_layout()
    
    # Export the figure if export_filename is provided
    if export_filename:
        save_figure_path = os.path.join(save_figure_directory, export_filename)
        plt.savefig(save_figure_path, format=export_format, dpi=dpi)
    # Otherwise show the plot in Jupyter Notebook
    else:
        plt.show()


# In[47]:


# Added column titles parameter
color_map_fuel = {
    'Electricity': 'seagreen',
    'Natural Gas': 'steelblue',
    'Propane': 'orange',
    'Fuel Oil': 'firebrick',
}

# Define a function to plot the boxplots
def create_subplot_boxplot(ax, df, y_col, x_col, x_label=None, y_label=None, lower_percentile=1, upper_percentile=99, show_outliers=True, include_zero=True):
    df_copy = df.copy()

    if not include_zero:
        df_copy[x_col] = df_copy[x_col].replace(0, np.nan)

    # Get the corresponding color for each fuel category
    colors = [color_map_fuel.get(fuel, 'gray') for fuel in df_copy[y_col].unique()]

    # Set the order to match the unique fuel categories and their corresponding colors
    order = [fuel for fuel in df_copy[y_col].unique() if fuel in color_map_fuel]

    ax = sns.boxplot(data=df_copy, x=x_col, y=y_col, order=order, palette=colors, showfliers=show_outliers, ax=ax)

    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=22)  # Set font size for x-axis label

    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=22)  # Set font size for y-axis label

    # Calculate number of obs per group & median to position labels
    medians = df_copy.groupby([y_col])[x_col].median().values
    num_obs = df_copy[y_col].value_counts().values
    num_obs = [str(x) for x in num_obs.tolist()]
    num_obs = ["n = " + i for i in num_obs]

    # Set custom y-axis labels using num_obs without changing y-axis positions
    ax.set_yticklabels([f"""{label.get_text()} 
    ({num})""" for label, num in zip(ax.get_yticklabels(), num_obs)])

    # Set font size for tick labels
    ax.tick_params(axis='both', labelsize=22)
    
    sns.despine()

def create_subplot_grid_boxplot(df, subplot_positions, x_cols, x_labels, suptitle=None, y_label=None, show_outliers=False, include_zero=True, column_titles=None, figure_size=(12, 10), sharex=False, sharey=False, export_filename=None, export_format='png', dpi=300):
    num_subplots = len(subplot_positions)
    num_cols = max(pos[1] for pos in subplot_positions) + 1
    num_rows = max(pos[0] for pos in subplot_positions) + 1

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figure_size, sharex=sharex, sharey=sharey)

    # Create a dictionary to map subplot positions to their respective axes
    subplot_axes = {(pos[0], pos[1]): axes[pos[0], pos[1]] for pos in subplot_positions}

    # Define the parameters for each boxplot subplot
    plot_params = [{'ax': subplot_axes[pos], 'y_col': 'base_fuel', 'x_col': col, 'x_label': label, 'y_label': y_label, 'show_outliers': show_outliers, 'include_zero': include_zero}
                   for pos, col, label in zip(subplot_positions, x_cols, x_labels)]

    # Plot each boxplot subplot using the defined parameters
    for params in plot_params:
        create_subplot_boxplot(df=df, **params)

    # Add titles over the columns
    if column_titles:
        for col_index, title in enumerate(column_titles):
            axes[0, col_index].set_title(title, fontsize=22, fontweight='bold')

    # Add a super title to the entire figure if suptitle is provided
    if suptitle:
        plt.suptitle(suptitle, fontweight='bold')
        
    if sharey:
        for row_index in range(num_rows):
            for col_index in range(num_cols):
                if col_index > 0:
                    axes[row_index, col_index].set_yticklabels([])
                    axes[row_index, col_index].set_ylabel('')
                else:
                    # Add the y-axis label in the leftmost column
                    axes[row_index, col_index].set_ylabel(y_label)

                    # Add the fuel type labels based on unique values in 'base_fuel' column
                    if row_index == 0:
                        fuel_types = df['base_fuel'].unique()
                        axes[row_index, col_index].set_yticks(range(len(fuel_types)))
                        axes[row_index, col_index].set_yticklabels(fuel_types)

    # Add a legend for the color mapping at the bottom of the entire figure
    legend_labels = list(color_map_fuel.keys())
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_map_fuel[label]) for label in legend_labels]
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=len(legend_labels), prop={'size': 22}, labelspacing=0.5, bbox_to_anchor=(0.5, -0.05))             
        
    # Adjust the layout
    plt.tight_layout()
    
    # Export the figure if export_filename is provided
    if export_filename:
        save_figure_path = os.path.join(save_figure_directory, export_filename)
        plt.savefig(save_figure_path, format=export_format, dpi=dpi)
    # Otherwise show the plot in Jupyter Notebook
    else:
        plt.show()


# # Adoption Rate Scenario Comparison

# In[49]:


color_mapping = {
    'Existing Equipment': 'gray',
    'Adoption': 'steelblue',
    'Potential Adoption with Subsidy': 'lightblue', 
    'Averse to Adoption': 'lightsalmon',
}

def create_subplot_adoption(df, main_data_column, groups, groupby1, groupby2=None, x_label=None, y_label=None, plot_title=None, ax=None, desired_order=None, display_obs=None):
    """
    Creates a subplot showing the adoption rates across different groups using stacked bar charts.
    
    Parameters:
    - df (DataFrame): The pandas DataFrame containing the data.
    - main_data_column (str): The name of the column in df representing the main data to plot.
    - groups (int or str): Determines whether to group the data by one or two dimensions.
    - groupby1 (str): The name of the first grouping column.
    - groupby2 (str, optional): The name of the second grouping column, if applicable.
    - x_label (str, optional): Label for the x-axis.
    - y_label (str, optional): Label for the y-axis.
    - plot_title (str, optional): Title for the plot.
    - ax (matplotlib.axes.Axes, optional): The matplotlib Axes object to plot on.
    - desired_order (list, optional): The desired order of categories to display.
    - display_obs (str, optional): Determines whether to display counts or percentages of observations.
    
    Returns:
    - ax (matplotlib.axes.Axes): The Axes object with the plot.
    """  
#     # Filter out the 'Existing Equipment' category from the dataframe
#     df = df[df[main_data_column] != 'Existing Equipment']
    
    if groups == 1 or groups == '1':
        # Calculate the percentages for each combination of categories
        counts = df.groupby(f'{groupby1}')[f'{main_data_column}'].value_counts(normalize=True).unstack()

        # Calculate the total count of each groupby1 category
        total_counts = df[f'{groupby1}'].value_counts()

        # Manually define the desired order of categories
        desired_order = []

        # Check if different adoption decisions are present in counts, and add them if present
        if 'Existing Equipment' in counts.columns:
            desired_order.append('Existing Equipment') 
        if 'Adoption' in counts.columns:
            desired_order.append('Adoption')
        if 'Potential Adoption with Subsidy' in counts.columns:
            desired_order.append('Potential Adoption with Subsidy')
        if 'Averse to Adoption' in counts.columns:
            desired_order.append('Averse to Adoption')
            
        # Create a stacked bar chart
        if ax is not None:
            counts = counts[desired_order]  # Reorder the columns
            ax = counts.plot(kind='barh', stacked=True, color=[color_mapping.get(label, 'gray') for label in counts.columns], ax=ax, width=0.8)

        else:
            counts = counts[desired_order]  # Reorder the columns
            ax = counts.plot(kind='barh', stacked=True, color=[color_mapping.get(label, 'gray') for label in counts.columns], width=0.8)

#         # After plotting, remove the 'Existing Equipment' from the desired_order if present
#         desired_order = [category for category in desired_order if category != 'Existing Equipment']

        # Display the number of observations next to each bar
        if display_obs is not None:
            # Add total count of observations next to bar
            if display_obs == 'count':
                # Add the total count as text on top of each groupby1 bar
                for i, group1 in enumerate(counts.index):
                    total_count = total_counts[group1]
                    ax.text(1.1, i, total_count, va='center', ha='left')
            
            # Add percentage of housing stock represented
            elif display_obs == 'percentage':
                # Add the percentage of observation as text on top of each bar
                for i, group1 in enumerate(counts.index):
                    total_count = total_counts[group1]
                    total_observations = total_counts.sum()
                    percentage = total_count / total_observations * 100
                    ha = 'left' if total_count < 0.5 else 'right'  # Adjust the horizontal alignment based on the count
                    ax.text(1.12, i, f'{percentage:.2f}%', va='center', ha=ha, fontweight='bold')
                    
    if groups == 2 or groups == '2':
        if groupby2 is None:
            groupby2 = 'federal_poverty_level'
        if x_label is None:
            x_label = f'{main_data_column}'
        if y_label is None:
            y_label = groupby1
        if plot_title is None:
            plot_title = f'Stacked Bar Chart: {groupby1}'

        # Calculate the percentages for each combination of categories
        counts = df.groupby([groupby1, groupby2])[f'{main_data_column}'].value_counts(normalize=True).unstack()

        # Calculate the total count of each combination of groupby1 and groupby2
        total_counts = df.groupby([groupby1, groupby2]).size()

        # Manually define the desired order of categories
        desired_order = []

        # Check if different adoption decisions are present in counts, and add them if present
        if 'Existing Equipment' in counts.columns:
            desired_order.append('Existing Equipment')  
        if 'Adoption' in counts.columns:
            desired_order.append('Adoption')
        if 'Potential Adoption with Subsidy' in counts.columns:
            desired_order.append('Potential Adoption with Subsidy')
        if 'Averse to Adoption' in counts.columns:
            desired_order.append('Averse to Adoption')

        # Create a stacked bar chart
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        counts = counts[desired_order]  # Reorder the columns
        ax = counts.plot(kind='barh', stacked=True, color=[color_mapping.get(label, 'gray') for label in counts.columns], ax=ax, width=0.8)

#         # After plotting, remove the 'Existing Equipment' from the desired_order if present
#         desired_order = [category for category in desired_order if category != 'Existing Equipment']
        
        # Display the number of observations next to each bar
        if display_obs is not None:
            # Add total count of observations next to bar
            if display_obs == 'count':
                # Add the total count as text on top of each bar
                for i, (group1, group2) in enumerate(counts.index):
                    total_count = total_counts[(group1, group2)]
                    ha = 'left' if total_count < 0.5 else 'right'  # Adjust the horizontal alignment based on the count
                    ax.text(1.1, i, total_count, va='center', ha=ha, fontweight='bold')

            # Add percentage of housing stock represented
            elif display_obs == 'percentage':
                # Add the percentage of observation as text on top of each bar
                for i, (group1, group2) in enumerate(counts.index):
                    total_count = total_counts[(group1, group2)]
                    total_observations = total_counts.sum()
                    percentage = total_count / total_observations * 100
                    ha = 'left' if total_count < 0.5 else 'right'  # Adjust the horizontal alignment based on the count
                    ax.text(1.12, i, f'{percentage:.2f}%', va='center', ha=ha, fontweight='bold')

    # Set the labels and title
    ax.set_xlabel(f'{x_label}', fontsize=32)  # Set font size for x-axis label
    ax.set_ylabel(f'{y_label}', fontsize=32)  # Set font size for y-axis label
    ax.set_title(f'{plot_title}', fontweight='bold', fontsize=32)  # Set font size and bold for title
#     ax.set_title(f'{plot_title}', fontweight='bold', fontsize=40)  # Set font size and bold for title
    
    # Format the x-axis labels as percentages
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, symbol='%'))
       
    # Set background to white
    ax.set_facecolor('white')      

    # Add gridlines to the plot 
    ax.set_axisbelow(False)
    ax.grid(which='both', axis='x', linestyle='--', color='black')
    ax.grid(which='both', axis='y', linestyle='', color='none')
    
    # Remove the legend
    ax.get_legend().remove()   
    
    # Set font size for tick labels on the x-axis
    ax.tick_params(axis='x', labelsize=32)

#     # Set font size for tick labels on the y-axis
#     ax.tick_params(axis='y', labelsize=28)

    # Remove y-ticks
    ax.set_yticks([])
    
    # Set x-axis limits to include bounds at the figure edge
    ax.set_xlim(0, 1)  # Assuming percentages, adjust if necessary
    
    return ax


# In[50]:


def create_subplot_grid_adoption(dataframes, subplot_positions, x_cols, groups=2, groupby1='base_fuel', groupby2='default_groupby2', x_labels=None, plot_titles=None, y_labels=None, suptitle=None, figure_size=(12, 10), sharex=False, sharey=False, rotate_landscape=False, filter_fuel=None, display_obs=None, export_filename=None, export_format='png', dpi=300):
    """
    Creates a grid of subplots to visualize adoption rates across different groups, with an option to include 'Existing Equipment' in the analysis.

    Parameters:
    - dataframes (list of pd.DataFrame): A list of pandas DataFrames to plot.
    - subplot_positions (list of tuples): Positions of subplots in the grid, specified as (row, col) tuples.
    - x_cols (list of str): Names of the columns in each DataFrame to be used for the main data in plots.
    - groups (int, optional): Determines whether to group the data by one or two dimensions (default is 2).
    - groupby1 (str, optional): The name of the first grouping column (default is f'base_{category}_fuel').
    - groupby2 (str, optional): The name of the second grouping column, with a placeholder default 'default_groupby2'.
    - x_labels (list of str, optional): Labels for the x-axis of each subplot (defaults to names in x_cols if not provided).
    - plot_titles (list of str, optional): Titles for each subplot (defaults to empty strings if not provided).
    - y_labels (list of str, optional): Labels for the y-axis of each subplot (defaults to empty strings if not provided).
    - suptitle (str, optional): A central title for the entire figure.
    - figure_size (tuple, optional): Size of the entire figure (width, height) in inches (default is (12, 10)).
    - sharex (bool, optional): Whether subplots should share the same x-axis (default is False).
    - sharey (bool, optional): Whether subplots should share the same y-axis (default is False).
    - rotate_landscape (bool, optional): Rotates the grid for a landscape orientation (default is False).
    - filter_fuel (list, optional): Filters the data to include only specified f'base_{category}_fuel' values before plotting.
    - display_obs (str, optional): Determines whether to display counts or percentages of observations next to each bar.
    - export_filename (str, optional): If provided, the figure will be saved to this filename instead of displayed.
    - export_format (str, optional): The file format for saving the figure (default is 'png').
    - dpi (int, optional): The resolution of the figure for saving (default is 300).

    Returns:
    None. Displays or saves the figure based on the provided parameters.

    Note:
    - Ensure all provided DataFrames, column names, and other parameters are valid and consistent.
    - The function dynamically adjusts subplot arrangements and legends based on input parameters.
    - 'default_groupby2' is a placeholder and should be replaced or handled appropriately within the function.
    """   
    num_subplots = len(subplot_positions)
    num_cols = max(pos[1] for pos in subplot_positions) + 1
    num_rows = max(pos[0] for pos in subplot_positions) + 1

    if rotate_landscape:
        num_cols, num_rows = num_rows, num_cols  # Swap rows and columns for landscape

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figure_size, sharex=sharex, sharey=sharey)

    # Convert axes to a 2A array to match the grid
    axes = axes.reshape((num_rows, num_cols))

    # Default x_labels if not provided
    if x_labels is None:
        x_labels = x_cols

    # Default plot_titles if not provided
    if plot_titles is None:
        plot_titles = [''] * num_subplots

    # Default y_labels if not provided
    if y_labels is None:
        y_labels = [''] * num_subplots

    # Define the parameters for each subplot
    for idx, df in enumerate(dataframes):
        # Apply the filter_fuel if provided
        if filter_fuel:
            df = df[df['base_fuel'].isin(filter_fuel)]

        pos = subplot_positions[idx]  # Define the subplot position

        params = {
            'ax': axes[pos[0], pos[1]] if not rotate_landscape else axes[pos[1], pos[0]],
            'df': df,
            'main_data_column': x_cols[idx],
            'groups': groups,
            'groupby1': groupby1,
            'groupby2': groupby2,
            'x_label': x_labels[idx],
            'y_label': y_labels[idx],
            'plot_title': plot_titles[idx],
            'desired_order': ['Existing Equipment', 'Adoption', 'Potential Adoption with Subsidy', 'Averse to Adoption'],  # Pass the desired_order as a parameter
            'display_obs': display_obs  # Pass display_obs to the create_subplot_adoption function
        }

        # Plot each subplot using the defined parameters
        create_subplot_adoption(**params)  # Call your custom function

        if suptitle:
            plt.suptitle(suptitle, fontweight='bold')

    # Add a legend for the color mapping at the bottom of the entire figure
    legend_labels = list(color_mapping.keys())
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping[label]) for label in legend_labels]
            
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=len(legend_labels), prop={'size': 32}, labelspacing=0.5, bbox_to_anchor=(0.5, -0.05))

    # Adjust the layout
    plt.tight_layout()
    
    # Export the figure if export_filename is provided
    if export_filename:
        save_figure_path = os.path.join(save_figure_directory, export_filename)
        plt.savefig(save_figure_path, format=export_format, dpi=dpi)
    # Otherwise show the plot in Jupyter Notebook
    else:
        plt.show()


# # Adoption Rate Percentages

# In[52]:


def format_group_percentages(counts, group):
    # Initialize total adoption with subsidy to 0
    total_adoption_with_subsidy = 0
    
    # Check and sum 'Adoption' and 'Potential Adoption with Subsidy' if they exist
    if 'Adoption' in counts.columns:
        total_adoption_with_subsidy += counts.loc[group, 'Adoption']
    if 'Potential Adoption with Subsidy' in counts.columns:
        total_adoption_with_subsidy += counts.loc[group, 'Potential Adoption with Subsidy']

    # Format percentages, including checks for existence before accessing
    formatted_percentages = ', '.join(f"{decision_prefix}{counts.loc[group, decision]:.1f}%" 
                                      for decision, decision_prefix in [('Adoption', 'AD '), ('Potential Adoption with Subsidy', 'PAS ')]
                                      if decision in counts.columns)
    formatted_percentages += f", TAS {total_adoption_with_subsidy:.1f}%"
    return formatted_percentages

def print_combined_adoption_decision_percentages(dataframes, data_columns, groups, groupby1, groupby2=None, filter_fuel=None):
    # Initialize a dictionary to hold the results
    results = {}
    
    # Add a key for overall percentages
    overall_key = "('Overall')"
    results[overall_key] = []

    # Iterate over each DataFrame and corresponding main_data_column
    for df, data_column in zip(dataframes, data_columns):
#         df_filtered = df.copy()

        # Filter out the 'Existing Equipment' category from the dataframe
        df_filtered = df[df[data_column] != 'Existing Equipment']

        # Apply the filter_fuel if provided
        if filter_fuel:
            df_filtered = df_filtered[df_filtered['base_fuel'].isin(filter_fuel)]
        
        # Calculate overall percentages for the entire data column
        overall_counts = df_filtered[data_column].value_counts(normalize=True) * 100
        # Calculate Total Adoption with Subsidy
        total_adoption_with_subsidy = overall_counts.get('Adoption', 0) + overall_counts.get('Potential Adoption with Subsidy', 0)

        overall_percentages = ', '.join(f"{decision_prefix}{overall_counts[decision]:.1f}%" 
                                        for decision, decision_prefix in [('Adoption', 'AD '), ('Potential Adoption with Subsidy', 'PAS ')]
                                        if decision in overall_counts.index)
        overall_percentages += f", TAS {total_adoption_with_subsidy:.1f}%"
        results[overall_key].append(overall_percentages)
        
        if groups == 1 or groups == '1':
            # Calculate the percentages for each combination of categories
            counts = df_filtered.groupby(f'{groupby1}')[f'{data_column}'].value_counts(normalize=True).unstack() * 100
            for group in counts.index:
                key = f"('{groupby1}', '{group}')"
                if key not in results:
                    results[key] = []
                
                # Calculate and format percentages including Total Adoption with Subsidy
                formatted_percentages = format_group_percentages(counts, group)
                results[key].append(formatted_percentages)
                
        elif groups == 2 or groups == '2' and groupby2 is not None:
            # Calculate the percentages for each combination of categories
            counts = df_filtered.groupby([groupby1, groupby2])[f'{data_column}'].value_counts(normalize=True).unstack() * 100
            for group1_group2 in counts.index:
                key = f"('{group1_group2[0]}', '{group1_group2[1]}')"
                if key not in results:
                    results[key] = []

                # Calculate and format percentages including Total Adoption with Subsidy
                formatted_percentages = format_group_percentages(counts, group1_group2)
                results[key].append(formatted_percentages)
    
    # Print combined results for overall and then for each group
    for key, values in results.items():
        combined_values = ' | '.join(values)
        print(f"{key}: {combined_values}")


# In[53]:


# # Define a function to filter out specific decision columns
# def filter_columns(df):
#     # We want to keep columns containing 'Adoption' or 'Total Adoption with Subsidy' only
#     keep_columns = [col for col in df.columns if ('Adoption' in col[1] or 'Total Adoption with Subsidy' in col[1]) and 'Averse to Adoption' not in col[1] and 'Potential Adoption with Subsidy' not in col[1] and 'Existing Equipment' not in col[1]]
#     return df[keep_columns]


# In[54]:


# # Define a function to filter out specific decision columns
# def filter_columns(df):
#     # We want to keep columns containing 'Adoption' or 'Total Adoption with Subsidy' only
#     keep_columns = [col for col in df.columns if ('Adoption' in col[1] or 'Total Adoption with Subsidy' in col[1]) and 'Averse to Adoption' not in col[1] and 'Potential Adoption with Subsidy' not in col[1] and 'Existing Equipment' not in col[1]]
#     return df[keep_columns]

# def create_multiIndex_adoption_df(df, menu_mp, category, cost_assumptions):
#     # Create a copy of the dataframe
#     df_copy = df.copy()
        
#     # Begin df with these cols
#     adoption_cols = ['bldg_id', 'federal_poverty_level', 'lowModerateIncome_designation', 'state', 'end_use', 'base_fuel']

#     for cost_type in cost_assumptions:
#         cols = [
#             f'mp{menu_mp}_{category}_adoption_{cost_type}',
#             f'ira_mp{menu_mp}_{category}_adoption_{cost_type}',
#             f'ira_gridDecarb_mp{menu_mp}_{category}_adoption_{cost_type}',
#         ]
#         adoption_cols.extend(cols)
        
#     df_copy = df_copy[adoption_cols]
    
#     df_multi_index = round((df_copy.groupby(['base_fuel', 'lowModerateIncome_designation'])[adoption_columns].apply(lambda x: x.apply(lambda y: y.value_counts(normalize=True))).unstack().fillna(0) * 100), 2)
    
#     # Iterate through each adoption reference column and add the new category
#     for column in adoption_columns:
#         # Sum the 'Adoption' and 'Potential Adoption with Subsidy' categories
#         df_multi_index[(column, 'Total Adoption with Subsidy')] = (
#             df_multi_index[(column, 'Adoption')] + df_multi_index[(column, 'Potential Adoption with Subsidy')]
#         )
    
#     # Apply the function to your DataFrame
#     df_multi_index = filter_columns(df_multi_index)

#     # Specify new column order with 'Adoption' first and 'Total Adoption with Subsidy' second for each group
#     for col in adoption_cols:
#         new_order = [
#             (col, 'Adoption'),
#             (col, 'Total Adoption with Subsidy'),
#         ]

#     # Reorder the columns according to the new_order
#     df_multi_index = df_multi_index.loc[:, new_order]
    
#     # Set the 'lowModerateIncome_designation' index as a categorical index with a specified order
#     filtered_df.index = filtered_df.index.set_levels(
#         pd.Categorical(filtered_df.index.levels[1], categories=['Low-Income', 'Moderate-Income', 'Middle-to-Upper-Income'], ordered=True),
#         level=1
#     )
    
#     # Sort the DataFrame based on the entire index
#     df_multi_index = df_multi_index.sort_index(level=['base_fuel', 'lowModerateIncome_designation'])

#     return df_multi_index


# In[55]:


# cost_assumptions=['progressive', 'reference', 'conservative']
# menu_mp=8
# category='heating'

# adoption_cols = []

# for cost_type in cost_assumptions:
#     cols = [
#         f'mp{menu_mp}_{category}_adoption_{cost_type}',
#         f'ira_mp{menu_mp}_{category}_adoption_{cost_type}',
#         f'ira_gridDecarb_mp{menu_mp}_{category}_adoption_{cost_type}',
#     ]
#     adoption_cols.extend(cols)
# print(adoption_cols)

# # Group by 'base_fuel' and 'federal_poverty_level', then apply value_counts normalized
# percentages_df = round((df_basic_adoption_heating.groupby(['base_fuel', 'lowModerateIncome_designation'])[adoption_columns].apply(lambda x: x.apply(lambda y: y.value_counts(normalize=True))).unstack().fillna(0) * 100), 2)

# # Iterate through each adoption reference column and add the new category
# for column in adoption_columns:
#     # Sum the 'Adoption' and 'Potential Adoption with Subsidy' categories
#     percentages_df[(column, 'Total Adoption with Subsidy')] = (
#         percentages_df[(column, 'Adoption')] + percentages_df[(column, 'Potential Adoption with Subsidy')]
#     )

# # Rebuild the column MultiIndex to ensure proper structure and order
# percentages_df.columns = pd.MultiIndex.from_tuples(percentages_df.columns)

# # Sort the columns to ensure that 'Total Adoption with Subsidy' appears directly under its respective category
# percentages_df = percentages_df.sort_index(axis=1, ascending=False)
# # percentages_df

# def filter_columns(df):
#     # We want to keep columns containing 'Adoption' or 'Total Adoption with Subsidy' only
#     keep_columns = [col for col in df.columns if ('Adoption' in col[1] or 'Total Adoption with Subsidy' in col[1]) and 'Averse to Adoption' not in col[1] and 'Potential Adoption with Subsidy' not in col[1] and 'Existing Equipment' not in col[1]]
#     return df[keep_columns]

# # Apply the function to your DataFrame
# filtered_df = filter_columns(percentages_df)
# # filtered_df

# # Set the 'lowModerateIncome_designation' index as a categorical index with a specified order
# filtered_df.index = filtered_df.index.set_levels(
#     pd.Categorical(filtered_df.index.levels[1], categories=['Low-Income', 'Moderate-Income', 'Middle-to-Upper-Income'], ordered=True),
#     level=1
# )

# # Specify new column order with 'Adoption' first and 'Total Adoption with Subsidy' second for each group
# new_order = [
#     ('mp8_heating_adoption_reference', 'Adoption'),
#     ('mp8_heating_adoption_reference', 'Total Adoption with Subsidy'),
#     ('ira_mp8_heating_adoption_reference', 'Adoption'),
#     ('ira_mp8_heating_adoption_reference', 'Total Adoption with Subsidy'),
#     ('ira_gridDecarb_mp8_heating_adoption_reference', 'Adoption'),
#     ('ira_gridDecarb_mp8_heating_adoption_reference', 'Total Adoption with Subsidy')
# ]

# # Reorder the columns according to the new_order
# filtered_df = filtered_df.loc[:, new_order]

# # Sort the DataFrame based on the entire index
# filtered_df = filtered_df.sort_index(level=['base_fuel', 'lowModerateIncome_designation'])
# filtered_df

