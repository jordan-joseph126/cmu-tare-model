#!/usr/bin/env python
# coding: utf-8

# In[52]:


# Necessary Packages
# Finish updating this later
import numpy as np
import pandas as pd
from scipy.stats import norm


# In[53]:


menu_prompt = """
Would you like to filter for a specific state's data? Please enter one of the following:
N. I'd like to analyze all of the United States.
Y. I'd like to filter data for a specific state.
"""

city_prompt = """
To accurately characterize load profile, it is recommended to select subsets of data with >= 1000 models (~240,000 representative dwelling units).

The following cities (number of models also shown) are available for this state:
"""

city_menu_prompt = """
Would you like to filter a subset of city-level data? Please enter one of the following:
N. I'd like to analyze all of my selected state.
Y. I'd like to filter by city in the state.
"""

def get_menu_choice(prompt, choices):
    while True:
        choice = input(prompt).upper()
        if choice in choices:
            return choice
        print("Invalid option. Please try again.")

def get_state_choice(df_copy):
    while True:
        input_state = input("Which state would you like to analyze data for? Please enter the two-letter abbreviation: ").upper()
        if df_copy['in.state'].eq(input_state).any():
            return input_state
        print("Invalid state abbreviation. Please try again.")

def get_city_choice(df_copy, input_state):
    while True:
        input_cityFilter = input("Please enter the city name ONLY (e.g., Pittsburgh): ")
        city_filter = df_copy['in.city'].eq(f"{input_state}, {input_cityFilter}")
        if city_filter.any():
            return input_cityFilter
        print("Invalid city name. Please try again.")


# # Baseline

# In[ ]:


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
        # 'bldg_id': df_baseline['bldg_id'],
        'square_footage': df_baseline['in.sqft'],
        'census_region': df_baseline['in.census_region'],
        'census_division': df_baseline['in.census_division'],
        'census_division_recs': df_baseline['in.census_division_recs'],
        'building_america_climate_zone': df_baseline['in.building_america_climate_zone'],
        'reeds_balancing_area': df_baseline['in.reeds_balancing_area'],
        'gea_region': df_baseline['in.generation_and_emissions_assessment_region'],
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


# In[ ]:


# LAST UPDATED ON DECEMBER 5, 2024 @ 6:50 PM
# UPDATED TO RETURN BOTH DF_COPY AND DF_CONSUMPTION
# THIS FIXES THE ISSUE WITH MP_SCENARIO_DAMAGES AND PUBLIC NPV NOT BEING CALCULATED CORRECTLY
# df_consumption contains only the projected consumption data. df_copy contains all columns including the projected consumption data.
def project_future_consumption(df, hdd_factor_lookup, menu_mp):
    """
    Projects future energy consumption based on baseline or upgraded equipment specifications.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing baseline consumption data.
    hdd_factor_lookup (dict): A dictionary with Heating Degree Day (HDD) factors for different census divisions and years.
    menu_mp (int): Indicates the measure package to apply. 0 for baseline, 8/9/10 for retrofit scenarios.
    
    Returns:
    pd.DataFrame: A DataFrame with projected future energy consumption and reductions.
    """

    # Equipment lifetime specifications in years
    equipment_specs = {
        'heating': 15,
        'waterHeating': 12,
        'clothesDrying': 13,
        'cooking': 15
    }

    # Create a copy of the input DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Check if the 'census_division' column exists in the DataFrame
    if 'census_division' not in df_copy.columns:
        raise KeyError("'census_division' column is missing from the DataFrame")

    # Prepare a dictionary to hold new columns for projected consumption
    new_columns = {}

    # Baseline policy_scenario: Existing Equipment
    if menu_mp == 0:
        for category, lifetime in equipment_specs.items():
            print(f"Projecting Future Energy Consumption (Baseline Equipment): {category}")
            for year in range(1, lifetime + 1):
                year_label = 2023 + year

                # Adjust consumption based on HDD factors for heating and water heating
                if category in ['heating', 'waterHeating']:
                    hdd_factor = df_copy['census_division'].map(lambda x: hdd_factor_lookup.get(x, {}).get(year_label, hdd_factor_lookup['National'][year_label]))
                    new_columns[f'baseline_{year_label}_{category}_consumption'] = (df_copy[f'baseline_{category}_consumption'] * hdd_factor).round(2)

                else:
                    new_columns[f'baseline_{year_label}_{category}_consumption'] = df_copy[f'baseline_{category}_consumption'].round(2)

    # Retrofit policy_scenario: Upgraded Equipment (Measure Packages 8, 9, 10)
    else:
        for category, lifetime in equipment_specs.items():
            print(f"Projecting Future Energy Consumption (Upgraded Equipment): {category}")
            for year in range(1, lifetime + 1):
                year_label = 2023 + year

                # Adjust consumption based on HDD factors for heating and water heating
                if category in ['heating', 'waterHeating']:
                    hdd_factor = df_copy['census_division'].map(lambda x: hdd_factor_lookup.get(x, {}).get(year_label, hdd_factor_lookup['National'][year_label]))
                    new_columns[f'mp{menu_mp}_{year_label}_{category}_consumption'] = (df_copy[f'mp{menu_mp}_{category}_consumption'] * hdd_factor).round(2)

                    # Calculate the reduction in annual energy consumption
                    new_columns[f'mp{menu_mp}_{year_label}_{category}_reduction_consumption'] = df_copy[f'baseline_{year_label}_{category}_consumption'].sub(
                        new_columns[f'mp{menu_mp}_{year_label}_{category}_consumption'], axis=0, fill_value=0
                    ).round(2)
                else:
                    new_columns[f'mp{menu_mp}_{year_label}_{category}_consumption'] = df_copy[f'mp{menu_mp}_{category}_consumption'].round(2)

                    # Calculate the reduction in annual energy consumption
                    new_columns[f'mp{menu_mp}_{year_label}_{category}_reduction_consumption'] = df_copy[f'baseline_{year_label}_{category}_consumption'].sub(
                        new_columns[f'mp{menu_mp}_{year_label}_{category}_consumption'], axis=0, fill_value=0
                    ).round(2)

    # Calculate the new columns based on policy scenario and create dataframe based on df_copy index
    df_new_columns = pd.DataFrame(new_columns, index=df_copy.index)

    # Identify overlapping columns between the new and existing DataFrame.
    overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)

    # Drop overlapping columns from df_copy.
    if not overlapping_columns.empty:
        df_copy.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into df_copy, ensuring no duplicates or overwrites occur.
    df_copy = df_copy.join(df_new_columns, how='left')

    df_consumption = df_copy.copy()

    # Return the updated DataFrames. df_consumption contains only the projected consumption data. df_copy contains all columns including the projected consumption data.
    return df_copy, df_consumption


# In[ ]:


# def project_future_consumption(df, hdd_factor_lookup, menu_mp):
#     """
#     Projects future energy consumption based on baseline or upgraded equipment specifications.
    
#     Parameters:
#     df (pd.DataFrame): The input DataFrame containing baseline consumption data.
#     hdd_factor_lookup (dict): A dictionary with Heating Degree Day (HDD) factors for different census divisions and years.
#     menu_mp (int): Indicates the measure package to apply. 0 for baseline, 8/9/10 for retrofit scenarios.
    
#     Returns:
#     pd.DataFrame: A DataFrame with projected future energy consumption and reductions.
#     """

#     # Equipment lifetime specifications in years
#     equipment_specs = {
#         'heating': 15,
#         'waterHeating': 12,
#         'clothesDrying': 13,
#         'cooking': 15
#     }

#     # Create a copy of the input DataFrame to avoid modifying the original
#     df_copy = df.copy()

#     # Check if the 'census_division' column exists in the DataFrame
#     if 'census_division' not in df_copy.columns:
#         raise KeyError("'census_division' column is missing from the DataFrame")

#     # Prepare a dictionary to hold new columns for projected consumption
#     new_columns = {}

#     # Baseline policy_scenario: Existing Equipment
#     if menu_mp == 0:
#         for category, lifetime in equipment_specs.items():
#             print(f"Projecting Future Energy Consumption (Baseline Equipment): {category}")
#             for year in range(1, lifetime + 1):
#                 year_label = 2023 + year

#                 # Adjust consumption based on HDD factors for heating and water heating
#                 if category in ['heating', 'waterHeating']:
#                     hdd_factor = df_copy['census_division'].map(lambda x: hdd_factor_lookup.get(x, {}).get(year_label, hdd_factor_lookup['National'][year_label]))
#                     new_columns[f'baseline_{year_label}_{category}_consumption'] = (df_copy[f'baseline_{category}_consumption'] * hdd_factor).round(2)

#                 else:
#                     new_columns[f'baseline_{year_label}_{category}_consumption'] = df_copy[f'baseline_{category}_consumption'].round(2)

#     # Retrofit policy_scenario: Upgraded Equipment (Measure Packages 8, 9, 10)
#     else:
#         for category, lifetime in equipment_specs.items():
#             print(f"Projecting Future Energy Consumption (Upgraded Equipment): {category}")
#             for year in range(1, lifetime + 1):
#                 year_label = 2023 + year

#                 # Adjust consumption based on HDD factors for heating and water heating
#                 if category in ['heating', 'waterHeating']:
#                     hdd_factor = df_copy['census_division'].map(lambda x: hdd_factor_lookup.get(x, {}).get(year_label, hdd_factor_lookup['National'][year_label]))
#                     new_columns[f'mp{menu_mp}_{year_label}_{category}_consumption'] = (df_copy[f'mp{menu_mp}_{category}_consumption'] * hdd_factor).round(2)

#                     # Calculate the reduction in annual energy consumption
#                     new_columns[f'mp{menu_mp}_{year_label}_{category}_reduction_consumption'] = df_copy[f'baseline_{year_label}_{category}_consumption'].sub(
#                         new_columns[f'mp{menu_mp}_{year_label}_{category}_consumption'], axis=0, fill_value=0
#                     ).round(2)
#                 else:
#                     new_columns[f'mp{menu_mp}_{year_label}_{category}_consumption'] = df_copy[f'mp{menu_mp}_{category}_consumption'].round(2)

#                     # Calculate the reduction in annual energy consumption
#                     new_columns[f'mp{menu_mp}_{year_label}_{category}_reduction_consumption'] = df_copy[f'baseline_{year_label}_{category}_consumption'].sub(
#                         new_columns[f'mp{menu_mp}_{year_label}_{category}_consumption'], axis=0, fill_value=0
#                     ).round(2)

#     # Calculate the new columns based on policy scenario and create dataframe based on df_copy index
#     df_new_columns = pd.DataFrame(new_columns, index=df_copy.index)

#     # Identify overlapping columns between the new and existing DataFrame.
#     overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)

#     # Drop overlapping columns from df_copy.
#     if not overlapping_columns.empty:
#         df_copy.drop(columns=overlapping_columns, inplace=True)

#     # Merge new columns into df_copy, ensuring no duplicates or overwrites occur.
#     df_copy = df_copy.join(df_new_columns, how='left')

#     # Return the updated DataFrame.
#     return df_copy


# In[ ]:


# # LAST UPDATED/TESTED NOV 24, 2024 @ 5 PM
# # Calculate emissions factors for fossil fuels
# # This is before adjusting for natural gas leakage
# # Note: We use electricity marginal damages directly instead of multiplying
# # CEDM emissions factors by the EASIUR marginal damages. 
# def calculate_fossilFuel_emission_factor(fuel_type, so2_factor, nox_factor, pm25_factor, fuelConversion_factor1, fuelConversion_factor2):
#     """
#     Calculate Emissions Factors: FOSSIL FUELS
#     Fossil Fuels (Natural Gas, Fuel Oil, Propane):
#     - NOx, SO2, CO2: 
#         - RESNET Table 7.1.2 Emissions Factors for Household Combustion Fuels
#         - Source: https://www.resnet.us/wp-content/uploads/ANSIRESNETICC301-2022_resnetpblshd.pdf
#         - All factors are in units of lb/Mbtu so energy consumption in kWh need to be converted to kWh 
#         - (1 lb / Mbtu) * (1 Mbtu / 1x10^6 Btu) * (3412 Btu / 1 kWh)
#     - PM2.5: 
#         - A National Methodology and Emission Inventory for Residential Fuel Combustion
#         - Source: https://www3.epa.gov/ttnchie1/conference/ei12/area/haneke.pdf
#     """
    
#     # Create an empty dictionary called margEmis_factors to store the values
#     margEmis_factors = {}

#     # SO2, NOx, CO2: (_ lb / Mbtu) * (1 Mbtu / 1x10^6 Btu) * (3412 Btu / 1 kWh)
#     # PM2.5 - FUEL OIL: 0.83 lb/thousand gallons * (1 thousand gallons / 1000 gallons) * (1 gallon heating oil/138,500 BTU) * (3412 BTU/1 kWh)
#     # PM2.5 - NATURAL GAS: 1.9 lb/million cf * (million cf/1000000 cf) * (1 cf natural gas/1039 BTU) * (3412 BTU/1 kWh)
#     # PM2.5 - PROPANE: 0.17 lb/thousand gallons * (1 thousand gallons / 1000 gallons) * (1 gallon propane/91,452 BTU) * (3412 BTU/1 kWh)
#     margEmis_factors[f"{fuel_type}_so2"] = so2_factor * (1 / 1000000) * (3412 / 1)
#     margEmis_factors[f"{fuel_type}_nox"] = nox_factor * (1 / 1000000) * (3412 / 1)
#     margEmis_factors[f"{fuel_type}_pm25"] = pm25_factor * (1 / fuelConversion_factor1) * (1 / fuelConversion_factor2) * (3412 / 1)

#     # NATURAL GAS LEAKAGE: NATURAL GAS INFRASTRUCTURE
#     # leakage rate for natural gas infrastructure
#     # 1 Therm = 29.30 kWh --> 1.27 kg CO2e/therm * (1 therm/29.30 kWh) = 0.043 kg CO2e/kWh = 0.095 lb CO2e/kWh
#     naturalGas_leakage_mtCO2e_perkWh = 0.043 * (1/1000)

#     # CO2e include pre- and post-combustion emissions
#     margEmis_factors[f"naturalGas_co2e"] = (228.5 * (1/1000) * (1/1000)) + naturalGas_leakage_mtCO2e_perkWh
#     margEmis_factors[f"propane_co2e"]  = 275.8 * (1/1000) * (1/1000)
#     margEmis_factors[f"fuelOil_co2e"]  = 303.9 * (1/1000) * (1/1000)

#     return margEmis_factors


# In[ ]:


# LAST UPDATED DECEMBER 4, 2024
def calculate_fossil_fuel_emission_factor(fuel_type, so2_factor, nox_factor, pm25_factor, conversion_factor1, conversion_factor2):
    """
    Calculate Emission Factors for Fossil Fuels.

    Parameters:
    -----------
    fuel_type : str
        Type of fuel (e.g., "naturalGas", "fuelOil", "propane").
    so2_factor : float
        SO2 emission factor in lb/Mbtu.
    nox_factor : float
        NOx emission factor in lb/Mbtu.
    pm25_factor : float
        PM2.5 emission factor in lb per volume unit (varies by fuel).
    conversion_factor1 : int
        Conversion factor for volume units to gallons/thousand gallons.
    conversion_factor2 : int
        Conversion factor for energy content (e.g., BTU per gallon/cf).
    
    Returns:
    --------
    dict
        Dictionary containing emission factors for the given fuel type in lb/kWh or mt/kWh.
    """

    # Correct conversion factor from Mbtu to kWh
    # 1 Mbtu = 1,000,000 Btu
    # 1 kWh = 3,412 Btu
    # So, 1 Mbtu = 1,000,000 / 3,412 kWh
    mbtu_to_kwh = 1_000_000 / 3412  # Approximately 293.07107 kWh/Mbtu

    # Emission factors in lb/kWh
    emission_factors = {
        f"{fuel_type}_so2": so2_factor * (1/mbtu_to_kwh),
        f"{fuel_type}_nox": nox_factor * (1 / mbtu_to_kwh),
        f"{fuel_type}_pm25": pm25_factor * (1 / conversion_factor1) * (1 / conversion_factor2) * 3412,
    }

    # # Natural gas-specific CO2e calculation (including leakage)
    # leakage rate for natural gas infrastructure
    # 1 Therm = 29.30 kWh --> 1.27 kg CO2e/therm * (1 therm/29.30 kWh) = 0.043 kg CO2e/kWh = 0.095 lb CO2e/kWh
    naturalGas_leakage_mtCO2e_perkWh = 0.043 * (1 / 1000)

    if fuel_type == "naturalGas":
        # Convert units from kg/MWh to ton/MWh to ton/kWh
        emission_factors[f"{fuel_type}_co2e"] = (228.5 * (1 / 1000) * (1 / 1000)) + naturalGas_leakage_mtCO2e_perkWh

    # CO2e for propane and fuel oil
    # Convert units from kg/MWh to ton/MWh to ton/kWh
    elif fuel_type == "propane":
        emission_factors[f"{fuel_type}_co2e"] = 275.8 * (1 / 1000) * (1 / 1000)
    elif fuel_type == "fuelOil":
        emission_factors[f"{fuel_type}_co2e"] = 303.9 * (1 / 1000) * (1 / 1000)

    return emission_factors


# In[ ]:


# LAST UPDATED/TESTED NOV 24, 2024 @ 5 PM
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

def calculate_electricity_co2e_cambium(df_cambium_import):
    """
    Interpolates Cambium electricity emission factors and converts units.

    This function takes a dataframe containing Cambium electricity emission factors and performs the following:
    - Interpolates the Long Run Marginal Emissions Rates (LRMER) and Short Run Marginal Emissions Rates (SRMER)
      values for each scenario and GEA region on an annual basis.
    - Converts the LRMER and SRMER values from kg per MWh to tons per MWh and tons per kWh.

    Parameters
    ----------
    df_cambium_import : pandas.DataFrame
        DataFrame containing Cambium electricity emission factors with the following columns:
        - 'scenario': Scenario name or identifier.
        - 'gea_region': GEA region identifier.
        - 'year': Year of the data.
        - 'lrmer_co2e_kg_per_MWh': Long Run Marginal Emissions Rate in kg CO2e per MWh.
        - 'srmer_co2e_kg_per_MWh': Short Run Marginal Emissions Rate in kg CO2e per MWh.

    Returns
    -------
    df_cambium_import_copy : pandas.DataFrame
        DataFrame with interpolated LRMER and SRMER values for each year and additional columns for emission factors
        converted to tons per MWh and tons per kWh.

    Notes
    -----
    - The interpolation is performed linearly between the available years for each unique combination of scenario and GEA region.
    - The converted emission factors are added as new columns:
        - 'lrmer_co2e_ton_per_MWh'
        - 'lrmer_co2e_ton_per_kWh'
        - 'srmer_co2e_ton_per_MWh'
        - 'srmer_co2e_ton_per_kWh'
    - The conversion from kg to tons is done by dividing by 1,000 (1 ton = 1,000 kg).
    - The conversion from MWh to kWh is done by dividing by 1,000 (1 MWh = 1,000 kWh).

    """
    # Create a copy of the dataframe
    df_cambium_import_copy = df_cambium_import.copy()

    # Create a new DataFrame to store interpolated results
    interpolated_data = []

    # Group by 'scenario', 'state', and 'gea_region'
    grouped = df_cambium_import_copy.groupby(['scenario', 'gea_region'])

    for (scenario, gea_region), group in grouped:
        years = group['year'].values

        # Interpolate for LRMER (Long Run Marginal Emissions Rates)
        lrmer_values = group['lrmer_co2e_kg_per_MWh'].values
        lrmer_interp_func = interp1d(years, lrmer_values, kind='linear')

        # Interpolate for SRMER (Short Run Marginal Emissions Rates)
        srmer_values = group['srmer_co2e_kg_per_MWh'].values
        srmer_interp_func = interp1d(years, srmer_values, kind='linear')

        # Generate new years in 1-year increments
        new_years = np.arange(years.min(), years.max() + 1)

        # Interpolate the LRMER and SRMER values for these new years
        new_lrmer_values = lrmer_interp_func(new_years)
        new_srmer_values = srmer_interp_func(new_years)

        # Store the results in a DataFrame
        interpolated_group = pd.DataFrame({
            'scenario': scenario,
            'gea_region': gea_region,
            'year': new_years,
            'lrmer_co2e_kg_per_MWh': new_lrmer_values,
            'srmer_co2e_kg_per_MWh': new_srmer_values
        })

        interpolated_data.append(interpolated_group)

    # Concatenate all the interpolated data into a single DataFrame
    df_cambium_import_copy = pd.concat(interpolated_data).reset_index(drop=True)

    # Convert both LRMER and SRMER values to tons per MWh and tons per kWh
    df_cambium_import_copy['lrmer_co2e_ton_per_MWh'] = df_cambium_import_copy['lrmer_co2e_kg_per_MWh'] / 1000
    df_cambium_import_copy['lrmer_co2e_ton_per_kWh'] = df_cambium_import_copy['lrmer_co2e_kg_per_MWh'] / 1_000_000

    df_cambium_import_copy['srmer_co2e_ton_per_MWh'] = df_cambium_import_copy['srmer_co2e_kg_per_MWh'] / 1000
    df_cambium_import_copy['srmer_co2e_ton_per_kWh'] = df_cambium_import_copy['srmer_co2e_kg_per_MWh'] / 1_000_000

    return df_cambium_import_copy

def create_cambium_co2e_lookup(df_cambium_processed):
    """
    Creates a nested lookup dictionary for Cambium emission factors.

    This function takes a processed dataframe containing Cambium emission factors and constructs a nested dictionary
    that allows quick lookup of LRMER and SRMER emission factors based on scenario, GEA region, and year.

    Parameters
    ----------
    df_cambium_processed : pandas.DataFrame
        DataFrame containing processed Cambium emission factors with the following columns:
        - 'scenario': Scenario name or identifier.
        - 'gea_region': GEA region identifier.
        - 'year': Year of the data.
        - 'lrmer_co2e_ton_per_kWh': Long Run Marginal Emissions Rate in tons CO2e per kWh.
        - 'srmer_co2e_ton_per_kWh': Short Run Marginal Emissions Rate in tons CO2e per kWh.

    Returns
    -------
    emis_scenario_cambium_lookup : dict
        Nested dictionary structured as:
        {
            (scenario, gea_region): {
                year: {
                    'lrmer_co2e': lrmer_value,
                    'srmer_co2e': srmer_value
                },
                ...
            },
            ...
        }

    Notes
    -----
    - The outer keys of the dictionary are tuples containing (scenario, gea_region).
    - The inner dictionary maps years to a dictionary containing both LRMER and SRMER values.
    - This structure allows efficient retrieval of emission factors based on scenario, location, and year.

    """

    # Create a copy of the dataframe
    df_cambium_processed_copy = df_cambium_processed.copy()

    # Create the nested lookup dictionary for both LRMER and SRMER in tons CO2e per kWh
    emis_scenario_cambium_lookup = {}

    # Populate the dictionary
    for _, row in df_cambium_processed_copy.iterrows():
        outer_key = (row['scenario'], row['gea_region'])
        year = row['year']

        # Extract both LRMER and SRMER values in tons per kWh
        lrmer_value = row['lrmer_co2e_ton_per_kWh']
        srmer_value = row['srmer_co2e_ton_per_kWh']

        # Initialize the outer key if not already present
        if outer_key not in emis_scenario_cambium_lookup:
            emis_scenario_cambium_lookup[outer_key] = {}

        # Assign both LRMER and SRMER values in the inner dictionary for each year
        emis_scenario_cambium_lookup[outer_key][year] = {
            'lrmer_ton_per_kWh_co2e': lrmer_value,
            'srmer_ton_per_kWh_co2e': srmer_value
        }

    return emis_scenario_cambium_lookup


# In[ ]:


# LAST UPDATED/TESTED NOV 24, 2024 @ 5 PM
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def calculate_coal_projection_factors(df_cambium):
    """
    Interpolates coal_MWh and calculates coal projection factors for each region from 2018 to 2050,
    using 2018 coal generation as the reference point.

    Parameters
    ----------
    df_cambium : pandas.DataFrame
        DataFrame containing Cambium coal generation data with the following columns:
        - 'scenario': Scenario name or identifier.
        - 'gea_region': GEA region identifier.
        - 'year': Year of the data.
        - 'coal_MWh': Coal generation in MWh.

    Returns
    -------
    df_coal_factors : pandas.DataFrame
        DataFrame with interpolated coal_MWh values and a new column 'coal_projection_factors'.
    """
    # Create a copy of the dataframe
    df_cambium_copy = df_cambium.copy()

    # Create a new DataFrame to store interpolated results
    interpolated_data = []

    # Group by 'scenario' and 'gea_region'
    grouped = df_cambium_copy.groupby(['scenario', 'gea_region'])

    for (scenario, gea_region), group in grouped:
        # Extract existing years and coal_MWh values
        years = group['year'].values
        coal_MWh_values = group['coal_MWh'].values

        # Create interpolation function, allowing extrapolation
        coal_MWh_interp_func = interp1d(years, coal_MWh_values, kind='linear', bounds_error=False, fill_value="extrapolate")

        # Generate years from 2018 to 2050
        all_years = np.arange(2018, 2051)

        # Interpolate the coal_MWh values for these years
        interpolated_values = coal_MWh_interp_func(all_years)

        # Store the results in a DataFrame
        interpolated_group = pd.DataFrame({
            'scenario': scenario,
            'gea_region': gea_region,
            'year': all_years,
            'coal_MWh': interpolated_values
        })

        interpolated_data.append(interpolated_group)

    # Concatenate all the interpolated data into a single DataFrame
    df_interpolated = pd.concat(interpolated_data).reset_index(drop=True)

    # Get the coal_MWh value in 2018 for each scenario and gea_region
    coal_MWh_2018 = df_interpolated[df_interpolated['year'] == 2018][['scenario', 'gea_region', 'coal_MWh']]
    coal_MWh_2018 = coal_MWh_2018.set_index(['scenario', 'gea_region'])['coal_MWh']

    # Map the 2018 coal_MWh values to the DataFrame
    df_interpolated['coal_MWh_2018'] = df_interpolated.set_index(['scenario', 'gea_region']).index.map(coal_MWh_2018)

    # Avoid division by zero by replacing zero coal_MWh_2018 with NaN
    df_interpolated['coal_MWh_2018'] = df_interpolated['coal_MWh_2018'].replace(0, np.nan)

    # Conditions for regions other than CAMX
    condition_regions = (df_interpolated['gea_region'] != 'CAMX')

    # Calculate coal projection factors for regions other than CAMX
    df_interpolated.loc[condition_regions, 'coal_projection_factors'] = (
        df_interpolated.loc[condition_regions, 'coal_MWh'] / df_interpolated.loc[condition_regions, 'coal_MWh_2018']
    )

    # For CAMX region, assign coal_projection_factors as 1
    condition_CAMX = (df_interpolated['gea_region'] == 'CAMX')
    df_interpolated.loc[condition_CAMX, 'coal_projection_factors'] = 1

    # Replace any NaN or infinite values resulting from division by zero with 0
    df_interpolated['coal_projection_factors'] = df_interpolated['coal_projection_factors'].replace([np.inf, -np.inf, np.nan], 0)

    # Drop temporary columns
    df_interpolated.drop(columns=['coal_MWh_2018'], inplace=True)

    return df_interpolated


# In[ ]:


# UPDATED DEC 9, 2024 @ 3:00 PM
# Constants (Assuming these are defined elsewhere in your code)
TD_LOSSES = 0.06
TD_LOSSES_MULTIPLIER = 1 / (1 - TD_LOSSES)
EQUIPMENT_SPECS = {'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15}
POLLUTANTS = ['so2', 'nox', 'pm25', 'co2e']
# EPA_SCC_USD2023_PER_TON = 190 * cpi_ratio_2023_2020 # For co2e adjust SCC

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
    scenario_prefix, cambium_scenario, emis_fossil_fuel_lookup, emis_climate_electricity_lookup, damages_health_electricity_lookup = define_scenario_settings(menu_mp, policy_scenario)
    
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
        emis_climate_electricity_lookup=emis_climate_electricity_lookup,
        cambium_scenario=cambium_scenario,
        scenario_prefix=scenario_prefix,
        hdd_factors_df=hdd_factors_per_year,
        emis_fossil_fuel_lookup=emis_fossil_fuel_lookup,
        damages_health_electricity_lookup=damages_health_electricity_lookup,
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
        Tuple: (scenario_prefix, cambium_scenario, emis_fossil_fuel_lookup, emis_climate_electricity_lookup, damages_health_electricity_lookup)
    """
        
    if menu_mp == 0:
        print(f"""-- Scenario: Baseline -- 
              scenario_prefix: 'baseline_', cambium_scenario: 'MidCase', emis_fossil_fuel_lookup: 'emis_fossil_fuel_lookup', 
              emis_climate_electricity_lookup: 'emis_preIRA_co2e_cambium21_lookup', damages_health_electricity_lookup: 'damages_preIRA_health_damages_lookup'
              """)
        return "baseline_", "MidCase", emis_fossil_fuel_lookup, emis_preIRA_co2e_cambium21_lookup, damages_preIRA_health_damages_lookup

    if policy_scenario == 'No Inflation Reduction Act':
        print(f"""-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp{menu_mp}_', cambium_scenario: 'MidCase', emis_fossil_fuel_lookup: 'emis_fossil_fuel_lookup', 
              emis_climate_electricity_lookup: 'emis_preIRA_co2e_cambium21_lookup', damages_health_electricity_lookup: 'damages_preIRA_health_damages_lookup'
              """)
        return f"preIRA_mp{menu_mp}_", "MidCase", emis_fossil_fuel_lookup, emis_preIRA_co2e_cambium21_lookup, damages_preIRA_health_damages_lookup

    if policy_scenario == 'AEO2023 Reference Case':
        print(f"""-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp{menu_mp}_', cambium_scenario: 'MidCase', emis_fossil_fuel_lookup: 'emis_fossil_fuel_lookup', 
              emis_climate_electricity_lookup: 'emis_IRA_co2e_cambium22_lookup', damages_health_electricity_lookup: 'damages_iraRef_health_damages_lookup'
              """)
        return f"iraRef_mp{menu_mp}_", "MidCase", emis_fossil_fuel_lookup, emis_IRA_co2e_cambium22_lookup, damages_iraRef_health_damages_lookup

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
            lambda x: hdd_factor_lookup.get(x, hdd_factor_lookup['National']).get(year_label, 1.0)
        )

    return hdd_factors_df

def calculate_damages_grid_scenario(df_copy, df_baseline_damages_copy, df_detailed_damages, menu_mp, td_losses_multiplier, emis_climate_electricity_lookup, cambium_scenario, scenario_prefix, 
                                    hdd_factors_df, emis_fossil_fuel_lookup, damages_health_electricity_lookup, EPA_SCC_USD2023_PER_TON, equipment_specs):
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
                df_copy, category, adjusted_hdd_factor, emis_fossil_fuel_lookup, menu_mp
            )

            # Calculate climate data (annual)
            climate_results, annual_climate_emissions, annual_climate_damages = calculate_climate_emissions_and_damages(
                df=df_copy, category=category, year_label=year_label, adjusted_hdd_factor=adjusted_hdd_factor, td_losses_multiplier=td_losses_multiplier,
                emis_climate_electricity_lookup=emis_climate_electricity_lookup, cambium_scenario=cambium_scenario, EPA_SCC_USD2023_PER_TON=EPA_SCC_USD2023_PER_TON,
                total_fossil_emissions=total_fossil_emissions, scenario_prefix=scenario_prefix, menu_mp=menu_mp
            )

            # Calculate health data (annual)
            health_results, annual_health_damages = calculate_health_damages(
                df=df_copy, category=category, year_label=year_label, adjusted_hdd_factor=adjusted_hdd_factor, td_losses_multiplier=td_losses_multiplier,
                damages_health_electricity_lookup=damages_health_electricity_lookup, cambium_scenario=cambium_scenario, 
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

def calculate_fossil_fuel_emissions(df, category, adjusted_hdd_factor, emis_fossil_fuel_lookup, menu_mp):
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
                emis_factor = emis_fossil_fuel_lookup.get((fuel, pollutant), 0)
                total_fossil_emissions[pollutant] += fuel_consumption * emis_factor

    return total_fossil_emissions

def calculate_climate_emissions_and_damages(df, category, year_label, adjusted_hdd_factor, td_losses_multiplier, emis_climate_electricity_lookup, cambium_scenario, 
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
        return emis_climate_electricity_lookup.get(
            (cambium_scenario, region), {}
        ).get(year_label, {}).get('lrmer_ton_per_kWh_co2e', 0)

    def get_emission_factor_srmer(region):
        return emis_climate_electricity_lookup.get(
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


def calculate_health_damages(df, category, year_label, adjusted_hdd_factor, td_losses_multiplier, damages_health_electricity_lookup, cambium_scenario, 
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
                return damages_health_electricity_lookup.get(
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


# In[ ]:





# In[ ]:


# # UPDATED DEC 5, 2024 @ 3:15PM TO ADDRESS THE ISSUE WITH THE SUMMARY DATA BEING STORED IN THE WRONG DATAFRAME
# # ALSO FIXED THE DATA FRAGMENTATION AND PERFORMANCE WARNINGS
# # Constants (Assuming these are defined elsewhere in your code)
# TD_LOSSES = 0.06
# TD_LOSSES_MULTIPLIER = 1 / (1 - TD_LOSSES)
# EQUIPMENT_SPECS = {'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15}
# POLLUTANTS = ['so2', 'nox', 'pm25', 'co2e']
# # EPA_SCC_USD2023_PER_TON = 190 * cpi_ratio_2023_2020 # For co2e adjust SCC

# def calculate_marginal_damages(df, menu_mp, policy_scenario, df_detailed_damages=None):
#     """
#     Calculate marginal damages of pollutants based on equipment usage, emissions, and policy scenarios.
    
#     Parameters:
#         df (DataFrame): Input data with emissions and consumption data.
#         menu_mp (int): Measure package identifier.
#         policy_scenario (str): Specifies the policy scenario ('No Inflation Reduction Act' or 'AEO2023 Reference Case').
#         df_detailed_damages (DataFrame): Summary DataFrame to store aggregated results.
    
#     Returns:
#         Tuple[DataFrame, DataFrame]: Updated df with lifetime damages and df_detailed_damages with detailed annual data.
#     """

#     df_copy = df.copy()
#     if df_detailed_damages is None:
#         df_detailed_damages = pd.DataFrame(index=df_copy.index)

#     # Define policy-specific settings
#     scenario_prefix, cambium_scenario, emis_fossil_fuel_lookup, emis_climate_electricity_lookup, damages_health_electricity_lookup = define_scenario_settings(menu_mp, policy_scenario)

#     # Precompute HDD adjustment factors by region and year
#     hdd_factors_per_year = precompute_hdd_factors(df_copy)

#     # Note the two-variable unpacking here:
#     df_new_columns, df_detailed_damages = calculate_damages_grid_scenario(
#         df_copy, df_detailed_damages, menu_mp, TD_LOSSES_MULTIPLIER, emis_climate_electricity_lookup, cambium_scenario,
#         scenario_prefix, hdd_factors_per_year, emis_fossil_fuel_lookup, damages_health_electricity_lookup, EPA_SCC_USD2023_PER_TON
#     )

#     # Handle overlapping columns
#     overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)
#     if not overlapping_columns.empty:
#         df_copy.drop(columns=overlapping_columns, inplace=True)

#     # Merge newly calculated lifetime damages into df
#     df_copy = df_copy.join(df_new_columns, how='left')
#     return df_copy, df_detailed_damages

# def define_scenario_settings(menu_mp, policy_scenario):
#     """
#     Define scenario-specific settings based on menu and policy inputs.

#     Parameters:
#         menu_mp (int): Measure package identifier.
#         policy_scenario (str): Policy scenario.

#     Returns:
#         Tuple: (scenario_prefix, cambium_scenario, emis_fossil_fuel_lookup, emis_climate_electricity_lookup, damages_health_electricity_lookup)
#     """
        
#     if menu_mp == 0:
#         print(f"""-- Scenario: Baseline -- 
#               scenario_prefix: 'baseline_', cambium_scenario: 'MidCase', emis_fossil_fuel_lookup: 'emis_fossil_fuel_lookup', 
#               emis_climate_electricity_lookup: 'emis_preIRA_co2e_cambium21_lookup', damages_health_electricity_lookup: 'damages_preIRA_health_damages_lookup'
#               """)
#         return "baseline_", "MidCase", emis_fossil_fuel_lookup, emis_preIRA_co2e_cambium21_lookup, damages_preIRA_health_damages_lookup

#     if policy_scenario == 'No Inflation Reduction Act':
#         print(f"""-- Scenario: No Inflation Reduction Act -- 
#               scenario_prefix: f'preIRA_mp{menu_mp}_', cambium_scenario: 'MidCase', emis_fossil_fuel_lookup: 'emis_fossil_fuel_lookup', 
#               emis_climate_electricity_lookup: 'emis_preIRA_co2e_cambium21_lookup', damages_health_electricity_lookup: 'damages_preIRA_health_damages_lookup'
#               """)
#         return f"preIRA_mp{menu_mp}_", "MidCase", emis_fossil_fuel_lookup, emis_preIRA_co2e_cambium21_lookup, damages_preIRA_health_damages_lookup

#     if policy_scenario == 'AEO2023 Reference Case':
#         print(f"""-- Scenario: Inflation Reduction Act (IRA) Reference -- 
#               scenario_prefix: 'iraRef_mp{menu_mp}_', cambium_scenario: 'MidCase', emis_fossil_fuel_lookup: 'emis_fossil_fuel_lookup', 
#               emis_climate_electricity_lookup: 'emis_IRA_co2e_cambium22_lookup', damages_health_electricity_lookup: 'damages_iraRef_health_damages_lookup'
#               """)
#         return f"iraRef_mp{menu_mp}_", "MidCase", emis_fossil_fuel_lookup, emis_IRA_co2e_cambium22_lookup, damages_iraRef_health_damages_lookup

#     raise ValueError("Invalid Policy Scenario! Choose 'No Inflation Reduction Act' or 'AEO2023 Reference Case'.")
#     # Return the appropriate variables (assuming these lookups are defined elsewhere)

# def precompute_hdd_factors(df):
#     """
#     Precompute heating degree day (HDD) factors for each region and year.

#     Parameters:
#         df (DataFrame): Input data.

#     Returns:
#         dict: HDD factors mapped by year and region.
#     """
    
#     max_lifetime = max(EQUIPMENT_SPECS.values())
#     years = range(2024, 2024 + max_lifetime + 1)
#     hdd_factors_df = pd.DataFrame(index=df.index, columns=years)

#     for year_label in years:
#         hdd_factors_df[year_label] = df['census_division'].map(
#             lambda x: hdd_factor_lookup.get(x, hdd_factor_lookup['National']).get(year_label, 1.0)
#         )

#     return hdd_factors_df

# def calculate_damages_grid_scenario(df_copy, df_detailed_damages, menu_mp, td_losses_multiplier, emis_climate_electricity_lookup, cambium_scenario, scenario_prefix, 
#                                     hdd_factors_df, emis_fossil_fuel_lookup, damages_health_electricity_lookup, EPA_SCC_USD2023_PER_TON):
#     """
#     Calculate damages for the specified electricity grid scenario using helper functions.

#     This version avoids repeated DataFrame insertions by collecting annual and lifetime results in dictionaries,
#     then concatenating them to df_detailed_damages in bulk at the end of each iteration.
#     """

#     new_columns_data = {}  # Will hold lifetime aggregated results
#     # We'll also store annual results in dictionaries and concat once per year iteration
#     # and once after finishing each category for lifetime results.

#     for category, lifetime in EQUIPMENT_SPECS.items():
#         print(f"Calculating marginal emissions and marginal damages for {category}")

#         # Initialize lifetime accumulators
#         lifetime_climate_emissions = {'lrmer': pd.Series(0.0, index=df_copy.index),
#                                       'srmer': pd.Series(0.0, index=df_copy.index)}
#         lifetime_climate_damages = {'lrmer': pd.Series(0.0, index=df_copy.index),
#                                     'srmer': pd.Series(0.0, index=df_copy.index)}
#         lifetime_health_damages = pd.Series(0.0, index=df_copy.index)

#         for year in range(1, lifetime + 1):
#             year_label = year + 2023

#             # Get HDD factors for the year
#             hdd_factor = hdd_factors_df.get(year_label, pd.Series(1.0, index=df_copy.index))
#             # Adjust HDD factors based on the category
#             adjusted_hdd_factor = hdd_factor if category in ['heating', 'waterHeating'] else pd.Series(1.0, index=df_copy.index)

#             # Calculate fossil fuel emissions
#             total_fossil_emissions = calculate_fossil_fuel_emissions(
#                 df_copy, category, adjusted_hdd_factor, emis_fossil_fuel_lookup, menu_mp
#             )

#             # Calculate climate data (annual)
#             climate_results, annual_climate_emissions, annual_climate_damages = calculate_climate_emissions_and_damages(
#                 df=df_copy, category=category, year_label=year_label, adjusted_hdd_factor=adjusted_hdd_factor, td_losses_multiplier=td_losses_multiplier,
#                 emis_climate_electricity_lookup=emis_climate_electricity_lookup, cambium_scenario=cambium_scenario, EPA_SCC_USD2023_PER_TON=EPA_SCC_USD2023_PER_TON,
#                 total_fossil_emissions=total_fossil_emissions, scenario_prefix=scenario_prefix, menu_mp=menu_mp
#             )

#             # Calculate health data (annual)
#             health_results, annual_health_damages = calculate_health_damages(
#                 df=df_copy, category=category, year_label=year_label, adjusted_hdd_factor=adjusted_hdd_factor, td_losses_multiplier=td_losses_multiplier,
#                 damages_health_electricity_lookup=damages_health_electricity_lookup, cambium_scenario=cambium_scenario, 
#                 total_fossil_emissions=total_fossil_emissions, scenario_prefix=scenario_prefix, POLLUTANTS=POLLUTANTS, menu_mp=menu_mp
#             )

#             # Update lifetime accumulators
#             for mer_type in ['lrmer', 'srmer']:
#                 lifetime_climate_emissions[mer_type] += annual_climate_emissions[mer_type]
#                 lifetime_climate_damages[mer_type] += annual_climate_damages[mer_type]

#             lifetime_health_damages += annual_health_damages

#             # Now concatenate annual results once for this year
#             # Instead of assigning columns to df_detailed_damages one by one, we build a single DataFrame and concat
#             annual_data_all = {**climate_results, **health_results}
#             if annual_data_all:
#                 df_detailed_damages = pd.concat([df_detailed_damages, pd.DataFrame(annual_data_all, index=df_copy.index)], axis=1)

#         # After computing all years for this category, store lifetime values
#         # We also add them to df_detailed_damages once.
#         lifetime_dict = {}
#         for mer_type in ['lrmer', 'srmer']:
#             lifetime_emissions_col = f'{scenario_prefix}{category}_lifetime_tons_co2e_{mer_type}'
#             lifetime_damages_col = f'{scenario_prefix}{category}_lifetime_damages_climate_{mer_type}'
#             lifetime_dict[lifetime_emissions_col] = lifetime_climate_emissions[mer_type].round(2)
#             lifetime_dict[lifetime_damages_col] = lifetime_climate_damages[mer_type].round(2)

#             # Also store in new_columns_data for df_copy
#             new_columns_data[lifetime_emissions_col] = lifetime_climate_emissions[mer_type].round(2)
#             new_columns_data[lifetime_damages_col] = lifetime_climate_damages[mer_type].round(2)

#         # Store lifetime health damages
#         lifetime_health_damages_col = f'{scenario_prefix}{category}_lifetime_damages_health'
#         lifetime_dict[lifetime_health_damages_col] = lifetime_health_damages.round(2)
#         new_columns_data[lifetime_health_damages_col] = lifetime_health_damages.round(2)

#         # Concatenate lifetime results for this category to df_detailed_damages in one go
#         df_detailed_damages = pd.concat([df_detailed_damages, pd.DataFrame(lifetime_dict, index=df_copy.index)], axis=1)

#     df_new_columns = pd.DataFrame(new_columns_data, index=df_copy.index)
    
#     # Return both the new lifetime columns and the updated df_detailed_damages
#     return df_new_columns, df_detailed_damages

# def calculate_fossil_fuel_emissions(df, category, adjusted_hdd_factor, emis_fossil_fuel_lookup, menu_mp):
#     """
#     Calculate fossil fuel emissions for a given row and category.
#     """

#     total_fossil_emissions = {pollutant: pd.Series(0.0, index=df.index) for pollutant in POLLUTANTS}

#     if menu_mp == 0:
#         fuels = ['naturalGas', 'propane']
#         if category not in ['cooking', 'clothesDrying']:
#             fuels.append('fuelOil')

#         for fuel in fuels:
#             consumption_col = f'base_{fuel}_{category}_consumption'
#             fuel_consumption = df.get(consumption_col, pd.Series(0.0, index=df.index)).fillna(0) * adjusted_hdd_factor

#             for pollutant in total_fossil_emissions.keys():
#                 emis_factor = emis_fossil_fuel_lookup.get((fuel, pollutant), 0)
#                 total_fossil_emissions[pollutant] += fuel_consumption * emis_factor

#     return total_fossil_emissions

# def calculate_climate_emissions_and_damages(df, category, year_label, adjusted_hdd_factor, td_losses_multiplier, emis_climate_electricity_lookup, cambium_scenario, 
#                                             EPA_SCC_USD2023_PER_TON, total_fossil_emissions, scenario_prefix, menu_mp):
#     """
#     Calculate climate-related emissions and damages for a given row, category, and year.
#     Returns dicts of results and series for annual emissions/damages aggregation.
#     """

#     climate_results = {}
#     annual_climate_emissions = {}
#     annual_climate_damages = {}

#     total_fossil_emissions_co2e = total_fossil_emissions['co2e']

#     # Define functions for vectorized lookup of emission factors
#     def get_emission_factor_lrmer(region):
#         return emis_climate_electricity_lookup.get(
#             (cambium_scenario, region), {}
#         ).get(year_label, {}).get('lrmer_ton_per_kWh_co2e', 0)

#     def get_emission_factor_srmer(region):
#         return emis_climate_electricity_lookup.get(
#             (cambium_scenario, region), {}
#         ).get(year_label, {}).get('srmer_ton_per_kWh_co2e', 0)

#     mer_factors = {
#         'lrmer': df['gea_region'].map(get_emission_factor_lrmer),
#         'srmer': df['gea_region'].map(get_emission_factor_srmer)
#     }

#     # Electricity consumption depends on the scenario
#     if menu_mp == 0:
#         elec_consumption = df.get(f'base_electricity_{category}_consumption', pd.Series(0.0, index=df.index)).fillna(0) * adjusted_hdd_factor
#     else:
#         consumption_col = f'mp{menu_mp}_{year_label}_{category}_consumption'
#         elec_consumption = df.get(consumption_col, pd.Series(0.0, index=df.index)).fillna(0)

#     for mer_type in ['lrmer', 'srmer']:
#         annual_emis_electricity = elec_consumption * td_losses_multiplier * mer_factors[mer_type]

#         total_annual_climate_emissions = total_fossil_emissions_co2e + annual_emis_electricity
#         total_annual_climate_damages = total_annual_climate_emissions * EPA_SCC_USD2023_PER_TON

#         # Store annual CO2e emissions and climate damages in the dictionary
#         emis_col = f'{scenario_prefix}{year_label}_{category}_tons_co2e_{mer_type}'
#         damage_col = f'{scenario_prefix}{year_label}_{category}_damages_climate_{mer_type}'
#         climate_results[emis_col] = total_annual_climate_emissions.round(2)
#         climate_results[damage_col] = total_annual_climate_damages.round(2)

#         annual_climate_emissions[mer_type] = total_annual_climate_emissions
#         annual_climate_damages[mer_type] = total_annual_climate_damages

#     return climate_results, annual_climate_emissions, annual_climate_damages


# def calculate_health_damages(df, category, year_label, adjusted_hdd_factor, td_losses_multiplier, damages_health_electricity_lookup, cambium_scenario, 
#                              scenario_prefix, POLLUTANTS, total_fossil_emissions, menu_mp):
#     """
#     Calculate health-related damages for a given row, category, and year.
#     Returns a dict of annual pollutant damage results and a series of aggregated health damages.
#     """
    
#     health_results = {}
#     annual_health_damages = pd.Series(0.0, index=df.index)

#     for pollutant in POLLUTANTS:
#         if pollutant != 'co2e':
#             # Fossil fuel damages
#             fossil_emissions = total_fossil_emissions.get(pollutant, pd.Series(0.0, index=df.index))
#             marginal_damage_col = f'marginal_damages_{pollutant}'
#             if marginal_damage_col in df.columns:
#                 marginal_damages = df[marginal_damage_col]
#             else:
#                 marginal_damages = pd.Series(0.0, index=df.index)
#             fossil_fuel_damage = fossil_emissions * marginal_damages

#             # Define a function for vectorized lookup
#             def get_electricity_damage_factor(region):
#                 pollutant_damage_key = f'{pollutant}_dollarPerkWh_adjustVSL'
#                 return damages_health_electricity_lookup.get(
#                     (cambium_scenario, region), {}
#                 ).get(year_label, {}).get(pollutant_damage_key, 0)

#             elec_damage_factor = df['gea_region'].map(get_electricity_damage_factor)

#             # Electricity consumption depends on the scenario
#             if menu_mp == 0:
#                 elec_consumption = df.get(f'base_electricity_{category}_consumption', pd.Series(0.0, index=df.index)).fillna(0) * adjusted_hdd_factor
#             else:
#                 consumption_col = f'mp{menu_mp}_{year_label}_{category}_consumption'
#                 elec_consumption = df.get(consumption_col, pd.Series(0.0, index=df.index)).fillna(0)

#             electricity_damages = elec_consumption * td_losses_multiplier * elec_damage_factor

#             total_pollutant_damage = fossil_fuel_damage + electricity_damages

#             # Store the annual health damages in the dictionary
#             damage_col = f'{scenario_prefix}{year_label}_{category}_damages_{pollutant}'
#             health_results[damage_col] = total_pollutant_damage.round(2)

#             # Accumulate annual health-related damages
#             annual_health_damages += total_pollutant_damage

#     # Calculate total health damages for the category/year
#     health_damage_col = f'{scenario_prefix}{year_label}_{category}_damages_health'
#     # Ensure required columns exist in health_results before summation
#     so2_col = f'{scenario_prefix}{year_label}_{category}_damages_so2'
#     nox_col = f'{scenario_prefix}{year_label}_{category}_damages_nox'
#     pm25_col = f'{scenario_prefix}{year_label}_{category}_damages_pm25'

#     if menu_mp == 0:
#         if all(col in health_results for col in [so2_col, nox_col, pm25_col]):
#             health_results[health_damage_col] = round(health_results[so2_col] +
#                                                       health_results[nox_col] +
#                                                       health_results[pm25_col], 2)
#         else:
#             health_results[health_damage_col] = annual_health_damages.round(2)
#     else:
#         if all(col in health_results for col in [so2_col, nox_col, pm25_col]):
#             health_results[health_damage_col] = round((health_results[so2_col] +
#                                                        health_results[nox_col] +
#                                                        health_results[pm25_col]), 2)
#         else:
#             health_results[health_damage_col] = annual_health_damages.round(2)
                
#     return health_results, annual_health_damages

# # # calculate_marginal_damages(df, menu_mp, policy_scenario)
# # df_euss_am_baseline_home, df_baseline_scenario_damages = calculate_marginal_damages(df=df_euss_am_baseline_home,
# #                                                                                     menu_mp=menu_mp,
# #                                                                                     policy_scenario='No Inflation Reduction Act',
# #                                                                                     df_detailed_damages=df_baseline_scenario_damages
# #                                                                                     )
# # # df_euss_am_baseline_home


# In[ ]:


# # UPDATED DECEMBER 4, 2024 @ 7 PM
# # Constants (Assuming these are defined elsewhere in your code)
# # TD_LOSSES = 0.06
# # TD_LOSSES_MULTIPLIER = 1 / (1 - TD_LOSSES)
# # EQUIPMENT_SPECS = {'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15}
# # POLLUTANTS = ['so2', 'nox', 'pm25', 'co2e']
# # EPA_SCC_USD2023_PER_TON = 190 * cpi_ratio_2023_2020 # For co2e adjust SCC

# def calculate_marginal_damages(df, menu_mp, policy_scenario, df_summary):
#     """
#     Calculate marginal damages of pollutants based on equipment usage, emissions, and policy scenarios.

#     Parameters:
#         df (DataFrame): Input data with emissions and consumption data.
#         menu_mp (int): Measure package identifier.
#         policy_scenario (str): Specifies the policy scenario ('No Inflation Reduction Act' or 'AEO2023 Reference Case').
#         df_summary (DataFrame): Summary DataFrame to store aggregated results.

#     Returns:
#         DataFrame: Updated DataFrame with calculated marginal emissions and damages.
#     """

#     df_copy = df.copy()

#     # Define policy-specific settings
#     scenario_prefix, cambium_scenario, emis_fossil_fuel_lookup, emis_climate_electricity_lookup, damages_health_electricity_lookup = define_scenario_settings(menu_mp, policy_scenario)

#     # Precompute HDD adjustment factors by region and year
#     hdd_factors_per_year = precompute_hdd_factors(df_copy)

#     df_new_columns = calculate_damages_grid_scenario(
#         df_copy, df_summary, menu_mp, TD_LOSSES_MULTIPLIER, emis_climate_electricity_lookup, cambium_scenario,
#         scenario_prefix, hdd_factors_per_year, emis_fossil_fuel_lookup, damages_health_electricity_lookup, EPA_SCC_USD2023_PER_TON
#     )

#     # Handle overlapping columns
#     overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)
#     if not overlapping_columns.empty:
#         df_copy.drop(columns=overlapping_columns, inplace=True)

#     # Merge newly calculated columns
#     df_copy = df_copy.join(df_new_columns, how='left')
#     return df_copy

# def define_scenario_settings(menu_mp, policy_scenario):
#     """
#     Define scenario-specific settings based on menu and policy inputs.

#     Parameters:
#         menu_mp (int): Measure package identifier.
#         policy_scenario (str): Policy scenario.

#     Returns:
#         Tuple: (scenario_prefix, cambium_scenario, emis_fossil_fuel_lookup, emis_climate_electricity_lookup, damages_health_electricity_lookup)
#     """
        
#     if menu_mp == 0:
#         print(f"""-- Scenario: Baseline -- 
#               scenario_prefix: 'baseline_', cambium_scenario: 'MidCase', emis_fossil_fuel_lookup: 'emis_fossil_fuel_lookup', 
#               emis_climate_electricity_lookup: 'emis_preIRA_co2e_cambium21_lookup', damages_health_electricity_lookup: 'damages_preIRA_health_damages_lookup'
#               """)
#         return "baseline_", "MidCase", emis_fossil_fuel_lookup, emis_preIRA_co2e_cambium21_lookup, damages_preIRA_health_damages_lookup

#     if policy_scenario == 'No Inflation Reduction Act':
#         print(f"""-- Scenario: No Inflation Reduction Act -- 
#               scenario_prefix: f'preIRA_mp{menu_mp}_', cambium_scenario: 'MidCase', emis_fossil_fuel_lookup: 'emis_fossil_fuel_lookup', 
#               emis_climate_electricity_lookup: 'emis_preIRA_co2e_cambium21_lookup', damages_health_electricity_lookup: 'damages_preIRA_health_damages_lookup'
#               """)
#         return f"preIRA_mp{menu_mp}_", "MidCase", emis_fossil_fuel_lookup, emis_preIRA_co2e_cambium21_lookup, damages_preIRA_health_damages_lookup

#     if policy_scenario == 'AEO2023 Reference Case':
#         print(f"""-- Scenario: Inflation Reduction Act (IRA) Reference -- 
#               scenario_prefix: 'iraRef_mp{menu_mp}_', cambium_scenario: 'MidCase', emis_fossil_fuel_lookup: 'emis_fossil_fuel_lookup', 
#               emis_climate_electricity_lookup: 'emis_IRA_co2e_cambium22_lookup', damages_health_electricity_lookup: 'damages_iraRef_health_damages_lookup'
#               """)
#         return f"iraRef_mp{menu_mp}_", "MidCase", emis_fossil_fuel_lookup, emis_IRA_co2e_cambium22_lookup, damages_iraRef_health_damages_lookup

#     raise ValueError("Invalid Policy Scenario! Choose 'No Inflation Reduction Act' or 'AEO2023 Reference Case'.")
#     # Return the appropriate variables (assuming these lookups are defined elsewhere)

# def precompute_hdd_factors(df):
#     """
#     Precompute heating degree day (HDD) factors for each region and year.

#     Parameters:
#         df (DataFrame): Input data.

#     Returns:
#         dict: HDD factors mapped by year and region.
#     """
    
#     max_lifetime = max(EQUIPMENT_SPECS.values())
#     years = range(2024, 2024 + max_lifetime + 1)
#     hdd_factors_df = pd.DataFrame(index=df.index, columns=years)

#     for year_label in years:
#         hdd_factors_df[year_label] = df['census_division'].map(
#             lambda x: hdd_factor_lookup.get(x, hdd_factor_lookup['National']).get(year_label, 1.0)
#         )

#     return hdd_factors_df

# def calculate_damages_grid_scenario(df_copy, df_summary, menu_mp, td_losses_multiplier, emis_climate_electricity_lookup, cambium_scenario, scenario_prefix, 
#                                     hdd_factors_df, emis_fossil_fuel_lookup, damages_health_electricity_lookup, EPA_SCC_USD2023_PER_TON):
#     """
#     Calculate damages for the specified electricity grid scenario using helper functions.

#     Parameters:
#         df_copy (DataFrame): The DataFrame containing consumption data.
#         df_summary (DataFrame): Summary DataFrame to store aggregated results (optional).
#         menu_mp (int): Measure package identifier.
#         td_losses_multiplier (float): Transmission and distribution losses multiplier.
#         emis_climate_electricity_lookup (dict): Lookup table for electricity emissions factors.
#         cambium_scenario (str): Cambium scenario identifier.
#         scenario_prefix (str): Prefix for scenario columns.
#         hdd_factors_per_year (dict): Precomputed HDD factors per year.
#         emis_fossil_fuel_lookup (dict): Lookup table for fossil fuel emissions factors.
#         damages_health_electricity_lookup (dict): Lookup table for health damages from electricity.
#         EPA_SCC_USD2023_PER_TON (float): Social cost of carbon dioxide emissions.
        
#     Returns:
#         DataFrame: DataFrame with calculated damages columns.
#     """

#     new_columns_data = {}

#     for category, lifetime in EQUIPMENT_SPECS.items():
#         print(f"Calculating marginal emissions and marginal damages for {category}")

#         # Initialize lifetime accumulators
#         lifetime_climate_emissions = {'lrmer': pd.Series(0.0, index=df_copy.index),
#                                       'srmer': pd.Series(0.0, index=df_copy.index)}
#         lifetime_climate_damages = {'lrmer': pd.Series(0.0, index=df_copy.index),
#                                     'srmer': pd.Series(0.0, index=df_copy.index)}
#         lifetime_health_damages = pd.Series(0.0, index=df_copy.index)

#         for year in range(1, lifetime + 1):
#             year_label = year + 2023

#             # Get HDD factors for the year
#             hdd_factor = hdd_factors_df.get(year_label, pd.Series(1.0, index=df_copy.index))

#             # Adjust HDD factors based on the category
#             if category in ['heating', 'waterHeating']:
#                 adjusted_hdd_factor = hdd_factor
#             else:
#                 adjusted_hdd_factor = pd.Series(1.0, index=df_copy.index)

#             # Calculate fossil fuel emissions
#             total_fossil_emissions = calculate_fossil_fuel_emissions(
#                 df_copy, category, adjusted_hdd_factor, emis_fossil_fuel_lookup, menu_mp
#             )

#             climate_results, annual_climate_emissions, annual_climate_damages = calculate_climate_emissions_and_damages(
#                 df=df_copy, category=category, year_label=year_label, adjusted_hdd_factor=adjusted_hdd_factor, td_losses_multiplier=td_losses_multiplier,
#                 emis_climate_electricity_lookup=emis_climate_electricity_lookup, cambium_scenario=cambium_scenario, EPA_SCC_USD2023_PER_TON=EPA_SCC_USD2023_PER_TON,
#                 total_fossil_emissions=total_fossil_emissions, scenario_prefix=scenario_prefix, menu_mp=menu_mp
#                 )

#             # Update lifetime accumulators
#             for mer_type in ['lrmer', 'srmer']:
#                 lifetime_climate_emissions[mer_type] += annual_climate_emissions[mer_type]
#                 lifetime_climate_damages[mer_type] += annual_climate_damages[mer_type]

#             # Add climate results to new_columns_data
#             new_columns_data.update(climate_results)

#             health_results, annual_health_damages = calculate_health_damages(
#                 df=df_copy, category=category, year_label=year_label, adjusted_hdd_factor=adjusted_hdd_factor, td_losses_multiplier=td_losses_multiplier,
#                 damages_health_electricity_lookup=damages_health_electricity_lookup, cambium_scenario=cambium_scenario, 
#                 total_fossil_emissions=total_fossil_emissions, scenario_prefix=scenario_prefix, POLLUTANTS=POLLUTANTS, menu_mp=menu_mp
#                 )

#             # Update lifetime health damages
#             lifetime_health_damages += annual_health_damages

#             # Add health results to new_columns_data
#             new_columns_data.update(health_results)

#         # Store the lifetime values in new_columns_data and df_summary
#         for mer_type in ['lrmer', 'srmer']:
#             lifetime_emissions_col = f'{scenario_prefix}{category}_lifetime_tons_co2e_{mer_type}'
#             lifetime_damages_col = f'{scenario_prefix}{category}_lifetime_damages_climate_{mer_type}'
#             new_columns_data[lifetime_emissions_col] = lifetime_climate_emissions[mer_type].round(2)
#             df_summary[lifetime_emissions_col] = lifetime_climate_emissions[mer_type].round(2)

#             new_columns_data[lifetime_damages_col] = lifetime_climate_damages[mer_type].round(2)
#             df_summary[lifetime_damages_col] = lifetime_climate_damages[mer_type].round(2)

#         # Store lifetime health damages
#         lifetime_health_damages_col = f'{scenario_prefix}{category}_lifetime_damages_health'
#         new_columns_data[lifetime_health_damages_col] = lifetime_health_damages.round(2)
#         df_summary[lifetime_health_damages_col] = lifetime_health_damages.round(2)

#     df_new_columns = pd.DataFrame(new_columns_data, index=df_copy.index)
#     return df_new_columns

# def calculate_fossil_fuel_emissions(df, category, adjusted_hdd_factor, emis_fossil_fuel_lookup, menu_mp):
#     """
#     Calculate fossil fuel emissions for a given row and category.
#     """

#     total_fossil_emissions = {pollutant: pd.Series(0.0, index=df.index) for pollutant in POLLUTANTS}

#     if menu_mp == 0:
#         fuels = ['naturalGas', 'propane']
#         if category not in ['cooking', 'clothesDrying']:
#             fuels.append('fuelOil')

#         for fuel in fuels:
#             consumption_col = f'base_{fuel}_{category}_consumption'
#             fuel_consumption = df.get(consumption_col, pd.Series(0.0, index=df.index)).fillna(0) * adjusted_hdd_factor

#             for pollutant in total_fossil_emissions.keys():
#                 emis_factor = emis_fossil_fuel_lookup.get((fuel, pollutant), 0)
#                 total_fossil_emissions[pollutant] += fuel_consumption * emis_factor

#     return total_fossil_emissions

# def calculate_climate_emissions_and_damages(df, category, year_label, adjusted_hdd_factor, td_losses_multiplier, emis_climate_electricity_lookup, cambium_scenario, 
#                                             EPA_SCC_USD2023_PER_TON, total_fossil_emissions, scenario_prefix, menu_mp):
#     """
#     Calculate climate-related emissions and damages for a given row, category, and year.
#     """

#     climate_results = {}
#     annual_climate_emissions = {}
#     annual_climate_damages = {}

#     total_fossil_emissions_co2e = total_fossil_emissions['co2e']

#     # Define functions for vectorized lookup of emission factors
#     def get_emission_factor_lrmer(region):
#         return emis_climate_electricity_lookup.get(
#             (cambium_scenario, region), {}
#         ).get(year_label, {}).get('lrmer_ton_per_kWh_co2e', 0)

#     def get_emission_factor_srmer(region):
#         return emis_climate_electricity_lookup.get(
#             (cambium_scenario, region), {}
#         ).get(year_label, {}).get('srmer_ton_per_kWh_co2e', 0)

#     mer_factors = {
#         'lrmer': df['gea_region'].map(get_emission_factor_lrmer),
#         'srmer': df['gea_region'].map(get_emission_factor_srmer)
#     }

#     # Electricity consumption depends on the scenario
#     if menu_mp == 0:
#         elec_consumption = df.get(f'base_electricity_{category}_consumption', pd.Series(0.0, index=df.index)).fillna(0) * adjusted_hdd_factor
#     else:
#         consumption_col = f'mp{menu_mp}_{year_label}_{category}_consumption'
#         elec_consumption = df.get(consumption_col, pd.Series(0.0, index=df.index)).fillna(0)

#     for mer_type in ['lrmer', 'srmer']:
#         annual_emis_electricity = elec_consumption * td_losses_multiplier * mer_factors[mer_type]

#         total_annual_climate_emissions = total_fossil_emissions_co2e + annual_emis_electricity
#         total_annual_climate_damages = total_annual_climate_emissions * EPA_SCC_USD2023_PER_TON

#         # Store annual CO2e emissions and climate damages
#         emis_col = f'{scenario_prefix}{year_label}_{category}_tons_co2e_{mer_type}'
#         damage_col = f'{scenario_prefix}{year_label}_{category}_damages_climate_{mer_type}'
#         climate_results[emis_col] = total_annual_climate_emissions.round(2)
#         climate_results[damage_col] = total_annual_climate_damages.round(2)

#         # Accumulate lifetime CO2e emissions and damages
#         if mer_type not in annual_climate_emissions:
#             annual_climate_emissions[mer_type] = pd.Series(0.0, index=df.index)
#             annual_climate_damages[mer_type] = pd.Series(0.0, index=df.index)

#         annual_climate_emissions[mer_type] = total_annual_climate_emissions
#         annual_climate_damages[mer_type] = total_annual_climate_damages

#     return climate_results, annual_climate_emissions, annual_climate_damages

# def calculate_health_damages(df, category, year_label, adjusted_hdd_factor, td_losses_multiplier, damages_health_electricity_lookup, cambium_scenario, 
#                              scenario_prefix, POLLUTANTS, total_fossil_emissions, menu_mp):
#     """
#     Calculate health-related damages for a given row, category, and year.
#     """
    
#     health_results = {}
#     annual_health_damages = pd.Series(0.0, index=df.index)

#     for pollutant in POLLUTANTS:
#         if pollutant != 'co2e':
#             # Fossil fuel damages
#             fossil_emissions = total_fossil_emissions.get(pollutant, pd.Series(0.0, index=df.index))
#             marginal_damage_col = f'marginal_damages_{pollutant}'
#             if marginal_damage_col in df.columns:
#                 marginal_damages = df[marginal_damage_col]
#             else:
#                 marginal_damages = pd.Series(0.0, index=df.index)
#             fossil_fuel_damage = fossil_emissions * marginal_damages

#             # Define a function for vectorized lookup
#             def get_electricity_damage_factor(region):
#                 pollutant_damage_key = f'{pollutant}_dollarPerkWh_adjustVSL'
#                 return damages_health_electricity_lookup.get(
#                     (cambium_scenario, region), {}
#                 ).get(year_label, {}).get(pollutant_damage_key, 0)

#             elec_damage_factor = df['gea_region'].map(get_electricity_damage_factor)

#             # Electricity consumption depends on the scenario
#             if menu_mp == 0:
#                 elec_consumption = df.get(f'base_electricity_{category}_consumption', pd.Series(0.0, index=df.index)).fillna(0) * adjusted_hdd_factor
#             else:
#                 consumption_col = f'mp{menu_mp}_{year_label}_{category}_consumption'
#                 elec_consumption = df.get(consumption_col, pd.Series(0.0, index=df.index)).fillna(0)

#             electricity_damages = elec_consumption * td_losses_multiplier * elec_damage_factor

#             total_pollutant_damage = fossil_fuel_damage + electricity_damages

#             # Store the annual health damages
#             damage_col = f'{scenario_prefix}{year_label}_{category}_damages_{pollutant}'
#             health_results[damage_col] = total_pollutant_damage.round(2)

#             # Accumulate lifetime health-related damages
#             annual_health_damages += total_pollutant_damage

#     return health_results, annual_health_damages


# In[ ]:


# LAST UPDATED NOVEMBER 24 @ 5 PM
# HEALTH RELATED EMISSIONS VALIDATION
def process_Schmitt_emissions_data(df_grid_mix, df_grid_emis_factors):
    # Check unique fuel sources in both dataframes
    fuel_sources_mix = set(df_grid_mix['fuel_source'].unique())
    fuel_sources_emis = set(df_grid_emis_factors['fuel_source'].unique())

    print("Fuel sources in df_grid_mix:", fuel_sources_mix)
    print("Fuel sources in df_grid_emis_factors:", fuel_sources_emis)

    # Merge the dataframes
    df_combined = pd.merge(
        df_grid_mix,
        df_grid_emis_factors,
        on=['cambium_gea_region', 'fuel_source'],
        how='inner'
    )

    # Calculate emissions contribution
    df_combined['emis_contribution'] = df_combined['fraction_generation'] * df_combined['emis_rate']

    # Sum emissions contributions
    df_emis_factors = df_combined.groupby(
        ['year', 'cambium_gea_region', 'pollutant']
    )['emis_contribution'].sum().reset_index()

    # Pivot the dataframe
    df_emis_factors_pivot = df_emis_factors.pivot_table(
        index=['year', 'cambium_gea_region'],
        columns='pollutant',
        values='emis_contribution'
    ).reset_index()

    # Rename columns
    df_emis_factors_pivot.rename(columns={
        'NH3': 'delta_egrid_nh3',
        'NOx': 'delta_egrid_nox',
        'PM25': 'delta_egrid_pm25',
        'SO2': 'delta_egrid_so2',
        'VOC': 'delta_egrid_voc'
    }, inplace=True)

    return df_emis_factors_pivot


# In[50]:


import pandas as pd

# Define function to create a fuel price lookup dictionary without policy_scenario from row
def create_fuel_price_lookup(df, policy_scenario):
    lookup_dict = {}
    
    for _, row in df.iterrows():
        location = row['location_map']
        fuel_type = row['fuel_type']
        
        if location not in lookup_dict:
            lookup_dict[location] = {}
        
        if fuel_type not in lookup_dict[location]:
            lookup_dict[location][fuel_type] = {}
        
        if policy_scenario not in lookup_dict[location][fuel_type]:
            lookup_dict[location][fuel_type][policy_scenario] = {}
        
        for year in range(2022, 2051):
            column_name = f"{year}_fuelPrice_perkWh"
            lookup_dict[location][fuel_type][policy_scenario][year] = row[column_name]
    
    return lookup_dict

# Define function to project future prices with fallback to 'National'
def project_future_prices(row, factor_dict, policy_scenario):
    loc = row['census_division']
    fuel = row['fuel_type']
    price_2022 = row['2022_fuelPrice_perkWh']

    print(f"\nProcessing location: {loc}, fuel: {fuel}, policy_scenario: {policy_scenario}")
    print(f"Initial price for 2022: {price_2022}")

    # First, try to fetch the projection factors for the specific region
    projection_factors = factor_dict.get((loc, fuel, policy_scenario))
    
    # If no factors are found for the specific region, default to 'National'
    if not projection_factors:
        print(f"No projection factors found for {loc}, {fuel}, {policy_scenario}. Defaulting to 'National'.")
        projection_factors = factor_dict.get(('National', fuel, policy_scenario))
        
    if projection_factors:
        print(f"Using projection factors for {loc if projection_factors else 'National'}, {fuel}, {policy_scenario}: {projection_factors}")
    else:
        print(f"No projection factors found for 'National', {fuel}, {policy_scenario} either. Cannot project future prices.")
        return pd.Series()  # Return an empty Series if no factors are found

    future_prices = {}
    for year in range(2022, 2051):
        if projection_factors and year in projection_factors:
            factor = projection_factors[year]
            future_price = price_2022 * factor
            future_prices[f'{year}_fuelPrice_perkWh'] = future_price
            print(f"Year: {year}, Factor: {factor}, Future Price: {future_price}")
        else:
            print(f"Missing factor for year {year} in {loc if projection_factors else 'National'}, {fuel}, {policy_scenario}. Skipping this year.")
    
    return pd.Series(future_prices)


# In[51]:


# LAST UPDATED SEPTEMBER 5, 2024 @ 9:37 PM
def calculate_annual_fuelCost(df, menu_mp, policy_scenario, drop_fuel_cost_columns):
    """
    Calculate the annual fuel cost for baseline and measure packages.

    Parameters:
    df (pd.DataFrame): DataFrame containing baseline fuel consumption data.
    menu_mp (int): Measure package identifier
    policy_scenario (str): Name of EIA AEO policy_scenario used to project fuel prices

    Returns:
    pd.DataFrame: DataFrame with additional columns for annual fuel costs, savings, and changes.
    """
    df_copy = df.copy()

    # Determine the scenario prefix and fuel price lookup based on menu_mp and policy_scenario
    if menu_mp == 0:
        scenario_prefix = "baseline_"
        fuel_price_lookup = preIRA_fuel_price_lookup
    else:
        if policy_scenario == 'No Inflation Reduction Act':
            scenario_prefix = f"preIRA_mp{menu_mp}_"
            fuel_price_lookup = preIRA_fuel_price_lookup
        elif policy_scenario == 'AEO2023 Reference Case':
            scenario_prefix = f"iraRef_mp{menu_mp}_"
            fuel_price_lookup = iraRef_fuel_price_lookup
        else:
            raise ValueError("Invalid Policy policy_scenario! Please choose from 'No Inflation Reduction Act' or 'AEO2023 Reference Case'.")

    # Fuel type mapping and equipment lifetime specifications
    fuel_mapping = {'Electricity': 'electricity', 'Natural Gas': 'naturalGas', 'Fuel Oil': 'fuelOil', 'Propane': 'propane'}
    equipment_specs = {'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15}

    # Initialize a dictionary to hold new columns
    new_columns = {}

    # If baseline calculations are required
    if menu_mp == 0:
        for category in equipment_specs:
            df_copy[f'fuel_type_{category}'] = df_copy[f'base_{category}_fuel'].map(fuel_mapping)

        for category, lifetime in equipment_specs.items():
            print(f"Calculating BASELINE (no retrofit) fuel costs from 2024 to {2024 + lifetime} for {category}")
            for year in range(1, lifetime + 1):
                year_label = year + 2023

                fuel_costs = df_copy.apply(lambda row: round(
                    row[f'baseline_{year_label}_{category}_consumption'] *
                    fuel_price_lookup.get(
                        row['state'] if row[f'fuel_type_{category}'] in ['electricity', 'naturalGas'] else row['census_division'],
                        {}
                    ).get(row[f'fuel_type_{category}'], {}).get(policy_scenario, {}).get(year_label, 0), 2),
                    axis=1
                )

                new_columns[f'baseline_{year_label}_{category}_fuelCost'] = fuel_costs

    else:
        for category, lifetime in equipment_specs.items():
            print(f"Calculating POST-RETROFIT (MP{menu_mp}) fuel costs from 2024 to {2024 + lifetime} for {category}")
            for year in range(1, lifetime + 1):
                year_label = year + 2023

                fuel_costs = df_copy.apply(lambda row: round(
                    row[f'mp{menu_mp}_{year_label}_{category}_consumption'] *
                    fuel_price_lookup.get(row['state'], {}).get('electricity', {}).get(policy_scenario, {}).get(year_label, 0), 2),
                    axis=1
                )

                # Store all new columns in the dictionary first
                new_columns[f'{scenario_prefix}{year_label}_{category}_fuelCost'] = fuel_costs
                
                new_columns[f'{scenario_prefix}{year_label}_{category}_savings_fuelCost'] = (
                    df_copy[f'baseline_{year_label}_{category}_fuelCost'] - fuel_costs
                )

        # Only drop if annual fuel cost savings have already been calculated
        # Drop fuel cost columns if the flag is True
        if drop_fuel_cost_columns:
            print("Dropping Annual Fuel Costs for Baseline Scenario and Retrofit. Storing Fuel Savings for Private NPV Calculation.")
            fuel_cost_columns = [col for col in df_copy.columns if '_fuelCost' in col and '_savings_fuelCost' not in col]
            df_copy.drop(columns=fuel_cost_columns, inplace=True)

    # Calculate the new columns based on policy scenario and create dataframe based on df_copy index
    df_new_columns = pd.DataFrame(new_columns, index=df_copy.index)

    # Identify overlapping columns between the new and existing DataFrame.
    overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)

    # Drop overlapping columns from df_copy.
    if not overlapping_columns.empty:
        df_copy.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into df_copy, ensuring no duplicates or overwrites occur.
    df_copy = df_copy.join(df_new_columns, how='left')

    # Return the updated DataFrame.
    return df_copy


# # Retrofit Packages

# ## Basic Retrofit

# In[ ]:


def df_enduse_compare(df_mp, input_mp, menu_mp, df_baseline):
    # Create a new DataFrame named df_compare
    # using pd.DataFrame constructor and initialize it with columns from df_mp
    df_compare = pd.DataFrame({
        # 'bldg_id':df_mp['bldg_id'],
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
    # df_compare = pd.merge(df_baseline, df_compare, how='inner', on = 'bldg_id')
    # calculate_consumption_reduction(df_compare, category)    

    # If both df_baseline and df_compare now have bldg_id set as their index, modify it as:
    df_compare = pd.merge(df_baseline, df_compare, how='inner', left_index=True, right_index=True)
    # Make sure df_baseline and df_compare both have bldg_id as their index before doing this.

    return df_compare


# In[ ]:


# import pandas as pd
# import numpy as np

# def summarize_stats_table(df, category, data_columns, column_name_mapping, number_formatting, include_zero=True):
#     """
#     Generate a formatted summary statistics table for specified columns in a DataFrame, grouped by 'base_fuel' and 'lowModerateIncome_designation'.

#     Parameters:
#     - df (DataFrame): The input DataFrame from which to compute statistics.
#     - data_columns (list of str): The columns to include in the summary statistics.
#     - column_name_mapping (dict): A dictionary to rename the columns in the summary statistics output.
#     - number_formatting (str): The format string to use for numeric values in the output.
#     - include_zero (bool, optional): Whether to include zero values in the statistics. Defaults to True.
#       If False, zeros are replaced with NaN, which are then ignored in the computations.

#     Returns:
#     - DataFrame: A DataFrame containing the summary statistics, with formatted numeric values
#       and renamed columns according to the input specifications, grouped by 'base_fuel' and 'lowModerateIncome_designation'.
#     """

#     # Ensure 'lowModerateIncome_designation' is treated as a categorical variable with a specific order
#     income_categories = ['Low-Income', 'Moderate-Income', 'Middle-to-Upper-Income']
#     df['lowModerateIncome_designation'] = pd.Categorical(df['lowModerateIncome_designation'], categories=income_categories, ordered=True)

#     # Filter out the 'Middle-to-Upper-Income' rows if needed (similar to your earlier function)
#     df_filtered = df[df['lowModerateIncome_designation'] != 'Middle-to-Upper-Income']

#     # Replace 0 values with NaN in the selected columns if include_zero is set to False
#     if not include_zero:
#         df_filtered[data_columns] = df_filtered[data_columns].replace(0, np.nan)

#     # Group by 'base_fuel' and 'lowModerateIncome_designation' and calculate summary statistics
#     summary_stats = df_filtered.groupby(by=[f'base_{category}_fuel', 'lowModerateIncome_designation'], observed=False)[data_columns].describe().unstack()

#     # # Apply formatting to each number in these statistics according to the given format
#     # summary_stats = summary_stats.applymap(lambda x: f"{x:{number_formatting}}" if pd.notnull(x) else "")

#     # Rename the columns in the summary statistics DataFrame according to the provided mapping
#     summary_stats.rename(columns=column_name_mapping, inplace=True)

#     return summary_stats

# # Example usage of the function:
# # Assume 'df' is a DataFrame with relevant data and columns:
# df_multiIndex_summary = summarize_stats_table(df_basic_summary_heating, category='heating', data_columns=['iraRef_heating_usd2023_per_mtCO2e'], 
#                                                column_name_mapping={'iraRef_heating_usd2023_per_mtCO2e': 'CO2 Abatement Cost (USD/mtCO2e)'},
#                                                number_formatting=".2f", include_zero=True)
# df_multiIndex_summary


# In[56]:


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


# In[ ]:


# LAST UPDATED DECEMBER 5, 2024 @ 9 PM
# UPDATE THE SET OF FUNCTIONS TO CALCULATE CLIMATE, HEALTH, AND PUBLIC NPV WITH THE MER_TYPE PARAMETER
# NEXT NEED TO UPDATE THE ADOPTION DECISION FUNCTION TO INCLUDE HEALTH 
def calculate_public_npv(df, df_baseline_damages, df_mp_damages, menu_mp, policy_scenario, interest_rate=0.02):
    """
    Calculate the public Net Present Value (NPV) for specific categories of damages,
    considering different policy scenarios related to grid decarbonization.

    Parameters:
    - df (DataFrame): A pandas DataFrame containing the relevant data.
    - menu_mp (str): Menu identifier used in column names.
    - policy_scenario (str): Policy policy_scenario that determines electricity grid projections. 
                             Accepted values: 'AEO2023 Reference Case'.
    - interest_rate (float): The discount rate used in the NPV calculation. Default is 2% for Social Discount Rate.

    Returns:
    - DataFrame: The input DataFrame with additional columns containing the calculated public NPVs for each enduse.
    """
    equipment_specs = {
        'heating': 15,
        'waterHeating': 12,
        'clothesDrying': 13,
        'cooking': 15
    }
    
    df_copy = df.copy()
    df_baseline_damages_copy = df_baseline_damages.copy()
    df_mp_damages_copy = df_mp_damages.copy()

    # Calculate the lifetime damages and corresponding NPV based on the policy policy_scenario
    df_new_columns = calculate_lifetime_damages_grid_scenario(df_copy, df_baseline_damages_copy, df_mp_damages_copy, menu_mp, equipment_specs, policy_scenario, interest_rate)

    # Drop any overlapping columns from df_copy
    overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)
    if not overlapping_columns.empty:
        df_copy.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into the original DataFrame
    df_copy = df_copy.join(df_new_columns, how='left')

    return df_copy

def calculate_lifetime_damages_grid_scenario(df_copy, df_baseline_damages_copy, df_mp_damages_copy, menu_mp, equipment_specs, policy_scenario, interest_rate):
    """
    Calculate the NPV of climate, health, and public damages over the equipment's lifetime
    under different grid decarbonization scenarios.

    Parameters:
    - df_copy (DataFrame): A copy of the original DataFrame to store NPV calculations.
    - menu_mp (str): Menu identifier used in column names.
    - equipment_specs (dict): Dictionary containing lifetimes for each equipment category.
    - policy_scenario (str): Specifies the grid policy_scenario ('No Inflation Reduction Act', 'AEO2023 Reference Case').
    - interest_rate (float): Discount rate for NPV calculation.

    Returns:
    - DataFrame: A DataFrame containing the calculated NPV values for each category.
    """
    # Determine the policy_scenario prefix based on the policy policy_scenario
    if policy_scenario == 'No Inflation Reduction Act':
        scenario_prefix = f"preIRA_mp{menu_mp}_"
    elif policy_scenario == 'AEO2023 Reference Case':
        scenario_prefix = f"iraRef_mp{menu_mp}_"
    else:
        raise ValueError("Invalid Policy policy_scenario! Please choose from 'No Inflation Reduction Act' or 'AEO2023 Reference Case'.")
    
    # Create a DataFrame to hold the NPV calculations
    npv_columns = {}
    
    for category, lifetime in equipment_specs.items():
        print(f"""\nCalculating Public NPV for {category}...
            lifetime: {lifetime}, interest_rate: {interest_rate}, policy_scenario: {policy_scenario}""")
        # For LRMER and SRMER
        for mer_type in ['lrmer', 'srmer']:
            print(f"Type of Marginal Emissions Rate Factor: {mer_type}")           
            # Initialize NPV columns for each category
            climate_npv_key = f'{scenario_prefix}{category}_climate_npv_{mer_type}'
            health_npv_key = f'{scenario_prefix}{category}_health_npv'
            public_npv_key = f'{scenario_prefix}{category}_public_npv_{mer_type}'
            
            # Initialize NPV columns in the dictionary if they don't exist
            npv_columns[climate_npv_key] = npv_columns.get(climate_npv_key, 0)
            npv_columns[health_npv_key] = npv_columns.get(health_npv_key, 0)
            npv_columns[public_npv_key] = npv_columns.get(public_npv_key, 0)
                
            for year in range(1, lifetime + 1):
                year_label = year + 2023
                
                # Base Damages for Climate and Health
                base_annual_climate_damages = df_baseline_damages_copy[f'baseline_{year_label}_{category}_damages_climate_{mer_type}']
                base_annual_health_damages = df_baseline_damages_copy[f'baseline_{year_label}_{category}_damages_health']
                base_annual_damages = base_annual_climate_damages + base_annual_health_damages
                
                # Post-Retrofit Damages for Climate and Health
                retrofit_annual_climate_damages = df_mp_damages_copy[f'{scenario_prefix}{year_label}_{category}_damages_climate_{mer_type}']
                retrofit_annual_health_damages = df_mp_damages_copy[f'{scenario_prefix}{year_label}_{category}_damages_health']
                retrofit_annual_damages = retrofit_annual_climate_damages + retrofit_annual_health_damages

                # Apply the discount factor to each year's damages
                discount_factor = 1 / ((1 + interest_rate) ** year)
    
                npv_columns[climate_npv_key] += ((base_annual_climate_damages - retrofit_annual_climate_damages) * discount_factor).round(2)
                npv_columns[health_npv_key] += ((base_annual_health_damages - retrofit_annual_health_damages) * discount_factor).round(2)
                npv_columns[public_npv_key] += ((base_annual_damages - retrofit_annual_damages) * discount_factor).round(2)

    # Convert the dictionary to a DataFrame and return it
    df_npv = pd.DataFrame(npv_columns, index=df_copy.index)
    return df_npv


# In[ ]:


# # LAST UPDATED SEPTEMBER 20, 2024 @ 12:15 AM
# def calculate_public_npv(df, df_damages, menu_mp, policy_scenario, interest_rate=0.02):
#     """
#     Calculate the public Net Present Value (NPV) for specific categories of damages,
#     considering different policy scenarios related to grid decarbonization.

#     Parameters:
#     - df (DataFrame): A pandas DataFrame containing the relevant data.
#     - menu_mp (str): Menu identifier used in column names.
#     - policy_scenario (str): Policy policy_scenario that determines electricity grid projections. 
#                              Accepted values: 'AEO2023 Reference Case'.
#     - interest_rate (float): The discount rate used in the NPV calculation. Default is 2% for Social Discount Rate.

#     Returns:
#     - DataFrame: The input DataFrame with additional columns containing the calculated public NPVs for each enduse.
#     """
#     equipment_specs = {
#         'heating': 15,
#         'waterHeating': 12,
#         'clothesDrying': 13,
#         'cooking': 15
#     }
    
#     df_copy = df.copy()
#     df_damages_copy = df_damages.copy()

#     # Calculate the lifetime damages and corresponding NPV based on the policy policy_scenario
#     df_new_columns = calculate_lifetime_damages_grid_scenario(df_copy, df_damages_copy, menu_mp, equipment_specs, policy_scenario, interest_rate)

#     # Drop any overlapping columns from df_copy
#     overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)
#     if not overlapping_columns.empty:
#         df_copy.drop(columns=overlapping_columns, inplace=True)

#     # Merge new columns into the original DataFrame
#     df_copy = df_copy.join(df_new_columns, how='left')

#     return df_copy

# def calculate_lifetime_damages_grid_scenario(df_copy, df_damages_copy, menu_mp, equipment_specs, policy_scenario, interest_rate):
#     """
#     Calculate the NPV of climate, health, and public damages over the equipment's lifetime
#     under different grid decarbonization scenarios.

#     Parameters:
#     - df_copy (DataFrame): A copy of the original DataFrame to store NPV calculations.
#     - menu_mp (str): Menu identifier used in column names.
#     - equipment_specs (dict): Dictionary containing lifetimes for each equipment category.
#     - policy_scenario (str): Specifies the grid policy_scenario ('No Inflation Reduction Act', 'AEO2023 Reference Case').
#     - interest_rate (float): Discount rate for NPV calculation.

#     Returns:
#     - DataFrame: A DataFrame containing the calculated NPV values for each category.
#     """
#     # Determine the policy_scenario prefix based on the policy policy_scenario
#     if policy_scenario == 'No Inflation Reduction Act':
#         scenario_prefix = f"preIRA_mp{menu_mp}_"
#     elif policy_scenario == 'AEO2023 Reference Case':
#         scenario_prefix = f"iraRef_mp{menu_mp}_"
#     else:
#         raise ValueError("Invalid Policy policy_scenario! Please choose from 'No Inflation Reduction Act' or 'AEO2023 Reference Case'.")
    
#     # Create a DataFrame to hold the NPV calculations
#     npv_columns = {}
    
#     for category, lifetime in equipment_specs.items():
#         print(f"""\nCalculating Public NPV for {category}...
#             lifetime: {lifetime}, interest_rate: {interest_rate}, policy_scenario: {policy_scenario}""")
#         # For LRMER and SRMER
#         for mer_type in ['lrmer', 'srmer']:
#             print(f"Type of Marginal Emissions Rate Factor: {mer_type}")
#             # Initialize NPV columns for each category
#             public_npv_key = f'{scenario_prefix}{category}_public_npv_{mer_type}'
            
#             # Initialize NPV columns in the dictionary if they don't exist
#             npv_columns[public_npv_key] = npv_columns.get(public_npv_key, 0)
                
#             for year in range(1, lifetime + 1):
#                 year_label = year + 2023
                
#                 base_climate_damages = df_damages_copy[f'baseline_{year_label}_{category}_damages_climate_{mer_type}']
                
#                 retrofit_climate_damages = df_damages_copy[f'{scenario_prefix}{year_label}_{category}_damages_climate_{mer_type}']
                
#                 # Apply the discount factor to each year's damages
#                 discount_factor = 1 / ((1 + interest_rate) ** year)
                    
#                 npv_columns[public_npv_key] += ((base_climate_damages - retrofit_climate_damages) * discount_factor).round(2)
    
#     # Convert the dictionary to a DataFrame and return it
#     npv_df = pd.DataFrame(npv_columns, index=df_copy.index)
#     return npv_df


# In[15]:


# Use CCI to adjust for cost differences when compared to the national average
# Function to map city to its average cost
def map_average_cost(city):
    if city in average_cost_map:
        return average_cost_map[city]
    elif city == 'Not in a census Place' or city == 'In another census Place':
        return average_cost_map.get('+30 City Average')
    else:
        return average_cost_map.get('+30 City Average')


# In[16]:


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


# In[17]:


def calculate_heating_installation_premium(df, rsMeans_national_avg, cpi_ratio_2023_2013):
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
            premium_cost = 400 * cpi_ratio_2023_2013
        
        # Installation cost for homes without central AC and an existing boiler as heating system
        # Deetjen: Install SEER 15, 8.5 HSPF ASHP: NREL REMDB High Cost is $4800 USD-2013        
        elif 'Boiler' in row['heating_type']:
            premium_cost = 1500 * cpi_ratio_2023_2013
        
        # Apply CPI adjustment above and regional cost index adjustment below
        adjusted_cost = round(premium_cost * (row['rsMeans_CCI_avg'] / rsMeans_national_avg), 2)
        df.at[index, f'mp{menu_mp}_heating_installation_premium'] = adjusted_cost
        
    return df


# In[18]:


# UPDATED AUGUST 22, 2024 @ 9:40 PM (~ENSURE COLS UPDATE WHEN FUNCTION RE-RUN. DROP OLD OVERLAPPING COLS~)
# Replacement Cost Function and Helper Functions (Parametes, Formula)

# Helper function to get parameters based on end use
def get_end_use_replacement_parameters(df, end_use):
    parameters = {
        'heating': {
            'conditions': [
                (df['base_heating_fuel'] == 'Propane'),
                (df['base_heating_fuel'] == 'Fuel Oil'),
                (df['base_heating_fuel'] == 'Natural Gas'),
                (df['base_heating_fuel'] == 'Electricity') & (df['heating_type'] == 'Electricity ASHP'),
                (df['base_heating_fuel'] == 'Electricity')
            ],
            'tech_eff_pairs': [
                ('Propane Furnace', '94 AFUE'),
                ('Fuel Oil Furnace', '95 AFUE'),
                ('Natural Gas Furnace', '95 AFUE'),
                ('Electric ASHP', 'SEER 18, 9.3 HSPF'),
                ('Electric Furnace', '100 AFUE')
            ],
            'cost_components': ['unitCost', 'otherCost', 'cost_per_kBtuh']
        },
        'waterHeating': {
            'conditions': [
                (df['base_waterHeating_fuel'] == 'Fuel Oil'),
                (df['base_waterHeating_fuel'] == 'Natural Gas'),
                (df['base_waterHeating_fuel'] == 'Propane'),
                (df['water_heater_efficiency'].isin(['Electric Standard', 'Electric Premium'])),
                (df['water_heater_efficiency'] == 'Electric Heat Pump, 80 gal')
            ],
            'tech_eff_pairs': [
                ('Fuel Oil Water Heater', 0.68),
                ('Natural Gas Water Heater', 0.67),
                ('Propane Water Heater', 0.67),
                ('Electric Water Heater', 0.95),
                ('Electric Heat Pump Water Heater, 80 gal', 2.35)
            ],
            'cost_components': ['unitCost', 'cost_per_gallon']
        },
        'clothesDrying': {
            'conditions': [
                (df['base_clothesDrying_fuel'] == 'Electricity'),
                (df['base_clothesDrying_fuel'] == 'Natural Gas'),
                (df['base_clothesDrying_fuel'] == 'Propane')
            ],
            'tech_eff_pairs': [
                ('Electric Clothes Dryer', 3.1),
                ('Natural Gas Clothes Dryer', 2.75),
                ('Propane Clothes Dryer', 2.75)
            ],
            'cost_components': ['unitCost']
        },
        'cooking': {
            'conditions': [
                (df['base_cooking_fuel'] == 'Electricity'),
                (df['base_cooking_fuel'] == 'Natural Gas'),
                (df['base_cooking_fuel'] == 'Propane')
            ],
            'tech_eff_pairs': [
                ('Electric Range', 0.74),
                ('Natural Gas Range', 0.4),
                ('Propane Range', 0.4)
            ],
            'cost_components': ['unitCost']
        }
    }
    if end_use not in parameters:
        raise ValueError(f"Invalid end_use specified: {end_use}")
    return parameters[end_use]

# UPDATED AUGUST 22, 2024 @ 9:40 PM (~ENSURE COLS UPDATE WHEN FUNCTION RE-RUN. DROP OLD OVERLAPPING COLS~)
def calculate_replacement_cost_per_row(df_valid, sampled_costs_dict, rsMeans_national_avg, menu_mp, end_use):
    """
    Helper function to calculate the replacement cost for each row based on the end use.

    Parameters:
    df_valid (pd.DataFrame): Filtered DataFrame containing valid rows.
    sampled_costs_dict (dict): Dictionary with sampled costs for each component.
    rsMeans_national_avg (float): National average value for cost adjustment.
    menu_mp (int): Menu option identifier.
    end_use (str): Type of end-use to calculate replacement cost for ('heating', 'waterHeating', 'clothesDrying', 'cooking').

    Returns:
    tuple: Tuple containing the calculated replacement costs and the cost column name.
    """
    if end_use == 'heating':
        replacement_cost = (
            sampled_costs_dict['unitCost'] +
            sampled_costs_dict['otherCost'] +
            (df_valid['total_heating_load_kBtuh'] * sampled_costs_dict['cost_per_kBtuh'])
        ) * (df_valid['rsMeans_CCI_avg'] / rsMeans_national_avg)
        cost_column_name = f'mp{menu_mp}_heating_replacementCost'
    elif end_use == 'waterHeating':
        replacement_cost = (
            sampled_costs_dict['unitCost'] +
            (sampled_costs_dict['cost_per_gallon'] * df_valid['size_water_heater_gal'])
        ) * (df_valid['rsMeans_CCI_avg'] / rsMeans_national_avg)
        cost_column_name = f'mp{menu_mp}_waterHeating_replacementCost'
    else:
        replacement_cost = sampled_costs_dict['unitCost'] * (df_valid['rsMeans_CCI_avg'] / rsMeans_national_avg)
        cost_column_name = f'mp{menu_mp}_{end_use}_replacementCost'
    
    return replacement_cost, cost_column_name

# UPDATED AUGUST 22, 2024 @ 9:40 PM (~ENSURE COLS UPDATE WHEN FUNCTION RE-RUN. DROP OLD OVERLAPPING COLS~)
def calculate_replacement_cost(df, cost_dict, rsMeans_national_avg, menu_mp, end_use):
    """
    General function to calculate replacement costs for various end-uses based on fuel types, costs, and efficiency.

    Parameters:
    df (pd.DataFrame): DataFrame containing data for different scenarios.
    cost_dict (dict): Dictionary with cost information for different technology and efficiency combinations.
    rsMeans_national_avg (float): National average value for cost adjustment.
    menu_mp (int): Menu option identifier.
    end_use (str): Type of end-use to calculate replacement cost for ('heating', 'waterHeating', 'clothesDrying', 'cooking').

    Returns:
    pd.DataFrame: Updated DataFrame with calculated replacement costs.
    """
    
    # Validate menu_mp
    valid_menu_mps = [7, 8, 9, 10]
    if menu_mp not in valid_menu_mps:
        raise ValueError("Please enter a valid measure package number for menu_mp. Should be 7, 8, 9, or 10.")
    
    # Get conditions, technology-efficiency pairs, and cost components for the specified end_use
    params = get_end_use_replacement_parameters(df, end_use)
    conditions = params['conditions']
    tech_eff_pairs = params['tech_eff_pairs']
    cost_components = params['cost_components']
   
    # Map each condition to its tech and efficiency
    tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
    eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default=np.nan)

    # Convert efficiency values to appropriate types
    if end_use == 'heating':
        eff = np.array([str(e) if e != 'unknown' else np.nan for e in eff])
    else:
        eff = np.array([float(e) if e != 'unknown' else np.nan for e in eff])

    # Filter out rows with unknown technology and NaN efficiency
    valid_indices = tech != 'unknown'
    tech = tech[valid_indices]
    eff = eff[valid_indices]
    df_valid = df.loc[valid_indices].copy()

    # Initialize dictionaries to store sampled costs
    sampled_costs_dict = {}

    # Calculate costs for each component
    for cost_component in cost_components:
        progressive_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_progressive', np.nan) for t, e in zip(tech, eff)])
        reference_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_reference', np.nan) for t, e in zip(tech, eff)])
        conservative_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_conservative', np.nan) for t, e in zip(tech, eff)])

        # Handle missing cost data
        if np.isnan(progressive_costs).any() or np.isnan(reference_costs).any() or np.isnan(conservative_costs).any():
            missing_indices = np.where(np.isnan(progressive_costs) | np.isnan(reference_costs) | np.isnan(conservative_costs))
            print(f"Missing data at indices: {missing_indices}")
            print(f"Tech with missing data: {tech[missing_indices]}")
            print(f"Efficiencies with missing data: {eff[missing_indices]}")
            
            raise ValueError(f"Missing cost data for some technology and efficiency combinations in cost_component {cost_component}")

        # Calculate mean and standard deviation assuming the costs represent the 10th, 50th, and 90th percentiles of a normal distribution
        mean_costs = reference_costs
        std_costs = (conservative_costs - progressive_costs) / (norm.ppf(0.90) - norm.ppf(0.10))

        # Sample from the normal distribution for each row
        sampled_costs = np.random.normal(loc=mean_costs, scale=std_costs)
        sampled_costs_dict[cost_component] = sampled_costs

    # Calculate the replacement cost for each row
    replacement_cost, cost_column_name = calculate_replacement_cost_per_row(df_valid, sampled_costs_dict, rsMeans_national_avg, menu_mp, end_use)

    # Add the calculated costs to a new DataFrame, rounded to 2 decimal places
    df_new_columns = pd.DataFrame({cost_column_name: np.round(replacement_cost, 2)}, index=df_valid.index)

    # Identify overlapping columns between the new and existing DataFrame
    overlapping_columns = df_new_columns.columns.intersection(df.columns)

    # Drop overlapping columns from the original DataFrame
    if not overlapping_columns.empty:
        df.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into the original DataFrame, ensuring no duplicates or overwrites occur
    df = df.join(df_new_columns, how='left')

    return df


# In[19]:


# UPDATED AUGUST 22, 2024 @ 9:30 PM (~ENSURE COLS UPDATE WHEN FUNCTION RE-RUN. DROP OLD OVERLAPPING COLS~)

# Installation Cost Function and Helper Functions (Parametes, Formula)
# Helper function to get parameters based on end use
def get_end_use_installation_parameters(df, end_use, menu_mp):
    parameters = {
        'heating': {
            'conditions': [
                (df['hvac_has_ducts'] == 'Yes') & (menu_mp == 7),
                (df['hvac_has_ducts'] == 'No') & (menu_mp == 7),
                (df['hvac_has_ducts'] == 'Yes') & (menu_mp != 7),
                (df['hvac_has_ducts'] == 'No') & (menu_mp != 7)
            ],
            'tech_eff_pairs': [
                ('Electric ASHP', 'SEER 18, 9.3 HSPF'),
                ('Electric MSHP', 'SEER 18, 9.6 HSPF'),
                ('Electric MSHP - Ducted', 'SEER 15.5, 10 HSPF'),
                ('Electric MSHP', 'SEER 29.3, 14 HSPF')
            ],
            'cost_components': ['unitCost', 'otherCost', 'cost_per_kBtuh']
        },
        'waterHeating': {
            'conditions': [
                (df['upgrade_water_heater_efficiency'] == 'Electric Heat Pump, 50 gal, 3.45 UEF'),
                (df['upgrade_water_heater_efficiency'] == 'Electric Heat Pump, 66 gal, 3.35 UEF'),
                (df['upgrade_water_heater_efficiency'] == 'Electric Heat Pump, 80 gal, 3.45 UEF')
            ],
            'tech_eff_pairs': [
                ('Electric Heat Pump Water Heater, 50 gal', 3.45),
                ('Electric Heat Pump Water Heater, 66 gal', 3.35),
                ('Electric Heat Pump Water Heater, 80 gal', 3.45),
            ],
            'cost_components': ['unitCost', 'cost_per_gallon']
        },
        'clothesDrying': {
            'conditions': [
                df['upgrade_clothes_dryer'].str.contains('Electric, Premium, Heat Pump, Ventless', na=False),
                ~df['upgrade_clothes_dryer'].str.contains('Electric, Premium, Heat Pump, Ventless', na=False),
            ],
            'tech_eff_pairs': [
                ('Electric HP Clothes Dryer', 5.2),
                ('Electric Clothes Dryer', 3.1),
            ],
            'cost_components': ['unitCost']
        },
        'cooking': {
            'conditions': [
                df['upgrade_cooking_range'].str.contains('Electric, Induction', na=False),
                ~df['upgrade_cooking_range'].str.contains('Electric, Induction', na=False),
            ],
            'tech_eff_pairs': [
                ('Electric Induction Range', 0.84),
                ('Electric Range, Modern', 0.74),
            ],
            'cost_components': ['unitCost']
        }
    }
    if end_use not in parameters:
        raise ValueError(f"Invalid end_use specified: {end_use}")
    return parameters[end_use]

# UPDATED AUGUST 22, 2024 @ 9:30 PM (~ENSURE COLS UPDATE WHEN FUNCTION RE-RUN. DROP OLD OVERLAPPING COLS~)
def calculate_installation_cost_per_row(df_valid, sampled_costs_dict, rsMeans_national_avg, menu_mp, end_use):
    """
    Helper function to calculate the installation cost for each row based on the end use.

    Parameters:
    df_valid (pd.DataFrame): Filtered DataFrame containing valid rows.
    sampled_costs_dict (dict): Dictionary with sampled costs for each component.
    rsMeans_national_avg (float): National average value for cost adjustment.
    menu_mp (int): Menu option identifier.
    end_use (str): Type of end-use to calculate installation cost for ('heating', 'waterHeating', 'clothesDrying', 'cooking').

    Returns:
    tuple: Tuple containing the calculated installation costs and the cost column name.
    """
    if end_use == 'heating':
        installation_cost = (
            sampled_costs_dict['unitCost'] +
            sampled_costs_dict['otherCost'] +
            (df_valid['total_heating_load_kBtuh'] * sampled_costs_dict['cost_per_kBtuh'])
        ) * (df_valid['rsMeans_CCI_avg'] / rsMeans_national_avg)
        cost_column_name = f'mp{menu_mp}_heating_installationCost'
    elif end_use == 'waterHeating':
        installation_cost = (
            sampled_costs_dict['unitCost'] +
            (sampled_costs_dict['cost_per_gallon'] * df_valid['size_water_heater_gal'])
        ) * (df_valid['rsMeans_CCI_avg'] / rsMeans_national_avg)
        cost_column_name = f'mp{menu_mp}_waterHeating_installationCost'
    else:
        installation_cost = sampled_costs_dict['unitCost'] * (df_valid['rsMeans_CCI_avg'] / rsMeans_national_avg)
        cost_column_name = f'mp{menu_mp}_{end_use}_installationCost'
    
    return installation_cost, cost_column_name

# UPDATED AUGUST 22, 2024 @ 9:30 PM (~ENSURE COLS UPDATE WHEN FUNCTION RE-RUN. DROP OLD OVERLAPPING COLS~)
def calculate_installation_cost(df, cost_dict, rsMeans_national_avg, menu_mp, end_use):
    """
    General function to calculate installation costs for various end-uses based on fuel types, costs, and efficiency.

    Parameters:
    df (pd.DataFrame): DataFrame containing data for different scenarios.
    cost_dict (dict): Dictionary with cost information for different technology and efficiency combinations.
    rsMeans_national_avg (float): National average value for cost adjustment.
    menu_mp (int): Menu option identifier.
    end_use (str): Type of end-use to calculate installation cost for ('heating', 'waterHeating', 'clothesDrying', 'cooking').

    Returns:
    pd.DataFrame: Updated DataFrame with calculated installation costs.
    """
    
    # Validate menu_mp 
    valid_menu_mps = [7, 8, 9, 10]
    if menu_mp not in valid_menu_mps:
        raise ValueError("Please enter a valid measure package number for menu_mp. Should be 7, 8, 9, or 10.")
    
    # Get conditions, technology-efficiency pairs, and cost components for the specified end_use
    params = get_end_use_installation_parameters(df, end_use, menu_mp)
    conditions = params['conditions']
    tech_eff_pairs = params['tech_eff_pairs']
    cost_components = params['cost_components']
   
    # Map each condition to its tech and efficiency
    tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
    eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default=np.nan)

    # Convert efficiency values to appropriate types
    if end_use == 'heating':
        eff = np.array([str(e) if e != 'unknown' else np.nan for e in eff])
    else:
        eff = np.array([float(e) if e != 'unknown' else np.nan for e in eff])

    # Filter out rows with unknown technology and NaN efficiency
    valid_indices = tech != 'unknown'
    tech = tech[valid_indices]
    eff = eff[valid_indices]
    df_valid = df.loc[valid_indices].copy()

    # Initialize dictionaries to store sampled costs
    sampled_costs_dict = {}

    # Calculate costs for each component
    for cost_component in cost_components:
        progressive_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_progressive', np.nan) for t, e in zip(tech, eff)])
        reference_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_reference', np.nan) for t, e in zip(tech, eff)])
        conservative_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_conservative', np.nan) for t, e in zip(tech, eff)])

        # Handle missing cost data
        if np.isnan(progressive_costs).any() or np.isnan(reference_costs).any() or np.isnan(conservative_costs).any():
            missing_indices = np.where(np.isnan(progressive_costs) | np.isnan(reference_costs) | np.isnan(conservative_costs))
            print(f"Missing data at indices: {missing_indices}")
            print(f"Tech with missing data: {tech[missing_indices]}")
            print(f"Efficiencies with missing data: {eff[missing_indices]}")
            
            raise ValueError(f"Missing cost data for some technology and efficiency combinations in cost_component {cost_component}")

        # Calculate mean and standard deviation assuming the costs represent the 10th, 50th, and 90th percentiles of a normal distribution
        mean_costs = reference_costs
        std_costs = (conservative_costs - progressive_costs) / (norm.ppf(0.90) - norm.ppf(0.10))

        # Sample from the normal distribution for each row
        sampled_costs = np.random.normal(loc=mean_costs, scale=std_costs)
        sampled_costs_dict[cost_component] = sampled_costs

    # Calculate the installation cost for each row
    installation_cost, cost_column_name = calculate_installation_cost_per_row(df_valid, sampled_costs_dict, rsMeans_national_avg, menu_mp, end_use)

    # Add the calculated costs to a new DataFrame, rounded to 2 decimal places
    df_new_columns = pd.DataFrame({cost_column_name: np.round(installation_cost, 2)}, index=df_valid.index)

    # Identify overlapping columns between the new and existing DataFrame
    overlapping_columns = df_new_columns.columns.intersection(df.columns)

    # Drop overlapping columns from the original DataFrame
    if not overlapping_columns.empty:
        df.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into the original DataFrame, ensuring no duplicates or overwrites occur
    df = df.join(df_new_columns, how='left')

    return df


# In[ ]:


# LAST UPDATE AUGUST 21, 2024 @ 11:40 PM

# POTENTIALLY UPDATE CODE IN THE FUTURE TO ACCOUNT FOR CHANGES IN CAPITAL COSTS BASED ON SCENARIOS (BESIDES IRA REBATES)
# Note: CURRENT MODELING ASSUMES EQUIPMENT PRICES ARE THE SAME UNDER IRA REF AND IRA HIGH
# THIS MAY BE UPDATED IN THE FUTURE, SO WE STILL USE policy_scenario PREFIXES FOR TOTAL AND NET CAPITAL COSTS
# COSTS ARE DIFFERENT FOR PRE-IRA BECAUSE NO REBATES ARE APPLIED
def calculate_private_NPV(df, df_fuelCosts, interest_rate, input_mp, menu_mp, policy_scenario):
    """
    Calculate the private net present value (NPV) for various equipment categories,
    considering different cost assumptions and potential IRA rebates. The function adjusts
    equipment costs for inflation and regional cost differences, and calculates NPV based
    on cost savings between baseline and retrofit scenarios.

    Parameters:
        df (DataFrame): Input DataFrame with installation costs, fuel savings, and potential rebates.
        interest_rate (float): Annual discount rate used for NPV calculation.
        menu_mp (str): Prefix for columns in the DataFrame.
        input_mp (str): Input policy_scenario for calculating costs.
        policy_scenario (str): Policy policy_scenario that determines electricity grid projections. 
                               Accepted values: 'AEO2023 Reference Case'.

    Returns:
        DataFrame: The input DataFrame updated with calculated private NPV and adjusted equipment costs.
    """
    # POTENTIALLY UPDATE CODE IN THE FUTURE TO ACCOUNT FOR CHANGES IN CAPITAL COSTS BASED ON SCENARIOS (BESIDES IRA REBATES)
    # Note: CURRENT MODELING ASSUMES EQUIPMENT PRICES ARE THE SAME UNDER IRA REF AND IRA HIGH
    # THIS MAY BE UPDATED IN THE FUTURE, SO WE STILL USE policy_scenario PREFIXES FOR TOTAL AND NET CAPITAL COSTS
    # COSTS ARE DIFFERENT FOR PRE-IRA BECAUSE NO REBATES ARE APPLIED   
    equipment_specs = {
        'heating': 15,
        'waterHeating': 12,
        'clothesDrying': 13,
        'cooking': 15
    }
    
    df_copy = df.copy()

    df_fuelCosts_copy = df_fuelCosts.copy()

    df_new_columns = pd.DataFrame(index=df_copy.index)
    
    for category, lifetime in equipment_specs.items():
        # print(f"\nCalculating for category: {category} with lifetime: {lifetime}")
        
        total_capital_cost, net_capital_cost = calculate_costs(df_copy, category, input_mp, menu_mp, policy_scenario)
        
        # print(f"Total capital cost for {category}: {total_capital_cost}")
        # print(f"Net capital cost for {category}: {net_capital_cost}")
        
        calculate_and_update_npv(df_new_columns, df_fuelCosts_copy, category, menu_mp, interest_rate, lifetime, total_capital_cost, net_capital_cost, policy_scenario)
      
    overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)
    if not overlapping_columns.empty:
        df_copy.drop(columns=overlapping_columns, inplace=True)

    df_copy = df_copy.join(df_new_columns, how='left')
    # print("Final DataFrame after NPV calculations:\n", df_copy.head())
    return df_copy

def calculate_costs(df_copy, category, input_mp, menu_mp, policy_scenario):
    """
    Calculate total and net capital costs based on the equipment category and cost assumptions.

    Parameters:
        df_copy (DataFrame): DataFrame containing cost data.
        category (str): Equipment category.
        menu_mp (str): Prefix for columns in the DataFrame.
        input_mp (str): Input policy_scenario for calculating costs.
        ira_rebates (bool): Flag indicating whether IRA rebates are applied.

    Returns:
        tuple: Total and net capital costs.
    """
    print(f"""\nCalculating costs for {category}...
          input_mp: {input_mp}, menu_mp: {menu_mp}, policy_scenario: {policy_scenario}""")


    # POTENTIALLY UPDATE CODE IN THE FUTURE TO ACCOUNT FOR CHANGES IN CAPITAL COSTS BASED ON SCENARIOS (BESIDES IRA REBATES)
    # Note: CURRENT MODELING ASSUMES EQUIPMENT PRICES ARE THE SAME UNDER IRA REF AND IRA HIGH
    # THIS MAY BE UPDATED IN THE FUTURE, SO WE STILL USE policy_scenario PREFIXES FOR TOTAL AND NET CAPITAL COSTS
    # COSTS ARE DIFFERENT FOR PRE-IRA BECAUSE NO REBATES ARE APPLIED
    if policy_scenario == 'No Inflation Reduction Act':
        if category == 'heating':
            if input_mp == 'upgrade09':            
                weatherization_cost = df_copy[f'mp9_enclosure_upgradeCost'].fillna(0)
            elif input_mp == 'upgrade10':
                weatherization_cost = df_copy[f'mp10_enclosure_upgradeCost'].fillna(0)
            else:
                weatherization_cost = 0.0
            # print(f"Weatherization cost (no IRA rebates): {weatherization_cost}")
            
            total_capital_cost = (df_copy[f'mp{menu_mp}_{category}_installationCost'].fillna(0) + 
                                  weatherization_cost + 
                                  df_copy[f'mp{menu_mp}_heating_installation_premium'].fillna(0))
            net_capital_cost = total_capital_cost - df_copy[f'mp{menu_mp}_{category}_replacementCost'].fillna(0)
            
        else:
            total_capital_cost = df_copy[f'mp{menu_mp}_{category}_installationCost'].fillna(0)
            net_capital_cost = total_capital_cost - df_copy[f'mp{menu_mp}_{category}_replacementCost'].fillna(0)
    
    # POTENTIALLY UPDATE CODE IN THE FUTURE TO ACCOUNT FOR CHANGES IN CAPITAL COSTS BASED ON SCENARIOS (BESIDES IRA REBATES)
    # Note: CURRENT MODELING ASSUMES EQUIPMENT PRICES ARE THE SAME UNDER IRA REF AND IRA HIGH
    # THIS MAY BE UPDATED IN THE FUTURE, SO WE STILL USE policy_scenario PREFIXES FOR TOTAL AND NET CAPITAL COSTS
    # COSTS ARE DIFFERENT FOR PRE-IRA BECAUSE NO REBATES ARE APPLIED
    else:
        if category == 'heating':
            if input_mp == 'upgrade09':            
                weatherization_cost = df_copy[f'mp9_enclosure_upgradeCost'].fillna(0) - df_copy[f'weatherization_rebate_amount'].fillna(0)
            elif input_mp == 'upgrade10':
                weatherization_cost = df_copy[f'mp10_enclosure_upgradeCost'].fillna(0) - df_copy[f'weatherization_rebate_amount'].fillna(0)
            else:
                weatherization_cost = 0.0       
            # print(f"Weatherization cost (with IRA rebates): {weatherization_cost}")
            
            installation_cost = (df_copy[f'mp{menu_mp}_{category}_installationCost'].fillna(0) + 
                                 weatherization_cost + 
                                 df_copy[f'mp{menu_mp}_{category}_installation_premium'].fillna(0))
            
            rebate_amount = df_copy[f'mp{menu_mp}_{category}_rebate_amount'].fillna(0)
            total_capital_cost = installation_cost - rebate_amount
            net_capital_cost = total_capital_cost - df_copy[f'mp{menu_mp}_{category}_replacementCost'].fillna(0)
        
        else:
            installation_cost = df_copy[f'mp{menu_mp}_{category}_installationCost'].fillna(0)
            rebate_amount = df_copy[f'mp{menu_mp}_{category}_rebate_amount'].fillna(0)
            total_capital_cost = installation_cost - rebate_amount
            net_capital_cost = total_capital_cost - df_copy[f'mp{menu_mp}_{category}_replacementCost'].fillna(0)

    # print(f"Calculated total_capital_cost: {total_capital_cost}, net_capital_cost: {net_capital_cost}")
    return total_capital_cost, net_capital_cost

def calculate_and_update_npv(df_new_columns, df_fuelCosts_copy, category, menu_mp, interest_rate, lifetime, total_capital_cost, net_capital_cost, policy_scenario):
    """
    Calculate and update the NPV values in the DataFrame based on provided capital costs.

    Parameters:
        df_new_columns (DataFrame): DataFrame to update.
        df_fuelCosts_copy (DataFrame): Original DataFrame containing savings data.
        category (str): Equipment category.
        menu_mp (str): Prefix for columns in the DataFrame.
        interest_rate (float): Discount rate for NPV calculation.
        lifetime (int): Expected lifetime of the equipment.
        total_capital_cost (float): Total capital cost of the equipment.
        net_capital_cost (float): Net capital cost after considering replacements.
        ira_rebates (bool): Flag to consider IRA rebates in calculations.
    """
    # Determine the policy_scenario prefix based on the policy policy_scenario
    if policy_scenario == 'No Inflation Reduction Act':
        scenario_prefix = f"preIRA_mp{menu_mp}_"
    elif policy_scenario == 'AEO2023 Reference Case':
        scenario_prefix = f"iraRef_mp{menu_mp}_"
    else:
        raise ValueError("Invalid Policy policy_scenario! Please choose from 'AEO2023 Reference Case'.")
        
    print(f"""\nCalculating Private NPV for {category}...
          lifetime: {lifetime}, interest_rate: {interest_rate}, policy_scenario: {policy_scenario}
          """)

    # Calculate the discounted savings for each year
    discounted_savings = []
    for year in range(1, lifetime + 1):
        year_label = year + 2023  # Adjust the start year as necessary
        annual_savings = df_fuelCosts_copy[f'{scenario_prefix}{year_label}_{category}_savings_fuelCost'].fillna(0)
        discount_factor = (1 / ((1 + interest_rate) ** year))
        discounted_savings.append(annual_savings * discount_factor)
        # print(f"Year {year_label} savings for {category}: {annual_savings}, discounted: {annual_savings * discount_factor}")
    
    # Sum up the discounted savings over the lifetime
    total_discounted_savings = sum(discounted_savings)
    # print(f"Total discounted savings over {lifetime} years for {category}: {total_discounted_savings}")
    
    # Calculate NPV for less WTP and more WTP scenarios
    npv_lessWTP = round(total_discounted_savings - total_capital_cost, 2)
    npv_moreWTP = round(total_discounted_savings - net_capital_cost, 2)
    
    # POTENTIALLY UPDATE CODE IN THE FUTURE TO ACCOUNT FOR CHANGES IN CAPITAL COSTS BASED ON SCENARIOS (BESIDES IRA REBATES)
    # Note: CURRENT MODELING ASSUMES EQUIPMENT PRICES ARE THE SAME UNDER IRA REF AND IRA HIGH
    # THIS MAY BE UPDATED IN THE FUTURE, SO WE STILL USE policy_scenario PREFIXES FOR TOTAL AND NET CAPITAL COSTS
    # COSTS ARE DIFFERENT FOR PRE-IRA BECAUSE NO REBATES ARE APPLIED
    df_new_columns[f'{scenario_prefix}{category}_total_capitalCost'] = total_capital_cost
    df_new_columns[f'{scenario_prefix}{category}_net_capitalCost'] = net_capital_cost
        
    df_new_columns[f'{scenario_prefix}{category}_private_npv_lessWTP'] = npv_lessWTP
    df_new_columns[f'{scenario_prefix}{category}_private_npv_moreWTP'] = npv_moreWTP
        
    # print(f"Updated df_new_columns with NPV for {category}:\n", df_new_columns[[col for col in df_new_columns.columns if category in col]].head())


# In[ ]:


# # UPDATED SEPTEMBER 14, 2024 @ 4:23 PM
# def adoption_decision(df, policy_scenario):
#     """
#     Updates the provided DataFrame with new columns that reflect decisions about equipment adoption
#     and public impacts based on net present values (NPV). The function handles different scenarios
#     based on input flags for incentives and grid decarbonization.

#     Parameters:
#         df (pandas.DataFrame): The DataFrame containing home equipment data.
#         policy_scenario (str): Policy policy_scenario that determines electricity grid projections. 
#                                Accepted values: 'AEO2023 Reference Case'.

#     Returns:
#         pandas.DataFrame: The modified DataFrame with additional columns for decisions and impacts.

#     Notes:
#         - It adds columns for both individual and public economic evaluations.
#         - Adoption decisions and public impacts are dynamically calculated based on the input parameters.
#     """
#     df_copy = df.copy()
    
#     # Define the lifetimes of different equipment categories
#     upgrade_columns = {
#         'heating': 'upgrade_hvac_heating_efficiency',
#         'waterHeating': 'upgrade_water_heater_efficiency',
#         'clothesDrying': 'upgrade_clothes_dryer',
#         'cooking': 'upgrade_cooking_range'
#     }
    
#     df_new_columns = pd.DataFrame(index=df_copy.index)  # DataFrame to hold new or modified columns

#     # Determine the policy_scenario prefix based on the policy policy_scenario
#     if policy_scenario == 'No Inflation Reduction Act':
#         scenario_prefix = f"preIRA_mp{menu_mp}_"
#     elif policy_scenario == 'AEO2023 Reference Case':
#         scenario_prefix = f"iraRef_mp{menu_mp}_"
#     else:
#         raise ValueError("Invalid Policy Scenario! Please choose from 'No Inflation Reduction Act' or 'AEO2023 Reference Case'.")

#     # Iterate over each equipment category and its respective upgrade column
#     for category, upgrade_column in upgrade_columns.items():
#         # Column names for net NPV, private NPV, and public NPV
#         lessWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_lessWTP' # LESS WTP: BREAK EVEN ON TOTAL CAPITAL COSTS
#         moreWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_moreWTP' # MORE WTP: BREAK EVEN ON NET CAPITAL COSTS (BETTER THAN ALTERNATIVE)

#         public_npv_col = f'{scenario_prefix}{category}_public_npv'
#         rebate_col = f'mp{menu_mp}_{category}_rebate_amount'
#         addition_public_benefit = f'{scenario_prefix}{category}_additional_public_benefit'

#         lessWTP_total_npv_col = f'{scenario_prefix}{category}_total_npv_lessWTP' # LESS WTP: BREAK EVEN ON TOTAL CAPITAL COSTS
#         moreWTP_total_npv_col = f'{scenario_prefix}{category}_total_npv_moreWTP' # MORE WTP: BREAK EVEN ON NET CAPITAL COSTS (BETTER THAN ALTERNATIVE)
#         # Ensure columns are numeric if they exist and convert them
#         for col in [lessWTP_private_npv_col, moreWTP_private_npv_col, public_npv_col, rebate_col]:
#             if col in df.columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
#             else:
#                 print(f"Warning: {col} does not exist in the DataFrame.")

#         # Ensure the columns are present after conversion
#         if lessWTP_private_npv_col in df.columns and moreWTP_private_npv_col in df.columns and public_npv_col in df.columns:
#             # No IRA Rebate so no "Additional Public Benefit"
#             if policy_scenario == 'No Inflation Reduction Act':
#                 df_new_columns[addition_public_benefit] = 0.0
#             else:
#                 # Calculate Additional Public Benefit with IRA Rebates Accounted For and clip at 0
#                 df_new_columns[addition_public_benefit] = (df[public_npv_col] - df[rebate_col]).clip(lower=0)
            
#             # Calculate Total NPV by summing private and public NPVs
#             df_new_columns[lessWTP_total_npv_col] = df[lessWTP_private_npv_col] + df_new_columns[addition_public_benefit] # LESS WTP: BREAK EVEN ON TOTAL CAPITAL COSTS
#             df_new_columns[moreWTP_total_npv_col] = df[moreWTP_private_npv_col] + df_new_columns[addition_public_benefit] # MORE WTP: BREAK EVEN ON NET CAPITAL COSTS (BETTER THAN ALTERNATIVE)

#             # Initialize columns for adoption decisions and public impact
#             adoption_col_name = f'{scenario_prefix}{category}_adoption'
#             retrofit_col_name = f'{scenario_prefix}{category}_retrofit_publicImpact'
#             df_new_columns[adoption_col_name] = 'Tier 4: Averse'  # Default value for all rows
#             df_new_columns[retrofit_col_name] = 'No Retrofit'  # Default public impact

#             # Conditions for determining adoption decisions
#             conditions = [
#                 df[upgrade_column].isna(),
#                 df[lessWTP_private_npv_col] > 0,
#                 (df[lessWTP_private_npv_col] < 0) & (df[moreWTP_private_npv_col] > 0),
#                 (df[lessWTP_private_npv_col] < 0) & (df[moreWTP_private_npv_col] <= 0) & (df_new_columns[moreWTP_total_npv_col] > 0) & (df_new_columns[addition_public_benefit] > 0), # Ensures only Tier 3 for IRA Scenario
#             ]

#             choices = ['Existing Equipment', 'Tier 1: Feasible', 'Tier 2: Feasible vs. Alternative', 'Tier 3: Subsidy-Dependent Feasibility']
#             df_new_columns[adoption_col_name] = np.select(conditions, choices, default='Tier 4: Averse')

#             # Conditions and choices for public impacts
#             public_conditions = [
#                 df[public_npv_col] > 0,
#                 df[public_npv_col] < 0
#             ]
            
#             public_choices = ['Public Benefit', 'Public Detriment']
#             df_new_columns[retrofit_col_name] = np.select(public_conditions, public_choices, default='No Retrofit')
#         else:
#             print(f"Warning: One or more columns ({lessWTP_private_npv_col}, {moreWTP_private_npv_col}, {public_npv_col}) are missing or not numeric.")
    
#     # Identify overlapping columns between the new and existing DataFrame.
#     overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)

#     # Drop overlapping columns from df_copy.
#     if not overlapping_columns.empty:
#         df_copy.drop(columns=overlapping_columns, inplace=True)

#     # Merge new columns into df_copy, ensuring no duplicates or overwrites occur.
#     df_copy = df_copy.join(df_new_columns, how='left')

#     # Return the updated DataFrame.
#     return df_copy


# In[ ]:


# UPDATED SEPTEMBER 20, 2024 @ 12:30 AM
def adoption_decision(df, policy_scenario):
    """
    Updates the provided DataFrame with new columns that reflect decisions about equipment adoption
    and public impacts based on net present values (NPV). The function handles different scenarios
    based on input flags for incentives and grid decarbonization.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing home equipment data.
        policy_scenario (str): Policy policy_scenario that determines electricity grid projections. 
                               Accepted values: 'AEO2023 Reference Case'.

    Returns:
        pandas.DataFrame: The modified DataFrame with additional columns for decisions and impacts.

    Notes:
        - It adds columns for both individual and public economic evaluations.
        - Adoption decisions and public impacts are dynamically calculated based on the input parameters.
    """
    df_copy = df.copy()
    
    # Define the lifetimes of different equipment categories
    upgrade_columns = {
        'heating': 'upgrade_hvac_heating_efficiency',
        'waterHeating': 'upgrade_water_heater_efficiency',
        'clothesDrying': 'upgrade_clothes_dryer',
        'cooking': 'upgrade_cooking_range'
    }
    
    df_new_columns = pd.DataFrame(index=df_copy.index)  # DataFrame to hold new or modified columns

    # Determine the policy_scenario prefix based on the policy policy_scenario
    if policy_scenario == 'No Inflation Reduction Act':
        scenario_prefix = f"preIRA_mp{menu_mp}_"
    elif policy_scenario == 'AEO2023 Reference Case':
        scenario_prefix = f"iraRef_mp{menu_mp}_"
    else:
        raise ValueError("Invalid Policy Scenario! Please choose from 'No Inflation Reduction Act' or 'AEO2023 Reference Case'.")

    # Iterate over each equipment category and its respective upgrade column
    for category, upgrade_column in upgrade_columns.items():
        print(f"\nCalculating Adoption Potential for {category} under '{policy_scenario}' Scenario...")

        # For LRMER and SRMER
        for mer_type in ['lrmer', 'srmer']:
            print(f"Type of Marginal Emissions Rate Factor: {mer_type}")

            # Column names for net NPV, private NPV, and public NPV
            lessWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_lessWTP' # LESS WTP: BREAK EVEN ON TOTAL CAPITAL COSTS
            moreWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_moreWTP' # MORE WTP: BREAK EVEN ON NET CAPITAL COSTS (BETTER THAN ALTERNATIVE)

            public_npv_col = f'{scenario_prefix}{category}_public_npv_{mer_type}'
            rebate_col = f'mp{menu_mp}_{category}_rebate_amount'
            addition_public_benefit = f'{scenario_prefix}{category}_additional_public_benefit_{mer_type}'

            lessWTP_total_npv_col = f'{scenario_prefix}{category}_total_npv_lessWTP_{mer_type}' # LESS WTP: BREAK EVEN ON TOTAL CAPITAL COSTS
            moreWTP_total_npv_col = f'{scenario_prefix}{category}_total_npv_moreWTP_{mer_type}' # MORE WTP: BREAK EVEN ON NET CAPITAL COSTS (BETTER THAN ALTERNATIVE)
            # Ensure columns are numeric if they exist and convert them
            for col in [lessWTP_private_npv_col, moreWTP_private_npv_col, public_npv_col, rebate_col]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    print(f"Warning: {col} does not exist in the DataFrame.")

            # Ensure the columns are present after conversion
            if lessWTP_private_npv_col in df.columns and moreWTP_private_npv_col in df.columns and public_npv_col in df.columns:
                # No IRA Rebate so no "Additional Public Benefit"
                if policy_scenario == 'No Inflation Reduction Act':
                    df_new_columns[addition_public_benefit] = 0.0
                else:
                    # Calculate Additional Public Benefit with IRA Rebates Accounted For and clip at 0
                    df_new_columns[addition_public_benefit] = (df[public_npv_col] - df[rebate_col]).clip(lower=0)
                
                # Calculate Total NPV by summing private and public NPVs
                df_new_columns[lessWTP_total_npv_col] = df[lessWTP_private_npv_col] + df_new_columns[addition_public_benefit] # LESS WTP: BREAK EVEN ON TOTAL CAPITAL COSTS
                df_new_columns[moreWTP_total_npv_col] = df[moreWTP_private_npv_col] + df_new_columns[addition_public_benefit] # MORE WTP: BREAK EVEN ON NET CAPITAL COSTS (BETTER THAN ALTERNATIVE)

                # Initialize columns for adoption decisions and public impact
                adoption_col_name = f'{scenario_prefix}{category}_adoption_{mer_type}'
                retrofit_col_name = f'{scenario_prefix}{category}_retrofit_publicImpact_{mer_type}'
                df_new_columns[adoption_col_name] = 'Tier 4: Averse'  # Default value for all rows
                df_new_columns[retrofit_col_name] = 'No Retrofit'  # Default public impact

                # Conditions for determining adoption decisions
                conditions = [
                    df[upgrade_column].isna(),
                    df[lessWTP_private_npv_col] > 0,
                    (df[lessWTP_private_npv_col] < 0) & (df[moreWTP_private_npv_col] > 0),
                    (df[lessWTP_private_npv_col] < 0) & (df[moreWTP_private_npv_col] <= 0) & (df_new_columns[moreWTP_total_npv_col] > 0) & (df_new_columns[addition_public_benefit] > 0), # Ensures only Tier 3 for IRA Scenario
                ]

                choices = ['Existing Equipment', 'Tier 1: Feasible', 'Tier 2: Feasible vs. Alternative', 'Tier 3: Subsidy-Dependent Feasibility']
                df_new_columns[adoption_col_name] = np.select(conditions, choices, default='Tier 4: Averse')

                # Conditions and choices for public impacts
                public_conditions = [
                    df[public_npv_col] > 0,
                    df[public_npv_col] < 0
                ]
                
                public_choices = ['Public Benefit', 'Public Detriment']
                df_new_columns[retrofit_col_name] = np.select(public_conditions, public_choices, default='No Retrofit')
            else:
                print(f"Warning: One or more columns ({lessWTP_private_npv_col}, {moreWTP_private_npv_col}, {public_npv_col}) are missing or not numeric.")
    
    # Identify overlapping columns between the new and existing DataFrame.
    overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)

    # Drop overlapping columns from df_copy.
    if not overlapping_columns.empty:
        df_copy.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into df_copy, ensuring no duplicates or overwrites occur.
    df_copy = df_copy.join(df_new_columns, how='left')

    # Return the updated DataFrame.
    return df_copy


# In[23]:


# def check_adoption_consistency(df, category, upgrade_column):
#     df_copy = df.copy()
    
#     cols_to_display = ['bldg_id',
#                        f'base_{category}_fuel',
#                        f'{upgrade_column}',
#                        f'baseline_{category}_consumption',
#                        f'mp{menu_mp}_{category}_consumption',
#                        f'mp{menu_mp}_{category}_reduction_consumption',
#                        f'baseline_{category}_fuelCost',
#                        f'mp{menu_mp}_{category}_fuelCost',        
#                        f'mp{menu_mp}_{category}_savings_fuelCost',
#                        f'mp{menu_mp}_{category}_net_capitalCost',
#                        f'mp{menu_mp}_{category}_private_npv',
#                        f'baseline_{category}_damages_health',
#                        f'baseline_{category}_damages_climate',
#                        f'mp{menu_mp}_{category}_damages_health',
#                        f'mp{menu_mp}_{category}_damages_climate',
#                        f'mp{menu_mp}_{category}_reduction_damages_health',
#                        f'mp{menu_mp}_{category}_reduction_damages_climate',
#                        f'mp{menu_mp}_{category}_public_npv',
#                        f'mp{menu_mp}_{category}_retrofit_publicImpact',
#                        f'mp{menu_mp}_{category}_total_npv',
#                        f'mp{menu_mp}_{category}_adoption',  
#                        ]    
        
#     # Filter the dataframe to show only the columns relevant for the current cost_type
#     df_filtered = df_copy[cols_to_display]
    
#     return df_filtered


# In[ ]:


# UPDATED SEPTEMBER 6, 2024 @ 12:10 AM
import pandas as pd
import numpy as np
from scipy.stats import norm

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


# In[ ]:


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


# In[26]:


# def check_ira_adoption_consistency(df, category, upgrade_column):
#     df_copy = df.copy()
    
#     cols_to_display = ['bldg_id',
#                        f'base_{category}_fuel',
#                        f'{upgrade_column}',
#                        f'baseline_{category}_consumption',
#                        f'mp{menu_mp}_{category}_consumption',
#                        f'mp{menu_mp}_{category}_reduction_consumption',
#                        f'baseline_{category}_fuelCost',
#                        f'mp{menu_mp}_{category}_fuelCost',        
#                        f'mp{menu_mp}_{category}_savings_fuelCost',
#                        f'mp{menu_mp}_{category}_net_capitalCost',
#                        f'mp{menu_mp}_{category}_private_npv',
#                        f'baseline_{category}_damages_health',
#                        f'baseline_{category}_damages_climate',
#                        f'mp{menu_mp}_{category}_damages_health',
#                        f'mp{menu_mp}_{category}_damages_climate',
#                        f'mp{menu_mp}_{category}_reduction_damages_health',
#                        f'mp{menu_mp}_{category}_reduction_damages_climate',
#                        f'mp{menu_mp}_{category}_public_npv',
#                        f'mp{menu_mp}_{category}_retrofit_publicImpact',
#                        f'mp{menu_mp}_{category}_total_npv',
#                        f'mp{menu_mp}_{category}_adoption',
#                        f'ira_mp{menu_mp}_{category}_net_capitalCost',
#                        f'ira_mp{menu_mp}_{category}_private_npv',
#                        f'ira_mp{menu_mp}_{category}_total_npv',
#                        f'ira_mp{menu_mp}_{category}_adoption',
#                        ]    

#     # Filter the dataframe to show only the relevant columns
#     df_filtered = df_copy[cols_to_display]
    
#     return df_filtered


# ## Moderate Retrofit (MP9): MP8 + Basic Enclosure

# ## Advanced Retrofit (MP10): MP8 + Enhanced Enclosure
# **Notes**
# - There are some inconsistencies for variable names and syntax for calculations
# - The calculations should still end up the same regardless because of order of operations
# - Plan to update for consistency to avoid user confusion.

# In[27]:


# UPDATED AUGUST 22, 2024 @ 7:00 PM
import numpy as np
import pandas as pd
from scipy.stats import norm

# Helper function to get conditions and tech-efficiency pairs for enclosure retrofit
def get_enclosure_parameters(df, retrofit_col):
    if retrofit_col == 'insulation_atticFloor_upgradeCost':
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
            (df['upgrade_insulation_atticFloor'] == 'R-60') & (df['base_insulation_atticFloor'] == 'Uninsulated')
        ]
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
            ('Attic Floor Insulation: R-60', 'Uninsulated')
        ]
    elif retrofit_col == 'infiltration_reduction_upgradeCost':
        conditions = [
            (df['upgrade_infiltration_reduction'] == '30%')
        ]
        tech_eff_pairs = [
            ('Air Leakage Reduction: 30% Reduction', 'Varies')
        ]
    elif retrofit_col == 'duct_sealing_upgradeCost':
        conditions = [
            (df['upgrade_duct_sealing'] == '10% Leakage, R-8') & (df['base_ducts'].str.contains('10% Leakage')),
            (df['upgrade_duct_sealing'] == '10% Leakage, R-8') & (df['base_ducts'].str.contains('20% Leakage')),
            (df['upgrade_duct_sealing'] == '10% Leakage, R-8') & (df['base_ducts'].str.contains('30% Leakage')),
        ]
        tech_eff_pairs = [
            ('Duct Sealing: 10% Leakage, R-8', '10% Leakage'),
            ('Duct Sealing: 10% Leakage, R-8', '20% Leakage'),
            ('Duct Sealing: 10% Leakage, R-8', '30% Leakage'),
        ]
    elif retrofit_col == 'insulation_wall_upgradeCost':
        conditions = [
            (df['upgrade_insulation_wall'] == 'Wood Stud, R-13')
        ]
        tech_eff_pairs = [
            ('Drill-and-fill Wall Insulation: Wood Stud, R-13', 'Wood Stud, Uninsulated')
        ]
    elif retrofit_col == 'insulation_foundation_wall_upgradeCost':
        conditions = [
            (df['upgrade_insulation_foundation_wall'] == 'Wall R-10, Interior')
        ]
        tech_eff_pairs = [
            ('Foundation Wall Insulation: Wall R-10, Interior', 'Uninsulated')
        ]
    elif retrofit_col == 'insulation_rim_joist_upgradeCost':
        conditions = [
            (df['base_insulation_foundation_wall'] == 'Uninsulated') & (df['base_foundation_type'].isin(['Unvented Crawlspace', 'Vented Crawlspace', 'Heated Basement']))
        ]
        tech_eff_pairs = [
            ('Rim Joist Insulation: Wall R-10, Exterior', 'Uninsulated')
        ]
    elif retrofit_col == 'seal_crawlspace_upgradeCost':
        conditions = [
            (df['upgrade_seal_crawlspace'] == 'Unvented Crawlspace')
        ]
        tech_eff_pairs = [
            ('Seal Vented Crawlspace: Unvented Crawlspace', 'Vented Crawlspace')
        ]
    elif retrofit_col == 'insulation_roof_upgradeCost':
        conditions = [
            (df['upgrade_insulation_roof'] == 'Finished, R-30')
        ]
        tech_eff_pairs = [
            ('Insulate Finished Attics and Cathedral Ceilings: Finished, R-30', 'R-30')
        ]
    else:
        raise ValueError(f"Invalid retrofit_col specified: {retrofit_col}")
    
    return {'conditions': conditions, 'tech_eff_pairs': tech_eff_pairs}

# UPDATED AUGUST 22, 2024 @ 7:00 PM
def calculate_enclosure_retrofit_upgradeCosts(df, cost_dict, retrofit_col, params_col, rsMeans_national_avg):
    """
    Calculate the enclosure retrofit upgrade costs based on given parameters and conditions.

    Parameters:
    df (pd.DataFrame): DataFrame containing data for different scenarios.
    cost_dict (dict): Dictionary with cost information for different technology and efficiency combinations.
    retrofit_col (str): Column name for the retrofit cost.
        - NaN value indicates that the retrofit was not performed.
    params_col (str): Column name for the parameter to use in the cost calculation.
    rsMeans_national_avg (float): National average value for cost adjustment.

    Returns:
    pd.DataFrame: Updated DataFrame with calculated retrofit costs.
    """
    
    # Create a copy of the original DataFrame to avoid modifying it directly
    df_copy = df.copy()

    # Get conditions and tech-efficiency pairs for the specified retrofit
    params = get_enclosure_parameters(df_copy, retrofit_col)
    conditions = params['conditions']
    tech_eff_pairs = params['tech_eff_pairs']

    # # Debug: Print the extracted parameters
    # print("Extracted Parameters:", params)

    # Map each condition to its tech and efficiency
    tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
    eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default='unknown')

    # # Debug: Print the mapped tech and efficiency pairs
    # print("Mapped Tech:", tech)
    # print("Mapped Efficiency:", eff)

    # Filter out rows with unknown technology and efficiency
    valid_indices = tech != 'unknown'
    tech = tech[valid_indices]
    eff = eff[valid_indices]
    df_valid = df_copy.loc[valid_indices].copy()

    # # Debug: Print the valid indices and corresponding tech-efficiency pairs
    # print("Valid Indices:", valid_indices)
    # print("Valid Tech:", tech)
    # print("Valid Efficiency:", eff)

    # Initialize dictionary to store sampled costs
    sampled_costs_dict = {}

    # Calculate costs for each component (normalized_cost)
    for cost_component in ['normalized_cost']:
        progressive_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_progressive', np.nan) for t, e in zip(tech, eff)])
        reference_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_reference', np.nan) for t, e in zip(tech, eff)])
        conservative_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_conservative', np.nan) for t, e in zip(tech, eff)])

        # Handle missing cost data
        if np.isnan(progressive_costs).any() or np.isnan(reference_costs).any() or np.isnan(conservative_costs).any():
            missing_indices = np.where(np.isnan(progressive_costs) | np.isnan(reference_costs) | np.isnan(conservative_costs))
            print(f"Missing data at indices: {missing_indices}")
            print(f"Tech with missing data: {tech[missing_indices]}")
            print(f"Efficiencies with missing data: {eff[missing_indices]}")
            
            raise ValueError(f"Missing cost data for some technology and efficiency combinations in cost_component {cost_component}")

        # Calculate mean and standard deviation assuming the costs represent the 10th, 50th, and 90th percentiles of a normal distribution
        mean_costs = reference_costs
        std_costs = (conservative_costs - progressive_costs) / (norm.ppf(0.90) - norm.ppf(0.10))

        # Sample from the normal distribution for each row
        sampled_costs = np.random.normal(loc=mean_costs, scale=std_costs)
        sampled_costs_dict[cost_component] = sampled_costs

    # Calculate the retrofit cost for each row
    retrofit_cost = (
        sampled_costs_dict['normalized_cost'] * df_valid[params_col]
    ) * (df_valid['rsMeans_CCI_avg'] / rsMeans_national_avg)

    # Add the calculated costs to a new DataFrame, rounded to 2 decimal places
    df_new_columns = pd.DataFrame({retrofit_col: np.round(retrofit_cost, 2)}, index=df_valid.index)

    # Identify overlapping columns between the new and existing DataFrame
    overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)

    # Drop overlapping columns from the original DataFrame
    if not overlapping_columns.empty:
        df_copy.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into the original DataFrame, ensuring no duplicates or overwrites occur
    df_copy = df_copy.join(df_new_columns, how='left')

    return df_copy


# # Storing Output Results and Data Visualization

# ## Save Results: Merge DFs and Export to CSV

# In[ ]:


# If bldg_id is now the index in all DataFrames (df_compare, df_results_IRA, and df_results_IRA_gridDecarb):
def clean_df_merge(df_compare, df_results_IRA, df_results_IRA_gridDecarb):
    # Identify common columns (excluding 'bldg_id' which is the merging key)
    common_columns_IRA = set(df_compare.columns) & set(df_results_IRA.columns)
    common_columns_IRA.discard('bldg_id')
        
    # Drop duplicate columns in df_results_IRA and merge
    df_results_IRA = df_results_IRA.drop(columns=common_columns_IRA)
    print(f"""Dropped the following duplicate columns before merge: 
    {common_columns_IRA}
    """)
    # merged_df = pd.merge(df_compare, df_results_IRA, on='bldg_id', how='inner')
    merged_df = pd.merge(df_compare, df_results_IRA, how='inner', left_index=True, right_index=True)

    # Repeat the steps above for the merged_df and df_results_IRA_gridDecarb
    common_columnsIRA_gridDecarb = set(merged_df.columns) & set(df_results_IRA_gridDecarb.columns)
    common_columnsIRA_gridDecarb.discard('bldg_id')
    df_results_IRA_gridDecarb = df_results_IRA_gridDecarb.drop(columns=common_columnsIRA_gridDecarb)
    print(f"""Dropped the following duplicate columns before merge: 
    {common_columnsIRA_gridDecarb}
    """)
        
    # Create cleaned, merged results df with no duplicate columns
    # df_results_export = pd.merge(merged_df, df_results_IRA_gridDecarb, on='bldg_id', how='inner')
    df_results_export = pd.merge(merged_df, df_results_IRA_gridDecarb, how='inner', left_index=True, right_index=True)
    print("Dataframes have been cleaned of duplicate columns and merged successfully. Ready to export!")
    return df_results_export


# In[ ]:


def export_model_run_output(df_results_export, results_category, menu_mp):
    """
    Exports data for results summaries (npv, adoption, impact) and supplemental info (consumption, damages, fuel costs)

    Parameters:
    df_results_export (pd.DataFrame): DataFrame containing data for different scenarios.
    results_category (str): Determines the type of info being exported.
        - Accepted: 'summary', 'consumption', 'damages', 'fuelCost'
    menu_mp (int or str): Determines the measure package or retrofit being conducted
    
    """
    print("-------------------------------------------------------------------------------------------------------")
    # Baseline model run results
    if results_category == 'summary':
        if menu_mp == '0' or menu_mp==0:
            results_filename = f"baseline_results_{location_id}_{results_export_formatted_date}.csv"
            print(f"BASELINE RESULTS:")
            print(f"Dataframe results will be saved in this csv file: {results_filename}")

            # Change the directory to the upload folder and export the file
            results_change_directory = "baseline_summary"

        # Measure Package model run results
        else:
            if menu_mp == '8' or menu_mp==8:
                print(f"MEASURE PACKAGE {menu_mp} (MP{menu_mp}) RESULTS:")
                results_filename = f"mp{menu_mp}_results_{location_id}_{results_export_formatted_date}.csv"
                print(f"Dataframe results will be saved in this csv file: {results_filename}")

                # Change the directory to the upload folder and export the file
                results_change_directory = "retrofit_basic_summary"

            elif menu_mp == '9' or menu_mp==9:
                results_filename = f"mp{menu_mp}_results_{location_id}_{results_export_formatted_date}.csv"
                print(f"MEASURE PACKAGE {menu_mp} (MP{menu_mp}) RESULTS:")
                print(f"Dataframe results will be saved in this csv file: {results_filename}")

                # Change the directory to the upload folder and export the file
                results_change_directory = "retrofit_moderate_summary"

            elif menu_mp == '10' or menu_mp==10:
                results_filename = f"mp{menu_mp}_results_{location_id}_{results_export_formatted_date}.csv"
                print(f"MEASURE PACKAGE {menu_mp} (MP{menu_mp}) RESULTS:")
                print(f"Dataframe results will be saved in this csv file: {results_filename}")

                # Change the directory to the upload folder and export the file
                results_change_directory = "retrofit_advanced_summary"

            else:
                print("No matching scenarios for this Measure Package (MP)")

    # This includes exported dataframes for calculated consumption, damages, and fuel costs
    else:
        results_filename = f"mp{menu_mp}_data_{results_category}_{location_id}_{results_export_formatted_date}.csv"
        print(f"SUPPLEMENTAL INFORMATION DATAFRAME: {results_category}")
        print(f"Dataframe results will be saved in this csv file: {results_filename}")

        # Change the directory to the upload folder and export the file
        results_change_directory = f"supplemental_data_{results_category}"

    # Export dataframe results as a csv to the specified filepath
    results_export_filepath = os.path.join(output_folder_path, results_change_directory, results_filename)
    df_results_export.to_csv(results_export_filepath)
    print(f"Dataframe for MP{menu_mp} {results_category} results were exported here: {results_export_filepath}")
    print("-------------------------------------------------------------------------------------------------------", "\n")


# ## Convert Results Output CSVs to Dataframes

# In[30]:


def load_scenario_data(end_use, output_folder_path, scenario_string, model_run_date_time, columns_to_string):
    # Construct the output folder path with the policy_scenario of interest
    scenario_folder_path = os.path.join(output_folder_path, scenario_string)
    print(f"Output Results Folder Path: {scenario_folder_path}")

    # List all files in the specified folder with the specified date in the filename
    files = [f for f in os.listdir(scenario_folder_path) if os.path.isfile(os.path.join(scenario_folder_path, f)) and model_run_date_time in f]

    # Initialize dataframe as None
    df_outputs = None

    # Assume there is one main file per policy_scenario that includes all necessary data
    if files:
        file_path = os.path.join(scenario_folder_path, files[0])  # Assumes the first file is the correct one

        if os.path.exists(file_path):
            df_outputs = pd.read_csv(file_path, index_col=0, dtype=columns_to_string)
            print(f"Loaded {end_use} data for policy_scenario '{scenario_string}'", "\n")
        else:
            print("File not found for the specified policy_scenario", "\n")

    if df_outputs is None:
        print(f"No {end_use} data found for policy_scenario '{scenario_string}'")

    return df_outputs


# ## Visuals for Public and Private Perspective

# In[31]:


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


# In[ ]:


# LAST UPDATED DECEMBER 9, 2024
def subplot_grid_co2_abatement(dataframes, subplot_positions, epa_scc_values, x_cols, y_cols, hues, plot_titles=None, x_labels=None, y_labels=None, suptitle=None, figure_size=(12, 10), sharex=False, sharey=False):
    """
    Creates a grid of subplots to visualize CO2 abatement cost effectiveness across different datasets and scenarios.
    """
    num_cols = max(pos[1] for pos in subplot_positions) + 1
    num_rows = max(pos[0] for pos in subplot_positions) + 1

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figure_size, sharex=sharex, sharey=sharey)
    axes = np.array(axes).reshape(num_rows, num_cols)  # Ensure axes is always 2D

    for idx, (df, epa_scc, x_col, y_col, hue) in enumerate(zip(dataframes, epa_scc_values, x_cols, y_cols, hues)):
        pos = subplot_positions[idx]
        ax = axes[pos[0], pos[1]]
        title = plot_titles[idx] if plot_titles else ""
        x_label = x_labels[idx] if x_labels else ""
        y_label = y_labels[idx] if y_labels else ""

        # Plot using the plot_co2_abatement function, passing the current axis to it
        plot_co2_abatement(df, x_col, y_col, hue, epa_scc, ax=ax)

        # Set custom labels and title if provided
        ax.set_xlabel(x_label, fontweight='bold', fontsize=18)
        ax.set_ylabel(y_label, fontweight='bold', fontsize=18)
        ax.set_title(title, fontweight='bold', fontsize=18)

        # Set font size for tick labels on the x-axis
        ax.tick_params(axis='x', labelsize=18)

        # Set font size for tick labels on the y-axis
        ax.tick_params(axis='y', labelsize=18)

    if suptitle:
        plt.suptitle(suptitle, fontweight='bold')

    # Create a consolidated legend by grabbing handles and labels from all subplots
    handles, labels = [], []
    for ax in axes.flatten():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:  # Avoid duplicates
                handles.append(handle)
                labels.append(label)

    # # Add the consolidated legend outside the plots
    # fig.legend(handles, labels, loc='lower center', ncol=5, prop={'size': 18}, labelspacing=0.25, bbox_to_anchor=(0.5, -0.01))

    # # Adjust the layout
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to leave space for the suptitle

    # Add the consolidated legend outside the plots
    fig.legend(handles, labels, loc='lower center', ncol=5, prop={'size': 16}, labelspacing=0.25, handletextpad=1, columnspacing=1, bbox_to_anchor=(0.5, -0.05), bbox_transform=fig.transFigure)

    # Fine-tune the layout adjustment if needed
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjusted the rect to leave space for the suptitle and legend

    plt.show()

def plot_co2_abatement(df, x_col, y_col, hue, EPA_SCC_USD2023_PER_TON, ax=None):
    """
    Plots a boxplot of CO2 abatement cost effectiveness.

    Parameters:
    - df: DataFrame containing the data.
    - x_col: Column name for the x-axis.
    - y_col: Column name for the y-axis.
    - hue: Column name for the hue (categorical variable for color).
    - EPA_SCC_USD2023_PER_TON: Value for the red dashed line indicating SCC.
    - ax: Axis object to plot on. If None, creates a new plot.
    
    Returns:
    - None: Displays the plot.
    """
    # Filter out the 'Middle-to-Upper-Income' rows and create a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    df_filtered = df_copy[df_copy[x_col] != 'Middle-to-Upper-Income']

    # If x_col is categorical, remove unused categories
    if df_filtered[x_col].dtype.name == 'category':
        df_filtered.loc[:, x_col] = df_filtered[x_col].cat.remove_unused_categories()

    # Color map for fuel types
    color_map_fuel = {
        'Electricity': 'seagreen',
        'Natural Gas': 'steelblue',
        'Propane': 'orange',
        'Fuel Oil': 'firebrick',
    }

    if ax is None:
        ax = plt.gca()

    # Create the boxplot
    sns.boxplot(
        data=df_filtered,
        x=x_col, 
        y=y_col, 
        hue=hue, 
        palette=color_map_fuel, 
        showfliers=False,
        width=0.8,
        ax=ax
    )

    # Add a red dashed line at the value of EPA_SCC_USD2023_PER_TON
    ax.axhline(y=EPA_SCC_USD2023_PER_TON, color='red', linestyle='--', linewidth=2, label=f'SCC (USD2023): ${int(round((EPA_SCC_USD2023_PER_TON), 0))}/mtCO2e')

    # Remove the individual legend for each subplot
    ax.legend_.remove()


# # Adoption Rate Scenario Comparison

# In[ ]:


# LAST UPDATED DECEMBER 9, 2024
def create_df_adoption(df, menu_mp, category):
    """
    Generates a new DataFrame with specific adoption columns based on provided parameters.
    
    Args:
    df (pd.DataFrame): Original DataFrame.
    menu_mp (int): Measure package identifier.

    Returns:
    pd.DataFrame: A DataFrame with the selected columns.
    """    
    # Create a copy of the dataframe
    df_copy = df.copy()

    # Begin df with these cols
    df_copy['scc_usd2023_per_ton'] = np.round(EPA_SCC_USD2023_PER_TON, 2)

    summary_cols = ['state', 'city', 'county', 'puma', 'percent_AMI', 'lowModerateIncome_designation', 'scc_usd2023_per_ton']

    df_copy[f'iraRef_mp{menu_mp}_{category}_usd2023_per_mtCO2e_lrmer'] = round((df_copy[f'mp{menu_mp}_{category}_rebate_amount'] / df_copy[f'iraRef_mp{menu_mp}_{category}_avoided_tons_co2e_lrmer']), 2)
    df_copy[f'iraRef_mp{menu_mp}_{category}_usd2023_per_mtCO2e_srmer'] = round((df_copy[f'mp{menu_mp}_{category}_rebate_amount'] / df_copy[f'iraRef_mp{menu_mp}_{category}_avoided_tons_co2e_srmer']), 2)

    cols_to_add = [f'base_{category}_fuel',
                   f'preIRA_mp{menu_mp}_{category}_private_npv_lessWTP', # PRE-IRA PRIVATE
                   f'preIRA_mp{menu_mp}_{category}_total_capitalCost', 
                   f'preIRA_mp{menu_mp}_{category}_private_npv_moreWTP', 
                   f'preIRA_mp{menu_mp}_{category}_net_capitalCost',
                   f'preIRA_mp{menu_mp}_{category}_avoided_tons_co2e_lrmer', # LRMER
                   f'preIRA_mp{menu_mp}_{category}_public_npv_lrmer',
                   f'preIRA_mp{menu_mp}_{category}_adoption_lrmer',
                   f'preIRA_mp{menu_mp}_{category}_avoided_tons_co2e_srmer', # SRMER
                   f'preIRA_mp{menu_mp}_{category}_public_npv_srmer',
                   f'preIRA_mp{menu_mp}_{category}_adoption_srmer',
                   f'mp{menu_mp}_{category}_rebate_amount', # IRA-REFERENCE PRIVATE
                   f'iraRef_mp{menu_mp}_{category}_private_npv_lessWTP', 
                   f'iraRef_mp{menu_mp}_{category}_total_capitalCost', 
                   f'iraRef_mp{menu_mp}_{category}_private_npv_moreWTP', 
                   f'iraRef_mp{menu_mp}_{category}_net_capitalCost',
                   f'iraRef_mp{menu_mp}_{category}_avoided_tons_co2e_lrmer', # LRMER
                   f'iraRef_mp{menu_mp}_{category}_usd2023_per_mtCO2e_lrmer',
                   f'iraRef_mp{menu_mp}_{category}_public_npv_lrmer',
                   f'iraRef_mp{menu_mp}_{category}_additional_public_benefit_lrmer',
                   f'iraRef_mp{menu_mp}_{category}_adoption_lrmer',
                   f'iraRef_mp{menu_mp}_{category}_avoided_tons_co2e_srmer', # SRMER
                   f'iraRef_mp{menu_mp}_{category}_usd2023_per_mtCO2e_srmer',
                   f'iraRef_mp{menu_mp}_{category}_public_npv_srmer',
                   f'iraRef_mp{menu_mp}_{category}_additional_public_benefit_srmer',
                   f'iraRef_mp{menu_mp}_{category}_adoption_srmer',
                   ]
            
    # Use extend instead of append to add each element of cols_to_add to summary_cols
    summary_cols.extend(cols_to_add)

    # Select the relevant columns
    df_copy = df_copy[summary_cols]

    return df_copy


# In[ ]:


# UPDATED SEPTEMBER 20, 2024 @ 1:30 AM
import pandas as pd

def filter_columns(df):
    keep_columns = [col for col in df.columns if 'Tier 1: Feasible' in col[1] or 
                    'Tier 2: Feasible vs. Alternative' in col[1] or 
                    'Tier 3: Subsidy-Dependent Feasibility' in col[1] or 
                    'Total Adoption Potential' in col[1] or 
                    'Total Adoption Potential (Additional Subsidy)' in col[1]]    
    
    return df.loc[:, keep_columns]

def create_multiIndex_adoption_df(df, menu_mp, category, mer_type):
    # Explicitly set 'lowModerateIncome_designation' as a categorical type with order
    income_categories = ['Low-Income', 'Moderate-Income', 'Middle-to-Upper-Income']

    df['lowModerateIncome_designation'] = pd.Categorical(df['lowModerateIncome_designation'], categories=income_categories, ordered=True)
    
    # Define the columns for adoption data
    adoption_cols = [f'preIRA_mp{menu_mp}_{category}_adoption_{mer_type}', 
                     f'iraRef_mp{menu_mp}_{category}_adoption_{mer_type}']

    # Group by f'base_{category}_fuel' and 'lowModerateIncome_designation', calculate normalized counts
    percentages_df = df.groupby([f'base_{category}_fuel', 'lowModerateIncome_designation'], observed=False)[adoption_cols].apply(
        lambda x: x.apply(lambda y: y.value_counts(normalize=True))).unstack().fillna(0) * 100
    percentages_df = percentages_df.round(0)

    # Ensure 'Tier 1: Feasible' columns exist, set to 0 if they don't
    for column in adoption_cols:
        if (column, 'Tier 1: Feasible') not in percentages_df.columns:
            percentages_df[(column, 'Tier 1: Feasible')] = 0
        if (column, 'Tier 2: Feasible vs. Alternative') not in percentages_df.columns:
            percentages_df[(column, 'Tier 2: Feasible vs. Alternative')] = 0
        if (column, 'Tier 3: Subsidy-Dependent Feasibility') not in percentages_df.columns:
            percentages_df[(column, 'Tier 3: Subsidy-Dependent Feasibility')] = 0

        percentages_df[(column, 'Total Adoption Potential')] = (
            percentages_df[(column, 'Tier 1: Feasible')] + 
            percentages_df[(column, 'Tier 2: Feasible vs. Alternative')]
        )

        percentages_df[(column, 'Total Adoption Potential (Additional Subsidy)')] = (
            percentages_df[(column, 'Tier 1: Feasible')] + 
            percentages_df[(column, 'Tier 2: Feasible vs. Alternative')] + 
            percentages_df[(column, 'Tier 3: Subsidy-Dependent Feasibility')]
        )

    # Rebuild the column MultiIndex
    percentages_df.columns = pd.MultiIndex.from_tuples(percentages_df.columns)
    
    # Filter DataFrame to keep relevant columns only
    filtered_df = filter_columns(percentages_df)

    new_order = []
    for prefix in ['preIRA_mp', 'iraRef_mp']:
        for suffix in ['Tier 1: Feasible', 'Tier 2: Feasible vs. Alternative', 'Tier 3: Subsidy-Dependent Feasibility', 'Total Adoption Potential', 'Total Adoption Potential (Additional Subsidy)']:
            col = (f'{prefix}{menu_mp}_{category}_adoption_{mer_type}', suffix)
            if col in filtered_df.columns:
                new_order.append(col)

    # Check if new_order is empty before reordering columns
    if new_order:
        # Reorder columns based on new_order
        filtered_df = filtered_df.loc[:, pd.MultiIndex.from_tuples(new_order)]
                    
        # Sort DataFrame by the entire index
        filtered_df.sort_index(level=[f'base_{category}_fuel', 'lowModerateIncome_designation'], inplace=True)
    else:
        print("Warning: No matching columns found for reordering")

    return filtered_df

# Usage example (assuming df_basic_adoption_heating is properly formatted and loaded):
# df_multiIndex_heating_adoption = create_multiIndex_adoption_df(df_basic_adoption_heating, 8, 'heating')
# df_multiIndex_heating_adoption


# In[ ]:


# import pandas as pd

# def filter_columns(df):
#     keep_columns = [col for col in df.columns if 'Tier 1: Feasible' in col[1] or 'Tier 2: Feasible vs. Alternative' in col[1] or 'Tier 2: Feasible vs. Alternative' in col[1] or 'Tier 3: Subsidy-Dependent Feasibility' in col[1]]
#     return df.loc[:, keep_columns]

# def create_multiIndex_adoption_df(df, menu_mp, category):
#     # Explicitly set 'lowModerateIncome_designation' as a categorical type with order
#     income_categories = ['Low-Income', 'Moderate-Income', 'Middle-to-Upper-Income']

#     df['lowModerateIncome_designation'] = pd.Categorical(df['lowModerateIncome_designation'], categories=income_categories, ordered=True)
    
#     # Define the columns for adoption data
#     adoption_cols = [f'preIRA_mp{menu_mp}_{category}_adoption', 
#                      f'iraRef_mp{menu_mp}_{category}_adoption']

#     # Group by f'base_{category}_fuel' and 'lowModerateIncome_designation', calculate normalized counts
#     percentages_df = df.groupby([f'base_{category}_fuel', 'lowModerateIncome_designation'], observed=False)[adoption_cols].apply(
#         lambda x: x.apply(lambda y: y.value_counts(normalize=True))).unstack().fillna(0) * 100
#     percentages_df = percentages_df.round(2)

#     # Ensure 'Tier 1: Feasible' columns exist, set to 0 if they don't
#     for column in adoption_cols:
#         if (column, 'Tier 1: Feasible') not in percentages_df.columns:
#             percentages_df[(column, 'Tier 1: Feasible')] = 0
#         if (column, 'Tier 2: Feasible vs. Alternative') not in percentages_df.columns:
#             percentages_df[(column, 'Tier 2: Feasible vs. Alternative')] = 0
#         if (column, 'Tier 3: Subsidy-Dependent Feasibility') not in percentages_df.columns:
#             percentages_df[(column, 'Tier 3: Subsidy-Dependent Feasibility')] = 0


#     # Create 'Total Adoption with Subsidy' by combining related columns
#     for column in adoption_cols:
#         percentages_df[(column, 'Total Adoption with Subsidy')] = percentages_df[(column, 'Tier 1: Feasible')] + percentages_df.get((column, 'Tier 2: Feasible vs. Alternative'), 0) + percentages_df.get((column, 'Tier 3: Subsidy-Dependent Feasibility'), 0)

#     # Rebuild the column MultiIndex
#     percentages_df.columns = pd.MultiIndex.from_tuples(percentages_df.columns)
    
#     # Filter DataFrame to keep relevant columns only
#     filtered_df = filter_columns(percentages_df)

#     # Dynamically build the new column order based on existing columns
#     new_order = []
#     for prefix in ['preIRA_mp', 'iraRef_mp']:
#         for suffix in ['Tier 1: Feasible', 'Tier 2: Feasible vs. Alternative', 'Tier 3: Subsidy-Dependent Feasibility', 'Total Adoption with Subsidy']:
#             col = (f'{prefix}{menu_mp}_{category}_adoption', suffix)
#             if col in filtered_df.columns:
#                 new_order.append(col)

#     # Check if new_order is empty before reordering columns
#     if new_order:
#         # Reorder columns based on new_order
#         filtered_df = filtered_df.loc[:, pd.MultiIndex.from_tuples(new_order)]
                    
#         # Sort DataFrame by the entire index
#         filtered_df.sort_index(level=[f'base_{category}_fuel', 'lowModerateIncome_designation'], inplace=True)
#     else:
#         print("Warning: No matching columns found for reordering")

#     return filtered_df

# # Usage example (assuming df_basic_adoption_heating is properly formatted and loaded):
# # df_multiIndex_heating_adoption = create_multiIndex_adoption_df(df_basic_adoption_heating, 8, 'heating')
# # df_multiIndex_heating_adoption


# In[ ]:


# # LAST UPDATED SEPTEMBER 6, 2024

# import matplotlib.pyplot as plt
# import numpy as np

# def subplot_grid_adoption_vBar(dataframes, scenarios_list, subplot_positions, filter_fuel=None, x_labels=None, plot_titles=None, y_labels=None, suptitle=None, figure_size=(12, 10), sharex=False, sharey=False):
#     """
#     Creates a grid of subplots to visualize adoption rates across different scenarios, with an option to plot specific data related to adoption.
#     """
#     num_subplots = len(subplot_positions)
#     num_cols = max(pos[1] for pos in subplot_positions) + 1
#     num_rows = max(pos[0] for pos in subplot_positions) + 1

#     fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figure_size, sharex=sharex, sharey=sharey)
#     axes = np.array(axes).reshape(num_rows, num_cols)  # Ensure axes is always 2D

#     for idx, (df, scenarios) in enumerate(zip(dataframes, scenarios_list)):
#         # Apply the filter_fuel if provided
#         if filter_fuel:
#             df = df.loc[(df.index.get_level_values('base_fuel').isin(filter_fuel)), :]
        
#         pos = subplot_positions[idx]
#         ax = axes[pos[0], pos[1]]
#         x_label = x_labels[idx] if x_labels else ""
#         y_label = y_labels[idx] if y_labels else ""
#         title = plot_titles[idx] if plot_titles else ""

#         plot_adoption_rate_bar(df, scenarios, title, x_label, y_label, ax)

#     if suptitle:
#         plt.suptitle(suptitle, fontweight='bold')

#     # Define the relevant tiers to display in the legend
#     relevant_tiers = [
#         'Tier 1: Feasible',
#         'Tier 2: Feasible vs. Alternative',
#         'Tier 3: Subsidy-Dependent Feasibility'
#     ]

#     # Add a legend for only the relevant tiers
#     legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping[label]) for label in relevant_tiers]
#     fig.legend(legend_handles, relevant_tiers, loc='lower center', ncol=len(relevant_tiers), prop={'size': 20}, labelspacing=0.5, bbox_to_anchor=(0.5, -0.05))

#     # Adjust the layout
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to leave space for the suptitle
#     plt.show()

# def plot_adoption_rate_bar(df, scenarios, title, x_label, y_label, ax):
#     # Assume the DataFrame 'df' has a suitable structure, similar to earlier examples
#     adoption_data = df.loc[:, df.columns.get_level_values(1).isin(['Tier 1: Feasible', 'Tier 2: Feasible vs. Alternative', 'Tier 3: Subsidy-Dependent Feasibility'])]
#     adoption_data.columns = adoption_data.columns.remove_unused_levels()

#     # Define the color mapping as specified
#     global color_mapping
#     color_mapping = {
#         'Tier 1: Feasible': 'steelblue',
#         'Tier 2: Feasible vs. Alternative': 'lightblue',
#         'Tier 3: Subsidy-Dependent Feasibility': 'lightsalmon'
#     }

#     # Plotting logic
#     n = len(adoption_data.index)
#     bar_width = 0.35  # Width of bars
#     index = list(range(n))  # Base index for bars

#     for scenario in scenarios:
#         if (scenario, 'Tier 1: Feasible') in adoption_data.columns and (scenario, 'Tier 2: Feasible vs. Alternative') in adoption_data.columns and (scenario, 'Tier 3: Subsidy-Dependent Feasibility') in adoption_data.columns:
#             tier3 = adoption_data[scenario, 'Tier 3: Subsidy-Dependent Feasibility'].values
#             tier2 = adoption_data[scenario, 'Tier 2: Feasible vs. Alternative'].values
#             tier1 = adoption_data[scenario, 'Tier 1: Feasible'].values
#             ax.bar(index, tier3, bar_width, color=color_mapping['Tier 3: Subsidy-Dependent Feasibility'], edgecolor='white')
#             ax.bar(index, tier2, bar_width, color=color_mapping['Tier 2: Feasible vs. Alternative'], edgecolor='white')
#             ax.bar(index, tier1, bar_width, color=color_mapping['Tier 1: Feasible'], edgecolor='white')
#             index = [i + bar_width for i in index]

#     ax.set_xlabel(x_label, fontweight='bold', fontsize=20)
#     ax.set_ylabel(y_label, fontweight='bold', fontsize=20)
#     ax.set_title(title, fontweight='bold', fontsize=20)
#     ax.set_xticks([i + bar_width / 2 for i in range(n)])
#     ax.set_xticklabels([f'{name[1]}' for name in adoption_data.index.tolist()], rotation=90, ha='right')

#     # Set font size for tick labels on the x-axis
#     ax.tick_params(axis='x', labelsize=20)

#     # Set font size for tick labels on the y-axis
#     ax.tick_params(axis='y', labelsize=20)


# In[ ]:


# # UPDATED SEPTEMBER 14, 2024 @ 12:46 AM
# def subplot_grid_adoption_vBar(dataframes, scenarios_list, subplot_positions, filter_fuel=None, x_labels=None, plot_titles=None, y_labels=None, suptitle=None, figure_size=(12, 10), sharex=False, sharey=False):
#     """
#     Creates a grid of subplots to visualize adoption rates across different scenarios, with an option to plot specific data related to adoption.

#     Parameters:
#     - dataframes (list of pd.DataFrame): List of pandas DataFrames, each DataFrame is assumed to be formatted for use in plot_adoption_rate_bar.
#     - scenarios_list (list of list): List of scenarios corresponding to each DataFrame.
#     - subplot_positions (list of tuples): Positions of subplots in the grid, specified as (row, col) tuples.
#     - filter_fuel (list of str, optional): List of fuel types to filter the DataFrames by 'base_fuel' column in a multi-index.
#     - x_labels (list of str, optional): Labels for the x-axis of each subplot.
#     - plot_titles (list of str, optional): Titles for each subplot.
#     - y_labels (list of str, optional): Labels for the y-axis of each subplot.
#     - suptitle (str, optional): A central title for the entire figure.
#     - figure_size (tuple, optional): Size of the entire figure (width, height) in inches.
#     - sharex (bool, optional): Whether subplots should share the same x-axis.
#     - sharey (bool, optional): Whether subplots should share the same y-axis.

#     Returns:
#     None. Displays the figure based on the provided parameters.
#     """
#     # Define the color mapping as specified
#     color_mapping = {
#         'Tier 1: Feasible': 'steelblue',
#         'Tier 2: Feasible vs. Alternative': 'lightblue',
#         'Tier 3: Subsidy-Dependent Feasibility': 'lightsalmon'
#     }

#     num_cols = max(pos[1] for pos in subplot_positions) + 1
#     num_rows = max(pos[0] for pos in subplot_positions) + 1

#     fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figure_size, sharex=sharex, sharey=sharey)
#     axes = np.array(axes).reshape(num_rows, num_cols)  # Ensure axes is always 2D

#     for idx, (df, scenarios) in enumerate(zip(dataframes, scenarios_list)):
#         # Apply the filter_fuel if provided
#         if filter_fuel:
#             df = df.loc[(df.index.get_level_values('base_fuel').isin(filter_fuel)), :]
        
#         pos = subplot_positions[idx]
#         ax = axes[pos[0], pos[1]]
#         x_label = x_labels[idx] if x_labels else ""
#         y_label = y_labels[idx] if y_labels else ""
#         title = plot_titles[idx] if plot_titles else ""

#         plot_adoption_rate_bar(df, scenarios, title, x_label, y_label, ax)

#     if suptitle:
#         plt.suptitle(suptitle, fontweight='bold')

#     # Add a legend for the color mapping at the bottom of the entire figure
#     legend_labels = list(color_mapping.keys())
#     legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping[label]) for label in legend_labels]
            
#     fig.legend(legend_handles, legend_labels, loc='lower center', ncol=len(legend_labels), prop={'size': 20}, labelspacing=0.5, bbox_to_anchor=(0.5, -0.05))

#     # Adjust the layout
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to leave space for the suptitle
#     plt.show()

# def plot_adoption_rate_bar(df, scenarios, title, x_label, y_label, ax):
#     # Assume the DataFrame 'df' has a suitable structure, similar to earlier examples
#     adoption_data = df.loc[:, df.columns.get_level_values(1).isin(['Tier 1: Feasible', 'Tier 2: Feasible vs. Alternative', 'Tier 3: Subsidy-Dependent Feasibility'])]
#     adoption_data.columns = adoption_data.columns.remove_unused_levels()

#     # Define the color mapping as specified
#     global color_mapping
#     color_mapping = {
#         'Tier 1: Feasible': 'steelblue',
#         'Tier 2: Feasible vs. Alternative': 'lightblue',
#         'Tier 3: Subsidy-Dependent Feasibility': 'lightsalmon'
#     }

#     # Plotting logic
#     n = len(adoption_data.index)
#     bar_width = 0.35  # Width of bars
#     index = list(range(n))  # Base index for bars

#     for i, scenario in enumerate(scenarios):
#         if (scenario, 'Tier 1: Feasible') in adoption_data.columns and (scenario, 'Tier 2: Feasible vs. Alternative') in adoption_data.columns and (scenario, 'Tier 3: Subsidy-Dependent Feasibility') in adoption_data.columns:
#             tier1 = adoption_data[scenario, 'Tier 1: Feasible'].values
#             tier2 = adoption_data[scenario, 'Tier 2: Feasible vs. Alternative'].values
#             tier3 = adoption_data[scenario, 'Tier 3: Subsidy-Dependent Feasibility'].values

#             # Adjust the index for this scenario
#             scenario_index = np.array(index) + i * bar_width
            
#             # Plot the bars for the scenario
#             ax.bar(scenario_index, tier1, bar_width, color=color_mapping['Tier 1: Feasible'], edgecolor='white')
#             ax.bar(scenario_index, tier2, bar_width, bottom=tier1, color=color_mapping['Tier 2: Feasible vs. Alternative'], edgecolor='white')
#             ax.bar(scenario_index, tier3, bar_width, bottom=(tier1+tier2), color=color_mapping['Tier 3: Subsidy-Dependent Feasibility'], edgecolor='white')


#     ax.set_xlabel(x_label, fontweight='bold', fontsize=20)
#     ax.set_ylabel(y_label, fontweight='bold', fontsize=20)
#     ax.set_title(title, fontweight='bold', fontsize=20)
    
#     ax.set_xticks([i + bar_width / 2 for i in range(n)])
#     ax.set_xticklabels([f'{name[1]}' for name in adoption_data.index.tolist()], rotation=90, ha='right')

#     # Set font size for tick labels on the x-axis
#     ax.tick_params(axis='x', labelsize=20)

#     # Set font size for tick labels on the y-axis
#     ax.tick_params(axis='y', labelsize=20)

#     # Set y-ticks from 0 to 100 in steps of 10%
#     ax.set_yticks(np.arange(0, 101, 10))
#     ax.set_ylim(0, 100)


# In[ ]:


# UPDATED SEPTEMBER 14, 2024 @ 12:46 AM
def subplot_grid_adoption_vBar(dataframes, scenarios_list, subplot_positions, filter_fuel=None, x_labels=None, plot_titles=None, y_labels=None, suptitle=None, figure_size=(12, 10), sharex=False, sharey=False):
    """
    Creates a grid of subplots to visualize adoption rates across different scenarios, with an option to plot specific data related to adoption.

    Parameters:
    - dataframes (list of pd.DataFrame): List of pandas DataFrames, each DataFrame is assumed to be formatted for use in plot_adoption_rate_bar.
    - scenarios_list (list of list): List of scenarios corresponding to each DataFrame.
    - subplot_positions (list of tuples): Positions of subplots in the grid, specified as (row, col) tuples.
    - filter_fuel (list of str, optional): List of fuel types to filter the DataFrames by 'base_fuel' column in a multi-index.
    - x_labels (list of str, optional): Labels for the x-axis of each subplot.
    - plot_titles (list of str, optional): Titles for each subplot.
    - y_labels (list of str, optional): Labels for the y-axis of each subplot.
    - suptitle (str, optional): A central title for the entire figure.
    - figure_size (tuple, optional): Size of the entire figure (width, height) in inches.
    - sharex (bool, optional): Whether subplots should share the same x-axis.
    - sharey (bool, optional): Whether subplots should share the same y-axis.

    Returns:
    None. Displays the figure based on the provided parameters.
    """
    # Define the color mapping as specified
    color_mapping = {
        'Tier 1: Feasible': 'steelblue',
        'Tier 2: Feasible vs. Alternative': 'lightblue',
        'Tier 3: Subsidy-Dependent Feasibility': 'lightsalmon'
    }

    num_cols = max(pos[1] for pos in subplot_positions) + 1
    num_rows = max(pos[0] for pos in subplot_positions) + 1

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figure_size, sharex=sharex, sharey=sharey)
    axes = np.array(axes).reshape(num_rows, num_cols)  # Ensure axes is always 2D

    for idx, (df, scenarios) in enumerate(zip(dataframes, scenarios_list)):
        # Apply the filter_fuel if provided
        if filter_fuel:
            df = df.loc[(df.index.get_level_values('base_fuel').isin(filter_fuel)), :]
        
        pos = subplot_positions[idx]
        ax = axes[pos[0], pos[1]]
        x_label = x_labels[idx] if x_labels else ""
        y_label = y_labels[idx] if y_labels else ""
        title = plot_titles[idx] if plot_titles else ""

        plot_adoption_rate_bar(df, scenarios, title, x_label, y_label, ax)

    if suptitle:
        plt.suptitle(suptitle, fontweight='bold')

    # Add a legend for the color mapping at the bottom of the entire figure
    legend_labels = list(color_mapping.keys())
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping[label]) for label in legend_labels]
            
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=len(legend_labels), prop={'size': 20}, labelspacing=0.5, bbox_to_anchor=(0.5, -0.05))

    # Adjust the layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to leave space for the suptitle
    plt.show()

def plot_adoption_rate_bar(df, scenarios, title, x_label, y_label, ax):
    # Assume the DataFrame 'df' has a suitable structure, similar to earlier examples
    adoption_data = df.loc[:, df.columns.get_level_values(1).isin(['Tier 1: Feasible', 'Tier 2: Feasible vs. Alternative', 'Tier 3: Subsidy-Dependent Feasibility'])]
    adoption_data.columns = adoption_data.columns.remove_unused_levels()

    # Define the color mapping as specified
    global color_mapping
    color_mapping = {
        'Tier 1: Feasible': 'steelblue',
        'Tier 2: Feasible vs. Alternative': 'lightblue',
        'Tier 3: Subsidy-Dependent Feasibility': 'lightsalmon'
    }

    # Plotting logic
    n = len(adoption_data.index)
    bar_width = 0.35  # Width of bars
    index = list(range(n))  # Base index for bars

    for i, scenario in enumerate(scenarios):
        if (scenario, 'Tier 1: Feasible') in adoption_data.columns and (scenario, 'Tier 2: Feasible vs. Alternative') in adoption_data.columns and (scenario, 'Tier 3: Subsidy-Dependent Feasibility') in adoption_data.columns:
            tier1 = adoption_data[scenario, 'Tier 1: Feasible'].values
            tier2 = adoption_data[scenario, 'Tier 2: Feasible vs. Alternative'].values
            tier3 = adoption_data[scenario, 'Tier 3: Subsidy-Dependent Feasibility'].values

            # Adjust the index for this scenario
            scenario_index = np.array(index) + i * bar_width
            
            # Plot the bars for the scenario
            ax.bar(scenario_index, tier1, bar_width, color=color_mapping['Tier 1: Feasible'], edgecolor='white')
            ax.bar(scenario_index, tier2, bar_width, bottom=tier1, color=color_mapping['Tier 2: Feasible vs. Alternative'], edgecolor='white')
            ax.bar(scenario_index, tier3, bar_width, bottom=(tier1+tier2), color=color_mapping['Tier 3: Subsidy-Dependent Feasibility'], edgecolor='white')


    ax.set_xlabel(x_label, fontweight='bold', fontsize=20)
    ax.set_ylabel(y_label, fontweight='bold', fontsize=20)
    ax.set_title(title, fontweight='bold', fontsize=20)
    
    ax.set_xticks([i + bar_width / 2 for i in range(n)])
    ax.set_xticklabels([f'{name[1]}' for name in adoption_data.index.tolist()], rotation=90, ha='right')

    # Set font size for tick labels on the x-axis
    ax.tick_params(axis='x', labelsize=20)

    # Set font size for tick labels on the y-axis
    ax.tick_params(axis='y', labelsize=20)

    # Set y-ticks from 0 to 100 in steps of 10%
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylim(0, 100)


# # Adoption Rate Percentages

# In[34]:


# UPDATED ON AUGUST 23, 2024 @ 2:00 AM
def format_group_percentages(counts, group):
    # Initialize total adoption with subsidy to 0
    total_adoption_with_subsidy = 0
    
    # Check and sum 'Tier 1: Feasible' and 'Tier 2: Feasible vs. Alternative' if they exist
    if 'Tier 1: Feasible' in counts.columns:
        total_adoption_with_subsidy += counts.loc[group, 'Tier 1: Feasible']
    if 'Tier 2: Feasible vs. Alternative' in counts.columns:
        total_adoption_with_subsidy += counts.loc[group, 'Tier 2: Feasible vs. Alternative']
    if 'Tier 3: Subsidy-Dependent Feasibility' in counts.columns:
        total_adoption_with_subsidy += counts.loc[group, 'Tier 3: Subsidy-Dependent Feasibility']

    # Format percentages, including checks for existence before accessing
    formatted_percentages = ', '.join(f"{decision_prefix}{counts.loc[group, decision]:.1f}%" 
                                      for decision, decision_prefix in [('Tier 1: Feasible', 'T1 '), ('Tier 2: Feasible vs. Alternative', 'T2 '),('Tier 3: Subsidy-Dependent Feasibility', 'T3 ')]
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
        total_adoption_with_subsidy = overall_counts.get('Tier 1: Feasible', 0) + overall_counts.get('Tier 2: Feasible vs. Alternative', 0) + overall_counts.get('Tier 3: Subsidy-Dependent Feasibility', 0)

        overall_percentages = ', '.join(f"{decision_prefix}{overall_counts[decision]:.1f}%" 
                                        for decision, decision_prefix in [('Tier 1: Feasible', 'T1 '), ('Tier 2: Feasible vs. Alternative', 'T2 '),('Tier 3: Subsidy-Dependent Feasibility', 'T3 ')]
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

