import pandas as pd
import os

# Get the current working directory of the project
# project_root = os.path.abspath(os.getcwd())
project_root = "C:\\Users\\14128\\Research\\cmu-tare-model"
print(f"Project root directory: {project_root}")

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
USER INPUT FOR GEOGRAPHIC FILTERS
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

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

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
LOAD EUSS/RESSTOCK DATA AND APPLY FILTERS
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

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

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
PROJECT FUTURE ENERGY CONSUMPTION
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
- HDD factors for different census divisions and years
- Functions to project future energy consumption based on HDD projections
"""

# HDD factors for different census divisions and years
# Factors for 2022 to 2050
filename = 'aeo_projections_2022_2050.xlsx'
relative_path = os.path.join(r"projections", filename)
file_path = os.path.join(project_root, relative_path)
df_hdd_projection_factors = pd.read_excel(io=file_path, sheet_name='hdd_factors_2022_2050')

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")

# Convert the factors dataframe into a lookup dictionary
hdd_factor_lookup = df_hdd_projection_factors.set_index(['census_division']).to_dict('index')
hdd_factor_lookup

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