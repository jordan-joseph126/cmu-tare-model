import pandas as pd
import numpy as np
import re

from config import PROJECT_ROOT

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
    """Prompts the user with a menu and returns a validated choice.

    This function loops until the user inputs a valid option
    that exists in the provided choices.

    Args:
        prompt (str): The message to display to the user.
        choices (list[str]): A list of valid choices (e.g., ['N', 'Y']).

    Returns:
        str: The validated user input.

    Raises:
        None: This function does not raise any exceptions.
    """
    while True:
        choice = input(prompt).upper()
        if choice in choices:
            return choice
        print("Invalid option. Please try again.")


def get_state_choice(df_copy):
    """Prompts the user to input a two-letter state abbreviation and validates it.

    This function repeats until the user provides a valid state abbreviation
    that exists within the 'in.state' column of the given DataFrame.

    Args:
        df_copy (pd.DataFrame): A pandas DataFrame that includes an 'in.state' column.

    Returns:
        str: The two-letter abbreviation of the state.

    Raises:
        None: This function does not raise any exceptions.
    """
    while True:
        input_state = input("Which state would you like to analyze data for? Please enter the two-letter abbreviation: ").upper()
        if df_copy['in.state'].eq(input_state).any():
            return input_state
        print("Invalid state abbreviation. Please try again.")


def get_city_choice(df_copy, input_state):
    """Prompts the user to input a city name within a given state and validates it.

    This function repeats until the user provides a valid city name
    that exists in the 'in.city' column of the DataFrame, matching
    the specified state.

    Args:
        df_copy (pd.DataFrame): A pandas DataFrame that includes an 'in.city' column.
        input_state (str): The two-letter abbreviation of the selected state.

    Returns:
        str: The validated city name.

    Raises:
        None: This function does not raise any exceptions.
    """
    while True:
        input_cityFilter = input("Please enter the city name ONLY (e.g., Pittsburgh): ")
        # city_filter checks for an exact match of state and city in the format 'ST, CityName'
        city_filter = df_copy['in.city'].eq(f"{input_state}, {input_cityFilter}")
        if city_filter.any():
            return input_cityFilter
        print("Invalid city name. Please try again.")


"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
LOAD EUSS/RESSTOCK DATA AND APPLY FILTERS
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def standardize_fuel_name(fuel_desc):
    """Standardizes a fuel description into a recognized category or None.

    This function inspects an input fuel description (e.g., "Electric Heater",
    "Gas Furnace", "Propane Heater") and maps it to one of the following strings:
    "Electricity", "Natural Gas", "Propane", or "Fuel Oil". If the input is NaN,
    not a string, or does not contain any recognizable fuel keyword, the function
    returns None.

    Args:
        fuel_desc (Any): A value representing the fuel description. It can be a string
            containing words like "Electric," "Gas," "Propane," or "Oil." It may
            also be NaN (pandas missing value) or another data type.

    Returns:
        str or None: One of the strings {"Electricity", "Natural Gas", "Propane", "Fuel Oil"}
            if a match is found, or None otherwise.

    Raises:
        None: This function does not raise any exceptions.
    """
    # Check if fuel_desc is NaN or not a string; return None if so
    if pd.isna(fuel_desc) or not isinstance(fuel_desc, str):
        return None

    # Convert the string to uppercase for case-insensitive matching
    fuel_desc_upper = fuel_desc.upper()

    # Match substrings for known fuel types
    if 'ELECTRIC' in fuel_desc_upper:
        return 'Electricity'
    elif 'GAS' in fuel_desc_upper:
        return 'Natural Gas'
    elif 'PROPANE' in fuel_desc_upper:
        return 'Propane'
    elif 'OIL' in fuel_desc_upper:
        return 'Fuel Oil'
    else:
        # If no match is found, return None
        return None


def preprocess_fuel_data(df, column_name):
    """Applies a standardization process to the specified fuel column in the DataFrame.

    This function applies 'standardize_fuel_name' to every value in the specified column
    and updates the DataFrame in-place.

    Args:
        df (pd.DataFrame): The input pandas DataFrame containing fuel data.
        column_name (str): The name of the column to standardize.

    Returns:
        pd.DataFrame: The updated DataFrame with standardized fuel names in the specified column.

    Raises:
        None: This function does not raise any exceptions.
    """
    print(f"Processing column: {column_name}")
    print(f"Initial data types: {df[column_name].dtype}")

    # Use .loc to avoid SettingWithCopyWarning when applying the function
    df.loc[:, column_name] = df[column_name].apply(standardize_fuel_name)

    print(f"Data types after processing: {df[column_name].dtype}")
    return df

def apply_fuel_filter(df, category, enable, mode='filter'):
    """
    Filters or masks the DataFrame rows based on allowed fuels for a given category.
    
    Args:
        df (pd.DataFrame): The DataFrame to modify.
        category (str): The end-use category (e.g., 'waterHeating').
        enable (str): 'Yes' to apply the filter.
        mode (str): 'filter' to drop invalid rows or 'mask' to assign NaN.
    
    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    if enable != 'Yes':
        return df

    allowed_fuel = ['Natural Gas', 'Electricity', 'Propane', 'Fuel Oil']
    fuel_col = f'base_{category}_fuel'
    valid_mask = df[fuel_col].isin(allowed_fuel)
    
    if mode == 'filter':
        df = df[valid_mask]
        print(f"Filtered for allowed fuels in {category}: {allowed_fuel}")
    elif mode == 'mask':
        # Determine consumption columns based on category.
        fuel_types = ['electricity', 'fuelOil', 'naturalGas', 'propane'] if category in ['heating', 'waterHeating'] else ['electricity', 'naturalGas', 'propane']
        for fuel in fuel_types:
            cons_col = f'base_{fuel}_{category}_consumption'
            df.loc[~valid_mask, cons_col] = np.nan
        print(f"Applied NaN to consumption columns in {category} for invalid fuels")
    else:
        raise ValueError("Mode must be 'filter' or 'mask'")
    
    return df

def apply_technology_filter(df, category, enable, mode='filter'):
    """
    Filters or masks the DataFrame rows based on allowed technology types for a given category.
    
    Args:
        df (pd.DataFrame): The DataFrame to modify.
        category (str): The consumption category (e.g., 'waterHeating').
        enable (str): 'Yes' to apply the filter.
        mode (str): 'filter' to drop invalid rows or 'mask' to assign NaN.
    
    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    if enable != 'Yes' or category not in ['heating', 'waterHeating']:
        return df

    if category == 'heating':
        allowed_tech = [
            'Electricity ASHP', 'Electricity Baseboard', 'Electricity Electric Boiler', 'Electricity Electric Furnace',
            'Fuel Oil Fuel Boiler', 'Fuel Oil Fuel Furnace', 
            'Natural Gas Fuel Boiler', 'Natural Gas Fuel Furnace',
            'Propane Fuel Boiler', 'Propane Fuel Furnace'
        ]
        tech_col = 'heating_type'
    elif category == 'waterHeating':
        allowed_tech = [
            'Electric Heat Pump, 80 gal', 'Electric Premium', 'Electric Standard',
            'Fuel Oil Premium', 'Fuel Oil Standard', 
            'Natural Gas Premium', 'Natural Gas Standard',
            'Propane Premium', 'Propane Standard'
        ]
        tech_col = 'waterHeating_type'
    
    valid_mask = df[tech_col].isin(allowed_tech)
    
    if mode == 'filter':
        df = df[valid_mask]
        print(f"Filtered for allowed technologies in {category}: {allowed_tech}")
    elif mode == 'mask':
        # Determine consumption columns based on category.
        fuel_types = ['electricity', 'fuelOil', 'naturalGas', 'propane']
        for fuel in fuel_types:
            cons_col = f'base_{fuel}_{category}_consumption'
            df.loc[~valid_mask, cons_col] = np.nan
        print(f"Applied NaN to consumption columns in {category} for invalid technologies")
    else:
        raise ValueError("Mode must be 'filter' or 'mask'")
    
    return df

def debug_filters(df, filter_name):
    """Prints a debugging statement about the DataFrame size after a filter is applied.

    This function helps to track the number of rows remaining in the DataFrame
    after a specific filter step.

    Args:
        df (pd.DataFrame): The DataFrame after a filter operation.
        filter_name (str): A short description of the filter operation.

    Returns:
        None

    Raises:
        None: This function does not raise any exceptions.
    """
    if df.empty:
        print(f"No rows left after applying {filter_name}")
    else:
        print(f"{len(df)} rows remain after applying {filter_name}")


def extract_city_name(row):
    """Extracts the city name from a string in the format 'ST, CityName'.

    If the input does not match the pattern of two uppercase letters,
    followed by a comma and a space, then the original string is returned.

    Args:
        row (str): A string in the format 'ST, CityName'.

    Returns:
        str: The extracted city name if the format matches; otherwise, the original string.

    Raises:
        None: This function does not raise any exceptions.
    """
    # Regex to match exactly two uppercase letters, then a comma and a space, capturing the remainder
    match = re.match(r'^[A-Z]{2}, (.+)$', row)
    return match.group(1) if match else row


def df_enduse_refactored(df_baseline, fuel_filter='Yes', tech_filter='Yes', invalid_row_handling='mask'):
    """
    Creates and returns a DataFrame named df_enduse based on a baseline DataFrame,
    optionally applying fuel and technology filters.

    Args:
        df_baseline (pd.DataFrame): The baseline DataFrame from which df_enduse is created.
        fuel_filter (str, optional): 'Yes' to apply the fuel filter, defaults to 'Yes'.
        tech_filter (str, optional): 'Yes' to apply the technology filter, defaults to 'Yes'.

    Returns:
        pd.DataFrame: A new DataFrame (df_enduse) with standardized fuel columns, 
            total consumption columns, and optional filters applied.

    Raises:
        None: This function does not raise any exceptions.
    """
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
    # Create a new DataFrame named df_enduse using pd.DataFrame constructor
    df_enduse = pd.DataFrame({
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
        'clothesDrying_type': df_baseline['in.clothes_dryer'],
        'base_electricity_clothesDrying_consumption': df_baseline['out.electricity.clothes_dryer.energy_consumption.kwh'],
        'base_naturalGas_clothesDrying_consumption': df_baseline['out.natural_gas.clothes_dryer.energy_consumption.kwh'],
        'base_propane_clothesDrying_consumption': df_baseline['out.propane.clothes_dryer.energy_consumption.kwh'],
        'base_cooking_fuel': df_baseline['in.cooking_range'],
        'cooking_type': df_baseline['in.cooking_range'],
        'base_electricity_cooking_consumption': df_baseline['out.electricity.range_oven.energy_consumption.kwh'],
        'base_naturalGas_cooking_consumption': df_baseline['out.natural_gas.range_oven.energy_consumption.kwh'],
        'base_propane_cooking_consumption': df_baseline['out.propane.range_oven.energy_consumption.kwh']
    })
    
    categories = ['heating', 'waterHeating', 'clothesDrying', 'cooking']
    for category in categories:
        # Determine fuel types based on the category
        if category in ['heating', 'waterHeating']:
            fuel_types = ['electricity', 'fuelOil', 'naturalGas', 'propane']
        else:
            fuel_types = ['electricity', 'naturalGas', 'propane']

        # Calculate total consumption by summing fuel-specific columns
        total_consumption = sum(
            df_enduse.get(f'base_{fuel}_{category}_consumption', pd.Series([], dtype=float)).fillna(0)
            for fuel in fuel_types
        )
        df_enduse[f'baseline_{category}_consumption'] = total_consumption.replace(0, np.nan)
        debug_filters(df_enduse, f"total {category} consumption calculation")
        
        # Apply filtering or masking using the updated helper functions
        if category in ['heating', 'waterHeating']:
            df_enduse = apply_fuel_filter(df_enduse, category, fuel_filter, mode=invalid_row_handling)
            debug_filters(df_enduse, f"{category} fuel filter/mask")
            df_enduse = apply_technology_filter(df_enduse, category, tech_filter, mode=invalid_row_handling)
            debug_filters(df_enduse, f"{category} technology filter/mask")
        else:
            df_enduse = apply_fuel_filter(df_enduse, category, fuel_filter, mode=invalid_row_handling)
            debug_filters(df_enduse, f"{category} fuel filter/mask")
    
    return df_enduse

def df_enduse_compare(df_mp, input_mp, menu_mp, df_baseline, df_cooking_range):
    """Creates a comparison DataFrame by merging multiple DataFrames based on measure packages.

    This function constructs a new DataFrame (df_compare) that includes columns
    from df_mp, df_cooking_range, and merges them with df_baseline to compare
    baseline vs. measure package outputs for heating, water heating, clothes drying,
    and cooking.

    Args:
        df_mp (pd.DataFrame): The main DataFrame containing modeling parameters and outputs.
        input_mp (str): The input measure package ID (e.g., 'upgrade09', 'upgrade10').
        menu_mp (int): The menu measure package number.
        df_baseline (pd.DataFrame): The baseline DataFrame to merge with df_compare.
        df_cooking_range (pd.DataFrame): Additional DataFrame for cooking range parameters/outputs.

    Returns:
        pd.DataFrame: A merged DataFrame (df_compare) that includes relevant columns for
            baseline and measure packages comparison.

    Raises:
        None: This function does not raise any exceptions.
    """
    # Build df_compare from relevant columns in df_mp and df_cooking_range
    df_compare = pd.DataFrame({
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
        'cooking_range_in_unit': df_cooking_range['in.cooking_range'],
        'upgrade_cooking_range': df_cooking_range['upgrade.cooking_range']
    })

    categories = ['heating', 'waterHeating', 'clothesDrying', 'cooking']
    for category in categories:
        if category == 'heating':
            # Special handling for measure packages 9 and 10 (MP9, MP10)
            if input_mp == 'upgrade09':
                menu_mp = 9
                df_compare[f'mp{menu_mp}_heating_consumption'] = df_mp['out.electricity.heating.energy_consumption.kwh'].round(2)

                # Basic Enclosure Package
                df_compare['base_insulation_atticFloor'] = df_mp['in.insulation_ceiling']
                df_compare['upgrade_insulation_atticFloor'] = df_mp['upgrade.insulation_ceiling']
                df_compare['out_params_floor_area_attic_ft_2'] = df_mp['out.params.floor_area_attic_ft_2']

                df_compare['upgrade_infiltration_reduction'] = df_mp['upgrade.infiltration_reduction']

                df_compare['base_ducts'] = df_mp['in.ducts']
                df_compare['upgrade_duct_sealing'] = df_mp['upgrade.ducts']
                df_compare['out_params_duct_unconditioned_surface_area_ft_2'] = df_mp['out.params.duct_unconditioned_surface_area_ft_2']

                df_compare['base_insulation_wall'] = df_mp['in.insulation_wall']
                df_compare['upgrade_insulation_wall'] = df_mp['upgrade.insulation_wall']
                df_compare['out_params_wall_area_above_grade_exterior_ft_2'] = df_mp['out.params.wall_area_above_grade_exterior_ft_2']

            elif input_mp == 'upgrade10':
                menu_mp = 10
                df_compare[f'mp{menu_mp}_heating_consumption'] = df_mp['out.electricity.heating.energy_consumption.kwh'].round(2)

                # Basic Enclosure Package
                df_compare['base_insulation_atticFloor'] = df_mp['in.insulation_ceiling']
                df_compare['upgrade_insulation_atticFloor'] = df_mp['upgrade.insulation_ceiling']
                df_compare['out_params_floor_area_attic_ft_2'] = df_mp['out.params.floor_area_attic_ft_2']

                df_compare['upgrade_infiltration_reduction'] = df_mp['upgrade.infiltration_reduction']

                df_compare['base_ducts'] = df_mp['in.ducts']
                df_compare['upgrade_duct_sealing'] = df_mp['upgrade.ducts']
                df_compare['out_params_duct_unconditioned_surface_area_ft_2'] = df_mp['out.params.duct_unconditioned_surface_area_ft_2']

                df_compare['base_insulation_wall'] = df_mp['in.insulation_wall']
                df_compare['upgrade_insulation_wall'] = df_mp['upgrade.insulation_wall']
                df_compare['out_params_wall_area_above_grade_exterior_ft_2'] = df_mp['out.params.wall_area_above_grade_exterior_ft_2']

                # Enhanced Enclosure Package
                df_compare['base_foundation_type'] = df_mp['in.geometry_foundation_type']
                df_compare['base_insulation_foundation_wall'] = df_mp['in.insulation_foundation_wall']
                df_compare['base_insulation_rim_joist'] = df_mp['in.insulation_rim_joist']
                df_compare['upgrade_insulation_foundation_wall'] = df_mp['upgrade.insulation_foundation_wall']
                df_compare['out_params_floor_area_foundation_ft_2'] = df_mp['out.params.floor_area_foundation_ft_2']
                df_compare['out_params_rim_joist_area_above_grade_exterior_ft_2'] = df_mp['out.params.rim_joist_area_above_grade_exterior_ft_2']

                df_compare['upgrade_seal_crawlspace'] = df_mp['upgrade.geometry_foundation_type']
                df_compare['base_insulation_roof'] = df_mp['in.insulation_roof']
                df_compare['upgrade_insulation_roof'] = df_mp['upgrade.insulation_roof']
                df_compare['out_params_roof_area_ft_2'] = df_mp['out.params.roof_area_ft_2']

            else:
                df_compare[f'mp{menu_mp}_heating_consumption'] = df_mp['out.electricity.heating.energy_consumption.kwh'].round(2)

        elif category == 'waterHeating':
            df_compare[f'mp{menu_mp}_waterHeating_consumption'] = df_mp['out.electricity.hot_water.energy_consumption.kwh'].round(2)

        elif category == 'clothesDrying':
            df_compare[f'mp{menu_mp}_clothesDrying_consumption'] = df_mp['out.electricity.clothes_dryer.energy_consumption.kwh'].round(2)

        elif category == 'cooking':
            df_compare[f'mp{menu_mp}_cooking_consumption'] = df_cooking_range['out.electricity.range_oven.energy_consumption.kwh'].round(2)

    # Merge the baseline DataFrame and df_compare on their index
    df_compare = pd.merge(df_baseline, df_compare, how='inner', left_index=True, right_index=True)
    return df_compare