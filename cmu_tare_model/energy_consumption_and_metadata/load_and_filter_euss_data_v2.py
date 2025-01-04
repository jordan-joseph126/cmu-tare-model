import os
import pandas as pd
import numpy as np
import re

# from config import PROJECT_ROOT

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
        # Updated to filter out ASHP ("Electricity ASHP") and those without HVAC ("None")
        if category == 'heating':
            tech_list = [
                'Electricity Baseboard', 'Electricity Electric Boiler', 'Electricity Electric Furnace',
                'Fuel Oil Fuel Boiler', 'Fuel Oil Fuel Furnace', 
                'Natural Gas Fuel Boiler', 'Natural Gas Fuel Furnace',
                'Propane Fuel Boiler', 'Propane Fuel Furnace'
                ]
            df_filtered = df[df['heating_type'].isin(tech_list)]
            print(f"Filtered for the following Heating technologies: {tech_list}")    
            return df_filtered
        
        # Updated to filter out HP water heaters ("Electric Heat Pump, 80 gal") and those without water heaters ("None")
        elif category == 'waterHeating':
            tech_list = [
                'Electric Premium', 'Electric Standard',
                'Fuel Oil Premium', 'Fuel Oil Standard', 
                'Natural Gas Premium', 'Natural Gas Standard',
                'Propane Premium', 'Propane Standard'
                ]
            df_filtered = df[df['waterHeating_type'].isin(tech_list)]
            print(f"Filtered for the following Water Heating technologies: {tech_list}")
            return df_filtered
        
        # Updated to filter out those without clothes dryers ("None")
        elif category == 'clothesDrying':
            tech_list = [
                "Electric, 100% Usage", "Electric, 120% Usage", "Electric, 80% Usage"
                "Gas, 100% Usage", "Gas, 120% Usage", "Gas, 80% Usage", 
                "Propane, 100% Usage", "Propane, 120% Usage", "Propane, 80% Usage"
                ]
            df_filtered = df[df['clothesDrying_type'].isin(tech_list)]
            print(f"Filtered for the following Clothes Drying technologies: {tech_list}")
            return df_filtered

        # Updated to filter out electric cooking ranges ("Electric, 100% Usage", "Electric, 120% Usage", "Electric, 80% Usage") and those without ranges ("None")
        elif category == 'cooking':
            tech_list = [
                "Gas, 100% Usage", "Gas, 120% Usage", "Gas, 80% Usage", 
                "Propane, 100% Usage", "Propane, 120% Usage", "Propane, 80% Usage"
                ]
            df_filtered = df[df['cooking_type'].isin(tech_list)]
            print(f"Filtered for the following Cooking technologies: {tech_list}")
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

        elif category == 'clothesDrying':
            fuel_types = ['electricity', 'naturalGas', 'propane']
            # Calculate and update total consumption
            total_consumption = sum(df_enduse.get(f'base_{fuel}_{category}_consumption', pd.Series([], dtype=float)).fillna(0) for fuel in fuel_types)
            df_enduse[f'baseline_{category}_consumption'] = total_consumption.replace(0, np.nan)

            debug_filters(df_enduse, f"total {category} consumption calculation")

            # Apply filters
            df_enduse = apply_fuel_filter(df_enduse, category, fuel_filter)
            debug_filters(df_enduse, f"{category} fuel filter")

            # df_enduse = apply_technology_filter(df_enduse, category, tech_filter)
            # debug_filters(df_enduse, f"{category} technology filter")

        # We filter out electric cooking because this tech is the same as the retrofit for MP7
        else:
            fuel_types = ['naturalGas', 'propane']
            # Calculate and update total consumption
            total_consumption = sum(df_enduse.get(f'base_{fuel}_{category}_consumption', pd.Series([], dtype=float)).fillna(0) for fuel in fuel_types)
            df_enduse[f'baseline_{category}_consumption'] = total_consumption.replace(0, np.nan)

            debug_filters(df_enduse, f"total {category} consumption calculation")

            # Apply filters
            df_enduse = apply_fuel_filter(df_enduse, category, fuel_filter)
            debug_filters(df_enduse, f"{category} fuel filter")
            
            # df_enduse = apply_technology_filter(df_enduse, category, tech_filter)
            # debug_filters(df_enduse, f"{category} technology filter")

    return df_enduse

def df_enduse_compare(df_mp, input_mp, menu_mp, df_baseline, df_cooking_range):
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
        'cooking_range_in_unit': df_cooking_range['in.cooking_range'],
        'upgrade_cooking_range': df_cooking_range['upgrade.cooking_range']
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
            df_compare[f'mp{menu_mp}_cooking_consumption'] = df_cooking_range['out.electricity.range_oven.energy_consumption.kwh'].round(2)
            
    # Merge dataframes on bldg id column so everything is lined up
    # df_compare = pd.merge(df_baseline, df_compare, how='inner', on = 'bldg_id')
    # calculate_consumption_reduction(df_compare, category)    

    # If both df_baseline and df_compare now have bldg_id set as their index, modify it as:
    df_compare = pd.merge(df_baseline, df_compare, how='inner', left_index=True, right_index=True)
    # Make sure df_baseline and df_compare both have bldg_id as their index before doing this.

    return df_compare

