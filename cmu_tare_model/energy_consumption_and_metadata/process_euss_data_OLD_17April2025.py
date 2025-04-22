import pandas as pd
import numpy as np
import re
from typing import Any, Tuple, Dict, List, Optional, Union

from config import PROJECT_ROOT
from cmu_tare_model.constants import ALLOWED_TECHNOLOGIES, EQUIPMENT_SPECS, FUEL_MAPPING

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
LOAD EUSS/RESSTOCK DATA AND APPLY FILTERS
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""


def extract_city_name(row: str) -> str:
    """Extracts the city name from a string in the format 'ST, CityName'.

    If the input does not match the pattern of two uppercase letters,
    followed by a comma and a space, then the original string is returned.

    Args:
        row: A string in the format 'ST, CityName'.

    Returns:
        The extracted city name if the format matches; otherwise, the original string.
    """
    if not isinstance(row, str):
        return row
        
    # Regex to match exactly two uppercase letters, then a comma and a space, capturing the remainder
    match = re.match(r'^[A-Z]{2}, (.+)$', row)
    return match.group(1) if match else row


def get_fuel_types_for_category(category: str) -> List[str]:
    """Returns the list of fuel types applicable for a given category.
    
    Args:
        category: Equipment category name.
        
    Returns:
        List of fuel type strings for column name construction.
    """
    if category in ['heating', 'waterHeating']:
        return list(FUEL_MAPPING.values())
    elif category == 'clothesDrying':
        # Valid Fuel Types: Electricity, Natural Gas, Propane
        return [v for k, v in FUEL_MAPPING.items() if k != 'Fuel Oil']
    elif category == 'cooking':
        # Valid Fuel Types: Natural Gas and Propane
        return [v for k, v in FUEL_MAPPING.items() if k != 'Fuel Oil' and k != 'Electricity']
    else:
        raise ValueError(f"Invalid category. Must be one of the following: {EQUIPMENT_SPECS.keys()}")
    

def standardize_fuel_name(fuel_desc: Any) -> Optional[str]:
    """Standardizes a fuel description into a recognized category or None.

    This function inspects an input fuel description (e.g., "Electric Heater",
    "Gas Furnace", "Propane Heater") and maps it to one of the following strings:
    "Electricity", "Natural Gas", "Propane", or "Fuel Oil". If the input is NaN,
    not a string, or does not contain any recognizable fuel keyword, the function
    returns None.

    Args:
        fuel_desc: A value representing the fuel description. It can be a string
            containing words like "Electric," "Gas," "Propane," or "Oil." It may
            also be NaN (pandas missing value) or another data type.

    Returns:
        One of the strings {"Electricity", "Natural Gas", "Propane", "Fuel Oil"}
        if a match is found, or None otherwise.
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

    # # Check if fuel_desc is NaN or not a string; return None if so
    # if pd.isna(fuel_desc) or not isinstance(fuel_desc, str):
    #     return None

    # # Convert the string to uppercase for case-insensitive matching
    # fuel_desc_upper = fuel_desc.upper()

    # # Match substrings for known fuel types using keys from FUEL_MAPPING
    # for fuel_name in FUEL_MAPPING.keys():
    #     if fuel_name.upper() in fuel_desc_upper:
    #         return fuel_name
    
    # # If no match is found, return None
    # return None


def preprocess_fuel_data(df: pd.DataFrame,
                         column_name: str
) -> pd.DataFrame:
    """Applies a standardization process to the specified fuel column in the DataFrame.

    This function applies 'standardize_fuel_name' to every value in the specified column
    and updates the DataFrame in-place.

    Args:
        df: The input pandas DataFrame containing fuel data.
        column_name: The name of the column to standardize.

    Returns:
        The updated DataFrame with standardized fuel names in the specified column.

    Raises:
        KeyError: If the specified column does not exist in the DataFrame.
        TypeError: If the DataFrame is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame")
    
    print(f"Processing column: {column_name}")
    print(f"Initial data types: {df[column_name].dtype}")

    # Use .loc to avoid SettingWithCopyWarning when applying the function
    df.loc[:, column_name] = df[column_name].apply(standardize_fuel_name)

    print(f"Data types after processing: {df[column_name].dtype}")
    return df


def identify_valid_homes(df: pd.DataFrame) -> pd.DataFrame:
    """Creates comprehensive data quality flags for all categories.
    
    This function adds columns to track the quality and validity of data
    across all equipment categories. Technology validation is only applied
    to heating and water heating categories.
    
    Args:
        df: DataFrame containing energy consumption data.
        
    Returns:
        DataFrame with added data quality flags.
    """
    # Initialize the overall inclusion flag
    df['include_all'] = True
    print("\nCreating data quality flags for all categories")
    
    for category in EQUIPMENT_SPECS.keys():
        print(f"\n--- Processing {category} ---")
        
        # Create fuel validity flag
        fuel_flag = f'valid_fuel_{category}'
        fuel_col = f'base_{category}_fuel'
        
        if fuel_col in df.columns:
            # Print some diagnostic info about the values
            print(f"Values in {fuel_col} (top 5):")
            print(df[fuel_col].value_counts().head(5))
            
            # UPDATED: use get_fuel_types_for_category to get the valid fuel types
            # Get the appropriate fuel keys for this category using reverse mapping
            valid_fuel_types = get_fuel_types_for_category(category)
            valid_fuel_keys = [k for k, v in FUEL_MAPPING.items() if v in valid_fuel_types]
            df[fuel_flag] = df[fuel_col].isin(valid_fuel_keys)

            # Invalid fuel count and percentage
            invalid_fuel_count = (~df[fuel_flag]).sum()
            invalid_fuel_pct = (invalid_fuel_count / len(df)) * 100 if len(df) > 0 else 0
            print(f"  {category}: Found {invalid_fuel_count} homes ({invalid_fuel_pct:.1f}%) with invalid fuel types")
            
            # Show what's being filtered
            if invalid_fuel_count > 0:
                invalid_fuels = df.loc[~df[fuel_flag], fuel_col].value_counts()
                print("  Invalid fuel types (top 5):")
                print(invalid_fuels.head(5))
        else:
            print(f"  Warning: Column {fuel_col} not found")
            df[fuel_flag] = True
        
        # Handle technology validation only for heating and water heating
        if category in ['heating', 'waterHeating']:
            # Create technology validity flag
            tech_flag = f'valid_tech_{category}'
            tech_col = f'{category}_type'
            
            if tech_col in df.columns and category in ALLOWED_TECHNOLOGIES:
                # Print some diagnostic info
                print(f"Values in {tech_col} (top 5):")
                print(df[tech_col].value_counts().head(5))
                
                print(f"Allowed values for {category}:")
                print(ALLOWED_TECHNOLOGIES[category])
                
                # Check if the technology type is in the allowed list
                df[tech_flag] = df[tech_col].isin(ALLOWED_TECHNOLOGIES[category])

                # Invalid technology count and percentage
                invalid_tech_count = (~df[tech_flag]).sum()
                invalid_tech_pct = (invalid_tech_count / len(df)) * 100 if len(df) > 0 else 0
                print(f"  {category}: Found {invalid_tech_count} homes ({invalid_tech_pct:.1f}%) with invalid technology types")
                
                # Show what's being filtered
                if invalid_tech_count > 0:
                    invalid_techs = df.loc[~df[tech_flag], tech_col].value_counts()
                    print("  Invalid technology types (top 5):")
                    print(invalid_techs.head(5))
                
                # Create category inclusion flag based on both fuel and tech validity
                include_col = f'include_{category}'
                df[include_col] = df[fuel_flag] & df[tech_flag]
            else:
                if category not in ALLOWED_TECHNOLOGIES:
                    print(f"  {category}: No allowed technologies defined")
                elif tech_col not in df.columns:
                    print(f"  {category}: Warning - Column {tech_col} not found")
                
                # Set inclusion flag based only on fuel validity
                include_col = f'include_{category}'
                df[include_col] = df[fuel_flag]
        else:
            # For clothes drying and cooking, only use fuel validation
            print(f"  {category}: Technology validation not applicable (no technology type column)")
            include_col = f'include_{category}'
            df[include_col] = df[fuel_flag]
        
        # Print exclusion summary
        excluded_count = (~df[include_col]).sum()
        excluded_pct = (excluded_count / len(df)) * 100 if len(df) > 0 else 0
        print(f"  {category}: Total {excluded_count} homes ({excluded_pct:.1f}%) excluded from analysis")
        
        # Update the overall inclusion flag
        df['include_all'] &= df[include_col]
    
    overall_excluded = (~df['include_all']).sum()
    overall_pct = (overall_excluded / len(df)) * 100 if len(df) > 0 else 0
    print(f"\nTotal {overall_excluded} homes ({overall_pct:.1f}%) excluded from all categories")
    return df


def mask_invalid_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sets consumption values to NaN based on inclusion flags.
    
    Args:
        df: DataFrame with inclusion flags already created.
        
    Returns:
        DataFrame with consumption values set to NaN for invalid records.
    """
    print("Applying NaN masking based on inclusion flags")
    
    for category in EQUIPMENT_SPECS.keys():
        include_col = f'include_{category}'
        
        if include_col not in df.columns:
            print(f"  {category}: Warning - Inclusion flag '{include_col}' not found. Skipping masking.")
            continue
        
        # Get appropriate fuel types for this category
        fuel_types = get_fuel_types_for_category(category)
        
        # Mask individual fuel consumption columns
        masked_count = 0
        for fuel in fuel_types:
            cons_col = f'base_{fuel}_{category}_consumption'
            if cons_col in df.columns:
                df.loc[~df[include_col], cons_col] = np.nan
                masked_count += 1
        
        # Also mask total consumption
        total_col = f'baseline_{category}_consumption'
        if total_col in df.columns:
            df.loc[~df[include_col], total_col] = np.nan
            masked_count += 1
        
        print(f"  {category}: Masked {masked_count} consumption columns for invalid records")
    
    return df


def df_enduse_refactored(df_baseline: pd.DataFrame) -> pd.DataFrame:
    """Creates a standardized energy usage DataFrame and applies data quality filters.

    This function creates a new DataFrame with standardized column names and structure,
    calculates total consumption by fuel type, creates data quality flags for analysis,
    and sets invalid consumption values to NaN.

    Args:
        df_baseline: The baseline DataFrame containing raw EUSS/ResStock data.

    Returns:
        A standardized DataFrame with processed consumption data and data quality flags.

    Raises:
        ValueError: If required columns are missing from the input DataFrame.
    """
    # Initial check
    if df_baseline.empty:
        print("Warning: Input DataFrame is empty")
        return df_baseline

    # Standardize fuel names in the base columns
    df_baseline = preprocess_fuel_data(df_baseline, 'in.clothes_dryer')
    df_baseline = preprocess_fuel_data(df_baseline, 'in.cooking_range')

    # Initialize df_enduse from df_baseline with all required columns
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
        'county_fips': df_baseline['in.county'].apply(lambda x: x[1:3] + x[4:7]),
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

    # Calculate total consumption for all categories
    for category in EQUIPMENT_SPECS.keys():
        # Get appropriate fuel types for this category
        fuel_types = get_fuel_types_for_category(category)

        # Calculate total consumption by summing fuel-specific columns
        total_consumption = sum(
            df_enduse.get(f'base_{fuel}_{category}_consumption', pd.Series([], dtype=float)).fillna(0)
            for fuel in fuel_types
        )
        df_enduse[f'baseline_{category}_consumption'] = total_consumption.replace(0, np.nan)
    
    # Step 1: Create data quality flags
    df_enduse = identify_valid_homes(df_enduse)
    
    # Step 2: Apply NaN masking based on inclusion flags
    df_enduse = mask_invalid_data(df_enduse)

    return df_enduse


def df_enduse_compare(df_mp: pd.DataFrame, 
                      input_mp: str, 
                      menu_mp: int, 
                      df_baseline: pd.DataFrame, 
                      df_cooking_range: pd.DataFrame
) -> pd.DataFrame:
    """Creates a comparison DataFrame by merging multiple DataFrames based on measure packages.

    This function constructs a new DataFrame (df_compare) that includes columns
    from df_mp, df_cooking_range, and merges them with df_baseline to compare
    baseline vs. measure package outputs for heating, water heating, clothes drying,
    and cooking.

    Args:
        df_mp: The main DataFrame containing modeling parameters and outputs.
        input_mp: The input measure package ID (e.g., 'upgrade09', 'upgrade10').
        menu_mp: The menu measure package number.
        df_baseline: The baseline DataFrame to merge with df_compare.
        df_cooking_range: Additional DataFrame for cooking range parameters/outputs.

    Returns:
        A merged DataFrame (df_compare) that includes relevant columns for
        baseline and measure packages comparison.
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
