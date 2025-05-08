import pandas as pd
import numpy as np
import re
from typing import Any, Optional

from cmu_tare_model.constants import EQUIPMENT_SPECS

from cmu_tare_model.utils.validation_framework import get_valid_calculation_mask
from cmu_tare_model.utils.calculation_utils import (
    get_all_possible_fuel_columns,
    identify_valid_homes
    )

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
LOAD EUSS/RESSTOCK DATA AND APPLY FILTERS
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""


def extract_city_name(row: str) -> str:
    """
    Extracts the city name from a string in the format 'ST, CityName'.

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

    # UPDATED: Uses get_all_possible_fuel_columns() for consumption calculation
    for category in EQUIPMENT_SPECS.keys():
        # Get consumption columns for this category
        consumption_columns = get_all_possible_fuel_columns(category)
        
        # Calculate total consumption by summing fuel-specific columns
        total_consumption = sum(
            df_enduse.get(col, pd.Series([], dtype=float)).fillna(0)
            for col in consumption_columns
        )
        df_enduse[f'baseline_{category}_consumption'] = total_consumption.replace(0, np.nan)
        print(f"Calculated total {category} consumption")
    
    # Step 1: Create data quality flags
    df_enduse = identify_valid_homes(df_enduse)
    
    # Step 2: Apply validation using the combined validation system
    # Since this is baseline data, menu_mp = 0
    print("\nApplying data validation (baseline only):")
    for category in EQUIPMENT_SPECS.keys():
        # Get validation mask (this already knows it's baseline)
        valid_mask = get_valid_calculation_mask(df_enduse, category, menu_mp=0, verbose=True)
        
        # Apply masking to consumption columns
        columns_to_mask = get_all_possible_fuel_columns(category)
        columns_to_mask.append(f'baseline_{category}_consumption')
        
        # Apply masking
        for col in columns_to_mask:
            if col in df_enduse.columns:
                non_nan_before = df_enduse[col].notna().sum()
                df_enduse.loc[~valid_mask, col] = np.nan
                non_nan_after = df_enduse[col].notna().sum()
                
                masked_count = non_nan_before - non_nan_after
                if masked_count > 0:
                    print(f"  {col}: Masked {masked_count} values")

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

    for category in EQUIPMENT_SPECS.keys():
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
    
    # Ensure validation flags are preserved from the baseline DataFrame
    validation_flags = [col for col in df_baseline.columns 
                       if col.startswith('valid_') or col.startswith('include_')]
    
    for flag in validation_flags:
        if flag in df_baseline.columns and flag not in df_compare.columns:
            df_compare[flag] = df_baseline[flag]
    
    # Apply combined validation (data quality + retrofit status)
    print("\nApplying combined validation (data quality + retrofit status):")
    for category in EQUIPMENT_SPECS.keys():
        # Get combined validation mask
        valid_mask = get_valid_calculation_mask(df_compare, category, menu_mp, verbose=True)
        
        # Determine which columns to mask for this category
        category_cols = []
        
        # Add basic consumption columns
        fuel_columns = get_all_possible_fuel_columns(category)
        category_cols.extend([col for col in fuel_columns if col in df_compare.columns])
        
        # Add total baseline column
        baseline_col = f'baseline_{category}_consumption'
        if baseline_col in df_compare.columns:
            category_cols.append(baseline_col)
        
        # Add measure package column
        mp_col = f'mp{menu_mp}_{category}_consumption'
        if mp_col in df_compare.columns:
            category_cols.append(mp_col)
        
        # Apply masking
        for col in category_cols:
            non_nan_before = df_compare[col].notna().sum()
            df_compare.loc[~valid_mask, col] = np.nan
            non_nan_after = df_compare[col].notna().sum()
            
            masked_count = non_nan_before - non_nan_after
            if masked_count > 0:
                print(f"  {col}: Masked {masked_count} values")

    return df_compare
