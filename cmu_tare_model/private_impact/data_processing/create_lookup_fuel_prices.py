import os
import pandas as pd

# import from cmu-tare-model package
from config import PROJECT_ROOT
from cmu_tare_model.utils.inflation_adjustment import (
    cpi_ratio_2023_2018, 
    cpi_ratio_2023_2019, 
    cpi_ratio_2023_2020,
    cpi_ratio_2023_2021,
    cpi_ratio_2023_2022,
    )
from cmu_tare_model.utils.data_visualization import print_truncated_dict

# ====================================================================================================================================================================================
# Set print_verbose to True for detailed output, or False for minimal output
# By default, verbose is set to False because define_scenario_params is imported multiple times in the codebase
# and we don't want to print the same information multiple times.
print_verbose = False
# ====================================================================================================================================================================================


"""
====================================================================================================================================================================
CREATE LOOKUP FOR FUEL PRICES
====================================================================================================================================================================
"""

def map_location_to_census_division(location):
    """
    Maps a state or location to its corresponding census division.

    Args:
        location (str): The state or location name.

    Returns:
        str: The corresponding census division or the original location if no mapping exists.
    """
    return state_to_census_division.get(location, location)

# LAST UPDATED APRIL 8, 2025 @ 2:15 PM
def process_fuel_price_data(df_fuel_prices_import):
    """
    Processes and adjusts fuel prices data to consistent units and inflation-adjusted values.
    """
    # Create a copy of the dataframe to avoid mutating the original
    df_fuel_prices_copy = df_fuel_prices_import.copy()

    # New units for the converted and inflated prices below
    df_fuel_prices_copy['units'] = 'USD2023 per kWh'

    years = ['2018', '2019', '2020', '2021', '2022']

    # Take dataframe with nominal prices in their base units and convert to $/kWh equivalent
    for year in years:
        for index, row in df_fuel_prices_copy.iterrows():
            
            # Propane: (dollars per gallon) * (1 gallon propane/91,452 BTU) * (3412 BTU/1 kWh)
            if row['fuel_type'] == 'propane':
                df_fuel_prices_copy.at[index, f'{year}_fuelPrice_perkWh'] = row[f'{year}_nominal_unit_price'] * (1/91452) * (3412/1)
            
            # Fuel Oil: (dollars/gallon) * (1 gallon heating oil/138,500 BTU) * (3412 BTU/1 kWh)
            elif row['fuel_type'] == 'fuelOil':
                df_fuel_prices_copy.at[index, f'{year}_fuelPrice_perkWh'] = row[f'{year}_nominal_unit_price'] * (1/138500) * (3412/1)
            
            # Natural Gas: (dollars/cf) * (thousand cf/1000 cf) * (1 cf natural gas/1039 BTU) * (3412 BTU/1 kWh)
            elif row['fuel_type'] == 'naturalGas':
                df_fuel_prices_copy.at[index, f'{year}_fuelPrice_perkWh'] = row[f'{year}_nominal_unit_price'] * (1/1000) * (1/1039) * (3412/1)
            
            # Electricity: convert cents per kWh to $ per kWh
            elif row['fuel_type'] == 'electricity':
                df_fuel_prices_copy.at[index, f'{year}_fuelPrice_perkWh'] = row[f'{year}_nominal_unit_price'] / 100

    # Convert nominal dollars to real 2023 US dollars (USD2023)
    df_fuel_prices_copy['2018_fuelPrice_perkWh'] = df_fuel_prices_copy['2018_fuelPrice_perkWh'] * cpi_ratio_2023_2018
    df_fuel_prices_copy['2019_fuelPrice_perkWh'] = df_fuel_prices_copy['2019_fuelPrice_perkWh'] * cpi_ratio_2023_2019
    df_fuel_prices_copy['2020_fuelPrice_perkWh'] = df_fuel_prices_copy['2020_fuelPrice_perkWh'] * cpi_ratio_2023_2020
    df_fuel_prices_copy['2021_fuelPrice_perkWh'] = df_fuel_prices_copy['2021_fuelPrice_perkWh'] * cpi_ratio_2023_2021
    df_fuel_prices_copy['2022_fuelPrice_perkWh'] = df_fuel_prices_copy['2022_fuelPrice_perkWh'] * cpi_ratio_2023_2022

    # Map states to census divisions
    map_states_census_divisions = {
        "New England": ["CT", "ME", "MA", "NH", "RI", "VT"],
        "Middle Atlantic": ["NJ", "NY", "PA"],
        "East North Central": ["IN", "IL", "MI", "OH", "WI"],
        "West North Central": ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
        "South Atlantic": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"],
        "East South Central": ["AL", "KY", "MS", "TN"],
        "West South Central": ["AR", "LA", "OK", "TX"],
        "Mountain": ["AZ", "CO", "ID", "NM", "MT", "UT", "NV", "WY"],
        "Pacific": ["AK", "CA", "HI", "OR", "WA"]
    }

    # Reverse the mapping to create a state-to-census-division map
    global state_to_census_division
    state_to_census_division = {}
    for division, states in map_states_census_divisions.items():
        for state in states:
            state_to_census_division[state] = division

    # Apply the function to map locations to census divisions
    df_fuel_prices_copy['census_division'] = df_fuel_prices_copy['location_map'].apply(map_location_to_census_division)

    return df_fuel_prices_copy

def create_projection_factors_dict(df_projection_factors):
    """
    Transforms the projection factors DataFrame into a nested dictionary.
    
    The input DataFrame is expected to have columns:
    - 'region': Region identifier
    - 'fuel_type': Type of fuel (electricity, naturalGas, etc.)
    - 'policy_scenario': Policy scenario identifier
    - Year columns (2022, 2023, ...): Values are projection factors
    
    Args:
        df_projection_factors (pd.DataFrame): DataFrame with projection factors
        
    Returns:
        dict: Nested dictionary mapping (region, fuel_type, policy_scenario) to {year: factor}
    """
    # Create the projection factors dictionary
    projection_factors = {}
    
    # Get the year columns (all numeric column names)
    year_columns = [col for col in df_projection_factors.columns if isinstance(col, int) or 
                   (isinstance(col, str) and col.isdigit())]
    
    # Process each row
    for _, row in df_projection_factors.iterrows():
        # Create the key tuple
        key = (row['region'], row['fuel_type'], row['policy_scenario'])
        
        # Initialize dictionary for this key if it doesn't exist
        if key not in projection_factors:
            projection_factors[key] = {}
        
        # Add factors for each year
        for year_col in year_columns:
            year = int(year_col) if isinstance(year_col, str) else year_col
            factor = row[year_col]
            projection_factors[key][year] = factor
    
    return projection_factors

def project_future_prices(row, factor_dict, policy_scenario):
    """
    Projects future fuel prices from 2022 to 2050 for a specific location and fuel type.
    
    Args:
        row (pd.Series): A row from the fuel prices dataframe
        factor_dict (dict): Dictionary of projection factors
        policy_scenario (str): Policy scenario to use
        
    Returns:
        pd.Series: Series with projected prices for years 2022-2050
    """
    loc = row['census_division']
    fuel = row['fuel_type']
    price_2022 = row['2022_fuelPrice_perkWh']
    
    # Create key for factor lookup
    key = (loc, fuel, policy_scenario)
    
    # If no factors for specific location, try national factors
    if key not in factor_dict:
        key = ('National', fuel, policy_scenario)
    
    # If still no factors, return empty series
    if key not in factor_dict:
        return pd.Series()
    
    # Get the factors
    factors = factor_dict[key]
    
    # Calculate prices for each year
    future_prices = {}
    for year in range(2022, 2051):
        if year in factors:
            future_prices[f'{year}_fuelPrice_perkWh'] = price_2022 * factors[year]
    
    return pd.Series(future_prices)

def create_lookup_fuel_price(df, policy_scenario):
    """
    Creates a nested dictionary for quick lookup of fuel prices.
    
    Args:
        df (pd.DataFrame): DataFrame with projected fuel prices
        policy_scenario (str): Policy scenario identifier
        
    Returns:
        dict: Nested dictionary for fuel price lookups
    """
    lookup_dict = {}
    
    for _, row in df.iterrows():
        location = row['location_map']
        fuel_type = row['fuel_type']
        
        # Initialize nested dictionary structure
        if location not in lookup_dict:
            lookup_dict[location] = {}
        if fuel_type not in lookup_dict[location]:
            lookup_dict[location][fuel_type] = {}
        if policy_scenario not in lookup_dict[location][fuel_type]:
            lookup_dict[location][fuel_type][policy_scenario] = {}
        
        # Add prices for each year using robust conditional check:
        for year in range(2022, 2051):
            column_name = f"{year}_fuelPrice_perkWh"
            if column_name in row.index:
                value = row[column_name]
                # If value is a Series, reduce it to a scalar.
                if isinstance(value, pd.Series):
                    value = value.item() if value.size == 1 else value.iloc[0]
                # Only add value if it is not missing (NA).
                if not pd.isna(value):
                    lookup_dict[location][fuel_type][policy_scenario][year] = value
         
    return lookup_dict

# ====================================================================================================================================================================
# LOAD AND PROCESS FUEL PRICE DATA
# ====================================================================================================================================================================
filename = 'fuel_prices_nominal.csv'
relative_path = os.path.join("cmu_tare_model", "data", "fuel_prices", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)
df_fuel_prices_nominal = pd.read_csv(file_path)

# Process and adjust the fuel prices
df_fuel_prices_adjusted = process_fuel_price_data(df_fuel_prices_nominal)

# ====================================================================================================================================================================
# LOAD PROJECTION FACTORS DATA AND PROJECT FUEL PRICES FOR PRE-IRA AND IRA REFERENCE SCENARIOS
# ====================================================================================================================================================================
filename = 'aeo_projections_2022_2050.xlsx'
relative_path = os.path.join("cmu_tare_model", "data", "projections", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)
df_projection_factors = pd.read_excel(io=file_path, sheet_name='fuel_price_factors_2022_2050')

# Print column names for debugging
if print_verbose:
    print("Columns in projection factors DataFrame:", df_projection_factors.columns.tolist())

# Create a dictionary from the projection factors DataFrame
# This approach handles the wide-format data where years are columns
projection_factors = create_projection_factors_dict(df_projection_factors)

# ====================================================================================================================================================================
# PROJECT FUEL PRICES USING THE SIMPLIFIED APPROACH
# ====================================================================================================================================================================

# Pre-IRA policy_scenario: No Inflation Reduction Act
preIRA_projected_prices_df = df_fuel_prices_adjusted.apply(
    lambda row: project_future_prices(row, projection_factors, 'No Inflation Reduction Act'), 
    axis=1
)

# Concatenate the projected prices with the original DataFrame
df_fuel_prices_preIRA = pd.concat([df_fuel_prices_adjusted, preIRA_projected_prices_df], axis=1)

# IRA-Reference policy_scenario: AEO2023 Reference Case
iraRef_projected_prices_df = df_fuel_prices_adjusted.apply(
    lambda row: project_future_prices(row, projection_factors, 'AEO2023 Reference Case'), 
    axis=1
)

# Concatenate the projected prices with the original DataFrame
df_fuel_prices_iraRef = pd.concat([df_fuel_prices_adjusted, iraRef_projected_prices_df], axis=1)

if print_verbose:
    print(f"""
    ====================================================================================================================================================================
    PROCESSING FUEL PRICE DATA FOR PRE-IRA AND IRA REFERENCE SCENARIOS
    - First, obtain the projection factors from the AEO 2022 to 2050 projections.
    - Then, project the fuel prices for the Pre-IRA and IRA Reference scenarios.
    ====================================================================================================================================================================

    DataFrame for Pre-IRA Scenario:
    {df_fuel_prices_preIRA}

    DataFrame for IRA Reference Scenario:
    {df_fuel_prices_iraRef}

    """)

# ====================================================================================================================================================================
# CREATE LOOKUP DICTIONARIES FOR FUEL PRICES
# ====================================================================================================================================================================

# Create the lookup dictionary for Pre-IRA scenario
lookup_fuel_prices_preIRA = create_lookup_fuel_price(
    df_fuel_prices_preIRA,
    'No Inflation Reduction Act'
)

# Create the lookup dictionary for IRA Reference scenario
lookup_fuel_prices_iraRef = create_lookup_fuel_price(
    df_fuel_prices_iraRef,
    'AEO2023 Reference Case'
)

if print_verbose:
    print(f"""
    ====================================================================================================================================================================
    CREATE LOOKUP DICTIONARIES FOR FUEL PRICES
    ====================================================================================================================================================================
    Lookup dictionary for Pre-IRA and IRA-Ref scenarios:
    """)
    print_truncated_dict(lookup_fuel_prices_preIRA, n=5)
    print_truncated_dict(lookup_fuel_prices_iraRef, n=5)

"""
As of April 8, 2025, the dataframes and lookup dictionaries for fuel prices have been created successfully and are identical to the original code.

The dataframes and lookup dictionaries are compared in the data_comparison_script.py file.

Starting data comparison...

=== Comparing df_preIRA_ORIGINAL vs df_preIRA_NEW ===
Same shape: True ((116, 46) vs (116, 46))
Same columns: True
Same index: True
Exact match: True

=== Comparing df_iraRef_ORIGINAL vs df_iraRef_NEW ===
Same shape: True ((116, 46) vs (116, 46))
Same columns: True
Same index: True
Exact match: True

=== Comparing lookup_preIRA_ORIGINAL vs lookup_preIRA_NEW ===
Same keys: True
All values match for common keys: True

=== Comparing lookup_iraRef_ORIGINAL vs lookup_iraRef_NEW ===
Same keys: True
All values match for common keys: True

Comparison complete!

"""