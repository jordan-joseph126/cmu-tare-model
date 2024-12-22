import pandas as pd
import os
from config import PROJECT_ROOT
from cmu_tare_model.functions.inflation_adjustment import (
    cpi_ratio_2023_2018, 
    cpi_ratio_2023_2019, 
    cpi_ratio_2023_2020,
    cpi_ratio_2023_2021,
    cpi_ratio_2023_2022
    )

# Print the project root
print(f"Project root directory: {PROJECT_ROOT}")

"""
===========================================================================================================================================================
FUNCTIONS: FUEL PRICES
===========================================================================================================================================================
"""

import pandas as pd

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

# Define function to create a fuel price lookup dictionary without policy_scenario from row
def create_lookup_fuel_price(df, policy_scenario):
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

"""
===========================================================================================================================================================
PROCESS FUEL PRICE DATA AND PREPARE FOR PROJECTION: 
1. Load nominal fuel price data from EIA sources (2018-2022)
2. Convert base units (like dollars per gallon and cf) in nominal fuel price data to USD2023/kWh
3. Inflate all nominal fuel prices to $USD2023 (example: 2018 data in USD2018... multiply by cpi_ratio_2023_2018)
4. Map states to census division using map_location_to_census_division function (allows us to apply regional projection factors)
===========================================================================================================================================================
**Data Sources for Excel workbook containing state average Residential fuel cost for each fuel in 2018**
- EIA State Electricity Price: https://www.eia.gov/electricity/state/archive/2018/
- EIA Natural Gas Prices: https://www.eia.gov/dnav/ng/ng_pri_sum_dcu_SPA_a.htm
- Propane and Fuel Oil: EIA March 2023 Short Term Energy Outlook
    - https://www.eia.gov/outlooks/steo/pdf/wf01.pdf
    - Table WF01: Average Consumer Prices and Expenditures for Heating Fuels During the Winter
    - US Average: 2018-2019 Data
===========================================================================================================================================================
"""

filename = 'fuel_prices_nominal.csv'
relative_path = os.path.join("cmu_tare_model", "data", "fuel_prices", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)
df_fuelPrices_perkWh = pd.read_csv(file_path)

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")

# New units for the converted and inflated prices below
# $USD-2023, PREVIOUSLY USED $USD-2021
df_fuelPrices_perkWh['units'] = 'USD2023 per kWh'

years = ['2018', '2019', '2020', '2021', '2022']

# Take dataframe with nominal prices in their base units and convert to $/kWh equivalent
# https://www.eia.gov/energyexplained/units-and-calculators/british-thermal-units.php
for year in years:
    for index, row in df_fuelPrices_perkWh.iterrows():
        
        # Propane: (dollars per gallon) * (1 gallon propane/91,452 BTU) * (3412 BTU/1 kWh)
        if row['fuel_type'] == 'propane':
            df_fuelPrices_perkWh.at[index, f'{year}_fuelPrice_perkWh'] = row[f'{year}_nominal_unit_price'] * (1/91452) * (3412/1)
        
        # Fuel Oil: (dollars/gallon) * (1 gallon heating oil/138,500 BTU) * (3412 BTU/1 kWh)
        elif row['fuel_type'] == 'fuelOil':
            df_fuelPrices_perkWh.at[index, f'{year}_fuelPrice_perkWh'] = row[f'{year}_nominal_unit_price'] * (1/138500) * (3412/1)
        
        # Natural Gas: (dollars/cf) * (thousand cf/1000 cf) * (1 cf natural gas/1039 BTU) * (3412 BTU/1 kWh)
        elif row['fuel_type'] == 'naturalGas':
            df_fuelPrices_perkWh.at[index, f'{year}_fuelPrice_perkWh'] = row[f'{year}_nominal_unit_price'] * (1/1000) * (1/1039) * (3412/1)
        
        # Electricity: convert cents per kWh to $ per kWh
        elif row['fuel_type'] == 'electricity':
            df_fuelPrices_perkWh.at[index, f'{year}_fuelPrice_perkWh'] = row[f'{year}_nominal_unit_price'] / 100

# Convert nominal dollars to real 2022 US dollars (USD2022)
# $USD-2023, PREVIOUSLY USED $USD-2021
df_fuelPrices_perkWh['2018_fuelPrice_perkWh'] = df_fuelPrices_perkWh['2018_fuelPrice_perkWh'] * cpi_ratio_2023_2018
df_fuelPrices_perkWh['2019_fuelPrice_perkWh'] = df_fuelPrices_perkWh['2019_fuelPrice_perkWh'] * cpi_ratio_2023_2019
df_fuelPrices_perkWh['2020_fuelPrice_perkWh'] = df_fuelPrices_perkWh['2020_fuelPrice_perkWh'] * cpi_ratio_2023_2020
df_fuelPrices_perkWh['2021_fuelPrice_perkWh'] = df_fuelPrices_perkWh['2021_fuelPrice_perkWh'] * cpi_ratio_2023_2021
df_fuelPrices_perkWh['2022_fuelPrice_perkWh'] = df_fuelPrices_perkWh['2022_fuelPrice_perkWh'] * cpi_ratio_2023_2022

# Original dictionary mapping census divisions to states
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
state_to_census_division = {}
for division, states in map_states_census_divisions.items():
    for state in states:
        state_to_census_division[state] = division

# Function to map location to census division
def map_location_to_census_division(location):
    if location in state_to_census_division:
        return state_to_census_division[location]
    return location

# Apply the function to map locations using .loc
df_fuelPrices_perkWh.loc[:, 'census_division'] = df_fuelPrices_perkWh['location_map'].apply(map_location_to_census_division)
# print(df_fuelPrices_perkWh)

"""
===========================================================================================================================================================
PROJECT FUTURE FUEL PRICES: 
5. Load 2022-2050 regional projection factor data from AEO2023 (Normalized all fuel prices by the 2022 value for each region)
6. For each policy scenario, project future fuel prices using the project_future_prices function
7. For each policy scenario, create a lookup dictionary using the create_lookup_fuel_price function
===========================================================================================================================================================
Pre-IRA Projections
(AEO 2023 'No Inflation Reduction Act' Scenario)
===========================================================================================================================================================
"""

# Project Fuel Prices from 2022 to 2050
filename = 'aeo_projections_2022_2050.xlsx'
relative_path = os.path.join("cmu_tare_model", "data", "projections", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)
df_fuelPrices_projection_factors = pd.read_excel(io=file_path, sheet_name='fuel_price_factors_2022_2050')

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")
# print(df_fuelPrices_projection_factors)

# Convert the factors dataframe into a lookup dictionary including policy_scenario
factor_dict = df_fuelPrices_projection_factors.set_index(['region', 'fuel_type', 'policy_scenario']).to_dict('index')
# print(factor_dict)

# Pre-IRA policy_scenario: No Inflation Reduction Act
# Pass the desired policy_scenario as a parameter when applying the function
preIRA_projected_prices_df = df_fuelPrices_perkWh.apply(lambda row: project_future_prices(row, factor_dict, 'No Inflation Reduction Act'), axis=1)

# Concatenate the projected prices with the original DataFrame
df_fuelPrices_perkWh_preIRA = pd.concat([df_fuelPrices_perkWh, preIRA_projected_prices_df], axis=1)

# Create Fuel Price Lookup with the policy_scenario included
lookup_fuel_prices_preIRA = create_lookup_fuel_price(df_fuelPrices_perkWh_preIRA, 'No Inflation Reduction Act')
# print(lookup_fuel_prices_preIRA)

"""
===========================================================================================================================================================
IRA-Reference Projections
(AEO 2023 'IRA Reference Case' Scenario)
===========================================================================================================================================================
"""

# IRA-Reference policy_scenario: AEO2023 Reference Case
# Pass the desired policy_scenario as a parameter when applying the function
iraRef_projected_prices_df = df_fuelPrices_perkWh.apply(lambda row: project_future_prices(row, factor_dict, 'AEO2023 Reference Case'), axis=1)

# Concatenate the projected prices with the original DataFrame
df_fuelPrices_perkWh_iraRef = pd.concat([df_fuelPrices_perkWh, iraRef_projected_prices_df], axis=1)

# Create Fuel Price Lookup with the policy_scenario included
lookup_fuel_prices_iraRef = create_lookup_fuel_price(df_fuelPrices_perkWh_iraRef, 'AEO2023 Reference Case')
# print(lookup_fuel_prices_iraRef)