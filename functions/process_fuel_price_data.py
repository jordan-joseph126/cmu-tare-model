import pandas as pd
import os

# Get the current working directory of the project
# project_root = os.path.abspath(os.getcwd())
project_root = "C:\\Users\\14128\\Research\\cmu-tare-model"
print(f"Project root directory: {project_root}")

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FUEL PRICES
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

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