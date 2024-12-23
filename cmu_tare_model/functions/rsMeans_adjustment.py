import pandas as pd
import os

from config import PROJECT_ROOT
print(f"Project root directory: {PROJECT_ROOT}")

from cmu_tare_model.functions.inflation_adjustment import *

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
RSMEANS CITY COST INDEX
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

### Adjustment Factors for Construction: 
#### RSMeans City Cost Index
#### Consumer Price Index for All Urban Consumers (CPI, CPI-U)
# Adjust for regional cost differences with RSMeans
filename = "rsMeans_cityCostIndex.csv"
relative_path = os.path.join("cmu_tare_model", "data", "inflation_data", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")
print("\n")

df_rsMeans_cityCostIndex = pd.read_csv(file_path)

df_rsMeans_cityCostIndex = pd.DataFrame({
    'State': df_rsMeans_cityCostIndex['State'],
    'City': df_rsMeans_cityCostIndex['City'],
    'Material': (df_rsMeans_cityCostIndex['Material']).round(2),
    'Installation': (df_rsMeans_cityCostIndex['Installation']).round(2),
    'Average': (df_rsMeans_cityCostIndex['Average']).round(2),
})
print(df_rsMeans_cityCostIndex)

# Assuming df_rsMeans_cityCostIndex is your DataFrame with average costs
# Accounts for the costs of materials, labor and equipment and compares it to a national average of 30 major U.S. cities
average_cost_map = df_rsMeans_cityCostIndex.set_index('City')['Average'].to_dict()
rsMeans_national_avg = round((3.00 * (cpi_ratio_2023_2019)), 2)

# Use CCI to adjust for cost differences when compared to the national average
# Function to map city to its average cost
def map_average_cost(city):
    if city in average_cost_map:
        return average_cost_map[city]
    elif city == 'Not in a census Place' or city == 'In another census Place':
        return average_cost_map.get('+30 City Average')
    else:
        return average_cost_map.get('+30 City Average')