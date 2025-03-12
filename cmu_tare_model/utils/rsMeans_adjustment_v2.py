import os
import pandas as pd

# import from cmu-tare-model
from config import PROJECT_ROOT
print(f"Project root directory: {PROJECT_ROOT}")

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
RSMEANS CITY COST INDEX
Adjustment Factors for Construction
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

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
    'cci_loc_adjust_factor_material': ((df_rsMeans_cityCostIndex['Material'] / 100) + 1.0),
    'cci_loc_adjust_factor_installation': ((df_rsMeans_cityCostIndex['Installation'] / 100) + 1.0),
    'cci_loc_adjust_factor_avg': ((df_rsMeans_cityCostIndex['Average'] / 100) + 1.0),
})
print(df_rsMeans_cityCostIndex)

# Use CCI to adjust for cost differences when compared to the national average
# Accounts for the costs of materials, labor and equipment and compares it to a national average of 30 major U.S. cities
loc_adjust_factor_map = df_rsMeans_cityCostIndex.set_index('City')['cci_loc_adjust_factor_avg'].to_dict()
loc_adjust_factor_30cities = (3.00 / 100) + 1.0
print(loc_adjust_factor_30cities)

# Use CCI to adjust for cost differences when compared to the national average
# Function to map city to its average cost
def map_loc_adjust_factor(city):
    if city in loc_adjust_factor_map:
        return loc_adjust_factor_map[city]
    elif city == 'Not in a census Place' or city == 'In another census Place':
        return loc_adjust_factor_map.get('+30 City Average')
    else:
        return loc_adjust_factor_map.get('+30 City Average')