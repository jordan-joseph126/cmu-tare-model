import pandas as pd
import os

from config import PROJECT_ROOT

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