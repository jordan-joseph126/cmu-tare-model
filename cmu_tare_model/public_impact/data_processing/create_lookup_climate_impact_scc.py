import os
import pandas as pd

# import from cmu-tare-model package
from config import PROJECT_ROOT

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CLIMATE CHANGE IMPACT SENSITIVITY: SCC LOOKUP
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def create_scc_lookup(df: pd.DataFrame) -> dict:
    """
    Reads the CSV/Excel of SCC data and returns a nested dictionary:
      lookup_climate_impact_scc["lower"][year]   -> float
      lookup_climate_impact_scc["central"][year] -> float
      lookup_climate_impact_scc["upper"][year]   -> float
    """
    
    # Initialize top-level keys for each SCC bound
    lookup_climate_impact_scc = {
        "lower": {},
        "central": {},
        "upper": {}
    }
    
    # Fill in year-specific values under each key
    for _, row in df.iterrows():
        year = int(row["emissions_year"])
        lookup_climate_impact_scc["lower"][year] = row["scc_lower_usd2023"]
        lookup_climate_impact_scc["central"][year] = row["scc_central_usd2023"]
        lookup_climate_impact_scc["upper"][year] = row["scc_upper_usd2023"]
    
    return lookup_climate_impact_scc

print("""
-------------------------------------------------------------------------------------------------------
CLIMATE CHANGE IMPACT SENSITIVITY: SCC LOOKUP
-------------------------------------------------------------------------------------------------------
LOWER BOUND: IWG 2021, 5% Discount Rate (Trump Previously Used 3-7% Discount Rate)
CENTRAL ESTIMATE: IWG 2021, 3% Discount Rate (Obama Administration, Pre-2017)
UPPER BOUND: Recent EPA Central Estimate (Biden Administration), Commonly Cited
""")

# CAMBIUM 2022 FOR IRA SCENARIO
filename = 'scc_climate_impact_sensitivity.xlsx'
relative_path = os.path.join("cmu_tare_model", "public_impact", "data_processing", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)
df_climate_impact_scc = pd.read_excel(io=file_path, sheet_name='scc_bounds')

print(f"""
Retrieved data for filename: {filename}
Located at filepath: {file_path}

Loading dataframe ...
Creating lookup dictionary for SCC Lower Bound, Central Estimate, and Upper Bound (2020-2050) ...
-------------------------------------------------------------------------------------------------------
""")

# Create the lookup dictionary
lookup_climate_impact_scc = create_scc_lookup(df_climate_impact_scc)

