import os
import pandas as pd

# import functions.tare_setup as tare_setup
from config import PROJECT_ROOT
from cmu_tare_model.constants import EQUIPMENT_SPECS

# HDD factors for different census divisions and years
# Factors for 2022 to 2050
# Define the relative path to the target file
filename = 'aeo_projections_2022_2050.xlsx'
relative_path = os.path.join("cmu_tare_model", "data", "projections", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)

df_hdd_projection_factors = pd.read_excel(io=file_path, sheet_name='hdd_factors_2022_2050')

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")

# Convert the factors dataframe into a lookup dictionary
lookup_hdd_factor = df_hdd_projection_factors.set_index(['census_division']).to_dict('index')

def precompute_hdd_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Precompute heating degree day (HDD) factors for each region and year.

    Parameters:
        df (DataFrame): Input data.

    Returns:
        dict: HDD factors mapped by year and region.
    """
    
    max_lifetime = max(EQUIPMENT_SPECS.values())
    years = range(2024, 2024 + max_lifetime + 1)
    hdd_factors_df = pd.DataFrame(index=df.index, columns=years)

    for year_label in years:
        hdd_factors_df[year_label] = df['census_division'].map(
            lambda x: lookup_hdd_factor.get(x, lookup_hdd_factor['National']).get(year_label, 1.0)
        )

    return hdd_factors_df
