import os
import pandas as pd
from typing import Dict

# import from cmu-tare-model package
from config import PROJECT_ROOT

# =======================================================================================================================
# Set print_verbose to True for detailed output, or False for minimal output
# By default, verbose is set to False because define_scenario_params is imported multiple times in the codebase
# and we don't want to print the same information multiple times.
print_verbose = False
# =======================================================================================================================

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CLIMATE CHANGE IMPACT SENSITIVITY: SCC LOOKUP
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def create_scc_lookup(df: pd.DataFrame) -> Dict[str, Dict[int, float]]:
    """
    Create a nested dictionary that maps SCC assumptions ('lower', 'central', 'upper')
    to a year-based lookup of the Social Cost of Carbon (USD 2023).

    Args:
        df (pd.DataFrame): DataFrame containing columns:
            'emissions_year', 'scc_lower_usd2023', 'scc_central_usd2023', 'scc_upper_usd2023'.

    Returns:
        Dict[str, Dict[int, float]]:
            Nested dictionary:
                {
                  "lower":   {year: scc_value, ...},
                  "central": {year: scc_value, ...},
                  "upper":   {year: scc_value, ...}
                }

    Raises:
        KeyError: If required columns are missing from df.
    """
    # Initialize top-level keys for each SCC bound
    lookup_climate_impact_scc = {
        "lower": {},
        "central": {},
        "upper": {}
    }
    
    # Fill in year-specific values under each key
    # Non-trivial iteration that ensures each row is inserted into the correct assumption dict
    for _, row in df.iterrows():
        year = int(row["emissions_year"])
        lookup_climate_impact_scc["lower"][year] = row["scc_lower_usd2023"]
        lookup_climate_impact_scc["central"][year] = row["scc_central_usd2023"]
        lookup_climate_impact_scc["upper"][year] = row["scc_upper_usd2023"]
    
    return lookup_climate_impact_scc

if print_verbose:
    print("""
    -------------------------------------------------------------------------------------------------------
    CLIMATE CHANGE IMPACT SENSITIVITY: SCC LOOKUP
    -------------------------------------------------------------------------------------------------------
    LOWER BOUND: IWG 2021, 5% Discount Rate (Trump Previously Used 3-7% Discount Rate)
    CENTRAL ESTIMATE: IWG 2021, 3% Discount Rate (Obama Administration, Pre-2017)
    UPPER BOUND: Recent EPA Central Estimate (Biden Administration), Commonly Cited
    """)

filename = 'scc_climate_impact_sensitivity.xlsx'
relative_path = os.path.join("cmu_tare_model", "data", "marginal_social_costs", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)
df_climate_impact_scc = pd.read_excel(io=file_path, sheet_name='scc_bounds')

if print_verbose:
    print(f"""
    Retrieved data for filename: {filename}
    Located at filepath: {file_path}

    Loading dataframe ...
    Creating lookup dictionary for SCC Lower Bound, Central Estimate, and Upper Bound (2020-2050) ...
    -------------------------------------------------------------------------------------------------------
    """)

# Create the lookup dictionary
lookup_climate_impact_scc = create_scc_lookup(df_climate_impact_scc)

