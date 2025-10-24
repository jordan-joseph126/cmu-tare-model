import os
import pandas as pd
from typing import Dict

# import from cmu-tare-model package
from config import PROJECT_ROOT

# ====================================================================================================================================================================================
# Set print_verbose to True for detailed output, or False for minimal output
# By default, verbose is set to False because define_scenario_params is imported multiple times in the codebase
# and we don't want to print the same information multiple times.
print_verbose = False
# ====================================================================================================================================================================================


"""
===================================================================================================================================================================================
Future VSL estimates were calculated using the following formula:
VSL_future = VSL_base * (1.01)^(year - base_year)

Where:
VSL_base and VSL_future are both in constant 2023 dollars.
VSL_base = 11.3M USD2021 * (CPI_2023 / CPI_2021) = 11.3M * 1.1244861 = 12.71M USD2023
            (This matches the VSL used in the marginal social cost data files)
base_year = 2023 (our adjustment factors are normalized to 2023, projecting from 2023 to 2050).
We assume a 1% annual growth rate for real income, consistent with the HHS VSL Guidance.

CORRECTED (October 2025):
- Fixed base year normalization from 2024 to 2023 to match MSC data timeframe
- Updated VSL source from EPA 11.0M USD2022 to EPA 11.3M USD2021 to match MSC data
- This ensures consistency between MSC data (based on VSL=12.71M) and adjustment factors
===================================================================================================================================================================================
"""

filename = "vsl_adjustment_factor_2023-2050.xlsx"
relative_path = os.path.join("cmu_tare_model", "data", "marginal_social_costs", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)
df_rcm_msc_data = pd.read_excel(io=file_path, sheet_name='vsl_adjustment_factor')

if print_verbose:
    print(f"""
    ===================================================================================================================================================================================
    HEALTH IMPACTS VSL ADJUSTMENT FACTOR (2023-2050)
    ===================================================================================================================================================================================

    DATAFRAME: HEALTH IMPACTS VSL ADJUSTMENT FACTOR (2023-2050)
        
    {df_rcm_msc_data}
        
    """)

# Create a lookup dictionary with year as the key and vsl_adjustment_factor as the value
lookup_health_vsl_adjustment: Dict[int, float] = df_rcm_msc_data.set_index('year')['vsl_adjustment_factor'].to_dict()

if print_verbose:
    print(f"""
    ===================================================================================================================================================================================
    LOOKUP DICTIONARY: HEALTH IMPACTS (MSC) FROM ELECTRICITY GENERATION
    ===================================================================================================================================================================================

    LOOKUP: VSL ADJUSTMENT FACTOR (2023-2050)
        
    {lookup_health_vsl_adjustment}

    """)
