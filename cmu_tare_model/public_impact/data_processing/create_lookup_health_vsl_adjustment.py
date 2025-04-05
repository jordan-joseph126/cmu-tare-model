import os
import pandas as pd
from typing import Dict

# import from cmu-tare-model package
from config import PROJECT_ROOT

"""
======================================================================================================================
Future VSL estimates were calculated using the following formula:
VSL_future = VSL_base * (1.01)^(year - base_year)

Where: 
VSL_base and VSL_future are both in constant 2023 dollars. 
VSL_base = 11.0M * (CPI_2023 / CPI_2022) = 12.71 in 2023 dollars.
base_year = 2023 (our adjustment factors project from 2024 to 2050, because we have a 2024 start year).
We assume a 1% annual growth rate for real earnings, consistent with the HHS VSL Guidance.
======================================================================================================================
"""

filename = "vsl_adjustment_factor_2023-2050.xlsx"
relative_path = os.path.join("cmu_tare_model", "data", "marginal_social_costs", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)
df_rcm_msc_data = pd.read_excel(io=file_path, sheet_name='vsl_adjustment_factor')

print(f"""
======================================================================================================================
HEALTH IMPACTS VSL ADJUSTMENT FACTOR (2023-2050)
======================================================================================================================

DATAFRAME: HEALTH IMPACTS VSL ADJUSTMENT FACTOR (2023-2050)
      
{df_rcm_msc_data}
      
""")

# Create a lookup dictionary with year as the key and vsl_adjustment_factor as the value
lookup_health_vsl_adjustment: Dict[int, float] = df_rcm_msc_data.set_index('year')['vsl_adjustment_factor'].to_dict()

print(f"""
======================================================================================================================
LOOKUP DICTIONARY: HEALTH IMPACTS (MSC) FROM ELECTRICITY GENERATION
======================================================================================================================

LOOKUP: VSL ADJUSTMENT FACTOR (2023-2050)
      
{lookup_health_vsl_adjustment}

""")
