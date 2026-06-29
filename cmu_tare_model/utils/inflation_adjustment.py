import os
import pandas as pd

from config import PROJECT_ROOT

# ==============================================================================================================================================================================
# INFLATION ADJUSTMENT
# ==============================================================================================================================================================================
# Inflate Marginal Social Cost (Damage) Factors using BLS CPI for All Urban Consumers (CPI-U)
# - Series Id:	CUUR0000SA0
# - Not Seasonally Adjusted
# - Series Title:	All items in U.S. city average, all urban consumers, not seasonally adjusted
# - Area:	U.S. city average
# - Item:	All items
# - Base Period:	1982-84=100
# ==============================================================================================================================================================================

# Load the BLS Inflation Data
filename = 'bls_cpiu_2005-2025.xlsx'
relative_path = os.path.join("cmu_tare_model", "data", "inflation_data", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)

# Create a pandas dataframe.
# The workbook is the raw BLS export: a single sheet named 'BLS Data Series'
# with 11 rows of series metadata above the 'Year' / 'Annual' table, so the
# header row is read from position 11.
df_bls_cpiu = pd.read_excel(
    file_path, sheet_name='BLS Data Series', header=11
)

df_bls_cpiu = pd.DataFrame({
    'year': df_bls_cpiu['Year'],
    'cpiu_annual': df_bls_cpiu['Annual']
})

# Obtain the Annual CPIU values for the years of interest
bls_cpi_annual_2008 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2008)].item()
bls_cpi_annual_2018 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2018)].item()
bls_cpi_annual_2020 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2020)].item()
bls_cpi_annual_2023 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2023)].item()
bls_cpi_annual_2024 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2024)].item()
bls_cpi_annual_2025 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2025)].item()

# Precompute constant values
cpi_ratio_2025_2024 = bls_cpi_annual_2025 / bls_cpi_annual_2024  # For adjusting the income data from USD 2024 to USD 2025
cpi_ratio_2025_2023 = bls_cpi_annual_2025 / bls_cpi_annual_2023  # For REMDB Capital Costs inflation adjustment from USD 2023 to USD 2025
cpi_ratio_2025_2020 = bls_cpi_annual_2025 / bls_cpi_annual_2020  # For SCC (reported in USD 2020) inflation adjustment to USD 2025
cpi_ratio_2025_2018 = bls_cpi_annual_2025 / bls_cpi_annual_2018  # ResStock is circa 2018, so assume that the costs are in USD 2018 and need to be inflated to USD 2025
cpi_ratio_2025_2008 = bls_cpi_annual_2025 / bls_cpi_annual_2008  # For EPA VSL and SCC
