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

# C:\Users\14128\Research\cmu-tare-model\cmu_tare_model\data\inflation_data\bls_cpiu_2005-2023.xlsx# Load the BLS Inflation Data
filename = 'bls_cpiu_2005-2023.xlsx'
relative_path = os.path.join("cmu_tare_model", "data", "inflation_data", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")

# Create a pandas dataframe
df_bls_cpiu = pd.read_excel(file_path, sheet_name='bls_cpiu')

df_bls_cpiu = pd.DataFrame({
    'year': df_bls_cpiu['Year'],
    'cpiu_annual': df_bls_cpiu['Annual']
})
print(df_bls_cpiu)

# Obtain the Annual CPIU values for the years of interest
bls_cpi_annual_2006 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2006)].item()
bls_cpi_annual_2008 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2008)].item()
bls_cpi_annual_2010 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2010)].item()
bls_cpi_annual_2013 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2013)].item()
bls_cpi_annual_2018 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2018)].item()
bls_cpi_annual_2019 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2019)].item()
bls_cpi_annual_2020 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2020)].item()
bls_cpi_annual_2021 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2021)].item()
bls_cpi_annual_2022 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2022)].item()
bls_cpi_annual_2023 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2023)].item()

# Precompute constant values
cpi_ratio_2023_2023 = bls_cpi_annual_2023 / bls_cpi_annual_2023
cpi_ratio_2023_2022 = bls_cpi_annual_2023 / bls_cpi_annual_2022
cpi_ratio_2023_2021 = bls_cpi_annual_2023 / bls_cpi_annual_2021  # For EPA VSL (11.3M USD-2021)
cpi_ratio_2023_2020 = bls_cpi_annual_2023 / bls_cpi_annual_2020  # For SCC
cpi_ratio_2023_2019 = bls_cpi_annual_2023 / bls_cpi_annual_2019 
cpi_ratio_2023_2018 = bls_cpi_annual_2023 / bls_cpi_annual_2018 
cpi_ratio_2023_2013 = bls_cpi_annual_2023 / bls_cpi_annual_2013
cpi_ratio_2023_2010 = bls_cpi_annual_2023 / bls_cpi_annual_2010
cpi_ratio_2023_2008 = bls_cpi_annual_2023 / bls_cpi_annual_2008  # For EPA VSL and SCC
cpi_ratio_2023_2006 = bls_cpi_annual_2023 / bls_cpi_annual_2006  # For CACES RCM Marginal Social Costs 
