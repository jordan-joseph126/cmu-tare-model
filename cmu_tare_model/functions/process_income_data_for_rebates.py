import pandas as pd
import os

# Get the current working directory of the project
# project_root = os.path.abspath(os.getcwd())
project_root = "C:\\Users\\14128\\Research\\cmu-tare-model"
print(f"Project root directory: {project_root}")

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
INCOME GROUPS, PERCENT AMI, AND REBATE ELIGIBILITY
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
## Area Median Income Data Used to determine LMI Designation and IRA Rebates Eligibility/Amount
### PUMA Median Income
# Collect Area Median Income Data at PUMA-resolution
filename = "nhgis0003_ds261_2022_puma.csv"
relative_path = os.path.join(r"equity_data", filename)
file_path = os.path.join(project_root, relative_path)

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")
print("\n")

df_puma_medianIncome = pd.read_csv(file_path, encoding='ISO-8859-1')
# df_puma_medianIncome = df_puma_medianIncome.drop(0)
df_puma_medianIncome = df_puma_medianIncome.reset_index(drop=True)

cols_interest = ['GISJOIN', 'STUSAB', 'PUMAA', 'NAME_E', 'AP2PE001', 'AP2PM001']
df_puma_medianIncome = df_puma_medianIncome[cols_interest]
df_puma_medianIncome = df_puma_medianIncome.rename(columns={"GISJOIN": "gis_joinID_puma", "STUSAB": "state_abbrev", "PUMAA": "puma_code", "NAME_E": "name_estimate", "AP2PE001": "median_income_USD2022", "AP2PM001": "median_income_USD2022_marginOfError"})
df_puma_medianIncome['median_income_USD2023'] = round((df_puma_medianIncome['median_income_USD2022'] * cpi_ratio_2023_2022), 2)
df_puma_medianIncome
### County Median Income
# Collect Area Median Income Data at PUMA-resolution
filename = "nhgis0005_ds261_2022_county.csv"
relative_path = os.path.join(r"equity_data", filename)
file_path = os.path.join(project_root, relative_path)

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")
print("\n")

df_county_medianIncome = pd.read_csv(file_path, encoding='ISO-8859-1')
# df_county_medianIncome = df_county_medianIncome.drop(0)
df_county_medianIncome = df_county_medianIncome.reset_index(drop=True)

cols_interest = ['GISJOIN', 'STUSAB', 'COUNTYA', 'NAME_E', 'AP2PE001', 'AP2PM001']
df_county_medianIncome = df_county_medianIncome[cols_interest]
df_county_medianIncome = df_county_medianIncome.rename(columns={"GISJOIN": "gis_joinID_county", "STUSAB": "state_abbrev", "COUNTYA": "county_code", "NAME_E": "name_estimate", "AP2PE001": "median_income_USD2022", "AP2PM001": "median_income_USD2022_marginOfError"})
df_county_medianIncome['median_income_USD2023'] = round((df_county_medianIncome['median_income_USD2022'] * cpi_ratio_2023_2022), 2)
df_county_medianIncome
### State Median Income
# Collect Area Median Income Data at PUMA-resolution
filename = "nhgis0004_ds261_2022_state.csv"
relative_path = os.path.join(r"equity_data", filename)
file_path = os.path.join(project_root, relative_path)

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")
print("\n")

df_state_medianIncome = pd.read_csv(file_path, encoding='ISO-8859-1')
# df_state_medianIncome = df_state_medianIncome.drop(0)
df_state_medianIncome = df_state_medianIncome.reset_index(drop=True)

cols_interest = ['GISJOIN', 'STUSAB','STATEA', 'NAME_E', 'AP2PE001', 'AP2PM001']
df_state_medianIncome = df_state_medianIncome[cols_interest]
df_state_medianIncome = df_state_medianIncome.rename(columns={"GISJOIN": "gis_joinID_state", "STUSAB": "state_abbrev", "STATEA": "state_code", "NAME_E": "name_estimate", "AP2PE001": "median_income_USD2022", "AP2PM001": "median_income_USD2022_marginOfError"})
df_state_medianIncome['median_income_USD2023'] = round((df_state_medianIncome['median_income_USD2022'] * cpi_ratio_2023_2022), 2)
df_state_medianIncome

