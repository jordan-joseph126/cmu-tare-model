#!/usr/bin/env python
# coding: utf-8

# # Baseline Scenario

# In[21]:


import os
from pathlib import Path
import subprocess

import os
from pathlib import Path
import subprocess

# Define the file path to the Jupyter notebook
file_path = Path(r'C:\Users\14128\Research\cmu-tare-model\tare_model_scenario_baseline_v2.1.ipynb')

# Check if the notebook file exists
if file_path.is_file():
    print("File exists")
    
    # Convert the notebook to a Python script
    command_convert = f'jupyter nbconvert --to script "{file_path}"'
    subprocess.run(command_convert, shell=True)
    
    # Get the converted script path
    script_path = file_path.with_suffix('.py')
    
    # Check if the conversion was successful
    if script_path.is_file():
        print(f"Converted script path: {script_path}")
        
        # Use the %run magic command to execute the script
        command_run = f'%run -i "{script_path}"'
        print(f'Executing command: {command_run}')
        get_ipython().run_line_magic('run', f'-i "{script_path}"')
        
        print("Baseline Scenario - Model Run Complete")
        
        # Clean up the converted script
        if script_path.is_file():
            os.remove(script_path)
            print("Cleaned up the converted script.")
    else:
        print("Conversion to script failed. Script file does not exist.")
else:
    print("File does not exist")


# In[17]:


# REPLACE THE FILE PATH WITH YOUR OWN!
# %run -i r"C:\Users\14128\Research\cmu-tare-model\tare_model_functions_v2.1.py"
import os
from pathlib import Path
import subprocess

# Define the file path using pathlib for better path handling
file_path = Path(r'C:\Users\14128\Research\cmu-tare-model\tare_model_functions_v2.1.ipynb')

# Optional: Run the converted Python script
command_run = f'python "{script_path}"'
subprocess.run(command_run, shell=True)

# if file_path.is_file():
#     print("File exists")
    
#     # Convert the notebook to a Python script using nbconvert
#     command_convert = f'jupyter nbconvert --to script "{file_path}"'
#     subprocess.run(command_convert, shell=True)
    
#     # Get the converted script path
#     script_path = file_path.with_suffix('.py')
    
#     if script_path.is_file():
#         print(f"Converted script path: {script_path}")
        
#         # Optional: Run the converted Python script
#         command_run = f'python "{script_path}"'
#         subprocess.run(command_run, shell=True)
    
#     else:
#         print("Conversion to script failed. Script file does not exist.")
# else:
#     print("File does not exist")


# In[15]:


get_ipython().run_line_magic('run', '-i r"C:\\Users\\14128\\Research\\cmu-tare-model\\tare_model_functions_v2.1.py"')
print("Loaded All TARE Model Functions")


# In[ ]:


# REPLACE THE FILE PATH WITH YOUR OWN!
# Program Directory ---> C:\Users\14128\Research\cmu_tare_model
program_directory = str(input("Copy and paste filepath for main folder here: "))

# Output Folder Path --> C:\Users\14128\Research\cmu_tare_model_outputs
output_folder_path = str(input("Enter the filepath for the output folder: "))

# Results Export ------> C:\Users\14128\Research\cmu_tare_model_outputs
# save_figure_directory = str(input("Enter the filepath for the folder you are saving figures: "))
save_figure_directory = str(output_folder_path)


# # Baseline: 

# ## Simulate Residential Energy Consumption
# 
# 

# In[ ]:


# Measure Package 0: Baseline
menu_mp = 0
input_mp = 'baseline'


# In[ ]:


# The ``inline`` flag will use the appropriate backend to make figures appear inline in the notebook.  
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np

# `plt` is an alias for the `matplotlib.pyplot` module
import matplotlib.pyplot as plt

# import seaborn library (wrapper of matplotlib)
import seaborn as sns
sns.set(style="darkgrid")

# For regex, import re
import re


# In[ ]:


from datetime import datetime

# Get the current datetime
# Start the timer
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# In[ ]:


# Import pandas and change column/row display restriction
pd.set_option("display.max_columns", None)
# pd.reset_option('display.max_columns')
# pd.set_option('display.max_rows', None)
pd.reset_option('display.max_rows')


# In[ ]:


print("-------------------------------------------------------------------------------------------------------")
print("""
Welcome to the Trade-off Analysis of residential Retrofits for energy Equity Tool (TARE Model)
Let's start by reading the data from the NREL EUSS Database.

Make sure that the zipped folders stay organized as they are once unzipped.
If changes are made to the file path, then the program will not run properly.""")
print("-------------------------------------------------------------------------------------------------------")
print("\n")


# In[ ]:


print("-------------------------------------------------------------------------------------------------------")
print("BASELINE (Measure Package 0)")
print("-------------------------------------------------------------------------------------------------------")


# In[ ]:


# # Program Directory --> C:\Users\14128\Research\cmu_tare_model
# program_directory = str(input("Copy and paste filepath for main folder here: "))
# print("\n")

input_mp = 'baseline'
filename = input_mp + "_" + "metadata_and_annual_results.csv"
print(f"Retrieving data for filename: {filename}")

change_directory = "euss_data\\resstock_amy2018_release_1.1\\state"
filepath = str(program_directory) + "\\" + str(change_directory) + "\\" + str(filename) 

print(f"Located at filepath: {filepath}")


# In[ ]:


# Results export directory --> C:\Users\14128\Research\cmu_tare_model_outputs
results_export_directory = str(save_figure_directory)


# ### Data Filters: Only occupied units and Single Family Homes

# In[ ]:


print("-------------------------------------------------------------------------------------------------------")
print("Data Filters: Only occupied units and Single Family Homes")
print("-------------------------------------------------------------------------------------------------------")


# In[ ]:


# Fix DtypeWarning error in columns:
# 'in.neighbors', 'in.geometry_stories_low_rise', 'in.iso_rto_region', 'in.pv_orientation', 'in.pv_system_size'
columns_to_string = {11: str, 61: str, 121: str, 103: str, 128: str, 129: str}
df_euss_am_baseline = pd.read_csv(filepath, dtype=columns_to_string)
occupancy_filter = df_euss_am_baseline['in.vacancy_status'] == 'Occupied'
df_euss_am_baseline = df_euss_am_baseline.loc[occupancy_filter]

# Filter for single family home building type
house_type_list = ['Single-Family Attached', 'Single-Family Detached']
house_type_filter = df_euss_am_baseline['in.geometry_building_type_recs'].isin(house_type_list)
df_euss_am_baseline = df_euss_am_baseline.loc[house_type_filter]

df_euss_am_baseline


# In[ ]:


while True:
    try:
        menu_state = str(input("""
Would you like to filter for a specific state's data? Please enter one of the following:
N. I'd like to analyze all of the United States.
Y. I'd like to filter data for a specific state. """)).upper()

        if menu_state == 'N':
            print("You chose to analyze all of the United States.")
            input_state = 'National'
            break

        elif menu_state == 'Y':
            while True:
                input_state = str(input("""
Which state would you like to analyze data for?
Please enter the two-letter abbreviation: """)).upper()
                state_filter = df_euss_am_baseline['in.state'] == input_state

                if state_filter.any():
                    print(f"""
You chose to filter for: {input_state}""")
                    df_euss_am_baseline = df_euss_am_baseline.loc[state_filter, :]
                    break
                else:
                    print("""
Invalid state abbreviation. Please try again.""")

            while True:
                try:
                    print("""
To accurately characterize load profile, it is recommended to select subsets of data with >= 1000 models (~240,000 representative dwelling units).

The following cities (number of models also shown) are available for this state:
                    """)
                    print(df_euss_am_baseline['in.city'].value_counts())
                    
                    menu_city = str(input("""
Would you like to filter a subset of city-level data? Please enter one of the following:
N. I'd like to analyze all of my selected state.
Y. I'd like to filter by city in the state.""")).upper()

                    if menu_city == 'N':
                        print(f"""
You chose to analyze all of state: {input_state}""")
                        break

                    elif menu_city == 'Y':
                        while True:
                            input_cityFilter = str(input("""
Please enter the city name ONLY (e.g., Pittsburgh): """))
                            city_filter = df_euss_am_baseline['in.city'] == (input_state + ", " + input_cityFilter)

                            if city_filter.any():
                                print(f"""
You chose to filter for: {input_state}, {input_cityFilter}""")
                                df_euss_am_baseline = df_euss_am_baseline.loc[city_filter, :]
                                break
                            else:
                                print("""
Invalid city name. Please try again.""")

                        break

                    else:
                        print("""
Please enter a valid option.""")

                except Exception as e:
                    print("""
Invalid input. Please try again.""")

            break

        else:
            print("""
Please enter a valid option
            """)

    except Exception as e:
        print("""
Invalid input. Please try again.""")
print("\n")
# df_euss_am_baseline


# In[ ]:


print("-------------------------------------------------------------------------------------------------------")
print("BASELINE CONSUMPTION")
print("-------------------------------------------------------------------------------------------------------")
print("\n")


# ## Baseline Energy Consumption

# In[ ]:


print("-------------------------------------------------------------------------------------------------------")
print("Baseline Consumption:")
print("-------------------------------------------------------------------------------------------------------")


# In[ ]:


# df_baseline_enduse(df_baseline, df_enduse, category, fuel_filter='Yes', tech_filter='Yes')
df_euss_am_baseline_home = df_enduse_refactored(df_baseline = df_euss_am_baseline,
                                                fuel_filter = 'Yes',
                                                tech_filter = 'Yes')
df_euss_am_baseline_home


# ## Public Perspective: Monetized Marginal Damages from Emissions

# ### Step 1: Calculate emissions factors for different fuel sources

# ### Marginal Emissions Factors
# #### Electricity
# - STATE Regional Aggregation is what is used in the Parth Analysis 
# - "Marginal Emissions Factors for Electricity"
# - Factor Type: Marginal
# - Calculation Method: Regression
# - Metric: Emissions [kg/MWh]")
# - Predictor: Year")
# - Pollutants: SO2, NOx, PM2.5, CO2")
# #### Fossil Fuels
# - NOx, SO2, CO2: 
#     - RESNET Table 7.1.2 Emissions Factors for Household Combustion Fuels
#     - Source: https://www.resnet.us/wp-content/uploads/ANSIRESNETICC301-2022_resnetpblshd.pdf
#     - All factors are in units of lb/Mbtu so energy consumption in kWh need to be converted to kWh 
#     - (1 lb / Mbtu) * (1 Mbtu / 1x10^6 Btu) * (3412 Btu / 1 kWh)
# - PM2.5: 
#     - A National Methodology and Emission Inventory for Residential Fuel Combustion
#     - Source: https://www3.epa.gov/ttnchie1/conference/ei12/area/haneke.pdf

# In[ ]:


print("\n")
print("""
-------------------------------------------------------------------------------------------------------
Public Perspective: Monetized Marginal Damages from Emissions
-------------------------------------------------------------------------------------------------------
Step 1: Calculate emissions factors for different fuel sources
- Electricity
- Natural Gas
- Fuel Oil 
- Propane
-------------------------------------------------------------------------------------------------------
""")


# In[ ]:


print("-------------------------------------------------------------------------------------------------------")
print("Calculate Emissions Factors: ELECTRICITY")
print("-------------------------------------------------------------------------------------------------------")
print("""
Electricity Marginal Emissions Factors:
- STATE Regional Aggregation is what is used in the Parth Analysis 
- "Marginal Emissions Factors for Electricity"
- Factor Type: Marginal
- Calculation Method: Regression
- Metric: Emissions [kg/MWh]")
- Predictor: Year")
- Pollutants: SO2, NOx, PM2.5, CO2")
""")
print("-------------------------------------------------------------------------------------------------------")

# Program Directory --> C:\Users\14128\Research\cmu_tare_model
# # C:\Users\14128\Research\cmu_tare_model\margEmis_electricity
filename = 'Generation-MARREG-EMIT-state-byYear.csv'
print(f"Retrieving data for filename: {filename}")

change_directory = "margEmis_electricity"
filepath = str(program_directory) + "\\" + str(change_directory) + "\\" + str(filename) 

print(f"Located at filepath: {filepath}")

df_margEmissions = pd.read_csv(filepath, index_col=0)

# Convert from kg/MWh to lb/kWh
# Obtain value from the CSV file and convert to lbs pollutant per kWh 
df_margEmis_electricity = pd.DataFrame({
    'state': df_margEmissions['region'],
    'fuel_type': 'electricity',
    'pollutant': df_margEmissions['pollutant'],
    'value': df_margEmissions['factor'] * (2.20462/1) * (1/1000),
    'unit': '[lb/kWh]'
})

df_margEmis_electricity


# In[ ]:


print("-------------------------------------------------------------------------------------------------------")
print("Calculate Emissions Factors: FOSSIL FUELS")
print("-------------------------------------------------------------------------------------------------------")
print("""
Fossil Fuels (Natural Gas, Fuel Oil, Propane):
- NOx, SO2, CO2: 
    - RESNET Table 7.1.2 Emissions Factors for Household Combustion Fuels
    - Source: https://www.resnet.us/wp-content/uploads/ANSIRESNETICC301-2022_resnetpblshd.pdf
    - All factors are in units of lb/Mbtu so energy consumption in kWh need to be converted to kWh 
    - (1 lb / Mbtu) * (1 Mbtu / 1x10^6 Btu) * (3412 Btu / 1 kWh)
- PM2.5: 
    - A National Methodology and Emission Inventory for Residential Fuel Combustion
    - Source: https://www3.epa.gov/ttnchie1/conference/ei12/area/haneke.pdf
""")
print("-------------------------------------------------------------------------------------------------------")

fuelOil_factors = calculate_fossilFuel_emission_factor("fuelOil", 0.0015, 0.1300, 0.83, 161.0, 1000, 138500)
naturalGas_factors = calculate_fossilFuel_emission_factor("naturalGas", 0.0006, 0.0922, 1.9, 117.6, 1000000, 1039)
propane_factors = calculate_fossilFuel_emission_factor("propane", 0.0002, 0.1421, 0.17, 136.6, 1000, 91452)

all_factors = {**fuelOil_factors, **naturalGas_factors, **propane_factors}

df_margEmis_factors = pd.DataFrame.from_dict(all_factors, orient="index", columns=["value"])
df_margEmis_factors.reset_index(inplace=True)
df_margEmis_factors.columns = ["pollutant", "value"]
df_margEmis_factors[["fuel_type", "pollutant"]] = df_margEmis_factors["pollutant"].str.split("_", expand=True)
df_margEmis_factors["unit"] = "[lb/kWh]"

# Add the 'state' column and assign 'National' to every row
df_margEmis_factors = df_margEmis_factors.assign(state='National')

df_margEmis_factors = df_margEmis_factors[["state", "fuel_type", "pollutant", "value", "unit"]]
df_margEmis_factors


# ### Step 2: Adjust Natural Gas & Electricity Emissions Factors for Natural Gas Leakage

# In[ ]:


print("-------------------------------------------------------------------------------------------------------")
print("Step 2: Adjust Natural Gas & Electricity Emissions Factors for Natural Gas Leakage")
print("-------------------------------------------------------------------------------------------------------")
print("""
Natural Gas (Deetjen et al.): 
"To account for the natural gas infrastructure's leakage of the greenhouse gas methane, 
we estimate the amount of methane leaked per therm of natural gas consumed for heating and 
convert to CO2-equivalent emissions via the GWP of methane. We assume that for every therm of 
natural gas consumed for heating, 0.023 therms of methane escape to the atmosphere [28]. 
Using the energy density of natural gas, we convert from therms to kilograms and multiply 
by 28—the GWP of methane [29]—to calculate a rate of 1.27 kg CO2-equivalent per therm of 
consumed natural gas."

Electricity NERC Regions (Deetjen et al): 
"To account for the natural gas infrastructure's leakage of the greenhouse gas methane, 
we estimate the amount of methane leaked per MWh of electricity generation in each NERC 
region and convert to CO2-equivalent emissions via the global warming potential (GWP) of methane. 
For example, we find that in 2017, the states comprising the western region (WECC) of 
the US electric grid consumed 1.45 million MMcf of natural gas in the power sector [27]. 
We assume that for every MMcf of consumed natural gas, 0.023 MMcf of methane is leaked into 
the atmosphere [28]. By multiplying that leakage rate by the 1.45 million MMcf of consumed 
natural gas, converting to tonnes, and multiplying by a GWP of 28 [29], we estimate 
that the 2017 WECC power sector contributed to methane leakage amounting to 18.6 Mt CO2-equivalent.
By dividing this 18.6 Mt by the 724 TWh of the WECC states' generated electricity [27], we 
calculate a methane leakage rate factor of 25.7 kg MWh−1. In the same manner, we calculate the 
methane leakage rate factors for the other NERC regions. We use the 100 years GWP value of 28 
for methane. Although there have been proposals to use 20 years GWP values, recent research 
shows that the benefits of this alternative 20 years time from are overstated [30]."
""")
print("-------------------------------------------------------------------------------------------------------")

# Program Directory --> C:\Users\14128\Research\cmu_tare_model
# # C:\Users\14128\Research\cmu_tare_model\margEmis_electricity
# "C:\Users\14128\Research\cmu_tare_model\margEmis_electricity\natural_gas_leakage_rate.csv"
filename = 'natural_gas_leakage_rate.csv'
print(f"Retrieving data for filename: {filename}")

change_directory = "margEmis_electricity"
filepath = str(program_directory) + "\\" + str(change_directory) + "\\" + str(filename) 
print(f"Located at filepath: {filepath}")

df_naturalGas_leakage_rate = pd.read_csv(filepath)

state_abbreviations = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'District of Columbia': 'DC',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}
# Map full state names to abbreviations
df_naturalGas_leakage_rate['state'] = df_naturalGas_leakage_rate['state_name'].map(state_abbreviations)

# thousand Mcf * (0.023 Mcf leak/1 Mcf) * (19.3 tonnes/1000 Mcf) * (1000 kg/1 tonne) * (2.205 lb/1 kg)) / (thousand MWh * (1000 MWh/thousand MWh)) 
df_naturalGas_leakage_rate['naturalGas_leakage_lbCH4_perMWh'] = (df_naturalGas_leakage_rate['naturalGas_electricity_generation'] * (0.023/1) * (19.3/1) * (1000/1) * (2.205/1)) / (df_naturalGas_leakage_rate['net_generation'] * (1000/1)) 

# (lb CH4/MWh) * (28 lb CO2e/1 lb CH4)
df_naturalGas_leakage_rate['naturalGas_leakage_lbCO2e_perMWh'] = df_naturalGas_leakage_rate['naturalGas_leakage_lbCH4_perMWh'] * (28/1)

# (lb CO2e/MWh) * (1 MWh / 1000 kWh)
df_naturalGas_leakage_rate['naturalGas_leakage_lbCO2e_perkWh'] = df_naturalGas_leakage_rate['naturalGas_leakage_lbCO2e_perMWh'] * (1/1000)
df_naturalGas_leakage_rate


# In[ ]:


# NATURAL GAS LEAKAGE: NATURAL GAS USED IN ELECTRICITY GENERATION
if 'naturalGas_leakage_lbCO2e_perkWh' in df_margEmis_electricity.columns:
    df_margEmis_electricity.drop(columns=['naturalGas_leakage_lbCO2e_perkWh'], inplace=True)

df_margEmis_electricity = df_margEmis_electricity.merge(
    df_naturalGas_leakage_rate[['state', 'naturalGas_leakage_lbCO2e_perkWh']],
    how='left',  # Use a left join to keep all rows from df_margEmis_electricity
    on=['state']  # Merge on the 'state' column
)
# Set 'naturalGas_leakage_lbCO2e_perkWh' to zero where 'pollutant' is not 'co2'
df_margEmis_electricity.loc[df_margEmis_electricity['pollutant'] != 'co2', 'naturalGas_leakage_lbCO2e_perkWh'] = 0.0

# Calculate adjusted marginal emissions factore with natural gas fugitive emissions
df_margEmis_electricity['margEmis_factor_adjusted'] = df_margEmis_electricity['value'] + df_margEmis_electricity['naturalGas_leakage_lbCO2e_perkWh'] 

# Create a factor to multiply marginal damages by
df_margEmis_electricity['naturalGas_leakage_factor'] = df_margEmis_electricity['margEmis_factor_adjusted'] / df_margEmis_electricity['value']

# Reorder columns
df_margEmis_electricity = df_margEmis_electricity[['state', 'fuel_type', 'pollutant', 'value', 'unit', 'naturalGas_leakage_lbCO2e_perkWh', 'margEmis_factor_adjusted', 'naturalGas_leakage_factor']]
df_margEmis_electricity


# In[ ]:


# NATURAL GAS LEAKAGE: NATURAL GAS INFRASTRUCTURE
# leakage rate for natural gas infrastructure
# 1 Therm = 29.30 kWh --> 1.27 kg CO2e/therm * (1 therm/29.30 kWh) = 0.043 kg CO2e/kWh = 0.095 lb CO2e/kWh
df_margEmis_factors['naturalGas_leakage_lbCO2e_perkWh'] = 0.095

# Set 'naturalGas_leakage_lbCO2e_perkWh' to zero where 'pollutant' is not 'co2'
df_margEmis_factors.loc[df_margEmis_factors['pollutant'] != 'co2', 'naturalGas_leakage_lbCO2e_perkWh'] = 0.0

# Set 'naturalGas_leakage_lbCO2e_perkWh' to zero where 'fuel_type' is not 'naturalGas'
df_margEmis_factors.loc[df_margEmis_factors['fuel_type'] != 'naturalGas', 'naturalGas_leakage_lbCO2e_perkWh'] = 0.0

# Calculate adjusted marginal emissions factor with natural gas fugitive emissions
df_margEmis_factors['margEmis_factor_adjusted'] = df_margEmis_factors['value'] + df_margEmis_factors['naturalGas_leakage_lbCO2e_perkWh'] 

# Create a factor to multiply marginal damages by
df_margEmis_factors['naturalGas_leakage_factor'] = df_margEmis_factors['margEmis_factor_adjusted'] / df_margEmis_factors['value']

# Reorder columns
df_margEmis_factors = df_margEmis_factors[['state', 'fuel_type', 'pollutant', 'value', 'unit', 'naturalGas_leakage_lbCO2e_perkWh', 'margEmis_factor_adjusted', 'naturalGas_leakage_factor']]
df_margEmis_factors


# In[ ]:


# Append df_margEmissions_electricity to df_margEmis_factors
# This produces a dataframe of marginal emissions rates for various fuel types
df_margEmis_factors = pd.concat([df_margEmis_factors, df_margEmis_electricity], ignore_index=True)
df_margEmis_factors


# ### Step 3: Quantify monitized damages using EASIUR Marginal Social Cost Factors

# **STEPS to obtain Marginal Damage Factors through Batch Conversion:**
# - Obtain all of the dwelling unit latitude and longitude values from the metadata columns
# - Make a new dataframe of just the longitude and latitude values 
#     - Make sure that the order is (longitude, latitude)
#     - Do not include the index or column name when exporting 
# - Export the CSV
# - **Upload csv to EASIUR Website:**
#     - Website: https://barney.ce.cmu.edu/~jinhyok/easiur/online/
#     - Dollar Year: 2010 (NOTE THERE IS NO 2018 OPTION, NEEDS TO BE 2010 OR EARLIER)
#     - Income Year: 2018 (SAME AS AMY2018, HOUSING CHARACTERISTICS DATA)
#     - Population Year: 2018 (SAME AS AMY2018, HOUSING CHARACTERISTICS DATA)
# - Download the file and put it in the 'easiur_batchConversion_download' folder
# - Copy and paste the name of the file EASIUR generated when prompted
# - Copy and paste the name of the filepath for the 'easiur_batchConversion_download' folder when prompted
# - **Match up the longitude and latitudes for each dwelling unit with the selected damages**

# In[ ]:


print('''
-------------------------------------------------------------------------------------------------------
Step 3: Quantify monitized damages using EASIUR Marginal Social Cost Factors
-------------------------------------------------------------------------------------------------------
**STEPS to obtain Marginal Damage Factors through Batch Conversion:**
- **Upload csv to EASIUR Website:**
    - Website: https://barney.ce.cmu.edu/~jinhyok/easiur/online/
    - Dollar Year: 2010 (NOTE THERE IS NO 2018 OPTION, NEEDS TO BE 2010 OR EARLIER)
    - Income Year: 2018 (SAME AS AMY2018, HOUSING CHARACTERISTICS DATA)
    - Population Year: 2018 (SAME AS AMY2018, HOUSING CHARACTERISTICS DATA)
- Download the file and put it in the 'easiur_batchConversion_download' folder
- Copy and paste the name of the file EASIUR generated when prompted
- Copy and paste the name of the filepath for the 'easiur_batchConversion_download' folder when prompted
- **Match up the longitude and latitudes for each dwelling unit with the selected damages**
-------------------------------------------------------------------------------------------------------
'''
)

from datetime import datetime

# Create a dataframe containing just the longitude and Latitude
df_EASIUR_batchConversion = pd.DataFrame({
    'Longitude':df_euss_am_baseline['in.weather_file_longitude'],
    'Latitude':df_euss_am_baseline['in.weather_file_latitude'],
})

# Drop duplicate rows based on 'Longitude' and 'Latitude' columns
df_EASIUR_batchConversion.drop_duplicates(subset=['Longitude', 'Latitude'], keep='first', inplace=True)

# Create a location ID for the name of the batch conversion file
while True:
    if menu_state == 'N':
        location_id = 'National'
        print("You chose to analyze all of the United States.")
        break
    elif menu_state == 'Y':
        if menu_city == 'N':
            try:
                location_id = str(input_state)
                print(f"Location ID is: {location_id}")
                break
            except ValueError:
                print("Invalid input for state!")
        elif menu_city == 'Y':
            try:
                location_id = input_cityFilter.replace(', ', '_').strip()
                print(f"Location ID is: {location_id}")
                break
            except AttributeError:
                print("Invalid input for city filter!")
        else:
            print("Incorrect state or city filter assignment!")
    else:
        print("Invalid data location. Check your inputs at the beginning of this notebook!")
        
# Format the name of the exported batch conversion file using the location ID
current_datetime = datetime.now()
formatted_date = current_datetime.strftime("%Y-%m-%d_%H-%M")
filename = f"{location_id}_{formatted_date}.csv"
print(f"Retrieving data for filename: {filename}")

# Change the directory to the upload folder and export the file
change_directory = "margDamages_EASIUR\\easiur_batchConversion_upload"
filepath = str(program_directory) + "\\" + str(change_directory) + "\\" + str(filename) 
print(f"Located at filepath: {filepath}")
df_EASIUR_batchConversion.to_csv(filepath, header=False, index=False)


# ### Fossil Fuels: EASIUR Marginal Damage (Social Cost) Factors Info
# - Factor Type: Marginal Social Cost
# - Calculation Method: Regression
# - Metric: Marginal Social Cost [USD per metric ton]
# - Dollar Year: 2010
# - Income Year: 2018
# - Population Year: 2018
# - Aggregation: Longitude, and Latitude Coordinates
# - Pollutants: Primary PM2.5, Sulfur Dioxide (SO2), Nitrogen Oxides (NOx), Ammonia (NH3)
# - Elevation (Ground, 150m, 300m) and Seasons (Winter, Spring, Summer, Fall)

# In[ ]:


print("""
-------------------------------------------------------------------------------------------------------
Information for EASIUR Marginal Damage (Social Cost) Factors
-------------------------------------------------------------------------------------------------------
- Factor Type: Marginal Social Cost
- Calculation Method: Regression
- Metric: Marginal Social Cost [$/metric ton]
- Dollar Year: 2010
- Income Year: 2018
- Population Year: 2018
- Aggregation: Longitude, and Latitude Coordinates
- Pollutants: Primary PM2.5, Sulfur Dioxide (SO2), Nitrogen Oxides (NOx), Ammonia (NH3)
- Elevation (Ground, 150m, 300m) and Seasons (Winter, Spring, Summer, Fall)
-------------------------------------------------------------------------------------------------------
"""
)

# Program Directory --> C:\Users\14128\Research\cmu_tare_model
# # C:\Users\14128\Research\cmu_tare_model\margDamages_EASIUR\easiur_batchConversion_download
filename = str(input("Copy and paste the name of the file that EASIUR generated here: "))
print("\n")

filename = filename + ".csv"
print(f"Retrieving data for filename: {filename}")

change_directory = "margDamages_EASIUR\\easiur_batchConversion_download"
filepath = str(program_directory) + "\\" + str(change_directory) + "\\" + str(filename) 

print(f"Located at filepath: {filepath}")
df_margSocialCosts = pd.read_csv(filepath)

# Convert from kg/MWh to lb/kWh
# Obtain value from the CSV file and convert to lbs pollutant per kWh 
# Inflate from 2010 to 2018

# Define df_margSocialCosts_EASIUR DataFrame first
df_margSocialCosts_EASIUR = pd.DataFrame({
    'Longitude': df_margSocialCosts['Longitude'],
    'Latitude': df_margSocialCosts['Latitude']
})


# ### Step 4: Inflate Marginal Social Cost (Damage) Factors using BLS CPI for All Urban Consumers (CPI-U)
# - Series Id:	CUUR0000SA0
# - Not Seasonally Adjusted
# - Series Title:	All items in U.S. city average, all urban consumers, not seasonally adjusted
# - Area:	U.S. city average
# - Item:	All items
# - Base Period:	1982-84=100
# 
# ### Use the updated Social Cost of Carbon (190 USD-2020/ton CO2) and inflate to USD-2021
# - EPA Median for 2% near term discount rate and most commonly mentioned value is 190 USD-2020 using the GIVE model.
# - 190 USD-2020 has some inconsistency with the VSL being used. An old study and 2008 VSL is noted
# - 190 USD value and inflate to USD 2021 because there is a clear source and ease of replicability.
# 
# ### Adjustment for VSL
# - EASIUR uses a VSL of 8.8M USD-2010 
# - New EPA VSL is 11.3M USD-2021

# In[ ]:


print("""
-------------------------------------------------------------------------------------------------------
Step 4: Inflate Marginal Social Cost (Damage) Factors using BLS CPI for All Urban Consumers (CPI-U)
-------------------------------------------------------------------------------------------------------
BLS CPI for All Urban Consumers (CPI-U) Info:
- Series Id: CUUR0000SA0
- Not Seasonally Adjusted
- Series Title: All items in U.S. city average, all urban consumers, not seasonally adjusted
- Area: U.S. city average
- Item: All items
- Base Period: 1982-84=100

-------------------------------------------------------------------------------------------------------
Use the updated Social Cost of Carbon (190 USD-2020/ton CO2) and inflate to USD-2021
-------------------------------------------------------------------------------------------------------
- EPA Median for 2% near term discount rate and most commonly mentioned value is 190 USD-2020 using the GIVE model.

-------------------------------------------------------------------------------------------------------
VSl Adjustment
-------------------------------------------------------------------------------------------------------
- EASIUR uses a VSL of 8.8M USD-2010 
- New EPA VSL is 11.3M USD-2021
-------------------------------------------------------------------------------------------------------
""")
# Load the BLS Inflation Data
filename = "bls_cpiu_2005-2023.xlsx"
print(f"Retrieving data for filename: {filename}")

change_directory = "inflation_data"
filepath = str(program_directory) + "\\" + str(change_directory) + "\\" + str(filename) 
print(f"Located at filepath: {filepath}")

# Create a pandas dataframe
df_bls_cpiu = pd.read_excel(filepath, sheet_name='bls_cpiu')

df_bls_cpiu = pd.DataFrame({
    'year': df_bls_cpiu['Year'],
    'cpiu_annual': df_bls_cpiu['Annual']
})

# Obtain the Annual CPIU values for the years of interest
bls_cpi_annual_2008 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2008)].item()
bls_cpi_annual_2010 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2010)].item()
bls_cpi_annual_2013 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2013)].item()
bls_cpi_annual_2018 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2018)].item()
bls_cpi_annual_2019 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2019)].item()
bls_cpi_annual_2020 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2020)].item()
bls_cpi_annual_2021 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2021)].item()
bls_cpi_annual_2022 = df_bls_cpiu['cpiu_annual'].loc[(df_bls_cpiu['year'] == 2022)].item()

# Precompute constant values
cpi_ratio_2021_2021 = bls_cpi_annual_2021 / bls_cpi_annual_2021  # This will be 1
cpi_ratio_2021_2020 = bls_cpi_annual_2021 / bls_cpi_annual_2020  # For SCC
cpi_ratio_2021_2019 = bls_cpi_annual_2021 / bls_cpi_annual_2019 
cpi_ratio_2021_2018 = bls_cpi_annual_2021 / bls_cpi_annual_2018 
cpi_ratio_2021_2013 = bls_cpi_annual_2021 / bls_cpi_annual_2013
cpi_ratio_2021_2010 = bls_cpi_annual_2021 / bls_cpi_annual_2010
cpi_ratio_2021_2008 = bls_cpi_annual_2021 / bls_cpi_annual_2008  # For EPA VSL and SCC

# 2021 US EPA VSL is $11.3M in 2021 USD
df_margSocialCosts_EASIUR['current_VSL_USD2021'] = 11.3

# Easiur uses a VSL of $8.8 M USD2010
# Inflate to 2021 $USD
df_margSocialCosts_EASIUR['easiur_VSL_USD2021'] = 8.8 * cpi_ratio_2021_2010

# Use df_margSocialCosts_EASIUR in the calculation of other columns
# Also adjust the VSL
df_margSocialCosts_EASIUR['margSocialCosts_pm25'] = round((df_margSocialCosts['PM25 Annual Ground'] * (1/2204.6) * (df_margSocialCosts_EASIUR['current_VSL_USD2021']/df_margSocialCosts_EASIUR['easiur_VSL_USD2021'])), 2)
df_margSocialCosts_EASIUR['margSocialCosts_so2'] = round((df_margSocialCosts['SO2 Annual Ground'] * (1/2204.6) * (df_margSocialCosts_EASIUR['current_VSL_USD2021']/df_margSocialCosts_EASIUR['easiur_VSL_USD2021'])), 2)
df_margSocialCosts_EASIUR['margSocialCosts_nox'] = round((df_margSocialCosts['NOX Annual Ground'] * (1/2204.6) * (df_margSocialCosts_EASIUR['current_VSL_USD2021']/df_margSocialCosts_EASIUR['easiur_VSL_USD2021'])), 2)

# Note that SCC of $190 USD-2020 has some inconsistency with the VSL being used. An old study and 2008 VSL is noted
# We use the $190 USD value and inflate to USD 2021 because there is a clear source and ease of replicability.
df_margSocialCosts_EASIUR['margSocialCosts_co2'] = round((190 * cpi_ratio_2021_2020 * (1/2204.6)), 2)

df_margSocialCosts_EASIUR['unit'] = '[$USD2021/lb]'

df_margSocialCosts_EASIUR


# ## Electricity CEDM-EASIUR Marginal Damages: Current and Decarbonizing Grid
# - Factor Type: Marginal
# - Calculation Method: Regression
# - Metric: Marginal Damages EASIUR [USD per MWh or kWh]
# - Year: 2018
# - Regional Aggregation: eGRID subregion (all regions)
# - Pollutants: SO2, NOx, PM2.5 CO2
# 
# SCC Adjustment: We use the EPA suggested 190 USD-2020 value for the social cost of carbon and inflate to 2021 USD. 
# 
# VSL: "We use a value of a statistical life (VSL) of USD 8.8 million (in 2010 dollars) for both our AP2 and EASIUR calculations. EASIUR reports damage intensities in USD/metric ton using this VSL and dollar year."

# In[ ]:


# For CO2 adjust SCC
# Create an adjustment factor for the new Social Cost of Carbon (SCC)
epa_scc = 190 * cpi_ratio_2021_2020
old_scc = 40 * cpi_ratio_2021_2010
scc_adjustment_factor = epa_scc / old_scc

# For Health-Related Emissions Adjust for different Value of a Statistical Life (VSL) values
# Current VSL is $11.3 M USD2021
current_VSL_USD2021 = 11.3

# Easiur uses a VSL of $8.8 M USD2010
easiur_VSL_USD2021 = 8.8 * (cpi_ratio_2021_2010)

# Calculate VSL adjustment factor
vsl_adjustment_factor = current_VSL_USD2021 / easiur_VSL_USD2021


# ### Damages from Climate Related Emissions

# In[ ]:


# Climate damages (co2) are expected to decline 68% linearlly by 2030 (% relative to 2005)
# Note only 2006 data available, used in place of 2005
filename = "Generation-MARREG-DAMEASIUR-egrid-byYear_climate2006"
print("\n")

filename = filename + ".csv"
print(f"Retrieving data for filename: {filename}")

change_directory = "margDamages_EASIUR"
filepath = str(program_directory) + "\\" + str(change_directory) + "\\" + str(filename) 

print(f"Located at filepath: {filepath}")
df_margDamages_climate2006 = pd.read_csv(filepath, index_col=0)
# df_margDamages_climate2006


# In[ ]:


# Climate damages (co2) are expected to decline 68% linearlly by 2030 (% relative to 2005)
# Note 2018 start year
filename = "Generation-MARREG-DAMEASIUR-egrid-byYear_climate2018"
print("\n")

filename = filename + ".csv"
print(f"Retrieving data for filename: {filename}")

change_directory = "margDamages_EASIUR"
filepath = str(program_directory) + "\\" + str(change_directory) + "\\" + str(filename) 

print(f"Located at filepath: {filepath}")
df_margDamages_climate2018 = pd.read_csv(filepath, index_col=0)
# df_margDamages_climate2018


# In[ ]:


# Marginal damages [$/kWh]
# Inflate from 2010 to 2021
# Note only 2006 data available, used in place of 2005
df_margDamages_EASIUR_climate = pd.DataFrame({
    'subregion_eGRID': df_margDamages_climate2006['region'],
    'pollutant': df_margDamages_climate2006['pollutant'],
    'unit': '[$/kWh]',
    '2030_decarb': '68% from 2005',
    'margDamages_dollarPerkWh_adjustVSL_ref': (df_margDamages_climate2006['factor'] * (scc_adjustment_factor) * (1/1000)) * (cpi_ratio_2021_2010),
    'margDamages_dollarPerkWh_adjustVSL_2018': (df_margDamages_climate2018['factor'] * (scc_adjustment_factor) * (1/1000)) * (cpi_ratio_2021_2010)
})
df_margDamages_EASIUR_climate['margDamages_decarb_2030'] = df_margDamages_EASIUR_climate['margDamages_dollarPerkWh_adjustVSL_ref'] - (df_margDamages_EASIUR_climate['margDamages_dollarPerkWh_adjustVSL_ref'] * 0.68)
df_margDamages_EASIUR_climate['reduction_margDamages_2030'] = df_margDamages_EASIUR_climate['margDamages_dollarPerkWh_adjustVSL_2018'] - df_margDamages_EASIUR_climate['margDamages_decarb_2030']
df_margDamages_EASIUR_climate['reduction_margDamages_annual'] = df_margDamages_EASIUR_climate['reduction_margDamages_2030'] / 12 # Relative to 2018, 
df_margDamages_EASIUR_climate


# ### Damages from Health Related Emissions

# In[ ]:


# Health damages (SO2, NOx, PM2.5) are expected to decline 65% by 2030 (% relative from 2021)
filename = "Generation-MARREG-DAMEASIUR-egrid-byYear_health2018"
print("\n")

filename = filename + ".csv"
print(f"Retrieving data for filename: {filename}")

change_directory = "margDamages_EASIUR"
filepath = str(program_directory) + "\\" + str(change_directory) + "\\" + str(filename) 

print(f"Located at filepath: {filepath}")
df_margDamages_health2018 = pd.read_csv(filepath, index_col=0)
# df_margDamages_health2018


# In[ ]:


# Marginal damages [$/kWh]
# Inflate from 2010 to 2021
# Note only 2018 data available, used in place of 2021
df_margDamages_EASIUR_health = pd.DataFrame({
    'subregion_eGRID': df_margDamages_health2018['region'],
    'pollutant': df_margDamages_health2018['pollutant'],
    'unit': '[$/kWh]',
    '2030_decarb': '65% from 2021',
    'margDamages_dollarPerkWh_adjustVSL_ref': (df_margDamages_health2018['factor'] * (vsl_adjustment_factor) * (1/1000)) * (cpi_ratio_2021_2010),
    'margDamages_dollarPerkWh_adjustVSL_2018': (df_margDamages_health2018['factor'] * (vsl_adjustment_factor) * (1/1000)) * (cpi_ratio_2021_2010)
})
df_margDamages_EASIUR_health['margDamages_decarb_2030'] = df_margDamages_EASIUR_health['margDamages_dollarPerkWh_adjustVSL_ref'] - (df_margDamages_EASIUR_health['margDamages_dollarPerkWh_adjustVSL_ref'] * 0.65)
df_margDamages_EASIUR_health['reduction_margDamages_2030'] = df_margDamages_EASIUR_health['margDamages_dollarPerkWh_adjustVSL_2018'] - df_margDamages_EASIUR_health['margDamages_decarb_2030']
df_margDamages_EASIUR_health['reduction_margDamages_annual'] = df_margDamages_EASIUR_health['reduction_margDamages_2030'] / 9
df_margDamages_EASIUR_health


# In[ ]:


# Combine them top to bottom
df_margDamages_EASIUR = pd.concat([df_margDamages_EASIUR_climate, df_margDamages_EASIUR_health], ignore_index=True)
df_margDamages_EASIUR


# In[ ]:


df_margDamages_gridDecarb = df_margDamages_EASIUR.copy()

years = list(range(2019, 2051))

# Apply reductions
for year in years:
    column_name = f'margDamages_dollarPerkWh_adjustVSL_{year}'
    df_margDamages_gridDecarb[column_name] = df_margDamages_gridDecarb['margDamages_dollarPerkWh_adjustVSL_ref']  # Initialize

    for index, row in df_margDamages_gridDecarb.iterrows():  # Correctly unpack the index and row
        if year <= 2030:
            # Climate reduction (C02) applicable from 2019 to 2030
            # No Health reductions before 2022
            if 2019 <= year < 2022:
                if row['pollutant'] == 'co2':
                    df_margDamages_gridDecarb.at[index, column_name] = df_margDamages_gridDecarb.at[index, f'margDamages_dollarPerkWh_adjustVSL_{year-1}'] - df_margDamages_gridDecarb.at[index, 'reduction_margDamages_annual']
                else:
                    df_margDamages_gridDecarb.at[index, column_name] = df_margDamages_gridDecarb.at[index, f'margDamages_dollarPerkWh_adjustVSL_{year-1}']
            
            # Health reduction applicable from 2022 to 2030
            # Climate reductions continue
            elif year >= 2022:
                df_margDamages_gridDecarb.at[index, column_name] = df_margDamages_gridDecarb.at[index, f'margDamages_dollarPerkWh_adjustVSL_{year-1}'] - df_margDamages_gridDecarb.at[index, 'reduction_margDamages_annual']

        # Post-2030, damage values should be at the 2030 level
        else:
            df_margDamages_gridDecarb.at[index, column_name] = df_margDamages_gridDecarb.at[index, f'margDamages_dollarPerkWh_adjustVSL_2030']
df_margDamages_gridDecarb


# In[ ]:


# cols_to_display = ['subregion_eGRID', 'pollutant', 'margDamages_decarb_2030', 'margDamages_dollarPerkWh_adjustVSL_2030']
# test_df = df_margDamages_gridDecarb[cols_to_display]
# test_df


# In[ ]:


# Create an empty dictionary to store the lookup data
dict_margDamages_gridDecarb = {}

for year in years:
    # Create an empty dictionary for the current year
    year_lookup = {}
    
    for _, row in df_margDamages_gridDecarb.iterrows():
        year_lookup[(row['subregion_eGRID'], row['pollutant'])] = row[f'margDamages_dollarPerkWh_adjustVSL_{str(year)}']
    
    # Add the year-specific lookup to the main lookup_data dictionary
    dict_margDamages_gridDecarb[year] = year_lookup

# Now, you have a lookup_data dictionary containing emissions factors for each state and year
dict_margDamages_gridDecarb


# In[ ]:


print("-------------------------------------------------------------------------------------------------------")
print("Calculate Emissions Factors: FOSSIL FUELS")
print("-------------------------------------------------------------------------------------------------------")
print("""
Fossil Fuels (Natural Gas, Fuel Oil, Propane):
- NOx, SO2, CO2: 
    - RESNET Table 7.1.2 Emissions Factors for Household Combustion Fuels
    - Source: https://www.resnet.us/wp-content/uploads/ANSIRESNETICC301-2022_resnetpblshd.pdf
    - All factors are in units of lb/Mbtu so energy consumption in kWh need to be converted to kWh 
    - (1 lb / Mbtu) * (1 Mbtu / 1x10^6 Btu) * (3412 Btu / 1 kWh)
- PM2.5: 
    - A National Methodology and Emission Inventory for Residential Fuel Combustion
    - Source: https://www3.epa.gov/ttnchie1/conference/ei12/area/haneke.pdf
""")
print("-------------------------------------------------------------------------------------------------------")

# Create a lookup dictionary for the national emissions factors
national_factors = df_margEmis_factors[df_margEmis_factors['state'] == 'National']
national_lookup = {(row['fuel_type'], row['pollutant']): row['margEmis_factor_adjusted'] for _, row in national_factors.iterrows()}

# Create a lookup dictionary for the state-specific emissions factors for electricity
electricity_factors = df_margEmis_factors[df_margEmis_factors['fuel_type'] == 'electricity']
electricity_lookup = {(row['pollutant'], row['state']): row['margEmis_factor_adjusted'] for _, row in electricity_factors.iterrows()}

pollutants = ['so2', 'nox', 'pm25', 'co2']

# ELECTRICITY CEDM DAMAGES LOOKUP
damages_CEDM_lookup = {(row['pollutant'], row['subregion_eGRID']): row['margDamages_dollarPerkWh_adjustVSL_ref'] for _, row in df_margDamages_EASIUR.iterrows()}

# FOSSIL FUELS DAMAGES LOOKUP
# Create a damages_fossilFuel_lookup dictionary from df_margSocialCosts_EASIUR
damages_fossilFuel_lookup = df_margSocialCosts_EASIUR.groupby(['Longitude', 'Latitude']).first().to_dict()


# ### Step 5: Calculate End-use specific marginal damages
# **I used the total emissions column for each of the end uses for the following reasons:**
# - Most homes only have 1 of each end-use, so it is unlikely that the homes have a significant consumption values from different fuel types. Thus, the total consumption and total emissions column (sum of each dwelling units consumption by end-use for each fuel) is fine to use to calculate marginal damages (social cost)
# - We can visualize the emissions in 2 by 2 grid (CO2, PM25, SO2, NOx) with each appliance's heating fuel in a different shape or color. 

# In[ ]:


print("""
-------------------------------------------------------------------------------------------------------
Step 5: Calculate End-use specific marginal damages
-------------------------------------------------------------------------------------------------------
""")


# ### Baseline Marginal Damages: WHOLE-HOME

# In[ ]:


print("-------------------------------------------------------------------------------------------------------")
print("Baseline Marginal Damages: WHOLE-HOME")
print("-------------------------------------------------------------------------------------------------------")


# In[ ]:


# calculate_marginal_damages(df, grid_decarb=False)
df_euss_am_baseline_home = calculate_marginal_damages(df=df_euss_am_baseline_home,
                                                      grid_decarb=False
                                                     )
df_euss_am_baseline_home


# ## Private Perspective: Annual Energy Costs

# In[ ]:


print("\n")
print("""
-------------------------------------------------------------------------------------------------------
Private Perspective: Annual Energy Costs
-------------------------------------------------------------------------------------------------------
- Step 1: Obtain Level Energy Fuel Cost Data from the EIA
- Step 2: Calculate Annual Operating (Fuel) Costs
-------------------------------------------------------------------------------------------------------
""")


# ### Step 1: Obtain Level Energy Fuel Cost Data from the EIA
# **Data Sources for Excel workbook containing state average Residential fuel cost for each fuel in 2018**
# - EIA State Electricity Price: https://www.eia.gov/electricity/state/archive/2018/
# - EIA Natural Gas Prices: https://www.eia.gov/dnav/ng/ng_pri_sum_dcu_SPA_a.htm
# - Propane and Fuel Oil: EIA March 2023 Short Term Energy Outlook
#     - https://www.eia.gov/outlooks/steo/pdf/wf01.pdf
#     - Table WF01: Average Consumer Prices and Expenditures for Heating Fuels During the Winter
#     - US Average: 2018-2019 Data

# In[ ]:


print("""
-------------------------------------------------------------------------------------------------------
Step 1: Obtain Level Energy Fuel Cost Data from the EIA
-------------------------------------------------------------------------------------------------------
**Data Sources for Excel workbook containing state average Residential fuel cost for each fuel in 2018**
- EIA State Electricity Price: https://www.eia.gov/electricity/state/archive/2018/
- EIA Natural Gas Prices: https://www.eia.gov/dnav/ng/ng_pri_sum_dcu_SPA_a.htm
- Propane and Fuel Oil: EIA March 2023 Short Term Energy Outlook
    - https://www.eia.gov/outlooks/steo/pdf/wf01.pdf
    - Table WF01: Average Consumer Prices and Expenditures for Heating Fuels During the Winter
    - US Average: 2018-2019 Data
-------------------------------------------------------------------------------------------------------
""")
# Program Directory --> C:\Users\14128\Research\cmu_tare_model
filename = 'fuel_prices.xlsx'
print(f"Retrieving data for filename: {filename}")

change_directory = "fuel_prices"
filepath = str(program_directory) + "\\" + str(change_directory) + "\\" + str(filename) 

print(f"Located at filepath: {filepath}")


# In[ ]:


df_fuelPrices_perkWh = pd.read_excel(filepath, sheet_name='fuel_prices')

# Create a new column called "cost_per_kWh"
df_fuelPrices_perkWh['cost_per_kWh'] = 0.0

# Convert to $/kWh equivalent
# https://www.eia.gov/energyexplained/units-and-calculators/british-thermal-units.php
for index, row in df_fuelPrices_perkWh.iterrows():
    
    # Propane: (dollars per gallon) * (1 gallon propane/91,452 BTU) * (3412 BTU/1 kWh)
    if row['fuel_type'] == 'propane':
        df_fuelPrices_perkWh.at[index, 'cost_per_kWh'] = row['cost_per_unit'] * (1/91452) * (3412/1)
    
    # Fuel Oil: (dollars/gallon) * (1 gallon heating oil/138,500 BTU) * (3412 BTU/1 kWh)
    elif row['fuel_type'] == 'fuelOil':
        df_fuelPrices_perkWh.at[index, 'cost_per_kWh'] = row['cost_per_unit'] * (1/138500) * (3412/1)
    
    # Natural Gas: (dollars/cf) * (thousand cf/1000 cf) * (1 cf natural gas/1039 BTU) * (3412 BTU/1 kWh)
    elif row['fuel_type'] == 'naturalGas':
        df_fuelPrices_perkWh.at[index, 'cost_per_kWh'] = row['cost_per_unit'] * (1/1000) * (1/1039) * (3412/1)
    
    # Electricity: convert cents per kWh to $ per kWh
    elif row['fuel_type'] == 'electricity':
        df_fuelPrices_perkWh.at[index, 'cost_per_kWh'] = row['cost_per_unit'] / 100
        
df_fuelPrices_perkWh


# ### Step 2: Calculate Annual Operating (Fuel) Costs

# In[ ]:


print("""
-------------------------------------------------------------------------------------------------------
Step 2: Calculate Annual Operating (Fuel) Costs
-------------------------------------------------------------------------------------------------------
- Create a mapping dictionary for fuel types
- Create new merge columns to ensure a proper match.
- Merge df_copy with df_fuel_prices to get fuel prices for electricity, natural gas, propane, and fuel oil
- Calculate the per kWh fuel costs for each fuel type and region
- Calculate the baseline fuel cost 
- Add fuel oil cost for non-cooking and non-clothesDrying categories
-------------------------------------------------------------------------------------------------------
""")


# ### Baseline Fuel Cost: WHOLE-HOME

# In[ ]:


print("\n")
print("-------------------------------------------------------------------------------------------------------")
print("Baseline Fuel Cost: WHOLE-HOME")
print("-------------------------------------------------------------------------------------------------------")


# In[ ]:


# df_euss_am_baseline_home = df_euss_am_baseline_home.copy()
# calculate_annual_fuelCost(df, category, state_region, df_fuelPrices_perkWh, cpi_ratio)
df_euss_am_baseline_home = calculate_annual_fuelCost(df=df_euss_am_baseline_home,
                                                        state_region=input_state,
                                                        df_fuelPrices_perkWh=df_fuelPrices_perkWh,
                                                        cpi_ratio=cpi_ratio_2021_2018                                                        
                                                       )
df_euss_am_baseline_home


# # Model Runtime

# In[ ]:


# Get the current datetime again
end_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Calculate the elapsed time
elapsed_time = datetime.strptime(end_time, "%Y-%m-%d_%H-%M-%S") - datetime.strptime(start_time, "%Y-%m-%d_%H-%M-%S")

# Format the elapsed time
elapsed_seconds = elapsed_time.total_seconds()
elapsed_minutes = int(elapsed_seconds // 60)
elapsed_seconds = int(elapsed_seconds % 60)

# Print the elapsed time
print(f"The code took {elapsed_minutes} minutes and {elapsed_seconds} seconds to execute.")


# In[ ]:




