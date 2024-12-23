# %%
# Set columns in display
# pd.set_option('display.max_columns', None)
# pd.reset_option('display.max_columns') # Reset options to default

# Set rows in display
# pd.set_option('display.max_rows', None)
# pd.reset_option('display.max_rows') # Reset options to default

# %% [markdown]
# # Load Util File with TARE Model Functions

# %%
import os

# Measure Package 0: Baseline
menu_mp = 0
input_mp = 'baseline'

from config import PROJECT_ROOT

import pandas as pd
import numpy as np

# `plt` is an alias for the `matplotlib.pyplot` module
import matplotlib.pyplot as plt

# import seaborn library (wrapper of matplotlib)
import seaborn as sns
sns.set(style="darkgrid")

# For regex, import re
import re

from datetime import datetime

# Get the current datetime
# Start the timer
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Storing Result Outputs in output_results folder
relative_path = os.path.join("cmu_tare_model", "output_results")
output_folder_path = os.path.join(PROJECT_ROOT, relative_path)

print(f"""
-------------------------------------------------------------------------------------------------------
Welcome to the Trade-off Analysis of residential Retrofits for energy Equity Tool (TARE Model)
Let's start by reading the data from the NREL EUSS Database.

Make sure that the zipped folders stay organized as they are once unzipped.
If changes are made to the file path, then the program will not run properly.
-------------------------------------------------------------------------------------------------------
      
Project root directory: {PROJECT_ROOT}
Result outputs will be exported here: {output_folder_path}

""")

# %% [markdown]
# # Simulate Residential Energy Consumption using NREL End-Use Savings Shapes

# %%
from cmu_tare_model.functions.load_and_filter_euss_data import *

# Measure Package 0: Baseline
menu_mp = 0
input_mp = 'baseline'

filename = "baseline_metadata_and_annual_results.csv"
relative_path = os.path.join("cmu_tare_model", "data", "euss_data", "resstock_amy2018_release_1.1", "state", filename)

file_path = os.path.join(PROJECT_ROOT, relative_path)

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")

# Fix DtypeWarning error in columns:
# 'in.neighbors', 'in.geometry_stories_low_rise', 'in.iso_rto_region', 'in.pv_orientation', 'in.pv_system_size'
columns_to_string = {11: str, 61: str, 121: str, 103: str, 128: str, 129: str}
df_euss_am_baseline = pd.read_csv(file_path, dtype=columns_to_string, index_col="bldg_id") # UPDATE: Set index to 'bldg_id'

# Filter for occupied homes
occupancy_filter = df_euss_am_baseline['in.vacancy_status'] == 'Occupied'
df_euss_am_baseline = df_euss_am_baseline.loc[occupancy_filter]

# Filter for single family home building type
house_type_list = ['Single-Family Attached', 'Single-Family Detached']
house_type_filter = df_euss_am_baseline['in.geometry_building_type_recs'].isin(house_type_list)
df_euss_am_baseline = df_euss_am_baseline.loc[house_type_filter]

# Choose between national or sub-national level analysis
menu_state = get_menu_choice(menu_prompt, {'N', 'Y'})   # This code is only run in baseline

# National Level 
if menu_state == 'N':
    print("You chose to analyze all of the United States.")
    input_state = 'National'
    location_id = 'National'

# Filter down to state or city
else:
    input_state = get_state_choice(df_euss_am_baseline)    
    print(f"You chose to filter for: {input_state}")

    location_id = str(input_state)
    print(f"Location ID is: {location_id}")


    state_filter = df_euss_am_baseline['in.state'].eq(input_state)
    df_euss_am_baseline = df_euss_am_baseline.loc[state_filter]

    print(city_prompt)
    print(df_euss_am_baseline['in.city'].value_counts())

    menu_city = get_menu_choice(city_menu_prompt, {'N', 'Y'})

    # Filter for the entire selected state
    if menu_city == 'N':
        print(f"You chose to analyze all of state: {input_state}")
        
        location_id = str(input_state)
        print(f"Location ID is: {location_id}")
        
    # Filter to a city within the selected state
    else:
        input_cityFilter = get_city_choice(df_euss_am_baseline, input_state)
        print(f"You chose to filter for: {input_state}, {input_cityFilter}")

        location_id = input_cityFilter.replace(', ', '_').strip()
        print(f"Location ID is: {location_id}")

        city_filter = df_euss_am_baseline['in.city'].eq(f"{input_state}, {input_cityFilter}")
        df_euss_am_baseline = df_euss_am_baseline.loc[city_filter]

print(f"""
-------------------------------------------------------------------------------------------------------
BASELINE (Measure Package 0)
-------------------------------------------------------------------------------------------------------
DATAFRAME: df_euss_am_baseline

DATA: NREL EUSS Database
HOUSING FILTERS: Occupied units and Single Family Homes
GEOGRAPHIC FILTERS: National, State, or City

Additional details and documentation: 
      data can be found in the EUSS documentation here: 
methods can be found in the load_and_filter_euss_data.py file.
      
DATAFRAME: df_euss_am_baseline
      
{df_euss_am_baseline}
""")

# %%
# df_baseline_enduse(df_baseline, df_enduse, category, fuel_filter='Yes', tech_filter='Yes')
df_euss_am_baseline_home = df_enduse_refactored(df_baseline = df_euss_am_baseline,
                                                fuel_filter = 'Yes',
                                                tech_filter = 'Yes')

print(f"""
In addition to the housing type and occupancy filters, the data has been filtered for fuel and technology:
      
FUEL FILTERS: 
      - Water Heating and Space Heating: Electricity, Fuel Oil, Natural Gas, Propane
      - Clothes Drying and Cooking: Electricity, Natural Gas, Propane

ACCEPTABLE TECH FILTERS (cost data available):
      - Water Heating: ['Electric Heat Pump, 80 gal', 'Electric Premium', 'Electric Standard', 
                        'Fuel Oil Premium', 'Fuel Oil Standard', 
                        'Natural Gas Premium', 'Natural Gas Standard', 
                        'Propane Premium', 'Propane Standard']
      - Space Heating: ['Electricity ASHP', 'Electricity Baseboard', 'Electricity Electric Boiler', 'Electricity Electric Furnace',
                        'Fuel Oil Fuel Boiler', 'Fuel Oil Fuel Furnace', 
                        'Natural Gas Fuel Boiler', 'Natural Gas Fuel Furnace', 
                        'Propane Fuel Boiler', 'Propane Fuel Furnace']
      - Clothes Drying and Cooking: Required no tech filter

df_enduse_refactored will display the number of representative homes after filtering for fuel and tech.
      
DATAFRAME: df_euss_am_baseline_home
      
{df_euss_am_baseline_home}
""")

# %% [markdown]
# ## Project Future Energy Consumption Using EIA Heating Degree Day (HDD) Forecasted Data (Factors)

# %%
from cmu_tare_model.functions.project_future_energy_consumption import *

# Make a copy of the dataframe
df_baseline_scenario_consumption = df_euss_am_baseline_home.copy()

# Project Future Energy Consumption
df_euss_am_baseline_home, df_baseline_scenario_consumption = project_future_consumption(df=df_euss_am_baseline_home,
                                                                                        lookup_hdd_factor=lookup_hdd_factor,
                                                                                        menu_mp=menu_mp
                                                                                        )
# Display the baseline scenario summary dataframe
# df_euss_am_baseline_home
print(F"""
-------------------------------------------------------------------------------------------------------
PROJECT FUTURE ENERGY CONSUMPTION: Baseline Consumption
-------------------------------------------------------------------------------------------------------
Creating dataframe to store annual energy consumption calculations ...
      
DATAFRAME: df_euss_am_baseline_home

{df_euss_am_baseline_home}      

DATAFRAME: df_baseline_scenario_consumption
      
{df_baseline_scenario_consumption}
""")

# %% [markdown]
# # Public Perspective: Monetized Marginal Damages from Emissions

# %% [markdown]
# ## Fossil Fuels: Climate and Health-Related Pollutants

# %%
"""
create_lookup_emissions_fossil_fuel.py uses the calculate_fossil_fuel_emission_factor function and RESNET data sources
to calculate the emission factors for fossil fuels. The function returns a dataframe of marginal emission factors

The create_lookup_emissions_fossil_fuel.py file contains the following dataframes and lookup dictionaries:
    - df_marg_emis_factors: Marginal Emission Factors for Fossil Fuels
    - lookup_emis_fossil_fuel: Lookup Dictionary for Fossil Fuel Emissions

Fossil Fuels (Natural Gas, Fuel Oil, Propane):
- NOx, SO2, CO2: 
    - RESNET Table 7.1.2 Emissions Factors for Household Combustion Fuels
    - Source: https://www.resnet.us/wp-content/uploads/ANSIRESNETICC301-2022_resnetpblshd.pdf
    - All factors are in units of lb/Mbtu; energy consumption in kWh needs conversion.
- PM2.5: 
    - A National Methodology and Emission Inventory for Residential Fuel Combustion
    - Source: https://www3.epa.gov/ttnchie1/conference/ei12/area/haneke.pdf
"""

from cmu_tare_model.functions.create_lookup_emissions_fossil_fuel import *

print(f"""
--------------------------------------------------------------------------------------------------------------------------------------
Fossil Fuels: Climate and Health-Related Pollutants
--------------------------------------------------------------------------------------------------------------------------------------
DATAFRAME: Marginal Emission Factors for Fossil Fuels
      
{df_marg_emis_factors}  

LOOKUP DICTIONARY: Fossil Fuel Emissions

{lookup_emis_fossil_fuel}
""")

# %% [markdown]
# ### Inflate Marginal Social Cost (Damage) Factors using BLS CPI for All Urban Consumers (CPI-U)

# %%
""""
We adjust all monetary values for inflation using the Bureau of Labor Statistics (BLS) 
Consumer Price Index (CPI) for All Urban Consumers (CPI-U).

inflation_adjustment.py does the following:
- Loads the BLS CPI for 2005-2023
- Uses this data to create a dataframe named df_bls_cpiu
- Calculates the CPI ratio constants for use later in the model (ex: cpi_ratio_2023_2020 for EPA SCC, etc.)

Additional information concerning the BLS CPI for All Urban Consumers (CPI-U) is provided the inflation_adjustment.py file.
"""

from cmu_tare_model.functions.inflation_adjustment import *
print(f"""
--------------------------------------------------------------------------------------------------------------------------------------
Inflate Marginal Social Cost (Damage) Factors using BLS CPI for All Urban Consumers (CPI-U)
--------------------------------------------------------------------------------------------------------------------------------------
DATAFRAME: Annual CPI-U for 2005-2023 used for cpi_ratio constants and inflation adjustments
      
{df_bls_cpiu}
""")

# %% [markdown]
# ### For Health-Related Emissions Adjust for different Value of a Statistical Life (VSL) values

# %%
# Current VSL is $11.3 M USD2021
# INFLATE TO USD2022, PREVIOUSLY USD2021
current_VSL_USD2022 = 11.3 * cpi_ratio_2023_2021

# Easiur uses a VSL of $8.8 M USD2010
# INFLATE TO USD2022, PREVIOUSLY USD2021
easiur_VSL_USD2022 = 8.8 * (cpi_ratio_2023_2010)

# Calculate VSL adjustment factor
vsl_adjustment_factor = current_VSL_USD2022 / easiur_VSL_USD2022

print(f"""
--------------------------------------------------------------------------------------------------------------------------------------
For Health-Related Emissions Adjust for different Value of a Statistical Life (VSL) values
    - EASIUR uses a VSL of 8.8M USD-2010 
    - New EPA VSL is 11.3M USD-2021
    - INFLATE TO $USD-2023
--------------------------------------------------------------------------------------------------------------------------------------
Current VSL: 
{current_VSL_USD2022} USD-2023
      
EASIUR VSL:
{easiur_VSL_USD2022} USD-2023

VSL Adjustment Factor:
{vsl_adjustment_factor}
"""
)

# %% [markdown]
# ### Quantify monitized HEALTH damages using EASIUR Marginal Social Cost Factors
# #### THE STEPS BELOW SUMMARIZE WHAT WAS DONE TO OBTAIN ALL NATIONAL EASIUR VALUES INCLUDED ON GITHUB
# - Obtain all of the dwelling unit latitude and longitude values from the metadata columns
# - Make a new dataframe of just the longitude and latitude values 
#     - Make sure that the order is (longitude, latitude)
#     - Do not include the index or column name when exporting 
# - Export the CSV
# - **Upload csv to EASIUR Website:**
#     - Website: https://barney.ce.cmu.edu/~jinhyok/easiur/online/
#     - See inputs in respective sections
# - Download the file and put it in the 'easiur_batchConversion_download' folder
# - Copy and paste the name of the file EASIUR generated when prompted
# - Copy and paste the name of the filepath for the 'easiur_batchConversion_download' folder when prompted
# - Match up the longitude and latitudes for each dwelling unit with the selected damages

# %%
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

# Updated GitHub code has EASIUR file with all unique latitude, longitude coordinates in the US
filename = 'easiur_National2024-06-1421-22.csv'
# filename = 'easiur_National_14June2024_2024IncPop2010Dollar.csv'
relative_path = os.path.join("cmu_tare_model", "data", "margDamages_EASIUR", "easiur_batchConversion_download", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")

df_margSocialCosts = pd.read_csv(file_path)

# Convert from kg/MWh to lb/kWh
# Obtain value from the CSV file and convert to lbs pollutant per kWh 

# Define df_marg_social_costs_EASIUR DataFrame first
df_marg_social_costs_EASIUR = pd.DataFrame({
    'Longitude': df_margSocialCosts['Longitude'],
    'Latitude': df_margSocialCosts['Latitude'],
})

# Use df_marg_social_costs_EASIUR in the calculation of other columns
# Also adjust the VSL
# df_marg_social_costs_EASIUR['marg_social_costs_pm25'] = round((df_margSocialCosts['PM25 Annual Ground'] * (1/2204.6) * vsl_adjustment_factor), 2)
# df_marg_social_costs_EASIUR['marg_social_costs_so2'] = round((df_margSocialCosts['SO2 Annual Ground'] * (1/2204.6) * vsl_adjustment_factor), 2)
# df_marg_social_costs_EASIUR['marg_social_costs_nox'] = round((df_margSocialCosts['NOX Annual Ground'] * (1/2204.6) * vsl_adjustment_factor), 2)
# df_marg_social_costs_EASIUR['unit'] = '[$USD2023/lb]'

df_marg_social_costs_EASIUR['pm25_usd2023_per_mt'] = round((df_margSocialCosts['PM25 Annual Ground'] * vsl_adjustment_factor), 2)
df_marg_social_costs_EASIUR['so2_usd2023_per_mt'] = round((df_margSocialCosts['SO2 Annual Ground'] * vsl_adjustment_factor), 2)
df_marg_social_costs_EASIUR['nox_usd2023_per_mt'] = round((df_margSocialCosts['NOX Annual Ground'] * vsl_adjustment_factor), 2)
df_marg_social_costs_EASIUR['unit'] = '[$USD2023/mt]'

# Create a damages_fossil_fuel_lookup dictionary from df_marg_social_costs_EASIUR
# First drop the 'unit' column
lookup_health_damages_fossil_fuel = df_marg_social_costs_EASIUR.drop(columns=['unit']).groupby(['Longitude', 'Latitude']).first().to_dict()
lookup_health_damages_fossil_fuel

# Dispalay the EASIUR marginal social costs df
print(f"""
--------------------------------------------------------------------------------------------------------------------------------------
EASIUR Marginal Social Costs for HEALTH-RELATED EMISSIONS (PM2.5, SO2, and NOx)
    - UPDATED  
    - New EPA VSL is 11.3M USD-2021
    - INFLATE TO $USD-2023
--------------------------------------------------------------------------------------------------------------------------------------
DATAFRAME: EASIUR Marginal Social Costs for HEALTH-RELATED EMISSIONS
      
{df_marg_social_costs_EASIUR}

LOOKUP DICTIONARY: Health Damages from Fossil Fuel Emissions

{lookup_health_damages_fossil_fuel}
"""
)

# %% [markdown]
# ## Emissions from Electricity Generation

# %% [markdown]
# ### Climate-Related Emissions from CAMBIUM LRMER/SRMER 
# ### Includes pre-combustion (fugitive) and combustion

# %%
from cmu_tare_model.functions.create_lookup_climate_damages_electricity import *
"""
-------------------------------------------------------------------------------------------------------
CLIMATE DAMAGES FROM CAMBIUM
-------------------------------------------------------------------------------------------------------
- Load CSV
- Convert MWh --> kWh and kg --> metric tons (mt)
- Inflate updated Social Cost of Carbon from $190 USD2020 to $USD2023
- Convert SCC to $USD2023/lb
- Calculate damage factors for CO2e: LRMER/SRMER [lb/kWh] * SCC[$USD2023/lb] = $USD2023/kWh
-------------------------------------------------------------------------------------------------------

Additional details and documentation:
      - LRMER/SRMER data can be found in the Cambium documentation here:
      - Functions and methods to process and create lookup dictionary can be found in the create_lookup_climate_damages_electricity.py file.
"""

print(f"""
=======================================================================================================
PRE-IRA:
LONG RUN AND SHORT RUN MARGINAL EMISSIONS RATES (LRMER, SRMER) FROM CAMBIUM 2021 RELEASE
=======================================================================================================
DATAFRAME: LRMER and SRMER for ELECTRICITY CO2e [mtCO2e/kWh]
      
{df_cambium21_processed}  

LOOKUP DICTIONARY: LRMER and SRMER for ELECTRICITY CO2e [mtCO2e/kWh]

{lookup_co2e_emis_electricity_preIRA}

=======================================================================================================
IRA-REFERENCE:
LONG RUN AND SHORT RUN MARGINAL EMISSIONS RATES (LRMER, SRMER) FROM CAMBIUM 2022 RELEASE
=======================================================================================================
DATAFRAME: LRMER and SRMER for ELECTRICITY CO2e [mtCO2e/kWh]
      
{df_cambium22_processed}  

LOOKUP DICTIONARY: LRMER and SRMER for ELECTRICITY CO2e [mtCO2e/kWh]

{lookup_co2e_emis_electricity_IRA}
""")

# %% [markdown]
# ### Use the updated Social Cost of Carbon (190 USD-2020/ton co2e) and inflate to USD-2023
# - EPA Median for 2% near term discount rate and most commonly mentioned value is 190 USD-2020 using the GIVE model.
# - 190 USD-2020 has some inconsistency with the VSL being used. An old study and 2008 VSL is noted
# - 190 USD value and inflate to USD 2023 because there is a clear source and ease of replicability.

# %%
# For co2e adjust SCC
EPA_SCC_USD2023_PER_TON = 190 * cpi_ratio_2023_2020

print(f"""
Steps 3 and 4: Obtain BLS CPI-U Data and Inflate Current Social Cost of Carbon (SCC) to USD2023
      
EPA Median for 2% near term discount rate and most commonly mentioned value is 190 USD-2020 using the GIVE model.
Inflate 190 $USD-2020 Social Cost of Carbon to $USD-2023

SCC Value used in analysis is: ${round(EPA_SCC_USD2023_PER_TON, 2)} per mt CO2e
""")

# %% [markdown]
# ## HEALTH-RELATED EMISSIONS: Grid Emissions Intensity Projections

# %% [markdown]
# ### PROJECTION FACTORS FOR FUTURE GRID EMISSIONS INTENSITY (Coal Generation Reduction)

# %%
from cmu_tare_model.functions.coal_projection_factors import *
print(f"""
=======================================================================================================
COAL USED IN ELECTRICITY GENERATION (Data Sources)
=======================================================================================================
Pre-process Coal Generation Data from EPA eGRID and Cambium 2021/2022
      - EPA eGRID Coal Generation data is used for 2018-2022
      - Map eGRID subregion to Cambium GEA region
      - Drop eGRID subregions not in Cambium GEA region (PR, AK, HI, etc)
      - Coal generation is aggregated by Cambium GEA region ('NYCW', 'NYLI', 'NYUP' --> 'NYSTc')

The coal_projection_factors.py file contains the following dataframes and lookup dictionaries:
      
DATAFRAME: EPA eGRID Coal Generation Data
{df_epa_eGRID_COAL_processed}

DARAFRAME: Cambium 2021
{df_cambium21_COAL_processed}
      
DATAFRAME: Cambium 2022
{df_cambium22_COAL_processed}

""")

# %%
print(f"""
=======================================================================================================
COAL USED IN ELECTRICITY GENERATION (Combined 2018-2050 Data Pre-IRA and IRA-Ref)
=======================================================================================================
For all other years up until 2050, Cambium 2021 and Cambium 2022 data is used
      - Combine with EPA eGRID data
            - Pre-IRA: EPA eGRID 2018-2022 + Cambium 2021
            - IRA-Reference: EPA eGRID 2018-2022 + Cambium 2022
      - Interpolate between years for missing data

The coal_projection_factors.py file contains the following dataframes and lookup dictionaries:

DATAFRAME: Pre-IRA Coal Generation 2018-2050
{df_preIRA_coal_generation}

DARAFRAME: IRA-Reference Coal Generation 2018-2050
{df_iraRef_coal_generation}
""")


# %%
print(f"""
=======================================================================================================
HEALTH-RELATED EMISSIONS PROJECTION FACTORS (Pre-IRA and IRA-Reference)
=======================================================================================================
Normalize projection factors relative to 2018. This is the last year of available data from CEDM Marginal Damages Factors (EASIUR Marginal Social Costs)

The coal_projection_factors.py file contains the following dataframes and lookup dictionaries:
    
DATAFRAME: Pre-IRA Health Projection Factors 2018-2050
{df_preIRA_coal_projection_factors}

DATAFRAME: IRA-Reference Health Projection Factors 2018-2050
{df_iraRef_coal_projection_factors}

""")


# %% [markdown]
# ## ADJUSTED Electricity CEDM-EASIUR Marginal Damages: Pre-IRA and IRA-Reference
# - Factor Type: Marginal
# - Calculation Method: Regression
# - Metric: Marginal Damages EASIUR [USD per MWh or kWh]
# - Year: 2018
# - Regional Aggregation: eGRID subregion (all regions)
# - Pollutants: SO2, NOx, PM2.5 CO2
# 
# SCC Adjustment: We use the EPA suggested 190 USD-2020 value for the social cost of carbon and inflate to 2022 USD. **PREVIOUSLY USED 2021 USD**
# 
# VSL: "We use a value of a statistical life (VSL) of USD 8.8 million (in 2010 dollars) for both our AP2 and EASIUR calculations. EASIUR reports damage intensities in USD/metric ton using this VSL and dollar year."

# %% [markdown]
# ### Pre-IRA Projections

# %%
#
from cmu_tare_model.functions.create_lookup_health_damages_electricity import *

print(f"""
===========================================================================================================================================================
PROJECT ELECTRICITY CEDM MARGINAL FACTORS (Adjust for VSL and Use Coal Projection Factors thru 2050)
===========================================================================================================================================================
1. Create a dataframe using the 2018 CEDM Marginal Damage Factors data
2. Create dictionaries mapping 'gea_region' to marginal damage factors for each pollutant
3. Map to the projection factors dataframe
4. Calculate the new columns by multiplying coal projection factors with marginal damages
5. Drop the intermediate marginal damage columns if they're no longer needed
6. Group the projection factors df by scenario and gea_region
7. Create a nested dictionary to serve as the lookup dictionary for pollutant damage factors (pollutant_dollarsPerKWh_adjustVSL)
===========================================================================================================================================================
Pre-IRA Scenario
The create_lookup_health_damages_electricity.py file contains the following dataframes and lookup dictionaries:

DATAFRAME: Adjusted CEDM Marginal Damage Factors with Updated VSL and Inflate to $USD2023
{df_margDamages_EASIUR_health}

LOOKUP: Pre-IRA Health Damages 2018-2050
{df_preIRA_health_damages_factors}

LOOKUP: Pre-IRA Health Damages 2018-2050
{lookup_health_damages_electricity_preIRA}
""")

# %% [markdown]
# ### IRA-Reference Projections

# %%
print(f"""
===========================================================================================================================================================
PROJECT ELECTRICITY CEDM MARGINAL FACTORS (Adjust for VSL and Use Coal Projection Factors thru 2050)
===========================================================================================================================================================
IRA-Reference Scenario
The create_lookup_health_damages_electricity.py file contains the following dataframes and lookup dictionaries:

DATAFRAME: Adjusted CEDM Marginal Damage Factors with Updated VSL and Inflate to $USD2023
{df_margDamages_EASIUR_health}

LOOKUP: IRA-Reference Health Damages 2018-2050
{df_iraRef_health_damages_factors}

LOOKUP: IRA-Reference Health Damages 2018-2050
{lookup_health_damages_electricity_iraRef}
""")

# %% [markdown]
# ### Calculate End-use specific marginal damages
# **I used the total emissions column for each of the end uses for the following reasons:**
# - Most homes only have 1 of each end-use, so it is unlikely that the homes have a significant consumption values from different fuel types. Thus, the total consumption and total emissions column (sum of each dwelling units consumption by end-use for each fuel) is fine to use to calculate marginal damages (social cost)
# - We can visualize the emissions in 2 by 2 grid (CO2, PM25, SO2, NOx) with each appliance's heating fuel in a different shape or color. 

# %% [markdown]
# ### Baseline Marginal Damages: WHOLE-HOME

# %%
from cmu_tare_model.functions.calculate_emissions_damages import *
print("""
-------------------------------------------------------------------------------------------------------
Step 5: Calculate End-use specific marginal damages
-------------------------------------------------------------------------------------------------------
      
-------------------------------------------------------------------------------------------------------
Baseline Marginal Damages: WHOLE-HOME
-------------------------------------------------------------------------------------------------------
""")
# Make copies from scenario consumption to keep df smaller
print("\n", "Creating dataframe to store marginal damages calculations ...")
df_baseline_scenario_damages = df_euss_am_baseline_home.copy()

# calculate_marginal_damages(df, menu_mp, policy_scenario)
df_euss_am_baseline_home, df_baseline_scenario_damages = calculate_marginal_damages(df=df_euss_am_baseline_home,
                                                                                    menu_mp=menu_mp,
                                                                                    policy_scenario='No Inflation Reduction Act',
                                                                                    df_detailed_damages=df_baseline_scenario_damages
                                                                                    )
df_euss_am_baseline_home

# %%
df_baseline_scenario_damages

# %% [markdown]
# ## Private Perspective: Annual Energy Costs

# %%
from cmu_tare_model.functions.process_fuel_price_data import *
print(f"""
===========================================================================================================================================================
PROCESS FUEL PRICE DATA AND PREPARE FOR PROJECTION: 
1. Load nominal fuel price data from EIA sources (2018-2022)
2. Convert base units (like dollars per gallon and cf) in nominal fuel price data to USD2023/kWh
3. Inflate all nominal fuel prices to $USD2023 (example: 2018 data in USD2018... multiply by cpi_ratio_2023_2018)
4. Map states to census division using map_location_to_census_division function (allows us to apply regional projection factors)
===========================================================================================================================================================
**Data Sources for Excel workbook containing state average Residential fuel cost for each fuel in 2018**
- EIA State Electricity Price: https://www.eia.gov/electricity/state/archive/2018/
- EIA Natural Gas Prices: https://www.eia.gov/dnav/ng/ng_pri_sum_dcu_SPA_a.htm
- Propane and Fuel Oil: EIA March 2023 Short Term Energy Outlook
    - https://www.eia.gov/outlooks/steo/pdf/wf01.pdf
    - Table WF01: Average Consumer Prices and Expenditures for Heating Fuels During the Winter
    - US Average: 2018-2019 Data
===========================================================================================================================================================

The process_fuel_price_data.py file contains the following dataframes and lookup dictionaries:

DATAFRAME: Processed fuel price data (normalized in per kWh price, set up location columns to allow mapping for projection factors)
{df_fuelPrices_perkWh}
"""
)

# %%
print(f"""
===========================================================================================================================================================
PROJECT FUTURE FUEL PRICES: 
5. Load 2022-2050 regional projection factor data from AEO2023 (Normalized all fuel prices by the 2022 value for each region)
6. For each policy scenario, project future fuel prices using the project_future_prices function
7. For each policy scenario, create a lookup dictionary using the create_fuel_price_lookup function
===========================================================================================================================================================

The process_fuel_price_data.py file contains the following dataframes and lookup dictionaries:

DATAFRAME: Regional Fuel Price Projection Factors
{df_fuelPrices_projection_factors}

PRE-IRA SCENARIO
----------------------------------------------------------------------------------------------------------------------------------------------------------
DATAFRAME: Pre-IRA Projection Factors
{df_fuelPrices_perkWh_preIRA}

LOOKUP: Pre-IRA Projection Factors
{lookup_fuel_prices_preIRA}

IRA-REFERENCE SCENARIO
----------------------------------------------------------------------------------------------------------------------------------------------------------
DATAFRAME: IRA-Reference Projection Factors
{df_fuelPrices_perkWh_iraRef}

DATAFRAME: IRA-Reference Projection Factors
{lookup_fuel_prices_iraRef}
"""
)

# %% [markdown]
# ### Step 2: Calculate Annual Operating (Fuel) Costs

# %% [markdown]
# ### Baseline Fuel Cost: WHOLE-HOME

# %%
from cmu_tare_model.functions.calculate_fuel_costs import *

print("""
-------------------------------------------------------------------------------------------------------
Step 2: Calculate Annual Operating (Fuel) Costs
-------------------------------------------------------------------------------------------------------
- Create a mapping dictionary for fuel types
- Create new merge columns to ensure a proper match.
- Merge df_copy with df_fuel_prices to get fuel prices for electricity, natural gas, propane, and fuel oil
- Calculate the per kWh fuel costs for each fuel type and region
- Calculate the baseline fuel cost 
-------------------------------------------------------------------------------------------------------
""")
# calculate_annual_fuelCost(df, menu_mp, policy_scenario, drop_fuel_cost_columns)
df_euss_am_baseline_home = calculate_annual_fuelCost(df=df_euss_am_baseline_home,
                                                     menu_mp=menu_mp,
                                                     policy_scenario='No Inflation Reduction Act',
                                                     drop_fuel_cost_columns=False
                                                     )
df_euss_am_baseline_home

# %% [markdown]
# # Model Runtime

# %%
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


