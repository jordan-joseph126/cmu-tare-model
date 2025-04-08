# %% [markdown]
# # Load Util File with TARE Model Functions

# %%
import os

# Measure Package 0: Baseline
menu_mp = 0
input_mp = 'baseline'

# import from cmu-tare-model package
from config import PROJECT_ROOT
from cmu_tare_model.constants import EPA_SCC_USD2023_PER_MT_LOW, EPA_SCC_USD2023_PER_MT_BASE, EPA_SCC_USD2023_PER_MT_HIGH
import pandas as pd

# Set columns in display
# pd.set_option('display.max_columns', None)
# pd.reset_option('display.max_columns') # Reset options to default

# Set rows in display
# pd.set_option('display.max_rows', None)
# pd.reset_option('display.max_rows') # Reset options to default

# import seaborn library (wrapper of matplotlib)
import seaborn as sns
sns.set_theme(style="darkgrid")

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
from cmu_tare_model.energy_consumption_and_metadata.load_and_filter_euss_data_v2 import *

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
                                                tech_filter = 'Yes',
                                                invalid_row_handling='mask'
                                                )

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
from cmu_tare_model.energy_consumption_and_metadata.project_future_energy_consumption import *

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

from cmu_tare_model.utils.inflation_adjustment import *
print(f"""
--------------------------------------------------------------------------------------------------------------------------------------
Inflate Marginal Social Cost (Damage) Factors using BLS CPI for All Urban Consumers (CPI-U)
--------------------------------------------------------------------------------------------------------------------------------------
DATAFRAME: Annual CPI-U for 2005-2023 used for cpi_ratio constants and inflation adjustments
      
{df_bls_cpiu}
""")

# %% [markdown]
# ### For Health-Related Emissions Adjust for different Value of a Statistical Life (VSL) values


print(f"""
Steps 3 and 4: Obtain BLS CPI-U Data and Inflate Current Social Cost of Carbon (SCC) to USD2023
      
EPA Median for 2% near term discount rate and most commonly mentioned value is 190 USD-2020 using the GIVE model.
Inflate 190 $USD-2020 Social Cost of Carbon to $USD-2023

SCC Values used in analysis are:
      LOW: ${round(EPA_SCC_USD2023_PER_MT_LOW, 2)} per mt CO2e
      BASE: ${round(EPA_SCC_USD2023_PER_MT_BASE, 2)} per mt CO2e
      HIGH: ${round(EPA_SCC_USD2023_PER_MT_HIGH, 2)} per mt CO2e
""")

# %%
print(f"""
===========================================================================================================================================================
LOOKUP MARGINAL SOCIAL COSTS FOR HEALTH-RELATED EMISSIONS
===========================================================================================================================================================
No Electricity Grid Uncertainty (Just Current Grid and Future Grid Projections)
      

""")

# %% [markdown]
# ### Calculate End-use specific marginal damages
# **I used the total emissions column for each of the end uses for the following reasons:**
# - Most homes only have 1 of each end-use, so it is unlikely that the homes have a significant consumption values from different fuel types. Thus, the total consumption and total emissions column (sum of each dwelling units consumption by end-use for each fuel) is fine to use to calculate marginal damages (social cost)
# - We can visualize the emissions in 2 by 2 grid (CO2, PM25, SO2, NOx) with each appliance's heating fuel in a different shape or color. 

# %% [markdown]
# ### Baseline Marginal Damages: WHOLE-HOME

# %%
from cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity import *
from cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity import *
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

# calculate_lifetime_climate_impacts(df, menu_mp, policy_scenario, df_baseline_damages=None)
df_euss_am_baseline_home, df_baseline_scenario_damages = calculate_lifetime_climate_impacts(df=df_euss_am_baseline_home,
                                                                                   menu_mp=menu_mp,
                                                                                   policy_scenario='No Inflation Reduction Act',
                                                                                   df_baseline_damages=df_baseline_scenario_damages,
                                                                                   )
df_euss_am_baseline_home

# %%
df_baseline_scenario_damages

# %%
# calculate_lifetime_climate_impacts(df, menu_mp, policy_scenario, df_baseline_damages=None)
df_euss_am_baseline_home, df_baseline_scenario_damages = calculate_lifetime_health_impacts(df=df_euss_am_baseline_home,
                                                                                  menu_mp=menu_mp,
                                                                                  policy_scenario='No Inflation Reduction Act',
                                                                                  df_baseline_damages=df_baseline_scenario_damages,
                                                                                  )
df_euss_am_baseline_home

# %%
df_baseline_scenario_damages

# %% [markdown]
# ## Private Perspective: Annual Energy Costs

# %%
from cmu_tare_model.private_impact.process_fuel_price_data import *
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
from cmu_tare_model.private_impact.calculate_fuel_costs import *

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
