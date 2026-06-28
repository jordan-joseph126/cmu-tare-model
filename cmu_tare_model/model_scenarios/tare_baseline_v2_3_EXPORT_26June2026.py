# %%
import os

# Measure Package 0: Baseline
menu_mp = 0
input_mp = 'baseline'

# import from cmu-tare-model package
from config import PROJECT_ROOT
from cmu_tare_model.constants import ALLOWED_HOUSING_TYPES, VERBOSE, PRINT_VERBOSE_DATAFRAMES, EQUIPMENT_SPECS, VALID_CATEGORIES
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
=========================================================================================================
Welcome to the Trade-off Analysis of residential Retrofits for energy Equity Tool (TARE Model)
Let's start by reading the data from the NREL EUSS Database.

Make sure that the zipped folders stay organized as they are once unzipped.
If changes are made to the file path, then the program will not run properly.
==========================================================================================================
      
Project root directory: {PROJECT_ROOT}
Result outputs will be exported here: {output_folder_path}

""")

# %% [markdown]
# # LOAD EUSS DATA: Annual Energy Consumption and Metadata
# ## BASELINE (MP0): Metadata, Space Heating, Water Heating, Clothes Drying, Cooking

# %%
from cmu_tare_model.energy_consumption_and_metadata.user_input_geographic_filter import *
from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import *

print(F"""
====================================================================================================================================================================
LOAD EUSS DATA FOR BASELINE SCENARIO: MEASURE PACKAGE {menu_mp} (MP{menu_mp})
====================================================================================================================================================================
Load data from the NREL EUSS Database for the baseline scenario and apply various filters.

DATA: NREL EUSS Database
HOUSING FILTERS: Occupied units and Single Family Homes
GEOGRAPHIC FILTERS: National, State, or City

Additional details data can be found in the official End-Use Load Profiles/Savings Shapes documentation.
Our methodology is detailed in the process_euss_data.py file and its associated imports.

""")

# Measure Package 0: Baseline
filename = "baseline_metadata_and_annual_results.csv"
# relative_path = os.path.join("cmu_tare_model", "data", "euss_data", "resstock_amy2018_release_1.1", "national", "csv", filename)
relative_path = os.path.join("cmu_tare_model", "data", "euss_data", "resstock_amy2018_release_1.1", "national", "csv", filename)

file_path = os.path.join(PROJECT_ROOT, relative_path)

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")

# Fix DtypeWarning error in columns:
# 'in.neighbors', 'in.geometry_stories_low_rise', 'in.iso_rto_region', 'in.pv_orientation', 'in.pv_system_size'
columns_to_string = {11: str, 61: str, 121: str, 103: str, 113: str, 128: str, 129: str}
df_euss_am_baseline = pd.read_csv(file_path, dtype=columns_to_string, index_col="bldg_id") # UPDATE: Set index to 'bldg_id'

# Filter for occupied homes
occupancy_filter = df_euss_am_baseline['in.vacancy_status'] == 'Occupied'
df_euss_am_baseline = df_euss_am_baseline.loc[occupancy_filter]

# Filter for allowed housing types
house_type_filter = df_euss_am_baseline['in.geometry_building_type_recs'].isin(ALLOWED_HOUSING_TYPES)
df_euss_am_baseline = df_euss_am_baseline.loc[house_type_filter]
print(f"Allowed housing types: {ALLOWED_HOUSING_TYPES}")
print(f"DATAFRAME SIZE after filtering for allowed housing types: {df_euss_am_baseline.shape}")

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
====================================================================================================================================================================      
DATAFRAME: df_euss_am_baseline
      
{df_euss_am_baseline}
""")

# %% [markdown]
# ## Project Future Energy Consumption

# %%
print(F"""
====================================================================================================================================================================
LOAD EUSS DATA FOR BASELINE SCENARIO: MEASURE PACKAGE {menu_mp} (MP{menu_mp})
====================================================================================================================================================================
In addition to the housing type and occupancy filters, the data has been filtered for fuel and technology
Please see the energy_consumption_and_metadata folder for more details on data processing.

We employ the df_enduse_refactored function to process the data and create a new dataframe, which is part of the 5-step data validation process:
1. Mask Initialization with initialize_validation_tracking()
2. Series Initialization with create_retrofit_only_series()
3. Valid-Only Calculation for qualifying homes
4. Valid-Only Updates using list-based collection
5. Final Masking with apply_final_masking()

----------------------------------------------------------------

RESULTS OUTPUT (BASELINE):
""")

# df_baseline_enduse(df_baseline, df_enduse, category, fuel_filter='Yes', tech_filter='Yes')
df_euss_am_baseline_home = df_enduse_refactored(
    df_baseline = df_euss_am_baseline
    )

if PRINT_VERBOSE_DATAFRAMES:
    print(f"""
    ====================================================================================================================================================================
    DATAFRAME: df_euss_am_baseline_home
        
    {df_euss_am_baseline_home}
    """)

# %% [markdown]
# # PUBLIC IMPACTS (BASELINE): Climate and Health Damages

# %%
from cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity import *
from cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity import *

print(f"""
====================================================================================================================================================================
PUBLIC IMPACTS (BASELINE): DAMAGES FROM CLIMATE AND HEALTH-RELATED EMISSIONS
====================================================================================================================================================================
Detailed documentation of the methods used to calculate the climate and public health impacts can be
found in the public_impacts folder of the cmu-tare-model package. All monetary values are in 2023
inflation-adjusted dollars ($USD2023).
- data_processing sub-folder: contains the data processing scripts used for lookup dictionaries
- calculations sub-folder: contains the calculations for emissions from fossul fuel appliances
- Main folder contains the scripts for calculating lifetime climate and public health impacts as
  as well as climate NPV, health NPV, and combined public NPV.

Step 1: Calculate the baseline marginal damages for climate and health-related emissions
Step 2: Calculate the post-retrofit marginal damages for climate and health-related emissions
Step 3: Discount climate and health impacts and calculate lifetime public impacts (public NPV)

----------------------------------------------------------------------------------------------
Step 1: Calculate the baseline marginal damages for climate and health-related emissions
----------------------------------------------------------------------------------------------

""")

# %% [markdown]
# ## PUBLIC IMPACTS (BASELINE): Climate Damages

# %%
# Make copies to prevent overwriting the original dataframe and compare the differences
df_euss_am_baseline_home = df_euss_am_baseline_home.copy()
df_baseline_damages_climate = df_euss_am_baseline_home.copy()

print(f"""
==================== PUBLIC IMPACTS (BASELINE): DAMAGES FROM CLIMATE-RELATED EMISSIONS ====================

""")

# Modified usage pattern - keeping df_euss_am_baseline_home as a single DataFrame
# while keeping detailed results separate
df_euss_am_baseline_home, df_baseline_damages_climate = calculate_lifetime_climate_impacts(
    df=df_euss_am_baseline_home,
    menu_mp=0,  # baseline
    policy_scenario='2025 Reference Case',
    verbose=VERBOSE  # Add this parameter
)

# %%
# Climate Change Impacts: Baseline Scenario

if PRINT_VERBOSE_DATAFRAMES:
    print(f"""
    ====================================================================================================================================================================

    df_euss_am_baseline_home: DataFrame containing the baseline scenario data
    {df_euss_am_baseline_home}

    df_baseline_damages_climate: DataFrame containing the baseline scenario data with climate damages      
    {df_baseline_damages_climate}

    """)

# %% [markdown]
# ## PUBLIC IMPACTS (BASELINE): Health Damages

# %%
# Make copies to prevent overwriting the original dataframe and compare the differences
df_euss_am_baseline_home = df_euss_am_baseline_home.copy()
df_baseline_damages_health = df_euss_am_baseline_home.copy()

print(f"""
==================== PUBLIC IMPACTS (BASELINE): DAMAGES FROM HEALTH-RELATED EMISSIONS ====================

""")

df_euss_am_baseline_home, df_baseline_damages_health = calculate_lifetime_health_impacts(
    df=df_euss_am_baseline_home,
    menu_mp=0,  # baseline
    policy_scenario='2025 Reference Case',
    debug=False,
    verbose=VERBOSE  # Add this parameter
)

# Health Impacts: Baseline Scenario
if PRINT_VERBOSE_DATAFRAMES:
    print(f"""
    ====================================================================================================================================================================
    df_euss_am_baseline_home: DataFrame containing the baseline scenario data
    {df_euss_am_baseline_home}
        
    df_baseline_damages_health: DataFrame containing the baseline scenario data with health damages
    {df_baseline_damages_health}

    """)

# %% [markdown]
# # PRIVATE IMPACTS (BASELINE): Lifetime Fuel Costs

# %%
from cmu_tare_model.private_impact.calculate_lifetime_fuel_costs import *

print(f"""
=====================================================================================================================================================================
LIFETIME FUEL COSTS: Baseline Scenario
=====================================================================================================================================================================
STEPS
- Create a mapping dictionary for fuel types
- Create new merge columns to ensure a proper match.
- Merge df_copy with df_fuel_prices to get fuel prices for electricity, natural gas, propane, and fuel oil
- Calculate the per kWh fuel costs for each fuel type and region
- Calculate the baseline fuel cost 

----------------------------------------------------------------------------------------------------------------------
Step 1: Calculate annual operating (fuel) costs
----------------------------------------------------------------------------------------------------------------------

====================================================================================================================================================================
FUEL COSTS RESULTS (BASELINE): 

""")

# Make copies to prevent overwriting the original dataframe and compare the differences
df_euss_am_baseline_home = df_euss_am_baseline_home.copy()
df_baseline_fuel_costs = df_euss_am_baseline_home.copy()

# Returns df_main, df_detailed
df_euss_am_baseline_home, df_baseline_fuel_costs = calculate_lifetime_fuel_costs(
    df=df_euss_am_baseline_home,
    menu_mp=menu_mp,
    policy_scenario='2025 Reference Case'
    )

if PRINT_VERBOSE_DATAFRAMES:
    print(f"""
    ====================================================================================================================================================================

    df_euss_am_baseline_home: DataFrame containing the baseline scenario data
    {df_euss_am_baseline_home}
        
    df_baseline_fuel_costs: DataFrame containing the baseline scenario data with fuel costs
    {df_baseline_fuel_costs}

    """)

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


