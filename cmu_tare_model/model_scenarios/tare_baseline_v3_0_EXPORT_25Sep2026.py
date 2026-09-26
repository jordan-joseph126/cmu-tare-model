# %%
import os

# Measure Package 0: Baseline
menu_mp = 0
input_mp = 'baseline'

# import from cmu-tare-model package
from config import PROJECT_ROOT
from cmu_tare_model.constants import (
    RESSTOCK_RELEASE_THIS_RUN,
    RESSTOCK_RELEASE_AND_MP,
    VALID_MENU_MPS,
    ALLOWED_HOUSING_TYPES,
    VERBOSE,
    PRINT_VERBOSE_DATAFRAMES,
    EQUIPMENT_SPECS,
    VALID_CATEGORIES
    )
import pandas as pd

PRINT_VERBOSE_DATAFRAMES = True

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
# # Load ResStock Data (0-Baseline): Annual Energy Consumption and Metadata

# %%
# TODO: Update to provide explicit imports and add techfilter to the table (should show 0% ASHP after that stage)

from cmu_tare_model.energy_consumption_and_metadata.user_input_geographic_filter import *
from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import *

# RESSTOCK_RELEASE_THIS_RUN is now imported from constants.py in cell 0

print(F"""
====================================================================================================================================================================
LOAD RESSTOCK DATA FOR BASELINE SCENARIO: 
MEASURE PACKAGE {menu_mp} (MP{menu_mp}) -- ResStock {RESSTOCK_RELEASE_THIS_RUN}
====================================================================================================================================================================
Load data from the NREL ResStock Database for the baseline scenario and apply various filters.

DATA: NREL ResStock Database
SCOPE FILTERS: ResStock applicability, occupied units, single-family homes, Alaska/Hawaii excluded (Phase 3)
GEOGRAPHIC FILTERS: National, State, or City

Additional details data can be found in the official End-Use Load Profiles/Savings Shapes documentation.
Our methodology is detailed in the process_euss_data.py file and its associated imports.

""")

# Measure Package 0: Baseline
# load_and_filter_upgrade replaces the inline occupancy/housing-type
# filtering this cell used to do directly, and adds two new leading filter
# stages ahead of them: ResStock's own applicability flag (Phase 3, D3) and
# the Alaska/Hawaii exclusion (Phase 3). state/city are left as None here --
# the geographic filter, if any, is applied below after the interactive
# prompt, since that prompt needs a DataFrame to validate the choice
# against before the filter itself can be applied.
print(f"Loading ResStock {RESSTOCK_RELEASE_THIS_RUN} baseline")
df_euss_am_baseline, df_funnel_baseline = load_and_filter_upgrade(
    menu_mp=0, verbose=True, release=RESSTOCK_RELEASE_THIS_RUN)
print(f"DATAFRAME SIZE after scope filters: {df_euss_am_baseline.shape}")

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

    # Record the geographic choice as its own funnel stage, the same way
    # load_and_filter_upgrade would if state/city had been known up
    # front -- kept as a separate step here because the interactive prompt
    # above needs a DataFrame to validate the choice against before the
    # filter can be applied.
    df_funnel_baseline = pd.concat([
        df_funnel_baseline,
        pd.DataFrame([compute_funnel_stage_row(
            df_euss_am_baseline, 'geographic_filter',
            heating_fuel_col='in.heating_fuel',
            heating_type_col='in.hvac_heating_type_and_fuel')]),
    ], ignore_index=True)
    print_masking_funnel_stage(
        df_euss_am_baseline, 'geographic_filter',
        heating_fuel_col='in.heating_fuel',
        heating_type_col='in.hvac_heating_type_and_fuel')

print(f"""
====================================================================================================================================================================      
DATAFRAME: df_euss_am_baseline (ResStock {RESSTOCK_RELEASE_THIS_RUN})
      
{df_euss_am_baseline}

FILTER FUNNEL:
{df_funnel_baseline}
""")


# %%
# =============================================================================
# STUDY SAMPLE -- which homes each measure package was applied to
# =============================================================================
# Load each package in the run with the same scope and geographic filters as
# the baseline. The bldg_ids feed include_sample (df_enduse_refactored) and the
# funnels feed the SI funnel table. Applicability is each funnel's first filter.
state_filter_for_packages = None if input_state == 'National' else input_state
city_filter_for_packages = (
    input_cityFilter if (state_filter_for_packages and menu_city == 'Y') else None)

applicable_bldg_ids = []
df_funnel_packages = []
for package_mp in [mp for mp in VALID_MENU_MPS if mp != 0]:
    df_package_filtered, df_package_funnel = load_and_filter_upgrade(
        menu_mp=package_mp,
        state=state_filter_for_packages,
        city=city_filter_for_packages,
        verbose=False,
        release=RESSTOCK_RELEASE_THIS_RUN,
    )
    applicable_bldg_ids.append(df_package_filtered.index)
    df_funnel_packages.append(df_package_funnel)
    print(f"MP{package_mp}: {len(df_package_filtered):,} rdu applicable "
          f"after scope filters")
    del df_package_filtered  # only the ids are needed; frees about 2 GB per package


# %% [markdown]
# ## Project Future Energy Consumption

# %%
print(F"""
====================================================================================================================================================================
LOAD RESSTOCK DATA FOR BASELINE SCENARIO: 
MEASURE PACKAGE {menu_mp} (MP{menu_mp}) -- ResStock {RESSTOCK_RELEASE_THIS_RUN}
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

df_euss_am_baseline_home = df_enduse_refactored(
    df_baseline=df_euss_am_baseline,
    applicable_bldg_ids=applicable_bldg_ids,
    release=RESSTOCK_RELEASE_THIS_RUN,
    )

# Study sample ids and the SI funnel table, both read from include_sample
from cmu_tare_model.energy_consumption_and_metadata.study_sample import (
    build_sample_funnel,
    build_tare_sample_ids,
)
TARE_SAMPLE_IDS = build_tare_sample_ids(
    df_euss_am_baseline_home, RESSTOCK_RELEASE_THIS_RUN)
df_sample_funnel = build_sample_funnel(
    df_funnel_packages, applicable_bldg_ids, df_euss_am_baseline_home)
print(df_sample_funnel[['stage', 'rdu_count', 'weighted_count',
                        'removed_rdu', 'removed_homes']].to_string())

if PRINT_VERBOSE_DATAFRAMES:
    print(f"""
    ====================================================================================================================================================================
    DATAFRAME: df_euss_am_baseline_home
        
    {df_euss_am_baseline_home}
    """)


# %% [markdown]
# # PUBLIC IMPACTS (BASELINE): Climate Damages

# %%
from cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity import *

print(f"""
====================================================================================================================================================================
CLIMATE IMPACTS (BASELINE): DAMAGES FROM CLIMATE-RELATED EMISSIONS
====================================================================================================================================================================
Detailed documentation of the methods used to calculate the climate impacts can be
found in the public_impacts folder of the cmu-tare-model package. All monetary values are in 2023
inflation-adjusted dollars ($USD2023).
- data_processing sub-folder: contains the data processing scripts used for lookup dictionaries
- calculations sub-folder: contains the calculations for emissions from fossil fuel appliances
- Main folder contains the scripts for calculating lifetime climate impacts and climate NPV.

Step 1: Calculate the baseline marginal damages for climate-related emissions

----------------------------------------------------------------------------------------------
Step 1: Calculate the baseline marginal damages for climate-related emissions
----------------------------------------------------------------------------------------------

""")

# %%
# Climate Change Impacts: Baseline Scenario
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

if PRINT_VERBOSE_DATAFRAMES:
    print(f"""
    ====================================================================================================================================================================

    df_euss_am_baseline_home: DataFrame containing the baseline scenario data
    {df_euss_am_baseline_home}

    df_baseline_damages_climate: DataFrame containing the baseline scenario data with climate damages      
    {df_baseline_damages_climate}

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


