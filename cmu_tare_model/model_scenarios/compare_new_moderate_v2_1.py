# %%
# import os

# # import from cmu-tare-model package
# from config import PROJECT_ROOT

# # Measure Package 0: Baseline
# menu_mp = 0
# input_mp = 'baseline'

# print(f"PROJECT_ROOT (from config.py): {PROJECT_ROOT}")

# # Construct the absolute path to the .py file
# relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_baseline_v2_1.ipynb")
# file_path = os.path.join(PROJECT_ROOT, relative_path)

# # On Windows, to avoid any path-escape quirks, convert backslashes to forward slashes
# file_path = file_path.replace("\\", "/")

# print(f"Running file: {file_path}")

# # %run magic command to run a .py file and import variables into the current IPython session
# # # If your path has spaces, wrap it in quotes:
# %run -i {file_path} # If your path has NO spaces, no quotes needed.

# print("Baseline Scenario - Model Run Complete")

# %%


# %% [markdown]
# # LOAD EUSS DATA: Annual Energy Consumption and Metadata
# ## MEASURE PACKAGE 7 (MP7): Data for Electric Resistance Cooking

# %%
print(f"""
====================================================================================================================================================================
We assume the use of Electric Resistance (MP7) rather than Induction (mp9).
Electric Resistance is significantly cheaper and only slightly less efficient than Induction.
====================================================================================================================================================================
""")

# Measure Package 7
menu_mp = 7
input_mp = 'upgrade07'

filename = "upgrade07_metadata_and_annual_results.csv"
relative_path = os.path.join("cmu_tare_model", "data", "euss_data", "resstock_amy2018_release_1.1", "state", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")
print("\n")

# Fix DtypeWarning error in columns 'in.neighbors' and 'in.geometry_stories_low_rise'
columns_to_string = {11: str, 61: str, 121: str, 103: str, 128: str, 129: str}
df_euss_am_mp7 = pd.read_csv(file_path, dtype=columns_to_string, index_col="bldg_id") # UPDATE: Set index to 'bldg_id' (unique identifier)
print(f"DATAFRAME SIZE before applying any filters: {df_euss_am_mp7.shape}")


# Filter for occupied homes
occupancy_filter = df_euss_am_mp7['in.vacancy_status'] == 'Occupied'
df_euss_am_mp7 = df_euss_am_mp7.loc[occupancy_filter]
print(f"DATAFRAME SIZE after filtering for 'Occupied' homes: {df_euss_am_mp7.shape}")

# Filter for single family home building type
house_type_list = ['Single-Family Attached', 'Single-Family Detached']
house_type_filter = df_euss_am_mp7['in.geometry_building_type_recs'].isin(house_type_list)
df_euss_am_mp7 = df_euss_am_mp7.loc[house_type_filter]
print(f"DATAFRAME SIZE after filtering for 'Single-Family Attached' and 'Single-Family Detached' homes: {df_euss_am_mp7.shape}")

# National Level 
if menu_state == 'N':
    print("You chose to analyze all of the United States.")
    input_state = 'National'

# Filter down to state or city
else:
    print(f"You chose to filter for: {input_state}")
    state_filter = df_euss_am_mp7['in.state'].eq(input_state)
    df_euss_am_mp7 = df_euss_am_mp7.loc[state_filter]

    # Filter for the entire selected state
    if menu_city == 'N':
        print(f"You chose to analyze all of state: {input_state}")
        
    # Filter to a city within the selected state
    else:
        print(f"You chose to filter for: {input_state}, {input_cityFilter}")
        city_filter = df_euss_am_mp7['in.city'].eq(f"{input_state}, {input_cityFilter}")
        df_euss_am_mp7 = df_euss_am_mp7.loc[city_filter]

# Display the filtered dataframe
print(f"DATAFRAME SIZE after applying geographic filter: {df_euss_am_mp7.shape}")
print(df_euss_am_mp7)

# %% [markdown]
# ## MEASURE PACKAGE 9 (MP9): Metadata, Space Heating, Water Heating, and Clothes Drying

# %%
# Print debugging information
print_debug = True

if print_debug:
    from cmu_tare_model.utils.create_sample_df import *

# %%
# Measure Package 9
menu_mp = 9
input_mp = 'upgrade09'
scenario_name = 'Moderate-BAU'
cost_scenario = 'BAU Costs'
grid_scenario = 'Current Electricity Grid'

print(f"""
====================================================================================================================================================================
MODEL SCENARIO
====================================================================================================================================================================
Scenario {scenario_name}:
Moderate Retrofit: Measure Package {menu_mp}
{cost_scenario}
{grid_scenario}
====================================================================================================================================================================
""")

filename = "upgrade09_metadata_and_annual_results.csv"
relative_path = os.path.join("cmu_tare_model", "data", "euss_data", "resstock_amy2018_release_1.1", "state", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")
print("\n")

# Fix DtypeWarning error in columns 'in.neighbors' and 'in.geometry_stories_low_rise'
columns_to_string = {11: str, 61: str, 121: str, 103: str, 128: str, 129: str}
df_euss_am_mp9 = pd.read_csv(file_path, dtype=columns_to_string, index_col="bldg_id") # UPDATE: Set index to 'bldg_id' (unique identifier)
print(f"DATAFRAME SIZE before applying any filters: {df_euss_am_mp9.shape}")

# Filter for occupied homes
occupancy_filter = df_euss_am_mp9['in.vacancy_status'] == 'Occupied'
df_euss_am_mp9 = df_euss_am_mp9.loc[occupancy_filter]
print(f"DATAFRAME SIZE after filtering for 'Occupied' homes: {df_euss_am_mp9.shape}")

# Filter for single family home building type
house_type_list = ['Single-Family Attached', 'Single-Family Detached']
house_type_filter = df_euss_am_mp9['in.geometry_building_type_recs'].isin(house_type_list)
df_euss_am_mp9 = df_euss_am_mp9.loc[house_type_filter]
print(f"DATAFRAME SIZE after filtering for 'Single-Family Attached' and 'Single-Family Detached' homes: {df_euss_am_mp9.shape}")

# National Level 
if menu_state == 'N':
    print("You chose to analyze all of the United States.")
    input_state = 'National'

# Filter down to state or city
else:
    print(f"You chose to filter for: {input_state}")
    state_filter = df_euss_am_mp9['in.state'].eq(input_state)
    df_euss_am_mp9 = df_euss_am_mp9.loc[state_filter]

    # Filter for the entire selected state
    if menu_city == 'N':
        print(f"You chose to analyze all of state: {input_state}")
        
    # Filter to a city within the selected state
    else:
        print(f"You chose to filter for: {input_state}, {input_cityFilter}")
        city_filter = df_euss_am_mp9['in.city'].eq(f"{input_state}, {input_cityFilter}")
        df_euss_am_mp9 = df_euss_am_mp9.loc[city_filter]

# Display the filtered dataframe
# Display the filtered dataframe
print(f"DATAFRAME SIZE after applying geographic filter: {df_euss_am_mp9.shape}")
print(df_euss_am_mp9)

# %% [markdown]
# # Project Future Energy Consumption

# %%
from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import df_enduse_compare

print(F"""
====================================================================================================================================================================
LOAD EUSS DATA FOR MEASURE PACKAGE {menu_mp} (MP{menu_mp})
====================================================================================================================================================================
You'll notice that the number of rows differs from df_euss_am_mp7 and df_euss_am_mp9.
      - df_euss_am_baseline_home has fewer rows (representative dwelling units) because a tech filter was applied. 
      - df_euss_am_mp9_home will have the same number of rows as df_euss_am_baseline_home after df_enduse_compare function is run.
      - df_enduse_compare function performs an inner merge on the two dataframes, keeping only the rows that are present in both dataframes.
====================================================================================================================================================================
df_euss_am_mp9_home will be created by running the df_enduse_compare function (contains post-retrofit consumption data for the entire home in 2024).
process_euss_data.py file contains the function definition.
      
Documentation for df_enduse_compare function:
{df_enduse_compare.__doc__}
----------------------------------------------------------------

RESULTING DATAFRAME:
""")

# df_enduse_compare(df_mp, category, df_baseline):
df_euss_am_mp9_home = df_enduse_compare(
    df_mp = df_euss_am_mp9,
    input_mp=input_mp,
    menu_mp=menu_mp,
    df_baseline = df_euss_am_baseline_home,
    df_cooking_range=df_euss_am_mp7,
    )


# %%
from cmu_tare_model.energy_consumption_and_metadata.project_future_energy_consumption import *

print(f"""
====================================================================================================================================================================
PROJECT FUTURE ENERGY CONSUMPTION
====================================================================================================================================================================
project_future_energy_consumption.py file was used to calculate/project the annual energy consumption 
for each home in the dataframe.
      
Documentation for project_future_energy_consumption function:
{project_future_consumption.__doc__}

RESULTS OUTPUT:
""")

# Make copies to avoid modifying the original dataframes
df_euss_am_mp9_home = df_euss_am_mp9_home.copy()
df_mp9_consumption = df_euss_am_mp9_home.copy()

# Project Future Energy Consumption
df_euss_am_mp9_home, df_mp9_consumption = project_future_consumption(
    df=df_euss_am_mp9_home,
    menu_mp=menu_mp,
    base_year=2024
    )

print(f"""
====================================================================================================================================================================
DATAFRAME FOR MEASURE PACKAGE {menu_mp} (MP{menu_mp}):

{df_euss_am_mp9_home}
     
DATAFRAME (Consumption Cols): df_mp9_consumption
      
{df_mp9_consumption}

""")

# %%
if print_debug:
    # Create a sample dataframe for the heating category
    df_sample_heating = create_sample_df(
        df=df_euss_am_mp9_home,
        include_groups=['base_equipment'],
        categories=['heating'],
        scenarios=['preIRA', 'iraRef'],
        metrics=[],
        mp_number=menu_mp,
        regex_patterns=['upgrade_heating', 'valid_fuel_heating', 'valid_tech_heating', 'include_heating', 'heating_consumption']
    )
    print(df_sample_heating)

    # Create a sample dataframe for the waterHeating category
    df_sample_waterHeating = create_sample_df(
        df=df_euss_am_mp9_home,
        include_groups=['base_equipment'],
        categories=['waterHeating'],
        scenarios=['preIRA', 'iraRef'],
        metrics=[],
        mp_number=menu_mp,
        regex_patterns=['upgrade_waterHeating', 'valid_fuel_waterHeating', 'valid_tech_waterHeating', 'include_waterHeating', 'waterHeating_consumption']
    )
    print(df_sample_waterHeating)

    # Create a sample dataframe for the clothesDrying category
    df_sample_clothesDrying = create_sample_df(
        df=df_euss_am_mp9_home,
        include_groups=['base_equipment'],
        categories=['clothesDrying'],
        scenarios=['preIRA', 'iraRef'],
        metrics=[],
        mp_number=menu_mp,
        regex_patterns=['upgrade_clothesDrying', 'valid_fuel_clothesDrying', 'include_clothesDrying', 'clothesDrying_consumption']
    )
    print(df_sample_clothesDrying)

    # Create a sample dataframe for the cooking category
    df_sample_cooking = create_sample_df(
        df=df_euss_am_mp9_home,
        include_groups=['base_equipment'],
        categories=['cooking'],
        scenarios=['preIRA', 'iraRef'],
        metrics=[],
        mp_number=menu_mp,
        regex_patterns=['upgrade_cooking', 'valid_fuel_cooking', 'include_cooking', 'cooking_consumption']
    )
    print(df_sample_cooking)

# %% [markdown]
# # PUBLIC IMPACTS: Climate and Health Damages
# ## Scenarios: No IRA and IRA-Reference

# %%
from cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity import calculate_lifetime_climate_impacts
from cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity import calculate_lifetime_health_impacts

print(f"""
====================================================================================================================================================================
PUBLIC IMPACTS: DAMAGES FROM CLIMATE AND HEALTH-RELATED EMISSIONS
====================================================================================================================================================================

----------------------------------------------------------------------------------------------
Step 1: Calculate the baseline marginal damages for climate and health-related emissions
----------------------------------------------------------------------------------------------
Done in baseline_v2_1.ipynb

----------------------------------------------------------------------------------------------
Step 2: Calculate the post-retrofit marginal damages for climate and health-related emissions 
----------------------------------------------------------------------------------------------      
Documentation for calculate_lifetime_climate_impacts_sensitivity function:
{calculate_lifetime_climate_impacts.__doc__}
      
Documentation for calculate_lifetime_health_impacts_sensitivity function:
{calculate_lifetime_health_impacts.__doc__}

---------------------------------------------------------------------------------------------
Step 3: Discount climate and health impacts and calculate lifetime public impacts (public NPV)
---------------------------------------------------------------------------------------------
Explained and calculated late in the notebook.

""")

# Make copies from scenario consumption to keep df smaller
print("\n", "Creating dataframe to store marginal damages calculations ...")
# Climate damages: No IRA and IRA-Reference
df_mp9_noIRA_damages_climate = df_mp9_consumption.copy()
df_mp9_IRA_damages_climate = df_mp9_consumption.copy()

# Health damages: No IRA and IRA-Reference
df_mp9_noIRA_damages_health = df_mp9_consumption.copy()
df_mp9_IRA_damages_health = df_mp9_consumption.copy()


# %%
print("""
==================== SCENARIO: No Inflation Reduction Act ==========
""")
df_euss_am_mp9_home, df_mp9_noIRA_damages_climate = calculate_lifetime_climate_impacts(
    df=df_euss_am_mp9_home,
    menu_mp=menu_mp, 
    policy_scenario='No Inflation Reduction Act', 
    df_baseline_damages=df_baseline_damages_climate,
    verbose=True  # Add this parameter
    )

df_euss_am_mp9_home, df_mp9_noIRA_damages_health = calculate_lifetime_health_impacts(
    df=df_euss_am_mp9_home,
    menu_mp=menu_mp, 
    policy_scenario='No Inflation Reduction Act', 
    df_baseline_damages=df_baseline_damages_health,
    debug=False,
    verbose=True  # Add this parameter
    )


print("""
==================== SCENARIO: Inflation Reduction Act (AEO2023 Reference Case) ==========
""")
df_euss_am_mp9_home, df_mp9_IRA_damages_climate = calculate_lifetime_climate_impacts(
    df=df_euss_am_mp9_home,
    menu_mp=menu_mp, 
    policy_scenario='AEO2023 Reference Case', 
    df_baseline_damages=df_baseline_damages_climate,
    verbose=True  # Add this parameter
    )


df_euss_am_mp9_home, df_mp9_IRA_damages_health = calculate_lifetime_health_impacts(
    df=df_euss_am_mp9_home,
    menu_mp=menu_mp, 
    policy_scenario='AEO2023 Reference Case', 
    df_baseline_damages=df_baseline_damages_health,
    debug=False,
    verbose=True  # Add this parameter
    )


print(f"""  
====================================================================================================================================================================
Post-Retrofit (mp9) Marginal Damages: WHOLE-HOME
Scenario: No Inflation Reduction Act and AEO2023 Reference Case
====================================================================================================================================================================
calculate_emissions_damages.py file contains the definition for the calculate_marginal_damages function.
Additional information on emissions and damage factor lookups can be found in the calculate_emissions_damages.py file as well. 
      
CLIMATE DAMAGES: No IRA and IRA-Reference
--------------------------------------------------------
Climate Damages (No IRA): df_mp9_noIRA_damages_climate
{df_mp9_noIRA_damages_climate}

Climate Damages (IRA): df_mp9_IRA_damages_climate
{df_mp9_IRA_damages_climate}

HEALTH DAMAGES: No IRA and IRA-Reference
--------------------------------------------------------
Health Damages (No IRA): df_mp9_noIRA_damages_health
{df_mp9_noIRA_damages_health}

Health Damages (IRA): df_mp9_IRA_damages_health
{df_mp9_IRA_damages_health}

SUMMARY DATAFRAME FOR mp9: df_euss_am_mp9_home
{df_euss_am_mp9_home}
====================================================================================================================================================================
""")

# %%
if print_debug:
   # Create a sample dataframe for the heating category
    df_sample_heating = create_sample_df(
        df=df_euss_am_mp9_home,
        include_groups=['base_equipment'],
        categories=['heating'],
        scenarios=['preIRA', 'iraRef'],
        metrics=[],
        mp_number=menu_mp,
        regex_patterns=['upgrade_heating', 'valid_fuel_heating', 'include_heating', 'heating_lifetime_damages']
    )
    print(df_sample_heating)

    # Create a sample dataframe for the waterHeating category
    df_sample_waterHeating = create_sample_df(
        df=df_euss_am_mp9_home,
        include_groups=['base_equipment'],
        categories=['waterHeating'],
        scenarios=['preIRA', 'iraRef'],
        metrics=[],
        mp_number=menu_mp,
        regex_patterns=['upgrade_waterHeating', 'valid_fuel_waterHeating', 'include_waterHeating', 'waterHeating_lifetime_damages']
    )
    print(df_sample_waterHeating)

    # Create a sample dataframe for the clothesDrying category
    df_sample_clothesDrying = create_sample_df(
        df=df_euss_am_mp9_home,
        include_groups=['base_equipment'],
        categories=['clothesDrying'],
        scenarios=['preIRA', 'iraRef'],
        metrics=[],
        mp_number=menu_mp,
        regex_patterns=['upgrade_clothesDrying', 'valid_fuel_clothesDrying', 'include_clothesDrying', 'clothesDrying_lifetime_damages']
    )
    print(df_sample_clothesDrying)

    # Create a sample dataframe for the cooking category
    df_sample_cooking = create_sample_df(
        df=df_euss_am_mp9_home,
        include_groups=['base_equipment'],
        categories=['cooking'],
        scenarios=['preIRA', 'iraRef'],
        metrics=[],
        mp_number=menu_mp,
        regex_patterns=['upgrade_cooking', 'valid_fuel_cooking', 'include_cooking', 'cooking_lifetime_damages']
    )
    print(df_sample_cooking)

# %%
if print_debug:
    print("""
==================== SUMMARY DATAFRAME WITH LIFETIME DAMAGES ==========
""")
    # Create a sample dataframe for the heating category
    df_main_sample = create_sample_df(
        df=df_euss_am_mp9_home,
        include_groups=['base_equipment'],
        categories=['cooking'],
        scenarios=['preIRA', 'iraRef'],
        metrics=[],
        mp_number=menu_mp,
        regex_patterns=['upgrade_cooking', 'valid_fuel_cooking', 'include_cooking', 'cooking_lifetime_damages']
    )
    print(f"""
df_main_sample dataframe is created using df_euss_am_mp9_home:
-----------------------------------------------------------------------------

{df_main_sample}

-----------------------------------------------------------------------------
""")


    print("""
==================== CLIMATE IMPACTS WITH ANNUAL AND LIFETIME ==========
""")

    # Create a sample dataframe for the heating category
    df_detailed_climate_noIRA = create_sample_df(
        df=df_mp9_noIRA_damages_climate,
        include_groups=['base_equipment'],
        categories=['cooking'],
        scenarios=['preIRA'],
        metrics=[],
        mp_number=menu_mp,
        regex_patterns=['upgrade_cooking', 'valid_fuel_cooking', 'include_cooking', f'cooking_damages_health']
    )
    print(f"""
df_detailed_climate_noIRA dataframe is created using df_mp9_noIRA_damages_climate:
-----------------------------------------------------------------------------
          
{df_detailed_climate_noIRA}

-----------------------------------------------------------------------------
""")

    # Create a sample dataframe for the heating category
    df_detailed_climate_IRA = create_sample_df(
        df=df_mp9_IRA_damages_climate,
        include_groups=['base_equipment'],
        categories=['cooking'],
        scenarios=['iraRef'],
        metrics=[],
        mp_number=menu_mp,
        regex_patterns=['upgrade_cooking', 'valid_fuel_cooking', 'include_cooking', f'cooking_damages_health']
    )
    print(f"""
df_detailed_climate_IRA dataframe is created using df_mp9_IRA_damages_climate:
-----------------------------------------------------------------------------
          
{df_detailed_climate_IRA}

-----------------------------------------------------------------------------
""")

    print("""
==================== HEALTH IMPACTS WITH ANNUAL AND LIFETIME ==========
""")
    # Create a sample dataframe for the heating category
    df_detailed_health_noIRA = create_sample_df(
        df=df_mp9_noIRA_damages_health,
        include_groups=['base_equipment'],
        categories=['cooking'],
        scenarios=['preIRA'],
        metrics=[],
        mp_number=menu_mp,
        regex_patterns=['upgrade_cooking', 'valid_fuel_cooking', 'include_cooking', f'cooking_damages_health']
    )
    print(f"""
df_detailed_health_noIRA dataframe is created using df_mp9_noIRA_damages_health:
-----------------------------------------------------------------------------
          
{df_detailed_health_noIRA}

-----------------------------------------------------------------------------
""")

    # Create a sample dataframe for the heating category
    df_detailed_health_IRA = create_sample_df(
        df=df_mp9_IRA_damages_health,
        include_groups=['base_equipment'],
        categories=['cooking'],
        scenarios=['iraRef'],
        metrics=[],
        mp_number=menu_mp,
        regex_patterns=['upgrade_cooking', 'valid_fuel_cooking', 'include_cooking', f'cooking_damages_health']
    )
    print(f"""
df_detailed_health_IRA dataframe is created using df_mp9_IRA_damages_health:
-----------------------------------------------------------------------------
          
{df_detailed_health_IRA}

-----------------------------------------------------------------------------
""")

# %% [markdown]
# # PRIVATE IMPACTS: FUEL COSTS
# ## Scenarios: No IRA and IRA-Reference

# %%
from cmu_tare_model.private_impact.calculate_lifetime_fuel_costs import *

print(f"""
====================================================================================================================================================================
PRIVATE IMPACTS: OVERVIEW
====================================================================================================================================================================
Step 1: Calculate annual operating (fuel) costs
Step 2: Calculate equipment capital costs (For space heating, include ductwork and weatherization (MP9 and MP10))
Step 3: Calculate replacement cost (replacing existing piece of eqipment with similar technology)
Step 4: Calculate net equipment capital costs and private NPV (less WTP and more WTP)

----------------------------------------------------------------------------------------------------------------------
Step 1: Calculate annual operating (fuel) costs
----------------------------------------------------------------------------------------------------------------------

Documentation for calculate_lifetime_fuel_costs function:
{calculate_lifetime_fuel_costs.__doc__}

====================================================================================================================================================================
FUEL COSTS RESULTS: No IRA and IRA-Reference

""")

print("\n", "Creating dataframe to store annual fuel cost calculations ...")
# Annual fuel costs: No IRA and IRA-Reference
df_mp9_fuel_costs_noIRA = df_mp9_consumption.copy()
df_mp9_fuel_costs_IRA = df_mp9_consumption.copy()

# %%
print("""
==================== SCENARIO: No Inflation Reduction Act ==========
""")
df_euss_am_mp9_home, df_mp9_noIRA_fuel_costs = calculate_lifetime_fuel_costs(
    df=df_euss_am_mp9_home,
    menu_mp=menu_mp,
    policy_scenario='No Inflation Reduction Act',
    df_baseline_costs=df_baseline_fuel_costs  # Add this line
    )

print("""
==================== SCENARIO: Inflation Reduction Act (AEO2023 Reference Case) ==========
""")
df_euss_am_mp9_home, df_mp9_IRA_fuel_costs = calculate_lifetime_fuel_costs(
    df=df_euss_am_mp9_home,
    menu_mp=menu_mp,
    policy_scenario='AEO2023 Reference Case',
    df_baseline_costs=df_baseline_fuel_costs  # Add this line
    )


print(f"""  
====================================================================================================================================================================
Creating Dataframes for Lifetime Fuel Costs ...

FUEL COSTS (No IRA): df_mp9_noIRA_fuel_costs
{df_mp9_noIRA_fuel_costs}

FUEL COSTS (IRA): df_mp9_IRA_fuel_costs
{df_mp9_IRA_fuel_costs}

SUMMARY DATAFRAME FOR mp9: df_euss_am_mp9_home
{df_euss_am_mp9_home}

====================================================================================================================================================================
""")

# %%
if print_debug:
    # Create a sample dataframe for the heating category
    df_main_sample = create_sample_df(
        df=df_euss_am_mp9_home,
        include_groups=['base_equipment', 'costs'],
        categories=['cooking'],
        scenarios=['preIRA', 'iraRef'],
        metrics=[],
        mp_number=menu_mp,
        regex_patterns=['upgrade_cooking', 'valid_fuel_cooking', 'include_cooking', 'cooking_lifetime_fuel_cost']
    )
    print(f"""
df_main_sample dataframe is created using df_euss_am_mp9_home:
-----------------------------------------------------------------------------

{df_main_sample}

-----------------------------------------------------------------------------
""")

    # Create a sample dataframe for the heating category
    df_detailed_test = create_sample_df(
        df=df_mp9_IRA_fuel_costs,
        include_groups=['base_equipment', 'costs'],
        categories=['cooking'],
        scenarios=['preIRA', 'iraRef'],
        metrics=[],
        mp_number=menu_mp,
        regex_patterns=['upgrade_cooking', 'valid_fuel_cooking', 'include_cooking', f'_fuel_cost']
    )
    print(f"""
df_detailed_test dataframe is created using df_mp9_IRA_fuel_costs:
-----------------------------------------------------------------------------
          
{df_detailed_test}

-----------------------------------------------------------------------------
""")

    # # Create a sample dataframe for the waterHeating category
    # df_sample_waterHeating = create_sample_df(
    #     df=df_euss_am_mp9_home,
    #     include_groups=['base_equipment', 'costs'],
    #     categories=['waterHeating'],
    #     scenarios=['preIRA', 'iraRef'],
    #     metrics=[],
    #     mp_number=menu_mp,
    #     regex_patterns=['upgrade_waterHeating', 'valid_fuel_waterHeating', 'include_waterHeating', 'waterHeating_lifetime_fuel_cost']
    # )
    # print(df_sample_waterHeating)

    # # Create a sample dataframe for the clothesDrying category
    # df_sample_clothesDrying = create_sample_df(
    #     df=df_euss_am_mp9_home,
    #     include_groups=['base_equipment', 'costs'],
    #     categories=['clothesDrying'],
    #     scenarios=['preIRA', 'iraRef'],
    #     metrics=[],
    #     mp_number=menu_mp,
    #     regex_patterns=['upgrade_clothesDrying', 'valid_fuel_clothesDrying', 'include_clothesDrying', 'clothesDrying_lifetime_fuel_cost']
    # )
    # print(df_sample_clothesDrying)

    # # Create a sample dataframe for the cooking category
    # df_sample_cooking = create_sample_df(
    #     df=df_euss_am_mp9_home,
    #     include_groups=['base_equipment', 'costs'],
    #     categories=['cooking'],
    #     scenarios=['preIRA', 'iraRef'],
    #     metrics=[],
    #     mp_number=menu_mp,
    #     regex_patterns=['upgrade_cooking', 'valid_fuel_cooking', 'include_cooking', 'cooking_lifetime_fuel_cost']
    # )
    # print(df_sample_cooking)

# %% [markdown]
# # PRIVATE IMPACTS: CAPITAL COSTS
# ## Scenarios: No IRA and IRA-Reference

# %%
# UPDATED MARCH 24, 2025 @ 4:30 PM - REMOVED RSMEANS CCI ADJUSTMENTS
from cmu_tare_model.utils.inflation_adjustment import *

# UPDATED MARCH 24, 2025 @ 4:30 PM - REMOVED RSMEANS CCI ADJUSTMENTS
from cmu_tare_model.private_impact.calculations.calculate_equipment_installation_costs import *
from cmu_tare_model.private_impact.calculations.calculate_equipment_replacement_costs import *

print(f"""
====================================================================================================================================================================
PRIVATE IMPACTS: NET CAPITAL COSTS AND TOTAL CAPITAL COSTS
====================================================================================================================================================================
Completed Steps:
1. Calculate annual operating (fuel) costs

----------------------------------------------------------------------------------------------------------------------
Step 2: Calculate equipment capital costs (For space heating, include ductwork and weatherization (MP9 and MP10))
----------------------------------------------------------------------------------------------------------------------
Documentation for obtain_heating_system_specs function (HVAC specific):
{obtain_heating_system_specs.__doc__}

Documentation for calculate_heating_installation_premium function (HVAC specific):
{calculate_heating_installation_premium.__doc__}

Moderate and Advanced scenarios also use calculate_enclosure_upgrade_costs function.


Documentation for calculate_installation_cost function:
{calculate_installation_cost.__doc__}

----------------------------------------------------------------------------------------------------------------------
Step 3: Calculate replacement cost (replacing existing piece of eqipment with similar technology)
----------------------------------------------------------------------------------------------------------------------
Documentation for calculate_replacement_cost function:
{calculate_replacement_cost.__doc__}

====================================================================================================================================================================
LIFETIME CAPITAL COSTS RESULTS: No IRA and IRA-Reference (Rebates)

""")

# %%
# Collect Capital Cost Data for different End-uses
filename = "tare_retrofit_costs_cpi.xlsx"
relative_path = os.path.join("cmu_tare_model", "data", "retrofit_costs", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")
print("\n")

df_heating_retrofit_costs = pd.read_excel(io=file_path, sheet_name='heating_costs')
df_waterHeating_retrofit_costs = pd.read_excel(io=file_path, sheet_name='waterHeating_costs')
df_clothesDrying_retrofit_costs = pd.read_excel(io=file_path, sheet_name='clothesDrying_costs')
df_cooking_retrofit_costs = pd.read_excel(io=file_path, sheet_name='cooking_costs')
df_enclosure_retrofit_costs = pd.read_excel(io=file_path, sheet_name='enclosure_upgrade_costs')

# %% [markdown]
# ### Space Heating and No Enclosure Upgrade

# %% [markdown]
# #### Space Heating Capital Costs

# %%
print("""
========================== Capital Costs: Space Heating ==========================

Obtaining Capital Cost Data from Retrofit Cost Spreadsheet ...
""")

# Columns to update
cost_columns = [
    'unitCost_progressive', 'unitCost_reference', 'unitCost_conservative',
    'cost_per_kBtuh_progressive', 'cost_per_kBtuh_reference', 'cost_per_kBtuh_conservative',
    'otherCost_progressive', 'otherCost_reference', 'otherCost_conservative'
]

# Update each cost column by multiplying with cpi_ratio and cost_multiplier
for column in cost_columns:
    df_heating_retrofit_costs[column] = round((df_heating_retrofit_costs[column] * df_heating_retrofit_costs['cpi_ratio'] * df_heating_retrofit_costs['cost_multiplier']), 2)

# Creating a dictionary from the DataFrame
dict_heating_equipment_cost = df_heating_retrofit_costs.set_index(['technology', 'efficiency']).to_dict(orient='index')

# Call the function and obtain equipment specifications
print("Obtaining heating system specs ...")
df_euss_am_mp9_home = obtain_heating_system_specs(df=df_euss_am_mp9_home)

# calculate_installation_cost(df, cost_dict, menu_mp, end_use)
print("Calculating Cost of Retrofit Upgrade: Heat Pump for Space Heating (No Enclosure Upgrade) ...")
df_euss_am_mp9_home = calculate_installation_cost(df=df_euss_am_mp9_home,
                                                  cost_dict=dict_heating_equipment_cost,
                                                  menu_mp=menu_mp,
                                                  end_use='heating')

# calculate_replacement_cost(df, cost_dict, menu_mp, end_use)
print("Calculating Cost of Replacing Existing Equipment with Similar Model/Efficiency ...")
df_euss_am_mp9_home = calculate_replacement_cost(df=df_euss_am_mp9_home,
                                                 cost_dict=dict_heating_equipment_cost,
                                                 menu_mp=menu_mp,
                                                 end_use='heating')

# Call the function and calculate installation premium based on existing housing characteristics
# calculate_heating_installation_premium(df, menu_mp, cpi_ratio_2023_2013)
print("Calculating Space Heating Specific Premiums (Ex: Removing Hydronic Boiler) ...")
df_euss_am_mp9_home = calculate_heating_installation_premium(df=df_euss_am_mp9_home,
                                                             menu_mp=menu_mp,
                                                             cpi_ratio_2023_2013=cpi_ratio_2023_2013)

# Display the df
print(df_euss_am_mp9_home)

# %%
if print_debug:
    # Create a sample dataframe for the heating category
    df_sample_heating = create_sample_df(
        df=df_euss_am_mp9_home,
        include_groups=['base_equipment'],
        categories=['heating'],
        scenarios=['preIRA', 'iraRef'],
        metrics=[],
        mp_number=menu_mp,
        regex_patterns=['valid_fuel_heating', 'include_heating', 'baseline_AFUE', 'baseline_SEER', 'baseline_HSPF', 
                        'hvac_heating_efficiency', 'upgrade_hvac_', 'upgrade_heating','ugrade_newInstall_HSPF']
    )
    print(df_sample_heating)


# %% [markdown]
# ### Water Heating

# %%
print("""
========================== Capital Costs: Water Heating ==========================

Obtaining Capital Cost Data from Retrofit Cost Spreadsheet ...
""")

cost_columns = [
    'unitCost_progressive', 'unitCost_reference', 'unitCost_conservative',
    'cost_per_gallon_progressive', 'cost_per_gallon_reference', 'cost_per_gallon_conservative',
]

# Update each cost column by multiplying with cpi_ratio and cost_multiplier
for column in cost_columns:
    df_waterHeating_retrofit_costs[column] = round((df_waterHeating_retrofit_costs[column] * df_waterHeating_retrofit_costs['cpi_ratio'] * df_waterHeating_retrofit_costs['cost_multiplier']), 2)

# Creating a dictionary from the DataFrame
dict_waterHeating_equipment_cost = df_waterHeating_retrofit_costs.set_index(['technology', 'efficiency']).to_dict(orient='index')
# dict_waterHeating_equipment_cost

# calculate_installation_cost(df, cost_dict, menu_mp, end_use)
print("Calculating Cost of Retrofit Upgrade: Electric Heat Pump Water Heater ...")
df_euss_am_mp9_home = calculate_installation_cost(df=df_euss_am_mp9_home,
                                                  cost_dict=dict_waterHeating_equipment_cost,
                                                  menu_mp=menu_mp,
                                                  end_use='waterHeating')

# calculate_replacement_cost(df, cost_dict, menu_mp, end_use)
print("Calculating Cost of Replacing Existing Equipment with Similar Model/Efficiency ...")
df_euss_am_mp9_home = calculate_replacement_cost(df=df_euss_am_mp9_home,
                                                 cost_dict=dict_waterHeating_equipment_cost,
                                                 menu_mp=menu_mp,
                                                 end_use='waterHeating')

# Display the df
print(df_euss_am_mp9_home)

# %% [markdown]
# ### Clothes Drying

# %%
print("""
========================== Capital Costs: Clothes Drying ==========================

Obtaining Capital Cost Data from Retrofit Cost Spreadsheet ... 
""")

# Columns to update
cost_columns = [
    'unitCost_progressive', 'unitCost_reference', 'unitCost_conservative',
]
 
# Update each cost column by multiplying with cpi_ratio and cost_multiplier
for column in cost_columns:
    df_clothesDrying_retrofit_costs[column] = round((df_clothesDrying_retrofit_costs[column] * df_clothesDrying_retrofit_costs['cpi_ratio'] * df_clothesDrying_retrofit_costs['cost_multiplier']), 2)

# Creating a dictionary from the DataFrame
dict_clothesDrying_equipment_cost = df_clothesDrying_retrofit_costs.set_index(['technology', 'efficiency']).to_dict(orient='index')
# dict_clothesDrying_equipment_cost

# calculate_installation_cost(df, cost_dict, menu_mp, end_use)
print("Calculating Cost of Retrofit Upgrade: Ventless Heat Pump Clothes Dryer ...")
df_euss_am_mp9_home = calculate_installation_cost(df=df_euss_am_mp9_home,
                                                  cost_dict=dict_clothesDrying_equipment_cost,
                                                  menu_mp=menu_mp,
                                                  end_use='clothesDrying')


# calculate_replacement_cost(df, cost_dict, menu_mp, end_use)
print("Calculating Cost of Replacing Existing Equipment with Similar Model/Efficiency ...")
df_euss_am_mp9_home = calculate_replacement_cost(df=df_euss_am_mp9_home,
                                                 cost_dict=dict_clothesDrying_equipment_cost,
                                                 menu_mp=menu_mp,
                                                 end_use='clothesDrying')

# Display the df
print(df_euss_am_mp9_home)

# %% [markdown]
# ### Cooking

# %%
print("""
========================== Capital Costs: Cooking ==========================

Obtaining Capital Cost Data from Retrofit Cost Spreadsheet ...      
""")

# Columns to update
cost_columns = [
    'unitCost_progressive', 'unitCost_reference', 'unitCost_conservative',
]
 
# Update each cost column by multiplying with cpi_ratio and cost_multiplier
for column in cost_columns:
    df_cooking_retrofit_costs[column] = round((df_cooking_retrofit_costs[column] * df_cooking_retrofit_costs['cpi_ratio'] * df_cooking_retrofit_costs['cost_multiplier']), 2)

# Creating a dictionary from the DataFrame
dict_cooking_equipment_cost = df_cooking_retrofit_costs.set_index(['technology', 'efficiency']).to_dict(orient='index')
# dict_cooking_equipment_cost

# calculate_installation_cost(df, cost_dict, menu_mp, end_use)
print("Calculating Cost of Retrofit Upgrade: Electric Resistance Range ...")
df_euss_am_mp9_home = calculate_installation_cost(df=df_euss_am_mp9_home,
                                                  cost_dict=dict_cooking_equipment_cost,
                                                  menu_mp=menu_mp,
                                                  end_use='cooking')

# calculate_replacement_cost(df, cost_dict, menu_mp, end_use)
print("Calculating Cost of Replacing Existing Equipment with Similar Model/Efficiency ...")
df_euss_am_mp9_home = calculate_replacement_cost(df=df_euss_am_mp9_home,
                                                 cost_dict=dict_cooking_equipment_cost,
                                                 menu_mp=menu_mp,
                                                 end_use='cooking')

# Display the df
print(df_euss_am_mp9_home)

# %% [markdown]
#  ## Calculate Rebate Amounts (Applicable to IRA-Reference)

# %%
from cmu_tare_model.private_impact.data_processing.determine_rebate_eligibility_and_amount import *

print(f"""
====================================================================================================================================================================
CALCULATE HOUSEHOLD PERCENT AREA MEDIAN INCOME (%AMI) AND REBATE ELIGIBILITY/AMOUNTS
====================================================================================================================================================================
determine_rebate_eligibility_and_amount.py file contains the function definitions for calculating rebate amounts and determining household %AMI.
process_income_data_for_rebates.py file contains additional information on data sources and procedures used to process data for determine_rebate_eligibility_and_amount.py file.

Documentation for calculate_percent_AMI function:
{calculate_percent_AMI.__doc__}

Documentation for calculate_rebateIRA function:
{calculate_rebateIRA.__doc__}
-------------------------------------------------------

""")

# Determine Percent AMI and Rebate Amounts
# This needs to be done before running the calculate_percent_AMI function
df_euss_am_mp9_home = df_euss_am_mp9_home.copy()

# calculate_percent_AMI(df_results_IRA, df_county_medianIncome):
df_euss_am_mp9_home = calculate_percent_AMI(df_results_IRA=df_euss_am_mp9_home)

print("Calculating rebate amounts for Space Heating ...")
df_euss_am_mp9_home = calculate_rebateIRA(df_results_IRA=df_euss_am_mp9_home,
                                          category="heating",
                                          menu_mp=menu_mp)

print("Calculating rebate amounts for Water Heating ...")
df_euss_am_mp9_home = calculate_rebateIRA(df_results_IRA=df_euss_am_mp9_home,
                                          category="waterHeating",
                                          menu_mp=menu_mp)

print("Calculating rebate amounts for Clothes Drying ...")
df_euss_am_mp9_home = calculate_rebateIRA(df_results_IRA=df_euss_am_mp9_home,
                                          category="clothesDrying",
                                          menu_mp=menu_mp)

print("Calculating rebate amounts for Cooking ...")
df_euss_am_mp9_home = calculate_rebateIRA(df_results_IRA=df_euss_am_mp9_home,
                                          category="cooking",
                                          menu_mp=menu_mp)

print(f"""
====================================================================================================================================================================
DATAFRAME: df_euss_am_mp9_home AFTER CALCULATING REBATE AMOUNTS
{df_euss_am_mp9_home}

====================================================================================================================================================================
""")

# %% [markdown]
# # SCENARIO ANALYSIS: 
# ## "No Inflation Reduction Act" and "AEO2023 Reference Case"
# ## Public Impact, Private Impact and Adoption Potential (Degree of Adoption Feasibility)

# %%
from cmu_tare_model.private_impact.calculate_lifetime_private_impact import *
from cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity import *
from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import *

print(f"""  
====================================================================================================================================================================
SCENARIO ANALYSIS: OVERVIEW
====================================================================================================================================================================

-------------------------------------------------------
PUBLIC IMPACTS: CLIMATE AND HEALTH
-------------------------------------------------------
calculate_lifetime_public_impact.py file contains the definition for the calculate_public_npv function.
Additional information on emissions/damage factor lookups as well as marginal damages calculation methods can be found in the public_impact folder. 
      
Documentation for the calculate_public_NPV function:
{calculate_public_npv.__doc__} 
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

--------------------------------------------------------
PRIVATE IMPACTS: LIFETIME CAPITAL COSTS AND SAVINGS
--------------------------------------------------------
calculate_lifetime_private_impact.py file contains the definition for the calculate_private_npv function.
Additional information on fuel price lookups as well as capital costs calculation methods can be found in the private_impact folder.

Documentation for the calculate_private_npv function:
{calculate_private_npv.__doc__}
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

--------------------------------------------------------
ADOPTION POTENTIAL
--------------------------------------------------------
determine_adoption_potential.py file contains the definition for the adoption_decision function.
      
Documentation for the adoption_decision function:
{adoption_decision.__doc__}
--------------------------------------------------------------------------------------------------------------------------------------------------------------------

""")

# %% [markdown]
# ## Moderate MEASURE PACKAGE (MP9): NO INFLATION REDUCTION ACT
# ## Assumptions
# - NREL End-Use Savings Shapes Database: Measure Package 9
# - AEO2023 No Inflation Reduction Act:  HDD and Fuel Price Projections
# - Cambium 2021 MidCase Scenario

# %%
# Measure Package 9
scenario_name = 'No Inflation Reduction Act'
cost_scenario = 'Fuel Costs: AEO2023 No Inflation Reduction Act'
grid_scenario = 'Electricity Grid: Cambium 2021 MidCase Scenario'

print(f"""
====================================================================================================================================================================
Scenario {scenario_name}:
Moderate Retrofit: Measure Package {menu_mp}
{cost_scenario}
{grid_scenario}
====================================================================================================================================================================
""")

# %%
print(f"""  
====================================================================================================================================================================
SCENARIO ANALYSIS (NO INFLATION REDUCTION ACT): PUBLIC IMPACT 
====================================================================================================================================================================
Completed Steps:
1. Calculate the baseline marginal damages for climate and health-related emissions
2. Calculate the post-retrofit marginal damages for climate and health-related emissions

---------------------------------------------------------------------------------------------
Step 3: Discount climate and health impacts and calculate lifetime public impacts (public NPV)
---------------------------------------------------------------------------------------------

RESULTS OUTPUT:
""")

# Create copies to prevent overwriting the original dataframe and compare the differences
df_euss_am_mp9_home_ap2 = df_euss_am_mp9_home.copy()
df_euss_am_mp9_home_easiur = df_euss_am_mp9_home.copy()
df_euss_am_mp9_home_inmap = df_euss_am_mp9_home.copy()

# ========================== AP2  ========================== 
df_euss_am_mp9_home_ap2 = calculate_public_npv(
    df=df_euss_am_mp9_home_ap2,
    df_baseline_climate=df_baseline_damages_climate,
    df_baseline_health=df_baseline_damages_health,
    df_mp_climate=df_mp9_noIRA_damages_climate,
    df_mp_health=df_mp9_noIRA_damages_health,
    menu_mp="9",
    policy_scenario='No Inflation Reduction Act',
    rcm_model='AP2',
    base_year=2024,
    discounting_method='public'
)

# ====================== EASIUR ========================== 
df_euss_am_mp9_home_easiur = calculate_public_npv(
    df=df_euss_am_mp9_home_easiur,
    df_baseline_climate=df_baseline_damages_climate,
    df_baseline_health=df_baseline_damages_health,
    df_mp_climate=df_mp9_noIRA_damages_climate,
    df_mp_health=df_mp9_noIRA_damages_health,
    menu_mp="9",
    policy_scenario='No Inflation Reduction Act',
    rcm_model='EASIUR',
    base_year=2024,
    discounting_method='public'
)

# ========================== InMAP ========================== 
df_euss_am_mp9_home_inmap = calculate_public_npv(
    df=df_euss_am_mp9_home_inmap,
    df_baseline_climate=df_baseline_damages_climate,
    df_baseline_health=df_baseline_damages_health,
    df_mp_climate=df_mp9_noIRA_damages_climate,
    df_mp_health=df_mp9_noIRA_damages_health,
    menu_mp="9",
    policy_scenario='No Inflation Reduction Act',
    rcm_model='InMAP',
    base_year=2024,
    discounting_method='public'
)

print(f"""  
====================================================================================================================================================================
DATAFRAME FOR mp9 AFTER CALCULATING PUBLIC NPV (NO IRA): df_euss_am_mp9_home

-----------------------------------------------
AP2: 
-----------------------------------------------

{df_euss_am_mp9_home_ap2}

-----------------------------------------------
EASIUR:
-----------------------------------------------

{df_euss_am_mp9_home_easiur}

-----------------------------------------------
InMAP:
-----------------------------------------------

{df_euss_am_mp9_home_inmap}
      
""")

# %%
print(f"""
====================================================================================================================================================================
SCENARIO ANALYSIS (NO INFLATION REDUCTION ACT): PRIVATE IMPACT
====================================================================================================================================================================
Completed Steps:
1. Calculate annual operating (fuel) costs
2. Calculate equipment capital costs (For space heating, include ductwork and weatherization (MP9 and MP10))
3. Calculate replacement cost (replacing existing piece of eqipment with similar technology)

----------------------------------------------------------------------------------------------------------------------
Step 4: Calculate net equipment capital costs and private NPV (less WTP and more WTP)
----------------------------------------------------------------------------------------------------------------------

RESULTS OUTPUT (NO INFLATION REDUCTION ACT):
""")


# ========================== AP2  ========================== 
df_euss_am_mp9_home_ap2 = calculate_private_npv(
    df=df_euss_am_mp9_home_ap2,
    df_fuel_costs=df_mp9_noIRA_fuel_costs,
    df_baseline_costs=df_baseline_fuel_costs,
    menu_mp=menu_mp,
    input_mp=input_mp,
    policy_scenario='No Inflation Reduction Act',
    discounting_method='private_fixed',
    base_year=2024,
    verbose=True  # Add this parameter
    )

# ====================== EASIUR ========================== 
df_euss_am_mp9_home_easiur = calculate_private_npv(
    df=df_euss_am_mp9_home_easiur,
    df_fuel_costs=df_mp9_noIRA_fuel_costs,
    df_baseline_costs=df_baseline_fuel_costs,
    menu_mp=menu_mp,
    input_mp=input_mp,
    policy_scenario='No Inflation Reduction Act',
    discounting_method='private_fixed',
    base_year=2024,
    verbose=True  # Add this parameter
    )

# ====================== InMAP ========================== 
df_euss_am_mp9_home_inmap = calculate_private_npv(
    df=df_euss_am_mp9_home_inmap,
    df_fuel_costs=df_mp9_noIRA_fuel_costs,
    df_baseline_costs=df_baseline_fuel_costs,
    menu_mp=menu_mp,
    input_mp=input_mp,
    policy_scenario='No Inflation Reduction Act',
    discounting_method='private_fixed',
    base_year=2024,
    verbose=True  # Add this parameter
    )


print(f"""  
====================================================================================================================================================================
DATAFRAME FOR mp9 AFTER CALCULATING PRIVATE NPV (NO INFLATION REDUCTION ACT): df_euss_am_mp9_home

-----------------------------------------------
AP2: 
-----------------------------------------------

{df_euss_am_mp9_home_ap2}

-----------------------------------------------
EASIUR:
-----------------------------------------------

{df_euss_am_mp9_home_easiur}

-----------------------------------------------
InMAP:
-----------------------------------------------

{df_euss_am_mp9_home_inmap}
      
""")

# %%
print(f"""
====================================================================================================================================================================
SCENARIO ANALYSIS (NO INFLATION REDUCTION ACT): ADOPTION POTENTIAL
====================================================================================================================================================================
determine_adoption_potential.py file contains the definition for the adoption_decision function.
      
RESULTS OUTPUT (NO INFLATION REDUCTION ACT):
""")

# ========================== AP2  ========================== 
df_euss_am_mp9_home_ap2 = adoption_decision(
    df=df_euss_am_mp9_home_ap2,
    menu_mp=menu_mp,
    policy_scenario='No Inflation Reduction Act',
    rcm_model='AP2',
    cr_function='acs',
    climate_sensitivity=False
    )

df_euss_am_mp9_home_ap2 = adoption_decision(
    df=df_euss_am_mp9_home_ap2,
    menu_mp=menu_mp,
    policy_scenario='No Inflation Reduction Act',
    rcm_model='AP2',
    cr_function='h6c',
    climate_sensitivity=False
    )

# ====================== EASIUR ========================== 
df_euss_am_mp9_home_easiur = adoption_decision(
    df=df_euss_am_mp9_home_easiur,
    menu_mp=menu_mp,
    policy_scenario='No Inflation Reduction Act',
    rcm_model='EASIUR',
    cr_function='acs',
    climate_sensitivity=False
    )

df_euss_am_mp9_home_easiur = adoption_decision(
    df=df_euss_am_mp9_home_easiur,
    menu_mp=menu_mp,
    policy_scenario='No Inflation Reduction Act',
    rcm_model='EASIUR',
    cr_function='h6c',
    climate_sensitivity=False
    )

# ====================== InMAP ========================== 
df_euss_am_mp9_home_inmap = adoption_decision(
    df=df_euss_am_mp9_home_inmap,
    menu_mp=menu_mp,
    policy_scenario='No Inflation Reduction Act',
    rcm_model='InMAP',
    cr_function='acs',
    climate_sensitivity=False
    )

df_euss_am_mp9_home_inmap = adoption_decision(
    df=df_euss_am_mp9_home_inmap,
    menu_mp=menu_mp,
    policy_scenario='No Inflation Reduction Act',
    rcm_model='InMAP',
    cr_function='h6c',
    climate_sensitivity=False
    )


print(f"""
====================================================================================================================================================================
DATAFRAME FOR mp9 AFTER DETERMINING ADOPTION FEASIBILITY (NO INFLATION REDUCTION ACT): df_euss_am_mp9_home

-----------------------------------------------
AP2: 
-----------------------------------------------

{df_euss_am_mp9_home_ap2}

-----------------------------------------------
EASIUR:
-----------------------------------------------

{df_euss_am_mp9_home_easiur}

-----------------------------------------------
InMAP:
-----------------------------------------------

{df_euss_am_mp9_home_inmap}
      
""")

# %% [markdown]
# ## Moderate MEASURE PACKAGE (MP9): AEO2023 REFERENCE CASE
# ## Assumptions
# - NREL End-Use Savings Shapes Database: Measure Package 9
# - AEO2023 REFERENCE CASE: HDD and Fuel Price Projections
# - Cambium 2023 MidCase Scenario

# %%
# Measure Package 9
scenario_name = 'Moderate IRA-Reference'
cost_scenario = 'Fuel Costs: AEO2023 Reference Case'
grid_scenario = 'Electricity Grid: Cambium 2023 MidCase Scenario'

print(f"""
====================================================================================================================================================================
MODEL SCENARIO
====================================================================================================================================================================
Scenario {scenario_name}:
Moderate Retrofit: Measure Package {menu_mp}
{cost_scenario}
{grid_scenario}
====================================================================================================================================================================
""")

# %%
print(f"""  
====================================================================================================================================================================
SCENARIO ANALYSIS (AEO2023 REFERENCE CASE): PUBLIC IMPACT 
====================================================================================================================================================================
Completed Steps:
1. Calculate the baseline marginal damages for climate and health-related emissions
2. Calculate the post-retrofit marginal damages for climate and health-related emissions

---------------------------------------------------------------------------------------------
Step 3: Discount climate and health impacts and calculate lifetime public impacts (public NPV)
---------------------------------------------------------------------------------------------

RESULTS OUTPUT (AEO2023 REFERENCE CASE):
""")

# ========================== AP2  ========================== 
df_euss_am_mp9_home_ap2 = calculate_public_npv(
    df=df_euss_am_mp9_home_ap2,
    df_baseline_climate=df_baseline_damages_climate,
    df_baseline_health=df_baseline_damages_health,
    df_mp_climate=df_mp9_IRA_damages_climate,
    df_mp_health=df_mp9_IRA_damages_health,
    menu_mp="9",
    policy_scenario='AEO2023 Reference Case',
    rcm_model='AP2',
    base_year=2024,
    discounting_method='public'
)

# ====================== EASIUR ========================== 
df_euss_am_mp9_home_easiur = calculate_public_npv(
    df=df_euss_am_mp9_home_easiur,
    df_baseline_climate=df_baseline_damages_climate,
    df_baseline_health=df_baseline_damages_health,
    df_mp_climate=df_mp9_IRA_damages_climate,
    df_mp_health=df_mp9_IRA_damages_health,
    menu_mp="9",
    policy_scenario='AEO2023 Reference Case',
    rcm_model='EASIUR',
    base_year=2024,
    discounting_method='public'
)

# ========================== InMAP ========================== 
df_euss_am_mp9_home_inmap = calculate_public_npv(
    df=df_euss_am_mp9_home_inmap,
    df_baseline_climate=df_baseline_damages_climate,
    df_baseline_health=df_baseline_damages_health,
    df_mp_climate=df_mp9_IRA_damages_climate,
    df_mp_health=df_mp9_IRA_damages_health,
    menu_mp="9",
    policy_scenario='AEO2023 Reference Case',
    rcm_model='InMAP',
    base_year=2024,
    discounting_method='public'
)

print(f"""  
=====================================================================================================================================================================
DATAFRAME FOR mp9 AFTER CALCULATING PUBLIC NPV (AEO2023 REFERENCE CASE): df_euss_am_mp9_home

-----------------------------------------------
AP2: 
-----------------------------------------------
      
{df_euss_am_mp9_home_ap2}

-----------------------------------------------
EASIUR:
-----------------------------------------------

{df_euss_am_mp9_home_easiur}

-----------------------------------------------
InMAP:
-----------------------------------------------

{df_euss_am_mp9_home_inmap}
      
""")

# %%
print(f"""
====================================================================================================================================================================
SCENARIO ANALYSIS (AEO2023 REFERENCE CASE): PRIVATE IMPACT
====================================================================================================================================================================
Completed Steps:
1. Calculate annual operating (fuel) costs
2. Calculate equipment capital costs (For space heating, include ductwork and weatherization (MP9 and MP10))
3. Calculate replacement cost (replacing existing piece of eqipment with similar technology)

----------------------------------------------------------------------------------------------------------------------
Step 4: Calculate net equipment capital costs and private NPV (less WTP and more WTP)
----------------------------------------------------------------------------------------------------------------------

RESULTS OUTPUT (AEO2023 REFERENCE CASE):
""")


# ========================== AP2  ========================== 
df_euss_am_mp9_home_ap2 = calculate_private_npv(
    df=df_euss_am_mp9_home_ap2,
    df_fuel_costs=df_mp9_IRA_fuel_costs,
    df_baseline_costs=df_baseline_fuel_costs,
    menu_mp=menu_mp,
    input_mp=input_mp,
    policy_scenario='AEO2023 Reference Case',
    discounting_method='private_fixed',
    base_year=2024,
    verbose=True  # Add this parameter
    )

# ====================== EASIUR ========================== 
df_euss_am_mp9_home_easiur = calculate_private_npv(
    df=df_euss_am_mp9_home_easiur,
    df_fuel_costs=df_mp9_IRA_fuel_costs,
    df_baseline_costs=df_baseline_fuel_costs,
    menu_mp=menu_mp,
    input_mp=input_mp,
    policy_scenario='AEO2023 Reference Case',
    discounting_method='private_fixed',
    base_year=2024,
    verbose=True  # Add this parameter
    )

# ====================== InMAP ========================== 
df_euss_am_mp9_home_inmap = calculate_private_npv(
    df=df_euss_am_mp9_home_inmap,
    df_fuel_costs=df_mp9_IRA_fuel_costs,
    df_baseline_costs=df_baseline_fuel_costs,
    menu_mp=menu_mp,
    input_mp=input_mp,
    policy_scenario='AEO2023 Reference Case',
    discounting_method='private_fixed',
    base_year=2024,
    verbose=True  # Add this parameter
    )



print(f"""  
=====================================================================================================================================================================
DATAFRAME FOR mp9 AFTER CALCULATING PRIVATE NPV (AEO2023 REFERENCE CASE): df_euss_am_mp9_home

-----------------------------------------------
AP2: 
-----------------------------------------------
      
{df_euss_am_mp9_home_ap2}

-----------------------------------------------
EASIUR:
-----------------------------------------------

{df_euss_am_mp9_home_easiur}

-----------------------------------------------
InMAP:
-----------------------------------------------

{df_euss_am_mp9_home_inmap}
      
""")

# %%
print(f"""
====================================================================================================================================================================
SCENARIO ANALYSIS (AEO2023 REFERENCE CASE): ADOPTION POTENTIAL
====================================================================================================================================================================
determine_adoption_potential.py file contains the definition for the adoption_decision function.
      
RESULTS OUTPUT (AEO2023 REFERENCE CASE):
""")

# ========================== AP2  ========================== 
df_euss_am_mp9_home_ap2 = adoption_decision(
    df=df_euss_am_mp9_home_ap2,
    menu_mp=menu_mp,
    policy_scenario='AEO2023 Reference Case',
    rcm_model='AP2',
    cr_function='acs',
    climate_sensitivity=False
    )

df_euss_am_mp9_home_ap2 = adoption_decision(
    df=df_euss_am_mp9_home_ap2,
    menu_mp=menu_mp,
    policy_scenario='AEO2023 Reference Case',
    rcm_model='AP2',
    cr_function='h6c',
    climate_sensitivity=False
    )

# ====================== EASIUR ========================== 
df_euss_am_mp9_home_easiur = adoption_decision(
    df=df_euss_am_mp9_home_easiur,
    menu_mp=menu_mp,
    policy_scenario='AEO2023 Reference Case',
    rcm_model='EASIUR',
    cr_function='acs',
    climate_sensitivity=False
    )

df_euss_am_mp9_home_easiur = adoption_decision(
    df=df_euss_am_mp9_home_easiur,
    menu_mp=menu_mp,
    policy_scenario='AEO2023 Reference Case',
    rcm_model='EASIUR',
    cr_function='h6c',
    climate_sensitivity=False
    )

# ====================== InMAP ========================== 
df_euss_am_mp9_home_inmap = adoption_decision(
    df=df_euss_am_mp9_home_inmap,
    menu_mp=menu_mp,
    policy_scenario='AEO2023 Reference Case',
    rcm_model='InMAP',
    cr_function='acs',
    climate_sensitivity=False
    )

df_euss_am_mp9_home_inmap = adoption_decision(
    df=df_euss_am_mp9_home_inmap,
    menu_mp=menu_mp,
    policy_scenario='AEO2023 Reference Case',
    rcm_model='InMAP',
    cr_function='h6c',
    climate_sensitivity=False
    )


print(f"""
====================================================================================================================================================================
DATAFRAME FOR mp9 AFTER DETERMINING ADOPTION FEASIBILITY (AEO2023 REFERENCE CASE): df_euss_am_mp9_home

-----------------------------------------------   
AP2: 
-----------------------------------------------
      
{df_euss_am_mp9_home_ap2}

-----------------------------------------------
EASIUR:
-----------------------------------------------

{df_euss_am_mp9_home_easiur}

-----------------------------------------------
InMAP:
-----------------------------------------------

{df_euss_am_mp9_home_inmap}
      
""")

# %%
df_euss_am_mp9_home_ap2

# %%
df_euss_am_mp9_home_easiur

# %%
df_euss_am_mp9_home_inmap

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

# %%



