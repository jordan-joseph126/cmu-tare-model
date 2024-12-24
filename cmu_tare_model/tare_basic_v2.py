# %%
# pd.set_option("display.max_columns", None)
# pd.reset_option('display.max_columns')
# pd.set_option('display.max_rows', None)
# pd.reset_option('display.max_rows')

# %% [markdown]
# # Baseline Scenario
# - Uncomment Code Below For Individual scenario runs (rather than full model MP8-Basic, MP9-Moderate, MP10-Advanced)

# # %%
# from config import PROJECT_ROOT
# # from cmu_tare_model.tare_baseline_v2 import *
# import os

# # Measure Package 0: Baseline
# menu_mp = 0
# input_mp = 'baseline'

# print(f"PROJECT_ROOT (from config.py): {PROJECT_ROOT}")

# # Construct the absolute path to the .py file
# relative_path = os.path.join("cmu_tare_model", "tare_baseline_v2.py")
# file_path = os.path.join(PROJECT_ROOT, relative_path)

# # On Windows, to avoid any path-escape quirks, convert backslashes to forward slashes
# file_path = file_path.replace("\\", "/")

# print(f"Running file: {file_path}")

# # # %run magic command to run a .py file and import variables into the current IPython session
# # # If your path has spaces, wrap it in quotes:
# # %run -i {file_path} # If your path has NO spaces, no quotes needed.

# # iPthon magic command to run a .py file and import variables into the current IPython session
# # Now run it, importing variables into your current IPython session
# from IPython import get_ipython
# get_ipython().run_line_magic('run', f'-i {file_path}')  # If your path has NO spaces, no quotes needed.

# # # exec() function to run a .py file and import variables into the current IPython session
# # with open(file_path) as f:
# #     code = compile(f.read(), file_path, 'exec')
# #     exec(code)


# print("Baseline Scenario - Model Run Complete")

# %% [markdown]
# ## Dataframe for Electric Resistance Cooking (MP7)

# %%
print(f"""
====================================================================================================================================================================
We assume the use of Electric Resistance (MP7) rather than Induction (MP8).
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

# %%
# # from cmu_tare_model.tare_baseline_v2 import menu_state, menu_city, input_state, input_cityFilter
# from cmu_tare_model.functions.mp7_electric_resistance_range import df_euss_am_mp7

# print(f"""
# ====================================================================================================================================================================
# We assume the use of Electric Resistance (MP7) rather than Induction (MP8).
# Electric Resistance is significantly cheaper and only slightly less efficient than Induction.
# ====================================================================================================================================================================
# Created a dataframe for Measure Package 7 (MP7) upgrades using the mp7_electric_resistance_range.py file. 
      
# DATAFRAME: df_euss_am_mp7

# {df_euss_am_mp7}
# """)

# %% [markdown]
# ## Dataframe used for other end-uses (MP8)

# %%
# Measure Package 8
menu_mp = 8
input_mp = 'upgrade08'
scenario_name = 'Basic-BAU'
cost_scenario = 'BAU Costs'
grid_scenario = 'Current Electricity Grid'

print(f"""
====================================================================================================================================================================
MODEL SCENARIO
====================================================================================================================================================================
Scenario {scenario_name}:
Basic Retrofit: Measure Package {menu_mp}
{cost_scenario}
{grid_scenario}
====================================================================================================================================================================
""")

filename = "upgrade08_metadata_and_annual_results.csv"
relative_path = os.path.join("cmu_tare_model", "data", "euss_data", "resstock_amy2018_release_1.1", "state", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")
print("\n")

# Fix DtypeWarning error in columns 'in.neighbors' and 'in.geometry_stories_low_rise'
columns_to_string = {11: str, 61: str, 121: str, 103: str, 128: str, 129: str}
df_euss_am_mp8 = pd.read_csv(file_path, dtype=columns_to_string, index_col="bldg_id") # UPDATE: Set index to 'bldg_id' (unique identifier)
print(f"DATAFRAME SIZE before applying any filters: {df_euss_am_mp8.shape}")

# Filter for occupied homes
occupancy_filter = df_euss_am_mp8['in.vacancy_status'] == 'Occupied'
df_euss_am_mp8 = df_euss_am_mp8.loc[occupancy_filter]
print(f"DATAFRAME SIZE after filtering for 'Occupied' homes: {df_euss_am_mp8.shape}")

# Filter for single family home building type
house_type_list = ['Single-Family Attached', 'Single-Family Detached']
house_type_filter = df_euss_am_mp8['in.geometry_building_type_recs'].isin(house_type_list)
df_euss_am_mp8 = df_euss_am_mp8.loc[house_type_filter]
print(f"DATAFRAME SIZE after filtering for 'Single-Family Attached' and 'Single-Family Detached' homes: {df_euss_am_mp8.shape}")

# National Level 
if menu_state == 'N':
    print("You chose to analyze all of the United States.")
    input_state = 'National'

# Filter down to state or city
else:
    print(f"You chose to filter for: {input_state}")
    state_filter = df_euss_am_mp8['in.state'].eq(input_state)
    df_euss_am_mp8 = df_euss_am_mp8.loc[state_filter]

    # Filter for the entire selected state
    if menu_city == 'N':
        print(f"You chose to analyze all of state: {input_state}")
        
    # Filter to a city within the selected state
    else:
        print(f"You chose to filter for: {input_state}, {input_cityFilter}")
        city_filter = df_euss_am_mp8['in.city'].eq(f"{input_state}, {input_cityFilter}")
        df_euss_am_mp8 = df_euss_am_mp8.loc[city_filter]

# Display the filtered dataframe
# Display the filtered dataframe
print(f"DATAFRAME SIZE after applying geographic filter: {df_euss_am_mp8.shape}")
print(df_euss_am_mp8)

# %% [markdown]
# # Project Future Energy Consumption

# %%
# from cmu_tare_model.tare_baseline_v2 import df_euss_am_baseline_home
print(f"""
====================================================================================================================================================================
Resulting Baseline Home Dataframe from running tare_baseline_v2.py file
====================================================================================================================================================================
You'll notice that the number of rows differs from df_euss_am_mp7 and df_euss_am_mp8.
      - df_euss_am_baseline_home has fewer rows (representative dwelling units) because a tech filter was applied. 
      - df_euss_am_mp8_home will have the same number of rows as df_euss_am_baseline_home after df_enduse_compare function is run.
      - df_enduse_compare function performs an inner merge on the two dataframes, keeping only the rows that are present in both dataframes.
====================================================================================================================================================================      

DATAFRAME: df_euss_am_baseline_home

{df_euss_am_baseline_home}

====================================================================================================================================================================
""")

# %%
print("""
====================================================================================================================================================================
Post-Retrofit (MP) Consumption: WHOLE-HOME
====================================================================================================================================================================
df_euss_am_mp8_home will be created by running the df_enduse_compare function (contains post-retrofit consumption data for the entire home in 2024).
load_and_filter_euss_data.py file contains the function definition.
""")

# df_enduse_compare(df_mp, category, df_baseline):
df_euss_am_mp8_home = df_enduse_compare(df_mp = df_euss_am_mp8,
                                        input_mp=input_mp,
                                        menu_mp=menu_mp,
                                        df_baseline = df_euss_am_baseline_home,
                                        df_cooking_range=df_euss_am_mp7,
                                        )
# df_euss_am_mp8_home
# print(df_euss_am_mp8_home)

# %%
# from cmu_tare_model.functions.project_future_energy_consumption import *

df_mp8_scenario_consumption = df_euss_am_mp8_home.copy()

# Project Future Energy Consumption
df_euss_am_mp8_home, df_mp8_scenario_consumption = project_future_consumption(df=df_euss_am_mp8_home,
                                                                              lookup_hdd_factor=lookup_hdd_factor,
                                                                              menu_mp=menu_mp
                                                                              )

print(f"""
====================================================================================================================================================================
PROJECT FUTURE ENERGY CONSUMPTION
====================================================================================================================================================================
project_future_energy_consumption.py file was used to calculate/project the annual energy consumption for each home in the dataframe.
      
DATAFRAME (Summary of MP8): df_euss_am_mp8_home

{df_euss_am_mp8_home}

DATAFRAME (Consumption Cols): df_mp8_scenario_consumption
      
{df_mp8_scenario_consumption}
""")

# %% [markdown]
# # Model Future Climate Damages and Annual Fuel Costs
# ## Scenarios: No IRA and IRA-Reference

# %%
print("""
Model Future Climate Damages and Annual Fuel Costs for Scenarios No IRA and IRA-Reference
""")

# Make copies from scenario consumption to keep df smaller
print("\n", "Creating dataframe to store marginal damages calculations ...")
df_mp8_scenario_damages = df_mp8_scenario_consumption.copy()

print("\n", "Creating dataframe to store annual fuel cost calculations ...")
df_mp8_scenario_fuelCosts = df_mp8_scenario_consumption.copy()

# %% [markdown]
# ## Future Climate Damages: No IRA and IRA-Reference

# %%
print("""
====================================================================================================================================================================
Public Perspective: Monetized Marginal Damages from Emissions
====================================================================================================================================================================
**Steps 1-4 were performed in the Baseline Scenario**
- Step 1: Calculate emissions factors for different fuel sources
- Step 2: Adjust Natural Gas & Electricity Emissions Factors for Natural Gas Leakage
- Step 3: Quantify monitized damages using EASIUR Marginal Social Cost Factors
- Step 4: Inflate Marginal Social Cost (Damage) Factors using BLS CPI for All Urban Consumers (CPI-U) 

Step 5: Calculate End-use specific marginal damages
====================================================================================================================================================================
""")

# %%
# from cmu_tare_model.functions.calculate_emissions_damages import *
# calculate_marginal_damages(df, menu_mp, policy_scenario, df_baseline_damages, df_detailed_damages)
print("\n", "Modeling Scenario: No Inflation Reduction Act")
df_euss_am_mp8_home, df_mp8_scenario_damages = calculate_marginal_damages(df=df_euss_am_mp8_home, menu_mp=menu_mp, policy_scenario='No Inflation Reduction Act', df_baseline_damages=df_baseline_scenario_damages, df_detailed_damages=df_mp8_scenario_damages)


print("\n","Modeling Scenario: AEO2023 Reference Case")
df_euss_am_mp8_home, df_mp8_scenario_damages = calculate_marginal_damages(df=df_euss_am_mp8_home, menu_mp=menu_mp, policy_scenario='AEO2023 Reference Case', df_baseline_damages=df_baseline_scenario_damages, df_detailed_damages=df_mp8_scenario_damages)

print(f"""  
====================================================================================================================================================================
Post-Retrofit (MP8) Marginal Damages: WHOLE-HOME
Scenario: No Inflation Reduction Act and AEO2023 Reference Case
====================================================================================================================================================================
calculate_emissions_damages.py file contains the definition for the calculate_marginal_damages function.
Additional information on emissions and damage factor lookups can be found in the calculate_emissions_damages.py file as well. 
      
DATAFRAME FOR MP8 SCENARIO DAMAGES: df_mp8_scenario_damages
{df_mp8_scenario_damages}
      
SUMMARY DATAFRAME FOR MP8: df_euss_am_mp8_home
{df_euss_am_mp8_home}
====================================================================================================================================================================
""")

# %% [markdown]
# ## Future Annual Fuel Costs: No IRA and IRA-Reference

# %%
print("""  
====================================================================================================================================================================
Private Perspective: Annual Energy Costs
====================================================================================================================================================================
- Step 1: Obtain Level Energy Fuel Cost Data from the EIA
- Step 2: Calculate Annual Operating (Fuel) Costs
====================================================================================================================================================================
""")

# calculate_annual_fuelCost(df, menu_mp, policy_scenario)
print("\n", "Modeling Scenario: No Inflation Reduction Act")
df_mp8_scenario_fuelCosts = calculate_annual_fuelCost(df=df_mp8_scenario_fuelCosts, menu_mp=menu_mp, policy_scenario='No Inflation Reduction Act', drop_fuel_cost_columns=False)

print("\n","Modeling Scenario: AEO2023 Reference Case")
df_mp8_scenario_fuelCosts = calculate_annual_fuelCost(df=df_mp8_scenario_fuelCosts, menu_mp=menu_mp, policy_scenario='AEO2023 Reference Case', drop_fuel_cost_columns=False)

print(f"""  
====================================================================================================================================================================
Post-Retrofit (MP8) Fuel Costs: WHOLE-HOME
Scenario: No Inflation Reduction Act and AEO2023 Reference Case
====================================================================================================================================================================
Calculating Fuel Costs for each end-use ...

DATAFRAME FOR MP8 Fuel Costs: df_mp8_scenario_fuelCosts
{df_mp8_scenario_fuelCosts}
      ====================================================================================================================================================================
""")

# %% [markdown]
# # Calculate Capital Costs and Rebate Amounts

# %% [markdown]
# ## Calculate Capital Costs (Applicable to All Scenarios)

# %%
print("""
====================================================================================================================================================================
PRIVATE PERSPECTIVE COSTS AND BENEFITS
====================================================================================================================================================================
- Step 1: Calculate annual operating (fuel) costs
- Step 2: Calculate equipment capital costs (For space heating, include ductwork and weatherization (MP9 and MP10))
- Step 3: Calculate replacement cost (replacing existing piece of eqipment with similar technology)
- Step 4: Calculate net equipment capital costs
- Step 5: Calculate private NPV
====================================================================================================================================================================
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

# %%
from cmu_tare_model.functions.rsMeans_adjustment import *
from cmu_tare_model.functions.inflation_adjustment import *

# Use CCI to adjust for cost differences when compared to the national average
# Call the function and map the values for CCI adjustment
df_euss_am_mp8_home['rsMeans_CCI_avg'] = df_euss_am_mp8_home['city'].apply(map_average_cost)
df_euss_am_mp8_home

# %% [markdown]
# ### Space Heating and No Enclosure Upgrade

# %% [markdown]
# #### Space Heating Capital Costs

# %%
print("""
====================================================================================================================================================================
Capital Costs: Space Heating
====================================================================================================================================================================

Obtaining Capital Cost Data from Retrofit Cost Spreadsheet ...
""")

from cmu_tare_model.functions.calculate_equipment_installation_costs import *
from cmu_tare_model.functions.calculate_equipment_replacement_costs import *

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
# dict_heating_equipment_cost

# Call the function and obtain equipment specifications
# obtain_heating_system_specs(df)
print("Obtaining system specs ...")
df_euss_am_mp8_home = obtain_heating_system_specs(df=df_euss_am_mp8_home)

# calculate_installation_cost(df, cost_dict, rsMeans_national_avg, menu_mp, end_use)
print("Calculating Cost of Retrofit Upgrade: Heat Pump for Space Heating (No Enclosure Upgrade) ...")
df_euss_am_mp8_home = calculate_installation_cost(df=df_euss_am_mp8_home,
                                                  cost_dict=dict_heating_equipment_cost,
                                                  rsMeans_national_avg=rsMeans_national_avg,
                                                  menu_mp=menu_mp,
                                                  end_use='heating')

# calculate_replacement_cost(df, cost_dict, rsMeans_national_avg, menu_mp, end_use)
print("Calculating Cost of Replacing Existing Equipment with Similar Model/Efficiency ...")
df_euss_am_mp8_home = calculate_replacement_cost(df=df_euss_am_mp8_home,
                                                 cost_dict=dict_heating_equipment_cost,
                                                 rsMeans_national_avg=rsMeans_national_avg,
                                                 menu_mp=menu_mp,
                                                 end_use='heating')

# Call the function and calculate installation premium based on existing housing characteristics
# calculate_heating_installation_premium(df, menu_mp, rsMeans_national_avg, cpi_ratio_2023_2013)
print("Calculating Space Heating Specific Premiums (Ex: Removing Hydronic Boiler) ...")
df_euss_am_mp8_home = calculate_heating_installation_premium(df=df_euss_am_mp8_home,
                                                             menu_mp=menu_mp,
                                                             rsMeans_national_avg=rsMeans_national_avg,
                                                             cpi_ratio_2023_2013=cpi_ratio_2023_2013)

# Display the df
print(df_euss_am_mp8_home)

# %% [markdown]
# ### Water Heating

# %%
print("""
====================================================================================================================================================================
Capital Costs: Water Heating
====================================================================================================================================================================

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

# calculate_installation_cost(df, cost_dict, rsMeans_national_avg, menu_mp, end_use)
print("Calculating Cost of Retrofit Upgrade: Electric Heat Pump Water Heater ...")
df_euss_am_mp8_home = calculate_installation_cost(df=df_euss_am_mp8_home,
                                                  cost_dict=dict_waterHeating_equipment_cost,
                                                  rsMeans_national_avg=rsMeans_national_avg,
                                                  menu_mp=menu_mp,
                                                  end_use='waterHeating')

# calculate_replacement_cost(df, cost_dict, rsMeans_national_avg, menu_mp, end_use)
print("Calculating Cost of Replacing Existing Equipment with Similar Model/Efficiency ...")
df_euss_am_mp8_home = calculate_replacement_cost(df=df_euss_am_mp8_home,
                                                 cost_dict=dict_waterHeating_equipment_cost,
                                                 rsMeans_national_avg=rsMeans_national_avg,
                                                 menu_mp=menu_mp,
                                                 end_use='waterHeating')

# Display the df
print(df_euss_am_mp8_home)

# %% [markdown]
# ### Clothes Drying

# %%
print("""
====================================================================================================================================================================
Capital Costs: Clothes Drying
====================================================================================================================================================================

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

# calculate_installation_cost(df, cost_dict, rsMeans_national_avg, menu_mp, end_use)
print("Calculating Cost of Retrofit Upgrade: Ventless Heat Pump Clothes Dryer ...")
df_euss_am_mp8_home = calculate_installation_cost(df=df_euss_am_mp8_home,
                                                  cost_dict=dict_clothesDrying_equipment_cost,
                                                  rsMeans_national_avg=rsMeans_national_avg,
                                                  menu_mp=menu_mp,
                                                  end_use='clothesDrying')


# calculate_replacement_cost(df, cost_dict, rsMeans_national_avg, menu_mp, end_use)
print("Calculating Cost of Replacing Existing Equipment with Similar Model/Efficiency ...")
df_euss_am_mp8_home = calculate_replacement_cost(df=df_euss_am_mp8_home,
                                                 cost_dict=dict_clothesDrying_equipment_cost,
                                                 rsMeans_national_avg=rsMeans_national_avg,
                                                 menu_mp=menu_mp,
                                                 end_use='clothesDrying')

# Display the df
print(df_euss_am_mp8_home)

# %% [markdown]
# ### Cooking

# %%
print("""
====================================================================================================================================================================
Capital Costs: Cooking
====================================================================================================================================================================

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

# calculate_installation_cost(df, cost_dict, rsMeans_national_avg, menu_mp, end_use)
print("Calculating Cost of Retrofit Upgrade: Electric Resistance Range ...")
df_euss_am_mp8_home = calculate_installation_cost(df=df_euss_am_mp8_home,
                                                  cost_dict=dict_cooking_equipment_cost,
                                                  rsMeans_national_avg=rsMeans_national_avg,
                                                  menu_mp=menu_mp,
                                                  end_use='cooking')

# calculate_replacement_cost(df, cost_dict, rsMeans_national_avg, menu_mp, end_use)
print("Calculating Cost of Replacing Existing Equipment with Similar Model/Efficiency ...")
df_euss_am_mp8_home = calculate_replacement_cost(df=df_euss_am_mp8_home,
                                                 cost_dict=dict_cooking_equipment_cost,
                                                 rsMeans_national_avg=rsMeans_national_avg,
                                                 menu_mp=menu_mp,
                                                 end_use='cooking')

# Display the df
print(df_euss_am_mp8_home)

# %% [markdown]
#  ## Calculate Rebate Amounts (Applicable to IRA-Reference)

# %%
from cmu_tare_model.functions.determine_rebate_eligibility_and_amount import *

# Determine Percent AMI and Rebate Amounts
# This needs to be done before running the calculate_percent_AMI function
df_euss_am_mp8_home = df_euss_am_mp8_home.copy()

# calculate_percent_AMI(df_results_IRA, df_county_medianIncome):
df_euss_am_mp8_home = calculate_percent_AMI(df_results_IRA=df_euss_am_mp8_home)

print("Calculating rebate amounts for Space Heating ...")
df_euss_am_mp8_home = calculate_rebateIRA(df_results_IRA=df_euss_am_mp8_home,
                                          category="heating",
                                          menu_mp=menu_mp)

print("Calculating rebate amounts for Water Heating ...")
df_euss_am_mp8_home = calculate_rebateIRA(df_results_IRA=df_euss_am_mp8_home,
                                          category="waterHeating",
                                          menu_mp=menu_mp)

print("Calculating rebate amounts for Clothes Drying ...")
df_euss_am_mp8_home = calculate_rebateIRA(df_results_IRA=df_euss_am_mp8_home,
                                          category="clothesDrying",
                                          menu_mp=menu_mp)

print("Calculating rebate amounts for Cooking ...")
df_euss_am_mp8_home = calculate_rebateIRA(df_results_IRA=df_euss_am_mp8_home,
                                          category="cooking",
                                          menu_mp=menu_mp)

print(f"""
====================================================================================================================================================================
CALCULATE HOUSEHOLD PERCENT AREA MEDIAN INCOME (%AMI) AND REBATE ELIGIBILITY/AMOUNTS
====================================================================================================================================================================
determine_rebate_eligibility_and_amount.py file contains the function definitions for calculating rebate amounts and determining household %AMI.
process_income_data_for_rebates.py file contains additional information on data sources and procedures used to process data for determine_rebate_eligibility_and_amount.py file.

DATAFRAME: df_euss_am_mp8_home AFTER CALCULATING REBATE AMOUNTS
{df_euss_am_mp8_home}

====================================================================================================================================================================
""")

# %% [markdown]
# # SCENARIO ANALYSIS: Basic Pre-IRA Scenario
# ## - NREL End-Use Savings Shapes Database: Measure Package 8
# ## - AEO2023 No Inflation Reduction Act
# ## - Cambium 2021 MidCase Scenario

# %%
# Measure Package 8
scenario_name = 'No Inflation Reduction Act'
cost_scenario = 'Fuel Costs: AEO2023 No Inflation Reduction Act'
grid_scenario = 'Electricity Grid: Cambium 2021 MidCase Scenario'

print(f"""
====================================================================================================================================================================
MODEL SCENARIO
====================================================================================================================================================================
Scenario {scenario_name}:
Basic Retrofit: Measure Package {menu_mp}
{cost_scenario}
{grid_scenario}
====================================================================================================================================================================
""")

# %%
from cmu_tare_model.functions.calculate_lifetime_public_impact import *

# calculate_public_npv(df, df_baseline_damages, df_mp_damages, menu_mp, policy_scenario, equipment_specs, interest_rate=0.02)
df_euss_am_mp8_home = calculate_public_npv(df=df_euss_am_mp8_home,
                                           df_baseline_damages=df_baseline_scenario_damages,
                                           df_mp_damages=df_mp8_scenario_damages,
                                           menu_mp=menu_mp,
                                           policy_scenario='No Inflation Reduction Act',
                                           interest_rate=0.02,
                                           )

print(f"""  
====================================================================================================================================================================
PUBLIC PERSPECTIVE COSTS AND BENEFITS: NO INFLATION REDUCTION ACT
====================================================================================================================================================================
calculate_lifetime_public_impact.py file contains the definition for the calculate_public_npv function.
Additional information on emissions/damage factor lookups as well as marginal damages calculation methods can be found in the calculate_emissions_damages.py file. 
      
DATAFRAME FOR MP8 AFTER CALCULATING PUBLIC NPV: df_euss_am_mp8_home
      
{df_euss_am_mp8_home}
      
""")

# %%
from cmu_tare_model.functions.calculate_lifetime_private_impact import *

# calculate_private_npv(df, df_fuelCosts, menu_mp, policy_scenario, equipment_specs, interest_rate=0.07)
df_euss_am_mp8_home = calculate_private_NPV(df=df_euss_am_mp8_home,
                                            df_fuelCosts=df_mp8_scenario_fuelCosts,
                                            menu_mp=menu_mp,
                                            input_mp=input_mp,
                                            policy_scenario='No Inflation Reduction Act',
                                            interest_rate=0.07,
                                            )

print(f"""  
====================================================================================================================================================================
PRIVATE PERSPECTIVE COSTS AND BENEFITS: NO INFLATION REDUCTION ACT
====================================================================================================================================================================
calculate_lifetime_private_impact.py file contains the definition for the calculate_private_NPV function.
Additional information on fuel price lookups can be found in the calculate_fuel_costs.py file. 
      
DATAFRAME FOR MP8 AFTER CALCULATING PRIVATE NPV: df_euss_am_mp8_home

{df_euss_am_mp8_home}
      
""")

# %%
from cmu_tare_model.functions.determine_adoption_potential import *

# adoption_decision(df, policy_scenario)
df_euss_am_mp8_home = adoption_decision(df=df_euss_am_mp8_home,
                                        menu_mp=menu_mp,
                                        policy_scenario='No Inflation Reduction Act'
                                        )

print(f"""
====================================================================================================================================================================
ADOPTION FEASIBILITY OF VARIOUS RETROFITS: NO INFLATION REDUCTION ACT
====================================================================================================================================================================
determine_adoption_potential.py file contains the definition for the adoption_decision function.

DATAFRAME FOR MP8 AFTER DETERMINING ADOPTION FEASIBILITY: df_euss_am_mp8_home
      
{df_euss_am_mp8_home}
      
""")

# %% [markdown]
# # Basic IRA-Reference Scenario:
# ## - NREL End-Use Savings Shapes Database: Measure Package 8
# ## - AEO2023 REFERENCE CASE - HDD and Fuel Price Projections
# ## - Cambium 2023 MidCase Scenario

# %%
# Measure Package 8
scenario_name = 'Basic IRA-Reference'
cost_scenario = 'Fuel Costs: AEO2023 Reference Case'
grid_scenario = 'Electricity Grid: Cambium 2023 MidCase Scenario'

print(f"""
====================================================================================================================================================================
MODEL SCENARIO
====================================================================================================================================================================
Scenario {scenario_name}:
Basic Retrofit: Measure Package {menu_mp}
{cost_scenario}
{grid_scenario}
====================================================================================================================================================================
""")

# %%
# from cmu_tare_model.functions.calculate_lifetime_public_impact import *

# calculate_public_npv(df, df_baseline_damages, df_mp_damages, menu_mp, policy_scenario, equipment_specs, interest_rate=0.02)
df_euss_am_mp8_home = calculate_public_npv(df=df_euss_am_mp8_home,
                                           df_baseline_damages=df_baseline_scenario_damages,
                                           df_mp_damages=df_mp8_scenario_damages,
                                           menu_mp=menu_mp,
                                           policy_scenario='AEO2023 Reference Case',
                                           interest_rate=0.02,
                                           )

print(f"""  
====================================================================================================================================================================
PUBLIC PERSPECTIVE COSTS AND BENEFITS: AEO2023 REFERENCE CASE
====================================================================================================================================================================
calculate_lifetime_public_impact.py file contains the definition for the calculate_public_npv function.
Additional information on emissions/damage factor lookups as well as marginal damages calculation methods can be found in the calculate_emissions_damages.py file. 
      
DATAFRAME FOR MP8 AFTER CALCULATING PUBLIC NPV: df_euss_am_mp8_home
      
{df_euss_am_mp8_home}
      
""")

# %%
# from cmu_tare_model.functions.calculate_lifetime_private_impact import *

# calculate_private_npv(df, df_fuelCosts, menu_mp, policy_scenario, equipment_specs, interest_rate=0.07)
df_euss_am_mp8_home = calculate_private_NPV(df=df_euss_am_mp8_home,
                                            df_fuelCosts=df_mp8_scenario_fuelCosts,
                                            menu_mp=menu_mp,
                                            input_mp=input_mp,
                                            policy_scenario='AEO2023 Reference Case',
                                            interest_rate=0.07,
                                            )

print(f"""  
====================================================================================================================================================================
PRIVATE PERSPECTIVE COSTS AND BENEFITS: AEO2023 REFERENCE CASE
====================================================================================================================================================================
calculate_lifetime_private_impact.py file contains the definition for the calculate_private_NPV function.
Additional information on fuel price lookups can be found in the calculate_fuel_costs.py file. 
      
DATAFRAME FOR MP8 AFTER CALCULATING PRIVATE NPV: df_euss_am_mp8_home

{df_euss_am_mp8_home}
      
""")

# %%
# from cmu_tare_model.functions.determine_adoption_potential import *

# adoption_decision(df, policy_scenario)
df_euss_am_mp8_home = adoption_decision(df=df_euss_am_mp8_home,
                                        menu_mp=menu_mp,
                                        policy_scenario='AEO2023 Reference Case'
                                        )

print(f"""
====================================================================================================================================================================
ADOPTION FEASIBILITY OF VARIOUS RETROFITS: AEO2023 REFERENCE CASE
====================================================================================================================================================================
determine_adoption_potential.py file contains the definition for the adoption_decision function.

DATAFRAME FOR MP8 AFTER DETERMINING ADOPTION FEASIBILITY: df_euss_am_mp8_home
      
{df_euss_am_mp8_home}
      
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


