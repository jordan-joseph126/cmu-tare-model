
"""
This file runs all of the code responsible for generating the output CSVs but does not visualize the data.
Data visualization is done in the main program.
"""

import os
import pandas as pd
from IPython import get_ipython

# # Format the name of the exported results file using the location ID
# from datetime import datetime
# result_export_time = datetime.now()
# results_export_formatted_date = result_export_time.strftime("%Y-%m-%d_%H-%M")

from config import PROJECT_ROOT

"""
=====================================================================================================================================================================================
FUNCTIONS USED TO RUN THE MODEL AND EXPORT RESULTS
=====================================================================================================================================================================================
"""
# SAVE MODEL RUN RESULTS AND EXPORT TO CSV
# If bldg_id is now the index in all DataFrames (df_compare, df_results_IRA, and df_results_IRA_gridDecarb):
def clean_df_merge(df_compare, df_results_IRA, df_results_IRA_gridDecarb):
    # Identify common columns (excluding 'bldg_id' which is the merging key)
    common_columns_IRA = set(df_compare.columns) & set(df_results_IRA.columns)
    common_columns_IRA.discard('bldg_id')
        
    # Drop duplicate columns in df_results_IRA and merge
    df_results_IRA = df_results_IRA.drop(columns=common_columns_IRA)
    print(f"""Dropped the following duplicate columns before merge: 
    {common_columns_IRA}
    """)
    # merged_df = pd.merge(df_compare, df_results_IRA, on='bldg_id', how='inner')
    merged_df = pd.merge(df_compare, df_results_IRA, how='inner', left_index=True, right_index=True)

    # Repeat the steps above for the merged_df and df_results_IRA_gridDecarb
    common_columnsIRA_gridDecarb = set(merged_df.columns) & set(df_results_IRA_gridDecarb.columns)
    common_columnsIRA_gridDecarb.discard('bldg_id')
    df_results_IRA_gridDecarb = df_results_IRA_gridDecarb.drop(columns=common_columnsIRA_gridDecarb)
    print(f"""Dropped the following duplicate columns before merge: 
    {common_columnsIRA_gridDecarb}
    """)
        
    # Create cleaned, merged results df with no duplicate columns
    # df_results_export = pd.merge(merged_df, df_results_IRA_gridDecarb, on='bldg_id', how='inner')
    df_results_export = pd.merge(merged_df, df_results_IRA_gridDecarb, how='inner', left_index=True, right_index=True)
    print("Dataframes have been cleaned of duplicate columns and merged successfully. Ready to export!")
    return df_results_export

def export_model_run_output(df_results_export, results_category, menu_mp):
    """
    Exports data for results summaries (npv, adoption, impact) and supplemental info (consumption, damages, fuel costs)

    Parameters:
    df_results_export (pd.DataFrame): DataFrame containing data for different scenarios.
    results_category (str): Determines the type of info being exported.
        - Accepted: 'summary', 'consumption', 'damages', 'fuelCost'
    menu_mp (int or str): Determines the measure package or retrofit being conducted
    
    """
    print("-------------------------------------------------------------------------------------------------------")
    # Baseline model run results
    if results_category == 'summary':
        if menu_mp == '0' or menu_mp==0:
            results_filename = f"baseline_results_{location_id}_{results_export_formatted_date}.csv"
            print(f"BASELINE RESULTS:")
            print(f"Dataframe results will be saved in this csv file: {results_filename}")

            # Change the directory to the upload folder and export the file
            results_change_directory = "baseline_summary"

        # Measure Package model run results
        else:
            if menu_mp == '8' or menu_mp==8:
                print(f"MEASURE PACKAGE {menu_mp} (MP{menu_mp}) RESULTS:")
                results_filename = f"mp{menu_mp}_results_{location_id}_{results_export_formatted_date}.csv"
                print(f"Dataframe results will be saved in this csv file: {results_filename}")

                # Change the directory to the upload folder and export the file
                results_change_directory = "retrofit_basic_summary"

            elif menu_mp == '9' or menu_mp==9:
                results_filename = f"mp{menu_mp}_results_{location_id}_{results_export_formatted_date}.csv"
                print(f"MEASURE PACKAGE {menu_mp} (MP{menu_mp}) RESULTS:")
                print(f"Dataframe results will be saved in this csv file: {results_filename}")

                # Change the directory to the upload folder and export the file
                results_change_directory = "retrofit_moderate_summary"

            elif menu_mp == '10' or menu_mp==10:
                results_filename = f"mp{menu_mp}_results_{location_id}_{results_export_formatted_date}.csv"
                print(f"MEASURE PACKAGE {menu_mp} (MP{menu_mp}) RESULTS:")
                print(f"Dataframe results will be saved in this csv file: {results_filename}")

                # Change the directory to the upload folder and export the file
                results_change_directory = "retrofit_advanced_summary"

            else:
                print("No matching scenarios for this Measure Package (MP)")

    # This includes exported dataframes for calculated consumption, damages, and fuel costs
    else:
        results_filename = f"mp{menu_mp}_data_{results_category}_{location_id}_{results_export_formatted_date}.csv"
        print(f"SUPPLEMENTAL INFORMATION DATAFRAME: {results_category}")
        print(f"Dataframe results will be saved in this csv file: {results_filename}")

        # Change the directory to the upload folder and export the file
        results_change_directory = f"supplemental_data_{results_category}"

    # Export dataframe results as a csv to the specified filepath
    results_export_filepath = os.path.join(output_folder_path, results_change_directory, results_filename)
    df_results_export.to_csv(results_export_filepath)
    print(f"Dataframe for MP{menu_mp} {results_category} results were exported here: {results_export_filepath}")
    print("-------------------------------------------------------------------------------------------------------", "\n")

# ===================================================================================================================================================================================
# Measure Package 0: Baseline
# ===================================================================================================================================================================================
# BASELINE Pre-IRA Scenario:
menu_mp = 0
input_mp = 'baseline'

print(f"PROJECT_ROOT (from config.py): {PROJECT_ROOT}")

# Construct the absolute path to the .py file
relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_baseline_v2.py")
file_path = os.path.join(PROJECT_ROOT, relative_path)

# On Windows, to avoid any path-escape quirks, convert backslashes to forward slashes
file_path = file_path.replace("\\", "/")

print(f"Running file: {file_path}")

# iPthon magic command to run a .py file and import variables into the current IPython session
get_ipython().run_line_magic('run', f'-i {file_path}')  # If your path has NO spaces, no quotes needed.

print("Baseline Scenario - Model Run Complete")

# ===================================================================================================================================================================================
# EXPORT RESULTS TO CSV
# ===================================================================================================================================================================================
# SUMMARY RESULTS
export_model_run_output(df_results_export=df_euss_am_baseline_home,
                        results_category='summary',
                        menu_mp=menu_mp
                        )

print("""
Scenarios for Basic, Moderate, and Advanced Retrofit
- Rows in the Baseline CSV are merged with the Post-Retrofit Measure Package.
- The same filters are applied.

-------------------------------------------------------------------------------------------------------
MODEL SCENARIOS
-------------------------------------------------------------------------------------------------------
- BASIC/MODERATE/ADVANCED Pre-IRA Scenario:
    - NREL End-Use Savings Shapes Database: Measure Package 8/9/10
    - AEO2023 No Inflation Reduction Act
    - Cambium 2021 MidCase
      
- BASIC/MODERATE/ADVANCED IRA-Reference Scenario:
    - NREL End-Use Savings Shapes Database: Measure Package 8/9/10
    - AEO2023 REFERENCE CASE - HDD and Fuel Price Projections
    - Cambium 2022 and 2023 MidCase
""")

# ===================================================================================================================================================================================
# # BASIC RETROFIT MEASURE PACKAGE 8
# ===================================================================================================================================================================================
# ## - BASIC Pre-IRA Scenario:
#     - NREL End-Use Savings Shapes Database: Measure Package 8
#     - AEO2023 No Inflation Reduction Act
#     - Cambium 2021 MidCase
#       
# ## - BASIC IRA-Reference Scenario:
#     - NREL End-Use Savings Shapes Database: Measure Package 8
#     - AEO2023 REFERENCE CASE - HDD and Fuel Price Projections
#     - Cambium 2022 and 2023 MidCase
# ===================================================================================================================================================================================

# Measure Package 8
menu_mp = 8
input_mp = 'upgrade08'

print(f"PROJECT_ROOT (from config.py): {PROJECT_ROOT}")

# Construct the absolute path to the .py file
relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_basic_v2.py")
file_path = os.path.join(PROJECT_ROOT, relative_path)

# On Windows, to avoid any path-escape quirks, convert backslashes to forward slashes
file_path = file_path.replace("\\", "/")

print(f"Running file: {file_path}")

# iPthon magic command to run a .py file and import variables into the current IPython session
get_ipython().run_line_magic('run', f'-i {file_path}')  # If your path has NO spaces, no quotes needed.

print("Basic Retrofit Scenario - Model Run Complete")

# ===================================================================================================================================================================================
# EXPORT RESULTS TO CSV
# ===================================================================================================================================================================================

# CONSUMPTION RESULTS
export_model_run_output(df_results_export=df_mp8_scenario_consumption,
                        results_category='consumption',
                        menu_mp=8
                        )

# DAMAGES RESULTS
export_model_run_output(df_results_export=df_mp8_scenario_damages,
                        results_category='damages',
                        menu_mp=8
                        )

# FUEL COSTS RESULTS
export_model_run_output(df_results_export=df_mp8_scenario_fuelCosts,
                        results_category='fuelCosts',
                        menu_mp=8
                        )

# SUMMARY RESULTS
export_model_run_output(df_results_export=df_euss_am_mp8_home,
                        results_category='summary',
                        menu_mp=8
                        )

# ===================================================================================================================================================================================
# # MODERATE RETROFIT MEASURE PACKAGE 9
# ===================================================================================================================================================================================
# ## - MODERATE Pre-IRA Scenario:
#     - NREL End-Use Savings Shapes Database: Measure Package 9
#     - AEO2023 No Inflation Reduction Act
#     - Cambium 2021 MidCase
#       
# ## - MODERATE IRA-Reference Scenario:
#     - NREL End-Use Savings Shapes Database: Measure Package 9
#     - AEO2023 REFERENCE CASE - HDD and Fuel Price Projections
#     - Cambium 2022 and 2023 MidCase      
# ===================================================================================================================================================================================

# Measure Package 9
menu_mp = 9
input_mp = 'upgrade09'

print(f"PROJECT_ROOT (from config.py): {PROJECT_ROOT}")

# Construct the absolute path to the .py file
relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_moderate_v2.py")
file_path = os.path.join(PROJECT_ROOT, relative_path)

# On Windows, to avoid any path-escape quirks, convert backslashes to forward slashes
file_path = file_path.replace("\\", "/")

print(f"Running file: {file_path}")

# iPthon magic command to run a .py file and import variables into the current IPython session
get_ipython().run_line_magic('run', f'-i {file_path}')  # If your path has NO spaces, no quotes needed.

print("Moderate Retrofit Scenario - Model Run Complete")

# ===================================================================================================================================================================================
# EXPORT RESULTS TO CSV
# ===================================================================================================================================================================================

# CONSUMPTION RESULTS
export_model_run_output(df_results_export=df_mp9_scenario_consumption,
                        results_category='consumption',
                        menu_mp=9
                        )

# DAMAGES RESULTS
export_model_run_output(df_results_export=df_mp9_scenario_damages,
                        results_category='damages',
                        menu_mp=9
                        )

# FUEL COSTS RESULTS
export_model_run_output(df_results_export=df_mp9_scenario_fuelCosts,
                        results_category='fuelCosts',
                        menu_mp=9
                        )

# SUMMARY RESULTS
export_model_run_output(df_results_export=df_euss_am_mp9_home,
                        results_category='summary',
                        menu_mp=9
                        )

# ===================================================================================================================================================================================
# # ADVANCED RETROFIT MEASURE PACKAGE 10
# ===================================================================================================================================================================================
# ## - ADVANCED Pre-IRA Scenario:
#     - NREL End-Use Savings Shapes Database: Measure Package 10
#     - AEO2023 No Inflation Reduction Act
#     - Cambium 2021 MidCase
#       
# ## - ADVANCED IRA-Reference Scenario:
#     - NREL End-Use Savings Shapes Database: Measure Package 10
#     - AEO2023 REFERENCE CASE - HDD and Fuel Price Projections
#     - Cambium 2022 and 2023 MidCase
# ===================================================================================================================================================================================

# Measure Package 10
menu_mp = 10
input_mp = 'upgrade10'

print(f"PROJECT_ROOT (from config.py): {PROJECT_ROOT}")

# Construct the absolute path to the .py file
relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_advanced_v2.py")
file_path = os.path.join(PROJECT_ROOT, relative_path)

# On Windows, to avoid any path-escape quirks, convert backslashes to forward slashes
file_path = file_path.replace("\\", "/")

print(f"Running file: {file_path}")

# iPthon magic command to run a .py file and import variables into the current IPython session
get_ipython().run_line_magic('run', f'-i {file_path}')  # If your path has NO spaces, no quotes needed.

print("Advanced Retrofit Scenario - Model Run Complete")

# ===================================================================================================================================================================================
# EXPORT RESULTS TO CSV
# ===================================================================================================================================================================================

# CONSUMPTION RESULTS
export_model_run_output(df_results_export=df_mp10_scenario_consumption,
                        results_category='consumption',
                        menu_mp=10
                        )

# DAMAGES RESULTS
export_model_run_output(df_results_export=df_mp10_scenario_damages,
                        results_category='damages',
                        menu_mp=10
                        )

# FUEL COST RESULTS
export_model_run_output(df_results_export=df_mp10_scenario_fuelCosts,
                        results_category='fuelCosts',
                        menu_mp=10
                        )

# SUMMARY RESULTS
export_model_run_output(df_results_export=df_euss_am_mp10_home,
                        results_category='summary',
                        menu_mp=10
                        )

# ===================================================================================================================================================================================
# TIME ELAPSED
# ===================================================================================================================================================================================
end_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Calculate the elapsed time
elapsed_time = datetime.strptime(end_time, "%Y-%m-%d_%H-%M-%S") - datetime.strptime(start_time, "%Y-%m-%d_%H-%M-%S")

# Format the elapsed time
elapsed_seconds = elapsed_time.total_seconds()
elapsed_minutes = int(elapsed_seconds // 60)
elapsed_seconds = int(elapsed_seconds % 60)

# Print the elapsed time
print(f"The code took {elapsed_minutes} minutes and {elapsed_seconds} seconds to execute.")