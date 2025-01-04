import os
import pandas as pd

# from config import PROJECT_ROOT

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
SAVE MODEL RUN RESULTS AND EXPORT TO CSV
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

def load_scenario_data(end_use, output_folder_path, scenario_string, model_run_date_time, columns_to_string):
    # Construct the output folder path with the policy_scenario of interest
    scenario_folder_path = os.path.join(output_folder_path, scenario_string)
    print(f"Output Results Folder Path: {scenario_folder_path}")

    # List all files in the specified folder with the specified date in the filename
    files = [f for f in os.listdir(scenario_folder_path) if os.path.isfile(os.path.join(scenario_folder_path, f)) and model_run_date_time in f]

    # Initialize dataframe as None
    df_outputs = None

    # Assume there is one main file per policy_scenario that includes all necessary data
    if files:
        file_path = os.path.join(scenario_folder_path, files[0])  # Assumes the first file is the correct one

        if os.path.exists(file_path):
            df_outputs = pd.read_csv(file_path, index_col=0, dtype=columns_to_string)
            print(f"Loaded {end_use} data for policy_scenario '{scenario_string}'", "\n")
        else:
            print("File not found for the specified policy_scenario", "\n")

    if df_outputs is None:
        print(f"No {end_use} data found for policy_scenario '{scenario_string}'")

    return df_outputs