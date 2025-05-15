import os
import pandas as pd
from typing import Optional, Dict, Any, Union

def load_scenario_data(
    end_use: str,
    output_folder_path: str,
    scenario_string: str,
    model_run_date_time: str,
    columns_to_string: Optional[Dict[str, str]] = None
) -> Optional[pd.DataFrame]:
    """Load scenario data from a specified folder and date.
    
    This function searches for files matching the given scenario and date
    in the specified folder path, and loads the first matching file as a
    pandas DataFrame.
    
    Args:
        end_use: Description of the data being loaded (used for logging).
        output_folder_path: Base path where scenario folders are located.
        scenario_string: Path to the scenario data (can include subdirectories).
        model_run_date_time: Date/time string to filter files by (must be in filename).
        columns_to_string: Dictionary mapping column names to string dtypes.
            Defaults to None.
        
    Returns:
        DataFrame containing the loaded data, or None if no matching file is found.
    
    Raises:
        FileNotFoundError: If the specified scenario folder doesn't exist.
    """
    # Construct the output folder path with the policy_scenario of interest
    scenario_folder_path = os.path.join(output_folder_path, scenario_string)
    print(f"Output Results Folder Path: {scenario_folder_path}")
    
    # Check if the folder exists
    if not os.path.isdir(scenario_folder_path):
        raise FileNotFoundError(f"Scenario folder not found: {scenario_folder_path}")
    
    # List all files in the specified folder with the specified date in the filename
    try:
        files = [f for f in os.listdir(scenario_folder_path) 
                if os.path.isfile(os.path.join(scenario_folder_path, f)) 
                and model_run_date_time in f]
    except Exception as e:
        print(f"Error accessing folder {scenario_folder_path}: {str(e)}")
        return None
    
    # Initialize dataframe as None
    df_outputs = None
    
    # Attempt to load the first matching file
    if files:
        file_path = os.path.join(scenario_folder_path, files[0])  # Assumes the first file is the correct one
        
        try:
            df_outputs = pd.read_csv(file_path, index_col=0, dtype=columns_to_string)
            print(f"Loaded {end_use} data for policy_scenario '{scenario_string}'", "\n")
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
    else:
        print(f"No files found matching date '{model_run_date_time}' in folder {scenario_folder_path}", "\n")
    
    if df_outputs is None:
        print(f"No {end_use} data found for policy_scenario '{scenario_string}'")
    
    return df_outputs
