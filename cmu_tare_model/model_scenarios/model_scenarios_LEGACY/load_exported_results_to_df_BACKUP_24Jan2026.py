import os
import pandas as pd
import gc
from typing import Optional, Dict, Union

def load_scenario_data(
    end_use: str,
    output_folder_path: str,
    scenario_string: str,
    model_run_date_time: str,
    columns_to_string: Optional[Dict[Union[str, int], str]] = None,
    use_chunked_loading: bool = True,
    chunk_size: int = 50000
) -> Optional[pd.DataFrame]:
    """Load scenario data from a specified folder and date.
    
    This function searches for files matching the given scenario and date
    in the specified folder path, and loads the first matching file as a
    pandas DataFrame. Includes optional chunked loading for memory efficiency
    when working with large datasets.
    
    Args:
        end_use: Description of the data being loaded (used for logging).
        output_folder_path: Base path where scenario folders are located.
        scenario_string: Path to the scenario data (can include subdirectories).
        model_run_date_time: Date/time string to filter files by (must be in filename).
        columns_to_string: Dictionary mapping column names/indices to string dtypes.
            Supports both string keys (column names) and integer keys (column indices).
            Defaults to None.
        use_chunked_loading: Whether to load the file in chunks to reduce memory usage.
            Useful for large files that may cause memory allocation errors.
            Defaults to False to maintain existing behavior.
        chunk_size: Number of rows to read per chunk when using chunked loading.
            Larger chunks use more memory but may be faster. Smaller chunks use less
            memory but may be slower. Defaults to 50,000 rows.
        
    Returns:
        DataFrame containing the loaded data, or None if no matching file is found
        or if loading fails.
    
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
            if use_chunked_loading:
                # Load file in chunks to reduce memory usage
                print(f"Loading {end_use} data in chunks of {chunk_size:,} rows...")
                
                # Read file in chunks using pandas built-in chunksize parameter
                chunk_reader = pd.read_csv(file_path, index_col=0, dtype=columns_to_string, chunksize=chunk_size)
                
                # Collect all chunks in a list for concatenation
                chunk_list = []
                chunk_count = 0
                
                for chunk in chunk_reader:
                    chunk_list.append(chunk)
                    chunk_count += 1
                    
                    # Provide progress updates every 5 chunks to monitor loading
                    if chunk_count % 5 == 0:
                        print(f"  Loaded chunk {chunk_count} ({len(chunk):,} rows)")
                
                # Combine all chunks into a single DataFrame preserving the original index
                df_outputs = pd.concat(chunk_list, ignore_index=False)
                
                # Clean up chunk list to free intermediate memory
                del chunk_list
                gc.collect()  # Force garbage collection to free memory immediately
                
                print(f"  Successfully combined {chunk_count} chunks into DataFrame with shape {df_outputs.shape}")
                
            else:
                # Use original loading method for backward compatibility
                df_outputs = pd.read_csv(file_path, index_col=0, dtype=columns_to_string)
            
            print(f"Loaded {end_use} data for policy_scenario '{scenario_string}'", "\n")
            
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            return None
    else:
        print(f"No files found matching date '{model_run_date_time}' in folder {scenario_folder_path}", "\n")
    
    if df_outputs is None:
        print(f"No {end_use} data found for policy_scenario '{scenario_string}'")
    
    return df_outputs
