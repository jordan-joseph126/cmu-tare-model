import os
import pandas as pd
import gc
from typing import Optional, Dict, Union

from cmu_tare_model.constants import RCM_MODELS
from cmu_tare_model.utils.discounting import PRIVATE_DISCOUNT_RATE_COLS

def load_model_run_output(
    results_category: str,
    menu_mp: Union[int, str],
    output_folder_path: str,
    location_id: str,
    results_export_formatted_date: str,
    rcm_model: Optional[str] = None,
    discount_rate_col: Optional[str] = None,
    columns_to_string: Optional[Dict[Union[str, int], str]] = None,
    use_chunked_loading: bool = True,
    chunk_size: int = 50000
) -> Optional[pd.DataFrame]:
    """Load model run results from CSV files (reverse of export_model_run_output).
    
    This function loads DataFrame results from CSV files organized by result type.
    It mirrors the export_model_run_output() function exactly, using the same
    parameters to construct the correct file path.
    
    Args:
        results_category: Category of results being loaded. Valid options:
            - 'summary_baseline': Baseline summary results
            - 'summary': Retrofit summary results (requires rcm_model and discount_rate_col)
            - 'damages_climate_IRA', 'damages_climate_noIRA': Climate damages
            - 'damages_health_IRA', 'damages_health_noIRA': Health damages
            - 'fuel_costs_IRA', 'fuel_costs_noIRA': Fuel costs
        menu_mp: Measure package identifier (0 for baseline, 8/9/10 for retrofits).
        output_folder_path: Base directory where results are stored.
        location_id: Location identifier in the filename (e.g., 'national', 'NYC').
        results_export_formatted_date: Date string in the filename (e.g., '2024-01-24_10-30').
        rcm_model: RCM model used for health damage calculations (e.g., 'ap2', 'easiur', 
            'inmap'). Required when results_category='summary' and menu_mp != 0.
        discount_rate_col: Discount rate method used (e.g., 'private_discount_rate_fixed_base',
            'private_discount_rate_variable'). Required when results_category='summary' and menu_mp != 0.
        columns_to_string: Dictionary mapping column indices/names to string dtype.
        use_chunked_loading: Whether to load the file in chunks to reduce memory usage.
        chunk_size: Number of rows to read per chunk when using chunked loading.
            
    Returns:
        DataFrame (df_model_run_output) containing the loaded data, or None if file not found or loading fails.
        
    Raises:
        ValueError: If any required parameter is missing, results_category is invalid,
            or sensitivity parameters are missing when required.
        FileNotFoundError: If the expected directory or file doesn't exist.
        
    Example:
        >>> # Load baseline
        >>> df_output_baseline = load_model_run_output(
        ...     results_category='summary_baseline',
        ...     menu_mp=0,
        ...     output_folder_path='./output_results',
        ...     location_id='national',
        ...     results_export_formatted_date='2024-01-24_10-30'
        ... )
        >>> 
        >>> # Load retrofit summary (with sensitivity parameters)
        >>> df_output_mp9_ap2_fixed_base = load_model_run_output(
        ...     results_category='summary',
        ...     menu_mp=9,
        ...     output_folder_path='./output_results',
        ...     location_id='national',
        ...     results_export_formatted_date='2024-01-24_10-30',
        ...     rcm_model='ap2',
        ...     discount_rate_col='private_discount_rate_fixed_base'
        ... )
    """
    # Validate required parameters
    if output_folder_path is None:
        raise ValueError("output_folder_path is required")
    if location_id is None:
        raise ValueError("location_id is required")
    if results_export_formatted_date is None:
        raise ValueError("results_export_formatted_date is required")

    # Standardize menu_mp to string
    menu_mp = str(menu_mp)

    # Build directory path and filename based on results_category
    if results_category == 'summary_baseline':
        # Baseline summary results
        directory_path = os.path.join("baseline_summary", "summary_baseline")
        filename = f"baseline_results_{location_id}_{results_export_formatted_date}.csv"
        
    elif results_category == 'summary':
        # Retrofit summary results with sensitivity tracking
        # Validate that sensitivity parameters are provided
        if rcm_model is None:
            raise ValueError("rcm_model is required for retrofit summary results (results_category='summary')")
        if discount_rate_col is None:
            raise ValueError("discount_rate_col is required for retrofit summary results (results_category='summary')")
        
        # Validate measure package (only for summary results)
        if menu_mp not in ['8', '9', '10']:
            raise ValueError(f"menu_mp must be 8, 9, or 10 (Basic, Moderate, or Advanced), got {menu_mp}")
        
        # Validate RCM model is valid
        if rcm_model not in RCM_MODELS:
            raise ValueError(f"rcm_model must be one of {RCM_MODELS}, got '{rcm_model}'")
        
        # Validate discount rate is valid
        if discount_rate_col not in PRIVATE_DISCOUNT_RATE_COLS:
            raise ValueError(f"discount_rate_col must be one of {PRIVATE_DISCOUNT_RATE_COLS}, got '{discount_rate_col}'")
                
        # Build directory path using sensitivity parameters
        directory_path = os.path.join(
            f"retrofit_mp{menu_mp}_results",
            f"summary_mp{menu_mp}_{rcm_model}_{discount_rate_col}"
        )
        filename = f"mp{menu_mp}_results_{location_id}_{results_export_formatted_date}.csv"
        
    elif results_category.startswith('damages_'):
        # Damages results (climate or health, IRA or noIRA)
        directory_path = os.path.join("supplemental_data_damages", results_category)
        filename = f"mp{menu_mp}_{results_category}_{location_id}_{results_export_formatted_date}.csv"
        
    elif results_category.startswith('fuel_costs_'):
        # Fuel costs results (IRA or noIRA)
        directory_path = os.path.join("supplemental_data_fuelCosts", results_category)
        filename = f"mp{menu_mp}_{results_category}_{location_id}_{results_export_formatted_date}.csv"
        
    else:
        raise ValueError(
            f"Unrecognized results_category: {results_category}. "
            f"Must be 'summary_baseline', 'summary', 'damages_*', or 'fuel_costs_*'"
        )

    # Construct full file path
    full_directory = os.path.join(output_folder_path, directory_path)
    full_filepath = os.path.join(full_directory, filename)
    
    # Check if directory exists
    if not os.path.isdir(full_directory):
        raise FileNotFoundError(f"Directory not found: {full_directory}")
    
    # Check if file exists
    if not os.path.isfile(full_filepath):
        print(f"Warning: File not found: {full_filepath}")
        return None
    
    # Load the DataFrame
    try:
        if use_chunked_loading:
            # Load file in chunks to reduce memory usage
            print(f"Loading {filename} in chunks of {chunk_size:,} rows...")
            
            # Read file in chunks using pandas built-in chunksize parameter
            chunk_reader = pd.read_csv(full_filepath, index_col=0, dtype=columns_to_string, chunksize=chunk_size)
            
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
            df_model_run_output = pd.concat(chunk_list, ignore_index=False)
            
            # Clean up chunk list to free intermediate memory
            del chunk_list
            gc.collect()  # Force garbage collection to free memory immediately
            
            print(f"  Successfully combined {chunk_count} chunks into DataFrame with shape {df_model_run_output.shape}")
            
        else:
            # Use standard loading method
            df_model_run_output = pd.read_csv(full_filepath, index_col=0, dtype=columns_to_string)
        
        print(f"Loaded: {filename}")
        print(f"Shape: {df_model_run_output.shape}")
        print()
        
        return df_model_run_output
        
    except Exception as e:
        print(f"Error loading file {full_filepath}: {str(e)}")
        return None
