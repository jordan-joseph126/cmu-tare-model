import os
from matplotlib.pylab import rint
import pandas as pd
import gc
from typing import Optional, Dict, Union

from cmu_tare_model.constants import RCM_MODELS, VERBOSE, VALID_MENU_MPS
from cmu_tare_model.constants import PRIVATE_DISCOUNT_RATE_SHORT_KEYS

def load_model_run_output(
    results_category: str,
    menu_mp: Union[int, str],
    output_folder_path: str,
    location_id: str,
    results_export_formatted_date: str,
    rcm_model: Optional[str] = None,
    discount_rate: Optional[str] = None,
    use_chunked_loading: bool = True,
    chunk_size: int = 50000,
    verbose: bool = VERBOSE
) -> Optional[pd.DataFrame]:
    """Load model run results from CSV files (reverse of export_model_run_output).
    
    This function loads DataFrame results from CSV files organized by result type.
    It mirrors the export_model_run_output() function exactly, using the same
    parameters to construct the correct file path.
    
    Args:
        results_category: Category of results being loaded. Valid options:
            - 'summary_baseline': Baseline summary results
            - 'summary': Retrofit summary results (requires rcm_model and discount_rate_col_name)
            - 'damages_climate_IRA', 'damages_climate_noIRA': Climate damages
            - 'damages_health_IRA', 'damages_health_noIRA': Health damages
            - 'fuel_costs_IRA', 'fuel_costs_noIRA': Fuel costs
        menu_mp: Measure package identifier (0 for baseline, VALID_MENU_MPS for retrofits).
        output_folder_path: Base directory where results are stored.
        location_id: Location identifier in the filename (e.g., 'national', 'NYC').
        results_export_formatted_date: Date string in the filename (e.g., '2024-01-24_10-30').
        rcm_model: RCM model used for health damage calculations (e.g., 'ap2', 'easiur', 
            'inmap'). Required when results_category='summary' and menu_mp != 0.
        discount_rate: Short key for discount rate method (e.g., 'fixed_base', 'variable').
            Required when results_category='summary' and menu_mp != 0.
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
        ...     discount_rate='fixed_base'
        ... )
    """
    # Validate required parameters
    if output_folder_path is None:
        raise ValueError("output_folder_path is required")
    if location_id is None:
        raise ValueError("location_id is required")
    if results_export_formatted_date is None:
        raise ValueError("results_export_formatted_date is required")

    # Standardize menu_mp to int for validation, then to string for file paths
    menu_mp_int = int(menu_mp)
    menu_mp_str = str(menu_mp)

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
        if discount_rate is None:
            raise ValueError("discount_rate is required for retrofit summary results (results_category='summary')")
        
        # Validate measure package (only for summary results)
        if menu_mp_int not in VALID_MENU_MPS:
            raise ValueError(f"menu_mp must be one of {VALID_MENU_MPS}, got {menu_mp_int}")
        
        # Validate RCM model is valid
        if rcm_model not in RCM_MODELS:
            raise ValueError(f"rcm_model must be one of {RCM_MODELS}, got '{rcm_model}'")
        
        # Validate discount rate is valid
        if discount_rate not in PRIVATE_DISCOUNT_RATE_SHORT_KEYS:
            raise ValueError(
                f"discount_rate must be one of {PRIVATE_DISCOUNT_RATE_SHORT_KEYS}, "
                f"got '{discount_rate}'"
            )
                
        # Build directory path using sensitivity parameters
        directory_path = os.path.join(
            f"retrofit_mp{menu_mp_str}_results",
            f"summary_mp{menu_mp_str}_{rcm_model}_{discount_rate}"
        )
        filename = f"mp{menu_mp_str}_results_{location_id}_{results_export_formatted_date}.csv"
        
    elif results_category.startswith('damages_'):
        # Damages results (climate or health, IRA or noIRA)
        directory_path = os.path.join("supplemental_data_damages", results_category)
        filename = f"mp{menu_mp_str}_{results_category}_{location_id}_{results_export_formatted_date}.csv"
        
    elif results_category.startswith('fuel_costs_'):
        # Fuel costs results (IRA or noIRA)
        directory_path = os.path.join("supplemental_data_fuelCosts", results_category)
        filename = f"mp{menu_mp_str}_{results_category}_{location_id}_{results_export_formatted_date}.csv"
        
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
        raise FileNotFoundError(f"File not found: {full_filepath}")
    
    # Load the DataFrame
    if use_chunked_loading:
        # Load file in chunks to reduce memory usage
        if verbose:
            print(f"Loading {filename} in chunks of {chunk_size:,} rows...")
        
        # Read file in chunks using pandas built-in chunksize parameter
        chunk_reader = pd.read_csv(full_filepath, index_col=0, low_memory=False, chunksize=chunk_size)
        
        # Collect all chunks in a list for concatenation
        chunk_list = []
        chunk_count = 0
        
        for chunk in chunk_reader:
            chunk_list.append(chunk)
            chunk_count += 1
            
            # Provide progress updates every 5 chunks to monitor loading
            if chunk_count % 5 == 0:
                if verbose:
                    print(f"  Loaded chunk {chunk_count} ({len(chunk):,} rows)")
        
        # Combine all chunks into a single DataFrame preserving the original index
        df_model_run_output = pd.concat(chunk_list, ignore_index=False)
        
        # Clean up chunk list to free intermediate memory
        del chunk_list
        gc.collect()  # Force garbage collection to free memory immediately
        
        if verbose:
            print(f"  Successfully combined {chunk_count} chunks into DataFrame with shape {df_model_run_output.shape}")
        
    else:
        # Use standard loading method
        df_model_run_output = pd.read_csv(full_filepath, index_col=0, low_memory=False)
        
    if verbose:
        print(f"Loaded: {filename}")
        print(f"Shape: {df_model_run_output.shape}")
        print()
    
    return df_model_run_output


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
from typing import Dict, Optional

def load_measure_package_data(
    menu_mp: int,
    output_folder_path: str,
    location_id: str,
    model_run_date_time: str,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load all discount rate × RCM combinations for a measure package.
    
    Creates a nested dictionary structure using short keys:
    {discount_rate: {rcm_model: DataFrame}}
    
    Args:
        menu_mp: Measure package identifier (VALID_MENU_MPS).
        output_folder_path: Base directory containing exported results.
        location_id: Geographic identifier used in filenames.
        model_run_date_time: Timestamp string from the model run.
    
    Returns:
        Nested dictionary: {discount_rate: {rcm_model: DataFrame}}
    """
    # Initialize nested dictionary with SHORT KEYS and proper level ordering
    dataframes = {
        discount_rate: {rcm: None for rcm in RCM_MODELS}
        for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS
    }
    
    print(f"Loading MP{menu_mp} data...")
    
    # Iterate in same order as dictionary structure: discount rate → RCM
    for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS:
        print(f"  {discount_rate}: ", end="")
        
        for rcm_model in RCM_MODELS:
            df = load_model_run_output(
                results_category='summary',
                menu_mp=menu_mp,
                output_folder_path=output_folder_path,
                location_id=location_id,
                results_export_formatted_date=model_run_date_time,
                rcm_model=rcm_model,
                discount_rate=discount_rate,  # Use short key
                use_chunked_loading=True,
                chunk_size=10000
            )
            
            dataframes[discount_rate][rcm_model] = df
            print("✓" if df is not None else "✗", end=" ")
        
        print()  # Newline after each discount rate
    
    print(f"MP{menu_mp} loading complete!\n")
    return dataframes
