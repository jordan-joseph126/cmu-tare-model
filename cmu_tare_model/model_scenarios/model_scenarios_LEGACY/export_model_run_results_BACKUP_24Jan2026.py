import os
import pandas as pd
from typing import Union, Optional
import pathlib

def export_model_run_output(
    df_results_export: pd.DataFrame,
    results_category: str,
    menu_mp: Union[int, str],
    output_folder_path: Optional[str] = None,
    location_id: Optional[str] = None,
    results_export_formatted_date: Optional[str] = None
) -> None:
    """Export data for various result categories to appropriate directories.
    
    This function exports DataFrame results to CSV files in the appropriate 
    directory based on the results category and measure package. It supports
    the expanded sensitivity analysis categories and directory structure.
    
    Args:
        df_results_export: DataFrame containing the data to be exported.
        results_category: Type of results being exported:
            - For summaries: 'summary_baseline', 'summary_basic_ap2', etc.
            - For damages: 'damages_climate_IRA', 'damages_climate_noIRA', etc.
            - For fuel costs: 'fuel_costs_IRA', 'fuel_costs_noIRA'
            - For consumption: 'consumption'
        menu_mp: Measure package identifier (0 for baseline, 8/9/10 for retrofits).
        output_folder_path: Base directory for exporting results. If None, uses
            a global/predefined variable.
        location_id: Location identifier for the filename. If None, uses
            a global/predefined variable.
        results_export_formatted_date: Formatted date string for the filename. If None,
            uses a global/predefined variable.
            
    Raises:
        ValueError: If the results_category is not recognized or if required
            parameters are missing.
        OSError: If there is an error creating directories or writing the file.
    """
    print("-------------------------------------------------------------------------------------------------------")
    
    # Use global variables if parameters are not provided
    global_vars = globals()
    if output_folder_path is None:
        if 'output_folder_path' in global_vars:
            output_folder_path = global_vars['output_folder_path']
        else:
            raise ValueError("output_folder_path must be provided either as a parameter or as a global variable")
    
    if location_id is None:
        if 'location_id' in global_vars:
            location_id = global_vars['location_id']
        else:
            raise ValueError("location_id must be provided either as a parameter or as a global variable")
    
    if results_export_formatted_date is None:
        if 'results_export_formatted_date' in global_vars:
            results_export_formatted_date = global_vars['results_export_formatted_date']
        else:
            raise ValueError("results_export_formatted_date must be provided either as a parameter or as a global variable")
    
    # Standardize menu_mp to string
    menu_mp = str(menu_mp)
    
    # Determine output directory and filename based on results_category
    if results_category.startswith('summary_'):
        # Handle summary results with different models
        model_type = results_category.split('_', 1)[1]  # e.g., 'baseline', 'basic_ap2'
        
        if model_type == 'baseline':
            results_filename = f"baseline_results_{location_id}_{results_export_formatted_date}.csv"
            results_change_directory = os.path.join("baseline_summary", "summary_baseline")
            print(f"BASELINE SUMMARY RESULTS:")
        else:
            # Parse retrofit level and model from model_type
            # e.g., 'basic_ap2' -> retrofit_level='basic', model='ap2'
            parts = model_type.split('_', 1)
            retrofit_level = parts[0]  # 'basic', 'moderate', 'advanced'
            model = parts[1] if len(parts) > 1 else ''  # 'ap2', 'easiur', 'inmap'
            
            # Map retrofit level to directory and expected menu_mp
            if retrofit_level == 'basic':
                retrofit_dir = "retrofit_basic_summary"
                menu_mp_expected = '8'
            elif retrofit_level == 'moderate':
                retrofit_dir = "retrofit_moderate_summary"
                menu_mp_expected = '9'
            elif retrofit_level == 'advanced':
                retrofit_dir = "retrofit_advanced_summary"
                menu_mp_expected = '10'
            else:
                raise ValueError(f"Unrecognized retrofit level in results_category: {retrofit_level}")
            
            # Validate that menu_mp matches the expected value for the retrofit level
            if menu_mp != menu_mp_expected:
                print(f"Warning: menu_mp={menu_mp} doesn't match expected value {menu_mp_expected} for {retrofit_level} retrofit")
            
            results_filename = f"mp{menu_mp}_results_{location_id}_{results_export_formatted_date}.csv"
            results_change_directory = os.path.join(retrofit_dir, f"summary_{retrofit_level}_{model}")
            print(f"MEASURE PACKAGE {menu_mp} (MP{menu_mp}) RESULTS for model {model}:")
            
    elif results_category.startswith(('damages_', 'fuel_costs_')):
        # Handle damages and fuel costs with different policy scenarios
        if results_category.startswith('damages_'):
            base_dir = "supplemental_data_damages"
        else:  # fuel_costs
            base_dir = "supplemental_data_fuelCosts"
        
        # Use the full results_category as the subdirectory name
        results_change_directory = os.path.join(base_dir, results_category)
        results_filename = f"mp{menu_mp}_{results_category}_{location_id}_{results_export_formatted_date}.csv"
        print(f"SUPPLEMENTAL INFORMATION DATAFRAME: {results_category}")
        
    elif results_category == 'consumption':
        # Handle consumption data
        results_change_directory = os.path.join("supplemental_data_consumption", "consumption_all_scenarios")
        results_filename = f"mp{menu_mp}_data_{results_category}_{location_id}_{results_export_formatted_date}.csv"
        print(f"SUPPLEMENTAL INFORMATION DATAFRAME: {results_category}")
        
    else:
        raise ValueError(f"Unrecognized results_category: {results_category}")
    
    # Create output directory if it doesn't exist
    full_directory = os.path.join(output_folder_path, results_change_directory)
    pathlib.Path(full_directory).mkdir(parents=True, exist_ok=True)
    
    # Export dataframe results as a CSV to the specified filepath
    results_export_filepath = os.path.join(full_directory, results_filename)
    
    try:
        df_results_export.to_csv(results_export_filepath)
        print(f"Dataframe results will be saved in this csv file: {results_filename}")
        print(f"Dataframe for MP{menu_mp} {results_category} results were exported here: {results_export_filepath}")
    except Exception as e:
        raise OSError(f"Error exporting data to {results_export_filepath}: {str(e)}")
    
    print("-------------------------------------------------------------------------------------------------------", "\n")
