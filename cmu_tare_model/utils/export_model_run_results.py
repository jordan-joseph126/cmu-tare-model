import os
import pandas as pd
from typing import Union, Optional
import pathlib


def export_model_run_output(
    df_results_export: pd.DataFrame,
    results_category: str,
    menu_mp: Union[int, str],
    output_folder_path: str,
    location_id: str,
    results_export_formatted_date: str,
    rcm_model: Optional[str] = None,
    discount_rate_col: Optional[str] = None
) -> None:
    """Export model run results to CSV files with sensitivity tracking.
    
    This function exports DataFrame results to CSV files organized by result type.
    For retrofit summary results, it adds sensitivity columns (RCM model and discount
    rate method) to the DataFrame before export, enabling easy filtering and analysis
    of sensitivity scenarios.
    
    Directory Structure:
        baseline_summary/summary_baseline/
            - baseline_results_{location_id}_{date}.csv
        
        retrofit_mp{menu_mp}_results/summary_mp{menu_mp}_{rcm_model}_{discount_rate_col}/
            - mp{menu_mp}_results_{location_id}_{date}.csv
            - DataFrame includes: rcm_model, discount_rate_col columns
        
        supplemental_data_damages/{results_category}/
            - mp{menu_mp}_{results_category}_{location_id}_{date}.csv
        
        supplemental_data_fuelCosts/{results_category}/
            - mp{menu_mp}_{results_category}_{location_id}_{date}.csv
    
    Args:
        df_results_export: DataFrame containing the results to export.
        results_category: Category of results being exported. Valid options:
            - 'summary_baseline': Baseline summary results
            - 'summary': Retrofit summary results (requires rcm_model and discount_rate_col)
            - 'damages_climate_IRA', 'damages_climate_noIRA': Climate damages
            - 'damages_health_IRA', 'damages_health_noIRA': Health damages
            - 'fuel_costs_IRA', 'fuel_costs_noIRA': Fuel costs
        menu_mp: Measure package identifier (0 for baseline, 8/9/10 for retrofits).
        output_folder_path: Base directory for all exports.
        location_id: Location identifier for the filename (e.g., 'NYC', 'LA').
        results_export_formatted_date: Date string for the filename (e.g., '2024_01_15').
        rcm_model: RCM model used for health damage calculations (e.g., 'ap2', 'easiur', 
            'inmap'). Required when results_category='summary' and menu_mp != 0.
        discount_rate_col: Discount rate method used (e.g., 'private_discount_rate_fixed_base',
            'private_discount_rate_variable'). Required when results_category='summary' and menu_mp != 0.
            
    Raises:
        ValueError: If any required parameter is missing, results_category is invalid,
            or sensitivity parameters are missing when required.
        OSError: If there is an error creating directories or writing the file.
    """
    print("---" * 35)
    
    # Validate required parameters
    if output_folder_path is None:
        raise ValueError("output_folder_path is required")
    if location_id is None:
        raise ValueError("location_id is required")
    if results_export_formatted_date is None:
        raise ValueError("results_export_formatted_date is required")
    
    # Standardize menu_mp to string
    menu_mp = str(menu_mp)
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_results_export_copy = df_results_export.copy()
    
    # Build directory path and filename based on results_category
    if results_category == 'summary_baseline':
        # Baseline summary results
        directory_path = os.path.join("baseline_summary", "summary_baseline")
        filename = f"baseline_results_{location_id}_{results_export_formatted_date}.csv"
        print(f"BASELINE SUMMARY RESULTS:")
        
    elif results_category == 'summary':
        # Retrofit summary results with sensitivity tracking
        # Validate that sensitivity parameters are provided
        if rcm_model is None:
            raise ValueError("rcm_model is required for retrofit summary results (results_category='summary')")
        if discount_rate_col is None:
            raise ValueError("discount_rate_col is required for retrofit summary results (results_category='summary')")
                
        # Build directory path using sensitivity parameters
        directory_path = os.path.join(
            f"retrofit_mp{menu_mp}_results",
            f"summary_mp{menu_mp}_{rcm_model}_{discount_rate_col}"
        )
        filename = f"mp{menu_mp}_results_{location_id}_{results_export_formatted_date}.csv"
        print(f"MEASURE PACKAGE {menu_mp} SUMMARY RESULTS:")
        print(f"  RCM Model: {rcm_model}")
        print(f"  Discount Rate: {discount_rate_col}")
        
    elif results_category.startswith('damages_'):
        # Damages results (climate or health, IRA or noIRA)
        directory_path = os.path.join("supplemental_data_damages", results_category)
        filename = f"mp{menu_mp}_{results_category}_{location_id}_{results_export_formatted_date}.csv"
        print(f"SUPPLEMENTAL DAMAGES: {results_category}")
        
    elif results_category.startswith('fuel_costs_'):
        # Fuel costs results (IRA or noIRA)
        directory_path = os.path.join("supplemental_data_fuelCosts", results_category)
        filename = f"mp{menu_mp}_{results_category}_{location_id}_{results_export_formatted_date}.csv"
        print(f"SUPPLEMENTAL FUEL COSTS: {results_category}")
        
    else:
        raise ValueError(
            f"Unrecognized results_category: {results_category}. "
            f"Must be 'summary_baseline', 'summary', 'damages_*', or 'fuel_costs_*'"
        )
    
    # Create full directory path (creates empty directory if it doesn't exist)
    full_directory = os.path.join(output_folder_path, directory_path)
    pathlib.Path(full_directory).mkdir(parents=True, exist_ok=True)
    
    # Export DataFrame to CSV
    full_filepath = os.path.join(full_directory, filename)
    
    try:
        df_results_export_copy.to_csv(full_filepath)
        print(f"Saved to: {filename}")
        print(f"Full path: {full_filepath}")
    except Exception as e:
        raise OSError(f"Error exporting data to {full_filepath}: {str(e)}")
    
    print("---" * 35, "\n")

# ===============================================================
# EXAMPLE USAGE:
# ===============================================================

"""
# ===================================================================================================================================================================================
# EXPORT RESULTS TO CSV - SUMMARY RESULTS FOR RCM MODELS AND DISCOUNT RATES SENSITIVITY ANALYSIS
# ===================================================================================================================================================================================
DATAFRAMES_MP8_RCM_DISCOUNT_RATE_RESULTS = {
    'ap2': {
        'private_discount_rate_fixed_low': df_euss_am_mp8_home_ap2_fixed_low,
        'private_discount_rate_fixed_base': df_euss_am_mp8_home_ap2_fixed_base,
        'private_discount_rate_fixed_high': df_euss_am_mp8_home_ap2_fixed_high,
        'private_discount_rate_variable': df_euss_am_mp8_home_ap2_variable
    },
    'easiur': {
        'private_discount_rate_fixed_low': df_euss_am_mp8_home_easiur_fixed_low,
        'private_discount_rate_fixed_base': df_euss_am_mp8_home_easiur_fixed_base,
        'private_discount_rate_fixed_high': df_euss_am_mp8_home_easiur_fixed_high,
        'private_discount_rate_variable': df_euss_am_mp8_home_easiur_variable
    },
    'inmap': {
        'private_discount_rate_fixed_low': df_euss_am_mp8_home_inmap_fixed_low,
        'private_discount_rate_fixed_base': df_euss_am_mp8_home_inmap_fixed_base,
        'private_discount_rate_fixed_high': df_euss_am_mp8_home_inmap_fixed_high,
        'private_discount_rate_variable': df_euss_am_mp8_home_inmap_variable
    }
}

# Process each RCM model
for rcm_model in RCM_MODELS:
    print(f"Exporting SUMMARY RESULTS for {rcm_model.upper()} model...")
    
    # Process each discount method
    for discount_rate_col in PRIVATE_DISCOUNT_RATE_COLS:
        print(f"Private Discounting Method: {discount_rate_col}")
        
        # Get the specific DataFrame for this RCM × discount rate combination
        df_results_export = DATAFRAMES_MP8_RCM_DISCOUNT_RATE_RESULTS[rcm_model][discount_rate_col]
        
        # Export summary results with explicit sensitivity parameters
        export_model_run_output(
            df_results_export=df_results_export,
            results_category='summary',  # Simplified - no parsing needed!
            menu_mp=menu_mp,
            output_folder_path=output_folder_path,
            location_id=location_id,
            results_export_formatted_date=model_run_date_time,
            rcm_model=rcm_model,  # Explicit sensitivity parameter
            discount_rate_col=discount_rate_col  # Explicit sensitivity parameter
        )
"""
