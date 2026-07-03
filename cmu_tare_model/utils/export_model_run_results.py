import os
import pandas as pd
from typing import Union, Optional
import pathlib
from cmu_tare_model.constants import PRIVATE_DISCOUNT_RATE_SHORT_KEYS

def export_model_run_output(
    df_results_export: pd.DataFrame,
    results_category: str,
    menu_mp: Union[int, str],
    output_folder_path: str,
    location_id: str,
    results_export_formatted_date: str,
    discount_rate: Optional[str] = None
) -> None:
    """Export model run results to CSV files with sensitivity tracking.

    This function exports DataFrame results to CSV files organized by result type.
    For retrofit summary results, it uses short discount rate keys for directory naming
    while maintaining backward compatibility with full column names in exported DataFrames.

    Directory Structure:
        baseline_summary/summary_baseline/
            - baseline_results_{location_id}_{date}.csv

        retrofit_mp{menu_mp}_results/summary_mp{menu_mp}_{discount_rate}/
            - mp{menu_mp}_results_{location_id}_{date}.csv
            - Directory uses short key (e.g., 'fixed_base')
            - DataFrame columns use full names (e.g., 'private_discount_rate_fixed_base')

    Args:
        df_results_export: DataFrame containing the results to export.
        results_category: Category of results being exported. Valid options:
            - 'summary_baseline': Baseline summary results
            - 'summary': Retrofit summary results (requires discount_rate)
            - 'damages_climate_IRA', 'damages_climate_noIRA': Climate damages
            - 'fuel_costs_IRA', 'fuel_costs_noIRA': Fuel costs
        menu_mp: Measure package identifier (0 for baseline, 8/9/10 for retrofits).
        output_folder_path: Base directory for all exports.
        location_id: Location identifier for the filename (e.g., 'NYC', 'LA').
        results_export_formatted_date: Date string for the filename (e.g., '2024_01_15').
        discount_rate: Short key for discount rate method (e.g., 'fixed_base', 'variable').
            Required when results_category='summary' and menu_mp != 0.

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
        # Validate that the discount rate is provided
        if discount_rate is None:
            raise ValueError("discount_rate is required for retrofit summary results (results_category='summary')")

        # Validate short key is valid
        if discount_rate not in PRIVATE_DISCOUNT_RATE_SHORT_KEYS:
            raise ValueError(
                f"discount_rate must be one of {list(PRIVATE_DISCOUNT_RATE_SHORT_KEYS)}, "
                f"got '{discount_rate}'"
            )

        # Build directory path using the discount rate key
        directory_path = os.path.join(
            f"retrofit_mp{menu_mp}_results",
            f"summary_mp{menu_mp}_{discount_rate}"
        )
        filename = f"mp{menu_mp}_results_{location_id}_{results_export_formatted_date}.csv"
        print(f"MEASURE PACKAGE {menu_mp} SUMMARY RESULTS:")
        print(f"  Discount Rate: {discount_rate}")
        
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
