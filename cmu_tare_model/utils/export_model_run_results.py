import os
import pandas as pd
from typing import Union, Optional
import pathlib
from cmu_tare_model.constants import PRIVATE_DISCOUNT_RATE_SHORT_KEYS, VERBOSE
from cmu_tare_model.utils.export_tepper_csv import (
    export_tepper_household,
    export_tepper_county,
)

def export_model_run_output(
    df_results_export: pd.DataFrame,
    results_category: str,
    menu_mp: Union[int, str],
    output_folder_path: str,
    location_id: str,
    results_export_formatted_date: str,
    discount_rate: Optional[str] = None,
    county_tables: Optional[dict] = None,
    df_annual_consumption: Optional[pd.DataFrame] = None,
    verbose: bool = VERBOSE
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
            - 'damages_climate_baseline', 'damages_climate_ref2025': Climate
              damages (any name starting with 'damages_')
            - 'fuel_costs_baseline', 'fuel_costs_ref2025': Fuel costs (any
              name starting with 'fuel_costs_')
            - 'tepper_household': One-time Tepper household CSVs, the main
              file and the detailed copy, limited to the study sample
              (delegates to export_tepper_household; uses df_results_export
              as the frame)
            - 'tepper_county': One-time Tepper county CSV (delegates to
              export_tepper_county; requires county_tables)
        menu_mp: Measure package identifier (0 for baseline, nonzero for a measure package).
        output_folder_path: Base directory for all exports.
        location_id: Location identifier for the filename (e.g., 'NYC', 'LA').
        results_export_formatted_date: Date string for the filename (e.g., '2024_01_15').
        discount_rate: Short key for discount rate method (e.g., 'fixed_base', 'variable').
            Required when results_category='summary' and menu_mp != 0.
        county_tables: Mapping with keys 'adoption', 'bill_savings', and
            'demand' holding the three per-MP county result frames. Required
            only when results_category='tepper_county'.
        df_annual_consumption: Supplemental fuel-cost frame for this measure
            package and run, indexed by bldg_id. It holds the per-year
            consumption columns, which are not in the summary frame. Required
            only when results_category='tepper_household'.
        verbose: Whether to print the dividers, the category heading, and the
            file name. When False, one 'Saved: <full path>' line is printed
            per file.

    Raises:
        ValueError: If any required parameter is missing, results_category is invalid,
            or sensitivity parameters are missing when required.
        OSError: If there is an error creating directories or writing the file.
    """
    if verbose:
        print("---" * 35)
    
    # Validate required parameters
    if output_folder_path is None:
        raise ValueError("output_folder_path is required")
    if location_id is None:
        raise ValueError("location_id is required")
    if results_export_formatted_date is None:
        raise ValueError("results_export_formatted_date is required")
    
    # One-time additive Tepper exports. These delegate to dedicated writers in
    # export_tepper_csv and return before the shared single-frame CSV path,
    # because each applies its own list of included columns or multi-table
    # assembly.
    if results_category == 'tepper_household':
        if df_annual_consumption is None:
            raise ValueError(
                "results_category='tepper_household' requires "
                "df_annual_consumption: the supplemental fuel-cost frame for "
                "this measure package and run, which holds the per-year "
                "consumption columns."
            )
        export_tepper_household(
            df_household=df_results_export,
            df_annual_consumption=df_annual_consumption,
            menu_mp=menu_mp,
            output_folder_path=output_folder_path,
            location_id=location_id,
            results_export_formatted_date=results_export_formatted_date,
        )
        if verbose:
            print("---" * 35, "\n")
        return
    if results_category == 'tepper_county':
        if county_tables is None:
            raise ValueError(
                "results_category='tepper_county' requires county_tables with "
                "keys 'adoption', 'bill_savings', and 'demand'"
            )
        export_tepper_county(
            df_adoption=county_tables['adoption'],
            df_bill_savings=county_tables['bill_savings'],
            df_demand=county_tables['demand'],
            menu_mp=menu_mp,
            output_folder_path=output_folder_path,
            location_id=location_id,
            results_export_formatted_date=results_export_formatted_date,
        )
        if verbose:
            print("---" * 35, "\n")
        return

    # Standardize menu_mp to string
    menu_mp = str(menu_mp)
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_results_export_copy = df_results_export.copy()
    
    # Build directory path and filename based on results_category
    if results_category == 'summary_baseline':
        # Baseline summary results
        directory_path = os.path.join("baseline_summary", "summary_baseline")
        filename = f"baseline_results_{location_id}_{results_export_formatted_date}.csv"
        if verbose:
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
        if verbose:
            print(f"MEASURE PACKAGE {menu_mp} SUMMARY RESULTS:")
            print(f"  Discount Rate: {discount_rate}")
        
    elif results_category.startswith('damages_'):
        # Climate damages (baseline or a measure package)
        directory_path = os.path.join("supplemental_data_damages", results_category)
        filename = f"mp{menu_mp}_{results_category}_{location_id}_{results_export_formatted_date}.csv"
        if verbose:
            print(f"SUPPLEMENTAL DAMAGES: {results_category}")
        
    elif results_category.startswith('fuel_costs_'):
        # Fuel costs (baseline or a measure package)
        directory_path = os.path.join("supplemental_data_fuelCosts", results_category)
        filename = f"mp{menu_mp}_{results_category}_{location_id}_{results_export_formatted_date}.csv"
        if verbose:
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
        # One line per file when quiet; the file name and path when verbose.
        if verbose:
            print(f"Saved to: {filename}")
            print(f"Full path: {full_filepath}")
        else:
            print(f"Saved: {full_filepath}")
    except Exception as e:
        raise OSError(f"Error exporting data to {full_filepath}: {str(e)}")

    if verbose:
        print("---" * 35, "\n")
