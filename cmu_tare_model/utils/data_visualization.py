import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Optional

# from config import PROJECT_ROOT

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FUNCTIONS FOR DATA VISUALIZATION
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# ====================================================================================================================================================================================
# DISPLAYING TRUNCATED DICTIONARIES
# ====================================================================================================================================================================================
def print_truncated_dict(
        dict: dict,
        n: int = 5):
    """
    Mimics Jupyter's truncated display for dictionaries.
    
    If the dictionary contains more than 2*n items, it prints the first n key–value
    pairs, then an ellipsis ('...'), followed by the last n key–value pairs.
    Otherwise, it prints the full dictionary.
    
    Parameters:
        dict (dict): The dictionary to print.
        n (int): The number of items to show from the beginning and end.
    """
    items = list(dict.items())
    total_items = len(items)
    
    if total_items <= 2 * n:
        print(dict)
    else:
        # Start of the dict representation
        print("{")
        # Print the first n items with some indentation for readability
        for key, value in items[:n]:
            print("  {}: {},".format(repr(key), repr(value)))
        # Print an ellipsis to indicate omitted items
        print("  ...")
        # Print the last n items
        for key, value in items[-n:]:
            print("  {}: {},".format(repr(key), repr(value)))
        # End of the dict representation
        print("}")

# # Build a sample dictionary with 20 key–value pairs
# sample_dict = {f'key{i}': i for i in range(1, 21)}
# print_truncated_dict(sample_dict, n=5)

# ===================================================================================================================================================================================
# FORMAT DATA USING .DESCRIBE() METHODS
# ===================================================================================================================================================================================

def print_summary_stats(
    df: Optional[pd.DataFrame] = None,
    dataframes: Optional[List[pd.DataFrame]] = None,
    dataframe_indices: Optional[List[int]] = None,
    column_names: List[str] = None,
    subplot_titles: Optional[List[str]] = None,
    percentiles: Optional[List[float]] = None,
    fuel_column: Optional[str] = None,
    by_fuel_type: bool = False
):
    """
    Print summary statistics for each specified column across single or multiple DataFrames,
    with optional breakdown by fuel type.
    
    Uses pandas .describe() to provide descriptive statistics (count, mean, std, min, 
    quartiles, max). Useful for understanding data distributions before creating histograms.
    
    Args:
        df: Single DataFrame for backward compatibility
        dataframes: List of DataFrames for multi-DataFrame analysis
        dataframe_indices: List of indices mapping columns to DataFrames
        column_names: List of column names to analyze
        subplot_titles: Optional list of display titles for each column
        percentiles: Optional list of percentiles to include (default: [25, 50, 75])
        fuel_column: Column name containing fuel type information (required if by_fuel_type=True)
        by_fuel_type: Whether to show statistics broken down by fuel type
    """
    # Input validation
    if df is not None and dataframes is not None:
        print("Error: Specify either 'df' or 'dataframes', not both")
        return
    if df is None and dataframes is None:
        print("Error: Must specify either 'df' or 'dataframes'")
        return
    if column_names is None or not column_names:
        print("Error: column_names must be provided and non-empty")
        return
    if by_fuel_type and fuel_column is None:
        print("Error: fuel_column must be specified when by_fuel_type=True")
        return
    
    # Set default percentiles if not provided
    if percentiles is None:
        percentiles = [25, 50, 75]
    
    # Set up DataFrame handling
    if df is not None:
        # Single DataFrame mode (original behavior)
        df_list = [df]
        df_indices = [0] * len(column_names)
    else:
        # Multiple DataFrame mode
        df_list = dataframes
        if dataframe_indices is None:
            df_indices = [i % len(dataframes) for i in range(len(column_names))]
        else:
            if len(dataframe_indices) != len(column_names):
                print("Error: dataframe_indices length must match number of columns")
                return
            df_indices = dataframe_indices
    
    # Get the known fuel types for breakdown (matching your COLOR_MAP_FUEL)
    known_fuel_types = ['Electricity', 'Natural Gas', 'Propane', 'Fuel Oil']
    
    for i, col in enumerate(column_names):
        # Get the appropriate DataFrame for this column
        current_df = df_list[df_indices[i]]
        
        # Check if column exists in the current DataFrame
        if col not in current_df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame at index {df_indices[i]}. Available columns: {list(current_df.columns)[:10]} ...")
            continue
            
        # Check if fuel column exists when needed
        if by_fuel_type and fuel_column not in current_df.columns:
            print(f"Warning: Fuel column '{fuel_column}' not found in DataFrame at index {df_indices[i]}. Available columns: {list(current_df.columns)[:10]} ...")
            continue
        
        # Get display name
        display_name = subplot_titles[i] if subplot_titles and i < len(subplot_titles) else col
        
        print(f"\n===== {display_name} =====")
        
        # Calculate and display overall statistics using pandas describe()
        overall_stats = current_df[col].describe(percentiles=[p/100 for p in percentiles])
        
        print("All Data:")
        print(overall_stats)
        
        # Calculate statistics by fuel type if requested
        if by_fuel_type:
            print(f"\nBy Fuel Type:")
            
            for fuel in known_fuel_types:
                # Filter by fuel type
                fuel_data = current_df[current_df[fuel_column] == fuel]
                
                # Skip if no data for this fuel type
                if len(fuel_data) == 0:
                    continue
                
                # Calculate and display statistics for this fuel type using pandas describe()
                fuel_stats = fuel_data[col].describe(percentiles=[p/100 for p in percentiles])
                
                print(f"\n  {fuel}:")
                # Add indentation to each line of the describe output
                for line in str(fuel_stats).split('\n'):
                    if line.strip():  # Skip empty lines
                        print(f"  {line}")
        
        # Report missing data information
        nan_count = current_df[col].isna().sum()
        total_rows = len(current_df)
        nan_percentage = (nan_count / total_rows * 100) if total_rows > 0 else 0
        
        if nan_count > 0:
            print(f"\nMissing Values: {nan_count:,} ({nan_percentage:.1f}%)")
        else:
            print(f"\nNo missing values.")
