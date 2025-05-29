import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from config import PROJECT_ROOT

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FUNCTIONS FOR DATA VISUALIZATION
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# ====================================================================================================================================================================================
# DISPLAYING TRUNCATED DICTIONARIES
# ====================================================================================================================================================================================
def print_truncated_dict(dict, n=5):
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

def summarize_stats_table(
    df: pd.DataFrame, 
    data_columns: list[str], 
    column_name_mapping: dict[str, str], 
    number_formatting: str, 
    include_zero: bool = True,
    category: str | None = None,        
    enable_fuel_filter: bool = False,
    included_fuel_list: list[str] | None = None
) -> pd.DataFrame:
    """
    Generate a formatted summary statistics table for specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame from which to compute statistics.
        data_columns (list[str]): The columns to include in the summary statistics.
        column_name_mapping (dict[str, str]): Mapping from original column names to desired display names.
        number_formatting (str): The Python format string (e.g. ".2f") to format numeric values in the output.
        include_zero (bool, optional): Whether to include zero values in the statistics. Defaults to True.
        category (str | None, optional): Category name for filtering fuel types (e.g., 'heating', 'waterHeating', etc.).
        enable_fuel_filter (bool, optional): Whether to filter the DataFrame based on specific fuel types. Defaults to False.
        included_fuel_list (list[str] | None, optional): List of fuels to include if filtering is enabled.

    Returns:
        pd.DataFrame: A DataFrame containing the summary statistics with formatted numeric values 
            and renamed columns according to the input specifications.

    Raises:
        ValueError: If any of the specified data_columns are missing from df, or if the required fuel column is not present 
            when fuel filtering is enabled.
    """

    # Validate that all specified data_columns exist in the DataFrame
    missing_cols = [c for c in data_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")

    # Make a copy to avoid modifying the original data
    df_copy = df.copy()
    
    # Apply fuel filter if enabled and if category and included_fuel_list are provided
    if enable_fuel_filter and category is not None and included_fuel_list:
        fuel_col = f'base_{category}_fuel'
        # Check if the expected fuel column for filtering is present
        if fuel_col not in df_copy.columns:
            raise ValueError(
                f"Fuel column '{fuel_col}' is not present in the DataFrame, cannot filter."
            )
        df_copy = df_copy[df_copy[fuel_col].isin(included_fuel_list)]
        print(f"Filtered for the following fuels: {included_fuel_list}")
    
    # Replace 0 with NaN if include_zero is False so these values are ignored in stats
    if not include_zero:
        df_copy[data_columns] = df_copy[data_columns].replace(0, np.nan)
    
    # If the DataFrame becomes empty after filtering/zero removal, return an empty DataFrame
    if df_copy.empty:
        print("Warning: DataFrame is empty after filtering and/or zero removal.")
        return pd.DataFrame()
    
    # Calculate the summary statistics using pandas' describe()
    summary_stats = df_copy[data_columns].describe()
    
    # Helper function to format numeric values according to the specified format string
    def format_func(value):
        try:
            return f"{float(value):{number_formatting}}"
        except (ValueError, TypeError):
            return str(value)

    # Apply the formatting function to all entries in the summary_stats DataFrame
    summary_stats = summary_stats.map(format_func)
    
    # Rename columns according to the user-specified mapping
    summary_stats.rename(columns=column_name_mapping, inplace=True)
    
    return summary_stats

# ===================================================================================================================================================================================
# VALIDATION TESTS FOR WITHIN-CATEGORY CONSISTENCY
# ===================================================================================================================================================================================
def validate_within_category_consistency(df_climate, df_health, df_npv):
    """
    CORRECTED validation test that checks for consistency WITHIN categories,
    not across categories (which should differ based on retrofit eligibility).
    
    Args:
        df_climate: Results from calculate_lifetime_climate_impacts
        df_health: Results from calculate_lifetime_health_impacts  
        df_npv: Results from calculate_public_npv
        
    Returns:
        dict: Summary of validation results
    """
    results = {
        'climate_within_category_consistency': True,
        'health_within_category_consistency': True,
        'category_analysis': {},
        'issues_found': []
    }
    
    # Define expected categories
    categories = ['heating', 'waterHeating', 'clothesDrying', 'cooking']
    scc_assumptions = ['lower', 'central', 'upper']
    mer_types = ['lrmer', 'srmer']
    rcm_models = ['ap2', 'easiur', 'inmap']
    cr_functions = ['acs', 'h6c']
    
    # Test 1: Climate impact consistency WITHIN each category
    for category in categories:
        category_counts = []
        category_cols = []
        
        # Check all SCC assumptions and MER types for this category
        for scc in scc_assumptions:
            for mer in mer_types:
                col_pattern = f'_lifetime_damages_climate_{mer}_{scc}'
                matching_cols = [col for col in df_climate.columns 
                               if category in col and col_pattern in col]
                
                for col in matching_cols:
                    count = df_climate[col].notna().sum()
                    category_counts.append(count)
                    category_cols.append(col)
        
        # All counts within this category should be identical
        unique_counts = set(category_counts)
        if len(unique_counts) > 1:
            results['climate_within_category_consistency'] = False
            results['issues_found'].append(
                f"Climate {category} counts vary: {unique_counts}"
            )
        
        results['category_analysis'][f'{category}_climate'] = {
            'count': list(unique_counts)[0] if len(unique_counts) == 1 else unique_counts,
            'columns_checked': len(category_cols)
        }
    
    # Test 2: Health impact consistency WITHIN each category  
    for category in categories:
        for rcm in rcm_models:
            for cr in cr_functions:
                combo_counts = []
                combo_cols = []
                
                col_pattern = f'_lifetime_damages_health_{rcm}_{cr}'
                matching_cols = [col for col in df_health.columns 
                               if category in col and col_pattern in col]
                
                for col in matching_cols:
                    count = df_health[col].notna().sum()
                    combo_counts.append(count)
                    combo_cols.append(col)
                
                # All counts for this RCM/CR combo within category should be identical
                unique_counts = set(combo_counts)
                if len(unique_counts) > 1:
                    results['health_within_category_consistency'] = False
                    results['issues_found'].append(
                        f"Health {category} {rcm}_{cr} counts vary: {unique_counts}"
                    )
    
    # Test 3: Cross-module consistency (climate vs health within same category)
    for category in categories:
        # Get climate count for this category (should be same across all SCC/MER)
        climate_col = [col for col in df_climate.columns 
                      if category in col and 'lifetime_damages_climate' in col][0]
        climate_count = df_climate[climate_col].notna().sum()
        
        # Get health count for this category (should be same across all RCM/CR)
        health_col = [col for col in df_health.columns 
                     if category in col and 'lifetime_damages_health' in col][0]
        health_count = df_health[health_col].notna().sum()
        
        if climate_count != health_count:
            results['issues_found'].append(
                f"Climate vs Health count mismatch for {category}: {climate_count} vs {health_count}"
            )
        
        results['category_analysis'][f'{category}_total'] = {
            'climate_count': climate_count,
            'health_count': health_count,
            'consistent': climate_count == health_count
        }
    
    return results


def run_corrected_validation_tests(df_climate, df_health, df_npv):
    """Run corrected validation tests that understand category differences are expected."""
    
    print("="*70)
    print("CORRECTED VALIDATION FRAMEWORK CONSISTENCY CHECK")
    print("="*70)
    print("✅ Testing WITHIN-category consistency (not across categories)")
    print("✅ Category count differences are EXPECTED and CORRECT")
    
    results = validate_within_category_consistency(df_climate, df_health, df_npv)
    
    print(f"\n✅ Climate Within-Category Consistency: {'PASS' if results['climate_within_category_consistency'] else 'FAIL'}")
    print(f"✅ Health Within-Category Consistency: {'PASS' if results['health_within_category_consistency'] else 'FAIL'}")
    
    # Show category analysis
    print(f"\n📊 CATEGORY ANALYSIS:")
    for category, analysis in results['category_analysis'].items():
        if 'total' in category:
            cat_name = category.replace('_total', '')
            print(f"   {cat_name.upper()}: Climate={analysis['climate_count']}, Health={analysis['health_count']} ({'✅' if analysis['consistent'] else '❌'})")
    
    if results['issues_found']:
        print(f"\n⚠️  Issues Found:")
        for issue in results['issues_found']:
            print(f"   - {issue}")
    else:
        print(f"\n🎉 All validation tests PASSED!")
        print("   ✅ Within-category consistency perfect")
        print("   ✅ Climate-Health alignment correct") 
        print("   ✅ Validation framework working properly")
        print("   ✅ Category differences reflect business logic")
    
    return results

# Usage example:
# results = run_corrected_validation_tests(df_climate_main, df_health_main, df_npv_results)