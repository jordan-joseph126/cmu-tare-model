import pandas as pd
import numpy as np

# # Import the original data from the backup module
# from cmu_tare_model.private_impact.data_processing.process_fuel_price_data_OLD import (
#     df_fuelPrices_perkWh_preIRA as df_preIRA_ORIGINAL,
#     df_fuelPrices_perkWh_iraRef as df_iraRef_ORIGINAL,
#     lookup_fuel_prices_preIRA as lookup_preIRA_ORIGINAL,
#     lookup_fuel_prices_iraRef as lookup_iraRef_ORIGINAL,
# )

# # Import the new data from the updated module
# from cmu_tare_model.private_impact.data_processing.create_lookup_fuel_prices import (
#     df_fuel_prices_preIRA as df_preIRA_NEW,
#     df_fuel_prices_iraRef as df_iraRef_NEW,
#     lookup_fuel_prices_preIRA as lookup_preIRA_NEW,
#     lookup_fuel_prices_iraRef as lookup_iraRef_NEW,
# )

def safe_equal(val1, val2):
    """
    Safely compares two values, ensuring that the result is a single Boolean.
    
    Handles:
      - Numeric values (using np.isclose)
      - Pandas objects (Series, DataFrame, Index) using their .equals() method
      - Numpy arrays using np.array_equal
      - Fallback comparisons wrapped in try/except to catch ambiguous truth evaluations.
    """
    # 1. Numeric types: compare using np.isclose.
    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
        return bool(np.isclose(val1, val2, rtol=1e-5, atol=1e-8))
    
    # 2. If either value is a pandas object with a dedicated equals method:
    if hasattr(val1, "equals") or hasattr(val2, "equals"):
        # If types differ, they cannot be equal.
        if type(val1) != type(val2):
            return False
        try:
            return val1.equals(val2)
        except Exception:
            # Fall back to the next block if .equals() fails for any reason.
            pass
    
    # 3. If either value is a numpy array, ensure both are arrays and compare.
    if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
        if not (isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray)):
            return False
        return np.array_equal(val1, val2)
    
    # 4. Fallback: attempt a direct equality comparison.
    try:
        eq = val1 == val2
    except ValueError:
        try:
            # Convert both values to numpy arrays and try again.
            eq = np.asarray(val1) == np.asarray(val2)
        except Exception:
            return False

    # If eq is a boolean, return it directly.
    if isinstance(eq, bool):
        return eq
    
    # If eq is an array-like (e.g. a Series or numpy array), aggregate its truth value.
    try:
        eq_array = np.asarray(eq)
        return bool(np.all(eq_array))
    except Exception:
        return False

# Function to compare dataframes
def compare_dataframes(df1, df2, name1, name2):
    print(f"\n=== Comparing {name1} vs {name2} ===")
    
    # Check if dataframes have the same shape
    shape_match = df1.shape == df2.shape
    print(f"Same shape: {shape_match} ({df1.shape} vs {df2.shape})")
    
    # Check if dataframes have the same columns (order is ignored)
    same_columns = set(df1.columns) == set(df2.columns)
    print(f"Same columns: {same_columns}")
    
    if not same_columns:
        only_in_df1 = set(df1.columns) - set(df2.columns)
        only_in_df2 = set(df2.columns) - set(df1.columns)
        if only_in_df1:
            print(f"Columns only in {name1}: {only_in_df1}")
        if only_in_df2:
            print(f"Columns only in {name2}: {only_in_df2}")
    
    # Check if dataframes have the same index
    same_index = df1.index.equals(df2.index)
    print(f"Same index: {same_index}")
    
    if not same_index:
        only_in_df1_idx = set(df1.index) - set(df2.index)
        only_in_df2_idx = set(df2.index) - set(df1.index)
        if only_in_df1_idx:
            print(f"Indices only in {name1}: {only_in_df1_idx}")
        if only_in_df2_idx:
            print(f"Indices only in {name2}: {only_in_df2_idx}")
    
    # For comparing values, use only columns present in both dataframes
    common_columns = list(set(df1.columns) & set(df2.columns))
    if common_columns:
        # Ensure indices match before comparing values
        if same_index and same_columns:
            # Check for exactly equal dataframes
            exact_match = df1.equals(df2)
            print(f"Exact match: {exact_match}")
            
            if not exact_match:
                # Compare numeric values with tolerance for floating point differences
                are_close = np.isclose(
                    df1.select_dtypes(include=['number']).fillna(0),
                    df2.select_dtypes(include=['number']).fillna(0),
                    rtol=1e-5, atol=1e-8, equal_nan=True
                ).all().all()
                
                print(f"Values are close (allowing for float precision): {are_close}")
                
                # If not close, find and display differences for numeric columns
                numeric_cols = df1.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    if col in df2.columns:
                        diff = ~np.isclose(
                            df1[col].fillna(0),
                            df2[col].fillna(0),
                            rtol=1e-5, atol=1e-8, equal_nan=True
                        )
                        if diff.any():
                            mismatch_indices = diff[diff].index
                            print(f"\nMismatches in column '{col}':")
                            for idx in mismatch_indices[:5]:  # Show first 5 mismatches
                                print(f"  Index {idx}: {name1}={df1.loc[idx, col]} vs {name2}={df2.loc[idx, col]}")
                            if len(mismatch_indices) > 5:
                                print(f"  ... and {len(mismatch_indices) - 5} more mismatches")
                
                # For non-numeric columns
                non_numeric_cols = set(common_columns) - set(numeric_cols)
                for col in non_numeric_cols:
                    mismatches = df1[col] != df2[col]
                    if mismatches.any():
                        mismatch_indices = mismatches[mismatches].index
                        print(f"\nMismatches in non-numeric column '{col}':")
                        for idx in mismatch_indices[:5]:  # Show first 5 mismatches
                            print(f"  Index {idx}: {name1}={df1.loc[idx, col]} vs {name2}={df2.loc[idx, col]}")
                        if len(mismatch_indices) > 5:
                            print(f"  ... and {len(mismatch_indices) - 5} more mismatches")
        else:
            print("Cannot compare values directly due to different indices or columns")
    else:
        print("No common columns to compare values")

# Function to compare lookup dictionaries
def compare_lookups(dict1, dict2, name1, name2):
    print(f"\n=== Comparing {name1} vs {name2} ===")
    
    # Check if dictionaries have the same keys
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    same_keys = keys1 == keys2
    print(f"Same keys: {same_keys}")
    
    if not same_keys:
        only_in_dict1 = keys1 - keys2
        only_in_dict2 = keys2 - keys1
        if only_in_dict1:
            print(f"Keys only in {name1}: {only_in_dict1}")
        if only_in_dict2:
            print(f"Keys only in {name2}: {only_in_dict2}")
    
    # Check if values for common keys are the same
    common_keys = keys1 & keys2
    all_values_match = True
    mismatched_keys = []
    
    for key in common_keys:
        # If the value is a nested dictionary, compare each nested key-value pair.
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            nested_keys1 = set(dict1[key].keys())
            nested_keys2 = set(dict2[key].keys())
            
            if nested_keys1 != nested_keys2:
                all_values_match = False
                mismatched_keys.append((key, "different nested keys"))
                continue
                
            for nested_key in nested_keys1:
                val1 = dict1[key][nested_key]
                val2 = dict2[key][nested_key]
                
                # Use safe_equal to compare the values.
                if not safe_equal(val1, val2):
                    all_values_match = False
                    mismatched_keys.append((key, nested_key, val1, val2))
        
        # For simple key-value pairs, also use safe_equal.
        elif not safe_equal(dict1[key], dict2[key]):
            all_values_match = False
            mismatched_keys.append((key, dict1[key], dict2[key]))
    
    print(f"All values match for common keys: {all_values_match}")
    
    if not all_values_match:
        print("First few mismatches:")
        for i, mismatch in enumerate(mismatched_keys[:5]):
            print(f"  {mismatch}")
        if len(mismatched_keys) > 5:
            print(f"  ... and {len(mismatched_keys) - 5} more mismatches")

# # Main execution block
# def main():
#     print("Starting data comparison...")
    
#     # Compare dataframes
#     compare_dataframes(df_preIRA_ORIGINAL, df_preIRA_NEW, "df_preIRA_ORIGINAL", "df_preIRA_NEW")
#     compare_dataframes(df_iraRef_ORIGINAL, df_iraRef_NEW, "df_iraRef_ORIGINAL", "df_iraRef_NEW")
    
#     # Compare lookup dictionaries
#     compare_lookups(lookup_preIRA_ORIGINAL, lookup_preIRA_NEW, "lookup_preIRA_ORIGINAL", "lookup_preIRA_NEW")
#     compare_lookups(lookup_iraRef_ORIGINAL, lookup_iraRef_NEW, "lookup_iraRef_ORIGINAL", "lookup_iraRef_NEW")
    
#     print("\nComparison complete!")

# if __name__ == "__main__":
#     main()
