# ===================================================================================================================================================================================
# COMPARISON: Find and validate results by timestamp
# ===================================================================================================================================================================================

import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set


# Columns that naturally vary due to random sampling and should be excluded from comparison
STOCHASTIC_PATTERNS = {
    'capital_cost', 'capitalcost', 'total_capitalcost', 'net_capitalcost',
    'installation_cost', 'installationcost',
    'replacement_cost', 'replacementcost',
    'rebate', 'rebate_amount',
    'private_npv', 'total_npv', 'npv', 'benefit',
    'household_income', 'percent_ami', 'private_discount_rate_variable'
}

def identify_stochastic_columns(df: pd.DataFrame) -> Set[str]:
    stochastic_cols = set()
    for col in df.columns:
        col_lower = col.strip().lower()
        if any(pat in col_lower for pat in STOCHASTIC_PATTERNS):
            stochastic_cols.add(col)
    return stochastic_cols


def find_files_by_timestamp(output_folder_path: str, timestamp: str, location_id: str) -> Dict[str, Path]:
    """
    Find all CSV files matching a timestamp in output_results folder and subfolders.
    
    Args:
        output_folder_path: Path to output_results directory
        timestamp: Timestamp string to search for (e.g., '2026-02-04_17-13')
        location_id: Location identifier (e.g., 'PA')
    
    Returns:
        Dictionary mapping file paths (relative) to Path objects
    """
    base_path = Path(output_folder_path)
    pattern = f"*_{location_id}_{timestamp}.csv"
    
    files = {}
    for csv_file in base_path.rglob(pattern):
        relative_path = csv_file.relative_to(base_path)
        files[str(relative_path)] = csv_file
    
    return files


def print_files_by_timestamp(output_folder_path: str, timestamp: str, location_id: str) -> None:
    """
    Print all files found for a given timestamp and location.
    
    Args:
        output_folder_path: Path to output_results directory
        timestamp: Timestamp string to search for
        location_id: Location identifier
    """
    files = find_files_by_timestamp(output_folder_path, timestamp, location_id)
    
    print(f"\n{'='*80}")
    print(f"FILES FOUND FOR TIMESTAMP: {timestamp}")
    print(f"Location: {location_id}")
    print(f"{'='*80}\n")
    
    if not files:
        print(f"No files found matching pattern: *_{location_id}_{timestamp}.csv\n")
        return
    
    print(f"Total files found: {len(files)}\n")
    
    # Group files by directory for cleaner output
    files_by_dir = {}
    for rel_path in sorted(files.keys()):
        directory = str(Path(rel_path).parent)
        if directory not in files_by_dir:
            files_by_dir[directory] = []
        files_by_dir[directory].append(Path(rel_path).name)
    
    for directory in sorted(files_by_dir.keys()):
        print(f"📁 {directory}/")
        for filename in sorted(files_by_dir[directory]):
            print(f"   📄 {filename}")
        print()


def compare_two_timestamps(output_folder_path: str, timestamp_v1: str, timestamp_v2: str, 
                           location_id: str, tolerance: float = 1e-6, 
                           exclude_stochastic: bool = True) -> Dict:
    """
    Compare results between two timestamps for matching files.
    
    Capital costs, installation costs, and NPV columns are excluded by default 
    since they vary due to random sampling.
    
    Args:
        output_folder_path: Path to output_results directory
        timestamp_v1: Earlier timestamp
        timestamp_v2: Later timestamp
        location_id: Location identifier
        tolerance: Numerical comparison tolerance (default 1e-6)
        exclude_stochastic: If True, exclude stochastic columns (default True)
    
    Returns:
        Dictionary with comparison results
    """
    files_v1 = find_files_by_timestamp(output_folder_path, timestamp_v1, location_id)
    files_v2 = find_files_by_timestamp(output_folder_path, timestamp_v2, location_id)
    
    print(f"\n{'='*80}")
    print(f"COMPARING TIMESTAMPS")
    print(f"{'='*80}")
    print(f"  v1 (Earlier): {timestamp_v1}")
    print(f"  v2 (Later):   {timestamp_v2}")
    print(f"  Location:     {location_id}")
    if exclude_stochastic:
        print(f"  Mode:         Comparing deterministic columns only")
        print(f"                (excluding stochastic: capital/installation costs, NPV)")
    print(f"\nFound {len(files_v1)} files for v1")
    print(f"Found {len(files_v2)} files for v2\n")
    
    results = {
        'timestamp_v1': timestamp_v1,
        'timestamp_v2': timestamp_v2,
        'location_id': location_id,
        'exclude_stochastic': exclude_stochastic,
        'comparisons': {}
    }
    
    # Compare only files that exist in both timestamps
    common_patterns = set()
    for path_v1 in files_v1.keys():
        # Extract common pattern by replacing timestamp
        pattern = path_v1.replace(timestamp_v1, '{TIMESTAMP}')
        common_patterns.add(pattern)
    
    for pattern in sorted(common_patterns):
        path_v1_pattern = pattern.replace('{TIMESTAMP}', timestamp_v1)
        path_v2_pattern = pattern.replace('{TIMESTAMP}', timestamp_v2)
        
        if path_v1_pattern not in files_v1 or path_v2_pattern not in files_v2:
            print(f"⚠️  Skipping (not in both): {path_v1_pattern}")
            continue
        
        file_desc = path_v1_pattern.replace(f"_{timestamp_v1}.csv", "")
        print(f"\nComparing: {file_desc}")
        
        try:
            df_v1 = pd.read_csv(files_v1[path_v1_pattern], low_memory=False)
            df_v2 = pd.read_csv(files_v2[path_v2_pattern], low_memory=False)
        except Exception as e:
            print(f"  ✗ Error reading files: {e}")
            results['comparisons'][file_desc] = {'status': 'read_error', 'error': str(e)}
            continue
        
        # Check shape
        if df_v1.shape != df_v2.shape:
            print(f"  ✗ Shape mismatch: v1={df_v1.shape}, v2={df_v2.shape}")
            results['comparisons'][file_desc] = {
                'status': 'shape_mismatch',
                'shape_v1': df_v1.shape,
                'shape_v2': df_v2.shape
            }
            continue
        
        print(f"  ✓ Shape matches: {df_v1.shape}")
        
        # Check columns
        if not df_v1.columns.equals(df_v2.columns):
            print(f"  ✗ Column mismatch")
            results['comparisons'][file_desc] = {'status': 'column_mismatch'}
            continue
        
        print(f"  ✓ Columns match: {len(df_v1.columns)} columns")
        
        # Identify stochastic columns to exclude
        stochastic_cols = identify_stochastic_columns(df_v1) if exclude_stochastic else set()
        
        # Get all numerical columns
        numerical_cols = df_v1.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Separate stochastic from deterministic columns
        deterministic_cols = [col for col in numerical_cols if col not in stochastic_cols]
        
        differences = {}
        excluded_cols = {}
        has_diffs = False
        
        # Only compare deterministic columns
        for col in deterministic_cols:
            max_diff = (df_v1[col] - df_v2[col]).abs().max()
            differences[col] = float(max_diff)
            
            if max_diff >= tolerance:
                has_diffs = True
                idx_max = (df_v1[col] - df_v2[col]).abs().idxmax()
                print(f"  ✗ {col}: max_diff={max_diff:.2e} (row {idx_max})")
        
        # Track excluded stochastic columns
        for col in stochastic_cols:
            max_val_v1 = df_v1[col].abs().max()
            max_val_v2 = df_v2[col].abs().max()
            excluded_cols[col] = {
                'max_v1': float(max_val_v1),
                'max_v2': float(max_val_v2)
            }
        
        # Report excluded stochastic columns (compact format)
        if excluded_cols:
            print(f"  ⊘ Excluded {len(excluded_cols)} stochastic columns (expected to vary)")
        
        if not has_diffs and len(deterministic_cols) > 0:
            print(f"  ✓ All {len(deterministic_cols)} deterministic columns match (tolerance={tolerance})")
            results['comparisons'][file_desc] = {
                'status': 'passed',
                'shape': df_v1.shape,
                'total_columns': len(df_v1.columns),
                'compared_columns': len(deterministic_cols),
                'excluded_stochastic': len(excluded_cols),
                'max_differences': differences
            }
        elif len(deterministic_cols) == 0:
            print(f"  ⊘ All numerical columns are stochastic")
            results['comparisons'][file_desc] = {
                'status': 'all_stochastic',
                'shape': df_v1.shape,
                'total_columns': len(df_v1.columns),
                'compared_columns': 0,
                'excluded_stochastic': len(excluded_cols)
            }
        else:
            results['comparisons'][file_desc] = {
                'status': 'numerical_diff',
                'shape': df_v1.shape,
                'total_columns': len(df_v1.columns),
                'compared_columns': len(deterministic_cols),
                'excluded_stochastic': len(excluded_cols),
                'differences': differences
            }
    
    return results


def print_comparison_summary(results: Dict) -> None:
    """Print a summary report of comparison results."""
    comparisons = results['comparisons']
    
    passed = sum(1 for r in comparisons.values() if r.get('status') == 'passed')
    failed = sum(1 for r in comparisons.values() if r.get('status') in ['shape_mismatch', 'column_mismatch', 'numerical_diff'])
    all_stochastic = sum(1 for r in comparisons.values() if r.get('status') == 'all_stochastic')
    errors = sum(1 for r in comparisons.values() if r.get('status') == 'read_error')
    
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"  v1 → v2: {results['timestamp_v1']} → {results['timestamp_v2']}")
    print(f"  Location: {results['location_id']}\n")
    
    if results.get('exclude_stochastic'):
        print(f"  Note: Stochastic columns excluded from comparison\n")
    
    print(f"  ✓ Passed:           {passed}")
    print(f"  ✗ Failed:           {failed}")
    print(f"  ⊘ All Stochastic:   {all_stochastic}")
    print(f"  ⚠️  Errors:         {errors}")
    print(f"  Total:             {len(comparisons)}\n")
    
    if failed > 0:
        print("Files with deterministic differences:")
        for name, result in comparisons.items():
            if result.get('status') == 'numerical_diff':
                print(f"  - {name}")
    
    print()


# # ===== USAGE EXAMPLES =====

# # Example 1: Find all files with a specific timestamp
# print_files_by_timestamp(
#     output_folder_path=output_folder_path,
#     timestamp=model_run_date_time,
#     location_id=location_id
# )

# # Example 2: Compare two timestamps (stochastic columns excluded by default)
# comparison_results = compare_two_timestamps(
#     output_folder_path=output_folder_path,
#     timestamp_v1='2026-02-04_17-13',
#     timestamp_v2='2026-02-04_15-39',
#     location_id=location_id,
#     tolerance=1e-6,
#     exclude_stochastic=True
# )

# print_comparison_summary(comparison_results)