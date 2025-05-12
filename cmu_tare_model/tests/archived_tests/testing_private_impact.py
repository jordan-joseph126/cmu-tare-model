import pandas as pd
import numpy as np
import sys
import os
from copy import deepcopy

# Import both versions of the function
from cmu_tare_model.private_impact import calculate_lifetime_private_impact as current
from archived_files import calculate_lifetime_private_impact_BACKUP_nodoc as backup
from cmu_tare_model.constants import EQUIPMENT_SPECS
from cmu_tare_model.utils.modeling_params import define_scenario_params

# Create synthetic test data that matches the expected structure
def create_test_data(menu_mp, policy_scenario):
    """
    Create test data with the specified menu_mp value.
    
    Args:
        menu_mp (int): Measure package identifier to use in column names.
        
    Returns:
        tuple: (df, df_fuel_costs) containing the test data
    """
    # Create a small synthetic DataFrame with the necessary columns
    np.random.seed(42)  # For reproducibility
    
    # Sample size
    n = 10

    # Create base DataFrame
    df = pd.DataFrame({
        'bldg_id': range(1, n + 1),
    })
    
    if menu_mp == 8:
        df['mp8_enclosure_upgradeCost'] = 0
        df['weatherization_rebate_amount'] = 0
    else:
        df[f'mp{menu_mp}_enclosure_upgradeCost'] = np.random.uniform(1000, 5000, n)
        df['weatherization_rebate_amount'] = np.random.uniform(100, 1000, n)
    
    scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp=menu_mp,policy_scenario=policy_scenario)
    
    # Prepare data dictionaries to use concat instead of insert
    capital_cost_data = {}
    fuel_cost_data = {}

    base_year = 2024  # Base year for calculations

    # Loop over each equipment category and its lifetime
    for category, lifetime in EQUIPMENT_SPECS.items():
        # =================================================================================================
        # Capital Costs
        # =================================================================================================
        if category == 'heating':
            # Installation premiums (only for heating)
            capital_cost_data[f'mp{menu_mp}_heating_installation_premium'] = np.random.uniform(100, 500, n)
        
        # Installation costs
        capital_cost_data[f'mp{menu_mp}_{category}_installationCost'] = np.random.uniform(2000, 8000, n)
        
        # Replacement costs
        capital_cost_data[f'mp{menu_mp}_{category}_replacementCost'] = np.random.uniform(1000, 4000, n)
                
        # Rebate amounts
        capital_cost_data[f'mp{menu_mp}_{category}_rebate_amount'] = np.random.uniform(500, 2000, n)
        
        # =================================================================================================
        # Generate fuel cost savings columns for each year, category, and scenario
        # =================================================================================================
        for year in range(1, lifetime + 1):
            year_label = year + (base_year - 1)  # Match current implementation's formula
            mp_savings_col = f'{scenario_prefix}{year_label}_{category}_savings_fuelCost'
            fuel_cost_data[mp_savings_col] = np.random.uniform(50, 500, n)
    
    # Combine all cost data with the base dataframe
    df = pd.concat([df, pd.DataFrame(capital_cost_data, index=range(n))], axis=1)
        
    # Create fuel costs dataframe all at once
    df_fuel_costs = pd.DataFrame(fuel_cost_data, index=range(n))
    
    return df, df_fuel_costs

# Function to compare the outputs of both implementations
def compare_outputs(df_current, df_backup):
    """
    Compare the outputs of both implementations and report differences.
    
    Args:
        df_current (DataFrame): Output from current implementation
        df_backup (DataFrame): Output from backup implementation
        
    Returns:
        bool: True if all columns match between implementations
    """
    # Get all columns from both DataFrames
    all_columns = set(df_current.columns).union(set(df_backup.columns))
    
    # Print comparison results
    print("\n===== COMPARISON RESULTS =====")
    matching_cols = 0
    total_cols = 0
    
    for col in sorted(all_columns):
        if col in df_current.columns and col in df_backup.columns:
            total_cols += 1
            # Check if values match (within a small tolerance for floating point differences)
            is_close = np.allclose(
                df_current[col].fillna(0),
                df_backup[col].fillna(0),
                rtol=1e-5,  # Relative tolerance
                atol=1e-8   # Absolute tolerance
            )
            
            if is_close:
                matching_cols += 1
                print(f"✓ Column '{col}' matches between implementations")
            else:
                print(f"✗ Column '{col}' differs between implementations")
                # Show some sample differences
                print("  Sample values from current implementation:")
                print(df_current[col].head(3))
                print("  Sample values from backup implementation:")
                print(df_backup[col].head(3))
                print("  Difference:")
                print(abs(df_current[col].fillna(0) - df_backup[col].fillna(0)).head(3))
                print()
        else:
            print(f"! Column '{col}' exists in only one implementation")
            if col in df_current.columns:
                print(f"  Only in current implementation")
            else:
                print(f"  Only in backup implementation")
            
    # Print summary
    print(f"\nSummary: {matching_cols}/{total_cols} columns match exactly between implementations")
    print(f"Equivalence rate: {matching_cols/total_cols*100:.2f}%")
    
    if matching_cols == total_cols:
        print("\n✓✓✓ BOTH IMPLEMENTATIONS ARE FUNCTIONALLY EQUIVALENT ✓✓✓")
        return True
    else:
        print("\n✗✗✗ IMPLEMENTATIONS HAVE DIFFERENCES ✗✗✗")
        return False

# Main test function with parameters
def test_equivalence(menu_mp, input_mp):
    """
    Test the equivalence of both implementations with specified parameters.
    
    Args:
        menu_mp (int): Measure package identifier for column naming
        input_mp (str): Input policy scenario (e.g., 'upgrade09')
        
    Returns:
        bool: True if implementations are equivalent, False otherwise
    """
    print(f"Creating test data for menu_mp={menu_mp}...")
    
    # Test both policy scenarios
    policy_scenarios = ['No Inflation Reduction Act', 'AEO2023 Reference Case']
    
    all_results_matching = True
    
    for policy_scenario in policy_scenarios:
        print(f"\n\nTesting with policy scenario: {policy_scenario}, menu_mp={menu_mp}, input_mp={input_mp}")
        
        # Create test data
        df, df_fuel_costs = create_test_data(menu_mp=menu_mp,
                                            policy_scenario=policy_scenario
                                            )

        # Run backup implementation
        interest_rate = 0.07  # 7% for private fixed rate (equivalent to 'private_fixed')
        df_backup_result = backup.calculate_private_NPV(
            df=df.copy(), 
            df_fuel_costs=df_fuel_costs.copy(),
            interest_rate=interest_rate,
            input_mp=input_mp,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
        )
        
        # Run current implementation
        df_current_result = current.calculate_private_NPV(
            df=df.copy(), 
            df_fuel_costs=df_fuel_costs.copy(),
            input_mp=input_mp,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            discounting_method='private_fixed',  # Equivalent to 0.07 interest rate
            base_year=2024  # Base year
        )
        
        # Compare results
        print(f"\nComparing results for policy scenario: {policy_scenario}")
        scenario_columns_match = compare_outputs(df_current_result, df_backup_result)
        all_results_matching = all_results_matching and scenario_columns_match
    
    return all_results_matching

if __name__ == "__main__":
    print("Starting equivalence testing between implementations...")
    
    # Define test cases with valid measure packages and input_mp values
    test_cases = [
        {'menu_mp': 8, 'input_mp': 'upgrade08'}, # Does not include weatherization/enclosure upgrade
        {'menu_mp': 9, 'input_mp': 'upgrade09'}, # Include upgrade09 as it's used in the original
        {'menu_mp': 10, 'input_mp': 'upgrade10'},  
    ]
    
    # Run all test cases
    all_tests_passed = True
    for test_case in test_cases:
        print(f"\n\n========== TESTING CASE: menu_mp={test_case['menu_mp']}, input_mp={test_case['input_mp']} ==========")
        test_result = test_equivalence(
            menu_mp=test_case['menu_mp'], 
            input_mp=test_case['input_mp']
        )
        all_tests_passed = all_tests_passed and test_result
    
    # Final summary
    if all_tests_passed:
        print("\n\n✅ ALL TESTS PASSED! Both implementations are functionally equivalent across all test cases.")
    else:
        print("\n\n❌ TESTS FAILED! There are differences between the implementations in at least one test case.")
