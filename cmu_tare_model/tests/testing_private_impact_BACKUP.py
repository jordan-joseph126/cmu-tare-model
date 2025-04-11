import pandas as pd
import numpy as np
import sys
import os
from copy import deepcopy

# Import both versions of the function
from cmu_tare_model.private_impact import calculate_lifetime_private_impact as current
from cmu_tare_model.private_impact import calculate_lifetime_private_impact_BACKUP as backup

from cmu_tare_model.constants import EQUIPMENT_SPECS

# Create synthetic test data that matches the expected structure
def create_test_data(menu_mp, input_mp):
    # Create a small synthetic DataFrame with the necessary columns
    np.random.seed(42)  # For reproducibility
    
    n = 10  # Number of rows for the test data

    # Create base DataFrame
    df = pd.DataFrame({
        'mp9_enclosure_upgradeCost': np.random.uniform(1000, 5000, n),
        'mp10_enclosure_upgradeCost': np.random.uniform(1000, 5000, n),
        'weatherization_rebate_amount': np.random.uniform(100, 1000, n),
    })
    

    # Prepare data dictionaries to use concat instead of insert
    fuel_cost_data = {}
    capital_cost_data = {}

    # Create scenario prefixes for cost data
    scenario_prefixes = [f'preIRA_mp{menu_mp}_', f'iraRef_mp{menu_mp}_']

    base_year = 2024  # Base year for the test data

    # Fuel Costs
    for category, lifetime in EQUIPMENT_SPECS.items():
        for scenario_prefix in scenario_prefixes:
            for year in range(1, lifetime + 1):
                year_label = year + (base_year - 1)
                    
                # Get the savings column name and retrieve values
                savings_col = f'{scenario_prefix}{year_label}_{category}_savings_fuelCost'
                fuel_cost_data[savings_col] = np.random.uniform(50, 500, n)
        
    # Capital costs
    for category, lifetime in EQUIPMENT_SPECS.items():
        # Capital Costs
        if category == 'heating':
            # Installation premiums (only for heating)
            capital_cost_data[f'mp{menu_mp}_heating_installation_premium'] = np.random.uniform(100, 500, n)
        
        # Installation costs
        capital_cost_data[f'mp{menu_mp}_{category}_installationCost'] = np.random.uniform(2000, 8000, n)
        
        # Replacement costs
        capital_cost_data[f'mp{menu_mp}_{category}_replacementCost'] = np.random.uniform(1000, 4000, n)
                
        # Rebate amounts
        capital_cost_data[f'mp{menu_mp}_{category}_rebate_amount'] = np.random.uniform(500, 2000, n)

    # Combine all cost data with the base dataframe
    df = pd.concat([df, pd.DataFrame(capital_cost_data, index=range(n))], axis=1)
    
    # Create fuel costs dataframe all at once
    df_fuelCosts = pd.DataFrame(fuel_cost_data, index=range(n))
    
    return df, df_fuelCosts

# Function to compare the outputs of both implementations
def compare_outputs(df_current, df_backup):
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
            
    # Print summary
    print(f"\nSummary: {matching_cols}/{total_cols} columns match exactly between implementations")
    print(f"Equivalence rate: {matching_cols/total_cols*100:.2f}%")
    
    if matching_cols == total_cols:
        print("\n✓✓✓ BOTH IMPLEMENTATIONS ARE FUNCTIONALLY EQUIVALENT ✓✓✓")
    else:
        print("\n✗✗✗ IMPLEMENTATIONS HAVE DIFFERENCES ✗✗✗")

# Main test function
def test_equivalence(menu_mp, input_mp):
    print("Creating test data...")
    df, df_fuelCosts = create_test_data()
    
    # Parameters for both functions
    policy_scenarios = ['No Inflation Reduction Act', 'AEO2023 Reference Case']
    
    all_results_matching = True
    
    for policy_scenario in policy_scenarios:
        print(f"\n\nTesting with policy scenario: {policy_scenario}")
        # Run backup implementation
        # def calculate_private_NPV(df, df_fuelCosts, interest_rate, input_mp, menu_mp, policy_scenario)
        interest_rate = 0.07  # 7% for private fixed rate
        df_backup_result = backup.calculate_private_NPV(
            df=df.copy(), 
            df_fuelCosts=df_fuelCosts.copy(),
            interest_rate=interest_rate,
            input_mp=input_mp,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
        )
        
        # Run current implementation
        # def calculate_private_NPV(df, df_fuelCosts, input_mp, menu_mp, policy_scenario, discounting_method, base_year=2024):
        df_current_result = current.calculate_private_NPV(
            df=df.copy(), 
            df_fuelCosts=df_fuelCosts.copy(),
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
    print("Starting equivalence test between implementations...")
    test_result = test_equivalence()
    
    if test_result:
        print("\nTest passed! Both implementations are functionally equivalent.")
    else:
        print("\nTest failed! There are differences between the implementations.")


"""
Prompt:
I'm running a testing script (testing_private_impact.py) that compares two implementations (the backup and the current versions) of the function calculate_private_NPV from our cmu_tare_model module. For the "No Inflation Reduction Act" policy scenario, the test runs through calculating costs and NPVs for several categories (heating, waterHeating, clothesDrying, cooking), and then compares the resulting DataFrames column by column.
For example, some of the output for the "No Inflation Reduction Act" scenario is as follows:

sql
Copy
... Comparing results for policy scenario: No Inflation Reduction Act ===== COMPARISON RESULTS ===== ✓ Column 'iraRef_mp5_heating_installationCost' matches between implementations ... ✓ Column 'preIRA_mp5_clothesDrying_installationCost' matches between implementations ✗ Column 'preIRA_mp5_clothesDrying_net_capitalCost' differs between implementations Sample values from current implementation: 2561.243443, 2901.940341, -168.663047, ... Sample values from backup implementation: 1623.277919, 3370.237490, 4621.642348, ... Difference: 937.965525, 468.297149, 4790.305395, ... ... ✗ Column 'preIRA_mp5_heating_total_capitalCost' differs between implementations Sample values from current implementation: 7160.861408, 10831.890272, 9979.387213, ... Sample values from backup implementation: 8606.735941, 8021.826724, 6476.243274, ... Difference: 1445.874533, 2810.063548, 3503.143939, ... ... Summary: 42/58 columns match exactly (equivalence rate: 72.41%) ✗✗✗ IMPLEMENTATIONS HAVE DIFFERENCES ✗✗✗ Test failed! There are differences between the implementations.
Below is the snippet from the updated test script that calls the current version of calculate_private_NPV:

python
Copy
df_current_result = current.calculate_private_NPV( df=df.copy(), df_fuelCosts=df_fuelCosts.copy(), input_mp=input_mp, menu_mp=menu_mp, policy_scenario=policy_scenario, discounting_method='private_fixed', # Equivalent to 0.07 interest rate base_year=2024 # Base year )
Could you please help me troubleshoot the following points?
* Discrepancies in Calculation: Several columns (e.g., preIRA_mp5_clothesDrying_net_capitalCost, preIRA_mp5_heating_total_capitalCost, and others) are producing different values between the current and backup implementations. What might be causing these numerical discrepancies? Is it possible that there have been changes in the formula or the order in which costs and rebates are applied?
* Impact of Scenario Prefixes: The test output logs both iraRef_mp5_... and preIRA_mp5_... columns. Could the differences be due to differences in how scenario prefixes are applied or handled between the two implementations?
* Validation of Input Data: Since the test data creation uses columns named like mp5_heating_installationCost and the implementations work with scenario-prefixed columns, could there be a mismatch in how the input data is transformed or expected between the backup and current versions?
* Suggested Fixes: What steps or modifications should I consider in the test script or in the functions themselves to resolve these differences, ensuring that both implementations produce equivalent outputs?
Any insights or guidance on how to diagnose and reconcile these differences would be greatly appreciated.

I've analyzed the code and found several issues causing the discrepancies between your current and backup implementations of calculate_private_NPV. Let me walk through the key differences:

Main Discrepancy Sources
Scenario Prefix Handling:
The backup version uses: f"preIRA_mp{menu_mp}_" or f"iraRef_mp{menu_mp}_"
The current version uses a scenario_prefix returned by define_scenario_params()
Column Naming in Installation Costs:
Backup version accesses: f'mp{menu_mp}_{category}_installationCost'
Current version accesses: f'{scenario_prefix}{category}_installationCost'
Inconsistent Synthetic Test Data Structure:
Your test data creation doesn't align with the column naming expected by the current implementation
Discounting Method Differences:
Backup uses manual calculation with interest rate
Current version uses a pre-calculated discount factor dictionary
Specific Issues to Fix
1. Fix the calculate_costs Function Call
In calculate_and_update_npv(), you're missing the menu_mp parameter when called from the main function. Look at this line in calculate_private_NPV:

python
calculate_and_update_npv(
    df_new_columns=df_new_columns,
    df_fuel_costs_copy=df_fuelCosts_copy,
    category=category,
    lifetime=lifetime,
    total_capital_cost=total_capital_cost,
    net_capital_cost=net_capital_cost,
    policy_scenario=policy_scenario,
    scenario_prefix=scenario_prefix,
    discount_factors=discount_factors,
    base_year=base_year
)
The function signature expects menu_mp but it's not being passed, which could lead to inconsistent column names.

2. Fix Test Data Creation
Your test data creation doesn't match the column naming expectations. You create columns like:

mp5_heating_installationCost
preIRA_mp5_heating_installationCost
But the current code expects:

mp5_heating_installationCost (for base costs)
preIRA_heating_installationCost (without the mp5_ in the middle)
Modify your test data creation to match the expected column structure:

python
# For current implementation
capital_cost_data[f'{prefix}{category}_installationCost'] = np.random.uniform(2000, 8000, n)
# Instead of
capital_cost_data[f'mp{menu_mp}_{category}_installationCost'] = np.random.uniform(2000, 8000, n)
3. Fix Scenario Prefix Handling in Test
Your test script should account for how define_scenario_params() generates prefixes. Since you don't have direct access to that function in your test, you need to ensure the test data has columns that match both naming schemes or modify the current implementation to use the same naming scheme as the backup.

4. Fix Column Access in Output Comparison
When comparing outputs, make sure to account for potentially different column naming patterns:

python
# Map between current and backup column names
column_mapping = {
    f'preIRA_{category}_total_capitalCost': f'preIRA_mp{menu_mp}_{category}_total_capitalCost',
    # Add more mappings as needed
}
Recommended Solution
Create a more flexible test data generator that includes all possible column name variations needed by both implementations.
Fix the parameter passing in the calculate_and_update_npv function call in the current implementation.
Temporarily modify the current implementation to use the exact same column naming as the backup for testing purposes:
python
# In calculate_capital_costs function
if policy_scenario == 'No Inflation Reduction Act':
    # Use the same column pattern as the backup version
    total_capital_cost = df_copy[f'mp{menu_mp}_{category}_installationCost'].fillna(0)
    # Instead of 
    total_capital_cost = df_copy[f'{scenario_prefix}{category}_installationCost'].fillna(0)
By implementing these changes, you should be able to achieve much closer alignment between the two implementations and identify any remaining calculation differences.

"""