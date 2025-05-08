# Updated
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union

from cmu_tare_model.constants import EQUIPMENT_SPECS

from cmu_tare_model.utils.modeling_params import define_scenario_params
from cmu_tare_model.utils.discounting import calculate_discount_factor
from cmu_tare_model.utils.validation_framework import (
    apply_final_masking,
    initialize_validation_tracking,
    create_retrofit_only_series
)

"""
========================================================================================================================================================================
OVERVIEW: CALCULATE LIFETIME PRIVATE IMPACTS
========================================================================================================================================================================
This module calculates the private net present value (NPV) for various equipment categories,
considering different cost assumptions and potential IRA rebates.

# UPDATED MARCH 24, 2025 @ 4:30 PM - REMOVED RSMEANS CCI ADJUSTMENTS
# UPDATED APRIL 9, 2025 @ 7:30 PM - IMPROVED DOCUMENTATION
# UPDATED APRIL 21, 2025 @ 10:30 AM - ADDED DATA VALIDATION CHECKS AND ERROR HANDLING
    def calculate_private_NPV:
        - Added comprehensive data validation checks and error handling for the function.
        - Applied a validation mask to ensure that only valid homes are considered in calculations.
        - Applied final verification masking to ensure data consistency before returning the result.
    def calculate_capital_costs:
        - Modified the function to accept a validation mask and applying masking to costs so that only valid homes have values.
    def calculate_and_update_npv:
        - Modified this function to accept a validation mask, initialize with valid values only, update only valid homes 
          during calculations, and return a dictionary of columns instead of directly updating the DataFrame.

"""

# ========================================================================================================================================================================
# LIFETIME PRIVATE IMPACT: NPV OF CAPITAL COST INVESTMENT AND LIFETIME FUEL COSTS
# ========================================================================================================================================================================

# UPDATED
def calculate_private_NPV(
        df: pd.DataFrame,
        df_fuel_costs: pd.DataFrame,
        input_mp: str,
        menu_mp: int,
        policy_scenario: str,
        discounting_method: str = 'private_fixed',
        base_year: int = 2024
) -> pd.DataFrame:    
    """
    Calculate the private net present value (NPV) for various equipment categories,
    considering different cost assumptions and potential IRA rebates. 
    The function adjusts equipment costs for inflation and calculates NPV based on 
    cost savings between baseline and retrofit scenarios.

    This function follows the five-step validation framework:
    1. Mask Initialization: Identifies valid homes using inclusion flags and retrofit status
    2. Series Initialization: Creates result series with zeros for valid homes, NaN for others
    3. Valid-Only Calculation: Performs calculations only for valid homes
    4. Valid-Only Updates: Uses list-based collection of yearly values instead of incremental updates
    5. Final Masking: Applies consistent masking to all result columns

    The list-based collection approach stores yearly values in lists and sums them using pandas
    vectorized operations after all years have been processed. This approach prevents accumulation
    errors that can occur with incremental updates.

    Args:
        df (DataFrame): Input DataFrame with installation costs, fuel savings, and potential rebates.
        df_fuel_costs (DataFrame): DataFrame containing fuel cost savings data.
        input_mp (str): Input policy_scenario for calculating costs.
        menu_mp (int): Measure package identifier.
        policy_scenario (str): Policy policy_scenario that determines electricity grid projections. 
                            Accepted values: 'AEO2023 Reference Case'.
        discounting_method (str): The method used for discounting. Default is 'private_fixed'.
        base_year (int): The base year for discounting calculations. Default is 2024.

    Returns:
        DataFrame: The input DataFrame updated with calculated private NPV and adjusted equipment costs.

    Raises:
        ValueError: If an invalid policy_scenario or menu_mp is provided.
    """
    # Add validation for menu_mp
    if not str(menu_mp).isdigit():
        raise ValueError(f"Invalid menu_mp: {menu_mp}. Must be a digit.")
        
    # Add validation for policy_scenario
    valid_scenarios = ['No Inflation Reduction Act', 'AEO2023 Reference Case']
    if policy_scenario not in valid_scenarios:
        raise ValueError(f"Invalid policy_scenario: {policy_scenario}. Must be one of {valid_scenarios}")
        
    # Add validation for discounting_method
    valid_methods = ['private_fixed']
    if discounting_method not in valid_methods:
        raise ValueError(f"Invalid discounting_method: {discounting_method}. Must be one of {valid_methods}")

    # Create copies to avoid modifying original dataframes
    df_copy = df.copy()
    df_fuel_costs_copy = df_fuel_costs.copy()
    df_new_columns = pd.DataFrame(index=df_copy.index)

    # Copy inclusion flags and validation columns from df_copy to df_detailed
    validation_prefixes = ["include_", "valid_tech_", "valid_fuel_"]
    validation_cols = []
    for prefix in validation_prefixes:
        validation_cols.extend([col for col in df_copy.columns if col.startswith(prefix)])
        
    for col in validation_cols:
        df_fuel_costs_copy[col] = df_copy[col]

    # Initialize dictionary to track columns for masking verification
    all_columns_to_mask = {category: [] for category in EQUIPMENT_SPECS}    # UPDATED

    # Determine the scenario prefix based on the policy scenario
    scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)
    
    # Pre-calculate discount factors for each year to avoid redundant calculations
    # This maps from year_label to its discount factor
    discount_factors: Dict[int, float] = {}
    
    # Calculate the maximum lifetime across all equipment to determine how many years to pre-calculate
    max_lifetime = max(EQUIPMENT_SPECS.values())
    for year in range(1, max_lifetime + 1):
        year_label = year + (base_year - 1)
        discount_factors[year_label] = calculate_discount_factor(base_year, year_label, discounting_method)

    # Updated
    for category, lifetime in EQUIPMENT_SPECS.items():
        print(f"\nDetermining lifetime private impacts for category: {category} with lifetime: {lifetime}")

        # STEP 1: Initialize validation tracking
        df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
            df_copy, category, menu_mp, verbose=True)

        # Calculate total and net capital costs based on policy scenario        
        total_capital_cost, net_capital_cost = calculate_capital_costs(
            df_copy=df_copy,
            category=category,
            input_mp=input_mp,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            valid_mask=valid_mask
        )
        
        # Calculate and get NPV values
        # UPDATED: Pass the valid_mask to the function
        new_columns = calculate_and_update_npv(
            df_fuel_costs_copy=df_fuel_costs_copy,
            category=category,
            lifetime=lifetime,
            total_capital_cost=total_capital_cost,
            net_capital_cost=net_capital_cost,
            policy_scenario=policy_scenario,
            scenario_prefix=scenario_prefix,
            discount_factors=discount_factors,
            valid_mask=valid_mask,
            menu_mp=menu_mp,
            base_year=base_year
        )
        
        # UPDATED: Add new columns to masking dictionary
        # Add the new columns to df_new_columns
        for col_name, values in new_columns.items():
            df_new_columns[col_name] = values
            category_columns_to_mask.append(col_name)
        
        # Add all columns for this category to the masking dictionary
        all_columns_to_mask[category].extend(category_columns_to_mask)

    # STEP 5: Apply final masking using the utility function
    print("\nVerifying masking for all calculated columns:")

    # Add temporary validation columns to df_new_columns for masking
    temp_validation_columns = {}
    for prefix in ["include_", "valid_tech_", "valid_fuel_"]:
        validation_cols = [col for col in df_copy.columns if col.startswith(prefix)]
        for col in validation_cols:
            if col not in df_new_columns.columns:
                temp_validation_columns[col] = True  # Mark this as temporary
                df_new_columns[col] = df_copy[col]

    # Apply final masking
    df_new_columns = apply_final_masking(df_new_columns, all_columns_to_mask, verbose=True)

    # Remove temporary validation columns after masking is done
    validation_columns = [col for col in temp_validation_columns.keys()]
    if validation_columns:
        df_new_columns = df_new_columns.drop(columns=validation_columns)

    # Check for overlapping columns AFTER removing temporary validation columns
    overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)
    if not overlapping_columns.empty:
        print(f"Dropping {len(overlapping_columns)} overlapping columns from the original DataFrame.")
        df_copy.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into the original DataFrame
    df_copy = df_copy.join(df_new_columns, how='left')

    return df_copy


# UPDATED
def calculate_capital_costs(
    df_copy: pd.DataFrame, 
    category: str, 
    input_mp: str, 
    menu_mp: int, 
    policy_scenario: str,
    valid_mask: pd.Series
) -> Tuple[pd.Series, pd.Series]:    
    """
    Calculate total and net capital costs for an equipment category.
    
    This function computes the total capital cost and net capital cost (after accounting
    for replacement costs) based on the equipment category, measure package, and whether
    IRA rebates are applied.
    
    Args:
        df: DataFrame containing cost data.
        category: Equipment category (e.g., 'heating', 'waterHeating').
        input_mp: Measure package identifier (string) used for column naming.
        menu_mp: Measure package identifier (integer) used for column naming.
        policy_scenario: Policy scenario that determines if IRA rebates are applied.
                       'No Inflation Reduction Act' means no rebates are applied.
                       'AEO2023 Reference Case' means IRA rebates are applied. 
        valid_mask: Series indicating which rows have valid data for the category.
        
    Returns:
        A tuple containing:
            - total_capital_cost: Series with total capital costs
            - net_capital_cost: Series with net capital costs (total - replacement)
            
    Notes:
        Current modeling assumes equipment prices are the same under IRA Reference
        and IRA High scenarios. Costs differ for pre-IRA because no rebates are applied.
    
    """

    print(f"""\nCalculating costs for {category}...
          input_mp: {input_mp}, menu_mp: {menu_mp}, policy_scenario: {policy_scenario}""")

    if policy_scenario == 'No Inflation Reduction Act':
        if category == 'heating':
            if input_mp == 'upgrade09':            
                weatherization_cost = df_copy[f'mp9_enclosure_upgradeCost'].fillna(0)
            elif input_mp == 'upgrade10':
                weatherization_cost = df_copy[f'mp10_enclosure_upgradeCost'].fillna(0)
            else:
                weatherization_cost = 0.0
            # print(f"Weatherization cost (no IRA rebates): {weatherization_cost}")
            
            total_capital_cost = (df_copy[f'mp{menu_mp}_{category}_installationCost'].fillna(0) + 
                                  weatherization_cost + 
                                  df_copy[f'mp{menu_mp}_heating_installation_premium'].fillna(0))
            net_capital_cost = total_capital_cost - df_copy[f'mp{menu_mp}_{category}_replacementCost'].fillna(0)
            
        else:
            total_capital_cost = df_copy[f'mp{menu_mp}_{category}_installationCost'].fillna(0)
            net_capital_cost = total_capital_cost - df_copy[f'mp{menu_mp}_{category}_replacementCost'].fillna(0)
    
    else:
        if category == 'heating':
            if input_mp == 'upgrade09':            
                weatherization_cost = df_copy[f'mp9_enclosure_upgradeCost'].fillna(0) - df_copy[f'weatherization_rebate_amount'].fillna(0)
            elif input_mp == 'upgrade10':
                weatherization_cost = df_copy[f'mp10_enclosure_upgradeCost'].fillna(0) - df_copy[f'weatherization_rebate_amount'].fillna(0)
            else:
                weatherization_cost = 0.0       
            # print(f"Weatherization cost (with IRA rebates): {weatherization_cost}")
            
            installation_cost = (df_copy[f'mp{menu_mp}_{category}_installationCost'].fillna(0) + 
                                 weatherization_cost + 
                                 df_copy[f'mp{menu_mp}_{category}_installation_premium'].fillna(0))
            
            rebate_amount = df_copy[f'mp{menu_mp}_{category}_rebate_amount'].fillna(0)
            total_capital_cost = installation_cost - rebate_amount
            net_capital_cost = total_capital_cost - df_copy[f'mp{menu_mp}_{category}_replacementCost'].fillna(0)
        
        else:
            installation_cost = df_copy[f'mp{menu_mp}_{category}_installationCost'].fillna(0)
            rebate_amount = df_copy[f'mp{menu_mp}_{category}_rebate_amount'].fillna(0)
            total_capital_cost = installation_cost - rebate_amount
            net_capital_cost = total_capital_cost - df_copy[f'mp{menu_mp}_{category}_replacementCost'].fillna(0)

    # UPDATED: Apply masking to costs based on valid_mask. Valid homes keep their values, invalid homes get NaN
    total_capital_cost_masked = pd.Series(np.nan, index=df_copy.index)
    net_capital_cost_masked = pd.Series(np.nan, index=df_copy.index)

    total_capital_cost_masked.loc[valid_mask] = total_capital_cost.loc[valid_mask]
    net_capital_cost_masked.loc[valid_mask] = net_capital_cost.loc[valid_mask]

    # print(f"Calculated total_capital_cost: {total_capital_cost}, net_capital_cost: {net_capital_cost}")
    return total_capital_cost_masked, net_capital_cost_masked


# ========================================================================================================================================================================
# UPDATED TO NOT USE THE PROBLEMATIC UPDATE_VALUES_FOR_RETROFITS FUNCTION
# ========================================================================================================================================================================
def calculate_and_update_npv(
    df_fuel_costs_copy: pd.DataFrame,
    category: str,
    lifetime: int,
    total_capital_cost: pd.Series,
    net_capital_cost: pd.Series,
    policy_scenario: str,
    scenario_prefix: str,
    discount_factors: Dict[int, float],
    valid_mask: pd.Series,
    menu_mp: int,
    base_year: int = 2024,
) -> Dict[str, pd.Series]:
    """Calculate and update NPV values in the results DataFrame.
    
    This function computes the NPV for two willingness-to-pay (WTP) scenarios:
    - Less WTP: Using total capital cost in calculations
    - More WTP: Using net capital cost (total - replacement) in calculations
    
    The NPV is based on discounted lifetime fuel cost savings minus the applicable capital cost.
    
    Args:
        df_fuel_costs_copy: DataFrame containing fuel cost savings data.
        category: Equipment category being processed.
        lifetime: Expected lifetime of the equipment in years.
        total_capital_cost: Series with total capital costs.
        net_capital_cost: Series with net capital costs.
        policy_scenario: Policy scenario that determines column naming.
        scenario_prefix: Prefix for column names based on policy scenario.
        discount_factors: Dictionary mapping years to discount factors.
        valid_mask: Series indicating which rows have valid data for the category.
        menu_mp: Measure package identifier (integer) used for column naming.
        base_year: Base year for calculations.

    Returns:
        A dictionary with new columns added to the DataFrame.

    Raises:
        ValueError: If the category is not recognized or if the DataFrame is empty.
    """

    print(f"""Calculating Private NPV for {category} with lifetime: {lifetime} years
          policy_scenario: {policy_scenario} --> scenario_prefix: {scenario_prefix}
          """)
    
    # Step 1 & 2: Initialize placeholder for the final result (not for accumulation)
    result_template = create_retrofit_only_series(df_fuel_costs_copy, valid_mask)

    # Create a list to store each year's discounted savings Series
    yearly_discounted_savings = []

    # Step 3: Calculate discounted fuel savings for each year in the equipment's lifetime
    for year in range(1, lifetime + 1):
        year_label = year + (base_year - 1)
            
        # Get the savings column name and retrieve values
        savings_col = f'{scenario_prefix}{year_label}_{category}_savings_fuelCost'
        if savings_col in df_fuel_costs_copy.columns:
            # Get annual savings
            annual_savings = df_fuel_costs_copy[savings_col].fillna(0).copy()
            
            # STEP 3: Apply validation mask for measure packages
            if menu_mp != 0:  # Only for measure packages
                annual_savings.loc[~valid_mask] = 0.0
                
            # Apply pre-calculated discount factor
            discount_factor = discount_factors[year_label]
            discounted_savings = annual_savings * discount_factor        

            # Add to our list of yearly savings
            yearly_discounted_savings.append(discounted_savings)  

    # Step 4: Create total discounted savings by summing all years at once
    # If no years were processed, use the initialized zeros
    if yearly_discounted_savings:
        # Convert list of Series to DataFrame
        savings_df = pd.concat(yearly_discounted_savings, axis=1)
        
        # Sum across all years
        total_discounted_savings = savings_df.sum(axis=1)
        
        # Apply validation mask to ensure only valid homes have values
        if menu_mp != 0:  # Only for measure packages
            total_discounted_savings = pd.Series(
                np.where(valid_mask, total_discounted_savings, np.nan),
                index=total_discounted_savings.index
            )
    else:
        # If no yearly savings were found, use the initialized template
        total_discounted_savings = result_template    
    
    # Calculate NPV for less WTP and more WTP scenarios using updated total_discounted_savings
    npv_less_wtp = round(total_discounted_savings - total_capital_cost, 2)
    npv_more_wtp = round(total_discounted_savings - net_capital_cost, 2)

    # Create a dictionary to hold the results
    result_columns = {
        f'{scenario_prefix}{category}_total_capitalCost': total_capital_cost,
        f'{scenario_prefix}{category}_net_capitalCost': net_capital_cost,
        f'{scenario_prefix}{category}_private_npv_lessWTP': npv_less_wtp,
        f'{scenario_prefix}{category}_private_npv_moreWTP': npv_more_wtp
    }

    return result_columns
