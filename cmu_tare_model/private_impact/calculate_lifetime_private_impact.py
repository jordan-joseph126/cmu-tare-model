# Updated
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union

from cmu_tare_model.constants import EQUIPMENT_SPECS
from cmu_tare_model.utils.modeling_params import define_scenario_params
from cmu_tare_model.utils.discounting import calculate_discount_factor
from cmu_tare_model.utils.validation_framework import (
    create_retrofit_only_series,
    calculate_avoided_values,
    initialize_validation_tracking,
    replace_small_values_with_nan
)
from cmu_tare_model.utils.calculation_utils import (
    validate_common_parameters,
    apply_temporary_validation_and_mask
)

"""
========================================================================================================================================================================
OVERVIEW: CALCULATE LIFETIME PRIVATE IMPACTS
========================================================================================================================================================================
This module calculates the private net present value (NPV) for various equipment categories,
considering different cost assumptions and potential IRA rebates.

# UPDATED MARCH 24, 2025 @ 4:30 PM - REMOVED RSMEANS CCI ADJUSTMENTS
# UPDATED APRIL 9, 2025 @ 7:30 PM - IMPROVED DOCUMENTATION
# UPDATED APRIL 29, 2025 @ 10:30 AM - ADDED DATA VALIDATION CHECKS AND ERROR HANDLING
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

def calculate_private_NPV(
        df: pd.DataFrame,
        df_fuel_costs: pd.DataFrame,
        df_baseline_costs: pd.DataFrame,
        input_mp: str,
        menu_mp: int,
        policy_scenario: str,
        discounting_method: str = 'private_fixed',
        base_year: int = 2024,
        verbose: bool = True
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

    Args:
        df (DataFrame): Input DataFrame with installation costs, fuel savings, and potential rebates.
        df_fuel_costs (DataFrame): DataFrame containing measure package fuel costs.
        df_baseline_costs (DataFrame): DataFrame containing baseline fuel costs.
        input_mp (str): Input policy_scenario for calculating costs.
        menu_mp (int): Measure package identifier.
        policy_scenario (str): Policy scenario that determines electricity grid projections. 
                               Accepted values: 'No Inflation Reduction Act', 'AEO2023 Reference Case'.
        discounting_method (str): The method used for discounting. Default is 'private_fixed'.
        base_year (int): The base year for discounting calculations. Default is 2024.
        verbose (bool): Whether to print detailed processing information. Default is True.

    Returns:
        DataFrame: The input DataFrame updated with calculated private NPV and adjusted equipment costs.

    Raises:
        ValueError: If an invalid policy_scenario or menu_mp is provided.
    """
    # ===== STEP 0: Validate input parameters =====
    menu_mp, policy_scenario, discounting_method = validate_common_parameters(
        menu_mp, policy_scenario, discounting_method)

    # Create copies to avoid modifying original dataframes
    df_copy = df.copy()
    df_fuel_costs_copy = df_fuel_costs.copy()
    df_baseline_costs_copy = df_baseline_costs.copy()
    df_new_columns = pd.DataFrame(index=df_copy.index)

    # Copy inclusion flags and validation columns from df_copy to df_detailed
    validation_prefixes = ["include_", "valid_tech_", "valid_fuel_"]
    validation_cols = []
    for prefix in validation_prefixes:
        validation_cols.extend([col for col in df_copy.columns if col.startswith(prefix)])
        
    for col in validation_cols:
        if col not in df_fuel_costs_copy.columns:
            df_fuel_costs_copy[col] = df_copy[col]
        if col not in df_baseline_costs_copy.columns:
            df_baseline_costs_copy[col] = df_copy[col]

    # Initialize dictionary to track columns for masking verification
    all_columns_to_mask = {category: [] for category in EQUIPMENT_SPECS}

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

    # Process each equipment category
    for category, lifetime in EQUIPMENT_SPECS.items():
        if verbose:
            print(f"\nDetermining lifetime private impacts for category: {category} with lifetime: {lifetime}")

        # ===== STEP 1: Initialize validation tracking =====
        df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
            df_copy, category, menu_mp, verbose=verbose)
        
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
        new_columns = calculate_and_update_npv(
            df_measure_costs=df_fuel_costs_copy,
            df_baseline_costs=df_baseline_costs_copy,
            category=category,
            lifetime=lifetime,
            total_capital_cost=total_capital_cost,
            net_capital_cost=net_capital_cost,
            policy_scenario=policy_scenario,
            scenario_prefix=scenario_prefix,
            discount_factors=discount_factors,
            valid_mask=valid_mask,
            menu_mp=menu_mp,
            base_year=base_year,
            verbose=verbose
        )
        
        # Add new columns to df_new_columns
        for col_name, values in new_columns.items():
            df_new_columns[col_name] = values
            category_columns_to_mask.append(col_name)
        
        # Add all columns for this category to the masking dictionary
        all_columns_to_mask[category].extend(category_columns_to_mask)

    # ===== STEP 5: Apply final verification masking for consistency =====
    df_result = apply_temporary_validation_and_mask(df_copy, df_new_columns, all_columns_to_mask, verbose=verbose)
    
    if verbose:
        print(f"\nPrivate NPV calculation completed. Added {len(df_new_columns.columns)} new columns.")
    
    return df_result


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
        df_copy: DataFrame containing cost data.
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

    # Apply masking to costs based on valid_mask. Valid homes keep their values, invalid homes get NaN
    total_capital_cost_masked = pd.Series(np.nan, index=df_copy.index)
    net_capital_cost_masked = pd.Series(np.nan, index=df_copy.index)

    total_capital_cost_masked.loc[valid_mask] = total_capital_cost.loc[valid_mask]
    net_capital_cost_masked.loc[valid_mask] = net_capital_cost.loc[valid_mask]

    return total_capital_cost_masked, net_capital_cost_masked

def calculate_and_update_npv(
    df_measure_costs: pd.DataFrame,
    df_baseline_costs: pd.DataFrame,
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
    verbose: bool = False
) -> Dict[str, pd.Series]:
    """Calculate and update NPV values for fuel cost savings.
    
    This function computes the NPV for two willingness-to-pay (WTP) scenarios:
    - Less WTP: Using total capital cost in calculations
    - More WTP: Using net capital cost (total - replacement) in calculations
    
    The NPV is based on discounted lifetime fuel cost savings minus the applicable capital cost.
    Uses list-based collection of yearly values rather than incremental updates.
    
    Args:
        df_measure_costs: DataFrame containing measure package fuel costs.
        df_baseline_costs: DataFrame containing baseline fuel costs.
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
        verbose: Whether to print detailed progress messages.

    Returns:
        A dictionary with new columns (keys are column names, values are Series).

    Raises:
        ValueError: If the category is not recognized or if the DataFrame is empty.
    """
    print(f"""Calculating Private NPV for {category} with lifetime: {lifetime} years
          policy_scenario: {policy_scenario} --> scenario_prefix: {scenario_prefix}
          """)
    
    # ===== STEP 2: Initialize result series with template =====
    # Use create_retrofit_only_series to properly initialize with zeros for valid homes, NaN for others
    discounted_savings_template = create_retrofit_only_series(df_measure_costs, valid_mask)
    
    # Create lists to store yearly avoided costs
    yearly_avoided_costs = []
    
    # Track successful years processed
    years_processed = 0
    
    # ===== STEP 3 & 4: Valid-Only Calculation and Updates =====
    # Loop over each year in the equipment's lifetime
    for year in range(1, lifetime + 1):
        year_label = year + (base_year - 1)
        discount_factor = discount_factors[year_label]
        
        # Get column names for baseline and measure package fuel costs
        base_cost_col = f'baseline_{year_label}_{category}_fuelCost'
        measure_cost_col = f'{scenario_prefix}{year_label}_{category}_fuelCost'
        
        # Check if columns exist before calculation
        cols_exist = (base_cost_col in df_baseline_costs.columns and 
                      measure_cost_col in df_measure_costs.columns)
        
        # if cols_exist:
        #     # Calculate avoided costs for this year (baseline - measure)
        #     avoided_costs = ((df_baseline_costs[base_cost_col] - 
        #                     df_measure_costs[measure_cost_col]) * discount_factor)
            
        #     # Apply validation mask for measure packages
        #     if menu_mp != 0:
        #         avoided_costs.loc[~valid_mask] = 0.0
                
        #     yearly_avoided_costs.append(avoided_costs)
        #     years_processed += 1
            
        # elif verbose:
        #     print(f"  Warning: Fuel cost data missing for year {year_label}")

        if cols_exist:
            # Use calculate_avoided_values function for consistency
            avoided_costs = calculate_avoided_values(
            baseline_values=df_baseline_costs[base_cost_col],
            measure_values=df_measure_costs[measure_cost_col],
            retrofit_mask=(valid_mask if menu_mp != 0 else None)
            ) * discount_factor
            
            yearly_avoided_costs.append(avoided_costs)
            years_processed += 1
            
        elif verbose:
            print(f"  Warning: Fuel cost data missing for year {year_label}")
    
    # Sum up all yearly avoided costs using pandas operations
    if yearly_avoided_costs:
        # Convert list of Series to DataFrame and sum
        avoided_costs_df = pd.concat(yearly_avoided_costs, axis=1)
        total_discounted_savings = avoided_costs_df.sum(axis=1)
        
        # Apply validation mask for measure packages
        if menu_mp != 0:
            total_discounted_savings = pd.Series(
                np.where(valid_mask, total_discounted_savings, np.nan),
                index=total_discounted_savings.index
            )
    else:
        total_discounted_savings = discounted_savings_template
    
    # Replace tiny values with NaN to avoid numerical artifacts
    total_discounted_savings = replace_small_values_with_nan(total_discounted_savings)
    
    # Check if any data was processed
    if verbose:
        if years_processed == 0:
            print(f"  Warning: No fuel cost data found for {category}")
        elif years_processed < lifetime:
            print(f"  Warning: Only processed {years_processed}/{lifetime} years for fuel costs")
    
    # Calculate NPV for less WTP and more WTP scenarios
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
