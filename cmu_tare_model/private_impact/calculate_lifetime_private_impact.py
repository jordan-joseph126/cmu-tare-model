# Updated
from tabnanny import verbose
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union

from cmu_tare_model.constants import EQUIPMENT_SPECS, PRIVATE_DISCOUNTING_METHOD_SUFFIXES, REBATE_ELIGIBLE_HEATING_MPS
from cmu_tare_model.utils.modeling_params import define_scenario_params
from cmu_tare_model.utils.discounting import calculate_discount_factors
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
from cmu_tare_model.utils.column_names import (
    create_fuel_cost_col,
    create_cost_col,
    create_rebate_col,
    create_capital_col,
    create_npv_col,
    create_enclosure_cost_col,
    create_weatherization_rebate_col,
    create_installation_premium_col
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
# UPDATED JANUARY 23, 2026
    1.Add defensive column validation in calculate_capital_costs().
    2. Add _validate_required_columns() helper function and early column existence
    checking in calculate_capital_costs() to prevent KeyError when required
    installation cost columns are missing (e.g., mp10_heating_upgrade_installed_cost).

    The fix:
    - Builds a list of required columns based on category and policy scenario
    - Validates all required columns exist before accessing them
    - Raises a clear, actionable KeyError with the list of missing columns
    and guidance to ensure installation costs are calculated first

    This prevents cryptic KeyError messages and helps debug data pipeline
    issues where installation cost columns weren't created upstream.
"""

# ========================================================================================================================================================================
# LIFETIME PRIVATE IMPACT: NPV OF CAPITAL COST INVESTMENT AND LIFETIME FUEL COSTS
# ========================================================================================================================================================================

def calculate_private_npv(
        df: pd.DataFrame,
        df_fuel_costs: pd.DataFrame,
        df_baseline_costs: pd.DataFrame,
        input_mp: str,
        menu_mp: int,
        policy_scenario: str,
        discount_rate_col_name: str,
        cost_scenario: str = 'v4MID',
        base_year: int = 2024,
        verbose: bool = True
) -> pd.DataFrame:
    """
    Calculate private net present value (NPV) using BOTH fixed and variable discount rates.
    
    This function automatically computes NPV for all equipment categories using:
    - 'private_discount_rate_fixed_{low|base|high}': Constant fixed private discount rates for all households
    - 'private_discount_rate_variable': Private discount rate (household-specific), inverse relationship proportional to AMI

    This function follows the five-step validation framework:
    1. Mask Initialization: Identifies valid homes using inclusion flags and retrofit status
    2. Series Initialization: Creates result series with zeros for valid homes, NaN for others
    3. Valid-Only Calculation: Performs calculations only for valid homes
    4. Valid-Only Updates: Uses list-based collection of yearly values instead of incremental updates
    5. Final Masking: Applies consistent masking to all result columns

    Args:
        df: Input DataFrame with installation costs, fuel savings, and potential rebates.
            IMPORTANT: Must contain discount rate columns created by prepare_discount_rates().
        df_fuel_costs: DataFrame containing measure package fuel costs.
        df_baseline_costs: DataFrame containing baseline fuel costs.
        input_mp: Input policy_scenario for calculating costs.
        menu_mp: Measure package identifier.
        policy_scenario: Policy scenario that determines electricity grid projections. 
            Accepted values: 'No Inflation Reduction Act', 'AEO2023 Reference Case'.
        discount_rate_col_name: Discount rate column name for private discounting.
        cost_scenario: Cost scenario identifier for column naming. Supported values:
            'v3', 'v4LOW', 'v4MID' (default), 'v4HIGH'.
        base_year: The base year for discounting calculations. Default is 2024.
        verbose: Whether to print detailed processing information. Default is True.

    Returns:
        DataFrame with 2-8 new NPV columns per category (2 WTP scenarios × 1-4 discount methods).

    Raises:
        ValueError: If an invalid policy_scenario or menu_mp is provided.
    """
    # ===== STEP 0: Validate input parameters =====
    menu_mp, policy_scenario = validate_common_parameters(
        menu_mp, policy_scenario)

    if verbose:
        print(f"""\nCalculating Private NPV with parameters:
          input_mp: {input_mp}, menu_mp: {menu_mp}, policy_scenario: {policy_scenario}""")

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
        
    # Calculate the maximum lifetime across all equipment to determine how many years to pre-calculate
    max_lifetime = max(EQUIPMENT_SPECS.values())

    # Pre-calculate discount factors for each year to avoid redundant calculations
    # This maps from year_label to its discount factor. Series not scalar.
    discount_factors: Dict[int, pd.Series] = {}

    for year in range(1, max_lifetime + 1):
        year_label = year + (base_year - 1)
        
        # ===== Calculate private discount factors for fixed and variable methods =====
        discount_factors[year_label] = calculate_discount_factors(
            df=df_copy,
            base_year=base_year, 
            target_year=year_label,
            discount_rate_col_name=discount_rate_col_name
        )

    method_suffix = PRIVATE_DISCOUNTING_METHOD_SUFFIXES[discount_rate_col_name]

    # Process each equipment category
    for category, lifetime in EQUIPMENT_SPECS.items():
        if verbose:
            print(f"\nDetermining lifetime private impacts for category: {category} with lifetime: {lifetime}")

        # ===== STEP 1: Initialize validation tracking =====
        # MEMORY OPTIMIZATION: copy=False since df_copy was already copied at the start
        _, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
            df_copy, category, menu_mp, verbose=verbose, copy=False)
        
        # Calculate total and net capital costs based on policy scenario        
        total_capital_cost, net_capital_cost = calculate_capital_costs(
            df_copy=df_copy,
            category=category,
            input_mp=input_mp,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            cost_scenario=cost_scenario,
            valid_mask=valid_mask
        )
        
        # ===== Calculate private discount factors for fixed and variable methods =====
        # Calculate and get NPV values using FIXED discounting method
        result_cols = calculate_and_update_npv(
            df_measure_costs=df_fuel_costs_copy,
            df_baseline_costs=df_baseline_costs_copy,
            category=category,
            lifetime=lifetime,
            total_capital_cost=total_capital_cost,
            net_capital_cost=net_capital_cost,
            policy_scenario=policy_scenario,
            scenario_prefix=scenario_prefix,
            discount_factors=discount_factors,
            method_suffix=method_suffix,
            valid_mask=valid_mask,
            menu_mp=menu_mp,
            base_year=base_year,
            cost_scenario=cost_scenario,
            verbose=verbose
        )

        for col_name, values in result_cols.items():
            df_new_columns[col_name] = values
            category_columns_to_mask.append(col_name)

        # Add all columns for this category to the masking dictionary
        all_columns_to_mask[category].extend(category_columns_to_mask)

    # ===== STEP 5: Apply final verification masking for consistency =====
    df_result = apply_temporary_validation_and_mask(df_copy, df_new_columns, all_columns_to_mask, verbose=verbose)
    
    if verbose:
        print(f"\nPrivate NPV calculation completed. Added {len(df_new_columns.columns)} new columns.")
    
    return df_result


def _validate_required_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    context: str
) -> List[str]:
    """
    Validate that required columns exist in the DataFrame.

    Args:
        df: DataFrame to check.
        required_cols: List of column names that must exist.
        context: Description of the calculation context for error messages.

    Returns:
        List of missing column names (empty if all columns exist).
    """
    missing = [col for col in required_cols if col not in df.columns]
    return missing


def calculate_capital_costs(
    df_copy: pd.DataFrame,
    category: str,
    input_mp: str,
    menu_mp: int,
    policy_scenario: str,
    cost_scenario: str,
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
        cost_scenario: Cost scenario identifier used for column naming (e.g., 'mid').
        valid_mask: Series indicating which rows have valid data for the category.

    Returns:
        A tuple containing:
            - total_capital_cost: Series with total capital costs
            - net_capital_cost: Series with net capital costs (total - replacement)

    Raises:
        KeyError: If required installation cost columns are missing from the DataFrame.

    Notes:
        Current modeling assumes equipment prices are the same under IRA Reference
        and IRA High scenarios. Costs differ for pre-IRA because no rebates are applied.

    """
    if verbose:
        print(f"\nCalculating costs for {category}... ")

    # Build list of required columns based on category and policy scenario
    upgrade_cost_col_name = create_cost_col(menu_mp=menu_mp, category=category, cost_type='upgrade', cost_scenario=cost_scenario)
    replacement_cost_col_name = create_cost_col(menu_mp=menu_mp, category=category, cost_type='replacement', cost_scenario=cost_scenario)
    required_cols = [upgrade_cost_col_name, replacement_cost_col_name]

    if category == 'heating':
        required_cols.append(create_installation_premium_col(menu_mp, category))
        if input_mp in ['upgrade09', 'upgrade10']:
            required_cols.append(create_enclosure_cost_col(menu_mp=menu_mp, cost_scenario=cost_scenario))

            # Weatherization rebate only applies to MP9 and MP10 under IRA scenarios
            if policy_scenario != 'No Inflation Reduction Act':
                required_cols.append(create_weatherization_rebate_col(cost_scenario=cost_scenario))

        # Only high-efficiency MPs are eligible for heating rebates
        if policy_scenario != 'No Inflation Reduction Act' and menu_mp in REBATE_ELIGIBLE_HEATING_MPS:
            required_cols.append(create_rebate_col(menu_mp=menu_mp, category=category, cost_scenario=cost_scenario))

    elif policy_scenario != 'No Inflation Reduction Act':
        required_cols.append(create_rebate_col(menu_mp=menu_mp, category=category, cost_scenario=cost_scenario))

    # Validate required columns exist
    missing_cols = _validate_required_columns(df_copy, required_cols,
        f"{category} capital cost calculation for MP{menu_mp}")

    if missing_cols:
        raise KeyError(
            f"Missing required columns for {category} capital cost calculation "
            f"(MP{menu_mp}, {policy_scenario}): {missing_cols}. "
            f"Ensure installation costs are calculated before calling calculate_private_npv()."
        )

    if policy_scenario == 'No Inflation Reduction Act':
        if category == 'heating':
            if input_mp == 'upgrade09':            
                weatherization_cost = df_copy[create_enclosure_cost_col(menu_mp=menu_mp, cost_scenario=cost_scenario)].fillna(0)
            elif input_mp == 'upgrade10':
                weatherization_cost = df_copy[create_enclosure_cost_col(menu_mp=menu_mp, cost_scenario=cost_scenario)].fillna(0)
            else:
                weatherization_cost = 0.0
            
            total_capital_cost = (df_copy[create_cost_col(menu_mp=menu_mp, category=category, cost_type='upgrade', cost_scenario=cost_scenario)].fillna(0) + 
                                  weatherization_cost + 
                                  df_copy[create_installation_premium_col(menu_mp=menu_mp, category='heating')].fillna(0))
            net_capital_cost = total_capital_cost - df_copy[create_cost_col(menu_mp=menu_mp, category=category, cost_type='replacement', cost_scenario=cost_scenario)].fillna(0)
            
        else:
            total_capital_cost = df_copy[create_cost_col(menu_mp=menu_mp, category=category, cost_type='upgrade', cost_scenario=cost_scenario)].fillna(0)
            net_capital_cost = total_capital_cost - df_copy[create_cost_col(menu_mp=menu_mp, category=category, cost_type='replacement', cost_scenario=cost_scenario)].fillna(0)
    
    else:
        if category == 'heating':
            if input_mp == 'upgrade09':
                # menu_mp should be 9            
                weatherization_cost = df_copy[create_enclosure_cost_col(menu_mp=menu_mp, cost_scenario=cost_scenario)].fillna(0) - df_copy[create_weatherization_rebate_col(cost_scenario=cost_scenario)].fillna(0)
            elif input_mp == 'upgrade10':
                # menu_mp should be 10
                weatherization_cost = df_copy[create_enclosure_cost_col(menu_mp=menu_mp, cost_scenario=cost_scenario)].fillna(0) - df_copy[create_weatherization_rebate_col(cost_scenario=cost_scenario)].fillna(0)
            else:
                weatherization_cost = 0.0       
            
            installation_cost = (df_copy[create_cost_col(menu_mp=menu_mp, category=category, cost_type='upgrade', cost_scenario=cost_scenario)].fillna(0) + 
                                 weatherization_cost + 
                                 df_copy[create_installation_premium_col(menu_mp=menu_mp, category=category)].fillna(0))
            
            # Only high-efficiency MPs are eligible for heating rebates
            if menu_mp in REBATE_ELIGIBLE_HEATING_MPS:
                rebate_amount = df_copy[create_rebate_col(menu_mp=menu_mp, category=category, cost_scenario=cost_scenario)].fillna(0)
            else:
                rebate_amount = 0.0
            total_capital_cost = installation_cost - rebate_amount
            net_capital_cost = total_capital_cost - df_copy[create_cost_col(menu_mp=menu_mp, category=category, cost_type='replacement', cost_scenario=cost_scenario)].fillna(0)
        
        else:
            installation_cost = df_copy[create_cost_col(menu_mp=menu_mp, category=category, cost_type='upgrade', cost_scenario=cost_scenario)].fillna(0)
            rebate_amount = df_copy[create_rebate_col(menu_mp=menu_mp, category=category, cost_scenario=cost_scenario)].fillna(0)
            total_capital_cost = installation_cost - rebate_amount
            net_capital_cost = total_capital_cost - df_copy[create_cost_col(menu_mp=menu_mp, category=category, cost_type='replacement', cost_scenario=cost_scenario)].fillna(0)

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
    discount_factors: Dict[int, pd.Series],
    method_suffix: str,
    valid_mask: pd.Series,
    menu_mp: int,
    base_year: int = 2024,
    cost_scenario: str = 'v3',
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
        method_suffix: Suffix indicating discounting method 
            - Comes from DISCOUNTING_METHOD_SUFFIXES values ('_fixed_low', '_fixed_base', '_fixed_high', or '_variable')
        valid_mask: Series indicating which rows have valid data for the category.
        menu_mp: Measure package identifier (integer) used for column naming.
        base_year: Base year for calculations.
        cost_scenario: Cost methodology key ('v3' or 'v4LOW/MID/HIGH').
            Determines the REMDB suffix on output capital/NPV column names
            (e.g., '_v3', '_v4MID').
        verbose: Whether to print detailed progress messages.

    Returns:
        A dictionary with new columns (keys are column names, values are Series).

    Raises:
        ValueError: If the category is not recognized or if the DataFrame is empty.
    """    
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
        base_cost_col_name = create_fuel_cost_col('baseline_', year_label, category)
        measure_cost_col_name = create_fuel_cost_col(scenario_prefix, year_label, category)
        
        # Check if columns exist before calculation
        cols_exist = (base_cost_col_name in df_baseline_costs.columns and 
                      measure_cost_col_name in df_measure_costs.columns)
        
        if cols_exist:
            # Use calculate_avoided_values function for consistency
            avoided_costs = calculate_avoided_values(
            baseline_values=df_baseline_costs[base_cost_col_name],
            measure_values=df_measure_costs[measure_cost_col_name],
            retrofit_mask=(valid_mask if menu_mp != 0 else None)
            ) * discount_factor
            
            yearly_avoided_costs.append(avoided_costs)
            years_processed += 1
            
        else:
            raise ValueError(f"  Warning: Fuel cost data missing for year {year_label}")
    
    # Sum up all yearly avoided costs using pandas operations
    if yearly_avoided_costs:
        # Convert list of Series to DataFrame and sum
        avoided_costs_df = pd.concat(yearly_avoided_costs, axis=1)
        total_discounted_savings = avoided_costs_df.sum(axis=1, skipna=False)  # Use skipna=False to propagate NaN values

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
            raise ValueError(f"  Warning: No fuel cost data found for {category}")
        elif years_processed < lifetime:
            raise ValueError(f"  Warning: Only processed {years_processed}/{lifetime} years for fuel costs")
    
    # Calculate NPV for less WTP and more WTP scenarios
    npv_less_wtp = round(total_discounted_savings - total_capital_cost, 2)
    npv_more_wtp = round(total_discounted_savings - net_capital_cost, 2)
    
    # Create a dictionary to hold the results
    result_columns = {
        create_capital_col(scenario_prefix=scenario_prefix, category=category, net=False, cost_scenario=cost_scenario): total_capital_cost,
        create_capital_col(scenario_prefix=scenario_prefix, category=category, net=True, cost_scenario=cost_scenario): net_capital_cost,
        create_npv_col(scenario_prefix=scenario_prefix, category=category, wtp='lessWTP', cost_scenario=cost_scenario, method_suffix=method_suffix): npv_less_wtp,
        create_npv_col(scenario_prefix=scenario_prefix, category=category, wtp='moreWTP', cost_scenario=cost_scenario, method_suffix=method_suffix): npv_more_wtp
    }

    return result_columns
