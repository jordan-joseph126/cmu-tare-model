import pandas as pd
from typing import Tuple, Dict

from config import PROJECT_ROOT
from cmu_tare_model.constants import EQUIPMENT_SPECS
from cmu_tare_model.utils.modeling_params import define_scenario_params
from cmu_tare_model.utils.discounting import calculate_discount_factor
print(f"Project root directory: {PROJECT_ROOT}")

"""
========================================================================================================================================================================
OVERVIEW: CALCULATE LIFETIME PRIVATE IMPACTS
========================================================================================================================================================================
This module calculates the private net present value (NPV) for various equipment categories,
considering different cost assumptions and potential IRA rebates.

# UPDATED MARCH 24, 2025 @ 4:30 PM - REMOVED RSMEANS CCI ADJUSTMENTS
# UPDATED APRIL 9, 2025 @ 7:30 PM - IMPROVED DOCUMENTATION
"""

# ========================================================================================================================================================================
# LIFETIME PRIVATE IMPACT: NPV OF CAPITAL COST INVESTMENT AND LIFETIME FUEL COSTS
# ========================================================================================================================================================================

def calculate_private_NPV(
        df: pd.DataFrame,
        df_fuelCosts: pd.DataFrame,
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

    Args:
        df (DataFrame): Input DataFrame with installation costs, fuel savings, and potential rebates.
        df_fuelCosts (DataFrame): DataFrame containing fuel cost savings data.
        menu_mp (str): Prefix for columns in the DataFrame.
        input_mp (str): Input policy_scenario for calculating costs.
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
    df_fuelCosts_copy = df_fuelCosts.copy()
    df_new_columns = pd.DataFrame(index=df_copy.index)
    
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

    for category, lifetime in EQUIPMENT_SPECS.items():
        print(f"\nDetermining lifetime private impacts for category: {category} with lifetime: {lifetime}")

        # Calculate total and net capital costs based on policy scenario        
        total_capital_cost, net_capital_cost = calculate_capital_costs(df_copy=df_copy,
                                                                       category=category,
                                                                       input_mp=input_mp,
                                                                       menu_mp=menu_mp,
                                                                       policy_scenario=policy_scenario,
                                                                       )
        
        # print(f"Total capital cost for {category}: {total_capital_cost}")
        # print(f"Net capital cost for {category}: {net_capital_cost}")

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

    # Drop any overlapping columns from df_copy
    overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)
    if not overlapping_columns.empty:
        df_copy.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into the original DataFrame
    df_copy = df_copy.join(df_new_columns, how='left')
    # print("Final DataFrame after NPV calculations:\n", df_copy.head())
    return df_copy


def calculate_capital_costs(
    df_copy: pd.DataFrame, 
    category: str, 
    input_mp: str, 
    menu_mp: int, 
    policy_scenario: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate total and net capital costs for an equipment category.
    
    This function computes the total capital cost and net capital cost (after accounting
    for replacement costs) based on the equipment category, measure package, and whether
    IRA rebates are applied.
    
    Args:
        df: DataFrame containing cost data.
        category: Equipment category (e.g., 'heating', 'waterHeating').
        menu_mp: Measure package identifier (integer) used for column naming.
        policy_scenario: Policy scenario that determines if IRA rebates are applied.
                       'No Inflation Reduction Act' means no rebates are applied.
    
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

    # print(f"Calculated total_capital_cost: {total_capital_cost}, net_capital_cost: {net_capital_cost}")
    return total_capital_cost, net_capital_cost

# Update function signature:
def calculate_and_update_npv(
    df_new_columns: pd.DataFrame,
    df_fuel_costs_copy: pd.DataFrame,
    category: str,
    lifetime: int,
    total_capital_cost: pd.Series,
    net_capital_cost: pd.Series,
    policy_scenario: str,
    scenario_prefix: str,
    discount_factors: Dict[int, float],
    base_year: int = 2024,
) -> None:
    """Calculate and update NPV values in the results DataFrame.
    
    This function computes the NPV for two willingness-to-pay (WTP) scenarios:
    - Less WTP: Using total capital cost in calculations
    - More WTP: Using net capital cost (total - replacement) in calculations
    
    The NPV is based on discounted lifetime fuel cost savings minus the applicable capital cost.
    
    Args:
        df_new_columns: DataFrame to update with calculated NPV values.
        df_fuel_costs_copy: DataFrame containing fuel cost savings data.
        category: Equipment category being processed.
        lifetime: Expected lifetime of the equipment in years.
        total_capital_cost: Series with total capital costs.
        net_capital_cost: Series with net capital costs.
        policy_scenario: Policy scenario that determines column naming.
        scenario_prefix: Prefix for column names based on policy scenario.
        discount_factors: Dictionary mapping years to discount factors.
        base_year: Base year for calculations.
    """
    print(f"""Calculating Private NPV for {category} with lifetime: {lifetime} years
          policy_scenario: {policy_scenario} --> scenario_prefix: {scenario_prefix}
          """)
    
    # Initialize total_discounted_savings
    total_discounted_savings = 0
    
    # Calculate discounted fuel savings for each year in the equipment's lifetime
    for year in range(1, lifetime + 1):
        year_label = year + (base_year - 1)
            
        # Get the savings column name and retrieve values
        savings_col = f'{scenario_prefix}{year_label}_{category}_savings_fuelCost'
        if savings_col in df_fuel_costs_copy.columns:
            annual_savings = df_fuel_costs_copy[savings_col].fillna(0)
                
            # Apply pre-calculated discount factor
            discount_factor = discount_factors[year_label]
            total_discounted_savings += annual_savings * discount_factor
    
    # Calculate NPV for less WTP and more WTP scenarios
    npv_less_wtp = round(total_discounted_savings - total_capital_cost, 2)
    npv_more_wtp = round(total_discounted_savings - net_capital_cost, 2)
    
    # Update capital cost columns
    df_new_columns[f'{scenario_prefix}{category}_total_capitalCost'] = total_capital_cost
    df_new_columns[f'{scenario_prefix}{category}_net_capitalCost'] = net_capital_cost
    
    # Update NPV columns
    df_new_columns[f'{scenario_prefix}{category}_private_npv_lessWTP'] = npv_less_wtp
    df_new_columns[f'{scenario_prefix}{category}_private_npv_moreWTP'] = npv_more_wtp
