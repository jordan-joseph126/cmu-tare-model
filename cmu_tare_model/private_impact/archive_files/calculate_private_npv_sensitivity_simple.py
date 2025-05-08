import pandas as pd
from typing import Dict

from cmu_tare_model.constants import EQUIPMENT_SPECS
from cmu_tare_model.utils.modeling_params import define_scenario_params
from cmu_tare_model.utils.discounting import calculate_discount_factor

print(f"""
===================================================================================
LIFETIME PRIVATE IMPACT: NPV OF CAPITAL COST INVESTMENT AND LIFETIME FUEL COSTS
===================================================================================
""")

def calculate_private_npv(
    df: pd.DataFrame, 
    df_fuel_costs: pd.DataFrame, 
    interest_rate: float, 
    input_mp: str, 
    menu_mp: int, 
    policy_scenario: str,
    discounting_method: str = 'private_fixed',
    base_year: int = 2024
) -> pd.DataFrame:
    """Calculate the private net present value (NPV) for various equipment categories.
    
    This function computes the private NPV by considering different cost assumptions
    including capital costs, installation costs, and potential IRA rebates. The NPV
    calculation takes into account lifetime fuel cost savings between baseline and 
    retrofit scenarios.
    
    Args:
        df: Input DataFrame with installation costs, fuel savings, and potential rebates.
        df_fuel_costs: DataFrame containing fuel cost savings data.
        interest_rate: Annual discount rate used for NPV calculation.
        input_mp: Input measure package (e.g., 'upgrade09', 'upgrade10') for calculating costs.
        menu_mp: Measure package identifier (integer) used for column naming.
        policy_scenario: Policy scenario that determines electricity grid projections. 
                       Accepted values: 'No Inflation Reduction Act', 'AEO2023 Reference Case'.
        discounting_method: The method used for discounting. Default is 'private_fixed'.
        base_year: The base year for discounting calculations. Default is 2024.
    
    Returns:
        The input DataFrame updated with calculated private NPV and adjusted equipment costs.
        
    Raises:
        ValueError: If an invalid policy_scenario or menu_mp is provided.
        KeyError: If required columns are missing from the input DataFrame.
        RuntimeError: If there's an error configuring scenario parameters.
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
    
    try:
        # Determine the scenario prefix based on menu_mp and policy_scenario
        scenario_prefix, _, _, _, _, _ = define_scenario_params(
            menu_mp=menu_mp,
            policy_scenario=policy_scenario
        )
    except ValueError as e:
        raise ValueError(f"Invalid policy scenario: {policy_scenario}. {str(e)}")
    
    # Initialize a DataFrame to store all NPV and capital cost results
    all_results: Dict[str, pd.Series] = {}
    
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
        print(f"""\nCalculating Private NPV for {category}...
              lifetime: {lifetime}, discounting_method: {discounting_method}, policy_scenario: {policy_scenario}""")
        
        # Calculate total and net capital costs based on policy scenario
        if policy_scenario == 'No Inflation Reduction Act':
            # Handle pre-IRA scenario (no rebates)
            if category == 'heating':
                # Calculate weatherization cost
                if input_mp == 'upgrade09':            
                    weatherization_cost = df_copy['mp9_enclosure_upgradeCost'].fillna(0)
                elif input_mp == 'upgrade10':
                    weatherization_cost = df_copy['mp10_enclosure_upgradeCost'].fillna(0)
                else:
                    weatherization_cost = pd.Series(0.0, index=df_copy.index)
                
                total_capital_cost = (
                    df_copy[f'mp{menu_mp}_{category}_installationCost'].fillna(0) + 
                    weatherization_cost + 
                    df_copy[f'mp{menu_mp}_heating_installation_premium'].fillna(0)
                )
            else:
                # Simpler calculation for non-heating categories
                total_capital_cost = df_copy[f'mp{menu_mp}_{category}_installationCost'].fillna(0)
        else:
            # Handle IRA scenarios (with rebates)
            if category == 'heating':
                # Calculate weatherization cost with rebates
                if input_mp == 'upgrade09':            
                    weatherization_cost = df_copy['mp9_enclosure_upgradeCost'].fillna(0) - df_copy['weatherization_rebate_amount'].fillna(0)
                elif input_mp == 'upgrade10':
                    weatherization_cost = df_copy['mp10_enclosure_upgradeCost'].fillna(0) - df_copy['weatherization_rebate_amount'].fillna(0)
                else:
                    weatherization_cost = pd.Series(0.0, index=df_copy.index)
                
                installation_cost = (
                    df_copy[f'mp{menu_mp}_{category}_installationCost'].fillna(0) + 
                    weatherization_cost + 
                    df_copy[f'mp{menu_mp}_{category}_installation_premium'].fillna(0)
                )
            else:
                # Simpler calculation for non-heating categories
                installation_cost = df_copy[f'mp{menu_mp}_{category}_installationCost'].fillna(0)
            
            # Apply equipment rebates for IRA scenarios
            rebate_amount = df_copy[f'mp{menu_mp}_{category}_rebate_amount'].fillna(0)
            total_capital_cost = installation_cost - rebate_amount
        
        # Calculate net capital cost (total cost minus replacement cost)
        net_capital_cost = total_capital_cost - df_copy[f'mp{menu_mp}_{category}_replacementCost'].fillna(0)
        
        # Store capital costs in results dictionary
        all_results[f'{scenario_prefix}{category}_total_capitalCost'] = total_capital_cost
        all_results[f'{scenario_prefix}{category}_net_capitalCost'] = net_capital_cost
        
        # Calculate NPV in a consolidated loop
        total_discounted_savings = pd.Series(0.0, index=df_copy.index)
        
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
        
        # Calculate NPV for different willingness-to-pay scenarios
        npv_less_wtp = (total_discounted_savings - total_capital_cost).round(2)
        npv_more_wtp = (total_discounted_savings - net_capital_cost).round(2)
        
        # Store NPV results
        all_results[f'{scenario_prefix}{category}_private_npv_lessWTP'] = npv_less_wtp
        all_results[f'{scenario_prefix}{category}_private_npv_moreWTP'] = npv_more_wtp
    
    # Convert the dictionary of Series to a DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Drop any overlapping columns from df_copy
    overlapping_columns = df_results.columns.intersection(df_copy.columns)
    if not overlapping_columns.empty:
        df_copy.drop(columns=overlapping_columns, inplace=True)
    
    # Merge results with the original DataFrame
    df_copy = df_copy.join(df_results, how='left')
    
    return df_copy


