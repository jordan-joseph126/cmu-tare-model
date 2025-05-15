import pandas as pd

from config import PROJECT_ROOT
print(f"Project root directory: {PROJECT_ROOT}")

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
LIFETIME PRIVATE IMPACT: NPV OF CAPITAL COST INVESTMENT AND LIFETIME FUEL COSTS
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# UPDATED MARCH 24, 2025 @ 4:30 PM - REMOVED RSMEANS CCI ADJUSTMENTS
def calculate_private_npv(df, df_fuel_costs, interest_rate, input_mp, menu_mp, policy_scenario):
    """
    Calculate the private net present value (NPV) for various equipment categories,
    considering different cost assumptions and potential IRA rebates. 
    The function adjusts equipment costs for inflation and calculates NPV based on 
    cost savings between baseline and retrofit scenarios.

    Parameters:
        df (DataFrame): Input DataFrame with installation costs, fuel savings, and potential rebates.
        df_fuel_costs (DataFrame): DataFrame containing fuel cost savings data.
        interest_rate (float): Annual discount rate used for NPV calculation.
        menu_mp (str): Prefix for columns in the DataFrame.
        input_mp (str): Input policy_scenario for calculating costs.
        policy_scenario (str): Policy policy_scenario that determines electricity grid projections. 
                               Accepted values: 'AEO2023 Reference Case'.

    Returns:
        DataFrame: The input DataFrame updated with calculated private NPV and adjusted equipment costs.
    """
    # POTENTIALLY UPDATE CODE IN THE FUTURE TO ACCOUNT FOR CHANGES IN CAPITAL COSTS BASED ON SCENARIOS (BESIDES IRA REBATES)
    # Note: CURRENT MODELING ASSUMES EQUIPMENT PRICES ARE THE SAME UNDER IRA REF AND IRA HIGH
    # THIS MAY BE UPDATED IN THE FUTURE, SO WE STILL USE policy_scenario PREFIXES FOR TOTAL AND NET CAPITAL COSTS
    # COSTS ARE DIFFERENT FOR PRE-IRA BECAUSE NO REBATES ARE APPLIED   
    equipment_specs = {
        'heating': 15,
        'waterHeating': 12,
        'clothesDrying': 13,
        'cooking': 15
    }
    
    df_copy = df.copy()

    df_fuel_costs_copy = df_fuel_costs.copy()

    df_new_columns = pd.DataFrame(index=df_copy.index)
    
    for category, lifetime in equipment_specs.items():
        # print(f"\nCalculating for category: {category} with lifetime: {lifetime}")
        
        total_capital_cost, net_capital_cost = calculate_costs(df_copy, category, input_mp, menu_mp, policy_scenario)
        
        # print(f"Total capital cost for {category}: {total_capital_cost}")
        # print(f"Net capital cost for {category}: {net_capital_cost}")
        
        calculate_and_update_npv(df_new_columns, df_fuel_costs_copy, category, menu_mp, interest_rate, lifetime, total_capital_cost, net_capital_cost, policy_scenario)
      
    overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)
    if not overlapping_columns.empty:
        df_copy.drop(columns=overlapping_columns, inplace=True)

    df_copy = df_copy.join(df_new_columns, how='left')
    # print("Final DataFrame after NPV calculations:\n", df_copy.head())
    return df_copy

def calculate_costs(df_copy, category, input_mp, menu_mp, policy_scenario):
    """
    Calculate total and net capital costs based on the equipment category and cost assumptions.

    Parameters:
        df_copy (DataFrame): DataFrame containing cost data.
        category (str): Equipment category.
        menu_mp (str): Prefix for columns in the DataFrame.
        input_mp (str): Input policy_scenario for calculating costs.
        ira_rebates (bool): Flag indicating whether IRA rebates are applied.

    Returns:
        tuple: Total and net capital costs.
    """
    print(f"""\nCalculating costs for {category}...
          input_mp: {input_mp}, menu_mp: {menu_mp}, policy_scenario: {policy_scenario}""")


    # POTENTIALLY UPDATE CODE IN THE FUTURE TO ACCOUNT FOR CHANGES IN CAPITAL COSTS BASED ON SCENARIOS (BESIDES IRA REBATES)
    # Note: CURRENT MODELING ASSUMES EQUIPMENT PRICES ARE THE SAME UNDER IRA REF AND IRA HIGH
    # THIS MAY BE UPDATED IN THE FUTURE, SO WE STILL USE policy_scenario PREFIXES FOR TOTAL AND NET CAPITAL COSTS
    # COSTS ARE DIFFERENT FOR PRE-IRA BECAUSE NO REBATES ARE APPLIED
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
    
    # POTENTIALLY UPDATE CODE IN THE FUTURE TO ACCOUNT FOR CHANGES IN CAPITAL COSTS BASED ON SCENARIOS (BESIDES IRA REBATES)
    # Note: CURRENT MODELING ASSUMES EQUIPMENT PRICES ARE THE SAME UNDER IRA REF AND IRA HIGH
    # THIS MAY BE UPDATED IN THE FUTURE, SO WE STILL USE policy_scenario PREFIXES FOR TOTAL AND NET CAPITAL COSTS
    # COSTS ARE DIFFERENT FOR PRE-IRA BECAUSE NO REBATES ARE APPLIED
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

def calculate_and_update_npv(df_new_columns, df_fuel_costs_copy, category, menu_mp, interest_rate, lifetime, total_capital_cost, net_capital_cost, policy_scenario):
    """
    Calculate and update the NPV values in the DataFrame based on provided capital costs.

    Parameters:
        df_new_columns (DataFrame): DataFrame to update.
        df_fuel_costs_copy (DataFrame): Original DataFrame containing savings data.
        category (str): Equipment category.
        menu_mp (str): Prefix for columns in the DataFrame.
        interest_rate (float): Discount rate for NPV calculation.
        lifetime (int): Expected lifetime of the equipment.
        total_capital_cost (float): Total capital cost of the equipment.
        net_capital_cost (float): Net capital cost after considering replacements.
        ira_rebates (bool): Flag to consider IRA rebates in calculations.
    """
    # Determine the policy_scenario prefix based on the policy policy_scenario
    if policy_scenario == 'No Inflation Reduction Act':
        scenario_prefix = f"preIRA_mp{menu_mp}_"
    elif policy_scenario == 'AEO2023 Reference Case':
        scenario_prefix = f"iraRef_mp{menu_mp}_"
    else:
        raise ValueError("Invalid Policy policy_scenario! Please choose from 'AEO2023 Reference Case'.")
        
    print(f"""\nCalculating Private NPV for {category}...
          lifetime: {lifetime}, interest_rate: {interest_rate}, policy_scenario: {policy_scenario}
          """)

    # Calculate the discounted savings for each year
    discounted_savings = []
    for year in range(1, lifetime + 1):
        year_label = year + 2023  # Adjust the start year as necessary
        annual_savings = df_fuel_costs_copy[f'{scenario_prefix}{year_label}_{category}_savings_fuel_cost'].fillna(0)
        discount_factor = (1 / ((1 + interest_rate) ** year))
        discounted_savings.append(annual_savings * discount_factor)
        # print(f"Year {year_label} savings for {category}: {annual_savings}, discounted: {annual_savings * discount_factor}")
    
    # Sum up the discounted savings over the lifetime
    total_discounted_savings = sum(discounted_savings)
    # print(f"Total discounted savings over {lifetime} years for {category}: {total_discounted_savings}")
    
    # Calculate NPV for less WTP and more WTP scenarios
    npv_lessWTP = round(total_discounted_savings - total_capital_cost, 2)
    npv_moreWTP = round(total_discounted_savings - net_capital_cost, 2)
    
    # POTENTIALLY UPDATE CODE IN THE FUTURE TO ACCOUNT FOR CHANGES IN CAPITAL COSTS BASED ON SCENARIOS (BESIDES IRA REBATES)
    # Note: CURRENT MODELING ASSUMES EQUIPMENT PRICES ARE THE SAME UNDER IRA REF AND IRA HIGH
    # THIS MAY BE UPDATED IN THE FUTURE, SO WE STILL USE policy_scenario PREFIXES FOR TOTAL AND NET CAPITAL COSTS
    # COSTS ARE DIFFERENT FOR PRE-IRA BECAUSE NO REBATES ARE APPLIED
    df_new_columns[f'{scenario_prefix}{category}_total_capitalCost'] = total_capital_cost
    df_new_columns[f'{scenario_prefix}{category}_net_capitalCost'] = net_capital_cost
        
    df_new_columns[f'{scenario_prefix}{category}_private_npv_lessWTP'] = npv_lessWTP
    df_new_columns[f'{scenario_prefix}{category}_private_npv_moreWTP'] = npv_moreWTP
        
    # print(f"Updated df_new_columns with NPV for {category}:\n", df_new_columns[[col for col in df_new_columns.columns if category in col]].head())