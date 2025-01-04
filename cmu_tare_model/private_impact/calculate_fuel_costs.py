import pandas as pd
# from config import PROJECT_ROOT
from cmu_tare_model.utils.process_fuel_price_data import lookup_fuel_prices_preIRA, lookup_fuel_prices_iraRef

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ANNUAL FUEL COST
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# LAST UPDATED DECEMBER 22, 2024 @ 3 PM
def calculate_annual_fuelCost(df, menu_mp, policy_scenario, drop_fuel_cost_columns):
    """
    Calculate the annual fuel cost for baseline and measure packages.

    Args:
        df (pd.DataFrame): DataFrame containing baseline fuel consumption data.
        menu_mp (int): Measure package identifier.
        policy_scenario (str): Name of EIA AEO policy_scenario used to project fuel prices.
        drop_fuel_cost_columns (bool): Flag indicating whether to drop annual fuel cost columns 
            after calculating savings.

    Returns:
        pd.DataFrame: DataFrame with additional columns for annual fuel costs, savings, and changes.

    Raises:
        ValueError: If an invalid policy_scenario is provided. Must be one of:
            'No Inflation Reduction Act' or 'AEO2023 Reference Case'.
    """
    df_copy = df.copy()

    # Determine the scenario prefix and fuel price lookup based on menu_mp and policy_scenario
    if menu_mp == 0:
        scenario_prefix = "baseline_"
        fuel_price_lookup = lookup_fuel_prices_preIRA
    else:
        if policy_scenario == 'No Inflation Reduction Act':
            scenario_prefix = f"preIRA_mp{menu_mp}_"
            fuel_price_lookup = lookup_fuel_prices_preIRA
        elif policy_scenario == 'AEO2023 Reference Case':
            scenario_prefix = f"iraRef_mp{menu_mp}_"
            fuel_price_lookup = lookup_fuel_prices_iraRef
        else:
            raise ValueError("Invalid Policy policy_scenario! Please choose from 'No Inflation Reduction Act' or 'AEO2023 Reference Case'.")

    # Fuel type mapping and equipment lifetime specifications
    fuel_mapping = {'Electricity': 'electricity', 'Natural Gas': 'naturalGas', 'Fuel Oil': 'fuelOil', 'Propane': 'propane'}
    equipment_specs = {'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15}

    # Initialize a dictionary to hold new columns
    new_columns = {}

    # If baseline calculations are required
    if menu_mp == 0:
        # Map each baseline fuel to its lower-case version
        for category in equipment_specs:
            df_copy[f'fuel_type_{category}'] = df_copy[f'base_{category}_fuel'].map(fuel_mapping)

        for category, lifetime in equipment_specs.items():
            print(f"Calculating BASELINE (no retrofit) fuel costs from 2024 to {2024 + lifetime} for {category}")

            # Create a boolean mask indicating which rows use 'state' vs. 'census_division'
            is_elec_or_gas = df_copy[f'fuel_type_{category}'].isin(['electricity', 'naturalGas'])

            for year in range(1, lifetime + 1):
                year_label = year + 2023

                # Build a list/Series of per-row prices using a dictionary lookup based on state/census_division
                df_copy['_temp_price'] = [
                    fuel_price_lookup
                        .get(
                            # Use 'state' if electric/natural gas, else use 'census_division'
                            state_val if use_state else cdiv_val,
                            {}
                        )
                        .get(fueltype_val, {})
                        .get(policy_scenario, {})
                        .get(year_label, 0)
                    for state_val, cdiv_val, fueltype_val, use_state in zip(
                        df_copy['state'],
                        df_copy['census_division'],
                        df_copy[f'fuel_type_{category}'],
                        is_elec_or_gas
                    )
                ]

                # Multiply consumption by the temp price in a vectorized manner
                fuel_costs = round(
                    df_copy[f'baseline_{year_label}_{category}_consumption'] * df_copy['_temp_price'],
                    2
                )

                # Add resulting fuel cost column to the dictionary
                new_columns[f'baseline_{year_label}_{category}_fuelCost'] = fuel_costs

            # Drop the temporary column after finishing this category
            if '_temp_price' in df_copy.columns:
                df_copy.drop(columns=['_temp_price'], inplace=True)

    else:
        # For measure packages, everything is mapped to electricity (via 'state')
        for category, lifetime in equipment_specs.items():
            print(f"Calculating POST-RETROFIT (MP{menu_mp}) fuel costs from 2024 to {2024 + lifetime} for {category}")

            for year in range(1, lifetime + 1):
                year_label = year + 2023

                # Build a list/Series of per-row prices for electricity
                df_copy['_temp_price'] = [
                    fuel_price_lookup
                        .get(state_name, {})
                        .get('electricity', {})
                        .get(policy_scenario, {})
                        .get(year_label, 0)
                    for state_name in df_copy['state']
                ]

                # Multiply consumption by the temp price in a vectorized manner
                fuel_costs = round(
                    df_copy[f'mp{menu_mp}_{year_label}_{category}_consumption'] * df_copy['_temp_price'],
                    2
                )

                # Store the new columns
                new_columns[f'{scenario_prefix}{year_label}_{category}_fuelCost'] = fuel_costs

                # Calculate the difference between baseline and measure package fuel costs
                new_columns[f'{scenario_prefix}{year_label}_{category}_savings_fuelCost'] = (
                    df_copy[f'baseline_{year_label}_{category}_fuelCost'] - fuel_costs
                )

            # Drop the temporary column after finishing this category
            if '_temp_price' in df_copy.columns:
                df_copy.drop(columns=['_temp_price'], inplace=True)

        # Optionally drop the annual fuel cost columns if savings alone are needed
        if drop_fuel_cost_columns:
            print("Dropping Annual Fuel Costs for Baseline Scenario and Retrofit. Storing Fuel Savings for Private NPV Calculation.")
            fuel_cost_columns = [col for col in df_copy.columns if '_fuelCost' in col and '_savings_fuelCost' not in col]
            df_copy.drop(columns=fuel_cost_columns, inplace=True)

    # Create a DataFrame from new columns based on df_copy index
    df_new_columns = pd.DataFrame(new_columns, index=df_copy.index)

    # Identify overlapping columns between the new and existing DataFrame
    overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)
    if not overlapping_columns.empty:
        # Drop overlapping columns from df_copy to avoid duplication
        df_copy.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into df_copy
    df_copy = df_copy.join(df_new_columns, how='left')

    # Return the updated DataFrame with calculated costs and optional savings
    return df_copy