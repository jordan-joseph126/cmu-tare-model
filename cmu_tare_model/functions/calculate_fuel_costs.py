import pandas as pd

project_root = "C:\\Users\\14128\\Research\\cmu-tare-model"


"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ANNUAL FUEL COST
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# LAST UPDATED SEPTEMBER 5, 2024 @ 9:37 PM
def calculate_annual_fuelCost(df, menu_mp, policy_scenario, drop_fuel_cost_columns):
    """
    Calculate the annual fuel cost for baseline and measure packages.

    Parameters:
    df (pd.DataFrame): DataFrame containing baseline fuel consumption data.
    menu_mp (int): Measure package identifier
    policy_scenario (str): Name of EIA AEO policy_scenario used to project fuel prices

    Returns:
    pd.DataFrame: DataFrame with additional columns for annual fuel costs, savings, and changes.
    """
    df_copy = df.copy()

    # Determine the scenario prefix and fuel price lookup based on menu_mp and policy_scenario
    if menu_mp == 0:
        scenario_prefix = "baseline_"
        fuel_price_lookup = preIRA_fuel_price_lookup
    else:
        if policy_scenario == 'No Inflation Reduction Act':
            scenario_prefix = f"preIRA_mp{menu_mp}_"
            fuel_price_lookup = preIRA_fuel_price_lookup
        elif policy_scenario == 'AEO2023 Reference Case':
            scenario_prefix = f"iraRef_mp{menu_mp}_"
            fuel_price_lookup = iraRef_fuel_price_lookup
        else:
            raise ValueError("Invalid Policy policy_scenario! Please choose from 'No Inflation Reduction Act' or 'AEO2023 Reference Case'.")

    # Fuel type mapping and equipment lifetime specifications
    fuel_mapping = {'Electricity': 'electricity', 'Natural Gas': 'naturalGas', 'Fuel Oil': 'fuelOil', 'Propane': 'propane'}
    equipment_specs = {'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15}

    # Initialize a dictionary to hold new columns
    new_columns = {}

    # If baseline calculations are required
    if menu_mp == 0:
        for category in equipment_specs:
            df_copy[f'fuel_type_{category}'] = df_copy[f'base_{category}_fuel'].map(fuel_mapping)

        for category, lifetime in equipment_specs.items():
            print(f"Calculating BASELINE (no retrofit) fuel costs from 2024 to {2024 + lifetime} for {category}")
            for year in range(1, lifetime + 1):
                year_label = year + 2023

                fuel_costs = df_copy.apply(lambda row: round(
                    row[f'baseline_{year_label}_{category}_consumption'] *
                    fuel_price_lookup.get(
                        row['state'] if row[f'fuel_type_{category}'] in ['electricity', 'naturalGas'] else row['census_division'],
                        {}
                    ).get(row[f'fuel_type_{category}'], {}).get(policy_scenario, {}).get(year_label, 0), 2),
                    axis=1
                )

                new_columns[f'baseline_{year_label}_{category}_fuelCost'] = fuel_costs

    else:
        for category, lifetime in equipment_specs.items():
            print(f"Calculating POST-RETROFIT (MP{menu_mp}) fuel costs from 2024 to {2024 + lifetime} for {category}")
            for year in range(1, lifetime + 1):
                year_label = year + 2023

                fuel_costs = df_copy.apply(lambda row: round(
                    row[f'mp{menu_mp}_{year_label}_{category}_consumption'] *
                    fuel_price_lookup.get(row['state'], {}).get('electricity', {}).get(policy_scenario, {}).get(year_label, 0), 2),
                    axis=1
                )

                # Store all new columns in the dictionary first
                new_columns[f'{scenario_prefix}{year_label}_{category}_fuelCost'] = fuel_costs
                
                new_columns[f'{scenario_prefix}{year_label}_{category}_savings_fuelCost'] = (
                    df_copy[f'baseline_{year_label}_{category}_fuelCost'] - fuel_costs
                )

        # Only drop if annual fuel cost savings have already been calculated
        # Drop fuel cost columns if the flag is True
        if drop_fuel_cost_columns:
            print("Dropping Annual Fuel Costs for Baseline Scenario and Retrofit. Storing Fuel Savings for Private NPV Calculation.")
            fuel_cost_columns = [col for col in df_copy.columns if '_fuelCost' in col and '_savings_fuelCost' not in col]
            df_copy.drop(columns=fuel_cost_columns, inplace=True)

    # Calculate the new columns based on policy scenario and create dataframe based on df_copy index
    df_new_columns = pd.DataFrame(new_columns, index=df_copy.index)

    # Identify overlapping columns between the new and existing DataFrame.
    overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)

    # Drop overlapping columns from df_copy.
    if not overlapping_columns.empty:
        df_copy.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into df_copy, ensuring no duplicates or overwrites occur.
    df_copy = df_copy.join(df_new_columns, how='left')

    # Return the updated DataFrame.
    return df_copy