import pandas as pd

from config import PROJECT_ROOT
print(f"Project root directory: {PROJECT_ROOT}")

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
LIFETIME PUBLIC IMPACT: NPV OF LIFETIME MONETIZED DAMAGES (CLIMATE AND HEALTH)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# LAST UPDATED DECEMBER 5, 2024 @ 9 PM
# UPDATE THE SET OF FUNCTIONS TO CALCULATE CLIMATE, HEALTH, AND PUBLIC NPV WITH THE MER_TYPE PARAMETER
# NEXT NEED TO UPDATE THE ADOPTION DECISION FUNCTION TO INCLUDE HEALTH 
def calculate_public_npv(df, df_baseline_damages, df_mp_damages, menu_mp, policy_scenario, interest_rate=0.02):
    """
    Calculate the public Net Present Value (NPV) for specific categories of damages,
    considering different policy scenarios related to grid decarbonization.

    Parameters:
    - df (DataFrame): A pandas DataFrame containing the relevant data.
    - menu_mp (str): Menu identifier used in column names.
    - policy_scenario (str): Policy policy_scenario that determines electricity grid projections. 
                             Accepted values: 'AEO2023 Reference Case'.
    - interest_rate (float): The discount rate used in the NPV calculation. Default is 2% for Social Discount Rate.

    Returns:
    - DataFrame: The input DataFrame with additional columns containing the calculated public NPVs for each enduse.
    """
    equipment_specs = {
        'heating': 15,
        'waterHeating': 12,
        'clothesDrying': 13,
        'cooking': 15
    }
    
    df_copy = df.copy()
    df_baseline_damages_copy = df_baseline_damages.copy()
    df_mp_damages_copy = df_mp_damages.copy()

    # Calculate the lifetime damages and corresponding NPV based on the policy policy_scenario
    df_new_columns = calculate_lifetime_damages_grid_scenario(df_copy, df_baseline_damages_copy, df_mp_damages_copy, menu_mp, equipment_specs, policy_scenario, interest_rate)

    # Drop any overlapping columns from df_copy
    overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)
    if not overlapping_columns.empty:
        df_copy.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into the original DataFrame
    df_copy = df_copy.join(df_new_columns, how='left')

    return df_copy

def calculate_lifetime_damages_grid_scenario(df_copy, df_baseline_damages_copy, df_mp_damages_copy, menu_mp, equipment_specs, policy_scenario, interest_rate):
    """
    Calculate the NPV of climate, health, and public damages over the equipment's lifetime
    under different grid decarbonization scenarios.

    Parameters:
    - df_copy (DataFrame): A copy of the original DataFrame to store NPV calculations.
    - menu_mp (str): Menu identifier used in column names.
    - equipment_specs (dict): Dictionary containing lifetimes for each equipment category.
    - policy_scenario (str): Specifies the grid policy_scenario ('No Inflation Reduction Act', 'AEO2023 Reference Case').
    - interest_rate (float): Discount rate for NPV calculation.

    Returns:
    - DataFrame: A DataFrame containing the calculated NPV values for each category.
    """
    # Determine the policy_scenario prefix based on the policy policy_scenario
    if policy_scenario == 'No Inflation Reduction Act':
        scenario_prefix = f"preIRA_mp{menu_mp}_"
    elif policy_scenario == 'AEO2023 Reference Case':
        scenario_prefix = f"iraRef_mp{menu_mp}_"
    else:
        raise ValueError("Invalid Policy policy_scenario! Please choose from 'No Inflation Reduction Act' or 'AEO2023 Reference Case'.")
    
    # Create a DataFrame to hold the NPV calculations
    npv_columns = {}
    
    for category, lifetime in equipment_specs.items():
        print(f"""\nCalculating Public NPV for {category}...
            lifetime: {lifetime}, interest_rate: {interest_rate}, policy_scenario: {policy_scenario}""")
        # For LRMER and SRMER
        for mer_type in ['lrmer', 'srmer']:
            print(f"Type of Marginal Emissions Rate Factor: {mer_type}")           
            # Initialize NPV columns for each category
            climate_npv_key = f'{scenario_prefix}{category}_climate_npv_{mer_type}'
            health_npv_key = f'{scenario_prefix}{category}_health_npv'
            public_npv_key = f'{scenario_prefix}{category}_public_npv_{mer_type}'
            
            # Initialize NPV columns in the dictionary if they don't exist
            npv_columns[climate_npv_key] = npv_columns.get(climate_npv_key, 0)
            npv_columns[health_npv_key] = npv_columns.get(health_npv_key, 0)
            npv_columns[public_npv_key] = npv_columns.get(public_npv_key, 0)
                
            for year in range(1, lifetime + 1):
                year_label = year + 2023
                
                # Base Damages for Climate and Health
                base_annual_climate_damages = df_baseline_damages_copy[f'baseline_{year_label}_{category}_damages_climate_{mer_type}']
                base_annual_health_damages = df_baseline_damages_copy[f'baseline_{year_label}_{category}_damages_health']
                base_annual_damages = base_annual_climate_damages + base_annual_health_damages
                
                # Post-Retrofit Damages for Climate and Health
                retrofit_annual_climate_damages = df_mp_damages_copy[f'{scenario_prefix}{year_label}_{category}_damages_climate_{mer_type}']
                retrofit_annual_health_damages = df_mp_damages_copy[f'{scenario_prefix}{year_label}_{category}_damages_health']
                retrofit_annual_damages = retrofit_annual_climate_damages + retrofit_annual_health_damages

                # Apply the discount factor to each year's damages
                discount_factor = 1 / ((1 + interest_rate) ** year)
    
                npv_columns[climate_npv_key] += ((base_annual_climate_damages - retrofit_annual_climate_damages) * discount_factor).round(2)
                npv_columns[health_npv_key] += ((base_annual_health_damages - retrofit_annual_health_damages) * discount_factor).round(2)
                npv_columns[public_npv_key] += ((base_annual_damages - retrofit_annual_damages) * discount_factor).round(2)

    # Convert the dictionary to a DataFrame and return it
    df_npv = pd.DataFrame(npv_columns, index=df_copy.index)
    return df_npv