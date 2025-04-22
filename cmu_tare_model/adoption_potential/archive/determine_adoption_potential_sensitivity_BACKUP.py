import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import PROJECT_ROOT
from cmu_tare_model.constants import SCC_ASSUMPTIONS
from cmu_tare_model.utils.modeling_params import define_scenario_params


print(f"Project root directory: {PROJECT_ROOT}")

# from cmu_tare_model.functions.calculate_emissions_damages import EPA_SCC_USD2023_PER_MT

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FUNCTION: ADOPTION POTENTIAL
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# UPDATED ON APRIL 10, 2025 @ 11:45 PM
def adoption_decision(df: pd.DataFrame,
                      menu_mp: int,
                      policy_scenario: str,
                      rcm_model: str,
                      cr_function: str,
                      climate_sensitivity: bool = False, # Default is false because we use $190USD2020/mt in Joseph et al. (2025)
) -> pd.DataFrame: 
    """
    Updates the provided DataFrame with new columns that reflect decisions about equipment adoption
    and public impacts based on net present values (NPV). The function handles different scenarios
    based on input flags for incentives and grid decarbonization.

    Parameters:
        df_copy (pandas.DataFrame): The DataFrame containing home equipment data.
        policy_scenario (str): Policy policy_scenario that determines electricity grid projections.
            Accepted values: 'AEO2023 Reference Case'.
        menu_mp (int): Measure package identifier to use in column names.
        rcm_model (str): The RCM model to use for the analysis. 
            Accepted values: ['AP2', 'EASIUR', 'InMAP']
        cr_function (str): The concentration response function to use for the analysis. 
            Accepted values: ['acs', 'h6c']
        climate_sensitivity (bool): List of climate sensitivity assumptions to consider. 
            Default is ['upper'] or $190USD2020/mt as per Joseph et al. (2025).
    Returns:
        pandas.DataFrame: The modified DataFrame with additional columns for decisions and impacts.

    Notes:
        - It adds columns for both individual and public economic evaluations.
        - Adoption decisions and public impacts are dynamically calculated based on the input parameters.
    """
    df_copy = df.copy()
    
    # Define the lifetimes of different equipment categories
    upgrade_columns = {
        'heating': 'upgrade_hvac_heating_efficiency',
        'waterHeating': 'upgrade_water_heater_efficiency',
        'clothesDrying': 'upgrade_clothes_dryer',
        'cooking': 'upgrade_cooking_range'
    }
    
    df_new_columns = pd.DataFrame(index=df_copy.index)  # DataFrame to hold new or modified columns

    # Determine the scenario prefix based on the policy scenario
    scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)

    # Define the SCC assumptions
    if climate_sensitivity:
        scc_assumptions = SCC_ASSUMPTIONS
    else:
        scc_assumptions = ['upper']  # Default to upper bound for climate sensitivity as per Joseph et al. (2025)

    # Iterate over each equipment category and its respective upgrade column
    for category, upgrade_column in upgrade_columns.items():
        print(f"\nCalculating Adoption Potential for {category} under '{policy_scenario}' Scenario...")

        # Process each SCC assumption for climate damages
        for scc in scc_assumptions:
            print(f"""\n --- Public NPV Sensitivity ---
                  Climate Impact Sensitivity:
                    SCC Assumption (Bound): {scc}

                  Health Impact Sensitivity:
                    rcm_model Model: {rcm_model} | cr_function Function: {cr_function}""")

            # Create column for sensitivity analysis identification
            health_sensitivity_col = f'{scenario_prefix}{category}_health_sensitivity'
            df_new_columns[health_sensitivity_col] = f'{rcm_model}, {cr_function}'

            # Column names for net NPV, private NPV, and public NPV
            lessWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_lessWTP' # LESS WTP: BREAK EVEN ON TOTAL CAPITAL COSTS
            moreWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_moreWTP' # MORE WTP: BREAK EVEN ON NET CAPITAL COSTS (BETTER THAN ALTERNATIVE)

            public_npv_col = f'{scenario_prefix}{category}_{scc}_{rcm_model}_{cr_function}'
            rebate_col = f'mp{menu_mp}_{category}_rebate_amount'
            addition_public_benefit = f'{scenario_prefix}{category}_additional_public_benefit_{scc}'

            lessWTP_total_npv_col = f'{scenario_prefix}{category}_total_npv_lessWTP_{scc}' # LESS WTP: BREAK EVEN ON TOTAL CAPITAL COSTS
            moreWTP_total_npv_col = f'{scenario_prefix}{category}_total_npv_moreWTP_{scc}' # MORE WTP: BREAK EVEN ON NET CAPITAL COSTS (BETTER THAN ALTERNATIVE)
            # Ensure columns are numeric if they exist and convert them
            for col in [lessWTP_private_npv_col, moreWTP_private_npv_col, public_npv_col, rebate_col]:
                if col in df_copy.columns:
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                else:
                    print(f"Warning: {col} does not exist in the DataFrame.")

            # Ensure the columns are present after conversion
            if lessWTP_private_npv_col in df_copy.columns and moreWTP_private_npv_col in df_copy.columns and public_npv_col in df_copy.columns:
                # No IRA Rebate so no "Additional Public Benefit"
                if policy_scenario == 'No Inflation Reduction Act':
                    df_new_columns[addition_public_benefit] = 0.0
                else:
                    # Calculate Additional Public Benefit with IRA Rebates Accounted For and clip at 0
                    df_new_columns[addition_public_benefit] = (df_copy[public_npv_col] - df_copy[rebate_col]).clip(lower=0)
                
                # Calculate Total NPV by summing private and public NPVs
                df_new_columns[lessWTP_total_npv_col] = df_copy[lessWTP_private_npv_col] + df_copy[public_npv_col] # LESS WTP: BREAK EVEN ON TOTAL CAPITAL COSTS
                df_new_columns[moreWTP_total_npv_col] = df_copy[moreWTP_private_npv_col] + df_copy[public_npv_col] # MORE WTP: BREAK EVEN ON NET CAPITAL COSTS (BETTER THAN ALTERNATIVE)

                # Initialize columns for adoption decisions and public impact
                adoption_col_name = f'{scenario_prefix}{category}_adoption_{scc}SCC'
                retrofit_col_name = f'{scenario_prefix}{category}_retrofit_publicImpact_{scc}SCC'
                df_new_columns[adoption_col_name] = 'Tier 4: Averse'  # Default value for all rows
                df_new_columns[retrofit_col_name] = 'No Retrofit'  # Default public impact

                # Conditions for determining adoption decisions
                conditions = [
                    df_copy[upgrade_column].isna(),
                    df_copy[lessWTP_private_npv_col] > 0,
                    (df_copy[lessWTP_private_npv_col] < 0) & (df_copy[moreWTP_private_npv_col] > 0),
                    (df_copy[lessWTP_private_npv_col] < 0) & (df_copy[moreWTP_private_npv_col] < 0) & (df_new_columns[moreWTP_total_npv_col] > 0), # Ensures only Tier 3 for IRA Scenario
                ]

                choices = ['Existing Equipment', 'Tier 1: Feasible', 'Tier 2: Feasible vs. Alternative', 'Tier 3: Subsidy-Dependent Feasibility']
                df_new_columns[adoption_col_name] = np.select(conditions, choices, default='Tier 4: Averse')

                # Conditions and choices for public impacts
                public_conditions = [
                    df_copy[public_npv_col] > 0,
                    df_copy[public_npv_col] < 0
                ]
                
                public_choices = ['Public Benefit', 'Public Detriment']
                df_new_columns[retrofit_col_name] = np.select(public_conditions, public_choices, default='No Retrofit')
            else:
                print(f"Warning: One or more columns ({lessWTP_private_npv_col}, {moreWTP_private_npv_col}, {public_npv_col}) are missing or not numeric.")
    
    # Identify overlapping columns between the new and existing DataFrame.
    overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)

    # Drop overlapping columns from df_copy.
    if not overlapping_columns.empty:
        df_copy.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into df_copy, ensuring no duplicates or overwrites occur.
    df_copy = df_copy.join(df_new_columns, how='left')

    # Return the updated DataFrame.
    return df_copy
