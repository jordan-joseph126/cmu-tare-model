import pandas as pd
from typing import Dict

# Constants
from cmu_tare_model.constants import EQUIPMENT_SPECS, CR_FUNCTIONS, RCM_MODELS, SCC_ASSUMPTIONS
from cmu_tare_model.utils.discounting import calculate_discount_factor
from cmu_tare_model.utils.modeling_params import define_scenario_params

print(f"""
===================================================================================
LIFETIME PUBLIC IMPACT: NPV OF LIFETIME MONETIZED DAMAGES (CLIMATE AND HEALTH)
===================================================================================
- CLIMATE: 
    SCC_ASSUMPTIONS = {SCC_ASSUMPTIONS}
- HEALTH: 
    RCM_MODELS = {RCM_MODELS}
    CR_FUNCTIONS = {CR_FUNCTIONS}
""")

# LAST UPDATED APRIL 6, 2025 @ 2:30PM
def calculate_public_npv(
    df: pd.DataFrame, 
    df_baseline_damages: pd.DataFrame, 
    df_mp_damages: pd.DataFrame, 
    menu_mp: str, 
    policy_scenario: str, 
    rcm_model: str,
    base_year: int = 2024,
    discounting_method: str = 'public',
) -> pd.DataFrame:
    """
    Calculate the public Net Present Value (NPV) for specific categories of damages,
    considering different policy scenarios related to grid decarbonization.
    
    The function compares baseline damages with post-measure package (mp) damages
    to determine the avoided damages (benefits) from implementing retrofits.

    Args:
        df: Input DataFrame containing base data for calculations.
        df_baseline_damages: DataFrame containing baseline damage projections.
        df_mp_damages: DataFrame containing post-retrofit damage projections.
        menu_mp: Menu identifier used to construct column names for the measure package.
        policy_scenario: Policy scenario that determines electricity grid projections.
            Accepted values: 'No Inflation Reduction Act', 'AEO2023 Reference Case'.
        rcm_model: The Reduced Complexity Model used for health impact calculations. Loops through CR_FUNCTIONS.
        base_year: The base year for discounting calculations. Default is 2024.
        discounting_method: The method used for discounting. Default is 'public'.

    Returns:
        DataFrame with additional columns containing the calculated public NPVs for each 
        equipment category, damage type, and sensitivity analysis combination.
    """
    # Add validation for menu_mp
    if not menu_mp.isdigit():
        raise ValueError(f"Invalid menu_mp: {menu_mp}. Must be a digit.")
        
    # Add validation for policy_scenario
    valid_scenarios = ['No Inflation Reduction Act', 'AEO2023 Reference Case']
    if policy_scenario not in valid_scenarios:
        raise ValueError(f"Invalid policy_scenario: {policy_scenario}. Must be one of {valid_scenarios}")
        
    # Add validation for discounting_method
    valid_methods = ['public', 'private_fixed']
    if discounting_method not in valid_methods:
        raise ValueError(f"Invalid discounting_method: {discounting_method}. Must be one of {valid_methods}")

    # Create copies to avoid modifying original dataframes
    df_copy = df.copy()
    df_baseline_damages_copy = df_baseline_damages.copy()
    df_mp_damages_copy = df_mp_damages.copy()

    for cr_function in CR_FUNCTIONS:
        # Calculate the lifetime damages and corresponding NPV based on the policy scenario
        df_new_columns = calculate_lifetime_damages_grid_scenario(
            df_copy, 
            df_baseline_damages_copy, 
            df_mp_damages_copy, 
            menu_mp,
            policy_scenario, 
            rcm_model, 
            cr_function, 
            base_year, 
            discounting_method
        )

    # Drop any overlapping columns from df_copy
    overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)
    if not overlapping_columns.empty:
        df_copy.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into the original DataFrame
    df_copy = df_copy.join(df_new_columns, how='left')

    return df_copy

def calculate_lifetime_damages_grid_scenario(
    df_copy: pd.DataFrame, 
    df_baseline_damages_copy: pd.DataFrame, 
    df_mp_damages_copy: pd.DataFrame, 
    menu_mp: str, 
    policy_scenario: str, 
    rcm_model: str,
    cr_function: str,
    base_year: int = 2024,
    discounting_method: str = 'public',
) -> pd.DataFrame:
    """
    Calculate the NPV of climate, health, and public damages over the equipment's lifetime
    under different grid decarbonization scenarios.
    
    This function performs sensitivity analysis across multiple dimensions:
    - Equipment categories (with varying lifetimes)
    - Social Cost of Carbon (SCC) assumptions for climate damages
    - Reduced Complexity Models (rcm_model) for health impacts
    - Concentration-Response (cr_function) functions for health impacts

    Args:
        df_copy: Copy of the original DataFrame to use for index alignment.
        df_baseline_damages_copy: DataFrame containing baseline damage projections.
        df_mp_damages_copy: DataFrame containing post-retrofit damage projections.
        menu_mp: Menu identifier used to construct column names for the measure package.
        policy_scenario: Specifies the grid scenario ('No Inflation Reduction Act', 'AEO2023 Reference Case').
        rcm_model: The Reduced Complexity Model used for health impact calculations.
        cr_function: The Concentration-Response function used for health impact calculations.
        base_year: The base year for discounting calculations. Default is 2024.
        discounting_method: The method used for discounting. Default is 'public'.

    Returns:
        DataFrame containing the calculated NPV values for each category, damage type,
        and sensitivity combination, using the index from df_copy.
        
    Raises:
        ValueError: Indirectly through define_scenario_params if policy_scenario is invalid.
    """
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
    
    # Initialize a dictionary to store all NPV results
    # Each key will be a column name, and each value will be a pandas Series of NPVs
    all_npvs: Dict[str, pd.Series] = {}
    
    # Process each equipment category
    for category, lifetime in EQUIPMENT_SPECS.items():
        print(f"""\nCalculating Public NPV for {category}...
            lifetime: {lifetime}, discounting_method: {discounting_method}, policy_scenario: {policy_scenario}""")
        
        # Process each SCC assumption for climate damages
        for scc in SCC_ASSUMPTIONS:
            print(f"""\n --- Public NPV Sensitivity ---
                  Climate Impact Sensitivity:
                    SCC Assumption (Bound): {scc}

                  Health Impact Sensitivity:
                    rcm_model Model: {rcm_model} | cr_function Function: {cr_function}""")
            
            # Define column names for NPV results
            climate_npv_key = f'{scenario_prefix}{category}_climate_npv_{scc}'
            health_npv_key = f'{scenario_prefix}{category}_health_npv_{rcm_model}_{cr_function}'
            public_npv_key = f'{scenario_prefix}{category}_public_npv_{scc}_{rcm_model}_{cr_function}'
            
            # Initialize NPV series with zeros
            climate_npv = pd.Series(0.0, index=df_copy.index)
            health_npv = pd.Series(0.0, index=df_copy.index)
            
            # Calculate NPVs for each year in the equipment's lifetime
            for year in range(1, lifetime + 1):
                year_label = year + (base_year - 1)
                discount_factor = discount_factors[year_label]
                
                # Get column names for baseline damages
                base_climate_col = f'baseline_{year_label}_{category}_damages_climate_lrmer_{scc}'
                base_health_col = f'baseline_{year_label}_{category}_damages_health_{rcm_model}_{cr_function}'
                
                # Get column names for retrofit damages
                retrofit_climate_col = f'{scenario_prefix}{year_label}_{category}_damages_climate_lrmer_{scc}'
                retrofit_health_col = f'{scenario_prefix}{year_label}_{category}_damages_health_{rcm_model}_{cr_function}'
                
                # Calculate avoided damages (baseline - retrofit) and apply discount factor
                if base_climate_col in df_baseline_damages_copy and retrofit_climate_col in df_mp_damages_copy:
                    climate_npv += ((df_baseline_damages_copy[base_climate_col] - 
                                    df_mp_damages_copy[retrofit_climate_col]) * discount_factor)
                
                if base_health_col in df_baseline_damages_copy and retrofit_health_col in df_mp_damages_copy:
                    health_npv += ((df_baseline_damages_copy[base_health_col] - 
                                  df_mp_damages_copy[retrofit_health_col]) * discount_factor)
            
            # First, store the unrounded values
            climate_npv_unrounded = climate_npv.copy()
            health_npv_unrounded = health_npv.copy()

            # Then round for display/storage
            climate_npv = climate_npv_unrounded.round(2)
            health_npv = health_npv_unrounded.round(2)

            # Calculate public NPV from sum of climate and health NPVs (unrounded values), then round
            public_npv = (climate_npv_unrounded + health_npv_unrounded).round(2)
            
            # Store NPVs in the results dictionary
            all_npvs[climate_npv_key] = climate_npv
            all_npvs[health_npv_key] = health_npv
            all_npvs[public_npv_key] = public_npv
            
            print(f"Calculated Public NPV: {public_npv_key}")
    
    # Convert the dictionary of Series to a DataFrame
    df_npv = pd.DataFrame(all_npvs)
    return df_npv
