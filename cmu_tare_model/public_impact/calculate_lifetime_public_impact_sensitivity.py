import pandas as pd
from typing import Dict, Tuple, List

# Constants
from cmu_tare_model.constants import EQUIPMENT_SPECS, CR_FUNCTIONS, RCM_MODELS, SCC_ASSUMPTIONS
from cmu_tare_model.utils.discounting import calculate_discount_factor
from cmu_tare_model.utils.modeling_params import define_scenario_params
from cmu_tare_model.utils.data_validation.retrofit_status_utils import (
    create_retrofit_only_series,
    replace_small_values_with_nan,
    update_values_for_retrofits
)
from cmu_tare_model.utils.data_validation.data_quality_utils import (
    mask_category_specific_data,
    get_valid_calculation_mask
)
from cmu_tare_model.public_impact.data_processing.validate_damages_dataframes import validate_damage_dataframes

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

# LAST UPDATED APRIL 13, 2025 @ 12:30PM
def calculate_public_npv(
    df: pd.DataFrame, 
    df_baseline_climate: pd.DataFrame, 
    df_baseline_health: pd.DataFrame,
    df_mp_climate: pd.DataFrame,
    df_mp_health: pd.DataFrame,
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
        df_baseline_climate: DataFrame containing baseline climate damage projections.
        df_baseline_health: DataFrame containing baseline health damage projections.
        df_mp_climate: DataFrame containing post-retrofit climate damage projections.
        df_mp_health: DataFrame containing post-retrofit health damage projections.
        menu_mp: Menu identifier used to construct column names for the measure package.
        policy_scenario: Policy scenario that determines electricity grid projections.
            Accepted values: 'No Inflation Reduction Act', 'AEO2023 Reference Case'.
        rcm_model: The Reduced Complexity Model used for health impact calculations. Loops through CR_FUNCTIONS.
        base_year: The base year for discounting calculations. Default is 2024.
        discounting_method: The method used for discounting. Default is 'public'.

    Returns:
        DataFrame with additional columns containing the calculated public NPVs for each 
        equipment category, damage type, and sensitivity analysis combination.
        
    Raises:
        ValueError: If input parameters are invalid or if required data columns are missing.
    """
    print("\n" + "="*50)
    print("VALIDATING INPUT PARAMETERS")
    print("="*50)
    
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
    
    # Add validation for rcm_model
    if rcm_model not in RCM_MODELS:
        raise ValueError(f"Invalid rcm_model: {rcm_model}. Must be one of {RCM_MODELS}")
    
    print("✓ All input parameters are valid.")
    
    print("\n" + "="*50)
    print("VALIDATING INPUT DATA STRUCTURE")
    print("="*50)
    
    # Validate input DataFrames have the expected structure
    is_valid, messages = validate_damage_dataframes(
        df_baseline_climate,
        df_baseline_health,
        df_mp_climate,
        df_mp_health,
        menu_mp, 
        policy_scenario, 
        base_year, 
        EQUIPMENT_SPECS
    )
    
    # Print any validation messages
    for message in messages:
        print(message)
    
    if not is_valid:
        raise ValueError("Input DataFrames are missing required damage columns. See errors above.")
    
    print("✓ Input data structure validation passed.\n")

    # Create copies to avoid modifying original dataframes
    df_copy = df.copy()
    df_baseline_climate_copy = df_baseline_climate.copy()
    df_baseline_health_copy = df_baseline_health.copy()
    df_mp_climate_copy = df_mp_climate.copy()
    df_mp_health_copy = df_mp_health.copy()
    
    # Track all new dataframes that will be joined at the end
    all_new_dfs = []

    # Initialize dictionary to track columns for masking verification
    all_columns_to_mask = {category: [] for category in EQUIPMENT_SPECS}

    print("\n" + "="*50)
    print(f"CALCULATING NPV FOR EACH CONCENTRATION-RESPONSE FUNCTION")
    print("="*50)
    
    for cr_function in CR_FUNCTIONS:
        print(f"\nProcessing CR Function: {cr_function}")
        # Calculate the lifetime damages and corresponding NPV based on the policy scenario
        new_columns_dict = calculate_lifetime_damages_grid_scenario(
            df_copy=df_copy, 
            df_baseline_climate=df_baseline_climate_copy,
            df_baseline_health=df_baseline_health_copy,
            df_mp_climate=df_mp_climate_copy,
            df_mp_health=df_mp_health_copy,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario, 
            rcm_model=rcm_model, 
            cr_function=cr_function, 
            base_year=base_year, 
            discounting_method=discounting_method,
            all_columns_to_mask=all_columns_to_mask
        )
        
        # Convert to DataFrame and store for later joining
        df_new_columns = pd.DataFrame(new_columns_dict, index=df_copy.index)
        all_new_dfs.append(df_new_columns)

    # Combine all new DataFrames
    if all_new_dfs:
        df_combined_new = pd.concat(all_new_dfs, axis=1)
        
        # Apply final verification masking for consistency
        print("\nVerifying masking for all calculated columns:")
        for category, cols_to_mask in all_columns_to_mask.items():
            # Filter out columns that don't exist in the combined DataFrame
            combined_cols = [col for col in cols_to_mask if col in df_combined_new.columns]
            
            if combined_cols:
                # Copy inclusion flags to df_combined_new
                validation_prefixes = ["include_", "valid_tech", "valid_fuel"]
                validation_cols = []
                for prefix in validation_prefixes:
                    validation_cols.extend([col for col in df_copy.columns if col.startswith(prefix)])
                    
                # Ensure all validation columns exist in df_combined_new
                for col in validation_cols:
                    if col not in df_combined_new.columns:
                        df_combined_new[col] = df_copy[col]
                        
                df_combined_new = mask_category_specific_data(df_combined_new, combined_cols, category)
        
        # Drop any overlapping columns from df_copy
        overlapping_columns = df_combined_new.columns.intersection(df_copy.columns)
        if not overlapping_columns.empty:
            print(f"Dropping {len(overlapping_columns)} overlapping columns from the original DataFrame.")
            df_copy.drop(columns=overlapping_columns, inplace=True)
        
        # Merge new columns into the original DataFrame
        df_copy = df_copy.join(df_combined_new, how='left')
    else:
        print("WARNING: No new NPV columns were calculated.")

    print("\n" + "="*50)
    print("NPV CALCULATION COMPLETED")
    print("="*50)
    print(f"Added {len(df_copy.columns) - len(df.columns)} new NPV columns to the DataFrame.")
        
    return df_copy

def calculate_lifetime_damages_grid_scenario(
    df_copy: pd.DataFrame, 
    df_baseline_climate: pd.DataFrame,
    df_baseline_health: pd.DataFrame,
    df_mp_climate: pd.DataFrame,
    df_mp_health: pd.DataFrame,
    menu_mp: str, 
    policy_scenario: str, 
    rcm_model: str,
    cr_function: str,
    base_year: int = 2024,
    discounting_method: str = 'public',
    all_columns_to_mask: Dict[str, List[str]] = None
) -> Dict[str, pd.Series]:
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
        df_baseline_climate: DataFrame containing baseline climate damage projections.
        df_baseline_health: DataFrame containing baseline health damage projections.
        df_mp_climate: DataFrame containing post-retrofit climate damage projections.
        df_mp_health: DataFrame containing post-retrofit health damage projections.
        menu_mp: Menu identifier used to construct column names for the measure package.
        policy_scenario: Specifies the grid scenario ('No Inflation Reduction Act', 'AEO2023 Reference Case').
        rcm_model: The Reduced Complexity Model used for health impact calculations.
        cr_function: The Concentration-Response function used for health impact calculations.
        base_year: The base year for discounting calculations. Default is 2024.
        discounting_method: The method used for discounting. Default is 'public'.
        all_columns_to_mask: Dictionary to track columns for masking verification by category.
                     Keys are equipment categories and values are lists of column names.

    Returns:
        Dictionary where keys are column names (str) and values are pd.Series representing 
        calculated NPV values for each category, damage type, and sensitivity combination.
        
    Raises:
        ValueError: Indirectly through define_scenario_params if policy_scenario is invalid.
        ValueError: If required columns are missing from input DataFrames.
    """
    # Initialize the masking dictionary if None is provided
    if all_columns_to_mask is None:
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
            
            # Determine which homes have valid data for this category
            valid_mask = get_valid_calculation_mask(df_copy, category, menu_mp)

            # Track columns for this category
            category_columns_to_mask = []

            # Initialize NPV series with zeros for homes with valid data, NaN for others
            climate_npv = create_retrofit_only_series(df_copy, valid_mask)
            health_npv = create_retrofit_only_series(df_copy, valid_mask)

            # Track if any year's data was successfully processed
            climate_years_processed = 0
            health_years_processed = 0
            
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
                
                # Check if climate columns exist before calculation
                climate_cols_exist = (base_climate_col in df_baseline_climate.columns and 
                                     retrofit_climate_col in df_mp_climate.columns)
                
                # Check if health columns exist before calculation
                health_cols_exist = (base_health_col in df_baseline_health.columns and 
                                    retrofit_health_col in df_mp_health.columns)
                
                # Calculate avoided climate damages if columns exist
                if climate_cols_exist:
                    # Calculate avoided damages for this year
                    avoided_climate = ((df_baseline_climate[base_climate_col] - 
                                    df_mp_climate[retrofit_climate_col]) * discount_factor)
                    
                    # Update NPV only for homes with valid data
                    climate_npv = update_values_for_retrofits(
                        climate_npv, avoided_climate, valid_mask, menu_mp)
                    climate_years_processed += 1

                else:
                    # Print clear warning about missing columns
                    print(f"  WARNING: Cannot calculate climate NPV for year {year_label}.")
                    print(f"    Missing column(s): "
                          f"{'base_climate_col' if base_climate_col not in df_baseline_climate.columns else ''}"
                          f"{' and ' if (base_climate_col not in df_baseline_climate.columns and retrofit_climate_col not in df_mp_climate.columns) else ''}"
                          f"{'retrofit_climate_col' if retrofit_climate_col not in df_mp_climate.columns else ''}")
                
                # Calculate avoided health damages if columns exist
                if health_cols_exist:
                    # Calculate avoided damages for this year
                    avoided_health = ((df_baseline_health[base_health_col] - 
                                    df_mp_health[retrofit_health_col]) * discount_factor)
                    
                    # Update NPV only for homes with valid data
                    health_npv = update_values_for_retrofits(
                        health_npv, avoided_health, valid_mask, menu_mp)
                    health_years_processed += 1

                else:
                    # Print clear warning about missing columns
                    print(f"  WARNING: Cannot calculate health NPV for year {year_label}.")
                    print(f"    Missing column(s): "
                          f"{'base_health_col' if base_health_col not in df_baseline_health.columns else ''}"
                          f"{' and ' if (base_health_col not in df_baseline_health.columns and retrofit_health_col not in df_mp_health.columns) else ''}"
                          f"{'retrofit_health_col' if retrofit_health_col not in df_mp_health.columns else ''}")
            
            # Replace tiny values with NaN to avoid numerical artifacts
            climate_npv = replace_small_values_with_nan(climate_npv)
            health_npv = replace_small_values_with_nan(health_npv)

            # Store the unrounded values for final calculation
            climate_npv_unrounded = climate_npv.copy()
            health_npv_unrounded = health_npv.copy()

            # Check if any data was processed
            if climate_years_processed == 0:
                print(f"ERROR: No climate damage data was found for {category}. NPV will be zero.")
            elif climate_years_processed < lifetime:
                print(f"WARNING: Only processed {climate_years_processed}/{lifetime} years for climate damages.")
                
            if health_years_processed == 0:
                print(f"ERROR: No health damage data was found for {category}. NPV will be zero.")
            elif health_years_processed < lifetime:
                print(f"WARNING: Only processed {health_years_processed}/{lifetime} years for health damages.")
            
            # Round values for display/storage
            climate_npv = climate_npv_unrounded.round(2)
            health_npv = health_npv_unrounded.round(2)

            # Calculate public NPV from sum of climate and health NPVs (unrounded values), then round
            public_npv = (climate_npv_unrounded + health_npv_unrounded).round(2)
            
            # Check for zero NPVs and warn
            if (climate_npv == 0).all():
                print(f"WARNING: {climate_npv_key} has all zero values. This may indicate missing data.")
            if (health_npv == 0).all():
                print(f"WARNING: {health_npv_key} has all zero values. This may indicate missing data.")
            if (public_npv == 0).all():
                print(f"WARNING: {public_npv_key} has all zero values. This may indicate missing data.")
            
            # Store NPVs in the results dictionary
            all_npvs[climate_npv_key] = climate_npv
            all_npvs[health_npv_key] = health_npv
            all_npvs[public_npv_key] = public_npv

            # Track NPV columns for masking verification
            category_columns_to_mask.extend([climate_npv_key, health_npv_key, public_npv_key])
                        
            # Add all columns for this category to the masking dictionary
            all_columns_to_mask[category].extend(category_columns_to_mask)
                        
            print(f"Calculated Public NPV: {public_npv_key}")
    
    # Return the dictionary of Series (don't convert to DataFrame here)
    return all_npvs
