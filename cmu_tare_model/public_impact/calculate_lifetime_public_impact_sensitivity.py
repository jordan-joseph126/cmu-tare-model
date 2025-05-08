import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional

# Constants
from cmu_tare_model.constants import EQUIPMENT_SPECS, CR_FUNCTIONS, RCM_MODELS, SCC_ASSUMPTIONS
from cmu_tare_model.utils.discounting import calculate_discount_factor
from cmu_tare_model.utils.modeling_params import define_scenario_params
from cmu_tare_model.utils.validation_framework import (
    calculate_avoided_values,
    create_retrofit_only_series,
    initialize_validation_tracking,
    replace_small_values_with_nan,
    # apply_final_masking
)
from cmu_tare_model.utils.calculation_utils import (
    validate_common_parameters,
    apply_temporary_validation_and_mask
)
from cmu_tare_model.public_impact.data_processing.validate_damages_dataframes import validate_damage_dataframes

def calculate_public_npv(
    df: pd.DataFrame, 
    df_baseline_climate: pd.DataFrame, 
    df_baseline_health: pd.DataFrame,
    df_mp_climate: pd.DataFrame,
    df_mp_health: pd.DataFrame,
    menu_mp: int, 
    policy_scenario: str, 
    rcm_model: str,
    base_year: int = 2024,
    discounting_method: str = 'public',
    verbose: bool = False
) -> pd.DataFrame:
    """
    Calculate the public Net Present Value (NPV) for specific categories of damages,
    considering different policy scenarios related to grid decarbonization.

    The function compares baseline damages with post-measure package (mp) damages
    to determine the avoided damages (benefits) from implementing retrofits.

    This function follows the five-step validation framework:
    1. Mask Initialization: Identifies valid homes using inclusion flags and retrofit status
    2. Series Initialization: Creates result series with zeros for valid homes, NaN for others
    3. Valid-Only Calculation: Performs calculations only for valid homes
    4. Valid-Only Updates: Uses list-based collection of yearly values instead of incremental updates
    5. Final Masking: Applies consistent masking to all result columns

    The list-based collection approach stores yearly values in lists and sums them using pandas
    vectorized operations after all years have been processed. This approach prevents accumulation
    errors that can occur with incremental updates.

    Args:
        df (pd.DataFrame): Input DataFrame containing base data for calculations.
        df_baseline_climate (pd.DataFrame): DataFrame containing baseline climate damage projections.
        df_baseline_health (pd.DataFrame): DataFrame containing baseline health damage projections.
        df_mp_climate (pd.DataFrame): DataFrame containing post-retrofit climate damage projections.
        df_mp_health (pd.DataFrame): DataFrame containing post-retrofit health damage projections.
        menu_mp (int): Menu identifier used to construct column names for the measure package.
        policy_scenario (str): Policy scenario that determines electricity grid projections.
            Accepted values: 'No Inflation Reduction Act', 'AEO2023 Reference Case'.
        rcm_model (str): The Reduced Complexity Model used for health impact calculations.
        base_year (int, optional): The base year for discounting calculations. Default is 2024.
        discounting_method (str, optional): The method used for discounting. Default is 'public'.
        verbose (bool, optional): Whether to print detailed progress messages. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with additional columns containing the calculated public NPVs for each 
        equipment category, damage type, and sensitivity analysis combination.
        
    Raises:
        ValueError: If input parameters are invalid or if required data columns are missing.
        RuntimeError: If processing fails at the category or combination level.
    """
    # ===== STEP 0: Validate input parameters =====
    menu_mp, policy_scenario, discounting_method = validate_common_parameters(
        menu_mp, policy_scenario, discounting_method)
    
    # Validate RCM model
    if rcm_model not in RCM_MODELS:
        raise ValueError(f"Invalid rcm_model: {rcm_model}. Must be one of {RCM_MODELS}")
    
    # Validate input data structure
    if verbose:
        print("\nValidating input data structure...")
    
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
    if verbose and messages:
        for message in messages:
            print(message)
    
    if not is_valid:
        raise ValueError("Input DataFrames are missing required damage columns. See errors above.")
    
    if verbose:
        print("✓ Input data validation passed.")

    # Create copies to avoid modifying original dataframes
    df_copy = df.copy()
    df_baseline_climate_copy = df_baseline_climate.copy()
    df_baseline_health_copy = df_baseline_health.copy()
    df_mp_climate_copy = df_mp_climate.copy()
    df_mp_health_copy = df_mp_health.copy()
    
    # Initialize dictionary to track columns for masking verification
    all_columns_to_mask = {category: [] for category in EQUIPMENT_SPECS}
    df_new_columns = pd.DataFrame(index=df_copy.index)

    if verbose:
        print("\nCalculating NPV for each concentration-response function...")
    
    for cr_function in CR_FUNCTIONS:
        if verbose:
            print(f"\nProcessing CR Function: {cr_function}")
        
        try:
            # Calculate the lifetime damages and corresponding NPV 
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
                all_columns_to_mask=all_columns_to_mask,
                verbose=verbose
            )

            # Collect NPV results in a dictionary first
            if new_columns_dict:
                # Create a temporary DataFrame from the collected columns
                temp_df = pd.DataFrame(new_columns_dict, index=df_copy.index)
                # Add all columns at once with concat
                df_new_columns = pd.concat([df_new_columns, temp_df], axis=1)

        except Exception as e:
            raise RuntimeError(f"Error processing CR function '{cr_function}': {e}")

    # STEP 5: Apply final masking to all new columns
    if verbose:
        print("\nApplying final masking to calculated columns...")
        
    df_result = apply_temporary_validation_and_mask(df_copy, df_new_columns, all_columns_to_mask, verbose=verbose)
    
    if verbose:
        print(f"\nNPV calculation completed. Added {len(df_new_columns.columns)} new columns.")
        
    return df_result

def calculate_lifetime_damages_grid_scenario(
    df_copy: pd.DataFrame, 
    df_baseline_climate: pd.DataFrame,
    df_baseline_health: pd.DataFrame,
    df_mp_climate: pd.DataFrame,
    df_mp_health: pd.DataFrame,
    menu_mp: int, 
    policy_scenario: str, 
    rcm_model: str,
    cr_function: str,
    base_year: int = 2024,
    discounting_method: str = 'public',
    all_columns_to_mask: Optional[Dict[str, List[str]]] = None,
    verbose: bool = False
) -> Dict[str, pd.Series]:
    """
    Calculate the NPV of climate, health, and public damages over the equipment's lifetime.
    
    This function performs sensitivity analysis across multiple dimensions:
    - Equipment categories (with varying lifetimes)
    - Social Cost of Carbon (SCC) assumptions for climate damages
    - Health impacts based on specified RCM model and C-R function

    Args:
        df_copy: Copy of the original DataFrame to use for calculations.
        df_baseline_climate: DataFrame containing baseline climate damage projections.
        df_baseline_health: DataFrame containing baseline health damage projections.
        df_mp_climate: DataFrame containing post-retrofit climate damage projections.
        df_mp_health: DataFrame containing post-retrofit health damage projections.
        menu_mp: Menu identifier used to construct column names for the measure package.
        policy_scenario: Specifies the grid scenario.
        rcm_model: The Reduced Complexity Model used for health impact calculations.
        cr_function: The Concentration-Response function used for health impact calculations.
        base_year: The base year for discounting calculations. Default is 2024.
        discounting_method: The method used for discounting. Default is 'public'.
        all_columns_to_mask: Dictionary to track columns for masking verification by category.
        verbose: Whether to print detailed progress messages.

    Returns:
        Dictionary mapping column names to Series of calculated NPV values for 
        each category, damage type, and sensitivity combination.
        
    Raises:
        ValueError: If required columns are missing from input DataFrames.
        RuntimeError: If processing fails for a specific category or SCC assumption.
    """
    # Initialize the masking dictionary if None is provided
    if all_columns_to_mask is None:
        all_columns_to_mask = {category: [] for category in EQUIPMENT_SPECS}

    # Determine the scenario prefix based on the policy scenario
    scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)
    
    # Pre-calculate discount factors for efficiency
    discount_factors: Dict[int, float] = {}
    max_lifetime = max(EQUIPMENT_SPECS.values())
    for year in range(1, max_lifetime + 1):
        year_label = year + (base_year - 1)
        discount_factors[year_label] = calculate_discount_factor(base_year, year_label, discounting_method)
    
    # Initialize a dictionary to store all NPV results
    all_npvs: Dict[str, pd.Series] = {}
    
    # Process each equipment category
    for category, lifetime in EQUIPMENT_SPECS.items():
        if verbose:
            print(f"  Calculating Public NPV for {category}...")
        
        # Process each SCC assumption for climate damages
        for scc in SCC_ASSUMPTIONS:
            try:
                if verbose:
                    print(f"    SCC: {scc}, RCM: {rcm_model}, C-R: {cr_function}")
                
                # Define column names for NPV results
                climate_npv_key = f'{scenario_prefix}{category}_climate_npv_{scc}'
                health_npv_key = f'{scenario_prefix}{category}_health_npv_{rcm_model}_{cr_function}'
                public_npv_key = f'{scenario_prefix}{category}_public_npv_{scc}_{rcm_model}_{cr_function}'
                
                # ===== STEP 1: Initialize validation tracking =====
                df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
                    df_copy, category, menu_mp, verbose=False)

                # ===== STEP 2: Initialize result series with template =====
                # Use create_retrofit_only_series to properly initialize with zeros for valid homes, NaN for others
                climate_npv_template = create_retrofit_only_series(df_copy, valid_mask)
                health_npv_template = create_retrofit_only_series(df_copy, valid_mask)

                # Create lists to store yearly avoided damages
                yearly_climate_avoided = []
                yearly_health_avoided = []

                # Track if any year's data was successfully processed
                climate_years_processed = 0
                health_years_processed = 0
                            
                # ===== STEP 3: Valid-Only Calculation =====
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
                                        
                    # ===== STEP 4: Valid-Only Updates =====
                    # NOTES: WE ARE DISCOUNTING THE AVOIDED DAMAGES HERE. USING MARGINAL SOCIAL COSTS SO MAY NOT BE NECESSARY.
                    # Calculate avoided climate damages if columns exist (store in list instead of incremental update)
                    if climate_cols_exist:
                        # Use calculate_avoided_values function for consistency 
                        avoided_climate = calculate_avoided_values(
                            baseline_values=df_baseline_climate[base_climate_col],
                            measure_values=df_mp_climate[retrofit_climate_col],
                            retrofit_mask=(valid_mask if menu_mp != 0 else None)
                        ) * discount_factor
                    
                        yearly_climate_avoided.append(avoided_climate)
                        climate_years_processed += 1
                    elif verbose:
                        print(f"    Warning: Climate data missing for year {year_label}")
                    
                    # Calculate avoided health damages if columns exist (store in list instead of incremental update)
                    if health_cols_exist:
                        # Use calculate_avoided_values function for consistency 
                        avoided_health = calculate_avoided_values(
                            baseline_values=df_baseline_health[base_health_col],
                            measure_values=df_mp_health[retrofit_health_col],
                            retrofit_mask=(valid_mask if menu_mp != 0 else None)
                        ) * discount_factor
                    
                        yearly_health_avoided.append(avoided_health)
                        health_years_processed += 1
                    elif verbose:
                        print(f"    Warning: Health data missing for year {year_label}")

                # Sum up all yearly avoided damages using pandas operations
                if yearly_climate_avoided:
                    # Convert list of Series to DataFrame and sum
                    climate_df = pd.concat(yearly_climate_avoided, axis=1)
                    climate_npv = climate_df.sum(axis=1)
                    
                    # Apply validation mask for measure packages
                    if menu_mp != 0:
                        climate_npv = pd.Series(
                            np.where(valid_mask, climate_npv, np.nan),
                            index=climate_npv.index
                        )
                else:
                    climate_npv = climate_npv_template

                if yearly_health_avoided:
                    # Convert list of Series to DataFrame and sum
                    health_df = pd.concat(yearly_health_avoided, axis=1)
                    health_npv = health_df.sum(axis=1)
                    
                    # Apply validation mask for measure packages
                    if menu_mp != 0:
                        health_npv = pd.Series(
                            np.where(valid_mask, health_npv, np.nan),
                            index=health_npv.index
                        )
                else:
                    health_npv = health_npv_template

                # Replace tiny values with NaN to avoid numerical artifacts
                climate_npv = replace_small_values_with_nan(climate_npv)
                health_npv = replace_small_values_with_nan(health_npv)

                # Store the unrounded values for final calculation
                climate_npv_unrounded = climate_npv.copy()
                health_npv_unrounded = health_npv.copy()

                # Check if any data was processed
                if verbose:
                    if climate_years_processed == 0:
                        print(f"    Warning: No climate data found for {category}")
                    elif climate_years_processed < lifetime:
                        print(f"    Warning: Only processed {climate_years_processed}/{lifetime} years for climate")
                        
                    if health_years_processed == 0:
                        print(f"    Warning: No health data found for {category}")
                    elif health_years_processed < lifetime:
                        print(f"    Warning: Only processed {health_years_processed}/{lifetime} years for health")
                
                # Round values for display/storage
                climate_npv = climate_npv_unrounded.round(2)
                health_npv = health_npv_unrounded.round(2)

                # Calculate public NPV from sum of climate and health NPVs (unrounded values), then round
                public_npv = (climate_npv_unrounded + health_npv_unrounded).round(2)
                
                # Store NPVs in the results dictionary
                all_npvs[climate_npv_key] = climate_npv
                all_npvs[health_npv_key] = health_npv
                all_npvs[public_npv_key] = public_npv

                # Track NPV columns for masking verification
                category_columns_to_mask.extend([climate_npv_key, health_npv_key, public_npv_key])
                
                # Add all columns for this category to the masking dictionary
                all_columns_to_mask[category].extend(category_columns_to_mask)
                
                if verbose:
                    print(f"    Completed {public_npv_key}")
                    
            except Exception as e:
                raise RuntimeError(f"Error processing {category} with SCC assumption '{scc}': {e}")
    
    # Return the dictionary of Series
    return all_npvs
