import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from cmu_tare_model.constants import SCC_ASSUMPTIONS, RCM_MODELS, CR_FUNCTIONS, UPGRADE_COLUMNS
from cmu_tare_model.utils.modeling_params import define_scenario_params

from cmu_tare_model.utils.validation_framework import (
    create_retrofit_only_series,
    initialize_validation_tracking,
    apply_new_columns_to_dataframe,
    apply_final_masking
    )


def validate_input_parameters(
        menu_mp: int,
        policy_scenario: str,
        rcm_model: str,
        cr_function: str
) -> None:
    """
    Validates that input parameters meet expected criteria.
    
    Args:
        menu_mp: Measure package identifier.
        policy_scenario: Policy scenario name.
        rcm_model: RCM model name.
        cr_function: Concentration response function name.
        
    Raises:
        ValueError: If any parameter is invalid.
    """
    # Validate policy_scenario
    valid_scenarios = ['No Inflation Reduction Act', 'AEO2023 Reference Case']
    if policy_scenario not in valid_scenarios:
        raise ValueError(f"Invalid policy_scenario: '{policy_scenario}'. Must be one of {valid_scenarios}")
    
    # Validate rcm_model
    if rcm_model not in RCM_MODELS:
        raise ValueError(f"Invalid rcm_model: '{rcm_model}'. Must be one of {RCM_MODELS}")
    
    # Validate cr_function
    if cr_function not in CR_FUNCTIONS:
        raise ValueError(f"Invalid cr_function: '{cr_function}'. Must be one of {CR_FUNCTIONS}")
    
    # Validate menu_mp is an integer
    if not isinstance(menu_mp, int):
        try:
            int(menu_mp)  # Test if convertible to int
        except (ValueError, TypeError):
            raise ValueError(f"menu_mp must be an integer, got {type(menu_mp).__name__}: {menu_mp}")


def adoption_decision(
        df: pd.DataFrame,
        menu_mp: int,
        policy_scenario: str,
        rcm_model: str,
        cr_function: str,
        climate_sensitivity: bool = False  # Default is false because we use $190USD2020/mt in Joseph et al. (2025)
) -> pd.DataFrame:
    """
    Updates the provided DataFrame with new columns that reflect decisions about equipment adoption
    and public impacts based on net present values (NPV).
    
    This function evaluates equipment adoption potential by categorizing upgrades into four tiers
    based on economic feasibility. It also assesses public impacts (benefit or detriment) of retrofits.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing home equipment data.
        menu_mp (int): Measure package identifier to use in column names.
        policy_scenario (str): Policy scenario that determines electricity grid projections.
            Accepted values: 'AEO2023 Reference Case', 'No Inflation Reduction Act'.
        rcm_model (str): The RCM model to use for the analysis. 
            Accepted values: ['AP2', 'EASIUR', 'InMAP']
        cr_function (str): The concentration response function to use for the analysis. 
            Accepted values: ['acs', 'h6c']
        climate_sensitivity (bool): Whether to consider multiple climate sensitivity assumptions.
            Default is False, which uses only the 'upper' bound ($190USD2020/mt) as per Joseph et al. (2025).
    
    Returns:
        pandas.DataFrame: The modified DataFrame with additional columns for adoption decisions and public impacts.
    
    Raises:
        ValueError: If any input parameters are invalid or not supported.
        KeyError: If required columns are missing from the input DataFrame.
        TypeError: If column data types cannot be converted as expected.
        
    Notes:
        - The function categorizes adoption decisions into four tiers:
          - Tier 1: Feasible (positive private NPV with total capital costs)
          - Tier 2: Feasible vs. Alternative (negative NPV with total capital costs, positive NPV with net costs)
          - Tier 3: Subsidy-Dependent Feasibility (negative private NPV, positive total NPV)
          - Tier 4: Averse (negative private and total NPV)
        - Public impacts are classified as either 'Public Benefit' or 'Public Detriment'
    """
    try:
        # ========== SETUP AND VALIDATION ==========
        
        # Validate input parameters
        validate_input_parameters(menu_mp, policy_scenario, rcm_model, cr_function)
        
        # Make a copy of the input DataFrame to avoid modifying it
        df_copy = df.copy()
        
        # Create a DataFrame to hold new columns
        df_new_columns = pd.DataFrame(index=df_copy.index)
                
        # Check that required upgrade columns exist in the DataFrame
        missing_columns = [col for col in UPGRADE_COLUMNS.values() if col not in df_copy.columns]
        if missing_columns:
            raise KeyError(f"Required upgrade columns missing from DataFrame: {missing_columns}")
        
        # Determine the scenario prefix based on the policy scenario
        try:
            scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)
        except Exception as e:
            raise ValueError(f"Error determining scenario parameters: {str(e)}")
        
        # Determine which SCC assumptions to use based on climate sensitivity
        if climate_sensitivity:
            scc_assumptions = SCC_ASSUMPTIONS
        else:
            scc_assumptions = ['upper']  # Default to upper bound for climate sensitivity
            
        # Validate that SCC_ASSUMPTIONS is not empty
        if len(scc_assumptions) == 0:
            raise ValueError("SCC_ASSUMPTIONS is empty. Cannot perform climate sensitivity analysis.")
        
        # ========== PROCESS EACH EQUIPMENT CATEGORY ==========
        for category, upgrade_column in UPGRADE_COLUMNS.items():
            try:
                print(f"\nCalculating Adoption Potential for {category} under '{policy_scenario}' Scenario...")
                
                # Step 1): Initialize validation tracking for this category (Get valid mask)
                df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
                    df_copy, category, menu_mp, verbose=True)
                
                print(f"Found {valid_mask.sum()} valid homes out of {len(df_copy)} for {category} adoption potential")
                
                # Process each SCC assumption for climate damages
                for scc in scc_assumptions:
                    try:
                        # Create a DataFrame for new columns
                        df_new_columns = pd.DataFrame(index=df_copy.index)
                        
                        # Log processing information
                        print(f"""\n --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): {scc}
                
                              Health Impact Sensitivity:
                                rcm_model Model: {rcm_model} | cr_function Function: {cr_function}""")
                        
                        # ========== PREPARE COLUMN NAMES ==========
                        # Create variable names for NPV columns (private and public impact progams)
                        lessWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_lessWTP'
                        moreWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_moreWTP'
                        public_npv_col = f'{scenario_prefix}{category}_public_npv_{scc}_{rcm_model}_{cr_function}'
                        rebate_col = f'mp{menu_mp}_{category}_rebate_amount'
                        
                        # Define column names for calculated values using a dictionary
                        new_col_names = {
                            'health_sensitivity': f'{scenario_prefix}{category}_health_sensitivity',
                            'benefit': f'{scenario_prefix}{category}_benefit_{scc}_{rcm_model}_{cr_function}',
                            'lessWTP_total_npv': f'{scenario_prefix}{category}_total_npv_lessWTP_{scc}_{rcm_model}_{cr_function}',
                            'moreWTP_total_npv': f'{scenario_prefix}{category}_total_npv_moreWTP_{scc}_{rcm_model}_{cr_function}',
                            'adoption': f'{scenario_prefix}{category}_adoption_{scc}_{rcm_model}_{cr_function}',
                            'impact': f'{scenario_prefix}{category}_impact_{scc}_{rcm_model}_{cr_function}'
                        }

                        # Track all output columns for masking
                        category_columns_to_mask.extend(new_col_names.values())

                        # Step 2.) Initialize result columns with zeros for valid homes, NaN for others
                        for col_name in new_col_names.values():
                            if col_name == new_col_names['health_sensitivity']:
                                df_new_columns[col_name] = f'{rcm_model}, {cr_function}'
                            else:
                                df_new_columns[col_name] = create_retrofit_only_series(df_copy, valid_mask)

                        # ========== VALIDATE AND CONVERT COLUMNS ==========
                        # Check for required NPV columns
                        required_cols = [lessWTP_private_npv_col, moreWTP_private_npv_col, public_npv_col]
                        missing_required_cols = [col for col in required_cols if col not in df_copy.columns]
                        if missing_required_cols:
                            raise KeyError(f"Required NPV columns missing from DataFrame: {missing_required_cols}")
                            
                        # Ensure columns are numeric
                        for col in required_cols + [rebate_col]:
                            if col in df_copy.columns:
                                try:
                                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                                except Exception as e:
                                    raise TypeError(f"Failed to convert column '{col}' to numeric: {str(e)}")
                            else:
                                # Only rebate_col is optional
                                if col != rebate_col:
                                    raise KeyError(f"Required column '{col}' does not exist in the DataFrame.")
                                else:
                                    print(f"Warning: Optional column '{col}' does not exist in the DataFrame.")
                        
                        # ========== CALCULATE DERIVED VALUES ==========
                        # Step 3 & 4.) Calculate and update ONLY valid homes
                        # Calculate additional public benefit
                        if policy_scenario == 'No Inflation Reduction Act':
                            # No IRA Rebate so no "Additional Public Benefit"
                            df_new_columns.loc[valid_mask, new_col_names['benefit']] = 0.0
                        else:
                            # If rebate column exists, use it; otherwise, assume zero
                            if rebate_col in df_copy.columns:
                                valid_rows = valid_mask & df_copy[public_npv_col].notna() & df_copy[rebate_col].notna()
                                df_new_columns.loc[valid_rows, new_col_names['benefit']] = (
                                    df_copy.loc[valid_rows, public_npv_col] - 
                                    df_copy.loc[valid_rows, rebate_col]).clip(lower=0)
                            else:
                                valid_rows = valid_mask & df_copy[public_npv_col].notna()
                                df_new_columns.loc[valid_rows, new_col_names['benefit']] = (df_copy.loc[valid_rows, public_npv_col]).clip(lower=0)

                        # Calculate total NPV values - only for valid homes with non-null inputs
                        valid_npv_rows = valid_mask & df_copy[lessWTP_private_npv_col].notna() & df_copy[public_npv_col].notna()
                        df_new_columns.loc[valid_npv_rows, new_col_names['lessWTP_total_npv']] = (
                            df_copy.loc[valid_npv_rows, lessWTP_private_npv_col] + 
                            df_copy.loc[valid_npv_rows, public_npv_col]
                        )

                        valid_npv_rows = valid_mask & df_copy[moreWTP_private_npv_col].notna() & df_copy[public_npv_col].notna()
                        df_new_columns.loc[valid_npv_rows, new_col_names['moreWTP_total_npv']] = (
                            df_copy.loc[valid_npv_rows, moreWTP_private_npv_col] + 
                            df_copy.loc[valid_npv_rows, public_npv_col]
                        )

                        # ========== DETERMINE ADOPTION TIERS AND PUBLIC IMPACTS ==========
                        # # Initialize with more descriptive default values for ALL homes
                        # df_new_columns[new_col_names['adoption']] = 'N/A: Invalid Fuel/Tech or No Upgrade'
                        # df_new_columns[new_col_names['impact']] = 'N/A: Invalid Fuel/Tech or No Upgrade'

                        # Initialize with more descriptive default values for ALL homes
                        df_new_columns[new_col_names['adoption']] = 'N/A: Invalid Baseline Fuel/Tech'
                        df_new_columns[new_col_names['impact']] = 'N/A: Invalid Baseline Fuel/Tech'

                        # Set Tier 4 as default for valid homes with NPV data
                        valid_homes_with_npv = valid_mask & df_copy[lessWTP_private_npv_col].notna() & df_copy[moreWTP_private_npv_col].notna()
                        df_new_columns.loc[valid_homes_with_npv, new_col_names['adoption']] = 'Tier 4: Averse'

                        # Process homes with no upgrades
                        no_upgrade_mask = valid_mask & df_copy[upgrade_column].isna()
                        df_new_columns.loc[no_upgrade_mask, new_col_names['adoption']] = 'N/A: Already Upgraded!'

                        # Process Tier 1: Economically feasible
                        tier1_mask = valid_mask & df_copy[lessWTP_private_npv_col].notna() & (df_copy[lessWTP_private_npv_col] > 0)
                        df_new_columns.loc[tier1_mask, new_col_names['adoption']] = 'Tier 1: Feasible'

                        # Process Tier 2: Feasible vs alternative
                        tier2_mask = valid_mask & df_copy[lessWTP_private_npv_col].notna() & df_copy[moreWTP_private_npv_col].notna() & \
                                    (df_copy[lessWTP_private_npv_col] < 0) & (df_copy[moreWTP_private_npv_col] > 0)
                        df_new_columns.loc[tier2_mask, new_col_names['adoption']] = 'Tier 2: Feasible vs. Alternative'

                        # Process Tier 3: Subsidy-dependent
                        tier3_mask = valid_mask & df_copy[lessWTP_private_npv_col].notna() & df_copy[moreWTP_private_npv_col].notna() & \
                                    df_new_columns[new_col_names['moreWTP_total_npv']].notna() & \
                                    (df_copy[lessWTP_private_npv_col] < 0) & (df_copy[moreWTP_private_npv_col] < 0) & \
                                    (df_new_columns[new_col_names['moreWTP_total_npv']] > 0)
                        df_new_columns.loc[tier3_mask, new_col_names['adoption']] = 'Tier 3: Subsidy-Dependent Feasibility'

                        # Determine public impacts - only for valid homes
                        # Initialize impact to "No Impact" for valid homes
                        df_new_columns.loc[valid_mask, new_col_names['impact']] = 'N/A: Already Upgraded!'

                        # Zero public NPV = zero impact
                        zero_impact_mask = valid_mask & df_copy[public_npv_col].notna() & (df_copy[public_npv_col] == 0)
                        df_new_columns.loc[zero_impact_mask, new_col_names['impact']] = 'Public NPV is Zero'

                        # Positive public NPV = benefit
                        benefit_mask = valid_mask & df_copy[public_npv_col].notna() & (df_copy[public_npv_col] > 0)
                        df_new_columns.loc[benefit_mask, new_col_names['impact']] = 'Public Benefit'

                        # Negative public NPV = detriment
                        detriment_mask = valid_mask & df_copy[public_npv_col].notna() & (df_copy[public_npv_col] < 0)
                        df_new_columns.loc[detriment_mask, new_col_names['impact']] = 'Public Detriment'

                        # Apply new columns to DataFrame with proper tracking
                        # This utility function handles common tasks when adding new calculated columns:
                        # - Tracks columns for validation
                        # - Handles overlapping columns
                        # - Joins new columns to the original DataFrame
                        df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
                            df_copy, df_new_columns, category, category_columns_to_mask, all_columns_to_mask
                        )

                    except Exception as e:
                        print(f"Error processing SCC assumption {scc} for {category}: {str(e)}")
                        # Continue with next SCC assumption rather than failing completely
                        continue
                        
            except Exception as e:
                print(f"Error processing equipment category {category}: {str(e)}")
                # Continue with next equipment category rather than failing completely
                continue
        
        # ========== APPLY FINAL VERIFICATION MASKING ==========
        # Step 5.) Apply final verification masking for all tracked columns
        df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=True)
        
        # Return the updated DataFrame
        return df_copy
        
    except Exception as e:
        # Catch any unhandled exceptions
        error_message = f"An unexpected error occurred in adoption_decision: {str(e)}"
        print(f"ERROR: {error_message}")
        raise RuntimeError(error_message) from e
