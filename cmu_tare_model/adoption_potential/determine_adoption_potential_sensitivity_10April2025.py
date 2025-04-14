import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from config import PROJECT_ROOT
from cmu_tare_model.constants import SCC_ASSUMPTIONS, RCM_MODELS, CR_FUNCTIONS
from cmu_tare_model.utils.modeling_params import define_scenario_params


def validate_input_parameters(menu_mp: int,
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


def adoption_decision(df: pd.DataFrame,
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
        
        # Define equipment categories and their corresponding upgrade columns
        upgrade_columns = {
            'heating': 'upgrade_hvac_heating_efficiency',
            'waterHeating': 'upgrade_water_heater_efficiency',
            'clothesDrying': 'upgrade_clothes_dryer',
            'cooking': 'upgrade_cooking_range'
        }
        
        # Check that required upgrade columns exist in the DataFrame
        missing_columns = [col for col in upgrade_columns.values() if col not in df_copy.columns]
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
        
        for category, upgrade_column in upgrade_columns.items():
            try:
                print(f"\nCalculating Adoption Potential for {category} under '{policy_scenario}' Scenario...")
                
                # Process each SCC assumption for climate damages
                for scc in scc_assumptions:
                    try:
                        # Log processing information
                        print(f"""\n --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): {scc}
                
                              Health Impact Sensitivity:
                                rcm_model Model: {rcm_model} | cr_function Function: {cr_function}""")
                        
                        # ========== PREPARE COLUMN NAMES ==========
                        
                        # Create column for sensitivity analysis identification
                        health_sensitivity_col = f'{scenario_prefix}{category}_health_sensitivity'
                        df_new_columns[health_sensitivity_col] = f'{rcm_model}, {cr_function}'
                        
                        # Define column names for NPV values
                        lessWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_lessWTP'
                        moreWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_moreWTP'
                        public_npv_col = f'{scenario_prefix}{category}_{scc}_{rcm_model}_{cr_function}'
                        rebate_col = f'mp{menu_mp}_{category}_rebate_amount'
                        
                        # Define column names for calculated values
                        addition_public_benefit = f'{scenario_prefix}{category}_additional_public_benefit_{scc}'
                        lessWTP_total_npv_col = f'{scenario_prefix}{category}_total_npv_lessWTP_{scc}'
                        moreWTP_total_npv_col = f'{scenario_prefix}{category}_total_npv_moreWTP_{scc}'
                        adoption_col_name = f'{scenario_prefix}{category}_adoption_{scc}SCC'
                        retrofit_col_name = f'{scenario_prefix}{category}_retrofit_publicImpact_{scc}SCC'
                        
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
                        
                        # Calculate additional public benefit
                        if policy_scenario == 'No Inflation Reduction Act':
                            # No IRA Rebate so no "Additional Public Benefit"
                            df_new_columns[addition_public_benefit] = 0.0
                        else:
                            # If rebate column exists, use it; otherwise, assume zero
                            if rebate_col in df_copy.columns:
                                df_new_columns[addition_public_benefit] = (df_copy[public_npv_col] - df_copy[rebate_col]).clip(lower=0)
                            else:
                                rebate_values = pd.Series(0.0, index=df_copy.index)
                                df_new_columns[addition_public_benefit] = (df_copy[public_npv_col] - rebate_values).clip(lower=0)
                        
                        # Calculate total NPV values
                        df_new_columns[lessWTP_total_npv_col] = df_copy[lessWTP_private_npv_col] + df_copy[public_npv_col]
                        df_new_columns[moreWTP_total_npv_col] = df_copy[moreWTP_private_npv_col] + df_copy[public_npv_col]
                        
                        # ========== DETERMINE ADOPTION TIERS AND PUBLIC IMPACTS ==========
                        
                        # Initialize with default values
                        df_new_columns[adoption_col_name] = 'Tier 4: Averse'
                        df_new_columns[retrofit_col_name] = 'No Retrofit'
                        
                        # Determine adoption tiers
                        conditions = [
                            df_copy[upgrade_column].isna(),  # Existing equipment (no upgrade)
                            df_copy[lessWTP_private_npv_col] > 0,  # Tier 1: Economically feasible
                            (df_copy[lessWTP_private_npv_col] < 0) & (df_copy[moreWTP_private_npv_col] > 0),  # Tier 2: Feasible vs alternative
                            (df_copy[lessWTP_private_npv_col] < 0) & (df_copy[moreWTP_private_npv_col] < 0) & (df_new_columns[moreWTP_total_npv_col] > 0),  # Tier 3: Subsidy-dependent
                        ]
                        
                        choices = [
                            'Existing Equipment', 
                            'Tier 1: Feasible', 
                            'Tier 2: Feasible vs. Alternative', 
                            'Tier 3: Subsidy-Dependent Feasibility'
                        ]
                        
                        df_new_columns[adoption_col_name] = np.select(conditions, choices, default='Tier 4: Averse')
                        
                        # Determine public impacts
                        public_conditions = [
                            df_copy[public_npv_col] > 0,  # Positive public NPV = benefit
                            df_copy[public_npv_col] < 0   # Negative public NPV = detriment
                        ]
                        
                        public_choices = ['Public Benefit', 'Public Detriment']
                        df_new_columns[retrofit_col_name] = np.select(public_conditions, public_choices, default='No Retrofit')
                        
                    except Exception as e:
                        print(f"Error processing SCC assumption {scc} for {category}: {str(e)}")
                        # Continue with next SCC assumption rather than failing completely
                        continue
                        
            except Exception as e:
                print(f"Error processing equipment category {category}: {str(e)}")
                # Continue with next equipment category rather than failing completely
                continue
        
        # ========== MERGE RESULTS AND RETURN ==========
        
        # Identify overlapping columns between the new and existing DataFrame
        overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)
        
        # Drop overlapping columns from df_copy
        if not overlapping_columns.empty:
            df_copy.drop(columns=overlapping_columns, inplace=True)
        
        # Merge new columns into df_copy
        df_copy = df_copy.join(df_new_columns, how='left')
        
        # Return the updated DataFrame
        return df_copy
        
    except Exception as e:
        # Catch any unhandled exceptions
        error_message = f"An unexpected error occurred in adoption_decision: {str(e)}"
        print(f"ERROR: {error_message}")
        raise RuntimeError(error_message) from e
    