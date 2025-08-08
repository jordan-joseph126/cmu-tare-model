"""
Alternative Version 1: Minimal Structural Changes

This version preserves the existing working functions EXACTLY and adds
independent climate and health analysis functions with minimal modifications.

Key Principles:
- Maximum compatibility with current architecture
- Preserve all working logic from backup version
- Add new functions without modifying existing ones
- Simple, direct validation and error handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from cmu_tare_model.constants import SCC_ASSUMPTIONS, RCM_MODELS, CR_FUNCTIONS, UPGRADE_COLUMNS, EQUIPMENT_SPECS
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
    UNCHANGED - Validates that input parameters meet expected criteria.
    
    This function is preserved exactly from the working backup version.
    
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
) -> pd.DataFrame:
    """
    UNCHANGED - Updates DataFrame with adoption decisions and public impacts.
    
    This function is preserved exactly from the working backup version.
    This ensures computational consistency and avoids introducing new bugs.
    
    Args:
        df: The DataFrame containing home equipment data.
        menu_mp: Measure package identifier to use in column names.
        policy_scenario: Policy scenario that determines electricity grid projections.
        rcm_model: The RCM model to use for analysis.
        cr_function: The concentration response function to use.
    
    Returns:
        DataFrame with additional adoption decision and public impact columns.
        
    Raises:
        ValueError: If any input parameters are invalid or not supported.
        KeyError: If required columns are missing from the input DataFrame.
        TypeError: If column data types cannot be converted as expected.
    """
    try:
        # ========== SETUP AND VALIDATION ==========
        validate_input_parameters(menu_mp, policy_scenario, rcm_model, cr_function)
        
        df_copy = df.copy()
        df_new_columns = pd.DataFrame(index=df_copy.index)
                
        missing_columns = [col for col in UPGRADE_COLUMNS.values() if col not in df_copy.columns]
        if missing_columns:
            raise KeyError(f"Required upgrade columns missing from DataFrame: {missing_columns}")
        
        try:
            scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)
        except Exception as e:
            raise ValueError(f"Error determining scenario parameters: {str(e)}")
                
        # ========== PROCESS EACH EQUIPMENT CATEGORY ==========
        for category, upgrade_column in UPGRADE_COLUMNS.items():
            try:
                print(f"\nCalculating Adoption Potential for {category} under '{policy_scenario}' Scenario...")
                
                # Step 1: Initialize validation tracking
                df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
                    df_copy, category, menu_mp, verbose=True)
                
                print(f"Found {valid_mask.sum()} valid homes out of {len(df_copy)} for {category} adoption potential")
                
                # Process each SCC assumption
                for scc in SCC_ASSUMPTIONS:
                    try:
                        df_new_columns = pd.DataFrame(index=df_copy.index)
                        
                        print(f"""\n --- Public NPV Sensitivity ---
                              Climate Impact Sensitivity:
                                SCC Assumption (Bound): {scc}
                
                              Health Impact Sensitivity:
                                rcm_model Model: {rcm_model} | cr_function Function: {cr_function}""")
                        
                        # ========== PREPARE COLUMN NAMES ==========
                        lessWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_lessWTP'
                        moreWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_moreWTP'
                        rebate_col = f'mp{menu_mp}_{category}_rebate_amount'
                        public_npv_col = f'{scenario_prefix}{category}_public_npv_{scc}_{rcm_model}_{cr_function}'          

                        new_col_names = {
                            'health_sensitivity': f'{scenario_prefix}{category}_health_sensitivity',
                            'benefit': f'{scenario_prefix}{category}_benefit_{scc}_{rcm_model}_{cr_function}',
                            'lessWTP_total_npv': f'{scenario_prefix}{category}_total_npv_lessWTP_{scc}_{rcm_model}_{cr_function}',
                            'moreWTP_total_npv': f'{scenario_prefix}{category}_total_npv_moreWTP_{scc}_{rcm_model}_{cr_function}',
                            'adoption': f'{scenario_prefix}{category}_adoption_{scc}_{rcm_model}_{cr_function}',
                            'impact': f'{scenario_prefix}{category}_impact_{scc}_{rcm_model}_{cr_function}'
                        }

                        category_columns_to_mask.extend(new_col_names.values())

                        # Step 2: Initialize result columns
                        for col_name in new_col_names.values():
                            if col_name == new_col_names['health_sensitivity']:
                                df_new_columns[col_name] = f'{rcm_model}, {cr_function}'
                            else:
                                df_new_columns[col_name] = create_retrofit_only_series(df_copy, valid_mask)

                        # ========== VALIDATE COLUMNS ==========
                        required_cols = [lessWTP_private_npv_col, moreWTP_private_npv_col, public_npv_col]
                        missing_required_cols = [col for col in required_cols if col not in df_copy.columns]
                        if missing_required_cols:
                            raise KeyError(f"Required NPV columns missing from DataFrame: {missing_required_cols}")
                            
                        # Convert to numeric
                        for col in required_cols + [rebate_col]:
                            if col in df_copy.columns:
                                try:
                                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                                except Exception as e:
                                    raise TypeError(f"Failed to convert column '{col}' to numeric: {str(e)}")
                            else:
                                if col != rebate_col:
                                    raise KeyError(f"Required column '{col}' does not exist in the DataFrame.")
                                else:
                                    print(f"Warning: Optional column '{col}' does not exist in the DataFrame.")
                        
                        # ========== CALCULATE VALUES ==========
                        # Step 3 & 4: Calculate and update only valid homes
                        if policy_scenario == 'No Inflation Reduction Act':
                            df_new_columns.loc[valid_mask, new_col_names['benefit']] = 0.0
                        else:
                            if rebate_col in df_copy.columns:
                                valid_rows = valid_mask & df_copy[public_npv_col].notna() & df_copy[rebate_col].notna()
                                df_new_columns.loc[valid_rows, new_col_names['benefit']] = (
                                    df_copy.loc[valid_rows, public_npv_col] - 
                                    df_copy.loc[valid_rows, rebate_col]).clip(lower=0)
                            else:
                                valid_rows = valid_mask & df_copy[public_npv_col].notna()
                                df_new_columns.loc[valid_rows, new_col_names['benefit']] = (df_copy.loc[valid_rows, public_npv_col]).clip(lower=0)

                        # Calculate total NPV values
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

                        # ========== ADOPTION TIERS AND PUBLIC IMPACTS ==========
                        df_new_columns[new_col_names['adoption']] = 'N/A: Invalid Baseline Fuel/Tech'
                        df_new_columns[new_col_names['impact']] = 'N/A: Invalid Baseline Fuel/Tech'

                        valid_homes_with_npv = valid_mask & df_copy[lessWTP_private_npv_col].notna() & df_copy[moreWTP_private_npv_col].notna()
                        df_new_columns.loc[valid_homes_with_npv, new_col_names['adoption']] = 'Tier 4: Averse'

                        no_upgrade_mask = valid_mask & df_copy[upgrade_column].isna()
                        df_new_columns.loc[no_upgrade_mask, new_col_names['adoption']] = 'N/A: Already Upgraded!'

                        # Tier 1: Economically feasible
                        tier1_mask = valid_mask & df_copy[lessWTP_private_npv_col].notna() & (df_copy[lessWTP_private_npv_col] > 0)
                        df_new_columns.loc[tier1_mask, new_col_names['adoption']] = 'Tier 1: Feasible'

                        # Tier 2: Feasible vs alternative
                        tier2_mask = valid_mask & df_copy[lessWTP_private_npv_col].notna() & df_copy[moreWTP_private_npv_col].notna() & \
                                    (df_copy[lessWTP_private_npv_col] < 0) & (df_copy[moreWTP_private_npv_col] > 0)
                        df_new_columns.loc[tier2_mask, new_col_names['adoption']] = 'Tier 2: Feasible vs. Alternative'

                        # Tier 3: Subsidy-dependent
                        tier3_mask = valid_mask & df_copy[lessWTP_private_npv_col].notna() & df_copy[moreWTP_private_npv_col].notna() & \
                                    df_new_columns[new_col_names['moreWTP_total_npv']].notna() & \
                                    (df_copy[lessWTP_private_npv_col] < 0) & (df_copy[moreWTP_private_npv_col] < 0) & \
                                    (df_new_columns[new_col_names['moreWTP_total_npv']] > 0)
                        df_new_columns.loc[tier3_mask, new_col_names['adoption']] = 'Tier 3: Subsidy-Dependent Feasibility'

                        # Determine public impacts
                        df_new_columns.loc[valid_mask, new_col_names['impact']] = 'N/A: Already Upgraded!'

                        zero_impact_mask = valid_mask & df_copy[public_npv_col].notna() & (df_copy[public_npv_col] == 0)
                        df_new_columns.loc[zero_impact_mask, new_col_names['impact']] = 'Public NPV is Zero'

                        benefit_mask = valid_mask & df_copy[public_npv_col].notna() & (df_copy[public_npv_col] > 0)
                        df_new_columns.loc[benefit_mask, new_col_names['impact']] = 'Public Benefit'

                        detriment_mask = valid_mask & df_copy[public_npv_col].notna() & (df_copy[public_npv_col] < 0)
                        df_new_columns.loc[detriment_mask, new_col_names['impact']] = 'Public Detriment'

                        # Apply new columns to DataFrame
                        df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
                            df_copy, df_new_columns, category, category_columns_to_mask, all_columns_to_mask
                        )

                    except Exception as e:
                        print(f"Error processing SCC assumption {scc} for {category}: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"Error processing equipment category {category}: {str(e)}")
                continue
                        
        # ========== APPLY FINAL MASKING ==========
        # Step 5: Apply final verification masking
        df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=True)
        return df_copy
        
    except ValueError as e:
        print(f"ERROR: {str(e)}")
        raise
    except Exception as e:
        error_message = f"An unexpected error occurred in adoption_decision: {str(e)}"
        print(f"ERROR: {error_message}")
        raise RuntimeError(error_message) from e


def calculate_climate_only_adoption(
    df: pd.DataFrame,
    menu_mp: int,
    policy_scenario: str,
    scc_assumptions: List[str] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Independent climate-only adoption analysis using working backup logic.
    
    This function adapts the working backup's logic specifically for climate-only
    analysis, maintaining the same successful patterns but focusing on climate impacts.
    
    Args:
        df: DataFrame containing home equipment data and climate NPV columns.
        menu_mp: Measure package identifier.
        policy_scenario: Policy scenario for analysis.
        scc_assumptions: List of SCC assumptions to analyze. Defaults to all.
        verbose: Whether to print detailed progress messages.
        
    Returns:
        DataFrame with climate-only adoption analysis columns.
        
    Raises:
        ValueError: If input parameters are invalid.
        KeyError: If required climate NPV columns are missing.
    """
    if scc_assumptions is None:
        scc_assumptions = SCC_ASSUMPTIONS
        
    # Validate policy scenario
    valid_scenarios = ['No Inflation Reduction Act', 'AEO2023 Reference Case']
    if policy_scenario not in valid_scenarios:
        raise ValueError(f"Invalid policy_scenario: '{policy_scenario}'. Must be one of {valid_scenarios}")
    
    df_copy = df.copy()
    
    # Get scenario prefix
    try:
        scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)
    except Exception as e:
        raise ValueError(f"Error determining scenario parameters: {str(e)}")
    
    # Check required upgrade columns
    missing_columns = [col for col in UPGRADE_COLUMNS.values() if col not in df_copy.columns]
    if missing_columns:
        raise KeyError(f"Required upgrade columns missing from DataFrame: {missing_columns}")
    
    if verbose:
        print(f"Starting climate-only adoption analysis for {policy_scenario}")
    
    # Process each equipment category
    for category, upgrade_column in UPGRADE_COLUMNS.items():
        try:
            if verbose:
                print(f"\nProcessing climate-only analysis for {category}...")
            
            # Step 1: Initialize validation tracking
            df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
                df_copy, category, menu_mp, verbose=verbose)
            
            # Process each SCC assumption
            for scc in scc_assumptions:
                try:
                    df_new_columns = pd.DataFrame(index=df_copy.index)
                    
                    # Define column names
                    lessWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_lessWTP'
                    moreWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_moreWTP'
                    climate_npv_col = f'{scenario_prefix}{category}_climate_npv_{scc}'
                    
                    # Check if climate NPV column exists
                    if climate_npv_col not in df_copy.columns:
                        if verbose:
                            print(f"  Skipping {scc}: Climate NPV column {climate_npv_col} not found")
                        continue
                    
                    # Define output column names
                    climate_col_names = {
                        'lessWTP_total_npv': f'{scenario_prefix}{category}_total_npv_lessWTP_climateOnly_{scc}',
                        'moreWTP_total_npv': f'{scenario_prefix}{category}_total_npv_moreWTP_climateOnly_{scc}',
                        'adoption': f'{scenario_prefix}{category}_adoption_climateOnly_{scc}',
                        'impact': f'{scenario_prefix}{category}_impact_climateOnly_{scc}'
                    }
                    
                    # Step 2: Initialize result columns
                    for col_name in climate_col_names.values():
                        df_new_columns[col_name] = create_retrofit_only_series(df_copy, valid_mask)
                    
                    # Validate required columns exist
                    required_cols = [lessWTP_private_npv_col, moreWTP_private_npv_col, climate_npv_col]
                    missing_cols = [col for col in required_cols if col not in df_copy.columns]
                    if missing_cols:
                        if verbose:
                            print(f"  Skipping {scc}: Missing columns {missing_cols}")
                        continue
                    
                    # Convert to numeric (following working backup pattern)
                    for col in required_cols:
                        try:
                            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                        except Exception as e:
                            if verbose:
                                print(f"  Warning: Failed to convert {col} to numeric: {e}")
                            continue
                    
                    # Step 3 & 4: Calculate total NPV values (climate-only)
                    valid_npv_rows = valid_mask & df_copy[lessWTP_private_npv_col].notna() & df_copy[climate_npv_col].notna()
                    df_new_columns.loc[valid_npv_rows, climate_col_names['lessWTP_total_npv']] = (
                        df_copy.loc[valid_npv_rows, lessWTP_private_npv_col] + 
                        df_copy.loc[valid_npv_rows, climate_npv_col]
                    )
                    
                    valid_npv_rows = valid_mask & df_copy[moreWTP_private_npv_col].notna() & df_copy[climate_npv_col].notna()
                    df_new_columns.loc[valid_npv_rows, climate_col_names['moreWTP_total_npv']] = (
                        df_copy.loc[valid_npv_rows, moreWTP_private_npv_col] + 
                        df_copy.loc[valid_npv_rows, climate_npv_col]
                    )
                    
                    # Apply adoption decision logic (adapted from working backup)
                    df_new_columns[climate_col_names['adoption']] = 'N/A: Invalid Baseline Fuel/Tech'
                    df_new_columns[climate_col_names['impact']] = 'N/A: Invalid Baseline Fuel/Tech'
                    
                    # Set defaults for valid homes
                    valid_homes_with_npv = valid_mask & df_copy[lessWTP_private_npv_col].notna() & df_copy[moreWTP_private_npv_col].notna()
                    df_new_columns.loc[valid_homes_with_npv, climate_col_names['adoption']] = 'Tier 4: Averse'
                    
                    # Handle no upgrade cases
                    no_upgrade_mask = valid_mask & df_copy[upgrade_column].isna()
                    df_new_columns.loc[no_upgrade_mask, climate_col_names['adoption']] = 'N/A: Already Upgraded!'
                    
                    # Tier 1: Economically feasible
                    tier1_mask = valid_mask & df_copy[lessWTP_private_npv_col].notna() & (df_copy[lessWTP_private_npv_col] > 0)
                    df_new_columns.loc[tier1_mask, climate_col_names['adoption']] = 'Tier 1: Feasible'
                    
                    # Tier 2: Feasible vs alternative  
                    tier2_mask = valid_mask & df_copy[lessWTP_private_npv_col].notna() & df_copy[moreWTP_private_npv_col].notna() & \
                                (df_copy[lessWTP_private_npv_col] < 0) & (df_copy[moreWTP_private_npv_col] > 0)
                    df_new_columns.loc[tier2_mask, climate_col_names['adoption']] = 'Tier 2: Feasible vs. Alternative'
                    
                    # Tier 3: Subsidy-dependent (using climate-only total NPV)
                    tier3_mask = valid_mask & df_copy[lessWTP_private_npv_col].notna() & df_copy[moreWTP_private_npv_col].notna() & \
                                df_new_columns[climate_col_names['moreWTP_total_npv']].notna() & \
                                (df_copy[lessWTP_private_npv_col] < 0) & (df_copy[moreWTP_private_npv_col] < 0) & \
                                (df_new_columns[climate_col_names['moreWTP_total_npv']] > 0)
                    df_new_columns.loc[tier3_mask, climate_col_names['adoption']] = 'Tier 3: Subsidy-Dependent Feasibility'
                    
                    # Determine public impacts based on climate NPV
                    df_new_columns.loc[valid_mask, climate_col_names['impact']] = 'N/A: Already Upgraded!'
                    
                    zero_impact_mask = valid_mask & df_copy[climate_npv_col].notna() & (df_copy[climate_npv_col] == 0)
                    df_new_columns.loc[zero_impact_mask, climate_col_names['impact']] = 'Public NPV is Zero'
                    
                    benefit_mask = valid_mask & df_copy[climate_npv_col].notna() & (df_copy[climate_npv_col] > 0)
                    df_new_columns.loc[benefit_mask, climate_col_names['impact']] = 'Public Benefit'
                    
                    detriment_mask = valid_mask & df_copy[climate_npv_col].notna() & (df_copy[climate_npv_col] < 0)
                    df_new_columns.loc[detriment_mask, climate_col_names['impact']] = 'Public Detriment'
                    
                    # Track columns for masking
                    category_columns_to_mask.extend(climate_col_names.values())
                    
                    # Apply new columns
                    df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
                        df_copy, df_new_columns, category, category_columns_to_mask, all_columns_to_mask
                    )
                    
                    if verbose:
                        print(f"  Completed climate-only analysis for {category}/{scc}")
                        
                except Exception as e:
                    if verbose:
                        print(f"  Error processing {scc} for {category}: {e}")
                    continue
                    
        except Exception as e:
            if verbose:
                print(f"Error processing category {category}: {e}")
            continue
    
    # Step 5: Apply final masking
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=verbose)
    
    if verbose:
        print("Climate-only adoption analysis completed")
    
    return df_copy


def calculate_health_only_adoption(
    df: pd.DataFrame,
    menu_mp: int,
    policy_scenario: str,
    rcm_model: str,
    cr_function: str,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Independent health-only adoption analysis using working backup logic.
    
    This function adapts the working backup's logic specifically for health-only
    analysis, maintaining the same successful patterns but focusing on health impacts.
    
    Args:
        df: DataFrame containing home equipment data and health NPV columns.
        menu_mp: Measure package identifier.
        policy_scenario: Policy scenario for analysis.
        rcm_model: RCM model for health impact analysis.
        cr_function: Concentration response function for health analysis.
        verbose: Whether to print detailed progress messages.
        
    Returns:
        DataFrame with health-only adoption analysis columns.
        
    Raises:
        ValueError: If input parameters are invalid.
        KeyError: If required health NPV columns are missing.
    """
    # Validate inputs
    valid_scenarios = ['No Inflation Reduction Act', 'AEO2023 Reference Case']
    if policy_scenario not in valid_scenarios:
        raise ValueError(f"Invalid policy_scenario: '{policy_scenario}'. Must be one of {valid_scenarios}")
    
    if rcm_model not in RCM_MODELS:
        raise ValueError(f"Invalid rcm_model: '{rcm_model}'. Must be one of {RCM_MODELS}")
    
    if cr_function not in CR_FUNCTIONS:
        raise ValueError(f"Invalid cr_function: '{cr_function}'. Must be one of {CR_FUNCTIONS}")
    
    df_copy = df.copy()
    
    # Get scenario prefix
    try:
        scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)
    except Exception as e:
        raise ValueError(f"Error determining scenario parameters: {str(e)}")
    
    # Check required upgrade columns
    missing_columns = [col for col in UPGRADE_COLUMNS.values() if col not in df_copy.columns]
    if missing_columns:
        raise KeyError(f"Required upgrade columns missing from DataFrame: {missing_columns}")
    
    if verbose:
        print(f"Starting health-only adoption analysis for {policy_scenario}, {rcm_model}, {cr_function}")
    
    # Process each equipment category
    for category, upgrade_column in UPGRADE_COLUMNS.items():
        try:
            if verbose:
                print(f"\nProcessing health-only analysis for {category}...")
            
            # Step 1: Initialize validation tracking
            df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
                df_copy, category, menu_mp, verbose=verbose)
            
            try:
                df_new_columns = pd.DataFrame(index=df_copy.index)
                
                # Define column names
                lessWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_lessWTP'
                moreWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_moreWTP'
                health_npv_col = f'{scenario_prefix}{category}_health_npv_{rcm_model}_{cr_function}'
                
                # Check if health NPV column exists
                if health_npv_col not in df_copy.columns:
                    if verbose:
                        print(f"  Skipping {category}: Health NPV column {health_npv_col} not found")
                    continue
                
                # Define output column names
                health_col_names = {
                    'lessWTP_total_npv': f'{scenario_prefix}{category}_total_npv_lessWTP_healthOnly_{rcm_model}_{cr_function}',
                    'moreWTP_total_npv': f'{scenario_prefix}{category}_total_npv_moreWTP_healthOnly_{rcm_model}_{cr_function}',
                    'adoption': f'{scenario_prefix}{category}_adoption_healthOnly_{rcm_model}_{cr_function}',
                    'impact': f'{scenario_prefix}{category}_impact_healthOnly_{rcm_model}_{cr_function}'
                }
                
                # Step 2: Initialize result columns
                for col_name in health_col_names.values():
                    df_new_columns[col_name] = create_retrofit_only_series(df_copy, valid_mask)
                
                # Validate required columns exist
                required_cols = [lessWTP_private_npv_col, moreWTP_private_npv_col, health_npv_col]
                missing_cols = [col for col in required_cols if col not in df_copy.columns]
                if missing_cols:
                    if verbose:
                        print(f"  Skipping {category}: Missing columns {missing_cols}")
                    continue
                
                # Convert to numeric (following working backup pattern)
                for col in required_cols:
                    try:
                        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                    except Exception as e:
                        if verbose:
                            print(f"  Warning: Failed to convert {col} to numeric: {e}")
                        continue
                
                # Step 3 & 4: Calculate total NPV values (health-only)
                valid_npv_rows = valid_mask & df_copy[lessWTP_private_npv_col].notna() & df_copy[health_npv_col].notna()
                df_new_columns.loc[valid_npv_rows, health_col_names['lessWTP_total_npv']] = (
                    df_copy.loc[valid_npv_rows, lessWTP_private_npv_col] + 
                    df_copy.loc[valid_npv_rows, health_npv_col]
                )
                
                valid_npv_rows = valid_mask & df_copy[moreWTP_private_npv_col].notna() & df_copy[health_npv_col].notna()
                df_new_columns.loc[valid_npv_rows, health_col_names['moreWTP_total_npv']] = (
                    df_copy.loc[valid_npv_rows, moreWTP_private_npv_col] + 
                    df_copy.loc[valid_npv_rows, health_npv_col]
                )
                
                # Apply adoption decision logic (adapted from working backup)
                df_new_columns[health_col_names['adoption']] = 'N/A: Invalid Baseline Fuel/Tech'
                df_new_columns[health_col_names['impact']] = 'N/A: Invalid Baseline Fuel/Tech'
                
                # Set defaults for valid homes
                valid_homes_with_npv = valid_mask & df_copy[lessWTP_private_npv_col].notna() & df_copy[moreWTP_private_npv_col].notna()
                df_new_columns.loc[valid_homes_with_npv, health_col_names['adoption']] = 'Tier 4: Averse'
                
                # Handle no upgrade cases
                no_upgrade_mask = valid_mask & df_copy[upgrade_column].isna()
                df_new_columns.loc[no_upgrade_mask, health_col_names['adoption']] = 'N/A: Already Upgraded!'
                
                # Tier 1: Economically feasible
                tier1_mask = valid_mask & df_copy[lessWTP_private_npv_col].notna() & (df_copy[lessWTP_private_npv_col] > 0)
                df_new_columns.loc[tier1_mask, health_col_names['adoption']] = 'Tier 1: Feasible'
                
                # Tier 2: Feasible vs alternative  
                tier2_mask = valid_mask & df_copy[lessWTP_private_npv_col].notna() & df_copy[moreWTP_private_npv_col].notna() & \
                            (df_copy[lessWTP_private_npv_col] < 0) & (df_copy[moreWTP_private_npv_col] > 0)
                df_new_columns.loc[tier2_mask, health_col_names['adoption']] = 'Tier 2: Feasible vs. Alternative'
                
                # Tier 3: Subsidy-dependent (using health-only total NPV)
                tier3_mask = valid_mask & df_copy[lessWTP_private_npv_col].notna() & df_copy[moreWTP_private_npv_col].notna() & \
                            df_new_columns[health_col_names['moreWTP_total_npv']].notna() & \
                            (df_copy[lessWTP_private_npv_col] < 0) & (df_copy[moreWTP_private_npv_col] < 0) & \
                            (df_new_columns[health_col_names['moreWTP_total_npv']] > 0)
                df_new_columns.loc[tier3_mask, health_col_names['adoption']] = 'Tier 3: Subsidy-Dependent Feasibility'
                
                # Determine public impacts based on health NPV
                df_new_columns.loc[valid_mask, health_col_names['impact']] = 'N/A: Already Upgraded!'
                
                zero_impact_mask = valid_mask & df_copy[health_npv_col].notna() & (df_copy[health_npv_col] == 0)
                df_new_columns.loc[zero_impact_mask, health_col_names['impact']] = 'Public NPV is Zero'
                
                benefit_mask = valid_mask & df_copy[health_npv_col].notna() & (df_copy[health_npv_col] > 0)
                df_new_columns.loc[benefit_mask, health_col_names['impact']] = 'Public Benefit'
                
                detriment_mask = valid_mask & df_copy[health_npv_col].notna() & (df_copy[health_npv_col] < 0)
                df_new_columns.loc[detriment_mask, health_col_names['impact']] = 'Public Detriment'
                
                # Track columns for masking
                category_columns_to_mask.extend(health_col_names.values())
                
                # Apply new columns
                df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
                    df_copy, df_new_columns, category, category_columns_to_mask, all_columns_to_mask
                )
                
                if verbose:
                    print(f"  Completed health-only analysis for {category}")
                    
            except Exception as e:
                if verbose:
                    print(f"  Error processing {category}: {e}")
                continue
                
        except Exception as e:
            if verbose:
                print(f"Error processing category {category}: {e}")
            continue
    
    # Step 5: Apply final masking
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=verbose)
    
    if verbose:
        print("Health-only adoption analysis completed")
    
    return df_copy
