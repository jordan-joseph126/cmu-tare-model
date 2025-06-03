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
    Validates input parameters with clear error messages.
    
    Args:
        menu_mp: Measure package identifier.
        policy_scenario: Policy scenario name.
        rcm_model: RCM model name.
        cr_function: Concentration response function name.
        
    Raises:
        ValueError: If any parameter is invalid.
    """
    errors = []
    
    # Validate menu_mp
    if not isinstance(menu_mp, int):
        try:
            int(menu_mp)
        except (ValueError, TypeError):
            errors.append(f"menu_mp must be an integer, got {type(menu_mp).__name__}: {menu_mp}")
    
    # Validate policy_scenario
    valid_scenarios = ['No Inflation Reduction Act', 'AEO2023 Reference Case']
    if policy_scenario not in valid_scenarios:
        errors.append(f"Invalid policy_scenario: '{policy_scenario}'. Must be one of {valid_scenarios}")
    
    # Validate rcm_model
    if rcm_model not in RCM_MODELS:
        errors.append(f"Invalid rcm_model: '{rcm_model}'. Must be one of {RCM_MODELS}")
    
    # Validate cr_function
    if cr_function not in CR_FUNCTIONS:
        errors.append(f"Invalid cr_function: '{cr_function}'. Must be one of {CR_FUNCTIONS}")
    
    if errors:
        error_msg = "Parameter validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        raise ValueError(error_msg)


def fix_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate columns if found, keeping first occurrence.
    Silent operation unless duplicates are actually fixed.
    
    Args:
        df: DataFrame with potential duplicate columns.
        
    Returns:
        DataFrame with duplicates removed.
    """
    duplicate_count = len(df.columns) - len(df.columns.unique())
    if duplicate_count == 0:
        return df
    
    # Only print if action taken
    print(f"Fixed {duplicate_count} duplicate columns")
    return df.loc[:, ~df.columns.duplicated(keep='first')]


def adoption_decision(
    df: pd.DataFrame,
    menu_mp: int,
    policy_scenario: str,
    rcm_model: str,
    cr_function: str,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Updates DataFrame with adoption decisions and public impacts based on NPV analysis.
    
    Simplified output for nation-level analysis while maintaining full functionality.
    
    Args:
        df: DataFrame containing home equipment data.
        menu_mp: Measure package identifier.
        policy_scenario: Policy scenario for electricity grid projections.
            Accepted values: 'No Inflation Reduction Act', 'AEO2023 Reference Case'.
        rcm_model: RCM model for health impact analysis ('ap2', 'easiur', 'inmap').
        cr_function: Concentration response function ('acs', 'h6c').
        verbose: Enable detailed output for debugging (default: False).
        
    Returns:
        DataFrame with adoption tier and public impact classifications.
        
    Raises:
        ValueError: If input parameters are invalid.
        KeyError: If required columns are missing.
        TypeError: If data type conversion fails.
    """
    try:
        # Validate inputs
        validate_input_parameters(menu_mp, policy_scenario, rcm_model, cr_function)
        
        # Check required upgrade columns
        missing_columns = [col for col in UPGRADE_COLUMNS.values() if col not in df.columns]
        if missing_columns:
            raise KeyError(f"Required upgrade columns missing: {missing_columns}")
        
        # Setup
        df_copy = df.copy()
        df_copy = fix_duplicate_columns(df_copy)
        
        # Get scenario prefix
        try:
            scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)
        except Exception as e:
            raise ValueError(f"Error determining scenario parameters: {str(e)}")
        
        # Single header for nation-level analysis
        if not verbose:
            print(f"\nAdoption Analysis: {policy_scenario} | {rcm_model}-{cr_function}")
        
        all_columns_to_mask = {cat: [] for cat in UPGRADE_COLUMNS}
        category_summaries = []
        
        # Process each equipment category
        for category, upgrade_column in UPGRADE_COLUMNS.items():
            try:
                # Initialize validation tracking (silent unless verbose)
                df_copy, valid_mask, _, category_columns_to_mask = initialize_validation_tracking(
                    df_copy, category, menu_mp, verbose=verbose)
                
                valid_count = valid_mask.sum()
                total_count = len(df_copy)
                
                # Process all SCC assumptions
                scc_processed = 0
                for scc in SCC_ASSUMPTIONS:
                    try:
                        # Define column names
                        lessWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_lessWTP'
                        moreWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_moreWTP'
                        public_npv_col = f'{scenario_prefix}{category}_public_npv_{scc}_{rcm_model}_{cr_function}'
                        climate_npv_col = f'{scenario_prefix}{category}_climate_npv_{scc}'
                        health_npv_col = f'{scenario_prefix}{category}_health_npv_{rcm_model}_{cr_function}'
                        rebate_col = f'mp{menu_mp}_{category}_rebate_amount'
                        
                        new_col_names = {
                            'health_sensitivity': f'{scenario_prefix}{category}_health_sensitivity',
                            'benefit': f'{scenario_prefix}{category}_benefit_{scc}_{rcm_model}_{cr_function}',
                            'total_npv': f'{scenario_prefix}{category}_total_npv_{scc}_{rcm_model}_{cr_function}',
                            'total_npv_climate': f'{scenario_prefix}{category}_total_npv_climateOnly_{scc}',
                            'total_npv_health': f'{scenario_prefix}{category}_total_npv_healthOnly_{rcm_model}_{cr_function}',
                            'adoption': f'{scenario_prefix}{category}_adoption_{scc}_{rcm_model}_{cr_function}',
                            'impact': f'{scenario_prefix}{category}_impact_{scc}_{rcm_model}_{cr_function}'
                        }
                        
                        category_columns_to_mask.extend(new_col_names.values())
                        
                        # Validate required columns
                        required_cols = [lessWTP_private_npv_col, moreWTP_private_npv_col, public_npv_col]
                        missing_required = [col for col in required_cols if col not in df_copy.columns]
                        if missing_required:
                            raise KeyError(f"Required NPV columns missing: {missing_required}")
                        
                        # Convert to numeric
                        for col in required_cols + ([rebate_col] if rebate_col in df_copy.columns else []):
                            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                        
                        # Create new columns DataFrame
                        df_new_columns = pd.DataFrame(index=df_copy.index)
                        
                        # Initialize columns
                        for col_name in new_col_names.values():
                            if col_name == new_col_names['health_sensitivity']:
                                df_new_columns[col_name] = f'{rcm_model}, {cr_function}'
                            else:
                                df_new_columns[col_name] = create_retrofit_only_series(df_copy, valid_mask)
                        
                        # Calculate additional public benefit
                        if policy_scenario == 'No Inflation Reduction Act':
                            df_new_columns.loc[valid_mask, new_col_names['benefit']] = 0.0
                        else:
                            if rebate_col in df_copy.columns:
                                valid_rows = valid_mask & df_copy[public_npv_col].notna() & df_copy[rebate_col].notna()
                                df_new_columns.loc[valid_rows, new_col_names['benefit']] = (
                                    df_copy.loc[valid_rows, public_npv_col] - 
                                    df_copy.loc[valid_rows, rebate_col]
                                ).clip(lower=0)
                            else:
                                valid_rows = valid_mask & df_copy[public_npv_col].notna()
                                df_new_columns.loc[valid_rows, new_col_names['benefit']] = (
                                    df_copy.loc[valid_rows, public_npv_col]
                                ).clip(lower=0)
                        
                        # ===== CLIMATE AND HEALTH COMBINED TOTAL NPV (ORIGINAL LOGIC) =====                  
                        valid_npv_rows = valid_mask & df_copy[moreWTP_private_npv_col].notna() & df_copy[public_npv_col].notna()
                        df_new_columns.loc[valid_npv_rows, new_col_names['total_npv']] = (
                            df_copy.loc[valid_npv_rows, moreWTP_private_npv_col] + 
                            df_copy.loc[valid_npv_rows, public_npv_col]
                        )
                        
                        # ===== CLIMATE ONLY TOTAL NPV =====                  
                        valid_npv_climate_rows = (
                            valid_mask & 
                            df_copy[moreWTP_private_npv_col].notna() & 
                            df_copy[climate_npv_col].notna())
                        
                        if valid_npv_climate_rows.any():
                            df_new_columns.loc[valid_npv_climate_rows, new_col_names['total_npv_climate']] = (
                                df_copy.loc[valid_npv_climate_rows, moreWTP_private_npv_col] + 
                                df_copy.loc[valid_npv_climate_rows, climate_npv_col]
                            )

                        # ===== HEALTH ONLY TOTAL NPV =====                  
                        valid_npv_health_rows = (
                            valid_mask & 
                            df_copy[moreWTP_private_npv_col].notna() & 
                            df_copy[health_npv_col].notna())
                        
                        if valid_npv_health_rows.any():
                            df_new_columns.loc[valid_npv_health_rows, new_col_names['total_npv_health']] = (
                                df_copy.loc[valid_npv_health_rows, moreWTP_private_npv_col] + 
                                df_copy.loc[valid_npv_health_rows, health_npv_col]
                            )

                        # Set defaults
                        df_new_columns[new_col_names['adoption']] = 'N/A: Invalid Baseline Fuel/Tech'
                        df_new_columns[new_col_names['impact']] = 'N/A: Invalid Baseline Fuel/Tech'
                        
                        # Adoption tier classification
                        valid_homes_with_npv = valid_mask & df_copy[lessWTP_private_npv_col].notna() & df_copy[moreWTP_private_npv_col].notna()
                        df_new_columns.loc[valid_homes_with_npv, new_col_names['adoption']] = 'Tier 4: Averse'
                        
                        no_upgrade_mask = valid_mask & df_copy[upgrade_column].isna()
                        df_new_columns.loc[no_upgrade_mask, new_col_names['adoption']] = 'N/A: Already Upgraded!'
                        
                        tier1_mask = valid_mask & df_copy[lessWTP_private_npv_col].notna() & (df_copy[lessWTP_private_npv_col] > 0)
                        df_new_columns.loc[tier1_mask, new_col_names['adoption']] = 'Tier 1: Feasible'
                        
                        tier2_mask = (valid_mask & 
                                     df_copy[lessWTP_private_npv_col].notna() & 
                                     df_copy[moreWTP_private_npv_col].notna() & 
                                     (df_copy[lessWTP_private_npv_col] < 0) & 
                                     (df_copy[moreWTP_private_npv_col] > 0))
                        df_new_columns.loc[tier2_mask, new_col_names['adoption']] = 'Tier 2: Feasible vs. Alternative'
                        
                        tier3_mask = (valid_mask & 
                                     df_copy[lessWTP_private_npv_col].notna() & 
                                     df_copy[moreWTP_private_npv_col].notna() & 
                                     df_new_columns[new_col_names['total_npv']].notna() & 
                                     (df_copy[lessWTP_private_npv_col] < 0) & 
                                     (df_copy[moreWTP_private_npv_col] < 0) & 
                                     (df_new_columns[new_col_names['total_npv']] > 0))
                        df_new_columns.loc[tier3_mask, new_col_names['adoption']] = 'Tier 3: Subsidy-Dependent Feasibility'
                        
                        # Public impact classification
                        df_new_columns.loc[valid_mask, new_col_names['impact']] = 'N/A: Already Upgraded!'
                        
                        zero_impact_mask = valid_mask & df_copy[public_npv_col].notna() & (df_copy[public_npv_col] == 0)
                        df_new_columns.loc[zero_impact_mask, new_col_names['impact']] = 'Public NPV is Zero'
                        
                        benefit_mask = valid_mask & df_copy[public_npv_col].notna() & (df_copy[public_npv_col] > 0)
                        df_new_columns.loc[benefit_mask, new_col_names['impact']] = 'Public Benefit'
                        
                        detriment_mask = valid_mask & df_copy[public_npv_col].notna() & (df_copy[public_npv_col] < 0)
                        df_new_columns.loc[detriment_mask, new_col_names['impact']] = 'Public Detriment'
                        
                        # Apply new columns
                        df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
                            df_copy, df_new_columns, category, category_columns_to_mask, all_columns_to_mask
                        )
                        
                        scc_processed += 1
                        
                    except Exception as e:
                        if verbose:
                            print(f"Error processing {scc} for {category}: {str(e)}")
                        continue
                
                # Category summary
                category_summaries.append(f"  {category}: {valid_count:,}/{total_count:,} valid homes, {scc_processed} scenarios")
                
            except Exception as e:
                print(f"ERROR: Failed to process {category}: {str(e)}")
                continue
        
        # Apply final masking
        total_columns = sum(len(cols) for cols in all_columns_to_mask.values())
        df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=False)
        
        # Final summary output
        if not verbose:
            print("\nSummary:")
            for summary in category_summaries:
                print(summary)
            print(f"  Total columns: {total_columns}")
        
        return df_copy
        
    except (ValueError, KeyError, TypeError) as e:
        print(f"ERROR: {str(e)}")
        raise
    except Exception as e:
        error_message = f"An unexpected error occurred in adoption_decision: {str(e)}"
        print(f"ERROR: {error_message}")
        raise RuntimeError(error_message) from e
