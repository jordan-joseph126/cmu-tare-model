import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from cmu_tare_model.constants import SCC_ASSUMPTIONS, RCM_MODELS, CR_FUNCTIONS, UPGRADE_COLUMNS, VERBOSE
from cmu_tare_model.utils.modeling_params import define_scenario_params
from cmu_tare_model.utils.validation_framework import (
    create_retrofit_only_series,
    initialize_validation_tracking,
    apply_new_columns_to_dataframe,
    apply_final_masking
)
from cmu_tare_model.utils.discounting import PRIVATE_DISCOUNTING_METHOD_SUFFIXES

# =============================================================
# HELPER FUNCTIONS
# =============================================================
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


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    context_params: Dict[str, str]
) -> None:
    """
    Validates that all required columns exist in DataFrame.
    
    Args:
        df: DataFrame to validate.
        required_columns: List of column names that must exist.
        context_params: Dictionary of parameters for error message context.
        
    Raises:
        KeyError: If any required columns are missing, with complete list.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        unique_missing = sorted(set(missing_columns))
        context_str = "\n".join(f"  {key}: {value}" for key, value in context_params.items())
        
        error_msg = (
            f"Required columns missing:\n"
            f"{context_str}\n"
            f"\nMissing columns ({len(unique_missing)}):\n"
        )
        error_msg += "\n".join(f"  - {col}" for col in unique_missing)
        raise KeyError(error_msg)

def _calculate_total_npv(
    df: pd.DataFrame,
    valid_mask: pd.Series,
    npv_col1: str,
    npv_col2: str,
    output_col: str
) -> pd.DataFrame:
    """
    Calculates total NPV by summing two NPV columns.
    
    Args:
        df: DataFrame containing NPV columns.
        valid_mask: Boolean mask for valid homes.
        npv_col1: First NPV column name.
        npv_col2: Second NPV column name.
        output_col: Output column name for total NPV.
        
    Returns:
        DataFrame with single column containing total NPV values.
    """
    df_result = pd.DataFrame(index=df.index)
    df_result[output_col] = create_retrofit_only_series(df, valid_mask)
    
    valid_rows = valid_mask & df[npv_col1].notna() & df[npv_col2].notna()
    if valid_rows.any():
        df_result.loc[valid_rows, output_col] = (
            df.loc[valid_rows, npv_col1] + df.loc[valid_rows, npv_col2]
        )
    
    return df_result


# =============================================================
# MAIN FUNCTIONS 
# =============================================================

def adoption_decision(
    df: pd.DataFrame,
    menu_mp: int,
    policy_scenario: str,
    rcm_model: str,
    cr_function: str,
    discount_rate_col: str,
    verbose: bool = VERBOSE,
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
        discount_rate_col: Discount rate column name for private discounting.
        verbose: Enable detailed output for debugging (default: False).
        
    Returns:
        DataFrame with adoption tier and public impact classifications.
        
    Raises:
        ValueError: If input parameters are invalid.
        KeyError: If required columns are missing.
    """
    # Validate inputs (fail fast on invalid parameters)
    validate_input_parameters(menu_mp, policy_scenario, rcm_model, cr_function)
    
    # Check required upgrade columns exist
    missing_upgrades = [col for col in UPGRADE_COLUMNS.values() if col not in df.columns]
    if missing_upgrades:
        raise KeyError(f"Required upgrade columns missing: {missing_upgrades}")
    
    # Setup
    df_copy = df.copy()
    df_copy = fix_duplicate_columns(df_copy)
    
    # Get scenario prefix
    scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)
    
    method_suffix = PRIVATE_DISCOUNTING_METHOD_SUFFIXES[discount_rate_col]

    # ===== Check ALL required columns exist before processing =====
    # Build list of all required columns
    required_columns = []
    for category in UPGRADE_COLUMNS.keys():
        for scc in SCC_ASSUMPTIONS:
            lessWTP_col = f'{scenario_prefix}{category}_private_npv_lessWTP{method_suffix}'
            moreWTP_col = f'{scenario_prefix}{category}_private_npv_moreWTP{method_suffix}'
            public_npv_col = f'{scenario_prefix}{category}_public_npv_{scc}_{rcm_model}_{cr_function}'
            required_columns.extend([lessWTP_col, moreWTP_col, public_npv_col])
        
        if policy_scenario == 'AEO2023 Reference Case':
            rebate_col = f'mp{menu_mp}_{category}_rebate_amount'
            required_columns.append(rebate_col)
    
    # Validate all required columns exist
    _validate_required_columns(
        df=df_copy,
        required_columns=required_columns,
        context_params={'Analysis': 'Adoption', 'Method': method_suffix, 'Policy': policy_scenario, 'RCM Model': rcm_model, 'CR Function': cr_function}
    )

    # Single header for nation-level analysis
    if verbose:
        print(f"\nAdoption Analysis: {policy_scenario} | {rcm_model}-{cr_function}")
    
    all_columns_to_mask = {cat: [] for cat in UPGRADE_COLUMNS}
    category_summaries = []
    
    # Process each equipment category
    for category, upgrade_column in UPGRADE_COLUMNS.items():
        # Initialize validation tracking (silent unless verbose)
        df_copy, valid_mask, _, category_columns_to_mask = initialize_validation_tracking(
            df_copy, category, menu_mp, verbose=verbose)
        
        valid_count = valid_mask.sum()
        total_count = len(df_copy)
        
        # Process all SCC assumptions
        scc_processed = 0
        for scc in SCC_ASSUMPTIONS:
            # Define column names (validation guarantees these exist)
            lessWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_lessWTP{method_suffix}'
            moreWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_moreWTP{method_suffix}'
            public_npv_col = f'{scenario_prefix}{category}_public_npv_{scc}_{rcm_model}_{cr_function}'
            rebate_col = f'mp{menu_mp}_{category}_rebate_amount'
            
            new_col_names = {
                'health_sensitivity': f'{scenario_prefix}{category}_health_sensitivity',
                'benefit': f'{scenario_prefix}{category}_benefit_{scc}_{rcm_model}_{cr_function}',
                'total_npv': f'{scenario_prefix}{category}_total_npv_{scc}_{rcm_model}_{cr_function}{method_suffix}',
                'adoption': f'{scenario_prefix}{category}_adoption_{scc}_{rcm_model}_{cr_function}{method_suffix}',
                'impact': f'{scenario_prefix}{category}_impact_{scc}_{rcm_model}_{cr_function}'
            }
            
            category_columns_to_mask.extend(new_col_names.values())
            
            # Convert to numeric
            for col in [lessWTP_private_npv_col, moreWTP_private_npv_col, public_npv_col]:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            
            if rebate_col in df_copy.columns:
                df_copy[rebate_col] = pd.to_numeric(df_copy[rebate_col], errors='coerce')
            
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
            
            # Calculate total NPV values
            valid_npv_rows = valid_mask & df_copy[moreWTP_private_npv_col].notna() & df_copy[public_npv_col].notna()
            df_new_columns.loc[valid_npv_rows, new_col_names['total_npv']] = (
                df_copy.loc[valid_npv_rows, moreWTP_private_npv_col] + 
                df_copy.loc[valid_npv_rows, public_npv_col]
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
        
        # Category summary
        category_summaries.append(f"  {category}: {valid_count:,}/{total_count:,} valid homes, {scc_processed} scenarios")
    
    # Apply final masking
    total_columns = sum(len(cols) for cols in all_columns_to_mask.values())
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=verbose)
    
    # Final summary output
    if verbose:
        print("\nSummary:")
        for summary in category_summaries:
            print(summary)
        print(f"  Total columns: {total_columns}")
        
    return df_copy


def calculate_climate_only_adoption_robust(
    df: pd.DataFrame,
    menu_mp: int,
    policy_scenario: str,
    discount_rate_col: str,
    scc_assumptions: List[str] = None,
    verbose: bool = VERBOSE
) -> pd.DataFrame:
    """
    Climate-only adoption analysis with simplified output.
    
    Args:
        df: Input DataFrame.
        menu_mp: Measure package identifier.
        policy_scenario: Policy scenario name.
        discount_rate_col: Discount rate column name for private discounting.
        scc_assumptions: List of SCC assumptions to process.
        verbose: Enable detailed output.
        
    Returns:
        DataFrame with climate-only adoption analysis columns.
        
    Raises:
        ValueError: If input parameters are invalid.
        KeyError: If required columns are missing.
    """
    # Validate policy_scenario
    valid_scenarios = ['No Inflation Reduction Act', 'AEO2023 Reference Case']
    if policy_scenario not in valid_scenarios:
        raise ValueError(f"Invalid policy_scenario: '{policy_scenario}'. Must be one of {valid_scenarios}")
    
    # Use all SCC assumptions if not specified
    if scc_assumptions is None:
        scc_assumptions = SCC_ASSUMPTIONS
    
    # Validate SCC assumptions
    invalid_scc = [scc for scc in scc_assumptions if scc not in SCC_ASSUMPTIONS]
    if invalid_scc:
        raise ValueError(f"Invalid SCC assumptions: {invalid_scc}. Must be from {SCC_ASSUMPTIONS}")
    
    df_copy = df.copy()
    df_copy = fix_duplicate_columns(df_copy)
    
    # Get scenario prefix
    scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)
    
    method_suffix = PRIVATE_DISCOUNTING_METHOD_SUFFIXES[discount_rate_col]

    # Build list of all required columns
    required_columns = []
    for category in UPGRADE_COLUMNS.keys():
        for scc in scc_assumptions:
            required_columns.extend([
                f'{scenario_prefix}{category}_private_npv_moreWTP{method_suffix}',
                f'{scenario_prefix}{category}_climate_npv_{scc}'
            ])
    
    # Validate all required columns exist
    _validate_required_columns(
        df=df_copy, 
        required_columns=required_columns, 
        context_params={'Analysis': 'Climate-only', 'Method': method_suffix, 'Policy': policy_scenario}
        )
    
    if verbose:
        print(f"\nClimate-only Analysis: {policy_scenario}")
    
    all_columns_to_mask = {cat: [] for cat in UPGRADE_COLUMNS}
    
    # Process each equipment category
    for category, upgrade_column in UPGRADE_COLUMNS.items():
        # Initialize validation tracking
        df_copy, valid_mask, _, category_columns_to_mask = initialize_validation_tracking(
            df_copy, category, menu_mp, verbose=verbose)
        
        for scc in scc_assumptions:
            # Define column names (validation guarantees these exist)
            moreWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_moreWTP{method_suffix}'
            climate_npv_col = f'{scenario_prefix}{category}_climate_npv_{scc}'
            
            # Convert to numeric
            df_copy[moreWTP_private_npv_col] = pd.to_numeric(df_copy[moreWTP_private_npv_col], errors='coerce')
            df_copy[climate_npv_col] = pd.to_numeric(df_copy[climate_npv_col], errors='coerce')
            
            # Calculate total NPV (moreWTP + climate)
            output_col = f'{scenario_prefix}{category}_total_npv_climateOnly_{scc}{method_suffix}'
            df_new_columns = _calculate_total_npv(
                df_copy, valid_mask, moreWTP_private_npv_col, climate_npv_col, output_col
            )

            # Track and apply columns
            category_columns_to_mask.extend([output_col])
            df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
                df_copy, df_new_columns, category, category_columns_to_mask, all_columns_to_mask
            )
    
    # Apply final masking
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=verbose)
    
    if verbose:
        total_columns = sum(len(cols) for cols in all_columns_to_mask.values())
        print(f"  Climate-only: {total_columns} columns added")
    
    return df_copy


def calculate_health_only_adoption_robust(
    df: pd.DataFrame,
    menu_mp: int,
    policy_scenario: str,
    rcm_model: str,
    cr_function: str,
    discount_rate_col: str,
    verbose: bool = VERBOSE
) -> pd.DataFrame:
    """
    Health-only adoption analysis with simplified output.
    
    Args:
        df: Input DataFrame.
        menu_mp: Measure package identifier.
        policy_scenario: Policy scenario name.
        rcm_model: RCM model name.
        cr_function: Concentration response function.
        discount_rate_col: Discount rate column name for private discounting.
        verbose: Enable detailed output.
        
    Returns:
        DataFrame with health-only adoption analysis columns.
        
    Raises:
        ValueError: If input parameters are invalid.
        KeyError: If required columns are missing.
    """
    # Validate inputs
    validate_input_parameters(menu_mp, policy_scenario, rcm_model, cr_function)
    
    df_copy = df.copy()
    df_copy = fix_duplicate_columns(df_copy)
    
    # Get scenario prefix
    scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)

    method_suffix = PRIVATE_DISCOUNTING_METHOD_SUFFIXES[discount_rate_col]

    # Build list of all required columns
    required_columns = []
    for category in UPGRADE_COLUMNS.keys():
        required_columns.extend([
            f'{scenario_prefix}{category}_private_npv_moreWTP{method_suffix}',
            f'{scenario_prefix}{category}_health_npv_{rcm_model}_{cr_function}'
        ])
    
    # Validate all required columns exist
    _validate_required_columns(
        df=df_copy, 
        required_columns=required_columns,
        context_params={'Analysis': 'Health-only', 'Method': method_suffix, 'Policy': policy_scenario, 'RCM Model': rcm_model, 'CR Function': cr_function}
        )    

    if verbose:
        print(f"\nHealth-only Analysis: {policy_scenario} | {rcm_model}-{cr_function}")
    
    all_columns_to_mask = {cat: [] for cat in UPGRADE_COLUMNS}
    
    # Process each equipment category
    for category, upgrade_column in UPGRADE_COLUMNS.items():
        # Initialize validation tracking
        df_copy, valid_mask, _, category_columns_to_mask = initialize_validation_tracking(
            df_copy, category, menu_mp, verbose=verbose)
        
        # Define column names (validation guarantees these exist)
        moreWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_moreWTP{method_suffix}'
        health_npv_col = f'{scenario_prefix}{category}_health_npv_{rcm_model}_{cr_function}'
        
        # Convert to numeric
        df_copy[moreWTP_private_npv_col] = pd.to_numeric(df_copy[moreWTP_private_npv_col], errors='coerce')
        df_copy[health_npv_col] = pd.to_numeric(df_copy[health_npv_col], errors='coerce')
        
        # Calculate total NPV (moreWTP + health)
        output_col = f'{scenario_prefix}{category}_total_npv_healthOnly_{rcm_model}_{cr_function}{method_suffix}'
        df_new_columns = _calculate_total_npv(
            df_copy, valid_mask, moreWTP_private_npv_col, health_npv_col, output_col
        )
        
        # Track and apply columns
        category_columns_to_mask.extend([output_col])
        df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
            df_copy, df_new_columns, category, category_columns_to_mask, all_columns_to_mask
        )
    
    # Apply final masking
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=verbose)
    
    if verbose:
        total_columns = sum(len(cols) for cols in all_columns_to_mask.values())
        print(f"  Health-only: {total_columns} columns added")
    
    return df_copy
