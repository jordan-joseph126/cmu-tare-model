import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

from cmu_tare_model.constants import SCC_ASSUMPTIONS, RCM_MODELS, CR_FUNCTIONS, UPGRADE_COLUMNS
from cmu_tare_model.utils.modeling_params import define_scenario_params
from cmu_tare_model.utils.validation_framework import (
    create_retrofit_only_series,
    initialize_validation_tracking,
    apply_new_columns_to_dataframe,
    apply_final_masking
)

# =============================================================================
# ENHANCED VALIDATION FUNCTIONS - FAIL FAST, FAIL CLEARLY
# =============================================================================

def validate_input_parameters(
    menu_mp: int,
    policy_scenario: str,
    rcm_model: str,
    cr_function: str
) -> None:
    """
    Validates input parameters with detailed error messages and corrective actions.
    
    Args:
        menu_mp: Measure package identifier.
        policy_scenario: Policy scenario name.
        rcm_model: RCM model name.
        cr_function: Concentration response function name.
        
    Raises:
        ValueError: If any parameter is invalid with specific guidance.
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
        errors.append(f"Invalid policy_scenario: '{policy_scenario}'. "
                     f"Must be one of {valid_scenarios}")
    
    # Validate rcm_model
    if rcm_model not in RCM_MODELS:
        errors.append(f"Invalid rcm_model: '{rcm_model}'. "
                     f"Must be one of {RCM_MODELS}")
    
    # Validate cr_function
    if cr_function not in CR_FUNCTIONS:
        errors.append(f"Invalid cr_function: '{cr_function}'. "
                     f"Must be one of {CR_FUNCTIONS}")
    
    if errors:
        error_msg = "Parameter validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        error_msg += "\n\nCorrection: Please check the constants module for valid parameter values."
        raise ValueError(error_msg)


def validate_dataframe_preconditions(df: pd.DataFrame, context: str = "") -> None:
    """
    Validates DataFrame meets basic requirements for processing.
    
    Args:
        df: DataFrame to validate.
        context: Context string for error messages.
        
    Raises:
        ValueError: If DataFrame fails validation with specific guidance.
    """
    context_msg = f" for {context}" if context else ""
    
    if df is None:
        raise ValueError(f"DataFrame is None{context_msg}. Provide a valid DataFrame.")
    
    if df.empty:
        raise ValueError(f"DataFrame is empty{context_msg}. Ensure data is loaded before processing.")
    
    if len(df.columns) == 0:
        raise ValueError(f"DataFrame has no columns{context_msg}. Check data loading process.")
    
    # Check for required upgrade columns early
    missing_upgrade_cols = [col for col in UPGRADE_COLUMNS.values() if col not in df.columns]
    if missing_upgrade_cols:
        raise KeyError(f"Required upgrade columns missing from DataFrame{context_msg}: "
                      f"{missing_upgrade_cols}. "
                      f"Expected columns: {list(UPGRADE_COLUMNS.values())}")


def validate_and_convert_npv_columns(
    df_copy: pd.DataFrame,
    required_cols: List[str],
    optional_cols: List[str] = None,
    context: str = ""
) -> None:
    """
    Validates column existence and converts to numeric with comprehensive error handling.
    
    Args:
        df_copy: DataFrame to validate and modify in place.
        required_cols: Columns that must exist.
        optional_cols: Columns that may exist.
        context: Context for error messages.
        
    Raises:
        KeyError: If required columns are missing.
        TypeError: If numeric conversion fails.
    """
    if optional_cols is None:
        optional_cols = []
    
    context_msg = f" for {context}" if context else ""
    
    # Check required columns
    missing_required = [col for col in required_cols if col not in df_copy.columns]
    if missing_required:
        available_similar = []
        for missing_col in missing_required:
            similar = [col for col in df_copy.columns if any(part in col.lower() 
                      for part in missing_col.lower().split('_'))]
            if similar:
                available_similar.extend(similar[:3])  # Show up to 3 similar columns
        
        error_msg = f"Required NPV columns missing from DataFrame{context_msg}: {missing_required}"
        if available_similar:
            error_msg += f"\nSimilar available columns: {available_similar}"
        error_msg += f"\nCorrection: Ensure NPV calculation has been run before adoption analysis."
        raise KeyError(error_msg)
    
    # Convert columns to numeric
    conversion_errors = []
    all_cols_to_convert = required_cols + [col for col in optional_cols if col in df_copy.columns]
    
    for col in all_cols_to_convert:
        try:
            original_dtype = df_copy[col].dtype
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            
            # Check if conversion was successful
            if df_copy[col].isna().all():
                conversion_errors.append(f"Column '{col}' converted to all NaN values "
                                       f"(original dtype: {original_dtype})")
        except Exception as e:
            conversion_errors.append(f"Column '{col}': {str(e)}")
    
    if conversion_errors:
        error_msg = f"Numeric conversion failed{context_msg}:\n"
        error_msg += "\n".join(f"  - {error}" for error in conversion_errors)
        error_msg += "\nCorrection: Check that NPV columns contain numeric data."
        raise TypeError(error_msg)


# =============================================================================
# CORE HELPER FUNCTIONS - DRY PRINCIPLE IMPLEMENTATION
# =============================================================================

def calculate_total_npv_values(
    df_new_columns: pd.DataFrame,
    df_copy: pd.DataFrame,
    valid_mask: pd.Series,
    private_npv_cols: Tuple[str, str],
    public_npv_col: str,
    output_col_name: str  # Single output column
) -> pd.DataFrame:
    """
    Calculates only moreWTP total NPV values (lessWTP removed for performance).
    
    Args:
        output_col_name: Single output column name for moreWTP total NPV.
    """
    lessWTP_private_col, moreWTP_private_col = private_npv_cols
    
    # Validate column existence (lessWTP still needed for tier classification)
    required_cols = [lessWTP_private_col, moreWTP_private_col, public_npv_col]
    missing_cols = [col for col in required_cols if col not in df_copy.columns]
    if missing_cols:
        raise KeyError(f"Required columns missing for total NPV calculation: {missing_cols}")
    
    try:
        # Calculate only moreWTP total NPV (used for Tier 3 classification)
        valid_more_rows = (valid_mask & 
                          df_copy[moreWTP_private_col].notna() & 
                          df_copy[public_npv_col].notna())
        
        if valid_more_rows.any():
            df_new_columns.loc[valid_more_rows, output_col_name] = (
                df_copy.loc[valid_more_rows, moreWTP_private_col] + 
                df_copy.loc[valid_more_rows, public_npv_col]
            )
            
    except Exception as e:
        if "DataFrame" in str(e):
            raise ValueError(f"Column access returned DataFrame instead of Series. "
                           f"This typically indicates duplicate column names. "
                           f"Run fix_duplicate_columns() first. Error: {str(e)}")
        else:
            raise ValueError(f"Failed to calculate total NPV values: {str(e)}")
    
    return df_new_columns


def classify_adoption_tiers(
    df_new_columns: pd.DataFrame,
    df_copy: pd.DataFrame,
    valid_mask: pd.Series,
    upgrade_column: str,
    private_npv_cols: Tuple[str, str],
    moreWTP_total_col: str,  # Single column
    adoption_col: str
) -> pd.DataFrame:
    """
    Classifies homes into adoption tiers (updated to use only moreWTP total NPV).
    
    The lessWTP total NPV calculation has been removed for performance optimization
    since it's not used in the final adoption decision logic.

    This function implements the standard 4-tier adoption classification:
    - Tier 1: Feasible (positive private NPV with total capital costs)
    - Tier 2: Feasible vs. Alternative (negative total cost NPV, positive net cost NPV)
    - Tier 3: Subsidy-Dependent (negative private NPV, positive total NPV)
    - Tier 4: Averse (negative private and total NPV)
    
    Args:
        df_new_columns: DataFrame to store classifications.
        df_copy: Source DataFrame with NPV data.
        valid_mask: Boolean mask for valid homes.
        upgrade_column: Column indicating equipment upgrade.
        private_npv_cols: Tuple of (lessWTP, moreWTP) private NPV columns.
        total_npv_cols: Tuple of (lessWTP, moreWTP) total NPV columns.
        adoption_col: Output column name for adoption classification.
        
    Returns:
        Updated df_new_columns with adoption tier classifications.
    """
    lessWTP_private_col, moreWTP_private_col = private_npv_cols    
    
    # Column already initialized with default value, just set defaults for valid homes
    valid_homes_with_npv = (valid_mask & 
                           df_copy[lessWTP_private_col].notna() & 
                           df_copy[moreWTP_private_col].notna())
    df_new_columns.loc[valid_homes_with_npv, adoption_col] = 'Tier 4: Averse'
    
    # Handle no upgrade cases
    no_upgrade_mask = valid_mask & df_copy[upgrade_column].isna()
    df_new_columns.loc[no_upgrade_mask, adoption_col] = 'N/A: Already Upgraded!'
    
    # Tier 1: Economically feasible (positive private NPV with total capital costs)
    tier1_mask = (valid_mask & 
                  df_copy[lessWTP_private_col].notna() & 
                  (df_copy[lessWTP_private_col] > 0))
    df_new_columns.loc[tier1_mask, adoption_col] = 'Tier 1: Feasible'
    
    # Tier 2: Feasible vs alternative (negative total cost, positive net cost)
    tier2_mask = (valid_mask & 
                  df_copy[lessWTP_private_col].notna() & 
                  df_copy[moreWTP_private_col].notna() & 
                  (df_copy[lessWTP_private_col] < 0) & 
                  (df_copy[moreWTP_private_col] > 0))
    df_new_columns.loc[tier2_mask, adoption_col] = 'Tier 2: Feasible vs. Alternative'
    
    # Tier 3: Subsidy-dependent (negative private NPV, positive total NPV)
    tier3_mask = (valid_mask & 
                  df_copy[lessWTP_private_col].notna() & 
                  df_copy[moreWTP_private_col].notna() & 
                  df_new_columns[moreWTP_total_col].notna() & 
                  (df_copy[lessWTP_private_col] < 0) & 
                  (df_copy[moreWTP_private_col] < 0) & 
                  (df_new_columns[moreWTP_total_col] > 0))
    df_new_columns.loc[tier3_mask, adoption_col] = 'Tier 3: Subsidy-Dependent Feasibility'
    
    return df_new_columns


def determine_public_impacts(
    df_new_columns: pd.DataFrame,
    df_copy: pd.DataFrame,
    valid_mask: pd.Series,
    public_npv_col: str,
    impact_col: str
) -> pd.DataFrame:
    """
    Determines public impact classification based on public NPV values.
    
    This function classifies the public impact of retrofits as:
    - Public Benefit: Positive public NPV
    - Public Detriment: Negative public NPV
    - Public NPV is Zero: Zero public NPV
    - N/A: Already Upgraded: For homes that don't need upgrades
    
    Args:
        df_new_columns: DataFrame to store impact classifications.
        df_copy: Source DataFrame with public NPV data.
        valid_mask: Boolean mask for valid homes.
        public_npv_col: Column name for public NPV values.
        impact_col: Output column name for impact classification.
        
    Returns:
        Updated df_new_columns with public impact classifications.
    """
    # Column already initialized with default value, just update valid homes

    # Zero impact
    zero_impact_mask = (valid_mask & 
                       df_copy[public_npv_col].notna() & 
                       (df_copy[public_npv_col] == 0))
    df_new_columns.loc[zero_impact_mask, impact_col] = 'Public NPV is Zero'
    
    # Public benefit
    benefit_mask = (valid_mask & 
                   df_copy[public_npv_col].notna() & 
                   (df_copy[public_npv_col] > 0))
    df_new_columns.loc[benefit_mask, impact_col] = 'Public Benefit'
    
    # Public detriment
    detriment_mask = (valid_mask & 
                     df_copy[public_npv_col].notna() & 
                     (df_copy[public_npv_col] < 0))
    df_new_columns.loc[detriment_mask, impact_col] = 'Public Detriment'
    
    return df_new_columns


def calculate_additional_public_benefit(
    df_new_columns: pd.DataFrame,
    df_copy: pd.DataFrame,
    valid_mask: pd.Series,
    policy_scenario: str,
    public_npv_col: str,
    rebate_col: str,
    benefit_col: str
) -> pd.DataFrame:
    """
    Calculates additional public benefit accounting for IRA rebates.
    
    Args:
        df_new_columns: DataFrame to store calculated benefits.
        df_copy: Source DataFrame with NPV and rebate data.
        valid_mask: Boolean mask for valid homes.
        policy_scenario: Policy scenario determining rebate applicability.
        public_npv_col: Column name for public NPV values.
        rebate_col: Column name for rebate amounts.
        benefit_col: Output column name for additional benefit.
        
    Returns:
        Updated df_new_columns with additional public benefit values.
    """
    if policy_scenario == 'No Inflation Reduction Act':
        df_new_columns.loc[valid_mask, benefit_col] = 0.0
    else:
        if rebate_col in df_copy.columns:
            valid_rows = (valid_mask & 
                         df_copy[public_npv_col].notna() & 
                         df_copy[rebate_col].notna())
            df_new_columns.loc[valid_rows, benefit_col] = (
                df_copy.loc[valid_rows, public_npv_col] - 
                df_copy.loc[valid_rows, rebate_col]
            ).clip(lower=0)
        else:
            valid_rows = valid_mask & df_copy[public_npv_col].notna()
            df_new_columns.loc[valid_rows, benefit_col] = (
                df_copy.loc[valid_rows, public_npv_col]
            ).clip(lower=0)
    
    return df_new_columns


# =============================================================================
# ROBUST COLUMN ACCESS UTILITIES
# =============================================================================

def fix_duplicate_columns(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Fix duplicate column names by removing duplicates, keeping the first occurrence.
    
    This follows the pandas pattern used in validation_framework.py for handling
    overlapping columns by using built-in pandas methods.
    
    Args:
        df: DataFrame with potential duplicate columns.
        verbose: Whether to print detailed information.
        
    Returns:
        DataFrame with duplicate columns removed.
    """
    if verbose:
        print("\n=== FIXING DUPLICATE COLUMNS ===")
    
    # Count duplicates for reporting
    original_col_count = len(df.columns)
    duplicate_count = original_col_count - len(df.columns.unique())
    
    if duplicate_count == 0:
        if verbose:
            print("✅ No duplicate columns found")
        return df
    
    if verbose:
        print(f"Found {duplicate_count} duplicate columns")
        # Show which columns are duplicated
        duplicated_cols = df.columns[df.columns.duplicated(keep=False)].unique()
        print(f"Duplicated column names: {duplicated_cols.tolist()}")
    
    # Remove duplicates using pandas built-in method (follows validation_framework pattern)
    df_fixed = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    if verbose:
        final_col_count = len(df_fixed.columns)
        removed_count = original_col_count - final_col_count
        print(f"✅ Removed {removed_count} duplicate columns. "
              f"DataFrame now has {final_col_count} unique column names")
    
    return df_fixed


def safe_series_access(df: pd.DataFrame, column_name: str, context: str = "") -> pd.Series:
    """
    Safely access a column ensuring it returns a Series, not a DataFrame.
    
    Args:
        df: DataFrame to access.
        column_name: Name of column to access.
        context: Context for error messages.
        
    Returns:
        Series from the DataFrame.
        
    Raises:
        KeyError: If column doesn't exist.
        ValueError: If multiple columns match (returns DataFrame).
    """
    if column_name not in df.columns:
        available_cols = [col for col in df.columns if column_name.lower() in col.lower()]
        context_msg = f" for {context}" if context else ""
        if available_cols:
            raise KeyError(f"Column '{column_name}' not found{context_msg}. "
                          f"Similar columns: {available_cols[:5]}")
        else:
            raise KeyError(f"Column '{column_name}' not found{context_msg}")
    
    duplicate_count = df.columns.tolist().count(column_name)
    if duplicate_count > 1:
        context_msg = f" for {context}" if context else ""
        raise ValueError(f"Multiple columns named '{column_name}' found{context_msg} "
                        f"(count: {duplicate_count}). Use fix_duplicate_columns() first.")
    
    result = df[column_name]
    
    if not isinstance(result, pd.Series):
        context_msg = f" for {context}" if context else ""
        raise ValueError(f"Column '{column_name}' returned {type(result)} instead of Series{context_msg}")
    
    return result


def robust_numeric_conversion(df: pd.DataFrame, column_name: str, context: str = "") -> pd.Series:
    """
    Robustly convert a column to numeric with comprehensive error handling.
    
    Args:
        df: DataFrame containing the column.
        column_name: Name of column to convert.
        context: Context for error messages.
        
    Returns:
        Numeric Series.
        
    Raises:
        KeyError: If column doesn't exist.
        ValueError: If column access returns DataFrame.
        TypeError: If numeric conversion fails.
    """
    try:
        series_data = safe_series_access(df, column_name, context)
    except (KeyError, ValueError) as e:
        raise e
    
    try:
        converted = pd.to_numeric(series_data, errors='coerce')
        return converted
    except Exception as e:
        context_msg = f" for {context}" if context else ""
        raise TypeError(f"Failed to convert column '{column_name}' to numeric{context_msg}: {str(e)}")


# =============================================================================
# REFACTORED MAIN FUNCTIONS
# =============================================================================

# =============================================================================================================
# REMOVE LESS WTP TOTAL NPV. LESS WTP TOTAL NPV IS NOT USED IN ADOPTION ANALYSIS
# =============================================================================================================
def adoption_decision(
    df: pd.DataFrame,
    menu_mp: int,
    policy_scenario: str,
    rcm_model: str,
    cr_function: str,
) -> pd.DataFrame:
    """
    Updates DataFrame with adoption decisions and public impacts based on NPV analysis.
    
    This function has been refactored to use helper functions following the DRY principle.
    All common logic patterns have been extracted to eliminate code duplication.
    
    Args:
        df: DataFrame containing home equipment data.
        menu_mp: Measure package identifier.
        policy_scenario: Policy scenario for electricity grid projections.
        rcm_model: RCM model for health impact analysis.
        cr_function: Concentration response function for health analysis.
        
    Returns:
        DataFrame with additional adoption decision and public impact columns.
        
    Raises:
        ValueError: If input parameters are invalid.
        KeyError: If required columns are missing.
        TypeError: If data type conversion fails.
    """
    try:
        # ========== FAIL FAST VALIDATION ==========
        validate_input_parameters(menu_mp, policy_scenario, rcm_model, cr_function)
        validate_dataframe_preconditions(df, "adoption_decision")
        
        df_copy = df.copy()
        
        # Fix duplicate columns early to prevent DataFrame vs Series issues
        df_copy = fix_duplicate_columns(df_copy, verbose=False)
        
        # Get scenario parameters
        try:
            scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)
        except Exception as e:
            raise ValueError(f"Error determining scenario parameters: {str(e)}")
        
        all_columns_to_mask = []
        
        # ========== PROCESS EACH EQUIPMENT CATEGORY ==========
        for category, upgrade_column in UPGRADE_COLUMNS.items():
            try:
                print(f"\nCalculating Adoption Potential for {category} under '{policy_scenario}' Scenario...")
                
                # Initialize validation tracking
                df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
                    df_copy, category, menu_mp, verbose=True)
                
                print(f"Found {valid_mask.sum()} valid homes out of {len(df_copy)} for {category} adoption potential")
                
                # Process each SCC assumption
                for scc in SCC_ASSUMPTIONS:
                    try:
                        context = f"{category}/{scc}/{rcm_model}/{cr_function}"
                        print(f"\nProcessing SCC assumption: {scc} for {context}")
                        
                        # ========== DEFINE COLUMN NAMES ==========
                        lessWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_lessWTP'
                        moreWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_moreWTP'
                        public_npv_col = f'{scenario_prefix}{category}_public_npv_{scc}_{rcm_model}_{cr_function}'
                        rebate_col = f'mp{menu_mp}_{category}_rebate_amount'

                        # =============================================================================================================
                        # REMOVE LESS WTP TOTAL NPV. LESS WTP TOTAL NPV IS NOT USED IN ADOPTION ANALYSIS
                        # =============================================================================================================
                        new_col_names = {
                            'health_sensitivity': f'{scenario_prefix}{category}_health_sensitivity',
                            'benefit': f'{scenario_prefix}{category}_benefit_{scc}_{rcm_model}_{cr_function}',
                            'moreWTP_total_npv': f'{scenario_prefix}{category}_total_npv_moreWTP_{scc}_{rcm_model}_{cr_function}',
                            'adoption': f'{scenario_prefix}{category}_adoption_{scc}_{rcm_model}_{cr_function}',
                            'impact': f'{scenario_prefix}{category}_impact_{scc}_{rcm_model}_{cr_function}'
                        }
                        
                        category_columns_to_mask.extend(new_col_names.values())
                        
                        # ========== VALIDATE AND CONVERT COLUMNS ==========
                        required_cols = [lessWTP_private_npv_col, moreWTP_private_npv_col, public_npv_col]
                        optional_cols = [rebate_col]
                        
                        validate_and_convert_npv_columns(df_copy, required_cols, optional_cols, context)
                        
                        # ========== CREATE NEW COLUMNS DATAFRAME ==========
                        df_new_columns = pd.DataFrame(index=df_copy.index)
                        
                        # Initialize result columns                        
                        for col_name in new_col_names.values():
                            if col_name == new_col_names['health_sensitivity']:
                                df_new_columns[col_name] = f'{rcm_model}, {cr_function}'
                            elif 'adoption' in col_name or 'impact' in col_name:
                                # Initialize categorical columns with object dtype to prevent FutureWarning
                                df_new_columns[col_name] = pd.Series('N/A: Invalid Baseline Fuel/Tech', 
                                                                    index=df_copy.index, dtype='object')
                            else:
                                # Use existing pattern for numeric columns (NPV, benefit calculations)
                                df_new_columns[col_name] = create_retrofit_only_series(df_copy, valid_mask)

                        # ========== USE HELPER FUNCTIONS (DRY PRINCIPLE) ==========
                        
                        # Calculate additional public benefit using helper
                        df_new_columns = calculate_additional_public_benefit(
                            df_new_columns, df_copy, valid_mask, policy_scenario,
                            public_npv_col, rebate_col, new_col_names['benefit']
                        )
                        
                        # Calculate only moreWTP total NPV (lessWTP removed for performance)
                        df_new_columns = calculate_total_npv_values(
                            df_new_columns, df_copy, valid_mask,
                            private_npv_cols=(lessWTP_private_npv_col, moreWTP_private_npv_col),
                            public_npv_col=public_npv_col,
                            output_col_name=new_col_names['moreWTP_total_npv']
                        )

                        # Classify adoption tiers using updated helper
                        df_new_columns = classify_adoption_tiers(
                            df_new_columns, df_copy, valid_mask, upgrade_column,
                            private_npv_cols=(lessWTP_private_npv_col, moreWTP_private_npv_col),
                            moreWTP_total_col=new_col_names['moreWTP_total_npv'],
                            adoption_col=new_col_names['adoption']
                        )
                        
                        # Determine public impacts using helper
                        df_new_columns = determine_public_impacts(
                            df_new_columns=df_new_columns,
                            df_copy=df_copy,
                            valid_mask=valid_mask,
                            public_npv_col=public_npv_col,
                            impact_col=new_col_names['impact']
                        )
                        
                        # Apply new columns to DataFrame
                        df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
                            df_copy, df_new_columns, category, category_columns_to_mask, all_columns_to_mask
                        )
                        
                    except Exception as e:
                        error_msg = f"Error processing SCC assumption {scc} for {category}: {str(e)}"
                        print(f"❌ {error_msg}")
                        raise RuntimeError(error_msg) from e
                        
            except Exception as e:
                error_msg = f"Error processing equipment category {category}: {str(e)}"
                print(f"❌ {error_msg}")
                raise RuntimeError(error_msg) from e
        
        # Apply final verification masking
        df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=True)
        
        return df_copy
        
    except (ValueError, KeyError, TypeError) as e:
        print(f"ERROR: {str(e)}")
        raise
    except Exception as e:
        error_message = f"An unexpected error occurred in adoption_decision: {str(e)}"
        print(f"ERROR: {error_message}")
        raise RuntimeError(error_message) from e


def calculate_climate_only_adoption_robust(
    df: pd.DataFrame,
    menu_mp: int,
    policy_scenario: str,
    scc_assumptions: List[str] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Climate-only adoption analysis using refactored helper functions.
    
    This function has been refactored to eliminate code duplication by using
    the same helper functions as the main adoption_decision function.
    """
    if scc_assumptions is None:
        scc_assumptions = SCC_ASSUMPTIONS
    
    # Validate inputs
    validate_input_parameters(menu_mp, policy_scenario, 'ap2', 'acs')  # Use dummy values for validation
    validate_dataframe_preconditions(df, "climate-only analysis")
    
    df_copy = df.copy()
    df_copy = fix_duplicate_columns(df_copy, verbose=verbose)
    
    # Get scenario prefix
    try:
        scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)
    except Exception as e:
        raise ValueError(f"Error determining scenario parameters: {str(e)}")
    
    if verbose:
        print(f"Starting climate-only adoption analysis for {policy_scenario}")
    
    all_columns_to_mask = []
    
    # Process each equipment category
    for category, upgrade_column in UPGRADE_COLUMNS.items():
        try:
            if verbose:
                print(f"\nProcessing climate-only analysis for {category}...")
            
            # Initialize validation tracking
            df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
                df_copy, category, menu_mp, verbose=verbose)
            
            # Process each SCC assumption
            for scc in scc_assumptions:
                try:
                    context = f"{category}/{scc}/climate-only"
                    if verbose:
                        print(f"  Processing SCC assumption: {scc}")
                    
                    # Define column names
                    lessWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_lessWTP'
                    moreWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_moreWTP'
                    climate_npv_col = f'{scenario_prefix}{category}_climate_npv_{scc}'
                    
                    # Validate columns
                    required_cols = [lessWTP_private_npv_col, moreWTP_private_npv_col, climate_npv_col]
                    validate_and_convert_npv_columns(df_copy, required_cols, context=context)
                    
                    # Define output column names
                    climate_col_names = {
                        'moreWTP_total_npv': f'{scenario_prefix}{category}_total_npv_moreWTP_climateOnly_{scc}',
                    }
                    
                    # Create new columns DataFrame
                    df_new_columns = pd.DataFrame(index=df_copy.index)

                    # Since we only have moreWTP_total_npv in climate_col_names
                    # Simplified initialization for climate-only
                    df_new_columns[climate_col_names['moreWTP_total_npv']] = create_retrofit_only_series(df_copy, valid_mask)

                    # ========== USE HELPER FUNCTIONS (DRY PRINCIPLE) ==========
                    
                    # Calculate only moreWTP total NPV for visualization purposes
                    df_new_columns = calculate_total_npv_values(
                        df_new_columns, df_copy, valid_mask,
                        private_npv_cols=(lessWTP_private_npv_col, moreWTP_private_npv_col),
                        public_npv_col=climate_npv_col,
                        output_col_name=climate_col_names['moreWTP_total_npv']
                    )

                    # Note: Adoption classification and impact determination removed for 
                    # simplified climate-only analysis focused on visualization needs

                    # Track and apply columns
                    category_columns_to_mask.extend(climate_col_names.values())
                    df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
                        df_copy, df_new_columns, category, category_columns_to_mask, all_columns_to_mask
                    )
                    
                    if verbose:
                        print(f"    ✅ Completed climate-only analysis for {category}/{scc}")
                        
                except Exception as e:
                    error_msg = f"Error processing SCC '{scc}' for category '{category}': {str(e)}"
                    print(f"❌ {error_msg}")
                    raise RuntimeError(error_msg) from e
                    
        except Exception as e:
            error_msg = f"Error processing category '{category}': {str(e)}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
    
    # Apply final masking
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=verbose)
    
    if verbose:
        print("✅ Climate-only adoption analysis completed successfully")
    
    return df_copy


def calculate_health_only_adoption_robust(
    df: pd.DataFrame,
    menu_mp: int,
    policy_scenario: str,
    rcm_model: str,
    cr_function: str,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Health-only adoption analysis using refactored helper functions.
    
    This function has been refactored to eliminate code duplication by using
    the same helper functions as the main adoption_decision function.
    """
    # Validate inputs
    validate_input_parameters(menu_mp, policy_scenario, rcm_model, cr_function)
    validate_dataframe_preconditions(df, "health-only analysis")
    
    df_copy = df.copy()
    df_copy = fix_duplicate_columns(df_copy, verbose=verbose)
    
    # Get scenario prefix
    try:
        scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)
    except Exception as e:
        raise ValueError(f"Error determining scenario parameters: {str(e)}")
    
    if verbose:
        print(f"Starting health-only adoption analysis for {policy_scenario}, {rcm_model}, {cr_function}")
    
    all_columns_to_mask = []
    
    # Process each equipment category
    for category, upgrade_column in UPGRADE_COLUMNS.items():
        try:
            if verbose:
                print(f"\nProcessing health-only analysis for {category}...")
            
            # Initialize validation tracking
            df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
                df_copy, category, menu_mp, verbose=verbose)
            
            try:
                context = f"{category}/{rcm_model}/{cr_function}/health-only"
                if verbose:
                    print(f"  Processing health model: {rcm_model}, {cr_function}")
                
                # Define column names
                lessWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_lessWTP'
                moreWTP_private_npv_col = f'{scenario_prefix}{category}_private_npv_moreWTP'
                health_npv_col = f'{scenario_prefix}{category}_health_npv_{rcm_model}_{cr_function}'
                
                # Validate columns
                required_cols = [lessWTP_private_npv_col, moreWTP_private_npv_col, health_npv_col]
                validate_and_convert_npv_columns(df_copy, required_cols, context=context)
                
                # Define output column names
                health_col_names = {
                    'moreWTP_total_npv': f'{scenario_prefix}{category}_total_npv_moreWTP_healthOnly_{rcm_model}_{cr_function}',
                }
                
                # Create new columns DataFrame
                df_new_columns = pd.DataFrame(index=df_copy.index)

                # Since we only have moreWTP_total_npv in health_col_names
                # # Simplified initialization for health-only
                df_new_columns[health_col_names['moreWTP_total_npv']] = create_retrofit_only_series(df_copy, valid_mask)

                # ========== USE HELPER FUNCTIONS (DRY PRINCIPLE) ==========
                
                # Calculate only moreWTP total NPV for visualization purposes
                df_new_columns = calculate_total_npv_values(
                    df_new_columns, df_copy, valid_mask,
                    private_npv_cols=(lessWTP_private_npv_col, moreWTP_private_npv_col),
                    public_npv_col=health_npv_col,
                    output_col_name=health_col_names['moreWTP_total_npv']
                )

                # Note: Adoption classification and impact determination removed for 
                # simplified climate-only analysis focused on visualization needs
                
                # Track and apply columns
                category_columns_to_mask.extend(health_col_names.values())
                df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
                    df_copy, df_new_columns, category, category_columns_to_mask, all_columns_to_mask
                )
                
                if verbose:
                    print(f"    ✅ Completed health-only analysis for {category}/{rcm_model}/{cr_function}")
                    
            except Exception as e:
                error_msg = f"Error processing health model '{rcm_model}/{cr_function}' for category '{category}': {str(e)}"
                print(f"❌ {error_msg}")
                raise RuntimeError(error_msg) from e
                
        except Exception as e:
            error_msg = f"Error processing category '{category}': {str(e)}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
    
    # Apply final masking
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=verbose)
    
    if verbose:
        print("✅ Health-only adoption analysis completed successfully")
    
    return df_copy


# =============================================================================
# DIAGNOSTIC FUNCTIONS (PRESERVED FOR DEBUGGING)
# =============================================================================

def diagnose_dataframe_vs_series_issue(df: pd.DataFrame, column_name: str, context: str = ""):
    """
    Comprehensive diagnostic function to understand why df[column_name] returns DataFrame.
    
    This function is preserved for debugging purposes when the DataFrame vs Series
    issue occurs despite our safeguards.
    """
    print(f"\n=== DATAFRAME vs SERIES DIAGNOSIS: {context} ===")
    print(f"Problematic column: '{column_name}'")
    print(f"DataFrame shape: {df.shape}")
    
    exact_match = column_name in df.columns
    print(f"Exact column match: {exact_match}")
    
    if exact_match:
        print("✅ Column exists exactly - investigating why it returns DataFrame...")
        
        duplicate_count = df.columns.tolist().count(column_name)
        print(f"Column name appears {duplicate_count} times in DataFrame")
        
        if duplicate_count > 1:
            print("❌ FOUND DUPLICATE COLUMNS! This causes df[col] to return DataFrame")
            duplicate_positions = [i for i, col in enumerate(df.columns) if col == column_name]
            print(f"Duplicate positions: {duplicate_positions}")
            
            for i, pos in enumerate(duplicate_positions):
                try:
                    col_data = df.iloc[:, pos]
                    print(f"  Duplicate {i+1} (position {pos}): dtype={col_data.dtype}, non-null={col_data.notna().sum()}")
                    print(f"    Sample values: {col_data.dropna().head(2).tolist()}")
                except Exception as e:
                    print(f"  Duplicate {i+1} (position {pos}): Error accessing - {e}")
        else:
            print("Single column found - checking data access...")
            try:
                result = df[column_name]
                print(f"df[column_name] returns type: {type(result)}")
                print(f"Result shape: {result.shape}")
                
                if isinstance(result, pd.DataFrame):
                    print("❌ Confirmed: Returns DataFrame when it should return Series")
                    print(f"DataFrame columns: {result.columns.tolist()}")
                else:
                    print("✅ Returns Series as expected")
                    print(f"Series dtype: {result.dtype}")
                    
            except Exception as e:
                print(f"❌ Error accessing column: {e}")
    else:
        print("❌ Column does not exist exactly - checking for similar names...")
        
        similar_columns = [col for col in df.columns if column_name.lower() in col.lower()]
        print(f"Similar columns found: {len(similar_columns)}")
        
        if similar_columns:
            print("Similar column names:")
            for col in similar_columns[:10]:
                print(f"  - '{col}'")
            
            if len(similar_columns) > 10:
                print(f"  ... and {len(similar_columns) - 10} more")
