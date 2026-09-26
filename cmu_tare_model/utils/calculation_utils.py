"""
calculation_utils.py

Specialized utilities for calculations related to equipment costs, 
consumption, and operational savings.

This module contains utilities that support specific calculation operations
but aren't part of the core validation framework.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.stats import norm

from cmu_tare_model.constants import EQUIPMENT_SPECS, FUEL_MAPPING, ALLOWED_TECHNOLOGIES, VERBOSE
from cmu_tare_model.utils.validation_framework import (
    apply_final_masking,
    get_valid_fuel_types,
    mask_category_specific_data
    )

def get_all_possible_fuel_columns(category: str) -> List[str]:
    """
    Returns all possible fuel consumption columns for a category.

    This function identifies which consumption columns exist in the dataset for a given
    equipment category. The logic mirrors get_valid_fuel_types() to ensure consistency
    between validation rules and data retrieval operations.

    Note:
        This function determines which columns to RETRIEVE from the dataframe.
        See get_valid_fuel_types() for validation rules on which fuel types are ACCEPTABLE.
    
    Args:
        category: Equipment category name.
        
    Returns:
        List of column names for all possible fuel consumption measurements.
        
    Raises:
        ValueError: If an invalid category is provided.
    """  
    if category not in EQUIPMENT_SPECS:
        raise ValueError(f"Invalid category. Must be one of the following: {EQUIPMENT_SPECS.keys()}")
    
    # Heating and water heating have all four fuel types available in the dataset.
    # Tech filters handle excluding heat pump technologies, so electricity remains valid.
    if category in ['heating', 'waterHeating']:
        return [f'base_{fuel}_{category}_consumption' for fuel in FUEL_MAPPING.values()]
    
    # Heat pump clothes dryers are different from existing electric resistance dryers in EUSS.
    # Dataset contains: electricity, natural gas, and propane (no fuel oil).
    elif category == 'clothesDrying':
        return [f'base_{fuel}_{category}_consumption' for fuel in FUEL_MAPPING.values() 
                if fuel != 'fuelOil']
    
    # Cooking data excludes electricity because the electric upgrade in MP7 is the same technology.
    # Dataset contains: natural gas and propane only (no electricity, no fuel oil).
    elif category == 'cooking':
        return [f'base_{fuel}_{category}_consumption' for fuel in FUEL_MAPPING.values() 
                if fuel not in ['electricity', 'fuelOil']]
    
    # Cooling equipment is exclusively electric (air conditioners, heat pumps in cooling mode).
    # Dataset contains: electricity only.
    elif category == 'cooling':
        return [f'base_electricity_{category}_consumption']
    
    else:
        raise ValueError(f"Invalid category: {category}. Must be one of {list(EQUIPMENT_SPECS.keys())}")


def get_post_retrofit_columns(
    category: str,
    menu_mp: int
) -> List[str]:
    """
    Returns the post-retrofit consumption column name for a category and measure package.
    
    Args:
        category: Equipment category name.
        menu_mp: The measure package number.
        
    Returns:
        List containing the post-retrofit consumption column name.
        
    Raises:
        ValueError: If an invalid category is provided.
    """    
    if category not in EQUIPMENT_SPECS:
        raise ValueError(f"Invalid category. Must be one of the following: {EQUIPMENT_SPECS.keys()}")
    
    # Just return the basic consumption column for this measure package and category
    return [f'mp{menu_mp}_{category}_consumption']


# Energy ResStock reports separately from a system's primary heating or cooling
# energy, as (fuel, column token, ResStock logical name). Counted on BOTH the
# baseline and retrofit side. Earlier versions of this model compared primary
# energy only, assuming these components were the same before and after the
# retrofit. They are not: a furnace blower is not a heat pump's fan, and a
# dual-fuel heat pump's backup furnace burns gas. So every component is counted.
CONSUMPTION_COMPONENTS: Dict[str, List[Tuple[str, str, str]]] = {
    'heating': [
        ('electricity', 'fansPumps', 'heating_fans_pumps'),
        ('electricity', 'hpBackup', 'heating_hp_backup_electricity'),
        ('electricity', 'hpBackupFans', 'heating_hp_backup_fans'),
        ('naturalGas', 'hpBackup', 'heating_hp_backup_natural_gas'),
        ('propane', 'hpBackup', 'heating_hp_backup_propane'),
        ('fuelOil', 'hpBackup', 'heating_hp_backup_fuel_oil'),
    ],
    'cooling': [
        ('electricity', 'fansPumps', 'cooling_fans_pumps'),
    ],
}


def get_consumption_component_columns(
    category: str,
    menu_mp: int
) -> List[Tuple[str, str]]:
    """Returns every (fuel, column) pair that makes up one category's energy use.

    Covers the primary heating or cooling energy plus each separately reported
    component in CONSUMPTION_COMPONENTS, for the baseline (menu_mp=0) or one
    measure package's retrofit. Summing a fuel's columns gives that fuel's full
    energy use for the category.

    Args:
        category: Equipment category name.
        menu_mp: Measure package number; 0 for the baseline.

    Returns:
        List of (fuel, column name) pairs, with fuel spelled as in
        FUEL_MAPPING's values (e.g. 'naturalGas'). A fuel can appear more
        than once.

    Raises:
        TypeError: If menu_mp is not an integer.
        ValueError: If category is invalid or menu_mp is negative.
    """
    if not isinstance(menu_mp, (int, np.integer)):
        raise TypeError(
            f"menu_mp must be an integer, got {type(menu_mp).__name__}")
    if menu_mp < 0:
        raise ValueError(f"menu_mp must be 0 or greater, got {menu_mp}")

    # Step 1 -- primary energy. The baseline may use any fuel valid for the
    # category; every retrofit's primary heating or cooling energy is electric.
    if menu_mp == 0:
        baseline_cols = get_all_possible_fuel_columns(category)
        pairs = [
            (fuel, f'base_{fuel}_{category}_consumption')
            for fuel in FUEL_MAPPING.values()
            if f'base_{fuel}_{category}_consumption' in baseline_cols
        ]
        prefix = 'base'
    else:
        pairs = [('electricity', col)
                 for col in get_post_retrofit_columns(category, menu_mp)]
        prefix = f'mp{menu_mp}'

    # Step 2 -- separately reported components (fans, heat-pump backup)
    for fuel, component, _ in CONSUMPTION_COMPONENTS.get(category, []):
        pairs.append(
            (fuel, f'{prefix}_{fuel}_{category}_{component}_consumption'))
    return pairs


def identify_valid_homes(
    df: pd.DataFrame,
    verbose: bool = VERBOSE
) -> pd.DataFrame:
    """Creates comprehensive data quality flags for all categories.
    
    This function adds columns to track the quality and validity of data
    across all equipment categories. Technology validation is only applied
    to heating and water heating categories.
    
    Args:
        df: DataFrame containing energy consumption data.
        
    Returns:
        DataFrame with added data quality flags.
    """    
    # Initialize the overall inclusion flag
    df['include_all'] = True

    if verbose:
        print("\nCreating data quality flags for all categories")
    
    for category in EQUIPMENT_SPECS.keys():
        if verbose:
            print(f"\n--- Processing {category} ---")
        
        # Create fuel validity flag
        fuel_flag = f'valid_fuel_{category}'
        fuel_col = f'base_{category}_fuel'
        
        # UPDATED: Uses get_valid_fuel_types() instead of previous validation approach
        if fuel_col in df.columns:
            if verbose:
                # Print some diagnostic info about the values
                print(f"Values in {fuel_col} (top 5):")
                print(df[fuel_col].value_counts().head(5))
            
            # Get valid fuel types for this category
            valid_fuel_types = get_valid_fuel_types(category)
            df[fuel_flag] = df[fuel_col].isin(valid_fuel_types)

            # Invalid fuel count and percentage
            invalid_fuel_count = (~df[fuel_flag]).sum()
            invalid_fuel_pct = (invalid_fuel_count / len(df)) * 100 if len(df) > 0 else 0
            if verbose:
                print(f"  {category}: Found {invalid_fuel_count} homes ({invalid_fuel_pct:.1f}%) with invalid fuel types")  
            
            # Show what's being filtered
            if invalid_fuel_count > 0:
                invalid_fuels = df.loc[~df[fuel_flag], fuel_col].value_counts()
                if verbose:
                    print("  Invalid fuel types (top 5):")
                    (invalid_fuels.head(5))
        else:
            if verbose:
                print(f"  {category}: Warning - Column {fuel_col} not found")
            df[fuel_flag] = True
        
        # Handle technology validation only for heating, cooling, and water heating
        # Clothes drying and cooking:
        # - Do not have technology type columns, only fuel type specified
        # - All valid fuels are already filtered above
        if category in ['heating', 'cooling', 'waterHeating']:
            # Create technology validity flag
            tech_flag = f'valid_tech_{category}'
            tech_col = f'{category}_type'
            
            if tech_col in df.columns and category in ALLOWED_TECHNOLOGIES:
                # Print some diagnostic info
                if verbose:
                    print(f"Values in {tech_col} (top 5):")
                    print(df[tech_col].value_counts().head(5))
                    print(f"Allowed values for {category}:")
                    print(ALLOWED_TECHNOLOGIES[category])
                
                # Check if the technology type is in the allowed list
                df[tech_flag] = df[tech_col].isin(ALLOWED_TECHNOLOGIES[category])

                # Invalid technology count and percentage
                invalid_tech_count = (~df[tech_flag]).sum()
                invalid_tech_pct = (invalid_tech_count / len(df)) * 100 if len(df) > 0 else 0
                if verbose:
                    print(f"  {category}: Found {invalid_tech_count} homes ({invalid_tech_pct:.1f}%) with invalid technology types")
                
                # Show what's being filtered
                if invalid_tech_count > 0:
                    invalid_techs = df.loc[~df[tech_flag], tech_col].value_counts()
                    if verbose:
                        print("  Invalid technology types (top 5):")
                        print(invalid_techs.head(5))
                
                # Create category inclusion flag based on both fuel and tech validity
                include_col = f'include_{category}'
                df[include_col] = df[fuel_flag] & df[tech_flag]
            else:
                if category not in ALLOWED_TECHNOLOGIES:
                    if verbose:
                        print(f"  {category}: No allowed technologies defined")
                elif tech_col not in df.columns:
                    if verbose:
                        print(f"  {category}: Warning - Column {tech_col} not found")
                
                # Set inclusion flag based only on fuel validity
                include_col = f'include_{category}'
                df[include_col] = df[fuel_flag]
        else:
            # For clothes drying and cooking, only use fuel validation
            if verbose:
                print(f"  {category}: Technology validation not applicable (no technology type column)")
            include_col = f'include_{category}'
            df[include_col] = df[fuel_flag]
        
        # Print exclusion summary
        excluded_count = (~df[include_col]).sum()
        excluded_pct = (excluded_count / len(df)) * 100 if len(df) > 0 else 0
        if verbose:
            print(f"  {category}: Total {excluded_count} homes ({excluded_pct:.1f}%) excluded from analysis")
        
        # Update the overall inclusion flag
        df['include_all'] &= df[include_col]
    
    overall_excluded = (~df['include_all']).sum()
    overall_pct = (overall_excluded / len(df)) * 100 if len(df) > 0 else 0
    if verbose:
        print(f"\nOverall: Total {overall_excluded} homes ({overall_pct:.1f}%) excluded from ALL categories")
    return df


# Fuel bucket order shared by compute_funnel_stage_row and
# print_masking_funnel_stage, so the two funnel-reporting entry points can
# never drift out of sync on which buckets exist or what order they print in.
_FUNNEL_FUEL_ORDER = [
    'Electricity', 'Electricity ASHP', 'Fuel Oil', 'Natural Gas', 'Propane']

# Existing heat pumps are excluded from the study (nothing to replace).
EXISTING_HEAT_PUMP_TYPES = ['Electricity ASHP', 'Electricity MSHP']


def compute_funnel_stage_row(
    df: pd.DataFrame,
    stage_label: str,
    stage_mask: Optional[pd.Series] = None,
    weight_col: str = 'weight',
    heating_fuel_col: str = 'base_heating_fuel',
    heating_type_col: str = 'heating_type',
) -> Dict[str, Union[str, int, float]]:
    """Computes one row of the masking filter funnel, without printing it.

    Holds the rdu-count/weighted-count/fuel-share math in exactly one place,
    shared by print_masking_funnel_stage (console reporting) and any caller
    that accumulates funnel stages into a DataFrame instead (for example
    load_and_filter_2025_1_upgrade's df_funnel).

    Existing electric heat pumps ('Electricity ASHP' and its variants, such
    as MSHP) are broken out of the general 'Electricity' fuel bucket, since
    CLAUDE.md treats "any variant" of an existing heat pump as excluded for
    a different reason than a home's baseline fuel (Documented Limitation
    8).

    Args:
        df: The DataFrame at this filter stage.
        stage_label: A short name for this stage, e.g. 'applicability'.
        stage_mask: Optional boolean mask aligned to df's index; if given,
            only rows where True are counted, so a stage can be reported
            without pre-filtering df itself. If None, every row in df is
            counted.
        weight_col: Name of the dwelling-unit weight column.
        heating_fuel_col: Name of the baseline heating fuel column.
        heating_type_col: Name of the baseline heating type-and-fuel column
            (used only to identify existing heat pumps).

    Returns:
        A dict with keys 'stage', 'rdu_count', 'weighted_count', and one
        '{fuel}_pct' key per bucket in _FUNNEL_FUEL_ORDER.

    Raises:
        KeyError: If a required column is missing from df.
    """
    required_cols = [weight_col, heating_fuel_col, heating_type_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"compute_funnel_stage_row requires columns {required_cols}, "
            f"missing: {missing_cols}")

    df_stage = df if stage_mask is None else df.loc[stage_mask]

    rdu_count = len(df_stage)
    weighted_count = df_stage[weight_col].sum()

    is_existing_heat_pump = df_stage[heating_type_col].isin(
        EXISTING_HEAT_PUMP_TYPES)
    fuel_bucket = df_stage[heating_fuel_col].where(
        ~is_existing_heat_pump, 'Electricity ASHP')

    weight_by_fuel = df_stage.groupby(fuel_bucket)[weight_col].sum()
    share_by_fuel = {
        fuel: (weight_by_fuel.get(fuel, 0.0) / weighted_count * 100
               if weighted_count > 0 else 0.0)
        for fuel in _FUNNEL_FUEL_ORDER
    }

    row: Dict[str, Union[str, int, float]] = {
        'stage': stage_label,
        'rdu_count': rdu_count,
        'weighted_count': weighted_count,
    }
    row.update({f'{fuel}_pct': share_by_fuel[fuel] for fuel in _FUNNEL_FUEL_ORDER})
    return row


def print_masking_funnel_stage(
    df: pd.DataFrame,
    stage_label: str,
    stage_mask: Optional[pd.Series] = None,
    weight_col: str = 'weight',
    heating_fuel_col: str = 'base_heating_fuel',
    heating_type_col: str = 'heating_type',
) -> None:
    """Prints one row of the 2025.1 masking filter funnel.

    A thin console-reporting wrapper around compute_funnel_stage_row -- see
    that function's docstring for the rdu/weighted-count/fuel-share math and
    the existing-heat-pump bucketing rule.

    The applicability, occupancy, and housing-type filter stages run on the
    raw ResStock parquet, before df_enduse_refactored/df_enduse_compare have
    renamed anything -- so their fuel and technology columns are still the
    raw names ('in.heating_fuel', 'in.hvac_heating_type_and_fuel'). Later
    stages run on the TARE-side frame, where those columns are renamed to
    'base_heating_fuel'/'heating_type'. heating_fuel_col and heating_type_col
    let the same funnel call report either.

    Args:
        df: The DataFrame at this filter stage.
        stage_label: A short name for this stage, e.g. 'applicability'.
        stage_mask: Optional boolean mask aligned to df's index; if given,
            only rows where True are counted, so a stage can be reported
            without pre-filtering df itself. If None, every row in df is
            counted.
        weight_col: Name of the dwelling-unit weight column.
        heating_fuel_col: Name of the baseline heating fuel column.
        heating_type_col: Name of the baseline heating type-and-fuel column
            (used only to identify existing heat pumps).

    Returns:
        None. Prints the funnel row.

    Raises:
        KeyError: If a required column is missing from df.
    """
    row = compute_funnel_stage_row(
        df, stage_label, stage_mask,
        weight_col=weight_col, heating_fuel_col=heating_fuel_col,
        heating_type_col=heating_type_col)

    share_str = " | ".join(
        f"{fuel}={row[f'{fuel}_pct']:.2f}%" for fuel in _FUNNEL_FUEL_ORDER)

    print(
        f"[FUNNEL] {row['stage']}: {row['rdu_count']:,} rdu | "
        f"{row['weighted_count']:,.0f} weighted homes | {share_str}")


def mask_invalid_data(
    df: pd.DataFrame,
    menu_mp: Optional[int] = None,
    verbose: bool = VERBOSE
) -> pd.DataFrame:
    """
    Sets consumption values to NaN based on inclusion flags.
    
    Args:
        df: DataFrame with inclusion flags already created.
        menu_mp: Optional measure package number for post-retrofit masking.
        
    Returns:
        DataFrame with consumption values set to NaN for invalid records.
    """
    if verbose:    
        print("Applying NaN masking based on inclusion flags")
    
    for category in EQUIPMENT_SPECS.keys():
        include_col = f'include_{category}'
        
        if include_col not in df.columns:
            if verbose:
                print(f"  {category}: Warning - Inclusion flag '{include_col}' not found. Skipping masking.")
            continue
        
        # Get all baseline consumption columns for this category
        columns_to_mask = get_all_possible_fuel_columns(category)
        
        # Add the total baseline consumption column
        total_col = f'baseline_{category}_consumption'
        if total_col in df.columns:
            columns_to_mask.append(total_col)
            
        # Add post-retrofit column if menu_mp is provided
        if menu_mp != 0:
            post_retrofit_cols = get_post_retrofit_columns(category, menu_mp)
            columns_to_mask.extend(post_retrofit_cols)
        
        # Apply masking to all collected columns
        df = mask_category_specific_data(df, columns_to_mask, category, verbose=verbose)
    
    return df


def filter_valid_tech_homes(
    df: pd.DataFrame,
    valid_mask: pd.Series,
    tech: np.ndarray,
    eff: np.ndarray,
    default_value: str = 'unknown',
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    """
    Filter homes to those that have both valid data and identifiable technology.
    
    Args:
        df: DataFrame containing all homes
        valid_mask: Boolean Series indicating homes with valid data
        tech: Array of technology types for each home
        eff: Array of efficiency values for each home
        default_value: Value indicating an unknown technology
        
    Returns:
        Tuple containing:
        - Filtered DataFrame
        - Series of valid calculation indices
        - Filtered technology array
        - Filtered efficiency array
    """
    # Create tech validity mask and combine with data validation mask
    tech_valid_mask = tech != default_value
    tech_valid_series = pd.Series(tech_valid_mask, index=df.index)
    combined_valid_mask = valid_mask & tech_valid_series
    
    # Get indices of homes that meet both criteria
    valid_calculation_indices = combined_valid_mask[combined_valid_mask].index
    
    if len(valid_calculation_indices) > 0:
        # Filter df using combined_valid_mask
        df_valid = df.loc[valid_calculation_indices].copy()
        
        # FIXED: Filter tech and eff using combined_valid_mask instead of tech_valid_mask
        combined_valid_array = combined_valid_mask.values
        tech_filtered = tech[combined_valid_array]
        eff_filtered = eff[combined_valid_array]
    else:
        tech_filtered = np.array([])
        eff_filtered = np.array([])
        df_valid = pd.DataFrame()
    
    return df_valid, valid_calculation_indices, tech_filtered, eff_filtered

# ========================================================================
# FUNCTIONS FOR PRIVATE AND PUBLIC IMPACT CALCULATIONS
# ========================================================================
# Deleted sample_costs_from_distributions as it is no longer used in REMDB v4 cost calculations
def sample_costs_from_distributions(
    tech: np.ndarray,
    eff: np.ndarray,
    cost_dict: Dict,
    cost_components: List[str],
    verbose: bool = VERBOSE
) -> Dict[str, np.ndarray]:
    """
    Sample costs from distributions defined by progressive, reference, and conservative estimates.
    
    This utility function samples from normal distributions derived from 
    percentile-based cost estimates (10th, 50th, and 90th percentiles).
    
    Args:
        tech: Array of technology types
        eff: Array of efficiency values
        cost_dict: Dictionary mapping (tech, eff) pairs to cost components
        cost_components: List of cost component names to sample
        verbose: Whether to print detailed information about missing data

    Returns:
        Dictionary mapping cost component names to sampled cost arrays
        
    Raises:
        ValueError: If cost data is missing for any technology/efficiency combination
    """
    # Initialize dictionary to store sampled costs
    sampled_costs_dict = {}
    
    # Calculate costs for each component
    for cost_component in cost_components:
        # Extract the progressive (10th), reference (50th), and conservative (90th) costs
        progressive_costs = np.array([
            cost_dict.get((t, e), {}).get(f'{cost_component}_progressive', np.nan) 
            for t, e in zip(tech, eff)
        ])
        reference_costs = np.array([
            cost_dict.get((t, e), {}).get(f'{cost_component}_reference', np.nan) 
            for t, e in zip(tech, eff)
        ])
        conservative_costs = np.array([
            cost_dict.get((t, e), {}).get(f'{cost_component}_conservative', np.nan) 
            for t, e in zip(tech, eff)
        ])

        # Handle missing cost data
        if np.isnan(progressive_costs).any() or np.isnan(reference_costs).any() or np.isnan(conservative_costs).any():
            missing_indices = np.where(np.isnan(progressive_costs) | np.isnan(reference_costs) | np.isnan(conservative_costs))
            if verbose:
                print(f"Missing data at indices: {missing_indices}")
                print(f"Tech with missing data: {tech[missing_indices]}")
                print(f"Efficiencies with missing data: {eff[missing_indices]}")
            
            raise ValueError(f"Missing cost data for some technology and efficiency combinations in cost_component {cost_component}")

        # Calculate mean and standard deviation for normal distribution
        mean_costs = reference_costs  # 50th percentile becomes the mean
        
        # Calculate standard deviation using the difference between 90th and 10th percentiles
        std_costs = (conservative_costs - progressive_costs) / (norm.ppf(0.90) - norm.ppf(0.10))

        # Sample from the normal distribution for each row
        sampled_costs = np.random.normal(loc=mean_costs, scale=std_costs)
        sampled_costs_dict[cost_component] = sampled_costs
    
    return sampled_costs_dict


# ===== Input Parameter Validation =====
def validate_common_parameters(
    menu_mp: Union[int, str],
    policy_scenario: str
) -> Tuple[int, str]:
    """
    Validates common input parameters used across calculation functions.
    
    Args:
        menu_mp: Measure package identifier (int or str).
        policy_scenario: Policy scenario name.

    Returns:
        Tuple containing:
        - menu_mp_int: Validated menu_mp as integer
        - policy_scenario: Validated policy scenario string
        
    Raises:
        ValueError: If any parameter is invalid.
    """
    # Validate menu_mp
    try:
        menu_mp_int = int(menu_mp)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid menu_mp: {menu_mp}. Must be convertible to an integer.")
    
    # Validate policy_scenario
    valid_scenarios = ['No Inflation Reduction Act', 'AEO2023 Reference Case',
                       '2025 Reference Case']
    if policy_scenario not in valid_scenarios:
        raise ValueError(f"Invalid policy_scenario: {policy_scenario}. Must be one of {valid_scenarios}")
       
    # No longer validating discounting_method because using both methods for private NPV

    return menu_mp_int, policy_scenario


# ===== Shared DataFrame Helpers =====
# Relocated here so non-deprecated modules (e.g., the economic adoption module)
# do not have to import from determine_adoption_potential_sensitivity.py.
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
        context_str = "\n".join(
            f"  {key}: {value}" for key, value in context_params.items()
        )

        error_msg = (
            f"Required columns missing:\n"
            f"{context_str}\n"
            f"\nMissing columns ({len(unique_missing)}):\n"
        )
        error_msg += "\n".join(f"  - {col}" for col in unique_missing)
        raise KeyError(error_msg)


# ===== Apply Temporary Validation and Masking, Remove Duplicate Columns =====
def apply_temporary_validation_and_mask(
    df_copy: pd.DataFrame,
    df_new: pd.DataFrame,
    all_columns_to_mask: Dict[str, List[str]],
    verbose: bool = VERBOSE
) -> pd.DataFrame:
    """
    Applies temporary validation columns, performs masking, and joins the DataFrames.
    
    Args:
        df_copy: Main DataFrame.
        df_new: DataFrame with new columns.
        all_columns_to_mask: Dictionary mapping categories to columns to mask.
        verbose: Whether to print verbose output.
        
    Returns:
        Updated DataFrame with masked and joined columns.
    """
    # Add temporary validation columns for masking
    temp_columns = {}
    for prefix in ["include_", "valid_tech_", "valid_fuel_"]:
        cols = [col for col in df_copy.columns if col.startswith(prefix)]
        for col in cols:
            if col not in df_new.columns:
                temp_columns[col] = True
                df_new[col] = df_copy[col]
    
    # Apply final masking using the utility function
    if verbose:
        print("\nVerifying masking for all calculated columns:")

    df_new = apply_final_masking(df_new, all_columns_to_mask, verbose=verbose)
    
    # Remove temporary validation columns after masking is done
    if temp_columns:
        df_new = df_new.drop(columns=list(temp_columns.keys()))
    
    # FIXED: Before a bug led to the columns only being dropped if verbose was True
    # Remove any columns from df_new that already exist in df_copy to avoid duplication
    overlapping = df_new.columns.intersection(df_copy.columns)
    if not overlapping.empty:
        if verbose:
            print(f"WARNING: Replacing {len(overlapping)} existing columns. "
                f"Function was called on data that already contains results.")
            print(f"Columns being replaced: {overlapping.tolist()}")
        
        df_copy = df_copy.drop(columns=overlapping)

    # Join DataFrames
    df_main = df_copy.join(df_new, how='left')
        
    return df_main
