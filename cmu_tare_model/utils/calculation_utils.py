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
    valid_scenarios = ['No Inflation Reduction Act', 'AEO2023 Reference Case']
    if policy_scenario not in valid_scenarios:
        raise ValueError(f"Invalid policy_scenario: {policy_scenario}. Must be one of {valid_scenarios}")
       
    # No longer validating discounting_method because using both methods for private NPV

    return menu_mp_int, policy_scenario

    
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
