import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import List, Dict, Tuple

from cmu_tare_model.utils.validation_framework import (
    create_retrofit_only_series,
    initialize_validation_tracking,
    apply_new_columns_to_dataframe,
    apply_final_masking,
)

from cmu_tare_model.utils.calculation_utils import (
    sample_costs_from_distributions,
    filter_valid_tech_homes,
)

 
"""
========================================================================================================================================================================
OVERVIEW: CALCULATE INSTALLED COSTS FOR ENCLOSURE UPGRADE RETROFIT MEASURES
========================================================================================================================================================================
This module calculates the installed costs for enclosure upgrade retrofit measurers.It uses a 
probabilistic approach to sample costs from distributions defined by progressive (10th percentile), 
reference (50th percentile), and conservative (90th percentile) cost estimates.

# UPDATED MARCH 24, 2025 @ 4:30 PM - REMOVED RSMEANS CCI ADJUSTMENTS
# UPDATED APRIL 9, 2025 @ 7:30 PM - IMPROVED DOCUMENTATION
"""

# ========================================================================================================================================================================
# FUNCTIONS: CALCULATE COST OF ENCLOSURE UPGRADES
# ========================================================================================================================================================================

def get_enclosure_parameters(
        df: pd.DataFrame,
        retrofit_col: str) -> dict:
    """
    Get conditions and technology-efficiency pairs for enclosure retrofit based on the retrofit column.
    
    Args:
        df: DataFrame containing enclosure data
        retrofit_col: Column name for the retrofit cost to be calculated
    
    Returns:
        A dictionary containing conditions and technology-efficiency pairs
        
    Raises:
        ValueError: If an invalid retrofit_col is specified
    """
    if retrofit_col == 'insulation_atticFloor_upgradeCost':
        conditions = [
            (df['upgrade_insulation_atticFloor'] == 'R-30') & (df['base_insulation_atticFloor'] == 'R-13'),
            (df['upgrade_insulation_atticFloor'] == 'R-30') & (df['base_insulation_atticFloor'] == 'R-7'),
            (df['upgrade_insulation_atticFloor'] == 'R-30') & (df['base_insulation_atticFloor'] == 'Uninsulated'),
            (df['upgrade_insulation_atticFloor'] == 'R-49') & (df['base_insulation_atticFloor'] == 'R-30'),
            (df['upgrade_insulation_atticFloor'] == 'R-49') & (df['base_insulation_atticFloor'] == 'R-19'),
            (df['upgrade_insulation_atticFloor'] == 'R-49') & (df['base_insulation_atticFloor'] == 'R-13'),
            (df['upgrade_insulation_atticFloor'] == 'R-49') & (df['base_insulation_atticFloor'] == 'R-7'),
            (df['upgrade_insulation_atticFloor'] == 'R-49') & (df['base_insulation_atticFloor'] == 'Uninsulated'),
            (df['upgrade_insulation_atticFloor'] == 'R-60') & (df['base_insulation_atticFloor'] == 'R-38'),
            (df['upgrade_insulation_atticFloor'] == 'R-60') & (df['base_insulation_atticFloor'] == 'R-30'),
            (df['upgrade_insulation_atticFloor'] == 'R-60') & (df['base_insulation_atticFloor'] == 'R-19'),
            (df['upgrade_insulation_atticFloor'] == 'R-60') & (df['base_insulation_atticFloor'] == 'R-13'),
            (df['upgrade_insulation_atticFloor'] == 'R-60') & (df['base_insulation_atticFloor'] == 'R-7'),
            (df['upgrade_insulation_atticFloor'] == 'R-60') & (df['base_insulation_atticFloor'] == 'Uninsulated')
        ]
        tech_eff_pairs = [
            ('Attic Floor Insulation: R-30', 'R-13'),
            ('Attic Floor Insulation: R-30', 'R-7'),
            ('Attic Floor Insulation: R-30', 'Uninsulated'),
            ('Attic Floor Insulation: R-49', 'R-30'),
            ('Attic Floor Insulation: R-49', 'R-19'),
            ('Attic Floor Insulation: R-49', 'R-13'),
            ('Attic Floor Insulation: R-49', 'R-7'),
            ('Attic Floor Insulation: R-49', 'Uninsulated'),
            ('Attic Floor Insulation: R-60', 'R-38'),
            ('Attic Floor Insulation: R-60', 'R-30'),
            ('Attic Floor Insulation: R-60', 'R-19'),
            ('Attic Floor Insulation: R-60', 'R-13'),
            ('Attic Floor Insulation: R-60', 'R-7'),
            ('Attic Floor Insulation: R-60', 'Uninsulated')
        ]
    elif retrofit_col == 'infiltration_reduction_upgradeCost':
        conditions = [
            (df['upgrade_infiltration_reduction'] == '30%')
        ]
        tech_eff_pairs = [
            ('Air Leakage Reduction: 30% Reduction', 'Varies')
        ]
    elif retrofit_col == 'duct_sealing_upgradeCost':
        conditions = [
            (df['upgrade_duct_sealing'] == '10% Leakage, R-8') & (df['base_ducts'].str.contains('10% Leakage')),
            (df['upgrade_duct_sealing'] == '10% Leakage, R-8') & (df['base_ducts'].str.contains('20% Leakage')),
            (df['upgrade_duct_sealing'] == '10% Leakage, R-8') & (df['base_ducts'].str.contains('30% Leakage')),
        ]
        tech_eff_pairs = [
            ('Duct Sealing: 10% Leakage, R-8', '10% Leakage'),
            ('Duct Sealing: 10% Leakage, R-8', '20% Leakage'),
            ('Duct Sealing: 10% Leakage, R-8', '30% Leakage'),
        ]
    elif retrofit_col == 'insulation_wall_upgradeCost':
        conditions = [
            (df['upgrade_insulation_wall'] == 'Wood Stud, R-13')
        ]
        tech_eff_pairs = [
            ('Drill-and-fill Wall Insulation: Wood Stud, R-13', 'Wood Stud, Uninsulated')
        ]
    elif retrofit_col == 'insulation_foundation_wall_upgradeCost':
        conditions = [
            (df['upgrade_insulation_foundation_wall'] == 'Wall R-10, Interior')
        ]
        tech_eff_pairs = [
            ('Foundation Wall Insulation: Wall R-10, Interior', 'Uninsulated')
        ]
    elif retrofit_col == 'insulation_rim_joist_upgradeCost':
        conditions = [
            (df['base_insulation_foundation_wall'] == 'Uninsulated') & (df['base_foundation_type'].isin(['Unvented Crawlspace', 'Vented Crawlspace', 'Heated Basement']))
        ]
        tech_eff_pairs = [
            ('Rim Joist Insulation: Wall R-10, Exterior', 'Uninsulated')
        ]
    elif retrofit_col == 'seal_crawlspace_upgradeCost':
        conditions = [
            (df['upgrade_seal_crawlspace'] == 'Unvented Crawlspace')
        ]
        tech_eff_pairs = [
            ('Seal Vented Crawlspace: Unvented Crawlspace', 'Vented Crawlspace')
        ]
    elif retrofit_col == 'insulation_roof_upgradeCost':
        conditions = [
            (df['upgrade_insulation_roof'] == 'Finished, R-30')
        ]
        tech_eff_pairs = [
            ('Insulate Finished Attics and Cathedral Ceilings: Finished, R-30', 'R-30')
        ]
    else:
        raise ValueError(f"Invalid retrofit_col specified: {retrofit_col}")
    
    return {'conditions': conditions, 'tech_eff_pairs': tech_eff_pairs}


def calculate_enclosure_retrofit_upgradeCosts(
        df: pd.DataFrame,
        menu_mp: int,
        cost_dict: dict,
        retrofit_col: str,
        params_col: str
) -> pd.DataFrame:
    """
    Calculate the enclosure retrofit upgrade costs based on given parameters and conditions.

    This function uses a probabilistic approach to sample costs from distributions defined
    by progressive, reference, and conservative estimates. It applies data validation to
    ensure only valid homes are included in calculations.

    Args:
        df: DataFrame containing data for different scenarios
        menu_mp: Measure package identifier (0 for baseline, 8/9/10 for retrofits)
        cost_dict: Dictionary with cost information for different technology and efficiency combinations
        retrofit_col: Column name for the retrofit cost to be calculated
        params_col: Column name for the parameter to use in the cost calculation

    Returns:
        Updated DataFrame with calculated retrofit costs
        
    Raises:
        ValueError: If missing cost data for certain technology and efficiency combinations
        RuntimeError: If an unexpected error occurs during calculation
        
    Notes:
        This function implements the validation framework:
        1. Uses initialize_validation_tracking() to determine valid homes
        2. Creates retrofit-only series with NaN for invalid homes
        3. Calculates values only for valid homes with identifiable technology
        4. Applies final verification masking
    """
    # Category for enclosure upgrades is 'heating' because they are related to heating system performance and total upgrade costs
    category = 'heating'
    
    # Add logging for calculation start
    print(f"Calculating enclosure retrofit upgrade costs with validation framework ('include_{category}' flags): {retrofit_col}")

    # Initialize validation tracking
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        df, category, menu_mp, verbose=True)

    print(f"Found {valid_mask.sum()} valid homes out of {len(df_copy)} for {retrofit_col}")

    # Get conditions and tech-efficiency pairs for the specified retrofit
    params = get_enclosure_parameters(df_copy, retrofit_col)
    conditions = params['conditions']
    tech_eff_pairs = params['tech_eff_pairs']

    # Map each condition to its tech and efficiency
    tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
    eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default='unknown')

    # Use the standard filtering function to get only homes with both valid data and identifiable tech
    try:
        df_valid, valid_calculation_indices, tech_filtered, eff_filtered = filter_valid_tech_homes(
            df_copy, valid_mask, tech, eff)
        
        print(f"After tech filtering: {len(valid_calculation_indices)} homes remain valid for {retrofit_col}")

        if df_valid.empty:
            print(f"Warning: No valid homes found for {category} retrofit calculation.")
            
        # Define cost components for enclosure upgrades
        cost_components = ['normalized_cost']
        
        # Sample costs from distributions only if we have valid homes
        if not df_valid.empty:
            sampled_costs_dict = sample_costs_from_distributions(tech_filtered, eff_filtered, cost_dict, cost_components)
            
            # Calculate the retrofit cost for each row
            retrofit_cost = sampled_costs_dict['normalized_cost'] * df_valid[params_col]
        
        # Initialize the result series properly
        result_series = create_retrofit_only_series(df_copy, valid_mask)
        
        # Update only for homes that have valid data AND match our tech criteria
        if not df_valid.empty:
            result_series.loc[valid_calculation_indices] = np.round(retrofit_cost, 2)
    except Exception as e:
        raise RuntimeError(f"Error in enclosure retrofit calculation for {retrofit_col}: {str(e)}")

    # Then create the DataFrame column
    df_new_columns = pd.DataFrame({retrofit_col: result_series})    
    
    # Track the column for masking
    category_columns_to_mask.append(retrofit_col)
    
    # Apply new columns to DataFrame with proper tracking
    df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
        df_copy, df_new_columns, category, category_columns_to_mask, all_columns_to_mask)
    
    # Apply final verification masking for consistency
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=True)

    return df_copy
