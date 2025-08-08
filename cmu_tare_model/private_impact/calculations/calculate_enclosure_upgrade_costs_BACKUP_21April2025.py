import pandas as pd
import numpy as np
from scipy.stats import norm

from config import PROJECT_ROOT
print(f"Project root directory: {PROJECT_ROOT}")
 
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

def get_enclosure_parameters(df: pd.DataFrame,
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


def calculate_enclosure_retrofit_upgradeCosts(df: pd.DataFrame,
                                              cost_dict: dict,
                                              retrofit_col: str,
                                              params_col: str) -> pd.DataFrame:
    """
    Calculate the enclosure retrofit upgrade costs based on given parameters and conditions.

    Args:
        df: DataFrame containing data for different scenarios
        cost_dict: Dictionary with cost information for different technology and efficiency combinations
        retrofit_col: Column name for the retrofit cost to be calculated
        params_col: Column name for the parameter to use in the cost calculation

    Returns:
        Updated DataFrame with calculated retrofit costs
        
    Raises:
        ValueError: If missing cost data for certain technology and efficiency combinations
    """
    
    # Create a copy of the original DataFrame to avoid modifying it directly
    df_copy = df.copy()

    # Get conditions and tech-efficiency pairs for the specified retrofit
    params = get_enclosure_parameters(df_copy, retrofit_col)
    conditions = params['conditions']
    tech_eff_pairs = params['tech_eff_pairs']

    # Map each condition to its tech and efficiency
    tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
    eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default='unknown')

    # Filter out rows with unknown technology and efficiency
    valid_indices = tech != 'unknown'
    tech = tech[valid_indices]
    eff = eff[valid_indices]
    df_valid = df_copy.loc[valid_indices].copy()

    # Initialize dictionary to store sampled costs
    sampled_costs_dict = {}

    # Calculate costs for each component (normalized_cost)
    for cost_component in ['normalized_cost']:
        progressive_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_progressive', np.nan) for t, e in zip(tech, eff)])
        reference_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_reference', np.nan) for t, e in zip(tech, eff)])
        conservative_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_conservative', np.nan) for t, e in zip(tech, eff)])

        # Handle missing cost data
        if np.isnan(progressive_costs).any() or np.isnan(reference_costs).any() or np.isnan(conservative_costs).any():
            missing_indices = np.where(np.isnan(progressive_costs) | np.isnan(reference_costs) | np.isnan(conservative_costs))
            print(f"Missing data at indices: {missing_indices}")
            print(f"Tech with missing data: {tech[missing_indices]}")
            print(f"Efficiencies with missing data: {eff[missing_indices]}")
            
            raise ValueError(f"Missing cost data for some technology and efficiency combinations in cost_component {cost_component}")

        # Calculate mean and standard deviation assuming the costs represent the 10th, 50th, and 90th percentiles of a normal distribution
        mean_costs = reference_costs  # 50th percentile is the mean of the normal distribution
        # Calculate standard deviation based on the difference between 90th and 10th percentiles
        std_costs = (conservative_costs - progressive_costs) / (norm.ppf(0.90) - norm.ppf(0.10))

        # Sample from the normal distribution for each row
        sampled_costs = np.random.normal(loc=mean_costs, scale=std_costs)
        sampled_costs_dict[cost_component] = sampled_costs

    # Calculate the retrofit cost for each row
    retrofit_cost = (
        sampled_costs_dict['normalized_cost'] * df_valid[params_col])

    # Add the calculated costs to a new DataFrame, rounded to 2 decimal places
    df_new_columns = pd.DataFrame({retrofit_col: np.round(retrofit_cost, 2)}, index=df_valid.index)

    # Identify overlapping columns between the new and existing DataFrame
    overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)

    # Drop overlapping columns from the original DataFrame
    if not overlapping_columns.empty:
        df_copy.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into the original DataFrame, ensuring no duplicates or overwrites occur
    df_copy = df_copy.join(df_new_columns, how='left')

    return df_copy
