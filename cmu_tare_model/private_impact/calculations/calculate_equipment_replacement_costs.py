import pandas as pd
import numpy as np
from scipy.stats import norm

from config import PROJECT_ROOT
print(f"Project root directory: {PROJECT_ROOT}")

"""
===========================================================================================================
OVERVIEW: CALCULATE REPLACEMENT COSTS FOR VARIOUS END USES
===========================================================================================================
This module calculates the replacement costs for various end uses such as space heating, water heating,
clothes drying, and cooking. It uses a probabilistic approach to sample costs from distributions 
defined by progressive (10th percentile), reference (50th percentile), and conservative (90th percentile)
cost estimates.

# UPDATED MARCH 24, 2025 @ 4:30 PM - REMOVED RSMEANS CCI ADJUSTMENTS
# UPDATED APRIL 9, 2025 @ 7:30 PM - IMPROVED DOCUMENTATION
"""

# ===========================================================================================================
# FUNCTIONS: CALCULATE COST OF REPLACING EXISTING EQUIPMENT
# ===========================================================================================================

# Replacement Cost Function and Helper Functions (Parametes, Formula)
def get_end_use_replacement_parameters(df: pd.DataFrame, end_use: str) -> dict:
    """
    Retrieve parameters for equipment replacement cost calculations based on end use type.
    
    Args:
        df (pd.DataFrame): DataFrame containing equipment data.
        end_use (str): Type of equipment ('heating', 'waterHeating', 'clothesDrying', 'cooking').
        
    Returns:
        dict: Dictionary containing conditions, technology-efficiency pairs, and cost components for the specified end use.
        
    Raises:
        ValueError: If an invalid end_use is specified.
    """
    parameters = {
        'heating': {
            'conditions': [
                (df['base_heating_fuel'] == 'Propane'),
                (df['base_heating_fuel'] == 'Fuel Oil'),
                (df['base_heating_fuel'] == 'Natural Gas'),
                (df['base_heating_fuel'] == 'Electricity') & (df['heating_type'] == 'Electricity ASHP'),
                (df['base_heating_fuel'] == 'Electricity')
            ],
            'tech_eff_pairs': [
                ('Propane Furnace', '94 AFUE'),
                ('Fuel Oil Furnace', '95 AFUE'),
                ('Natural Gas Furnace', '95 AFUE'),
                ('Electric ASHP', 'SEER 18, 9.3 HSPF'),
                ('Electric Furnace', '100 AFUE')
            ],
            'cost_components': ['unitCost', 'otherCost', 'cost_per_kBtuh']
        },
        'waterHeating': {
            'conditions': [
                (df['base_waterHeating_fuel'] == 'Fuel Oil'),
                (df['base_waterHeating_fuel'] == 'Natural Gas'),
                (df['base_waterHeating_fuel'] == 'Propane'),
                (df['water_heater_efficiency'].isin(['Electric Standard', 'Electric Premium'])),
                (df['water_heater_efficiency'] == 'Electric Heat Pump, 80 gal')
            ],
            'tech_eff_pairs': [
                ('Fuel Oil Water Heater', 0.68),
                ('Natural Gas Water Heater', 0.67),
                ('Propane Water Heater', 0.67),
                ('Electric Water Heater', 0.95),
                ('Electric Heat Pump Water Heater, 80 gal', 2.35)
            ],
            'cost_components': ['unitCost', 'cost_per_gallon']
        },
        'clothesDrying': {
            'conditions': [
                (df['base_clothesDrying_fuel'] == 'Electricity'),
                (df['base_clothesDrying_fuel'] == 'Natural Gas'),
                (df['base_clothesDrying_fuel'] == 'Propane')
            ],
            'tech_eff_pairs': [
                ('Electric Clothes Dryer', 3.1),
                ('Natural Gas Clothes Dryer', 2.75),
                ('Propane Clothes Dryer', 2.75)
            ],
            'cost_components': ['unitCost']
        },
        'cooking': {
            'conditions': [
                (df['base_cooking_fuel'] == 'Electricity'),
                (df['base_cooking_fuel'] == 'Natural Gas'),
                (df['base_cooking_fuel'] == 'Propane')
            ],
            'tech_eff_pairs': [
                ('Electric Range', 0.74),
                ('Natural Gas Range', 0.4),
                ('Propane Range', 0.4)
            ],
            'cost_components': ['unitCost']
        }
    }
    if end_use not in parameters:
        raise ValueError(f"Invalid end_use specified: {end_use}")
    return parameters[end_use]


def calculate_replacement_cost_per_row(df_valid: pd.DataFrame, 
                                       sampled_costs_dict: dict, 
                                       menu_mp: int, 
                                       end_use: str) -> tuple:
    """
    Calculate replacement cost for each row based on the end use and associated costs.
    
    Args:
        df_valid (pd.DataFrame): Filtered DataFrame containing valid rows.
        sampled_costs_dict (dict): Dictionary with sampled costs for each component.
        menu_mp (int): Menu option identifier for column naming.
        end_use (str): Type of end-use ('heating', 'waterHeating', 'clothesDrying', 'cooking').
    
    Returns:
        tuple: (replacement_cost, cost_column_name) where:
            - replacement_cost: Array of calculated costs for each row
            - cost_column_name: String name for the cost column in the output DataFrame
    """
    if end_use == 'heating':
        # For heating, cost includes base unit cost, other costs, and capacity-based costs
        replacement_cost = (
            sampled_costs_dict['unitCost'] +
            sampled_costs_dict['otherCost'] +
            (df_valid['total_heating_load_kBtuh'] * sampled_costs_dict['cost_per_kBtuh']))
        
        cost_column_name = f'mp{menu_mp}_heating_replacementCost'

    elif end_use == 'waterHeating':
        # For water heating, cost includes base unit cost and gallon-based costs
        replacement_cost = (
            sampled_costs_dict['unitCost'] +
            (sampled_costs_dict['cost_per_gallon'] * df_valid['size_water_heater_gal']))
        
        cost_column_name = f'mp{menu_mp}_waterHeating_replacementCost'

    else:
        # For other end uses (cooking, clothes drying), only unit cost applies
        replacement_cost = sampled_costs_dict['unitCost'] 
        cost_column_name = f'mp{menu_mp}_{end_use}_replacementCost'
    
    return replacement_cost, cost_column_name


def calculate_replacement_cost(df: pd.DataFrame, 
                              cost_dict: dict, 
                              menu_mp: int, 
                              end_use: str) -> pd.DataFrame:
    """
    Calculate replacement costs for various end-uses based on fuel types, costs, and efficiency.
    
    This function applies probabilistic cost calculation using normal distribution sampling
    from progressive, reference, and conservative cost estimates.
    
    Args:
        df (pd.DataFrame): DataFrame containing data for different scenarios.
        cost_dict (dict): Dictionary with cost information for different technology and efficiency combinations.
            Expected format: {(tech, efficiency): {'unitCost_progressive': float, 'unitCost_reference': float, ...}}
        menu_mp (int): Menu option identifier (valid values: 7, 8, 9, 10).
        end_use (str): Type of end-use ('heating', 'waterHeating', 'clothesDrying', 'cooking').
    
    Returns:
        pd.DataFrame: Updated DataFrame with calculated replacement costs added as new columns.
        
    Raises:
        ValueError: If menu_mp is invalid or if cost data is missing for technology/efficiency combinations.
    """
    
    # Validate menu_mp
    valid_menu_mps = [7, 8, 9, 10]
    if menu_mp not in valid_menu_mps:
        raise ValueError("Please enter a valid measure package number for menu_mp. Should be 7, 8, 9, or 10.")
    
    # Get conditions, technology-efficiency pairs, and cost components for the specified end_use
    params = get_end_use_replacement_parameters(df, end_use)
    conditions = params['conditions']
    tech_eff_pairs = params['tech_eff_pairs']
    cost_components = params['cost_components']
   
    # Map each condition to its tech and efficiency using numpy.select
    # This creates arrays of technology and efficiency values based on matching conditions
    tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
    eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default=np.nan)

    # Convert efficiency values to appropriate types based on end use
    if end_use == 'heating':
        # For heating, efficiencies are strings (e.g., "94 AFUE")
        eff = np.array([str(e) if e != 'unknown' else np.nan for e in eff])
    else:
        # For other end uses, efficiencies are numeric values
        eff = np.array([float(e) if e != 'unknown' else np.nan for e in eff])

    # Filter out rows with unknown technology and NaN efficiency
    valid_indices = tech != 'unknown'
    tech = tech[valid_indices]
    eff = eff[valid_indices]
    df_valid = df.loc[valid_indices].copy()

    # Initialize dictionary to store sampled costs for each cost component
    sampled_costs_dict = {}

    # Calculate costs for each component (unitCost, otherCost, cost_per_kBtuh, etc.)
    for cost_component in cost_components:
        # Extract the three cost scenarios (progressive, reference, conservative) for each tech-eff pair
        # These represent low (10th percentile), median (50th), and high (90th) cost estimates
        progressive_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_progressive', np.nan) for t, e in zip(tech, eff)])
        reference_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_reference', np.nan) for t, e in zip(tech, eff)])
        conservative_costs = np.array([cost_dict.get((t, e), {}).get(f'{cost_component}_conservative', np.nan) for t, e in zip(tech, eff)])

        # Handle missing cost data with detailed error reporting
        if np.isnan(progressive_costs).any() or np.isnan(reference_costs).any() or np.isnan(conservative_costs).any():
            missing_indices = np.where(np.isnan(progressive_costs) | np.isnan(reference_costs) | np.isnan(conservative_costs))
            print(f"Missing data at indices: {missing_indices}")
            print(f"Tech with missing data: {tech[missing_indices]}")
            print(f"Efficiencies with missing data: {eff[missing_indices]}")
            
            raise ValueError(f"Missing cost data for some technology and efficiency combinations in cost_component {cost_component}")

        # Calculate mean and standard deviation for normal distribution
        # Assumes costs represent 10th, 50th, and 90th percentiles of a normal distribution
        mean_costs = reference_costs  # 50th percentile = mean for normal distribution
        # Calculate standard deviation using the difference between 90th and 10th percentiles
        std_costs = (conservative_costs - progressive_costs) / (norm.ppf(0.90) - norm.ppf(0.10))

        # Sample from the normal distribution for each row to create cost variability
        sampled_costs = np.random.normal(loc=mean_costs, scale=std_costs)
        sampled_costs_dict[cost_component] = sampled_costs

    # Calculate the replacement cost for each row based on the end use
    replacement_cost, cost_column_name = calculate_replacement_cost_per_row(df_valid, sampled_costs_dict, menu_mp, end_use)

    # Add the calculated costs to a new DataFrame, rounded to 2 decimal places
    df_new_columns = pd.DataFrame({cost_column_name: np.round(replacement_cost, 2)}, index=df_valid.index)

    # Identify overlapping columns between the new and existing DataFrame to avoid duplicates
    overlapping_columns = df_new_columns.columns.intersection(df.columns)

    # Drop overlapping columns from the original DataFrame if they exist
    if not overlapping_columns.empty:
        df.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into the original DataFrame using join
    # Left join ensures all original rows are preserved
    df = df.join(df_new_columns, how='left')

    return df
