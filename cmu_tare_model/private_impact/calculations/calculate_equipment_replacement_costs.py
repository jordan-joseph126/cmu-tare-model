import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from cmu_tare_model.utils.validation_framework import (
    create_retrofit_only_series,
    initialize_validation_tracking,
    apply_new_columns_to_dataframe,
    apply_final_masking
    )

from cmu_tare_model.utils.calculation_utils import (
    sample_costs_from_distributions,
    filter_valid_tech_homes
    )


"""
========================================================================================================================================================================
OVERVIEW: CALCULATE REPLACEMENT COSTS FOR VARIOUS END USES
========================================================================================================================================================================
This module calculates the replacement costs for various end uses such as space heating, water heating,
clothes drying, and cooking. It uses a probabilistic approach to sample costs from distributions 
defined by progressive (10th percentile), reference (50th percentile), and conservative (90th percentile)
cost estimates.

# UPDATED MARCH 24, 2025 @ 4:30 PM - REMOVED RSMEANS CCI ADJUSTMENTS
# UPDATED APRIL 9, 2025 @ 7:30 PM - IMPROVED DOCUMENTATION
"""

# ========================================================================================================================================================================
# FUNCTIONS: CALCULATE COST OF REPLACING EXISTING EQUIPMENT
# ========================================================================================================================================================================

# Replacement Cost Function and Helper Functions (Parametes, Formula)
def get_end_use_replacement_parameters(
        df: pd.DataFrame,
        end_use: str) -> dict:
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


def calculate_replacement_cost_per_row(
        df_valid: pd.DataFrame, 
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
            
    Raises:
        ValueError: If required columns are missing from the DataFrame
        KeyError: If required cost components are missing from sampled_costs_dict
    """
    try:
        if end_use == 'heating':
            # Validate required columns and cost components
            if 'total_heating_load_kBtuh' not in df_valid.columns:
                raise ValueError("Required column 'total_heating_load_kBtuh' not found in DataFrame")
            
            required_components = ['unitCost', 'otherCost', 'cost_per_kBtuh']
            for comp in required_components:
                if comp not in sampled_costs_dict:
                    raise KeyError(f"Required cost component '{comp}' not found for heating calculation")
                
            # For heating, cost includes base unit cost, other costs, and capacity-based costs
            replacement_cost = (
                sampled_costs_dict['unitCost'] +
                sampled_costs_dict['otherCost'] +
                (df_valid['total_heating_load_kBtuh'] * sampled_costs_dict['cost_per_kBtuh']))
            
            cost_column_name = f'mp{menu_mp}_heating_replacementCost'

        elif end_use == 'waterHeating':
            # Validate required columns and cost components
            if 'size_water_heater_gal' not in df_valid.columns:
                raise ValueError("Required column 'size_water_heater_gal' not found in DataFrame")
                
            required_components = ['unitCost', 'cost_per_gallon']
            for comp in required_components:
                if comp not in sampled_costs_dict:
                    raise KeyError(f"Required cost component '{comp}' not found for water heating calculation")
                
            # For water heating, cost includes base unit cost and gallon-based costs
            replacement_cost = (
                sampled_costs_dict['unitCost'] +
                (sampled_costs_dict['cost_per_gallon'] * df_valid['size_water_heater_gal']))
            
            cost_column_name = f'mp{menu_mp}_waterHeating_replacementCost'

        else:
            # Validate cost components for clothes drying and cooking
            if 'unitCost' not in sampled_costs_dict:
                raise KeyError(f"Required cost component 'unitCost' not found for {end_use} calculation")
                
            # For other end uses (cooking, clothes drying), only unit cost applies
            replacement_cost = sampled_costs_dict['unitCost'] 
            cost_column_name = f'mp{menu_mp}_{end_use}_replacementCost'
        
        return replacement_cost, cost_column_name
        
    except Exception as e:
        raise RuntimeError(f"Error calculating {end_use} replacement cost: {str(e)}")
    

def calculate_replacement_cost(
        df: pd.DataFrame, 
        cost_dict: dict, 
        menu_mp: int, 
        end_use: str) -> pd.DataFrame:
    """
    Calculate replacement costs for various end-uses based on fuel types, costs, and efficiency.
    
    This function applies probabilistic cost calculation using normal distribution sampling
    from progressive, reference, and conservative cost estimates. It also applies data validation
    to ensure only valid homes are included in calculations.
    
    Args:
        df: DataFrame containing data for different scenarios.
        cost_dict: Dictionary with cost information for different technology and efficiency combinations.
            Expected format: {(tech, efficiency): {'unitCost_progressive': float, 'unitCost_reference': float, ...}}
        menu_mp: Menu option identifier (valid values: 7, 8, 9, 10).
        end_use: Type of end-use ('heating', 'waterHeating', 'clothesDrying', 'cooking').
    
    Returns:
        pd.DataFrame: Updated DataFrame with calculated replacement costs added as new columns.
        
    Raises:
        ValueError: If menu_mp is invalid or if cost data is missing for technology/efficiency combinations.
        RuntimeError: If an unexpected error occurs during calculation.
        
    Notes:
        This function implements the validation framework:
        1. Uses initialize_validation_tracking() to determine valid homes
        2. Creates retrofit-only series with NaN for invalid homes
        3. Calculates values only for valid homes with identifiable technology
        4. Applies final verification masking
    """
    # Add logging for calculation start
    print(f"Starting {end_use} replacement cost calculation with validation framework")

    # Initialize validation tracking
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        df, end_use, menu_mp, verbose=True)
    
    print(f"Found {valid_mask.sum()} valid homes out of {len(df_copy)} for {end_use} replacement")

    # Validate menu_mp
    valid_menu_mps = [7, 8, 9, 10]
    if menu_mp not in valid_menu_mps:
        raise ValueError("Please enter a valid measure package number for menu_mp. Should be 7, 8, 9, or 10.")
    
    # Get conditions, technology-efficiency pairs, and cost components for the specified end_use
    params = get_end_use_replacement_parameters(df_copy, end_use)
    conditions = params['conditions']
    tech_eff_pairs = params['tech_eff_pairs']
    cost_components = params['cost_components']
   
    # Map each condition to its tech and efficiency using numpy.select
    tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
    eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default=np.nan)

    # Convert efficiency values to appropriate types based on end use
    if end_use == 'heating':
        eff = np.array([str(e) if e != 'unknown' else np.nan for e in eff])
    else:
        eff = np.array([float(e) if e != 'unknown' else np.nan for e in eff])

    try:
        # Use the standard filtering function to get only homes with both valid data and identifiable tech
        df_valid, valid_calculation_indices, tech_filtered, eff_filtered = filter_valid_tech_homes(
            df_copy, valid_mask, tech, eff)
        
        print(f"After tech filtering: {len(valid_calculation_indices)} homes remain valid for {end_use} replacement")

        if df_valid.empty:
            print(f"Warning: No valid homes found for {end_use} replacement cost calculation.")
            
        # Sample costs from distributions only if we have valid homes
        if not df_valid.empty:
            sampled_costs_dict = sample_costs_from_distributions(tech_filtered, eff_filtered, cost_dict, cost_components)
            
            # Calculate the replacement cost for each row
            replacement_cost, cost_column_name = calculate_replacement_cost_per_row(df_valid, sampled_costs_dict, menu_mp, end_use)
        else:
            cost_column_name = f'mp{menu_mp}_{end_use}_replacementCost'
        
        # Initialize the result series properly
        result_series = create_retrofit_only_series(df_copy, valid_mask)
        
        # Update only for homes that have valid data AND match our tech criteria
        if not df_valid.empty:
            result_series.loc[valid_calculation_indices] = np.round(replacement_cost, 2)
    except Exception as e:
        raise RuntimeError(f"Error in {end_use} replacement cost calculation: {str(e)}")

    # Then create the DataFrame column
    df_new_columns = pd.DataFrame({cost_column_name: result_series})    

    # Track the column for masking
    category_columns_to_mask.append(cost_column_name)
    
    # Apply new columns to DataFrame with proper tracking
    df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
        df_copy, df_new_columns, end_use, category_columns_to_mask, all_columns_to_mask)
    
    # Apply final verification masking for consistency
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=True)

    return df_copy

