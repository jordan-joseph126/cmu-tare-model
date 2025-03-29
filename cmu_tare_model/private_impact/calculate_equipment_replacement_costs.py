import pandas as pd
import numpy as np
from scipy.stats import norm

from config import PROJECT_ROOT
print(f"Project root directory: {PROJECT_ROOT}")

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FUNCTIONS: HELPER FUNCTIONS FOR SPACE HEATING
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
""" 
def obtain_heating_system_specs(df):
    # Check if necessary columns are in the DataFrame
    necessary_columns = ['size_heating_system_primary_k_btu_h', 'size_heat_pump_backup_primary_k_btu_h',
                         'size_heating_system_secondary_k_btu_h', 'baseline_heating_type']
    if not all(column in df.columns for column in necessary_columns):
        raise ValueError("DataFrame does not contain all necessary columns.")

    # Total heating load in kBtuh
    df['total_heating_load_kBtuh'] = df['size_heating_system_primary_k_btu_h'] + df['size_heat_pump_backup_primary_k_btu_h'] + df['size_heating_system_secondary_k_btu_h']
    
#     # Total heating load in kW
#     df['total_heating_load_kW'] = df['total_heating_load_kBtuh'] * 1000 / 3412.142
   
    # Use regex to remove the fuel and leave only the heating type:
    df['baseline_heating_type'] = df['baseline_heating_type'].str.extract(r'^(?:\d+\s+)?(?:Natural Gas|Electricity|Propane|Fuel Oil|Fuel)\s+(?:Fuel\s+)?(?:Electric\s+)?(.+)$')
    
    # AFUE extraction for existing, baseline equipment (Replacement Costs)
    df['baseline_AFUE'] = df['hvac_heating_efficiency'].str.extract(r'([\d.]+)%').astype(float)
    
    # SEER extraction for existing, baseline equipment (Replacement Costs)
    df['baseline_SEER'] = df['hvac_heating_efficiency'].str.extract(r'SEER ([\d.]+)').astype(float)
    
    # HSPF extraction for existing, baseline equipment (Replacement Costs)
    df['baseline_HSPF'] = df['hvac_heating_efficiency'].str.extract(r'([\d.]+) HSPF').astype(float)

    # HSPF extraction for upgraded equipment (New Install Costs)
    df['ugrade_newInstall_HSPF'] = df['upgrade_hvac_heating_efficiency'].str.extract(r'(\d+\.\d+)')
    
    return df

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FUNCTIONS: CALCULATE COST OF REPLACING EXISTING EQUIPMENT
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# UPDATED MARCH 24, 2025 @ 4:30 PM - REMOVED RSMEANS CCI ADJUSTMENTS
# Replacement Cost Function and Helper Functions (Parametes, Formula)

# Helper function to get parameters based on end use
def get_end_use_replacement_parameters(df, end_use):
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

# UPDATED MARCH 24, 2025 @ 4:30 PM - REMOVED RSMEANS CCI ADJUSTMENTS
def calculate_replacement_cost_per_row(df_valid, sampled_costs_dict, menu_mp, end_use):
    """
    Helper function to calculate the replacement cost for each row based on the end use.

    Parameters:
    df_valid (pd.DataFrame): Filtered DataFrame containing valid rows.
    sampled_costs_dict (dict): Dictionary with sampled costs for each component.
    menu_mp (int): Menu option identifier.
    end_use (str): Type of end-use to calculate replacement cost for ('heating', 'waterHeating', 'clothesDrying', 'cooking').

    Returns:
    tuple: Tuple containing the calculated replacement costs and the cost column name.
    """
    if end_use == 'heating':
        replacement_cost = (
            sampled_costs_dict['unitCost'] +
            sampled_costs_dict['otherCost'] +
            (df_valid['total_heating_load_kBtuh'] * sampled_costs_dict['cost_per_kBtuh']))
        
        cost_column_name = f'mp{menu_mp}_heating_replacementCost'

    elif end_use == 'waterHeating':
        replacement_cost = (
            sampled_costs_dict['unitCost'] +
            (sampled_costs_dict['cost_per_gallon'] * df_valid['size_water_heater_gal']))
        
        cost_column_name = f'mp{menu_mp}_waterHeating_replacementCost'

    else:
        replacement_cost = sampled_costs_dict['unitCost'] 
        cost_column_name = f'mp{menu_mp}_{end_use}_replacementCost'
    
    return replacement_cost, cost_column_name

# UPDATED MARCH 24, 2025 @ 4:30 PM - REMOVED RSMEANS CCI ADJUSTMENTS
def calculate_replacement_cost(df, cost_dict, menu_mp, end_use):
    """
    General function to calculate replacement costs for various end-uses based on fuel types, costs, and efficiency.

    Parameters:
    df (pd.DataFrame): DataFrame containing data for different scenarios.
    cost_dict (dict): Dictionary with cost information for different technology and efficiency combinations.
    menu_mp (int): Menu option identifier.
    end_use (str): Type of end-use to calculate replacement cost for ('heating', 'waterHeating', 'clothesDrying', 'cooking').

    Returns:
    pd.DataFrame: Updated DataFrame with calculated replacement costs.
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
   
    # Map each condition to its tech and efficiency
    tech = np.select(conditions, [pair[0] for pair in tech_eff_pairs], default='unknown')
    eff = np.select(conditions, [pair[1] for pair in tech_eff_pairs], default=np.nan)

    # Convert efficiency values to appropriate types
    if end_use == 'heating':
        eff = np.array([str(e) if e != 'unknown' else np.nan for e in eff])
    else:
        eff = np.array([float(e) if e != 'unknown' else np.nan for e in eff])

    # Filter out rows with unknown technology and NaN efficiency
    valid_indices = tech != 'unknown'
    tech = tech[valid_indices]
    eff = eff[valid_indices]
    df_valid = df.loc[valid_indices].copy()

    # Initialize dictionaries to store sampled costs
    sampled_costs_dict = {}

    # Calculate costs for each component
    for cost_component in cost_components:
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
        mean_costs = reference_costs
        std_costs = (conservative_costs - progressive_costs) / (norm.ppf(0.90) - norm.ppf(0.10))

        # Sample from the normal distribution for each row
        sampled_costs = np.random.normal(loc=mean_costs, scale=std_costs)
        sampled_costs_dict[cost_component] = sampled_costs

    # Calculate the replacement cost for each row
    replacement_cost, cost_column_name = calculate_replacement_cost_per_row(df_valid, sampled_costs_dict, menu_mp, end_use)

    # Add the calculated costs to a new DataFrame, rounded to 2 decimal places
    df_new_columns = pd.DataFrame({cost_column_name: np.round(replacement_cost, 2)}, index=df_valid.index)

    # Identify overlapping columns between the new and existing DataFrame
    overlapping_columns = df_new_columns.columns.intersection(df.columns)

    # Drop overlapping columns from the original DataFrame
    if not overlapping_columns.empty:
        df.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into the original DataFrame, ensuring no duplicates or overwrites occur
    df = df.join(df_new_columns, how='left')

    return df