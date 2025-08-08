import pandas as pd
import numpy as np
from scipy.stats import norm

from config import PROJECT_ROOT
print(f"Project root directory: {PROJECT_ROOT}")
 
"""
========================================================================================================================================================================
OVERVIEW: CALCULATE INSTALLATION COSTS FOR VARIOUS END USES
========================================================================================================================================================================
This module calculates the installation costs for various end uses such as heating, water heating,
clothes drying, and cooking. It uses a probabilistic approach to sample costs from distributions defined by
progressive (10th percentile), reference (50th percentile), and conservative (90th percentile) cost estimates.

# UPDATED MARCH 24, 2025 @ 4:30 PM - REMOVED RSMEANS CCI ADJUSTMENTS
# UPDATED APRIL 9, 2025 @ 7:30 PM - IMPROVED DOCUMENTATION
"""

# ========================================================================================================================================================================
# FUNCTIONS: HELPER FUNCTIONS FOR SPACE HEATING
# ========================================================================================================================================================================

def obtain_heating_system_specs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and process heating system specifications from input dataframe.
    
    Calculates total heating load and extracts efficiency metrics from raw data. 
    
    Args:
        df (pd.DataFrame): Input dataframe containing heating system data
        
    Returns:
        pd.DataFrame: Updated dataframe with calculated heating system specs
        
    Raises:
        ValueError: If dataframe is missing required columns
    """
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


def calculate_heating_installation_premium(df: pd.DataFrame,
                                           menu_mp: int,
                                           cpi_ratio_2023_2013: float) -> pd.DataFrame:
    """
    Calculate premium costs for heating system installation based on existing infrastructure.
    
    Adds costs for homes without existing central AC or with boiler systems that
    need additional modifications for heat pump installation.
    
    Args:
        df (pd.DataFrame): Input dataframe containing heating system data
        menu_mp (int): Menu package identifier 
        cpi_ratio_2023_2013 (float): Consumer price index ratio for adjusting 2013 costs to 2023
        
    Returns:
        pd.DataFrame: Updated dataframe with heating installation premium costs
        
    Raises:
        ValueError: If dataframe is missing required columns
    """
    necessary_columns = ['hvac_cooling_type', 'heating_type']
    if not all(column in df.columns for column in necessary_columns):
        raise ValueError("DataFrame does not contain all necessary columns.")
    
    for index, row in df.iterrows():
        # Initialization to zero
        premium_cost = 0
        
        # Installation cost for homes with existing AC
        # Deetjen: Replace SEER 15, 8.5 HSPF ASHP with SEER 15, 8.5 HSPF ASHP: NREL REMDB 50th Percentile Cost is $3300 USD-2013        
        if row['hvac_cooling_type'] != 'None':
            premium_cost = 0
        
        # Installation cost for homes without central AC, but an existing furnace or baseboard
        # Deetjen: Install SEER 15, 8.5 HSPF ASHP: NREL REMDB 50th Percentile Cost is $3700 USD-2013        
        elif 'Furnace' in row['heating_type'] or 'Baseboard' in row['heating_type']:
            premium_cost = 400 * cpi_ratio_2023_2013
        
        # Installation cost for homes without central AC and an existing boiler as heating system
        # Deetjen: Install SEER 15, 8.5 HSPF ASHP: NREL REMDB High Cost is $4800 USD-2013        
        elif 'Boiler' in row['heating_type']:
            premium_cost = 1500 * cpi_ratio_2023_2013
        
        # Apply CPI adjustment above
        adjusted_cost = round(premium_cost, 2)
        df.at[index, f'mp{menu_mp}_heating_installation_premium'] = adjusted_cost
        
    return df


# ========================================================================================================================================================================
# FUNCTIONS: CALCULATE COST OF INSTALLING NEW EQUIPMENT (RETROFIT/UPGRADES)
# ========================================================================================================================================================================

def get_end_use_installation_parameters(df: pd.DataFrame,
                                        end_use: str,
                                        menu_mp: int) -> dict:
    """
    Retrieve parameters for installation cost calculations based on end use type.
    
    Returns conditions, technology-efficiency pairs, and cost components tailored 
    to specific end uses like heating, water heating, clothes drying, and cooking.
    
    Args:
        df (pd.DataFrame): Input dataframe with equipment data
        end_use (str): Type of end use ('heating', 'waterHeating', 'clothesDrying', 'cooking')
        menu_mp (int): Menu package identifier
        
    Returns:
        dict: Dictionary containing conditions, technology-efficiency pairs, and cost components
        
    Raises:
        ValueError: If an invalid end_use is specified
    """
    parameters = {
        'heating': {
            'conditions': [
                (df['hvac_has_ducts'] == 'Yes') & (menu_mp == 7),
                (df['hvac_has_ducts'] == 'No') & (menu_mp == 7),
                (df['hvac_has_ducts'] == 'Yes') & (menu_mp != 7),
                (df['hvac_has_ducts'] == 'No') & (menu_mp != 7)
            ],
            'tech_eff_pairs': [
                ('Electric ASHP', 'SEER 18, 9.3 HSPF'),
                ('Electric MSHP', 'SEER 18, 9.6 HSPF'),
                ('Electric MSHP - Ducted', 'SEER 15.5, 10 HSPF'),
                ('Electric MSHP', 'SEER 29.3, 14 HSPF')
            ],
            'cost_components': ['unitCost', 'otherCost', 'cost_per_kBtuh']
        },
        'waterHeating': {
            'conditions': [
                (df['upgrade_water_heater_efficiency'] == 'Electric Heat Pump, 50 gal, 3.45 UEF'),
                (df['upgrade_water_heater_efficiency'] == 'Electric Heat Pump, 66 gal, 3.35 UEF'),
                (df['upgrade_water_heater_efficiency'] == 'Electric Heat Pump, 80 gal, 3.45 UEF')
            ],
            'tech_eff_pairs': [
                ('Electric Heat Pump Water Heater, 50 gal', 3.45),
                ('Electric Heat Pump Water Heater, 66 gal', 3.35),
                ('Electric Heat Pump Water Heater, 80 gal', 3.45),
            ],
            'cost_components': ['unitCost', 'cost_per_gallon']
        },
        'clothesDrying': {
            'conditions': [
                df['upgrade_clothes_dryer'].str.contains('Electric, Premium, Heat Pump, Ventless', na=False),
                ~df['upgrade_clothes_dryer'].str.contains('Electric, Premium, Heat Pump, Ventless', na=False),
            ],
            'tech_eff_pairs': [
                ('Electric HP Clothes Dryer', 5.2),
                ('Electric Clothes Dryer', 3.1),
            ],
            'cost_components': ['unitCost']
        },
        'cooking': {
            'conditions': [
                df['upgrade_cooking_range'].str.contains('Electric, Induction', na=False),
                ~df['upgrade_cooking_range'].str.contains('Electric, Induction', na=False),
            ],
            'tech_eff_pairs': [
                ('Electric Induction Range', 0.84),
                ('Electric Range, Modern', 0.74),
            ],
            'cost_components': ['unitCost']
        }
    }
    if end_use not in parameters:
        raise ValueError(f"Invalid end_use specified: {end_use}")
    return parameters[end_use]


def calculate_installation_cost_per_row(df_valid: pd.DataFrame,
                                        sampled_costs_dict: dict,
                                        menu_mp: int,
                                        end_use: str) -> tuple:
    """
    Calculate the installation cost for each row based on the end use type.
    
    Applies specific cost formulas depending on the end use category.

    Args:
        df_valid (pd.DataFrame): Filtered DataFrame containing valid rows
        sampled_costs_dict (dict): Dictionary with sampled costs for each component
        menu_mp (int): Menu option identifier
        end_use (str): Type of end-use ('heating', 'waterHeating', 'clothesDrying', 'cooking')

    Returns:
        tuple: (installation_cost, cost_column_name) containing the calculated costs and column name
    """
    if end_use == 'heating':
        # For heating: base cost + other costs + capacity-based costs
        installation_cost = (
            sampled_costs_dict['unitCost'] +
            sampled_costs_dict['otherCost'] +
            (df_valid['total_heating_load_kBtuh'] * sampled_costs_dict['cost_per_kBtuh']))
        cost_column_name = f'mp{menu_mp}_heating_installationCost'
    elif end_use == 'waterHeating':
        # For water heating: base cost + volume-based costs
        installation_cost = (
            sampled_costs_dict['unitCost'] +
            (sampled_costs_dict['cost_per_gallon'] * df_valid['size_water_heater_gal']))
        cost_column_name = f'mp{menu_mp}_waterHeating_installationCost'
    else:
        # For clothes drying and cooking: only base unit cost
        installation_cost = sampled_costs_dict['unitCost']
        cost_column_name = f'mp{menu_mp}_{end_use}_installationCost'
    
    return installation_cost, cost_column_name


def calculate_installation_cost(df: pd.DataFrame,
                                cost_dict: dict,
                                menu_mp: int,
                                end_use: str) -> pd.DataFrame:
    """
    Calculate installation costs for various end-uses based on technology types, efficiency, and cost data.

    This function applies a probabilistic approach by sampling from cost distributions defined
    by progressive (10th percentile), reference (50th percentile), and conservative (90th percentile)
    cost estimates for different technology and efficiency combinations.

    Args:
        df (pd.DataFrame): DataFrame containing data for different scenarios
        cost_dict (dict): Dictionary mapping (technology, efficiency) tuples to cost components
        menu_mp (int): Menu option identifier (valid values: 7, 8, 9, 10)
        end_use (str): Type of end-use ('heating', 'waterHeating', 'clothesDrying', 'cooking')

    Returns:
        pd.DataFrame: Updated DataFrame with calculated installation costs

    Raises:
        ValueError: If menu_mp is not valid or if cost data is missing for any technology/efficiency combination
    """
    
    # Validate menu_mp 
    valid_menu_mps = [7, 8, 9, 10]
    if menu_mp not in valid_menu_mps:
        raise ValueError("Please enter a valid measure package number for menu_mp. Should be 7, 8, 9, or 10.")
    
    # Get conditions, technology-efficiency pairs, and cost components for the specified end_use
    params = get_end_use_installation_parameters(df, end_use, menu_mp)
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
        # Extract the progressive (10th percentile), reference (50th percentile), and conservative (90th percentile) costs
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
        mean_costs = reference_costs  # 50th percentile becomes the mean
        # Calculate standard deviation using the difference between 90th and 10th percentiles
        std_costs = (conservative_costs - progressive_costs) / (norm.ppf(0.90) - norm.ppf(0.10))

        # Sample from the normal distribution for each row
        sampled_costs = np.random.normal(loc=mean_costs, scale=std_costs)
        sampled_costs_dict[cost_component] = sampled_costs

    # Calculate the installation cost for each row
    installation_cost, cost_column_name = calculate_installation_cost_per_row(df_valid, sampled_costs_dict, menu_mp, end_use)

    # Add the calculated costs to a new DataFrame, rounded to 2 decimal places
    df_new_columns = pd.DataFrame({cost_column_name: np.round(installation_cost, 2)}, index=df_valid.index)

    # Identify overlapping columns between the new and existing DataFrame
    overlapping_columns = df_new_columns.columns.intersection(df.columns)

    # Drop overlapping columns from the original DataFrame
    if not overlapping_columns.empty:
        df.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into the original DataFrame, ensuring no duplicates or overwrites occur
    df = df.join(df_new_columns, how='left')

    return df
