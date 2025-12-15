import os
import pandas as pd
import numpy as np
from typing import Literal, Optional


"""
========================================================================================================================================================================
OVERVIEW: EXTRACT EQUIPMENT PERFORMANCE METRICS FOR REMDB V4 COST CALCULATIONS
========================================================================================================================================================================
This module extracts performance metrics from equipment specifications (baseline or upgrade).
These metrics feed into REMDB v4 regression formulas for cost estimation.

These functions support the regression-based cost methodology:
Material_Price = (pm1_coef × Metric1) + (pm2_coef × Metric2) + Intercept
Installed_Cost = Material_Price × multiplier_retrofit (or + adder_retrofit)

WORKFLOW:
1. Call metric extraction function FIRST (in notebook)
2. Then call cost calculation function
"""

def load_remdb_v4_data(
        data_dir: Optional[str] = None,
        filename: str = "remdb_v4_tare_retrofit_costs.csv"
) -> pd.DataFrame:
    """
    Load REMDB v4 retrofit cost data.
    
    Args:
        data_dir: Optional custom directory path
        filename: CSV filename (default: "remdb_v4_tare_retrofit_costs.csv")
        
    Returns:
        DataFrame indexed by row_id with cost regression parameters
        
    Raises:
        FileNotFoundError: If cost data file doesn't exist
        ValueError: If required columns missing
    """
    # Determine file path
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', 'data', 'retrofit_costs'
        )
    
    file_path = os.path.join(data_dir, filename)
    
    # Check file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"REMDB v4 cost data not found at: {file_path}\n"
            f"Expected file: {filename}"
        )
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Data cols in the REMDB v4 cost dataframe
    required_cols = [
        'row_id',
        'tare_category',
        'multiplier_retrofit',
        'adder_retrofit'
    ]
  
    for percentile in ['low', 'mid', 'high']:
        required_cols.extend([
            f'pm1_coef_{percentile}',
            f'pm2_coef_{percentile}',
            f'intercept_{percentile}'
        ])

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"REMDB v4 data missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    
    # Set index
    df = df.set_index('row_id')
    
    return df


# ========== Extract performance metrics from equipment specifications (REPLACEMENT) ==========
# IMPORT UTILITY FUNCTION FROM utils/remdb_v4_installed_cost_utils.py
def add_remdb_replacement_metrics(
    df: pd.DataFrame,
    end_use: Literal['heating', 'cooling', 'waterHeating', 'clothesDrying', 'cooking']
) -> pd.DataFrame:
    """Extract performance metrics from BASELINE equipment for REMDB v4 cost calculations.
    
    This function extracts metrics from EXISTING equipment (not upgrade specs).
    
    Args:
        df: DataFrame with baseline equipment specifications.
        end_use: Equipment category.
        
    Returns:
        DataFrame with {end_use}_replace_metric1 and {end_use}_replace_metric2 columns.
        
    Note:
        Values are documented with sources from EUSS enumeration dictionary.
    """
    df_copy = df.copy()

    replace_or_upgrade = 'replace'
    
    if not isinstance(df_copy, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df_copy).__name__}")
    
    valid_categories = ['heating', 'cooling', 'waterHeating', 'clothesDrying', 'cooking']
    if end_use not in valid_categories:
        raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of {valid_categories}")
        
    # ===== HEATING =====
    if end_use == 'heating':
        # Metric1: Capacity in tons     
        # =============================================================================================================
        # No longer using summing the loads for system size and cost estimation
        # The supplemental heating (electric strip heat) is implicitly included in the REMDB v4 costs
        # Also, the primary system size is the same for both heating and cooling. You wouldnt have two different ASHP tonnages.
        # =============================================================================================================
        if 'size_heating_system_primary_k_btu_h' not in df_copy.columns:
            raise KeyError("Missing heating load columns")

        df_copy[f'{end_use}_{replace_or_upgrade}_metric1'] = df_copy['size_heating_system_primary_k_btu_h'] / 12.0
        
        # Metric2: Efficiency (SEER for heat pumps, AFUE for furnaces)
        if 'hvac_heating_efficiency' not in df_copy.columns:
            raise KeyError("Missing 'hvac_heating_efficiency' column")
        
        seer_extract = df_copy['hvac_heating_efficiency'].str.extract(
            r'SEER\s*(\d+\.?\d*)', expand=False, flags=0)
        afue_extract = df_copy['hvac_heating_efficiency'].str.extract(
            r'(\d+\.?\d*)%?\s*AFUE', expand=False, flags=0)
        
        df_copy[f'{end_use}_{replace_or_upgrade}_metric2'] = pd.to_numeric(
            seer_extract.fillna(afue_extract), errors='coerce')
    
    # ===== COOLING =====
    elif end_use == 'cooling':
        # Metric1: Capacity in tons
        # The primary system size is the same for both heating and cooling
        if 'size_cooling_system_primary_k_btu_h' not in df_copy.columns:
            raise KeyError("Missing cooling load columns")

        df_copy[f'{end_use}_{replace_or_upgrade}_metric1'] = df_copy['size_cooling_system_primary_k_btu_h'] / 12.0
        
        # Metric2: SEER
        if 'hvac_cooling_efficiency' in df_copy.columns:
            df_copy[f'{end_use}_{replace_or_upgrade}_metric2'] = df_copy['hvac_cooling_efficiency'].str.extract(
                r'SEER\s*(\d+\.?\d*)', expand=False).astype(float)
        elif f'heating_{replace_or_upgrade}_metric2' in df_copy.columns:
            df_copy[f'{end_use}_{replace_or_upgrade}_metric2'] = df_copy[f'heating_{replace_or_upgrade}_metric2']
        else:
            raise KeyError("Missing 'hvac_cooling_efficiency' column")
    
    # ===== WATER HEATING =====
    elif end_use == 'waterHeating':
        # Metric1: UEF
        if 'water_heater_efficiency' not in df_copy.columns:
            raise KeyError("Missing 'water_heater_efficiency' column")
        
        # Try to extract UEF from string
        uef_extract = df_copy['water_heater_efficiency'].str.extract(r'(\d+\.?\d*)\s*UEF', expand=False)
        
        # Assign defaults by fuel type
        # Source: EUSS enumeration dictionary
        # - Natural Gas Standard: EF = 0.59
        # - Electric Standard: EF = 0.92
        # - Electric Heat Pump, 80 gal: UEF = 3.45
        if 'base_waterHeating_fuel' in df_copy.columns:
            uef_defaults = pd.Series(np.nan, index=df_copy.index)
            
            fossil_mask = df_copy['base_waterHeating_fuel'].isin(['Natural Gas', 'Fuel Oil', 'Propane'])
            uef_defaults.loc[fossil_mask] = 0.59  # Standard efficiency
            
            electric_mask = (df_copy['base_waterHeating_fuel'] == 'Electricity') & \
                           ~df_copy['water_heater_efficiency'].str.contains('Heat Pump', case=False, na=False)
            uef_defaults.loc[electric_mask] = 0.92  # Standard efficiency
            
            hp_mask = df_copy['water_heater_efficiency'].str.contains('Heat Pump', case=False, na=False)
            uef_defaults.loc[hp_mask] = 3.45  # Heat pump
            
            df_copy[f'{end_use}_{replace_or_upgrade}_metric1'] = pd.to_numeric(
                uef_extract.fillna(uef_defaults), errors='coerce')
        else:
            df_copy[f'{end_use}_{replace_or_upgrade}_metric1'] = pd.to_numeric(uef_extract, errors='coerce')
        
        # Metric2: Tank capacity
        if 'size_water_heater_gal' not in df_copy.columns:
            raise KeyError("Missing 'size_water_heater_gal' column")
        df_copy[f'{end_use}_{replace_or_upgrade}_metric2'] = df_copy['size_water_heater_gal'].astype(float)
    
    # ===== CLOTHES DRYING =====
    elif end_use == 'clothesDrying':
        # Metric1: Drum volume - will be calculated from REMDB bounds in cost function
        df_copy[f'{end_use}_{replace_or_upgrade}_metric1'] = np.nan
        
        # Metric2: CEF by fuel type
        # Source: EUSS enumeration dictionary
        # - Electric: CEF = 2.7
        # - Gas/Propane: CEF = 2.39
        df_copy[f'{end_use}_{replace_or_upgrade}_metric2'] = 2.7  # Electric default
        
        if 'base_clothesDrying_fuel' in df_copy.columns:
            gas_mask = df_copy['base_clothesDrying_fuel'].isin(['Natural Gas', 'Propane'])
            df_copy.loc[gas_mask, f'{end_use}_{replace_or_upgrade}_metric2'] = 2.39
    
    # ===== COOKING =====
    elif end_use == 'cooking':
        # Metric1: Oven volume - will be calculated from REMDB bounds in cost function
        df_copy[f'{end_use}_{replace_or_upgrade}_metric1'] = np.nan
        
        # Metric2: Not used for cooking
        df_copy[f'{end_use}_{replace_or_upgrade}_metric2'] = np.nan
    
    return df_copy


# ========== Extract performance metrics from equipment specifications (REPLACEMENT) ==========
def add_remdb_upgrade_metrics(
    df: pd.DataFrame,
    end_use: Literal['heating', 'cooling', 'waterHeating', 'clothesDrying', 'cooking']
) -> pd.DataFrame:
    """Extract performance metrics from UPGRADE equipment for REMDB v4 cost calculations.
    
    This function extracts metrics from UPGRADE specs (not baseline).
    
    Args:
        df: DataFrame with upgrade equipment specifications.
        end_use: Equipment category.
        
    Returns:
        DataFrame with {end_use}_upgrade_metric1 and {end_use}_upgrade_metric2 columns.
    """
    df_copy = df.copy()

    replace_or_upgrade = 'upgrade'
    
    if not isinstance(df_copy, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df_copy).__name__}")
    
    valid_categories = ['heating', 'cooling', 'waterHeating', 'clothesDrying', 'cooking']
    if end_use not in valid_categories:
        raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of {valid_categories}")
        
    # ===== HEATING =====
    if end_use == 'heating':
        # Metric1: Capacity in tons     
        # =============================================================================================================
        # No longer using summing the loads for system size and cost estimation
        # The supplemental heating (electric strip heat) is implicitly included in the REMDB v4 costs
        # Also, the primary system size is the same for both heating and cooling. You wouldnt have two different ASHP tonnages.
        # =============================================================================================================
        if 'size_heating_system_primary_k_btu_h' not in df_copy.columns:
            raise KeyError("Missing heating load columns")

        df_copy[f'{end_use}_{replace_or_upgrade}_metric1'] = df_copy['size_heating_system_primary_k_btu_h'] / 12.0
        
        # Metric2: SEER from upgrade spec
        if 'upgrade_hvac_heating_efficiency' not in df_copy.columns:
            raise KeyError("Missing 'upgrade_hvac_heating_efficiency'")
        df_copy[f'{end_use}_{replace_or_upgrade}_metric2'] = df_copy['upgrade_hvac_heating_efficiency'].str.extract(
            r'SEER (\d+\.?\d*)', expand=False).astype(float)
    
    # ===== COOLING =====
    elif end_use == 'cooling':
        # Metric1: Capacity in tons
        # The primary system size is the same for both heating and cooling
        if 'size_cooling_system_primary_k_btu_h' not in df_copy.columns:
            raise KeyError("Missing cooling load columns")

        df_copy[f'{end_use}_{replace_or_upgrade}_metric1'] = df_copy['size_cooling_system_primary_k_btu_h'] / 12.0
        
        # Metric2: SEER
        if 'upgrade_hvac_cooling_efficiency' in df_copy.columns:
            df_copy[f'{end_use}_{replace_or_upgrade}_metric2'] = df_copy['upgrade_hvac_cooling_efficiency'].str.extract(
                r'SEER (\d+\.?\d*)', expand=False).astype(float)
        elif 'upgrade_SEER' in df_copy.columns:
            df_copy[f'{end_use}_{replace_or_upgrade}_metric2'] = df_copy['upgrade_SEER'].astype(float)
        else:
            raise KeyError("Missing upgrade_hvac_cooling_efficiency or upgrade_SEER")
    
    # ===== WATER HEATING =====
    elif end_use == 'waterHeating':
        # Metric1: UEF
        if 'upgrade_water_heater_efficiency' not in df_copy.columns:
            raise KeyError("Missing 'upgrade_water_heater_efficiency'")
        df_copy[f'{end_use}_{replace_or_upgrade}_metric1'] = df_copy['upgrade_water_heater_efficiency'].str.extract(
            r'(\d+\.?\d*)\s*UEF', expand=False).astype(float)
        
        # Metric2: Tank capacity
        if 'size_water_heater_gal' not in df_copy.columns:
            raise KeyError("Missing 'size_water_heater_gal'")
        df_copy[f'{end_use}_{replace_or_upgrade}_metric2'] = df_copy['size_water_heater_gal'].astype(float)
    
    # ===== CLOTHES DRYING =====
    elif end_use == 'clothesDrying':
        # Metric1: Drum volume - will be calculated from REMDB bounds
        df_copy[f'{end_use}_{replace_or_upgrade}_metric1'] = np.nan
        
        # Metric2: CEF
        if 'upgrade_clothes_dryer' not in df_copy.columns:
            raise KeyError("Missing 'upgrade_clothes_dryer'")
        is_hp = df_copy['upgrade_clothes_dryer'].str.contains('Heat Pump|HP', case=False, na=False)
        df_copy[f'{end_use}_{replace_or_upgrade}_metric2'] = is_hp.map({True: 5.2, False: 2.7})
    
    # ===== COOKING =====
    elif end_use == 'cooking':
        # Metric1: Oven volume - will be calculated from REMDB bounds
        df_copy[f'{end_use}_{replace_or_upgrade}_metric1'] = np.nan
        
        # Metric2: Not used for cooking
        df_copy[f'{end_use}_{replace_or_upgrade}_metric2'] = np.nan
    
    return df_copy

# ========== Mapping cost parameters from REMDB v4 database using unique row_id ==========
# IMPORT UTILITY FUNCTION FROM utils/remdb_v4_installed_cost_utils.py
def map_remdb_cost_parameters(
    df: pd.DataFrame,
    remdb_v4_costs: pd.DataFrame,
    end_use: str,
    replace_or_upgrade: str,
    percentile: str = 'mid'
) -> pd.DataFrame:
    """Map REMDB v4 cost parameters to main DataFrame.
    
    Uses row_id to look up regression coefficients and installation multipliers
    from the REMDB v4 cost database.
    
    Args:
        df: Main DataFrame with row_id_{end_use}_{replace_or_upgrade} column
        remdb_v4_costs: REMDB v4 cost data (indexed by row_id)
        end_use: Equipment category
        replace_or_upgrade: 'replace' or 'upgrade'
        percentile: Cost percentile ('low', 'mid', 'high')
        
    Returns:
        DataFrame with mapped REMDB parameters as new columns
    """
    df_copy = df.copy()
    
    if percentile not in ['low', 'mid', 'high']:
        raise ValueError(f"Invalid percentile: {percentile}. Must be 'low', 'mid', or 'high'")
    
    # REMDB v4 parameters needed for cost calculation
    # Data cols in the REMDB v4 cost dataframe
    param_cols = [
        f'pm1_coef_{percentile}',
        f'pm2_coef_{percentile}',
        f'intercept_{percentile}',
        'multiplier_retrofit',
        'adder_retrofit'
    ]
    
    # Verify REMDB data has required columns
    missing = [col for col in param_cols if col not in remdb_v4_costs.columns]
    if missing:
        raise ValueError(f"REMDB v4 data missing columns: {missing}")
    
    row_id_col = f'row_id_{end_use}_{replace_or_upgrade}'
    if row_id_col not in df_copy.columns:
        raise ValueError(f"DataFrame missing column: {row_id_col}")
    
    # Map each parameter using row_id as lookup key
    for param in param_cols:
        df_copy[f'{end_use}_{replace_or_upgrade}_{param}'] = df_copy[row_id_col].map(remdb_v4_costs[param])
    
    return df_copy


# ========== Calculate metric from REMDB bounds ==========
def calculate_metric_from_remdb_bounds(
    df: pd.DataFrame,
    remdb_v4_costs: pd.DataFrame,
    end_use: str,
    replace_or_upgrade: Literal['replace', 'upgrade'],
    lower_bound_col: str = 'pm1_lower_bound',
    upper_bound_col: str = 'pm1_upper_bound'
) -> pd.Series:
    """Calculate missing performance metric from REMDB v4 bounds.
    
    For equipment where physical dimensions aren't in home metadata
    (e.g., drum volume for clothes dryers, oven volume for cooking ranges),
    calculate metrics as the midpoint of bounds from REMDB v4 database.
    
    This function should be called AFTER row_id assignment but BEFORE cost calculation.
    
    Args:
        df: DataFrame with row_id_{end_use}_{replace_or_upgrade} column.
           Can be the full dataframe or a filtered subset (e.g., only rows with missing metrics).
        remdb_v4_costs: REMDB v4 cost database (indexed by row_id).
        end_use: Equipment category (e.g., 'clothesDrying', 'cooking').
        replace_or_upgrade: 'replace' or 'upgrade'.
        lower_bound_col: Column name in REMDB for lower bound (default: 'pm1_lower_bound').
        upper_bound_col: Column name in REMDB for upper bound (default: 'pm1_upper_bound').
        
    Returns:
        Series with calculated metric values (midpoint of REMDB bounds), 
        indexed to match input DataFrame.

    Raises:
        KeyError: If required columns are missing.

    Example:
        >>> # Calculate drum volume for clothes dryers with missing metric1
        >>> metric1_col = f'{end_use}_{replace_or_upgrade}_metric1'
        >>> missing_mask = df[metric1_col].isna()
        >>> df.loc[missing_mask, metric1_col] = calculate_metric_from_remdb_bounds(
        ...     df=df[missing_mask],  # Pass only rows that need calculation
        ...     remdb_v4_costs=remdb_costs,
        ...     end_use='clothesDrying',
        ...     replace_or_upgrade='replace',
        ...     lower_bound_col='pm1_lower_bound',
        ...     upper_bound_col='pm1_upper_bound'
        ... )   # Note: metric_col parameter removed
    """

    row_id_col = f'row_id_{end_use}_{replace_or_upgrade}'
    
    # Validate required columns exist
    if row_id_col not in df.columns:
        raise KeyError(
            f"Missing column: '{row_id_col}'. "
            f"Row IDs must be assigned before calculating bounds."
        )
    
    if lower_bound_col not in remdb_v4_costs.columns:
        raise KeyError(
            f"REMDB v4 data missing '{lower_bound_col}' column. "
            f"Available columns: {list(remdb_v4_costs.columns)}"
        )
    
    if upper_bound_col not in remdb_v4_costs.columns:
        raise KeyError(
            f"REMDB v4 data missing '{upper_bound_col}' column. "
            f"Available columns: {list(remdb_v4_costs.columns)}"
        )
    
    # Map bounds from REMDB database
    pm_lower = df[row_id_col].map(remdb_v4_costs[lower_bound_col])
    pm_upper = df[row_id_col].map(remdb_v4_costs[upper_bound_col])
    
    # Calculate metric as midpoint of bounds
    # Result automatically preserves the index from input df
    calculated_metric = (pm_lower + pm_upper) / 2.0
    
    return calculated_metric


# ========== Calculating costs using REMDB v4 regression formula ==========
# IMPORT UTILITY FUNCTION FROM utils/remdb_v4_installed_cost_utils.py
def remdb_cost_regression_formula(
    df: pd.DataFrame,
    replace_or_upgrade: str,
    end_use: str,
    percentile: str = 'mid'
) -> pd.Series:
    """
    Calculate installed costs for REPLACEMENT AND UPGRADE costs using REMDB v4 regression.
    
    Applies the formula:
    Material_Price = (metric1 × pm1_coef) + (metric2 × pm2_coef) + intercept
    Installed_Cost = (Material_Price × multiplier_retrofit) + adder_retrofit
    
    Args:
        df: DataFrame with metrics and mapped REMDB parameters
        replace_or_upgrade: 'replace' or 'upgrade'
        end_use: Equipment category
        percentile: Cost percentile being used
        
    Returns:
        Series with calculated installed costs (works for both replacement and upgrade)        
    """
    df_copy = df.copy()

    # Define column names based on end_use
    metric1 = f'{end_use}_{replace_or_upgrade}_metric1'
    metric2 = f'{end_use}_{replace_or_upgrade}_metric2'
    pm1_coef = f'{end_use}_{replace_or_upgrade}_pm1_coef_{percentile}'
    pm2_coef = f'{end_use}_{replace_or_upgrade}_pm2_coef_{percentile}'
    intercept = f'{end_use}_{replace_or_upgrade}_intercept_{percentile}'
    multiplier_retrofit = f'{end_use}_{replace_or_upgrade}_multiplier_retrofit'
    adder_retrofit = f'{end_use}_{replace_or_upgrade}_adder_retrofit'
    
    # Validate required columns exist
    required = [metric1, pm1_coef, intercept, multiplier_retrofit, adder_retrofit]
    missing = [col for col in required if col not in df_copy.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Calculate material price using regression formula
    material_price = (
        df_copy[metric1] * df_copy[pm1_coef] + 
        df_copy[intercept]
    )
    
    # Add metric2 component if present (not all end-uses use metric2)
    if metric2 in df_copy.columns and pm2_coef in df_copy.columns:
        # Use fillna(0) so missing metric2 doesn't break calculation
        material_price += df_copy[metric2].fillna(0) * df_copy[pm2_coef].fillna(0)
    
    # Apply installation multiplier_retrofit and adder_retrofit
    installed_cost = (material_price * df_copy[multiplier_retrofit]) + df_copy[adder_retrofit]
    
    # Apply validation mask - only valid homes get cost values
    include_col = f'include_{end_use}'
    if include_col in df_copy.columns:
        # FIXED: Convert np.where result to Series with proper index
        installed_cost_array = np.where(df_copy[include_col], installed_cost, np.nan)
        installed_cost = pd.Series(installed_cost_array, index=df_copy.index)
    
    return installed_cost.round(2)
