"""
========================================================================================================================================================================
REMDB v4 Installed Cost Utilities (SIMPLIFIED)
========================================================================================================================================================================

This module prepares equipment metrics for REMDB v4 cost calculations.

The main function `add_remdb_replacement_metrics` AND 'add_remdb_upgrade_metrics' performs ALL preparation:
1. Assigns row_id based on equipment type
2. Maps REMDB coefficients and unit specifications  
3. Extracts metrics from EUSS data with correct unit conversions
4. Fills missing values from REMDB bounds

Output columns ready for cost calculation:
- {end_use}_{replacement_or_upgrade}_pm1: Performance metric 1 (in REMDB units)
- {end_use}_{replacement_or_upgrade}_pm2: Performance metric 2 (in REMDB units)
- Plus coefficient, intercept, multiplier, adder columns

# UPDATED DECEMBER 15, 2025 - Simplified architecture, data-driven unit conversion

NO LONGER USING THE SUM OF THE HEATING AND COOLING LOADS FOR SYSTEM SIZE AND COST ESTIMATION
- The supplemental heating (electric strip heat) is implicitly included in the REMDB v4 costs
- Also, the primary system size is the same for both heating and cooling. You wouldnt have two different ASHP tonnages.

"""

import os
import pandas as pd
import numpy as np
from typing import Optional

from cmu_tare_model.constants import EQUIPMENT_SPECS

# ============================================================
# HELPER FUNCTIONS
# ============================================================

# ===== DATA LOADING FUNCTION =====
def load_remdb_v4_data(
    data_dir: Optional[str] = None,
    filename: str = "remdb_v4_tare_retrofit_costs.csv"
) -> pd.DataFrame:
    """Load REMDB v4 retrofit cost data.
    
    Args:
        data_dir: Optional custom directory path.
        filename: CSV filename.
        
    Returns:
        DataFrame indexed by row_id with cost regression parameters.
    """
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', 'data', 'retrofit_costs'
        )
    
    file_path = os.path.join(data_dir, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"REMDB v4 cost data not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    df = df.set_index('row_id')
    
    return df


# ===== Assign REMDB row_id based on technology =====
def _assign_replacement_row_id(df: pd.DataFrame, end_use: str) -> pd.DataFrame:
    """Assign REMDB row_id based on baseline equipment type."""

    df_copy = df.copy()
    row_id_col = f'row_id_{end_use}_replacement'
        
    # ========== HVAC OPTIONS: HEATING & COOLING ==========
    # The efficiency level does not impact row_id mapping in REMDB v4 but instead pm1/pm2 in the regression formula
    # Generally we use multi-zone non-ducted for homes without ducts, but may update to single-zone in the future for smaller homes 
    # New circuit will be addressed in future versions, but excluded here for simplicity.
    if end_use == 'heating':
        conditions = [
            (df_copy['base_heating_fuel'] == 'Propane'),
            (df_copy['base_heating_fuel'] == 'Fuel Oil'),
            (df_copy['base_heating_fuel'] == 'Natural Gas'),
            (df_copy['base_heating_fuel'] == 'Electricity') & (df_copy['heating_type'] != 'Electricity ASHP'),
            (df_copy['heating_type'] == 'Electricity ASHP') & (df_copy['hvac_has_ducts'] == 'Yes'),
            (df_copy['heating_type'] == 'Electricity ASHP') & (df_copy['hvac_has_ducts'] == 'No')
            ]

        choices = [
            'furnaces_gas_furnace',  # Proxy for propane
            'furnaces_gas_furnace',  # Proxy for fuel oil
            'furnaces_gas_furnace',
            'electric_baseboard_default',
            'air_source_heat_pump_centrally_ducted',
            'air_source_heat_pump_non_ducted_multi_zone'
            ]
        
        df_copy[row_id_col] = np.select(conditions, choices, default='unknown')
    
    elif end_use == 'cooling':
        conditions = [
            (df_copy['hvac_cooling_type'] == 'Room AC'),
            (df_copy['hvac_cooling_type'] == 'Central AC'),
            (df_copy['hvac_cooling_type'] == 'Heat Pump') & (df_copy['hvac_has_ducts'] == 'Yes'),
            (df_copy['hvac_cooling_type'] == 'Heat Pump') & (df_copy['hvac_has_ducts'] == 'No')
        ]

        choices = [
            'air_conditioner_room_ac_window_or_through_wall',
            'air_conditioner_centrally_ducted',
            'air_source_heat_pump_centrally_ducted',
            'air_source_heat_pump_non_ducted_multi_zone'
        ]

        df_copy[row_id_col] = np.select(conditions, choices, default='unknown')
        
    # =========================================
    # DELETE FOR NOW - NON-HVAC END USES
    # =========================================

    # else:
    #     # raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of: 'heating', 'cooling', 'waterHeating', 'clothesDrying', 'cooking'")
    #     raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of: 'heating', 'cooling'")

    return df_copy


# ===== Assign REMDB row_id based on technology =====
def _assign_upgrade_row_id(df: pd.DataFrame, end_use: str) -> pd.DataFrame:
    """Assign REMDB row_id based on baseline equipment type."""

    df_copy = df.copy()
    row_id_col = f'row_id_{end_use}_upgrade'
        
    # ========== HVAC OPTIONS: HEATING & COOLING ---> HEAT PUMP ==========
    # MP7 Standard heat pumps (SEER 18) | MP8-10: High-efficiency heat pumps
    # However, the efficiency level does not impact row_id mapping in REMDB v4 but instead pm1/pm2 in the regression formula
    # Generally we use multi-zone non-ducted for homes without ducts, but may update to single-zone in the future for smaller homes 
    # New circuit will be addressed in future versions, but excluded here for simplicity.
    if end_use == 'heating':        
        conditions = [
            (df_copy['hvac_has_ducts'] == 'Yes'),
            (df_copy['hvac_has_ducts'] == 'No'),
            # (df_copy['hvac_has_ducts'] == 'No') & (df_copy['square_footage'] < 1200)
        ]
        
        choices = [
            'air_source_heat_pump_centrally_ducted',
            'air_source_heat_pump_non_ducted_multi_zone',
            # 'air_source_heat_pump_non_ducted_single_zone',
        ]

        df_copy[row_id_col] = np.select(conditions, choices, default='unknown')
    
    elif end_use == 'cooling':
        # Similar logic to heating - cooling upgrades mirror heat pump installations
        conditions = [
            (df_copy['hvac_has_ducts'] == 'Yes'),
            (df_copy['hvac_has_ducts'] == 'No'),
            # (df_copy['hvac_has_ducts'] == 'No') & (df_copy['square_footage'] < 1200)
        ]
        
        choices = [
            'air_source_heat_pump_centrally_ducted',
            'air_source_heat_pump_non_ducted_multi_zone',
            # 'air_source_heat_pump_non_ducted_single_zone',
        ]

        df_copy[row_id_col] = np.select(conditions, choices, default='unknown')
        
    # =========================================
    # DELETE FOR NOW - NON-HVAC END USES
    # =========================================

    # else:
    #     # raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of: 'heating', 'cooling', 'waterHeating', 'clothesDrying', 'cooking'")
    #     raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of: 'heating', 'cooling'")

    return df_copy


# ===== Map REMDB parameters to DataFrame =====
def _map_remdb_parameters(
    df: pd.DataFrame, 
    remdb_v4_costs: pd.DataFrame, 
    end_use: str,
    replacement_or_upgrade: str,
    percentile: str = 'mid'
) -> pd.DataFrame:
    """
    Map REMDB coefficients and unit specifications to DataFrame.
    Documentation and error handling have been moved to the main functions.
    """
    df_copy = df.copy()
    row_id_col = f'row_id_{end_use}_{replacement_or_upgrade}'
    prefix = f'{end_use}_{replacement_or_upgrade}_'
    
    # Parameters to map from REMDB
    params = [
        'pm1_metric', 'pm1_unit', f'pm1_coef_{percentile}', 
        'pm2_metric', 'pm2_unit', f'pm2_coef_{percentile}',
        f'intercept_{percentile}', 'multiplier_retrofit', 'adder_retrofit'
    ]
    
    for param in params:
        if param in remdb_v4_costs.columns:
            df_copy[prefix + param] = df_copy[row_id_col].map(remdb_v4_costs[param])
    
    return df_copy


# ===== Convert capacity metric (EUSS) to REMDB units =====
def _convert_capacity_by_unit(
    df: pd.DataFrame, 
    capacity_col: str, 
    pm1_unit_col: str
) -> pd.Series:
    """Convert capacity to REMDB units based on pm1_unit column.
    
    Unit Conversions:
        - "Tons"   → kBtu/h ÷ 12
        - "BTU/hr" → kBtu/h × 1000
    """
    result = pd.Series(np.nan, index=df.index)
    
    # Tons: kBtu/h ÷ 12 (heat pumps, central ACs)
    tons_mask = df[pm1_unit_col] == 'Tons'
    if tons_mask.any():
        result.loc[tons_mask] = df.loc[tons_mask, capacity_col] / 12.0
    
    # BTU/hr: kBtu/h × 1000 (furnaces, boilers, baseboard)
    btuh_mask = df[pm1_unit_col].str.lower() == 'btu/hr'
    if btuh_mask.any():
        result.loc[btuh_mask] = df.loc[btuh_mask, capacity_col] * 1000.0
    
    return result


# ===== Convert efficiency metric (EUSS) to REMDB units =====
def _convert_efficiency_by_metric(
    df: pd.DataFrame, 
    efficiency_col: str, 
    pm2_metric_col: str
) -> pd.Series:
    """Convert efficiency to REMDB units based on pm2_metric column.
    
    Unit Conversions:
        - "SEER1" → Extract as-is (13-30 range)
        - "AFUE"  → Extract ÷ 100 (80% → 0.80)
        - "CEER"  → Extract as-is (9-15 range)
        - Empty   → Set to 0 (no efficiency metric)
    """
    result = pd.Series(np.nan, index=df.index)
    
    # SEER: Extract as-is
    seer_mask = df[pm2_metric_col] == 'SEER1'
    if seer_mask.any():
        seer_extract = df.loc[seer_mask, efficiency_col].str.extract(
            r'SEER\s*(\d+\.?\d*)', expand=False)
        result.loc[seer_mask] = pd.to_numeric(seer_extract, errors='coerce')
    
    # AFUE: Extract and divide by 100 (CRITICAL FIX!)
    afue_mask = df[pm2_metric_col] == 'AFUE'
    if afue_mask.any():
        afue_extract = df.loc[afue_mask, efficiency_col].str.extract(
            r'(\d+\.?\d*)%?\s*AFUE', expand=False)
        result.loc[afue_mask] = pd.to_numeric(afue_extract, errors='coerce') / 100.0
    
    # CEER: Extract as-is
    # EER IS WHAT IS SPECIFIED IN EUSS, BUT REMDB USES CEER FOR ROOM ACs
    # THERE IS NO DIRECT CONVERSION SO THIS WILL BE SET TO NAN --> THEN 0.0 
    ceer_mask = df[pm2_metric_col] == 'CEER'
    if ceer_mask.any():
        ceer_extract = df.loc[ceer_mask, efficiency_col].str.extract(
            r'CEER\s*(\d+\.?\d*)', expand=False)
        
        result.loc[ceer_mask] = pd.to_numeric(ceer_extract, errors='coerce')
    
    # Empty: Set to 0 (no efficiency metric used)
    empty_mask = df[pm2_metric_col].isna() | (df[pm2_metric_col] == '')
    if empty_mask.any():
        result.loc[empty_mask] = 0.0
    
    return result


# ===== Fill missing pm1/pm2 values from REMDB bounds =====
def _fill_missing_from_bounds(
    df: pd.DataFrame,
    remdb_v4_costs: pd.DataFrame,
    end_use: str,
    replacement_or_upgrade: str,
    pm1_col: str,
    pm2_col: str
) -> pd.DataFrame:
    """Fill missing pm1/pm2 values from REMDB bounds."""
    df_copy = df.copy()
    row_id_col = f'row_id_{end_use}_{replacement_or_upgrade}'
    
    # Fill pm1 from bounds
    pm1_missing = df_copy[pm1_col].isna()
    if pm1_missing.any() and 'pm1_lower_bound' in remdb_v4_costs.columns:
        lower = df_copy.loc[pm1_missing, row_id_col].map(remdb_v4_costs['pm1_lower_bound'])
        upper = df_copy.loc[pm1_missing, row_id_col].map(remdb_v4_costs['pm1_upper_bound'])
        df_copy.loc[pm1_missing, pm1_col] = (lower + upper) / 2.0
        print(f"  Filled {pm1_missing.sum():,} missing {pm1_col} from REMDB bounds")
    
    # Fill pm2 from bounds  
    pm2_missing = df_copy[pm2_col].isna()
    if pm2_missing.any() and 'pm2_lower_bound' in remdb_v4_costs.columns:
        lower = df_copy.loc[pm2_missing, row_id_col].map(remdb_v4_costs['pm2_lower_bound'])
        upper = df_copy.loc[pm2_missing, row_id_col].map(remdb_v4_costs['pm2_upper_bound'])
        df_copy.loc[pm2_missing, pm2_col] = (lower + upper) / 2.0
        print(f"  Filled {pm2_missing.sum():,} missing {pm2_col} from REMDB bounds")
    
    return df_copy


# ============================================================
# MAIN FUNCTIONS TO MAP MATCHING EQUIPMENT AND PREPARE METRICS FOR COST CALCULATION
# ============================================================

# ==== Prepare REMDB replacement metrics =====
def add_remdb_replacement_metrics(
    df: pd.DataFrame,
    remdb_v4_costs: pd.DataFrame,
    end_use: str,
    percentile: str = 'mid'
) -> pd.DataFrame:
    """Prepare replacement cost metrics from BASELINE equipment for REMDB v4 calculations.
    
    This function performs ALL preparation steps:
    1. Assigns row_id based on equipment type
    2. Maps REMDB coefficients and unit specifications
    3. Extracts metrics from EUSS data with correct unit conversions
    4. Fills missing values from REMDB bounds
    
    Args:
        df: DataFrame with baseline equipment specifications.
        remdb_v4_costs: REMDB v4 cost database (indexed by row_id).
        end_use: Equipment category ('heating', 'cooling').
        percentile: Cost percentile ('low', 'mid', 'high').
        
    Returns:
        DataFrame with columns ready for cost calculation:
        - row_id_{end_use}_replace
        - {end_use}_replacement_pm1 (in REMDB units)
        - {end_use}_replacement_pm2 (in REMDB units)
        - {end_use}_replacement_pm1_coef_{percentile}
        - {end_use}_replacement_pm2_coef_{percentile}
        - {end_use}_replacement_intercept_{percentile}
        - {end_use}_replacement_multiplier_retrofit
        - {end_use}_replace_adder_retrofit
    """
    df_copy = df.copy()
    replacement_or_upgrade = 'replacement'
    
    # Validate inputs
    if not isinstance(df_copy, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df_copy).__name__}")
    
    # Valid categories are defined in EQUIPMENT_SPECS
    valid_categories = list(EQUIPMENT_SPECS.keys())

    if end_use not in valid_categories:
        raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of {valid_categories}")
    
    if percentile not in ['low', 'mid', 'high']:
        raise ValueError(f"Invalid percentile: '{percentile}'")
    
    print(f"\nPreparing {end_use} replacement metrics (REMDB v4)")
    
    # ===== STEP 1: Assign row_id based on equipment type =====
    df_copy = _assign_replacement_row_id(df_copy, end_use)
    row_id_col = f'row_id_{end_use}_{replacement_or_upgrade}'
    unknown_count = (df_copy[row_id_col] == 'unknown').sum()
    if unknown_count > 0:
        print(f"  Warning: {unknown_count:,} homes with unknown row_id")
    
    # ===== STEP 2: Map REMDB parameters =====
    df_copy = _map_remdb_parameters(df_copy, remdb_v4_costs, end_use, replacement_or_upgrade, percentile)
    
    # ===== STEP 3: Extract metrics based on REMDB unit specifications =====
    # Determine source columns
    if end_use == 'heating':
        capacity_col = 'size_heating_system_primary_k_btu_h'
        efficiency_col = 'hvac_heating_efficiency'
    elif end_use == 'cooling':
        capacity_col = 'size_cooling_system_primary_k_btu_h'
        efficiency_col = 'hvac_cooling_efficiency'
    
    # Validate source columns
    if capacity_col not in df_copy.columns:
        raise KeyError(f"Missing capacity column: '{capacity_col}'")
    if efficiency_col not in df_copy.columns:
        raise KeyError(f"Missing efficiency column: '{efficiency_col}'")
    
    # Convert pm1 based on pm1_unit
    pm1_col = f'euss_{end_use}_replacement_pm1'
    pm1_unit_col = f'{end_use}_{replacement_or_upgrade}_pm1_unit'
    df_copy[pm1_col] = _convert_capacity_by_unit(df_copy, capacity_col, pm1_unit_col)
    
    # Convert pm2 based on pm2_metric
    pm2_col = f'euss_{end_use}_replacement_pm2'
    pm2_metric_col = f'{end_use}_{replacement_or_upgrade}_pm2_metric'
    df_copy[pm2_col] = _convert_efficiency_by_metric(df_copy, efficiency_col, pm2_metric_col)
    
    # ===== STEP 4: Fill missing values from REMDB bounds =====
    df_copy = _fill_missing_from_bounds(
        df_copy, remdb_v4_costs, end_use, replacement_or_upgrade, pm1_col, pm2_col
    )
    
    # Report summary
    pm1_valid = df_copy[pm1_col].notna().sum()
    pm2_valid = df_copy[pm2_col].notna().sum()
    print(f"  Prepared {pm1_valid:,} pm1 values, {pm2_valid:,} pm2 values")
    
    return df_copy


# ==== Prepare REMDB upgrade metrics =====
def add_remdb_upgrade_metrics(
    df: pd.DataFrame,
    remdb_v4_costs: pd.DataFrame,
    end_use: str,
    percentile: str = 'mid'
) -> pd.DataFrame:
    """Prepare upgrade cost metrics from UPGRADE equipment specs for REMDB v4 calculations.
    
    Similar to add_remdb_replacement_metrics but uses upgrade equipment specifications.
    
    Args:
        df: DataFrame with upgrade equipment specifications.
        remdb_v4_costs: REMDB v4 cost database (indexed by row_id).
        end_use: Equipment category ('heating', 'cooling').
        percentile: Cost percentile ('low', 'mid', 'high').
        
    Returns:
        DataFrame with columns ready for cost calculation.
    """
    df_copy = df.copy()
    replacement_or_upgrade = 'upgrade'
    
    # Validate inputs
    if not isinstance(df_copy, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df_copy).__name__}")
    
    # Valid categories are defined in EQUIPMENT_SPECS
    valid_categories = list(EQUIPMENT_SPECS.keys())

    if end_use not in valid_categories:
        raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of {valid_categories}")
    
    if percentile not in ['low', 'mid', 'high']:
        raise ValueError(f"Invalid percentile: '{percentile}'")
    
    print(f"\nPreparing {end_use} UPGRADE metrics (REMDB v4)")
    
    # ===== STEP 1: Assign row_id based on equipment type =====
    df_copy = _assign_upgrade_row_id(df_copy, end_use)
    row_id_col = f'row_id_{end_use}_{replacement_or_upgrade}'
    unknown_count = (df_copy[row_id_col] == 'unknown').sum()
    if unknown_count > 0:
        print(f"  Warning: {unknown_count:,} homes with unknown row_id")
        
    # ===== STEP 2: Map REMDB parameters =====
    df_copy = _map_remdb_parameters(df_copy, remdb_v4_costs, end_use, replacement_or_upgrade, percentile)
    
    # ===== STEP 3: Extract metrics based on REMDB unit specifications =====
    # Determine source columns
    if end_use == 'heating':
        capacity_col = 'size_heating_system_primary_k_btu_h'
        efficiency_col = 'upgrade_hvac_heating_efficiency'
    elif end_use == 'cooling':
        capacity_col = 'size_cooling_system_primary_k_btu_h'
        efficiency_col = 'upgrade_hvac_cooling_efficiency'
    
    # Validate source columns
    if capacity_col not in df_copy.columns:
        raise KeyError(f"Missing capacity column: '{capacity_col}'")
    if efficiency_col not in df_copy.columns:
        raise KeyError(f"Missing efficiency column: '{efficiency_col}'")
    
    # Convert pm1 based on pm1_unit
    pm1_col = f'euss_{end_use}_upgrade_pm1'
    pm1_unit_col = f'{end_use}_{replacement_or_upgrade}_pm1_unit'
    df_copy[pm1_col] = _convert_capacity_by_unit(df_copy, capacity_col, pm1_unit_col)
    
    # Convert pm2 based on pm2_metric
    pm2_col = f'euss_{end_use}_upgrade_pm2'
    pm2_metric_col = f'{end_use}_{replacement_or_upgrade}_pm2_metric'
    df_copy[pm2_col] = _convert_efficiency_by_metric(df_copy, efficiency_col, pm2_metric_col)
    
    # ===== STEP 4: Fill missing values from REMDB bounds =====
    df_copy = _fill_missing_from_bounds(
        df_copy, remdb_v4_costs, end_use, replacement_or_upgrade, pm1_col, pm2_col
    )
    
    # Report summary
    pm1_valid = df_copy[pm1_col].notna().sum()
    pm2_valid = df_copy[pm2_col].notna().sum()
    print(f"  Prepared {pm1_valid:,} pm1 values, {pm2_valid:,} pm2 values")
    
    return df_copy
