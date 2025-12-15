import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional

from cmu_tare_model.utils.validation_framework import (
    apply_new_columns_to_dataframe,
    apply_final_masking,
    initialize_validation_tracking,
    create_retrofit_only_series
    )
from cmu_tare_model.utils.calculation_utils import (
    filter_valid_tech_homes
    )

"""
========================================================================================================================================================================
OVERVIEW: CALCULATE INSTALLATION COSTS FOR VARIOUS END USES
========================================================================================================================================================================
This module calculates the installation costs for various end uses such as heating, cooling, water heating,
clothes drying, and cooking. It uses REMDB v4 regression methodology to calculate costs based on 
performance metrics (capacity, efficiency) rather than probabilistic sampling from cost distributions.

Key changes from REMDB v3 to v4:
- Regression-based calculation: Material_Price = (pm1_coef × Metric1) + (pm2_coef × Metric2) + Intercept
- Costs already in 2023$ (no CPI adjustment needed)
- Installation costs via multipliers OR adders (component-specific)
- Added cooling as new end-use category
- Dynamic row_id mapping replaces hardcoded technology-efficiency pairs

# UPDATED MARCH 24, 2025 @ 4:30 PM - REMOVED RSMEANS CCI ADJUSTMENTS
# UPDATED APRIL 9, 2025 @ 7:30 PM - IMPROVED DOCUMENTATION
# UPDATED APRIL 21, 2025 @ 11:45 PM - COST UTILITY FUNCTION REPLACED REDUNDANT CODE (SEE UTILS FOLDER)
# UPDATED DECEMBER 2, 2025 @ 5:00 PM - UPDATED TO REMDB V4 METHODOLOGY
"""

# ========================================================================================================================================================================
# FUNCTIONS: HELPER FUNCTIONS FOR SPACE HEATING AND COOLING
# ========================================================================================================================================================================


def obtain_heating_system_specs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and process heating system specifications from input dataframe.
    
    Calculates total heating load and extracts efficiency metrics from raw data.
    Updated for REMDB v4 to extract both baseline and upgrade efficiency values.
    
    Args:
        df: Input dataframe containing heating system data
        
    Returns:
        Updated dataframe with calculated heating system specs
        
    Raises:
        TypeError: If df is not a pandas DataFrame
        KeyError: If required columns are missing
        ValueError: If heating load cannot be calculated or efficiency extraction fails
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
    
    # Check for required efficiency columns
    required_columns = ['hvac_heating_efficiency', 'upgrade_hvac_heating_efficiency']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Required columns missing: {missing_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )
    
    # Calculate total heating load if columns exist
    heating_load_columns = [
        'size_heating_system_primary_k_btu_h',
        'size_heat_pump_backup_primary_k_btu_h',
        'size_heating_system_secondary_k_btu_h'
    ]
    
    available_load_columns = [col for col in heating_load_columns if col in df.columns]
    
    if len(available_load_columns) == 0:
        raise ValueError(
            f"Cannot calculate heating load: none of the required columns found.\n"
            f"Required: {heating_load_columns}\n"
            f"Available: {df.columns.tolist()}"
        )
    elif len(available_load_columns) < len(heating_load_columns):
        missing = set(heating_load_columns) - set(available_load_columns)
        warnings.warn(
            f"Some heating load columns missing: {missing}. "
            f"Calculation will use only available columns: {available_load_columns}"
        )
    
    # Sum available heating load columns
    df['total_heating_load_kBtuh'] = sum(
        df[col].fillna(0) for col in available_load_columns
    )
    
    # Validate that we have non-zero loads
    if (df['total_heating_load_kBtuh'] == 0).all():
        raise ValueError("All heating loads are zero - cannot calculate costs")
    
    # Extract baseline efficiency values using regex
    # SEER from pattern "SEER XX" or "SEER XX.X"
    df['baseline_SEER'] = df['hvac_heating_efficiency'].str.extract(
        r'SEER (\d+\.?\d*)', expand=False
    ).astype(float)
    
    # HSPF from pattern "XX HSPF" or "XX.X HSPF"
    df['baseline_HSPF'] = df['hvac_heating_efficiency'].str.extract(
        r'(\d+\.?\d*) HSPF', expand=False
    ).astype(float)
    
    # AFUE from pattern "XX% AFUE" or "XX% Efficiency" (for electric baseboard)
    df['baseline_AFUE'] = df['hvac_heating_efficiency'].str.extract(
        r'(\d+\.?\d*)% (?:AFUE|Efficiency)', expand=False
    ).astype(float)
    
    # Extract upgrade efficiency values
    df['upgrade_SEER'] = df['upgrade_hvac_heating_efficiency'].str.extract(
        r'SEER (\d+\.?\d*)', expand=False
    ).astype(float)
    
    df['upgrade_HSPF'] = df['upgrade_hvac_heating_efficiency'].str.extract(
        r'(\d+\.?\d*) HSPF', expand=False
    ).astype(float)
    
    # Validate that efficiency extraction worked
    if df['upgrade_SEER'].isna().all() and df['upgrade_HSPF'].isna().all():
        sample_data = df['upgrade_hvac_heating_efficiency'].head(3).tolist()
        raise ValueError(
            f"Failed to extract upgrade efficiency values.\n"
            f"Sample data: {sample_data}\n"
            f"Expected patterns: 'SEER XX.X' or 'XX.X HSPF'"
        )
    
    return df


def obtain_cooling_system_specs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and process cooling system specifications from input dataframe.
    
    Calculates total cooling load and extracts efficiency metrics for cooling equipment.
    Includes smart fallback to heating load if cooling-specific load is unavailable.
    
    Args:
        df: Input dataframe containing cooling system data
        
    Returns:
        Updated dataframe with calculated cooling system specs
        
    Raises:
        TypeError: If df is not a pandas DataFrame
        ValueError: If cooling load cannot be calculated or efficiency extraction fails
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
    
    # Calculate total cooling load with smart fallback
    if 'size_cooling_system_primary_k_btu_h' in df.columns:
        df['total_cooling_load_kBtuh'] = df['size_cooling_system_primary_k_btu_h'].fillna(0)
    elif 'total_heating_load_kBtuh' in df.columns:
        warnings.warn(
            "Cooling load column not found. Using total_heating_load_kBtuh as fallback. "
            "This is common for heat pumps that serve both heating and cooling."
        )
        df['total_cooling_load_kBtuh'] = df['total_heating_load_kBtuh']
    else:
        raise ValueError(
            "Cannot calculate cooling load: neither 'size_cooling_system_primary_k_btu_h' "
            "nor 'total_heating_load_kBtuh' found in DataFrame"
        )
    
    # Extract baseline cooling efficiency if column exists
    if 'hvac_cooling_efficiency' in df.columns:
        # SEER from pattern "SEER XX" or "AC, SEER XX"
        df['baseline_cooling_SEER'] = df['hvac_cooling_efficiency'].str.extract(
            r'SEER (\d+\.?\d*)', expand=False
        ).astype(float)
        
        # EER for room AC units
        df['baseline_cooling_EER'] = df['hvac_cooling_efficiency'].str.extract(
            r'EER (\d+\.?\d*)', expand=False
        ).astype(float)
    
    # Extract upgrade cooling efficiency with smart fallback
    if 'upgrade_hvac_cooling_efficiency' in df.columns:
        df['upgrade_cooling_SEER'] = df['upgrade_hvac_cooling_efficiency'].str.extract(
            r'SEER (\d+\.?\d*)', expand=False
        ).astype(float)
    elif 'upgrade_SEER' in df.columns:
        warnings.warn(
            "upgrade_hvac_cooling_efficiency column not found. "
            "Using upgrade_SEER from heating (common for heat pumps)."
        )
        df['upgrade_cooling_SEER'] = df['upgrade_SEER']
    else:
        raise ValueError(
            "Cannot extract upgrade cooling efficiency: neither "
            "'upgrade_hvac_cooling_efficiency' nor 'upgrade_SEER' found"
        )
    
    return df


def kbtuh_to_tons(kbtuh: np.ndarray) -> np.ndarray:
    """
    Convert heating/cooling capacity from kBtu/h to tons.
    
    Uses standard conversion: 1 ton = 12 kBtu/h = 12,000 Btu/h
    
    Args:
        kbtuh: Array of capacities in kBtu/h
        
    Returns:
        Array of capacities in tons
    """
    return kbtuh / 12.0


# ========================================================================================================================================================================
# FUNCTIONS: REMDB V4 COST CALCULATION HELPERS
# ========================================================================================================================================================================


def get_end_use_installation_parameters(
        df: pd.DataFrame,
        end_use: str,
        menu_mp: int) -> dict:
    """
    Get REMDB v4 row_id mapping and metric extraction logic for end-use.
    
    Returns dictionary with row_id mapping conditions and metric column names,
    replacing v3's hardcoded technology-efficiency pairs with dynamic mapping
    to REMDB v4 component identifiers.
    
    Args:
        df: DataFrame containing home equipment specifications
        end_use: Equipment category ('heating', 'cooling', 'waterHeating', 
                'clothesDrying', 'cooking')
        menu_mp: Measure package identifier (7, 8, 9, 10)
        
    Returns:
        dict with keys:
            - 'row_id_conditions': List of boolean conditions for np.select()
            - 'row_id_choices': List of REMDB v4 row_ids corresponding to conditions
            - 'metric1_column': Column name for primary performance metric (or None)
            - 'metric2_column': Column name for secondary metric (or None)
            - 'capacity_column': Column name for capacity (if applicable)
    
    Raises:
        ValueError: If invalid end_use specified
    """
    if end_use == 'heating':
        return {
            'row_id_conditions': [
                (df['hvac_has_ducts'] == 'Yes') & (menu_mp == 7),
                (df['hvac_has_ducts'] == 'No') & (menu_mp == 7),
                (df['hvac_has_ducts'] == 'Yes') & (menu_mp != 7),
                (df['hvac_has_ducts'] == 'No') & (menu_mp != 7)
            ],
            'row_id_choices': [
                'air_source_heat_pump_centrally_ducted',
                'air_source_heat_pump_non_ducted_single_zone',
                'air_source_heat_pump_centrally_ducted_with_new_circuit',
                'air_source_heat_pump_non_ducted_multi_zone'
            ],
            'capacity_column': 'total_heating_load_kBtuh',
            'metric1_column': 'total_heating_load_kBtuh',  # Will be converted to tons
            'metric2_column': 'upgrade_SEER'
        }
    
    elif end_use == 'cooling':
        return {
            'row_id_conditions': [
                (df['hvac_has_ducts'] == 'Yes') & (menu_mp == 7),
                (df['hvac_has_ducts'] == 'No') & (menu_mp == 7),
                (df['hvac_has_ducts'] == 'Yes') & (menu_mp != 7),
                (df['hvac_has_ducts'] == 'No') & (menu_mp != 7)
            ],
            'row_id_choices': [
                'air_source_heat_pump_centrally_ducted',
                'air_source_heat_pump_non_ducted_single_zone',
                'air_source_heat_pump_centrally_ducted_with_new_circuit',
                'air_source_heat_pump_non_ducted_multi_zone'
            ],
            'capacity_column': 'total_cooling_load_kBtuh',
            'metric1_column': 'total_cooling_load_kBtuh',  # Will be converted to tons
            'metric2_column': 'upgrade_cooling_SEER'
        }
    
    elif end_use == 'waterHeating':
        return {
            'row_id_conditions': [
                df['upgrade_water_heater_efficiency'].str.contains('Electric Heat Pump', na=False)
            ],
            'row_id_choices': ['water_heater_hp_tank'],
            'capacity_column': 'size_water_heater_gal',
            'metric1_column': None,  # UEF - extract from string
            'metric2_column': 'size_water_heater_gal'
        }
    
    elif end_use == 'clothesDrying':
        return {
            'row_id_conditions': [
                df['upgrade_clothes_dryer'].str.contains('Heat Pump', na=False),
                ~df['upgrade_clothes_dryer'].str.contains('Heat Pump', na=False)
            ],
            'row_id_choices': ['clothes_dryer_heat_pump', 'clothes_dryer_electric'],
            'capacity_column': None,  # No capacity for dryers
            'metric1_column': None,  # Volume - use default 7.0 cu ft
            'metric2_column': None   # CEF - infer from upgrade type
        }
    
    elif end_use == 'cooking':
        return {
            'row_id_conditions': [
                df['upgrade_cooking_range'].str.contains('Induction', na=False),
                ~df['upgrade_cooking_range'].str.contains('Induction', na=False)
            ],
            'row_id_choices': ['cooking_range_induction', 'cooking_range_electric'],
            'capacity_column': None,
            'metric1_column': None,  # Volume - use default 5.0 cu ft
            'metric2_column': None   # Not used in REMDB v4 cooking
        }
    
    else:
        raise ValueError(f"Invalid end_use: {end_use}. Must be one of: heating, cooling, waterHeating, clothesDrying, cooking")


def calculate_v4_costs(
        row_ids: np.ndarray,
        metric1_values: np.ndarray,
        metric2_values: Optional[np.ndarray],
        df_tare_remdb_data: pd.DataFrame,
        percentile: str = 'mid') -> np.ndarray:
    """
    Calculate costs using REMDB v4 regression methodology.
    
    Replaces v3's probabilistic sampling with deterministic regression:
    Material_Price = (pm1_coef × Metric1) + (pm2_coef × Metric2) + Intercept
    Installed_Cost = Material_Price × Multiplier (or + Adder)
    
    Args:
        row_ids: REMDB v4 row_id for each home (e.g., 'air_source_heat_pump_centrally_ducted')
        metric1_values: Primary performance metric values (e.g., capacity in tons)
        metric2_values: Secondary performance metric values (e.g., SEER) or None
        df_tare_remdb_data: REMDB v4 cost data DataFrame with row_id as index
        percentile: Cost percentile ('low', 'mid', 'high'). Default 'mid'.
        
    Returns:
        Array of installed costs in 2023$ for each home
        
    Raises:
        KeyError: If row_id not found in df_tare_remdb_data
        ValueError: If required coefficient columns missing
    """
    n = len(row_ids)
    costs = np.zeros(n)
    
    # Get coefficient column names based on percentile
    pm1_coef_col = f'pm1_coef_{percentile}'
    pm2_coef_col = f'pm2_coef_{percentile}'
    intercept_col = f'intercept_{percentile}'
    
    # Validate required columns exist
    required_cols = [pm1_coef_col, pm2_coef_col, intercept_col, 
                     'install_mult_retrofit', 'install_add_retrofit']
    missing_cols = [col for col in required_cols if col not in df_tare_remdb_data.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns missing from df_tare_remdb_data: {missing_cols}\n"
            f"Available columns: {df_tare_remdb_data.columns.tolist()}"
        )
    
    # Calculate cost for each unique row_id
    for row_id in np.unique(row_ids):
        if row_id == 'unknown':
            continue  # Skip unmapped homes (will remain zero)
        
        # Get homes with this row_id
        mask = (row_ids == row_id)
        
        # Get REMDB coefficients for this component
        try:
            remdb_row = df_tare_remdb_data.loc[row_id]
        except KeyError:
            warnings.warn(f"row_id '{row_id}' not found in REMDB data. Homes will have zero cost.")
            continue
        
        pm1_coef = remdb_row[coef1_col]
        pm2_coef = remdb_row[coef2_col] if metric2_values is not None else 0
        intercept = remdb_row[intercept_col]
        
        # Calculate material price using regression
        material_price = pm1_coef * metric1_values[mask]
        if metric2_values is not None:
            material_price += pm2_coef * metric2_values[mask]
        material_price += intercept
        
        # Apply installation multiplier or adder
        install_mult = remdb_row['install_mult_retrofit']
        install_add = remdb_row['install_add_retrofit']
        
        if install_mult != 1.0:
            # Use multiplier method (typical for equipment)
            installed_cost = material_price * install_mult
        else:
            # Use adder method (typical for insulation, enclosure)
            installed_cost = material_price + install_add
        
        costs[mask] = installed_cost
    
    return costs


def calculate_installation_cost_per_row(
        df_valid: pd.DataFrame,
        params: dict,
        end_use: str,
        menu_mp: int) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """
    Extract metric values from DataFrame for v4 cost calculation.
    
    Replaces v3's component-based cost formula with metric extraction for
    regression-based cost calculation.
    
    Args:
        df_valid: Filtered DataFrame with valid homes only
        params: Parameter dict from get_end_use_installation_parameters()
        end_use: Equipment category
        menu_mp: Measure package identifier (for column naming)
        
    Returns:
        Tuple of (metric1_values, metric2_values, cost_column_name)
        metric2_values is None if not used for this end_use
        
    Raises:
        KeyError: If required columns missing from df_valid
    """
    n = len(df_valid)
    
    # Extract metric1 (primary performance metric)
    if params['metric1_column'] is not None:
        # Get from DataFrame column
        if params['metric1_column'] not in df_valid.columns:
            raise KeyError(
                f"Required column '{params['metric1_column']}' not found in DataFrame. "
                f"Available: {df_valid.columns.tolist()}"
            )
        
        if end_use in ['heating', 'cooling']:
            # Convert kBtu/h to tons (1 ton = 12 kBtu/h)
            metric1_values = kbtuh_to_tons(df_valid[params['metric1_column']].values)
        else:
            metric1_values = df_valid[params['metric1_column']].values
    else:
        # Use defaults for end-uses without explicit metric columns
        if end_use == 'clothesDrying':
            metric1_values = np.full(n, 7.0)  # Standard 7 cu ft dryer
        elif end_use == 'cooking':
            metric1_values = np.full(n, 5.0)  # Standard 5 cu ft oven
        elif end_use == 'waterHeating':
            # Extract UEF from string: "Electric Heat Pump, 50 gal, 3.45 UEF"
            uef = df_valid['upgrade_water_heater_efficiency'].str.extract(
                r'(\d+\.?\d*)\s+UEF', expand=False
            ).astype(float)
            metric1_values = uef.fillna(3.45).values  # Default UEF = 3.45
        else:
            metric1_values = np.zeros(n)
    
    # Extract metric2 (secondary performance metric)
    if params['metric2_column'] is not None:
        if params['metric2_column'] not in df_valid.columns:
            raise KeyError(
                f"Required column '{params['metric2_column']}' not found in DataFrame. "
                f"Available: {df_valid.columns.tolist()}"
            )
        metric2_values = df_valid[params['metric2_column']].values
    elif end_use == 'clothesDrying':
        # CEF: 5.2 for heat pump, 2.7 for standard electric
        is_hp = df_valid['upgrade_clothes_dryer'].str.contains('Heat Pump', na=False)
        metric2_values = np.where(is_hp, 5.2, 2.7)
    else:
        metric2_values = None
    
    # Generate cost column name
    cost_column_name = f'mp{menu_mp}_{end_use}_installationCost'
    
    return metric1_values, metric2_values, cost_column_name


# ========================================================================================================================================================================
# FUNCTIONS: CALCULATE COST OF INSTALLING NEW EQUIPMENT (RETROFIT/UPGRADES)
# ========================================================================================================================================================================


def calculate_installation_cost(
        df: pd.DataFrame,
        df_tare_remdb_data: pd.DataFrame,
        menu_mp: int,
        end_use: str,
        percentile: str = 'mid') -> pd.DataFrame:
    """
    Calculate installation costs for various end-uses using REMDB v4 methodology.

    This function uses regression-based cost calculation with performance metrics:
    Material_Price = (Coef1 × Metric1) + (pm2_coef × Metric2) + Intercept
    Installed_Cost = Material_Price × Multiplier (or + Adder)
    
    Costs are already in 2023$ (no CPI adjustment needed).

    Args:
        df: DataFrame containing data for different scenarios
        df_tare_remdb_data: REMDB v4 cost data (from load_remdb_v4_data())
        menu_mp: Menu option identifier (valid values: 7, 8, 9, 10)
        end_use: Type of end-use ('heating', 'cooling', 'waterHeating', 'clothesDrying', 'cooking')
        percentile: Cost percentile ('low', 'mid', 'high'). Default 'mid'.

    Returns:
        pd.DataFrame: Updated DataFrame with calculated installation costs

    Raises:
        ValueError: If menu_mp is not valid or if cost data is missing
        RuntimeError: If an unexpected error occurs during calculation
        
    Notes:
        This function implements the validation framework:
        1. Uses initialize_validation_tracking() to determine valid homes
        2. Creates retrofit-only series with NaN for invalid homes
        3. Calculates values only for valid homes with identifiable technology
        4. Applies final verification masking
    """
    # Add logging for calculation start
    print(f"Starting {end_use} installation cost calculation with REMDB v4 methodology")

    # Initialize validation tracking
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        df, end_use, menu_mp, verbose=True)
    
    print(f"Found {valid_mask.sum()} valid homes out of {len(df_copy)} for {end_use} installation")

    # Validate menu_mp 
    valid_menu_mps = [7, 8, 9, 10]
    if menu_mp not in valid_menu_mps:
        raise ValueError("Please enter a valid measure package number for menu_mp. Should be 7, 8, 9, or 10.")
    
    # Get row_id mapping and metric column names
    params = get_end_use_installation_parameters(df_copy, end_use, menu_mp)
    
    # Map homes to REMDB v4 row_ids
    row_ids = np.select(
        params['row_id_conditions'],
        params['row_id_choices'],
        default='unknown'
    )

    try:
        # Use the standard filtering function to get only homes with both valid data and identifiable tech
        # Note: For v4, we use row_ids for both tech and eff parameters
        df_valid, valid_calculation_indices, row_ids_filtered, _ = filter_valid_tech_homes(
            df_copy, valid_mask, row_ids, row_ids)
        
        print(f"After tech filtering: {len(valid_calculation_indices)} homes remain valid for {end_use} installation")

        if df_valid.empty:
            print(f"Warning: No valid homes found for {end_use} installation cost calculation.")
            cost_column_name = f'mp{menu_mp}_{end_use}_installationCost'
        else:
            # Extract performance metrics from DataFrame
            metric1, metric2, cost_column_name = calculate_installation_cost_per_row(
                df_valid, params, end_use, menu_mp)
            
            # Calculate costs using v4 regression methodology
            installation_cost = calculate_v4_costs(
                row_ids_filtered, metric1, metric2, df_tare_remdb_data, percentile)
        
        # Initialize the result series properly
        result_series = create_retrofit_only_series(df_copy, valid_mask)
        
        # Update only for homes that have valid data AND match our tech criteria
        if not df_valid.empty:
            result_series.loc[valid_calculation_indices] = np.round(installation_cost, 2)
            
    except Exception as e:
        raise RuntimeError(f"Error in {end_use} installation cost calculation: {str(e)}")

    # Create the DataFrame column
    df_new_columns = pd.DataFrame({cost_column_name: result_series})    

    # Apply new columns to DataFrame with proper tracking
    df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
        df_copy, df_new_columns, end_use, category_columns_to_mask, all_columns_to_mask)
    
    # Apply final verification masking for consistency
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=True)

    return df_copy


# ============================================================================
# REMOVED: calculate_heating_installation_premium()
# This v3 function is no longer needed in v4 methodology.
# Installation complexity is now handled through REMDB v4 component classes
# (e.g., "Centrally ducted, with new circuit")
# Removed: December 2, 2025
# ============================================================================


# ============================================================================
# REMOVED: get_end_use_installation_parameters()
# This v3 function used hardcoded technology-efficiency pairs.
# V4 uses get_installation_row_id() with dynamic DataFrame-based mapping.
# Removed: December 2, 2025
# ============================================================================


# ============================================================================
# REMOVED: calculate_installation_cost_per_row()
# This v3 function used component-based cost formulas.
# V4 uses regression-based calculation in calculate_material_price_v4().
# Removed: December 2, 2025
# ============================================================================