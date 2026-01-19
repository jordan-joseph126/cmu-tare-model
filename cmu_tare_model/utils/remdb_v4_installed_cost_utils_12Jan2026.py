"""
========================================================================================================================================================================
REMDB v4 Installed Cost Utilities (REFACTORED)
========================================================================================================================================================================

This module prepares equipment metrics for REMDB v4 cost calculations.

Key Functions:
- add_remdb_metrics(): Unified function for both replacement and upgrade metrics

Features:
- Percentile-based filtering to exclude outliers before processing
- Unit conversion based on REMDB specifications
- Automatic bounds checking with configurable handling

Refactored: January 2026
- Combined duplicate replacement/upgrade logic into single function
- Added percentile filtering for capacity values
- Removed dead/commented code
- Simplified column management

NO LONGER USING THE SUM OF THE HEATING AND COOLING LOADS FOR SYSTEM SIZE AND COST ESTIMATION
- The supplemental heating (electric strip heat) is implicitly included in the REMDB v4 costs
- Also, the primary system size is the same for both heating and cooling. You wouldnt have two different ASHP tonnages.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Literal

from cmu_tare_model.constants import EQUIPMENT_SPECS


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

OutOfBoundMethod = Literal["masking", "keep_as_is", "keep_as_is_filter_ci"]
MetricType = Literal["replacement", "upgrade"]


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
    
    # Cooling only considered as replacement cost because heat pumps are the upgrade option for both heating and cooling

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
        'pm1_lower_bound', 'pm1_upper_bound', 
        'pm2_metric', 'pm2_unit', f'pm2_coef_{percentile}',
        'pm2_lower_bound', 'pm2_upper_bound',  
        f'intercept_{percentile}', 'multiplier_retrofit', 'adder_retrofit'
    ]
    
    for param in params:
        if param in remdb_v4_costs.columns:
            df_copy[prefix + param] = df_copy[row_id_col].map(remdb_v4_costs[param])
    
    return df_copy


def _convert_pm1(
    df: pd.DataFrame,
    pm1_euss_value_col: str,
    pm1_metric_col: str,
    pm1_unit_col: str
) -> pd.Series:
    """
    Convert PM1 (Performance Metric 1) to REMDB-expected units.
    
    Handles different metric types based on pm1_metric and target pm1_unit:
    - Cooling/Heating Capacity: kBtu/h → Tons or BTU/hr
    - Leakage Reduction: % → decimal or keep as %
    - R-value: Keep as-is
    - Volume: Keep as-is
    - etc.
    
    Args:
        df: DataFrame
        pm1_euss_value_col: Column with PM1 values from the End Use Saving Shapes (EUSS) (e.g., capacity in kBtu/h)
        pm1_metric_col: Column with metric name from REMDB (e.g., "Cooling Capacity")
        pm1_unit_col: Column with target unit from REMDB (e.g., "Tons", "BTU/hr")
        
    Returns:
        Series with PM1 converted to REMDB units
    """
    df_copy = df.copy()
    
    pm1_euss_value = df_copy[pm1_euss_value_col].copy()
    pm1_unit = df_copy[pm1_unit_col].copy()
    
    # Normalize units (case-insensitive, no whitespace)
    pm1_unit = pm1_unit.str.lower().str.strip()
    
    result = pd.Series(np.nan, index=df_copy.index)
    
    # CONVERSION 1: Tons (heat pumps, ACs)
    # Input: kBtu/h → Output: Tons (÷12)
    mask_tons = (pm1_unit == 'tons')
    if mask_tons.any():
        result.loc[mask_tons] = pm1_euss_value.loc[mask_tons] / 12.0
    
    # CONVERSION 2: BTU/hr (furnaces, boilers, baseboard)
    # Input: kBtu/h → Output: BTU/hr (×1000)
    mask_btu = pm1_unit.str.contains('btu', na=False)
    if mask_btu.any():
        result.loc[mask_btu] = pm1_euss_value.loc[mask_btu] * 1000.0
        
    # Everything else (R-value, volume, etc.): keep as-is
    mask_other = ~mask_tons & ~mask_btu & pm1_unit.notna()
    if mask_other.any():
        result.loc[mask_other] = pm1_euss_value.loc[mask_other]
    
    return result


def _convert_pm2(
    df: pd.DataFrame,
    pm2_euss_value_col: str,
    pm2_metric_col: str,
    pm2_unit_col: str
) -> pd.Series:
    """
    Convert PM2 (Performance Metric 2) to REMDB-expected format.
    
    Handles different metric types based on pm2_metric:
    - AFUE: % → decimal (÷100)
    - SEER, SEER1, SEER2: Extract numeric, keep as-is
    - HSPF, HSPF2: Extract numeric, keep as-is
    - EER, CEER: Extract numeric, keep as-is
    - UEF: Extract numeric, keep as-is
    - Combined Energy Factor: Extract numeric, keep as-is
    
    Args:
        df: DataFrame
        pm2_euss_value_col: Column with PM2 values from the End Use Saving Shapes (EUSS) (e.g., "80% AFUE", "SEER 15")
        pm2_metric_col: Column with metric name from REMDB (e.g., "AFUE", "SEER1")
        pm2_unit_col: Column with unit from REMDB (e.g., "Unitless")
        
    Returns:
        Series with PM2 in REMDB format
    """
    df_copy = df.copy()
    pm2_euss_value = df_copy[pm2_euss_value_col].astype(str)

    # Normalize metric names (case-insensitive)
    pm2_metric = df_copy[pm2_metric_col].str.lower().str.strip()
    
    # Extract numeric value from strings like "SEER 15", "80% AFUE"
    numeric_value = pm2_euss_value.str.extract(r'([\d.]+)', expand=False).astype(float)
        
    result = pd.Series(0.0, index=df_copy.index)
    
    # CONVERSION 1: AFUE (Annual Fuel Utilization Efficiency)
    # Input: "80% AFUE" → Extract: 80 → Output: 0.80 (÷100)
    mask_afue = (pm2_metric == 'afue')
    if mask_afue.any():
        result.loc[mask_afue] = numeric_value.loc[mask_afue] / 100.0
    
    # CONVERSION 2: All other metrics (SEER, HSPF, EER, UEF, CEF, etc.)
    # Input: "SEER 15" → Extract: 15 → Output: 15 (no conversion)
    mask_other = (~mask_afue) & pm2_metric.notna()
    if mask_other.any():
        result.loc[mask_other] = numeric_value.loc[mask_other]
    
    return result


def _check_out_of_bounds(
    df: pd.DataFrame,
    metric_col: str,
    lower_col: str,
    upper_col: str,
    metric_name: str,
    row_id_col: str,
    method: OutOfBoundMethod = "keep_as_is",
    verbose: bool = True
) -> Tuple[pd.Series, pd.Series]:
    """Check for out-of-bounds metrics and optionally mask them.
    
    Returns:
        Tuple of (processed_metric_series, out_of_bounds_mask)
    """
    df_copy = df.copy()
    
    metric = df_copy[metric_col]
    lower = df_copy[lower_col]
    upper = df_copy[upper_col]
    
    checkable = metric.notna() & lower.notna() & upper.notna()
    out_of_bounds = checkable & ((metric < lower) | (metric > upper))
    
    if verbose and out_of_bounds.any():
        total = out_of_bounds.sum()
        print(f"\n  WARNING: {total:,} homes have out-of-bounds {metric_name} (method={method})")
        
        # Summary by technology
        for row_id in df_copy.loc[out_of_bounds, row_id_col].value_counts().head(5).index:
            row_id_out_of_bounds = out_of_bounds & (df_copy[row_id_col] == row_id)
            vals = df_copy.loc[row_id_out_of_bounds, metric_col]
            bounds = f"{lower[row_id_out_of_bounds].iloc[0]:.2f}-{upper[row_id_out_of_bounds].iloc[0]:.2f}"
            print(f"    • {row_id}: {row_id_out_of_bounds.sum():,} out_of_bounds (bounds: {bounds}, actual: {vals.min():.2f}-{vals.max():.2f})")
    
    if method == "masking":
        metric.loc[out_of_bounds] = np.nan
    
    return metric, out_of_bounds


# ===== Fill missing pm1/pm2 values from REMDB bounds =====
def _fill_missing_from_bounds(
    df: pd.DataFrame,
    remdb_v4_costs: pd.DataFrame,
    row_id_col: str,
    pm1_col: str,
    pm2_col: str,
    verbose: bool = True
) -> pd.DataFrame:
    """Fill missing pm1/pm2 values from REMDB bounds."""
    df_copy = df.copy()
    
    # Fill pm1 from bounds
    pm1_missing = df_copy[pm1_col].isna()
    if pm1_missing.any() and 'pm1_lower_bound' in remdb_v4_costs.columns:
        lower = df_copy.loc[pm1_missing, row_id_col].map(remdb_v4_costs['pm1_lower_bound'])
        upper = df_copy.loc[pm1_missing, row_id_col].map(remdb_v4_costs['pm1_upper_bound'])
        df_copy.loc[pm1_missing, pm1_col] = (lower + upper) / 2.0
        
        if verbose:
            print(f"  Filled {pm1_missing.sum():,} missing {pm1_col} from REMDB bounds")
    
    # Fill pm2 from bounds  
    pm2_missing = df_copy[pm2_col].isna()
    if pm2_missing.any() and 'pm2_lower_bound' in remdb_v4_costs.columns:
        lower = df_copy.loc[pm2_missing, row_id_col].map(remdb_v4_costs['pm2_lower_bound'])
        upper = df_copy.loc[pm2_missing, row_id_col].map(remdb_v4_costs['pm2_upper_bound'])
        df_copy.loc[pm2_missing, pm2_col] = (lower + upper) / 2.0
        
        if verbose:
            print(f"  Filled {pm2_missing.sum():,} missing {pm2_col} from REMDB bounds")
    
    return df_copy


# =============================================================================
# PERCENTILE FILTERING
# =============================================================================

def filter_by_percentile(
    df: pd.DataFrame,
    column: str,
    lower_percentile: Optional[float] = None,
    upper_percentile: Optional[float] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Filter DataFrame to keep only values within specified percentile range.
    
    Args:
        df: Input DataFrame
        column: Column name to filter on
        lower_percentile: Lower percentile bound (0-100), or None for no lower bound
        upper_percentile: Upper percentile bound (0-100), or None for no upper bound
        verbose: Print filtering statistics
        
    Returns:
        df_filtered: DataFrame filtered to specified percentile range
        
    Example:
        # Keep values between 2.5th and 97.5th percentile (95% CI)
        df_filtered = filter_by_percentile(df, 'capacity', 2.5, 97.5)
    """
    df_copy = df.copy()

    if column not in df_copy.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")
    
    n_original = len(df_copy)
    values = df_copy[column]
    
    # Calculate bounds
    lower_bound = values.quantile(lower_percentile / 100) if lower_percentile is not None else values.min()
    upper_bound = values.quantile(upper_percentile / 100) if upper_percentile is not None else values.max()
    
    # Apply filter
    mask = (values >= lower_bound) & (values <= upper_bound)
    df_filtered = df_copy[mask]
    
    n_filtered = len(df_filtered)
    n_removed = n_original - n_filtered
    
    stats = {
        'n_original': n_original,
        'n_filtered': n_filtered,
        'n_removed': n_removed,
        'pct_removed': 100 * n_removed / n_original if n_original > 0 else 0,
        'lower_percentile': lower_percentile,
        'upper_percentile': upper_percentile,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'original_min': values.min(),
        'original_max': values.max(),
        'filtered_min': df_filtered[column].min() if n_filtered > 0 else np.nan,
        'filtered_max': df_filtered[column].max() if n_filtered > 0 else np.nan,
    }
    
    if verbose:
        print(f"\n📊 Percentile Filtering on '{column}':")
        print(f"   Percentile range: {lower_percentile or 0:.1f}% - {upper_percentile or 100:.1f}%")
        print(f"   Value bounds: {lower_bound:.2f} - {upper_bound:.2f}")
        print(f"   Original range: {stats['original_min']:.2f} - {stats['original_max']:.2f}")
        print(f"   Rows: {n_original:,} → {n_filtered:,} (removed {n_removed:,}, {stats['pct_removed']:.2f}%)")
    
    return df_filtered


# =============================================================================
# MAIN FUNCTION - UNIFIED METRICS PREPARATION
# =============================================================================

def add_remdb_metrics(
    df: pd.DataFrame,
    remdb_v4_costs: pd.DataFrame,
    end_use: str,
    metric_type: MetricType,
    percentile: str = 'mid',
    capacity_lower_percentile: Optional[float] = None,
    capacity_upper_percentile: Optional[float] = None,
    out_of_bound_method: OutOfBoundMethod = "keep_as_is",
    verbose: bool = True,
    return_detailed: bool = False
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare REMDB v4 metrics for cost calculations.
    
    This unified function handles both replacement and upgrade metrics with
    optional percentile-based filtering for capacity outliers.
    
    Args:
        df: DataFrame with equipment specifications
        remdb_v4_costs: REMDB v4 cost database (indexed by row_id)
        end_use: Equipment category ('heating' or 'cooling')
        metric_type: 'replacement' (baseline equipment) or 'upgrade' (heat pump)
        percentile: Cost percentile ('low', 'mid', 'high')
        capacity_lower_percentile: Lower percentile for capacity filtering (0-100), or None
        capacity_upper_percentile: Upper percentile for capacity filtering (0-100), or None
        out_of_bound_method: How to handle out-of-bounds values:
            - 'masking': Set to NaN
            - 'keep_as_is': Keep original values (default)
            - 'keep_as_is_filter_ci': Keep values but use mid coefficients for out_of_bounds rows
        verbose: Print progress and statistics
        return_detailed: If True, return (df_main, df_detailed) tuple
        
    Returns:
        DataFrame with REMDB metrics added, or tuple of (main, detailed) DataFrames
        
    Example:
        # Basic usage
        df = add_remdb_metrics(df, remdb_costs, 'heating', 'replacement')
        
        # With percentile filtering (95% CI)
        df = add_remdb_metrics(
            df, remdb_costs, 'heating', 'upgrade',
            capacity_lower_percentile=2.5,
            capacity_upper_percentile=97.5
        )
    """
    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
    
    valid_end_uses = list(EQUIPMENT_SPECS.keys()) + ['cooling']
    if end_use not in valid_end_uses:
        raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of {valid_end_uses}")
    
    if metric_type not in ('replacement', 'upgrade'):
        raise ValueError(f"Invalid metric_type: '{metric_type}'. Must be 'replacement' or 'upgrade'")
    
    if percentile not in ['low', 'mid', 'high']:
        raise ValueError(f"Invalid percentile: '{percentile}'")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Preparing {end_use} {metric_type.upper()} metrics (REMDB v4)")
        print(f"{'='*60}")
    
    df_copy = df.copy()
    prefix = f"{end_use}_{metric_type}_"
    row_id_col = f'row_id_{end_use}_{metric_type}'
    
    # Determine source columns
    if end_use == 'heating':
        capacity_col = 'size_heating_system_primary_k_btu_h'
        efficiency_col = 'upgrade_hvac_heating_efficiency' if metric_type == 'upgrade' else 'hvac_heating_efficiency'
    elif end_use == 'cooling':
        capacity_col = 'size_cooling_system_primary_k_btu_h'
        efficiency_col = 'hvac_cooling_efficiency'
    else:
        raise ValueError(f"Unsupported end_use: {end_use}")
    
    # Validate required columns
    for col in [capacity_col, efficiency_col]:
        if col not in df_copy.columns:
            raise KeyError(f"Missing required column: '{col}'")
    
    # =========================================================================
    # STEP 0: Optional percentile filtering
    # =========================================================================
    if capacity_lower_percentile is not None or capacity_upper_percentile is not None:
        df_copy = filter_by_percentile(
            df_copy, 
            capacity_col,
            capacity_lower_percentile,
            capacity_upper_percentile,
            verbose=verbose
        )
    
    # =========================================================================
    # STEP 1: Assign row_id
    # =========================================================================
    if metric_type == 'replacement':
        df_copy = _assign_replacement_row_id(df_copy, end_use)
    elif metric_type == 'upgrade':
        df_copy = _assign_upgrade_row_id(df_copy, end_use)
    else:
        raise ValueError(f"Invalid metric_type: '{metric_type}'. Must be 'replacement' or 'upgrade'")
        
    unknown_count = (df_copy[row_id_col] == 'unknown').sum()
    if unknown_count > 0 and verbose:
        print(f"  Warning: {unknown_count:,} homes with unknown row_id")
    
    # =========================================================================
    # STEP 2: Map REMDB parameters
    # =========================================================================
    df_copy = _map_remdb_parameters(df_copy, remdb_v4_costs, end_use, metric_type, percentile)
    
    # =========================================================================
    # STEP 3: Convert metrics to REMDB units
    # =========================================================================
    pm1_col = f'{prefix}pm1_euss'
    pm2_col = f'{prefix}pm2_euss'
    
    df_copy[pm1_col] = _convert_pm1(
        df=df_copy, 
        pm1_euss_value_col=capacity_col, 
        pm1_metric_col=f'{prefix}pm1_metric',
        pm1_unit_col=f'{prefix}pm1_unit'
    )
    
    df_copy[pm2_col] = _convert_pm2(
        df=df_copy, 
        pm2_euss_value_col=efficiency_col, 
        pm2_metric_col=f'{prefix}pm2_metric',
        pm2_unit_col=f'{prefix}pm2_unit'
    )
    
    # =========================================================================
    # STEP 4: Fill missing values from REMDB bounds
    # =========================================================================
    df_copy = _fill_missing_from_bounds(
        df_copy, remdb_v4_costs, row_id_col, pm1_col, pm2_col, verbose=verbose
    )
    
    # =========================================================================
    # STEP 5: Check out-of-bounds metrics
    # =========================================================================
    df_copy[pm1_col], out_of_bounds_pm1 = _check_out_of_bounds(
        df_copy, pm1_col,
        f"{prefix}pm1_lower_bound", f"{prefix}pm1_upper_bound",
        "capacity", row_id_col, out_of_bound_method, verbose
    )
    
    df_copy[pm2_col], out_of_bounds_pm2 = _check_out_of_bounds(
        df_copy, pm2_col,
        f"{prefix}pm2_lower_bound", f"{prefix}pm2_upper_bound",
        "efficiency", row_id_col, out_of_bound_method, verbose
    )
    
    # Apply mid coefficients for out_of_bounds rows if requested
    if out_of_bound_method == "keep_as_is_filter_ci" and percentile != "mid":
        out_of_bounds_any = out_of_bounds_pm1 | out_of_bounds_pm2
        if out_of_bounds_any.any():
            for coef in ['pm1_coef', 'pm2_coef', 'intercept']:
                df_copy.loc[out_of_bounds_any, f"{prefix}{coef}_{percentile}"] = \
                    df_copy.loc[out_of_bounds_any, row_id_col].map(remdb_v4_costs[f"{coef}_mid"])
            if verbose:
                print(f"  Applied mid coefficients to {out_of_bounds_any.sum():,} out-of-bounds homes")
    
    # Report summary
    if verbose:
        pm1_valid = df_copy[pm1_col].notna().sum()
        pm2_valid = df_copy[pm2_col].notna().sum()
        print(f"\n  ✓ Valid metrics: {pm1_valid:,} pm1, {pm2_valid:,} pm2")
    
    # =========================================================================
    # STEP 6: Prepare output columns
    # =========================================================================
    summary_cols = [
        row_id_col,
        f'{prefix}pm1_lower_bound', f'{prefix}pm1_upper_bound', pm1_col,
        f'{prefix}pm2_lower_bound', f'{prefix}pm2_upper_bound', pm2_col,
    ]
    
    detailed_cols = summary_cols + [
        f'{prefix}pm1_metric', f'{prefix}pm1_unit', f'{prefix}pm1_coef_{percentile}',
        f'{prefix}pm2_metric', f'{prefix}pm2_unit', f'{prefix}pm2_coef_{percentile}',
        f'{prefix}intercept_{percentile}',
        f'{prefix}multiplier_retrofit', f'{prefix}adder_retrofit',
    ]
    
    # Build output DataFrame
    existing_cols = [c for c in summary_cols if c in df_copy.columns]
    df_main = pd.concat([df.loc[df_copy.index], df_copy[existing_cols]], axis=1)
    
    if return_detailed:
        detailed_existing = [c for c in detailed_cols if c in df_copy.columns]
        df_detailed = df_copy[detailed_existing].copy()
        return df_main, df_detailed
    
    return df_main
