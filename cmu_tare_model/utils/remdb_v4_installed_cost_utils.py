"""REMDB v4 installed-cost metric preparation.

This module turns raw EUSS equipment fields into the performance metrics that
the REMDB v4 cost regression consumes. The single public entry point is
add_remdb_metrics(), which handles both replacement (counterfactual, like-for-
like) and upgrade (heat pump) metrics.

What the pipeline does, in order (see add_remdb_metrics):
  1. Optional percentile filtering of capacity outliers.
  2. Assign a REMDB row_id from the baseline (replacement) or heat-pump
     (upgrade) equipment type.
  3. Map the REMDB regression coefficients and unit specs onto each home.
  4. Convert capacity (pm1) and efficiency (pm2) into the units the regression
     expects.
  5. Replacement only: raise below-floor efficiencies (pm2) up to the minimum
     efficiency equipment sold today, preserving the raw value in a
     ``{pm2_col}_original`` column.
  6. Report diagnostics (including any capacity values outside the REMDB
     training bounds) and return a summary frame plus a detailed frame.

Capacity outliers are reported but never modified: pm1 is used as converted.
Homes far outside the training range are handled by the upstream percentile
filter and by NaN propagation, not by clamping.

NaN handling: homes with invalid fuel/technology types resolve to row_id
'unknown' and carry NaN metrics, which propagate to NaN costs downstream. This
is intentional and matches the validation framework's masking.

System sizing: a single primary system size drives both the heating and cooling
cost; supplemental electric-strip heat is already priced into the REMDB v4
figures, so heating and cooling loads are not summed into one larger system.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Literal

from config import PROJECT_ROOT

from cmu_tare_model.constants import (
    EQUIPMENT_SPECS,
    EFFICIENCY_FLOORS_PM2,
)

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

MetricType = Literal["replacement", "upgrade"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

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
            PROJECT_ROOT, 'cmu_tare_model', 'data', 'retrofit_costs'
        )
    
    file_path = os.path.join(data_dir, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"REMDB v4 cost data not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    df = df.set_index('row_id')
    
    return df


def _assign_replacement_row_id(df: pd.DataFrame, end_use: str) -> pd.DataFrame:
    """Assign REMDB row_id based on baseline equipment type.
    
    Args:
        df: DataFrame with equipment specifications.
        end_use: Equipment category ('heating' or 'cooling').
        
    Returns:
        DataFrame with row_id column added.
    """
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


def _assign_upgrade_row_id(df: pd.DataFrame, end_use: str) -> pd.DataFrame:
    """Assign REMDB row_id for upgrade equipment (heat pumps).
    
    Args:
        df: DataFrame with equipment specifications.
        end_use: Equipment category ('heating' or 'cooling').
        
    Returns:
        DataFrame with row_id column added.
    """
    df_copy = df.copy()
    row_id_col = f'row_id_{end_use}_upgrade'

    # ========== HVAC OPTIONS: HEATING & COOLING ---> HEAT PUMP ==========
    # MP3/MP7 standard-efficiency heat pumps (SEER 15) | MP4/MP8-10: high-efficiency (SEER 24+)
    # However, the efficiency level does not impact row_id mapping in REMDB v4 but instead pm1/pm2 in the regression formula
    # Generally we use multi-zone non-ducted for homes without ducts, but may update to single-zone in the future for smaller homes 
    # New circuit will be addressed in future versions, but excluded here for simplicity.
    if end_use == 'heating':        
        conditions = [
            (df_copy['hvac_has_ducts'] == 'Yes'),
            (df_copy['hvac_has_ducts'] == 'No'),
        ]
        
        choices = [
            'air_source_heat_pump_centrally_ducted',
            'air_source_heat_pump_non_ducted_multi_zone',
        ]

        df_copy[row_id_col] = np.select(conditions, choices, default='unknown')

    # Cooling is priced only as a replacement cost: the heat pump upgrade already
    # serves both the heating and the cooling load, so there is no separate
    # cooling upgrade row_id.

    # =========================================
    # DELETE FOR NOW - NON-HVAC END USES
    # =========================================

    # else:
    #     # raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of: 'heating', 'cooling', 'waterHeating', 'clothesDrying', 'cooking'")
    #     raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of: 'heating', 'cooling'")

    return df_copy


def _map_remdb_parameters(
    df: pd.DataFrame, 
    remdb_v4_costs: pd.DataFrame, 
    end_use: str,
    replacement_or_upgrade: str,
    percentile: str = 'mid'
) -> pd.DataFrame:
    """Map REMDB coefficients and unit specifications to DataFrame.
    
    Args:
        df: DataFrame with row_id column.
        remdb_v4_costs: REMDB v4 cost database (indexed by row_id).
        end_use: Equipment category.
        replacement_or_upgrade: Metric type.
        percentile: Cost percentile ('low', 'mid', 'high').
        
    Returns:
        DataFrame with REMDB parameters mapped.
    """
    df_copy = df.copy()
    row_id_col = f'row_id_{end_use}_{replacement_or_upgrade}'
    prefix = f'{end_use}_{replacement_or_upgrade}_'
    
    # Parameters to map from REMDB (bounds columns included for diagnostic reporting)
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
    """Convert PM1 (capacity) to REMDB-expected units.
    
    Conversion rules based on pm1_remdb_col:
    - "Tons": kBtu/h / 12 (heat pumps, central ACs)
    - "BTU/hr": kBtu/h x 1000 (furnaces, boilers, baseboard)
    - Other: keep as-is (R-value, volume, etc.)
    
    Args:
        df: DataFrame with capacity and unit columns.
        pm1_euss_value_col: Column to extract numeric value from (e.g., capacity in kBtu/h).
        pm1_metric_col: Column with metric name from REMDB.
        pm1_unit_col: Column with target unit from REMDB.
        
    Returns:
        Series with PM1 converted to REMDB units. (pm1_remdb_col)
    """
    df_copy = df.copy()

    # Ensure numeric value (e.g., capacity may be numeric or string)
    pm1_euss_value = df_copy[pm1_euss_value_col]
    if not pd.api.types.is_numeric_dtype(pm1_euss_value):
        pm1_euss_value = df_copy[pm1_euss_value_col].astype(str)
        numeric_value = pm1_euss_value.str.extract(r'([\d.]+)', expand=False).astype(float)
    else:
        numeric_value = pm1_euss_value

    # Normalize units (case-insensitive, strip whitespace)
    pm1_remdb_col = df_copy[pm1_unit_col].str.lower().str.strip()

    result = pd.Series(np.nan, index=df_copy.index)
    
    # Tons (heat pumps, ACs): kBtu/h -> Tons (/12)
    # mask_tons = (pm1_remdb_col == 'tons')
    mask_tons = pm1_remdb_col.str.contains('ton', na=False)
    if mask_tons.any():
        result.loc[mask_tons] = numeric_value.loc[mask_tons] / 12.0
    
    # BTU/hr (furnaces, boilers, baseboard): kBtu/h -> BTU/hr (x1000)    
    mask_btu = pm1_remdb_col.str.contains('btu', na=False)
    if mask_btu.any():
        result.loc[mask_btu] = numeric_value.loc[mask_btu] * 1000.0
        
    # Other data that is not NaN (R-value, volume, etc.): keep as-is
    mask_other = ~mask_tons & ~mask_btu & pm1_remdb_col.notna()
    if mask_other.any():
        result.loc[mask_other] = numeric_value.loc[mask_other]

    # NEW: Equipment with no pm_metric defined (e.g., electric baseboard)
    # These have pm_coef=0 in REMDB, so pm doesn't contribute to cost.
    # Return 0.0 to prevent NaN propagation in the cost formula.
    mask_no_metric = pm1_remdb_col.isna()
    if mask_no_metric.any():
        result.loc[mask_no_metric] = 0.0

    return result


def _convert_pm2(
    df: pd.DataFrame,
    pm2_euss_value_col: str,
    pm2_metric_col: str,
    pm2_unit_col: str
) -> pd.Series:
    """Convert PM2 (efficiency) to REMDB-expected format.
    
    Conversion rules based on pm2_metric:
    - AFUE: Extract numeric, divide by 100 (e.g., "80% AFUE" -> 0.80)
    - Other (SEER, HSPF, EER, UEF, CEF): Extract numeric, keep as-is
    
    Args:
        df: DataFrame with efficiency and metric columns.
        pm2_euss_value_col: Column to extract numeric value from (e.g., SEER or AFUE).
        pm2_metric_col: Column with metric name from REMDB.
        pm2_unit_col: Column with unit from REMDB.
        
    Returns:
        Series with PM2 in REMDB format.
    """
    df_copy = df.copy()

    # Ensure numeric value (e.g., efficiency may be numeric or string)
    pm2_euss_value = df_copy[pm2_euss_value_col]
    if not pd.api.types.is_numeric_dtype(pm2_euss_value):
        # Efficiency may be stored as string (e.g., SEER 15", "80% AFUE")
        pm2_euss_value = df_copy[pm2_euss_value_col].astype(str)
        numeric_value = pm2_euss_value.str.extract(r'([\d.]+)', expand=False).astype(float)
    else:
        numeric_value = pm2_euss_value

    # Normalize units (case-insensitive, strip whitespace)
    pm2_remdb_col = df_copy[pm2_metric_col].str.lower().str.strip()
    
    result = pd.Series(np.nan, index=df_copy.index)
    
    # AFUE: divide by 100 to convert percentage to decimal
    # mask_afue = (pm2_metric == 'afue')
    mask_afue = pm2_remdb_col.str.contains('afue', na=False)
    if mask_afue.any():
        result.loc[mask_afue] = numeric_value.loc[mask_afue] / 100.0
    
    # Other data that is not NaN (SEER, HSPF, EER, UEF, CEF): keep as-is
    mask_other = (~mask_afue) & pm2_remdb_col.notna()
    if mask_other.any():
        result.loc[mask_other] = numeric_value.loc[mask_other]
    
    # NEW: Equipment with no pm_metric defined (e.g., electric baseboard)
    # These have pm_coef=0 in REMDB, so pm doesn't contribute to cost.
    # Return 0.0 to prevent NaN propagation in the cost formula.
    mask_no_metric = pm2_remdb_col.isna()
    if mask_no_metric.any():
        result.loc[mask_no_metric] = 0.0

    return result


def _apply_efficiency_floor(
    df: pd.DataFrame,
    row_id_col: str,
    pm2_col: str,
    efficiency_floors: Dict[str, float],
    verbose: bool = False
) -> pd.DataFrame:
    """Clamp pm2 (efficiency) upward to a hard minimum floor per equipment type.

    For replacement cost estimation, the EUSS housing stock may contain
    legacy equipment with efficiencies far below what is available or legal
    today (e.g., SEER 8, AFUE 60%).  Since the replacement cost represents
    buying TODAY's minimum-efficiency equipment, we clamp ALL below-floor
    pm2 values up to the floor.

    The original (pre-clamping) pm2 values are preserved in a new column
    named ``{pm2_col}_original`` so that downstream reporting can show both
    the raw EUSS efficiency and the floored replacement efficiency.

    Only modifies rows whose row_id appears in *efficiency_floors*.
    Rows with NaN pm2 are left untouched.

    Args:
        df: DataFrame with pm2 values already converted by _convert_pm2().
        row_id_col: Column containing the REMDB row_id.
        pm2_col: Column containing the converted pm2 values.
        efficiency_floors: Dict mapping row_id -> minimum pm2 value.
        verbose: If True, print diagnostic info about clamped homes.

    Returns:
        DataFrame with:
          - ``pm2_col`` clamped upward where applicable (used by cost regression)
          - ``{pm2_col}_original`` preserving pre-clamping values (used by validation)
    """
    df_out = df.copy()

    # Preserve the raw EUSS efficiency before any flooring is applied.
    original_col = f'{pm2_col}_original'
    df_out[original_col] = df_out[pm2_col].copy()

    # Map each home's row_id to its floor. 
    # Rows whose row_id is not in the floors dict get NaN
    # Series.clip treats a NaN lower bound as "do not clip", so NaN rows pass through unchanged.
    # This raises every below-floor value up to its floor in one vectorized step.
    floor_by_row = df_out[row_id_col].map(efficiency_floors)
    df_out[pm2_col] = df_out[pm2_col].clip(lower=floor_by_row)

    if verbose:
        # A home was raised when its floored pm2 differs from the original and
        # the original was present (NaN originals are never touched).
        raised = (
            (df_out[pm2_col] != df_out[original_col])
            & df_out[original_col].notna()
        )
        total_clamped = int(raised.sum())
        if total_clamped == 0:
            print("    No homes required efficiency floor clamping.")
        else:
            print(f"    Total clamped: {total_clamped:,} homes")
            print(f"    Original values preserved in: {original_col}")

    return df_out


def _report_bounds_comparison(
    df: pd.DataFrame,
    row_id_col: str,
    pm_col: str,
    lower_bound_col: str,
    upper_bound_col: str,
    metric_name: str,
    verbose: bool = True,
    max_groups_to_print: int = 10
) -> int:
    """Report out-of-bounds metrics for diagnostic purposes (no data modification).
    
    Compares EUSS values against REMDB expected bounds and prints a summary
    of values falling outside the expected range, separated into BELOW and
    ABOVE categories. This is informational only - values are NOT modified.
    
    Args:
        df: DataFrame with metric and bounds columns.
        row_id_col: Column containing REMDB row_id.
        pm_col: Column with the performance metric values.
        lower_bound_col: Column with REMDB lower bounds.
        upper_bound_col: Column with REMDB upper bounds.
        metric_name: Human-readable name for reporting (e.g., "capacity", "efficiency").
        verbose: Whether to print the report.
        max_groups_to_print: Maximum number of row_id groups to display.
        
    Returns:
        Total count of out-of-bounds values.
    """
    if not verbose:
        return 0
    
    # Check if bounds columns exist
    if lower_bound_col not in df.columns or upper_bound_col not in df.columns:
        return 0
    
    # Coerce to numeric for safe comparison
    metric = pd.to_numeric(df[pm_col], errors='coerce')
    lower = pd.to_numeric(df[lower_bound_col], errors='coerce')
    upper = pd.to_numeric(df[upper_bound_col], errors='coerce')
    
    # Only check rows where all values are present
    comparable = metric.notna() & lower.notna() & upper.notna()
    below = comparable & (metric < lower)
    above = comparable & (metric > upper)
    out_of_bounds = below | above
    
    total_oob = int(out_of_bounds.sum())
    
    if total_oob == 0:
        return 0
    
    # Exclude unknown row_ids from detailed reporting
    known = df[row_id_col].notna() & (df[row_id_col] != 'unknown')
    oob_known = out_of_bounds & known
    
    if not oob_known.any():
        print(f"\n  INFO: {total_oob:,} homes have {metric_name} outside REMDB bounds (row_id='unknown')")
        return total_oob
    
    # Print header
    print(f"\n  INFO: {total_oob:,} homes have {metric_name} outside REMDB bounds")
    
    # Get unique row_ids with out-of-bounds values, sorted by count
    row_id_counts = df.loc[oob_known, row_id_col].value_counts()
    
    for i, (row_id, count) in enumerate(row_id_counts.items()):
        if i >= max_groups_to_print:
            remaining = len(row_id_counts) - max_groups_to_print
            if remaining > 0:
                print(f"        ... {remaining:,} additional row_ids omitted")
            break
        
        # Get data for this row_id
        mask = oob_known & (df[row_id_col] == row_id)
        row_metric = metric[mask]
        row_lower = lower[mask].iloc[0]
        row_upper = upper[mask].iloc[0]
        
        # Split into below and above
        below_mask = row_metric < row_lower
        above_mask = row_metric > row_upper
        
        print(f"\n    {row_id}: {count:,} values")
        print(f"    REMDB Bounds: {row_lower:.2f} - {row_upper:.2f}")
        print(f"    EUSS Bounds:")
        
        if below_mask.any():
            below_vals = row_metric[below_mask]
            print(f"        - BELOW ({below_mask.sum():,}): {below_vals.min():.2f} - {below_vals.max():.2f}")
        if above_mask.any():
            above_vals = row_metric[above_mask]
            print(f"        - ABOVE ({above_mask.sum():,}): {above_vals.min():.2f} - {above_vals.max():.2f}")
    
    return total_oob


def filter_by_percentile(
    df: pd.DataFrame,
    column: str,
    lower_percentile: Optional[float] = None,
    upper_percentile: Optional[float] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Filter DataFrame to keep only values within specified percentile range.
    
    Args:
        df: Input DataFrame.
        column: Column name to filter on.
        lower_percentile: Lower percentile bound (0-100), or None for no lower bound.
        upper_percentile: Upper percentile bound (0-100), or None for no upper bound.
        verbose: Print filtering statistics.
        
    Returns:
        DataFrame filtered to specified percentile range.
        
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
    df_filtered = df_copy[mask].copy()
    
    n_filtered = len(df_filtered)
    n_removed = n_original - n_filtered
    
    if verbose:
        pct_removed = 100 * n_removed / n_original if n_original > 0 else 0
        print(f"\nPercentile Filtering on '{column}':")
        print(f"   Percentile range: {lower_percentile or 0:.1f}% - {upper_percentile or 100:.1f}%")
        print(f"   Value bounds: {lower_bound:.2f} - {upper_bound:.2f}")
        print(f"   Original range: {values.min():.2f} - {values.max():.2f}")
        print(f"   Rows: {n_original:,} -> {n_filtered:,} (removed {n_removed:,}, {pct_removed:.2f}%)")
    
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
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare REMDB v4 metrics for cost calculations.
    
    This function handles both replacement and upgrade metrics with optional
    percentile-based filtering for capacity outliers. NaN values propagate
    naturally for homes with invalid fuel/technology types per the validation
    framework.
    
    Args:
        df: DataFrame with equipment specifications.
        remdb_v4_costs: REMDB v4 cost database (indexed by row_id).
        end_use: Equipment category ('heating' or 'cooling').
        metric_type: 'replacement' (avoided-replacement cost for the existing
            system) or 'upgrade' (the heat pump). Both are costed using the
            same capacity column -- see the capacity_col comment below.
        percentile: Cost percentile ('low', 'mid', 'high').
        capacity_lower_percentile: Lower percentile for capacity filtering (0-100), or None.
        capacity_upper_percentile: Upper percentile for capacity filtering (0-100), or None.
        verbose: Print progress and statistics.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - df_main: Main DataFrame with summary metrics
            - df_detailed: Detailed DataFrame with all REMDB parameters

    Example:
        # Basic usage
        df_main, df_detailed = add_remdb_metrics(df, remdb_costs, 'heating', 'replacement')
        
        # With percentile filtering (95% CI)
        df_main, df_detailed = add_remdb_metrics(
            df, remdb_costs, 'heating', 'upgrade',
            capacity_lower_percentile=2.5,
            capacity_upper_percentile=97.5
        )
    """
    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================
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
    
    # Create defensive copy to prevent mutation on re-execution
    df_copy = df.copy()
    
    prefix = f"{end_use}_{metric_type}_"
    row_id_col = f'row_id_{end_use}_{metric_type}'
    
    # Determine source columns based on end_use and metric_type
    # UPDATE VARIABLE NAMES FOR WATER HEATING, CLOTHES DRYING, COOKING LATER
    if end_use == 'heating':
        # size_heating_system_primary_k_btu_h is the retrofit heat pump's
        # ResStock-autosized capacity for this measure package, not the
        # baseline furnace's nameplate size (see process_euss_data.py,
        # df_enduse_compare). It is used for BOTH metric_type values: the
        # 'replacement' cost (avoided baseline-system replacement) is sized
        # off the same heat-pump capacity because no separate baseline
        # capacity value is carried in this pipeline.
        #
        # FOLLOW-UP FLAGGED 19 Aug 2026: this is the point where the
        # mismatch actually enters the replacement-cost regression -- pm1
        # (capacity) for metric_type='replacement' is built from the
        # retrofit capacity, not the baseline system's own. Dollar magnitude
        # and whether cooling is affected to the same degree are unconfirmed.
        # A fix is planned for a separate, value-critical session; see
        # docs/SESSION_CHANGELOG_2026-08-19.md. Do not change this line here.
        capacity_col = 'size_heating_system_primary_k_btu_h'
        efficiency_col = 'upgrade_hvac_heating_efficiency' if metric_type == 'upgrade' else 'hvac_heating_efficiency'
    elif end_use == 'cooling':
        capacity_col = 'size_cooling_system_primary_k_btu_h'
        efficiency_col = 'hvac_cooling_efficiency'
    else:
        raise ValueError(f"Unsupported end_use: {end_use}")
    
    # Validate required columns exist
    for col in [capacity_col, efficiency_col]:
        if col not in df_copy.columns:
            raise KeyError(f"Missing required column: '{col}'")
    
    # =========================================================================
    # STEP 1: Optional percentile filtering (handles capacity outliers)
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
    # STEP 2: Assign row_id based on equipment type
    # =========================================================================
    if metric_type == 'replacement':
        df_copy = _assign_replacement_row_id(df_copy, end_use)
    else:
        df_copy = _assign_upgrade_row_id(df_copy, end_use)
        
    unknown_count = (df_copy[row_id_col] == 'unknown').sum()
    if unknown_count > 0 and verbose:
        print(f"  {unknown_count:,} homes with unknown row_id (will have NaN costs)")
    
    # Diagnostic: Show row_id distribution
    if verbose:
        print(f"\n  Row ID Distribution ({row_id_col}):")
        row_id_counts = df_copy[row_id_col].value_counts()
        for row_id, count in row_id_counts.items():
            print(f"    {row_id}: {count:,}")

    # =========================================================================
    # STEP 3: Map REMDB parameters (coefficients, units)
    # =========================================================================
    df_copy = _map_remdb_parameters(df_copy, remdb_v4_costs, end_use, metric_type, percentile)
    
    # =========================================================================
    # STEP 4: Convert metrics to REMDB units
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
    
    # Diagnostic: Explain NaN sources for pm1 and pm2
    if verbose:
        pm1_metric_col = f'{prefix}pm1_metric'
        pm2_metric_col = f'{prefix}pm2_metric'
        
        print(f"\n  NaN Diagnostics:")
        
        # pm1 NaN breakdown
        pm1_nan_mask = df_copy[pm1_col].isna()
        pm1_nan_from_unknown = pm1_nan_mask & (df_copy[row_id_col] == 'unknown')
        pm1_nan_from_source = pm1_nan_mask & df_copy[capacity_col].isna()
        pm1_nan_other = pm1_nan_mask & ~pm1_nan_from_unknown & ~pm1_nan_from_source
        
        print(f"    pm1 NaN breakdown ({pm1_nan_mask.sum():,} total):")
        print(f"      - Unknown row_id: {pm1_nan_from_unknown.sum():,}")
        print(f"      - Missing source data ({capacity_col}): {pm1_nan_from_source.sum():,}")
        if pm1_nan_other.sum() > 0:
            print(f"      - Other: {pm1_nan_other.sum():,}")
        
        # pm2 NaN breakdown
        pm2_nan_mask = df_copy[pm2_col].isna()
        pm2_nan_from_unknown = pm2_nan_mask & (df_copy[row_id_col] == 'unknown')
        pm2_nan_from_no_metric = pm2_nan_mask & df_copy[pm2_metric_col].isna()
        pm2_nan_from_source = pm2_nan_mask & df_copy[efficiency_col].isna()
        pm2_nan_other = (pm2_nan_mask
                         & ~pm2_nan_from_unknown
                         & ~pm2_nan_from_no_metric
                         & ~pm2_nan_from_source)
        
        print(f"    pm2 NaN breakdown ({pm2_nan_mask.sum():,} total):")
        print(f"      - Unknown row_id: {pm2_nan_from_unknown.sum():,}")
        print(f"      - No pm2_metric defined (e.g., electric baseboard): {(pm2_nan_from_no_metric & ~pm2_nan_from_unknown).sum():,}")
        print(f"      - Missing source data ({efficiency_col}): {pm2_nan_from_source.sum():,}")
        if pm2_nan_other.sum() > 0:
            print(f"      - Extraction failure (has metric + source, but numeric parse failed): {pm2_nan_other.sum():,}")
            # Show sample values to aid debugging
            sample_vals = df_copy.loc[pm2_nan_other, efficiency_col].value_counts().head(5)
            for val, cnt in sample_vals.items():
                print(f"          '{val}': {cnt:,} homes")
        
        # Show pm2_metric distribution to explain why some have no metric
        print(f"\n  pm2_metric Distribution ({pm2_metric_col}):")
        pm2_metric_counts = df_copy[pm2_metric_col].value_counts(dropna=False)
        for metric, count in pm2_metric_counts.items():
            metric_display = metric if pd.notna(metric) else "(NaN - no efficiency metric)"
            print(f"    {metric_display}: {count:,}")
    
    # =========================================================================
    # STEP 4.5a: Apply efficiency floors (replacement only)
    # =========================================================================
    # For replacement costs only: clamp ALL pm2 values below the floor UP
    # to the minimum efficiency equipment available today.
    #   SEER 8 -> SEER 15, AFUE 60% -> AFUE 80%, etc.    
    # Upgrade efficiencies are set by the measure package definition, so
    # no clamping is needed for upgrades.
    if metric_type == 'replacement':
        if verbose:
            print(f"\n  Step 4.5a: Applying efficiency floors (replacement only)")
        df_copy = _apply_efficiency_floor(
            df=df_copy,
            row_id_col=row_id_col,
            pm2_col=pm2_col,
            efficiency_floors=EFFICIENCY_FLOORS_PM2,
            verbose=verbose
        )

    # Capacity (pm1) is used exactly as converted. Values outside the REMDB
    # training bounds are reported in Step 5 but never modified; the upstream
    # percentile filter and NaN propagation handle genuine outliers.

    # =========================================================================
    # STEP 5: Report bounds comparison (diagnostic only - no data modification)
    # =========================================================================
    if verbose:
        _report_bounds_comparison(
            df=df_copy,
            row_id_col=row_id_col, 
            pm_col=pm1_col,  # <-- Corrected to 'pm_col'
            lower_bound_col=f'{prefix}pm1_lower_bound',
            upper_bound_col=f'{prefix}pm1_upper_bound',
            metric_name='capacity',
            verbose=verbose
        )
        _report_bounds_comparison(
            df=df_copy,
            row_id_col=row_id_col,
            pm_col=pm2_col,  # <-- Corrected to 'pm_col'
            lower_bound_col=f'{prefix}pm2_lower_bound',
            upper_bound_col=f'{prefix}pm2_upper_bound',
            metric_name='efficiency',
            verbose=verbose
        )    
    # =========================================================================
    # STEP 6: Report summary statistics
    # =========================================================================
    if verbose:
        pm1_valid = df_copy[pm1_col].notna().sum()
        pm2_valid = df_copy[pm2_col].notna().sum()
        pm1_nan = df_copy[pm1_col].isna().sum()
        pm2_nan = df_copy[pm2_col].isna().sum()
        print(f"\n  Valid metrics: {pm1_valid:,} pm1, {pm2_valid:,} pm2")
        if pm1_nan > 0 or pm2_nan > 0:
            print(f"  NaN metrics: {pm1_nan:,} pm1, {pm2_nan:,} pm2 (per validation framework)")

    # =========================================================================
    # STEP 7: Prepare output columns
    # =========================================================================
    # Original (pre-clamping) pm2 column, created by _apply_efficiency_floor()
    pm2_original_col = f'{pm2_col}_original'

    summary_cols = [
        row_id_col,
        pm1_col,
        pm2_col,
    ]
    # Include original pm2 in summary if it exists (replacement metrics only)
    if pm2_original_col in df_copy.columns:
        summary_cols.append(pm2_original_col)
    
    detailed_cols = summary_cols + [
        f'{prefix}pm1_metric', f'{prefix}pm1_unit', f'{prefix}pm1_coef_{percentile}',
        f'{prefix}pm1_lower_bound', f'{prefix}pm1_upper_bound',
        f'{prefix}pm2_metric', f'{prefix}pm2_unit', f'{prefix}pm2_coef_{percentile}',
        f'{prefix}pm2_lower_bound', f'{prefix}pm2_upper_bound',
        f'{prefix}intercept_{percentile}',
        f'{prefix}multiplier_retrofit', f'{prefix}adder_retrofit',
    ]

    # Build both outputs
    existing_cols = [c for c in summary_cols if c in df_copy.columns]
    df_original_aligned = df.loc[df_copy.index].copy()
    
    cols_to_drop = [c for c in existing_cols if c in df_original_aligned.columns]
    if cols_to_drop:
        df_original_aligned = df_original_aligned.drop(columns=cols_to_drop)
    
    df_main = pd.concat([df_original_aligned, df_copy[existing_cols]], axis=1)
    
    detailed_existing = [c for c in detailed_cols if c in df_copy.columns]
    df_detailed = df_copy[detailed_existing].copy()
    
    return df_main, df_detailed  # ← Always return both
