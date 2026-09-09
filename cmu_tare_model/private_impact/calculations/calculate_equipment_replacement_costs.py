"""
========================================================================================================================================================================
UNIFIED MODULE: CALCULATE REPLACEMENT INSTALLED COSTS FOR VARIOUS END USES
========================================================================================================================================================================
This module provides a unified interface for calculating REPLACEMENT installed costs
using either REMDB v3 (probabilistic sampling) or REMDB v4 (regression) methodology.

Supported cost scenarios (via ``cost_scenario`` parameter):
    - ``'v3'``       : Probabilistic sampling from NREL REMDB v3 cost distributions
    - ``'v4LOW'``   : REMDB v4 regression, 10th percentile coefficients
    - ``'v4MID'``   : REMDB v4 regression, 50th percentile coefficients (default)
    - ``'v4HIGH'``  : REMDB v4 regression, 90th percentile coefficients

CRITICAL DISTINCTION: REPLACEMENT vs UPGRADE
    - **Replacement costs**: Cost to replace existing equipment with LIKE-FOR-LIKE technology (counterfactual)
    - **Upgrade costs**: Cost to retrofit to improved technology (e.g., gas furnace -> heat pump)

REMDB v3 methodology:
    - Probabilistic sampling from cost distributions (progressive/reference/conservative)
    - Technology mapping based on baseline fuel type
    - Costs from Excel-based dictionaries with CPI adjustment

REMDB v4 methodology:
    - Regression-based: Material_Price = (pm1 * pm1_coef) + (pm2 * pm2_coef) + intercept
    - Installed_Cost = (Material_Price * multiplier) + adder
    - Regression coefficients are published in 2023$; the public API inflates
      the result to the model reference year (USD2025) via cpi_ratio_2025_2023
    - PREREQUISITE: Call add_remdb_replacement_metrics() first

Consolidates:
    - calculations/calculate_equipment_replacement_costs.py (REMDB v3, 314 lines)
    - remdb_v4_update/calculate_equipment_replacement_costs_remdb_v4.py (REMDB v4, 211 lines)

# CREATED FEBRUARY 10, 2026 - UNIFIED v3/v4 MODULE
# Based on:
#   - calculations/calculate_equipment_replacement_costs.py (v3)
#   - remdb_v4_update/calculate_equipment_replacement_costs_remdb_v4.py (v4)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from cmu_tare_model.constants import (
    VERBOSE, VALID_MENU_MPS, EQUIPMENT_SPECS, VALID_CATEGORIES,
    REMDB_COST_SCENARIO_KEYS
)
from cmu_tare_model.utils.column_names import create_cost_col
from cmu_tare_model.utils.validation_framework import (
    apply_new_columns_to_dataframe,
    apply_final_masking,
    initialize_validation_tracking,
    create_retrofit_only_series
)
from cmu_tare_model.utils.calculation_utils import (
    filter_valid_tech_homes,
    sample_costs_from_distributions
)
from cmu_tare_model.utils.inflation_adjustment import cpi_ratio_2025_2023


# ========================================================================================================================================================================
# PUBLIC API
# ========================================================================================================================================================================

def calculate_replacement_installed_cost(
    df: pd.DataFrame,
    df_detailed: pd.DataFrame,
    menu_mp: int,
    end_use: str,
    cost_scenario: str,
    cost_dict: Optional[dict] = None,
    verbose: bool = VERBOSE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate REPLACEMENT installed costs using unified interface.

    Routes to REMDB v3 (probabilistic) or v4 (regression) based on ``cost_scenario``.

    FIXED 20 Aug 2026 (v4 only) -- the v4 path's pm1_euss capacity, built in
    add_remdb_metrics, now reads the existing system's own size
    (base_size_heating_system_primary_k_btu_h / base_size_cooling_...),
    added in process_euss_data.py's df_enduse_refactored. See
    docs/SESSION_CHANGELOG_2026-08-20.md for the before/after numbers.

    The v3 path below (_calculate_replacement_cost_per_row) still reads
    size_heating_system_primary_k_btu_h, the retrofit heat pump's capacity,
    not the old system's -- same bug, not fixed. This is harmless today
    because REMDB_COST_SCENARIO_KEYS only runs 'v4MID' (constants.py), so v3
    never executes. Apply the same fix there if v3 is ever turned back on.

    Args:
        df: DataFrame with home data and prepared metrics.
        df_detailed: Detailed DataFrame with regression parameters (v4) or cost lookup data.
        menu_mp: Measure package number.
        end_use: Equipment category ('heating', 'waterHeating', 'clothesDrying', 'cooking', 'cooling').
        cost_scenario: Cost scenario key. One of:
            'v3', 'v4LOW', 'v4MID', 'v4HIGH'.
        cost_dict: Required for REMDB v3. Dictionary mapping (technology, efficiency) tuples
            to cost component distributions. Ignored for v4.
        verbose: Whether to print detailed processing information.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - df_copy: Updated DataFrame with new cost column.
            - df_detailed_out: Updated detailed DataFrame with new cost column.

    Raises:
        ValueError: If menu_mp, end_use, or cost_scenario is invalid.
        KeyError: If required prerequisite columns are missing (v4).
        RuntimeError: If calculation fails unexpectedly.
    """

    # ===== Validate common inputs =====
    _validate_inputs(menu_mp, end_use, cost_scenario)

    # Derive routing method and percentile from cost_scenario
    if cost_scenario == 'v3':
        method, pct = 'v3', None
    else:
        method = 'remdb_v4'
        pct = cost_scenario[2:].lower()

    replacement_or_upgrade = 'replacement'

    if verbose:
        print(f"\nCalculating {end_use} replacement costs ({cost_scenario})")

    # ===== SPECIAL HANDLING FOR COOLING (METADATA-ONLY) =====
    # Cooling is not in EQUIPMENT_SPECS so doesn't have include_cooling flag.
    # Create temporary flag for validation framework, then clean up.
    added_cooling_to_categories = False
    created_temp_flag = False

    if end_use == 'cooling':
        if 'cooling' not in EQUIPMENT_SPECS and 'cooling' not in VALID_CATEGORIES:
            VALID_CATEGORIES.append('cooling')
            added_cooling_to_categories = True

        if f'include_{end_use}' not in df.columns:
            df_copy = df.copy()
            df_copy['include_cooling'] = (
                df_copy['hvac_cooling_type'].notna() &
                (df_copy['hvac_cooling_type'] != 'None')
            )
            created_temp_flag = True
            if verbose:
                print("  Created temporary include_cooling flag for metadata calculation")
        else:
            df_copy = df.copy()
    else:
        df_copy = df.copy()

    # ===== STEP 1: Initialize validation tracking =====
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        df_copy, end_use, menu_mp=menu_mp, verbose=verbose)

    # Initialize tracking dictionary key for cooling if needed
    if end_use == 'cooling' and end_use not in all_columns_to_mask:
        all_columns_to_mask['cooling'] = []
        if verbose:
            print("  Initialized all_columns_to_mask['cooling'] for metadata tracking")

    # ===== STEP 2: Initialize result series with template =====
    result_series = create_retrofit_only_series(df_copy, valid_mask)

    # ===== STEP 3 & 4: Route to appropriate calculation method =====
    if method == 'remdb_v4':
        installed_cost = _calculate_v4_replacement(df_detailed, end_use, pct)
        # The REMDB v4 regression returns costs in 2023 dollars. Inflate to the
        # model reference year so capital costs share the same USD2025 basis as
        # the incomes, fuel costs, and rebates used elsewhere in the model.
        installed_cost = installed_cost * cpi_ratio_2025_2023
    else:
        # REMDB v3 requires cost_dict
        if cost_dict is None:
            raise ValueError(
                "cost_dict is required for REMDB v3 calculations. "
                "Provide the dictionary mapping (technology, efficiency) tuples to cost distributions."
            )
        installed_cost, _ = _calculate_v3_replacement(
            df_copy, cost_dict, end_use, menu_mp, valid_mask, verbose)

    # Safety net: Ensure costs never go negative
    installed_cost = installed_cost.clip(lower=0)

    # Update result series with calculated values (only for valid homes)
    result_series.loc[valid_mask] = installed_cost.loc[valid_mask].round(2)

    # Build column name using centralized utility
    cost_col = create_cost_col(menu_mp=menu_mp, category=end_use, cost_type='replacement', cost_scenario=cost_scenario)

    # Create DataFrame with new column
    df_new_columns = pd.DataFrame({cost_col: result_series})

    # Apply new columns to DataFrame with proper tracking
    df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
        df_copy, df_new_columns, end_use, category_columns_to_mask, all_columns_to_mask)

    # ===== STEP 5: Apply final verification masking for consistency =====
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=verbose)

    # ===== Add cost column to detailed DataFrame =====
    df_detailed_out = df_detailed.copy()
    df_detailed_out[cost_col] = df_copy[cost_col]

    # ===== CLEANUP: Remove temporary cooling flag if created =====
    if created_temp_flag:
        df_copy = df_copy.drop(columns=['include_cooling'])
        if verbose:
            print("  Removed temporary include_cooling flag")

    # ===== CLEANUP: Remove cooling from VALID_CATEGORIES if we added it =====
    if added_cooling_to_categories:
        VALID_CATEGORIES.remove('cooling')
        if verbose:
            print("  Removed 'cooling' from VALID_CATEGORIES")

    # Report summary
    valid_count = df_copy[cost_col].notna().sum()
    mean_cost = df_copy[cost_col].mean()
    if verbose:
        print(f"  Calculated costs for {valid_count:,} homes (mean: ${mean_cost:,.2f})\n")

    return df_copy, df_detailed_out


# ========================================================================================================================================================================
# INTERNAL: VALIDATION
# ========================================================================================================================================================================

def _validate_inputs(menu_mp: int, end_use: str, cost_scenario: str) -> None:
    """Validate common inputs for cost calculations."""
    if menu_mp not in VALID_MENU_MPS:
        raise ValueError(
            f"Please enter a valid measure package number for menu_mp. "
            f"Should be one of {VALID_MENU_MPS}."
        )
    # Allow 'cooling' for replacement even if not in standard VALID_CATEGORIES
    valid_end_uses = list(VALID_CATEGORIES) + (['cooling'] if 'cooling' not in VALID_CATEGORIES else [])
    if end_use not in valid_end_uses:
        raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of {valid_end_uses}")
    if cost_scenario not in REMDB_COST_SCENARIO_KEYS:
        raise ValueError(
            f"Invalid cost_scenario: '{cost_scenario}'. "
            f"Must be one of {REMDB_COST_SCENARIO_KEYS}"
        )


# ========================================================================================================================================================================
# INTERNAL: REMDB V4 REGRESSION CALCULATION
# ========================================================================================================================================================================

def _calculate_v4_replacement(
    df_detailed: pd.DataFrame,
    end_use: str,
    percentile: str
) -> pd.Series:
    """
    Calculate replacement installed costs using REMDB v4 regression formula.

    Formula:
        Material_Price = (pm1 * pm1_coef) + (pm2 * pm2_coef) + intercept
        Installed_Cost = (Material_Price * multiplier) + adder

    Args:
        df_detailed: DataFrame with REMDB v4 regression parameters.
        end_use: Equipment category.
        percentile: Cost percentile ('low', 'mid', 'high').

    Returns:
        Series of installed costs.

    Raises:
        KeyError: If prerequisite columns are missing.
    """
    prefix = f'{end_use}_replacement_'

    # Verify prerequisite columns
    required_cols = [
        f'{prefix}pm1_euss',
        f'{prefix}pm2_euss',
        f'{prefix}pm1_coef_{percentile}',
        f'{prefix}pm2_coef_{percentile}',
        f'{prefix}intercept_{percentile}',
        f'{prefix}multiplier_retrofit',
        f'{prefix}adder_retrofit',
    ]
    missing = [c for c in required_cols if c not in df_detailed.columns]
    if missing:
        raise KeyError(
            f"Missing columns: {missing}\n"
            f"Call add_remdb_replacement_metrics() first to prepare these columns."
        )

    # REMDB v4 regression formula
    pm1 = df_detailed[f'{prefix}pm1_euss']
    pm2 = df_detailed[f'{prefix}pm2_euss']
    material_price = (
        (pm1 * df_detailed[f'{prefix}pm1_coef_{percentile}']) +
        (pm2 * df_detailed[f'{prefix}pm2_coef_{percentile}']) +
        df_detailed[f'{prefix}intercept_{percentile}']
    )
    installed_cost = (
        (material_price * df_detailed[f'{prefix}multiplier_retrofit']) +
        df_detailed[f'{prefix}adder_retrofit']
    )

    return installed_cost


# ========================================================================================================================================================================
# INTERNAL: REMDB V3 PROBABILISTIC CALCULATION
# ========================================================================================================================================================================

def _calculate_v3_replacement(
    df_copy: pd.DataFrame,
    cost_dict: dict,
    end_use: str,
    menu_mp: int,
    valid_mask: pd.Series,
    verbose: bool = VERBOSE
) -> Tuple[pd.Series, str]:
    """
    Calculate replacement installed costs using REMDB v3 probabilistic sampling.

    Args:
        df_copy: DataFrame with home data.
        cost_dict: Dictionary mapping (technology, efficiency) to cost distributions.
        end_use: Equipment category.
        menu_mp: Measure package number.
        valid_mask: Series indicating which rows have valid data.
        verbose: Whether to print progress.

    Returns:
        Tuple of (installed_cost Series, cost_column_name string).
    """
    # Get conditions, technology-efficiency pairs, and cost components
    params = _get_end_use_replacement_parameters(df_copy, end_use)
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
        # Filter to valid homes with identifiable technology
        df_valid, valid_calculation_indices, tech_filtered, eff_filtered = filter_valid_tech_homes(
            df_copy, valid_mask, tech, eff)

        if verbose:
            print(f"  After tech filtering: {len(valid_calculation_indices)} homes remain valid for {end_use} replacement")

        if df_valid.empty:
            raise ValueError(f"Warning: No valid homes found for {end_use} replacement cost calculation.")

        # Sample costs from distributions
        sampled_costs_dict = sample_costs_from_distributions(
            tech_filtered, eff_filtered, cost_dict, cost_components)

        # Calculate the replacement cost per row
        replacement_cost, cost_column_name = _calculate_replacement_cost_per_row(
            df_valid, sampled_costs_dict, menu_mp, end_use)

        # Build result series
        result_series = create_retrofit_only_series(df_copy, valid_mask)
        result_series.loc[valid_calculation_indices] = np.round(replacement_cost, 2)

        return result_series, cost_column_name

    except Exception as e:
        raise RuntimeError(f"Error in {end_use} replacement cost calculation: {str(e)}")


# ========================================================================================================================================================================
# INTERNAL: V3 HELPER FUNCTIONS
# ========================================================================================================================================================================

def _get_end_use_replacement_parameters(
    df: pd.DataFrame,
    end_use: str
) -> dict:
    """
    Retrieve parameters for replacement cost calculations based on end use type.

    Args:
        df: DataFrame containing equipment data.
        end_use: Type of equipment.

    Returns:
        Dictionary containing conditions, technology-efficiency pairs, and cost components.
    """
    parameters = {}

    if 'heating' in VALID_CATEGORIES:
        parameters['heating'] = {
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
        }

    if 'waterHeating' in VALID_CATEGORIES:
        parameters['waterHeating'] = {
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
        }

    if 'clothesDrying' in VALID_CATEGORIES:
        parameters['clothesDrying'] = {
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
        }

    if 'cooking' in VALID_CATEGORIES:
        parameters['cooking'] = {
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

    return parameters[end_use]


def _calculate_replacement_cost_per_row(
    df_valid: pd.DataFrame,
    sampled_costs_dict: dict,
    menu_mp: int,
    end_use: str
) -> tuple:
    """
    Calculate replacement cost for each row based on the end use type.

    Args:
        df_valid: Filtered DataFrame containing valid rows.
        sampled_costs_dict: Dictionary with sampled costs for each component.
        menu_mp: Menu option identifier.
        end_use: Type of end-use.

    Returns:
        Tuple of (replacement_cost, cost_column_name).
    """
    try:
        if end_use == 'heating':
            if 'size_heating_system_primary_k_btu_h' not in df_valid.columns:
                raise ValueError("Required column 'size_heating_system_primary_k_btu_h' not found in DataFrame")

            required_components = ['unitCost', 'otherCost', 'cost_per_kBtuh']
            for comp in required_components:
                if comp not in sampled_costs_dict:
                    raise KeyError(f"Required cost component '{comp}' not found for heating calculation")

            replacement_cost = (
                sampled_costs_dict['unitCost'] +
                sampled_costs_dict['otherCost'] +
                (df_valid['size_heating_system_primary_k_btu_h'] * sampled_costs_dict['cost_per_kBtuh']))
            cost_column_name = create_cost_col(menu_mp=menu_mp, category='heating', cost_type='replacement', cost_scenario='v3')

        elif end_use == 'waterHeating':
            if 'size_water_heater_gal' not in df_valid.columns:
                raise ValueError("Required column 'size_water_heater_gal' not found in DataFrame")

            required_components = ['unitCost', 'cost_per_gallon']
            for comp in required_components:
                if comp not in sampled_costs_dict:
                    raise KeyError(f"Required cost component '{comp}' not found for water heating calculation")

            replacement_cost = (
                sampled_costs_dict['unitCost'] +
                (sampled_costs_dict['cost_per_gallon'] * df_valid['size_water_heater_gal']))
            cost_column_name = create_cost_col(menu_mp=menu_mp, category='waterHeating', cost_type='replacement', cost_scenario='v3')

        else:
            if 'unitCost' not in sampled_costs_dict:
                raise KeyError(f"Required cost component 'unitCost' not found for {end_use} calculation")

            replacement_cost = sampled_costs_dict['unitCost']
            cost_column_name = create_cost_col(menu_mp=menu_mp, category=end_use, cost_type='replacement', cost_scenario='v3')

        return replacement_cost, cost_column_name

    except Exception as e:
        raise RuntimeError(f"Error calculating {end_use} replacement cost: {str(e)}")
