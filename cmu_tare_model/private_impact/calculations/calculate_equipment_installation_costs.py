"""
========================================================================================================================================================================
UNIFIED MODULE: CALCULATE UPGRADE INSTALLED COSTS FOR VARIOUS END USES
========================================================================================================================================================================
This module provides a unified interface for calculating UPGRADE (retrofit) installed costs
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
    - Costs from Excel-based dictionaries with CPI adjustment
    - Technology-efficiency pair mapping for each end-use

REMDB v4 methodology:
    - Regression-based: Material_Price = (pm1 * pm1_coef) + (pm2 * pm2_coef) + intercept
    - Installed_Cost = (Material_Price * multiplier) + adder
    - Regression coefficients are published in 2023$; the public API inflates
      the result to the model reference year (USD2025) via cpi_ratio_2025_2023
    - PREREQUISITE: Call add_remdb_upgrade_metrics() first to prepare pm1/pm2 columns

Consolidates:
    - calculate_equipment_installation_costs.py (REMDB v3, 402 lines)
    - calculate_equipment_installation_costs_remdb_v4.py (REMDB v4, 163 lines)

# CREATED FEBRUARY 10, 2026 - UNIFIED v3/v4 MODULE
# Based on:
#   - calculations/calculate_equipment_installation_costs.py (v3)
#   - remdb_v4_update/calculate_equipment_installation_costs_remdb_v4.py (v4)
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

def calculate_upgrade_installed_cost(
    df: pd.DataFrame,
    df_detailed: pd.DataFrame,
    menu_mp: int,
    end_use: str,
    cost_scenario: str,
    cost_dict: Optional[dict] = None,
    verbose: bool = VERBOSE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate UPGRADE installed costs using unified interface.

    Routes to REMDB v3 (probabilistic) or v4 (regression) based on ``cost_scenario``.

    Args:
        df: DataFrame with home data and prepared metrics.
        df_detailed: Detailed DataFrame with regression parameters (v4) or cost lookup data.
        menu_mp: Measure package number.
        end_use: Equipment category ('heating', 'waterHeating', 'clothesDrying', 'cooking').
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
        method, percentile = 'v3', None
    else:
        method = 'remdb_v4'
        percentile = cost_scenario[2:].lower()  # 'v4LOW' -> 'low', 'v4MID' -> 'mid', 'v4HIGH' -> 'high'

    if verbose:
        print(f"\nCalculating {end_use} upgrade costs ({cost_scenario})")

    # ===== STEP 1: Initialize validation tracking =====
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        df, end_use, menu_mp=menu_mp, verbose=verbose)

    # ===== STEP 2: Initialize result series with template =====
    result_series = create_retrofit_only_series(df_copy, valid_mask)

    # ===== STEP 3 & 4: Route to appropriate calculation method =====
    if method == 'remdb_v4':
        installed_cost = _calculate_v4_upgrade(df_detailed, end_use, percentile)
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
        installed_cost, _ = _calculate_v3_upgrade(df_copy, df_detailed, cost_dict, end_use, menu_mp, valid_mask, verbose)

    # Safety net: Ensure costs never go negative (defense against extreme extrapolation)
    installed_cost = installed_cost.clip(lower=0)

    # Update result series with calculated values (only for valid homes)
    result_series.loc[valid_mask] = installed_cost.loc[valid_mask].round(2)

    # Build column name using centralized utility
    cost_col = create_cost_col(menu_mp=menu_mp, category=end_use, cost_type='upgrade', cost_scenario=cost_scenario)

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
    if end_use not in VALID_CATEGORIES:
        raise ValueError(f"Invalid end_use: '{end_use}'. Must be one of {VALID_CATEGORIES}")
    if cost_scenario not in REMDB_COST_SCENARIO_KEYS:
        raise ValueError(
            f"Invalid cost_scenario: '{cost_scenario}'. "
            f"Must be one of {REMDB_COST_SCENARIO_KEYS}"
        )


# ========================================================================================================================================================================
# INTERNAL: REMDB V4 REGRESSION CALCULATION
# ========================================================================================================================================================================

def _calculate_v4_upgrade(
    df_detailed: pd.DataFrame,
    end_use: str,
    percentile: str
) -> pd.Series:
    """
    Calculate upgrade installed costs using REMDB v4 regression formula.

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
    prefix = f'{end_use}_upgrade_'

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
        raise KeyError(f"Missing columns: {missing}. Call add_remdb_upgrade_metrics() first.")

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

def _calculate_v3_upgrade(
    df_copy: pd.DataFrame,
    df_detailed: pd.DataFrame,
    cost_dict: dict,
    end_use: str,
    menu_mp: int,
    valid_mask: pd.Series,
    verbose: bool = VERBOSE
) -> Tuple[pd.Series, str]:
    """
    Calculate upgrade installed costs using REMDB v3 probabilistic sampling.

    Args:
        df_copy: DataFrame with home data.
        df_detailed: Detailed DataFrame (not used in v3; present for API consistency).
        cost_dict: Dictionary mapping (technology, efficiency) to cost distributions.
        end_use: Equipment category.
        menu_mp: Measure package number.
        valid_mask: Series indicating which rows have valid data.
        verbose: Whether to print progress.

    Returns:
        Tuple of (installed_cost Series, cost_column_name string).
    """
    # Get conditions, technology-efficiency pairs, and cost components
    params = _get_end_use_installation_parameters(df_copy, end_use, menu_mp)
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
            print(f"  After tech filtering: {len(valid_calculation_indices)} homes remain valid for {end_use} upgrade")

        if df_valid.empty:
            raise ValueError(f"Warning: No valid homes found for {end_use} upgrade cost calculation.")

        # Sample costs from distributions
        sampled_costs_dict = sample_costs_from_distributions(
            tech_filtered, eff_filtered, cost_dict, cost_components)

        # Calculate the installation cost per row
        installation_cost, cost_column_name = _calculate_installation_cost_per_row(
            df_valid, sampled_costs_dict, menu_mp, end_use)

        # Build result series
        result_series = create_retrofit_only_series(df_copy, valid_mask)
        result_series.loc[valid_calculation_indices] = np.round(installation_cost, 2)

        return result_series, cost_column_name

    except Exception as e:
        raise RuntimeError(f"Error in {end_use} upgrade cost calculation: {str(e)}")


# ========================================================================================================================================================================
# INTERNAL: V3 HELPER FUNCTIONS
# ========================================================================================================================================================================

def obtain_heating_system_specs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and process heating system specifications from input dataframe.

    Used to calculate total heating load. Code uses primary system capacity for heating and cooling costs.

    Args:
        df: Input dataframe containing heating system data.

    Returns:
        Updated dataframe with calculated heating system specs.

    Raises:
        ValueError: If dataframe is missing required columns.
    """
    necessary_columns = ['hvac_heating_efficiency', 'upgrade_hvac_heating_efficiency']
    if not all(column in df.columns for column in necessary_columns):
        raise ValueError("DataFrame does not contain all necessary columns.")

    # Extract AFUE from pattern "XX% AFUE"
    df['baseline_AFUE'] = df['hvac_heating_efficiency'].str.extract(r'(\d+\.?\d*)% AFUE').astype(float)

    # For Electric Baseboard with "XX% Efficiency" pattern
    mask = df['baseline_AFUE'].isna()
    efficiency_values = df.loc[mask, 'hvac_heating_efficiency'].str.extract(r'(\d+)% Efficiency')
    efficiency_float = pd.to_numeric(efficiency_values[0], errors='coerce')
    df.loc[mask & efficiency_float.notna(), 'baseline_AFUE'] = efficiency_float

    # SEER extraction from hvac_heating_efficiency
    df['baseline_SEER'] = df['hvac_heating_efficiency'].str.extract(r'SEER (\d+\.?\d*)').astype(float)

    # HSPF extraction from hvac_heating_efficiency
    df['baseline_HSPF'] = df['hvac_heating_efficiency'].str.extract(r'(\d+\.?\d*) HSPF').astype(float)

    # For upgrade_newInstall_HSPF, extract the HSPF value
    df['ugrade_newInstall_HSPF'] = df['upgrade_hvac_heating_efficiency'].str.extract(r'(\d+) HSPF').astype(float)

    return df


def calculate_heating_installation_premium(
    df: pd.DataFrame,
    menu_mp: int,
    cpi_ratio_2023_2013: float
) -> pd.DataFrame:
    """
    Calculate premium costs for heating system installation based on existing infrastructure.

    Adds costs for homes without existing central AC or with boiler systems that
    need additional modifications for heat pump installation.

    Args:
        df: Input dataframe containing heating system data.
        menu_mp: Menu package identifier.
        cpi_ratio_2023_2013: Consumer price index ratio for adjusting 2013 costs to 2023.

    Returns:
        Updated dataframe with heating installation premium costs.
    """
    necessary_columns = ['hvac_cooling_type', 'heating_type']
    if not all(column in df.columns for column in necessary_columns):
        raise ValueError("DataFrame does not contain all necessary columns.")

    for index, row in df.iterrows():
        premium_cost = 0

        if row['hvac_cooling_type'] != 'None':
            premium_cost = 0
        elif 'Furnace' in row['heating_type'] or 'Baseboard' in row['heating_type']:
            premium_cost = 400 * cpi_ratio_2023_2013
        elif 'Boiler' in row['heating_type']:
            premium_cost = 1500 * cpi_ratio_2023_2013

        adjusted_cost = round(premium_cost, 2)
        df.at[index, f'mp{menu_mp}_heating_installation_premium'] = adjusted_cost

    return df


def _get_end_use_installation_parameters(
    df: pd.DataFrame,
    end_use: str,
    menu_mp: int
) -> dict:
    """
    Retrieve parameters for installation cost calculations based on end use type.

    Returns conditions, technology-efficiency pairs, and cost components tailored
    to specific end uses like heating, water heating, clothes drying, and cooking.

    Args:
        df: Input dataframe with equipment data.
        end_use: Type of end use.
        menu_mp: Menu package identifier.

    Returns:
        Dictionary containing conditions, technology-efficiency pairs, and cost components.
    """
    parameters = {}

    if 'heating' in VALID_CATEGORIES:
        parameters['heating'] = {
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
        }

    if 'waterHeating' in VALID_CATEGORIES:
        parameters['waterHeating'] = {
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
        }

    if 'clothesDrying' in VALID_CATEGORIES:
        parameters['clothesDrying'] = {
            'conditions': [
                df['upgrade_clothes_dryer'].str.contains('Electric, Premium, Heat Pump, Ventless', na=False),
                ~df['upgrade_clothes_dryer'].str.contains('Electric, Premium, Heat Pump, Ventless', na=False),
            ],
            'tech_eff_pairs': [
                ('Electric HP Clothes Dryer', 5.2),
                ('Electric Clothes Dryer', 3.1),
            ],
            'cost_components': ['unitCost']
        }

    if 'cooking' in VALID_CATEGORIES:
        parameters['cooking'] = {
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

    return parameters[end_use]


def _calculate_installation_cost_per_row(
    df_valid: pd.DataFrame,
    sampled_costs_dict: dict,
    menu_mp: int,
    end_use: str
) -> tuple:
    """
    Calculate the installation cost for each row based on the end use type.

    Args:
        df_valid: Filtered DataFrame containing valid rows.
        sampled_costs_dict: Dictionary with sampled costs for each component.
        menu_mp: Menu option identifier.
        end_use: Type of end-use.

    Returns:
        Tuple of (installation_cost, cost_column_name).
    """
    try:
        if end_use == 'heating':
            if 'size_heating_system_primary_k_btu_h' not in df_valid.columns:
                raise ValueError("Required column 'size_heating_system_primary_k_btu_h' not found in DataFrame")

            required_components = ['unitCost', 'otherCost', 'cost_per_kBtuh']
            for comp in required_components:
                if comp not in sampled_costs_dict:
                    raise KeyError(f"Required cost component '{comp}' not found for heating calculation")

            installation_cost = (
                sampled_costs_dict['unitCost'] +
                sampled_costs_dict['otherCost'] +
                (df_valid['size_heating_system_primary_k_btu_h'] * sampled_costs_dict['cost_per_kBtuh']))
            cost_column_name = create_cost_col(menu_mp=menu_mp, category='heating', cost_type='upgrade', cost_scenario='v3')

        elif end_use == 'waterHeating':
            if 'size_water_heater_gal' not in df_valid.columns:
                raise ValueError("Required column 'size_water_heater_gal' not found in DataFrame")

            required_components = ['unitCost', 'cost_per_gallon']
            for comp in required_components:
                if comp not in sampled_costs_dict:
                    raise KeyError(f"Required cost component '{comp}' not found for water heating calculation")

            installation_cost = (
                sampled_costs_dict['unitCost'] +
                (sampled_costs_dict['cost_per_gallon'] * df_valid['size_water_heater_gal']))
            cost_column_name = create_cost_col(menu_mp=menu_mp, category='waterHeating', cost_type='upgrade', cost_scenario='v3')

        else:
            if 'unitCost' not in sampled_costs_dict:
                raise KeyError(f"Required cost component 'unitCost' not found for {end_use} calculation")

            installation_cost = sampled_costs_dict['unitCost']
            cost_column_name = create_cost_col(menu_mp=menu_mp, category=end_use, cost_type='upgrade', cost_scenario='v3')

        return installation_cost, cost_column_name

    except Exception as e:
        raise RuntimeError(f"Error calculating {end_use} installation cost: {str(e)}")
