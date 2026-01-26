import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional

# Constants
from cmu_tare_model.constants import EQUIPMENT_SPECS, CR_FUNCTIONS, RCM_MODELS, SCC_ASSUMPTIONS, VERBOSE
from cmu_tare_model.utils.discounting import calculate_discount_factors, PUBLIC_DISCOUNTING_METHOD_SUFFIXES
from cmu_tare_model.utils.modeling_params import define_scenario_params
from cmu_tare_model.utils.validation_framework import (
    calculate_avoided_values,
    create_retrofit_only_series,
    initialize_validation_tracking,
    replace_small_values_with_nan,
)
from cmu_tare_model.utils.calculation_utils import (
    validate_common_parameters,
    apply_temporary_validation_and_mask
)
from cmu_tare_model.public_impact.data_processing.validate_damages_dataframes import validate_damage_dataframes


def _sum_yearly_damages(
    yearly_damages: List[pd.Series],
    template_series: pd.Series,
    valid_mask: pd.Series,
    menu_mp: int
) -> pd.Series:
    """Sum yearly avoided damages into total NPV with proper masking.
    
    Args:
        yearly_damages: List of Series containing yearly avoided damages.
        template_series: Template with zeros for valid homes, NaN otherwise.
        valid_mask: Boolean Series indicating valid homes.
        menu_mp: Measure package ID (0=baseline, >0=retrofit).
        
    Returns:
        Total NPV Series with validation masking applied.
    """
    if yearly_damages:
        # Convert list of Series to DataFrame and sum
        damages_df = pd.concat(yearly_damages, axis=1)
        npv = damages_df.sum(axis=1, skipna=False)
        
        # Apply validation mask for measure packages
        if menu_mp != 0:
            npv = pd.Series(
                np.where(valid_mask, npv, np.nan),
                index=npv.index
            )
    else:
        npv = template_series
    
    return npv


def calculate_climate_npv(
    df_copy: pd.DataFrame,
    df_baseline_climate: pd.DataFrame,
    df_mp_climate: pd.DataFrame,
    menu_mp: int,
    policy_scenario: str,
    base_year: int,
    all_columns_to_mask: Dict[str, List[str]],
    verbose: bool = VERBOSE
) -> Dict[str, pd.Series]:
    """
    Calculate climate NPV for all equipment categories and SCC assumptions.
    - Only calculate Climate NPV ONCE: Does NOT vary by CR function or RCM model
    - NO additional discounting: Climate damages SCC values are already discounted back to the specific emission-year SCC.
    
    Args:
        df_copy: DataFrame for validation tracking.
        df_baseline_climate: Baseline climate damage projections.
        df_mp_climate: Post-retrofit climate damage projections.
        menu_mp: Measure package identifier.
        policy_scenario: Policy scenario name.
        base_year: Base year for calculations.
        all_columns_to_mask: Dictionary tracking columns for masking.
        verbose: Whether to print progress messages.
        
    Returns:
        Dictionary mapping column names to climate NPV Series (unrounded).
    """
    scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)
    
    all_npvs: Dict[str, pd.Series] = {}
    
    for category, lifetime in EQUIPMENT_SPECS.items():
        if verbose:
            print(f"    Climate NPV for {category}...")
        
        df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = \
            initialize_validation_tracking(df_copy, category, menu_mp, verbose=False)
        
        climate_npv_template = create_retrofit_only_series(df_copy, valid_mask)
        
        for scc in SCC_ASSUMPTIONS:
            climate_npv_key = f'{scenario_prefix}{category}_climate_npv_{scc}'
            yearly_climate_avoided = []
            
            for year in range(1, lifetime + 1):
                year_label = year + (base_year - 1)
                
                base_climate_col = f'baseline_{year_label}_{category}_damages_climate_lrmer_{scc}'
                retrofit_climate_col = f'{scenario_prefix}{year_label}_{category}_damages_climate_lrmer_{scc}'
                
                if (base_climate_col in df_baseline_climate.columns and 
                    retrofit_climate_col in df_mp_climate.columns):
                    
                    # No discount factor - SCC already represents NPV of future damages
                    avoided_climate = calculate_avoided_values(
                        baseline_values=df_baseline_climate[base_climate_col],
                        measure_values=df_mp_climate[retrofit_climate_col],
                        retrofit_mask=(valid_mask if menu_mp != 0 else None)
                    )
                    yearly_climate_avoided.append(avoided_climate)
            
            # Sum yearly avoided climate damages (scc was already discounted) into total NPV
            climate_npv = _sum_yearly_damages(
                yearly_damages=yearly_climate_avoided,
                template_series=climate_npv_template,
                valid_mask=valid_mask,
                menu_mp=menu_mp
            )
            
            climate_npv = replace_small_values_with_nan(climate_npv)
        
            all_npvs[climate_npv_key] = climate_npv
            
            if climate_npv_key not in category_columns_to_mask:
                category_columns_to_mask.append(climate_npv_key)
        
        all_columns_to_mask[category].extend(category_columns_to_mask)
    
    return all_npvs


def calculate_health_npv(
    df_copy: pd.DataFrame,
    df_baseline_health: pd.DataFrame,
    df_mp_health: pd.DataFrame,
    menu_mp: int,
    policy_scenario: str,
    rcm_model: str,
    cr_function: str,
    base_year: int,
    discount_rate_col: str,
    all_columns_to_mask: Dict[str, List[str]],
    verbose: bool = VERBOSE
) -> Dict[str, pd.Series]:
    """
    Calculate health NPV for all equipment categories for a specific CR function.
    
    Health NPV varies by RCM model and CR function. Unlike climate, health damages 
    require discounting because they represent annual marginal social costs.
    
    Args:
        df_copy: DataFrame for validation tracking.
        df_baseline_health: Baseline health damage projections.
        df_mp_health: Post-retrofit health damage projections.
        menu_mp: Measure package identifier.
        policy_scenario: Policy scenario name.
        rcm_model: RCM model name.
        cr_function: Concentration-response function name.
        base_year: Base year for discounting.
        discount_rate_col: Discount rate column name.
        all_columns_to_mask: Dictionary tracking columns for masking.
        verbose: Whether to print progress messages.
        
    Returns:
        Dictionary mapping column names to health NPV Series (unrounded).
    """
    scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)
    max_lifetime = max(EQUIPMENT_SPECS.values())
    
    # Pre-calculate discount factors
    discount_factors: Dict[int, pd.Series] = {}
    for year in range(1, max_lifetime + 1):
        year_label = year + (base_year - 1)
        discount_factors[year_label] = calculate_discount_factors(
            df=df_copy,
            base_year=base_year,
            target_year=year_label,
            discount_rate_col=discount_rate_col
        )
    
    all_npvs: Dict[str, pd.Series] = {}
    
    for category, lifetime in EQUIPMENT_SPECS.items():
        if verbose:
            print(f"      Health NPV for {category}...")
        
        # ===== STEP 1: Initialize validation tracking =====
        # Moved outside of SCC loop: Validation only depends on category, not SCC
        # MEMORY OPTIMIZATION: copy=False since df_copy was already copied at the start
        _, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
            df_copy, category, menu_mp, verbose=verbose, copy=False)
        
        # # ===== STEP 2: Initialize result series for health NPV =====
        health_npv_template = create_retrofit_only_series(df_copy, valid_mask)
        health_npv_key = f'{scenario_prefix}{category}_health_npv_{rcm_model}_{cr_function}'
        yearly_health_avoided = []
        
        # ===== STEP 3 & 4: Valid-Only Calculation and Updates =====
        for year in range(1, lifetime + 1):
            year_label = year + (base_year - 1)
            discount_factor = discount_factors[year_label]
            
            base_health_col = f'baseline_{year_label}_{category}_damages_health_{rcm_model}_{cr_function}'
            retrofit_health_col = f'{scenario_prefix}{year_label}_{category}_damages_health_{rcm_model}_{cr_function}'
            
            if (base_health_col in df_baseline_health.columns and 
                retrofit_health_col in df_mp_health.columns):
                
                avoided_health = calculate_avoided_values(
                    baseline_values=df_baseline_health[base_health_col],
                    measure_values=df_mp_health[retrofit_health_col],
                    retrofit_mask=(valid_mask if menu_mp != 0 else None)
                ) * discount_factor
                
                yearly_health_avoided.append(avoided_health)
        
        # Sum discounted yearly avoided health damages into total NPV
        health_npv = _sum_yearly_damages(
            yearly_damages=yearly_health_avoided,
            template_series=health_npv_template,
            valid_mask=valid_mask,
            menu_mp=menu_mp
        )
        
        health_npv = replace_small_values_with_nan(health_npv)

        all_npvs[health_npv_key] = health_npv
        
        if health_npv_key not in category_columns_to_mask:
            category_columns_to_mask.append(health_npv_key)
        
        all_columns_to_mask[category].extend(category_columns_to_mask)
    
    return all_npvs


def calculate_public_npv(
    df: pd.DataFrame, 
    df_baseline_climate: pd.DataFrame, 
    df_baseline_health: pd.DataFrame,
    df_mp_climate: pd.DataFrame,
    df_mp_health: pd.DataFrame,
    menu_mp: int, 
    policy_scenario: str, 
    rcm_model: str,
    discount_rate_col: str = 'public_discount_rate',
    base_year: int = 2024,
    verbose: bool = VERBOSE
) -> pd.DataFrame:
    """
    Calculate the public Net Present Value (NPV) for climate and health damages.

    This function compares baseline damages with post-retrofit damages to determine
    avoided damages (benefits) from implementing retrofits.
    
    Key change from v2.2: Climate NPV is calculated ONCE (outside CR function loop)
    since it doesn't vary by CR function. This prevents duplicate column creation.

    Args:
        df: Input DataFrame containing base data for calculations.
        df_baseline_climate: DataFrame containing baseline climate damage projections.
        df_baseline_health: DataFrame containing baseline health damage projections.
        df_mp_climate: DataFrame containing post-retrofit climate damage projections.
        df_mp_health: DataFrame containing post-retrofit health damage projections.
        menu_mp: Menu identifier for the measure package.
        policy_scenario: Policy scenario for grid projections.
        rcm_model: Reduced Complexity Model for health impact calculations.
        discount_rate_col: Column name for discount rate. Default 'public_discount_rate'.
        base_year: Base year for discounting. Default 2024.
        verbose: Whether to print progress messages.

    Returns:
        DataFrame with calculated public NPV columns.
        
    Raises:
        ValueError: If input parameters are invalid or required columns are missing.
    """
    # ===== STEP 0: Validate input parameters =====
    menu_mp, policy_scenario = validate_common_parameters(menu_mp, policy_scenario)
    
    if rcm_model not in RCM_MODELS:
        raise ValueError(f"Invalid rcm_model: {rcm_model}. Must be one of {RCM_MODELS}")
    
    if verbose:
        print("\nValidating input data structure...")
    
    is_valid, messages = validate_damage_dataframes(
        df_baseline_climate, df_baseline_health,
        df_mp_climate, df_mp_health,
        menu_mp, policy_scenario, base_year, EQUIPMENT_SPECS
    )
    
    if verbose and messages:
        for message in messages:
            print(message)
    
    if not is_valid:
        raise ValueError("Input DataFrames are missing required damage columns. See errors above.")
    
    if verbose:
        print("✓ Input data validation passed.")

    # MEMORY OPTIMIZATION: Only copy the main DataFrame since we'll add columns to it.
    # The baseline and measure package DataFrames are read-only, so no copy needed.
    df_copy = df.copy()
    
    # Use references to read-only DataFrames (no copy needed - saves ~4.9 GB for large datasets)
    df_baseline_climate_ref = df_baseline_climate
    df_baseline_health_ref = df_baseline_health
    df_mp_climate_ref = df_mp_climate
    df_mp_health_ref = df_mp_health
    
    all_columns_to_mask = {category: [] for category in EQUIPMENT_SPECS}
    df_new_columns = pd.DataFrame(index=df_copy.index)
    
    scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)

    # ===== Calculate climate NPV ONCE (doesn't vary by CR function) =====
    if verbose:
        print("\nCalculating climate NPV (independent of CR function)...")
    
    climate_npvs = calculate_climate_npv(
        df_copy=df_copy,
        df_baseline_climate=df_baseline_climate_ref,
        df_mp_climate=df_mp_climate_ref,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        base_year=base_year,
        all_columns_to_mask=all_columns_to_mask,
        verbose=verbose
    )
    
    if climate_npvs:
        temp_df = pd.DataFrame(climate_npvs, index=df_copy.index)
        df_new_columns = pd.concat([df_new_columns, temp_df], axis=1)

    # ===== Calculate health NPV for each CR function =====
    if verbose:
        print("\nCalculating health NPV for each CR function...")
    
    for cr_function in CR_FUNCTIONS:
        if verbose:
            print(f"  Processing CR Function: {cr_function}")
        
        health_npvs = calculate_health_npv(
            df_copy=df_copy,
            df_baseline_health=df_baseline_health_ref,
            df_mp_health=df_mp_health_ref,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            rcm_model=rcm_model,
            cr_function=cr_function,
            base_year=base_year,
            discount_rate_col=discount_rate_col,
            all_columns_to_mask=all_columns_to_mask,
            verbose=verbose
        )
        
        if health_npvs:
            temp_df = pd.DataFrame(health_npvs, index=df_copy.index)
            df_new_columns = pd.concat([df_new_columns, temp_df], axis=1)

    # ===== Calculate combined public NPV (climate + health) =====
    if verbose:
        print("\nCalculating combined public NPV...")
    
    for category in EQUIPMENT_SPECS.keys():
        category_columns_to_mask = []
        
        # Get the climate NPV key once (same for all CR functions)
        for scc in SCC_ASSUMPTIONS:
            climate_npv_key = f'{scenario_prefix}{category}_climate_npv_{scc}'

            # Now loop over CR functions to create health npv key and combined public NPV key
            for cr_function in CR_FUNCTIONS:
                health_npv_key = f'{scenario_prefix}{category}_health_npv_{rcm_model}_{cr_function}'
                public_npv_key = f'{scenario_prefix}{category}_public_npv_{scc}_{rcm_model}_{cr_function}'
                
                # Check if both climate and health NPV columns exist, then calculate combined public NPV
                if climate_npv_key in df_new_columns.columns and health_npv_key in df_new_columns.columns:
                    # Sum unrounded values first, then round
                    public_npv = (
                        df_new_columns[climate_npv_key] + 
                        df_new_columns[health_npv_key]
                    )
                    
                    public_npv_rounded = public_npv.round(2)

                    df_new_columns[public_npv_key] = public_npv_rounded
                    category_columns_to_mask.append(public_npv_key)
        
        all_columns_to_mask[category].extend(category_columns_to_mask)

    # ===== Apply deferred rounding to climate and health columns =====
    # This matches original behavior: public_npv calculated from unrounded values
    for col in df_new_columns.columns:
        if '_climate_npv_' in col or '_health_npv_' in col:
            df_new_columns[col] = df_new_columns[col].round(2)

    # ===== Apply final masking =====
    if verbose:
        print("\nApplying final masking to calculated columns...")
        
    df_result = apply_temporary_validation_and_mask(
        df_copy, df_new_columns, all_columns_to_mask, verbose=verbose
    )
    
    if verbose:
        print(f"\nNPV calculation completed. Added {len(df_new_columns.columns)} new columns.")
        
    return df_result
