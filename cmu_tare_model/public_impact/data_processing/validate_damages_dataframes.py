import pandas as pd
from typing import Dict, List, Tuple
from cmu_tare_model.utils.modeling_params import define_scenario_params

def validate_damage_dataframes(
    df_baseline_climate: pd.DataFrame,
    df_baseline_health: pd.DataFrame,
    df_mp_climate: pd.DataFrame,
    df_mp_health: pd.DataFrame,
    menu_mp: str,
    policy_scenario: str,
    base_year: int,
    equipment_specs: Dict[str, int],
    rcm_model: str,
    cr_function: str,
    check_health: bool = False,
    verbose: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate that the baseline and retrofit damage DataFrames have the expected columns
    before performing NPV calculations.
    
    Args:
        df_baseline_climate: DataFrame containing baseline climate damage projections.
        df_baseline_health: DataFrame containing baseline health damage projections.
        df_mp_climate: DataFrame containing post-retrofit climate damage projections.
        df_mp_health: DataFrame containing post-retrofit health damage projections.
        menu_mp: Menu identifier used to construct column names for the measure package.
        policy_scenario: Specifies the grid scenario.
        base_year: The base year for calculations.
        equipment_specs: Dictionary mapping equipment categories to their lifetimes.
        rcm_model: Reduced Complexity Model name (e.g., 'inmap', 'ap2', 'easiur').
        cr_function: Concentration-response function name (e.g., 'acs', 'h6c').
        check_health: Whether to validate health damage columns. Health is
            dormant (Session 3), so this defaults to False and only climate
            columns are required; set True to re-enable health validation.
        verbose: Whether to print detailed validation messages. Defaults to False.

    Returns:
        Tuple containing:
            - Boolean indicating if the DataFrames are valid
            - List of warning/error messages
    """
    scenario_prefix, _, _, _, _, _ = define_scenario_params(menu_mp, policy_scenario)
    messages = []
    
    # Track column existence for each DataFrame
    found_baseline_climate = False
    found_retrofit_climate = False
    # Health is dormant (Session 3): when not checking health, treat it as
    # already satisfied so climate-only validation can still pass.
    found_baseline_health = not check_health
    found_retrofit_health = not check_health
    
    # Example missing columns for reporting
    example_baseline_climate = ""
    example_baseline_health = ""
    example_retrofit_climate = ""
    example_retrofit_health = ""
    
    # Generate expected column patterns and check each DataFrame
    for category, lifetime in equipment_specs.items():
        for year in range(1, lifetime + 1):
            year_label = year + (base_year - 1)
            
            # Check climate damage columns (baseline)
            base_climate_col = f'baseline_{year_label}_{category}_damages_climate_lrmer_central'
            if base_climate_col in df_baseline_climate.columns:
                found_baseline_climate = True
            elif not example_baseline_climate:
                example_baseline_climate = base_climate_col
            
            # Check health damage columns (baseline) -- only when health is active
            if check_health:
                base_health_col = f'baseline_{year_label}_{category}_damages_health_{rcm_model}_{cr_function}'
                if base_health_col in df_baseline_health.columns:
                    found_baseline_health = True
                elif not example_baseline_health:
                    example_baseline_health = base_health_col
            
            # Check climate damage columns (retrofit)
            retrofit_climate_col = f'{scenario_prefix}{year_label}_{category}_damages_climate_lrmer_central'
            if retrofit_climate_col in df_mp_climate.columns:
                found_retrofit_climate = True
            elif not example_retrofit_climate:
                example_retrofit_climate = retrofit_climate_col
            
            # Check health damage columns (retrofit) -- only when health is active
            if check_health:
                retrofit_health_col = f'{scenario_prefix}{year_label}_{category}_damages_health_{rcm_model}_{cr_function}'
                if retrofit_health_col in df_mp_health.columns:
                    found_retrofit_health = True
                elif not example_retrofit_health:
                    example_retrofit_health = retrofit_health_col
            
            # Once we find at least one column of each type, we can break
            if (found_baseline_climate and found_baseline_health and 
                found_retrofit_climate and found_retrofit_health):
                break
        
        # If we found all column types, no need to check other categories
        if (found_baseline_climate and found_baseline_health and 
            found_retrofit_climate and found_retrofit_health):
            break
        
    # Return True if all expected column patterns were found
    is_valid = (found_baseline_climate and found_baseline_health and 
                found_retrofit_climate and found_retrofit_health)
    
    if verbose or not is_valid:
        ok_label = "[OK] Valid"
        missing_label = "[MISSING] expected columns"
        print(f"Baseline Climate DataFrame: "
              f"{ok_label if found_baseline_climate else missing_label}")
        print(f"Retrofit Climate DataFrame: "
              f"{ok_label if found_retrofit_climate else missing_label}")
        if check_health:
            print(f"Baseline Health DataFrame: "
                  f"{ok_label if found_baseline_health else missing_label}")
            print(f"Retrofit Health DataFrame: "
                  f"{ok_label if found_retrofit_health else missing_label}")

    # Generate warning messages for missing columns
    if not found_baseline_climate:
        messages.append(f"ERROR: No baseline climate damage columns found. Expected pattern: 'baseline_YEAR_CATEGORY_damages_climate_lrmer_BOUND'")
        messages.append(f"Example missing column: {example_baseline_climate}")
    
    if not found_baseline_health:
        messages.append(f"ERROR: No baseline health damage columns found. Expected pattern: 'baseline_YEAR_CATEGORY_damages_health_MODEL_FUNCTION'")
        messages.append(f"Example missing column: {example_baseline_health}")
    
    if not found_retrofit_climate:
        messages.append(f"ERROR: No retrofit climate damage columns found. Expected pattern: '{scenario_prefix}YEAR_CATEGORY_damages_climate_lrmer_BOUND'")
        messages.append(f"Example missing column: {example_retrofit_climate}")
    
    if not found_retrofit_health:
        messages.append(f"ERROR: No retrofit health damage columns found. Expected pattern: '{scenario_prefix}YEAR_CATEGORY_damages_health_MODEL_FUNCTION'")
        messages.append(f"Example missing column: {example_retrofit_health}")
    
    return is_valid, messages
