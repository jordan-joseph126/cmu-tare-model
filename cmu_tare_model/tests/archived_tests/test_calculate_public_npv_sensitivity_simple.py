import pytest
import pandas as pd
import numpy as np
from cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity import (
    calculate_public_npv,
    calculate_lifetime_damages_grid_scenario
)
from cmu_tare_model.constants import EQUIPMENT_SPECS, SCC_ASSUMPTIONS, RCM_MODELS, CR_FUNCTIONS

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_define_scenario_settings(monkeypatch):
    """
    Mock replacement for define_scenario_params.
    Given a menu_mp and policy_scenario, returns a tuple:
        (scenario_prefix, cambium_scenario, lookup1, lookup2, lookup3)
    """
    def mock_function(menu_mp, policy_scenario):
        if menu_mp == "0":
            scenario_prefix = "baseline_"
        elif policy_scenario == "No Inflation Reduction Act":
            scenario_prefix = f"preIRA_mp{menu_mp}_"
        elif policy_scenario == "AEO2023 Reference Case":
            scenario_prefix = f"iraRef_mp{menu_mp}_"
        else:
            raise ValueError(f"Invalid Policy Scenario: {policy_scenario}")
        return scenario_prefix, "MidCase", {}, {}, {}, {}
    
    monkeypatch.setattr(
        "cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.define_scenario_params",
        mock_function
    )
    return mock_function


@pytest.fixture
def create_damage_columns():
    """
    Helper fixture that returns a function to create damage columns for DataFrames.
    This centralizes column creation logic to reduce duplication.
    """
    def _create_columns(df, scenario_prefix, base_year, value_multiplier=1.0):
        """
        Creates damage columns with the specified scenario_prefix for each equipment category.
        
        Args:
            df: DataFrame to add columns to
            scenario_prefix: Column name scenario_prefix
            base_year: Base year for calculations
            value_multiplier: Multiplier for damage values (default 1.0)
        """
        # Create a dictionary to hold all the new columns
        new_columns = {}
        
        for category, lifetime in EQUIPMENT_SPECS.items():
            for year in range(1, lifetime + 1):
                year_label = year + (base_year - 1)
                
                # Climate damage columns
                for scc in SCC_ASSUMPTIONS:
                    col_name = f'{scenario_prefix}{year_label}_{category}_damages_climate_lrmer_{scc}'
                    new_columns[col_name] = [value_multiplier * 100.0 * i for i in range(1, len(df.index) + 1)]
                
                # Health damage columns
                for rcm in RCM_MODELS:
                    for cr in CR_FUNCTIONS:
                        col_name = f'{scenario_prefix}{year_label}_{category}_damages_health_{rcm}_{cr}'
                        new_columns[col_name] = [value_multiplier * 50.0 * i for i in range(1, len(df.index) + 1)]
        
        # Create a new DataFrame with all columns and concatenate it with original
        new_df = pd.DataFrame(new_columns, index=df.index)
        result_df = pd.concat([df, new_df], axis=1)
        return result_df
    
    return _create_columns


@pytest.fixture
def sample_dataframes(mock_define_scenario_settings, create_damage_columns):
    """
    Creates sample input DataFrames for testing calculate_public_npv.
    Returns a tuple: (df, df_baseline_damages, df_mp_damages).
    """
    base_year = 2024
    
    # Create base DataFrames with simple indices
    df = pd.DataFrame({'id': [1, 2, 3]}).set_index('id')
    df_baseline_damages = pd.DataFrame({'id': [1, 2, 3]}).set_index('id')
    df_mp_damages = pd.DataFrame({'id': [1, 2, 3]}).set_index('id')
    
    # Add damage columns to baseline DataFrame
    df_baseline_damages = create_damage_columns(df_baseline_damages, "baseline_", base_year)
    
    # Add damage columns to measure-package DataFrame
    scenario_prefix, _, _, _, _, _ = mock_define_scenario_settings("8", "AEO2023 Reference Case")
    df_mp_damages = create_damage_columns(df_mp_damages, scenario_prefix, base_year, 0.8)  # 20% avoided damages
    
    return df, df_baseline_damages, df_mp_damages


@pytest.fixture
def mock_discount_factor(monkeypatch):
    """
    Mock the calculate_discount_factor function to return predictable values for testing.
    Makes testing discount calculations easier and more deterministic.
    """
    def mock_function(base_year, year_label, discounting_method):
        if discounting_method not in ['public', 'private_fixed']:
            raise ValueError(f"Invalid discounting method: {discounting_method}")
        
        if year_label == base_year:
            return 1.0  # No discounting for base year
        
        years_diff = year_label - base_year
        if discounting_method == 'public':
            return 1.0 / ((1 + 0.02) ** years_diff)  # 2% discount rate
        else:  # private_fixed
            return 1.0 / ((1 + 0.07) ** years_diff)  # 7% discount rate
    
    monkeypatch.setattr(
        "cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.calculate_discount_factor",
        mock_function
    )
    return mock_function


@pytest.fixture
def custom_equipment_specs(monkeypatch):
    """
    Fixture for temporarily modifying EQUIPMENT_SPECS for specific test cases.
    """
    def _custom_specs(specs_update):
        test_specs = EQUIPMENT_SPECS.copy()
        test_specs.update(specs_update)
        monkeypatch.setattr(
            "cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.EQUIPMENT_SPECS",
            test_specs
        )
        return test_specs
    
    return _custom_specs

# ============================================================================
# Helper Functions for Tests
# ============================================================================

def add_special_category_damages(df_baseline_damages, df_mp_damages, scenario_prefix, category, base_year, multiplier=0.8):
    """
    Helper function to add damage columns for special test categories.
    """
    # Create dictionaries to hold all new columns
    baseline_columns = {}
    mp_columns = {}
    
    for scc in SCC_ASSUMPTIONS:
        baseline_col = f'baseline_{base_year}_{category}_damages_climate_lrmer_{scc}'
        baseline_columns[baseline_col] = [100.0 * i for i in range(1, len(df_baseline_damages) + 1)]
        
        mp_col = f'{scenario_prefix}{base_year}_{category}_damages_climate_lrmer_{scc}'
        mp_columns[mp_col] = [multiplier * 100.0 * i for i in range(1, len(df_mp_damages) + 1)]
    
    for rcm in RCM_MODELS:
        for cr in CR_FUNCTIONS:
            baseline_col = f'baseline_{base_year}_{category}_damages_health_{rcm}_{cr}'
            baseline_columns[baseline_col] = [50.0 * i for i in range(1, len(df_baseline_damages) + 1)]
            
            mp_col = f'{scenario_prefix}{base_year}_{category}_damages_health_{rcm}_{cr}'
            mp_columns[mp_col] = [multiplier * 50.0 * i for i in range(1, len(df_mp_damages) + 1)]

    # Add all columns at once
    baseline_new_df = pd.DataFrame(baseline_columns, index=df_baseline_damages.index)
    mp_new_df = pd.DataFrame(mp_columns, index=df_mp_damages.index)
    
    # Concatenate with original DataFrames
    result_baseline_damages = pd.concat([df_baseline_damages, baseline_new_df], axis=1)
    result_mp_damages = pd.concat([df_mp_damages, mp_new_df], axis=1)
    
    # Return updated DataFrames
    return result_baseline_damages, result_mp_damages


def verify_npv_values(result, scenario_prefix, category, scc, rcm_model, cr_function):
    """
    Helper function to verify NPV values are calculated and formatted correctly.
    """
    climate_npv_col = f'{scenario_prefix}{category}_climate_npv_{scc}'
    health_npv_col = f'{scenario_prefix}{category}_health_npv_{rcm_model}_{cr_function}'
    public_npv_col = f'{scenario_prefix}{category}_public_npv_{scc}_{rcm_model}_{cr_function}'
    
    # Check columns exist
    assert all(col in result.columns for col in [climate_npv_col, health_npv_col, public_npv_col])
    
    # Check values are rounded to 2 decimal places
    for col in [climate_npv_col, health_npv_col, public_npv_col]:
        for value in result[col].dropna():
            assert value == round(value, 2), f"Value in {col} is not rounded to 2 decimal places"
    
    # Check public NPV equals climate NPV + health NPV
    for i in range(len(result)):
        if not (pd.isna(result[climate_npv_col].iloc[i]) or 
                pd.isna(result[health_npv_col].iloc[i]) or
                pd.isna(result[public_npv_col].iloc[i])):
            
            expected_sum = result[climate_npv_col].iloc[i] + result[health_npv_col].iloc[i]
            # Allow for small differences due to rounding
            assert abs(result[public_npv_col].iloc[i] - expected_sum) <= 0.02, f"Public NPV differs significantly from climate + health NPV"

# ============================================================================
# Tests for calculate_public_npv
# ============================================================================

@pytest.mark.parametrize("menu_mp, policy_scenario, rcm_model, cr_function, discounting_method", [
    ("0", "No Inflation Reduction Act", RCM_MODELS[0], CR_FUNCTIONS[0], "public"),
    ("8", "AEO2023 Reference Case", RCM_MODELS[0], CR_FUNCTIONS[0], "public"),
    ("8", "AEO2023 Reference Case", RCM_MODELS[1], CR_FUNCTIONS[1], "private_fixed"),
])
def test_calculate_public_npv_successful_execution(
    sample_dataframes, 
    mock_define_scenario_settings, 
    mock_discount_factor,
    menu_mp, 
    policy_scenario, 
    rcm_model, 
    cr_function, 
    discounting_method
):
    """
    Test that calculate_public_npv executes successfully with valid inputs.
    Verifies expected columns are created and values are properly calculated and rounded.
    """
    df, df_baseline_damages, df_mp_damages = sample_dataframes
    
    result = calculate_public_npv(
        df, 
        df_baseline_damages, 
        df_mp_damages, 
        menu_mp, 
        policy_scenario, 
        rcm_model, 
        cr_function, 
        base_year=2024, 
        discounting_method=discounting_method
    )
    
    # Check that the result is a DataFrame with the same index as input df
    assert isinstance(result, pd.DataFrame)
    assert result.index.equals(df.index)
    
    # Determine expected scenario scenario_prefix based on inputs
    scenario_prefix, _, _, _, _, _ = mock_define_scenario_settings(menu_mp, policy_scenario)
    
    # Check NPV columns and values for each equipment category
    for category in EQUIPMENT_SPECS.keys():
        for scc in SCC_ASSUMPTIONS:
            verify_npv_values(result, scenario_prefix, category, scc, rcm_model, cr_function)


@pytest.mark.parametrize("param_name, invalid_value, error_type", [
    ("menu_mp", "invalid_mp", ValueError),
    ("policy_scenario", "Invalid Policy", ValueError),
    ("discounting_method", "invalid_method", ValueError),
])
def test_invalid_parameters(sample_dataframes, mock_define_scenario_settings, mock_discount_factor, param_name, invalid_value, error_type):
    """
    Test that invalid parameters cause the function to raise appropriate exceptions.
    """
    df, df_baseline_damages, df_mp_damages = sample_dataframes
    
    # Default valid parameters
    params = {
        "df": df,
        "df_baseline_damages": df_baseline_damages,
        "df_mp_damages": df_mp_damages,
        "menu_mp": "8",
        "policy_scenario": "AEO2023 Reference Case",
        "rcm_model": RCM_MODELS[0],
        "cr_function": CR_FUNCTIONS[0],
        "discounting_method": "public"
    }
    
    # Override the specified parameter with an invalid value
    params[param_name] = invalid_value
    
    with pytest.raises(error_type):
        calculate_public_npv(**params)


@pytest.mark.parametrize("test_case, column_type", [
    ("baseline", "damages_climate_lrmer"),
    ("mp", "damages_health"),
])
def test_missing_columns(sample_dataframes, mock_define_scenario_settings, mock_discount_factor, test_case, column_type):
    """
    Test that missing required columns in the DataFrames causes the function to fail gracefully.
    """
    df, df_baseline_damages, df_mp_damages = sample_dataframes
    
    if test_case == "baseline":
        # Remove a climate damage column from the baseline DataFrame
        columns_to_drop = [col for col in df_baseline_damages.columns if column_type in col]
        df_modified = df_baseline_damages.drop(columns=[columns_to_drop[0]])
        
        # Function should handle missing columns by skipping them
        result = calculate_public_npv(
            df, 
            df_modified, 
            df_mp_damages, 
            "8", 
            "AEO2023 Reference Case", 
            RCM_MODELS[0], 
            CR_FUNCTIONS[0]
        )
    else:  # mp case
        # Remove a health damage column from the mp DataFrame
        columns_to_drop = [col for col in df_mp_damages.columns if column_type in col]
        df_modified = df_mp_damages.drop(columns=[columns_to_drop[0]])
        
        # Function should handle missing columns by skipping them
        result = calculate_public_npv(
            df, 
            df_baseline_damages, 
            df_modified, 
            "8", 
            "AEO2023 Reference Case", 
            RCM_MODELS[0], 
            CR_FUNCTIONS[0]
        )
    
    # Verify the function completed and the result is a DataFrame
    assert isinstance(result, pd.DataFrame)


def test_non_numeric_damage_values(sample_dataframes, mock_define_scenario_settings, mock_discount_factor):
    """
    Test that non-numeric damage values in the DataFrames cause the function to raise an exception.
    """
    df, df_baseline_damages, df_mp_damages = sample_dataframes
    
    # Set one of the baseline damage columns to a non-numeric value
    col_to_modify = [col for col in df_baseline_damages.columns if 'damages_climate_lrmer' in col][0]
    df_baseline_nonnumeric = df_baseline_damages.copy()
    df_baseline_nonnumeric[col_to_modify] = "non-numeric"
    
    with pytest.raises(Exception):
        calculate_public_npv(
            df, 
            df_baseline_nonnumeric, 
            df_mp_damages, 
            "8", 
            "AEO2023 Reference Case", 
            RCM_MODELS[0], 
            CR_FUNCTIONS[0]
        )


def test_index_misalignment(sample_dataframes, mock_define_scenario_settings, mock_discount_factor):
    """
    Test that index misalignment between DataFrames is handled properly.
    """
    df, df_baseline_damages, df_mp_damages = sample_dataframes
    
    # Create a DataFrame with a different index
    df_misaligned = pd.DataFrame({'id': [4, 5, 6]})
    df_misaligned.set_index('id', inplace=True)
    
    # When df's index doesn't match the damage DataFrames, result should follow df's index
    result = calculate_public_npv(
        df_misaligned, 
        df_baseline_damages, 
        df_mp_damages, 
        "8", 
        "AEO2023 Reference Case", 
        RCM_MODELS[0], 
        CR_FUNCTIONS[0]
    )
    
    # Verify the result has the same index as df_misaligned
    assert result.index.equals(df_misaligned.index)
    
    # All NPV values should be NaN for the misaligned indices
    for col in result.columns:
        if any(term in col for term in ['_npv_', '_climate_', '_health_', '_public_']):
            assert result[col].isna().all(), f"Column {col} should contain only NaN values"


@pytest.mark.parametrize("category, lifetime, expected_behavior", [
    ("zero_lifetime_cat", 0, "zeros"),
    ("negative_lifetime_cat", -1, "handle_gracefully"),
])
def test_boundary_lifetime_equipment(
    sample_dataframes, 
    mock_define_scenario_settings, 
    mock_discount_factor, 
    custom_equipment_specs,
    category, 
    lifetime, 
    expected_behavior
):
    """
    Test boundary conditions for equipment with zero or negative lifetime.
    """
    df, df_baseline_damages, df_mp_damages = sample_dataframes
    
    # Create test categories with specified lifetimes
    custom_equipment_specs({category: lifetime})
    
    # Function should handle boundary cases appropriately
    result = calculate_public_npv(
        df, 
        df_baseline_damages, 
        df_mp_damages, 
        "8", 
        "AEO2023 Reference Case", 
        RCM_MODELS[0], 
        CR_FUNCTIONS[0]
    )
    
    # Verify the function completed and the result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # For zero lifetime, check that columns exist but contain zeros if they're present
    if expected_behavior == "zeros":
        scenario_prefix, _, _, _, _, _ = mock_define_scenario_settings("8", "AEO2023 Reference Case")
        for scc in SCC_ASSUMPTIONS:
            climate_col = f'{scenario_prefix}{category}_climate_npv_{scc}'
            if climate_col in result.columns:
                assert (result[climate_col] == 0.0).all(), f"Expected zeros for {climate_col}"


@pytest.mark.parametrize("category, lifetime", [
    ("base_year_cat", 1),
    ("min_lifetime_cat", 1),
])
def test_single_year_equipment(
    sample_dataframes, 
    mock_define_scenario_settings, 
    mock_discount_factor, 
    custom_equipment_specs,
    category, 
    lifetime
):
    """
    Test cases where equipment has a single year lifetime (boundary condition).
    """
    df, df_baseline_damages, df_mp_damages = sample_dataframes
    base_year = 2024
    
    # Create test category with specified lifetime
    custom_equipment_specs({category: lifetime})
    
    # Add damage columns for this category
    scenario_prefix, _, _, _, _, _ = mock_define_scenario_settings("8", "AEO2023 Reference Case")
    df_baseline_damages, df_mp_damages = add_special_category_damages(df_baseline_damages, df_mp_damages, scenario_prefix, category, base_year)

    # Calculate NPV for this category
    result = calculate_public_npv(
        df, 
        df_baseline_damages, 
        df_mp_damages, 
        "8", 
        "AEO2023 Reference Case", 
        RCM_MODELS[0], 
        CR_FUNCTIONS[0],
        base_year=base_year
    )
    
    # Verify NPV columns exist and values are calculated correctly
    for scc in SCC_ASSUMPTIONS[:1]:  # Test just one SCC value for simplicity
        verify_npv_values(result, scenario_prefix, category, scc, RCM_MODELS[0], CR_FUNCTIONS[0])
        
        # Check NPV values match expected for single-year case (no discounting)
        climate_npv_col = f'{scenario_prefix}{category}_climate_npv_{scc}'
        health_npv_col = f'{scenario_prefix}{category}_health_npv_{RCM_MODELS[0]}_{CR_FUNCTIONS[0]}'
        
        for i in range(len(df)):
            climate_avoided = 100.0 * (i + 1) - 80.0 * (i + 1)  # baseline - mp
            health_avoided = 50.0 * (i + 1) - 40.0 * (i + 1)
            
            assert result[climate_npv_col].iloc[i] == climate_avoided
            assert result[health_npv_col].iloc[i] == health_avoided


def test_overlapping_columns_handling(sample_dataframes, mock_define_scenario_settings, mock_discount_factor):
    """
    Test that calculate_public_npv correctly handles overlapping columns.
    """
    df, df_baseline_damages, df_mp_damages = sample_dataframes
    
    # Add a column to df that will overlap with a result column
    scenario_prefix, _, _, _, _, _ = mock_define_scenario_settings("8", "AEO2023 Reference Case")
    overlap_col = f'{scenario_prefix}{list(EQUIPMENT_SPECS.keys())[0]}_climate_npv_{SCC_ASSUMPTIONS[0]}'
    df[overlap_col] = 999.99  # This should be dropped in the result
    
    result = calculate_public_npv(
        df,
        df_baseline_damages,
        df_mp_damages,
        "8",
        "AEO2023 Reference Case",
        RCM_MODELS[0],
        CR_FUNCTIONS[0]
    )
    
    # Check that the overlapping column in the result has the calculated value, not 999.99
    assert overlap_col in result.columns
    assert not any(result[overlap_col] == 999.99), "Overlapping column was not properly handled"

# ============================================================================
# Tests for calculate_lifetime_damages_grid_scenario
# ============================================================================

def test_calculate_lifetime_damages_grid_scenario(sample_dataframes, mock_define_scenario_settings, mock_discount_factor):
    """
    Test that calculate_lifetime_damages_grid_scenario correctly calculates NPV values.
    """
    df, df_baseline_damages, df_mp_damages = sample_dataframes
    
    # Call the function directly to test it
    result = calculate_lifetime_damages_grid_scenario(
        df,
        df_baseline_damages,
        df_mp_damages,
        "8",
        "AEO2023 Reference Case",
        RCM_MODELS[0],
        CR_FUNCTIONS[0],
        base_year=2024,
        discounting_method="public"
    )
    
    # Verify result is a DataFrame with expected columns
    assert isinstance(result, pd.DataFrame)
    
    # Check that NPV columns exist for all equipment categories
    scenario_prefix, _, _, _, _, _ = mock_define_scenario_settings("8", "AEO2023 Reference Case")
    for category in EQUIPMENT_SPECS.keys():
        # Check just one SCC value and health model combination for simplicity
        verify_npv_values(result, scenario_prefix, category, SCC_ASSUMPTIONS[0], RCM_MODELS[0], CR_FUNCTIONS[0])
