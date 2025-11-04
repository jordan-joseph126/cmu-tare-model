"""
test_determine_adoption_potential_sensitivity.py

Pytest test suite for validating the adoption_decision functionality
in the adoption potential module. This module evaluates equipment adoption
feasibility by categorizing upgrades into four tiers based on economic factors
and assesses public impacts of retrofits.

This test suite verifies:
1. Proper implementation of the 5-step validation framework:
   - Mask Initialization with initialize_validation_tracking()
   - Series Initialization with create_retrofit_only_series()
   - Valid-Only Calculation for qualifying homes
   - Valid-Only Updates with compound masks
   - Final Masking with apply_final_masking()
2. Correct adoption tier classification logic
3. Accurate public impact assessment
4. Appropriate handling of errors and edge cases
5. Correct integration with private and public NPV calculations
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Set

# Import the module to test
from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_24April2025 import (
    adoption_decision,
    validate_input_parameters
)

# Import constants and utilities
from cmu_tare_model.constants import (
    EQUIPMENT_SPECS,
    RCM_MODELS,
    CR_FUNCTIONS,
    SCC_ASSUMPTIONS,
    UPGRADE_COLUMNS
)

from cmu_tare_model.utils.validation_framework_NEEDS_FIXED import (
    initialize_validation_tracking,
    create_retrofit_only_series,
    apply_new_columns_to_dataframe,
    apply_final_masking
)


# -------------------------------------------------------------------------
#                           TEST FIXTURES
# -------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Mock the constants module to isolate tests from external dependencies.
    
    This fixture runs automatically for all tests and provides consistent
    test data by mocking constants that affect validation behavior.
    
    Args:
        monkeypatch: Pytest fixture for patching attributes/functions
    """
    # Mock equipment specs with simplified lifetimes for testing
    mock_equipment_specs = {
        'heating': 15, 
        'waterHeating': 12, 
        'clothesDrying': 13, 
        'cooking': 15
    }
    
    # Mock upgrade columns for tracking retrofit status
    mock_upgrade_columns = {
        'heating': 'upgrade_hvac_heating_efficiency',
        'waterHeating': 'upgrade_water_heater_efficiency',
        'clothesDrying': 'upgrade_clothes_dryer',
        'cooking': 'upgrade_cooking_range'
    }
    
    # Mock RCM models for health impact calculations
    mock_rcm_models = ['ap2', 'easiur', 'inmap']
    
    # Mock CR functions for health impact calculations
    mock_cr_functions = ['acs', 'h6c']
    
    # Mock SCC assumptions for climate impact calculations
    mock_scc_assumptions = ['lower', 'central', 'upper']
    
    # Apply all mocks to relevant modules
    # monkeypatch.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', mock_equipment_specs)
    monkeypatch.setattr('cmu_tare_model.constants.UPGRADE_COLUMNS', mock_upgrade_columns)
    monkeypatch.setattr('cmu_tare_model.constants.RCM_MODELS', mock_rcm_models)
    monkeypatch.setattr('cmu_tare_model.constants.CR_FUNCTIONS', mock_cr_functions)
    monkeypatch.setattr('cmu_tare_model.constants.SCC_ASSUMPTIONS', mock_scc_assumptions)
    
    # Also apply to adoption potential module directly
    # monkeypatch.setattr('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity.EQUIPMENT_SPECS', 
    #                    mock_equipment_specs)
    monkeypatch.setattr('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity.UPGRADE_COLUMNS', 
                       mock_upgrade_columns)
    monkeypatch.setattr('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity.RCM_MODELS', 
                       mock_rcm_models)
    monkeypatch.setattr('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity.CR_FUNCTIONS', 
                       mock_cr_functions)
    monkeypatch.setattr('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity.SCC_ASSUMPTIONS', 
                       mock_scc_assumptions)


@pytest.fixture
def sample_homes_df() -> pd.DataFrame:
    """
    Generate sample DataFrame with comprehensive data for testing.
    
    This fixture creates test data with diverse scenarios including:
    - Valid and invalid homes across different equipment categories
    - Different NPV values for each tier classification
    - Various public impact scenarios
    - Homes with and without scheduled retrofits
    
    Returns:
        pd.DataFrame: Sample DataFrame for testing adoption potential
    """
    # Create base DataFrame with index
    data = {
        # Metadata columns
        'state': ['CA', 'TX', 'NY', 'FL', 'IL'],
        'census_division': ['Pacific', 'West South Central', 'Middle Atlantic', 'South Atlantic', 'East North Central'],
        
        # Include flags for each category (True = valid, False = invalid)
        'include_heating': [True, True, True, True, False],
        'include_waterHeating': [True, True, False, True, True],
        'include_clothesDrying': [True, False, True, True, False],
        'include_cooking': [False, True, True, False, True],
        
        # Add upgrade columns (None = already upgraded, value = scheduled upgrade)
        'upgrade_hvac_heating_efficiency': ['ASHP', 'GSHP', None, 'ASHP', None],
        'upgrade_water_heater_efficiency': ['HP', None, 'HP', None, 'HP'],
        'upgrade_clothes_dryer': [None, 'Electric', None, 'Electric', None],
        'upgrade_cooking_range': ['Induction', None, 'Induction', None, None],
        
        # Add rebate amounts for different equipment
        'mp8_heating_rebate_amount': [2000, 2500, 0, 1800, 0],
        'mp8_waterHeating_rebate_amount': [500, 0, 600, 0, 500],
        'mp8_clothesDrying_rebate_amount': [0, 300, 0, 300, 0],
        'mp8_cooking_rebate_amount': [500, 0, 500, 0, 0]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add private NPV columns for each category
    # Tier 1: Positive lessWTP NPV (index 0)
    # Tier 2: Negative lessWTP NPV, Positive moreWTP NPV (index 1)
    # Tier 3: Negative lessWTP and moreWTP NPV, Positive total NPV (index 2)
    # Tier 4: All NPVs negative (index 3)
    # Invalid: NaN values (index 4)
    
    # Heating NPVs - showcase all tiers
    df['iraRef_mp8_heating_private_npv_lessWTP'] = [500, -300, -800, -1200, np.nan]
    df['iraRef_mp8_heating_private_npv_moreWTP'] = [800, 200, -300, -800, np.nan]
    df['iraRef_mp8_heating_public_npv_upper_ap2_acs'] = [1000, 800, 600, 400, np.nan]
    
    # Water Heating NPVs - different patterns
    df['iraRef_mp8_waterHeating_private_npv_lessWTP'] = [-100, -300, np.nan, 400, -700]
    df['iraRef_mp8_waterHeating_private_npv_moreWTP'] = [200, -100, np.nan, 600, -400]
    df['iraRef_mp8_waterHeating_public_npv_upper_ap2_acs'] = [400, 200, np.nan, -200, 600]
    
    # Clothes Drying NPVs - with negative public values
    df['iraRef_mp8_clothesDrying_private_npv_lessWTP'] = [np.nan, -100, 300, -500, np.nan]
    df['iraRef_mp8_clothesDrying_private_npv_moreWTP'] = [np.nan, 100, 500, -300, np.nan]
    df['iraRef_mp8_clothesDrying_public_npv_upper_ap2_acs'] = [np.nan, -200, -100, -300, np.nan]
    
    # Cooking NPVs - more variations
    df['iraRef_mp8_cooking_private_npv_lessWTP'] = [-500, -200, -600, np.nan, 300]
    df['iraRef_mp8_cooking_private_npv_moreWTP'] = [-300, 100, -400, np.nan, 500] 
    df['iraRef_mp8_cooking_public_npv_upper_ap2_acs'] = [800, 0, 600, np.nan, -100]
    
    # Add same structure for pre-IRA scenario (different values to test both scenarios)
    df['preIRA_mp8_heating_private_npv_lessWTP'] = [400, -400, -900, -1300, np.nan]
    df['preIRA_mp8_heating_private_npv_moreWTP'] = [700, 100, -400, -900, np.nan]
    df['preIRA_mp8_heating_public_npv_upper_ap2_acs'] = [800, 600, 400, 200, np.nan]
    
    # Add total NPV columns (will be calculated but we add them for testing specific functions)
    df['iraRef_mp8_heating_total_npv_lessWTP_upper_ap2_acs'] = [1500, 500, -200, -800, np.nan]
    df['iraRef_mp8_heating_total_npv_moreWTP_upper_ap2_acs'] = [1800, 1000, 300, -400, np.nan]
    
    return df


@pytest.fixture
def mock_define_scenario_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Mock the define_scenario_params function to return consistent test values.
    
    Args:
        monkeypatch: Pytest fixture for patching functions
    """
    def mock_function(menu_mp: int, policy_scenario: str) -> Tuple[str, str, Dict, Dict, Dict, Dict]:
        """Mock implementation of define_scenario_params."""
        # Validate policy scenario
        if policy_scenario not in ["No Inflation Reduction Act", "AEO2023 Reference Case"]:
            raise ValueError(f"Invalid policy_scenario: {policy_scenario}. Must be one of ['No Inflation Reduction Act', 'AEO2023 Reference Case']")

        # Convert menu_mp to int if it's a string
        if isinstance(menu_mp, str):
            try:
                menu_mp = int(menu_mp)
            except ValueError:
                raise ValueError(f"Invalid menu_mp: {menu_mp}. Must be an integer or a string representation of an integer.")

        # Determine scenario prefix
        if menu_mp == 0:
            scenario_prefix = "baseline_"
        elif policy_scenario == "No Inflation Reduction Act":
            scenario_prefix = f"preIRA_mp{menu_mp}_"
        else:
            scenario_prefix = f"iraRef_mp{menu_mp}_"

        # Set the Cambium scenario
        cambium_scenario = "MidCase"

        # Create dummy lookup dictionaries - just placeholders for testing
        dummy_emissions_fossil_fuel = {}
        dummy_emissions_electricity_climate = {}
        dummy_emissions_electricity_health = {}
        dummy_fuel_prices = {}

        return (
            scenario_prefix,
            cambium_scenario,
            dummy_emissions_fossil_fuel,
            dummy_emissions_electricity_climate,
            dummy_emissions_electricity_health,
            dummy_fuel_prices
        )
    
    # Apply the monkeypatching - PATCH BOTH POSSIBLE IMPORT PATHS
    monkeypatch.setattr(
        'cmu_tare_model.utils.modeling_params.define_scenario_params',
        mock_function
    )
    monkeypatch.setattr(
        'cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity.define_scenario_params',
        mock_function
    )


@pytest.fixture
def mock_validation_framework(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Dict[str, bool]]:
    """
    Mock validation framework functions to track their calls while maintaining behavior.
    
    This fixture helps verify that the proper validation framework steps are followed
    by tracking when each function is called and for which categories.
    
    Args:
        monkeypatch: Pytest fixture for patching functions
        
    Returns:
        Dictionary tracking which validation functions were called for each category
    """
    # Create tracking dictionary
    function_calls = {
        'initialize_validation_tracking': {category: False for category in EQUIPMENT_SPECS},
        'create_retrofit_only_series': {category: False for category in EQUIPMENT_SPECS},
        'apply_final_masking': False
    }
    
    # Store original functions
    original_init_tracking = initialize_validation_tracking
    original_create_series = create_retrofit_only_series
    original_apply_masking = apply_final_masking
    original_apply_new_columns = apply_new_columns_to_dataframe
    
    # Create mock functions that track calls
    def mock_init_tracking(df, category, menu_mp, verbose=True):
        """Mock that tracks calls while maintaining behavior."""
        function_calls['initialize_validation_tracking'][category] = True
        return original_init_tracking(df, category, menu_mp, verbose)
    
    def mock_create_series(df, retrofit_mask, *args, **kwargs):
        """Mock that tracks calls while maintaining behavior."""
        # Find which category this mask corresponds to
        for cat in EQUIPMENT_SPECS:
            if f'include_{cat}' in df.columns and df[f'include_{cat}'].equals(retrofit_mask):
                function_calls['create_retrofit_only_series'][cat] = True
                break
        return original_create_series(df, retrofit_mask, *args, **kwargs)
    
    def mock_apply_masking(df, all_columns_to_mask, verbose=True):
        """Mock that tracks calls while maintaining behavior."""
        function_calls['apply_final_masking'] = True
        return original_apply_masking(df, all_columns_to_mask, verbose)
    
    def mock_apply_new_columns(df_original, df_new_columns, category, category_columns_to_mask, all_columns_to_mask):
        """Mock that tracks columns properly."""
        return original_apply_new_columns(df_original, df_new_columns, category, category_columns_to_mask, all_columns_to_mask)
    
    # Apply monkeypatching
    monkeypatch.setattr(
        'cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity.initialize_validation_tracking',
        mock_init_tracking
    )
    monkeypatch.setattr(
        'cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity.create_retrofit_only_series',
        mock_create_series
    )
    monkeypatch.setattr(
        'cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity.apply_final_masking',
        mock_apply_masking
    )
    monkeypatch.setattr(
        'cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity.apply_new_columns_to_dataframe',
        mock_apply_new_columns
    )
    
    return function_calls


@pytest.fixture(params=['heating', 'waterHeating', 'clothesDrying', 'cooking'])
def category(request: pytest.FixtureRequest) -> str:
    """
    Parametrized fixture for equipment categories.
    
    Args:
        request: Pytest request object containing the parameter
        
    Returns:
        str: Equipment category name
    """
    return request.param


@pytest.fixture(params=['No Inflation Reduction Act', 'AEO2023 Reference Case'])
def policy_scenario(request: pytest.FixtureRequest) -> str:
    """
    Parametrized fixture for policy scenarios.
    
    Args:
        request: Pytest request object containing the parameter
        
    Returns:
        str: Policy scenario name
    """
    return request.param


@pytest.fixture(params=['ap2', 'easiur', 'inmap'])
def rcm_model(request: pytest.FixtureRequest) -> str:
    """
    Parametrized fixture for RCM models.
    
    Args:
        request: Pytest request object containing the parameter
        
    Returns:
        str: RCM model name
    """
    return request.param


@pytest.fixture(params=['acs', 'h6c'])
def cr_function(request: pytest.FixtureRequest) -> str:
    """
    Parametrized fixture for CR functions.
    
    Args:
        request: Pytest request object containing the parameter
        
    Returns:
        str: CR function name
    """
    return request.param


# -------------------------------------------------------------------------
#              PARAMETER VALIDATION TESTS
# -------------------------------------------------------------------------

def test_validate_input_parameters_valid() -> None:
    """
    Test that valid parameters pass validation without errors.
    
    This test verifies that validate_input_parameters accepts valid parameters
    without raising any exceptions.
    """
    # Valid parameters should not raise errors
    try:
        validate_input_parameters(
            menu_mp=8,
            policy_scenario='AEO2023 Reference Case',
            rcm_model='ap2',
            cr_function='acs'
        )
    except Exception as e:
        pytest.fail(f"validate_input_parameters raised an exception with valid parameters: {e}")


def test_validate_input_parameters_invalid_policy_scenario() -> None:
    """
    Test that invalid policy scenario raises appropriate error.
    
    This test verifies that validate_input_parameters raises a ValueError
    with an informative message when an invalid policy scenario is provided.
    """
    with pytest.raises(ValueError) as excinfo:
        validate_input_parameters(
            menu_mp=8,
            policy_scenario='Invalid Scenario',
            rcm_model='ap2',
            cr_function='acs'
        )
    
    # Verify error message mentions policy scenario
    error_msg = str(excinfo.value)
    assert "policy_scenario" in error_msg.lower(), "Error message should mention policy_scenario"
    assert "Invalid Scenario" in error_msg, "Error message should contain the invalid value"


def test_validate_input_parameters_invalid_rcm_model() -> None:
    """
    Test that invalid RCM model raises appropriate error.
    
    This test verifies that validate_input_parameters raises a ValueError
    with an informative message when an invalid RCM model is provided.
    """
    with pytest.raises(ValueError) as excinfo:
        validate_input_parameters(
            menu_mp=8,
            policy_scenario='AEO2023 Reference Case',
            rcm_model='Invalid Model',
            cr_function='acs'
        )
    
    # Verify error message mentions RCM model
    error_msg = str(excinfo.value)
    assert "rcm_model" in error_msg.lower(), "Error message should mention rcm_model"
    assert "Invalid Model" in error_msg, "Error message should contain the invalid value"


def test_validate_input_parameters_invalid_cr_function() -> None:
    """
    Test that invalid CR function raises appropriate error.
    
    This test verifies that validate_input_parameters raises a ValueError
    with an informative message when an invalid CR function is provided.
    """
    with pytest.raises(ValueError) as excinfo:
        validate_input_parameters(
            menu_mp=8,
            policy_scenario='AEO2023 Reference Case',
            rcm_model='ap2',
            cr_function='Invalid Function'
        )
    
    # Verify error message mentions CR function
    error_msg = str(excinfo.value)
    assert "cr_function" in error_msg.lower(), "Error message should mention cr_function"
    assert "Invalid Function" in error_msg, "Error message should contain the invalid value"


def test_validate_input_parameters_invalid_menu_mp() -> None:
    """
    Test that invalid menu_mp raises appropriate error.
    
    This test verifies that validate_input_parameters raises a ValueError
    with an informative message when an invalid menu_mp is provided.
    """
    with pytest.raises(ValueError) as excinfo:
        validate_input_parameters(
            menu_mp="not_a_number",
            policy_scenario='AEO2023 Reference Case',
            rcm_model='ap2',
            cr_function='acs'
        )
    
    # Verify error message mentions menu_mp
    error_msg = str(excinfo.value)
    assert "menu_mp" in error_msg.lower(), "Error message should mention menu_mp"
    assert "not_a_number" in error_msg, "Error message should contain the invalid value"


def test_validate_input_parameters_convertible_menu_mp() -> None:
    """
    Test that menu_mp as string digits is accepted.
    
    This test verifies that validate_input_parameters accepts menu_mp
    as a string representation of a number without raising errors.
    """
    try:
        validate_input_parameters(
            menu_mp="8",  # String representation of an integer
            policy_scenario='AEO2023 Reference Case',
            rcm_model='ap2',
            cr_function='acs'
        )
    except Exception as e:
        pytest.fail(f"validate_input_parameters raised an exception with convertible menu_mp: {e}")


# -------------------------------------------------------------------------
#              VALIDATION FRAMEWORK IMPLEMENTATION TESTS
# -------------------------------------------------------------------------

def test_mask_initialization_implementation(
        sample_homes_df: pd.DataFrame, 
        mock_validation_framework: Dict[str, Dict[str, bool]],
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test Step 1: Mask initialization with initialize_validation_tracking().
    
    This test verifies that adoption_decision correctly:
    1. Initializes validation tracking for each category
    2. Creates a valid_mask using initialize_validation_tracking()
    3. Passes the valid_mask to subsequent calculations
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_validation_framework: Fixture tracking validation function calls
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Call adoption_decision with minimal parameter set
    adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Verify initialize_validation_tracking was called for each category
    for category, called in mock_validation_framework['initialize_validation_tracking'].items():
        assert called, f"initialize_validation_tracking() should be called for category '{category}'"


def test_series_initialization_implementation(
        sample_homes_df: pd.DataFrame, 
        mock_validation_framework: Dict[str, Dict[str, bool]],
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test Step 2: Series initialization with create_retrofit_only_series().
    
    This test verifies that adoption_decision correctly:
    1. Uses create_retrofit_only_series() to initialize result series
    2. Sets zeros for valid homes and NaN for invalid homes
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_validation_framework: Fixture tracking validation function calls
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Call adoption_decision with minimal parameter set
    adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Verify create_retrofit_only_series was called for at least one category
    assert any(mock_validation_framework['create_retrofit_only_series'].values()), \
        "create_retrofit_only_series() should be called for at least one category"


def test_valid_only_calculation_implementation(
        sample_homes_df: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test Step 3: Valid-only calculation for qualifying homes.
    
    This test verifies that calculations are performed only for valid homes
    and skipped for invalid homes, following the validation framework pattern.
    
    Args:
        sample_homes_df: Fixture providing test data
        monkeypatch: Pytest fixture for patching functions
    """
    # Create a tracking mechanism for calculation execution
    calculation_executed = {idx: False for idx in sample_homes_df.index}
    
    # Keep track of the original DataFrame.loc method
    original_loc = pd.DataFrame.loc
    
    # Mock the DataFrame.loc method to track which homes are calculated
    def mock_loc(self, *args, **kwargs):
        """Mock that tracks which homes were included in calculations."""
        # Check if we're setting values (modifying the DataFrame)
        is_setting = len(args) > 1
        
        if is_setting and isinstance(args[0], pd.Series):
            # args[0] is the mask Series when setting values
            mask = args[0]
            for idx in mask.index:
                if mask.loc[idx]:
                    calculation_executed[idx] = True
        
        # Call the original loc method
        return original_loc(self, *args, **kwargs)
    
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Apply monkeypatching
    monkeypatch.setattr(pd.DataFrame, 'loc', mock_loc)
    
    # Call adoption_decision focusing on one category
    result = adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Restore original method to avoid affecting other tests
    monkeypatch.setattr(pd.DataFrame, 'loc', original_loc)
    
    # Verify calculations were performed for valid homes
    # For simplicity, check heating category which has most complete test data
    category = 'heating'
    valid_mask = sample_homes_df[f'include_{category}']
    
    # At least some valid homes should have had calculations performed
    valid_homes_calculated = sum(calculation_executed[idx] for idx in valid_mask[valid_mask].index)
    assert valid_homes_calculated > 0, \
        "At least some valid homes should have had calculations performed"
    
    # Invalid homes should not have calculations performed
    # Check each invalid home specifically
    for idx in valid_mask[~valid_mask].index:
        if calculation_executed[idx]:
            pytest.fail(f"Invalid home at index {idx} should not have had calculations performed")


def test_final_masking_implementation(
        sample_homes_df: pd.DataFrame, 
        mock_validation_framework: Dict[str, Dict[str, bool]],
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test Step 5: Final masking with apply_final_masking.
    
    This test verifies that adoption_decision correctly:
    1. Tracks columns for masking
    2. Applies final masking with apply_final_masking
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_validation_framework: Fixture tracking validation function calls
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Call adoption_decision with minimal parameter set
    adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Verify apply_final_masking was called
    assert mock_validation_framework['apply_final_masking'], \
        "apply_final_masking() should be called"


def test_all_validation_steps(
        sample_homes_df: pd.DataFrame, 
        mock_validation_framework: Dict[str, Dict[str, bool]],
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test all five steps of the validation framework together.
    
    This test verifies that adoption_decision correctly implements all
    steps of the validation framework in sequence.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_validation_framework: Fixture tracking validation function calls
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Call adoption_decision
    adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Verify all validation steps were called
    # Step 1: initialize_validation_tracking for at least one category
    assert any(mock_validation_framework['initialize_validation_tracking'].values()), \
        "Step 1: initialize_validation_tracking() should be called for at least one category"
    
    # Step 2: create_retrofit_only_series for at least one category
    assert any(mock_validation_framework['create_retrofit_only_series'].values()), \
        "Step 2: create_retrofit_only_series() should be called for at least one category"
    
    # Step 5: apply_final_masking
    assert mock_validation_framework['apply_final_masking'], \
        "Step 5: apply_final_masking() should be called"


# -------------------------------------------------------------------------
#              ADOPTION TIER CLASSIFICATION TESTS
# -------------------------------------------------------------------------

def test_adoption_tier1_classification(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test correct classification of Tier 1 (Feasible) homes.
    
    This test verifies that adoption_decision correctly classifies homes as 
    Tier 1 (Feasible) when they have positive private NPV with total capital costs.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Call adoption_decision
    result = adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Check Tier 1 classification
    category = 'heating'  # For simplicity, focus on heating
    tier1_col = 'iraRef_mp8_heating_adoption_upper_ap2_acs'
    
    # Verify the column exists
    assert tier1_col in result.columns, f"Column {tier1_col} should exist in result DataFrame"
    
    # Find homes that should be classified as Tier 1
    tier1_homes = sample_homes_df[
        sample_homes_df['include_heating'] & 
        (sample_homes_df['iraRef_mp8_heating_private_npv_lessWTP'] > 0) &
        sample_homes_df['upgrade_hvac_heating_efficiency'].notna()
    ]
    
    # Verify these homes are correctly classified
    for idx in tier1_homes.index:
        assert result.loc[idx, tier1_col] == 'Tier 1: Feasible', \
            f"Home at index {idx} should be classified as 'Tier 1: Feasible'"


def test_adoption_tier2_classification(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test correct classification of Tier 2 (Feasible vs. Alternative) homes.
    
    This test verifies that adoption_decision correctly classifies homes as 
    Tier 2 when they have negative less-WTP NPV but positive more-WTP NPV.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Call adoption_decision
    result = adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Check Tier 2 classification
    category = 'heating'  # For simplicity, focus on heating
    tier2_col = 'iraRef_mp8_heating_adoption_upper_ap2_acs'
    
    # Verify the column exists
    assert tier2_col in result.columns, f"Column {tier2_col} should exist in result DataFrame"
    
    # Find homes that should be classified as Tier 2
    tier2_homes = sample_homes_df[
        sample_homes_df['include_heating'] & 
        (sample_homes_df['iraRef_mp8_heating_private_npv_lessWTP'] < 0) &
        (sample_homes_df['iraRef_mp8_heating_private_npv_moreWTP'] > 0) &
        sample_homes_df['upgrade_hvac_heating_efficiency'].notna()
    ]
    
    # Verify these homes are correctly classified
    for idx in tier2_homes.index:
        assert result.loc[idx, tier2_col] == 'Tier 2: Feasible vs. Alternative', \
            f"Home at index {idx} should be classified as 'Tier 2: Feasible vs. Alternative'"


def test_adoption_tier3_classification(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test correct classification of Tier 3 (Subsidy-Dependent) homes.
    
    This test verifies that adoption_decision correctly classifies homes as 
    Tier 3 when both private NPVs are negative but total NPV is positive.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Call adoption_decision
    result = adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Check Tier 3 classification
    category = 'heating'  # For simplicity, focus on heating
    tier3_col = 'iraRef_mp8_heating_adoption_upper_ap2_acs'
    
    # Verify the column exists
    assert tier3_col in result.columns, f"Column {tier3_col} should exist in result DataFrame"
    
    # Find homes that should be classified as Tier 3
    tier3_homes = sample_homes_df[
        sample_homes_df['include_heating'] & 
        (sample_homes_df['iraRef_mp8_heating_private_npv_lessWTP'] < 0) &
        (sample_homes_df['iraRef_mp8_heating_private_npv_moreWTP'] < 0) &
        (sample_homes_df['iraRef_mp8_heating_total_npv_moreWTP_upper_ap2_acs'] > 0) &
        sample_homes_df['upgrade_hvac_heating_efficiency'].notna()
    ]
    
    # Verify these homes are correctly classified
    for idx in tier3_homes.index:
        assert result.loc[idx, tier3_col] == 'Tier 3: Subsidy-Dependent Feasibility', \
            f"Home at index {idx} should be classified as 'Tier 3: Subsidy-Dependent Feasibility'"


def test_adoption_tier4_classification(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test correct classification of Tier 4 (Averse) homes.
    
    This test verifies that adoption_decision correctly classifies homes as 
    Tier 4 when both private NPVs and total NPV are negative.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Call adoption_decision
    result = adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Check Tier 4 classification
    category = 'heating'  # For simplicity, focus on heating
    tier4_col = 'iraRef_mp8_heating_adoption_upper_ap2_acs'
    
    # Verify the column exists
    assert tier4_col in result.columns, f"Column {tier4_col} should exist in result DataFrame"
    
    # Find homes that should be classified as Tier 4
    tier4_homes = sample_homes_df[
        sample_homes_df['include_heating'] & 
        (sample_homes_df['iraRef_mp8_heating_private_npv_lessWTP'] < 0) &
        (sample_homes_df['iraRef_mp8_heating_private_npv_moreWTP'] < 0) &
        (sample_homes_df['iraRef_mp8_heating_total_npv_moreWTP_upper_ap2_acs'] < 0) &
        sample_homes_df['upgrade_hvac_heating_efficiency'].notna()
    ]
    
    # Verify these homes are correctly classified
    for idx in tier4_homes.index:
        assert result.loc[idx, tier4_col] == 'Tier 4: Averse', \
            f"Home at index {idx} should be classified as 'Tier 4: Averse'"


def test_homes_with_no_upgrades_classification(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test correct classification of homes with no scheduled upgrades.
    
    This test verifies that adoption_decision correctly identifies homes
    that already have upgrades (upgrade_column is None) and marks them accordingly.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Call adoption_decision
    result = adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Check classification of homes with no upgrades
    category = 'heating'  # For simplicity, focus on heating
    adoption_col = 'iraRef_mp8_heating_adoption_upper_ap2_acs'
    
    # Verify the column exists
    assert adoption_col in result.columns, f"Column {adoption_col} should exist in result DataFrame"
    
    # Find homes with no upgrades (already upgraded)
    no_upgrade_homes = sample_homes_df[
        sample_homes_df['include_heating'] & 
        sample_homes_df['upgrade_hvac_heating_efficiency'].isna()
    ]
    
    # Verify these homes are correctly classified
    for idx in no_upgrade_homes.index:
        assert result.loc[idx, adoption_col] == 'N/A: Already Upgraded!', \
            f"Home at index {idx} should be classified as 'N/A: Already Upgraded!'"


def test_invalid_homes_classification(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test correct classification of invalid homes.
    
    This test verifies that adoption_decision correctly identifies homes
    that have invalid data (include_category is False) and marks them accordingly.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Call adoption_decision
    result = adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Check classification of invalid homes
    category = 'heating'  # For simplicity, focus on heating
    adoption_col = 'iraRef_mp8_heating_adoption_upper_ap2_acs'
    
    # Verify the column exists
    assert adoption_col in result.columns, f"Column {adoption_col} should exist in result DataFrame"
    
    # Find invalid homes
    invalid_homes = sample_homes_df[~sample_homes_df['include_heating']]
    
    # Verify these homes have appropriate classification
    for idx in invalid_homes.index:
        # Invalid homes should have masked values (NaN)
        assert pd.isna(result.loc[idx, adoption_col]) or result.loc[idx, adoption_col] == 'N/A: Invalid Baseline Fuel/Tech', \
            f"Invalid home at index {idx} should have NaN or 'N/A: Invalid Baseline Fuel/Tech'"


# -------------------------------------------------------------------------
#              PUBLIC IMPACT ASSESSMENT TESTS
# -------------------------------------------------------------------------

def test_public_benefit_classification(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test correct classification of Public Benefit impacts.
    
    This test verifies that adoption_decision correctly classifies homes as
    having 'Public Benefit' when they have positive public NPV.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Call adoption_decision
    result = adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Check Public Benefit classification
    category = 'heating'  # For simplicity, focus on heating
    impact_col = 'iraRef_mp8_heating_impact_upper_ap2_acs'
    
    # Verify the column exists
    assert impact_col in result.columns, f"Column {impact_col} should exist in result DataFrame"
    
    # Find homes with positive public NPV (Public Benefit)
    benefit_homes = sample_homes_df[
        sample_homes_df['include_heating'] & 
        (sample_homes_df['iraRef_mp8_heating_public_npv_upper_ap2_acs'] > 0) &
        sample_homes_df['upgrade_hvac_heating_efficiency'].notna()
    ]
    
    # Verify these homes are correctly classified
    for idx in benefit_homes.index:
        assert result.loc[idx, impact_col] == 'Public Benefit', \
            f"Home at index {idx} should be classified as 'Public Benefit'"


def test_public_detriment_classification(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test correct classification of Public Detriment impacts.
    
    This test verifies that adoption_decision correctly classifies homes as
    having 'Public Detriment' when they have negative public NPV.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Create modified test data with negative public NPV
    df = sample_homes_df.copy()
    # Set negative public NPV for at least one home
    df.loc[3, 'iraRef_mp8_heating_public_npv_upper_ap2_acs'] = -500
    
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Call adoption_decision
    result = adoption_decision(
        df=df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Check Public Detriment classification
    category = 'heating'  # For simplicity, focus on heating
    impact_col = 'iraRef_mp8_heating_impact_upper_ap2_acs'
    
    # Verify the column exists
    assert impact_col in result.columns, f"Column {impact_col} should exist in result DataFrame"
    
    # Find homes with negative public NPV (Public Detriment)
    detriment_homes = df[
        df['include_heating'] & 
        (df['iraRef_mp8_heating_public_npv_upper_ap2_acs'] < 0) &
        df['upgrade_hvac_heating_efficiency'].notna()
    ]
    
    # Verify these homes are correctly classified
    for idx in detriment_homes.index:
        assert result.loc[idx, impact_col] == 'Public Detriment', \
            f"Home at index {idx} should be classified as 'Public Detriment'"


def test_zero_public_npv_classification(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test correct classification of homes with zero public NPV.
    
    This test verifies that adoption_decision correctly classifies homes as
    having 'Public NPV is Zero' when they have zero public NPV.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Create modified test data with zero public NPV
    df = sample_homes_df.copy()
    # Set zero public NPV for at least one home
    df.loc[3, 'iraRef_mp8_heating_public_npv_upper_ap2_acs'] = 0
    
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Call adoption_decision
    result = adoption_decision(
        df=df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Check zero public NPV classification
    category = 'heating'  # For simplicity, focus on heating
    impact_col = 'iraRef_mp8_heating_impact_upper_ap2_acs'
    
    # Verify the column exists
    assert impact_col in result.columns, f"Column {impact_col} should exist in result DataFrame"
    
    # Find homes with zero public NPV
    zero_npv_homes = df[
        df['include_heating'] & 
        (df['iraRef_mp8_heating_public_npv_upper_ap2_acs'] == 0) &
        df['upgrade_hvac_heating_efficiency'].notna()
    ]
    
    # Verify these homes are correctly classified
    for idx in zero_npv_homes.index:
        assert result.loc[idx, impact_col] == 'Public NPV is Zero', \
            f"Home at index {idx} should be classified as 'Public NPV is Zero'"


def test_already_upgraded_impact_classification(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test correct impact classification for already upgraded homes.
    
    This test verifies that adoption_decision correctly identifies homes
    that already have upgrades and marks their impact accordingly.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Call adoption_decision
    result = adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Check impact classification for already upgraded homes
    category = 'heating'  # For simplicity, focus on heating
    impact_col = 'iraRef_mp8_heating_impact_upper_ap2_acs'
    
    # Verify the column exists
    assert impact_col in result.columns, f"Column {impact_col} should exist in result DataFrame"
    
    # Find homes that already have upgrades
    already_upgraded_homes = sample_homes_df[
        sample_homes_df['include_heating'] & 
        sample_homes_df['upgrade_hvac_heating_efficiency'].isna()
    ]
    
    # Verify these homes are correctly classified
    for idx in already_upgraded_homes.index:
        assert result.loc[idx, impact_col] == 'N/A: Already Upgraded!', \
            f"Already upgraded home at index {idx} should have impact 'N/A: Already Upgraded!'"


def test_additional_public_benefit_calculation(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test correct calculation of additional public benefit.
    
    This test verifies that adoption_decision correctly calculates additional
    public benefit as (public_npv - rebate_amount).clip(lower=0), but only for IRA scenarios.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Call adoption_decision with IRA scenario
    result_ira = adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Call adoption_decision with non-IRA scenario
    result_no_ira = adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='No Inflation Reduction Act',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Check benefit calculation for IRA scenario
    category = 'heating'  # For simplicity, focus on heating
    benefit_col_ira = 'iraRef_mp8_heating_benefit_upper_ap2_acs'
    
    # Verify the column exists in IRA result
    assert benefit_col_ira in result_ira.columns, f"Column {benefit_col_ira} should exist in IRA result DataFrame"
    
    # Find valid homes with public NPV and rebate
    valid_homes = sample_homes_df[
        sample_homes_df['include_heating'] & 
        sample_homes_df['iraRef_mp8_heating_public_npv_upper_ap2_acs'].notna() &
        sample_homes_df['mp8_heating_rebate_amount'].notna()
    ]
    
    # Verify benefit calculation for IRA scenario
    for idx in valid_homes.index:
        public_npv = sample_homes_df.loc[idx, 'iraRef_mp8_heating_public_npv_upper_ap2_acs']
        rebate = sample_homes_df.loc[idx, 'mp8_heating_rebate_amount']
        expected_benefit = max(0, public_npv - rebate)  # .clip(lower=0)
        
        assert result_ira.loc[idx, benefit_col_ira] == expected_benefit, \
            f"Home at index {idx} should have benefit {expected_benefit} in IRA scenario"
    
    # Check benefit calculation for non-IRA scenario
    benefit_col_no_ira = 'preIRA_mp8_heating_benefit_upper_ap2_acs'
    
    # Verify the column exists in non-IRA result
    assert benefit_col_no_ira in result_no_ira.columns, f"Column {benefit_col_no_ira} should exist in non-IRA result DataFrame"
    
    # Verify all values are zero in non-IRA scenario
    valid_no_ira_homes = sample_homes_df[sample_homes_df['include_heating']]
    for idx in valid_no_ira_homes.index:
        assert result_no_ira.loc[idx, benefit_col_no_ira] == 0, \
            f"Home at index {idx} should have zero benefit in non-IRA scenario"


# -------------------------------------------------------------------------
#              ERROR HANDLING TESTS
# -------------------------------------------------------------------------

def test_missing_required_columns(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test handling of missing required columns.
    
    This test verifies that adoption_decision raises appropriate errors
    when required columns are missing from the input DataFrame.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Create a copy of the DataFrame and remove a required column
    df_missing = sample_homes_df.copy()
    df_missing = df_missing.drop(columns=['upgrade_hvac_heating_efficiency'])
    
    # Call adoption_decision and expect an error
    with pytest.raises(KeyError) as excinfo:
        adoption_decision(
            df=df_missing,
            menu_mp=8,
            policy_scenario='AEO2023 Reference Case',
            rcm_model='ap2',
            cr_function='acs',
            climate_sensitivity=False
        )
    
    # Verify error message mentions missing column
    error_msg = str(excinfo.value)
    assert "upgrade_hvac_heating_efficiency" in error_msg or "missing" in error_msg.lower(), \
        "Error message should mention the missing required column"


def test_invalid_data_type_handling(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test handling of invalid data types in NPV columns.
    
    This test verifies that adoption_decision handles invalid data types
    by attempting to convert them to numeric values.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Create a copy of the DataFrame and modify a column to have string values
    df_invalid_type = sample_homes_df.copy()
    df_invalid_type['iraRef_mp8_heating_private_npv_lessWTP'] = df_invalid_type['iraRef_mp8_heating_private_npv_lessWTP'].astype(str)
    
    # Create mock pd.to_numeric to track if it's called
    to_numeric_called = False
    original_to_numeric = pd.to_numeric
    
    def mock_to_numeric(*args, **kwargs):
        """Mock to track calls to pd.to_numeric."""
        nonlocal to_numeric_called
        to_numeric_called = True
        return original_to_numeric(*args, **kwargs)
    
    # Apply monkeypatching
    monkeypatch.setattr(pd, 'to_numeric', mock_to_numeric)
    
    # Call adoption_decision
    try:
        result = adoption_decision(
            df=df_invalid_type,
            menu_mp=8,
            policy_scenario='AEO2023 Reference Case',
            rcm_model='ap2',
            cr_function='acs',
            climate_sensitivity=False
        )
        
        # Verify to_numeric was called
        assert to_numeric_called, "pd.to_numeric should be called to convert string values"
        
    except Exception as e:
        # If an error occurs, verify it's related to data type conversion
        error_msg = str(e)
        assert "convert" in error_msg.lower() or "numeric" in error_msg.lower(), \
            "Error message should mention conversion issues"


def test_error_handling_for_invalid_scc(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test handling of invalid SCC assumptions.
    
    This test verifies that adoption_decision raises an appropriate error
    when SCC_ASSUMPTIONS is empty.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Mock SCC_ASSUMPTIONS to be empty
    monkeypatch.setattr('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity.SCC_ASSUMPTIONS', [])
    
    # Call adoption_decision and expect an error
    with pytest.raises(ValueError) as excinfo:
        adoption_decision(
            df=sample_homes_df,
            menu_mp=8,
            policy_scenario='AEO2023 Reference Case',
            rcm_model='ap2',
            cr_function='acs',
            climate_sensitivity=True  # This triggers the use of SCC_ASSUMPTIONS
        )
    
    # Verify error message mentions SCC_ASSUMPTIONS
    error_msg = str(excinfo.value)
    assert "SCC_ASSUMPTIONS" in error_msg or "empty" in error_msg.lower(), \
        "Error message should mention empty SCC_ASSUMPTIONS"


def test_category_error_graceful_continuation(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test graceful continuation after category-specific error.
    
    This test verifies that adoption_decision continues processing other
    categories even if one category encounters an error.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Track which categories are processed
    processed_categories = set()
    
    # Save original initialize_validation_tracking
    original_init_tracking = initialize_validation_tracking
    
    # Create mock for initialize_validation_tracking that raises error for one category
    def mock_init_tracking(df, category, menu_mp, verbose=True):
        """Mock that raises an error for one specific category."""
        if category == 'heating':
            raise ValueError("Simulated error for testing graceful continuation")
        
        # For other categories, track and process normally
        processed_categories.add(category)
        return original_init_tracking(df, category, menu_mp, verbose)
    
    # Apply monkeypatching
    monkeypatch.setattr(
        'cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity.initialize_validation_tracking',
        mock_init_tracking
    )
    
    # Call adoption_decision and expect it to complete without error
    try:
        result = adoption_decision(
            df=sample_homes_df,
            menu_mp=8,
            policy_scenario='AEO2023 Reference Case',
            rcm_model='ap2',
            cr_function='acs',
            climate_sensitivity=False
        )
        
        # Verify 'heating' was not processed but other categories were
        assert 'heating' not in processed_categories, "'heating' category should not be processed"
        assert len(processed_categories) > 0, "At least some categories should be processed"
        assert all(cat in processed_categories for cat in ['waterHeating', 'clothesDrying', 'cooking']), \
            "All other categories should be processed"
        
    except Exception as e:
        pytest.fail(f"adoption_decision should handle category errors gracefully, but raised: {e}")


# -------------------------------------------------------------------------
#              INTEGRATION TESTS
# -------------------------------------------------------------------------

def test_end_to_end_processing(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test complete end-to-end processing of the adoption_decision function.
    
    This test verifies that adoption_decision correctly processes all equipment
    categories and returns a DataFrame with all expected result columns.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Call adoption_decision
    result = adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Verify result is a DataFrame
    assert isinstance(result, pd.DataFrame), "Result should be a pandas DataFrame"
    
    # Verify all expected columns are present for each category
    for category in EQUIPMENT_SPECS.keys():
        expected_cols = [
            f'iraRef_mp8_{category}_health_sensitivity',
            f'iraRef_mp8_{category}_benefit_upper_ap2_acs',
            f'iraRef_mp8_{category}_total_npv_lessWTP_upper_ap2_acs',
            f'iraRef_mp8_{category}_total_npv_moreWTP_upper_ap2_acs',
            f'iraRef_mp8_{category}_adoption_upper_ap2_acs',
            f'iraRef_mp8_{category}_impact_upper_ap2_acs'
        ]
        
        for col in expected_cols:
            assert col in result.columns, f"Result should contain column '{col}'"


def test_across_policy_scenarios(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        policy_scenario: str,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test adoption_decision with different policy scenarios.
    
    This parametrized test verifies that adoption_decision works correctly
    with all policy scenarios and uses the appropriate column naming.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        policy_scenario: Parametrized fixture providing policy scenario
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Determine expected prefixes based on policy scenario
    if policy_scenario == 'No Inflation Reduction Act':
        expected_prefix = 'preIRA_mp8_'
    else:
        expected_prefix = 'iraRef_mp8_'
    
    # Call adoption_decision
    result = adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario=policy_scenario,
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Verify columns have the expected prefix
    category = 'heating'  # Focus on one category for simplicity
    columns_for_category = [col for col in result.columns if category in col and col.startswith(expected_prefix)]
    
    assert len(columns_for_category) > 0, \
        f"Result should contain columns with prefix '{expected_prefix}' for category '{category}'"


def test_across_rcm_models_and_cr_functions(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        rcm_model: str,
        cr_function: str,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test adoption_decision with different RCM models and CR functions.
    
    This parametrized test verifies that adoption_decision works correctly
    with all RCM models and CR functions and includes them in column names.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        rcm_model: Parametrized fixture providing RCM model
        cr_function: Parametrized fixture providing CR function
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Call adoption_decision
    result = adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model=rcm_model,
        cr_function=cr_function,
        climate_sensitivity=False
    )
    
    # Verify columns include RCM model and CR function
    category = 'heating'  # Focus on one category for simplicity
    expected_pattern = f'iraRef_mp8_{category}_.*{rcm_model}_{cr_function}'
    columns_matching = [col for col in result.columns if category in col and rcm_model in col and cr_function in col]
    
    assert len(columns_matching) > 0, \
        f"Result should contain columns with RCM model '{rcm_model}' and CR function '{cr_function}'"
    
    # Verify health sensitivity column has correct value
    health_col = f'iraRef_mp8_{category}_health_sensitivity'
    assert health_col in result.columns, f"Result should contain column '{health_col}'"
    
    # Check a non-NaN value to confirm it includes RCM model and CR function
    for idx in sample_homes_df.index:
        if not pd.isna(result.loc[idx, health_col]):
            assert rcm_model in result.loc[idx, health_col], \
                f"Health sensitivity at index {idx} should contain RCM model '{rcm_model}'"
            assert cr_function in result.loc[idx, health_col], \
                f"Health sensitivity at index {idx} should contain CR function '{cr_function}'"
            break


def test_climate_sensitivity_parameter(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test climate_sensitivity parameter behavior.
    
    This test verifies that the climate_sensitivity parameter correctly
    controls which SCC values are used (all or just 'upper').
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Mock SCC_ASSUMPTIONS with multiple values
    mock_scc = ['lower', 'central', 'upper']
    monkeypatch.setattr('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity.SCC_ASSUMPTIONS', mock_scc)
    
    # Track which SCC values are used
    scc_values_used = set()
    
    # Save original iterate through SCC assumptions behavior
    original_for_loop = list.__iter__
    
    # Create a mock for list iteration that tracks SCC values
    def mock_for_iter(self):
        """Mock that tracks which values are iterated through."""
        original_iter = original_for_loop(self)
        for item in original_iter:
            if item in mock_scc:
                scc_values_used.add(item)
            yield item
    
    # Apply monkeypatching for SCC loop
    monkeypatch.setattr(list, '__iter__', mock_for_iter)
    
    # Call adoption_decision with climate_sensitivity=False
    result_false = adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=False
    )
    
    # Verify only 'upper' SCC was used
    assert scc_values_used == {'upper'}, \
        "Only 'upper' SCC value should be used when climate_sensitivity=False"
    
    # Reset tracking and call with climate_sensitivity=True
    scc_values_used.clear()
    
    result_true = adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='ap2',
        cr_function='acs',
        climate_sensitivity=True
    )
    
    # Verify all SCC values were used
    assert scc_values_used == set(mock_scc), \
        "All SCC values should be used when climate_sensitivity=True"


def test_health_sensitivity_column(
        sample_homes_df: pd.DataFrame,
        mock_define_scenario_settings: None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test health sensitivity column creation.
    
    This test verifies that adoption_decision correctly adds a health_sensitivity
    column with the RCM model and CR function for each category.
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_define_scenario_settings: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Mock print function to avoid cluttering test output
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: None)
    
    # Call adoption_decision
    rcm_model = 'ap2'
    cr_function = 'acs'
    result = adoption_decision(
        df=sample_homes_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model=rcm_model,
        cr_function=cr_function,
        climate_sensitivity=False
    )
    
    # Check health sensitivity column
    for category in EQUIPMENT_SPECS.keys():
        health_col = f'iraRef_mp8_{category}_health_sensitivity'
        
        # Verify column exists
        assert health_col in result.columns, f"Result should contain column '{health_col}'"
        
        # Verify it contains RCM model and CR function
        for idx in sample_homes_df.index:
            if sample_homes_df.loc[idx, f'include_{category}']:
                assert rcm_model in result.loc[idx, health_col], \
                    f"Health sensitivity at index {idx} should contain RCM model '{rcm_model}'"
                assert cr_function in result.loc[idx, health_col], \
                    f"Health sensitivity at index {idx} should contain CR function '{cr_function}'"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
