"""
test_calculate_lifetime_public_impact_sensitivity.py

Pytest test suite for validating the lifetime public impact calculation functionality
and its implementation of the 5-step validation framework:
1. Mask Initialization with initialize_validation_tracking()
2. Series Initialization with create_retrofit_only_series()
3. Valid-Only Calculation for qualifying homes
4. Valid-Only Updates using list-based collection pattern
5. Final Masking with apply_final_masking()

This test suite verifies the implementation of these steps in public impact calculations,
along with testing error handling and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Set

# Import the specific module being tested
from cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity import (
    calculate_public_npv,
    calculate_lifetime_damages_grid_scenario
)

# Import constants and validation framework utilities for validation
from cmu_tare_model.constants import (
    EQUIPMENT_SPECS, 
    FUEL_MAPPING, 
    RCM_MODELS,
    CR_FUNCTIONS,
    SCC_ASSUMPTIONS
)
from cmu_tare_model.utils.validation_framework import (
    initialize_validation_tracking,
    create_retrofit_only_series,
    apply_final_masking,
    get_valid_calculation_mask,
    calculate_avoided_values
)

# -------------------------------------------------------------------------
#                           TEST FIXTURES
# -------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_constants(monkeypatch):
    """
    Mock the constants module to isolate tests from external dependencies.
    
    This fixture runs automatically for all tests and ensures consistent test data
    by mocking constants that affect validation behavior.
    
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
    
    # Mock fuel mapping for baseline fuel types
    mock_fuel_mapping = {
        'Electricity': 'electricity', 
        'Natural Gas': 'naturalGas', 
        'Fuel Oil': 'fuelOil', 
        'Propane': 'propane'
    }
    
    # Mock RCM models for health impact calculations
    mock_rcm_models = ['AP2', 'EASIUR', 'InMAP']
    
    # Mock CR functions for health impact calculations
    mock_cr_functions = ['acs', 'h6c']
    
    # Mock SCC assumptions for climate impact calculations
    mock_scc_assumptions = ['lower', 'central', 'upper']
    
    # Apply all mocks to relevant modules
    monkeypatch.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', mock_equipment_specs)
    monkeypatch.setattr('cmu_tare_model.constants.FUEL_MAPPING', mock_fuel_mapping)
    monkeypatch.setattr('cmu_tare_model.constants.RCM_MODELS', mock_rcm_models)
    monkeypatch.setattr('cmu_tare_model.constants.CR_FUNCTIONS', mock_cr_functions)
    monkeypatch.setattr('cmu_tare_model.constants.SCC_ASSUMPTIONS', mock_scc_assumptions)
    monkeypatch.setattr('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.EQUIPMENT_SPECS', 
                       mock_equipment_specs)
    monkeypatch.setattr('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.RCM_MODELS', 
                       mock_rcm_models)
    monkeypatch.setattr('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.CR_FUNCTIONS', 
                       mock_cr_functions)
    monkeypatch.setattr('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.SCC_ASSUMPTIONS', 
                       mock_scc_assumptions)


@pytest.fixture
def sample_homes_df():
    """
    Generate sample DataFrame with comprehensive data for testing.
    
    This fixture creates a DataFrame with all required columns for public impact calculations:
    - Geographic identifiers (county_fips, state, census_division, etc.)
    - Validation flags for each equipment category
    - Fuel consumption data for all categories
    - Measure package consumption data for multiple years
    
    Returns:
        pd.DataFrame: Sample DataFrame for testing
    """
    # Build all data first in a dictionary
    data = {
        'county_fips': ['01001', '02002', '03003', '04004', '05005'],
        'state': ['CA', 'TX', 'NY', 'FL', 'IL'],
        'year': [2023, 2023, 2023, 2023, 2023],
        'census_division': ['Pacific', 'West South Central', 'Middle Atlantic', 'South Atlantic', 'East North Central'],
        'gea_region': ['Region1', 'Region2', 'Region3', 'Region4', 'Region5'],
        'cambium_gea_region': ['Region1', 'Region2', 'Region3', 'Region4', 'Region5'],
    }
    
    # Add validation flags for each equipment category
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        data[f'include_{category}'] = [True, True, False, True, False]
        data[f'valid_fuel_{category}'] = [True, True, False, True, False]
        data[f'valid_tech_{category}'] = [True, True, False, True, False]
        data[f'base_{category}_fuel'] = ['Electricity', 'Natural Gas', 'Propane', 'Fuel Oil', 'Invalid']
    
    # Add baseline consumption columns for each fuel type and category
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        for fuel in ['electricity', 'naturalGas', 'fuelOil', 'propane']:
            data[f'base_{fuel}_{category}_consumption'] = [100, 200, 300, 400, 500]
        
        # Add baseline total consumption
        data[f'baseline_{category}_consumption'] = [500, 600, 700, 800, 900]
    
    # Add measure package consumption for multiple years
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        for year in range(2024, 2040):  # Multiple years for lifetime calculations
            data[f'mp8_{year}_{category}_consumption'] = [80, 90, 100, 110, 120]
    
    # Add upgrade columns for each category
    data['upgrade_hvac_heating_efficiency'] = ['ASHP', 'GSHP', None, 'ASHP', None]
    data['upgrade_water_heater_efficiency'] = ['HP', None, 'HP', None, 'HP']
    data['upgrade_clothes_dryer'] = [None, 'Electric', None, 'Electric', None]
    data['upgrade_cooking_range'] = ['Induction', None, 'Induction', None, None]
    
    # Create the DataFrame all at once
    return pd.DataFrame(data)


@pytest.fixture
def create_damage_columns():
    """
    Helper fixture that returns a function to create damage columns for DataFrames.
    
    Returns:
        function: A function that creates damage columns with consistent values
    """
    def _create_columns(df, scenario_prefix, base_year, damage_type, value_multiplier=1.0):
        """
        Creates damage columns with the specified scenario_prefix for each equipment category.
        
        Args:
            df: DataFrame to add columns to
            scenario_prefix: Column name prefix (e.g., 'baseline_', 'iraRef_mp8_')
            base_year: Base year for calculations
            damage_type: 'climate' for climate damages, 'health' for health damages
            value_multiplier: Multiplier for damage values (default 1.0)
        
        Returns:
            pd.DataFrame: DataFrame with added damage columns
        """
        # Create a dictionary to hold all the new columns
        new_columns = {}
        
        for category, lifetime in EQUIPMENT_SPECS.items():
            for year in range(0, lifetime):
                year_label = base_year + year
                
                if damage_type == 'climate':
                    # Climate damage columns for each SCC assumption
                    for scc in SCC_ASSUMPTIONS:
                        col_name = f'{scenario_prefix}{year_label}_{category}_damages_climate_lrmer_{scc}'
                        new_columns[col_name] = [value_multiplier * 100.0 * (i + 1) 
                                               for i in range(len(df.index))]
                
                elif damage_type == 'health':
                    # Health damage columns for each RCM model and CR function
                    for rcm in RCM_MODELS:
                        for cr in CR_FUNCTIONS:
                            col_name = f'{scenario_prefix}{year_label}_{category}_damages_health_{rcm}_{cr}'
                            new_columns[col_name] = [value_multiplier * 50.0 * (i + 1) 
                                                   for i in range(len(df.index))]
        
        # Create a new DataFrame with all columns and concatenate it with original
        new_df = pd.DataFrame(new_columns, index=df.index)
        return new_df
    
    return _create_columns


@pytest.fixture
def df_climate_health_damages(sample_homes_df, create_damage_columns, mock_define_scenario_settings):
    """
    Creates sample climate and health damage DataFrames for testing.
    
    Args:
        sample_homes_df: Sample DataFrame for index matching
        create_damage_columns: Function to create damage columns
        mock_define_scenario_settings: Mock for scenario settings
        
    Returns:
        tuple: (df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health)
    """
    base_year = 2024
    
    # Create base DataFrames with the same index as sample_homes_df
    df_baseline_climate = pd.DataFrame(index=sample_homes_df.index)
    df_baseline_health = pd.DataFrame(index=sample_homes_df.index)
    df_mp_climate = pd.DataFrame(index=sample_homes_df.index)
    df_mp_health = pd.DataFrame(index=sample_homes_df.index)
    
    # Add damage columns to baseline DataFrames
    baseline_prefix = "baseline_"
    climate_cols = create_damage_columns(df_baseline_climate, baseline_prefix, base_year, 'climate')
    health_cols = create_damage_columns(df_baseline_health, baseline_prefix, base_year, 'health')
    
    df_baseline_climate = pd.concat([df_baseline_climate, climate_cols], axis=1)
    df_baseline_health = pd.concat([df_baseline_health, health_cols], axis=1)
    
    # Add damage columns to measure-package DataFrames
    measure_prefix, _, _, _, _, _ = mock_define_scenario_settings(8, "AEO2023 Reference Case")
    mp_climate_cols = create_damage_columns(df_mp_climate, measure_prefix, base_year, 'climate', 0.6)  # 40% avoided damages
    mp_health_cols = create_damage_columns(df_mp_health, measure_prefix, base_year, 'health', 0.7)     # 30% avoided damages
    
    df_mp_climate = pd.concat([df_mp_climate, mp_climate_cols], axis=1)
    df_mp_health = pd.concat([df_mp_health, mp_health_cols], axis=1)
    
    # Add validation columns for each equipment category
    for category in EQUIPMENT_SPECS.keys():
        include_col = f'include_{category}'
        if include_col in sample_homes_df.columns:
            df_baseline_climate[include_col] = sample_homes_df[include_col]
            df_baseline_health[include_col] = sample_homes_df[include_col]
            df_mp_climate[include_col] = sample_homes_df[include_col]
            df_mp_health[include_col] = sample_homes_df[include_col]
    
    return df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health


@pytest.fixture
def mock_define_scenario_settings(monkeypatch):
    """
    Mock the define_scenario_params function to return consistent test values.
    
    Args:
        monkeypatch: Pytest fixture for patching functions
        
    Returns:
        function: Mock function for define_scenario_params
    """
    def mock_function(menu_mp, policy_scenario):
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
        'cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.define_scenario_params',
        mock_function
    )
    
    return mock_function


@pytest.fixture
def mock_discount_factor(monkeypatch):
    """
    Mock the calculate_discount_factor function to return predictable values for testing.
    
    Args:
        monkeypatch: Pytest fixture for patching functions
        
    Returns:
        None
    """
    def mock_function(base_year, year_label, discounting_method):
        """Mock implementation of calculate_discount_factor."""
        if discounting_method not in ['public', 'private_fixed']:
            raise ValueError(f"Invalid discounting method: {discounting_method}")
        
        if year_label == base_year:
            return 1.0  # No discounting for base year
        
        years_diff = year_label - base_year
        if discounting_method == 'public':
            return 1.0 / ((1 + 0.03) ** years_diff)  # 3% discount rate for public
        else:  # private_fixed
            return 1.0 / ((1 + 0.07) ** years_diff)  # 7% discount rate for private
    
    monkeypatch.setattr(
        'cmu_tare_model.utils.discounting.calculate_discount_factor',
        mock_function
    )
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.calculate_discount_factor',
        mock_function
    )


@pytest.fixture
def custom_equipment_specs(monkeypatch):
    """
    Fixture for temporarily modifying EQUIPMENT_SPECS for specific test cases.
    
    Args:
        monkeypatch: Pytest fixture for patching attributes
        
    Returns:
        function: A function that updates EQUIPMENT_SPECS for testing
    """
    def _custom_specs(specs_update):
        test_specs = EQUIPMENT_SPECS.copy()
        test_specs.update(specs_update)
        monkeypatch.setattr(
            'cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.EQUIPMENT_SPECS',
            test_specs
        )
        return test_specs
    
    return _custom_specs


@pytest.fixture(params=['heating', 'waterHeating', 'clothesDrying', 'cooking'])
def category(request):
    """
    Parametrized fixture for equipment categories.
    
    Args:
        request: Pytest request object containing the parameter
        
    Returns:
        str: Equipment category name
    """
    return request.param


@pytest.fixture(params=[0, 8])
def menu_mp(request):
    """
    Parametrized fixture for measure package values.
    
    Args:
        request: Pytest request object containing the parameter
        
    Returns:
        int: Measure package identifier (0=baseline, 8=measure package)
    """
    return request.param


@pytest.fixture(params=['No Inflation Reduction Act', 'AEO2023 Reference Case'])
def policy_scenario(request):
    """
    Parametrized fixture for policy scenarios.
    
    Args:
        request: Pytest request object containing the parameter
        
    Returns:
        str: Policy scenario name
    """
    return request.param


@pytest.fixture(params=['AP2', 'InMAP'])
def rcm_model(request):
    """
    Parametrized fixture for RCM models.
    
    Args:
        request: Pytest request object containing the parameter
        
    Returns:
        str: RCM model name
    """
    return request.param


@pytest.fixture(params=['acs', 'h6c'])
def cr_function(request):
    """
    Parametrized fixture for CR functions.
    
    Args:
        request: Pytest request object containing the parameter
        
    Returns:
        str: CR function name
    """
    return request.param


@pytest.fixture(params=['public', 'private_fixed'])
def discounting_method(request):
    """
    Parametrized fixture for discounting methods.
    
    Args:
        request: Pytest request object containing the parameter
        
    Returns:
        str: Discounting method name
    """
    return request.param


@pytest.fixture(params=['lower', 'central', 'upper'])
def scc_assumption(request):
    """
    Parametrized fixture for SCC assumptions.
    
    Args:
        request: Pytest request object containing the parameter
        
    Returns:
        str: SCC assumption name
    """
    return request.param


# -------------------------------------------------------------------------
#              VALIDATION FRAMEWORK IMPLEMENTATION TESTS
# -------------------------------------------------------------------------

def test_mask_initialization_implementation(
        sample_homes_df, 
        df_climate_health_damages,
        mock_define_scenario_settings,
        mock_discount_factor,
        monkeypatch):
    """
    Test Step 1: Mask initialization with initialize_validation_tracking().
    
    This test verifies that calculate_public_npv correctly:
    1. Initializes validation tracking for each category
    2. Creates a valid_mask using initialize_validation_tracking()
    3. Passes the valid_mask to subsequent calculations
    """
    # Track if initialize_validation_tracking is called
    init_tracking_called = {category: False for category in EQUIPMENT_SPECS}
    original_init_tracking = initialize_validation_tracking
    
    def mock_init_tracking(df, category, menu_mp, verbose=True):
        """Mock to track calls to initialize_validation_tracking."""
        init_tracking_called[category] = True
        return original_init_tracking(df, category, menu_mp, verbose)
    
    # Apply monkeypatching
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.initialize_validation_tracking',
        mock_init_tracking
    )
    
    # Call the main function
    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = df_climate_health_damages
    
    result = calculate_public_npv(
        df=sample_homes_df,
        df_baseline_climate=df_baseline_climate,
        df_baseline_health=df_baseline_health,
        df_mp_climate=df_mp_climate,
        df_mp_health=df_mp_health,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model=RCM_MODELS[0],
        base_year=2024,
        discounting_method='public',
        verbose=False
    )
    
    # Verify initialize_validation_tracking was called for each category
    for category in EQUIPMENT_SPECS:
        assert init_tracking_called[category], \
            f"initialize_validation_tracking() should be called for category '{category}'"


def test_series_initialization_implementation(
        sample_homes_df, 
        df_climate_health_damages,
        mock_define_scenario_settings,
        mock_discount_factor,
        monkeypatch):
    """
    Test Step 2: Series initialization with create_retrofit_only_series().
    
    This test verifies that calculate_public_npv correctly:
    1. Uses create_retrofit_only_series() to initialize result series
    2. Sets zeros for valid homes and NaN for invalid homes
    """
    # Track if create_retrofit_only_series is called
    create_series_called = False
    original_create_series = create_retrofit_only_series
    
    def mock_create_series(df, retrofit_mask, *args, **kwargs):
        """Mock to track calls to create_retrofit_only_series."""
        nonlocal create_series_called
        create_series_called = True
        return original_create_series(df, retrofit_mask, *args, **kwargs)
    
    # Apply monkeypatching
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.create_retrofit_only_series',
        mock_create_series
    )
    
    # Call the main function
    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = df_climate_health_damages
    
    result = calculate_public_npv(
        df=sample_homes_df,
        df_baseline_climate=df_baseline_climate,
        df_baseline_health=df_baseline_health,
        df_mp_climate=df_mp_climate,
        df_mp_health=df_mp_health,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model=RCM_MODELS[0],
        base_year=2024,
        discounting_method='public',
        verbose=False
    )
    
    # Verify create_retrofit_only_series was called
    assert create_series_called, \
        "create_retrofit_only_series() should be called to initialize result series"


def test_valid_only_calculation_implementation(
        sample_homes_df, 
        df_climate_health_damages,
        mock_define_scenario_settings,
        mock_discount_factor,
        monkeypatch):
    """
    Test Step 3: Valid-only calculation for qualifying homes.
    
    This test verifies that calculations are performed only for valid homes
    and skipped for invalid homes, following the validation framework pattern.
    """
    # Create a tracking mechanism for calculation execution
    calculation_executed = {idx: False for idx in sample_homes_df.index}
    
    # Mock the calculate_avoided_values function to track which homes are calculated
    original_avoided_values = calculate_avoided_values
    
    def mock_avoided_values(baseline_values, measure_values, retrofit_mask=None):
        """Mock that tracks which homes were included in calculations."""
        # Mark valid homes as executed if retrofit_mask is provided
        if retrofit_mask is not None:
            for idx in retrofit_mask[retrofit_mask].index:
                calculation_executed[idx] = True
        
        # Return original result
        return original_avoided_values(baseline_values, measure_values, retrofit_mask)
    
    # Apply monkeypatching
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.calculate_avoided_values',
        mock_avoided_values
    )
    
    # Call the main function
    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = df_climate_health_damages
    
    result = calculate_public_npv(
        df=sample_homes_df,
        df_baseline_climate=df_baseline_climate,
        df_baseline_health=df_baseline_health,
        df_mp_climate=df_mp_climate,
        df_mp_health=df_mp_health,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model=RCM_MODELS[0],
        base_year=2024,
        discounting_method='public',
        verbose=False
    )
    
    # Verify calculations were only performed for valid homes
    category_to_check = 'heating'  # Focus on one category for simplicity
    valid_mask = sample_homes_df[f'include_{category_to_check}']
    
    for idx in sample_homes_df.index:
        if valid_mask[idx]:
            assert calculation_executed[idx], \
                f"Calculation should have been executed for valid home at index {idx}"
        else:
            assert not calculation_executed[idx], \
                f"Calculation should NOT have been executed for invalid home at index {idx}"


def test_list_based_collection_implementation(
        sample_homes_df, 
        df_climate_health_damages,
        mock_define_scenario_settings,
        mock_discount_factor,
        monkeypatch):
    """
    Test Step 4: Valid-only updates using list-based collection.
    
    This test verifies that calculate_public_npv correctly:
    1. Uses list-based collection pattern for yearly values
    2. Avoids inefficient incremental updates
    3. Combines yearly values using pandas operations
    """
    # Track how yearly values are collected and summed
    list_collection_used = False
    pandas_sum_used = False
    
    # Original pandas concat and sum functions
    original_concat = pd.concat
    original_sum = pd.DataFrame.sum
    
    def mock_concat(*args, **kwargs):
        """Mock to track if pd.concat is used for list collection."""
        nonlocal list_collection_used
        list_collection_used = True
        return original_concat(*args, **kwargs)
    
    def mock_sum(self, *args, **kwargs):
        """Mock to track if DataFrame.sum is used to combine yearly values."""
        nonlocal pandas_sum_used
        axis = kwargs.get('axis', args[0] if args else None)
        if axis == 1:  # Summing across columns (years)
            pandas_sum_used = True
        return original_sum(self, *args, **kwargs)
    
    # Apply monkeypatching
    monkeypatch.setattr('pandas.concat', mock_concat)
    monkeypatch.setattr('pandas.DataFrame.sum', mock_sum)
    
    # Call the main function
    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = df_climate_health_damages
    
    result = calculate_public_npv(
        df=sample_homes_df,
        df_baseline_climate=df_baseline_climate,
        df_baseline_health=df_baseline_health,
        df_mp_climate=df_mp_climate,
        df_mp_health=df_mp_health,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model=RCM_MODELS[0],
        base_year=2024,
        discounting_method='public',
        verbose=False
    )
    
    # Verify list-based collection with pandas operations
    assert list_collection_used, \
        "List-based collection should use pd.concat to combine yearly values"
    assert pandas_sum_used, \
        "Pandas DataFrame.sum should be used to sum across years"


def test_final_masking_implementation(
        sample_homes_df, 
        df_climate_health_damages,
        mock_define_scenario_settings,
        mock_discount_factor,
        monkeypatch):
    """
    Test Step 5: Final masking with apply_final_masking().
    
    This test verifies that public impact columns are properly tracked and masked
    at the end of the calculation process.
    """
    # Track which columns are passed to apply_temporary_validation_and_mask
    masking_columns_captured = {}
    
    def mock_apply_masking(df_original, df_new_columns, all_columns_to_mask, verbose=True):
        """Mock to track calls to apply_temporary_validation_and_mask."""
        nonlocal masking_columns_captured
        # Make a deep copy to ensure we store all columns
        masking_columns_captured = {k: list(v) for k, v in all_columns_to_mask.items()}
        
        # Return the original df for simplicity
        return df_original
    
    # Apply monkeypatching
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.apply_temporary_validation_and_mask',
        mock_apply_masking
    )
    
    # Call the main function
    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = df_climate_health_damages
    
    result = calculate_public_npv(
        df=sample_homes_df,
        df_baseline_climate=df_baseline_climate,
        df_baseline_health=df_baseline_health,
        df_mp_climate=df_mp_climate,
        df_mp_health=df_mp_health,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model=RCM_MODELS[0],
        base_year=2024,
        discounting_method='public',
        verbose=False
    )
    
    # Verify NPV columns are tracked for masking
    # Should have columns for climate, health, and public NPVs
    for category in EQUIPMENT_SPECS.keys():
        assert category in masking_columns_captured, \
            f"Category '{category}' should be in masking columns"
            
        # Check if climate, health, and public NPV columns are tracked
        tracked_columns = masking_columns_captured[category]
        
        # There should be at least one NPV column for this category
        assert any('_npv_' in col for col in tracked_columns), \
            f"No NPV columns found for category '{category}'"


def test_all_validation_steps_integrated(
        sample_homes_df, 
        df_climate_health_damages,
        mock_define_scenario_settings,
        mock_discount_factor,
        monkeypatch):
    """
    Test integration of all 5 steps of the validation framework.
    
    This test verifies that calculate_public_npv correctly
    implements all five steps of the validation framework in sequence,
    with each step building on the previous ones.
    """
    # Track which steps have been executed
    executed_steps = {
        'mask_initialization': False,
        'series_initialization': False,
        'valid_calculation': False,
        'list_collection': False,
        'final_masking': False
    }
    
    # Mock each step to track execution
    original_init_tracking = initialize_validation_tracking
    def mock_init_tracking(*args, **kwargs):
        executed_steps['mask_initialization'] = True
        return original_init_tracking(*args, **kwargs)
    
    original_create_series = create_retrofit_only_series
    def mock_create_series(*args, **kwargs):
        executed_steps['series_initialization'] = True
        return original_create_series(*args, **kwargs)
    
    original_avoided_values = calculate_avoided_values
    def mock_avoided_values(*args, **kwargs):
        if 'retrofit_mask' in kwargs and kwargs['retrofit_mask'] is not None:
            executed_steps['valid_calculation'] = True
        return original_avoided_values(*args, **kwargs)
    
    original_concat = pd.concat
    def mock_concat(*args, **kwargs):
        executed_steps['list_collection'] = True
        return original_concat(*args, **kwargs)
    
    original_apply_masking = apply_temporary_validation_and_mask
    def mock_apply_masking(*args, **kwargs):
        executed_steps['final_masking'] = True
        return original_apply_masking(*args, **kwargs)
    
    # Apply monkeypatching
    with monkeypatch.context() as m:
        m.setattr('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.initialize_validation_tracking', mock_init_tracking)
        m.setattr('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.create_retrofit_only_series', mock_create_series)
        m.setattr('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.calculate_avoided_values', mock_avoided_values)
        m.setattr('pandas.concat', mock_concat)
        m.setattr('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.apply_temporary_validation_and_mask', mock_apply_masking)
        
        # Call the main function
        df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = df_climate_health_damages
        
        result = calculate_public_npv(
            df=sample_homes_df,
            df_baseline_climate=df_baseline_climate,
            df_baseline_health=df_baseline_health,
            df_mp_climate=df_mp_climate,
            df_mp_health=df_mp_health,
            menu_mp=8,
            policy_scenario='AEO2023 Reference Case',
            rcm_model=RCM_MODELS[0],
            base_year=2024,
            discounting_method='public',
            verbose=False
        )
    
    # Verify all steps were executed
    for step, executed in executed_steps.items():
        assert executed, f"Validation step '{step}' was not executed"


# -------------------------------------------------------------------------
#                 BASIC CALCULATION TESTS
# -------------------------------------------------------------------------

@pytest.mark.parametrize("menu_mp, policy_scenario, rcm_model, cr_function, discounting_method", [
    (0, "No Inflation Reduction Act", RCM_MODELS[0], CR_FUNCTIONS[0], "public"),
    (8, "AEO2023 Reference Case", RCM_MODELS[0], CR_FUNCTIONS[0], "public"),
    (8, "AEO2023 Reference Case", RCM_MODELS[1], CR_FUNCTIONS[1], "private_fixed"),
])
def test_calculate_public_npv_successful_execution(
    sample_homes_df, 
    df_climate_health_damages,
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
    
    This test verifies that the function:
    1. Returns a DataFrame with expected columns
    2. Calculates NPV values correctly for each category
    3. Produces properly formatted output with correct value ranges
    """
    # Get scenario specific prefix and damages
    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = df_climate_health_damages
    
    # Call the function with the specified parameters
    result = calculate_public_npv(
        df=sample_homes_df,
        df_baseline_climate=df_baseline_climate,
        df_baseline_health=df_baseline_health,
        df_mp_climate=df_mp_climate,
        df_mp_health=df_mp_health,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        rcm_model=rcm_model,
        base_year=2024,
        discounting_method=discounting_method,
        verbose=False
    )
    
    # Check that the result is a DataFrame with the same index as input df
    assert isinstance(result, pd.DataFrame)
    assert result.index.equals(sample_homes_df.index)
    
    # Determine expected scenario prefix based on inputs
    scenario_prefix, _, _, _, _, _ = mock_define_scenario_settings(menu_mp, policy_scenario)
    
    # Check that expected columns are present (at least for the first SCC assumption)
    for category in EQUIPMENT_SPECS.keys():
        # Check climate NPV columns
        for scc in SCC_ASSUMPTIONS[:1]:  # Check just first SCC assumption for brevity
            climate_npv_col = f'{scenario_prefix}{category}_climate_npv_{scc}'
            assert climate_npv_col in result.columns, f"Missing climate NPV column: {climate_npv_col}"
        
        # Check health NPV columns
        health_npv_col = f'{scenario_prefix}{category}_health_npv_{rcm_model}_{cr_function}'
        assert health_npv_col in result.columns, f"Missing health NPV column: {health_npv_col}"
        
        # Check public NPV columns (combined climate and health)
        for scc in SCC_ASSUMPTIONS[:1]:  # Check just first SCC assumption for brevity
            public_npv_col = f'{scenario_prefix}{category}_public_npv_{scc}_{rcm_model}_{cr_function}'
            assert public_npv_col in result.columns, f"Missing public NPV column: {public_npv_col}"
    
    # Check that valid homes have non-NaN values and invalid homes have NaN values
    for category in EQUIPMENT_SPECS.keys():
        valid_mask = sample_homes_df[f'include_{category}']
        
        # Check just one NPV column for brevity
        scc = SCC_ASSUMPTIONS[0]
        public_npv_col = f'{scenario_prefix}{category}_public_npv_{scc}_{rcm_model}_{cr_function}'
        
        if public_npv_col in result.columns:
            # Valid homes should have non-NaN values
            for idx in valid_mask[valid_mask].index:
                assert not pd.isna(result.loc[idx, public_npv_col]), \
                    f"Valid home at index {idx} should have a non-NaN value for {public_npv_col}"
            
            # Invalid homes should have NaN values
            for idx in valid_mask[~valid_mask].index:
                assert pd.isna(result.loc[idx, public_npv_col]), \
                    f"Invalid home at index {idx} should have NaN for {public_npv_col}"


def test_calculate_lifetime_damages_grid_scenario_basic(
        sample_homes_df, 
        df_climate_health_damages,
        mock_define_scenario_settings,
        mock_discount_factor):
    """
    Test that calculate_lifetime_damages_grid_scenario correctly calculates NPV values.
    
    This test verifies that the function:
    1. Calculates climate and health NPVs correctly
    2. Returns a dictionary with the expected keys
    3. Produces values with the correct format and ranges
    """
    # Get scenario specific damages
    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = df_climate_health_damages
    
    # Call the function directly to test it
    menu_mp = 8
    policy_scenario = "AEO2023 Reference Case"
    rcm_model = RCM_MODELS[0]
    cr_function = CR_FUNCTIONS[0]
    
    # Get the expected scenario prefix
    scenario_prefix, _, _, _, _, _ = mock_define_scenario_settings(menu_mp, policy_scenario)
    
    # Create all_columns_to_mask dictionary
    all_columns_to_mask = {category: [] for category in EQUIPMENT_SPECS.keys()}
    
    # Call the function being tested
    result_dict = calculate_lifetime_damages_grid_scenario(
        df_copy=sample_homes_df,
        df_baseline_climate=df_baseline_climate,
        df_baseline_health=df_baseline_health,
        df_mp_climate=df_mp_climate,
        df_mp_health=df_mp_health,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        rcm_model=rcm_model,
        cr_function=cr_function,
        base_year=2024,
        discounting_method="public",
        all_columns_to_mask=all_columns_to_mask,
        verbose=False
    )
    
    # Verify result is a dictionary with expected keys
    assert isinstance(result_dict, dict), "Result should be a dictionary"
    
    # Check if NPV columns exist for each equipment category
    for category in EQUIPMENT_SPECS.keys():
        # Check just one SCC assumption for simplicity
        scc = SCC_ASSUMPTIONS[0]
        
        climate_npv_key = f'{scenario_prefix}{category}_climate_npv_{scc}'
        health_npv_key = f'{scenario_prefix}{category}_health_npv_{rcm_model}_{cr_function}'
        public_npv_key = f'{scenario_prefix}{category}_public_npv_{scc}_{rcm_model}_{cr_function}'
        
        assert climate_npv_key in result_dict, f"Missing climate NPV key: {climate_npv_key}"
        assert health_npv_key in result_dict, f"Missing health NPV key: {health_npv_key}"
        assert public_npv_key in result_dict, f"Missing public NPV key: {public_npv_key}"
        
        # Verify values are Series with the correct index
        assert isinstance(result_dict[climate_npv_key], pd.Series)
        assert isinstance(result_dict[health_npv_key], pd.Series)
        assert isinstance(result_dict[public_npv_key], pd.Series)
        
        assert result_dict[climate_npv_key].index.equals(sample_homes_df.index)
        assert result_dict[health_npv_key].index.equals(sample_homes_df.index)
        assert result_dict[public_npv_key].index.equals(sample_homes_df.index)
        
        # Verify values are rounded to 2 decimal places
        for key in [climate_npv_key, health_npv_key, public_npv_key]:
            for value in result_dict[key].dropna():
                assert value == round(value, 2), f"Value in {key} is not rounded to 2 decimal places"
        
        # Verify public NPV equals climate NPV + health NPV for valid homes
        valid_mask = sample_homes_df[f'include_{category}']
        for idx in valid_mask[valid_mask].index:
            climate_npv = result_dict[climate_npv_key].loc[idx]
            health_npv = result_dict[health_npv_key].loc[idx]
            public_npv = result_dict[public_npv_key].loc[idx]
            
            if not pd.isna(climate_npv) and not pd.isna(health_npv):
                # Allow for small differences due to rounding
                assert abs((climate_npv + health_npv) - public_npv) < 0.02, \
                    f"Public NPV should equal climate NPV + health NPV for index {idx}"
    
    # Check that all_columns_to_mask has been updated
    for category in EQUIPMENT_SPECS.keys():
        assert len(all_columns_to_mask[category]) > 0, \
            f"all_columns_to_mask[{category}] should contain tracked columns"


# -------------------------------------------------------------------------
#                 ERROR HANDLING TESTS
# -------------------------------------------------------------------------

@pytest.mark.parametrize("param_name, invalid_value, error_type", [
    ("menu_mp", "invalid_mp", ValueError),
    ("policy_scenario", "Invalid Policy", ValueError),
    ("rcm_model", "InvalidModel", ValueError),
    ("discounting_method", "invalid_method", ValueError),
])
def test_invalid_parameters(
    sample_homes_df, 
    df_climate_health_damages,
    mock_define_scenario_settings, 
    mock_discount_factor, 
    param_name, 
    invalid_value, 
    error_type
):
    """
    Test that invalid parameters cause the function to raise appropriate exceptions.
    """
    # Get damages dataframes
    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = df_climate_health_damages
    
    # Default valid parameters
    params = {
        "df": sample_homes_df,
        "df_baseline_climate": df_baseline_climate,
        "df_baseline_health": df_baseline_health,
        "df_mp_climate": df_mp_climate,
        "df_mp_health": df_mp_health,
        "menu_mp": 8,
        "policy_scenario": "AEO2023 Reference Case",
        "rcm_model": RCM_MODELS[0],
        "base_year": 2024,
        "discounting_method": "public"
    }
    
    # Override the specified parameter with an invalid value
    params[param_name] = invalid_value
    
    with pytest.raises(error_type):
        calculate_public_npv(**params)


def test_missing_required_columns(
    sample_homes_df, 
    df_climate_health_damages,
    mock_define_scenario_settings, 
    mock_discount_factor
):
    """
    Test that missing required columns cause the function to handle errors appropriately.
    """
    # Get damages dataframes
    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = df_climate_health_damages
    
    # Create modified DataFrames with missing columns
    df_baseline_climate_modified = df_baseline_climate.drop(columns=[df_baseline_climate.columns[0]])
    
    with pytest.raises((ValueError, KeyError)):
        calculate_public_npv(
            df=sample_homes_df,
            df_baseline_climate=df_baseline_climate_modified,
            df_baseline_health=df_baseline_health,
            df_mp_climate=df_mp_climate,
            df_mp_health=df_mp_health,
            menu_mp=8,
            policy_scenario="AEO2023 Reference Case",
            rcm_model=RCM_MODELS[0]
        )


def test_index_misalignment(
    sample_homes_df, 
    df_climate_health_damages,
    mock_define_scenario_settings, 
    mock_discount_factor
):
    """
    Test that index misalignment between DataFrames is handled properly.
    """
    # Get damages dataframes
    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = df_climate_health_damages
    
    # Create a DataFrame with a different index
    df_misaligned = pd.DataFrame({'id': [10, 11, 12]})
    df_misaligned.set_index('id', inplace=True)
    
    # Function should handle index misalignment gracefully
    result = calculate_public_npv(
        df=df_misaligned,
        df_baseline_climate=df_baseline_climate,
        df_baseline_health=df_baseline_health,
        df_mp_climate=df_mp_climate,
        df_mp_health=df_mp_health,
        menu_mp=8,
        policy_scenario="AEO2023 Reference Case",
        rcm_model=RCM_MODELS[0]
    )
    
    # Verify the result has the same index as df_misaligned
    assert result.index.equals(df_misaligned.index)
    
    # All NPV columns should have NaN values due to index mismatch
    for col in result.columns:
        if '_npv_' in col:
            assert result[col].isna().all(), \
                f"Column {col} should have only NaN values due to index misalignment"


@pytest.mark.parametrize("category, lifetime, expected_behavior", [
    ("zero_lifetime_cat", 0, "zeros"),
    ("negative_lifetime_cat", -1, "handle_gracefully"),
])
def test_boundary_lifetime_equipment(
    sample_homes_df, 
    df_climate_health_damages,
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
    # Get damages dataframes
    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = df_climate_health_damages
    
    # Create test categories with specified lifetimes
    custom_equipment_specs({category: lifetime})
    
    # Function should handle boundary cases appropriately
    result = calculate_public_npv(
        df=sample_homes_df,
        df_baseline_climate=df_baseline_climate,
        df_baseline_health=df_baseline_health,
        df_mp_climate=df_mp_climate,
        df_mp_health=df_mp_health,
        menu_mp=8,
        policy_scenario="AEO2023 Reference Case",
        rcm_model=RCM_MODELS[0]
    )
    
    # Verify the function completed and the result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # For zero lifetime, check that NPV columns contain zeros or NaNs
    if expected_behavior == "zeros":
        scenario_prefix, _, _, _, _, _ = mock_define_scenario_settings(8, "AEO2023 Reference Case")
        climate_col = f'{scenario_prefix}{category}_climate_npv_{SCC_ASSUMPTIONS[0]}'
        
        if climate_col in result.columns:
            # All non-NaN values should be zero
            for value in result[climate_col].dropna():
                assert value == 0.0, f"Expected zeros for {climate_col}"


@pytest.mark.parametrize("category, lifetime", [
    ("single_year_cat", 1),
])
def test_single_year_equipment(
    sample_homes_df, 
    df_climate_health_damages,
    mock_define_scenario_settings, 
    mock_discount_factor, 
    custom_equipment_specs,
    category, 
    lifetime
):
    """
    Test cases where equipment has a single year lifetime (boundary condition).
    """
    # Get damages dataframes
    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = df_climate_health_damages
    
    # Create test category with specified lifetime
    custom_equipment_specs({category: lifetime})
    
    # Call the function
    result = calculate_public_npv(
        df=sample_homes_df,
        df_baseline_climate=df_baseline_climate,
        df_baseline_health=df_baseline_health,
        df_mp_climate=df_mp_climate,
        df_mp_health=df_mp_health,
        menu_mp=8,
        policy_scenario="AEO2023 Reference Case",
        rcm_model=RCM_MODELS[0],
        base_year=2024
    )
    
    # Verify the function completed and the result is a DataFrame
    assert isinstance(result, pd.DataFrame)


# -------------------------------------------------------------------------
#                 PARAMETRIZED TESTS ACROSS CATEGORIES
# -------------------------------------------------------------------------

def test_different_categories(
    sample_homes_df, 
    df_climate_health_damages,
    mock_define_scenario_settings, 
    mock_discount_factor, 
    category
):
    """
    Test calculation across different equipment categories.
    
    This parametrized test verifies that calculate_public_npv
    works correctly for all equipment categories, applying category-specific
    validation.
    """
    # Get damages dataframes
    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = df_climate_health_damages
    
    # Call the main function
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    rcm_model = RCM_MODELS[0]
    
    result = calculate_public_npv(
        df=sample_homes_df,
        df_baseline_climate=df_baseline_climate,
        df_baseline_health=df_baseline_health,
        df_mp_climate=df_mp_climate,
        df_mp_health=df_mp_health,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        rcm_model=rcm_model
    )
    
    # Get expected scenario prefix
    scenario_prefix, _, _, _, _, _ = mock_define_scenario_settings(menu_mp, policy_scenario)
    
    # Verify at least one NPV column exists for this category
    scc = SCC_ASSUMPTIONS[0]
    cr = CR_FUNCTIONS[0]
    
    climate_npv_col = f'{scenario_prefix}{category}_climate_npv_{scc}'
    health_npv_col = f'{scenario_prefix}{category}_health_npv_{rcm_model}_{cr}'
    public_npv_col = f'{scenario_prefix}{category}_public_npv_{scc}_{rcm_model}_{cr}'
    
    # At least one of these columns should exist
    assert any(col in result.columns for col in [climate_npv_col, health_npv_col, public_npv_col]), \
        f"No NPV columns found for category '{category}'"
    
    # Verify values are masked based on inclusion flags
    valid_mask = sample_homes_df[f'include_{category}']
    
    for col in [col for col in [climate_npv_col, health_npv_col, public_npv_col] if col in result.columns]:
        # Valid homes should have non-NaN values
        for idx in valid_mask[valid_mask].index:
            assert not pd.isna(result.loc[idx, col]), \
                f"Valid home at index {idx} should have a non-NaN value for {col}"
        
        # Invalid homes should have NaN values
        for idx in valid_mask[~valid_mask].index:
            assert pd.isna(result.loc[idx, col]), \
                f"Invalid home at index {idx} should have NaN for {col}"


def test_different_rcm_cr_combinations(
    sample_homes_df, 
    df_climate_health_damages,
    mock_define_scenario_settings, 
    mock_discount_factor, 
    rcm_model, 
    cr_function
):
    """
    Test calculation with different RCM models and CR functions.
    
    This parametrized test verifies that calculate_public_npv
    works correctly with different model combinations.
    """
    # Get damages dataframes
    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = df_climate_health_damages
    
    # Call the main function
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    
    result = calculate_public_npv(
        df=sample_homes_df,
        df_baseline_climate=df_baseline_climate,
        df_baseline_health=df_baseline_health,
        df_mp_climate=df_mp_climate,
        df_mp_health=df_mp_health,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        rcm_model=rcm_model
    )
    
    # Get expected scenario prefix
    scenario_prefix, _, _, _, _, _ = mock_define_scenario_settings(menu_mp, policy_scenario)
    
    # Verify health NPV columns exist for this RCM/CR combination
    category = 'heating'  # Test with one category for simplicity
    health_npv_col = f'{scenario_prefix}{category}_health_npv_{rcm_model}_{cr_function}'
    
    assert health_npv_col in result.columns, \
        f"Health NPV column '{health_npv_col}' should exist for RCM={rcm_model}, CR={cr_function}"


def test_different_policy_scenarios(
    sample_homes_df, 
    df_climate_health_damages,
    mock_define_scenario_settings, 
    mock_discount_factor, 
    policy_scenario
):
    """
    Test calculation with different policy scenarios.
    
    This parametrized test verifies that calculate_public_npv
    works correctly for different policy scenarios, using appropriate
    column naming.
    """
    # Get damages dataframes
    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = df_climate_health_damages
    
    # Call the main function
    menu_mp = 8
    rcm_model = RCM_MODELS[0]
    
    result = calculate_public_npv(
        df=sample_homes_df,
        df_baseline_climate=df_baseline_climate,
        df_baseline_health=df_baseline_health,
        df_mp_climate=df_mp_climate,
        df_mp_health=df_mp_health,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        rcm_model=rcm_model
    )
    
    # Determine expected scenario prefix based on inputs
    scenario_prefix, _, _, _, _, _ = mock_define_scenario_settings(menu_mp, policy_scenario)
    
    # Verify at least one NPV column exists with the correct prefix
    category = 'heating'  # Test with one category for simplicity
    scc = SCC_ASSUMPTIONS[0]
    cr = CR_FUNCTIONS[0]
    
    public_npv_col = f'{scenario_prefix}{category}_public_npv_{scc}_{rcm_model}_{cr}'
    
    assert public_npv_col in result.columns, \
        f"Public NPV column '{public_npv_col}' should exist for policy scenario '{policy_scenario}'"


def test_different_discounting_methods(
    sample_homes_df, 
    df_climate_health_damages,
    mock_define_scenario_settings, 
    mock_discount_factor, 
    discounting_method
):
    """
    Test calculation with different discounting methods.
    
    This parametrized test verifies that calculate_public_npv
    works correctly with different discounting methods.
    """
    # Get damages dataframes
    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = df_climate_health_damages
    
    # Call the main function
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    rcm_model = RCM_MODELS[0]
    
    result = calculate_public_npv(
        df=sample_homes_df,
        df_baseline_climate=df_baseline_climate,
        df_baseline_health=df_baseline_health,
        df_mp_climate=df_mp_climate,
        df_mp_health=df_mp_health,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        rcm_model=rcm_model,
        discounting_method=discounting_method
    )
    
    # Get expected scenario prefix
    scenario_prefix, _, _, _, _, _ = mock_define_scenario_settings(menu_mp, policy_scenario)
    
    # Verify at least one NPV column exists
    category = 'heating'  # Test with one category for simplicity
    scc = SCC_ASSUMPTIONS[0]
    cr = CR_FUNCTIONS[0]
    
    public_npv_col = f'{scenario_prefix}{category}_public_npv_{scc}_{rcm_model}_{cr}'
    
    assert public_npv_col in result.columns, \
        f"Public NPV column '{public_npv_col}' should exist for discounting method '{discounting_method}'"


# -------------------------------------------------------------------------
#                 EDGE CASE TESTS
# -------------------------------------------------------------------------

def test_all_invalid_homes(
    sample_homes_df, 
    df_climate_health_damages,
    mock_define_scenario_settings, 
    mock_discount_factor
):
    """
    Test calculation when all homes are invalid.
    """
    # Get damages dataframes
    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = df_climate_health_damages
    
    # Create modified DataFrame with all homes invalid
    df_all_invalid = sample_homes_df.copy()
    for category in EQUIPMENT_SPECS.keys():
        df_all_invalid[f'include_{category}'] = False
    
    # Call the main function
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    rcm_model = RCM_MODELS[0]
    
    result = calculate_public_npv(
        df=df_all_invalid,
        df_baseline_climate=df_baseline_climate,
        df_baseline_health=df_baseline_health,
        df_mp_climate=df_mp_climate,
        df_mp_health=df_mp_health,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        rcm_model=rcm_model
    )
    
    # Get expected scenario prefix
    scenario_prefix, _, _, _, _, _ = mock_define_scenario_settings(menu_mp, policy_scenario)
    
    # Verify all NPV columns contain only NaN values
    for category in EQUIPMENT_SPECS.keys():
        for scc in SCC_ASSUMPTIONS[:1]:  # Check just first SCC assumption for brevity
            # Climate NPV
            climate_npv_col = f'{scenario_prefix}{category}_climate_npv_{scc}'
            if climate_npv_col in result.columns:
                assert result[climate_npv_col].isna().all(), \
                    f"Column {climate_npv_col} should contain only NaN values when all homes are invalid"
            
            # Health NPV
            for cr in CR_FUNCTIONS[:1]:  # Check just first CR function for brevity
                health_npv_col = f'{scenario_prefix}{category}_health_npv_{rcm_model}_{cr}'
                if health_npv_col in result.columns:
                    assert result[health_npv_col].isna().all(), \
                        f"Column {health_npv_col} should contain only NaN values when all homes are invalid"
                    

if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
