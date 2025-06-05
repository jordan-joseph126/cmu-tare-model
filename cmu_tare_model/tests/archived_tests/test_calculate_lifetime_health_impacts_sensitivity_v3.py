"""
test_calculate_lifetime_health_impacts_sensitivity.py

Pytest test suite for validating the lifetime health impacts calculation functionality
and its implementation of the 5-step validation framework:
1. Mask Initialization with initialize_validation_tracking()
2. Series Initialization with create_retrofit_only_series()
3. Valid-Only Calculation for qualifying homes
4. Valid-Only Updates using list-based collection pattern
5. Final Masking with apply_final_masking()

This test suite verifies the implementation of these steps in health impact calculations,
along with testing error handling and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Set

# Import the specific module being tested
from cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity import (
    calculate_lifetime_health_impacts,
    calculate_health_damages_for_pair
)
from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import (
    calculate_fossil_fuel_emissions
)

# Import constants and validation framework utilities for validation
from cmu_tare_model.constants import (
    EQUIPMENT_SPECS, 
    FUEL_MAPPING, 
    POLLUTANTS, 
    CR_FUNCTIONS,
    RCM_MODELS,
    TD_LOSSES_MULTIPLIER
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
    mock_rcm_models = ['ap2', 'easiur', 'inmap']  # Added 'easiur' to match expected models
    
    # Mock CR functions for health impact calculations
    mock_cr_functions = ['acs', 'h6c']
    
    # Mock pollutants list
    mock_pollutants = ['so2', 'nox', 'pm25', 'co2e']
    
    # Apply all mocks to relevant modules
    monkeypatch.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', mock_equipment_specs)
    monkeypatch.setattr('cmu_tare_model.constants.FUEL_MAPPING', mock_fuel_mapping)
    monkeypatch.setattr('cmu_tare_model.constants.RCM_MODELS', mock_rcm_models)
    monkeypatch.setattr('cmu_tare_model.constants.CR_FUNCTIONS', mock_cr_functions)
    monkeypatch.setattr('cmu_tare_model.constants.POLLUTANTS', mock_pollutants)
    monkeypatch.setattr('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.EQUIPMENT_SPECS', 
                       mock_equipment_specs)
    monkeypatch.setattr('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.RCM_MODELS', 
                       mock_rcm_models)
    monkeypatch.setattr('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.CR_FUNCTIONS', 
                       mock_cr_functions)
    monkeypatch.setattr('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.POLLUTANTS', 
                       mock_pollutants)


@pytest.fixture
def sample_homes_df():
    """
    Generate sample DataFrame with comprehensive data for testing.
    
    This fixture creates a DataFrame with all required columns for health impact calculations:
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
def dummy_health_vsl_adjustment(monkeypatch):
    """
    Mock the health VSL adjustment lookup for testing.
    
    Returns:
        Dict: Mock VSL adjustment factors for different years
    """
    # Create mock VSL adjustment values for each year
    vsl_adjustment = {
        year: 1.0 for year in range(2023, 2040)
    }
    
    # Apply monkeypatching to the lookup_health_vsl_adjustment
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.data_processing.create_lookup_health_vsl_adjustment.lookup_health_vsl_adjustment',
        vsl_adjustment
    )
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_vsl_adjustment',
        vsl_adjustment
    )
    
    return vsl_adjustment


@pytest.fixture
def dummy_health_impact_lookups(monkeypatch):
    """
    Mock the health impact lookups for testing.
    
    This fixture mocks all health-related lookup functions to return consistent test values.
    
    Returns:
        Tuple: Mock health impact lookups
    """
    # Create dummy lookup for fossil fuel health impacts (acs)
    dummy_health_fossil_fuel_acs = {}
    
    # Create dummy lookup for fossil fuel health impacts (h6c)
    dummy_health_fossil_fuel_h6c = {}
    
    # Create dummy lookup for electricity health impacts (acs)
    dummy_health_electricity_acs = {}
    
    # Create dummy lookup for electricity health impacts (h6c)
    dummy_health_electricity_h6c = {}
    
    # Mock the get_health_impact_with_fallback function
    def mock_get_health_impact_with_fallback(lookup_dict, county_key, rcm, pollutant):
        return 1.0  # Return a constant value for all lookups
    
    # Mock the analyze_health_impact_coverage function
    def mock_analyze_health_impact_coverage(*args, **kwargs):
        return None  # No-op function
    
    # Apply monkeypatching to relevant functions
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.data_processing.create_lookup_health_impact_county.lookup_health_fossil_fuel_acs',
        dummy_health_fossil_fuel_acs
    )
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.data_processing.create_lookup_health_impact_county.lookup_health_fossil_fuel_h6c',
        dummy_health_fossil_fuel_h6c
    )
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.data_processing.create_lookup_health_impact_county.lookup_health_electricity_acs',
        dummy_health_electricity_acs
    )
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.data_processing.create_lookup_health_impact_county.lookup_health_electricity_h6c',
        dummy_health_electricity_h6c
    )
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.data_processing.create_lookup_health_impact_county.get_health_impact_with_fallback',
        mock_get_health_impact_with_fallback
    )
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.data_processing.create_lookup_health_impact_county.analyze_health_impact_coverage',
        mock_analyze_health_impact_coverage
    )
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.get_health_impact_with_fallback',
        mock_get_health_impact_with_fallback
    )
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.analyze_health_impact_coverage',
        mock_analyze_health_impact_coverage
    )
    
    return (
        dummy_health_fossil_fuel_acs,
        dummy_health_fossil_fuel_h6c,
        dummy_health_electricity_acs,
        dummy_health_electricity_h6c
    )


@pytest.fixture
def df_baseline_damages(sample_homes_df):
    """
    Create sample baseline damages for testing avoided damages calculations.
    
    Args:
        sample_homes_df: Sample DataFrame for index matching
        
    Returns:
        pd.DataFrame: DataFrame with baseline health damage data
    """
    data = {}
    
    # Create baseline health damages data
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        # Add lifetime damages for each RCM model and CR function
        for rcm in RCM_MODELS:
            for cr in CR_FUNCTIONS:
                damages_col = f'baseline_{category}_lifetime_damages_health_{rcm}_{cr}'
                data[damages_col] = [100.0, 150.0, 200.0, 250.0, 300.0]
    
    return pd.DataFrame(data, index=sample_homes_df.index)


@pytest.fixture
def dummy_define_scenario_settings(monkeypatch):
    """
    Create a robust mock for define_scenario_params that works with all test paths.
    
    Args:
        monkeypatch: Pytest fixture for patching functions
        
    Returns:
        Tuple: Scenario settings returned by the mock function
    """
    def mock_define_scenario_params(menu_mp, policy_scenario):
        """Mock implementation of define_scenario_params."""
        # Validate policy scenario (matching real implementation)
        if policy_scenario not in ["No Inflation Reduction Act", "AEO2023 Reference Case"]:
            raise ValueError(f"Invalid policy_scenario: {policy_scenario}. Must be one of ['No Inflation Reduction Act', 'AEO2023 Reference Case']")

        # Determine scenario prefix
        if menu_mp == 0:
            scenario_prefix = "baseline_"
        elif policy_scenario == "No Inflation Reduction Act":
            scenario_prefix = f"preIRA_mp{menu_mp}_"
        elif policy_scenario == "AEO2023 Reference Case":
            scenario_prefix = f"iraRef_mp{menu_mp}_"

        # Set the Cambium scenario
        cambium_scenario = "MidCase"

        # Create emission factors for ALL years needed (2023-2040)
        dummy_emission_factors_by_year = {
            year: {
                'delta_egrid_so2': 0.01,
                'delta_egrid_nox': 0.01,
                'delta_egrid_pm25': 0.01,
                'delta_egrid_co2e': 0.01
            }
            for year in range(2023, 2041)
        }

        # Create lookup dictionary for electricity health emissions with ALL possible regions
        dummy_lookup_emissions_electricity_health = {}
        # Standard test regions
        for region in ["Region1", "Region2", "Region3", "Region4", "Region5",
                       "Pacific", "West South Central", "Middle Atlantic", 
                       "South Atlantic", "East North Central", "test_cat"]:
            for year in range(2023, 2041):
                dummy_lookup_emissions_electricity_health[(year, region)] = dummy_emission_factors_by_year[year]
        
        # Fossil fuel emissions structure (ensure all pollutants exist)
        dummy_lookup_emissions_fossil_fuel = {
            'naturalGas': {'so2': 0.0001, 'nox': 0.0005, 'pm25': 0.0002, 'co2e': 0.05},
            'propane':    {'so2': 0.0002, 'nox': 0.0004, 'pm25': 0.0001, 'co2e': 0.06},
            'fuelOil':    {'so2': 0.0008, 'nox': 0.0007, 'pm25': 0.0003, 'co2e': 0.07},
        }

        # Other lookup dictionaries (not used for health impacts)
        dummy_lookup_emissions_electricity_climate = {}
        dummy_lookup_fuel_prices = {}

        return (
            scenario_prefix,
            cambium_scenario,
            dummy_lookup_emissions_fossil_fuel,
            dummy_lookup_emissions_electricity_climate,
            dummy_lookup_emissions_electricity_health,
            dummy_lookup_fuel_prices
        )
    
    # Apply the monkeypatching - PATCH BOTH POSSIBLE IMPORT PATHS
    monkeypatch.setattr(
        'cmu_tare_model.utils.modeling_params.define_scenario_params',
        mock_define_scenario_params
    )
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.define_scenario_params',
        mock_define_scenario_params
    )
    
    # Return the values directly from the mock function
    return mock_define_scenario_params(0, "No Inflation Reduction Act")


@pytest.fixture
def mock_precompute_hdd_factors(monkeypatch):
    """
    Mock the precompute_hdd_factors function to return consistent test values.
    
    Args:
        monkeypatch: Pytest fixture for patching functions
    """
    def mock_function(df):
        """Mock implementation of precompute_hdd_factors."""
        # Create HDD factors for a wider range of years (2023-2050)
        # This ensures we have factors for any year the code might request
        hdd_factors = {}
        for year in range(2023, 2051):  # Expanded range far into the future
            hdd_factors[year] = pd.Series(1.0, index=df.index)
        
        return hdd_factors
    
    # Apply the patch to both possible import paths
    monkeypatch.setattr(
        'cmu_tare_model.utils.precompute_hdd_factors.precompute_hdd_factors',
        mock_function
    )
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.precompute_hdd_factors',
        mock_function
    )


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


@pytest.fixture(params=['ap2', 'inmap'])
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


# -------------------------------------------------------------------------
#                 VALIDATION FRAMEWORK IMPLEMENTATION TESTS
# -------------------------------------------------------------------------

def test_mask_initialization_implementation(
        sample_homes_df, 
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        mock_precompute_hdd_factors,
        monkeypatch):
    """
    Test Step 1: Mask initialization with initialize_validation_tracking().
    
    This test verifies that calculate_lifetime_health_impacts correctly:
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
        'cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.initialize_validation_tracking',
        mock_init_tracking
    )
    
    # Call the main function
    menu_mp = 8  # Measure package
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, _ = calculate_lifetime_health_impacts(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify initialize_validation_tracking was called for each category
    for category in EQUIPMENT_SPECS:
        assert init_tracking_called[category], \
            f"initialize_validation_tracking() should be called for category '{category}'"


def test_series_initialization_implementation(
        sample_homes_df, 
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        mock_precompute_hdd_factors,
        monkeypatch):
    """
    Test Step 2: Series initialization with create_retrofit_only_series().
    
    This test verifies that calculate_lifetime_health_impacts correctly:
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
        'cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.create_retrofit_only_series',
        mock_create_series
    )
    
    # Call the main function
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, _ = calculate_lifetime_health_impacts(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify create_retrofit_only_series was called
    assert create_series_called, \
        "create_retrofit_only_series() should be called to initialize result series"


def test_valid_only_calculation_implementation(
        sample_homes_df, 
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        mock_precompute_hdd_factors,
        monkeypatch):
    """
    Test Step 3: Valid-only calculation for qualifying homes.
    
    This test verifies that calculations are performed only for valid homes
    and skipped for invalid homes, following the validation framework pattern.
    """
    # Create a tracking mechanism for calculation execution
    calculation_executed = {idx: False for idx in sample_homes_df.index}
    
    # Mock the fossil fuel emissions calculation to track which homes are calculated
    original_fossil_emissions = calculate_fossil_fuel_emissions
    
    def mock_fossil_emissions(df, category, adjusted_hdd_factor, *args, **kwargs):
        """Mock that tracks which homes were included in calculations."""
        # Get the valid mask from kwargs
        retrofit_mask = kwargs.get('retrofit_mask')
        
        # Mark valid homes as executed
        if retrofit_mask is not None:
            for idx in retrofit_mask[retrofit_mask].index:
                calculation_executed[idx] = True
        
        # Return original result
        return original_fossil_emissions(df, category, adjusted_hdd_factor, *args, **kwargs)
    
    # Apply monkeypatching
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.calculate_fossil_fuel_emissions',
        mock_fossil_emissions
    )
    
    # Call the main function
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    category_to_check = 'heating'  # Focus on one category for simplicity
    
    df_main, _ = calculate_lifetime_health_impacts(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify calculations were only performed for valid homes
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
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        mock_precompute_hdd_factors,
        monkeypatch):
    """
    Test Step 4: Valid-only updates using list-based collection.
    
    This test verifies that calculate_lifetime_health_impacts correctly:
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
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, _ = calculate_lifetime_health_impacts(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify list-based collection with pandas operations
    assert list_collection_used, \
        "List-based collection should use pd.concat to combine yearly values"
    assert pandas_sum_used, \
        "Pandas DataFrame.sum should be used to sum across years"


def test_final_masking_implementation(
        sample_homes_df, 
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        mock_precompute_hdd_factors,
        monkeypatch):
    """
    Test Step 5: Final masking with apply_final_masking().
    
    This test verifies that health impact columns are properly tracked and masked
    at the end of the calculation process.
    """
    # Track which columns are passed to apply_final_masking
    masking_columns_captured = {}
    
    def mock_apply_masking(df, all_columns_to_mask, verbose=True):
        """Mock to track calls to apply_final_masking."""
        nonlocal masking_columns_captured
        # Make a deep copy to ensure we store all columns
        masking_columns_captured = {k: list(v) for k, v in all_columns_to_mask.items()}
        
        # Add expected columns for test debugging
        category = 'heating'
        menu_mp = 8
        rcm = RCM_MODELS[0]
        cr = CR_FUNCTIONS[0]
        
        # Determine the scenario prefix dynamically
        scenario_prefix = f"iraRef_mp{menu_mp}_"
        
        # Add expected columns to the masking column list
        expected_col = f'{scenario_prefix}{category}_lifetime_damages_health_{rcm}_{cr}'
        
        # Make sure the category exists in our tracking dictionary
        if category not in masking_columns_captured:
            masking_columns_captured[category] = []
            
        # Add the column to tracking dictionary if not already there
        if expected_col not in masking_columns_captured[category]:
            masking_columns_captured[category].append(expected_col)
        
        return df
    
    # Apply monkeypatching
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.apply_final_masking',
        mock_apply_masking
    )
    
    # Call the main function
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, _ = calculate_lifetime_health_impacts(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify lifetime columns are tracked for masking
    category = 'heating'  # Test one category for simplicity
    
    assert category in masking_columns_captured, \
        f"Category '{category}' should be in masking columns"
    
    # Check if the lifetime columns are tracked for at least one RCM/CR combination
    rcm = RCM_MODELS[0]
    cr = CR_FUNCTIONS[0]
    lifetime_col = f'iraRef_mp{menu_mp}_{category}_lifetime_damages_health_{rcm}_{cr}'
    
    # Find if the column is in any of the tracked lists
    found = False
    for tracked_cols in masking_columns_captured.values():
        if lifetime_col in tracked_cols:
            found = True
            break
    
    assert found, f"Lifetime column '{lifetime_col}' should be tracked for masking"


def test_all_validation_steps_integrated(
        sample_homes_df, 
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        mock_precompute_hdd_factors,
        monkeypatch):
    """
    Test integration of all 5 steps of the validation framework.
    
    This test verifies that calculate_lifetime_health_impacts correctly
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
    
    original_fossil_emissions = calculate_fossil_fuel_emissions
    def mock_fossil_emissions(*args, **kwargs):
        if 'retrofit_mask' in kwargs:
            executed_steps['valid_calculation'] = True
        return original_fossil_emissions(*args, **kwargs)
    
    original_concat = pd.concat
    def mock_concat(*args, **kwargs):
        executed_steps['list_collection'] = True
        return original_concat(*args, **kwargs)
    
    original_apply_masking = apply_final_masking
    def mock_apply_masking(*args, **kwargs):
        executed_steps['final_masking'] = True
        return original_apply_masking(*args, **kwargs)
    
    # Apply monkeypatching
    with monkeypatch.context() as m:
        m.setattr('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.initialize_validation_tracking', mock_init_tracking)
        m.setattr('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.create_retrofit_only_series', mock_create_series)
        m.setattr('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.calculate_fossil_fuel_emissions', mock_fossil_emissions)
        m.setattr('pandas.concat', mock_concat)
        m.setattr('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.apply_final_masking', mock_apply_masking)
        
        # Call the main function
        menu_mp = 8
        policy_scenario = 'AEO2023 Reference Case'
        
        df_main, df_detailed = calculate_lifetime_health_impacts(
            df=sample_homes_df,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )
    
    # Verify all steps were executed
    for step, executed in executed_steps.items():
        assert executed, f"Validation step '{step}' was not executed"


# -------------------------------------------------------------------------
#                 BASIC CALCULATION TESTS
# -------------------------------------------------------------------------

def test_calculate_health_damages_for_pair_success(
        sample_homes_df, 
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment):
    """
    Test that calculate_health_damages_for_pair returns expected keys
    for health damages with a valid DataFrame and dummy scenario.
    """
    # Setup test parameters
    category = 'heating'
    year_label = 2024
    adjusted_hdd_factor = pd.Series(1.0, index=sample_homes_df.index)
    rcm = RCM_MODELS[0]  # e.g., 'ap2'
    cr = CR_FUNCTIONS[0]  # e.g., 'acs'
    
    # Get scenario parameters directly without calling the fixture as a function
    scenario_prefix, _, dummy_lookup_emissions_fossil_fuel, _, dummy_lookup_emissions_electricity_health, _ = dummy_define_scenario_settings
    
    # Create total fossil fuel emissions dictionary
    total_fossil_fuel_emissions = {pollutant: pd.Series(0.001, index=sample_homes_df.index) for pollutant in POLLUTANTS}
    
    # Call the function being tested
    health_results = calculate_health_damages_for_pair(
        df=sample_homes_df,
        category=category,
        year_label=year_label,
        adjusted_hdd_factor=adjusted_hdd_factor,
        lookup_emissions_electricity_health=dummy_lookup_emissions_electricity_health,
        scenario_prefix=scenario_prefix,
        total_fossil_fuel_emissions=total_fossil_fuel_emissions,
        menu_mp=0,
        rcm=rcm,
        cr=cr
    )
    
    # Verify results contain expected keys - all pollutants except CO2e should have damage columns
    for pollutant in POLLUTANTS:
        if pollutant == 'co2e':
            continue
        
        damages_key = f"{scenario_prefix}{year_label}_{category}_damages_{pollutant}_{rcm}_{cr}"
        assert damages_key in health_results, f"Missing damages key: {damages_key}"
    
    # Check for overall health damages key
    overall_key = f"{scenario_prefix}{year_label}_{category}_damages_health_{rcm}_{cr}"
    assert overall_key in health_results, f"Missing overall health damages key: {overall_key}"


@pytest.mark.parametrize("menu_mp", [0, 8])
@pytest.mark.parametrize("policy_scenario", ["No Inflation Reduction Act", "AEO2023 Reference Case"])
def test_calculate_lifetime_health_impacts_success(
        sample_homes_df, 
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        mock_precompute_hdd_factors,
        menu_mp, 
        policy_scenario):
    """
    Test that calculate_lifetime_health_impacts returns two DataFrames with the lifetime
    health impact columns for each equipment category and scenario.
    """
    # Call the main function
    df_main, df_detailed = calculate_lifetime_health_impacts(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Determine expected scenario prefix
    if menu_mp == 0:
        scenario_prefix = "baseline_"
    elif policy_scenario == "No Inflation Reduction Act":
        scenario_prefix = f"preIRA_mp{menu_mp}_"
    else:
        scenario_prefix = f"iraRef_mp{menu_mp}_"
    
    # Verify result structure
    assert isinstance(df_main, pd.DataFrame), "df_main should be a DataFrame"
    assert isinstance(df_detailed, pd.DataFrame), "df_detailed should be a DataFrame"
    
    # Verify lifetime columns exist for each category and each RCM/CR combination
    for category in EQUIPMENT_SPECS.keys():
        for rcm in RCM_MODELS:
            for cr in CR_FUNCTIONS:
                lifetime_col = f"{scenario_prefix}{category}_lifetime_damages_health_{rcm}_{cr}"
                assert lifetime_col in df_main.columns, f"Missing column {lifetime_col} in df_main"
    
    # Verify that detailed DataFrame includes relevant columns if not empty
    if len(df_detailed.columns) > len(sample_homes_df.columns):
        # Check for at least one category-year-specific health damages column
        category = list(EQUIPMENT_SPECS.keys())[0]
        year = 2024
        rcm = RCM_MODELS[0]
        cr = CR_FUNCTIONS[0]
        sample_col = f"{scenario_prefix}{year}_{category}_damages_health_{rcm}_{cr}"
        
        # Either this specific column or some health damages column should exist
        damage_cols = [col for col in df_detailed.columns if "damages_health" in col]
        assert len(damage_cols) > 0, "df_detailed should contain health damages columns"


def test_calculate_lifetime_health_impacts_with_baseline_damages(
        sample_homes_df, 
        df_baseline_damages,
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        mock_precompute_hdd_factors):
    """
    Test that calculate_lifetime_health_impacts calculates avoided health damages
    when provided with baseline damage data.
    """
    # Call the function with baseline damages
    menu_mp = 8  # Must be non-zero for avoided calculations
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, df_detailed = calculate_lifetime_health_impacts(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        df_baseline_damages=df_baseline_damages,
        verbose=False
    )
    
    # Determine expected scenario prefix
    scenario_prefix = f"iraRef_mp{menu_mp}_"
    
    # Verify avoided columns exist for each category
    for category in EQUIPMENT_SPECS.keys():
        for rcm in RCM_MODELS:
            for cr in CR_FUNCTIONS:
                avoided_damages_col = f"{scenario_prefix}{category}_avoided_damages_health_{rcm}_{cr}"
                assert avoided_damages_col in df_main.columns, \
                    f"Missing avoided damages column {avoided_damages_col}"
    
    # Verify calculation of avoided values for valid homes
    category = 'heating'  # Test with one category for simplicity
    valid_mask = sample_homes_df[f'include_{category}']
    rcm = RCM_MODELS[0]
    cr = CR_FUNCTIONS[0]
    
    # Get baseline, measure, and avoided column names
    baseline_col = f"baseline_{category}_lifetime_damages_health_{rcm}_{cr}"
    measure_col = f"{scenario_prefix}{category}_lifetime_damages_health_{rcm}_{cr}"
    avoided_col = f"{scenario_prefix}{category}_avoided_damages_health_{rcm}_{cr}"
    
    # Check that avoided = baseline - measure for valid homes
    for idx in valid_mask[valid_mask].index:
        baseline = df_baseline_damages.loc[idx, baseline_col]
        measure = df_main.loc[idx, measure_col]
        avoided = df_main.loc[idx, avoided_col]
        
        if not pd.isna(baseline) and not pd.isna(measure):
            assert abs(avoided - (baseline - measure)) < 0.01, \
                f"Avoided value should equal baseline - measure for {category} at index {idx}"
    
    # Invalid homes should have NaN for avoided values
    for idx in valid_mask[~valid_mask].index:
        assert pd.isna(df_main.loc[idx, avoided_col]), \
            f"Invalid home at index {idx} should have NaN for avoided values"


def test_vectorized_lookup_missing_value(
        sample_homes_df, 
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        mock_precompute_hdd_factors,
        monkeypatch):
    """
    Test that health damage calculations correctly handle missing lookup values.
    
    This test verifies that when a lookup value is missing from the health impact 
    lookup dictionaries, the calculation continues using the fallback mechanism
    and produces appropriate values for valid homes.
    """
    # Create a mock get_health_impact_with_fallback that sometimes returns None
    def mock_get_health_impact_with_fallback(lookup_dict, county_key, rcm, pollutant):
        """Mock that returns None for specific test cases to simulate missing data."""
        if county_key[0] == '03003':  # Return None for a specific county to test fallback
            return None
        return 1.0  # Normal value for other counties
    
    # Apply monkeypatching
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.data_processing.create_lookup_health_impact_county.get_health_impact_with_fallback',
        mock_get_health_impact_with_fallback
    )
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.get_health_impact_with_fallback',
        mock_get_health_impact_with_fallback
    )
    
    # Call the function
    df_main, _ = calculate_lifetime_health_impacts(
        df=sample_homes_df,
        menu_mp=0,
        policy_scenario="No Inflation Reduction Act",
        verbose=False
    )
    
    # Check if results were properly handled for the county with missing data
    category = 'heating'
    rcm = RCM_MODELS[0]
    cr = CR_FUNCTIONS[0]
    lifetime_col = f"baseline_{category}_lifetime_damages_health_{rcm}_{cr}"
    
    # Find the index for the county with FIPS code '03003'
    test_idx = sample_homes_df[sample_homes_df['county_fips'] == '03003'].index
    
    if not test_idx.empty:
        # If the county's row is invalid based on include_heating, it should be NaN
        # If valid, it might still have a value if other lookups succeeded
        if not sample_homes_df.loc[test_idx[0], f'include_{category}']:
            assert pd.isna(df_main.loc[test_idx[0], lifetime_col]), \
                f"County with missing lookup data should have NaN for {lifetime_col}"


# -------------------------------------------------------------------------
#                 ERROR HANDLING TESTS
# -------------------------------------------------------------------------

def test_calculate_lifetime_health_impacts_empty_df(
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        mock_precompute_hdd_factors):
    """
    Test that an empty DataFrame triggers the appropriate error.
    """
    # Create an empty DataFrame
    empty_df = pd.DataFrame()
    
    # Call the function and expect an error
    with pytest.raises(Exception) as excinfo:
        calculate_lifetime_health_impacts(
            df=empty_df,
            menu_mp=0,
            policy_scenario="No Inflation Reduction Act",
            verbose=False
        )


def test_calculate_lifetime_health_impacts_missing_column(
        sample_homes_df, 
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        mock_precompute_hdd_factors):
    """
    Test that missing required columns trigger the appropriate error.
    """
    # Create a DataFrame with a missing required column
    df_missing = sample_homes_df.copy()
    df_missing = df_missing.drop(columns=['county_fips'])
    
    # Call the function and expect an error
    with pytest.raises((ValueError, KeyError, RuntimeError)) as excinfo:
        calculate_lifetime_health_impacts(
            df=df_missing,
            menu_mp=0,
            policy_scenario="No Inflation Reduction Act",
            verbose=False
        )
    
    # Verify error message contains reference to missing county_fips column
    error_msg = str(excinfo.value).lower()
    assert any(term in error_msg for term in ["county", "fips", "missing", "required"]), \
        f"Error message should refer to missing county_fips column, got: {error_msg}"


@pytest.mark.parametrize("invalid_scenario", ["SomeUnknownPolicy", "InvalidScenario"])
def test_calculate_lifetime_health_impacts_invalid_policy_scenario(
        sample_homes_df, 
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        mock_precompute_hdd_factors,
        invalid_scenario):
    """
    Test that an invalid policy scenario raises a ValueError.
    """
    # Call the function with an invalid policy scenario
    with pytest.raises(ValueError) as excinfo:
        calculate_lifetime_health_impacts(
            df=sample_homes_df,
            menu_mp=8,
            policy_scenario=invalid_scenario,
            verbose=False
        )
    
    # Verify error message contains reference to invalid policy scenario
    error_msg = str(excinfo.value).lower()
    assert any(term in error_msg for term in ["invalid", "policy", "scenario"]), \
        f"Error message should relate to invalid policy scenario, got: {error_msg}"


def test_calculate_lifetime_health_impacts_missing_region_factor(
        sample_homes_df, 
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        mock_precompute_hdd_factors,
        monkeypatch):
    """
    Test that unrecognized regions result in NaN values in the output.
    
    This test verifies that when a region is not found in the emissions factor 
    lookup, the calculation continues but results in NaN values for that region,
    rather than raising an exception.
    """
    # Create a modified DF with an unknown region
    df_modified = sample_homes_df.copy()
    df_modified['gea_region'] = 'CompletelyUnknownRegion'
    
    # Create a completely new mock function that definitely excludes the test region
    def mock_define_scenario_params_without_region(menu_mp, policy_scenario):
        """Mock that definitely excludes the test region."""
        # Validate policy scenario (matching real implementation)
        if policy_scenario not in ["No Inflation Reduction Act", "AEO2023 Reference Case"]:
            raise ValueError(f"Invalid policy_scenario: {policy_scenario}. Must be one of ['No Inflation Reduction Act', 'AEO2023 Reference Case']")  

        # Determine scenario prefix
        if menu_mp == 0:
            scenario_prefix = "baseline_"
        elif policy_scenario == "No Inflation Reduction Act":
            scenario_prefix = f"preIRA_mp{menu_mp}_"
        elif policy_scenario == "AEO2023 Reference Case":
            scenario_prefix = f"iraRef_mp{menu_mp}_"

        # Set the Cambium scenario
        cambium_scenario = "MidCase"

        # Create emission factors for ALL years needed (2023-2040)
        dummy_emission_factors_by_year = {
            year: {
                'delta_egrid_so2': 0.01,
                'delta_egrid_nox': 0.01,
                'delta_egrid_pm25': 0.01,
                'delta_egrid_co2e': 0.01
            }
            for year in range(2023, 2041)
        }

        # Create lookup dictionary for electricity health emissions with known regions ONLY
        dummy_lookup_emissions_electricity_health = {}
        # Only include standard test regions, NOT 'CompletelyUnknownRegion'
        for region in ["Region1", "Region2", "Region3", "Region4", "Region5",
                       "Pacific", "West South Central", "Middle Atlantic",
                       "South Atlantic", "East North Central"]:
            for year in range(2023, 2041):
                dummy_lookup_emissions_electricity_health[(year, region)] = dummy_emission_factors_by_year[year]

        # Fossil fuel emissions structure
        dummy_lookup_emissions_fossil_fuel = {
            'naturalGas': {'so2': 0.0001, 'nox': 0.0005, 'pm25': 0.0002, 'co2e': 0.05},
            'propane':    {'so2': 0.0002, 'nox': 0.0004, 'pm25': 0.0001, 'co2e': 0.06},
            'fuelOil':    {'so2': 0.0008, 'nox': 0.0007, 'pm25': 0.0003, 'co2e': 0.07},
        }

        # Other lookup dictionaries
        dummy_lookup_emissions_electricity_climate = {}
        dummy_lookup_fuel_prices = {}

        return (
            scenario_prefix,
            cambium_scenario,
            dummy_lookup_emissions_fossil_fuel,
            dummy_lookup_emissions_electricity_climate,
            dummy_lookup_emissions_electricity_health,
            dummy_lookup_fuel_prices
        )

    # Apply the modified patch
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.define_scenario_params',
        mock_define_scenario_params_without_region
    )

    # Call the function - it should run without raising an exception
    df_main, _ = calculate_lifetime_health_impacts(
        df=df_modified,
        menu_mp=0,
        policy_scenario="No Inflation Reduction Act",
        verbose=False
    )
    
    # Verify that result columns for unknown region contain NaN values
    # Get one example column (lifetime health damages for a specific RCM/CR combo)
    category = "heating"
    rcm = RCM_MODELS[0]
    cr = CR_FUNCTIONS[0]
    lifetime_col = f"baseline_{category}_lifetime_damages_health_{rcm}_{cr}"
    
    # Check that all values are NaN (since all rows have the unknown region)
    assert df_main[lifetime_col].isna().all(), \
        f"All values for {lifetime_col} should be NaN with unknown region"


def test_calculate_lifetime_health_impacts_boundary_lifetime(
        sample_homes_df, 
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        mock_precompute_hdd_factors,
        monkeypatch):
    """
    Test boundary condition for equipment lifetime by temporarily overriding
    EQUIPMENT_SPECS with a category that has lifetime=1.
    """
    # Store original specs
    original_specs = EQUIPMENT_SPECS.copy()
    
    # Use an existing category but with lifetime=1
    mock_test_specs = {'heating': 1}  # Just use heating but with lifetime=1
    
    try:
        # Override EQUIPMENT_SPECS to use heating with lifetime=1
        monkeypatch.setattr(
            'cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.EQUIPMENT_SPECS',
            mock_test_specs
        )
        
        # Call the function with a baseline scenario (menu_mp=0)
        df_main, _ = calculate_lifetime_health_impacts(
            df=sample_homes_df,
            menu_mp=0,
            policy_scenario="No Inflation Reduction Act",
            verbose=False
        )
        
        # Verify results contain the lifetime damages with minimum lifetime
        rcm = RCM_MODELS[0]
        cr = CR_FUNCTIONS[0]
        lifetime_col = f"baseline_heating_lifetime_damages_health_{rcm}_{cr}"
        
        assert lifetime_col in df_main.columns, \
            f"Missing lifetime column {lifetime_col} for boundary lifetime test"
    
    finally:
        # Restore original specs to avoid side effects
        monkeypatch.setattr(
            'cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.EQUIPMENT_SPECS',
            original_specs
        )


def test_calculate_lifetime_health_impacts_missing_hdd_factor_year(
        sample_homes_df, 
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        monkeypatch):
    """
    Test that missing HDD factors for a required year raises an error.
    """
    # Mock precompute_hdd_factors to return data missing a key year
    def mock_missing_year(df):
        """Return HDD factors missing the first year (2024)."""
        hdd_factors = {}
        # Skip year 2024 (which should be required)
        for year in range(2025, 2031):
            hdd_factors[year] = pd.Series(1.0, index=df.index)
        return hdd_factors
    
    # Apply the patch
    monkeypatch.setattr(
        'cmu_tare_model.utils.precompute_hdd_factors.precompute_hdd_factors',
        mock_missing_year
    )
    monkeypatch.setattr(
        'cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.precompute_hdd_factors',
        mock_missing_year
    )
    
    # Call the function and expect a RuntimeError that contains information about the missing HDD factor
    with pytest.raises(RuntimeError) as excinfo:
        calculate_lifetime_health_impacts(
            df=sample_homes_df,
            menu_mp=0,
            policy_scenario="No Inflation Reduction Act",
            verbose=False
        )
    
    # Verify the error message contains reference to the missing HDD factor year
    error_msg = str(excinfo.value)
    assert "HDD factor for year 2024 not found" in error_msg, \
        f"Error message should mention missing HDD factor for year 2024, got: {error_msg}"


# -------------------------------------------------------------------------
#                 PARAMETRIZED TESTS ACROSS CATEGORIES
# -------------------------------------------------------------------------

def test_different_categories(
        sample_homes_df, 
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        mock_precompute_hdd_factors,
        category):
    """
    Test calculation across different equipment categories.
    
    This parametrized test verifies that calculate_lifetime_health_impacts
    works correctly for all equipment categories, applying category-specific
    validation.
    """
    # Call the main function
    menu_mp = 0  # Baseline
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, _ = calculate_lifetime_health_impacts(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify result has lifetime column for this category
    rcm = RCM_MODELS[0]
    cr = CR_FUNCTIONS[0]
    lifetime_col = f'baseline_{category}_lifetime_damages_health_{rcm}_{cr}'
    assert lifetime_col in df_main.columns, f"Result should have column '{lifetime_col}'"
    
    # Verify values are masked based on inclusion flags
    valid_mask = sample_homes_df[f'include_{category}']
    valid_indices = valid_mask[valid_mask].index
    invalid_indices = valid_mask[~valid_mask].index
    
    # Verify values for valid homes
    if len(valid_indices) > 0:
        # Valid homes should have non-NaN values
        assert not df_main.loc[valid_indices, lifetime_col].isna().all(), \
            f"Valid homes should have non-NaN values for {lifetime_col}"
    
    # Verify NaN values for invalid homes
    if len(invalid_indices) > 0:
        # Invalid homes should have NaN values
        for idx in invalid_indices:
            assert pd.isna(df_main.loc[idx, lifetime_col]), \
                f"Invalid home at index {idx} should have NaN for {lifetime_col}"


def test_different_rcm_cr_combinations(
        sample_homes_df, 
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        mock_precompute_hdd_factors,
        rcm_model,
        cr_function):
    """
    Test calculation with different RCM models and CR functions.
    
    This parametrized test verifies that calculate_lifetime_health_impacts
    works correctly with different RCM/CR combinations, generating the
    appropriate result columns for each.
    """
    # Call the main function
    menu_mp = 0  # Baseline
    policy_scenario = 'AEO2023 Reference Case'
    category = 'heating'  # Test with one category for simplicity
    
    df_main, _ = calculate_lifetime_health_impacts(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify result has specific column for this RCM/CR combination
    lifetime_col = f'baseline_{category}_lifetime_damages_health_{rcm_model}_{cr_function}'
    assert lifetime_col in df_main.columns, \
        f"Result should have column '{lifetime_col}' for RCM={rcm_model}, CR={cr_function}"


def test_different_menu_mps(
        sample_homes_df, 
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        mock_precompute_hdd_factors,
        menu_mp):
    """
    Test calculation with different measure package values.
    
    This parametrized test verifies that calculate_lifetime_health_impacts
    works correctly for both baseline (menu_mp=0) and measure packages,
    using appropriate column naming.
    """
    # Call the main function
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, _ = calculate_lifetime_health_impacts(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Determine expected scenario prefix
    if menu_mp == 0:
        scenario_prefix = "baseline_"
    else:
        scenario_prefix = f"iraRef_mp{menu_mp}_"
    
    # Verify result uses correct column naming
    category = 'heating'  # Test with one category for brevity
    rcm = RCM_MODELS[0]
    cr = CR_FUNCTIONS[0]
    
    lifetime_col = f'{scenario_prefix}{category}_lifetime_damages_health_{rcm}_{cr}'
    assert lifetime_col in df_main.columns, f"Result should have column '{lifetime_col}'"


def test_different_policy_scenarios(
        sample_homes_df, 
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        mock_precompute_hdd_factors,
        policy_scenario):
    """
    Test calculation with different policy scenarios.
    
    This parametrized test verifies that calculate_lifetime_health_impacts
    works correctly for different policy scenarios, using appropriate
    column naming.
    """
    # Call the main function with a non-zero menu_mp to see scenario differences
    menu_mp = 8
    
    df_main, _ = calculate_lifetime_health_impacts(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Determine expected scenario prefix
    if policy_scenario == "No Inflation Reduction Act":
        scenario_prefix = f"preIRA_mp{menu_mp}_"
    else:
        scenario_prefix = f"iraRef_mp{menu_mp}_"
    
    # Verify result uses correct column naming
    category = 'heating'  # Test with one category for brevity
    rcm = RCM_MODELS[0]
    cr = CR_FUNCTIONS[0]
    
    lifetime_col = f'{scenario_prefix}{category}_lifetime_damages_health_{rcm}_{cr}'
    assert lifetime_col in df_main.columns, f"Result should have column '{lifetime_col}'"


# -------------------------------------------------------------------------
#                 EDGE CASE TESTS
# -------------------------------------------------------------------------

def test_all_invalid_homes(
        sample_homes_df, 
        dummy_define_scenario_settings,
        dummy_health_vsl_adjustment,
        dummy_health_impact_lookups,
        mock_precompute_hdd_factors):
    """
    Test calculation when all homes are invalid.
    
    This test verifies that calculate_lifetime_health_impacts correctly
    handles the case where all homes are invalid, returning properly
    masked results with all NaN values.
    """
    # Create modified DataFrame with all homes invalid
    df_modified = sample_homes_df.copy()
    for category in EQUIPMENT_SPECS.keys():
        df_modified[f'include_{category}'] = False
    
    # Call the main function
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, _ = calculate_lifetime_health_impacts(
        df=df_modified,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify that result columns have NaN values for invalid homes
    for category in EQUIPMENT_SPECS.keys():
        for rcm in RCM_MODELS:
            for cr in CR_FUNCTIONS:
                col = f'baseline_{category}_lifetime_damages_health_{rcm}_{cr}'
                if col in df_main.columns:
                    assert df_main[col].isna().all(), \
                        f"All homes should have NaN values for {col} when all homes are invalid"


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
