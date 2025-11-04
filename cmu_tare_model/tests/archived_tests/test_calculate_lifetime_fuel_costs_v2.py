"""
test_lifetime_fuel_costs.py

Pytest test suite for validating the implementation of the 5-step validation framework
in the calculate_lifetime_fuel_costs module.

This test suite follows the structure outlined in the project test template
and validates each step of the framework:
1. Mask Initialization with initialize_validation_tracking()
2. Series Initialization with create_retrofit_only_series()
3. Valid-Only Calculation for qualifying homes
4. Valid-Only Updates using list-based collection
5. Final Masking with apply_final_masking()
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Import the specific modules being tested
from cmu_tare_model.private_impact.calculate_lifetime_fuel_costs import (
    calculate_lifetime_fuel_costs,
    calculate_annual_fuel_costs
)

# Import constants and utilities
from cmu_tare_model.constants import EQUIPMENT_SPECS, FUEL_MAPPING, UPGRADE_COLUMNS
from cmu_tare_model.utils.validation_framework_NEEDS_FIXED import (
    initialize_validation_tracking,
    create_retrofit_only_series,
    apply_final_masking,
    calculate_avoided_values
)


# -------------------------------------------------------------------------
#                           UTILITY FUNCTIONS
# -------------------------------------------------------------------------

def debug_print_columns(df, prefix=None):
    """
    Print columns in DataFrame to help debug missing columns.
    
    Args:
        df: DataFrame to inspect
        prefix: Optional prefix to filter columns
    """
    columns = [col for col in df.columns if prefix is None or col.startswith(prefix)]
    print(f"\nColumns{f' starting with {prefix}' if prefix else ''}:")
    for col in sorted(columns):
        print(f"  {col}")

def debug_compare_masks(expected, actual, label="masks"):
    """
    Compare expected and actual masks and print differences.
    
    Args:
        expected: Expected mask Series
        actual: Actual mask Series
        label: Label for the comparison
        
    Returns:
        bool: True if masks match, False otherwise
    """
    if expected.equals(actual):
        print(f"✓ {label} match exactly")
        return True
    
    diff_indices = expected.index[expected != actual]
    print(f"✗ {label} differ at {len(diff_indices)} positions:")
    for idx in diff_indices[:5]:
        print(f"  Index {idx}: expected={expected[idx]}, actual={actual[idx]}")
    return False


# -------------------------------------------------------------------------
#                           TEST FIXTURES
# -------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_constants(monkeypatch):
    """
    Mock the constants module to isolate tests from external dependencies.
    
    This fixture runs automatically for all tests and ensures consistent test data
    by mocking out the constants that affect validation behavior.
    
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
    
    # Mock upgrade columns for tracking retrofit status
    mock_upgrade_columns = {
        'heating': 'upgrade_hvac_heating_efficiency',
        'waterHeating': 'upgrade_water_heater_efficiency',
        'clothesDrying': 'upgrade_clothes_dryer',
        'cooking': 'upgrade_cooking_range'
    }
    
    # Apply all mocks
    monkeypatch.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.EQUIPMENT_SPECS', 
                        mock_equipment_specs)
    monkeypatch.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.FUEL_MAPPING', 
                        mock_fuel_mapping)
    monkeypatch.setattr('cmu_tare_model.utils.validation_framework.UPGRADE_COLUMNS', 
                        mock_upgrade_columns)
    
    # Also patch the constants in the imported modules to ensure consistency
    monkeypatch.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', mock_equipment_specs)
    monkeypatch.setattr('cmu_tare_model.constants.FUEL_MAPPING', mock_fuel_mapping)
    monkeypatch.setattr('cmu_tare_model.constants.UPGRADE_COLUMNS', mock_upgrade_columns)


@pytest.fixture
def sample_homes_df():
    """
    Generate sample DataFrame with consistent data for testing.
    
    This fixture creates a DataFrame with 5 homes having diverse characteristics
    to test various validation and calculation scenarios.
    
    Returns:
        DataFrame with sample data for testing
    """
    data = {
        # Metadata
        'state': ['CA', 'TX', 'NY', 'FL', 'IL'],
        'census_division': ['Pacific', 'West South Central', 'Middle Atlantic', 'South Atlantic', 'East North Central'],
        
        # Validation flags
        'include_heating': [True, True, True, True, False],
        'include_waterHeating': [True, True, True, False, True],
        'include_clothesDrying': [True, True, False, True, False],
        'include_cooking': [False, True, True, False, True],
        
        # Base equipment info
        'base_heating_fuel': ['Electricity', 'Natural Gas', 'Propane', 'Fuel Oil', 'Invalid'],
        'base_waterHeating_fuel': ['Electricity', 'Natural Gas', 'Electricity', 'Fuel Oil', 'Propane'],
        'heating_type': ['Electricity Baseboard', 'Natural Gas Fuel Furnace', 'Propane Fuel Furnace', 'Fuel Oil Fuel Furnace', 'Invalid'],
        'waterHeating_type': ['Electric Standard', 'Natural Gas Standard', 'Electric Heat Pump', 'Fuel Oil Standard', 'Propane Standard'],
        
        # Retrofit flags
        'upgrade_hvac_heating_efficiency': ['ASHP', None, 'ASHP', None, 'ASHP'],
        'upgrade_water_heater_efficiency': ['HP', None, None, 'HP', None],
        'upgrade_clothes_dryer': [None, 'Electric', None, None, 'Electric'],
        'upgrade_cooking_range': ['Induction', None, 'Resistance', None, None],
    }
    
    # Generate consumption data for multiple years and categories
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        # Only include years needed for tests (not all 15+ years)
        for year in range(2024, 2027):  # Just 3 years for test efficiency
            # Baseline consumption (with small increases for each home and year)
            data[f'baseline_{year}_{category}_consumption'] = [
                1000 + (home_idx * 100) + ((year - 2024) * 50) 
                for home_idx in range(5)
            ]
            
            # Measure package consumption (30% reduction from baseline)
            data[f'mp8_{year}_{category}_consumption'] = [
                int(data[f'baseline_{year}_{category}_consumption'][home_idx] * 0.7)
                for home_idx in range(5)
            ]
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_fuel_prices():
    """
    Create mock fuel price data for testing.
    
    This fixture creates a nested dictionary that mimics the structure
    of fuel price lookups in the production system, but with simplified
    values for testing.
    
    Returns:
        Nested dictionary with mock fuel price data
    """
    # Create the basic structure: location -> fuel_type -> policy_scenario -> year -> price
    mock_prices = {}
    
    # Add state-level data (for electricity and naturalGas)
    for state, multiplier in {'CA': 1.5, 'TX': 0.8, 'NY': 1.3, 'FL': 1.0, 'IL': 1.1}.items():
        mock_prices[state] = {
            'electricity': {
                'No Inflation Reduction Act': {
                    year: round(0.15 * multiplier * (1 + (year - 2024) * 0.02), 4) 
                    for year in range(2024, 2040)
                },
                'AEO2023 Reference Case': {
                    year: round(0.14 * multiplier * (1 + (year - 2024) * 0.015), 4) 
                    for year in range(2024, 2040)
                },
            },
            'naturalGas': {
                'No Inflation Reduction Act': {
                    year: round(0.08 * multiplier * (1 + (year - 2024) * 0.01), 4) 
                    for year in range(2024, 2040)
                },
                'AEO2023 Reference Case': {
                    year: round(0.075 * multiplier * (1 + (year - 2024) * 0.008), 4) 
                    for year in range(2024, 2040)
                },
            }
        }
    
    # Add census division level data (for propane and fuelOil)
    for division, multiplier in {
        'Pacific': 1.2, 
        'West South Central': 0.9, 
        'Middle Atlantic': 1.1, 
        'South Atlantic': 1.0, 
        'East North Central': 1.05
    }.items():
        mock_prices[division] = {
            'propane': {
                'No Inflation Reduction Act': {
                    year: round(0.12 * multiplier * (1 + (year - 2024) * 0.015), 4) 
                    for year in range(2024, 2040)
                },
                'AEO2023 Reference Case': {
                    year: round(0.115 * multiplier * (1 + (year - 2024) * 0.012), 4) 
                    for year in range(2024, 2040)
                },
            },
            'fuelOil': {
                'No Inflation Reduction Act': {
                    year: round(0.14 * multiplier * (1 + (year - 2024) * 0.018), 4) 
                    for year in range(2024, 2040)
                },
                'AEO2023 Reference Case': {
                    year: round(0.135 * multiplier * (1 + (year - 2024) * 0.015), 4) 
                    for year in range(2024, 2040)
                },
            }
        }
    
    return mock_prices


@pytest.fixture
def mock_scenario_params(mock_fuel_prices, monkeypatch):
    """
    Mock the define_scenario_params function to control test scenarios.
    
    This fixture patches the define_scenario_params function to return
    controlled test values without requiring the actual module.
    
    Args:
        mock_fuel_prices: Mock fuel price data from the fixture
        monkeypatch: Pytest fixture for patching attributes/functions
        
    Returns:
        None (patches function for test duration)
    """
    def mock_function(menu_mp, policy_scenario):
        """Mock implementation of define_scenario_params."""
        if menu_mp == 0:
            scenario_prefix = "baseline_"
        elif policy_scenario == 'No Inflation Reduction Act':
            scenario_prefix = f"preIRA_mp{menu_mp}_"
        else:
            scenario_prefix = f"iraRef_mp{menu_mp}_"
            
        # Return mock values that match expected structure
        return (
            scenario_prefix,             # scenario_prefix
            "MidCase",                   # cambium_scenario
            {},                          # lookup_emissions_fossil_fuel
            {},                          # lookup_emissions_electricity_climate
            {},                          # lookup_emissions_electricity_health
            mock_fuel_prices             # lookup_fuel_prices
        )
    
    # Apply the patch
    monkeypatch.setattr(
        'cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.define_scenario_params',
        mock_function
    )


@pytest.fixture
def df_baseline_costs(sample_homes_df):
    """
    Create sample baseline cost DataFrame for testing.
    
    This fixture generates a DataFrame with predefined fuel costs for the
    baseline scenario, used for measure package avoided cost calculations.
    
    Args:
        sample_homes_df: Sample homes DataFrame for index reference
        
    Returns:
        DataFrame with baseline fuel costs
    """
    data = {}
    
    # Generate baseline fuel costs (annual and lifetime)
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        # Generate annual fuel costs for years 2024-2026
        for year in range(2024, 2027):
            col_name = f'baseline_{year}_{category}_fuelCost'
            # Base cost depends on home index and category
            data[col_name] = [
                (1000 + (home_idx * 100)) * (0.2 if category == 'heating' else 
                                             0.15 if category == 'waterHeating' else
                                             0.1 if category == 'clothesDrying' else 0.08)
                for home_idx in range(5)
            ]
        
        # Generate lifetime fuel costs (approximately annual * lifetime factor)
        lifetime_factor = {
            'heating': 12,
            'waterHeating': 10,
            'clothesDrying': 11,
            'cooking': 12
        }[category]
        
        col_name = f'baseline_{category}_lifetime_fuelCost'
        data[col_name] = [
            data[f'baseline_2024_{category}_fuelCost'][home_idx] * lifetime_factor
            for home_idx in range(5)
        ]
    
    return pd.DataFrame(data, index=sample_homes_df.index)


@pytest.fixture(params=['heating', 'waterHeating', 'clothesDrying', 'cooking'])
def category(request):
    """
    Parametrized fixture for equipment categories.
    
    Args:
        request: Pytest request object
        
    Returns:
        String with equipment category name
    """
    return request.param


@pytest.fixture(params=[0, 8])
def menu_mp(request):
    """
    Parametrized fixture for measure package values.
    
    Args:
        request: Pytest request object
        
    Returns:
        Integer with measure package identifier
    """
    return request.param


@pytest.fixture(params=['No Inflation Reduction Act', 'AEO2023 Reference Case'])
def policy_scenario(request):
    """
    Parametrized fixture for policy scenarios.
    
    Args:
        request: Pytest request object
        
    Returns:
        String with policy scenario name
    """
    return request.param


# -------------------------------------------------------------------------
#              STEP 1: MASK INITIALIZATION TESTS
# -------------------------------------------------------------------------

def test_mask_initialization(sample_homes_df, category):
    """
    Test proper initialization of validation tracking.
    
    This test validates that the mask initialization step correctly:
    1. Identifies valid homes based on include_X flags
    2. Sets up tracking dictionaries for columns to mask
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        category: Equipment category being tested
    """
    # Use a non-zero menu_mp to test both validation and retrofit logic
    menu_mp = 8
    
    # Call the initialize_validation_tracking function directly
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        sample_homes_df, category, menu_mp, verbose=False)
    
    # Verify mask matches the include_X column
    include_col = f'include_{category}'
    assert valid_mask.equals(sample_homes_df[include_col]), \
        f"Valid mask should match the {include_col} column"
    
    # Verify tracking dictionaries are properly initialized
    assert category in all_columns_to_mask, \
        f"Category '{category}' should be in all_columns_to_mask dictionary"
    
    assert isinstance(all_columns_to_mask[category], list), \
        f"all_columns_to_mask[{category}] should be a list"
    
    assert len(category_columns_to_mask) == 0, \
        "category_columns_to_mask should be an empty list initially"


# -------------------------------------------------------------------------
#              STEP 2: SERIES INITIALIZATION TESTS
# -------------------------------------------------------------------------

def test_series_initialization(sample_homes_df, category):
    """
    Test proper initialization of result series.
    
    This test validates that the series initialization step correctly:
    1. Creates a Series with zeros for valid homes
    2. Sets NaN for invalid homes
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        category: Equipment category being tested
    """
    # Get the valid mask from include_X column
    valid_mask = sample_homes_df[f'include_{category}']
    
    # Call the create_retrofit_only_series function directly
    result = create_retrofit_only_series(sample_homes_df, valid_mask)
    
    # Verify structure and type
    assert isinstance(result, pd.Series), "Result should be a pandas Series"
    assert len(result) == len(sample_homes_df), "Result should have same length as DataFrame"
    
    # Check valid homes have zeros
    valid_indices = valid_mask[valid_mask].index
    assert (result.loc[valid_indices] == 0.0).all(), \
        "All valid homes should have value 0.0"
    
    # Check invalid homes have NaN
    invalid_indices = valid_mask[~valid_mask].index
    assert result.loc[invalid_indices].isna().all(), \
        "All invalid homes should have value NaN"


# -------------------------------------------------------------------------
#              STEP 3: VALID-ONLY CALCULATION TESTS
# -------------------------------------------------------------------------

def test_calculate_annual_fuel_costs_basic(sample_homes_df, mock_fuel_prices, mock_scenario_params):
    """
    Test basic calculation of annual fuel costs with validation masking.
    
    This test verifies that the calculation step:
    1. Only performs calculations for valid homes
    2. Correctly applies fuel prices based on location and fuel type
    3. Respects the valid_mask parameter
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_fuel_prices: Mock fuel price data
        mock_scenario_params: Mock for define_scenario_params function
    """
    # Test parameters
    category = 'heating'
    year_label = 2024
    menu_mp = 0  # Baseline
    policy_scenario = 'No Inflation Reduction Act'
    scenario_prefix = 'baseline_'
    
    # Setup fuel type mapping for baseline calculations
    df = sample_homes_df.copy()
    fuel_col = f'base_{category}_fuel'
    df[f'fuel_type_{category}'] = df[fuel_col].map(FUEL_MAPPING)
    is_elec_or_gas = df[f'fuel_type_{category}'].isin(['electricity', 'naturalGas'])
    
    # Get the valid mask from include_X column
    valid_mask = df[f'include_{category}']
    
    # Call the function with valid_mask
    annual_costs, annual_cost_value = calculate_annual_fuel_costs(
        df=df,
        category=category,
        year_label=year_label,
        menu_mp=menu_mp,
        lookup_fuel_prices=mock_fuel_prices,
        policy_scenario=policy_scenario,
        scenario_prefix=scenario_prefix,
        is_elec_or_gas=is_elec_or_gas,
        valid_mask=valid_mask
    )
    
    # Verify the result contains the expected column
    cost_col = f'{scenario_prefix}{year_label}_{category}_fuelCost'
    assert cost_col in annual_costs, f"Result should contain column '{cost_col}'"
    
    # Verify invalid homes have zero values in the annual_cost_value
    invalid_indices = valid_mask[~valid_mask].index
    assert (annual_cost_value.loc[invalid_indices] == 0.0).all(), \
        "Invalid homes should have fuel cost of 0.0"
    
    # Verify calculation is correct for California with electricity (first home)
    if valid_mask.iloc[0]:  # If first home is valid
        ca_consumption = df.loc[0, f'baseline_{year_label}_{category}_consumption']
        ca_price = mock_fuel_prices['CA']['electricity'][policy_scenario][year_label]
        expected_ca_cost = ca_consumption * ca_price
        
        assert abs(annual_cost_value.iloc[0] - expected_ca_cost) < 0.01, \
            f"Cost calculation for first home should be approximately {expected_ca_cost}"


def test_calculate_annual_fuel_costs_measure_package(sample_homes_df, mock_fuel_prices, mock_scenario_params):
    """
    Test calculation of annual fuel costs for measure packages.
    
    This test verifies that the calculation:
    1. Uses electricity prices for all measure package calculations
    2. Correctly uses state-based prices
    3. Applies the valid_mask to filter invalid homes
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_fuel_prices: Mock fuel price data
        mock_scenario_params: Mock for define_scenario_params function
    """
    # Test parameters
    category = 'heating'
    year_label = 2024
    menu_mp = 8  # Measure package
    policy_scenario = 'AEO2023 Reference Case'
    scenario_prefix = f'iraRef_mp{menu_mp}_'
    
    # Get the valid mask from include_X column
    df = sample_homes_df.copy()
    valid_mask = df[f'include_{category}']
    
    # Call the function with valid_mask
    annual_costs, annual_cost_value = calculate_annual_fuel_costs(
        df=df,
        category=category,
        year_label=year_label,
        menu_mp=menu_mp,
        lookup_fuel_prices=mock_fuel_prices,
        policy_scenario=policy_scenario,
        scenario_prefix=scenario_prefix,
        is_elec_or_gas=None,  # Not needed for measure packages
        valid_mask=valid_mask
    )
    
    # Verify the result contains the expected column
    cost_col = f'{scenario_prefix}{year_label}_{category}_fuelCost'
    assert cost_col in annual_costs, f"Result should contain column '{cost_col}'"
    
    # Verify invalid homes have zero values in the annual_cost_value
    invalid_indices = valid_mask[~valid_mask].index
    assert (annual_cost_value.loc[invalid_indices] == 0.0).all(), \
        "Invalid homes should have fuel cost of 0.0"
    
    # Verify calculation is correct for a measure package (first home)
    if valid_mask.iloc[0]:  # If first home is valid
        consumption = df.loc[0, f'mp{menu_mp}_{year_label}_{category}_consumption']
        price = mock_fuel_prices['CA']['electricity'][policy_scenario][year_label]
        expected_cost = consumption * price
        
        assert abs(annual_cost_value.iloc[0] - expected_cost) < 0.01, \
            f"Cost calculation for first home should be approximately {expected_cost}"


# -------------------------------------------------------------------------
#              STEP 4: VALID-ONLY UPDATES TESTS
# -------------------------------------------------------------------------

def test_list_based_collection(sample_homes_df, mock_fuel_prices, mock_scenario_params, monkeypatch):
    """
    Test list-based collection for yearly values.
    
    This test verifies that the module:
    1. Uses a list to collect yearly values instead of incremental updates
    2. Only includes values for valid homes
    3. Properly aggregates yearly values into lifetime costs
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_fuel_prices: Mock fuel price data
        mock_scenario_params: Mock for define_scenario_params function
        monkeypatch: Pytest fixture for patching
    """
    # This test requires inspecting the inner workings of calculate_lifetime_fuel_costs
    # We'll use a spy to capture the yearly_costs_list
    
    original_function = calculate_annual_fuel_costs
    yearly_costs_captured = []
    
    def spy_annual_costs(*args, **kwargs):
        """Spy function to capture annual costs."""
        annual_costs, annual_cost_value = original_function(*args, **kwargs)
        yearly_costs_captured.append(annual_cost_value.copy())
        return annual_costs, annual_cost_value
    
    # Apply the patch
    monkeypatch.setattr(
        'cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.calculate_annual_fuel_costs',
        spy_annual_costs
    )
    
    # Call the main function
    category = 'heating'
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify we captured some yearly values
    assert len(yearly_costs_captured) > 0, "Should have captured yearly cost values"
    
    # Verify the values match our expected pattern
    valid_mask = sample_homes_df[f'include_{category}']
    for yearly_cost in yearly_costs_captured:
        # Valid homes should have non-zero values
        for idx in valid_mask.index:
            if valid_mask[idx]:
                assert yearly_cost[idx] > 0, f"Valid home at index {idx} should have non-zero value"
            else:
                assert yearly_cost[idx] == 0, f"Invalid home at index {idx} should have zero value"
    
    # Verify the lifetime column contains the sum of yearly values
    lifetime_col = f'iraRef_mp{menu_mp}_{category}_lifetime_fuelCost'
    assert lifetime_col in df_main.columns, f"Result should contain column '{lifetime_col}'"
    
    # Get the first few yearly values for the first valid home
    valid_idx = valid_mask[valid_mask].index[0]
    yearly_values_for_valid = [yearly[valid_idx] for yearly in yearly_costs_captured 
                               if category in yearly]
    
    # Check the lifetime value is approximately the sum of yearly values
    if yearly_values_for_valid:
        approx_sum = sum(yearly_values_for_valid)
        lifetime_value = df_main.loc[valid_idx, lifetime_col]
        
        # The lifetime value should be at least as large as our captured yearly values
        # (It might be larger since we might not have captured all years)
        assert lifetime_value >= approx_sum * 0.9, \
            f"Lifetime value {lifetime_value} should be approximately the sum of yearly values {approx_sum}"


# -------------------------------------------------------------------------
#              STEP 5: FINAL MASKING TESTS
# -------------------------------------------------------------------------

def test_final_masking(sample_homes_df, mock_fuel_prices, mock_scenario_params):
    """
    Test final masking of result columns.
    
    This test verifies that:
    1. Final masking is properly applied to all result columns
    2. Invalid homes have NaN values for all result columns
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_fuel_prices: Mock fuel price data
        mock_scenario_params: Mock for define_scenario_params function
    """
    # Call the main function
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Check final masking for each category
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        valid_mask = sample_homes_df[f'include_{category}']
        lifetime_col = f'iraRef_mp8_{category}_lifetime_fuelCost'
        
        # Skip categories that might not have results
        if lifetime_col not in df_main.columns:
            continue
            
        # Check that invalid homes have NaN values
        invalid_indices = valid_mask[~valid_mask].index
        assert df_main.loc[invalid_indices, lifetime_col].isna().all(), \
            f"Invalid homes should have NaN for {lifetime_col}"
        
        # Check that valid homes have non-NaN values
        valid_indices = valid_mask[valid_mask].index
        assert not df_main.loc[valid_indices, lifetime_col].isna().all(), \
            f"At least some valid homes should have non-NaN values for {lifetime_col}"


# -------------------------------------------------------------------------
#              INTEGRATION TESTS
# -------------------------------------------------------------------------

def test_lifetime_fuel_costs_basic(sample_homes_df, mock_fuel_prices, mock_scenario_params):
    """
    Test basic functionality of calculate_lifetime_fuel_costs.
    
    This test verifies that:
    1. The function runs without errors
    2. The result DataFrames have the expected structure
    3. Values are properly masked based on validation flags
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_fuel_prices: Mock fuel price data
        mock_scenario_params: Mock for define_scenario_params function
    """
    # Call the main function with baseline scenario
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify the shape of the result DataFrames
    assert len(df_main) == len(sample_homes_df), "Main DataFrame should have same number of rows"
    assert len(df_detailed) == len(sample_homes_df), "Detailed DataFrame should have same number of rows"
    
    # Verify the main DataFrame has lifetime columns for all categories
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        lifetime_col = f'baseline_{category}_lifetime_fuelCost'
        assert lifetime_col in df_main.columns, f"Main DataFrame should have column '{lifetime_col}'"
    
    # Verify the detailed DataFrame has yearly columns
    year = 2024
    category = 'heating'
    yearly_col = f'baseline_{year}_{category}_fuelCost'
    assert yearly_col in df_detailed.columns, f"Detailed DataFrame should have column '{yearly_col}'"
    
    # Check proper masking for one category
    category = 'heating'
    valid_mask = sample_homes_df[f'include_{category}']
    lifetime_col = f'baseline_{category}_lifetime_fuelCost'
    
    # Invalid homes should have NaN
    invalid_indices = valid_mask[~valid_mask].index
    assert df_main.loc[invalid_indices, lifetime_col].isna().all(), \
        f"Invalid homes should have NaN for {lifetime_col}"


def test_lifetime_fuel_costs_with_baseline(sample_homes_df, df_baseline_costs, mock_fuel_prices, mock_scenario_params):
    """
    Test calculation with measure package and baseline costs.
    
    This test verifies that:
    1. The function handles df_baseline_costs correctly
    2. Avoided costs are calculated as (baseline - measure)
    3. All values are properly masked
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        df_baseline_costs: Sample baseline costs DataFrame
        mock_fuel_prices: Mock fuel price data
        mock_scenario_params: Mock for define_scenario_params function
    """
    # Call the main function with measure package scenario
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        df_baseline_costs=df_baseline_costs,
        verbose=False
    )
    
    # Verify the main DataFrame has avoided cost columns
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        savings_col = f'iraRef_mp8_{category}_lifetime_savings_fuelCost'
        
        # The column might not exist for all categories depending on data
        if savings_col in df_main.columns:
            valid_mask = sample_homes_df[f'include_{category}']
            valid_indices = valid_mask[valid_mask].index
            
            # At least some valid homes should have non-NaN values
            if not valid_indices.empty:
                assert not df_main.loc[valid_indices, savings_col].isna().all(), \
                    f"At least some valid homes should have values for {savings_col}"
                
                # Spot check calculation for one valid home
                idx = valid_indices[0]
                baseline_col = f'baseline_{category}_lifetime_fuelCost'
                measure_col = f'iraRef_mp8_{category}_lifetime_fuelCost'
                
                if baseline_col in df_baseline_costs.columns and measure_col in df_main.columns:
                    baseline_value = df_baseline_costs.loc[idx, baseline_col]
                    measure_value = df_main.loc[idx, measure_col]
                    savings_value = df_main.loc[idx, savings_col]
                    
                    assert abs((baseline_value - measure_value) - savings_value) < 0.01, \
                        f"Savings should be baseline - measure for home at index {idx}"


# -------------------------------------------------------------------------
#              PARAMETRIZED TESTS
# -------------------------------------------------------------------------

def test_across_categories(sample_homes_df, mock_fuel_prices, mock_scenario_params, category):
    """
    Test calculation across different equipment categories.
    
    This parametrized test verifies that:
    1. The function handles all equipment categories correctly
    2. Lifetimes are properly applied based on category
    3. Validation is correctly applied per category
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_fuel_prices: Mock fuel price data
        mock_scenario_params: Mock for define_scenario_params function
        category: Parametrized equipment category
    """
    # Call the main function with baseline scenario
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify the result has a lifetime column for this category
    lifetime_col = f'baseline_{category}_lifetime_fuelCost'
    assert lifetime_col in df_main.columns, f"Result should have column '{lifetime_col}'"
    
    # Verify proper masking for this category
    valid_mask = sample_homes_df[f'include_{category}']
    
    # Invalid homes should have NaN
    invalid_indices = valid_mask[~valid_mask].index
    if not invalid_indices.empty:
        assert df_main.loc[invalid_indices, lifetime_col].isna().all(), \
            f"Invalid homes should have NaN for {lifetime_col}"
    
    # Valid homes should have non-NaN values
    valid_indices = valid_mask[valid_mask].index
    if not valid_indices.empty:
        assert not df_main.loc[valid_indices, lifetime_col].isna().all(), \
            f"At least some valid homes should have values for {lifetime_col}"


def test_across_policy_scenarios(sample_homes_df, mock_fuel_prices, mock_scenario_params, policy_scenario):
    """
    Test calculation across different policy scenarios.
    
    This parametrized test verifies that:
    1. The function handles all policy scenarios correctly
    2. Column prefixes match the policy scenario
    3. Fuel prices are applied according to the policy scenario
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_fuel_prices: Mock fuel price data
        mock_scenario_params: Mock for define_scenario_params function
        policy_scenario: Parametrized policy scenario
    """
    # Call the main function with measure package
    menu_mp = 8
    category = 'heating'  # Just test one category for efficiency
    
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Determine the expected column prefix
    if policy_scenario == 'No Inflation Reduction Act':
        expected_prefix = f"preIRA_mp{menu_mp}_"
    else:
        expected_prefix = f"iraRef_mp{menu_mp}_"
    
    # Verify the result has properly prefixed columns
    lifetime_col = f'{expected_prefix}{category}_lifetime_fuelCost'
    assert lifetime_col in df_main.columns, f"Result should have column '{lifetime_col}'"
    
    # Verify column values reflect the policy scenario
    # This is indirectly testing that the fuel prices were applied properly
    valid_mask = sample_homes_df[f'include_{category}']
    valid_indices = valid_mask[valid_mask].index
    
    if not valid_indices.empty:
        # Get a valid home to check
        idx = valid_indices[0]
        fuel_cost = df_main.loc[idx, lifetime_col]
        
        # Can't easily predict exact value, but should be > 0
        assert fuel_cost > 0, f"Fuel cost should be positive for {policy_scenario}"


# -------------------------------------------------------------------------
#              EDGE CASE TESTS
# -------------------------------------------------------------------------

def test_all_invalid_homes(sample_homes_df, mock_fuel_prices, mock_scenario_params):
    """
    Test calculation when all homes are invalid.
    
    This test verifies that:
    1. The function handles the case where all homes are invalid
    2. The result has NaN values for all homes
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_fuel_prices: Mock fuel price data
        mock_scenario_params: Mock for define_scenario_params function
    """
    # Create a modified DataFrame where all homes are invalid for a category
    df_modified = sample_homes_df.copy()
    category = 'heating'
    df_modified[f'include_{category}'] = False
    
    # Call the main function
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=df_modified,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify the result has a lifetime column for this category
    lifetime_col = f'baseline_{category}_lifetime_fuelCost'
    assert lifetime_col in df_main.columns, f"Result should have column '{lifetime_col}'"
    
    # All homes should have NaN values since all are invalid
    assert df_main[lifetime_col].isna().all(), \
        "All homes should have NaN values when all are invalid"


def test_missing_columns(sample_homes_df, mock_fuel_prices, mock_scenario_params):
    """
    Test error handling for missing required columns.
    
    This test verifies that:
    1. The function properly detects and reports missing columns
    2. Error messages are informative
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_fuel_prices: Mock fuel price data
        mock_scenario_params: Mock for define_scenario_params function
    """
    # Create DataFrame missing required columns
    df_missing = sample_homes_df.copy()
    df_missing = df_missing.drop(columns=['state', 'census_division'])
    
    # Call the main function and expect a KeyError
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    with pytest.raises(KeyError) as excinfo:
        calculate_lifetime_fuel_costs(
            df=df_missing,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )
    
    # Verify error message mentions missing columns
    error_msg = str(excinfo.value)
    assert "state" in error_msg and "census_division" in error_msg, \
        "Error message should mention missing state and census_division columns"
