"""
test_calculate_lifetime_fuel_costs.py

Pytest test suite for validating the implementation of the 5-step validation framework
in the calculate_lifetime_fuel_costs module.

This test suite verifies each step of the framework:
1. Mask Initialization with initialize_validation_tracking()
2. Series Initialization with create_retrofit_only_series()
3. Valid-Only Calculation for qualifying homes
4. Valid-Only Updates using list-based collection
5. Final Masking with apply_final_masking()
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

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
    get_valid_calculation_mask
)


# -------------------------------------------------------------------------
#                           UTILITY FUNCTIONS
# -------------------------------------------------------------------------

def debug_print_columns(df: pd.DataFrame, prefix: Optional[str] = None) -> None:
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


def debug_compare_masks(expected: pd.Series, actual: pd.Series, label: str = "masks") -> bool:
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
    
    # Find indices where masks differ
    diff_indices = expected.index[expected != actual]
    print(f"✗ {label} differ at {len(diff_indices)} positions:")
    for idx in diff_indices[:5]:  # Show at most 5 differences
        print(f"  Index {idx}: expected={expected[idx]}, actual={actual[idx]}")
    return False


# -------------------------------------------------------------------------
#                           TEST FIXTURES
# -------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_constants(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', mock_equipment_specs)
    monkeypatch.setattr('cmu_tare_model.constants.FUEL_MAPPING', mock_fuel_mapping)
    monkeypatch.setattr('cmu_tare_model.constants.UPGRADE_COLUMNS', mock_upgrade_columns)
    monkeypatch.setattr('cmu_tare_model.utils.validation_framework.EQUIPMENT_SPECS', mock_equipment_specs)
    monkeypatch.setattr('cmu_tare_model.utils.validation_framework.UPGRADE_COLUMNS', mock_upgrade_columns)


@pytest.fixture
def sample_homes_df() -> pd.DataFrame:
    """
    Generate sample DataFrame with comprehensive data for testing.
    
    This fixture creates a DataFrame with 5 homes having diverse characteristics
    to test various validation and calculation scenarios. It includes all required
    columns for each equipment category.
    
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
        
        # Base equipment fuel types for all categories
        'base_heating_fuel': ['Electricity', 'Natural Gas', 'Propane', 'Fuel Oil', 'Invalid'],
        'base_waterHeating_fuel': ['Electricity', 'Natural Gas', 'Electricity', 'Fuel Oil', 'Propane'],
        'base_clothesDrying_fuel': ['Electricity', 'Natural Gas', 'Propane', 'Electricity', 'Natural Gas'],
        'base_cooking_fuel': ['Natural Gas', 'Propane', 'Natural Gas', 'Propane', 'Natural Gas'],
        
        # Equipment types for categories that need them
        'heating_type': ['Electricity Baseboard', 'Natural Gas Fuel Furnace', 'Propane Fuel Furnace', 'Fuel Oil Fuel Furnace', 'Invalid'],
        'waterHeating_type': ['Electric Standard', 'Natural Gas Standard', 'Electric Heat Pump', 'Fuel Oil Standard', 'Propane Standard'],
        
        # Retrofit flags
        'upgrade_hvac_heating_efficiency': ['ASHP', None, 'ASHP', None, 'ASHP'],
        'upgrade_water_heater_efficiency': ['HP', None, None, 'HP', None],
        'upgrade_clothes_dryer': [None, 'Electric', None, None, 'Electric'],
        'upgrade_cooking_range': ['Induction', None, 'Resistance', None, None],
    }
    
    # Generate baseline consumption columns for each fuel type and category
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        for fuel in ['electricity', 'naturalGas', 'fuelOil', 'propane']:
            # Create columns like base_electricity_heating_consumption
            col_name = f'base_{fuel}_{category}_consumption'
            # Set values based on home index (0-indexed from 1-5)
            data[col_name] = [100 * (i + 1) if i % 2 == 0 else 0 for i in range(5)]
    
    # Generate baseline and measure package consumption data for multiple years and categories
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
def mock_fuel_prices() -> Dict[str, Dict[str, Dict[str, Dict[int, float]]]]:
    """
    Create mock fuel price data for testing.
    
    This fixture creates a nested dictionary that mimics the structure
    of fuel price lookups in the production system, but with simplified
    values for testing.
    
    Returns:
        Nested dictionary with mock fuel price data. The structure is:
        location -> fuel_type -> policy_scenario -> year -> price
    """
    # Create the basic structure
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
def mock_scenario_params(mock_fuel_prices: Dict[str, Dict[str, Dict[str, Dict[int, float]]]], 
                        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Mock the define_scenario_params function to control test scenarios.
    
    This fixture patches the define_scenario_params function to return
    controlled test values without requiring the actual module.
    
    Args:
        mock_fuel_prices: Mock fuel price data from the fixture
        monkeypatch: Pytest fixture for patching attributes/functions
    """
    def mock_function(menu_mp: int, policy_scenario: str) -> Tuple[str, str, Dict, Dict, Dict, Dict]:
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
def df_baseline_costs(sample_homes_df: pd.DataFrame) -> pd.DataFrame:
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
            col_name = f'baseline_{year}_{category}_fuel_cost'
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
        
        col_name = f'baseline_{category}_lifetime_fuel_cost'
        data[col_name] = [
            data[f'baseline_2024_{category}_fuel_cost'][home_idx] * lifetime_factor
            for home_idx in range(5)
        ]
    
    return pd.DataFrame(data, index=sample_homes_df.index)


@pytest.fixture
def mock_annual_calculation(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Mock the annual calculation to avoid needing consumption data for every year.
    
    This fixture replaces the calculate_annual_fuel_costs function with a simplified 
    version that returns predetermined values. This allows testing the framework 
    without needing detailed consumption data.
    
    Args:
        monkeypatch: Pytest fixture for patching attributes/functions
    """
    
    def mock_calculate_annual(df: pd.DataFrame, 
                             category: str, 
                             year_label: int, 
                             menu_mp: int, 
                             lookup_fuel_prices: Dict, 
                             policy_scenario: str, 
                             scenario_prefix: str, 
                             is_elec_or_gas: Optional[pd.Series] = None,
                             valid_mask: Optional[pd.Series] = None) -> Tuple[Dict[str, pd.Series], pd.Series]:
        """
        Mock implementation for calculate_annual_fuel_costs.
        
        Args:
            df: DataFrame containing home data
            category: Equipment category
            year_label: Year for calculation
            menu_mp: Measure package identifier
            lookup_fuel_prices: Mock fuel price data (unused in mock)
            policy_scenario: Policy scenario (unused in mock)
            scenario_prefix: Prefix for column names
            is_elec_or_gas: Mask indicating which homes use electricity/gas
            valid_mask: Mask indicating valid homes
            
        Returns:
            Tuple containing:
            - Dictionary of annual costs (column_name -> Series)
            - Series of annual cost values
        """
        # If no valid_mask provided, use all homes
        if valid_mask is None:
            valid_mask = pd.Series(True, index=df.index)
        
        # Create results dictionary and cost column
        annual_costs = {}
        cost_col = f"{scenario_prefix}{year_label}_{category}_fuel_cost"
        
        # Create a Series with zeros for all homes
        cost_series = pd.Series(0.0, index=df.index)
        
        # Set values for valid homes only
        valid_homes = valid_mask[valid_mask].index
        if len(valid_homes) > 0:
            # Use home index to create different values
            for i, idx in enumerate(valid_homes):
                # Base value on home index, year, and category for variability
                base_value = 100.0 + (i * 10) + ((year_label - 2024) * 5)
                multiplier = 1.0
                if category == 'heating':
                    multiplier = 2.0
                elif category == 'waterHeating':
                    multiplier = 1.5
                elif category == 'clothesDrying':
                    multiplier = 1.2
                elif category == 'cooking':
                    multiplier = 0.8
                
                cost_series.loc[idx] = base_value * multiplier
            
        # Create annual costs dictionary with the cost column
        annual_costs[cost_col] = cost_series
        
        # For measure packages, add savings column
        if menu_mp != 0:
            savings_col = f"{scenario_prefix}{year_label}_{category}_savings_fuel_cost"
            savings_series = pd.Series(0.0, index=df.index)
            
            # Set values for valid homes only (50% of costs as savings)
            for idx in valid_homes:
                savings_series.loc[idx] = cost_series.loc[idx] * 0.5
                
            annual_costs[savings_col] = savings_series
            
        return annual_costs, cost_series
        
    monkeypatch.setattr(
        'cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.calculate_annual_fuel_costs',
        mock_calculate_annual
    )


@pytest.fixture(params=['heating', 'waterHeating', 'clothesDrying', 'cooking'])
def category(request) -> str:
    """
    Parametrized fixture for equipment categories.
    
    Args:
        request: Pytest request object
        
    Returns:
        String with equipment category name
    """
    return request.param


@pytest.fixture(params=[0, 8])
def menu_mp(request) -> int:
    """
    Parametrized fixture for measure package values.
    
    Args:
        request: Pytest request object
        
    Returns:
        Integer with measure package identifier
    """
    return request.param


@pytest.fixture(params=['No Inflation Reduction Act', 'AEO2023 Reference Case'])
def policy_scenario(request) -> str:
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

def test_mask_initialization(sample_homes_df: pd.DataFrame, category: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test proper initialization of validation tracking.
    
    This test verifies that:
    1. initialize_validation_tracking creates the appropriate mask
    2. All columns are properly tracked for masking
    3. The function returns the expected structure
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        category: Equipment category being tested
        monkeypatch: Pytest fixture for patching attributes/functions
    """
    menu_mp = 8
    
    # Mock get_valid_calculation_mask to ensure predictable behavior
    def mock_get_valid_mask(df: pd.DataFrame, cat: str, mp: Union[int, str], verbose: bool = True) -> pd.Series:
        """Return a simple mask based on include column."""
        return df[f'include_{cat}']
        
    monkeypatch.setattr(
        'cmu_tare_model.utils.validation_framework.get_valid_calculation_mask',
        mock_get_valid_mask
    )
    
    # Call initialize_validation_tracking
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        sample_homes_df, category, menu_mp, verbose=False)
    
    # Verify valid_mask matches include_category
    assert valid_mask.equals(sample_homes_df[f'include_{category}']), \
        f"Valid mask should match the include_{category} column"
    
    # Verify tracking dictionaries are initialized properly
    assert isinstance(df_copy, pd.DataFrame), "df_copy should be a DataFrame"
    assert len(df_copy) == len(sample_homes_df), "df_copy should have same length as input"
    
    assert category in all_columns_to_mask, \
        f"Category '{category}' should be in all_columns_to_mask dictionary"
    
    assert isinstance(all_columns_to_mask[category], list), \
        f"all_columns_to_mask[{category}] should be a list"
    
    assert len(category_columns_to_mask) == 0, \
        "category_columns_to_mask should be an empty list initially"


def test_mask_initialization_missing_include(sample_homes_df: pd.DataFrame) -> None:
    """
    Test initialization with missing include flag column.
    
    This test verifies that:
    1. An appropriate error is raised when the include flag is missing
    2. The error message is informative
    
    Args:
        sample_homes_df: Sample DataFrame with home data
    """
    # Create a copy without the include_heating column
    df_modified = sample_homes_df.copy()
    df_modified = df_modified.drop(columns=['include_heating'])
    
    # Call initialize_validation_tracking and expect an error
    with pytest.raises(ValueError) as excinfo:
        initialize_validation_tracking(df_modified, 'heating', 8, verbose=False)
    
    # Verify error message
    error_msg = str(excinfo.value)
    assert "include_heating" in error_msg or "Inclusion flag" in error_msg, \
        "Error message should mention the missing include flag"


# -------------------------------------------------------------------------
#              STEP 2: SERIES INITIALIZATION TESTS
# -------------------------------------------------------------------------

def test_series_initialization(sample_homes_df: pd.DataFrame, category: str) -> None:
    """
    Test proper initialization of result series.
    
    This test verifies that:
    1. create_retrofit_only_series creates a Series with zeros for valid homes
    2. Invalid homes get NaN values
    3. The Series has the correct index and type
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        category: Equipment category being tested
    """
    # Get the valid mask from include_X column
    valid_mask = sample_homes_df[f'include_{category}']
    
    # Call create_retrofit_only_series
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
    if len(invalid_indices) > 0:
        assert result.loc[invalid_indices].isna().all(), \
            "All invalid homes should have value NaN"


def test_series_initialization_all_valid(sample_homes_df: pd.DataFrame) -> None:
    """
    Test series initialization when all homes are valid.
    
    This test verifies that:
    1. When all homes are valid, all get zeros
    2. No homes get NaN values
    
    Args:
        sample_homes_df: Sample DataFrame with home data
    """
    # Create mask with all homes valid
    all_valid = pd.Series(True, index=sample_homes_df.index)
    
    # Call create_retrofit_only_series
    result = create_retrofit_only_series(sample_homes_df, all_valid)
    
    # Check all homes have zeros
    assert (result == 0.0).all(), "All homes should have value 0.0"
    assert not result.isna().any(), "No homes should have NaN values"


def test_series_initialization_all_invalid(sample_homes_df: pd.DataFrame) -> None:
    """
    Test series initialization when all homes are invalid.
    
    This test verifies that:
    1. When all homes are invalid, all get NaN values
    2. No homes get zero values
    
    Args:
        sample_homes_df: Sample DataFrame with home data
    """
    # Create mask with all homes invalid
    all_invalid = pd.Series(False, index=sample_homes_df.index)
    
    # Call create_retrofit_only_series
    result = create_retrofit_only_series(sample_homes_df, all_invalid)
    
    # Check all homes have NaN
    assert result.isna().all(), "All homes should have NaN values"


# -------------------------------------------------------------------------
#              STEP 3: VALID-ONLY CALCULATION TESTS
# -------------------------------------------------------------------------

def test_calculate_annual_fuel_costs_basic(sample_homes_df: pd.DataFrame, mock_fuel_prices: Dict, mock_scenario_params: None) -> None:
    """
    Test basic calculation of annual fuel costs with validation masking.
    
    This test verifies that:
    1. Calculations are only performed for valid homes
    2. Fuel prices are applied correctly based on location and fuel type
    3. The valid_mask parameter properly filters calculations
    
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
    
    # Call calculate_annual_fuel_costs
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
    
    # Verify result structure
    cost_col = f'{scenario_prefix}{year_label}_{category}_fuel_cost'
    assert cost_col in annual_costs, f"Result should contain column '{cost_col}'"
    assert isinstance(annual_cost_value, pd.Series), "annual_cost_value should be a Series"
    assert len(annual_cost_value) == len(df), "annual_cost_value should have same length as DataFrame"
    
    # Check only valid homes have non-zero values
    invalid_indices = valid_mask[~valid_mask].index
    if len(invalid_indices) > 0:
        assert (annual_cost_value.loc[invalid_indices] == 0.0).all(), \
            "Invalid homes should have fuel cost of 0.0 in annual_cost_value"


def test_calculate_annual_fuel_costs_measure_package(sample_homes_df: pd.DataFrame, mock_fuel_prices: Dict, mock_scenario_params: None) -> None:
    """
    Test calculation of annual fuel costs for measure packages.
    
    This test verifies that:
    1. Electricity prices are used for all measure package calculations
    2. State-based prices are correctly applied
    3. The valid_mask is respected for filtering calculations
    
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
    
    # Call calculate_annual_fuel_costs
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
    
    # Verify result structure
    cost_col = f'{scenario_prefix}{year_label}_{category}_fuel_cost'
    assert cost_col in annual_costs, f"Result should contain column '{cost_col}'"
    
    # Check column in annual_costs contains only values for valid homes
    valid_indices = valid_mask[valid_mask].index
    invalid_indices = valid_mask[~valid_mask].index
    
    # Valid homes should have non-zero values
    if len(valid_indices) > 0:
        assert not annual_costs[cost_col].loc[valid_indices].isna().all(), \
            "At least some valid homes should have values"
    
    # Invalid homes should have 0 values in the annual_cost_value 
    if len(invalid_indices) > 0:
        assert (annual_cost_value.loc[invalid_indices] == 0.0).all(), \
            "Invalid homes should have fuel cost of 0.0 in annual_cost_value"


# -------------------------------------------------------------------------
#              STEP 4: VALID-ONLY UPDATES TESTS
# -------------------------------------------------------------------------

def test_list_based_collection(sample_homes_df: pd.DataFrame, mock_scenario_params: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test list-based collection for yearly values.
    
    This test verifies that:
    1. The module uses a list to collect yearly values
    2. Only valid homes get updated values
    3. Yearly values are properly aggregated into lifetime costs
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_scenario_params: Mock for define_scenario_params function
        monkeypatch: Pytest fixture for patching attributes/functions
    """
    # Create a spy function to track calls to calculate_annual_fuel_costs
    original_function = calculate_annual_fuel_costs
    yearly_costs_captured = []
    
    def spy_annual_costs(*args, **kwargs):
        """Spy function to capture annual costs."""
        annual_costs, annual_cost_value = original_function(*args, **kwargs)
        yearly_costs_captured.append(annual_cost_value.copy())
        return annual_costs, annual_cost_value
    
    # Replace the original function with our spy
    monkeypatch.setattr(
        'cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.calculate_annual_fuel_costs',
        spy_annual_costs
    )
    
    # Call the main function with a small lifetime for efficiency
    monkeypatch.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', {'heating': 3, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
    monkeypatch.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.EQUIPMENT_SPECS', 
                        {'heating': 3, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
    
    # Set up test parameters
    category = 'heating'
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call the main function with verbose=False
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify we captured some yearly values
    assert len(yearly_costs_captured) > 0, "Should have captured yearly cost values"
    
    # Verify the lifetime column contains the sum of yearly values for valid homes
    lifetime_col = f'iraRef_mp{menu_mp}_{category}_lifetime_fuel_cost'
    assert lifetime_col in df_main.columns, f"Result should contain column '{lifetime_col}'"
    
    # Get valid mask for the category
    valid_mask = sample_homes_df[f'include_{category}']
    valid_indices = valid_mask[valid_mask].index
    invalid_indices = valid_mask[~valid_mask].index
    
    # Check that invalid homes have NaN values in the final result
    if len(invalid_indices) > 0:
        assert df_main.loc[invalid_indices, lifetime_col].isna().all(), \
            "Invalid homes should have NaN values in final result"
    
    # For at least one valid home, verify sum of yearly values approximately matches lifetime value
    if len(valid_indices) > 0:
        idx = valid_indices[0]
        yearly_sum = sum(yearly_cost[idx] for yearly_cost in yearly_costs_captured[:3])  # First 3 years for heating
        assert abs(df_main.loc[idx, lifetime_col] - yearly_sum) < 0.1, \
            f"Lifetime cost should approximately match sum of yearly costs for home at index {idx}"


# -------------------------------------------------------------------------
#              STEP 5: FINAL MASKING TESTS
# -------------------------------------------------------------------------

def test_final_masking(sample_homes_df: pd.DataFrame, mock_scenario_params: None, mock_annual_calculation: None) -> None:
    """
    Test final masking of result columns.
    
    This test verifies that:
    1. All result columns are properly masked
    2. Invalid homes have NaN values in all result columns
    3. Valid homes have non-NaN values in result columns
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_scenario_params: Mock for define_scenario_params function
        mock_annual_calculation: Mock for calculate_annual_fuel_costs
    """
    # Set up shorter lifetimes for testing efficiency
    # Set up test parameters
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call the main function
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Check masking for each category
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        valid_mask = sample_homes_df[f'include_{category}']
        lifetime_col = f'iraRef_mp{menu_mp}_{category}_lifetime_fuel_cost'
        
        # Skip categories that don't have results
        if lifetime_col not in df_main.columns:
            continue
            
        # Check invalid homes have NaN values
        invalid_indices = valid_mask[~valid_mask].index
        if len(invalid_indices) > 0:
            assert df_main.loc[invalid_indices, lifetime_col].isna().all(), \
                f"Invalid homes should have NaN values for {lifetime_col}"
        
        # Check valid homes have non-NaN values
        valid_indices = valid_mask[valid_mask].index
        if len(valid_indices) > 0:
            assert not df_main.loc[valid_indices, lifetime_col].isna().all(), \
                f"At least some valid homes should have non-NaN values for {lifetime_col}"


def test_column_tracking_for_masking(sample_homes_df: pd.DataFrame, mock_scenario_params: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test proper tracking of columns for final masking.
    
    This test verifies that:
    1. Columns are properly tracked in all_columns_to_mask
    2. The tracked columns are passed to apply_final_masking
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_scenario_params: Mock for define_scenario_params function
        monkeypatch: Pytest fixture for patching attributes/functions
    """
    # Create a spy function to track calls to apply_final_masking
    captured_columns_to_mask = {}
    
    def spy_apply_final_masking(df, all_columns_to_mask, verbose=True):
        """Spy function to capture columns to mask."""
        # Make a deep copy to avoid reference issues
        captured_columns_to_mask.update(all_columns_to_mask)
        # Return the DataFrame unchanged for this test
        return df
    
    # Replace apply_final_masking with our spy
    monkeypatch.setattr(
        'cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.apply_final_masking',
        spy_apply_final_masking
    )
    
    # Set shorter lifetimes for testing efficiency
    monkeypatch.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', {'heating': 2, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
    monkeypatch.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.EQUIPMENT_SPECS', 
                        {'heating': 2, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
    
    # Set up test parameters
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call the main function
    calculate_lifetime_fuel_costs(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify columns were tracked
    assert len(captured_columns_to_mask) > 0, "Should have captured columns to mask"
    
    # Verify we have entries for all categories
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        assert category in captured_columns_to_mask, f"Category '{category}' should be tracked"
        # Verify at least lifetime column is tracked
        col_names = captured_columns_to_mask[category]
        assert len(col_names) > 0, f"Should have tracked columns for {category}"
        
        # Check for lifetime column
        lifetime_col = f'iraRef_mp{menu_mp}_{category}_lifetime_fuel_cost'
        assert any(col for col in col_names if col.endswith('lifetime_fuel_cost')), \
            f"Should track lifetime column for {category}"


# -------------------------------------------------------------------------
#              INTEGRATION TESTS
# -------------------------------------------------------------------------

def test_lifetime_fuel_costs_basic(sample_homes_df: pd.DataFrame, mock_scenario_params: None, mock_annual_calculation: None) -> None:
    """
    Test basic functionality of calculate_lifetime_fuel_costs.
    
    This test verifies that:
    1. The function runs without errors
    2. The result DataFrames have the expected structure
    3. Values are properly masked based on validation flags
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_scenario_params: Mock for define_scenario_params function
        mock_annual_calculation: Mock for calculate_annual_fuel_costs
    """
    # Set up test parameters
    menu_mp = 0  # Baseline
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call the main function
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify result structure
    assert isinstance(df_main, pd.DataFrame), "df_main should be a DataFrame"
    assert isinstance(df_detailed, pd.DataFrame), "df_detailed should be a DataFrame"
    assert len(df_main) == len(sample_homes_df), "df_main should have same number of rows"
    assert len(df_detailed) == len(sample_homes_df), "df_detailed should have same number of rows"
    
    # Verify main DataFrame has columns for all categories
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        lifetime_col = f'baseline_{category}_lifetime_fuel_cost'
        assert lifetime_col in df_main.columns, f"df_main should have column '{lifetime_col}'"
        
        # Verify values are masked based on inclusion flags
        valid_mask = sample_homes_df[f'include_{category}']
        invalid_indices = valid_mask[~valid_mask].index
        
        if len(invalid_indices) > 0:
            assert df_main.loc[invalid_indices, lifetime_col].isna().all(), \
                f"Invalid homes should have NaN values for {lifetime_col}"


def test_lifetime_fuel_costs_with_baseline(sample_homes_df: pd.DataFrame, df_baseline_costs: pd.DataFrame, 
                                          mock_scenario_params: None, mock_annual_calculation: None) -> None:
    """
    Test calculation with measure package and baseline costs.
    
    This test verifies that:
    1. The function processes baseline costs correctly
    2. Savings columns are calculated as baseline - measure
    3. All values are properly masked
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        df_baseline_costs: Baseline costs DataFrame
        mock_scenario_params: Mock for define_scenario_params function
        mock_annual_calculation: Mock for calculate_annual_fuel_costs
    """
    # Set up test parameters
    menu_mp = 8  # Measure package
    policy_scenario = 'AEO2023 Reference Case'
    
    # Set up annual calculation to ensure consistent values for comparison
    def consistent_annual_calc(df, category, year_label, **kwargs):
        """Return consistent values for testing savings."""
        scenario_prefix = kwargs.get('scenario_prefix', 'baseline_')
        menu_mp = kwargs.get('menu_mp', 0)
        valid_mask = kwargs.get('valid_mask', pd.Series(True, index=df.index))
        
        # Create cost column with predictable values
        cost_col = f"{scenario_prefix}{year_label}_{category}_fuel_cost"
        cost_series = pd.Series(np.nan, index=df.index)
        
        # Set consistent values for valid homes
        for idx in valid_mask[valid_mask].index:
            # Make measure costs consistently 70% of baseline
            multiplier = 1.0 if menu_mp == 0 else 0.7
            # Base value on index and category for consistency
            base_value = 100.0 + (idx * 20)
            if category == 'heating':
                base_value *= 2.0
            elif category == 'waterHeating':
                base_value *= 1.5
            
            cost_series.loc[idx] = base_value * multiplier
        
        # Create result dictionary
        annual_costs = {cost_col: cost_series}
        
        # For measure packages, calculate savings column
        if menu_mp != 0:
            # Look up baseline cost from baseline_costs DataFrame
            baseline_col = f"baseline_{year_label}_{category}_fuel_cost"
            if baseline_col in df_baseline_costs.columns:
                savings_col = f"{scenario_prefix}{year_label}_{category}_savings_fuel_cost"
                savings_series = pd.Series(np.nan, index=df.index)
                
                # Calculate savings only for valid homes
                for idx in valid_mask[valid_mask].index:
                    baseline_cost = df_baseline_costs.loc[idx, baseline_col]
                    measure_cost = cost_series.loc[idx]
                    savings_series.loc[idx] = baseline_cost - measure_cost
                
                annual_costs[savings_col] = savings_series
        
        return annual_costs, cost_series
    
    # Only mock for this specific test
    original_function = calculate_annual_fuel_costs
    with monkeypatch.context() as m:
        m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.calculate_annual_fuel_costs', 
                  consistent_annual_calc)
        
        # Call the main function
        df_main, df_detailed = calculate_lifetime_fuel_costs(
            df=sample_homes_df,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            df_baseline_costs=df_baseline_costs,
            verbose=False
        )
    
    # Verify result structure
    for category in ['heating', 'waterHeating']:  # Test just a couple categories for efficiency
        costs_col = f'iraRef_mp{menu_mp}_{category}_lifetime_fuel_cost'
        savings_col = f'iraRef_mp{menu_mp}_{category}_lifetime_savings_fuel_cost'
        baseline_col = f'baseline_{category}_lifetime_fuel_cost'
        
        # Verify columns exist
        assert costs_col in df_main.columns, f"df_main should have column '{costs_col}'"
        assert savings_col in df_main.columns, f"df_main should have column '{savings_col}'"
        
        # Get valid homes
        valid_mask = sample_homes_df[f'include_{category}']
        valid_indices = valid_mask[valid_mask].index
        
        # For at least one valid home, verify savings calculation
        if len(valid_indices) > 0:
            idx = valid_indices[0]
            baseline_cost = df_baseline_costs.loc[idx, baseline_col]
            measure_cost = df_main.loc[idx, costs_col]
            savings = df_main.loc[idx, savings_col]
            
            # Allow for small rounding differences
            assert abs((baseline_cost - measure_cost) - savings) < 0.1, \
                f"Savings should be baseline - measure for home at index {idx}"


def test_end_to_end_workflow(sample_homes_df: pd.DataFrame, mock_fuel_prices: Dict, mock_scenario_params: None) -> None:
    """
    Test the end-to-end workflow with real calculation logic.
    
    This test verifies that:
    1. The full workflow from data input to final output works properly
    2. Calculations without mocked annual calculation function are correct
    3. The module properly interacts with define_scenario_params
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_fuel_prices: Mock fuel price data
        mock_scenario_params: Mock for define_scenario_params function
    """
    # Use shorter lifetimes for testing efficiency
    with monkeypatch.context() as m:
        m.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', 
                  {'heating': 2, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
        m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.EQUIPMENT_SPECS', 
                  {'heating': 2, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
        
        # Set up test parameters - use a different policy scenario than other tests
        menu_mp = 0  # Baseline
        policy_scenario = 'No Inflation Reduction Act'
        
        # Call the main function without mocking calculate_annual_fuel_costs
        df_main, df_detailed = calculate_lifetime_fuel_costs(
            df=sample_homes_df,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )
    
    # Verify result structure
    assert isinstance(df_main, pd.DataFrame), "df_main should be a DataFrame"
    assert isinstance(df_detailed, pd.DataFrame), "df_detailed should be a DataFrame"
    
    # Check at least one category
    category = 'heating'
    lifetime_col = f'baseline_{category}_lifetime_fuel_cost'
    
    assert lifetime_col in df_main.columns, f"df_main should have column '{lifetime_col}'"
    
    # Verify values are masked based on inclusion flags
    valid_mask = sample_homes_df[f'include_{category}']
    valid_indices = valid_mask[valid_mask].index
    invalid_indices = valid_mask[~valid_mask].index
    
    # Valid homes should have non-NaN values
    if len(valid_indices) > 0:
        assert not df_main.loc[valid_indices, lifetime_col].isna().all(), \
            "At least some valid homes should have non-NaN values"
    
    # Invalid homes should have NaN values
    if len(invalid_indices) > 0:
        assert df_main.loc[invalid_indices, lifetime_col].isna().all(), \
            "Invalid homes should have NaN values"


# -------------------------------------------------------------------------
#              PARAMETRIZED TESTS
# -------------------------------------------------------------------------

def test_across_categories(sample_homes_df: pd.DataFrame, mock_fuel_prices: Dict, 
                          mock_scenario_params: None, category: str, mock_annual_calculation: None) -> None:
    """
    Test calculation across different equipment categories.
    
    This parametrized test verifies that:
    1. The function works for all equipment categories
    2. Lifetimes are correctly applied per category
    3. Validation is correctly applied per category
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_fuel_prices: Mock fuel price data
        mock_scenario_params: Mock for define_scenario_params function
        category: Parametrized equipment category
        mock_annual_calculation: Mock for calculate_annual_fuel_costs
    """
    # Use shorter lifetimes for testing efficiency
    with monkeypatch.context() as m:
        m.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', 
                  {'heating': 2, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
        m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.EQUIPMENT_SPECS', 
                  {'heating': 2, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
        
        # Set up test parameters
        menu_mp = 0  # Baseline
        policy_scenario = 'AEO2023 Reference Case'
        
        # Call the main function
        df_main, df_detailed = calculate_lifetime_fuel_costs(
            df=sample_homes_df,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )
    
    # Verify result has lifetime column for this category
    lifetime_col = f'baseline_{category}_lifetime_fuel_cost'
    assert lifetime_col in df_main.columns, f"Result should have column '{lifetime_col}'"
    
    # Verify values are masked based on inclusion flags
    valid_mask = sample_homes_df[f'include_{category}']
    valid_indices = valid_mask[valid_mask].index
    invalid_indices = valid_mask[~valid_mask].index
    
    # Valid homes should have non-NaN values
    if len(valid_indices) > 0:
        assert not df_main.loc[valid_indices, lifetime_col].isna().all(), \
            "At least some valid homes should have non-NaN values"
    
    # Invalid homes should have NaN values
    if len(invalid_indices) > 0:
        assert df_main.loc[invalid_indices, lifetime_col].isna().all(), \
            "Invalid homes should have NaN values"


def test_across_policy_scenarios(sample_homes_df: pd.DataFrame, mock_fuel_prices: Dict, 
                                mock_scenario_params: None, policy_scenario: str, mock_annual_calculation: None) -> None:
    """
    Test calculation across different policy scenarios.
    
    This parametrized test verifies that:
    1. The function works for all policy scenarios
    2. Column prefixes match the policy scenario
    3. Fuel prices are applied according to the policy scenario
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_fuel_prices: Mock fuel price data
        mock_scenario_params: Mock for define_scenario_params function
        policy_scenario: Parametrized policy scenario
        mock_annual_calculation: Mock for calculate_annual_fuel_costs
    """
    # Use shorter lifetimes for testing efficiency
    with monkeypatch.context() as m:
        m.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', 
                  {'heating': 2, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
        m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.EQUIPMENT_SPECS', 
                  {'heating': 2, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
        
        # Set up test parameters
        menu_mp = 8  # Measure package
        category = 'heating'  # Test one category for efficiency
        
        # Call the main function
        df_main, df_detailed = calculate_lifetime_fuel_costs(
            df=sample_homes_df,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )
    
    # Determine expected column prefix
    if policy_scenario == 'No Inflation Reduction Act':
        expected_prefix = f"preIRA_mp{menu_mp}_"
    else:
        expected_prefix = f"iraRef_mp{menu_mp}_"
    
    # Verify result has properly prefixed columns
    lifetime_col = f'{expected_prefix}{category}_lifetime_fuel_cost'
    assert lifetime_col in df_main.columns, f"Result should have column '{lifetime_col}'"
    
    # Verify values are present for valid homes
    valid_mask = sample_homes_df[f'include_{category}']
    valid_indices = valid_mask[valid_mask].index
    
    if len(valid_indices) > 0:
        # Check at least one valid home
        idx = valid_indices[0]
        assert not pd.isna(df_main.loc[idx, lifetime_col]), \
            f"Valid home should have non-NaN value for {policy_scenario}"


def test_across_measure_packages(sample_homes_df: pd.DataFrame, mock_fuel_prices: Dict, 
                               mock_scenario_params: None, menu_mp: int, mock_annual_calculation: None) -> None:
    """
    Test calculation across different measure packages.
    
    This parametrized test verifies that:
    1. The function works for both baseline and measure packages
    2. Measure package calculations include scenario-specific prefixes
    3. Baseline calculations have the baseline_ prefix
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_fuel_prices: Mock fuel price data
        mock_scenario_params: Mock for define_scenario_params function
        menu_mp: Parametrized measure package identifier
        mock_annual_calculation: Mock for calculate_annual_fuel_costs
    """
    # Use shorter lifetimes for testing efficiency
    with monkeypatch.context() as m:
        m.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', 
                  {'heating': 2, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
        m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.EQUIPMENT_SPECS', 
                  {'heating': 2, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
        
        # Set up test parameters
        policy_scenario = 'AEO2023 Reference Case'
        category = 'heating'  # Test one category for efficiency
        
        # Call the main function
        df_main, df_detailed = calculate_lifetime_fuel_costs(
            df=sample_homes_df,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )
    
    # Determine expected column prefix
    if menu_mp == 0:
        expected_prefix = "baseline_"
    else:
        expected_prefix = f"iraRef_mp{menu_mp}_"
    
    # Verify result has properly prefixed columns
    lifetime_col = f'{expected_prefix}{category}_lifetime_fuel_cost'
    assert lifetime_col in df_main.columns, f"Result should have column '{lifetime_col}'"
    
    # Verify values are present for valid homes
    valid_mask = sample_homes_df[f'include_{category}']
    valid_indices = valid_mask[valid_mask].index
    
    if len(valid_indices) > 0:
        # Check at least one valid home
        idx = valid_indices[0]
        assert not pd.isna(df_main.loc[idx, lifetime_col]), \
            f"Valid home should have non-NaN value for menu_mp={menu_mp}"


# -------------------------------------------------------------------------
#              EDGE CASE TESTS
# -------------------------------------------------------------------------

def test_empty_dataframe(mock_fuel_prices: Dict, mock_scenario_params: None) -> None:
    """
    Test handling of empty DataFrame.
    
    This test verifies that:
    1. The function handles empty DataFrames gracefully
    2. The result DataFrames are also empty but have the expected structure
    
    Args:
        mock_fuel_prices: Mock fuel price data
        mock_scenario_params: Mock for define_scenario_params function
    """
    # Create an empty DataFrame with required columns
    empty_df = pd.DataFrame(columns=['state', 'census_division'])
    
    # Set up test parameters
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call the main function
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=empty_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify result structure
    assert isinstance(df_main, pd.DataFrame), "df_main should be a DataFrame"
    assert isinstance(df_detailed, pd.DataFrame), "df_detailed should be a DataFrame"
    assert len(df_main) == 0, "df_main should be empty"
    assert len(df_detailed) == 0, "df_detailed should be empty"


def test_all_invalid_homes(sample_homes_df: pd.DataFrame, mock_scenario_params: None, mock_annual_calculation: None) -> None:
    """
    Test calculation when all homes are invalid.
    
    This test verifies that:
    1. The function handles the case where all homes are invalid
    2. All result columns contain NaN values for all homes
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_scenario_params: Mock for define_scenario_params function
        mock_annual_calculation: Mock for calculate_annual_fuel_costs
    """
    # Create modified DataFrame with all homes invalid
    df_modified = sample_homes_df.copy()
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        df_modified[f'include_{category}'] = False
    
    # Use shorter lifetimes for testing efficiency
    with monkeypatch.context() as m:
        m.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', 
                  {'heating': 2, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
        m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.EQUIPMENT_SPECS', 
                  {'heating': 2, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
        
        # Set up test parameters
        menu_mp = 0
        policy_scenario = 'AEO2023 Reference Case'
        
        # Call the main function
        df_main, df_detailed = calculate_lifetime_fuel_costs(
            df=df_modified,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )
    
    # Verify that all result columns have NaN values
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        lifetime_col = f'baseline_{category}_lifetime_fuel_cost'
        if lifetime_col in df_main.columns:
            assert df_main[lifetime_col].isna().all(), \
                f"All homes should have NaN values for {lifetime_col}"


def test_missing_columns(mock_fuel_prices: Dict, mock_scenario_params: None) -> None:
    """
    Test error handling for missing required columns.
    
    This test verifies that:
    1. The function properly detects and reports missing columns
    2. Error messages are informative
    
    Args:
        mock_fuel_prices: Mock fuel price data
        mock_scenario_params: Mock for define_scenario_params function
    """
    # Create a DataFrame missing required columns
    df_missing = pd.DataFrame({
        'include_heating': [True, True],
        'include_waterHeating': [True, True],
        'include_clothesDrying': [True, True],
        'include_cooking': [True, True]
    })
    
    # Set up test parameters
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call the main function and expect an error
    with pytest.raises(KeyError) as excinfo:
        calculate_lifetime_fuel_costs(
            df=df_missing,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )
    
    # Verify error message mentions missing columns
    error_msg = str(excinfo.value)
    assert "state" in error_msg or "census_division" in error_msg, \
        "Error message should mention at least one of the missing columns"


def test_extreme_values(sample_homes_df: pd.DataFrame, mock_scenario_params: None) -> None:
    """
    Test handling of extreme consumption and price values.
    
    This test verifies that:
    1. The function handles very high consumption values
    2. The function handles zero consumption values
    3. Calculations remain numerically stable
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_scenario_params: Mock for define_scenario_params function
    """
    # Create a DataFrame with extreme values
    df_extreme = sample_homes_df.copy()
    
    # Add extreme consumption values
    category = 'heating'
    year = 2024
    consumption_col = f'baseline_{year}_{category}_consumption'
    
    # Set a very high value for first home
    df_extreme.loc[0, consumption_col] = 1e9  # 1 billion
    
    # Set zero for second home
    df_extreme.loc[1, consumption_col] = 0.0
    
    # Use a custom mock for annual calculation to test extreme values
    def extreme_values_annual_calc(df, category, year_label, **kwargs):
        """Handle extreme values."""
        scenario_prefix = kwargs.get('scenario_prefix', 'baseline_')
        menu_mp = kwargs.get('menu_mp', 0)
        valid_mask = kwargs.get('valid_mask', pd.Series(True, index=df.index))
        
        # Create result dictionary
        annual_costs = {}
        cost_col = f"{scenario_prefix}{year_label}_{category}_fuel_cost"
        
        # Get consumption column
        consumption_col = f"{scenario_prefix}{year_label}_{category}_consumption"
        if consumption_col not in df.columns:
            # For measure packages, use mp column
            if menu_mp != 0:
                consumption_col = f"mp{menu_mp}_{year_label}_{category}_consumption"
        
        # Use a fixed price for testing
        price = 0.1
        
        # Calculate costs for valid homes
        costs = pd.Series(np.nan, index=df.index)
        for idx in valid_mask[valid_mask].index:
            if consumption_col in df.columns:
                costs.loc[idx] = df.loc[idx, consumption_col] * price
            else:
                costs.loc[idx] = 100.0  # default value
        
        annual_costs[cost_col] = costs
        return annual_costs, costs
    
    # Use shorter lifetimes for testing efficiency
    with monkeypatch.context() as m:
        m.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', 
                  {'heating': 2, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
        m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.EQUIPMENT_SPECS', 
                  {'heating': 2, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
        m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.calculate_annual_fuel_costs',
                 extreme_values_annual_calc)
        
        # Set up test parameters
        menu_mp = 0
        policy_scenario = 'AEO2023 Reference Case'
        
        # Call the main function
        df_main, df_detailed = calculate_lifetime_fuel_costs(
            df=df_extreme,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )
    
    # Verify that function handled extreme values
    lifetime_col = f'baseline_{category}_lifetime_fuel_cost'
    assert lifetime_col in df_main.columns, f"Result should have column '{lifetime_col}'"
    
    # Check first home with very high consumption has a finite value
    assert np.isfinite(df_main.loc[0, lifetime_col]), \
        "Home with very high consumption should have finite lifetime cost"
    
    # Check second home with zero consumption has a value of zero
    assert df_main.loc[1, lifetime_col] == 0.0, \
        "Home with zero consumption should have lifetime cost of 0.0"


def test_invalid_policy_scenario(sample_homes_df: pd.DataFrame, mock_fuel_prices: Dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test error handling for invalid policy scenario.
    
    This test verifies that:
    1. The function properly detects and reports invalid policy scenarios
    2. Error messages are informative
    
    Args:
        sample_homes_df: Sample DataFrame with home data
        mock_fuel_prices: Mock fuel price data
        monkeypatch: Pytest fixture for patching attributes/functions
    """
    # Create a mock that raises an error for invalid policy scenario
    def mock_function(menu_mp, policy_scenario):
        """Mock implementation that raises error for invalid policy scenario."""
        if policy_scenario not in ['No Inflation Reduction Act', 'AEO2023 Reference Case']:
            raise ValueError(f"Invalid policy scenario: {policy_scenario}")
        
        # Return some default values for valid scenarios
        if menu_mp == 0:
            scenario_prefix = "baseline_"
        elif policy_scenario == 'No Inflation Reduction Act':
            scenario_prefix = f"preIRA_mp{menu_mp}_"
        else:
            scenario_prefix = f"iraRef_mp{menu_mp}_"
            
        return (scenario_prefix, "MidCase", {}, {}, {}, mock_fuel_prices)
    
    # Apply the patch
    monkeypatch.setattr(
        'cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.define_scenario_params',
        mock_function
    )
    
    # Set up test parameters - use invalid policy scenario
    menu_mp = 0
    policy_scenario = 'Invalid Scenario'
    
    # Call the main function and expect an error
    with pytest.raises(ValueError) as excinfo:
        calculate_lifetime_fuel_costs(
            df=sample_homes_df,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )
    
    # Verify error message mentions invalid policy scenario
    error_msg = str(excinfo.value)
    assert "Invalid policy scenario" in error_msg, \
        "Error message should mention invalid policy scenario"
    assert policy_scenario in error_msg, \
        "Error message should include the invalid scenario name"


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])

    # -------------------------------------------------------------------------
    #              PER-CATEGORY VALIDATION TESTS
    # -------------------------------------------------------------------------

    def test_per_category_validation(sample_homes_df: pd.DataFrame, mock_scenario_params: None, mock_annual_calculation: None) -> None:
        """
        Test that validation is properly applied for each category independently.
        
        This test verifies that:
        1. Categories are validated independently of each other
        2. Invalid homes for one category don't affect calculations for other categories
        3. Each category's masking is properly applied to its respective columns
        
        Args:
            sample_homes_df: Sample DataFrame with home data
            mock_scenario_params: Mock for define_scenario_params function
            mock_annual_calculation: Mock for calculate_annual_fuel_costs
        """
        # Use shorter lifetimes for testing efficiency
        with monkeypatch.context() as m:
            m.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', 
                      {'heating': 2, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
            m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.EQUIPMENT_SPECS', 
                      {'heating': 2, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
            
            # Set up test parameters
            menu_mp = 0
            policy_scenario = 'AEO2023 Reference Case'
            
            # Call the main function
            df_main, df_detailed = calculate_lifetime_fuel_costs(
                df=sample_homes_df,
                menu_mp=menu_mp,
                policy_scenario=policy_scenario,
                verbose=False
            )
        
        # For each category, check that masking is applied independently
        for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
            lifetime_col = f'baseline_{category}_lifetime_fuel_cost'
            if lifetime_col not in df_main.columns:
                continue
            
            valid_mask = sample_homes_df[f'include_{category}']
            valid_indices = valid_mask[valid_mask].index
            invalid_indices = valid_mask[~valid_mask].index
            
            # Check valid homes have values
            if len(valid_indices) > 0:
                assert not df_main.loc[valid_indices, lifetime_col].isna().all(), \
                    f"Valid homes should have values for {category}"
            
            # Check invalid homes have NaN values
            if len(invalid_indices) > 0:
                assert df_main.loc[invalid_indices, lifetime_col].isna().all(), \
                    f"Invalid homes should have NaN values for {category}"


    # -------------------------------------------------------------------------
    #              VALIDATION FRAMEWORK COMPREHENSIVE TEST
    # -------------------------------------------------------------------------

    def test_validation_framework_full_flow(sample_homes_df: pd.DataFrame, mock_scenario_params: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Test the full flow of the 5-step validation framework.
        
        This comprehensive test verifies that:
        1. Mask initialization is performed correctly
        2. Series initialization creates proper starting values
        3. Valid-only calculation is restricted to valid homes
        4. List-based collection maintains proper values
        5. Final masking is applied to all relevant columns
        
        The test uses spy functions to track each step's execution.
        
        Args:
            sample_homes_df: Sample DataFrame with home data
            mock_scenario_params: Mock for define_scenario_params function
            monkeypatch: Pytest fixture for patching attributes/functions
        """
        # Create spy functions to track execution of each validation step
        tracking = {
            'initialized_mask': None,
            'initialized_series': None,
            'calculated_homes': [],
            'tracked_columns': {}
        }
        
        # Spy for mask initialization
        original_init_tracking = initialize_validation_tracking
        def spy_init_tracking(*args, **kwargs):
            result = original_init_tracking(*args, **kwargs)
            tracking['initialized_mask'] = result[1].copy()  # Store the mask
            return result
        
        # Spy for series initialization
        original_create_series = create_retrofit_only_series
        def spy_create_series(*args, **kwargs):
            result = original_create_series(*args, **kwargs)
            tracking['initialized_series'] = result.copy()
            return result
        
        # Spy for annual calculation (valid-only calculation)
        original_annual_calc = calculate_annual_fuel_costs
        def spy_annual_calc(*args, **kwargs):
            if 'valid_mask' in kwargs:
                # Track which homes were included in calculation
                tracking['calculated_homes'].append(kwargs['valid_mask'].copy())
            result = original_annual_calc(*args, **kwargs)
            return result
        
        # Spy for final masking
        original_final_masking = apply_final_masking
        def spy_final_masking(df, all_columns_to_mask, verbose=True):
            tracking['tracked_columns'] = all_columns_to_mask.copy()
            return original_final_masking(df, all_columns_to_mask, verbose)
        
        # Apply all the spy functions
        with monkeypatch.context() as m:
            m.setattr('cmu_tare_model.utils.validation_framework.initialize_validation_tracking', spy_init_tracking)
            m.setattr('cmu_tare_model.utils.validation_framework.create_retrofit_only_series', spy_create_series)
            m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.calculate_annual_fuel_costs', spy_annual_calc)
            m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.apply_final_masking', spy_final_masking)
            
            # Set shorter lifetimes for testing efficiency
            m.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', {'heating': 2, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
            m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.EQUIPMENT_SPECS', {'heating': 2, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
            
            # Call the main function with a single category for simplicity
            category = 'heating'
            df_copy = sample_homes_df.copy()
            
            # Run the calculation
            df_main, df_detailed = calculate_lifetime_fuel_costs(
                df=df_copy,
                menu_mp=8,  # Measure package
                policy_scenario='AEO2023 Reference Case',
                verbose=False
            )
        
        # Verify that all steps executed properly
        # 1. Mask initialization
        assert tracking['initialized_mask'] is not None, "Mask initialization should have executed"
        assert tracking['initialized_mask'].equals(sample_homes_df[f'include_{category}']), \
            "Initialized mask should match include_heating column"
        
        # 2. Series initialization
        assert tracking['initialized_series'] is not None, "Series initialization should have executed"
        assert len(tracking['initialized_series']) == len(sample_homes_df), \
            "Initialized series should have same length as DataFrame"
        
        # 3. Valid-only calculation 
        assert len(tracking['calculated_homes']) > 0, "At least one valid-only calculation should have executed"
        valid_mask = sample_homes_df[f'include_{category}']
        for calc_mask in tracking['calculated_homes']:
            # Every mask used for calculation should be a subset of the valid homes
            assert calc_mask[~valid_mask].sum() == 0, "Calculations should only be performed for valid homes"
        
        # 5. Final masking - check columns were tracked
        assert len(tracking['tracked_columns']) > 0, "Columns should be tracked for final masking"
        assert category in tracking['tracked_columns'], f"Category '{category}' should be in tracked columns"
        assert len(tracking['tracked_columns'][category]) > 0, f"Should track at least one column for {category}"


    # -------------------------------------------------------------------------
    #              CORRECT LIST-BASED COLLECTION TEST
    # -------------------------------------------------------------------------

    def test_correct_list_based_collection(sample_homes_df: pd.DataFrame, mock_scenario_params: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Test that list-based collection correctly accumulates values.
        
        This test fixes the previously failing test_list_based_collection
        by properly using monkeypatch.setattr instead of getattr.
        
        Args:
            sample_homes_df: Sample DataFrame with home data
            mock_scenario_params: Mock for define_scenario_params function
            monkeypatch: Pytest fixture for patching attributes/functions
        """
        # Create a spy function to track calls to calculate_annual_fuel_costs
        yearly_costs_captured = []
        
        def spy_annual_costs(*args, **kwargs):
            """Spy function to capture annual costs."""
            # Get the DataFrame from args or kwargs
            df = kwargs.get('df', args[0] if args else None)
            category = kwargs.get('category', args[1] if len(args) > 1 else None)
            year_label = kwargs.get('year_label', args[2] if len(args) > 2 else None)
            scenario_prefix = kwargs.get('scenario_prefix', '')
            valid_mask = kwargs.get('valid_mask', pd.Series(True, index=df.index))
            
            # Create a simple result with predictable values
            cost_col = f"{scenario_prefix}{year_label}_{category}_fuel_cost"
            cost_series = pd.Series(0.0, index=df.index)
            
            # Set values based on index for valid homes
            for idx in valid_mask[valid_mask].index:
                # Make value dependent on year and index for verification
                cost_series.loc[idx] = 100 + (year_label - 2024) * 10 + idx
            
            # Capture the cost series before returning
            yearly_costs_captured.append(cost_series.copy())
            
            return {cost_col: cost_series}, cost_series
        
        # Replace the original function with our spy
        with monkeypatch.context() as m:
            m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.calculate_annual_fuel_costs',
                      spy_annual_costs)
            
            # Set shorter lifetimes for testing efficiency
            m.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', {'heating': 3, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
            m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.EQUIPMENT_SPECS', 
                      {'heating': 3, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
            
            # Set up test parameters
            category = 'heating'
            menu_mp = 8
            policy_scenario = 'AEO2023 Reference Case'
            
            # Call the main function with verbose=False
            df_main, df_detailed = calculate_lifetime_fuel_costs(
                df=sample_homes_df,
                menu_mp=menu_mp,
                policy_scenario=policy_scenario,
                verbose=False
            )
        
        # Verify we captured yearly values
        assert len(yearly_costs_captured) > 0, "Should have captured yearly cost values"
        
        # Verify the lifetime column contains the sum of yearly values for valid homes
        lifetime_col = f'iraRef_mp{menu_mp}_{category}_lifetime_fuel_cost'
        assert lifetime_col in df_main.columns, f"Result should contain column '{lifetime_col}'"
        
        # Get valid mask for the category
        valid_mask = sample_homes_df[f'include_{category}']
        valid_indices = valid_mask[valid_mask].index
        
        # For at least one valid home, verify sum of yearly values matches lifetime value
        if len(valid_indices) > 0:
            idx = valid_indices[0]
            yearly_sum = sum(yearly_cost[idx] for yearly_cost in yearly_costs_captured if idx in yearly_cost.index)
            assert abs(df_main.loc[idx, lifetime_col] - yearly_sum) < 0.1, \
                f"Lifetime cost should match sum of yearly costs for home at index {idx}"


    # -------------------------------------------------------------------------
    #              CORRECT SAVINGS CALCULATION TEST
    # -------------------------------------------------------------------------

    def test_correct_savings_calculation(sample_homes_df: pd.DataFrame, mock_scenario_params: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Test that savings calculations are correct.
        
        This test fixes the previously failing test_lifetime_fuel_costs_with_baseline
        by ensuring baseline and measure package values are consistent for comparison.
        
        Args:
            sample_homes_df: Sample DataFrame with home data
            mock_scenario_params: Mock for define_scenario_params function
            monkeypatch: Pytest fixture for patching attributes/functions
        """
        # Create baseline costs DataFrame with predictable values
        df_baseline_costs = pd.DataFrame(index=sample_homes_df.index)
        
        # Add baseline costs for testing
        baseline_costs = {}
        for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
            # Create lifetime column with simple values (100 * index + offset based on category)
            offset = {'heating': 1000, 'waterHeating': 800, 'clothesDrying': 600, 'cooking': 400}
            col_name = f"baseline_{category}_lifetime_fuel_cost"
            baseline_costs[col_name] = [
                offset[category] + (idx * 100) for idx in range(len(sample_homes_df))
            ]
            
            # Add annual costs for 3 years
            for year in range(2024, 2027):
                col_name = f"baseline_{year}_{category}_fuel_cost"
                # Annual cost is lifetime / 10 for simplicity
                baseline_costs[col_name] = [
                    (offset[category] + (idx * 100)) / 10 for idx in range(len(sample_homes_df))
                ]
        
        df_baseline_costs = pd.DataFrame(baseline_costs, index=sample_homes_df.index)
        
        # Create a mock annual calculation function that returns predictable values
        # relative to the baseline values for consistent savings calculation
        def mock_annual_calc(df, category, year_label, **kwargs):
            """Return consistent values for testing savings."""
            scenario_prefix = kwargs.get('scenario_prefix', 'baseline_')
            menu_mp = kwargs.get('menu_mp', 0)
            valid_mask = kwargs.get('valid_mask', pd.Series(True, index=df.index))
            
            # Create cost column with predictable values
            cost_col = f"{scenario_prefix}{year_label}_{category}_fuel_cost"
            cost_series = pd.Series(np.nan, index=df.index)
            
            # Get baseline cost column for reference
            baseline_col = f"baseline_{year_label}_{category}_fuel_cost"
            
            # Set values for valid homes
            for idx in valid_mask[valid_mask].index:
                if menu_mp == 0:
                    # For baseline, use value from baseline costs
                    if baseline_col in df_baseline_costs.columns:
                        cost_series.loc[idx] = df_baseline_costs.loc[idx, baseline_col]
                    else:
                        # Default value based on index and category
                        offset = {'heating': 100, 'waterHeating': 80, 'clothesDrying': 60, 'cooking': 40}
                        cost_series.loc[idx] = offset[category] + (idx * 10)
                else:
                    # For measure package, use 60% of baseline cost (40% savings)
                    if baseline_col in df_baseline_costs.columns:
                        cost_series.loc[idx] = df_baseline_costs.loc[idx, baseline_col] * 0.6
                    else:
                        # Default value
                        offset = {'heating': 60, 'waterHeating': 48, 'clothesDrying': 36, 'cooking': 24}
                        cost_series.loc[idx] = offset[category] + (idx * 6)
            
            # Create result dictionary
            annual_costs = {cost_col: cost_series}
            
            # For measure packages, calculate savings column if baseline costs available
            if menu_mp != 0:
                savings_col = f"{scenario_prefix}{year_label}_{category}_savings_fuel_cost"
                savings_series = pd.Series(np.nan, index=df.index)
                
                if baseline_col in df_baseline_costs.columns:
                    # Calculate savings only for valid homes
                    for idx in valid_mask[valid_mask].index:
                        baseline_cost = df_baseline_costs.loc[idx, baseline_col]
                        measure_cost = cost_series.loc[idx]
                        savings_series.loc[idx] = baseline_cost - measure_cost
                    
                    annual_costs[savings_col] = savings_series
            
            return annual_costs, cost_series
        
        # Set up monkeypatch
        with monkeypatch.context() as m:
            m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.calculate_annual_fuel_costs',
                      mock_annual_calc)
            
            # Set shorter lifetimes for testing efficiency
            m.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', {'heating': 3, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
            m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.EQUIPMENT_SPECS', 
                      {'heating': 3, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
            
            # Set up test parameters
            menu_mp = 8
            policy_scenario = 'AEO2023 Reference Case'
            
            # Call the main function
            df_main, df_detailed = calculate_lifetime_fuel_costs(
                df=sample_homes_df,
                menu_mp=menu_mp,
                policy_scenario=policy_scenario,
                df_baseline_costs=df_baseline_costs,
                verbose=False
            )
        
        # Verify savings calculation for each category
        for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
            costs_col = f'iraRef_mp{menu_mp}_{category}_lifetime_fuel_cost'
            savings_col = f'iraRef_mp{menu_mp}_{category}_lifetime_savings_fuel_cost'
            baseline_col = f'baseline_{category}_lifetime_fuel_cost'
            
            # Skip categories that don't have these columns
            if baseline_col not in df_baseline_costs.columns or costs_col not in df_main.columns:
                continue
            
            # Get valid homes
            valid_mask = sample_homes_df[f'include_{category}']
            valid_indices = valid_mask[valid_mask].index
            
            # For valid homes, verify savings calculation
            for idx in valid_indices:
                baseline_cost = df_baseline_costs.loc[idx, baseline_col]
                measure_cost = df_main.loc[idx, costs_col]
                savings = df_main.loc[idx, savings_col]
                
                # Allow for small rounding differences
                assert abs((baseline_cost - measure_cost) - savings) < 0.1, \
                    f"Savings should be baseline - measure for home at index {idx} in {category}"


    if __name__ == '__main__':
        pytest.main(['-xvs', __file__])