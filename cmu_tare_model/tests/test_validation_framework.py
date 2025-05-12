"""
test_validation_framework_functional.py

Pytest test suite for validating the 5-step validation framework implementation.
This test suite verifies each step of the framework:
1. Mask Initialization with initialize_validation_tracking()
2. Series Initialization with create_retrofit_only_series()
3. Valid-Only Calculation for qualifying homes
4. Valid-Only Updates using list-based collection
5. Final Masking with apply_final_masking()

Tests are organized as simple functions with clear naming conventions
and leverage fixtures for reusable test data.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Set

# Import the validation framework utilities
from cmu_tare_model.utils.validation_framework import (
    initialize_validation_tracking,
    create_retrofit_only_series,
    apply_final_masking,
    get_valid_calculation_mask,
    apply_new_columns_to_dataframe,
    replace_small_values_with_nan,
    calculate_avoided_values
)

# Import constants
from cmu_tare_model.constants import EQUIPMENT_SPECS, FUEL_MAPPING, UPGRADE_COLUMNS


# -------------------------------------------------------------------------
#                           TEST FIXTURES
# -------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Mock the constants module to isolate tests from external dependencies.
    
    This fixture automatically applies to all tests and provides consistent
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
    
    # Apply all mocks to relevant modules
    monkeypatch.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', mock_equipment_specs)
    monkeypatch.setattr('cmu_tare_model.constants.FUEL_MAPPING', mock_fuel_mapping)
    monkeypatch.setattr('cmu_tare_model.constants.UPGRADE_COLUMNS', mock_upgrade_columns)
    monkeypatch.setattr('cmu_tare_model.utils.validation_framework.EQUIPMENT_SPECS', mock_equipment_specs)
    monkeypatch.setattr('cmu_tare_model.utils.validation_framework.UPGRADE_COLUMNS', mock_upgrade_columns)
    
    # Also apply to other modules to ensure consistency
    monkeypatch.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.EQUIPMENT_SPECS', 
                        mock_equipment_specs)
    monkeypatch.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.FUEL_MAPPING', 
                        mock_fuel_mapping)


@pytest.fixture
def sample_homes_df() -> pd.DataFrame:
    """
    Generate sample DataFrame with comprehensive data for testing.
    
    This fixture creates test data with diverse scenarios including:
    - Valid and invalid homes across different equipment categories
    - Different fuel types and technologies
    - Homes with and without scheduled retrofits
    - Mix of valid/invalid data across categories
    
    Returns:
        pd.DataFrame: Sample DataFrame with test data for validation testing
    """
    data = {
        # Metadata columns for lookup operations
        'state': ['CA', 'TX', 'NY', 'FL', 'IL'],
        'census_division': ['Pacific', 'West South Central', 'Middle Atlantic', 'South Atlantic', 'East North Central'],
        
        # Category-specific validation flags (True = valid, False = invalid)
        'include_heating': [True, True, True, True, False],
        'include_waterHeating': [True, True, True, False, True],
        'include_clothesDrying': [True, True, False, True, False],
        'include_cooking': [False, True, True, False, True],
        
        # Baseline fuel types (some valid, some invalid)
        'base_heating_fuel': ['Electricity', 'Natural Gas', 'Propane', 'Fuel Oil', 'Invalid'],
        'base_waterHeating_fuel': ['Electricity', 'Natural Gas', 'Electricity', 'Fuel Oil', 'Propane'],
        'base_clothesDrying_fuel': ['Electricity', 'Natural Gas', 'Propane', 'Electricity', 'Natural Gas'],
        'base_cooking_fuel': ['Natural Gas', 'Propane', 'Natural Gas', 'Propane', 'Natural Gas'],
    }
    
    # Add equipment types and retrofit flags
    data.update({
        'heating_type': ['Electricity Baseboard', 'Natural Gas Fuel Furnace', 'Propane Fuel Furnace', 'Fuel Oil Fuel Furnace', 'Invalid'],
        'waterHeating_type': ['Electric Standard', 'Natural Gas Standard', 'Electric Heat Pump', 'Fuel Oil Standard', 'Propane Standard'],
        'upgrade_hvac_heating_efficiency': ['ASHP', None, 'ASHP', None, 'ASHP'],
        'upgrade_water_heater_efficiency': ['HP', None, None, 'HP', None],
        'upgrade_clothes_dryer': [None, 'Electric', None, None, 'Electric'],
        'upgrade_cooking_range': ['Induction', None, 'Resistance', None, None],
    })
    
    # Generate baseline consumption columns for all fuel types
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        for fuel in ['electricity', 'naturalGas', 'fuelOil', 'propane']:
            col_name = f'base_{fuel}_{category}_consumption'
            data[col_name] = [100 * (i + 1) if i % 2 == 0 else 0 for i in range(5)]
    
    # Generate consumption data for multiple years
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        for year in range(2024, 2027):
            # Baseline consumption increases each year and for each home
            data[f'baseline_{year}_{category}_consumption'] = [
                1000 + (home_idx * 100) + ((year - 2024) * 50) 
                for home_idx in range(5)
            ]
            # Measure package consumption is 70% of baseline (30% savings)
            data[f'mp8_{year}_{category}_consumption'] = [
                int(data[f'baseline_{year}_{category}_consumption'][home_idx] * 0.7)
                for home_idx in range(5)
            ]
    
    return pd.DataFrame(data)


@pytest.fixture
def edge_case_df() -> pd.DataFrame:
    """
    Generate DataFrame with edge cases for testing validation logic.
    
    This fixture creates test data with various edge cases including:
    - Empty strings and null values
    - Zero, negative, and extremely large values
    - Mixed data types
    
    Returns:
        pd.DataFrame: DataFrame with edge case data for robust testing
    """
    data = {
        # Metadata columns
        'state': ['CA', 'TX', 'NY', 'AK', 'HI'],
        'census_division': ['Pacific', 'West South Central', 'Middle Atlantic', 'Pacific', 'Pacific'],
        
        # Empty or null category values
        'base_heating_fuel': ['Electricity', None, '', np.nan, 'Natural Gas'],
        'heating_type': ['Electricity Baseboard', '', None, np.nan, 'Natural Gas Fuel Furnace'],
        
        # Zero or negative consumption values
        'base_electricity_heating_consumption': [0, -10, np.nan, 100, 200],
        'base_naturalGas_heating_consumption': [100, 0, -20, np.nan, 300],
        'baseline_heating_consumption': [100, -10, -20, 100, 500],
        
        # Include flags with mixed types and values
        'include_heating': [True, False, None, np.nan, 'invalid'],
        'include_waterHeating': [1, 0, True, False, np.nan],
        'include_clothesDrying': [True, True, True, True, True],
        'include_cooking': [False, False, False, False, False],
        
        # Retrofit flags with mixed values
        'upgrade_hvac_heating_efficiency': ['ASHP', 123, None, True, 'String']
    }
    
    return pd.DataFrame(data)


@pytest.fixture(params=['heating', 'waterHeating', 'clothesDrying', 'cooking'])
def category(request) -> str:
    """
    Parametrized fixture for equipment categories.
    
    This fixture provides all equipment categories in sequence, allowing
    tests to be run against each category without code duplication.
    
    Args:
        request: Pytest request object containing the parameter
        
    Returns:
        str: Equipment category name
    """
    return request.param


@pytest.fixture(params=[0, 8])
def menu_mp(request) -> int:
    """
    Parametrized fixture for measure package values.
    
    This fixture provides both baseline (0) and measure package (8) values
    to ensure tests work with both scenarios.
    
    Args:
        request: Pytest request object containing the parameter
        
    Returns:
        int: Measure package identifier
    """
    return request.param


@pytest.fixture
def test_utils() -> Dict[str, Callable]:
    """
    Utility functions for common test operations.
    
    These functions simplify repetitive test operations and promote code reuse 
    across multiple test functions.
    
    Returns:
        Dict[str, Callable]: Dictionary of utility functions for test validation
    """
    def create_test_columns(df: pd.DataFrame, category: str, prefix: str) -> Dict[str, pd.Series]:
        """
        Create test columns for validation testing.
        
        Args:
            df: Source DataFrame
            category: Equipment category
            prefix: Prefix for column names
        
        Returns:
            Dictionary of test columns
        """
        valid_mask = df[f'include_{category}']
        result = {}
        
        # Create test calculated columns
        result[f'{prefix}{category}_test_value'] = create_retrofit_only_series(df, valid_mask)
        result[f'{prefix}{category}_capital_cost'] = create_retrofit_only_series(df, valid_mask)
        result[f'{prefix}{category}_savings'] = create_retrofit_only_series(df, valid_mask)
        
        return result
    
    def verify_valid_only_values(df: pd.DataFrame, column: str, valid_mask: pd.Series) -> bool:
        """
        Verify that only valid homes have non-NaN values.
        
        Args:
            df: DataFrame to check
            column: Column name to verify
            valid_mask: Mask identifying valid homes
        
        Returns:
            True if validation passes, False otherwise
        """
        if column not in df.columns:
            return False
        
        # Valid homes should have non-NaN values
        valid_homes_valid = df.loc[valid_mask, column].notna().all()
        
        # Invalid homes should have NaN values
        invalid_homes_valid = df.loc[~valid_mask, column].isna().all()
        
        return valid_homes_valid and invalid_homes_valid
    
    return {
        'create_test_columns': create_test_columns,
        'verify_valid_only_values': verify_valid_only_values
    }


# -------------------------------------------------------------------------
#              STEP 1: MASK INITIALIZATION TESTS
# -------------------------------------------------------------------------

def test_mask_initialization_basic(sample_homes_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test basic initialization of validation tracking.
    
    This test verifies that initialize_validation_tracking correctly:
    1. Creates a valid_mask based on the inclusion flag
    2. Initializes tracking dictionaries properly
    3. Returns all expected components in the correct structure
    
    Args:
        sample_homes_df: Fixture providing test data
        monkeypatch: Pytest fixture for patching functions
    """
    # Test parameters
    category = 'heating'
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
    
    # Verify valid_mask matches include_category column
    assert valid_mask.equals(sample_homes_df[f'include_{category}']), \
        f"Valid mask should exactly match the include_{category} column"
    
    # Verify DataFrame copying works correctly
    assert isinstance(df_copy, pd.DataFrame), "df_copy should be a DataFrame"
    assert len(df_copy) == len(sample_homes_df), "df_copy should have same length as input"
    assert df_copy is not sample_homes_df, "df_copy should be a new object, not the original"
    
    # Verify tracking dictionaries are initialized properly
    assert category in all_columns_to_mask, \
        f"Category '{category}' should be in all_columns_to_mask dictionary"
    
    assert isinstance(all_columns_to_mask[category], list), \
        f"all_columns_to_mask[{category}] should be a list"
    
    # Verify other categories are also initialized
    for cat in EQUIPMENT_SPECS.keys():
        assert cat in all_columns_to_mask, f"Category '{cat}' should be in all_columns_to_mask"
    
    # Verify category-specific tracking is empty initially
    assert len(category_columns_to_mask) == 0, \
        "category_columns_to_mask should be an empty list initially"


def test_mask_initialization_different_categories(sample_homes_df: pd.DataFrame, category: str, 
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test initialization with different equipment categories.
    
    This parametrized test verifies that initialize_validation_tracking works 
    correctly for all equipment categories, using the correct inclusion flag.
    
    Args:
        sample_homes_df: Fixture providing test data
        category: Parametrized fixture providing different categories
        monkeypatch: Pytest fixture for patching functions
    """
    # Test parameters
    menu_mp = 8
    
    # Mock get_valid_calculation_mask for predictable behavior
    def mock_get_valid_mask(df: pd.DataFrame, cat: str, mp: Union[int, str], verbose: bool = True) -> pd.Series:
        """Return a simple mask based on include column."""
        return df[f'include_{cat}']
        
    monkeypatch.setattr(
        'cmu_tare_model.utils.validation_framework.get_valid_calculation_mask',
        mock_get_valid_mask
    )
    
    # Call initialize_validation_tracking for this category
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        sample_homes_df, category, menu_mp, verbose=False)
    
    # Verify valid_mask matches the specific category's include column
    assert valid_mask.equals(sample_homes_df[f'include_{category}']), \
        f"Valid mask should match include_{category} column"
    
    # Verify all categories are initialized in the tracking dictionary
    for cat in EQUIPMENT_SPECS.keys():
        assert cat in all_columns_to_mask, f"Category '{cat}' should be in all_columns_to_mask"


def test_mask_initialization_missing_include_flag(sample_homes_df: pd.DataFrame) -> None:
    """
    Test initialization with missing include flag column.
    
    This test verifies that an appropriate error is raised when the 
    required inclusion flag column is missing from the DataFrame.
    
    Args:
        sample_homes_df: Fixture providing test data
    """
    # Create a copy without the include_heating column
    df_modified = sample_homes_df.copy()
    df_modified = df_modified.drop(columns=['include_heating'])
    
    # Call initialize_validation_tracking and expect an error
    with pytest.raises(ValueError) as excinfo:
        initialize_validation_tracking(df_modified, 'heating', 8, verbose=False)
    
    # Verify error message contains useful information
    error_msg = str(excinfo.value)
    assert "include_heating" in error_msg or "Inclusion flag" in error_msg, \
        "Error message should mention the missing include flag"


def test_mask_initialization_empty_dataframe() -> None:
    """
    Test initialization with an empty DataFrame.
    
    This test verifies that initialize_validation_tracking raises an appropriate
    error when provided with an empty DataFrame, as required columns will be missing.
    """
    # Create an empty DataFrame
    empty_df = pd.DataFrame()
    
    # Test parameters
    category = 'heating'
    menu_mp = 8
    
    # Call initialize_validation_tracking and expect an error
    with pytest.raises((ValueError, KeyError)) as excinfo:
        initialize_validation_tracking(empty_df, category, menu_mp, verbose=False)
    
    # Verify error message mentions missing column
    error_msg = str(excinfo.value)
    assert "include_heating" in error_msg or "column" in error_msg, \
        "Error message should mention missing include column"


def test_retrofit_status_integration(sample_homes_df: pd.DataFrame) -> None:
    """
    Test integration with retrofit status tracking.
    
    This test verifies that the get_valid_calculation_mask function correctly:
    1. Combines data validation and retrofit status for measure packages
    2. Uses only data validation for baseline scenarios
    3. Returns the expected mask
    
    Args:
        sample_homes_df: Fixture providing test data
    """
    # Test cases: one baseline and one measure package
    test_cases = [
        {'category': 'heating', 'menu_mp': 0, 'expected_mask': sample_homes_df['include_heating']},
        {'category': 'heating', 'menu_mp': 8, 'expected_mask': 
            sample_homes_df['include_heating'] & sample_homes_df['upgrade_hvac_heating_efficiency'].notna()}
    ]
    
    for tc in test_cases:
        # Call the function
        result_mask = get_valid_calculation_mask(
            sample_homes_df, tc['category'], tc['menu_mp'], verbose=False)
        
        # For baseline, should only use data validation
        if tc['menu_mp'] == 0:
            pd.testing.assert_series_equal(result_mask, tc['expected_mask'], 
                check_names=False, check_index=True, check_dtype=False)
        
        # For measure packages, check retrofit integration
        else:
            # Retrofit mask should combine validation and retrofit status
            expected_mask = tc['expected_mask']
            
            # Check that all homes flagged as valid have both valid data and scheduled retrofits
            for idx, is_valid in result_mask.items():
                expected_valid = expected_mask.loc[idx]
                assert is_valid == expected_valid, \
                    f"Mask mismatch at index {idx}: got {is_valid}, expected {expected_valid}"


# -------------------------------------------------------------------------
#              STEP 2: SERIES INITIALIZATION TESTS
# -------------------------------------------------------------------------

def test_series_initialization_basic(sample_homes_df: pd.DataFrame, category: str) -> None:
    """
    Test basic creation of retrofit-only series.
    
    This test verifies that create_retrofit_only_series correctly:
    1. Creates a Series with zeros for valid homes
    2. Sets NaN for invalid homes
    3. Returns a Series with the correct index and values
    
    Args:
        sample_homes_df: Fixture providing test data
        category: Parametrized fixture providing different categories
    """
    # Get the valid mask from include_X column
    valid_mask = sample_homes_df[f'include_{category}']
    
    # Call create_retrofit_only_series
    result = create_retrofit_only_series(sample_homes_df, valid_mask)
    
    # Verify structure and type
    assert isinstance(result, pd.Series), "Result should be a pandas Series"
    assert len(result) == len(sample_homes_df), "Result should have same length as DataFrame"
    assert result.index.equals(sample_homes_df.index), "Result should have same index as DataFrame"
    
    # Check valid homes have zeros
    valid_indices = valid_mask[valid_mask].index
    if len(valid_indices) > 0:
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
    
    This test verifies that when all homes are valid, all get zeros
    and no homes get NaN values.
    
    Args:
        sample_homes_df: Fixture providing test data
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
    
    This test verifies that when all homes are invalid, all get NaN values
    and no homes get zero values.
    
    Args:
        sample_homes_df: Fixture providing test data
    """
    # Create mask with all homes invalid
    all_invalid = pd.Series(False, index=sample_homes_df.index)
    
    # Call create_retrofit_only_series
    result = create_retrofit_only_series(sample_homes_df, all_invalid)
    
    # Check all homes have NaN
    assert result.isna().all(), "All homes should have NaN values"
    assert not (result == 0.0).any(), "No homes should have zero values"


def test_series_initialization_empty_dataframe() -> None:
    """
    Test series initialization with empty DataFrame.
    
    This test verifies that create_retrofit_only_series handles empty
    DataFrames gracefully, returning an empty Series with the correct type.
    """
    # Create an empty DataFrame
    empty_df = pd.DataFrame()
    empty_mask = pd.Series(dtype=bool)
    
    # Call create_retrofit_only_series
    result = create_retrofit_only_series(empty_df, empty_mask)
    
    # Verify result is an empty Series
    assert isinstance(result, pd.Series), "Result should be a pandas Series"
    assert len(result) == 0, "Result should be empty for empty DataFrame"


def test_series_initialization_non_standard_index() -> None:
    """
    Test series initialization with non-standard DataFrame index.
    
    This test verifies that create_retrofit_only_series works correctly
    with non-sequential indices, preserving the original index structure.
    """
    # Create DataFrame with non-sequential index
    df = pd.DataFrame({'value': [1, 2, 3]}, index=[100, 200, 300])
    mask = pd.Series([True, False, True], index=[100, 200, 300])
    
    # Call create_retrofit_only_series
    result = create_retrofit_only_series(df, mask)
    
    # Verify index and values
    assert result.index.equals(df.index), "Result should have same index as DataFrame"
    assert result.loc[100] == 0.0, "Valid home at index 100 should have value 0.0"
    assert pd.isna(result.loc[200]), "Invalid home at index 200 should have value NaN"
    assert result.loc[300] == 0.0, "Valid home at index 300 should have value 0.0"


def test_series_initialization_direct_mask_derivation(sample_homes_df: pd.DataFrame) -> None:
    """
    Test using a directly derived mask from a condition.
    
    This test verifies that create_retrofit_only_series works with masks
    derived from conditions rather than inclusion flags.
    
    Args:
        sample_homes_df: Fixture providing test data
    """
    # Create a mask using a condition
    condition_mask = sample_homes_df['base_electricity_heating_consumption'] > 50
    
    # Call create_retrofit_only_series
    result = create_retrofit_only_series(sample_homes_df, condition_mask)
    
    # Check that homes meeting the condition have zeros
    for idx, meets_condition in condition_mask.items():
        if meets_condition:
            assert result[idx] == 0.0, \
                f"Home meeting condition at index {idx} should have value 0.0"
        else:
            assert pd.isna(result[idx]), \
                f"Home not meeting condition at index {idx} should have value NaN"


# -------------------------------------------------------------------------
#              STEP 3: VALID-ONLY CALCULATION TESTS
# -------------------------------------------------------------------------

def test_valid_only_calculation_basic(sample_homes_df: pd.DataFrame) -> None:
    """
    Test basic calculation with valid-only filtering.
    
    This test verifies that calculations can be correctly performed only
    on rows where the valid_mask is True, demonstrating the pattern used
    in step 3 of the validation framework.
    
    Args:
        sample_homes_df: Fixture providing test data
    """
    # Initialize with a simple validity mask
    category = 'heating'
    valid_mask = sample_homes_df[f'include_{category}']
    
    # Initialize result Series with NaN
    result = pd.Series(np.nan, index=sample_homes_df.index)
    
    # Perform a calculation only on valid homes (doubling consumption)
    consumption_col = 'base_electricity_heating_consumption'
    valid_calc_mask = valid_mask & sample_homes_df[consumption_col].notna()
    
    # Step 3: Valid-Only Calculation
    result.loc[valid_calc_mask] = sample_homes_df.loc[valid_calc_mask, consumption_col] * 2
    
    # Verify calculations were only performed on valid homes
    for idx in sample_homes_df.index:
        if valid_calc_mask[idx]:
            expected = sample_homes_df.loc[idx, consumption_col] * 2
            assert result[idx] == expected, \
                f"Valid home at index {idx} should have calculated value {expected}"
        else:
            assert pd.isna(result[idx]), \
                f"Invalid home or home with missing data at index {idx} should have NaN"


def test_valid_only_calculation_compound_mask(sample_homes_df: pd.DataFrame) -> None:
    """
    Test calculation with compound masking conditions.
    
    This test verifies that complex conditions combining multiple masks work
    correctly to filter calculations, which is a common pattern in the validation
    framework.
    
    Args:
        sample_homes_df: Fixture providing test data
    """
    # Initialize validation
    valid_mask = sample_homes_df['include_heating']
    
    # Create a more complex mask combining multiple conditions
    complex_mask = (
        valid_mask & 
        (sample_homes_df['base_electricity_heating_consumption'] > 50) &
        (sample_homes_df['base_heating_fuel'] == 'Electricity')
    )
    
    # Initialize result
    result = pd.Series(np.nan, index=sample_homes_df.index)
    
    # Calculate only for homes that meet all conditions
    result.loc[complex_mask] = 100  # Fixed value for simplicity
    
    # Check that only homes meeting all conditions have values
    assert (result[complex_mask] == 100).all(), \
        "All homes meeting complex conditions should have the calculated value"
        
    # Check that homes not meeting all conditions have NaN
    assert result[~complex_mask].isna().all(), \
        "All homes not meeting complex conditions should have NaN"
    
    # Verify the number of homes that match
    matching_count = complex_mask.sum()
    assert matching_count < len(sample_homes_df), \
        "Not all homes should match the complex conditions"


def test_valid_only_calculation_edge_cases(edge_case_df: pd.DataFrame) -> None:
    """
    Test calculation with edge cases like zero or negative values.
    
    This test verifies that the framework correctly handles zero, negative,
    and NaN values during calculation, with different handling logic for
    different cases.
    
    Args:
        edge_case_df: Fixture providing edge case data
    """
    # Create a simple valid mask
    valid_mask = pd.Series(True, index=edge_case_df.index)
    
    # Create a test column with various edge cases
    edge_case_df['test_values'] = [100, 0, -50, np.nan, 200]
    
    # Create masks for different edge cases
    zero_mask = valid_mask & (edge_case_df['test_values'] == 0)
    negative_mask = valid_mask & (edge_case_df['test_values'] < 0)
    nan_mask = valid_mask & edge_case_df['test_values'].isna()
    positive_mask = valid_mask & (edge_case_df['test_values'] > 0)
    
    # Initialize result
    result = pd.Series(np.nan, index=edge_case_df.index)
    
    # Apply different calculations for each case
    # Standard calculation for positive values
    result.loc[positive_mask] = edge_case_df.loc[positive_mask, 'test_values'] * 2
    
    # Special handling for zero values
    result.loc[zero_mask] = 0
    
    # Special handling for negative values (convert to positive)
    result.loc[negative_mask] = edge_case_df.loc[negative_mask, 'test_values'].abs()
    
    # NaN values remain NaN
    
    # Verify each case
    for idx in edge_case_df.index:
        val = edge_case_df.loc[idx, 'test_values']
        
        if pd.isna(val):
            assert pd.isna(result[idx]), f"NaN value at index {idx} should remain NaN"
        elif val == 0:
            assert result[idx] == 0, f"Zero value at index {idx} should remain 0"
        elif val < 0:
            assert result[idx] == abs(val), f"Negative value {val} at index {idx} should become {abs(val)}"
        else:  # val > 0
            assert result[idx] == val * 2, f"Positive value {val} at index {idx} should become {val * 2}"


def test_valid_only_calculation_multi_column(sample_homes_df: pd.DataFrame) -> None:
    """
    Test calculations involving multiple columns.
    
    This test verifies that valid-only calculations work correctly when
    combining data from multiple columns, which is common in real-world
    calculations.
    
    Args:
        sample_homes_df: Fixture providing test data
    """
    # Initialize validation
    category = 'heating'
    valid_mask = sample_homes_df[f'include_{category}']
    
    # Create a calculation that combines multiple consumption columns
    result = pd.Series(np.nan, index=sample_homes_df.index)
    
    # Get fuel consumption columns
    fuel_columns = [
        'base_electricity_heating_consumption',
        'base_naturalGas_heating_consumption',
        'base_propane_heating_consumption',
        'base_fuelOil_heating_consumption'
    ]
    
    # Calculate only for valid homes using vectorized operations
    # We'll sum all fuel types for total consumption
    valid_calc_mask = valid_mask
    result.loc[valid_calc_mask] = sample_homes_df.loc[valid_calc_mask, fuel_columns].sum(axis=1)
    
    # Verify calculations were only performed on valid homes
    for idx in sample_homes_df.index:
        if valid_calc_mask[idx]:
            expected = sum(sample_homes_df.loc[idx, col] for col in fuel_columns)
            assert result[idx] == expected, \
                f"Valid home at index {idx} should have calculated value {expected}"
        else:
            assert pd.isna(result[idx]), \
                f"Invalid home at index {idx} should have NaN"


# -------------------------------------------------------------------------
#              STEP 4: VALID-ONLY UPDATES TESTS
# -------------------------------------------------------------------------

def test_list_based_collection(sample_homes_df: pd.DataFrame) -> None:
    """
    Test list-based collection pattern for updates.
    
    This test verifies the list-based collection pattern for updates,
    which is more efficient than incremental DataFrame updates.
    
    Args:
        sample_homes_df: Fixture providing test data
    """
    # Initialize validation
    category = 'heating'
    valid_mask = sample_homes_df[f'include_{category}']
    
    # Create a list to store calculation results
    yearly_values = []
    
    # Generate multiple years of calculations
    for year in range(2024, 2027):
        # Create a yearly calculation
        year_values = pd.Series(np.nan, index=sample_homes_df.index)
        
        # Calculate only for valid homes
        year_values.loc[valid_mask] = year - 2023  # Simple year-based value
        
        # Add to collection
        yearly_values.append(year_values)
    
    # Verify list contains the correct number of Series
    assert len(yearly_values) == 3, "List should contain 3 Series (one for each year)"
    
    # Verify each Series in the list has correct values
    for i, year_series in enumerate(yearly_values):
        year_value = i + 1  # 1, 2, 3 for years 2024, 2025, 2026
        
        # Valid homes should have the year value
        for idx in valid_mask[valid_mask].index:
            assert year_series[idx] == year_value, \
                f"Valid home at index {idx} for year {2024+i} should have value {year_value}"
        
        # Invalid homes should have NaN
        for idx in valid_mask[~valid_mask].index:
            assert pd.isna(year_series[idx]), \
                f"Invalid home at index {idx} for year {2024+i} should have NaN"
    
    # Now combine the list into a DataFrame and verify
    yearly_df = pd.concat(yearly_values, axis=1)
    yearly_df.columns = [f'year_{2024+i}' for i in range(len(yearly_values))]
    
    # Calculate total - explicitly propagate NaN values
    total_series = yearly_df.sum(axis=1, skipna=False)  # Use skipna=False to propagate NaN

    # Verify total has values for valid homes, NaN for invalid homes
    valid_sum = 1 + 2 + 3  # Sum of year values
    for idx in sample_homes_df.index:
        if valid_mask[idx]:
            assert total_series[idx] == valid_sum, \
                f"Valid home at index {idx} should have total value {valid_sum}"
        else:
            assert pd.isna(total_series[idx]), \
                f"Invalid home at index {idx} should have NaN for total"


def test_calculate_avoided_values_utility(sample_homes_df: pd.DataFrame) -> None:
    """
    Test the calculate_avoided_values utility function.
    
    This test verifies that the calculate_avoided_values function correctly:
    1. Calculates avoided values (baseline - measure) only for valid homes
    2. Sets NaN for invalid homes
    3. Handles edge cases appropriately
    
    Args:
        sample_homes_df: Fixture providing test data
    """
    # Create sample values
    category = 'heating'
    valid_mask = sample_homes_df[f'include_{category}']
    baseline = pd.Series([100, 200, 300, 400, 500], index=sample_homes_df.index)
    measure = pd.Series([50, 100, 150, 200, 250], index=sample_homes_df.index)
    
    # Calculate avoided values
    avoided = calculate_avoided_values(baseline, measure, valid_mask)
    
    # Verify results
    for idx in sample_homes_df.index:
        if valid_mask[idx]:
            expected = baseline[idx] - measure[idx]
            assert avoided[idx] == expected, \
                f"Valid home at index {idx} should have avoided value {expected}"
        else:
            assert pd.isna(avoided[idx]), \
                f"Invalid home at index {idx} should have NaN for avoided value"


def test_replace_small_values_utility() -> None:
    """
    Test the replace_small_values_with_nan utility function.
    
    This test verifies that the replace_small_values_with_nan function correctly:
    1. Replaces values close to zero with NaN
    2. Preserves larger values
    3. Works with different input types (Series, DataFrame)
    """
    # Test with Series
    series = pd.Series([1.0, 1e-12, -1e-11, 0.1, -0.2])
    result_series = replace_small_values_with_nan(series)
    
    # Values below threshold should be NaN
    assert pd.isna(result_series[1]), "Value 1e-12 should be replaced with NaN"
    assert pd.isna(result_series[2]), "Value -1e-11 should be replaced with NaN"
    
    # Larger values should be preserved
    assert result_series[0] == 1.0, "Value 1.0 should be preserved"
    assert result_series[3] == 0.1, "Value 0.1 should be preserved"
    assert result_series[4] == -0.2, "Value -0.2 should be preserved"
    
    # Test with DataFrame
    df = pd.DataFrame({
        'col1': [1.0, 1e-12, -1e-11, 0.1, -0.2],
        'col2': [0.5, -1e-9, 2e-8, -0.3, 0.7]
    })
    result_df = replace_small_values_with_nan(df)
    
    # Check values in first column
    assert pd.isna(result_df.loc[1, 'col1']), "Value 1e-12 in col1 should be NaN"
    assert pd.isna(result_df.loc[2, 'col1']), "Value -1e-11 in col1 should be NaN"
    
    # Check values in second column - using a higher threshold so these should NOT be NaN
    assert not pd.isna(result_df.loc[1, 'col2']), "Value -1e-9 in col2 should NOT be NaN with default threshold"
    assert not pd.isna(result_df.loc[2, 'col2']), "Value 2e-8 in col2 should NOT be NaN with default threshold"


def test_replace_small_values_with_custom_threshold() -> None:
    """
    Test replace_small_values_with_nan with a custom threshold.
    
    This test verifies that replace_small_values_with_nan correctly:
    1. Uses the provided custom threshold
    2. Replaces values below threshold with NaN
    3. Preserves values above threshold
    """
    # Create a Series with values of different magnitudes
    series = pd.Series([1.0, 1e-3, 1e-6, 1e-9, 0.0])
    
    # Test with default threshold (1e-10)
    result_default = replace_small_values_with_nan(series)
    
    # Values below default threshold should be NaN
    assert not pd.isna(result_default[0]), "1.0 should remain unchanged with default threshold"
    assert not pd.isna(result_default[1]), "1e-3 should remain unchanged with default threshold"
    assert not pd.isna(result_default[2]), "1e-6 should remain unchanged with default threshold"
    assert not pd.isna(result_default[3]), "1e-9 should NOT be NaN with default threshold"
    assert pd.isna(result_default[4]), "0.0 should be NaN with default threshold"
    
    # Test with custom threshold (1e-5)
    result_custom = replace_small_values_with_nan(series, threshold=1e-5)
    
    # Values below custom threshold should be NaN
    assert not pd.isna(result_custom[0]), "1.0 should remain unchanged with custom threshold"
    assert not pd.isna(result_custom[1]), "1e-3 should remain unchanged with custom threshold"
    assert pd.isna(result_custom[2]), "1e-6 should be NaN with custom threshold 1e-5"
    assert pd.isna(result_custom[3]), "1e-9 should be NaN with custom threshold 1e-5"
    assert pd.isna(result_custom[4]), "0.0 should be NaN with custom threshold 1e-5"


def test_list_based_collection_with_nan_propagation(sample_homes_df: pd.DataFrame) -> None:
    """
    Test list-based collection with NaN propagation.
    
    This test verifies that:
    1. NaNs are properly propagated when summing across years
    2. If any year has NaN for a home, the total will also be NaN
    3. This ensures consistent masking of invalid homes
    """
    # Initialize validation
    category = 'heating'
    valid_mask = sample_homes_df[f'include_{category}']
    
    # Create yearly values with some NaNs even for valid homes
    yearly_values = []
    for year in range(2024, 2027):
        year_values = pd.Series(np.nan, index=sample_homes_df.index)
        
        # For year 2025, leave NaN for some valid homes
        if year == 2025:
            # Set values for only half of valid homes
            valid_indices = valid_mask[valid_mask].index
            half_indices = valid_indices[:len(valid_indices)//2]
            year_values.loc[half_indices] = year - 2023
        else:
            # Set values for all valid homes
            year_values.loc[valid_mask] = year - 2023
        
        # Add to collection
        yearly_values.append(year_values)
    
    # Combine into DataFrame and calculate totals
    yearly_df = pd.concat(yearly_values, axis=1)
    yearly_df.columns = [f'year_{year}' for year in range(2024, 2027)]
    
    # Calculate total with and without NaN propagation
    total_with_propagation = yearly_df.sum(axis=1, skipna=False)
    total_without_propagation = yearly_df.sum(axis=1, skipna=True)
    
    # Verify valid homes with NaN in any year
    for idx in valid_mask[valid_mask].index:
        # Check which case we expect
        valid_indices = valid_mask[valid_mask].index
        half_indices = valid_indices[:len(valid_indices)//2]
        
        if idx not in half_indices:  # Homes with NaN in year 2025
            # These homes should have NaN total with propagation
            assert pd.isna(total_with_propagation[idx]), \
                f"Valid home at index {idx} should have NaN with propagation"
            
            # Without propagation, they should have a value (sum of other years)
            assert not pd.isna(total_without_propagation[idx]), \
                f"Valid home at index {idx} should have value without propagation"
            

def test_validation_framework_with_different_menu_mp_formats(sample_homes_df: pd.DataFrame) -> None:
    """
    Test validation framework with different menu_mp format strings.
    
    This test verifies that the validation framework correctly handles:
    1. Different string formats for menu_mp parameter
    2. Case-insensitive baseline indicators ("0", "baseline", "BASELINE")
    3. Non-baseline formats (1, "8", "mp8", etc.)
    """
    category = 'heating'
    include_col = f'include_{category}'
    
    # Test with different menu_mp formats that should be treated as baseline
    baseline_formats = [0, "0", "baseline", "BASELINE", "Baseline"]
    
    for menu_mp in baseline_formats:
        valid_mask = get_valid_calculation_mask(sample_homes_df, category, menu_mp, verbose=False)
        assert valid_mask.equals(sample_homes_df[include_col]), \
            f"For baseline menu_mp={menu_mp}, mask should match include column"
    
    # Test with non-baseline formats
    non_baseline_formats = [1, "1", 8, "8", "mp8", "upgrade08"]
    
    for menu_mp in non_baseline_formats:
        # Get valid mask
        valid_mask = get_valid_calculation_mask(sample_homes_df, category, menu_mp, verbose=False)
        
        # For non-baseline, may combine with retrofit status
        # Just check that it's not identical to the include column if retrofits exist
        upgrade_col = UPGRADE_COLUMNS.get(category)
        if upgrade_col in sample_homes_df.columns and sample_homes_df[upgrade_col].notna().any():
            assert not valid_mask.equals(sample_homes_df[include_col]), \
                f"For non-baseline menu_mp={menu_mp}, mask should consider retrofit status"


def test_apply_new_columns_utility(sample_homes_df: pd.DataFrame) -> None:
    """
    Test the apply_new_columns_to_dataframe utility function.
    
    This test verifies that the apply_new_columns_to_dataframe function correctly:
    1. Tracks columns for validation
    2. Handles overlapping columns
    3. Joins new columns to the original DataFrame
    
    Args:
        sample_homes_df: Fixture providing test data
    """
    # Set up test data
    category = 'heating'
    category_columns_to_mask = []
    all_columns_to_mask = {cat: [] for cat in EQUIPMENT_SPECS}
    
    # Create new columns
    new_cols = {
        'test_col1': range(len(sample_homes_df)),
        'test_col2': [i * 10 for i in range(len(sample_homes_df))]
    }
    df_new_columns = pd.DataFrame(new_cols, index=sample_homes_df.index)
    
    # Apply new columns
    df_updated, updated_all_columns = apply_new_columns_to_dataframe(
        df_original=sample_homes_df,
        df_new_columns=df_new_columns,
        category=category,
        category_columns_to_mask=category_columns_to_mask,
        all_columns_to_mask=all_columns_to_mask
    )
    
    # Verify columns were added to the DataFrame
    for col in df_new_columns.columns:
        assert col in df_updated.columns, f"Column {col} should be in updated DataFrame"
        
        # Verify values match
        for idx in sample_homes_df.index:
            assert df_updated.loc[idx, col] == df_new_columns.loc[idx, col], \
                f"Value mismatch at index {idx} for column {col}"
    
    # Verify tracking was updated
    assert all(col in updated_all_columns[category] for col in df_new_columns.columns), \
        "All new columns should be tracked in all_columns_to_mask"


# -------------------------------------------------------------------------
#              STEP 5: FINAL MASKING TESTS
# -------------------------------------------------------------------------

def test_apply_final_masking_basic(sample_homes_df: pd.DataFrame) -> None:
    """
    Test basic application of final masking.
    
    This test verifies that apply_final_masking correctly:
    1. Masks columns based on category inclusion flags
    2. Sets NaN for invalid homes
    3. Preserves values for valid homes
    
    Args:
        sample_homes_df: Fixture providing test data
    """
    # Create test DataFrame with values for all homes
    df = pd.DataFrame(index=sample_homes_df.index)
    
    # Add some test columns for each category
    all_columns_to_mask = {}
    
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        # Create columns to mask
        column_names = [
            f"baseline_{category}_lifetime_fuelCost",
            f"mp8_{category}_lifetime_fuelCost",
            f"mp8_{category}_lifetime_savings_fuelCost"
        ]
        
        # Add to tracking dictionary
        all_columns_to_mask[category] = column_names
        
        # Add data with non-NaN values for all homes
        for col in column_names:
            df[col] = range(len(df))
    
    # Copy inclusion flags from sample data
    for category in EQUIPMENT_SPECS.keys():
        df[f'include_{category}'] = sample_homes_df[f'include_{category}']
    
    # Apply final masking
    result = apply_final_masking(df, all_columns_to_mask, verbose=False)
    
    # Verify masking for each category
    for category in EQUIPMENT_SPECS.keys():
        mask = df[f'include_{category}']
        
        for col in all_columns_to_mask[category]:
            # Valid homes should keep their values
            valid_indices = mask[mask].index
            if len(valid_indices) > 0:
                assert not result.loc[valid_indices, col].isna().all(), \
                    f"Valid homes should have non-NaN values for {col}"
            
            # Invalid homes should have NaN values
            invalid_indices = mask[~mask].index
            if len(invalid_indices) > 0:
                assert result.loc[invalid_indices, col].isna().all(), \
                    f"Invalid homes should have NaN values for {col}"


def test_apply_final_masking_missing_columns(sample_homes_df: pd.DataFrame) -> None:
    """
    Test masking with non-existent columns in all_columns_to_mask.
    
    This test verifies that apply_final_masking handles missing columns 
    gracefully, without raising errors, while still masking existing columns.
    
    Args:
        sample_homes_df: Fixture providing test data
    """
    # Create test DataFrame
    df = sample_homes_df.copy()
    
    # Add a test column to mask
    df['test_column'] = range(len(df))
    
    # Create dictionary with both existing and non-existent columns
    all_columns_to_mask = {
        'heating': ['test_column', 'non_existent_column']
    }
    
    # Apply masking
    result = apply_final_masking(df, all_columns_to_mask, verbose=False)
    
    # Verify existing column was masked
    valid_mask = df['include_heating']
    
    # Valid homes should keep their values
    valid_indices = valid_mask[valid_mask].index
    if len(valid_indices) > 0:
        assert not result.loc[valid_indices, 'test_column'].isna().all(), \
            "Valid homes should have non-NaN values for existing column"
    
    # Invalid homes should have NaN values
    invalid_indices = valid_mask[~valid_mask].index
    if len(invalid_indices) > 0:
        assert result.loc[invalid_indices, 'test_column'].isna().all(), \
            "Invalid homes should have NaN values for existing column"
    
    # Non-existent column should not be added
    assert 'non_existent_column' not in result.columns, \
        "Non-existent columns should not be added to the DataFrame"


def test_apply_final_masking_empty_dict(sample_homes_df: pd.DataFrame) -> None:
    """
    Test masking with empty dictionary.
    
    This test verifies that apply_final_masking handles empty dictionaries
    gracefully, returning the DataFrame unchanged.
    
    Args:
        sample_homes_df: Fixture providing test data
    """
    # Create a copy to ensure no changes
    df_copy = sample_homes_df.copy()
    
    # Apply masking with empty dictionary
    result = apply_final_masking(df_copy, {}, verbose=False)
    
    # Verify DataFrame is unchanged
    pd.testing.assert_frame_equal(result, df_copy), \
        "DataFrame should be unchanged with empty all_columns_to_mask"


def test_apply_final_masking_multiple_categories(sample_homes_df: pd.DataFrame) -> None:
    """
    Test masking with multiple equipment categories.
    
    This test verifies that apply_final_masking handles multiple categories 
    correctly, applying the appropriate mask for each category.
    
    Args:
        sample_homes_df: Fixture providing test data
    """
    # Create test DataFrame with columns for multiple categories
    df = pd.DataFrame(index=sample_homes_df.index)
    
    # Add validation flags
    for category in EQUIPMENT_SPECS.keys():
        df[f'include_{category}'] = sample_homes_df[f'include_{category}']
    
    # Add test columns for each category
    all_columns_to_mask = {}
    
    for category in EQUIPMENT_SPECS.keys():
        col_name = f'test_value_{category}'
        df[col_name] = range(len(df))
        all_columns_to_mask[category] = [col_name]
    
    # Apply final masking
    result = apply_final_masking(df, all_columns_to_mask, verbose=False)
    
    # Verify each category's columns are masked with its own flag
    for category in EQUIPMENT_SPECS.keys():
        col_name = f'test_value_{category}'
        mask = df[f'include_{category}']
        
        # Valid homes should keep values
        valid_indices = mask[mask].index
        if len(valid_indices) > 0:
            assert not result.loc[valid_indices, col_name].isna().all(), \
                f"Valid {category} homes should have values for {col_name}"
        
        # Invalid homes should have NaN
        invalid_indices = mask[~mask].index
        if len(invalid_indices) > 0:
            assert result.loc[invalid_indices, col_name].isna().all(), \
                f"Invalid {category} homes should have NaN for {col_name}"


# -------------------------------------------------------------------------
#              INTEGRATION TESTS: FULL VALIDATION FRAMEWORK
# -------------------------------------------------------------------------

def test_full_validation_framework(
        sample_homes_df: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Integration test for the full 5-step validation framework.
    
    This test manually performs each step of the validation framework
    in sequence to verify that all steps work together as expected:
    1. Mask Initialization with initialize_validation_tracking()
    2. Series Initialization with create_retrofit_only_series()
    3. Valid-Only Calculation (performs calculations only for valid homes)
    4. Valid-Only Updates (collects results in a list before final assembly)
    5. Final Masking with apply_final_masking()
    
    Args:
        sample_homes_df: Fixture providing test data
        monkeypatch: Pytest fixture for patching functions
    """
    # Track which steps have been executed
    executed_steps = {
        'mask_initialization': False,
        'series_initialization': False,
        'valid_calculation': False,
        'list_accumulation': False,
        'final_masking': False
    }
    
    # Save original functions
    original_init_tracking = initialize_validation_tracking
    original_create_series = create_retrofit_only_series
    original_apply_masking = apply_final_masking
    
    # Define mock functions that track execution
    def mock_init_tracking(*args, **kwargs):
        executed_steps['mask_initialization'] = True
        return original_init_tracking(*args, **kwargs)
    
    def mock_create_series(*args, **kwargs):
        executed_steps['series_initialization'] = True
        return original_create_series(*args, **kwargs)
    
    def mock_apply_masking(*args, **kwargs):
        executed_steps['final_masking'] = True
        return original_apply_masking(*args, **kwargs)
    
    # Import the module directly for patching
    import cmu_tare_model.utils.validation_framework as vf
    
    # Apply monkeypatching
    with monkeypatch.context() as m:
        # Patch the module functions directly
        m.setattr(vf, 'initialize_validation_tracking', mock_init_tracking)
        m.setattr(vf, 'create_retrofit_only_series', mock_create_series)
        m.setattr(vf, 'apply_final_masking', mock_apply_masking)
        
        # Now call through the patched module
        category = 'heating'
        menu_mp = 8
        df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = vf.initialize_validation_tracking(
            sample_homes_df, category, menu_mp, verbose=False)
        
        # Rest of test continues using vf...
        result_series = vf.create_retrofit_only_series(df_copy, valid_mask)
        
        # ===== STEP 3: Calculate values only for valid homes =====
        consumption = df_copy['base_electricity_heating_consumption'].fillna(0)
        factor = 0.5  # Example calculation factor
        
        # Create a compound mask for calculation
        valid_calc_mask = valid_mask & consumption.notna() & (consumption > 0)
        executed_steps['valid_calculation'] = True
        
        # List to collect yearly values (simulating multi-year calculation)
        yearly_values = []
        
        # Perform calculation for year 1
        year1_values = pd.Series(np.nan, index=df_copy.index)
        year1_values.loc[valid_calc_mask] = consumption.loc[valid_calc_mask] * factor
        yearly_values.append(year1_values)
        
        # Perform calculation for year 2
        year2_values = pd.Series(np.nan, index=df_copy.index)
        year2_values.loc[valid_calc_mask] = consumption.loc[valid_calc_mask] * factor * 1.1  # 10% increase
        yearly_values.append(year2_values)
        
        # ===== STEP 4: List-based collection of results =====
        # Combine yearly values into a DataFrame
        yearly_df = pd.concat(yearly_values, axis=1)
        yearly_df.columns = [f'year_{i+1}' for i in range(len(yearly_values))]
        executed_steps['list_accumulation'] = True
        
        # Calculate total (sum across years) with NaN propagation
        total_values = yearly_df.sum(axis=1, skipna=False)
        
        # Create a new DataFrame with the results
        results_df = pd.DataFrame({
            f'test_yearly_year_1': yearly_values[0],
            f'test_yearly_year_2': yearly_values[1],
            f'test_total': total_values
        }, index=df_copy.index)
        
        # Track the columns for masking
        category_columns_to_mask.extend(results_df.columns)
        all_columns_to_mask[category] = category_columns_to_mask
        
        # ===== STEP 5: Apply final masking =====
        # Add required columns before masking
        for cat in EQUIPMENT_SPECS.keys():
            if f'include_{cat}' not in results_df.columns and f'include_{cat}' in df_copy.columns:
                results_df[f'include_{cat}'] = df_copy[f'include_{cat}']
        
        # Use vf.apply_final_masking to ensure the patched version is called
        final_df = vf.apply_final_masking(results_df, all_columns_to_mask, verbose=False)
        
        # Verify all steps were executed
        for step, executed in executed_steps.items():
            assert executed, f"Validation step '{step}' was not executed"
        
        # Verify final result has masked values
        assert 'test_total' in final_df.columns, "Result should contain test_total column"
        
        # Check masking was applied correctly
        valid_indices = valid_mask[valid_mask].index
        invalid_indices = valid_mask[~valid_mask].index
        
        if len(valid_indices) > 0:
            # At least some valid homes should have non-NaN values if they had non-zero consumption
            valid_with_data = valid_calc_mask[valid_calc_mask].index
            if len(valid_with_data) > 0:
                assert not final_df.loc[valid_with_data, 'test_total'].isna().all(), \
                    "Valid homes with data should have non-NaN values"
        
        if len(invalid_indices) > 0:
            assert final_df.loc[invalid_indices, 'test_total'].isna().all(), \
                "Invalid homes should have NaN values"


def test_edge_cases_integration(edge_case_df: pd.DataFrame) -> None:
    """
    Integration test with edge case data.
    
    This test verifies that the validation framework properly handles
    various edge cases, including empty strings, null values, and mixed data types.
    
    Args:
        edge_case_df: Fixture providing edge case data
    """
    # Clean up inclusion flags for testing
    for col in ['include_heating', 'include_waterHeating', 'include_clothesDrying', 'include_cooking']:
        if col in edge_case_df.columns:
            # Convert to boolean, treating None, NaN, and non-boolean as False
            edge_case_df[col] = edge_case_df[col].map(lambda x: x is True or x == 1)
    
    # ===== STEP 1: Initialize validation tracking =====
    # Use heating category for test
    category = 'heating'
    menu_mp = 0  # Baseline for simplicity
    
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        edge_case_df, category, menu_mp, verbose=False)
    
    # ===== STEP 2: Initialize result series =====
    result_series = create_retrofit_only_series(df_copy, valid_mask)
    
    # ===== STEP 3: Calculate values only for valid homes =====
    # Use the consumption column with edge cases
    consumption_col = 'base_electricity_heating_consumption'
    valid_calc_mask = valid_mask & edge_case_df[consumption_col].notna()
    
    # Initialize calculation results
    calc_results = pd.Series(np.nan, index=edge_case_df.index)
    
    # Handle different cases
    # For positive values: multiply by 2
    positive_mask = valid_calc_mask & (edge_case_df[consumption_col] > 0)
    calc_results.loc[positive_mask] = edge_case_df.loc[positive_mask, consumption_col] * 2
    
    # For zero values: set to 0
    zero_mask = valid_calc_mask & (edge_case_df[consumption_col] == 0)
    calc_results.loc[zero_mask] = 0
    
    # For negative values: take absolute value
    negative_mask = valid_calc_mask & (edge_case_df[consumption_col] < 0)
    calc_results.loc[negative_mask] = edge_case_df.loc[negative_mask, consumption_col].abs()
    
    # ===== STEP 4: Create DataFrame with results =====
    results_df = pd.DataFrame({
        'test_calculation': calc_results
    }, index=edge_case_df.index)
    
    # Track the columns for masking
    category_columns_to_mask.append('test_calculation')
    all_columns_to_mask[category] = category_columns_to_mask
    
    # ===== STEP 5: Apply final masking =====
    # Add required columns before masking
    for cat in EQUIPMENT_SPECS.keys():
        if f'include_{cat}' not in results_df.columns and f'include_{cat}' in df_copy.columns:
            results_df[f'include_{cat}'] = df_copy[f'include_{cat}']
    
    final_df = apply_final_masking(results_df, all_columns_to_mask, verbose=False)
    
    # Verify masking was applied correctly
    for idx in edge_case_df.index:
        if valid_mask[idx]:
            # Valid homes with non-null consumption should have calculated values
            if not pd.isna(edge_case_df.loc[idx, consumption_col]):
                consumption_value = edge_case_df.loc[idx, consumption_col]
                if consumption_value > 0:
                    expected = consumption_value * 2
                elif consumption_value == 0:
                    expected = 0
                else:  # consumption_value < 0
                    expected = abs(consumption_value)
                
                assert final_df.loc[idx, 'test_calculation'] == expected, \
                    f"Valid home at index {idx} with consumption {consumption_value} should have calculated value {expected}"
            else:
                # Valid homes with null consumption should have NaN
                assert pd.isna(final_df.loc[idx, 'test_calculation']), \
                    f"Valid home at index {idx} with null consumption should have NaN"
        else:
            # Invalid homes should have NaN
            assert pd.isna(final_df.loc[idx, 'test_calculation']), \
                f"Invalid home at index {idx} should have NaN"


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
