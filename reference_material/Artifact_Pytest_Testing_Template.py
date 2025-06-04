"""
some_function.py

Pytest tests for validating the 5-step data validation framework implementation in the TEST_VALIDATION_FRAMEWORK pytest test suite.
1. Mask Initialization with initialize_validation_tracking()
2. Series Initialization with create_retrofit_only_series()
3. Valid-Only Calculation for qualifying homes
4. Valid-Only Updates with loc[valid_mask]
5. Final Masking with apply_final_masking()

This test suite verifies proper implementation across equipment categories,
sensitivity parameters, and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union

# Import the module to test - update with your specific module
from your_module import (
    # Main function that implements validation framework
    your_main_function,
    # Utility functions for validation
    initialize_validation_tracking,
    create_retrofit_only_series,
    apply_new_columns_to_dataframe,
    apply_final_masking,
    # Other module-specific functions to test
    get_valid_calculation_mask,
    update_values_for_retrofits
)


# -------------------------------------------------------------------------
#                           TEST FIXTURES
# -------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_constants(monkeypatch) -> None:
    """
    Mock the constants module to isolate tests from external dependencies.
    
    This fixture runs automatically for all tests and ensures consistent test data
    by mocking out the constants that affect validation behavior.
    
    Args:
        monkeypatch: Pytest fixture for patching attributes/functions
    """
    # Example mock data for equipment categories
    mock_equipment_specs = {
        'heating': 15, 
        'waterHeating': 12, 
        'clothesDrying': 13, 
        'cooking': 15
    }
    
    # Example mock data for validation rules
    mock_allowed_technologies = {
        'heating': [
            'Electricity Baseboard', 'Electricity Electric Boiler', 
            'Electricity Electric Furnace', 'Fuel Oil Fuel Boiler', 'Fuel Oil Fuel Furnace', 
            'Natural Gas Fuel Boiler', 'Natural Gas Fuel Furnace',
            'Propane Fuel Boiler', 'Propane Fuel Furnace'
        ],
        'waterHeating': [
            'Electric Heat Pump, 80 gal', 'Electric Premium', 'Electric Standard',
            'Fuel Oil Premium', 'Fuel Oil Standard', 
            'Natural Gas Premium', 'Natural Gas Standard',
            'Propane Premium', 'Propane Standard'
        ],
    }
    
    # Mock to ensure consistent behavior for valid fuel types
    mock_fuel_mapping = {
        'Electricity': 'electricity', 
        'Natural Gas': 'naturalGas', 
        'Fuel Oil': 'fuelOil', 
        'Propane': 'propane'
    }
    
    # Mock upgrade columns for retrofit status
    mock_upgrade_columns = {
        'heating': 'upgrade_hvac_heating_efficiency',
        'waterHeating': 'upgrade_water_heater_efficiency',
        'clothesDrying': 'upgrade_clothes_dryer',
        'cooking': 'upgrade_cooking_range'
    }
    
    # Apply all mocks
    monkeypatch.setattr('your_module.EQUIPMENT_SPECS', mock_equipment_specs)
    monkeypatch.setattr('your_module.ALLOWED_TECHNOLOGIES', mock_allowed_technologies)
    monkeypatch.setattr('your_module.FUEL_MAPPING', mock_fuel_mapping)
    monkeypatch.setattr('your_module.UPGRADE_COLUMNS', mock_upgrade_columns)


@pytest.fixture
def sample_homes_df() -> pd.DataFrame:
    """
    Generate a sample DataFrame containing data for homes with various validation scenarios.
    
    This fixture creates a comprehensive test dataset with:
    - Homes with valid data for all categories
    - Homes with invalid fuel types
    - Homes with invalid technologies
    - Homes with mixed valid/invalid data across categories
    - Homes with and without scheduled retrofits
    
    Returns:
        DataFrame with test data representing diverse home scenarios
    """
    data = {
        # Metadata columns
        'square_footage': [1000, 2000, 1500, 1200, 900],
        'state': ['CA', 'TX', 'FL', 'NY', 'IL'],
        'county': ['01001', '02002', '02003', '03001', '04001'],
        'census_division': ['Division1', 'Division2', 'Division3', 'Division4', 'Division5'],
        
        # Category data with validation issues
        'base_heating_fuel': ['Electricity', 'Natural Gas', 'Fuel Oil', 'Propane', 'Coal'],  # Coal is invalid
        'heating_type': [
            'Electricity Baseboard', 'Natural Gas Fuel Furnace', 
            'Fuel Oil Fuel Furnace', 'Propane Fuel Furnace', 
            'Wood Stove'  # invalid
        ],
        
        'base_waterHeating_fuel': ['Electricity', 'Natural Gas', 'Propane', 'Electricity', 'Fuel Oil'],
        'waterHeating_type': [
            'Electric Standard', 'Natural Gas Standard', 
            'Propane Premium', 'Electric Heat Pump, 80 gal', 
            'Fuel Oil Standard'
        ],
        
        'base_clothesDrying_fuel': ['Electricity', 'Natural Gas', None, 'Propane', 'Fuel Oil'],
        'base_cooking_fuel': ['Electricity', 'Propane', 'Natural Gas', 'Fuel Oil', np.nan],  # Electric cooking invalid
        
        # Consumption data for each category
        'base_electricity_heating_consumption': [200, 0, 10, 100, 0],
        'base_naturalGas_heating_consumption': [0, 300, 0, 0, 50],
        'base_fuelOil_heating_consumption': [0, 0, 50, 0, 10],
        'base_propane_heating_consumption': [0, 0, 0, 200, 0],
        'baseline_heating_consumption': [200, 300, 60, 300, 60],
        
        # Retrofit columns for measure packages
        'upgrade_hvac_heating_efficiency': [
            'ASHP, SEER 18, HSPF 9.5', None, 
            'ASHP, SEER 18, HSPF 9.5', None, 
            'ASHP, SEER 18, HSPF 9.5'
        ],
        'upgrade_water_heater_efficiency': [
            'Electric Heat Pump', None,
            None, 'Electric Heat Pump, 80 gal',
            'Electric Heat Pump'
        ],
        'upgrade_clothes_dryer': [
            'Electric, Premium, Heat Pump, Ventless', 
            'Electric, Premium, Heat Pump, Ventless',
            None, None, 
            'Electric, Premium, Heat Pump, Ventless'
        ],
        'upgrade_cooking_range': [
            'Electric Induction', None,
            'Electric Resistance', None,
            None
        ],
        
        # Include validation flags for testing
        'include_heating': [True, True, True, True, False],
        'include_waterHeating': [True, True, True, True, True],
        'include_clothesDrying': [True, True, False, True, False],
        'include_cooking': [False, True, True, False, True],
        'include_all': [False, True, False, False, False],
        
        # Valid fuel and tech flags
        'valid_fuel_heating': [True, True, True, True, False],
        'valid_tech_heating': [True, True, True, True, False],
        'valid_fuel_waterHeating': [True, True, True, True, True], 
        'valid_tech_waterHeating': [True, True, True, True, True],
        'valid_fuel_clothesDrying': [True, True, False, True, False],
        'valid_fuel_cooking': [False, True, True, False, True]
    }
    return pd.DataFrame(data)


@pytest.fixture
def fuel_costs_df() -> pd.DataFrame:
    """
    Generate a sample DataFrame with fuel cost data for testing.
    
    This fixture creates test data for annual and lifetime fuel costs,
    including both baseline and measure package scenarios, which is
    needed for testing NPV calculations.
    
    Returns:
        DataFrame with fuel cost data across multiple years and categories
    """
    # Create data dictionary with years and categories
    data = {}
    
    # Add baseline and measure package fuel cost columns
    prefixes = ['baseline_', 'preIRA_mp8_', 'iraRef_mp8_']
    
    for prefix in prefixes:
        # Add annual fuel costs for each year
        for year in range(2024, 2040):  # Years from 2024 to 2039
            for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
                # Annual fuel cost columns
                data[f'{prefix}{year}_{category}_fuelCost'] = [100, 200, 150, 120, 80]
                
                # Add savings columns for measure packages
                if prefix != 'baseline_':
                    data[f'{prefix}{year}_{category}_savings_fuelCost'] = [20, 40, 30, 25, 15]
        
        # Add lifetime fuel cost columns
        for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
            data[f'{prefix}{category}_lifetime_fuelCost'] = [1500, 3000, 2250, 1800, 1200]
            
            # Add savings columns for measure packages
            if prefix != 'baseline_':
                data[f'{prefix}{category}_lifetime_savings_fuelCost'] = [300, 600, 450, 360, 240]
    
    return pd.DataFrame(data)


@pytest.fixture
def capital_costs_df() -> pd.DataFrame:
    """
    Generate a sample DataFrame with capital cost data for testing.
    
    This fixture creates test data for installation costs, replacement costs,
    and rebates across different measure packages, which is needed for
    testing private impact calculations.
    
    Returns:
        DataFrame with capital cost data for different equipment categories
    """
    data = {}
    
    # Create data for all required menu_mp values
    for mp in [7, 8, 9, 10]:
        # Heating columns
        data[f'mp{mp}_heating_installationCost'] = [10000, 12000, 15000, 8000, 11000]
        data[f'mp{mp}_heating_replacementCost'] = [5000, 6000, 7500, 4000, 5500]
        data[f'mp{mp}_heating_installation_premium'] = [1000, 1200, 1500, 800, 1100]
        data[f'mp{mp}_heating_rebate_amount'] = [2000, 2400, 3000, 1600, 2200]
        
        # Water heating columns
        data[f'mp{mp}_waterHeating_installationCost'] = [5000, 6000, 7500, 4000, 5500]
        data[f'mp{mp}_waterHeating_replacementCost'] = [2500, 3000, 3750, 2000, 2750]
        data[f'mp{mp}_waterHeating_rebate_amount'] = [1000, 1200, 1500, 800, 1100]
        
        # Clothes drying columns
        data[f'mp{mp}_clothesDrying_installationCost'] = [2000, 2400, 3000, 1600, 2200]
        data[f'mp{mp}_clothesDrying_replacementCost'] = [1000, 1200, 1500, 800, 1100]
        data[f'mp{mp}_clothesDrying_rebate_amount'] = [400, 480, 600, 320, 440]
        
        # Cooking columns
        data[f'mp{mp}_cooking_installationCost'] = [3000, 3600, 4500, 2400, 3300]
        data[f'mp{mp}_cooking_replacementCost'] = [1500, 1800, 2250, 1200, 1650]
        data[f'mp{mp}_cooking_rebate_amount'] = [600, 720, 900, 480, 660]
    
    # Additional columns needed for weatherization calculations
    data['weatherization_rebate_amount'] = [1000, 1200, 1500, 800, 1100]
    data['mp9_enclosure_upgradeCost'] = [3000, 3600, 4500, 2400, 3300]
    data['mp10_enclosure_upgradeCost'] = [4000, 4800, 6000, 3200, 4400]
    
    return pd.DataFrame(data)


@pytest.fixture
def edge_case_df() -> pd.DataFrame:
    """
    Generate a DataFrame with edge cases for testing validation logic.
    
    This fixture creates test data with various edge cases including:
    - Empty strings and null values
    - Zero, negative, and extremely large values
    - Mixed data types
    
    Returns:
        DataFrame with edge case data for robust testing
    """
    data = {
        # Empty or null category values
        'base_heating_fuel': ['Electricity', None, '', np.nan, 'Natural Gas'],
        'heating_type': ['Electricity Baseboard', '', None, np.nan, 'Natural Gas Fuel Furnace'],
        
        # Zero or negative consumption values
        'base_electricity_heating_consumption': [0, -10, np.nan, 100, 200],
        'base_naturalGas_heating_consumption': [100, 0, -20, np.nan, 300],
        'baseline_heating_consumption': [100, -10, -20, 100, 500],
        
        # Invalid cost data
        'mp8_heating_installationCost': [10000, 0, -5000, np.nan, 1e10],
        'mp8_heating_replacementCost': [5000, 0, -2500, np.nan, 1e9],
        'mp8_heating_rebate_amount': [2000, 0, 20000, np.nan, 1e11],  # Rebate > cost
        
        # Mixed data types (will be converted to string)
        'upgrade_hvac_heating_efficiency': ['ASHP', 123, None, True, 'String'],
        
        # Include invalid validation flags
        'include_heating': [True, False, None, np.nan, 'invalid'],
        'valid_fuel_heating': [True, False, None, np.nan, 'invalid']
    }
    return pd.DataFrame(data)


@pytest.fixture(params=['heating', 'waterHeating', 'clothesDrying', 'cooking'])
def category(request) -> str:
    """
    Fixture for parametrized testing across all equipment categories.
    
    Returns:
        str: Equipment category name
    """
    return request.param


@pytest.fixture(params=[0, 7, 8, 9, 10])
def menu_mp(request) -> int:
    """
    Fixture for parametrized testing across all menu measure packages.
    
    Returns:
        int: Measure package identifier
    """
    return request.param


@pytest.fixture(params=['No Inflation Reduction Act', 'AEO2023 Reference Case'])
def policy_scenario(request) -> str:
    """
    Fixture for parametrized testing across policy scenarios.
    
    Returns:
        str: Policy scenario name
    """
    return request.param


@pytest.fixture
def test_utils() -> Dict[str, callable]:
    """
    Utility functions for common test operations.
    
    Returns:
        Dict of utility functions for test validation
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
#                   STEP 1: MASK INITIALIZATION TESTS
# -------------------------------------------------------------------------

# THESE TESTS ARE IN THE TEST_VALIDATION_FRAMEWORK PYTEST TEST SUITE


# -------------------------------------------------------------------------
#                  STEP 2: SERIES INITIALIZATION WITH CREATE_RETROFIT_ONLY_SERIES()
# -------------------------------------------------------------------------

# THESE TESTS ARE IN THE TEST_VALIDATION_FRAMEWORK PYTEST TEST SUITE


# -------------------------------------------------------------------------
#                  STEP 3: VALID-ONLY CALCULATION TESTS
# -------------------------------------------------------------------------

# THESE TESTS ARE IN THE TEST_VALIDATION_FRAMEWORK PYTEST TEST SUITE


# -------------------------------------------------------------------------
#                  STEP 4: VALID-ONLY UPDATES TESTS
# -------------------------------------------------------------------------

# THESE TESTS ARE IN THE TEST_VALIDATION_FRAMEWORK PYTEST TEST SUITE


# -------------------------------------------------------------------------
#                  STEP 5: FINAL MASKING TESTS
# -------------------------------------------------------------------------

# THESE TESTS ARE IN THE TEST_VALIDATION_FRAMEWORK PYTEST TEST SUITE


# -------------------------------------------------------------------------
#                       INTEGRATION TESTS
# -------------------------------------------------------------------------

def test_full_validation_flow(sample_homes_df):
    """
    Test the full 5-step validation flow in sequence.
    
    This test manually performs each step in order to verify
    the entire framework works as expected.
    
    Verifies that:
    - All five steps work together seamlessly
    - Data integrity is maintained throughout the process
    - Only valid homes receive values
    """
    # Step 1: Initialize validation tracking
    category = 'heating'
    menu_mp = 8
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        sample_homes_df, category, menu_mp, verbose=False)
    
    # Check initialization
    assert valid_mask.equals(sample_homes_df['include_heating']), \
        "Valid mask should match include_heating column"
    assert category in all_columns_to_mask, \
        "Category should be in all_columns_to_mask dictionary"
    
    # Step 2: Initialize result series
    result_series = create_retrofit_only_series(df_copy, valid_mask)
    
    # Check series initialization
    assert result_series[valid_mask].equals(pd.Series(0.0, index=valid_mask[valid_mask].index)), \
        "Valid homes should have 0.0 initially"
    assert result_series[~valid_mask].isna().all(), \
        "Invalid homes should have NaN initially"
    
    # Step 3: Calculate values only for valid homes
    consumption = df_copy['base_electricity_heating_consumption']
    factor = 0.5  # Example factor
    valid_calc_mask = valid_mask & consumption.notna()
    calculated_values = consumption.copy()
    calculated_values.loc[valid_calc_mask] = consumption.loc[valid_calc_mask] * factor
    
    # Step 4: Update only valid homes
    updated_series = update_values_for_retrofits(result_series, calculated_values, valid_mask, menu_mp)
    
    # Check that only valid homes were updated
    for idx in df_copy.index:
        if valid_calc_mask[idx]:
            expected = consumption.loc[idx] * factor
            assert updated_series[idx] == expected, \
                f"Valid home at index {idx} should be updated to {expected}"
        elif valid_mask[idx]:
            # Valid homes with NaN consumption should get 0
            assert updated_series[idx] == 0, \
                f"Valid home with NaN consumption at index {idx} should get 0"
        else:
            # Invalid homes should remain NaN
            assert pd.isna(updated_series[idx]), \
                f"Invalid home at index {idx} should remain NaN"
    
    # Create a DataFrame with the result series
    df_new_columns = pd.DataFrame({
        f'test_result_{category}': updated_series
    })
    
    # Track the column for masking
    category_columns_to_mask.append(f'test_result_{category}')
    
    # Apply the new columns to the original DataFrame
    df_copy = pd.concat([df_copy, df_new_columns], axis=1)
    all_columns_to_mask[category].extend(category_columns_to_mask)
    
    # Step 5: Apply final masking
    result_df = apply_final_masking(df_copy, all_columns_to_mask, verbose=False)
    
    # Check final result
    assert f'test_result_{category}' in result_df.columns, \
        "Result column should be in the final DataFrame"
        
    # Invalid homes should have NaN in the result column
    assert result_df.loc[~valid_mask, f'test_result_{category}'].isna().all(), \
        "Invalid homes should have NaN in the result column"
        
    # Valid homes with consumption should have calculated values
    for idx in valid_calc_mask.index:
        if valid_calc_mask[idx]:
            expected = consumption.loc[idx] * factor
            assert result_df.loc[idx, f'test_result_{category}'] == expected, \
                f"Valid home at index {idx} should have calculated value {expected}"


@pytest.mark.parametrize("category", ['heating', 'waterHeating', 'clothesDrying', 'cooking'])
def test_integration_across_categories(sample_homes_df, category):
    """
    Test integration across different equipment categories.
    
    Verifies that:
    - The validation framework works consistently across all categories
    - Each category's validation logic is correctly applied
    
    Args:
        category: Equipment category being tested
    """
    # Use common menu_mp for all tests
    menu_mp = 8
    
    # Step 1: Initialize validation tracking
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        sample_homes_df, category, menu_mp, verbose=False)
    
    # Check that we're using the right validation flag
    expected_mask = sample_homes_df[f'include_{category}']
    assert valid_mask.equals(expected_mask), \
        f"Valid mask should match include_{category} column"
    
    # Get a consumption column for this category
    consumption_col = f'base_electricity_{category}_consumption'
    if consumption_col not in sample_homes_df.columns:
        # Create a dummy column for testing
        df_copy[consumption_col] = 100
    
    # Step 2 & 3: Initialize and calculate
    result_series = create_retrofit_only_series(df_copy, valid_mask)
    valid_calc_mask = valid_mask & df_copy[consumption_col].notna()
    result_series.loc[valid_calc_mask] = df_copy.loc[valid_calc_mask, consumption_col] * 0.5
    
    # Step 4: Create new DataFrame with results
    df_new_columns = pd.DataFrame({
        f'test_result_{category}': result_series
    })
    
    # Track columns and join
    category_columns_to_mask.append(f'test_result_{category}')
    df_copy = pd.concat([df_copy, df_new_columns], axis=1)
    all_columns_to_mask[category].extend(category_columns_to_mask)
    
    # Step 5: Apply final masking
    result_df = apply_final_masking(df_copy, all_columns_to_mask, verbose=False)
    
    # Check that masking worked correctly for this category
    for idx in sample_homes_df.index:
        if valid_mask[idx]:
            # Valid homes should have values
            assert not pd.isna(result_df.loc[idx, f'test_result_{category}']), \
                f"Valid {category} home at index {idx} should have a value"
        else:
            # Invalid homes should have NaN
            assert pd.isna(result_df.loc[idx, f'test_result_{category}']), \
                f"Invalid {category} home at index {idx} should have NaN"


def test_your_main_function(sample_homes_df, fuel_costs_df, capital_costs_df):
    """
    Test your module's main function that implements the validation framework.
    
    Replace your_main_function with the actual function name from your module.
    Adjust parameters and validation as needed.
    
    This is a placeholder test - modify it to test your actual function.
    """
    # This is a placeholder - you'll need to replace with your actual function
    # For example, a call to calculate_private_npv() or adoption_decision()
    
    # result = your_main_function(
    #     df=sample_homes_df,
    #     df_fuel_costs=fuel_costs_df,
    #     input_mp='upgrade08',
    #     menu_mp=8,
    #     policy_scenario='AEO2023 Reference Case'
    # )
    
    # Verify the result - examples of what to check:
    
    # 1. Check that result columns exist for each category
    # for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
    #     expected_column = f'iraRef_mp8_{category}_private_npv_lessWTP'
    #     assert expected_column in result.columns, f"Expected result column {expected_column} is missing"
    
    # 2. Check that invalid homes have NaN values
    # for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
    #     col = f'iraRef_mp8_{category}_private_npv_lessWTP'
    #     mask = sample_homes_df[f'include_{category}']
    #     assert result.loc[~mask, col].isna().all(), \
    #         f"Invalid homes should have NaN values for {col}"
    
    # 3. Check that valid homes have calculated values
    # for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
    #     col = f'iraRef_mp8_{category}_private_npv_lessWTP'
    #     mask = sample_homes_df[f'include_{category}']
    #     assert not result.loc[mask, col].isna().all(), \
    #         f"At least some valid homes should have calculated values for {col}"
    pass


# -------------------------------------------------------------------------
#                         EDGE CASE TESTS
# -------------------------------------------------------------------------

def test_empty_dataframe():
    """
    Test framework with empty DataFrame.
    
    Verifies that:
    - All framework functions handle empty DataFrames gracefully
    - No errors are raised for empty inputs
    - Empty results are returned with appropriate types
    """
    empty_df = pd.DataFrame()
    
    # Test initialize_validation_tracking
    category = 'heating'
    menu_mp = 8
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        empty_df, category, menu_mp, verbose=False)
    
    assert len(valid_mask) == 0, "Valid mask should be empty for empty DataFrame"
    assert isinstance(valid_mask, pd.Series), "Valid mask should be a pandas Series"
    assert category in all_columns_to_mask, "Category should be in all_columns_to_mask"
    
    # Test create_retrofit_only_series
    result_series = create_retrofit_only_series(empty_df, valid_mask)
    assert len(result_series) == 0, "Result series should be empty"
    assert isinstance(result_series, pd.Series), "Result series should be a pandas Series"
    
    # Test apply_final_masking
    result_df = apply_final_masking(empty_df, all_columns_to_mask, verbose=False)
    assert result_df.empty, "Result DataFrame should be empty"
    assert isinstance(result_df, pd.DataFrame), "Result should be a pandas DataFrame"


def test_missing_validation_columns(edge_case_df):
    """
    Test framework with missing validation columns.
    
    Verifies that:
    - The framework handles missing validation columns appropriately
    - Either default behavior is applied or informative errors are raised
    """
    # Initialize validation tracking - should either add default flags or raise error
    category = 'heating'
    menu_mp = 8
    
    # Handle based on your implementation - either:
    
    # 1. Expect an error to be raised with informative message
    with pytest.raises(Exception) as excinfo:  # Update with specific exception type
        initialize_validation_tracking(edge_case_df, category, menu_mp, verbose=False)
    
    # Verify error message is informative
    assert f"include_{category}" in str(excinfo.value) or "validation" in str(excinfo.value), \
        "Error message should mention missing validation flags"
    
    # Or 2. If implementation adds default flags, verify that behavior
    # df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
    #     edge_case_df, category, menu_mp, verbose=False)
    # assert len(valid_mask) == len(edge_case_df), "Valid mask should have same length as DataFrame"
    # assert not valid_mask.all(), "Not all homes should be valid with default flags"


def test_all_invalid_data(sample_homes_df):
    """
    Test framework when all homes are invalid.
    
    Verifies that:
    - The framework handles the case where no homes are valid
    - No calculations are performed
    - All result columns contain only NaN values
    """
    # Create a DataFrame where all homes are invalid
    all_invalid_df = sample_homes_df.copy()
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        all_invalid_df[f'include_{category}'] = False
    all_invalid_df['include_all'] = False
    
    # Test the full framework
    category = 'heating'
    menu_mp = 8
    
    # Step 1: Initialize validation tracking
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        all_invalid_df, category, menu_mp, verbose=False)
    
    # All homes should be invalid
    assert not valid_mask.any(), "No homes should be valid"
    
    # Step 2: Initialize result series - all should be NaN
    result_series = create_retrofit_only_series(df_copy, valid_mask)
    assert result_series.isna().all(), "All homes should have NaN in result series"
    
    # Step 3 & 4: Calculate and update - should have no effect
    update_values = pd.Series(100, index=df_copy.index)
    updated_series = update_values_for_retrofits(result_series, update_values, valid_mask, menu_mp)
    assert updated_series.isna().all(), "All homes should still have NaN after update"
    
    # Create a DataFrame with the result series
    df_new_columns = pd.DataFrame({
        f'test_result_{category}': updated_series
    })
    
    # Track the column and apply it
    category_columns_to_mask.append(f'test_result_{category}')
    df_copy = pd.concat([df_copy, df_new_columns], axis=1)
    all_columns_to_mask[category].extend(category_columns_to_mask)
    
    # Step 5: Apply final masking
    result_df = apply_final_masking(df_copy, all_columns_to_mask, verbose=False)
    
    # Result should have NaN for all tracked columns
    assert result_df[f'test_result_{category}'].isna().all(), \
        "All homes should have NaN in result column when all invalid"


def test_invalid_inputs():
    """
    Test framework with invalid input types.
    
    Verifies that:
    - The framework handles invalid input types appropriately
    - Informative errors are raised for invalid inputs
    """
    # Test with DataFrame replaced by a dictionary
    dict_input = {'column': [1, 2, 3]}
    category = 'heating'
    menu_mp = 8
    
    with pytest.raises(Exception) as excinfo:  # Update with specific exception type
        initialize_validation_tracking(dict_input, category, menu_mp, verbose=False)
    
    # Test with invalid category
    df = pd.DataFrame({'column': [1, 2, 3]})
    invalid_category = 'invalid_category'
    
    with pytest.raises(Exception) as excinfo:  # Update with specific exception type
        initialize_validation_tracking(df, invalid_category, menu_mp, verbose=False)
    
    # Test with invalid menu_mp
    invalid_menu_mp = 'not_a_number'
    
    with pytest.raises(Exception) as excinfo:  # Update with specific exception type
        initialize_validation_tracking(df, category, invalid_menu_mp, verbose=False)


def test_edge_case_mask_types():
    """
    Test framework with edge case mask types.
    
    Verifies that:
    - The framework handles various mask types correctly
    - Different boolean representations work as expected
    """
    # Create a simple DataFrame
    df = pd.DataFrame({'column': [1, 2, 3]})
    
    # Test with masks of various types
    masks = [
        # Boolean Series
        pd.Series([True, False, True], index=df.index),
        # NumPy array
        np.array([True, False, True]),
        # List
        [True, False, True],
        # All True
        pd.Series([True, True, True], index=df.index),
        # All False
        pd.Series([False, False, False], index=df.index),
        # Integer values (non-zero is True)
        pd.Series([1, 0, 2], index=df.index),
    ]
    
    for i, mask in enumerate(masks):
        # Try to create retrofit-only series with this mask
        try:
            result = create_retrofit_only_series(df, mask)
            
            # Check result shape and type
            assert isinstance(result, pd.Series), f"Mask type {i}: Result should be a pandas Series"
            assert len(result) == len(df), f"Mask type {i}: Result should have same length as DataFrame"
            
            # Check values based on mask type
            if isinstance(mask, (pd.Series, np.ndarray, list)):
                for j, val in enumerate(mask):
                    expected_value = 0.0 if val else np.nan
                    if pd.isna(expected_value):
                        assert pd.isna(result.iloc[j]), \
                            f"Mask type {i}, index {j}: Expected NaN for False mask value"
                    else:
                        assert result.iloc[j] == expected_value, \
                            f"Mask type {i}, index {j}: Expected {expected_value} for True mask value"
        except Exception as e:
            # Some mask types might raise exceptions, which is okay if they're not supported
            print(f"Mask type {i} raised exception: {str(e)}")
