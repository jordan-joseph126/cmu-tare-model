"""
test_calculate_fossil_fuel_emissions_4June2025.py

Simplified pytest test suite for fossil fuel emissions calculations.
Follows validated test patterns with focus on core functionality and integration fixes.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List
from unittest.mock import patch, MagicMock

# Import the module being tested
from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import (
    calculate_fossil_fuel_emissions
)

from cmu_tare_model.constants import EQUIPMENT_SPECS, POLLUTANTS

# =============================================================================
# FIXTURE FUNCTIONS - STREAMLINED APPROACH
# =============================================================================

def create_sample_data():
    """Create minimal sample data for testing fossil fuel emissions."""
    np.random.seed(42)
    n_homes = 15
    
    df = pd.DataFrame({
        'home_id': range(n_homes),
        'census_division': np.random.choice(['Pacific', 'West South Central'], n_homes),
    })
    
    # Add consumption columns for different fuels and categories
    for category in EQUIPMENT_SPECS:
        for fuel in ['naturalGas', 'propane', 'fuelOil']:
            df[f'base_{fuel}_{category}_consumption'] = np.random.uniform(50, 300, n_homes)
    
    return df


def create_mock_emissions_factors():
    """Create simplified mock emission factors."""
    return {
        'naturalGas': {
            'so2': 0.0006, 'nox': 0.0922, 'pm25': 0.0076, 'co2e': 0.228
        },
        'propane': {
            'so2': 0.0002, 'nox': 0.1421, 'pm25': 0.0055, 'co2e': 0.276
        },
        'fuelOil': {
            'so2': 0.0015, 'nox': 0.1300, 'pm25': 0.0027, 'co2e': 0.304
        }
    }


@pytest.fixture
def sample_data():
    """Fixture providing sample data."""
    return create_sample_data()


@pytest.fixture
def mock_emissions_factors():
    """Fixture providing mock emission factors."""
    return create_mock_emissions_factors()


@pytest.fixture
def mock_hdd_functions():
    """Fixture to mock HDD utility functions."""
    with patch('cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions.get_hdd_factor_for_year') as mock_hdd_factor, \
         patch('cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions.apply_hdd_adjustment') as mock_hdd_adjust:
        
        # Mock HDD factor as Series of 1.1 (10% increase)
        mock_hdd_factor.return_value = pd.Series(1.1, index=range(15))
        
        # Mock HDD adjustment - only affects heating, others unchanged
        def mock_adjustment(consumption, category, hdd_factor):
            if category == 'heating':
                return consumption * hdd_factor
            return consumption
        mock_hdd_adjust.side_effect = mock_adjustment
        
        yield mock_hdd_factor, mock_hdd_adjust


# =============================================================================
# CORE FUNCTIONALITY TESTS - INTEGRATION FOCUS
# =============================================================================

def test_year_label_parameter_integration(sample_data, mock_emissions_factors, mock_hdd_functions):
    """
    Test that year_label parameter is properly integrated.
    
    This tests the critical fix where year_label parameter was added to resolve
    integration issues with climate and health modules.
    """
    df = sample_data
    category = 'heating'
    year_label = 2024
    menu_mp = 0  # Baseline only
    
    # Create simple retrofit mask
    retrofit_mask = pd.Series(True, index=df.index)
    
    # Call function with year_label parameter
    result = calculate_fossil_fuel_emissions(
        df=df,
        category=category,
        year_label=year_label,  # Critical parameter that was missing
        lookup_emissions_fossil_fuel=mock_emissions_factors,
        menu_mp=menu_mp,
        retrofit_mask=retrofit_mask,
        verbose=False
    )
    
    # Verify output structure
    assert isinstance(result, dict), "Should return dictionary"
    assert set(result.keys()) == set(POLLUTANTS), "Should have all pollutants"
    
    # Verify all results are Series with correct length
    for pollutant, series in result.items():
        assert isinstance(series, pd.Series), f"{pollutant} should be Series"
        assert len(series) == len(df), f"{pollutant} should match DataFrame length"


def test_hdd_integration_fix(sample_data, mock_emissions_factors, mock_hdd_functions):
    """
    Test that HDD utility functions are properly integrated.
    
    This tests the integration of get_hdd_factor_for_year and apply_hdd_adjustment
    instead of using pre-computed HDD factors.
    """
    df = sample_data
    category = 'heating'  # Should get HDD adjustment
    year_label = 2025
    menu_mp = 0
    retrofit_mask = pd.Series(True, index=df.index)
    
    mock_hdd_factor, mock_hdd_adjust = mock_hdd_functions
    
    # Call function
    result = calculate_fossil_fuel_emissions(
        df=df,
        category=category,
        year_label=year_label,
        lookup_emissions_fossil_fuel=mock_emissions_factors,
        menu_mp=menu_mp,
        retrofit_mask=retrofit_mask,
        verbose=False
    )
    
    # Verify HDD functions were called
    mock_hdd_factor.assert_called_once_with(df, year_label)
    
    # Verify HDD adjustment was applied for each fuel
    expected_calls = len(['naturalGas', 'propane', 'fuelOil'])  # fuels for heating
    assert mock_hdd_adjust.call_count == expected_calls, \
        f"HDD adjustment should be called {expected_calls} times"


def test_baseline_only_calculation(sample_data, mock_emissions_factors, mock_hdd_functions):
    """
    Test that emissions are only calculated for baseline (menu_mp=0).
    
    This verifies the core logic that fossil fuel emissions are only
    calculated for baseline scenarios, not measure packages.
    """
    df = sample_data
    category = 'heating'
    year_label = 2024
    retrofit_mask = pd.Series(True, index=df.index)
    
    # Test baseline calculation
    baseline_result = calculate_fossil_fuel_emissions(
        df=df,
        category=category,
        year_label=year_label,
        lookup_emissions_fossil_fuel=mock_emissions_factors,
        menu_mp=0,  # Baseline
        retrofit_mask=retrofit_mask,
        verbose=False
    )
    
    # Should have calculated emissions
    for pollutant in POLLUTANTS:
        assert not baseline_result[pollutant].eq(0).all(), \
            f"Baseline should calculate {pollutant} emissions"
    
    # Test measure package (should return zeros)
    mp_result = calculate_fossil_fuel_emissions(
        df=df,
        category=category,
        year_label=year_label,
        lookup_emissions_fossil_fuel=mock_emissions_factors,
        menu_mp=8,  # Measure package
        retrofit_mask=retrofit_mask,
        verbose=False
    )
    
    # Should return zero emissions for measure packages
    for pollutant in POLLUTANTS:
        assert mp_result[pollutant].eq(0).all(), \
            f"Measure package should have zero {pollutant} emissions"


def test_fuel_category_logic(sample_data, mock_emissions_factors, mock_hdd_functions):
    """
    Test that appropriate fuels are used for different categories.
    
    Cooking and clothes drying exclude fuel oil, while heating and water heating include it.
    """
    df = sample_data
    year_label = 2024
    menu_mp = 0
    retrofit_mask = pd.Series(True, index=df.index)
    
    # Test category that includes fuel oil (heating)
    heating_result = calculate_fossil_fuel_emissions(
        df=df,
        category='heating',
        year_label=year_label,
        lookup_emissions_fossil_fuel=mock_emissions_factors,
        menu_mp=menu_mp,
        retrofit_mask=retrofit_mask
    )
    
    # Test category that excludes fuel oil (cooking)
    cooking_result = calculate_fossil_fuel_emissions(
        df=df,
        category='cooking',
        year_label=year_label,
        lookup_emissions_fossil_fuel=mock_emissions_factors,
        menu_mp=menu_mp,
        retrofit_mask=retrofit_mask
    )
    
    # Both should calculate emissions, but cooking might be different due to fuel exclusion
    for pollutant in POLLUTANTS:
        assert isinstance(heating_result[pollutant], pd.Series), \
            f"Heating should calculate {pollutant}"
        assert isinstance(cooking_result[pollutant], pd.Series), \
            f"Cooking should calculate {pollutant}"


# =============================================================================
# PARAMETRIZED TESTS - FOLLOWING STREAMLINED PATTERNS
# =============================================================================

@pytest.mark.parametrize("category", ['heating', 'waterHeating', 'clothesDrying', 'cooking'])
def test_all_categories_integration(sample_data, mock_emissions_factors, mock_hdd_functions, category):
    """Test emissions calculation for all equipment categories."""
    df = sample_data
    year_label = 2024
    menu_mp = 0
    retrofit_mask = pd.Series(True, index=df.index)
    
    # Should work for all categories
    result = calculate_fossil_fuel_emissions(
        df=df,
        category=category,
        year_label=year_label,
        lookup_emissions_fossil_fuel=mock_emissions_factors,
        menu_mp=menu_mp,
        retrofit_mask=retrofit_mask
    )
    
    # Verify structure for each category
    assert isinstance(result, dict), f"Should return dict for {category}"
    assert set(result.keys()) == set(POLLUTANTS), f"Should have all pollutants for {category}"


@pytest.mark.parametrize("year_label", [2024, 2025, 2030, 2040])
def test_different_years_integration(sample_data, mock_emissions_factors, mock_hdd_functions, year_label):
    """Test that different years work with HDD integration."""
    df = sample_data
    category = 'heating'
    menu_mp = 0
    retrofit_mask = pd.Series(True, index=df.index)
    
    result = calculate_fossil_fuel_emissions(
        df=df,
        category=category,
        year_label=year_label,
        lookup_emissions_fossil_fuel=mock_emissions_factors,
        menu_mp=menu_mp,
        retrofit_mask=retrofit_mask
    )
    
    # Should work for all years
    assert isinstance(result, dict), f"Should work for year {year_label}"
    
    # Verify HDD function was called with correct year
    mock_hdd_factor, _ = mock_hdd_functions
    mock_hdd_factor.assert_called_with(df, year_label)


@pytest.mark.parametrize("menu_mp", [0, 7, 8, 9, 10])
def test_menu_mp_scenarios(sample_data, mock_emissions_factors, mock_hdd_functions, menu_mp):
    """Test different measure package scenarios."""
    df = sample_data
    category = 'heating'
    year_label = 2024
    retrofit_mask = pd.Series(True, index=df.index)
    
    result = calculate_fossil_fuel_emissions(
        df=df,
        category=category,
        year_label=year_label,
        lookup_emissions_fossil_fuel=mock_emissions_factors,
        menu_mp=menu_mp,
        retrofit_mask=retrofit_mask
    )
    
    # Should complete for all menu_mp values
    assert isinstance(result, dict), f"Should work for menu_mp={menu_mp}"
    
    # Check emissions logic
    if menu_mp == 0:
        # Baseline should have emissions
        assert any(result[p].sum() > 0 for p in POLLUTANTS), "Baseline should have emissions"
    else:
        # Measure packages should have zero emissions
        assert all(result[p].eq(0).all() for p in POLLUTANTS), "MP should have zero emissions"


# =============================================================================
# INPUT VALIDATION TESTS - ESSENTIAL ERROR HANDLING
# =============================================================================

def test_invalid_category_error(sample_data, mock_emissions_factors):
    """Test error handling for invalid category."""
    df = sample_data
    
    with pytest.raises(ValueError, match="Invalid category"):
        calculate_fossil_fuel_emissions(
            df=df,
            category='invalid_category',
            year_label=2024,
            lookup_emissions_fossil_fuel=mock_emissions_factors,
            menu_mp=0
        )


def test_invalid_menu_mp_error(sample_data, mock_emissions_factors):
    """Test error handling for invalid menu_mp."""
    df = sample_data
    
    with pytest.raises(ValueError, match="Invalid menu_mp"):
        calculate_fossil_fuel_emissions(
            df=df,
            category='heating',
            year_label=2024,
            lookup_emissions_fossil_fuel=mock_emissions_factors,
            menu_mp=-1  # Invalid negative value
        )


def test_missing_consumption_column_error(mock_emissions_factors, mock_hdd_functions):
    """Test error handling for missing consumption columns."""
    # Create DataFrame missing required consumption columns
    df = pd.DataFrame({
        'census_division': ['Pacific', 'West South Central'],
        # Missing consumption columns
    })
    
    with pytest.raises(KeyError, match="Required column"):
        calculate_fossil_fuel_emissions(
            df=df,
            category='heating',
            year_label=2024,
            lookup_emissions_fossil_fuel=mock_emissions_factors,
            menu_mp=0
        )


# =============================================================================
# INTEGRATION VERIFICATION TESTS
# =============================================================================

def test_retrofit_mask_application(sample_data, mock_emissions_factors, mock_hdd_functions):
    """Test that retrofit mask is properly applied."""
    df = sample_data
    category = 'heating'
    year_label = 2024
    menu_mp = 0
    
    # Create mask where only half the homes are valid
    mid_point = len(df) // 2
    retrofit_mask = pd.Series([True] * mid_point + [False] * (len(df) - mid_point), index=df.index)
    
    result = calculate_fossil_fuel_emissions(
        df=df,
        category=category,
        year_label=year_label,
        lookup_emissions_fossil_fuel=mock_emissions_factors,
        menu_mp=menu_mp,
        retrofit_mask=retrofit_mask
    )
    
    # Check that masking is applied correctly
    for pollutant in POLLUTANTS:
        valid_values = result[pollutant][retrofit_mask]
        invalid_values = result[pollutant][~retrofit_mask]
        
        # Valid homes should have calculated values (not all zero)
        if len(valid_values) > 0:
            assert not valid_values.eq(0).all(), f"Valid homes should have {pollutant} values"
        
        # Invalid homes should have NaN values
        if len(invalid_values) > 0:
            assert invalid_values.isna().all(), f"Invalid homes should have NaN {pollutant} values"


def test_empty_dataframe_handling(mock_emissions_factors, mock_hdd_functions):
    """Test handling of empty DataFrame."""
    # Create empty DataFrame with all required columns
    required_columns = ['home_id', 'census_division']
    
    # Add consumption columns for all categories and fuels
    for category in EQUIPMENT_SPECS:
        for fuel in ['naturalGas', 'propane', 'fuelOil']:
            required_columns.append(f'base_{fuel}_{category}_consumption')
    
    empty_df = pd.DataFrame(columns=required_columns)
    
    # Mock HDD functions to return empty Series
    mock_hdd_factor, mock_hdd_adjust = mock_hdd_functions
    mock_hdd_factor.return_value = pd.Series([], dtype=float)
    mock_hdd_adjust.side_effect = lambda consumption, category, hdd_factor: consumption
    
    result = calculate_fossil_fuel_emissions(
        df=empty_df,
        category='heating',
        year_label=2024,
        lookup_emissions_fossil_fuel=mock_emissions_factors,
        menu_mp=0,
        retrofit_mask=pd.Series([], dtype=bool)
    )
    
    # Should return empty Series for each pollutant
    for pollutant in POLLUTANTS:
        assert isinstance(result[pollutant], pd.Series), f"Should return Series for {pollutant}"
        assert result[pollutant].empty, f"{pollutant} should be empty Series"

if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
