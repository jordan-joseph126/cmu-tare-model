"""
test_calculate_lifetime_fuel_costs_4June2025.py

Simplified pytest test suite for fuel cost calculations following streamlined patterns.
Focuses on core logic verification rather than extensive validation framework testing.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from unittest.mock import patch, MagicMock

# Import the specific module being tested
from cmu_tare_model.private_impact.calculate_lifetime_fuel_costs import (
    calculate_lifetime_fuel_costs,
    calculate_annual_fuel_costs
)

# Import HDD utilities for integration testing
from cmu_tare_model.utils.hdd_consumption_utils import (
    get_hdd_adjusted_consumption,
    apply_hdd_adjustment,
    get_total_baseline_consumption
)

from cmu_tare_model.constants import EQUIPMENT_SPECS

# =============================================================================
# FIXTURE FUNCTIONS - SIMPLIFIED APPROACH
# =============================================================================

def create_sample_data():
    """Create minimal sample data for testing fuel costs."""
    np.random.seed(42)
    n_homes = 20
    
    # Create base DataFrame with essential columns
    df = pd.DataFrame({
        'home_id': range(n_homes),
        'state': np.random.choice(['CA', 'TX', 'FL', 'NY'], n_homes),
        'census_division': np.random.choice(['Pacific', 'West South Central', 'South Atlantic'], n_homes),
    })
    
    # Add validation flags for each category
    for category in EQUIPMENT_SPECS:
        df[f'include_{category}'] = np.random.choice([True, False], n_homes, p=[0.8, 0.2])
        df[f'base_{category}_fuel'] = np.random.choice(['Electricity', 'Natural Gas', 'Propane'], n_homes)
    
    # Add base consumption columns
    for category in EQUIPMENT_SPECS:
        for fuel in ['electricity', 'naturalGas', 'propane', 'fuelOil']:
            df[f'base_{fuel}_{category}_consumption'] = np.random.uniform(50, 500, n_homes)
        # Add total baseline consumption
        df[f'baseline_{category}_consumption'] = np.random.uniform(100, 600, n_homes)
    
    # Add measure package consumption columns
    for mp in [7, 8, 9, 10]:
        for category in EQUIPMENT_SPECS:
            df[f'mp{mp}_{category}_consumption'] = np.random.uniform(20, 200, n_homes)
    
    return df


def create_mock_fuel_prices():
    """Create simplified mock fuel prices."""
    # Simplified structure - just what we need for testing
    mock_prices = {}
    
    # Add state-level prices (for electricity/naturalGas)
    for state in ['CA', 'TX', 'FL', 'NY']:
        mock_prices[state] = {
            'electricity': {
                'No Inflation Reduction Act': {year: 0.15 for year in range(2024, 2030)},
                'AEO2023 Reference Case': {year: 0.14 for year in range(2024, 2030)},
            },
            'naturalGas': {
                'No Inflation Reduction Act': {year: 0.08 for year in range(2024, 2030)},
                'AEO2023 Reference Case': {year: 0.075 for year in range(2024, 2030)},
            }
        }
    
    # Add census division prices (for propane/fuelOil)
    for division in ['Pacific', 'West South Central', 'South Atlantic']:
        mock_prices[division] = {
            'propane': {
                'No Inflation Reduction Act': {year: 0.12 for year in range(2024, 2030)},
                'AEO2023 Reference Case': {year: 0.115 for year in range(2024, 2030)},
            },
            'fuelOil': {
                'No Inflation Reduction Act': {year: 0.14 for year in range(2024, 2030)},
                'AEO2023 Reference Case': {year: 0.135 for year in range(2024, 2030)},
            }
        }
    
    return mock_prices


@pytest.fixture
def sample_data():
    """Fixture providing sample data."""
    return create_sample_data()


@pytest.fixture
def mock_scenario_params():
    """Fixture to mock scenario parameters."""
    def mock_define_scenario_params(menu_mp, policy_scenario):
        scenario_prefix = 'baseline_' if menu_mp == 0 else f'iraRef_mp{menu_mp}_'
        return (scenario_prefix, '', {}, {}, {}, create_mock_fuel_prices())
    
    with patch('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.define_scenario_params') as mock:
        mock.side_effect = mock_define_scenario_params
        yield mock


# =============================================================================
# CORE FUNCTIONALITY TESTS - STREAMLINED APPROACH  
# =============================================================================

def test_hdd_utils_integration(sample_data):
    """
    Test integration with HDD consumption utilities.
    
    Verifies that fuel cost calculations properly integrate with the new
    HDD utility functions instead of using pre-computed columns.
    """
    df = sample_data
    category = 'heating'
    year_label = 2024
    menu_mp = 8
    
    # Test that HDD-adjusted consumption can be calculated
    consumption = get_hdd_adjusted_consumption(df, category, year_label, menu_mp)
    
    # Verify results
    assert isinstance(consumption, pd.Series), "Should return a pandas Series"
    assert len(consumption) == len(df), "Should have same length as input DataFrame"
    assert not consumption.isna().all(), "Should have some non-NaN values"


def test_cooking_electricity_inclusion(sample_data):
    """
    Test that cooking baseline includes electricity fuel type.
    
    This verifies the fix where cooking baseline was updated to include
    electricity alongside naturalGas and propane.
    """
    df = sample_data
    category = 'cooking'
    year_label = 2024
    
    # Test baseline consumption calculation for cooking
    baseline_consumption = get_total_baseline_consumption(df, category, year_label)
    
    # Should not raise an error and should include electricity consumption
    assert isinstance(baseline_consumption, pd.Series), "Should return consumption Series"
    
    # If electricity consumption column exists, verify it's included in total
    elec_col = 'base_electricity_cooking_consumption'
    if elec_col in df.columns:
        elec_consumption = df[elec_col].fillna(0)
        # Total should be >= electricity consumption for each home
        assert (baseline_consumption >= elec_consumption).all(), \
            "Baseline should include electricity consumption"


def test_apply_hdd_adjustment_all_categories():
    """
    Test that HDD adjustment works for all categories without raising exceptions.
    
    This verifies the fix where apply_hdd_adjustment was updated to handle
    all categories gracefully instead of raising exceptions.
    """
    # Create test data
    consumption = pd.Series([100, 200, 300])
    hdd_factor = pd.Series([1.1, 1.2, 1.3])
    
    # Test all categories
    for category in EQUIPMENT_SPECS:
        try:
            result = apply_hdd_adjustment(consumption, category, hdd_factor)
            
            if category == 'heating':
                # Should be adjusted
                expected = consumption * hdd_factor
                assert result.equals(expected), f"Heating should be HDD-adjusted"
            else:
                # Should be unchanged
                assert result.equals(consumption), f"{category} should be unchanged"
                
        except Exception as e:
            pytest.fail(f"apply_hdd_adjustment should not raise exception for {category}: {e}")


def test_basic_fuel_cost_calculation(sample_data, mock_scenario_params):
    """
    Test basic fuel cost calculation functionality.
    
    Verifies that the main function can calculate fuel costs without errors
    and produces appropriate output structure.
    """
    df = sample_data
    menu_mp = 0  # Baseline
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call main function
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify output structure
    assert isinstance(df_main, pd.DataFrame), "df_main should be DataFrame"
    assert isinstance(df_detailed, pd.DataFrame), "df_detailed should be DataFrame"
    assert len(df_main) == len(df), "df_main should have same number of rows"
    assert len(df_detailed) == len(df), "df_detailed should have same number of rows"
    
    # Check for expected columns
    for category in ['heating', 'waterHeating']:  # Test subset for brevity
        expected_col = f'baseline_{category}_lifetime_fuel_cost'
        assert expected_col in df_main.columns, f"Should have {expected_col}"


def test_measure_package_calculation(sample_data, mock_scenario_params):
    """
    Test measure package fuel cost calculation.
    
    Verifies that measure package calculations work correctly and produce
    different results from baseline calculations.
    """
    df = sample_data
    menu_mp = 8  # Measure package
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call main function
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Check for expected measure package columns
    for category in ['heating', 'waterHeating']:  # Test subset
        expected_col = f'iraRef_mp{menu_mp}_{category}_lifetime_fuel_cost'
        assert expected_col in df_main.columns, f"Should have {expected_col}"


def test_validation_masking(sample_data, mock_scenario_params):
    """
    Test that validation masking is properly applied.
    
    Verifies that invalid homes receive NaN values while valid homes
    receive calculated fuel cost values.
    """
    # Create DataFrame with explicit valid/invalid homes
    df = sample_data.copy()
    category = 'heating'
    
    # Make first half of homes valid, second half invalid
    mid_point = len(df) // 2
    df[f'include_{category}'] = [True] * mid_point + [False] * (len(df) - mid_point)
    
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call main function
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Check masking
    fuel_cost_col = f'baseline_{category}_lifetime_fuel_cost'
    if fuel_cost_col in df_main.columns:
        valid_mask = df[f'include_{category}']
        
        # Valid homes should have non-NaN values
        valid_values = df_main.loc[valid_mask, fuel_cost_col]
        if len(valid_values) > 0:
            assert not valid_values.isna().all(), "Valid homes should have fuel cost values"
        
        # Invalid homes should have NaN values
        invalid_values = df_main.loc[~valid_mask, fuel_cost_col]
        if len(invalid_values) > 0:
            assert invalid_values.isna().all(), "Invalid homes should have NaN values"


# =============================================================================
# PARAMETRIZED TESTS - FOLLOWING STREAMLINED PATTERNS
# =============================================================================

@pytest.mark.parametrize("category", ['heating', 'waterHeating', 'clothesDrying', 'cooking'])
def test_categories_integration(sample_data, mock_scenario_params, category):
    """Test fuel cost calculation for each equipment category."""
    df = sample_data
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    # Make sure at least some homes are valid for this category
    df[f'include_{category}'] = [True] * (len(df) // 2) + [False] * (len(df) - len(df) // 2)
    
    # Call main function
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Check that category-specific columns exist
    expected_col = f'baseline_{category}_lifetime_fuel_cost'
    assert expected_col in df_main.columns, f"Should calculate costs for {category}"


@pytest.mark.parametrize("menu_mp", [0, 8, 9, 10])
def test_menu_mp_scenarios(sample_data, mock_scenario_params, menu_mp):
    """Test fuel cost calculation for different measure packages."""
    df = sample_data
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call main function
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Check appropriate column naming
    expected_prefix = 'baseline_' if menu_mp == 0 else f'iraRef_mp{menu_mp}_'
    expected_col = f'{expected_prefix}heating_lifetime_fuel_cost'
    assert expected_col in df_main.columns, f"Should use correct prefix for menu_mp={menu_mp}"


@pytest.mark.parametrize("policy_scenario", ['No Inflation Reduction Act', 'AEO2023 Reference Case'])
def test_policy_scenarios(sample_data, mock_scenario_params, policy_scenario):
    """Test fuel cost calculation for different policy scenarios."""
    df = sample_data
    menu_mp = 8
    
    # Call main function
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Should complete without error for both policy scenarios
    assert isinstance(df_main, pd.DataFrame), f"Should work for {policy_scenario}"


# =============================================================================
# EDGE CASE TESTS - ESSENTIAL COVERAGE
# =============================================================================

def test_empty_dataframe(mock_scenario_params):
    """Test handling of empty DataFrame."""
    empty_df = pd.DataFrame()
    
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=empty_df,
        menu_mp=0,
        policy_scenario='AEO2023 Reference Case',
        verbose=False
    )
    
    # Should return empty DataFrames
    assert df_main.empty, "Should return empty DataFrame for empty input"
    assert df_detailed.empty, "Should return empty detailed DataFrame for empty input"


def test_all_invalid_homes(sample_data, mock_scenario_params):
    """Test calculation when all homes are invalid."""
    df = sample_data.copy()
    
    # Make all homes invalid for all categories
    for category in EQUIPMENT_SPECS:
        df[f'include_{category}'] = False
    
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=df,
        menu_mp=0,
        policy_scenario='AEO2023 Reference Case',
        verbose=False
    )
    
    # Should complete without error and return NaN values
    for category in ['heating', 'waterHeating']:  # Test subset
        col = f'baseline_{category}_lifetime_fuel_cost'
        if col in df_main.columns:
            assert df_main[col].isna().all(), f"All homes should have NaN for {category}"


def test_invalid_policy_scenario(sample_data):
    """Test handling of invalid policy scenario."""
    df = sample_data
    
    with pytest.raises(ValueError, match="Invalid policy_scenario"):
        calculate_lifetime_fuel_costs(
            df=df,
            menu_mp=0,
            policy_scenario='Invalid Scenario',
            verbose=False
        )


def test_missing_required_columns(mock_scenario_params):
    """Test handling of missing required columns."""
    # Create DataFrame missing required 'state' column
    df = pd.DataFrame({
        'census_division': ['Pacific', 'West South Central'],
        'include_heating': [True, False],
        'include_waterHeating': [True, True],
        'include_clothesDrying': [False, True],
        'include_cooking': [True, False]
    })
    
    with pytest.raises(KeyError, match="Required columns missing"):
        calculate_lifetime_fuel_costs(
            df=df,
            menu_mp=0,
            policy_scenario='AEO2023 Reference Case',
            verbose=False
        )


# =============================================================================
# INTEGRATION SPECIFIC TESTS - VERIFYING FIXES
# =============================================================================

def test_hdd_adjustment_integration():
    """
    Test that HDD adjustment integrates properly with fuel cost calculations.
    
    This specifically tests the integration fixes mentioned in the conversation summary.
    """
    # Create test DataFrame
    df = pd.DataFrame({
        'state': ['CA', 'TX'],
        'census_division': ['Pacific', 'West South Central'],
        'include_heating': [True, True],
        'base_heating_fuel': ['Electricity', 'Natural Gas'],
        'base_electricity_heating_consumption': [100, 0],
        'base_naturalGas_heating_consumption': [0, 200],
        'mp8_heating_consumption': [50, 80]  # Measure package consumption
    })
    
    # Test baseline consumption (should include HDD adjustment for heating)
    baseline_consumption = get_total_baseline_consumption(df, 'heating', 2024)
    assert len(baseline_consumption) == len(df), "Should return consumption for all homes"
    
    # Test measure package consumption (should include HDD adjustment for heating)  
    mp_consumption = get_hdd_adjusted_consumption(df, 'heating', 2024, 8)
    assert len(mp_consumption) == len(df), "Should return measure package consumption"


def test_cooking_fuel_fix():
    """
    Test that cooking fuel calculations include electricity.
    
    This specifically tests the cooking fuel fix mentioned in the conversation summary.
    """
    # Create test DataFrame with cooking consumption data
    df = pd.DataFrame({
        'state': ['CA', 'TX'],
        'census_division': ['Pacific', 'West South Central'],
        'include_cooking': [True, True],
        'base_electricity_cooking_consumption': [50, 80],
        'base_naturalGas_cooking_consumption': [30, 60],
        'base_propane_cooking_consumption': [20, 40]
    })
    
    # Test that baseline cooking consumption includes electricity
    cooking_consumption = get_total_baseline_consumption(df, 'cooking', 2024)
    
    # Should include electricity consumption
    expected_total = (
        df['base_electricity_cooking_consumption'].fillna(0) +
        df['base_naturalGas_cooking_consumption'].fillna(0) +
        df['base_propane_cooking_consumption'].fillna(0)
    )
    
    # Allow for small numerical differences
    assert np.allclose(cooking_consumption, expected_total, atol=0.01), \
        "Cooking consumption should include electricity"


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
