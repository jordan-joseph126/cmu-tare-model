"""
test_calculate_lifetime_fuel_costs.py

Pytest test suite for validating the lifetime fuel costs calculation functionality.
This test suite focuses on the core calculation logic, while the validation framework
tests have been moved to a separate module.
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

# Import validation framework test utilities
from cmu_tare_model.tests.test_validation_framework import (
    mock_constants,  # Re-use fixtures when needed
    sample_homes_df,
    mock_fuel_prices,
)

# Import constants
from cmu_tare_model.constants import EQUIPMENT_SPECS, FUEL_MAPPING, UPGRADE_COLUMNS
from cmu_tare_model.utils.validation_framework import get_valid_calculation_mask

# -------------------------------------------------------------------------
#                           TEST FIXTURES
# -------------------------------------------------------------------------


@pytest.fixture
def mock_scenario_params(mock_fuel_prices: Dict[str, Dict[str, Dict[str, Dict[int, float]]]], 
                        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Mock the define_scenario_params function to control test scenarios.
    """
    def mock_function(menu_mp: int, policy_scenario: str) -> Tuple[str, str, Dict, Dict, Dict, Dict]:
        """Mock implementation of define_scenario_params."""
        if menu_mp == 0:
            scenario_prefix = "baseline_"
        elif policy_scenario == 'No Inflation Reduction Act':
            scenario_prefix = f"preIRA_mp{menu_mp}_"
        else:
            scenario_prefix = f"iraRef_mp{menu_mp}_"
            
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
    """
    data = {}
    
    # Generate baseline fuel costs (annual and lifetime)
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        # Generate annual fuel costs for years 2024-2026
        for year in range(2024, 2027):
            col_name = f'baseline_{year}_{category}_fuelCost'
            data[col_name] = [
                (1000 + (home_idx * 100)) * (0.2 if category == 'heating' else 
                                             0.15 if category == 'waterHeating' else
                                             0.1 if category == 'clothesDrying' else 0.08)
                for home_idx in range(5)
            ]
        
        # Generate lifetime fuel costs
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


@pytest.fixture
def mock_annual_calculation(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Mock the annual calculation to provide consistent test values.
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
        """Mock implementation for calculate_annual_fuel_costs."""
        # If no valid_mask provided, use all homes
        if valid_mask is None:
            valid_mask = pd.Series(True, index=df.index)
        
        # Create results dictionary and cost column
        annual_costs = {}
        cost_col = f"{scenario_prefix}{year_label}_{category}_fuelCost"
        
        # Create a Series with zeros for all homes
        cost_series = pd.Series(0.0, index=df.index)
        
        # Set values for valid homes only
        valid_homes = valid_mask[valid_mask].index
        if len(valid_homes) > 0:
            # Use home index to create different values
            for i, idx in enumerate(valid_homes):
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
            
        # Create annual costs dictionary
        annual_costs[cost_col] = cost_series
        
        # For measure packages, add savings column
        if menu_mp != 0:
            savings_col = f"{scenario_prefix}{year_label}_{category}_savings_fuelCost"
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
    """Parametrized fixture for equipment categories."""
    return request.param


@pytest.fixture(params=[0, 8])
def menu_mp(request) -> int:
    """Parametrized fixture for measure package values."""
    return request.param


@pytest.fixture(params=['No Inflation Reduction Act', 'AEO2023 Reference Case'])
def policy_scenario(request) -> str:
    """Parametrized fixture for policy scenarios."""
    return request.param


# -------------------------------------------------------------------------
#                           BASIC TESTS
# -------------------------------------------------------------------------

def test_calculate_annual_fuel_costs_basic(sample_homes_df: pd.DataFrame, mock_fuel_prices: Dict, mock_scenario_params: None) -> None:
    """
    Test basic calculation of annual fuel costs.
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
    cost_col = f'{scenario_prefix}{year_label}_{category}_fuelCost'
    assert cost_col in annual_costs, f"Result should contain column '{cost_col}'"
    assert isinstance(annual_cost_value, pd.Series), "annual_cost_value should be a Series"
    assert len(annual_cost_value) == len(df), "annual_cost_value should have same length as DataFrame"
    
    # Check only valid homes have non-zero values
    invalid_indices = valid_mask[~valid_mask].index
    if len(invalid_indices) > 0:
        assert (annual_cost_value.loc[invalid_indices] == 0.0).all(), \
            "Invalid homes should have fuel cost of 0.0 in annual_cost_value"


def test_lifetime_fuel_costs_basic(sample_homes_df: pd.DataFrame, mock_scenario_params: None, mock_annual_calculation: None) -> None:
    """
    Test basic functionality of calculate_lifetime_fuel_costs.
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
        lifetime_col = f'baseline_{category}_lifetime_fuelCost'
        assert lifetime_col in df_main.columns, f"df_main should have column '{lifetime_col}'"
        
        # Verify values are masked based on inclusion flags
        valid_mask = sample_homes_df[f'include_{category}']
        invalid_indices = valid_mask[~valid_mask].index
        
        if len(invalid_indices) > 0:
            assert df_main.loc[invalid_indices, lifetime_col].isna().all(), \
                f"Invalid homes should have NaN values for {lifetime_col}"


# -------------------------------------------------------------------------
#              LIST-BASED COLLECTION TEST (FIXED)
# -------------------------------------------------------------------------

def test_list_based_collection(sample_homes_df: pd.DataFrame, mock_scenario_params: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that list-based collection correctly accumulates values."""
    # Create a spy function to track calls to calculate_annual_fuel_costs
    yearly_costs_captured = []
    
    def spy_annual_costs(*args, **kwargs):
        """Spy function to capture annual costs."""
        # Get parameters from args or kwargs
        df = kwargs.get('df', args[0] if args else None)
        category = kwargs.get('category', args[1] if len(args) > 1 else None)
        year_label = kwargs.get('year_label', args[2] if len(args) > 2 else None)
        scenario_prefix = kwargs.get('scenario_prefix', '')
        valid_mask = kwargs.get('valid_mask', pd.Series(True, index=df.index))
        
        # Create predictable values
        cost_col = f"{scenario_prefix}{year_label}_{category}_fuelCost"
        cost_series = pd.Series(0.0, index=df.index)
        
        for idx in valid_mask[valid_mask].index:
            cost_series.loc[idx] = 100 + (year_label - 2024) * 10 + idx
        
        yearly_costs_captured.append(cost_series.copy())
        return {cost_col: cost_series}, cost_series
    
    # Use context manager for monkeypatching
    with monkeypatch.context() as m:
        m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.calculate_annual_fuel_costs',
                  spy_annual_costs)
        
        # Set shorter lifetimes for testing efficiency
        m.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', 
                 {'heating': 3, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
        m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.EQUIPMENT_SPECS', 
                 {'heating': 3, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
        
        # Run calculation
        category = 'heating'
        menu_mp = 8
        policy_scenario = 'AEO2023 Reference Case'
        
        df_main, df_detailed = calculate_lifetime_fuel_costs(
            df=sample_homes_df,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )
    
    # Verify results
    assert len(yearly_costs_captured) > 0, "Should have captured yearly cost values"
    lifetime_col = f'iraRef_mp{menu_mp}_{category}_lifetime_fuelCost'
    assert lifetime_col in df_main.columns
    
    # Check if yearly costs sum up correctly
    valid_mask = sample_homes_df[f'include_{category}']
    valid_indices = valid_mask[valid_mask].index
    
    if len(valid_indices) > 0:
        idx = valid_indices[0]
        yearly_sum = sum(yearly_cost[idx] for yearly_cost in yearly_costs_captured if idx in yearly_cost.index)
        assert abs(df_main.loc[idx, lifetime_col] - yearly_sum) < 0.1, \
            f"Lifetime cost should match sum of yearly costs for home at index {idx}"


# -------------------------------------------------------------------------
#              LIFETIME FUEL COSTS WITH BASELINE TEST (FIXED)
# -------------------------------------------------------------------------

def test_lifetime_fuel_costs_with_baseline(sample_homes_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test calculation with measure package and baseline costs."""
    # Create baseline costs with predictable values
    df_baseline_costs = pd.DataFrame(index=sample_homes_df.index)
    baseline_costs = {}
    
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        # Create lifetime columns
        offset = {'heating': 1000, 'waterHeating': 800, 'clothesDrying': 600, 'cooking': 400}
        col_name = f"baseline_{category}_lifetime_fuelCost"
        baseline_costs[col_name] = [offset[category] + (idx * 100) for idx in range(len(sample_homes_df))]
        
        # Add annual costs
        for year in range(2024, 2027):
            col_name = f"baseline_{year}_{category}_fuelCost"
            baseline_costs[col_name] = [(offset[category] + (idx * 100)) / 10 for idx in range(len(sample_homes_df))]
    
    df_baseline_costs = pd.DataFrame(baseline_costs, index=sample_homes_df.index)
    
    # Create mock for consistent savings calculation
    def mock_annual_calc(df, category, year_label, **kwargs):
        scenario_prefix = kwargs.get('scenario_prefix', 'baseline_')
        menu_mp = kwargs.get('menu_mp', 0)
        valid_mask = kwargs.get('valid_mask', pd.Series(True, index=df.index))
        
        cost_col = f"{scenario_prefix}{year_label}_{category}_fuelCost"
        cost_series = pd.Series(np.nan, index=df.index)
        baseline_col = f"baseline_{year_label}_{category}_fuelCost"
        
        for idx in valid_mask[valid_mask].index:
            if menu_mp == 0:
                # For baseline, use baseline costs
                if baseline_col in df_baseline_costs.columns:
                    cost_series.loc[idx] = df_baseline_costs.loc[idx, baseline_col]
                else:
                    offset = {'heating': 100, 'waterHeating': 80, 'clothesDrying': 60, 'cooking': 40}
                    cost_series.loc[idx] = offset[category] + (idx * 10)
            else:
                # For measure package, use 60% of baseline (40% savings)
                if baseline_col in df_baseline_costs.columns:
                    cost_series.loc[idx] = df_baseline_costs.loc[idx, baseline_col] * 0.6
                else:
                    offset = {'heating': 60, 'waterHeating': 48, 'clothesDrying': 36, 'cooking': 24}
                    cost_series.loc[idx] = offset[category] + (idx * 6)
        
        annual_costs = {cost_col: cost_series}
        
        # Add savings column for measure packages
        if menu_mp != 0 and baseline_col in df_baseline_costs.columns:
            savings_col = f"{scenario_prefix}{year_label}_{category}_savings_fuelCost"
            savings_series = pd.Series(np.nan, index=df.index)
            
            for idx in valid_mask[valid_mask].index:
                baseline_cost = df_baseline_costs.loc[idx, baseline_col]
                measure_cost = cost_series.loc[idx]
                savings_series.loc[idx] = baseline_cost - measure_cost
            
            annual_costs[savings_col] = savings_series
        
        return annual_costs, cost_series
    
    # Run test with monkeypatched calculation
    with monkeypatch.context() as m:
        m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.calculate_annual_fuel_costs',
                  mock_annual_calc)
        
        # Set shorter lifetimes
        m.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', 
                 {'heating': 3, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
        m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.EQUIPMENT_SPECS', 
                 {'heating': 3, 'waterHeating': 2, 'clothesDrying': 2, 'cooking': 2})
        
        menu_mp = 8
        policy_scenario = 'AEO2023 Reference Case'
        
        # Call calculation function
        df_main, df_detailed = calculate_lifetime_fuel_costs(
            df=sample_homes_df,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            df_baseline_costs=df_baseline_costs,
            verbose=False
        )
    
    # Verify savings calculations
    for category in ['heating', 'waterHeating']:  # Test subset for efficiency
        costs_col = f'iraRef_mp{menu_mp}_{category}_lifetime_fuelCost'
        savings_col = f'iraRef_mp{menu_mp}_{category}_lifetime_savings_fuelCost'
        baseline_col = f'baseline_{category}_lifetime_fuelCost'
        
        if baseline_col not in df_baseline_costs.columns or costs_col not in df_main.columns:
            continue
        
        valid_mask = sample_homes_df[f'include_{category}']
        valid_indices = valid_mask[valid_mask].index
        
        for idx in valid_indices:
            baseline_cost = df_baseline_costs.loc[idx, baseline_col]
            measure_cost = df_main.loc[idx, costs_col]
            savings = df_main.loc[idx, savings_col]
            
            assert abs((baseline_cost - measure_cost) - savings) < 0.1, \
                f"Savings should be baseline - measure for {category}, index {idx}"


# -------------------------------------------------------------------------
#              PARAMETRIZED AND EDGE CASE TESTS
# -------------------------------------------------------------------------

def test_across_categories(sample_homes_df: pd.DataFrame, mock_fuel_prices: Dict, 
                          mock_scenario_params: None, category: str, 
                          mock_annual_calculation: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test calculation across different equipment categories."""
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
    lifetime_col = f'baseline_{category}_lifetime_fuelCost'
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


def test_empty_dataframe(mock_fuel_prices: Dict, mock_scenario_params: None) -> None:
    """Test handling of empty DataFrame."""
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


def test_all_invalid_homes(sample_homes_df: pd.DataFrame, mock_scenario_params: None, 
                          mock_annual_calculation: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test calculation when all homes are invalid."""
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
        lifetime_col = f'baseline_{category}_lifetime_fuelCost'
        if lifetime_col in df_main.columns:
            assert df_main[lifetime_col].isna().all(), \
                f"All homes should have NaN values for {lifetime_col}"


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])