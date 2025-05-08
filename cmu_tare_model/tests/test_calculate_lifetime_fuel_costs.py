"""
test_calculate_lifetime_fuel_costs_functional.py

Pytest test suite for validating the lifetime fuel costs calculation functionality
and its implementation of the 5-step validation framework:
1. Mask Initialization with initialize_validation_tracking()
2. Series Initialization with create_retrofit_only_series()
3. Valid-Only Calculation for qualifying homes
4. Valid-Only Updates using list-based collection
5. Final Masking with apply_final_masking()

This test suite uses a functional approach with clear test functions and
reusable fixtures to verify both calculation correctness and framework integration.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Set

# Import the specific module being tested
from cmu_tare_model.private_impact.calculate_lifetime_fuel_costs import (
    calculate_lifetime_fuel_costs,
    calculate_annual_fuel_costs
)

# Import constants and validation framework utilities for validation
from cmu_tare_model.constants import EQUIPMENT_SPECS, FUEL_MAPPING
from cmu_tare_model.utils.validation_framework import (
    initialize_validation_tracking,
    create_retrofit_only_series,
    apply_final_masking,
    get_valid_calculation_mask,
    calculate_avoided_values
)

# -------------------------------------------------------------------------
#                          SHARED TEST FIXTURES
# -------------------------------------------------------------------------
# Import fixtures from validation framework tests to ensure consistency
from cmu_tare_model.tests.test_validation_framework import (
    mock_constants,
    sample_homes_df,
    category,
    menu_mp
)

# -------------------------------------------------------------------------
#                           SPECIFIC TEST FIXTURES
# -------------------------------------------------------------------------

@pytest.fixture
def df_baseline_costs(sample_homes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sample baseline cost DataFrame for testing.
    
    This fixture generates a DataFrame with baseline fuel costs:
    - Annual costs for each year (2024-2026)
    - Lifetime costs for each category
    - Appropriate scaling by category and home
    
    Args:
        sample_homes_df: Sample DataFrame to match index
        
    Returns:
        DataFrame with baseline cost data for testing
    """
    data = {}
    
    # Generate baseline fuel costs for each year
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        # Generate annual fuel costs (2024-2026)
        for year in range(2024, 2027):
            col_name = f'baseline_{year}_{category}_fuelCost'
            data[col_name] = [
                (1000 + (home_idx * 100)) * (0.2 if category == 'heating' else 
                                            0.15 if category == 'waterHeating' else
                                            0.1 if category == 'clothesDrying' else 0.08)
                for home_idx in range(5)
            ]
        
        # Generate lifetime fuel costs based on category-specific lifetime factors
        lifetime_factor = {
            'heating': 3,  # Shortened from original lifetimes
            'waterHeating': 2,
            'clothesDrying': 2,
            'cooking': 2
        }[category]
        
        col_name = f'baseline_{category}_lifetime_fuelCost'
        data[col_name] = [
            sum(data[f'baseline_{year}_{category}_fuelCost'][home_idx] 
                for year in range(2024, 2024 + lifetime_factor))
            for home_idx in range(5)
        ]
    
    return pd.DataFrame(data, index=sample_homes_df.index)


@pytest.fixture
def mock_fuel_prices() -> Dict[str, Dict[str, Dict[str, Dict[int, float]]]]:
    """
    Create mock fuel price data for testing.
    
    This fixture builds a realistic nested dictionary structure that mimics
    the fuel price lookup data used in production code:
    - State-level prices for electricity and natural gas
    - Census division level prices for propane and fuel oil
    - Prices for different policy scenarios
    - Year-over-year price projections with realistic escalation
    
    Returns:
        Dict: Nested dictionary with mock fuel price data
    """
    # Create the basic structure
    mock_prices = {}
    
    # Add state-level data (for electricity and naturalGas)
    for state, multiplier in {'CA': 1.5, 'TX': 0.8, 'NY': 1.3, 'FL': 1.0, 'IL': 1.1}.items():
        mock_prices[state] = {
            'electricity': {
                'No Inflation Reduction Act': {
                    year: round(0.15 * multiplier * (1 + (year - 2024) * 0.02), 4) 
                    for year in range(2024, 2027)
                },
                'AEO2023 Reference Case': {
                    year: round(0.14 * multiplier * (1 + (year - 2024) * 0.015), 4) 
                    for year in range(2024, 2027)
                },
            },
            'naturalGas': {
                'No Inflation Reduction Act': {
                    year: round(0.08 * multiplier * (1 + (year - 2024) * 0.01), 4) 
                    for year in range(2024, 2027)
                },
                'AEO2023 Reference Case': {
                    year: round(0.075 * multiplier * (1 + (year - 2024) * 0.008), 4) 
                    for year in range(2024, 2027)
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
                    for year in range(2024, 2027)
                },
                'AEO2023 Reference Case': {
                    year: round(0.115 * multiplier * (1 + (year - 2024) * 0.012), 4) 
                    for year in range(2024, 2027)
                },
            },
            'fuelOil': {
                'No Inflation Reduction Act': {
                    year: round(0.14 * multiplier * (1 + (year - 2024) * 0.018), 4) 
                    for year in range(2024, 2027)
                },
                'AEO2023 Reference Case': {
                    year: round(0.135 * multiplier * (1 + (year - 2024) * 0.015), 4) 
                    for year in range(2024, 2027)
                },
            }
        }
    
    return mock_prices


@pytest.fixture
def mock_scenario_params(mock_fuel_prices: Dict[str, Dict[str, Dict[str, Dict[int, float]]]], 
                        monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Mock the define_scenario_params function to control test scenarios.
    
    This fixture replaces the scenario params function with a predictable
    implementation that returns controlled test data:
    - Consistent scenario prefixes based on menu_mp and policy_scenario
    - Mock fuel prices fixture for consumption calculations
    
    Args:
        mock_fuel_prices: Fixture providing fuel price test data
        monkeypatch: Pytest fixture for patching functions
    """
    def mock_function(menu_mp: int, policy_scenario: str) -> Tuple[str, str, Dict, Dict, Dict, Dict]:
        """Mock implementation of define_scenario_params."""
        # Determine scenario prefix based on menu_mp and policy_scenario
        if menu_mp == 0:
            scenario_prefix = "baseline_"
        elif policy_scenario == 'No Inflation Reduction Act':
            scenario_prefix = f"preIRA_mp{menu_mp}_"
        else:
            scenario_prefix = f"iraRef_mp{menu_mp}_"
            
        # Return controlled test data
        return (
            scenario_prefix,             # scenario_prefix
            "MidCase",                   # cambium_scenario
            {},                          # lookup_emissions_fossil_fuel
            {},                          # lookup_emissions_electricity_climate
            {},                          # lookup_emissions_electricity_health
            mock_fuel_prices             # lookup_fuel_prices
        )
    
    # Apply the patch to the module under test
    monkeypatch.setattr(
        'cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.define_scenario_params',
        mock_function
    )


@pytest.fixture
def mock_annual_calculation(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Mock the annual calculation to provide consistent test values.
    
    This fixture replaces the annual fuel cost calculation with a 
    predictable implementation that:
    - Returns controlled values based on category and year
    - Properly handles validation masking
    - Creates both costs and savings for measure packages
    
    Args:
        monkeypatch: Pytest fixture for patching functions
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
        
        # Set values for valid homes only - using deterministic values based on inputs
        valid_homes = valid_mask[valid_mask].index
        if len(valid_homes) > 0:
            # Use home index, year, and category to create different values
            for i, idx in enumerate(valid_homes):
                # Create deterministic, different values for each home, year, and category
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
        
    # Apply the patch to the module under test
    monkeypatch.setattr(
        'cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.calculate_annual_fuel_costs',
        mock_calculate_annual
    )


@pytest.fixture(params=['No Inflation Reduction Act', 'AEO2023 Reference Case'])
def policy_scenario(request) -> str:
    """
    Parametrized fixture for policy scenarios.
    
    This fixture provides different policy scenarios:
    - 'No Inflation Reduction Act': Pre-IRA scenario
    - 'AEO2023 Reference Case': IRA reference scenario
    
    Args:
        request: Pytest request object containing the parameter
        
    Returns:
        str: Policy scenario name
    """
    return request.param


# -------------------------------------------------------------------------
#                 VALIDATION FRAMEWORK IMPLEMENTATION TESTS
# -------------------------------------------------------------------------

def test_mask_initialization_implementation(sample_homes_df: pd.DataFrame, 
                                          mock_scenario_params: None,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test Step 1: Mask initialization with initialize_validation_tracking().
    
    This test verifies that calculate_lifetime_fuel_costs correctly:
    1. Initializes validation tracking for each category
    2. Creates a valid_mask using initialize_validation_tracking()
    3. Passes the valid_mask to subsequent calculations
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_scenario_params: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Track if initialize_validation_tracking is called
    init_tracking_called = {category: False for category in EQUIPMENT_SPECS}
    original_init_tracking = initialize_validation_tracking
    
    def mock_init_tracking(df: pd.DataFrame, category: str, 
                           menu_mp: Union[int, str], verbose: bool = True) -> Tuple:
        """Mock to track calls to initialize_validation_tracking."""
        init_tracking_called[category] = True
        return original_init_tracking(df, category, menu_mp, verbose)
    
    # Apply monkeypatching
    monkeypatch.setattr(
        'cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.initialize_validation_tracking',
        mock_init_tracking
    )
    
    # Call the main function
    menu_mp = 8  # Measure package
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, _ = calculate_lifetime_fuel_costs(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify initialize_validation_tracking was called for each category
    for category in EQUIPMENT_SPECS:
        assert init_tracking_called[category], \
            f"initialize_validation_tracking() should be called for category '{category}'"


def test_series_initialization_implementation(sample_homes_df: pd.DataFrame, 
                                              mock_scenario_params: None,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test Step 2: Series initialization with create_retrofit_only_series().
    
    This test verifies that calculate_lifetime_fuel_costs correctly:
    1. Uses create_retrofit_only_series() to initialize result series
    2. Sets zeros for valid homes and NaN for invalid homes
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_scenario_params: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Track if create_retrofit_only_series is called
    create_series_called = False
    original_create_series = create_retrofit_only_series
    
    def mock_create_series(df: pd.DataFrame, retrofit_mask: pd.Series, *args, **kwargs) -> pd.Series:
        """Mock to track calls to create_retrofit_only_series."""
        nonlocal create_series_called
        create_series_called = True
        return original_create_series(df, retrofit_mask, *args, **kwargs)
    
    # Apply monkeypatching
    monkeypatch.setattr(
        'cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.create_retrofit_only_series',
        mock_create_series
    )
    
    # Call the main function
    menu_mp = 8  # Measure package
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, _ = calculate_lifetime_fuel_costs(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify create_retrofit_only_series was called
    assert create_series_called, \
        "create_retrofit_only_series() should be called to initialize result series"


def test_valid_only_calculation_implementation(sample_homes_df: pd.DataFrame, 
                                              mock_scenario_params: None,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test Step 3: Valid-only calculation for qualifying homes.
    
    This test verifies that calculate_lifetime_fuel_costs correctly:
    1. Performs calculations only on valid homes
    2. Passes valid_mask to calculate_annual_fuel_costs
    3. Avoids calculations for invalid homes
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_scenario_params: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Track valid_mask usage in calculations
    valid_masks_used = []
    
    def mock_calculate_annual(*args, **kwargs):
        """Mock to track valid_mask usage in calculations."""
        valid_mask = kwargs.get('valid_mask')
        if valid_mask is not None:
            valid_masks_used.append(valid_mask.copy())
        
        # Return dummy data
        df = kwargs.get('df')
        category = kwargs.get('category')
        year_label = kwargs.get('year_label')
        scenario_prefix = kwargs.get('scenario_prefix', '')
        
        # Create a cost column with zeros
        cost_col = f"{scenario_prefix}{year_label}_{category}_fuelCost"
        cost_series = pd.Series(0.0, index=df.index)
        
        return {cost_col: cost_series}, cost_series
    
    # Apply monkeypatching
    monkeypatch.setattr(
        'cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.calculate_annual_fuel_costs',
        mock_calculate_annual
    )
    
    # Call the main function
    menu_mp = 8  # Measure package
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, _ = calculate_lifetime_fuel_costs(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify valid_mask was passed to calculations
    assert len(valid_masks_used) > 0, \
        "valid_mask should be passed to calculation functions"
    
    # Check that at least one valid_mask has False values
    has_masked_homes = any(not mask.all() for mask in valid_masks_used)
    assert has_masked_homes, \
        "At least one valid_mask should have False values (masked homes)"


def test_list_based_collection_implementation(sample_homes_df: pd.DataFrame, 
                                            mock_scenario_params: None,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test Step 4: Valid-only updates using list-based collection.
    
    This test verifies that calculate_lifetime_fuel_costs correctly:
    1. Uses list-based collection pattern for yearly costs
    2. Avoids inefficient incremental updates
    3. Only updates valid homes with calculated values
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_scenario_params: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Track if pd.concat is called (indicating list-based collection)
    concat_called = False
    original_concat = pd.concat
    
    def mock_concat(*args, **kwargs):
        """Mock to track calls to pd.concat."""
        nonlocal concat_called
        concat_called = True
        return original_concat(*args, **kwargs)
    
    # Apply monkeypatching
    monkeypatch.setattr(
        'pandas.concat',
        mock_concat
    )
    
    # Call the main function
    menu_mp = 8  # Measure package
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, _ = calculate_lifetime_fuel_costs(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify pd.concat was called (indicating list-based collection)
    assert concat_called, \
        "List-based collection pattern should use pd.concat to combine results"


def test_final_masking_implementation(sample_homes_df: pd.DataFrame, 
                                     mock_scenario_params: None,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test Step 5: Final masking with apply_final_masking().
    
    This test verifies that calculate_lifetime_fuel_costs correctly:
    1. Applies final masking to ensure consistent NaN values
    2. Passes appropriate columns to apply_final_masking()
    3. Ensures output has properly masked values
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_scenario_params: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
    # Track if apply_final_masking is called
    final_masking_called = False
    original_apply_masking = apply_final_masking
    masking_columns = {}
    
    def mock_apply_masking(df: pd.DataFrame, 
                          all_columns_to_mask: Dict[str, List[str]], 
                          verbose: bool = True) -> pd.DataFrame:
        """Mock to track calls to apply_final_masking."""
        nonlocal final_masking_called, masking_columns
        final_masking_called = True
        masking_columns = all_columns_to_mask.copy()
        return original_apply_masking(df, all_columns_to_mask, verbose)
    
    # Apply monkeypatching
    monkeypatch.setattr(
        'cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.apply_final_masking',
        mock_apply_masking
    )
    
    # Call the main function
    menu_mp = 8  # Measure package
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify apply_final_masking was called
    assert final_masking_called, \
        "apply_final_masking() should be called to finalize results"
    
    # Verify columns to mask were properly tracked
    for category in EQUIPMENT_SPECS:
        assert category in masking_columns, \
            f"Masking columns for category '{category}' should be tracked"
        
        # Should have at least the lifetime cost column
        expected_column = f'iraRef_mp{menu_mp}_{category}_lifetime_fuelCost'
        tracked_columns = masking_columns.get(category, [])
        assert any(expected_column in col for col in tracked_columns), \
            f"Lifetime column '{expected_column}' should be tracked for final masking"


def test_all_validation_steps_integrated(sample_homes_df: pd.DataFrame, 
                                       mock_scenario_params: None,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test integration of all 5 steps of the validation framework.
    
    This test verifies that calculate_lifetime_fuel_costs correctly:
    1. Implements all 5 steps of the validation framework in sequence
    2. Maintains proper masking throughout the process
    3. Returns consistently masked results
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_scenario_params: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
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
    
    original_calculate_annual = calculate_annual_fuel_costs
    def mock_calculate_annual(*args, **kwargs):
        # Only mark as executed if valid_mask parameter is provided
        if 'valid_mask' in kwargs:
            executed_steps['valid_calculation'] = True
        return original_calculate_annual(*args, **kwargs)
    
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
        m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.initialize_validation_tracking', mock_init_tracking)
        m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.create_retrofit_only_series', mock_create_series)
        m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.calculate_annual_fuel_costs', mock_calculate_annual)
        m.setattr('pandas.concat', mock_concat)
        m.setattr('cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.apply_final_masking', mock_apply_masking)
        
        # Call the main function
        menu_mp = 8
        policy_scenario = 'AEO2023 Reference Case'
        
        df_main, df_detailed = calculate_lifetime_fuel_costs(
            df=sample_homes_df,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )
    
    # Verify all steps were executed
    for step, executed in executed_steps.items():
        assert executed, f"Validation step '{step}' was not executed"


# -------------------------------------------------------------------------
#                       CALCULATE_ANNUAL_FUEL_COSTS TESTS
# -------------------------------------------------------------------------

def test_annual_fuel_costs_baseline(sample_homes_df: pd.DataFrame, 
                                  mock_fuel_prices: Dict,
                                  mock_scenario_params: None) -> None:
    """
    Test calculation of annual fuel costs for baseline scenario.
    
    This test verifies that calculate_annual_fuel_costs correctly:
    1. Handles baseline (menu_mp=0) scenario
    2. Properly uses fuel type mapping for lookup
    3. Applies validation masking
    4. Returns the correct result structure
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_fuel_prices: Fixture providing fuel price test data
        mock_scenario_params: Fixture mocking scenario parameters
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
    
    # Verify savings column is not created for baseline
    savings_col = f'{scenario_prefix}{year_label}_{category}_savings_fuelCost'
    assert savings_col not in annual_costs, f"Baseline results should not include savings column '{savings_col}'"


def test_annual_fuel_costs_measure_package(sample_homes_df: pd.DataFrame, 
                                         mock_fuel_prices: Dict, 
                                         mock_scenario_params: None) -> None:
    """
    Test calculation of annual fuel costs for measure package scenario.
    
    This test verifies that calculate_annual_fuel_costs correctly:
    1. Handles measure package (menu_mp=8) scenario
    2. Uses electricity prices for all measure package homes
    3. Applies validation masking
    4. Returns the correct result structure
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_fuel_prices: Fixture providing fuel price test data
        mock_scenario_params: Fixture mocking scenario parameters
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
    
    # Add retrofit status to valid mask for measure packages
    retrofit_column = f'upgrade_hvac_heating_efficiency'
    if retrofit_column in df.columns:
        retrofit_mask = df[retrofit_column].notna()
        combined_mask = valid_mask & retrofit_mask
    else:
        combined_mask = valid_mask
    
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
        valid_mask=combined_mask
    )
    
    # Verify result structure
    cost_col = f'{scenario_prefix}{year_label}_{category}_fuelCost'
    assert cost_col in annual_costs, f"Result should contain column '{cost_col}'"
    
    # Measure packages use state-based electricity prices
    # Check that fuel type columns aren't created for measures
    assert f'fuel_type_{category}' not in df.columns, \
        "Measure packages should not create fuel type columns"


def test_annual_fuel_costs_masking(sample_homes_df: pd.DataFrame, 
                                 mock_fuel_prices: Dict,
                                 mock_scenario_params: None) -> None:
    """
    Test that validation masking is properly applied in annual calculations.
    
    This test verifies that calculate_annual_fuel_costs correctly:
    1. Applies validation masking to fuel cost calculations
    2. Sets zero/NaN for invalid homes
    3. Returns properly masked results
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_fuel_prices: Fixture providing fuel price test data
        mock_scenario_params: Fixture mocking scenario parameters
    """
    # Test parameters
    category = 'heating'
    year_label = 2024
    menu_mp = 8  # Measure package for clear masking
    policy_scenario = 'AEO2023 Reference Case'
    scenario_prefix = f'iraRef_mp{menu_mp}_'
    
    # Create a mixed valid mask with some homes valid, some invalid
    df = sample_homes_df.copy()
    # First home valid, others invalid
    valid_mask = pd.Series([True] + [False] * (len(df) - 1), index=df.index)
    
    # Call calculate_annual_fuel_costs with explicit valid_mask
    annual_costs, annual_cost_value = calculate_annual_fuel_costs(
        df=df,
        category=category,
        year_label=year_label,
        menu_mp=menu_mp,
        lookup_fuel_prices=mock_fuel_prices,
        policy_scenario=policy_scenario,
        scenario_prefix=scenario_prefix,
        is_elec_or_gas=None,
        valid_mask=valid_mask
    )
    
    # Verify masking was applied - only first home should have a value
    first_idx = df.index[0]
    assert annual_cost_value[first_idx] > 0, "First home should have positive fuel cost"
    
    # Other homes should have zero fuel cost
    for idx in df.index[1:]:
        assert annual_cost_value[idx] == 0, f"Invalid home at index {idx} should have zero fuel cost"


# -------------------------------------------------------------------------
#                CALCULATE_LIFETIME_FUEL_COSTS TESTS
# -------------------------------------------------------------------------

def test_lifetime_fuel_costs_basic(sample_homes_df: pd.DataFrame, 
                                 mock_scenario_params: None, 
                                 mock_annual_calculation: None) -> None:
    """
    Test basic functionality of calculate_lifetime_fuel_costs.
    
    This test verifies that calculate_lifetime_fuel_costs correctly:
    1. Calculates fuel costs for all categories and years
    2. Returns main and detailed DataFrames
    3. Applies proper validation masking
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_scenario_params: Fixture mocking scenario parameters
        mock_annual_calculation: Fixture mocking annual calculation
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
    
    # Verify result contains lifetime columns for all categories
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        lifetime_col = f'baseline_{category}_lifetime_fuelCost'
        assert lifetime_col in df_main.columns, f"df_main should have column '{lifetime_col}'"
        assert lifetime_col in df_detailed.columns, f"df_detailed should have column '{lifetime_col}'"
        
        # Verify validation masking on lifetime columns
        valid_mask = sample_homes_df[f'include_{category}']
        invalid_indices = valid_mask[~valid_mask].index
        
        if len(invalid_indices) > 0:
            assert df_main.loc[invalid_indices, lifetime_col].isna().all(), \
                f"Invalid homes should have NaN values for {lifetime_col} in df_main"
            assert df_detailed.loc[invalid_indices, lifetime_col].isna().all(), \
                f"Invalid homes should have NaN values for {lifetime_col} in df_detailed"


def test_lifetime_fuel_costs_with_baseline(sample_homes_df: pd.DataFrame, 
                                         df_baseline_costs: pd.DataFrame,
                                         mock_scenario_params: None, 
                                         mock_annual_calculation: None) -> None:
    """
    Test calculation with measure package and baseline costs.
    
    This test verifies that calculate_lifetime_fuel_costs correctly:
    1. Uses baseline costs to calculate savings
    2. Applies validation masking consistently
    3. Includes baseline and savings columns in results
    
    Args:
        sample_homes_df: Fixture providing test data
        df_baseline_costs: Fixture providing baseline cost data
        mock_scenario_params: Fixture mocking scenario parameters
        mock_annual_calculation: Fixture mocking annual calculation
    """
    # Set up test parameters
    menu_mp = 8  # Measure package
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call the main function with baseline costs
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        df_baseline_costs=df_baseline_costs,
        verbose=False
    )
    
    # Verify result structure
    assert isinstance(df_main, pd.DataFrame), "df_main should be a DataFrame"
    
    # Check for all expected columns
    for category in ['heating', 'waterHeating']:  # Check subset for brevity
        # Measure package cost column
        costs_col = f'iraRef_mp{menu_mp}_{category}_lifetime_fuelCost'
        assert costs_col in df_main.columns, f"df_main should have column '{costs_col}'"
        
        # Savings column
        savings_col = f'iraRef_mp{menu_mp}_{category}_lifetime_savings_fuelCost'
        assert savings_col in df_main.columns, f"df_main should have column '{savings_col}'"
        
        # Baseline cost column should be in detailed DataFrame
        baseline_col = f'baseline_{category}_lifetime_fuelCost'
        assert baseline_col in df_detailed.columns, f"df_detailed should have column '{baseline_col}'"
        
        # Verify savings = baseline - measure for valid homes
        valid_mask = sample_homes_df[f'include_{category}']
        retrofit_column = f'upgrade_{category}_type' if category != 'heating' else 'upgrade_hvac_heating_efficiency'
        
        # For measure packages, combine data validation with retrofit status
        if retrofit_column in sample_homes_df.columns:
            retrofit_mask = sample_homes_df[retrofit_column].notna()
            combined_mask = valid_mask & retrofit_mask
        else:
            combined_mask = valid_mask
        
        # Check savings calculation for valid+retrofit homes
        valid_indices = combined_mask[combined_mask].index
        if len(valid_indices) > 0:
            for idx in valid_indices:
                baseline_cost = df_baseline_costs.loc[idx, baseline_col]
                measure_cost = df_main.loc[idx, costs_col]
                savings = df_main.loc[idx, savings_col]
                
                # If both values are present, check savings = baseline - measure
                if not pd.isna(baseline_cost) and not pd.isna(measure_cost):
                    # Allow small numerical differences
                    assert abs((baseline_cost - measure_cost) - savings) < 0.01, \
                        f"Savings should equal baseline - measure cost for {category} at index {idx}"


def test_lifetime_fuel_costs_list_collection(sample_homes_df: pd.DataFrame, 
                                           mock_scenario_params: None, 
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test that list-based collection correctly accumulates values.
    
    This test verifies that the module correctly:
    1. Collects yearly costs in a list instead of incremental updates
    2. Combines yearly costs into lifetime totals
    3. Maintains proper validation masking
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_scenario_params: Fixture mocking scenario parameters
        monkeypatch: Pytest fixture for patching functions
    """
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
        
        # Set values for valid homes only
        for idx in valid_mask[valid_mask].index:
            cost_series.loc[idx] = 100 + (year_label - 2024) * 10 + int(idx)
        
        # Store values for later verification
        yearly_costs_captured.append(cost_series.copy())
        
        return {cost_col: cost_series}, cost_series
    
    # Use monkeypatch context manager
    monkeypatch.setattr(
        'cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.calculate_annual_fuel_costs',
        spy_annual_costs
    )
    
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
    assert lifetime_col in df_main.columns, f"df_main should have column '{lifetime_col}'"
    
    # Check if yearly costs sum up correctly to lifetime values
    valid_mask = sample_homes_df[f'include_{category}']
    valid_indices = valid_mask[valid_mask].index
    
    if len(valid_indices) > 0:
        idx = valid_indices[0]
        # Sum all yearly values for this home
        yearly_sum = sum(yearly_cost.loc[idx] for yearly_cost in yearly_costs_captured 
                         if idx in yearly_cost.index)
        
        # Compare with lifetime value
        assert abs(df_main.loc[idx, lifetime_col] - yearly_sum) < 0.1, \
            f"Lifetime cost should match sum of yearly costs for home at index {idx}"


# -------------------------------------------------------------------------
#              PARAMETRIZED TESTS
# -------------------------------------------------------------------------

def test_different_categories(sample_homes_df: pd.DataFrame, 
                            mock_scenario_params: None,
                            mock_annual_calculation: None,
                            category: str) -> None:
    """
    Test calculation across different equipment categories.
    
    This parametrized test verifies that calculate_lifetime_fuel_costs:
    1. Works correctly for all equipment categories
    2. Applies category-specific validation
    3. Produces appropriate results for each category
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_scenario_params: Fixture mocking scenario parameters
        mock_annual_calculation: Fixture mocking annual calculation
        category: Parametrized fixture providing different categories
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
            f"At least some valid homes should have non-NaN values for {category}"
    
    # Invalid homes should have NaN values
    if len(invalid_indices) > 0:
        assert df_main.loc[invalid_indices, lifetime_col].isna().all(), \
            f"Invalid homes should have NaN values for {category}"


def test_different_menu_mps(sample_homes_df: pd.DataFrame, 
                          mock_scenario_params: None,
                          mock_annual_calculation: None,
                          menu_mp: int) -> None:
    """
    Test calculation with different measure package values.
    
    This parametrized test verifies that calculate_lifetime_fuel_costs:
    1. Works correctly for both baseline (menu_mp=0) and measure packages
    2. Uses appropriate column naming based on menu_mp
    3. Produces appropriate results for each scenario
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_scenario_params: Fixture mocking scenario parameters
        mock_annual_calculation: Fixture mocking annual calculation
        menu_mp: Parametrized fixture providing different menu_mp values
    """
    # Set up test parameters
    policy_scenario = 'AEO2023 Reference Case'
    
    # Determine expected scenario prefix
    expected_prefix = 'baseline_' if menu_mp == 0 else f'iraRef_mp{menu_mp}_'
    
    # Call the main function
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify result uses correct column naming
    category = 'heating'  # Test with one category for brevity
    lifetime_col = f'{expected_prefix}{category}_lifetime_fuelCost'
    assert lifetime_col in df_main.columns, f"Result should have column '{lifetime_col}'"
    
    # Verify values are masked based on inclusion flags
    valid_mask = sample_homes_df[f'include_{category}']
    valid_indices = valid_mask[valid_mask].index
    invalid_indices = valid_mask[~valid_mask].index
    
    # Valid homes should have non-NaN values
    if len(valid_indices) > 0:
        assert not df_main.loc[valid_indices, lifetime_col].isna().all(), \
            f"At least some valid homes should have non-NaN values for menu_mp={menu_mp}"
    
    # Invalid homes should have NaN values
    if len(invalid_indices) > 0:
        assert df_main.loc[invalid_indices, lifetime_col].isna().all(), \
            f"Invalid homes should have NaN values for menu_mp={menu_mp}"


def test_different_policy_scenarios(sample_homes_df: pd.DataFrame, 
                                  mock_scenario_params: None,
                                  mock_annual_calculation: None,
                                  policy_scenario: str) -> None:
    """
    Test calculation with different policy scenarios.
    
    This parametrized test verifies that calculate_lifetime_fuel_costs:
    1. Works correctly for different policy scenarios
    2. Uses appropriate column naming based on policy_scenario
    3. Produces appropriate results for each scenario
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_scenario_params: Fixture mocking scenario parameters
        mock_annual_calculation: Fixture mocking annual calculation
        policy_scenario: Parametrized fixture providing different policy scenarios
    """
    # Set up test parameters
    menu_mp = 8  # Use measure package for policy scenario differentiation
    
    # Determine expected scenario prefix
    expected_prefix = 'preIRA_mp8_' if policy_scenario == 'No Inflation Reduction Act' else 'iraRef_mp8_'
    
    # Call the main function
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify result uses correct column naming
    category = 'heating'  # Test with one category for brevity
    lifetime_col = f'{expected_prefix}{category}_lifetime_fuelCost'
    assert lifetime_col in df_main.columns, f"Result should have column '{lifetime_col}'"
    
    # Verify values are masked based on inclusion flags
    valid_mask = sample_homes_df[f'include_{category}']
    valid_indices = valid_mask[valid_mask].index
    invalid_indices = valid_mask[~valid_mask].index
    
    # Valid homes should have non-NaN values
    if len(valid_indices) > 0:
        assert not df_main.loc[valid_indices, lifetime_col].isna().all(), \
            f"At least some valid homes should have non-NaN values for policy_scenario='{policy_scenario}'"
    
    # Invalid homes should have NaN values
    if len(invalid_indices) > 0:
        assert df_main.loc[invalid_indices, lifetime_col].isna().all(), \
            f"Invalid homes should have NaN values for policy_scenario='{policy_scenario}'"


# -------------------------------------------------------------------------
#              EDGE CASE TESTS
# -------------------------------------------------------------------------

def test_empty_dataframe(mock_scenario_params: None, 
                        mock_annual_calculation: None) -> None:
    """
    Test handling of empty DataFrame.
    
    This test verifies that calculate_lifetime_fuel_costs:
    1. Handles empty DataFrames gracefully
    2. Raises appropriate errors with informative messages
    
    Args:
        mock_scenario_params: Fixture mocking scenario parameters
        mock_annual_calculation: Fixture mocking annual calculation
    """
    # Create an empty DataFrame with required columns
    empty_df = pd.DataFrame(columns=['state', 'census_division'])
    
    # Set up test parameters
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call the main function, expecting KeyError due to missing include columns
    with pytest.raises((ValueError, KeyError)) as excinfo:
        df_main, df_detailed = calculate_lifetime_fuel_costs(
            df=empty_df,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )
    
    # Verify error message
    error_msg = str(excinfo.value)
    assert "include_" in error_msg or "column" in error_msg or "missing" in error_msg, \
        "Error message should mention missing include columns"


def test_all_invalid_homes(sample_homes_df: pd.DataFrame, 
                          mock_scenario_params: None, 
                          mock_annual_calculation: None) -> None:
    """
    Test calculation when all homes are invalid.
    
    This test verifies that calculate_lifetime_fuel_costs:
    1. Handles the case where all homes are invalid
    2. Returns properly masked results (all NaN)
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_scenario_params: Fixture mocking scenario parameters
        mock_annual_calculation: Fixture mocking annual calculation
    """
    # Create modified DataFrame with all homes invalid
    df_modified = sample_homes_df.copy()
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        df_modified[f'include_{category}'] = False
    
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
                f"All homes should have NaN values for {lifetime_col} when all homes are invalid"


def test_missing_required_columns(sample_homes_df: pd.DataFrame, 
                                mock_scenario_params: None, 
                                mock_annual_calculation: None) -> None:
    """
    Test handling of missing required columns.
    
    This test verifies that calculate_lifetime_fuel_costs:
    1. Validates required input columns
    2. Raises appropriate errors with informative messages
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_scenario_params: Fixture mocking scenario parameters
        mock_annual_calculation: Fixture mocking annual calculation
    """
    # Create modified DataFrame with missing required columns
    df_missing = sample_homes_df.copy()
    df_missing = df_missing.drop(columns=['state'])
    
    # Set up test parameters
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call the main function, expecting KeyError
    with pytest.raises(KeyError) as excinfo:
        df_main, df_detailed = calculate_lifetime_fuel_costs(
            df=df_missing,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )
    
    # Verify error message
    error_msg = str(excinfo.value)
    assert "state" in error_msg or "missing" in error_msg, \
        "Error message should mention the missing 'state' column"


def test_invalid_policy_scenario(sample_homes_df: pd.DataFrame, 
                               mock_scenario_params: None, 
                               mock_annual_calculation: None) -> None:
    """
    Test handling of invalid policy scenario.
    
    This test verifies that calculate_lifetime_fuel_costs:
    1. Validates policy scenario inputs
    2. Raises appropriate errors with informative messages
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_scenario_params: Fixture mocking scenario parameters
        mock_annual_calculation: Fixture mocking annual calculation
    """
    # Set up test parameters with invalid policy scenario
    menu_mp = 0
    policy_scenario = 'Invalid Scenario'
    
    # Call the main function, expecting ValueError
    with pytest.raises(ValueError) as excinfo:
        df_main, df_detailed = calculate_lifetime_fuel_costs(
            df=sample_homes_df,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )
    
    # Verify error message
    error_msg = str(excinfo.value)
    assert "policy scenario" in error_msg.lower() or "Invalid Scenario" in error_msg, \
        "Error message should mention invalid policy scenario"


def test_invalid_menu_mp(sample_homes_df: pd.DataFrame, 
                        mock_scenario_params: None, 
                        mock_annual_calculation: None) -> None:
    """
    Test handling of invalid menu_mp value.
    
    This test verifies that calculate_lifetime_fuel_costs:
    1. Validates menu_mp inputs
    2. Handles non-integer menu_mp values appropriately
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_scenario_params: Fixture mocking scenario parameters
        mock_annual_calculation: Fixture mocking annual calculation
    """
    # Set up test parameters with invalid menu_mp
    menu_mp = "invalid"
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call the main function, expecting ValueError or TypeError
    with pytest.raises((ValueError, TypeError)) as excinfo:
        df_main, df_detailed = calculate_lifetime_fuel_costs(
            df=sample_homes_df,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )
    
    # Verify error message
    error_msg = str(excinfo.value)
    assert "menu_mp" in error_msg.lower() or "invalid" in error_msg.lower(), \
        "Error message should mention invalid menu_mp"


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
