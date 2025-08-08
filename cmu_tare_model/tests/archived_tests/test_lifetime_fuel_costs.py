"""
test_lifetime_fuel_costs.py

Pytest tests for validating the calculate_lifetime_fuel_costs module.
This test suite verifies proper implementation of the 5-step validation framework
and ensures computational consistency for fuel cost calculations.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Import the module to test
from cmu_tare_model.private_impact.calculate_lifetime_fuel_costs import (
    calculate_lifetime_fuel_costs,
    calculate_annual_fuel_costs
)

# Import utilities needed for tests
from cmu_tare_model.utils.validation_framework import (
    initialize_validation_tracking,
    create_retrofit_only_series,
    calculate_avoided_values,
    apply_final_masking
)

# -------------------------------------------------------------------------
#                           TEST FIXTURES
# -------------------------------------------------------------------------

@pytest.fixture
def sample_home_data() -> Dict[str, List[Any]]:
    """
    Generate sample data for creating test DataFrames.
    
    This fixture provides column values for multiple homes with diverse 
    characteristics to test various validation scenarios.
    
    Returns:
        Dict mapping column names to lists of values.
    """
    return {
        # Metadata columns
        'state': ['CA', 'TX', 'NY', 'FL', 'IL'],
        'census_division': ['Pacific', 'West South Central', 'Middle Atlantic', 'South Atlantic', 'East North Central'],
        
        # Fuel type columns
        'base_heating_fuel': ['Electricity', 'Natural Gas', 'Propane', 'Fuel Oil', 'Invalid'],
        'base_waterHeating_fuel': ['Electricity', 'Natural Gas', 'Electricity', 'Fuel Oil', 'Propane'],
        'base_clothesDrying_fuel': ['Electricity', 'Natural Gas', 'Electricity', None, 'Propane'],
        'base_cooking_fuel': ['Electricity', 'Natural Gas', 'Propane', 'Invalid', None],
        
        # Technology columns
        'heating_type': ['Electricity Baseboard', 'Natural Gas Fuel Furnace', 'Propane Fuel Furnace', 'Fuel Oil Fuel Furnace', 'Heat Pump'],
        'waterHeating_type': ['Electric Standard', 'Natural Gas Standard', 'Electric Heat Pump', 'Fuel Oil Standard', 'Propane Standard'],
        
        # Validation flags (would normally be created by identify_valid_homes)
        'include_heating': [True, True, True, True, False],
        'include_waterHeating': [True, True, True, False, True],
        'include_clothesDrying': [True, True, False, True, False],
        'include_cooking': [False, True, True, False, True],
        
        # Retrofit flags (would normally be created by get_retrofit_homes_mask)
        'upgrade_hvac_heating_efficiency': ['ASHP', None, 'ASHP', None, 'ASHP'],
        'upgrade_water_heater_efficiency': ['HP', None, None, 'HP', None],
        'upgrade_clothes_dryer': [None, 'Electric', None, None, 'Electric'],
        'upgrade_cooking_range': ['Induction', None, 'Resistance', None, None],
        
        # Consumption data - baseline (current year)
        'baseline_2024_heating_consumption': [1000, 2000, 1500, 1200, 800],
        'baseline_2024_waterHeating_consumption': [500, 800, 600, 400, 350],
        'baseline_2024_clothesDrying_consumption': [300, 400, 350, 250, 200],
        'baseline_2024_cooking_consumption': [200, 300, 250, 180, 150],
        
        # Baseline fuel costs (current year)
        'baseline_2024_heating_fuel_cost': [200, 300, 250, 180, 120],
        'baseline_2024_waterHeating_fuel_cost': [100, 150, 120, 90, 70],
        'baseline_2024_clothesDrying_fuel_cost': [60, 80, 70, 50, 40],
        'baseline_2024_cooking_fuel_cost': [40, 60, 50, 36, 30],
        
        # Consumption data - measure package 8 (current year)
        'mp8_2024_heating_consumption': [800, 1600, 1200, 1000, 700],
        'mp8_2024_waterHeating_consumption': [400, 700, 500, 350, 300],
        'mp8_2024_clothesDrying_consumption': [240, 320, 280, 200, 180],
        'mp8_2024_cooking_consumption': [160, 240, 200, 150, 130],
    }

@pytest.fixture
def sample_homes_df(sample_home_data) -> pd.DataFrame:
    """
    Create a sample DataFrame with home data for testing.
    
    This fixture converts the dictionary data into a properly structured DataFrame.
    
    Args:
        sample_home_data: Dictionary of column data from the sample_home_data fixture.
        
    Returns:
        DataFrame with sample data for testing.
    """
    return pd.DataFrame(sample_home_data)

@pytest.fixture
def multi_year_homes_df(sample_homes_df) -> pd.DataFrame:
    """
    Create a sample DataFrame with multi-year consumption data.
    
    This fixture extends the basic sample DataFrame with consumption data
    for multiple years to test lifetime calculations.
    
    Args:
        sample_homes_df: The basic sample DataFrame.
        
    Returns:
        DataFrame with multi-year data for testing.
    """
    df = sample_homes_df.copy()
    
    # Add consumption data for multiple years
    for year in range(2025, 2040):
        for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
            # Baseline consumption (slight increase each year)
            baseline_col = f'baseline_{year}_{category}_consumption'
            current_year_col = f'baseline_2024_{category}_consumption'
            year_factor = 1.0 + ((year - 2024) * 0.01)  # 1% increase per year
            df[baseline_col] = df[current_year_col] * year_factor
            
            # Baseline costs
            baseline_cost_col = f'baseline_{year}_{category}_fuel_cost'
            current_cost_col = f'baseline_2024_{category}_fuel_cost'
            df[baseline_cost_col] = df[current_cost_col] * year_factor * 1.02  # 2% price increase
            
            # Measure package consumption (mp8)
            mp_col = f'mp8_{year}_{category}_consumption'
            current_mp_col = f'mp8_2024_{category}_consumption'
            df[mp_col] = df[current_mp_col] * year_factor
    
    return df

@pytest.fixture
def mock_fuel_prices() -> Dict[str, Dict[str, Dict[str, Dict[int, float]]]]:
    """
    Create mock fuel price data for testing.
    
    This fixture creates a nested dictionary structure that mimics the actual
    fuel price lookup data used by the module.
    
    Returns:
        Nested dictionary with mock fuel price data.
    """
    # Structure: location -> fuel_type -> policy_scenario -> year -> price
    mock_prices = {
        # State-level prices (for electricity and natural gas)
        'CA': {
            'electricity': {
                'No Inflation Reduction Act': {year: 0.20 + (year - 2024) * 0.01 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.18 + (year - 2024) * 0.008 for year in range(2024, 2040)}
            },
            'naturalGas': {
                'No Inflation Reduction Act': {year: 0.12 + (year - 2024) * 0.005 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.11 + (year - 2024) * 0.004 for year in range(2024, 2040)}
            }
        },
        'TX': {
            'electricity': {
                'No Inflation Reduction Act': {year: 0.15 + (year - 2024) * 0.008 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.14 + (year - 2024) * 0.007 for year in range(2024, 2040)}
            },
            'naturalGas': {
                'No Inflation Reduction Act': {year: 0.10 + (year - 2024) * 0.004 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.09 + (year - 2024) * 0.003 for year in range(2024, 2040)}
            }
        },
        'NY': {
            'electricity': {
                'No Inflation Reduction Act': {year: 0.22 + (year - 2024) * 0.011 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.20 + (year - 2024) * 0.009 for year in range(2024, 2040)}
            },
            'naturalGas': {
                'No Inflation Reduction Act': {year: 0.14 + (year - 2024) * 0.006 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.13 + (year - 2024) * 0.005 for year in range(2024, 2040)}
            }
        },
        'FL': {
            'electricity': {
                'No Inflation Reduction Act': {year: 0.16 + (year - 2024) * 0.009 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.15 + (year - 2024) * 0.008 for year in range(2024, 2040)}
            },
            'naturalGas': {
                'No Inflation Reduction Act': {year: 0.11 + (year - 2024) * 0.004 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.10 + (year - 2024) * 0.003 for year in range(2024, 2040)}
            }
        },
        'IL': {
            'electricity': {
                'No Inflation Reduction Act': {year: 0.18 + (year - 2024) * 0.010 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.17 + (year - 2024) * 0.008 for year in range(2024, 2040)}
            },
            'naturalGas': {
                'No Inflation Reduction Act': {year: 0.13 + (year - 2024) * 0.005 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.12 + (year - 2024) * 0.004 for year in range(2024, 2040)}
            }
        },
        
        # Census division level prices (for propane and fuel oil)
        'Pacific': {
            'propane': {
                'No Inflation Reduction Act': {year: 0.25 + (year - 2024) * 0.012 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.24 + (year - 2024) * 0.011 for year in range(2024, 2040)}
            },
            'fuelOil': {
                'No Inflation Reduction Act': {year: 0.28 + (year - 2024) * 0.014 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.26 + (year - 2024) * 0.012 for year in range(2024, 2040)}
            }
        },
        'West South Central': {
            'propane': {
                'No Inflation Reduction Act': {year: 0.22 + (year - 2024) * 0.011 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.21 + (year - 2024) * 0.010 for year in range(2024, 2040)}
            },
            'fuelOil': {
                'No Inflation Reduction Act': {year: 0.26 + (year - 2024) * 0.013 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.24 + (year - 2024) * 0.011 for year in range(2024, 2040)}
            }
        },
        'Middle Atlantic': {
            'propane': {
                'No Inflation Reduction Act': {year: 0.28 + (year - 2024) * 0.014 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.26 + (year - 2024) * 0.012 for year in range(2024, 2040)}
            },
            'fuelOil': {
                'No Inflation Reduction Act': {year: 0.30 + (year - 2024) * 0.015 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.28 + (year - 2024) * 0.013 for year in range(2024, 2040)}
            }
        },
        'South Atlantic': {
            'propane': {
                'No Inflation Reduction Act': {year: 0.24 + (year - 2024) * 0.012 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.22 + (year - 2024) * 0.010 for year in range(2024, 2040)}
            },
            'fuelOil': {
                'No Inflation Reduction Act': {year: 0.28 + (year - 2024) * 0.014 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.26 + (year - 2024) * 0.012 for year in range(2024, 2040)}
            }
        },
        'East North Central': {
            'propane': {
                'No Inflation Reduction Act': {year: 0.26 + (year - 2024) * 0.013 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.24 + (year - 2024) * 0.011 for year in range(2024, 2040)}
            },
            'fuelOil': {
                'No Inflation Reduction Act': {year: 0.29 + (year - 2024) * 0.014 for year in range(2024, 2040)},
                'AEO2023 Reference Case': {year: 0.27 + (year - 2024) * 0.012 for year in range(2024, 2040)}
            }
        }
    }
    
    return mock_prices

@pytest.fixture
def mock_scenario_params(mock_fuel_prices, monkeypatch):
    """
    Mock the define_scenario_params function for testing.
    
    This fixture creates a mock version of the define_scenario_params function
    that returns consistent test values for all scenarios. This avoids having to
    mock the entire modeling_params module.
    
    Args:
        mock_fuel_prices: The mock fuel price data.
        monkeypatch: pytest's monkeypatch fixture for patching functions.
        
    Returns:
        None (the function is patched for the duration of the test).
    """
    def mock_define_scenario_params(menu_mp, policy_scenario):
        """Mock implementation of define_scenario_params."""
        if menu_mp == 0:
            scenario_prefix = "baseline_"
        elif policy_scenario == 'No Inflation Reduction Act':
            scenario_prefix = f"preIRA_mp{menu_mp}_"
        else:
            scenario_prefix = f"iraRef_mp{menu_mp}_"
            
        # Mock the nested return values
        return (
            scenario_prefix,  # scenario_prefix
            "MidCase",        # cambium_scenario
            {},               # lookup_emissions_fossil_fuel
            {},               # lookup_emissions_electricity_climate
            {},               # lookup_emissions_electricity_health
            mock_fuel_prices  # lookup_fuel_prices
        )
    
    # Apply the patch
    monkeypatch.setattr(
        'cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.define_scenario_params',
        mock_define_scenario_params
    )

@pytest.fixture
def benchmark_constants():
    """
    Define benchmark constants for testing computational results.
    
    This fixture provides expected values for various calculations
    to validate computational consistency.
    
    Returns:
        Dictionary with benchmark values.
    """
    return {
        'annual_cost_ca_electricity_2024': 200.0,
        'annual_cost_tx_naturalgas_2024': 200.0,
        'lifetime_heating_costs_baseline': [2000, 4000, 3000, 2400, 0],  # Expected results for each home
        'lifetime_heating_costs_mp8': [1600, 3200, 2400, 2000, 0],
        'lifetime_savings_heating': [400, 800, 600, 400, 0]
    }

@pytest.fixture(params=['heating', 'waterHeating', 'clothesDrying', 'cooking'])
def category(request) -> str:
    """
    Parametrized fixture for equipment categories.
    
    Args:
        request: pytest request object.
        
    Returns:
        String with the current category name.
    """
    return request.param

@pytest.fixture(params=[0, 7, 8, 9, 10])
def menu_mp(request) -> int:
    """
    Parametrized fixture for measure package values.
    
    Args:
        request: pytest request object.
        
    Returns:
        Integer with the current menu_mp value.
    """
    return request.param

@pytest.fixture(params=['No Inflation Reduction Act', 'AEO2023 Reference Case'])
def policy_scenario(request) -> str:
    """
    Parametrized fixture for policy scenarios.
    
    Args:
        request: pytest request object.
        
    Returns:
        String with the current policy scenario.
    """
    return request.param

# -------------------------------------------------------------------------
#              STEP 1: MASK INITIALIZATION TESTS
# -------------------------------------------------------------------------

def test_mask_initialization(sample_homes_df, category):
    """
    Test proper initialization of validation tracking.
    
    This test validates that the mask initialization step correctly identifies
    valid homes based on inclusion flags and retrofit status.
    
    Args:
        sample_homes_df: Sample DataFrame with home data.
        category: Equipment category being tested.
    """
    # Call the initialize_validation_tracking function
    menu_mp = 8  # Using a non-zero menu_mp to test retrofit logic
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        sample_homes_df, category, menu_mp, verbose=False)
    
    # Verify the mask matches the include column
    include_col = f'include_{category}'
    assert valid_mask.equals(sample_homes_df[include_col]), \
        f"Valid mask should match include_{category} column"
    
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
    
    This test validates that the series initialization step correctly creates
    Series objects with zeros for valid homes and NaN for invalid homes.
    
    Args:
        sample_homes_df: Sample DataFrame with home data.
        category: Equipment category being tested.
    """
    # Get the valid mask from the sample data
    valid_mask = sample_homes_df[f'include_{category}']
    
    # Create a retrofit-only series
    result = create_retrofit_only_series(sample_homes_df, valid_mask)
    
    # Verify the result is a pandas Series
    assert isinstance(result, pd.Series), "Result should be a pandas Series"
    assert len(result) == len(sample_homes_df), "Result should have same length as DataFrame"
    
    # Verify valid homes have value 0.0
    for idx in valid_mask.index:
        if valid_mask[idx]:
            assert result[idx] == 0.0, f"Valid home at index {idx} should have value 0.0"
        else:
            assert pd.isna(result[idx]), f"Invalid home at index {idx} should have value NaN"

# -------------------------------------------------------------------------
#              STEP 3: VALID-ONLY CALCULATION TESTS
# -------------------------------------------------------------------------

def test_calculate_annual_fuel_costs_basic(sample_homes_df, mock_fuel_prices, mock_scenario_params, benchmark_constants):
    """
    Test basic calculation of annual fuel costs.
    
    This test validates that the annual fuel cost calculation correctly computes
    costs based on consumption and fuel prices, without validation masking.
    
    Args:
        sample_homes_df: Sample DataFrame with home data.
        mock_fuel_prices: Mock fuel price data.
        mock_scenario_params: Mock scenario parameters (patches define_scenario_params).
        benchmark_constants: Expected benchmark values for computations.
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
    df[f'fuel_type_{category}'] = df[fuel_col].map({'Electricity': 'electricity', 'Natural Gas': 'naturalGas', 
                                                   'Propane': 'propane', 'Fuel Oil': 'fuelOil'})
    is_elec_or_gas = df[f'fuel_type_{category}'].isin(['electricity', 'naturalGas'])
    
    # Call the function without validation mask
    annual_costs, annual_cost_value = calculate_annual_fuel_costs(
        df=df,
        category=category,
        year_label=year_label,
        menu_mp=menu_mp,
        lookup_fuel_prices=mock_fuel_prices,
        policy_scenario=policy_scenario,
        scenario_prefix=scenario_prefix,
        is_elec_or_gas=is_elec_or_gas,
        valid_mask=None  # No validation mask
    )
    
    # Verify the result contains the expected column
    cost_col = f'{scenario_prefix}{year_label}_{category}_fuel_cost'
    assert cost_col in annual_costs, f"Result should contain column '{cost_col}'"
    
    # Verify calculations for specific homes
    # Home 0: CA, Electricity
    expected_ca_cost = df.loc[0, f'baseline_{year_label}_{category}_consumption'] * mock_fuel_prices['CA']['electricity'][policy_scenario][year_label]
    assert abs(annual_costs[cost_col].iloc[0] - expected_ca_cost) < 0.01, \
        f"Cost for CA electricity home should be approximately {expected_ca_cost}"
    
    # Home 1: TX, Natural Gas
    expected_tx_cost = df.loc[1, f'baseline_{year_label}_{category}_consumption'] * mock_fuel_prices['TX']['naturalGas'][policy_scenario][year_label]
    assert abs(annual_costs[cost_col].iloc[1] - expected_tx_cost) < 0.01, \
        f"Cost for TX natural gas home should be approximately {expected_tx_cost}"

def test_calculate_annual_fuel_costs_with_validation(sample_homes_df, mock_fuel_prices, mock_scenario_params):
    """
    Test annual fuel cost calculation with validation masking.
    
    This test validates that the annual fuel cost calculation correctly applies
    validation masking to exclude invalid homes from calculations.
    
    Args:
        sample_homes_df: Sample DataFrame with home data.
        mock_fuel_prices: Mock fuel price data.
        mock_scenario_params: Mock scenario parameters (patches define_scenario_params).
    """
    # Test parameters
    category = 'heating'
    year_label = 2024
    menu_mp = 8  # Measure package
    policy_scenario = 'AEO2023 Reference Case'
    scenario_prefix = f'iraRef_mp{menu_mp}_'
    
    # Get the valid mask from the sample data
    df = sample_homes_df.copy()
    valid_mask = df[f'include_{category}']
    
    # Call the function with validation mask
    annual_costs, annual_cost_value = calculate_annual_fuel_costs(
        df=df,
        category=category,
        year_label=year_label,
        menu_mp=menu_mp,
        lookup_fuel_prices=mock_fuel_prices,
        policy_scenario=policy_scenario,
        scenario_prefix=scenario_prefix,
        is_elec_or_gas=None,  # Not needed for measure packages
        valid_mask=valid_mask  # Apply validation mask
    )
    
    # Verify the result contains the expected column
    cost_col = f'{scenario_prefix}{year_label}_{category}_fuel_cost'
    assert cost_col in annual_costs, f"Result should contain column '{cost_col}'"
    
    # Verify values for invalid homes are zero
    for idx in valid_mask.index:
        if not valid_mask[idx]:
            assert annual_cost_value[idx] == 0.0, \
                f"Invalid home at index {idx} should have fuel cost of 0.0"

# -------------------------------------------------------------------------
#              STEP 4: VALID-ONLY UPDATES TESTS
# -------------------------------------------------------------------------

def test_list_based_collection(sample_homes_df, mock_fuel_prices, mock_scenario_params):
    """
    Test list-based collection of yearly values.
    
    This test validates that the list-based collection approach correctly
    accumulates yearly values and combines them efficiently.
    
    Args:
        sample_homes_df: Sample DataFrame with home data.
        mock_fuel_prices: Mock fuel price data.
        mock_scenario_params: Mock scenario parameters (patches define_scenario_params).
    """
    # Test parameters
    category = 'heating'
    menu_mp = 8  # Measure package
    policy_scenario = 'AEO2023 Reference Case'
    scenario_prefix = f'iraRef_mp{menu_mp}_'
    
    # Get the valid mask from the sample data
    df = sample_homes_df.copy()
    valid_mask = df[f'include_{category}']
    
    # Initialize with zeros for valid homes
    template = create_retrofit_only_series(df, valid_mask)
    
    # Create a list to store yearly costs
    yearly_costs_list = []
    
    # Calculate for multiple years
    years_to_test = [2024, 2025, 2026]
    
    for year_label in years_to_test:
        # Calculate annual costs
        annual_costs, annual_cost_value = calculate_annual_fuel_costs(
            df=df,
            category=category,
            year_label=year_label,
            menu_mp=menu_mp,
            lookup_fuel_prices=mock_fuel_prices,
            policy_scenario=policy_scenario,
            scenario_prefix=scenario_prefix,
            valid_mask=valid_mask
        )
        
        # Add to list
        yearly_costs_list.append(annual_cost_value)
    
    # Convert list to DataFrame and sum
    costs_df = pd.concat(yearly_costs_list, axis=1)
    # lifetime_fuel_costs = costs_df.sum(axis=1)
    lifetime_fuel_costs = costs_df.sum(axis=1, skipna=False)  # Use skipna=False to propagate NaN values

    # Verify length and structure
    assert len(costs_df.columns) == len(years_to_test), \
        f"Combined DataFrame should have {len(years_to_test)} columns"
    
    assert len(lifetime_fuel_costs) == len(df), \
        "Lifetime fuel costs Series should have same length as original DataFrame"
    
    # Verify values for invalid homes are still zero
    for idx in valid_mask.index:
        if not valid_mask[idx]:
            assert lifetime_fuel_costs[idx] == 0.0, \
                f"Invalid home at index {idx} should have lifetime fuel cost of 0.0"

# -------------------------------------------------------------------------
#              STEP 5: FINAL MASKING TESTS
# -------------------------------------------------------------------------

def test_final_masking(sample_homes_df, mock_fuel_prices, mock_scenario_params):
    """
    Test final masking of result columns.
    
    This test validates that the final masking step correctly applies masking
    to all tracked columns based on validation flags.
    
    Args:
        sample_homes_df: Sample DataFrame with home data.
        mock_fuel_prices: Mock fuel price data.
        mock_scenario_params: Mock scenario parameters (patches define_scenario_params).
    """
    # Test parameters
    category = 'heating'
    menu_mp = 8  # Measure package
    policy_scenario = 'AEO2023 Reference Case'
    
    # Get the valid mask from the sample data
    df = sample_homes_df.copy()
    valid_mask = df[f'include_{category}']
    
    # Initialize validation tracking
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
        df, category, menu_mp, verbose=False)
    
    # Create sample results DataFrame with cost columns
    yearly_cols = [f'iraRef_mp8_202{i}_{category}_fuel_cost' for i in range(4, 7)]
    lifetime_col = f'iraRef_mp8_{category}_lifetime_fuel_cost'
    
    results_data = {}
    for col in yearly_cols + [lifetime_col]:
        results_data[col] = np.ones(len(df))
    
    results_df = pd.DataFrame(results_data, index=df.index)
    
    # Track columns for masking
    all_columns_to_mask[category].extend(results_df.columns)
    
    # Apply final masking
    masked_df = apply_final_masking(results_df, all_columns_to_mask, verbose=False)
    
    # Verify values for invalid homes are NaN after masking
    for col in results_df.columns:
        for idx in valid_mask.index:
            if not valid_mask[idx]:
                assert pd.isna(masked_df.loc[idx, col]), \
                    f"Invalid home at index {idx} should have NaN for column '{col}'"

# -------------------------------------------------------------------------
#              INTEGRATION TESTS
# -------------------------------------------------------------------------

def test_lifetime_fuel_costs_basic(multi_year_homes_df, mock_fuel_prices, mock_scenario_params):
    """
    Test basic functionality of calculate_lifetime_fuel_costs.
    
    This test validates that the full function correctly calculates lifetime 
    fuel costs for baseline and measure packages, following the 5-step validation 
    framework.
    
    Args:
        multi_year_homes_df: Sample DataFrame with multi-year data.
        mock_fuel_prices: Mock fuel price data.
        mock_scenario_params: Mock scenario parameters (patches define_scenario_params).
    """
    # Calculate baseline fuel costs first
    menu_mp_baseline = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    df_baseline_main, df_baseline_detailed = calculate_lifetime_fuel_costs(
        df=multi_year_homes_df,
        menu_mp=menu_mp_baseline,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Now calculate measure package costs
    menu_mp = 8
    df_measure_main, df_measure_detailed = calculate_lifetime_fuel_costs(
        df=multi_year_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        df_baseline_costs=df_baseline_main,
        verbose=False
    )
    
    # Verify the main result DataFrame has the expected columns
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        baseline_col = f'baseline_{category}_lifetime_fuel_cost'
        measure_col = f'iraRef_mp8_{category}_lifetime_fuel_cost'
        savings_col = f'iraRef_mp8_{category}_lifetime_savings_fuel_cost'
        
        assert baseline_col in df_baseline_main.columns, \
            f"Baseline result should contain column '{baseline_col}'"
        
        assert measure_col in df_measure_main.columns, \
            f"Measure package result should contain column '{measure_col}'"
        
        assert savings_col in df_measure_main.columns, \
            f"Measure package result should contain column '{savings_col}'"
    
    # Verify values for invalid homes are NaN
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        baseline_col = f'baseline_{category}_lifetime_fuel_cost'
        measure_col = f'iraRef_mp8_{category}_lifetime_fuel_cost'
        valid_mask = multi_year_homes_df[f'include_{category}']
        
        for idx in valid_mask.index:
            if not valid_mask[idx]:
                assert pd.isna(df_baseline_main.loc[idx, baseline_col]), \
                    f"Invalid home at index {idx} should have NaN for column '{baseline_col}'"
                
                assert pd.isna(df_measure_main.loc[idx, measure_col]), \
                    f"Invalid home at index {idx} should have NaN for column '{measure_col}'"

def test_detailed_dataframe_structure(multi_year_homes_df, mock_fuel_prices, mock_scenario_params):
    """
    Test the structure of the detailed DataFrame from calculate_lifetime_fuel_costs.
    
    This test validates that the detailed DataFrame contains the expected columns
    for yearly and lifetime fuel costs.
    
    Args:
        multi_year_homes_df: Sample DataFrame with multi-year data.
        mock_fuel_prices: Mock fuel price data.
        mock_scenario_params: Mock scenario parameters (patches define_scenario_params).
    """
    # Calculate measure package costs
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=multi_year_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify the detailed DataFrame has yearly columns
    category = 'heating'
    for year in range(2024, 2040):
        yearly_col = f'iraRef_mp8_{year}_{category}_fuel_cost'
        
        # Only check years that should be included based on lifetime
        if year <= 2024 + 15:  # EQUIPMENT_SPECS['heating'] = 15
            assert yearly_col in df_detailed.columns, \
                f"Detailed result should contain column '{yearly_col}'"
    
    # Verify the detailed DataFrame also has lifetime columns
    lifetime_col = f'iraRef_mp8_{category}_lifetime_fuel_cost'
    assert lifetime_col in df_detailed.columns, \
        f"Detailed result should contain column '{lifetime_col}'"

def test_consistency_with_baseline(multi_year_homes_df, mock_fuel_prices, mock_scenario_params):
    """
    Test consistency between baseline and measure package calculations.
    
    This test validates that avoided costs are correctly calculated as the
    difference between baseline and measure package costs.
    
    Args:
        multi_year_homes_df: Sample DataFrame with multi-year data.
        mock_fuel_prices: Mock fuel price data.
        mock_scenario_params: Mock scenario parameters (patches define_scenario_params).
    """
    # Calculate baseline costs
    menu_mp_baseline = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    df_baseline_main, _ = calculate_lifetime_fuel_costs(
        df=multi_year_homes_df,
        menu_mp=menu_mp_baseline,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Calculate measure package costs with baseline costs
    menu_mp = 8
    df_measure_main, _ = calculate_lifetime_fuel_costs(
        df=multi_year_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        df_baseline_costs=df_baseline_main,
        verbose=False
    )
    
    # Verify avoided costs are baseline - measure
    category = 'heating'
    baseline_col = f'baseline_{category}_lifetime_fuel_cost'
    measure_col = f'iraRef_mp8_{category}_lifetime_fuel_cost'
    savings_col = f'iraRef_mp8_{category}_lifetime_savings_fuel_cost'
    
    # Get the valid mask from the sample data
    valid_mask = multi_year_homes_df[f'include_{category}']
    
    # Check avoided costs calculation
    for idx in valid_mask.index:
        if valid_mask[idx]:
            baseline_value = df_baseline_main.loc[idx, baseline_col]
            measure_value = df_measure_main.loc[idx, measure_col]
            savings_value = df_measure_main.loc[idx, savings_col]
            
            assert abs((baseline_value - measure_value) - savings_value) < 0.01, \
                f"Avoided costs should be baseline - measure for home at index {idx}"

# -------------------------------------------------------------------------
#              PARAMETRIZED TESTS
# -------------------------------------------------------------------------

def test_across_categories(multi_year_homes_df, mock_fuel_prices, mock_scenario_params, category):
    """
    Test calculation across different equipment categories.
    
    This parametrized test validates that the function correctly handles
    all equipment categories.
    
    Args:
        multi_year_homes_df: Sample DataFrame with multi-year data.
        mock_fuel_prices: Mock fuel price data.
        mock_scenario_params: Mock scenario parameters (patches define_scenario_params).
        category: Parametrized equipment category.
    """
    # Calculate baseline costs
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=multi_year_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify the result contains the expected column for this category
    lifetime_col = f'baseline_{category}_lifetime_fuel_cost'
    assert lifetime_col in df_main.columns, \
        f"Result should contain column '{lifetime_col}'"
    
    # Verify values for invalid homes are NaN
    valid_mask = multi_year_homes_df[f'include_{category}']
    for idx in valid_mask.index:
        if not valid_mask[idx]:
            assert pd.isna(df_main.loc[idx, lifetime_col]), \
                f"Invalid home at index {idx} should have NaN for column '{lifetime_col}'"

def test_across_measure_packages(multi_year_homes_df, mock_fuel_prices, mock_scenario_params, menu_mp):
    """
    Test calculation across different measure packages.
    
    This parametrized test validates that the function correctly handles
    all measure package values.
    
    Args:
        multi_year_homes_df: Sample DataFrame with multi-year data.
        mock_fuel_prices: Mock fuel price data.
        mock_scenario_params: Mock scenario parameters (patches define_scenario_params).
        menu_mp: Parametrized measure package value.
    """
    # Skip some combinations for efficiency (we already test these elsewhere)
    if menu_mp > 9:
        pytest.skip("Skipping higher measure packages for efficiency")
    
    # Calculate fuel costs
    policy_scenario = 'AEO2023 Reference Case'
    category = 'heating'
    
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=multi_year_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Determine expected column prefix
    if menu_mp == 0:
        prefix = 'baseline_'
    else:
        prefix = f'iraRef_mp{menu_mp}_'
    
    # Verify the result contains the expected column for this measure package
    lifetime_col = f'{prefix}{category}_lifetime_fuel_cost'
    assert lifetime_col in df_main.columns, \
        f"Result should contain column '{lifetime_col}'"

def test_across_policy_scenarios(multi_year_homes_df, mock_fuel_prices, mock_scenario_params, policy_scenario):
    """
    Test calculation across different policy scenarios.
    
    This parametrized test validates that the function correctly handles
    all policy scenarios.
    
    Args:
        multi_year_homes_df: Sample DataFrame with multi-year data.
        mock_fuel_prices: Mock fuel price data.
        mock_scenario_params: Mock scenario parameters (patches define_scenario_params).
        policy_scenario: Parametrized policy scenario.
    """
    # Calculate fuel costs for a measure package
    menu_mp = 8
    category = 'heating'
    
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=multi_year_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Determine expected column prefix
    if policy_scenario == 'No Inflation Reduction Act':
        prefix = f'preIRA_mp{menu_mp}_'
    else:
        prefix = f'iraRef_mp{menu_mp}_'
    
    # Verify the result contains the expected column for this policy scenario
    lifetime_col = f'{prefix}{category}_lifetime_fuel_cost'
    assert lifetime_col in df_main.columns, \
        f"Result should contain column '{lifetime_col}'"

# -------------------------------------------------------------------------
#              EDGE CASE TESTS
# -------------------------------------------------------------------------

def test_empty_dataframe(mock_fuel_prices, mock_scenario_params):
    """
    Test calculation with an empty DataFrame.
    
    This test validates that the function handles empty input gracefully.
    
    Args:
        mock_fuel_prices: Mock fuel price data.
        mock_scenario_params: Mock scenario parameters (patches define_scenario_params).
    """
    # Create an empty DataFrame
    df_empty = pd.DataFrame()
    
    # Calculate fuel costs
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    # Expect a KeyError for missing required columns
    with pytest.raises(KeyError) as excinfo:
        calculate_lifetime_fuel_costs(
            df=df_empty,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )
    
    # Verify the error message mentions missing columns
    assert "Required columns missing" in str(excinfo.value), \
        "Error message should mention missing required columns"

def test_all_invalid_homes(multi_year_homes_df, mock_fuel_prices, mock_scenario_params):
    """
    Test calculation when all homes are invalid.
    
    This test validates that the function handles the case where no homes
    are valid for a category.
    
    Args:
        multi_year_homes_df: Sample DataFrame with multi-year data.
        mock_fuel_prices: Mock fuel price data.
        mock_scenario_params: Mock scenario parameters (patches define_scenario_params).
    """
    # Create a DataFrame where all homes are invalid for a category
    df = multi_year_homes_df.copy()
    category = 'heating'
    df[f'include_{category}'] = False
    
    # Calculate fuel costs
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify lifetime column exists but all values are NaN
    lifetime_col = f'iraRef_mp8_{category}_lifetime_fuel_cost'
    assert lifetime_col in df_main.columns, \
        f"Result should contain column '{lifetime_col}'"
    
    assert df_main[lifetime_col].isna().all(), \
        "All values should be NaN when all homes are invalid"

def test_missing_consumption_data(multi_year_homes_df, mock_fuel_prices, mock_scenario_params):
    """
    Test calculation with missing consumption data.
    
    This test validates that the function handles missing consumption data gracefully.
    
    Args:
        multi_year_homes_df: Sample DataFrame with multi-year data.
        mock_fuel_prices: Mock fuel price data.
        mock_scenario_params: Mock scenario parameters (patches define_scenario_params).
    """
    # Create a DataFrame with missing consumption data for one year
    df = multi_year_homes_df.copy()
    category = 'heating'
    year = 2025
    
    # Remove consumption column for this year
    if f'baseline_{year}_{category}_consumption' in df.columns:
        df = df.drop(columns=[f'baseline_{year}_{category}_consumption'])
    
    # Calculate fuel costs
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    # Expect a ValueError for missing consumption column
    with pytest.raises(ValueError) as excinfo:
        calculate_annual_fuel_costs(
            df=df,
            category=category,
            year_label=year,
            menu_mp=menu_mp,
            lookup_fuel_prices=mock_fuel_prices,
            policy_scenario=policy_scenario,
            scenario_prefix='baseline_',
            is_elec_or_gas=df[f'fuel_type_{category}'].isin(['electricity', 'naturalGas']),
            valid_mask=None
        )
    
    # Verify the error message mentions missing consumption column
    assert "Required consumption column" in str(excinfo.value), \
        "Error message should mention missing consumption column"
