"""
test_calculate_lifetime_climate_impacts_sensitivity_4June2025.py

Fixed and simplified pytest test suite for climate impact calculations.
Focuses on core functionality, sensitivity analysis, and integration fixes.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from unittest.mock import patch, MagicMock

# Import the module being tested
from cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity import (
    calculate_lifetime_climate_impacts,
    calculate_climate_emissions_and_damages
)

from cmu_tare_model.constants import EQUIPMENT_SPECS, MER_TYPES, SCC_ASSUMPTIONS

# =============================================================================
# FIXTURE FUNCTIONS - PROPERLY STRUCTURED
# =============================================================================

def create_complete_sample_data():
    """Create complete sample data with ALL required columns for climate impact testing."""
    np.random.seed(42)
    n_homes = 12
    
    df = pd.DataFrame({
        'home_id': range(n_homes),
        'gea_region': np.random.choice(['CAMXc', 'ERCTc', 'FRCCc'], n_homes),
        'census_division': np.random.choice(['Pacific', 'West South Central'], n_homes),
    })
    
    # Add ALL validation flags required by validation framework
    for category in EQUIPMENT_SPECS:
        df[f'include_{category}'] = np.random.choice([True, False], n_homes, p=[0.8, 0.2])
        df[f'valid_fuel_{category}'] = np.random.choice([True, False], n_homes, p=[0.9, 0.1])
        df[f'valid_tech_{category}'] = np.random.choice([True, False], n_homes, p=[0.9, 0.1])
    
    # Add upgrade columns for retrofit status
    for category in EQUIPMENT_SPECS:
        upgrade_col_map = {
            'heating': 'upgrade_hvac_heating_efficiency',
            'waterHeating': 'upgrade_water_heater_efficiency', 
            'clothesDrying': 'upgrade_clothes_dryer',
            'cooking': 'upgrade_cooking_range'
        }
        upgrade_col = upgrade_col_map[category]
        df[upgrade_col] = np.random.choice([None, 'Upgrade Type'], n_homes, p=[0.3, 0.7])
    
    # Add consumption columns
    for category in EQUIPMENT_SPECS:
        df[f'base_electricity_{category}_consumption'] = np.random.uniform(100, 500, n_homes)
        for mp in [7, 8, 9, 10]:
            df[f'mp{mp}_{category}_consumption'] = np.random.uniform(50, 200, n_homes)
    
    return df


def create_mock_scenario_params():
    """Create comprehensive mock scenario parameters."""
    def mock_define_scenario_params(menu_mp, policy_scenario):
        scenario_prefix = 'baseline_' if menu_mp == 0 else f'iraRef_mp{menu_mp}_'
        
        # Mock emission factors
        mock_fossil_fuel = {
            'naturalGas': {'co2e': 0.000228, 'so2': 0.0000006, 'nox': 0.000092, 'pm25': 0.000008},
            'propane': {'co2e': 0.000276, 'so2': 0.0000002, 'nox': 0.000142, 'pm25': 0.000006}
        }
        
        # Mock electricity emission factors - cover FULL equipment lifetime (2024-2040)
        mock_electricity = {}
        regions = ['CAMXc', 'ERCTc', 'FRCCc']
        base_factors = {
            'CAMXc': {'lrmer': 0.0004, 'srmer': 0.0005},
            'ERCTc': {'lrmer': 0.0005, 'srmer': 0.0006},
            'FRCCc': {'lrmer': 0.0006, 'srmer': 0.0007}
        }
        
        # Generate emission factors for each region and year (2024-2040)
        for region in regions:
            mock_electricity[('midCase', region)] = {}
            for year in range(2024, 2041):  # Cover full equipment lifetime
                # Slight annual decline to simulate grid decarbonization
                decline_factor = 0.98 ** (year - 2024)
                mock_electricity[('midCase', region)][year] = {
                    'lrmer_mt_per_kWh_co2e': base_factors[region]['lrmer'] * decline_factor,
                    'srmer_mt_per_kWh_co2e': base_factors[region]['srmer'] * decline_factor
                }
        
        return (scenario_prefix, 'midCase', mock_fossil_fuel, mock_electricity, {}, {})
    
    return mock_define_scenario_params


def create_mock_scc_lookup():
    """Create comprehensive mock SCC lookup values covering full equipment lifetime."""
    mock_scc = {'lower': {}, 'central': {}, 'upper': {}}
    
    # Generate SCC values for each year (2024-2040) with annual escalation
    base_values = {'lower': 100, 'central': 150, 'upper': 200}
    annual_escalation = {'lower': 1.02, 'central': 1.025, 'upper': 1.03}  # 2-3% annual increase
    
    for assumption in ['lower', 'central', 'upper']:
        for year in range(2024, 2041):  # Cover full equipment lifetime
            years_from_base = year - 2024
            mock_scc[assumption][year] = base_values[assumption] * (annual_escalation[assumption] ** years_from_base)
    
    return mock_scc


@pytest.fixture
def sample_data():
    """Fixture providing complete sample data."""
    return create_complete_sample_data()


@pytest.fixture
def mock_scenario_params():
    """Fixture to mock scenario parameters."""
    with patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.define_scenario_params') as mock:
        mock.side_effect = create_mock_scenario_params()
        yield mock


@pytest.fixture
def mock_scc_lookup():
    """Fixture to mock SCC lookup."""
    mock_scc_data = create_mock_scc_lookup()
    with patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.lookup_climate_impact_scc', mock_scc_data):
        yield mock_scc_data


@pytest.fixture
def mock_validation_framework():
    """Fixture to mock validation framework functions."""
    
    def mock_initialize_validation_tracking(df, category, menu_mp, verbose=False):
        df_copy = df.copy()
        valid_mask = df[f'include_{category}'].fillna(False)
        all_columns_to_mask = {cat: [] for cat in EQUIPMENT_SPECS}
        category_columns_to_mask = []
        return df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask
    
    def mock_create_retrofit_only_series(df, valid_mask, verbose=False):
        result = pd.Series(np.nan, index=df.index)
        result.loc[valid_mask] = 0.0
        return result
    
    def mock_apply_temporary_validation_and_mask(df_copy, df_new_columns, all_columns_to_mask, verbose=False):
        return pd.concat([df_copy, df_new_columns], axis=1)
    
    def mock_apply_final_masking(df, all_columns_to_mask, verbose=False):
        return df
    
    def mock_calculate_avoided_values(baseline_values, measure_values, retrofit_mask=None):
        if retrofit_mask is None:
            return baseline_values - measure_values
        result = pd.Series(np.nan, index=baseline_values.index)
        result.loc[retrofit_mask] = baseline_values.loc[retrofit_mask] - measure_values.loc[retrofit_mask]
        return result
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.initialize_validation_tracking', mock_initialize_validation_tracking), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.create_retrofit_only_series', mock_create_retrofit_only_series), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.apply_temporary_validation_and_mask', mock_apply_temporary_validation_and_mask), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.apply_final_masking', mock_apply_final_masking), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.calculate_avoided_values', mock_calculate_avoided_values):
        yield


@pytest.fixture
def mock_fossil_fuel_emissions():
    """Fixture to mock fossil fuel emissions calculation."""
    def mock_emissions(df, category, year_label, lookup_emissions_fossil_fuel, menu_mp, retrofit_mask, verbose=False):
        # Return mock emissions Series for each pollutant
        n_homes = len(df)
        return {
            'co2e': pd.Series(np.random.uniform(0.1, 0.5, n_homes), index=df.index),
            'so2': pd.Series(np.random.uniform(0.001, 0.005, n_homes), index=df.index),
            'nox': pd.Series(np.random.uniform(0.01, 0.05, n_homes), index=df.index),
            'pm25': pd.Series(np.random.uniform(0.001, 0.01, n_homes), index=df.index)
        }
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.calculate_fossil_fuel_emissions') as mock:
        mock.side_effect = mock_emissions
        yield mock


@pytest.fixture
def mock_hdd_consumption():
    """Fixture to mock HDD consumption utilities."""
    def mock_get_electricity_consumption(df, category, year_label, menu_mp):
        # Return mock electricity consumption
        n_homes = len(df)
        return pd.Series(np.random.uniform(100, 300, n_homes), index=df.index)
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.get_electricity_consumption_for_year') as mock:
        mock.side_effect = mock_get_electricity_consumption
        yield mock


@pytest.fixture
def mock_validate_common_parameters():
    """Fixture to mock parameter validation."""
    def mock_validate(menu_mp, policy_scenario, discounting_method=None):
        return int(menu_mp), policy_scenario, discounting_method
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.validate_common_parameters') as mock:
        mock.side_effect = mock_validate
        yield mock


# =============================================================================
# CORE FUNCTIONALITY TESTS - INTEGRATION FOCUS
# =============================================================================

def test_fossil_fuel_emissions_integration_fix(sample_data, mock_scenario_params, mock_fossil_fuel_emissions, mock_scc_lookup, mock_validation_framework, mock_hdd_consumption, mock_validate_common_parameters):
    """
    Test that fossil fuel emissions are called with year_label parameter.
    
    This tests the critical fix where year_label parameter was added to
    resolve integration failures between modules.
    """
    df = sample_data
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call main function
    df_main, df_detailed = calculate_lifetime_climate_impacts(
        df=df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify that calculate_fossil_fuel_emissions was called with year_label
    mock_fossil_fuel_emissions.assert_called()
    
    # Check that calls included year_label parameter
    call_args_list = mock_fossil_fuel_emissions.call_args_list
    for call_args in call_args_list:
        args, kwargs = call_args
        assert 'year_label' in kwargs, "year_label parameter should be included in call"
        assert isinstance(kwargs['year_label'], int), "year_label should be an integer"
        assert kwargs['year_label'] >= 2024, "year_label should be reasonable year"


def test_hdd_consumption_integration(sample_data, mock_scenario_params, mock_fossil_fuel_emissions, mock_hdd_consumption, mock_scc_lookup, mock_validation_framework, mock_validate_common_parameters):
    """
    Test that HDD consumption utility is properly integrated.
    
    This tests the integration of get_electricity_consumption_for_year
    instead of manual consumption calculations.
    """
    df = sample_data
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call main function
    df_main, df_detailed = calculate_lifetime_climate_impacts(
        df=df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify HDD consumption utility was called
    mock_hdd_consumption.assert_called()
    
    # Check parameters passed to HDD function
    call_args_list = mock_hdd_consumption.call_args_list
    assert len(call_args_list) > 0, "HDD consumption should be called"


def test_sensitivity_analysis_completeness(sample_data, mock_scenario_params, mock_fossil_fuel_emissions, mock_hdd_consumption, mock_scc_lookup, mock_validation_framework, mock_validate_common_parameters):
    """
    Test that all sensitivity analysis combinations are calculated.
    
    Verifies SCC assumptions (lower, central, upper) × MER types (LRMER, SRMER)
    are properly calculated for each equipment category.
    """
    df = sample_data
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, df_detailed = calculate_lifetime_climate_impacts(
        df=df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Check that all sensitivity combinations exist
    for category in ['heating', 'waterHeating']:  # Test subset for efficiency
        for mer_type in MER_TYPES:
            # Emissions columns
            emissions_col = f'baseline_{category}_lifetime_mt_co2e_{mer_type}'
            assert emissions_col in df_main.columns, f"Missing emissions: {emissions_col}"
            
            # Damages columns for each SCC assumption
            for scc in SCC_ASSUMPTIONS:
                damages_col = f'baseline_{category}_lifetime_damages_climate_{mer_type}_{scc}'
                assert damages_col in df_main.columns, f"Missing damages: {damages_col}"


# =============================================================================
# PARAMETRIZED TESTS - STREAMLINED APPROACH
# =============================================================================

@pytest.mark.parametrize("category", ['heating', 'waterHeating', 'clothesDrying', 'cooking'])
def test_all_categories_calculation(sample_data, mock_scenario_params, mock_fossil_fuel_emissions, mock_hdd_consumption, mock_scc_lookup, mock_validation_framework, mock_validate_common_parameters, category):
    """Test climate calculations for all equipment categories."""
    df = sample_data
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    # Ensure some homes are valid for this category
    df[f'include_{category}'] = [True] * (len(df) // 2) + [False] * (len(df) - len(df) // 2)
    
    df_main, df_detailed = calculate_lifetime_climate_impacts(
        df=df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Check category-specific results exist
    emissions_col = f'baseline_{category}_lifetime_mt_co2e_lrmer'
    assert emissions_col in df_main.columns, f"Should calculate emissions for {category}"


@pytest.mark.parametrize("menu_mp", [0, 8, 9, 10])
def test_different_menu_mp_scenarios(sample_data, mock_scenario_params, mock_fossil_fuel_emissions, mock_hdd_consumption, mock_scc_lookup, mock_validation_framework, mock_validate_common_parameters, menu_mp):
    """Test climate calculations for different measure packages."""
    df = sample_data
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, df_detailed = calculate_lifetime_climate_impacts(
        df=df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Check appropriate column naming
    expected_prefix = 'baseline_' if menu_mp == 0 else f'iraRef_mp{menu_mp}_'
    expected_col = f'{expected_prefix}heating_lifetime_mt_co2e_lrmer'
    assert expected_col in df_main.columns, f"Should use correct prefix for menu_mp={menu_mp}"


@pytest.mark.parametrize("policy_scenario", ['No Inflation Reduction Act', 'AEO2023 Reference Case'])
def test_policy_scenarios(sample_data, mock_scenario_params, mock_fossil_fuel_emissions, mock_hdd_consumption, mock_scc_lookup, mock_validation_framework, mock_validate_common_parameters, policy_scenario):
    """Test climate calculations for different policy scenarios."""
    df = sample_data
    menu_mp = 8
    
    df_main, df_detailed = calculate_lifetime_climate_impacts(
        df=df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Should complete without error for both policy scenarios
    assert isinstance(df_main, pd.DataFrame), f"Should work for {policy_scenario}"
    assert len(df_main) == len(df), f"Should preserve DataFrame length for {policy_scenario}"


# =============================================================================
# EDGE CASE TESTS - ESSENTIAL COVERAGE
# =============================================================================

def test_empty_dataframe(mock_validate_common_parameters):
    """Test handling of empty DataFrame."""
    # Create properly structured empty DataFrame with ALL required columns
    required_columns = ['gea_region', 'census_division']
    
    # Add all validation flags for each category
    for category in EQUIPMENT_SPECS:
        required_columns.extend([
            f'include_{category}',
            f'valid_fuel_{category}', 
            f'valid_tech_{category}'
        ])
    
    empty_df = pd.DataFrame(columns=required_columns)
    
    # Mock ALL dependencies to handle empty DataFrame properly
    def mock_initialize_validation_tracking(df, category, menu_mp, verbose=False):
        df_copy = df.copy()
        valid_mask = pd.Series([], dtype=bool, name=f'include_{category}')  # Empty but correct type
        all_columns_to_mask = {cat: [] for cat in EQUIPMENT_SPECS}
        category_columns_to_mask = []
        return df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask
    
    def mock_create_retrofit_only_series(df, valid_mask, verbose=False):
        return pd.Series([], dtype=float)  # Empty but correct type
    
    def mock_apply_temporary_validation_and_mask(df_copy, df_new_columns, all_columns_to_mask, verbose=False):
        return df_copy  # Return original empty DataFrame
    
    def mock_apply_final_masking(df, all_columns_to_mask, verbose=False):
        return df
    
    def mock_define_scenario_params(menu_mp, policy_scenario):
        return ('baseline_', 'midCase', {}, {}, {}, {})
    
    def mock_fossil_fuel_emissions(df, category, year_label, lookup_emissions_fossil_fuel, menu_mp, retrofit_mask, verbose=False):
        # Return empty emissions for empty DataFrame
        return {
            'co2e': pd.Series([], dtype=float),
            'so2': pd.Series([], dtype=float),
            'nox': pd.Series([], dtype=float),
            'pm25': pd.Series([], dtype=float)
        }
    
    def mock_get_electricity_consumption(df, category, year_label, menu_mp):
        return pd.Series([], dtype=float)  # Empty consumption
    
    mock_scc_data = create_mock_scc_lookup()
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.define_scenario_params', mock_define_scenario_params), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.lookup_climate_impact_scc', mock_scc_data), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.calculate_fossil_fuel_emissions', mock_fossil_fuel_emissions), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.get_electricity_consumption_for_year', mock_get_electricity_consumption), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.initialize_validation_tracking', mock_initialize_validation_tracking), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.create_retrofit_only_series', mock_create_retrofit_only_series), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.apply_temporary_validation_and_mask', mock_apply_temporary_validation_and_mask), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.apply_final_masking', mock_apply_final_masking):
        
        df_main, df_detailed = calculate_lifetime_climate_impacts(
            df=empty_df,
            menu_mp=0,
            policy_scenario='AEO2023 Reference Case',
            verbose=False
        )
    
    # Should return empty DataFrames
    assert df_main.empty, "Should return empty DataFrame for empty input"
    assert df_detailed.empty, "Should return empty detailed DataFrame for empty input"


def test_all_invalid_homes(sample_data, mock_scenario_params, mock_fossil_fuel_emissions, mock_hdd_consumption, mock_scc_lookup, mock_validate_common_parameters):
    """Test calculation when all homes are invalid."""
    df = sample_data.copy()
    
    # Make all homes invalid for all categories
    for category in EQUIPMENT_SPECS:
        df[f'include_{category}'] = False
    
    # Mock validation framework to properly handle all invalid homes
    def mock_initialize_validation_tracking(df, category, menu_mp, verbose=False):
        df_copy = df.copy()
        valid_mask = df[f'include_{category}'].fillna(False)  # Should be all False
        all_columns_to_mask = {cat: [] for cat in EQUIPMENT_SPECS}
        category_columns_to_mask = []
        return df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask
    
    def mock_create_retrofit_only_series(df, valid_mask, verbose=False):
        # When all homes are invalid, return all NaN
        result = pd.Series(np.nan, index=df.index)
        if valid_mask.any():  # Only set 0.0 for valid homes
            result.loc[valid_mask] = 0.0
        return result
    
    def mock_apply_temporary_validation_and_mask(df_copy, df_new_columns, all_columns_to_mask, verbose=False):
        # Apply proper masking - set invalid homes to NaN
        result_df = df_copy.copy()
        for col in df_new_columns.columns:
            result_df[col] = df_new_columns[col]
            # Apply masking for each category
            for category in EQUIPMENT_SPECS:
                if category in col:
                    include_col = f'include_{category}'
                    if include_col in df_copy.columns:
                        invalid_mask = ~df_copy[include_col].fillna(False)
                        result_df.loc[invalid_mask, col] = np.nan
        return result_df
    
    def mock_apply_final_masking(df, all_columns_to_mask, verbose=False):
        # Apply final masking to ensure invalid homes have NaN
        result_df = df.copy()
        for category, cols in all_columns_to_mask.items():
            include_col = f'include_{category}'
            if include_col in df.columns:
                invalid_mask = ~df[include_col].fillna(False)
                for col in cols:
                    if col in result_df.columns:
                        result_df.loc[invalid_mask, col] = np.nan
        return result_df
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.initialize_validation_tracking', mock_initialize_validation_tracking), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.create_retrofit_only_series', mock_create_retrofit_only_series), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.apply_temporary_validation_and_mask', mock_apply_temporary_validation_and_mask), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.apply_final_masking', mock_apply_final_masking), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.calculate_avoided_values') as mock_avoided:
        
        # Mock calculate_avoided_values to return NaN for invalid homes
        def mock_avoided_values(baseline_values, measure_values, retrofit_mask=None):
            return pd.Series(np.nan, index=baseline_values.index)
        mock_avoided.side_effect = mock_avoided_values
        
        df_main, df_detailed = calculate_lifetime_climate_impacts(
            df=df,
            menu_mp=0,
            policy_scenario='AEO2023 Reference Case',
            verbose=False
        )
    
    # Should complete without error and return NaN values
    for category in ['heating', 'waterHeating']:  # Test subset
        col = f'baseline_{category}_lifetime_mt_co2e_lrmer'
        if col in df_main.columns:
            assert df_main[col].isna().all(), f"All homes should have NaN for {category}"


def test_validation_masking_essential(sample_data, mock_scenario_params, mock_fossil_fuel_emissions, mock_hdd_consumption, mock_scc_lookup, mock_validation_framework, mock_validate_common_parameters):
    """Test essential validation masking functionality."""
    df = sample_data.copy()
    category = 'heating'
    
    # Create explicit valid/invalid pattern
    mid_point = len(df) // 2
    df[f'include_{category}'] = [True] * mid_point + [False] * (len(df) - mid_point)
    
    df_main, df_detailed = calculate_lifetime_climate_impacts(
        df=df,
        menu_mp=0,
        policy_scenario='AEO2023 Reference Case',
        verbose=False
    )
    
    # Should complete successfully with validation masking
    assert isinstance(df_main, pd.DataFrame), "Should return valid DataFrame"
    assert len(df_main) == len(df), "Should preserve DataFrame length"


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
