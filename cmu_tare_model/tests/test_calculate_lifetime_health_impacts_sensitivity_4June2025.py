"""
test_calculate_lifetime_health_impacts_sensitivity_4June2025.py

Fixed and simplified pytest test suite for health impact calculations.
Focuses on core functionality, RCM/CR sensitivity analysis, and integration fixes.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from unittest.mock import patch, MagicMock

# Import the module being tested
from cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity import (
    calculate_lifetime_health_impacts,
    calculate_health_damages_for_pair
)

from cmu_tare_model.constants import EQUIPMENT_SPECS, POLLUTANTS, RCM_MODELS, CR_FUNCTIONS

# =============================================================================
# FIXTURE FUNCTIONS - PROPERLY STRUCTURED
# =============================================================================

def create_complete_sample_data():
    """Create complete sample data with ALL required columns for health impact testing."""
    np.random.seed(42)
    n_homes = 10
    
    df = pd.DataFrame({
        'home_id': range(n_homes),
        'county_fips': [f'0600{i:02d}' for i in range(n_homes)],  # Mock county FIPS
        'state': np.random.choice(['CA', 'TX', 'FL'], n_homes),
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
        df[f'base_electricity_{category}_consumption'] = np.random.uniform(100, 400, n_homes)
        for mp in [7, 8, 9, 10]:
            df[f'mp{mp}_{category}_consumption'] = np.random.uniform(50, 150, n_homes)
    
    return df


def create_mock_scenario_params():
    """Create comprehensive mock scenario parameters."""
    def mock_define_scenario_params(menu_mp, policy_scenario):
        scenario_prefix = 'baseline_' if menu_mp == 0 else f'iraRef_mp{menu_mp}_'
        
        # Mock fossil fuel emission factors
        mock_fossil_fuel = {
            'naturalGas': {'so2': 0.0000006, 'nox': 0.000092, 'pm25': 0.000008, 'co2e': 0.000228},
            'propane': {'so2': 0.0000002, 'nox': 0.000142, 'pm25': 0.000006, 'co2e': 0.000276}
        }
        
        # Mock electricity emission factors for health (cover full lifetime)
        mock_electricity_health = {}
        regions = ['CAMXc', 'ERCTc', 'FRCCc']
        
        for region in regions:
            for year in range(2024, 2041):  # Cover full equipment lifetime
                mock_electricity_health[(year, region)] = {
                    'delta_egrid_so2': 0.00001,
                    'delta_egrid_nox': 0.0001, 
                    'delta_egrid_pm25': 0.00005
                }
        
        return (scenario_prefix, '', mock_fossil_fuel, {}, mock_electricity_health, {})
    
    return mock_define_scenario_params


def create_mock_health_lookups():
    """Create comprehensive mock health impact lookups."""
    # Mock county-level health impact values
    mock_health_values = {}
    
    # Create mock data for each county
    counties = [f'0600{i:02d}' for i in range(10)]
    states = ['CA', 'TX', 'FL']
    
    for county_fips in counties:
        for state in states:
            county_key = (county_fips, state)
            mock_health_values[county_key] = {
                'ap2': {'so2': 1000.0, 'nox': 500.0, 'pm25': 2000.0},
                'easiur': {'so2': 1200.0, 'nox': 600.0, 'pm25': 2400.0},
                'inmap': {'so2': 800.0, 'nox': 400.0, 'pm25': 1600.0}
            }
    
    # Add state-level fallback values
    for state in states:
        state_key = ('STATE_AVG', state)
        mock_health_values[state_key] = {
            'ap2': {'so2': 900.0, 'nox': 450.0, 'pm25': 1800.0},
            'easiur': {'so2': 1100.0, 'nox': 550.0, 'pm25': 2200.0},
            'inmap': {'so2': 700.0, 'nox': 350.0, 'pm25': 1400.0}
        }
    
    return mock_health_values


def create_mock_vsl_adjustment():
    """Create comprehensive mock VSL adjustment factors."""
    vsl_factors = {}
    
    # Generate VSL adjustment factors for full equipment lifetime
    for year in range(2024, 2041):
        years_from_base = year - 2024
        vsl_factors[year] = 1.0 * (1.01 ** years_from_base)  # 1% annual growth
    
    return vsl_factors


@pytest.fixture
def sample_data():
    """Fixture providing complete sample data."""
    return create_complete_sample_data()


@pytest.fixture
def mock_scenario_params():
    """Fixture to mock scenario parameters."""
    with patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.define_scenario_params') as mock:
        mock.side_effect = create_mock_scenario_params()
        yield mock


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
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.initialize_validation_tracking', mock_initialize_validation_tracking), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.create_retrofit_only_series', mock_create_retrofit_only_series), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.apply_temporary_validation_and_mask', mock_apply_temporary_validation_and_mask), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.apply_final_masking', mock_apply_final_masking), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.calculate_avoided_values', mock_calculate_avoided_values):
        yield


@pytest.fixture
def mock_health_lookups():
    """Fixture to mock health impact lookups."""
    mock_values = create_mock_health_lookups()
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_fossil_fuel_acs', mock_values), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_fossil_fuel_h6c', mock_values), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_electricity_acs', mock_values), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_electricity_h6c', mock_values):
        yield mock_values


@pytest.fixture
def mock_vsl_adjustment():
    """Fixture to mock VSL adjustment."""
    mock_vsl_data = create_mock_vsl_adjustment()
    with patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_vsl_adjustment', mock_vsl_data):
        yield mock_vsl_data


@pytest.fixture
def mock_fossil_fuel_emissions():
    """Fixture to mock fossil fuel emissions calculation."""
    def mock_emissions(df, category, year_label, lookup_emissions_fossil_fuel, menu_mp, retrofit_mask, verbose=False):
        # Return mock emissions Series for each pollutant
        n_homes = len(df)
        return {
            'so2': pd.Series(np.random.uniform(0.001, 0.01, n_homes), index=df.index),
            'nox': pd.Series(np.random.uniform(0.01, 0.1, n_homes), index=df.index),
            'pm25': pd.Series(np.random.uniform(0.001, 0.02, n_homes), index=df.index),
            'co2e': pd.Series(np.random.uniform(0.1, 0.5, n_homes), index=df.index)
        }
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.calculate_fossil_fuel_emissions') as mock:
        mock.side_effect = mock_emissions
        yield mock


@pytest.fixture
def mock_hdd_consumption():
    """Fixture to mock HDD consumption utilities."""
    def mock_get_electricity_consumption(df, category, year_label, menu_mp):
        # Return mock electricity consumption
        n_homes = len(df)
        return pd.Series(np.random.uniform(100, 300, n_homes), index=df.index)
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.get_electricity_consumption_for_year') as mock:
        mock.side_effect = mock_get_electricity_consumption
        yield mock


@pytest.fixture
def mock_health_impact_fallback():
    """Fixture to mock health impact fallback function."""
    def mock_fallback(lookup_dict, county_key, model, pollutant, debug=False):
        # Return mock health impact value with proper fallback logic
        value = lookup_dict.get(county_key, {}).get(model, {}).get(pollutant)
        if value is not None:
            return value
        
        # Try state fallback if county not found
        if county_key and len(county_key) >= 2:
            state = county_key[1]
            state_key = ('STATE_AVG', state)
            state_value = lookup_dict.get(state_key, {}).get(model, {}).get(pollutant)
            if state_value is not None:
                return state_value
        
        # Default fallback value
        return 1000.0
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.get_health_impact_with_fallback') as mock:
        mock.side_effect = mock_fallback
        yield mock


@pytest.fixture
def mock_validate_common_parameters():
    """Fixture to mock parameter validation."""
    def mock_validate(menu_mp, policy_scenario, discounting_method=None):
        return int(menu_mp), policy_scenario, discounting_method
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.validate_common_parameters') as mock:
        mock.side_effect = mock_validate
        yield mock


# =============================================================================
# CORE FUNCTIONALITY TESTS - INTEGRATION FOCUS
# =============================================================================

def test_fossil_fuel_emissions_integration_fix(sample_data, mock_scenario_params, mock_fossil_fuel_emissions, mock_vsl_adjustment, mock_validation_framework, mock_health_lookups, mock_health_impact_fallback, mock_hdd_consumption, mock_validate_common_parameters):
    """
    Test that fossil fuel emissions are called with year_label parameter.
    
    This tests the critical fix where year_label parameter was added to
    resolve integration failures between modules.
    """
    df = sample_data
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call main function
    df_main, df_detailed = calculate_lifetime_health_impacts(
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


def test_hdd_consumption_integration(sample_data, mock_scenario_params, mock_fossil_fuel_emissions, mock_hdd_consumption, mock_vsl_adjustment, mock_validation_framework, mock_health_lookups, mock_health_impact_fallback, mock_validate_common_parameters):
    """
    Test that HDD consumption utility is properly integrated.
    
    This tests the integration of get_electricity_consumption_for_year
    instead of manual consumption calculations.
    """
    df = sample_data
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call main function
    df_main, df_detailed = calculate_lifetime_health_impacts(
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


def test_rcm_cr_sensitivity_analysis(sample_data, mock_scenario_params, mock_fossil_fuel_emissions, mock_hdd_consumption, mock_health_lookups, mock_vsl_adjustment, mock_health_impact_fallback, mock_validation_framework, mock_validate_common_parameters):
    """
    Test that all RCM/CR sensitivity analysis combinations are calculated.
    
    Verifies RCM models (AP2, EASIUR, InMAP) × CR functions (ACS, H6C)
    are properly calculated for each equipment category.
    """
    df = sample_data
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, df_detailed = calculate_lifetime_health_impacts(
        df=df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Check that all RCM/CR combinations exist
    for category in ['heating', 'waterHeating']:  # Test subset for efficiency
        for rcm in RCM_MODELS:
            for cr in CR_FUNCTIONS:
                damages_col = f'baseline_{category}_lifetime_damages_health_{rcm}_{cr}'
                assert damages_col in df_main.columns, f"Missing health damages: {damages_col}"


def test_health_damages_calculation_logic(sample_data, mock_scenario_params, mock_health_lookups, mock_vsl_adjustment, mock_health_impact_fallback, mock_hdd_consumption):
    """
    Test the core health damages calculation logic.
    
    Tests the calculate_health_damages_for_pair function directly
    to verify proper MSC application and VSL adjustments.
    """
    df = sample_data
    category = 'heating'
    year_label = 2024
    rcm = 'ap2'
    cr = 'acs'
    
    # Mock electricity health emission factors
    mock_electricity_health = {
        (2024, 'CAMXc'): {
            'delta_egrid_so2': 0.00001, 'delta_egrid_nox': 0.0001, 'delta_egrid_pm25': 0.00005
        }
    }
    
    # Mock fossil fuel emissions
    mock_fossil_emissions = {
        'so2': pd.Series([0.001, 0.002, 0.0015], index=df.index[:3]),
        'nox': pd.Series([0.01, 0.02, 0.015], index=df.index[:3]),
        'pm25': pd.Series([0.005, 0.01, 0.007], index=df.index[:3]),
        'co2e': pd.Series([0.2, 0.3, 0.25], index=df.index[:3])
    }
    
    # Mock electricity consumption
    mock_consumption = pd.Series([200, 300, 250], index=df.index[:3])
    
    # Call the function
    health_results = calculate_health_damages_for_pair(
        df=df.iloc[:3],  # Use subset for clarity
        category=category,
        year_label=year_label,
        lookup_emissions_electricity_health=mock_electricity_health,
        scenario_prefix='baseline_',
        total_fossil_fuel_emissions=mock_fossil_emissions,
        menu_mp=0,
        rcm=rcm,
        cr=cr
    )
    
    # Verify results structure
    assert isinstance(health_results, dict), "Should return results dictionary"
    
    # Check that damages are calculated for each pollutant (excluding CO2e)
    health_pollutants = [p for p in POLLUTANTS if p != 'co2e']
    for pollutant in health_pollutants:
        col_name = f'baseline_{year_label}_{category}_damages_{pollutant}_{rcm}_{cr}'
        assert col_name in health_results, f"Should have damages for {pollutant}"
        assert len(health_results[col_name]) == 3, f"Should have 3 values for {pollutant}"
    
    # Check overall health damages
    overall_col = f'baseline_{year_label}_{category}_damages_health_{rcm}_{cr}'
    assert overall_col in health_results, "Should have overall health damages"


def test_county_key_creation(sample_data, mock_scenario_params, mock_fossil_fuel_emissions, mock_hdd_consumption, mock_health_lookups, mock_vsl_adjustment, mock_health_impact_fallback, mock_validation_framework, mock_validate_common_parameters):
    """
    Test that county_key is properly created for vectorized lookups.
    
    This ensures the county-level health impact lookups work correctly.
    """
    df = sample_data.copy()
    
    # Remove county_key if it exists to test creation
    if 'county_key' in df.columns:
        df = df.drop(columns=['county_key'])
    
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, df_detailed = calculate_lifetime_health_impacts(
        df=df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Should complete without error even without pre-existing county_key
    assert isinstance(df_main, pd.DataFrame), "Should work without pre-existing county_key"
    assert len(df_main) == len(df), "Should preserve DataFrame length"


# =============================================================================
# PARAMETRIZED TESTS - STREAMLINED APPROACH
# =============================================================================

@pytest.mark.parametrize("category", ['heating', 'waterHeating', 'clothesDrying', 'cooking'])
def test_all_categories_calculation(sample_data, mock_scenario_params, mock_fossil_fuel_emissions, mock_hdd_consumption, mock_health_lookups, mock_vsl_adjustment, mock_health_impact_fallback, mock_validation_framework, mock_validate_common_parameters, category):
    """Test health calculations for all equipment categories."""
    df = sample_data
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    # Ensure some homes are valid for this category
    df[f'include_{category}'] = [True] * (len(df) // 2) + [False] * (len(df) - len(df) // 2)
    
    df_main, df_detailed = calculate_lifetime_health_impacts(
        df=df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Check category-specific results exist
    damages_col = f'baseline_{category}_lifetime_damages_health_ap2_acs'
    assert damages_col in df_main.columns, f"Should calculate health damages for {category}"


@pytest.mark.parametrize("rcm", RCM_MODELS)
def test_all_rcm_models(sample_data, mock_scenario_params, mock_fossil_fuel_emissions, mock_hdd_consumption, mock_health_lookups, mock_vsl_adjustment, mock_health_impact_fallback, mock_validation_framework, mock_validate_common_parameters, rcm):
    """Test health calculations for all RCM models."""
    df = sample_data
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, df_detailed = calculate_lifetime_health_impacts(
        df=df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Check that this RCM model is included in results
    damages_col = f'baseline_heating_lifetime_damages_health_{rcm}_acs'
    assert damages_col in df_main.columns, f"Should calculate damages for RCM {rcm}"


@pytest.mark.parametrize("cr", CR_FUNCTIONS)
def test_all_cr_functions(sample_data, mock_scenario_params, mock_fossil_fuel_emissions, mock_hdd_consumption, mock_health_lookups, mock_vsl_adjustment, mock_health_impact_fallback, mock_validation_framework, mock_validate_common_parameters, cr):
    """Test health calculations for all concentration-response functions."""
    df = sample_data
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, df_detailed = calculate_lifetime_health_impacts(
        df=df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Check that this CR function is included in results
    damages_col = f'baseline_heating_lifetime_damages_health_ap2_{cr}'
    assert damages_col in df_main.columns, f"Should calculate damages for CR function {cr}"


@pytest.mark.parametrize("menu_mp", [0, 8, 9, 10])
def test_different_menu_mp_scenarios(sample_data, mock_scenario_params, mock_fossil_fuel_emissions, mock_hdd_consumption, mock_health_lookups, mock_vsl_adjustment, mock_health_impact_fallback, mock_validation_framework, mock_validate_common_parameters, menu_mp):
    """Test health calculations for different measure packages."""
    df = sample_data
    policy_scenario = 'AEO2023 Reference Case'
    
    df_main, df_detailed = calculate_lifetime_health_impacts(
        df=df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Check appropriate column naming
    expected_prefix = 'baseline_' if menu_mp == 0 else f'iraRef_mp{menu_mp}_'
    expected_col = f'{expected_prefix}heating_lifetime_damages_health_ap2_acs'
    assert expected_col in df_main.columns, f"Should use correct prefix for menu_mp={menu_mp}"


# =============================================================================
# EDGE CASE TESTS - ESSENTIAL COVERAGE
# =============================================================================

def test_empty_dataframe(mock_validate_common_parameters):
    """Test handling of empty DataFrame."""
    # Create properly structured empty DataFrame with ALL required columns
    required_columns = ['county_fips', 'state', 'gea_region', 'census_division']
    
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
        return ('baseline_', '', {}, {}, {}, {})
    
    def mock_fossil_fuel_emissions(df, category, year_label, lookup_emissions_fossil_fuel, menu_mp, retrofit_mask, verbose=False):
        # Return empty emissions for empty DataFrame
        return {
            'so2': pd.Series([], dtype=float),
            'nox': pd.Series([], dtype=float),
            'pm25': pd.Series([], dtype=float),
            'co2e': pd.Series([], dtype=float)
        }
    
    def mock_get_electricity_consumption(df, category, year_label, menu_mp):
        return pd.Series([], dtype=float)  # Empty consumption
    
    mock_vsl_data = create_mock_vsl_adjustment()
    mock_health_values = create_mock_health_lookups()
    
    def mock_health_impact_fallback(lookup_dict, county_key, model, pollutant, debug=False):
        return 1000.0  # Default fallback value
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.define_scenario_params', mock_define_scenario_params), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_vsl_adjustment', mock_vsl_data), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_fossil_fuel_acs', mock_health_values), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_fossil_fuel_h6c', mock_health_values), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_electricity_acs', mock_health_values), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_electricity_h6c', mock_health_values), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.get_health_impact_with_fallback', mock_health_impact_fallback), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.calculate_fossil_fuel_emissions', mock_fossil_fuel_emissions), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.get_electricity_consumption_for_year', mock_get_electricity_consumption), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.initialize_validation_tracking', mock_initialize_validation_tracking), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.create_retrofit_only_series', mock_create_retrofit_only_series), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.apply_temporary_validation_and_mask', mock_apply_temporary_validation_and_mask), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.apply_final_masking', mock_apply_final_masking):
        
        df_main, df_detailed = calculate_lifetime_health_impacts(
            df=empty_df,
            menu_mp=0,
            policy_scenario='AEO2023 Reference Case',
            verbose=False
        )
    
    # Should return empty DataFrames
    assert df_main.empty, "Should return empty DataFrame for empty input"
    assert df_detailed.empty, "Should return empty detailed DataFrame for empty input"


def test_all_invalid_homes(sample_data, mock_scenario_params, mock_fossil_fuel_emissions, mock_hdd_consumption, mock_health_lookups, mock_vsl_adjustment, mock_health_impact_fallback, mock_validate_common_parameters):
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
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.initialize_validation_tracking', mock_initialize_validation_tracking), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.create_retrofit_only_series', mock_create_retrofit_only_series), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.apply_temporary_validation_and_mask', mock_apply_temporary_validation_and_mask), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.apply_final_masking', mock_apply_final_masking), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.calculate_avoided_values') as mock_avoided:
        
        # Mock calculate_avoided_values to return NaN for invalid homes
        def mock_avoided_values(baseline_values, measure_values, retrofit_mask=None):
            return pd.Series(np.nan, index=baseline_values.index)
        mock_avoided.side_effect = mock_avoided_values
        
        df_main, df_detailed = calculate_lifetime_health_impacts(
            df=df,
            menu_mp=0,
            policy_scenario='AEO2023 Reference Case',
            verbose=False
        )
    
    # Should complete without error and return NaN values
    for category in ['heating', 'waterHeating']:  # Test subset
        col = f'baseline_{category}_lifetime_damages_health_ap2_acs'
        if col in df_main.columns:
            assert df_main[col].isna().all(), f"All homes should have NaN for {category}"


def test_missing_vsl_adjustment_error(sample_data, mock_scenario_params, mock_fossil_fuel_emissions, mock_validate_common_parameters):
    """Test handling of missing VSL adjustment factor."""
    df = sample_data
    
    # Mock VSL lookup to be missing a year (2024)
    incomplete_vsl_data = {2025: 1.01, 2026: 1.02}  # Missing 2024
    
    # Mock validation framework functions to avoid other errors
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
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_vsl_adjustment', incomplete_vsl_data), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.initialize_validation_tracking', mock_initialize_validation_tracking), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.create_retrofit_only_series', mock_create_retrofit_only_series):
        
        # Should raise RuntimeError (wrapping the original KeyError)
        with pytest.raises(RuntimeError, match="VSL adjustment factor not found"):
            calculate_lifetime_health_impacts(
                df=df,
                menu_mp=0,
                policy_scenario='AEO2023 Reference Case',
                verbose=False
            )


def test_validation_masking_essential(sample_data, mock_scenario_params, mock_fossil_fuel_emissions, mock_hdd_consumption, mock_health_lookups, mock_vsl_adjustment, mock_health_impact_fallback, mock_validation_framework, mock_validate_common_parameters):
    """Test essential validation masking functionality."""
    df = sample_data.copy()
    category = 'heating'
    
    # Create explicit valid/invalid pattern
    mid_point = len(df) // 2
    df[f'include_{category}'] = [True] * mid_point + [False] * (len(df) - mid_point)
    
    df_main, df_detailed = calculate_lifetime_health_impacts(
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
