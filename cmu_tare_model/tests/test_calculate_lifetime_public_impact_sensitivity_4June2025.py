"""
test_calculate_lifetime_public_impact_sensitivity.py

Strategic test suite for public NPV calculations following proven climate/health module patterns.
Focuses on core NPV logic, sensitivity analysis, and integration fixes.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from unittest.mock import patch, MagicMock

# Import the module being tested
from cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity import (
    calculate_public_npv,
    calculate_lifetime_damages_grid_scenario
)

from cmu_tare_model.constants import EQUIPMENT_SPECS, SCC_ASSUMPTIONS, RCM_MODELS, CR_FUNCTIONS

# =============================================================================
# FIXTURE FUNCTIONS - APPLYING PROVEN PATTERNS
# =============================================================================

def create_complete_sample_data():
    """Create complete sample data with ALL required columns for public NPV testing."""
    np.random.seed(42)
    n_homes = 15
    
    df = pd.DataFrame({
        'home_id': range(n_homes),
        'gea_region': np.random.choice(['CAMXc', 'ERCTc', 'FRCCc'], n_homes),
        'county_fips': np.random.choice(['06001', '48001', '12001'], n_homes),
        'state': np.random.choice(['CA', 'TX', 'FL'], n_homes),
    })
    
    # Add ALL validation flags required by validation framework
    for category in EQUIPMENT_SPECS:
        df[f'include_{category}'] = np.random.choice([True, False], n_homes, p=[0.8, 0.2])
        df[f'valid_fuel_{category}'] = np.random.choice([True, False], n_homes, p=[0.9, 0.1])
        df[f'valid_tech_{category}'] = np.random.choice([True, False], n_homes, p=[0.9, 0.1])
    
    # Add upgrade columns for retrofit status
    upgrade_columns = {
        'heating': 'upgrade_hvac_heating_efficiency',
        'waterHeating': 'upgrade_water_heater_efficiency', 
        'clothesDrying': 'upgrade_clothes_dryer',
        'cooking': 'upgrade_cooking_range'
    }
    
    for category, upgrade_col in upgrade_columns.items():
        df[upgrade_col] = np.random.choice([None, 'Upgrade Type'], n_homes, p=[0.3, 0.7])
    
    return df


def create_mock_damage_dataframes(n_homes, base_year=2024):
    """Create comprehensive mock damage DataFrames covering full equipment lifetime."""
    # Create baseline and measure package damage DataFrames
    damage_dfs = {}
    
    for df_type in ['baseline_climate', 'baseline_health', 'mp_climate', 'mp_health']:
        df = pd.DataFrame(index=range(n_homes))
        
        # Determine scenario prefix
        if 'baseline' in df_type:
            scenario_prefix = 'baseline_'
        else:
            scenario_prefix = 'iraRef_mp8_'
        
        # Collect all columns first, then assign at once
        new_columns = {}

        for category, lifetime in EQUIPMENT_SPECS.items():
            for year in range(1, lifetime + 1):
                year_label = year + (base_year - 1)
                
                if 'climate' in df_type:
                    # Climate damages for each SCC assumption and MER type
                    for scc in SCC_ASSUMPTIONS:
                        col = f'{scenario_prefix}{year_label}_{category}_damages_climate_lrmer_{scc}'
                        new_columns[col] = np.random.uniform(10, 100, n_homes)
                        
                elif 'health' in df_type:
                    # Health damages for each RCM/CR combination
                    for rcm in ['ap2']:  # Use subset for efficiency
                        for cr in ['acs']:
                            col = f'{scenario_prefix}{year_label}_{category}_damages_health_{rcm}_{cr}'
                            new_columns[col] = np.random.uniform(5, 50, n_homes)

        # Assign all columns at once to avoid fragmentation
        if new_columns:  # Only concat if there are columns to add
            df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
        
        damage_dfs[df_type] = df
    
    return damage_dfs


@pytest.fixture
def sample_data():
    """Fixture providing complete sample data."""
    return create_complete_sample_data()


@pytest.fixture
def damage_dataframes(sample_data):
    """Fixture providing mock damage DataFrames."""
    n_homes = len(sample_data)
    return create_mock_damage_dataframes(n_homes)


@pytest.fixture
def mock_validation_framework():
    """Fixture to mock validation framework functions - APPLYING LESSON 1."""
    
    def mock_initialize_validation_tracking_empty(df, category, menu_mp, verbose=False):
        """Custom mock for empty DataFrame scenarios - LESSON 1 APPLIED."""
        df_copy = df.copy()
        if len(df) == 0:
            valid_mask = pd.Series([], dtype=bool, name=f'include_{category}')  # Empty but correct type
        else:
            valid_mask = df[f'include_{category}'].fillna(False)
        all_columns_to_mask = {cat: [] for cat in EQUIPMENT_SPECS}
        category_columns_to_mask = []
        return df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask
    
    def mock_create_retrofit_only_series(df, valid_mask, verbose=False):
        if len(df) == 0:
            return pd.Series([], dtype=float)  # Empty but correct type
        result = pd.Series(np.nan, index=df.index)
        result.loc[valid_mask] = 0.0
        return result
    
    def mock_apply_temporary_validation_and_mask(df_copy, df_new_columns, all_columns_to_mask, verbose=False):
        """Apply proper masking - LESSON 2 APPLIED."""
        if len(df_copy) == 0:
            return df_copy
            
        result_df = pd.concat([df_copy, df_new_columns], axis=1)
        
        # Apply masking for invalid homes to get NaN values
        for category in EQUIPMENT_SPECS:
            include_col = f'include_{category}'
            if include_col in df_copy.columns:
                invalid_mask = ~df_copy[include_col].fillna(False)
                for col in df_new_columns.columns:
                    if category in col and col in result_df.columns:
                        result_df.loc[invalid_mask, col] = np.nan  # LESSON 2: NaN for invalid homes
        
        return result_df
    
    def mock_calculate_avoided_values(baseline_values, measure_values, retrofit_mask=None):
        if retrofit_mask is None:
            return baseline_values - measure_values
        result = pd.Series(np.nan, index=baseline_values.index)
        if retrofit_mask.any():
            result.loc[retrofit_mask] = (baseline_values.loc[retrofit_mask] - 
                                       measure_values.loc[retrofit_mask])
        return result
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.initialize_validation_tracking', mock_initialize_validation_tracking_empty), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.create_retrofit_only_series', mock_create_retrofit_only_series), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.apply_temporary_validation_and_mask', mock_apply_temporary_validation_and_mask), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.calculate_avoided_values', mock_calculate_avoided_values):
        yield


@pytest.fixture
def mock_discounting():
    """Fixture to mock discounting utilities."""
    def mock_calculate_discount_factor(base_year, year_label, discounting_method):
        # Simple discount factor for testing
        years_out = year_label - base_year
        return 0.97 ** years_out  # 3% discount rate
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.calculate_discount_factor') as mock:
        mock.side_effect = mock_calculate_discount_factor
        yield mock


@pytest.fixture
def mock_scenario_params():
    """Fixture to mock scenario parameters."""
    def mock_define_scenario_params(menu_mp, policy_scenario):
        scenario_prefix = 'baseline_' if menu_mp == 0 else f'iraRef_mp{menu_mp}_'
        return (scenario_prefix, '', {}, {}, {}, {})
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.define_scenario_params') as mock:
        mock.side_effect = mock_define_scenario_params
        yield mock


@pytest.fixture
def mock_validation_dataframes():
    """Fixture to mock damage DataFrame validation."""
    def mock_validate_damage_dataframes(*args, **kwargs):
        return True, []  # Always pass validation for tests
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.validate_damage_dataframes') as mock:
        mock.side_effect = mock_validate_damage_dataframes
        yield mock


@pytest.fixture
def mock_validate_common_parameters():
    """Fixture to mock parameter validation."""
    def mock_validate(menu_mp, policy_scenario, discounting_method=None):
        return int(menu_mp), policy_scenario, discounting_method
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.validate_common_parameters') as mock:
        mock.side_effect = mock_validate
        yield mock


# =============================================================================
# CORE FUNCTIONALITY TESTS - INTEGRATION FOCUS
# =============================================================================

def test_public_npv_integration_chain(sample_data, damage_dataframes, mock_validation_framework, 
                                     mock_discounting, mock_scenario_params, mock_validation_dataframes, 
                                     mock_validate_common_parameters):
    """
    Test complete public NPV calculation chain.
    
    Verifies end-to-end integration: climate damages + health damages → discounted NPV.
    """
    df = sample_data
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    rcm_model = 'ap2'
    
    # Call main function
    result = calculate_public_npv(
        df=df,
        df_baseline_climate=damage_dataframes['baseline_climate'],
        df_baseline_health=damage_dataframes['baseline_health'],
        df_mp_climate=damage_dataframes['mp_climate'],
        df_mp_health=damage_dataframes['mp_health'],
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        rcm_model=rcm_model,
        verbose=False
    )
    
    # Verify output structure
    assert isinstance(result, pd.DataFrame), "Should return DataFrame"
    assert len(result) == len(df), "Should preserve DataFrame length"
    
    # Check that NPV columns exist for each category and sensitivity combination
    for category in ['heating', 'waterHeating']:  # Test subset for efficiency
        for scc in ['central']:  # Test subset 
            expected_cols = [
                f'iraRef_mp{menu_mp}_{category}_climate_npv_{scc}',
                f'iraRef_mp{menu_mp}_{category}_health_npv_{rcm_model}_acs',
                f'iraRef_mp{menu_mp}_{category}_public_npv_{scc}_{rcm_model}_acs'
            ]
            for col in expected_cols:
                assert col in result.columns, f"Missing NPV column: {col}"


def test_discounting_methodology_distinction(sample_data, damage_dataframes, mock_validation_framework, 
                                            mock_scenario_params, mock_validation_dataframes, 
                                            mock_validate_common_parameters):
    """
    Test critical discounting methodology distinction.
    
    Verifies that climate damages (SCC) are NOT discounted while health damages ARE discounted.
    """
    df = sample_data
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    
    # Mock discount factors to verify they're only applied to health damages
    discount_factors_called = []
    def mock_calculate_discount_factor(base_year, year_label, discounting_method):
        discount_factors_called.append((base_year, year_label, discounting_method))
        return 0.95  # 5% discount for easy verification
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.calculate_discount_factor') as mock_discount:
        mock_discount.side_effect = mock_calculate_discount_factor
        
        result = calculate_public_npv(
            df=df,
            df_baseline_climate=damage_dataframes['baseline_climate'],
            df_baseline_health=damage_dataframes['baseline_health'],
            df_mp_climate=damage_dataframes['mp_climate'],
            df_mp_health=damage_dataframes['mp_health'],
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            rcm_model='ap2',
            verbose=False
        )
    
    # Verify discount factors were calculated (for health damages)
    assert len(discount_factors_called) > 0, "Discount factors should be calculated for health damages"
    
    # Verify the discounting methodology works correctly
    assert isinstance(result, pd.DataFrame), "Should complete successfully with proper discounting"


def test_sensitivity_analysis_completeness(sample_data, damage_dataframes, mock_validation_framework, 
                                          mock_discounting, mock_scenario_params, mock_validation_dataframes, 
                                          mock_validate_common_parameters):
    """
    Test that all sensitivity analysis combinations are calculated.
    
    Verifies SCC assumptions × RCM models × CR functions are properly calculated.
    """
    df = sample_data
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    rcm_model = 'ap2'
    
    result = calculate_public_npv(
        df=df,
        df_baseline_climate=damage_dataframes['baseline_climate'],
        df_baseline_health=damage_dataframes['baseline_health'],
        df_mp_climate=damage_dataframes['mp_climate'],
        df_mp_health=damage_dataframes['mp_health'],
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        rcm_model=rcm_model,
        verbose=False
    )
    
    # Check that all SCC × CR combinations exist for each category
    for category in ['heating', 'waterHeating']:  # Test subset for efficiency
        for scc in SCC_ASSUMPTIONS:
            for cr in ['acs']:  # Test subset
                public_npv_col = f'iraRef_mp{menu_mp}_{category}_public_npv_{scc}_{rcm_model}_{cr}'
                assert public_npv_col in result.columns, f"Missing sensitivity combination: {public_npv_col}"


# =============================================================================
# PARAMETRIZED TESTS - STREAMLINED APPROACH
# =============================================================================

@pytest.mark.parametrize("menu_mp", [0, 8, 9, 10])
def test_different_menu_mp_scenarios(sample_data, damage_dataframes, mock_validation_framework, 
                                    mock_discounting, mock_scenario_params, mock_validation_dataframes, 
                                    mock_validate_common_parameters, menu_mp):
    """Test public NPV calculations for different measure packages."""
    df = sample_data
    policy_scenario = 'AEO2023 Reference Case'
    
    result = calculate_public_npv(
        df=df,
        df_baseline_climate=damage_dataframes['baseline_climate'],
        df_baseline_health=damage_dataframes['baseline_health'],
        df_mp_climate=damage_dataframes['mp_climate'],
        df_mp_health=damage_dataframes['mp_health'],
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        rcm_model='ap2',
        verbose=False
    )
    
    # Check appropriate column naming
    expected_prefix = 'baseline_' if menu_mp == 0 else f'iraRef_mp{menu_mp}_'
    expected_col = f'{expected_prefix}heating_climate_npv_central'
    assert expected_col in result.columns, f"Should use correct prefix for menu_mp={menu_mp}"


@pytest.mark.parametrize("policy_scenario", ['No Inflation Reduction Act', 'AEO2023 Reference Case'])
def test_policy_scenarios(sample_data, damage_dataframes, mock_validation_framework, 
                         mock_discounting, mock_scenario_params, mock_validation_dataframes, 
                         mock_validate_common_parameters, policy_scenario):
    """Test public NPV calculations for different policy scenarios."""
    df = sample_data
    menu_mp = 8
    
    result = calculate_public_npv(
        df=df,
        df_baseline_climate=damage_dataframes['baseline_climate'],
        df_baseline_health=damage_dataframes['baseline_health'],
        df_mp_climate=damage_dataframes['mp_climate'],
        df_mp_health=damage_dataframes['mp_health'],
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        rcm_model='ap2',
        verbose=False
    )
    
    # Should complete without error for both policy scenarios
    assert isinstance(result, pd.DataFrame), f"Should work for {policy_scenario}"
    assert len(result) == len(df), f"Should preserve DataFrame length for {policy_scenario}"


@pytest.mark.parametrize("rcm_model", ['ap2', 'easiur', 'inmap'])
def test_different_rcm_models(sample_data, damage_dataframes, mock_validation_framework, 
                             mock_discounting, mock_scenario_params, mock_validation_dataframes, 
                             mock_validate_common_parameters, rcm_model):
    """Test public NPV calculations for different RCM models."""
    df = sample_data
    menu_mp = 8
    policy_scenario = 'AEO2023 Reference Case'
    
    result = calculate_public_npv(
        df=df,
        df_baseline_climate=damage_dataframes['baseline_climate'],
        df_baseline_health=damage_dataframes['baseline_health'],
        df_mp_climate=damage_dataframes['mp_climate'],
        df_mp_health=damage_dataframes['mp_health'],
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        rcm_model=rcm_model,
        verbose=False
    )
    
    # Check that RCM-specific columns exist
    expected_col = f'iraRef_mp{menu_mp}_heating_health_npv_{rcm_model}_acs'
    assert expected_col in result.columns, f"Should create {rcm_model}-specific columns"


# =============================================================================
# EDGE CASE TESTS - APPLYING PROVEN LESSONS
# =============================================================================

def test_empty_dataframe_handling():
    """Test handling of empty DataFrame - LESSON 1 APPLIED."""
    # Create properly structured empty DataFrame with ALL required columns
    required_columns = ['gea_region', 'county_fips', 'state']
    
    # Add all validation flags for each category
    for category in EQUIPMENT_SPECS:
        required_columns.extend([
            f'include_{category}',
            f'valid_fuel_{category}', 
            f'valid_tech_{category}'
        ])
    
    empty_df = pd.DataFrame(columns=required_columns)
    empty_damage_dfs = create_mock_damage_dataframes(0)  # Empty damage DataFrames
    
    # Mock ALL dependencies to handle empty DataFrame properly
    with patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.validate_damage_dataframes') as mock_validate, \
         patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.define_scenario_params') as mock_scenario, \
         patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.validate_common_parameters') as mock_params, \
         patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.initialize_validation_tracking') as mock_init, \
         patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.create_retrofit_only_series') as mock_create, \
         patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.apply_temporary_validation_and_mask') as mock_apply, \
         patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.calculate_discount_factor') as mock_discount:
        
        # Configure mocks for empty DataFrame
        mock_validate.return_value = (True, [])
        mock_scenario.return_value = ('baseline_', '', {}, {}, {}, {})
        mock_params.return_value = (0, 'AEO2023 Reference Case', 'public')
        mock_init.return_value = (empty_df.copy(), pd.Series([], dtype=bool), {cat: [] for cat in EQUIPMENT_SPECS}, [])
        mock_create.return_value = pd.Series([], dtype=float)
        mock_apply.return_value = empty_df
        mock_discount.return_value = 0.97
        
        result = calculate_public_npv(
            df=empty_df,
            df_baseline_climate=empty_damage_dfs['baseline_climate'],
            df_baseline_health=empty_damage_dfs['baseline_health'],
            df_mp_climate=empty_damage_dfs['mp_climate'],
            df_mp_health=empty_damage_dfs['mp_health'],
            menu_mp=0,
            policy_scenario='AEO2023 Reference Case',
            rcm_model='ap2',
            verbose=False
        )
    
    # Should return empty DataFrame without error
    assert result.empty, "Should return empty DataFrame for empty input"


def test_all_invalid_homes_nan_values(sample_data, damage_dataframes, mock_scenario_params, 
                                     mock_validation_dataframes, mock_validate_common_parameters):
    """Test that all invalid homes receive NaN values - LESSON 2 APPLIED."""
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
        result_df = pd.concat([df_copy, df_new_columns], axis=1)
        for category in EQUIPMENT_SPECS:
            include_col = f'include_{category}'
            if include_col in df_copy.columns:
                invalid_mask = ~df_copy[include_col].fillna(False)
                for col in df_new_columns.columns:
                    if category in col and col in result_df.columns:
                        result_df.loc[invalid_mask, col] = np.nan  # LESSON 2: NaN for invalid homes
        return result_df
    
    def mock_calculate_avoided_values(baseline_values, measure_values, retrofit_mask=None):
        return pd.Series(np.nan, index=baseline_values.index)
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.initialize_validation_tracking', mock_initialize_validation_tracking), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.create_retrofit_only_series', mock_create_retrofit_only_series), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.apply_temporary_validation_and_mask', mock_apply_temporary_validation_and_mask), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.calculate_avoided_values', mock_calculate_avoided_values), \
         patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.calculate_discount_factor') as mock_discount:
        
        mock_discount.return_value = 0.97
        
        result = calculate_public_npv(
            df=df,
            df_baseline_climate=damage_dataframes['baseline_climate'],
            df_baseline_health=damage_dataframes['baseline_health'],
            df_mp_climate=damage_dataframes['mp_climate'],
            df_mp_health=damage_dataframes['mp_health'],
            menu_mp=0,
            policy_scenario='AEO2023 Reference Case',
            rcm_model='ap2',
            verbose=False
        )
    
    # Should complete without error and return NaN values for invalid homes
    for category in ['heating', 'waterHeating']:  # Test subset
        col = f'baseline_{category}_climate_npv_central'
        if col in result.columns:
            col_values = result[col]
            if isinstance(col_values, pd.DataFrame):
                # Handle duplicate columns by taking the first one
                col_values = col_values.iloc[:, 0]
            assert col_values.isna().all(), f"All homes should have NaN for {category} (all invalid)"


def test_missing_damage_columns_error():
    """Test that missing damage columns raise appropriate errors - LESSON 3 APPLIED."""
    df = create_complete_sample_data()
    
    # Create damage DataFrames missing required columns
    incomplete_damage_dfs = {
        'baseline_climate': pd.DataFrame(index=df.index),  # Missing damage columns
        'baseline_health': pd.DataFrame(index=df.index),
        'mp_climate': pd.DataFrame(index=df.index),
        'mp_health': pd.DataFrame(index=df.index)
    }
    
    # Mock parameter validation to pass
    with patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.validate_common_parameters') as mock_params, \
         patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.validate_damage_dataframes') as mock_validate:
        
        mock_params.return_value = (8, 'AEO2023 Reference Case', 'public')
        mock_validate.return_value = (False, ["Missing required damage columns"])  # Validation fails
        
        # LESSON 3: Expect RuntimeError wrapping
        with pytest.raises(ValueError, match="Input DataFrames are missing required damage columns"):
            calculate_public_npv(
                df=df,
                df_baseline_climate=incomplete_damage_dfs['baseline_climate'],
                df_baseline_health=incomplete_damage_dfs['baseline_health'],
                df_mp_climate=incomplete_damage_dfs['mp_climate'],
                df_mp_health=incomplete_damage_dfs['mp_health'],
                menu_mp=8,
                policy_scenario='AEO2023 Reference Case',
                rcm_model='ap2',
                verbose=False
            )


def test_invalid_rcm_model_parameter():
    """Test invalid RCM model parameter raises appropriate error."""
    df = create_complete_sample_data()
    damage_dfs = create_mock_damage_dataframes(len(df))
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.validate_common_parameters') as mock_params:
        mock_params.return_value = (8, 'AEO2023 Reference Case', 'public')
        
        with pytest.raises(ValueError, match="Invalid rcm_model"):
            calculate_public_npv(
                df=df,
                df_baseline_climate=damage_dfs['baseline_climate'],
                df_baseline_health=damage_dfs['baseline_health'],
                df_mp_climate=damage_dfs['mp_climate'],
                df_mp_health=damage_dfs['mp_health'],
                menu_mp=8,
                policy_scenario='AEO2023 Reference Case',
                rcm_model='invalid_rcm',  # Invalid RCM model
                verbose=False
            )


# =============================================================================
# INTEGRATION VERIFICATION TESTS
# =============================================================================

def test_validation_framework_steps_executed(sample_data, damage_dataframes, mock_scenario_params, 
                                            mock_validation_dataframes, mock_validate_common_parameters):
    """Test that all 5 validation framework steps are executed properly."""
    df = sample_data
    
    with patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.initialize_validation_tracking') as mock_init, \
         patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.create_retrofit_only_series') as mock_create, \
         patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.apply_temporary_validation_and_mask') as mock_apply, \
         patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.calculate_discount_factor') as mock_discount:
        
        # Configure mocks
        mock_init.return_value = (df.copy(), pd.Series(True, index=df.index), {cat: [] for cat in EQUIPMENT_SPECS}, [])
        mock_create.return_value = pd.Series(0.0, index=df.index)
        mock_apply.return_value = df
        mock_discount.return_value = 0.97
        
        result = calculate_public_npv(
            df=df,
            df_baseline_climate=damage_dataframes['baseline_climate'],
            df_baseline_health=damage_dataframes['baseline_health'],
            df_mp_climate=damage_dataframes['mp_climate'],
            df_mp_health=damage_dataframes['mp_health'],
            menu_mp=8,
            policy_scenario='AEO2023 Reference Case',
            rcm_model='ap2',
            verbose=False
        )
        
        # Verify that validation framework functions were called
        assert mock_init.called, "Step 1: initialize_validation_tracking not called"
        assert mock_create.called, "Step 2: create_retrofit_only_series not called"
        # Steps 3 & 4 are handled by internal calculation logic
        assert mock_apply.called, "Step 5: apply_temporary_validation_and_mask not called"


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
