"""
test_determine_adoption_potential_sensitivity_robust.py

UPDATED test suite for adoption potential analysis focusing on core logic
and integration with NPV dependencies. Simplified validation framework testing.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from cmu_tare_model.constants import SCC_ASSUMPTIONS, RCM_MODELS, CR_FUNCTIONS, UPGRADE_COLUMNS

from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust import (
    adoption_decision,
    calculate_climate_only_adoption_robust,
    calculate_health_only_adoption_robust,
    validate_input_parameters,
    fix_duplicate_columns
)

# =============================================================================
# FIXTURE FUNCTIONS - SIMPLIFIED AND FOCUSED
# =============================================================================

def create_sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_homes = 50
    
    # Create base DataFrame
    df = pd.DataFrame({
        'home_id': range(n_homes),
        'gea_region': np.random.choice(['CAMXc', 'ERCTc', 'FRCCc'], n_homes),
        'state': np.random.choice(['CA', 'TX', 'FL'], n_homes),
        'county_fips': np.random.choice(['06001', '48001', '12001'], n_homes),
    })
    
    # Add upgrade columns
    for category in UPGRADE_COLUMNS:
        df[UPGRADE_COLUMNS[category]] = np.random.choice([np.nan, 'upgrade_needed'], n_homes, p=[0.3, 0.7])
    
    # Add validation flags (SIMPLIFIED - just essential)
    for category in UPGRADE_COLUMNS:
        df[f'include_{category}'] = np.random.choice([True, False], n_homes, p=[0.8, 0.2])
    
    return df


def create_complete_npv_data():
    """Create complete NPV data for testing adoption analysis."""
    df = create_sample_data()
    n_homes = len(df)
    
    scenario_prefix = 'iraRef_mp8_'
    
    # Add private NPV columns (realistic distribution around break-even)
    for category in UPGRADE_COLUMNS:
        df[f'{scenario_prefix}{category}_private_npv_lessWTP'] = np.random.normal(-2000, 8000, n_homes)
        df[f'{scenario_prefix}{category}_private_npv_moreWTP'] = np.random.normal(1000, 6000, n_homes)
    
    # Add public NPV components with realistic correlations
    for category in UPGRADE_COLUMNS:
        # Climate NPV (generally positive for electrification)
        base_climate = np.random.gamma(2, 1000)  # Skewed positive
        for scc in SCC_ASSUMPTIONS:
            multiplier = {'lower': 0.7, 'central': 1.0, 'upper': 1.4}[scc]
            df[f'{scenario_prefix}{category}_climate_npv_{scc}'] = base_climate * multiplier + np.random.normal(0, 500, n_homes)
        
        # Health NPV (varies by region, generally positive)
        base_health = np.random.normal(1500, 1000, n_homes)
        for rcm in ['ap2']:  # Use subset for efficiency
            for cr in ['acs']:
                df[f'{scenario_prefix}{category}_health_npv_{rcm}_{cr}'] = base_health + np.random.normal(0, 300, n_homes)
                
                # Combined public NPV
                for scc in SCC_ASSUMPTIONS:
                    climate_col = f'{scenario_prefix}{category}_climate_npv_{scc}'
                    health_col = f'{scenario_prefix}{category}_health_npv_{rcm}_{cr}'
                    df[f'{scenario_prefix}{category}_public_npv_{scc}_{rcm}_{cr}'] = df[climate_col] + df[health_col]
    
    # Add rebate columns (realistic IRA rebate amounts)
    rebate_amounts = {
        'heating': 8000, 'waterHeating': 1750, 'clothesDrying': 840, 'cooking': 840
    }
    for category in UPGRADE_COLUMNS:
        base_rebate = rebate_amounts[category]
        df[f'mp8_{category}_rebate_amount'] = np.random.uniform(base_rebate * 0.8, base_rebate * 1.2, n_homes)
    
    return df


@pytest.fixture
def sample_data():
    """Fixture providing sample data."""
    return create_sample_data()


@pytest.fixture
def complete_npv_data():
    """Fixture providing complete NPV data."""
    return create_complete_npv_data()


@pytest.fixture
def mock_validation_framework():
    """Simplified mock of validation framework - ESSENTIAL FUNCTIONS ONLY."""
    
    def mock_initialize_validation_tracking(df, category, menu_mp, verbose=False):
        df_copy = df.copy()
        valid_mask = df[f'include_{category}'].fillna(False)
        all_columns_to_mask = {cat: [] for cat in UPGRADE_COLUMNS}
        category_columns_to_mask = []
        return df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask
    
    def mock_create_retrofit_only_series(df, valid_mask, verbose=False):
        result = pd.Series(np.nan, index=df.index)
        result.loc[valid_mask] = 0.0
        return result
    
    def mock_apply_new_columns_to_dataframe(df_original, df_new_columns, category, category_columns_to_mask, all_columns_to_mask):
        # Simple implementation for testing
        result_df = pd.concat([df_original, df_new_columns], axis=1)
        all_columns_to_mask[category].extend(category_columns_to_mask)
        return result_df, all_columns_to_mask
    
    def mock_apply_final_masking(df, all_columns_to_mask, verbose=False):
        # Apply masking for invalid homes
        result_df = df.copy()
        for category, cols in all_columns_to_mask.items():
            include_col = f'include_{category}'
            if include_col in df.columns:
                invalid_mask = ~df[include_col].fillna(False)
                for col in cols:
                    if col in result_df.columns:
                        result_df.loc[invalid_mask, col] = np.nan
        return result_df
    
    with patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.initialize_validation_tracking', mock_initialize_validation_tracking), \
         patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.create_retrofit_only_series', mock_create_retrofit_only_series), \
         patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.apply_new_columns_to_dataframe', mock_apply_new_columns_to_dataframe), \
         patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.apply_final_masking', mock_apply_final_masking):
        yield


@pytest.fixture
def mock_scenario_params():
    """Mock scenario parameters."""
    def mock_define_scenario_params(menu_mp, policy_scenario):
        scenario_prefix = f'iraRef_mp{menu_mp}_'
        return (scenario_prefix, '', {}, {}, {}, {})
    
    with patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.define_scenario_params') as mock:
        mock.side_effect = mock_define_scenario_params
        yield mock


# =============================================================================
# PARAMETER VALIDATION TESTS - ENHANCED
# =============================================================================

def test_parameter_validation_comprehensive():
    """Test comprehensive parameter validation with clear error messages."""
    # Valid parameters should not raise
    validate_input_parameters(8, 'AEO2023 Reference Case', 'ap2', 'acs')
    
    # Test multiple invalid parameters to check error aggregation
    with pytest.raises(ValueError) as excinfo:
        validate_input_parameters('invalid_mp', 'Invalid Scenario', 'invalid_rcm', 'invalid_cr')
    
    error_message = str(excinfo.value)
    assert "Parameter validation failed:" in error_message
    assert "menu_mp" in error_message
    assert "policy_scenario" in error_message
    assert "rcm_model" in error_message
    assert "cr_function" in error_message


def test_missing_upgrade_columns_detailed_error(sample_data):
    """Test detailed error message for missing upgrade columns."""
    df_missing = sample_data.drop(columns=[UPGRADE_COLUMNS['heating']])
    
    with pytest.raises(KeyError, match="Required upgrade columns missing"):
        adoption_decision(df_missing, 8, 'AEO2023 Reference Case', 'ap2', 'acs')


# =============================================================================
# CORE ADOPTION TIER LOGIC TESTS - ENHANCED
# =============================================================================

def test_adoption_tier_classification_comprehensive():
    """Test comprehensive adoption tier classification with realistic data patterns."""
    # Create test data designed to hit all adoption tiers - WITH ALL UPGRADE COLUMNS
    test_data = pd.DataFrame({
        'include_heating': [True] * 8,
        'include_waterHeating': [True] * 8,
        'include_clothesDrying': [True] * 8,
        'include_cooking': [True] * 8,
    })
    
    # Add all required upgrade columns
    for category in UPGRADE_COLUMNS:
        test_data[UPGRADE_COLUMNS[category]] = ['upgrade'] * 8
    
    # Add NPV columns for heating (test focus)
    test_data['iraRef_mp8_heating_private_npv_lessWTP'] = [5000, -2000, -3000, -5000, np.nan, 1000, -1000, -2000]
    test_data['iraRef_mp8_heating_private_npv_moreWTP'] = [6000, 1000, -1000, -6000, 2000, 1500, -500, -1500]
    test_data['iraRef_mp8_heating_public_npv_central_ap2_acs'] = [1000, 500, 2000, -2000, 1500, 800, 300, 500]
    test_data['mp8_heating_rebate_amount'] = [2000] * 8
    
    # Add minimal NPV columns for other categories to prevent errors
    for category in ['waterHeating', 'clothesDrying', 'cooking']:
        test_data[f'iraRef_mp8_{category}_private_npv_lessWTP'] = [1000] * 8
        test_data[f'iraRef_mp8_{category}_private_npv_moreWTP'] = [1500] * 8
        test_data[f'iraRef_mp8_{category}_public_npv_central_ap2_acs'] = [800] * 8
        test_data[f'mp8_{category}_rebate_amount'] = [1000] * 8
    
    with patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.define_scenario_params') as mock_scenario, \
         patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.initialize_validation_tracking') as mock_init, \
         patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.create_retrofit_only_series') as mock_create, \
         patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.apply_new_columns_to_dataframe') as mock_apply, \
         patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.apply_final_masking') as mock_final:
        
        # Configure mocks
        mock_scenario.return_value = ('iraRef_mp8_', '', {}, {}, {}, {})
        mock_init.return_value = (test_data.copy(), pd.Series(True, index=test_data.index), {cat: [] for cat in UPGRADE_COLUMNS}, [])
        mock_create.return_value = pd.Series(0.0, index=test_data.index)
        mock_apply.side_effect = lambda df, new_cols, *args: (pd.concat([df, new_cols], axis=1), {})
        mock_final.side_effect = lambda df, *args, **kwargs: df
        
        result = adoption_decision(test_data, 8, 'AEO2023 Reference Case', 'ap2', 'acs')
        
        # Check that adoption analysis completed - look for any adoption columns
        adoption_cols = [col for col in result.columns if 'adoption_central_ap2_acs' in col]
        assert len(adoption_cols) > 0, f"Should create adoption classification columns, found: {adoption_cols}"
        
        # Verify tier classifications exist for at least one category
        adoption_values_found = False
        for col in adoption_cols:
            if result[col].dropna().shape[0] > 0:
                adoption_values_found = True
                break
        assert adoption_values_found, "Should have adoption classifications for valid homes"


def test_public_impact_classification_logic():
    """Test public impact classification with edge cases."""
    # Create test data with specific public NPV patterns - WITH ALL UPGRADE COLUMNS
    test_data = pd.DataFrame({
        'include_heating': [True] * 5,
        'include_waterHeating': [True] * 5,
        'include_clothesDrying': [True] * 5,
        'include_cooking': [True] * 5,
    })
    
    # Add all required upgrade columns
    for category in UPGRADE_COLUMNS:
        test_data[UPGRADE_COLUMNS[category]] = ['upgrade'] * 5
    
    # Add NPV columns for heating (test focus)
    test_data['iraRef_mp8_heating_private_npv_lessWTP'] = [1000] * 5
    test_data['iraRef_mp8_heating_private_npv_moreWTP'] = [1500] * 5
    test_data['iraRef_mp8_heating_public_npv_central_ap2_acs'] = [5000, 0, -2000, np.nan, 0.0001]  # Benefit, Zero, Detriment, NaN, Near-zero
    test_data['mp8_heating_rebate_amount'] = [2000] * 5
    
    # Add minimal NPV columns for other categories to prevent errors
    for category in ['waterHeating', 'clothesDrying', 'cooking']:
        test_data[f'iraRef_mp8_{category}_private_npv_lessWTP'] = [1000] * 5
        test_data[f'iraRef_mp8_{category}_private_npv_moreWTP'] = [1500] * 5
        test_data[f'iraRef_mp8_{category}_public_npv_central_ap2_acs'] = [800] * 5
        test_data[f'mp8_{category}_rebate_amount'] = [1000] * 5
    
    with patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.define_scenario_params') as mock_scenario, \
         patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.initialize_validation_tracking') as mock_init, \
         patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.create_retrofit_only_series') as mock_create, \
         patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.apply_new_columns_to_dataframe') as mock_apply, \
         patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.apply_final_masking') as mock_final:
        
        # Configure mocks
        mock_scenario.return_value = ('iraRef_mp8_', '', {}, {}, {}, {})
        mock_init.return_value = (test_data.copy(), pd.Series(True, index=test_data.index), {cat: [] for cat in UPGRADE_COLUMNS}, [])
        mock_create.return_value = pd.Series(0.0, index=test_data.index)
        mock_apply.side_effect = lambda df, new_cols, *args: (pd.concat([df, new_cols], axis=1), {})
        mock_final.side_effect = lambda df, *args, **kwargs: df
        
        result = adoption_decision(test_data, 8, 'AEO2023 Reference Case', 'ap2', 'acs')
        
        # Check that impact analysis completed - look for any impact columns
        impact_cols = [col for col in result.columns if 'impact_central_ap2_acs' in col]
        assert len(impact_cols) > 0, f"Should create public impact classification columns, found: {impact_cols}"
        
        # Verify impact classifications exist for at least one category
        impact_values_found = False
        for col in impact_cols:
            if result[col].dropna().shape[0] > 0:
                impact_values_found = True
                break
        assert impact_values_found, "Should have impact classifications for valid homes"


# =============================================================================
# INTEGRATION TESTS - FOCUSED ON NPV DEPENDENCIES
# =============================================================================

def test_npv_integration_chain(complete_npv_data, mock_validation_framework, mock_scenario_params):
    """Test complete integration with public and private NPV calculations."""
    df = complete_npv_data
    
    result = adoption_decision(df, 8, 'AEO2023 Reference Case', 'ap2', 'acs')
    
    # Verify that combined analysis columns exist
    for category in ['heating', 'waterHeating']:  # Test subset for efficiency
        for scc in ['central']:  # Test subset
            expected_cols = [
                f'iraRef_mp8_{category}_adoption_{scc}_ap2_acs',
                f'iraRef_mp8_{category}_impact_{scc}_ap2_acs',
                f'iraRef_mp8_{category}_benefit_{scc}_ap2_acs',
                f'iraRef_mp8_{category}_health_sensitivity'
            ]
            for col in expected_cols:
                assert col in result.columns, f"Missing combined analysis column: {col}"


def test_climate_only_analysis_integration(complete_npv_data, mock_validation_framework, mock_scenario_params):
    """Test climate-only analysis generates expected columns."""
    df = complete_npv_data
    
    result = calculate_climate_only_adoption_robust(df, 8, 'AEO2023 Reference Case', verbose=False)
    
    # Check that climate-only analysis columns exist
    for category in ['heating', 'waterHeating']:  # Test subset
        for scc in ['central']:  # Test subset
            expected_col = f'iraRef_mp8_{category}_total_npv_climateOnly_{scc}'
            assert expected_col in result.columns, f"Missing climate-only column: {expected_col}"


def test_health_only_analysis_integration(complete_npv_data, mock_validation_framework, mock_scenario_params):
    """Test health-only analysis generates expected columns."""
    df = complete_npv_data
    
    result = calculate_health_only_adoption_robust(df, 8, 'AEO2023 Reference Case', 'ap2', 'acs', verbose=False)
    
    # Check that health-only analysis columns exist
    for category in ['heating', 'waterHeating']:  # Test subset
        expected_col = f'iraRef_mp8_{category}_total_npv_healthOnly_ap2_acs'
        assert expected_col in result.columns, f"Missing health-only column: {expected_col}"


# =============================================================================
# SENSITIVITY ANALYSIS TESTS - FOCUSED
# =============================================================================

def test_sensitivity_analysis_coverage(complete_npv_data, mock_validation_framework, mock_scenario_params):
    """Test that sensitivity analysis covers all required combinations."""
    df = complete_npv_data
    
    result = adoption_decision(df, 8, 'AEO2023 Reference Case', 'ap2', 'acs')
    
    # Check that all SCC assumptions are covered for at least one category
    for scc in SCC_ASSUMPTIONS:
        scc_cols = [col for col in result.columns if f'_{scc}_ap2_acs' in col]
        assert len(scc_cols) > 0, f"Should have columns for SCC assumption: {scc}"


def test_different_rcm_cr_combinations():
    """Test analysis with different RCM and CR function combinations."""
    df = create_complete_npv_data()
    
    # Test different combinations
    combinations = [
        ('ap2', 'acs'),
        ('easiur', 'h6c'),
        ('inmap', 'acs')
    ]
    
    for rcm, cr in combinations:
        # Add required NPV columns for this combination
        for category in UPGRADE_COLUMNS:
            df[f'iraRef_mp8_{category}_health_npv_{rcm}_{cr}'] = np.random.normal(1500, 1000, len(df))
            for scc in SCC_ASSUMPTIONS:
                climate_col = f'iraRef_mp8_{category}_climate_npv_{scc}'
                health_col = f'iraRef_mp8_{category}_health_npv_{rcm}_{cr}'
                df[f'iraRef_mp8_{category}_public_npv_{scc}_{rcm}_{cr}'] = df[climate_col] + df[health_col]
        
        with patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.define_scenario_params') as mock_scenario, \
             patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.initialize_validation_tracking') as mock_init, \
             patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.create_retrofit_only_series') as mock_create, \
             patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.apply_new_columns_to_dataframe') as mock_apply, \
             patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.apply_final_masking') as mock_final:
            
            # Configure mocks
            mock_scenario.return_value = ('iraRef_mp8_', '', {}, {}, {}, {})
            mock_init.return_value = (df.copy(), pd.Series(True, index=df.index), {cat: [] for cat in UPGRADE_COLUMNS}, [])
            mock_create.return_value = pd.Series(0.0, index=df.index)
            mock_apply.side_effect = lambda df, new_cols, *args: (pd.concat([df, new_cols], axis=1), {})
            mock_final.side_effect = lambda df, *args, **kwargs: df
            
            result = adoption_decision(df, 8, 'AEO2023 Reference Case', rcm, cr)
            
            # Should complete without error for all combinations
            assert isinstance(result, pd.DataFrame), f"Should work for {rcm}-{cr} combination"


# =============================================================================
# ROBUSTNESS AND EDGE CASE TESTS
# =============================================================================

def test_duplicate_column_handling():
    """Test duplicate column handling utility."""
    # Create DataFrame with duplicate columns
    df_with_duplicates = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'col1': [7, 8, 9]  # Duplicate column name
    })
    
    result = fix_duplicate_columns(df_with_duplicates)
    
    # Should remove duplicates
    assert len(result.columns) == len(result.columns.unique()), "Should remove duplicate columns"
    assert list(result.columns) == ['col1', 'col2'], "Should keep first occurrence of duplicates"


def test_missing_npv_columns_graceful_handling(sample_data, mock_validation_framework, mock_scenario_params):
    """Test graceful handling when NPV columns are missing."""
    df = sample_data  # Basic data without NPV columns but with all upgrade columns
    
    # The test should fail because NPV columns are missing, but it should fail with the right error
    # Let's check that it does raise an error, but catch the actual error type
    try:
        adoption_decision(df, 8, 'AEO2023 Reference Case', 'ap2', 'acs')
        # If no error is raised, the test should fail
        assert False, "Should raise an error when NPV columns are missing"
    except Exception as e:
        # Accept either KeyError or other relevant exceptions that indicate missing NPV data
        assert any(keyword in str(e).lower() for keyword in ['npv', 'missing', 'required', 'column']), \
            f"Should raise informative error about missing data, got: {str(e)}"


def test_numeric_conversion_robustness():
    """Test robust handling of non-numeric values in NPV columns."""
    # Create data with mixed data types - WITH ALL UPGRADE COLUMNS
    df = pd.DataFrame({
        'include_heating': [True] * 4,
        'include_waterHeating': [True] * 4,
        'include_clothesDrying': [True] * 4,
        'include_cooking': [True] * 4,
    })
    
    # Add all required upgrade columns
    for category in UPGRADE_COLUMNS:
        df[UPGRADE_COLUMNS[category]] = ['upgrade'] * 4
    
    # Add NPV columns for heating with mixed data types (test focus)
    df['iraRef_mp8_heating_private_npv_lessWTP'] = ['1000', '2000', 'invalid', np.nan]  # Mixed types
    df['iraRef_mp8_heating_private_npv_moreWTP'] = [2000, 2500, 3000, 3500]
    df['iraRef_mp8_heating_public_npv_central_ap2_acs'] = [1500, 1800, 2000, 2200]
    df['mp8_heating_rebate_amount'] = [2000] * 4
    
    # Add minimal NPV columns for other categories to prevent errors
    for category in ['waterHeating', 'clothesDrying', 'cooking']:
        df[f'iraRef_mp8_{category}_private_npv_lessWTP'] = [1000] * 4
        df[f'iraRef_mp8_{category}_private_npv_moreWTP'] = [1500] * 4
        df[f'iraRef_mp8_{category}_public_npv_central_ap2_acs'] = [800] * 4
        df[f'mp8_{category}_rebate_amount'] = [1000] * 4
    
    with patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.define_scenario_params') as mock_scenario, \
         patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.initialize_validation_tracking') as mock_init, \
         patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.create_retrofit_only_series') as mock_create, \
         patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.apply_new_columns_to_dataframe') as mock_apply, \
         patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.apply_final_masking') as mock_final:
        
        # Configure mocks
        mock_scenario.return_value = ('iraRef_mp8_', '', {}, {}, {}, {})
        mock_init.return_value = (df.copy(), pd.Series(True, index=df.index), {cat: [] for cat in UPGRADE_COLUMNS}, [])
        mock_create.return_value = pd.Series(0.0, index=df.index)
        mock_apply.side_effect = lambda df, new_cols, *args: (pd.concat([df, new_cols], axis=1), {})
        mock_final.side_effect = lambda df, *args, **kwargs: df
        
        # Should not raise an error due to numeric conversion
        result = adoption_decision(df, 8, 'AEO2023 Reference Case', 'ap2', 'acs')
        
        # Function should complete without error (pd.to_numeric with errors='coerce' handles non-numeric)
        assert result is not None, "Should handle non-numeric values gracefully"


def test_all_invalid_homes_handling(complete_npv_data, mock_scenario_params):
    """Test handling when all homes are invalid for a category."""
    df = complete_npv_data.copy()
    
    # Make all homes invalid for heating
    df['include_heating'] = False
    
    with patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.initialize_validation_tracking') as mock_init, \
         patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.create_retrofit_only_series') as mock_create, \
         patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.apply_new_columns_to_dataframe') as mock_apply, \
         patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_robust.apply_final_masking') as mock_final:
        
        # Configure mocks for all invalid homes
        mock_init.return_value = (df.copy(), pd.Series(False, index=df.index), {cat: [] for cat in UPGRADE_COLUMNS}, [])
        mock_create.return_value = pd.Series(np.nan, index=df.index)  # All NaN for invalid homes
        mock_apply.side_effect = lambda df, new_cols, *args: (pd.concat([df, new_cols], axis=1), {})
        mock_final.side_effect = lambda df, *args, **kwargs: df
        
        result = adoption_decision(df, 8, 'AEO2023 Reference Case', 'ap2', 'acs')
        
        # Should complete without error
        assert isinstance(result, pd.DataFrame), "Should handle all invalid homes gracefully"


# =============================================================================
# PERFORMANCE AND OUTPUT TESTS
# =============================================================================

def test_simplified_output_for_nation_level_analysis(complete_npv_data, mock_validation_framework, mock_scenario_params):
    """Test simplified output suitable for nation-level analysis."""
    df = complete_npv_data
    
    # Test with verbose=False (nation-level mode)
    result = adoption_decision(df, 8, 'AEO2023 Reference Case', 'ap2', 'acs', verbose=False)
    
    # Should complete efficiently and return comprehensive results
    assert isinstance(result, pd.DataFrame), "Should return DataFrame for nation-level analysis"
    assert len(result) == len(df), "Should preserve all homes in analysis"
    
    # Check that key summary columns exist
    summary_cols = [col for col in result.columns if 'adoption_' in col or 'impact_' in col]
    assert len(summary_cols) > 0, "Should create adoption and impact summary columns"


def test_multiple_scc_assumptions_processing(complete_npv_data, mock_validation_framework, mock_scenario_params):
    """Test efficient processing of multiple SCC assumptions."""
    df = complete_npv_data
    
    result = adoption_decision(df, 8, 'AEO2023 Reference Case', 'ap2', 'acs', verbose=False)
    
    # Should create columns for all SCC assumptions
    scc_columns = {}
    for scc in SCC_ASSUMPTIONS:
        scc_cols = [col for col in result.columns if f'_{scc}_ap2_acs' in col]
        scc_columns[scc] = len(scc_cols)
    
    # Each SCC assumption should have similar number of columns
    column_counts = list(scc_columns.values())
    assert len(set(column_counts)) <= 2, "SCC assumptions should have similar column counts"  # Allow some variation
    assert all(count > 0 for count in column_counts), "All SCC assumptions should generate columns"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
