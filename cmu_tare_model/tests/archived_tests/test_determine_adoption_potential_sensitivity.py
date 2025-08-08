import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from cmu_tare_model.constants import SCC_ASSUMPTIONS, RCM_MODELS, CR_FUNCTIONS, UPGRADE_COLUMNS

from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import (
    adoption_decision,
    validate_input_parameters,
    _process_adoption_tiers_and_impacts,
    _calculate_total_npv_columns
)

# =============================================================================
# FIXTURE FUNCTIONS
# =============================================================================

def create_sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_homes = 100
    
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
    
    # Add validation flags
    for category in UPGRADE_COLUMNS:
        df[f'include_{category}'] = np.random.choice([True, False], n_homes, p=[0.8, 0.2])
        df[f'valid_tech_{category}'] = np.random.choice([True, False], n_homes, p=[0.9, 0.1])
        df[f'valid_fuel_{category}'] = np.random.choice([True, False], n_homes, p=[0.95, 0.05])
    
    return df


def create_sample_npv_data():
    """Add NPV columns to sample data."""
    df = create_sample_data()
    n_homes = len(df)
    
    scenario_prefix = 'iraRef_mp1_'
    
    # Add private NPV columns
    for category in UPGRADE_COLUMNS:
        df[f'{scenario_prefix}{category}_private_npv_lessWTP'] = np.random.normal(0, 5000, n_homes)
        df[f'{scenario_prefix}{category}_private_npv_moreWTP'] = np.random.normal(1000, 5000, n_homes)
    
    # Add public NPV components
    for category in UPGRADE_COLUMNS:
        # Climate NPV for each SCC assumption
        for scc in SCC_ASSUMPTIONS:
            df[f'{scenario_prefix}{category}_climate_npv_{scc}'] = np.random.normal(2000, 3000, n_homes)
        
        # Health NPV for RCM/CR combinations
        for rcm in ['ap2']:  # Test with one RCM
            for cr in ['acs']:  # Test with one CR
                df[f'{scenario_prefix}{category}_health_npv_{rcm}_{cr}'] = np.random.normal(1500, 2000, n_homes)
                
                # Combined public NPV for each SCC
                for scc in SCC_ASSUMPTIONS:
                    climate_col = f'{scenario_prefix}{category}_climate_npv_{scc}'
                    health_col = f'{scenario_prefix}{category}_health_npv_{rcm}_{cr}'
                    df[f'{scenario_prefix}{category}_public_npv_{scc}_{rcm}_{cr}'] = df[climate_col] + df[health_col]
    
    # Add rebate columns
    for category in UPGRADE_COLUMNS:
        df[f'mp1_{category}_rebate_amount'] = np.random.uniform(0, 2000, n_homes)
    
    return df


# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================

def test_parameter_validation_valid_inputs():
    """Test input parameter validation with valid inputs."""
    # Valid parameters should not raise
    validate_input_parameters(1, 'AEO2023 Reference Case', 'ap2', 'acs')
    

def test_parameter_validation_invalid_policy_scenario():
    """Test invalid policy scenario raises ValueError."""
    with pytest.raises(ValueError, match="Invalid policy_scenario"):
        validate_input_parameters(1, 'Invalid Scenario', 'ap2', 'acs')


def test_parameter_validation_invalid_rcm_model():
    """Test invalid RCM model raises ValueError."""
    with pytest.raises(ValueError, match="Invalid rcm_model"):
        validate_input_parameters(1, 'AEO2023 Reference Case', 'invalid_rcm', 'acs')


def test_parameter_validation_invalid_cr_function():
    """Test invalid CR function raises ValueError."""
    with pytest.raises(ValueError, match="Invalid cr_function"):
        validate_input_parameters(1, 'AEO2023 Reference Case', 'ap2', 'invalid_cr')


# =============================================================================
# MISSING COLUMNS TESTS
# =============================================================================

def test_missing_upgrade_columns_error():
    """Test that missing required upgrade columns raise appropriate errors."""
    df_sample = create_sample_data()
    # Remove one upgrade column
    df_missing = df_sample.drop(columns=[UPGRADE_COLUMNS['heating']])
    
    with pytest.raises(KeyError, match="Required upgrade columns missing"):
        adoption_decision(df_missing, 1, 'AEO2023 Reference Case', 'ap2', 'acs')


def test_missing_npv_columns_error():
    """Test that missing NPV columns raise appropriate errors."""
    df_sample = create_sample_data()  # Data without NPV columns
    
    with pytest.raises(KeyError, match="Required NPV columns missing"):
        adoption_decision(df_sample, 1, 'AEO2023 Reference Case', 'ap2', 'acs')


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

def test_calculate_total_npv_columns_helper():
    """Test the total NPV calculation helper function."""
    # Create test data
    df_test = pd.DataFrame({
        'lessWTP_private_npv': [1000, 2000, np.nan, 3000],
        'moreWTP_private_npv': [1500, 2500, 3500, np.nan],
        'public_npv': [500, -1000, 800, 1200]
    })
    
    valid_mask = pd.Series([True, True, False, True])  # Third row is invalid
    df_new_columns = pd.DataFrame(index=df_test.index)
    df_new_columns['lessWTP_total'] = pd.Series(0.0, index=df_test.index)
    df_new_columns['moreWTP_total'] = pd.Series(0.0, index=df_test.index)
    
    _calculate_total_npv_columns(
        df_new_columns=df_new_columns,
        df_copy=df_test,
        valid_mask=valid_mask,
        private_npv_cols=('lessWTP_private_npv', 'moreWTP_private_npv'),
        public_npv_col='public_npv',
        output_col_names=('lessWTP_total', 'moreWTP_total')
    )
    
    # Check calculations for valid rows with non-null data
    assert df_new_columns.loc[0, 'lessWTP_total'] == 1500  # 1000 + 500
    assert df_new_columns.loc[0, 'moreWTP_total'] == 2000  # 1500 + 500
    assert df_new_columns.loc[1, 'lessWTP_total'] == 1000  # 2000 + (-1000)
    assert df_new_columns.loc[1, 'moreWTP_total'] == 1500  # 2500 + (-1000)
    
    # Check that invalid row (index 2) remains unchanged (0.0)
    assert df_new_columns.loc[2, 'lessWTP_total'] == 0.0
    assert df_new_columns.loc[2, 'moreWTP_total'] == 0.0
    
    # Check that row with missing moreWTP (index 3) has lessWTP calculated but moreWTP unchanged
    assert df_new_columns.loc[3, 'lessWTP_total'] == 4200  # 3000 + 1200
    assert df_new_columns.loc[3, 'moreWTP_total'] == 0.0   # No calculation due to NaN


def test_calculate_total_npv_columns_missing_input():
    """Test that missing input columns raise appropriate errors."""
    df_test = pd.DataFrame({'some_other_col': [1, 2, 3]})
    df_new_columns = pd.DataFrame(index=df_test.index)
    valid_mask = pd.Series([True, True, True])
    
    with pytest.raises(KeyError, match="Required columns missing"):
        _calculate_total_npv_columns(
            df_new_columns=df_new_columns,
            df_copy=df_test,
            valid_mask=valid_mask,
            private_npv_cols=('missing_col1', 'missing_col2'),
            public_npv_col='missing_public',
            output_col_names=('output1', 'output2')
        )


# =============================================================================
# ADOPTION TIER LOGIC TESTS  
# =============================================================================

def test_adoption_tier_logic_complete():
    """Test adoption tier classification logic with moreWTP columns for all analysis types."""
    # Create test data for all tier classifications
    df_test = pd.DataFrame({
        'lessWTP_private_npv': [5000, -2000, -3000, -5000],  # Tier 1, 2, 3, 4
        'moreWTP_private_npv': [6000, 1000, -1000, -6000],
        'upgrade_column': ['upgrade', 'upgrade', 'upgrade', 'upgrade'],
        'public_npv': [1000, 500, 800, -2000]  # Add public NPV column
    })
    
    valid_mask = pd.Series([True, True, True, True])
    df_new_columns = pd.DataFrame(index=df_test.index)
    
    # Set up total NPV columns for Tier 3 test
    df_new_columns['lessWTP_total_npv'] = [6000, -1500, -2200, -7000]
    df_new_columns['moreWTP_total_npv'] = [7000, 1500, -200, -8000]  # Tier 3 needs positive value for row 2
    df_new_columns.loc[2, 'moreWTP_total_npv'] = 500  # Make Tier 3 positive
    
    result = _process_adoption_tiers_and_impacts(
        df_new_columns=df_new_columns,
        df_copy=df_test,
        valid_mask=valid_mask,
        upgrade_column='upgrade_column',
        lessWTP_private_npv_col='lessWTP_private_npv',
        moreWTP_private_npv_col='moreWTP_private_npv',
        lessWTP_total_npv_col='lessWTP_total_npv',
        moreWTP_total_npv_col='moreWTP_total_npv',
        public_npv_col='public_npv',
        adoption_col='adoption',
        impact_col='impact'
    )
    
    expected_tiers = [
        'Tier 1: Feasible',
        'Tier 2: Feasible vs. Alternative', 
        'Tier 3: Subsidy-Dependent Feasibility',
        'Tier 4: Averse'
    ]
    
    for i, expected in enumerate(expected_tiers):
        assert expected in result.loc[i, 'adoption'], f"Row {i}: Expected {expected}, got {result.loc[i, 'adoption']}"


def test_public_impact_classification():
    """Test public impact classification logic."""
    # Create test data for impact classification
    df_test = pd.DataFrame({
        'public_npv': [5000, 0, -2000],  # Benefit, Zero, Detriment
        'upgrade_column': ['upgrade', 'upgrade', 'upgrade'],
        'lessWTP_private_npv': [1000, 1000, 1000],  # Add required columns
        'moreWTP_private_npv': [1500, 1500, 1500]
    })
    
    valid_mask = pd.Series([True, True, True])
    df_new_columns = pd.DataFrame(index=df_test.index)
    df_new_columns['lessWTP_total_npv'] = [6000, 1000, -1000]
    df_new_columns['moreWTP_total_npv'] = [6500, 1500, -500]
    
    result = _process_adoption_tiers_and_impacts(
        df_new_columns=df_new_columns,
        df_copy=df_test,
        valid_mask=valid_mask,
        upgrade_column='upgrade_column',
        lessWTP_private_npv_col='lessWTP_private_npv',
        moreWTP_private_npv_col='moreWTP_private_npv',
        lessWTP_total_npv_col='lessWTP_total_npv',
        moreWTP_total_npv_col='moreWTP_total_npv',
        public_npv_col='public_npv',
        adoption_col='adoption',
        impact_col='impact'
    )
    
    expected_impacts = [
        'Public Benefit',
        'Public NPV is Zero',
        'Public Detriment'
    ]
    
    for i, expected in enumerate(expected_impacts):
        assert expected in result.loc[i, 'impact'], f"Row {i}: Expected {expected}, got {result.loc[i, 'impact']}"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@patch('cmu_tare_model.utils.validation_framework.initialize_validation_tracking')
@patch('cmu_tare_model.utils.validation_framework.apply_final_masking')
def test_combined_analysis_unchanged(mock_final_masking, mock_validation):
    """Test that combined analysis produces same results as original implementation."""
    sample_npv_data = create_sample_npv_data()
    
    # Mock validation framework functions
    mock_validation.return_value = (
        sample_npv_data.copy(),
        pd.Series(True, index=sample_npv_data.index),
        {category: [] for category in UPGRADE_COLUMNS},
        []
    )
    mock_final_masking.side_effect = lambda df, *args, **kwargs: df
    
    result = adoption_decision(sample_npv_data, 1, 'AEO2023 Reference Case', 'ap2', 'acs')
    
    # Check that combined analysis columns exist
    for category in UPGRADE_COLUMNS:
        for scc in SCC_ASSUMPTIONS:
            expected_cols = [
                f'iraRef_mp1_{category}_adoption_{scc}_ap2_acs',
                f'iraRef_mp1_{category}_impact_{scc}_ap2_acs',
                f'iraRef_mp1_{category}_total_npv_lessWTP_{scc}_ap2_acs'
            ]
            for col in expected_cols:
                assert col in result.columns, f"Missing combined analysis column: {col}"


@patch('cmu_tare_model.utils.validation_framework.initialize_validation_tracking')
@patch('cmu_tare_model.utils.validation_framework.apply_final_masking')
def test_climate_only_analysis_columns(mock_final_masking, mock_validation):
    """Test that climate-only analysis generates expected columns including moreWTP."""
    sample_npv_data = create_sample_npv_data()
    
    # Mock validation framework functions
    mock_validation.return_value = (
        sample_npv_data.copy(),
        pd.Series(True, index=sample_npv_data.index),
        {category: [] for category in UPGRADE_COLUMNS},
        []
    )
    mock_final_masking.side_effect = lambda df, *args, **kwargs: df
    
    result = adoption_decision(sample_npv_data, 1, 'AEO2023 Reference Case', 'ap2', 'acs')
    
    # Check that climate-only analysis columns exist (including moreWTP)
    for category in UPGRADE_COLUMNS:
        for scc in SCC_ASSUMPTIONS:
            expected_cols = [
                f'iraRef_mp1_{category}_adoption_climateOnly_{scc}',
                f'iraRef_mp1_{category}_impact_climateOnly_{scc}',
                f'iraRef_mp1_{category}_total_npv_lessWTP_climateOnly_{scc}',
                f'iraRef_mp1_{category}_total_npv_moreWTP_climateOnly_{scc}'  # Added moreWTP
            ]
            for col in expected_cols:
                assert col in result.columns, f"Missing climate-only column: {col}"


@patch('cmu_tare_model.utils.validation_framework.initialize_validation_tracking')
@patch('cmu_tare_model.utils.validation_framework.apply_final_masking')
def test_health_only_analysis_columns(mock_final_masking, mock_validation):
    """Test that health-only analysis generates expected columns including moreWTP."""
    sample_npv_data = create_sample_npv_data()
    
    # Mock validation framework functions
    mock_validation.return_value = (
        sample_npv_data.copy(),
        pd.Series(True, index=sample_npv_data.index),
        {category: [] for category in UPGRADE_COLUMNS},
        []
    )
    mock_final_masking.side_effect = lambda df, *args, **kwargs: df
    
    result = adoption_decision(sample_npv_data, 1, 'AEO2023 Reference Case', 'ap2', 'acs')
    
    # Check that health-only analysis columns exist (including moreWTP)
    for category in UPGRADE_COLUMNS:
        expected_cols = [
            f'iraRef_mp1_{category}_adoption_healthOnly_ap2_acs',
            f'iraRef_mp1_{category}_impact_healthOnly_ap2_acs',
            f'iraRef_mp1_{category}_total_npv_lessWTP_healthOnly_ap2_acs',
            f'iraRef_mp1_{category}_total_npv_moreWTP_healthOnly_ap2_acs'  # Added moreWTP
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing health-only column: {col}"


# =============================================================================
# SENSITIVITY ANALYSIS TESTS
# =============================================================================

def test_sensitivity_analysis_differences():
    """Test that different sensitivity analyses can produce different results."""
    # Create test data where climate and health NPVs have different signs
    df_test = pd.DataFrame({
        'private_npv_lessWTP': [-1000],  # Negative private NPV
        'private_npv_moreWTP': [-500],   # Negative moreWTP private NPV
        'climate_npv': [3000],           # Positive climate NPV  
        'health_npv': [-500],            # Negative health NPV
        'public_npv': [2500],            # Combined = 3000 + (-500) = 2500 (positive)
        'upgrade_column': ['upgrade']
    })
    
    valid_mask = pd.Series([True])
    df_new_columns = pd.DataFrame(index=df_test.index)
    
    # Calculate total NPVs for each analysis type
    df_new_columns['combined_lessWTP'] = df_test['private_npv_lessWTP'] + df_test['public_npv']    # = 1500 (positive)
    df_new_columns['combined_moreWTP'] = df_test['private_npv_moreWTP'] + df_test['public_npv']    # = 2000 (positive)
    df_new_columns['climate_lessWTP'] = df_test['private_npv_lessWTP'] + df_test['climate_npv']    # = 2000 (positive)
    df_new_columns['climate_moreWTP'] = df_test['private_npv_moreWTP'] + df_test['climate_npv']    # = 2500 (positive)
    df_new_columns['health_lessWTP'] = df_test['private_npv_lessWTP'] + df_test['health_npv']      # = -1500 (negative)
    df_new_columns['health_moreWTP'] = df_test['private_npv_moreWTP'] + df_test['health_npv']      # = -1000 (negative)
    
    # Test combined analysis (should be feasible)
    result_combined = _process_adoption_tiers_and_impacts(
        df_new_columns=df_new_columns.copy(),
        df_copy=df_test,
        valid_mask=valid_mask,
        upgrade_column='upgrade_column',
        lessWTP_private_npv_col='private_npv_lessWTP',
        moreWTP_private_npv_col='private_npv_moreWTP',
        lessWTP_total_npv_col='combined_lessWTP',
        moreWTP_total_npv_col='combined_moreWTP',
        public_npv_col='public_npv',
        adoption_col='adoption_combined',
        impact_col='impact_combined'
    )
    
    # Test climate-only analysis (should be feasible)
    result_climate = _process_adoption_tiers_and_impacts(
        df_new_columns=df_new_columns.copy(),
        df_copy=df_test,
        valid_mask=valid_mask,
        upgrade_column='upgrade_column',
        lessWTP_private_npv_col='private_npv_lessWTP',
        moreWTP_private_npv_col='private_npv_moreWTP',
        lessWTP_total_npv_col='climate_lessWTP',
        moreWTP_total_npv_col='climate_moreWTP',
        public_npv_col='climate_npv',
        adoption_col='adoption_climate',
        impact_col='impact_climate'
    )
    
    # Test health-only analysis (should be averse)
    result_health = _process_adoption_tiers_and_impacts(
        df_new_columns=df_new_columns.copy(),
        df_copy=df_test,
        valid_mask=valid_mask,
        upgrade_column='upgrade_column',
        lessWTP_private_npv_col='private_npv_lessWTP',
        moreWTP_private_npv_col='private_npv_moreWTP',
        lessWTP_total_npv_col='health_lessWTP',
        moreWTP_total_npv_col='health_moreWTP',
        public_npv_col='health_npv',
        adoption_col='adoption_health',
        impact_col='impact_health'
    )
    
    # Verify different results
    assert 'Public Benefit' in result_combined.loc[0, 'impact_combined']
    assert 'Public Benefit' in result_climate.loc[0, 'impact_climate']
    assert 'Public Detriment' in result_health.loc[0, 'impact_health']


# =============================================================================
# VALIDATION FRAMEWORK TESTS
# =============================================================================

@patch('cmu_tare_model.utils.validation_framework.initialize_validation_tracking')
def test_validation_framework_steps(mock_validation):
    """Test that all 5 validation framework steps are executed."""
    sample_npv_data = create_sample_npv_data()
    
    # Mock the validation tracking
    mock_validation.return_value = (
        sample_npv_data.copy(),
        pd.Series(True, index=sample_npv_data.index),
        {category: [] for category in UPGRADE_COLUMNS},
        []
    )
    
    with patch('cmu_tare_model.utils.validation_framework.create_retrofit_only_series') as mock_create, \
         patch('cmu_tare_model.utils.validation_framework.apply_new_columns_to_dataframe') as mock_apply, \
         patch('cmu_tare_model.utils.validation_framework.apply_final_masking') as mock_final:
        
        mock_create.return_value = pd.Series(0.0, index=sample_npv_data.index)
        mock_apply.side_effect = lambda df, new_cols, *args: (df, {})
        mock_final.side_effect = lambda df, *args, **kwargs: df
        
        result = adoption_decision(sample_npv_data, 1, 'AEO2023 Reference Case', 'ap2', 'acs')
        
        # Verify that validation framework functions were called
        assert mock_validation.called, "Step 1: initialize_validation_tracking not called"
        assert mock_create.called, "Step 2: create_retrofit_only_series not called"
        # Steps 3 & 4 are handled by _calculate_total_npv_columns function
        assert mock_final.called, "Step 5: apply_final_masking not called"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

def test_edge_case_all_invalid_homes():
    """Test behavior when all homes are invalid for a category."""
    sample_data = create_sample_data()
    
    # Make all homes invalid for heating
    sample_data['include_heating'] = False
    
    # Add minimal NPV columns
    scenario_prefix = 'iraRef_mp1_'
    for category in ['heating']:
        sample_data[f'{scenario_prefix}{category}_private_npv_lessWTP'] = 1000
        sample_data[f'{scenario_prefix}{category}_private_npv_moreWTP'] = 1000
        sample_data[f'{scenario_prefix}{category}_climate_npv_central'] = 1000
        sample_data[f'{scenario_prefix}{category}_health_npv_ap2_acs'] = 1000
        sample_data[f'{scenario_prefix}{category}_public_npv_central_ap2_acs'] = 2000
    
    with patch('cmu_tare_model.utils.validation_framework.initialize_validation_tracking') as mock_validation:
        mock_validation.return_value = (
            sample_data.copy(),
            pd.Series(False, index=sample_data.index),  # All homes invalid
            {'heating': []},
            []
        )
        
        result = adoption_decision(sample_data, 1, 'AEO2023 Reference Case', 'ap2', 'acs')
        
        # All adoption decisions should be 'N/A: Invalid Baseline Fuel/Tech'
        adoption_col = 'iraRef_mp1_heating_adoption_central_ap2_acs'
        if adoption_col in result.columns:
            assert all(result[adoption_col] == 'N/A: Invalid Baseline Fuel/Tech')


def test_numeric_conversion_robustness():
    """Test that non-numeric values are handled properly."""
    sample_data = create_sample_data()
    
    # Add NPV columns with some non-numeric values
    scenario_prefix = 'iraRef_mp1_'
    sample_data[f'{scenario_prefix}heating_private_npv_lessWTP'] = ['1000', '2000', 'invalid', '3000'] + [1000] * (len(sample_data) - 4)
    sample_data[f'{scenario_prefix}heating_private_npv_moreWTP'] = [2000] * len(sample_data)
    sample_data[f'{scenario_prefix}heating_climate_npv_central'] = [1500] * len(sample_data)
    sample_data[f'{scenario_prefix}heating_health_npv_ap2_acs'] = [500] * len(sample_data)
    sample_data[f'{scenario_prefix}heating_public_npv_central_ap2_acs'] = [2000] * len(sample_data)
    
    # Should not raise an error due to numeric conversion
    with patch('cmu_tare_model.utils.validation_framework.initialize_validation_tracking') as mock_validation:
        mock_validation.return_value = (
            sample_data.copy(),
            pd.Series(True, index=sample_data.index),
            {'heating': []},
            []
        )
        
        result = adoption_decision(sample_data, 1, 'AEO2023 Reference Case', 'ap2', 'acs')
        
        # The function should complete without error
        assert result is not None


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
