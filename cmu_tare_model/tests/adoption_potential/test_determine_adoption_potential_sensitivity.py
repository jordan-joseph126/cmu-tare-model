"""Tests for determine_adoption_potential_sensitivity (adoption_potential/determine_adoption_potential_sensitivity.py)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from cmu_tare_model.tests.conftest import FULL_EQUIPMENT_SPECS, FULL_UPGRADE_COLUMNS, BASE_YEAR


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def mock_constants(monkeypatch):
    """Mock constants with full production lifetimes."""
    monkeypatch.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', FULL_EQUIPMENT_SPECS)
    monkeypatch.setattr('cmu_tare_model.constants.UPGRADE_COLUMNS', FULL_UPGRADE_COLUMNS)
    monkeypatch.setattr('cmu_tare_model.constants.VERBOSE', False)
    monkeypatch.setattr('cmu_tare_model.constants.CR_FUNCTIONS', ['acs', 'h6c'])
    monkeypatch.setattr('cmu_tare_model.constants.RCM_MODELS', ['ap2', 'easiur', 'inmap'])
    monkeypatch.setattr('cmu_tare_model.constants.SCC_ASSUMPTIONS', ['lower', 'central', 'upper'])
    monkeypatch.setattr('cmu_tare_model.constants.PRIVATE_DISCOUNTING_METHOD_SUFFIXES', {
        'private_discount_rate_fixed_low': '_fixed_low',
        'private_discount_rate_fixed_base': '_fixed_base',
        'private_discount_rate_fixed_high': '_fixed_high',
        'private_discount_rate_variable': '_variable',
    })


@pytest.fixture
def adoption_df():
    """DataFrame with all columns needed for adoption potential."""
    n = 8
    np.random.seed(42)
    data = {
        'include_heating': [True, True, False, True, True, False, True, True],
        'include_waterHeating': [True, False, True, True, False, True, True, False],
        'include_clothesDrying': [True, True, True, False, True, True, False, True],
        'include_cooking': [True, True, True, True, False, True, True, False],
        'valid_fuel_heating': [True, True, False, True, True, False, True, True],
        'valid_tech_heating': [True, True, False, True, True, False, True, True],
        'upgrade_hvac_heating_efficiency': ['ASHP', 'MSHP', None, 'ASHP', None, None, 'ASHP', 'MSHP'],
        'upgrade_water_heater_efficiency': ['HP', None, 'HP', None, None, 'HP', None, None],
        'upgrade_clothes_dryer': [None, 'HP', None, None, 'HP', None, None, 'HP'],
        'upgrade_cooking_range': [None, 'Electric', None, 'Electric', None, None, 'Electric', None],
    }

    # NPV columns for each category and SCC
    for cat in FULL_EQUIPMENT_SPECS:
        # Private NPV columns
        data[f'iraRef_mp8_{cat}_private_npv_lessWTP_v4MID_fixed_base'] = np.random.uniform(-5000, 10000, n)
        data[f'iraRef_mp8_{cat}_private_npv_moreWTP_v4MID_fixed_base'] = np.random.uniform(-3000, 12000, n)

        # Rebate columns
        data[f'mp8_{cat}_rebate_amount_v4MID'] = np.random.uniform(500, 3000, n)

        # Public NPV for each (SCC, RCM, CR) combination
        for scc in ['lower', 'central', 'upper']:
            for rcm in ['ap2', 'easiur', 'inmap']:
                for cr in ['acs', 'h6c']:
                    data[f'iraRef_mp8_{cat}_public_npv_{scc}_{rcm}_{cr}'] = np.random.uniform(-2000, 8000, n)

    return pd.DataFrame(data)


# =============================================================================
# validate_input_parameters
# =============================================================================

def test_validate_input_parameters_valid():
    """Accepts valid parameters without raising."""
    from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import validate_input_parameters

    validate_input_parameters(
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='inmap',
        cr_function='acs',
    )


def test_validate_input_parameters_invalid_scenario():
    """Raises ValueError for invalid policy_scenario."""
    from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import validate_input_parameters

    with pytest.raises(ValueError, match="Invalid policy_scenario"):
        validate_input_parameters(
            menu_mp=8,
            policy_scenario='Invalid',
            rcm_model='inmap',
            cr_function='acs',
        )


def test_validate_input_parameters_invalid_rcm():
    """Raises ValueError for invalid rcm_model."""
    from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import validate_input_parameters

    with pytest.raises(ValueError, match="Invalid rcm_model"):
        validate_input_parameters(
            menu_mp=8,
            policy_scenario='AEO2023 Reference Case',
            rcm_model='invalid',
            cr_function='acs',
        )


def test_validate_input_parameters_invalid_cr():
    """Raises ValueError for invalid cr_function."""
    from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import validate_input_parameters

    with pytest.raises(ValueError, match="Invalid cr_function"):
        validate_input_parameters(
            menu_mp=8,
            policy_scenario='AEO2023 Reference Case',
            rcm_model='inmap',
            cr_function='invalid',
        )


def test_validate_input_parameters_multiple_errors():
    """Reports all invalid parameters at once."""
    from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import validate_input_parameters

    with pytest.raises(ValueError) as exc_info:
        validate_input_parameters(
            menu_mp=8,
            policy_scenario='Invalid',
            rcm_model='invalid',
            cr_function='invalid',
        )
    error_msg = str(exc_info.value)
    assert 'policy_scenario' in error_msg
    assert 'rcm_model' in error_msg
    assert 'cr_function' in error_msg


# =============================================================================
# fix_duplicate_columns
# =============================================================================

def test_fix_duplicate_columns_no_duplicates():
    """Returns unchanged DataFrame when no duplicates."""
    from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import fix_duplicate_columns

    df = pd.DataFrame({'a': [1], 'b': [2]})
    result = fix_duplicate_columns(df)
    assert list(result.columns) == ['a', 'b']


def test_fix_duplicate_columns_removes_duplicates():
    """Removes duplicate columns, keeping first occurrence."""
    from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import fix_duplicate_columns

    df = pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'a'])
    result = fix_duplicate_columns(df)
    assert list(result.columns) == ['a', 'b']
    assert result.loc[0, 'a'] == 1  # First occurrence kept


# =============================================================================
# _validate_required_columns
# =============================================================================

def test_validate_required_columns_missing_raises():
    """Raises KeyError with descriptive message for missing columns."""
    from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import _validate_required_columns

    df = pd.DataFrame({'a': [1], 'b': [2]})
    with pytest.raises(KeyError, match="Required columns missing"):
        _validate_required_columns(
            df,
            required_columns=['a', 'missing_col'],
            context_params={'test': 'value'},
        )


def test_validate_required_columns_all_present():
    """Does not raise when all columns exist."""
    from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import _validate_required_columns

    df = pd.DataFrame({'a': [1], 'b': [2]})
    _validate_required_columns(df, ['a', 'b'], {'test': 'value'})  # Should not raise


# =============================================================================
# _calculate_total_npv
# =============================================================================

def test_calculate_total_npv_sums_correctly():
    """Total NPV is sum of two NPV columns for valid homes."""
    from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import _calculate_total_npv

    df = pd.DataFrame({
        'npv1': [100.0, 200.0, np.nan, 400.0],
        'npv2': [50.0, np.nan, 300.0, 100.0],
        'include_heating': [True, True, True, True],
    })
    valid_mask = pd.Series([True, True, True, True], index=df.index)

    result = _calculate_total_npv(df, valid_mask, 'npv1', 'npv2', 'total')
    assert result.loc[0, 'total'] == pytest.approx(150.0)
    assert result.loc[3, 'total'] == pytest.approx(500.0)
    # Where either NPV is NaN, total should remain at initial 0.0 (from template)
    assert result.loc[1, 'total'] == 0.0


def test_calculate_total_npv_invalid_homes_nan():
    """Invalid homes get NaN in total NPV."""
    from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import _calculate_total_npv

    df = pd.DataFrame({
        'npv1': [100.0, 200.0],
        'npv2': [50.0, 75.0],
    })
    valid_mask = pd.Series([True, False], index=df.index)

    result = _calculate_total_npv(df, valid_mask, 'npv1', 'npv2', 'total')
    assert result.loc[0, 'total'] == pytest.approx(150.0)
    assert np.isnan(result.loc[1, 'total'])


# =============================================================================
# adoption_decision (integration)
# =============================================================================

@patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity.define_scenario_params')
def test_adoption_decision_output_has_adoption_columns(mock_params, adoption_df):
    """Produces DataFrame with adoption tier and impact columns."""
    from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import adoption_decision

    mock_params.return_value = ('iraRef_mp8_', 'MidCase', {}, {}, {}, {})

    result = adoption_decision(
        df=adoption_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='inmap',
        cr_function='acs',
        discount_rate_col_name='private_discount_rate_fixed_base',
        cost_scenario='v4MID',
        verbose=False,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(adoption_df)

    # Should have adoption columns
    adoption_cols = [c for c in result.columns if '_adoption_' in c]
    assert len(adoption_cols) > 0

    # Should have impact columns
    impact_cols = [c for c in result.columns if '_impact_' in c]
    assert len(impact_cols) > 0


@patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity.define_scenario_params')
def test_adoption_decision_tier_classification(mock_params, adoption_df):
    """Adoption tiers are properly assigned based on NPV values."""
    from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import adoption_decision

    mock_params.return_value = ('iraRef_mp8_', 'MidCase', {}, {}, {}, {})

    # Force specific NPV values for predictable tier classification
    for cat in FULL_EQUIPMENT_SPECS:
        # Home 0: lessWTP > 0 → Tier 1
        adoption_df.loc[0, f'iraRef_mp8_{cat}_private_npv_lessWTP_v4MID_fixed_base'] = 5000.0
        adoption_df.loc[0, f'iraRef_mp8_{cat}_private_npv_moreWTP_v4MID_fixed_base'] = 8000.0

        # Home 1: lessWTP < 0, moreWTP > 0 → Tier 2
        adoption_df.loc[1, f'iraRef_mp8_{cat}_private_npv_lessWTP_v4MID_fixed_base'] = -2000.0
        adoption_df.loc[1, f'iraRef_mp8_{cat}_private_npv_moreWTP_v4MID_fixed_base'] = 3000.0

    result = adoption_decision(
        df=adoption_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='inmap',
        cr_function='acs',
        discount_rate_col_name='private_discount_rate_fixed_base',
        cost_scenario='v4MID',
        verbose=False,
    )

    # Check at least one SCC combination for heating
    adoption_cols = [c for c in result.columns if '_heating_adoption_' in c]
    if adoption_cols:
        col = adoption_cols[0]
        # Home 0 should be Tier 1 (if it's valid and has upgrade)
        if adoption_df.loc[0, 'include_heating'] and pd.notna(adoption_df.loc[0, 'upgrade_hvac_heating_efficiency']):
            assert 'Tier 1' in result.loc[0, col]


@patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity.define_scenario_params')
def test_adoption_decision_missing_upgrade_columns_raises(mock_params):
    """Raises KeyError when required upgrade columns are missing."""
    from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import adoption_decision

    mock_params.return_value = ('iraRef_mp8_', 'MidCase', {}, {}, {}, {})
    df = pd.DataFrame({'col': [1]})

    with pytest.raises(KeyError, match="upgrade columns"):
        adoption_decision(
            df=df,
            menu_mp=8,
            policy_scenario='AEO2023 Reference Case',
            rcm_model='inmap',
            cr_function='acs',
            discount_rate_col_name='private_discount_rate_fixed_base',
            cost_scenario='v4MID',
            verbose=False,
        )


# =============================================================================
# calculate_climate_only_adoption_robust
# =============================================================================

@patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity.define_scenario_params')
def test_climate_only_adoption_output(mock_params, adoption_df):
    """Climate-only adoption produces total NPV columns."""
    from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import (
        calculate_climate_only_adoption_robust,
    )

    mock_params.return_value = ('iraRef_mp8_', 'MidCase', {}, {}, {}, {})

    # Add climate NPV columns
    for cat in FULL_EQUIPMENT_SPECS:
        for scc in ['lower', 'central', 'upper']:
            adoption_df[f'iraRef_mp8_{cat}_climate_npv_{scc}'] = np.random.uniform(-1000, 5000, len(adoption_df))

    result = calculate_climate_only_adoption_robust(
        df=adoption_df,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        discount_rate_col_name='private_discount_rate_fixed_base',
        cost_scenario='v4MID',
        verbose=False,
    )

    assert isinstance(result, pd.DataFrame)
    total_npv_cols = [c for c in result.columns if 'total_npv_climateOnly' in c]
    assert len(total_npv_cols) > 0


@patch('cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity.define_scenario_params')
def test_climate_only_adoption_invalid_scenario_raises(mock_params):
    """Raises ValueError for invalid policy_scenario."""
    from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import (
        calculate_climate_only_adoption_robust,
    )

    mock_params.return_value = ('iraRef_mp8_', 'MidCase', {}, {}, {}, {})
    df = pd.DataFrame({'col': [1]})

    with pytest.raises(ValueError, match="Invalid policy_scenario"):
        calculate_climate_only_adoption_robust(
            df=df,
            menu_mp=8,
            policy_scenario='Invalid',
            discount_rate_col_name='private_discount_rate_fixed_base',
            verbose=False,
        )
