"""Tests for calculate_lifetime_public_impact_sensitivity (public_impact/calculate_lifetime_public_impact_sensitivity.py)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

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
    monkeypatch.setattr('cmu_tare_model.constants.PUBLIC_DISCOUNTING_METHOD_SUFFIXES', {
        'public_discount_rate': '',
    })


@pytest.fixture
def public_npv_df():
    """DataFrame for public NPV calculation."""
    n = 6
    np.random.seed(42)
    data = {
        'state': ['CA', 'TX', 'NY', 'FL', 'IL', 'CA'],
        'include_heating': [True, True, False, True, True, False],
        'include_waterHeating': [True, False, True, True, False, True],
        'include_clothesDrying': [True, True, True, False, True, True],
        'include_cooking': [True, True, True, True, False, True],
        'valid_fuel_heating': [True, True, False, True, True, False],
        'valid_tech_heating': [True, True, False, True, True, False],
        'upgrade_hvac_heating_efficiency': ['ASHP', 'MSHP', None, 'ASHP', 'MSHP', None],
        'upgrade_water_heater_efficiency': ['HP', None, 'HP', None, None, 'HP'],
        'upgrade_clothes_dryer': [None, 'HP', None, None, 'HP', None],
        'upgrade_cooking_range': [None, 'Electric', None, 'Electric', None, 'Electric'],
        'public_discount_rate': [0.02] * n,
    }
    return pd.DataFrame(data)


@pytest.fixture
def damage_dataframes(public_npv_df):
    """Create baseline and measure package climate and health damage DataFrames."""
    n = len(public_npv_df)
    df_baseline_climate = pd.DataFrame(index=public_npv_df.index)
    df_mp_climate = pd.DataFrame(index=public_npv_df.index)
    df_baseline_health = pd.DataFrame(index=public_npv_df.index)
    df_mp_health = pd.DataFrame(index=public_npv_df.index)

    for cat, lifetime in FULL_EQUIPMENT_SPECS.items():
        for year in range(1, lifetime + 1):
            year_label = year + (BASE_YEAR - 1)
            for scc in ['lower', 'central', 'upper']:
                base_col = f'baseline_{year_label}_{cat}_damages_climate_lrmer_{scc}'
                mp_col = f'iraRef_mp8_{year_label}_{cat}_damages_climate_lrmer_{scc}'
                df_baseline_climate[base_col] = np.random.uniform(10, 100, n)
                df_mp_climate[mp_col] = np.random.uniform(5, 50, n)

            for rcm in ['ap2', 'easiur', 'inmap']:
                for cr in ['acs', 'h6c']:
                    base_health = f'baseline_{year_label}_{cat}_damages_health_{rcm}_{cr}'
                    mp_health = f'iraRef_mp8_{year_label}_{cat}_damages_health_{rcm}_{cr}'
                    df_baseline_health[base_health] = np.random.uniform(5, 50, n)
                    df_mp_health[mp_health] = np.random.uniform(2, 25, n)

    return df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health


# =============================================================================
# _sum_yearly_damages
# =============================================================================

def test_sum_yearly_damages_basic():
    """Sums yearly Series with proper NaN handling."""
    from cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity import _sum_yearly_damages

    yearly = [
        pd.Series([10.0, 20.0, np.nan]),
        pd.Series([5.0, 15.0, np.nan]),
    ]
    template = pd.Series([0.0, 0.0, np.nan])
    mask = pd.Series([True, True, False])

    result = _sum_yearly_damages(yearly, template, mask, menu_mp=8)
    assert result.iloc[0] == pytest.approx(15.0)
    assert result.iloc[1] == pytest.approx(35.0)
    assert np.isnan(result.iloc[2])


def test_sum_yearly_damages_empty_list():
    """Returns template when no yearly data."""
    from cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity import _sum_yearly_damages

    template = pd.Series([0.0, np.nan, 0.0])
    mask = pd.Series([True, False, True])

    result = _sum_yearly_damages([], template, mask, menu_mp=8)
    pd.testing.assert_series_equal(result, template, check_names=False)


def test_sum_yearly_damages_nan_propagation():
    """NaN in any year propagates to total via skipna=False."""
    from cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity import _sum_yearly_damages

    yearly = [
        pd.Series([10.0, np.nan, 30.0]),
        pd.Series([5.0, 15.0, 20.0]),
    ]
    template = pd.Series([0.0, 0.0, 0.0])
    mask = pd.Series([True, True, True])

    result = _sum_yearly_damages(yearly, template, mask, menu_mp=0)
    assert result.iloc[0] == pytest.approx(15.0)
    assert np.isnan(result.iloc[1])  # NaN propagation
    assert result.iloc[2] == pytest.approx(50.0)


# =============================================================================
# calculate_climate_npv
# =============================================================================

@patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.define_scenario_params')
def test_climate_npv_returns_dict(mock_params, public_npv_df, damage_dataframes):
    """Returns dictionary of climate NPV Series for each (category, SCC) pair."""
    from cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity import calculate_climate_npv

    df_baseline_climate, _, df_mp_climate, _ = damage_dataframes
    mock_params.return_value = ('iraRef_mp8_', 'MidCase', {}, {}, {}, {})

    all_cols = {cat: [] for cat in FULL_EQUIPMENT_SPECS}

    result = calculate_climate_npv(
        df_copy=public_npv_df.copy(),
        df_baseline_climate=df_baseline_climate,
        df_mp_climate=df_mp_climate,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        base_year=BASE_YEAR,
        all_columns_to_mask=all_cols,
        verbose=False,
    )

    assert isinstance(result, dict)
    # Should have entries for each (category, SCC) combination
    for cat in FULL_EQUIPMENT_SPECS:
        for scc in ['lower', 'central', 'upper']:
            expected_key = f'iraRef_mp8_{cat}_climate_npv_{scc}'
            assert expected_key in result, f"Missing {expected_key}"


# =============================================================================
# calculate_health_npv
# =============================================================================

@patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.calculate_discount_factors')
@patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.define_scenario_params')
def test_health_npv_returns_dict(mock_params, mock_discount, public_npv_df, damage_dataframes):
    """Returns dictionary of health NPV Series for each category."""
    from cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity import calculate_health_npv

    _, df_baseline_health, _, df_mp_health = damage_dataframes
    mock_params.return_value = ('iraRef_mp8_', 'MidCase', {}, {}, {}, {})
    mock_discount.return_value = pd.Series(0.95, index=public_npv_df.index)

    all_cols = {cat: [] for cat in FULL_EQUIPMENT_SPECS}

    result = calculate_health_npv(
        df_copy=public_npv_df.copy(),
        df_baseline_health=df_baseline_health,
        df_mp_health=df_mp_health,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='inmap',
        cr_function='acs',
        base_year=BASE_YEAR,
        discount_rate_col_name='public_discount_rate',
        all_columns_to_mask=all_cols,
        verbose=False,
    )

    assert isinstance(result, dict)
    for cat in FULL_EQUIPMENT_SPECS:
        expected_key = f'iraRef_mp8_{cat}_health_npv_inmap_acs'
        assert expected_key in result, f"Missing {expected_key}"


# =============================================================================
# calculate_public_npv (integration)
# =============================================================================

@patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.validate_damage_dataframes')
@patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.calculate_discount_factors')
@patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.define_scenario_params')
def test_public_npv_output_structure(mock_params, mock_discount, mock_validate,
                                      public_npv_df, damage_dataframes):
    """End-to-end: produces DataFrame with climate, health, and combined NPV columns."""
    from cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity import calculate_public_npv

    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = damage_dataframes
    mock_params.return_value = ('iraRef_mp8_', 'MidCase', {}, {}, {}, {})
    mock_discount.return_value = pd.Series(0.95, index=public_npv_df.index)
    mock_validate.return_value = (True, [])

    result = calculate_public_npv(
        df=public_npv_df,
        df_baseline_climate=df_baseline_climate,
        df_baseline_health=df_baseline_health,
        df_mp_climate=df_mp_climate,
        df_mp_health=df_mp_health,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='inmap',
        base_year=BASE_YEAR,
        verbose=False,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(public_npv_df)

    # Check for climate NPV columns
    climate_cols = [c for c in result.columns if 'climate_npv' in c]
    assert len(climate_cols) > 0

    # Check for health NPV columns
    health_cols = [c for c in result.columns if 'health_npv' in c]
    assert len(health_cols) > 0

    # Check for combined public NPV columns
    public_cols = [c for c in result.columns if 'public_npv' in c]
    assert len(public_cols) > 0


@patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.validate_damage_dataframes')
@patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.calculate_discount_factors')
@patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.define_scenario_params')
def test_public_npv_invalid_rcm_raises(mock_params, mock_discount, mock_validate, public_npv_df, damage_dataframes):
    """Raises ValueError for invalid RCM model."""
    from cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity import calculate_public_npv

    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = damage_dataframes
    mock_params.return_value = ('iraRef_mp8_', 'MidCase', {}, {}, {}, {})
    mock_validate.return_value = (True, [])

    with pytest.raises(ValueError, match="Invalid rcm_model"):
        calculate_public_npv(
            df=public_npv_df,
            df_baseline_climate=df_baseline_climate,
            df_baseline_health=df_baseline_health,
            df_mp_climate=df_mp_climate,
            df_mp_health=df_mp_health,
            menu_mp=8,
            policy_scenario='AEO2023 Reference Case',
            rcm_model='invalid_rcm',
            verbose=False,
        )


@patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.validate_damage_dataframes')
@patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.calculate_discount_factors')
@patch('cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity.define_scenario_params')
def test_public_npv_invalid_homes_masked(mock_params, mock_discount, mock_validate,
                                          public_npv_df, damage_dataframes):
    """Invalid homes get NaN in NPV columns."""
    from cmu_tare_model.public_impact.calculate_lifetime_public_impact_sensitivity import calculate_public_npv

    df_baseline_climate, df_baseline_health, df_mp_climate, df_mp_health = damage_dataframes
    mock_params.return_value = ('iraRef_mp8_', 'MidCase', {}, {}, {}, {})
    mock_discount.return_value = pd.Series(0.95, index=public_npv_df.index)
    mock_validate.return_value = (True, [])

    result = calculate_public_npv(
        df=public_npv_df,
        df_baseline_climate=df_baseline_climate,
        df_baseline_health=df_baseline_health,
        df_mp_climate=df_mp_climate,
        df_mp_health=df_mp_health,
        menu_mp=8,
        policy_scenario='AEO2023 Reference Case',
        rcm_model='inmap',
        base_year=BASE_YEAR,
        verbose=False,
    )

    for cat in FULL_EQUIPMENT_SPECS:
        invalid_mask = ~public_npv_df[f'include_{cat}']
        if not invalid_mask.any():
            continue
        npv_cols = [c for c in result.columns if f'_{cat}_climate_npv_' in c or f'_{cat}_health_npv_' in c]
        for col in npv_cols:
            assert result.loc[invalid_mask, col].isna().all(), \
                f"Invalid homes should have NaN in {col}"
