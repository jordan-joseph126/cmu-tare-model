"""Tests for calculate_lifetime_health_impacts (public_impact/calculate_lifetime_health_impacts_sensitivity.py)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from cmu_tare_model.tests.conftest import FULL_EQUIPMENT_SPECS, FULL_UPGRADE_COLUMNS, FULL_FUEL_MAPPING, BASE_YEAR


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def mock_constants(monkeypatch):
    """Mock constants with full production lifetimes."""
    monkeypatch.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', FULL_EQUIPMENT_SPECS)
    monkeypatch.setattr('cmu_tare_model.constants.UPGRADE_COLUMNS', FULL_UPGRADE_COLUMNS)
    monkeypatch.setattr('cmu_tare_model.constants.FUEL_MAPPING', FULL_FUEL_MAPPING)
    monkeypatch.setattr('cmu_tare_model.constants.VERBOSE', False)
    monkeypatch.setattr('cmu_tare_model.constants.TD_LOSSES_MULTIPLIER', 1 / (1 - 0.05))
    monkeypatch.setattr('cmu_tare_model.constants.POLLUTANTS', ['so2', 'nox', 'pm25', 'co2e'])
    monkeypatch.setattr('cmu_tare_model.constants.CR_FUNCTIONS', ['acs', 'h6c'])
    monkeypatch.setattr('cmu_tare_model.constants.RCM_MODELS', ['ap2', 'easiur', 'inmap'])


@pytest.fixture
def health_df():
    """DataFrame with columns needed for health impact calculation."""
    n = 6
    np.random.seed(42)
    data = {
        'state': ['CA', 'TX', 'NY', 'FL', 'IL', 'CA'],
        'census_division': ['Pacific', 'WSC', 'MA', 'SA', 'ENC', 'Pacific'],
        'gea_region': ['CAL', 'TXS', 'NYC', 'FLA', 'CEN', 'CAL'],
        'county_fips': ['06001', '48201', '36061', '12086', '17031', '06037'],
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
    }

    # Consumption columns for full lifetime
    for fuel_key in FULL_FUEL_MAPPING.values():
        for cat in FULL_EQUIPMENT_SPECS:
            data[f'base_{fuel_key}_{cat}_consumption'] = np.random.uniform(500, 2000, n)

    for cat, lifetime in FULL_EQUIPMENT_SPECS.items():
        data[f'base_{cat}_fuel'] = np.random.choice(['Electricity', 'Natural Gas'], n)
        for year in range(BASE_YEAR, BASE_YEAR + lifetime):
            data[f'baseline_{year}_{cat}_consumption'] = np.random.uniform(800, 2500, n)
            for mp in [8]:
                data[f'mp{mp}_{year}_{cat}_consumption'] = np.random.uniform(300, 1200, n)
                data[f'mp{mp}_{cat}_consumption'] = np.random.uniform(300, 1200, n)

    return pd.DataFrame(data)


def _mock_get_health_impact(lookup, county_key, rcm, pollutant):
    """Helper mock for health impact lookups."""
    return 50.0  # Constant MSC value for testing


# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

def test_health_impacts_invalid_policy():
    """Raises ValueError for invalid policy_scenario."""
    from cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity import (
        calculate_lifetime_health_impacts,
    )
    df = pd.DataFrame({'col': [1]})
    with pytest.raises(ValueError):
        calculate_lifetime_health_impacts(df, menu_mp=0, policy_scenario='Invalid')


# =============================================================================
# BASELINE (menu_mp=0) OUTPUT STRUCTURE
# =============================================================================

@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.get_health_impact_with_fallback')
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_vsl_adjustment')
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_fossil_fuel_acs', {})
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_fossil_fuel_h6c', {})
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_electricity_acs', {})
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_electricity_h6c', {})
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.calculate_fossil_fuel_emissions')
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.get_electricity_consumption_for_year')
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.define_scenario_params')
def test_baseline_health_output_structure(mock_params, mock_elec, mock_fossil, mock_vsl,
                                           mock_fallback, health_df):
    """Baseline produces df_main and df_detailed with expected lifetime columns."""
    mock_emissions_elec_health = {}
    for year in range(BASE_YEAR, BASE_YEAR + 16):
        for region in ['CAL', 'TXS', 'NYC', 'FLA', 'CEN']:
            mock_emissions_elec_health[(year, region)] = {
                'delta_egrid_so2': 0.001,
                'delta_egrid_nox': 0.002,
                'delta_egrid_pm25': 0.0005,
            }

    mock_params.return_value = ('baseline_', 'MidCase', {}, {}, mock_emissions_elec_health, {})
    mock_elec.return_value = pd.Series(1000.0, index=health_df.index)
    mock_fossil.return_value = {
        'co2e': pd.Series(0.5, index=health_df.index),
        'so2': pd.Series(0.001, index=health_df.index),
        'nox': pd.Series(0.002, index=health_df.index),
        'pm25': pd.Series(0.0005, index=health_df.index),
    }

    # VSL adjustment: dict-like lookup by year
    vsl_dict = {year: 1.0 + (year - BASE_YEAR) * 0.01 for year in range(BASE_YEAR, BASE_YEAR + 16)}
    mock_vsl.__contains__ = lambda self, key: key in vsl_dict
    mock_vsl.__getitem__ = lambda self, key: vsl_dict[key]

    mock_fallback.return_value = 50.0

    from cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity import (
        calculate_lifetime_health_impacts,
    )

    df_main, df_detailed = calculate_lifetime_health_impacts(
        health_df, menu_mp=0, policy_scenario='AEO2023 Reference Case', verbose=False
    )

    assert isinstance(df_main, pd.DataFrame)
    assert isinstance(df_detailed, pd.DataFrame)
    assert len(df_main) == len(health_df)

    # Check lifetime health damages columns for each (RCM, CR) combination
    for cat in FULL_EQUIPMENT_SPECS:
        for rcm in ['ap2', 'easiur', 'inmap']:
            for cr in ['acs', 'h6c']:
                col = f'baseline_{cat}_lifetime_damages_health_{rcm}_{cr}'
                assert col in df_main.columns, f"Missing {col}"


# =============================================================================
# MASKING TESTS
# =============================================================================

@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.get_health_impact_with_fallback')
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_vsl_adjustment')
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_fossil_fuel_acs', {})
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_fossil_fuel_h6c', {})
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_electricity_acs', {})
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_electricity_h6c', {})
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.calculate_fossil_fuel_emissions')
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.get_electricity_consumption_for_year')
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.define_scenario_params')
def test_mp_health_invalid_homes_masked(mock_params, mock_elec, mock_fossil, mock_vsl,
                                         mock_fallback, health_df):
    """Invalid homes get NaN in measure package lifetime health columns."""
    mock_emissions_elec_health = {}
    for year in range(BASE_YEAR, BASE_YEAR + 16):
        for region in ['CAL', 'TXS', 'NYC', 'FLA', 'CEN']:
            mock_emissions_elec_health[(year, region)] = {
                'delta_egrid_so2': 0.001,
                'delta_egrid_nox': 0.002,
                'delta_egrid_pm25': 0.0005,
            }

    mock_params.return_value = ('iraRef_mp8_', 'MidCase', {}, {}, mock_emissions_elec_health, {})
    mock_elec.return_value = pd.Series(1000.0, index=health_df.index)
    mock_fossil.return_value = {
        'co2e': pd.Series(0.5, index=health_df.index),
        'so2': pd.Series(0.001, index=health_df.index),
        'nox': pd.Series(0.002, index=health_df.index),
        'pm25': pd.Series(0.0005, index=health_df.index),
    }

    vsl_dict = {year: 1.0 for year in range(BASE_YEAR, BASE_YEAR + 16)}
    mock_vsl.__contains__ = lambda self, key: key in vsl_dict
    mock_vsl.__getitem__ = lambda self, key: vsl_dict[key]
    mock_fallback.return_value = 50.0

    from cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity import (
        calculate_lifetime_health_impacts,
    )

    df_main, _ = calculate_lifetime_health_impacts(
        health_df, menu_mp=8, policy_scenario='AEO2023 Reference Case', verbose=False
    )

    for cat in FULL_EQUIPMENT_SPECS:
        include_mask = health_df[f'include_{cat}']
        upgrade_col = FULL_UPGRADE_COLUMNS.get(cat)
        if upgrade_col and upgrade_col in health_df.columns:
            retrofit_mask = health_df[upgrade_col].notna()
            invalid_mask = ~(include_mask & retrofit_mask)
        else:
            invalid_mask = ~include_mask
        if not invalid_mask.any():
            continue
        for rcm in ['ap2', 'easiur', 'inmap']:
            for cr in ['acs', 'h6c']:
                col = f'iraRef_mp8_{cat}_lifetime_damages_health_{rcm}_{cr}'
                if col in df_main.columns:
                    assert df_main.loc[invalid_mask, col].isna().all(), \
                        f"Invalid homes should have NaN in {col}"


# =============================================================================
# calculate_health_damages_for_pair (unit test)
# =============================================================================

@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.get_health_impact_with_fallback')
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_vsl_adjustment')
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_fossil_fuel_acs', {})
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_electricity_acs', {})
@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.get_electricity_consumption_for_year')
def test_health_damages_for_pair_returns_dict(mock_elec, mock_vsl, mock_fallback, health_df):
    """Returns dictionary with pollutant-specific and overall damage columns."""
    from cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity import (
        calculate_health_damages_for_pair,
    )

    mock_emissions_elec_health = {}
    for region in ['CAL', 'TXS', 'NYC', 'FLA', 'CEN']:
        mock_emissions_elec_health[(BASE_YEAR, region)] = {
            'delta_egrid_so2': 0.001,
            'delta_egrid_nox': 0.002,
            'delta_egrid_pm25': 0.0005,
        }

    vsl_dict = {BASE_YEAR: 1.0}
    mock_vsl.__contains__ = lambda self, key: key in vsl_dict
    mock_vsl.__getitem__ = lambda self, key: vsl_dict[key]
    mock_fallback.return_value = 50.0
    mock_elec.return_value = pd.Series(1000.0, index=health_df.index)

    result = calculate_health_damages_for_pair(
        df=health_df,
        category='heating',
        year_label=BASE_YEAR,
        lookup_emissions_electricity_health=mock_emissions_elec_health,
        scenario_prefix='baseline_',
        total_fossil_fuel_emissions={
            'co2e': pd.Series(0.5, index=health_df.index),
            'so2': pd.Series(0.001, index=health_df.index),
            'nox': pd.Series(0.002, index=health_df.index),
            'pm25': pd.Series(0.0005, index=health_df.index),
        },
        menu_mp=0,
        rcm='inmap',
        cr='acs',
    )

    assert isinstance(result, dict)

    # Should have per-pollutant damages (so2, nox, pm25 — not co2e)
    for pollutant in ['so2', 'nox', 'pm25']:
        key = f'baseline_{BASE_YEAR}_heating_damages_{pollutant}_inmap_acs'
        assert key in result, f"Missing {key}"

    # Should have overall health damages
    overall_key = f'baseline_{BASE_YEAR}_heating_damages_health_inmap_acs'
    assert overall_key in result


@patch('cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity.lookup_health_vsl_adjustment')
def test_health_damages_for_pair_invalid_cr(mock_vsl, health_df):
    """Raises ValueError for invalid CR function."""
    from cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity import (
        calculate_health_damages_for_pair,
    )

    vsl_dict = {BASE_YEAR: 1.0}
    mock_vsl.__contains__ = lambda self, key: key in vsl_dict
    mock_vsl.__getitem__ = lambda self, key: vsl_dict[key]

    with pytest.raises(ValueError, match="Invalid C-R function"):
        calculate_health_damages_for_pair(
            df=health_df,
            category='heating',
            year_label=BASE_YEAR,
            lookup_emissions_electricity_health={},
            scenario_prefix='baseline_',
            total_fossil_fuel_emissions={},
            menu_mp=0,
            rcm='inmap',
            cr='invalid_cr',
        )
