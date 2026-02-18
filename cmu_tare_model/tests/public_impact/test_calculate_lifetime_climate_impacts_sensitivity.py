"""Tests for calculate_lifetime_climate_impacts (public_impact/calculate_lifetime_climate_impacts_sensitivity.py)."""

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
    monkeypatch.setattr('cmu_tare_model.constants.MER_TYPES', ['lrmer', 'srmer'])
    monkeypatch.setattr('cmu_tare_model.constants.SCC_ASSUMPTIONS', ['lower', 'central', 'upper'])


@pytest.fixture
def climate_df():
    """DataFrame with columns needed for climate impact calculation."""
    n = 6
    np.random.seed(42)
    data = {
        'state': ['CA', 'TX', 'NY', 'FL', 'IL', 'CA'],
        'census_division': ['Pacific', 'WSC', 'MA', 'SA', 'ENC', 'Pacific'],
        'gea_region': ['CAL', 'TXS', 'NYC', 'FLA', 'CEN', 'CAL'],
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

    # Fuel-specific baseline consumption
    for fuel_key in FULL_FUEL_MAPPING.values():
        for cat in FULL_EQUIPMENT_SPECS:
            data[f'base_{fuel_key}_{cat}_consumption'] = np.random.uniform(500, 2000, n)

    # Annual consumption columns
    for cat, lifetime in FULL_EQUIPMENT_SPECS.items():
        data[f'base_{cat}_fuel'] = np.random.choice(['Electricity', 'Natural Gas'], n)
        for year in range(BASE_YEAR, BASE_YEAR + lifetime):
            data[f'baseline_{year}_{cat}_consumption'] = np.random.uniform(800, 2500, n)
            for mp in [8]:
                data[f'mp{mp}_{year}_{cat}_consumption'] = np.random.uniform(300, 1200, n)
                data[f'mp{mp}_{cat}_consumption'] = np.random.uniform(300, 1200, n)

    return pd.DataFrame(data)


@pytest.fixture
def mock_emissions_electricity_climate():
    """Lookup for electricity climate emissions by (scenario, region)."""
    lookup = {}
    for scenario in ['MidCase', 'LowREHighCost']:
        for region in ['CAL', 'TXS', 'NYC', 'FLA', 'CEN']:
            key = (scenario, region)
            lookup[key] = {}
            for year in range(BASE_YEAR, BASE_YEAR + 16):
                lookup[key][year] = {
                    'lrmer_mt_per_kWh_co2e': 0.0004 - (year - BASE_YEAR) * 0.00001,
                    'srmer_mt_per_kWh_co2e': 0.0005 - (year - BASE_YEAR) * 0.00001,
                }
    return lookup


@pytest.fixture
def mock_emissions_fossil_fuel():
    """Lookup for fossil fuel emission factors."""
    return {'Natural Gas': {'co2e': 0.005}, 'Fuel Oil': {'co2e': 0.007}, 'Propane': {'co2e': 0.006}}


@pytest.fixture
def mock_scc_lookup():
    """Lookup for SCC values by (assumption, year)."""
    lookup = {}
    for assumption in ['lower', 'central', 'upper']:
        lookup[assumption] = {}
        base_scc = {'lower': 20, 'central': 50, 'upper': 100}[assumption]
        for year in range(BASE_YEAR, BASE_YEAR + 16):
            lookup[assumption][year] = base_scc + (year - BASE_YEAR) * 2
    return lookup


# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

def test_climate_impacts_invalid_policy():
    """Raises ValueError for invalid policy_scenario."""
    from cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity import (
        calculate_lifetime_climate_impacts,
    )
    df = pd.DataFrame({'col': [1]})
    with pytest.raises(ValueError):
        calculate_lifetime_climate_impacts(df, menu_mp=0, policy_scenario='Invalid')


# =============================================================================
# BASELINE (menu_mp=0) OUTPUT STRUCTURE
# =============================================================================

@patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.lookup_climate_impact_scc')
@patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.calculate_fossil_fuel_emissions')
@patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.get_electricity_consumption_for_year')
@patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.define_scenario_params')
def test_baseline_climate_output_structure(mock_params, mock_elec, mock_fossil, mock_scc,
                                           climate_df, mock_emissions_electricity_climate,
                                           mock_scc_lookup):
    """Baseline produces df_main and df_detailed with expected column patterns."""
    mock_params.return_value = ('baseline_', 'MidCase', {}, mock_emissions_electricity_climate, {}, {})
    mock_elec.return_value = pd.Series(1000.0, index=climate_df.index)
    mock_fossil.return_value = {
        'co2e': pd.Series(0.5, index=climate_df.index),
        'so2': pd.Series(0.001, index=climate_df.index),
        'nox': pd.Series(0.002, index=climate_df.index),
        'pm25': pd.Series(0.0005, index=climate_df.index),
    }
    mock_scc.__getitem__ = lambda self, key: mock_scc_lookup[key]

    df_main, df_detailed = calculate_lifetime_climate_impacts(
        climate_df, menu_mp=0, policy_scenario='AEO2023 Reference Case', verbose=False
    )

    assert isinstance(df_main, pd.DataFrame)
    assert isinstance(df_detailed, pd.DataFrame)
    assert len(df_main) == len(climate_df)

    # Check lifetime emission columns for each category and MER type
    for cat in FULL_EQUIPMENT_SPECS:
        for mer in ['lrmer', 'srmer']:
            col = f'baseline_{cat}_lifetime_mt_co2e_{mer}'
            assert col in df_main.columns, f"Missing {col}"


@patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.lookup_climate_impact_scc')
@patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.calculate_fossil_fuel_emissions')
@patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.get_electricity_consumption_for_year')
@patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.define_scenario_params')
def test_baseline_climate_damages_columns_exist(mock_params, mock_elec, mock_fossil, mock_scc,
                                                 climate_df, mock_emissions_electricity_climate,
                                                 mock_scc_lookup):
    """Lifetime damages columns exist for all (MER, SCC) combinations."""
    mock_params.return_value = ('baseline_', 'MidCase', {}, mock_emissions_electricity_climate, {}, {})
    mock_elec.return_value = pd.Series(1000.0, index=climate_df.index)
    mock_fossil.return_value = {
        'co2e': pd.Series(0.5, index=climate_df.index),
        'so2': pd.Series(0.001, index=climate_df.index),
        'nox': pd.Series(0.002, index=climate_df.index),
        'pm25': pd.Series(0.0005, index=climate_df.index),
    }
    mock_scc.__getitem__ = lambda self, key: mock_scc_lookup[key]

    df_main, _ = calculate_lifetime_climate_impacts(
        climate_df, menu_mp=0, policy_scenario='AEO2023 Reference Case', verbose=False
    )

    for cat in FULL_EQUIPMENT_SPECS:
        for mer in ['lrmer', 'srmer']:
            for scc in ['lower', 'central', 'upper']:
                col = f'baseline_{cat}_lifetime_damages_climate_{mer}_{scc}'
                assert col in df_main.columns, f"Missing damages column {col}"


# =============================================================================
# MASKING TESTS
# =============================================================================

@patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.lookup_climate_impact_scc')
@patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.calculate_fossil_fuel_emissions')
@patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.get_electricity_consumption_for_year')
@patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.define_scenario_params')
def test_mp_invalid_homes_masked(mock_params, mock_elec, mock_fossil, mock_scc,
                                  climate_df, mock_emissions_electricity_climate,
                                  mock_scc_lookup):
    """Invalid homes get NaN in measure package lifetime climate columns."""
    mock_params.return_value = ('iraRef_mp8_', 'MidCase', {}, mock_emissions_electricity_climate, {}, {})
    mock_elec.return_value = pd.Series(1000.0, index=climate_df.index)
    mock_fossil.return_value = {
        'co2e': pd.Series(0.5, index=climate_df.index),
        'so2': pd.Series(0.001, index=climate_df.index),
        'nox': pd.Series(0.002, index=climate_df.index),
        'pm25': pd.Series(0.0005, index=climate_df.index),
    }
    mock_scc.__getitem__ = lambda self, key: mock_scc_lookup[key]

    df_main, _ = calculate_lifetime_climate_impacts(
        climate_df, menu_mp=8, policy_scenario='AEO2023 Reference Case', verbose=False
    )

    for cat in FULL_EQUIPMENT_SPECS:
        include_mask = climate_df[f'include_{cat}']
        upgrade_col = FULL_UPGRADE_COLUMNS.get(cat)
        if upgrade_col and upgrade_col in climate_df.columns:
            retrofit_mask = climate_df[upgrade_col].notna()
            invalid_mask = ~(include_mask & retrofit_mask)
        else:
            invalid_mask = ~include_mask
        if not invalid_mask.any():
            continue
        for mer in ['lrmer', 'srmer']:
            col = f'iraRef_mp8_{cat}_lifetime_mt_co2e_{mer}'
            if col in df_main.columns:
                assert df_main.loc[invalid_mask, col].isna().all(), \
                    f"Invalid homes should have NaN in {col}"


# =============================================================================
# FULL LIFETIME YEAR COVERAGE
# =============================================================================

@patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.lookup_climate_impact_scc')
@patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.calculate_fossil_fuel_emissions')
@patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.get_electricity_consumption_for_year')
@patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.define_scenario_params')
def test_all_years_in_detailed(mock_params, mock_elec, mock_fossil, mock_scc,
                                climate_df, mock_emissions_electricity_climate,
                                mock_scc_lookup):
    """df_detailed has annual columns for every year of the full lifetime."""
    mock_params.return_value = ('baseline_', 'MidCase', {}, mock_emissions_electricity_climate, {}, {})
    mock_elec.return_value = pd.Series(1000.0, index=climate_df.index)
    mock_fossil.return_value = {
        'co2e': pd.Series(0.5, index=climate_df.index),
        'so2': pd.Series(0.001, index=climate_df.index),
        'nox': pd.Series(0.002, index=climate_df.index),
        'pm25': pd.Series(0.0005, index=climate_df.index),
    }
    mock_scc.__getitem__ = lambda self, key: mock_scc_lookup[key]

    _, df_detailed = calculate_lifetime_climate_impacts(
        climate_df, menu_mp=0, policy_scenario='AEO2023 Reference Case', verbose=False
    )

    for cat, lifetime in FULL_EQUIPMENT_SPECS.items():
        for year in range(BASE_YEAR, BASE_YEAR + lifetime):
            # Check that at least one annual column exists for this year
            annual_cols = [c for c in df_detailed.columns if f'baseline_{year}_{cat}_' in c]
            assert len(annual_cols) > 0, \
                f"Missing annual columns for {cat} year {year}"


# =============================================================================
# calculate_climate_emissions_and_damages (unit test)
# =============================================================================

def test_calculate_climate_emissions_and_damages_returns_three_dicts(climate_df, mock_emissions_electricity_climate):
    """Returns (climate_results, annual_emissions, annual_damages) tuple."""
    from cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity import (
        calculate_climate_emissions_and_damages,
    )

    # Patch the SCC lookup at module level
    scc_lookup = {
        'lower': {BASE_YEAR: 20},
        'central': {BASE_YEAR: 50},
        'upper': {BASE_YEAR: 100},
    }

    with patch('cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.lookup_climate_impact_scc', scc_lookup):
        climate_results, annual_emissions, annual_damages = calculate_climate_emissions_and_damages(
            df=climate_df,
            category='heating',
            year_label=BASE_YEAR,
            lookup_emissions_electricity_climate=mock_emissions_electricity_climate,
            cambium_scenario='MidCase',
            total_fossil_fuel_emissions={'co2e': pd.Series(0.5, index=climate_df.index)},
            scenario_prefix='baseline_',
            menu_mp=0,
        )

    assert isinstance(climate_results, dict)
    assert isinstance(annual_emissions, dict)
    assert isinstance(annual_damages, dict)

    # Should have LRMER and SRMER emissions
    assert 'lrmer' in annual_emissions
    assert 'srmer' in annual_emissions

    # Should have damages for each (MER, SCC) pair
    for mer in ['lrmer', 'srmer']:
        for scc in ['lower', 'central', 'upper']:
            assert (mer, scc) in annual_damages


# Import helper for inline usage
def calculate_lifetime_climate_impacts(*args, **kwargs):
    from cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity import (
        calculate_lifetime_climate_impacts as fn,
    )
    return fn(*args, **kwargs)
