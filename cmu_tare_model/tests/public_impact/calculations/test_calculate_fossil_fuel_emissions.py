"""Tests for cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions module.

Verifies fossil fuel emissions for baseline (menu_mp=0) and retrofit (menu_mp>0)
scenarios, read from the consumption table (build_projected_consumption), which
is already degree-day adjusted and counts every component.
"""

import pytest
import pandas as pd
import numpy as np

from cmu_tare_model.constants import ANCHOR_YEAR

MODULE = 'cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions'


@pytest.fixture
def mock_constants(monkeypatch):
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {
        'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15
    })
    monkeypatch.setattr(f'{MODULE}.POLLUTANTS', ['so2', 'nox', 'pm25', 'co2e'])


@pytest.fixture
def emissions_df():
    """DataFrame with fossil fuel consumption columns."""
    n = 5
    return pd.DataFrame({
        'census_division': ['Pacific'] * n,
        'include_heating': [True, True, True, False, True],
        'include_cooking': [True, True, False, True, True],
        'base_heating_fuel': ['Natural Gas', 'Propane', 'Fuel Oil', 'Electricity', 'Natural Gas'],
        'base_cooking_fuel': ['Natural Gas', 'Propane', 'Natural Gas', 'Natural Gas', 'Propane'],
        'upgrade_hvac_heating_efficiency': ['ASHP', 'ASHP', 'ASHP', None, 'ASHP'],
        'base_naturalGas_heating_consumption': [2000.0, 0.0, 0.0, 500.0, 1500.0],
        'base_propane_heating_consumption': [0.0, 1500.0, 0.0, 0.0, 0.0],
        'base_fuelOil_heating_consumption': [0.0, 0.0, 1200.0, 0.0, 0.0],
        'base_naturalGas_cooking_consumption': [300.0, 0.0, 250.0, 200.0, 0.0],
        'base_propane_cooking_consumption': [0.0, 200.0, 0.0, 0.0, 150.0],
    })


@pytest.fixture
def fossil_fuel_lookup():
    """Emission factors lookup: fuel -> pollutant -> factor."""
    return {
        'naturalGas': {'so2': 1e-8, 'nox': 5e-7, 'pm25': 3e-8, 'co2e': 2.285e-4},
        'propane': {'so2': 2e-8, 'nox': 6e-7, 'pm25': 2e-8, 'co2e': 2.758e-4},
        'fuelOil': {'so2': 3e-8, 'nox': 7e-7, 'pm25': 4e-8, 'co2e': 3.039e-4},
    }


def _table(df: pd.DataFrame, prefix: str, category: str,
           year: int = ANCHOR_YEAR, source_prefix: str = 'base') -> pd.DataFrame:
    """Consumption-table columns for one category and year, built from the
    fixture's per-fuel columns. A fuel with no column gets no table column."""
    columns = {}
    total = pd.Series(0.0, index=df.index)
    for fuel in ['electricity', 'naturalGas', 'propane', 'fuelOil']:
        src = f'{source_prefix}_{fuel}_{category}_consumption'
        if src in df.columns:
            columns[f'{prefix}{year}_{category}_{fuel}_consumption'] = df[src]
            total = total + df[src].fillna(0)
    columns[f'{prefix}{year}_{category}_consumption'] = total
    return pd.DataFrame(columns, index=df.index)


# ── Validation ───────────────────────────────────────────────────────────────

def test_invalid_category_raises(mock_constants, emissions_df, fossil_fuel_lookup):
    from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import (
        calculate_fossil_fuel_emissions,
    )
    with pytest.raises(ValueError, match="Invalid category"):
        calculate_fossil_fuel_emissions(
            emissions_df, 'invalid', ANCHOR_YEAR, fossil_fuel_lookup, menu_mp=0,
            df_consumption=_table(emissions_df, 'baseline_', 'heating'),
            scenario_prefix='baseline_')


def test_negative_menu_mp_raises(mock_constants, emissions_df, fossil_fuel_lookup):
    from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import (
        calculate_fossil_fuel_emissions,
    )
    with pytest.raises(ValueError, match="Invalid menu_mp"):
        calculate_fossil_fuel_emissions(
            emissions_df, 'heating', ANCHOR_YEAR, fossil_fuel_lookup, menu_mp=-1,
            df_consumption=_table(emissions_df, 'baseline_', 'heating'),
            scenario_prefix='baseline_')


# ── Baseline (menu_mp=0) ────────────────────────────────────────────────────

def test_baseline_returns_all_pollutants(mock_constants, emissions_df, fossil_fuel_lookup):
    from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import (
        calculate_fossil_fuel_emissions,
    )
    result = calculate_fossil_fuel_emissions(
        emissions_df, 'heating', ANCHOR_YEAR, fossil_fuel_lookup, menu_mp=0,
        df_consumption=_table(emissions_df, 'baseline_', 'heating'),
        scenario_prefix='baseline_')
    assert isinstance(result, dict)
    for pollutant in ['so2', 'nox', 'pm25', 'co2e']:
        assert pollutant in result
        assert isinstance(result[pollutant], pd.Series)
        assert len(result[pollutant]) == len(emissions_df)


def test_baseline_emissions_nonzero_for_fossil_fuel_homes(mock_constants, emissions_df, fossil_fuel_lookup):
    from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import (
        calculate_fossil_fuel_emissions,
    )
    result = calculate_fossil_fuel_emissions(
        emissions_df, 'heating', ANCHOR_YEAR, fossil_fuel_lookup, menu_mp=0,
        df_consumption=_table(emissions_df, 'baseline_', 'heating'),
        scenario_prefix='baseline_')
    # First home burns 2,000 kWh of natural gas for heating.
    assert result['co2e'].iloc[0] == pytest.approx(2000.0 * 2.285e-4)


def test_fuel_without_table_column_adds_nothing(mock_constants, emissions_df, fossil_fuel_lookup):
    """Cooking's table has no fuel oil column, so cooking CO2e comes only from
    natural gas and propane."""
    from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import (
        calculate_fossil_fuel_emissions,
    )
    result = calculate_fossil_fuel_emissions(
        emissions_df, 'cooking', ANCHOR_YEAR, fossil_fuel_lookup, menu_mp=0,
        df_consumption=_table(emissions_df, 'baseline_', 'cooking'),
        scenario_prefix='baseline_')
    valid = emissions_df['include_cooking']
    expected = (emissions_df['base_naturalGas_cooking_consumption'] * 2.285e-4
                + emissions_df['base_propane_cooking_consumption'] * 2.758e-4)
    np.testing.assert_allclose(result['co2e'][valid], expected[valid])


# ── Retrofit (menu_mp>0) ────────────────────────────────────────────────────

def test_all_electric_retrofit_has_zero_fossil_emissions(mock_constants, emissions_df, fossil_fuel_lookup):
    """An all-electric retrofit's table has zero fossil use, so zero emissions."""
    from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import (
        calculate_fossil_fuel_emissions,
    )
    table = _table(emissions_df, 'ref2025_mp8_', 'heating')
    for col in table.columns:
        table[col] = 0.0
    result = calculate_fossil_fuel_emissions(
        emissions_df, 'heating', ANCHOR_YEAR, fossil_fuel_lookup, menu_mp=8,
        df_consumption=table, scenario_prefix='ref2025_mp8_')
    valid = emissions_df['include_heating']
    for pollutant in ['so2', 'nox', 'pm25', 'co2e']:
        assert (result[pollutant][valid] == 0.0).all()


def test_dual_fuel_retrofit_counts_gas_backup(mock_constants, emissions_df, fossil_fuel_lookup):
    """A retrofit that still burns gas (a dual-fuel heat pump's backup furnace)
    gets emissions from that gas."""
    from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import (
        calculate_fossil_fuel_emissions,
    )
    prefix = 'ref2025_mp8_'
    table = pd.DataFrame({
        f'{prefix}{ANCHOR_YEAR}_heating_electricity_consumption': 3000.0,
        f'{prefix}{ANCHOR_YEAR}_heating_naturalGas_consumption': 1000.0,
        f'{prefix}{ANCHOR_YEAR}_heating_consumption': 4000.0,
    }, index=emissions_df.index)
    result = calculate_fossil_fuel_emissions(
        emissions_df, 'heating', ANCHOR_YEAR, fossil_fuel_lookup, menu_mp=8,
        df_consumption=table, scenario_prefix=prefix)
    valid = emissions_df['include_heating']
    np.testing.assert_allclose(result['co2e'][valid], 1000.0 * 2.285e-4)


# ── Retrofit mask ────────────────────────────────────────────────────────────

def test_custom_retrofit_mask(mock_constants, emissions_df, fossil_fuel_lookup):
    """Passing a pre-computed retrofit_mask should work."""
    from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import (
        calculate_fossil_fuel_emissions,
    )
    custom_mask = pd.Series([True, True, False, False, True], index=emissions_df.index)
    result = calculate_fossil_fuel_emissions(
        emissions_df, 'heating', ANCHOR_YEAR, fossil_fuel_lookup, menu_mp=0,
        df_consumption=_table(emissions_df, 'baseline_', 'heating'),
        scenario_prefix='baseline_', retrofit_mask=custom_mask)
    assert isinstance(result, dict)
    # Row index 2 and 3 are False in mask -> their emissions should be NaN
    for pollutant in ['so2', 'nox', 'pm25', 'co2e']:
        assert pd.isna(result[pollutant].iloc[2])
        assert pd.isna(result[pollutant].iloc[3])


def test_missing_consumption_column_raises(mock_constants, emissions_df, fossil_fuel_lookup):
    """A table without this scenario's columns raises, rather than giving zero."""
    from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import (
        calculate_fossil_fuel_emissions,
    )
    wrong_prefix_table = _table(emissions_df, 'ref2025_mp8_', 'heating')
    with pytest.raises(ValueError, match="build_projected_consumption"):
        calculate_fossil_fuel_emissions(
            emissions_df, 'heating', ANCHOR_YEAR, fossil_fuel_lookup, menu_mp=0,
            df_consumption=wrong_prefix_table, scenario_prefix='baseline_')
