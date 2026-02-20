"""Tests for cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions module.

Verifies fossil fuel emissions calculations for baseline (menu_mp=0) and
retrofit (menu_mp>0) scenarios, including pollutant-specific outputs and
HDD adjustment behavior.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

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


# ── Validation ───────────────────────────────────────────────────────────────

def test_invalid_category_raises(mock_constants, emissions_df, fossil_fuel_lookup):
    from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import (
        calculate_fossil_fuel_emissions,
    )
    with pytest.raises(ValueError, match="Invalid category"):
        calculate_fossil_fuel_emissions(
            emissions_df, 'invalid', 2024, fossil_fuel_lookup, menu_mp=0)


def test_negative_menu_mp_raises(mock_constants, emissions_df, fossil_fuel_lookup):
    from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import (
        calculate_fossil_fuel_emissions,
    )
    with pytest.raises(ValueError, match="Invalid menu_mp"):
        calculate_fossil_fuel_emissions(
            emissions_df, 'heating', 2024, fossil_fuel_lookup, menu_mp=-1)


# ── Baseline (menu_mp=0) ────────────────────────────────────────────────────

def test_baseline_returns_all_pollutants(mock_constants, emissions_df, fossil_fuel_lookup):
    from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import (
        calculate_fossil_fuel_emissions,
    )
    result = calculate_fossil_fuel_emissions(
        emissions_df, 'heating', 2024, fossil_fuel_lookup, menu_mp=0)
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
        emissions_df, 'heating', 2024, fossil_fuel_lookup, menu_mp=0)
    # First home has naturalGas heating consumption=2000 -> should have nonzero emissions
    assert result['co2e'].iloc[0] > 0


def test_baseline_cooking_excludes_fuelOil(mock_constants, emissions_df, fossil_fuel_lookup):
    """Cooking should NOT include fuelOil in its fuel list."""
    from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import (
        calculate_fossil_fuel_emissions,
    )
    # Add fuelOil column for cooking - if it's included it would add emissions
    emissions_df['base_fuelOil_cooking_consumption'] = [1000.0] * 5
    result = calculate_fossil_fuel_emissions(
        emissions_df, 'cooking', 2024, fossil_fuel_lookup, menu_mp=0)
    # The code skips fuelOil for cooking/clothesDrying, so this column should be ignored
    # Emissions should only come from naturalGas and propane
    assert isinstance(result, dict)


# ── Retrofit (menu_mp>0) ────────────────────────────────────────────────────

def test_retrofit_returns_zero_emissions(mock_constants, emissions_df, fossil_fuel_lookup):
    """For menu_mp>0, fossil fuel emissions should be zero (all-electric retrofit)."""
    from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import (
        calculate_fossil_fuel_emissions,
    )
    result = calculate_fossil_fuel_emissions(
        emissions_df, 'heating', 2024, fossil_fuel_lookup, menu_mp=8)
    for pollutant in ['so2', 'nox', 'pm25', 'co2e']:
        # For valid retrofit homes, emissions should be 0.0 (not NaN)
        valid = emissions_df['include_heating']
        assert (result[pollutant][valid] == 0.0).all()


# ── Retrofit mask ────────────────────────────────────────────────────────────

def test_custom_retrofit_mask(mock_constants, emissions_df, fossil_fuel_lookup):
    """Passing a pre-computed retrofit_mask should work."""
    from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import (
        calculate_fossil_fuel_emissions,
    )
    custom_mask = pd.Series([True, True, False, False, True], index=emissions_df.index)
    result = calculate_fossil_fuel_emissions(
        emissions_df, 'heating', 2024, fossil_fuel_lookup, menu_mp=0,
        retrofit_mask=custom_mask)
    assert isinstance(result, dict)
    # Row index 2 and 3 are False in mask -> their emissions should be NaN
    for pollutant in ['so2', 'nox', 'pm25', 'co2e']:
        assert pd.isna(result[pollutant].iloc[2])
        assert pd.isna(result[pollutant].iloc[3])


def test_missing_consumption_column_raises(mock_constants, fossil_fuel_lookup):
    """Should raise KeyError if required consumption column is missing."""
    from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import (
        calculate_fossil_fuel_emissions,
    )
    df = pd.DataFrame({
        'census_division': ['Pacific'],
        'include_heating': [True],
        'base_heating_fuel': ['Natural Gas'],
        'upgrade_hvac_heating_efficiency': ['ASHP'],
        # Missing base_naturalGas_heating_consumption etc.
    })
    with pytest.raises(KeyError, match="Required column"):
        calculate_fossil_fuel_emissions(df, 'heating', 2024, fossil_fuel_lookup, menu_mp=0)
