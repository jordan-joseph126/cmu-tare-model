"""Shared fixtures for the TARE Model test suite.

This file pre-mocks data-loading modules to prevent FileNotFoundError
from module-level Excel/CSV reads during import.
"""

import sys
import types
from unittest.mock import MagicMock

# ── Pre-mock data-loading modules before ANY source code import ──────────────
# These modules load data files at module level. We inject mock modules into
# sys.modules so that when source code does `from ... import X`, it gets a
# MagicMock instead of triggering a FileNotFoundError.

_DATA_MODULES_TO_MOCK = [
    'cmu_tare_model.public_impact.data_processing.create_lookup_emissions_electricity_climate',
    'cmu_tare_model.public_impact.data_processing.create_lookup_climate_impact_scc',
    'cmu_tare_model.private_impact.data_processing.create_lookup_fuel_prices',
    'cmu_tare_model.private_impact.data_processing.process_income_data_for_rebates',
    'cmu_tare_model.utils.precompute_hdd_factors',
]

# Only mock the leaf modules themselves (not parent packages which contain real code).
for mod_path in _DATA_MODULES_TO_MOCK:
    if mod_path not in sys.modules:
        mock_mod = MagicMock()
        mock_mod.__name__ = mod_path
        mock_mod.__package__ = '.'.join(mod_path.split('.')[:-1])
        sys.modules[mod_path] = mock_mod

# ── Now safe to import standard libraries and project modules ────────────────
import pytest
import pandas as pd
import numpy as np

# Full production lifetimes — never shorten these
FULL_EQUIPMENT_SPECS = {
    'heating': 15,
    'waterHeating': 12,
    'clothesDrying': 13,
    'cooking': 15,
}

FULL_UPGRADE_COLUMNS = {
    'heating': 'upgrade_hvac_heating_efficiency',
    'waterHeating': 'upgrade_water_heater_efficiency',
    'clothesDrying': 'upgrade_clothes_dryer',
    'cooking': 'upgrade_cooking_range',
}

FULL_FUEL_MAPPING = {
    'Electricity': 'electricity',
    'Natural Gas': 'naturalGas',
    'Fuel Oil': 'fuelOil',
    'Propane': 'propane',
}

BASE_YEAR = 2024


def create_sample_homes_df(n_homes=10, categories=None, base_year=2024):
    """Create a minimal DataFrame with mixed valid/invalid homes.

    Provides data for all equipment categories across full production lifetimes.
    """
    if categories is None:
        categories = list(FULL_EQUIPMENT_SPECS.keys())

    np.random.seed(42)

    data = {
        'home_id': range(n_homes),
        'state': np.random.choice(['CA', 'TX', 'NY', 'FL', 'IL'], n_homes),
        'census_division': np.random.choice(
            ['Pacific', 'West South Central', 'Middle Atlantic', 'South Atlantic', 'East North Central'],
            n_homes,
        ),
        'county_fips': np.random.choice(['06001', '48201', '36061', '12086', '17031'], n_homes),
        'gea_region': np.random.choice(['CAL', 'TXS', 'NYC', 'FLA', 'CEN'], n_homes),
    }

    # Fuel type columns
    fuels = ['Electricity', 'Natural Gas', 'Propane', 'Fuel Oil']
    for cat in categories:
        # Inclusion flags — ~70% valid
        data[f'include_{cat}'] = np.random.choice([True, False], n_homes, p=[0.7, 0.3])
        data[f'valid_fuel_{cat}'] = data[f'include_{cat}']
        data[f'base_{cat}_fuel'] = np.random.choice(fuels, n_homes)

    # Technology type columns for heating
    if 'heating' in categories:
        data['heating_type'] = np.random.choice(
            ['Natural Gas Fuel Furnace', 'Electricity ASHP', 'Propane Fuel Furnace'],
            n_homes,
        )
        data['valid_tech_heating'] = data['include_heating']

    # Upgrade columns (NaN = no retrofit, non-NaN = retrofit)
    for cat, col in FULL_UPGRADE_COLUMNS.items():
        if cat in categories:
            vals = [None] * n_homes
            for i in range(n_homes):
                if data[f'include_{cat}'][i] and np.random.random() > 0.3:
                    vals[i] = 'ASHP' if cat == 'heating' else 'Electric Heat Pump'
            data[col] = vals

    # Consumption columns for ALL years of each category's lifetime
    for cat in categories:
        lifetime = FULL_EQUIPMENT_SPECS[cat]
        for year in range(base_year, base_year + lifetime):
            # Baseline consumption (fuel-specific for baseline)
            for fuel_name, fuel_key in FULL_FUEL_MAPPING.items():
                data[f'base_{fuel_key}_{cat}_consumption'] = np.random.uniform(500, 2000, n_homes)

            # Total baseline consumption
            data[f'baseline_{year}_{cat}_consumption'] = np.random.uniform(1000, 3000, n_homes)

            # Post-retrofit consumption for various MPs
            for mp in [3, 4, 8]:
                data[f'mp{mp}_{year}_{cat}_consumption'] = np.random.uniform(400, 1500, n_homes)
                data[f'mp{mp}_{cat}_consumption'] = np.random.uniform(400, 1500, n_homes)

    # Discount rate columns
    data['public_discount_rate'] = 0.02
    data['private_discount_rate_fixed_low'] = 0.02
    data['private_discount_rate_fixed_base'] = 0.07
    data['private_discount_rate_fixed_high'] = 0.12
    data['private_discount_rate_variable'] = np.random.uniform(0.07, 0.45, n_homes)

    return pd.DataFrame(data)


@pytest.fixture
def sample_homes_df():
    """10-row DataFrame with mixed valid/invalid homes and full lifetime data."""
    return create_sample_homes_df(n_homes=10)


@pytest.fixture
def small_df():
    """5-row DataFrame for lightweight tests."""
    return create_sample_homes_df(n_homes=5, categories=['heating'])


@pytest.fixture
def mock_scenario_params():
    """Mock define_scenario_params for predictable test behavior.

    Mirrors the current single-scenario 5-tuple return:
    (scenario_prefix, cambium_scenario, fossil, elec_climate, fuel_prices).
    """
    mock_fossil = {'Natural Gas': {'co2e': 0.005}}
    mock_elec_climate = {}
    mock_fuel_prices = {}

    def mock_define(menu_mp, policy_scenario='2025 Reference Case', verbose=False):
        if int(menu_mp) == 0:
            prefix = 'baseline_'
        else:
            prefix = f'ref2025_mp{menu_mp}_'
        return (prefix, 'MidCase', mock_fossil, mock_elec_climate, mock_fuel_prices)

    return mock_define


@pytest.fixture
def mock_fuel_prices():
    """Nested fuel price lookup dictionary for testing."""
    prices = {}
    for state in ['CA', 'TX', 'NY', 'FL', 'IL']:
        prices[state] = {}
        for fuel in ['electricity', 'naturalGas', 'fuelOil', 'propane']:
            prices[state][fuel] = {}
            for scenario in ['No Inflation Reduction Act', 'AEO2023 Reference Case']:
                prices[state][fuel][scenario] = {}
                for year in range(2024, 2040):
                    prices[state][fuel][scenario][year] = 0.12 + (year - 2024) * 0.005

    # Also add census divisions for non-electric/gas fuels
    for div in ['Pacific', 'West South Central', 'Middle Atlantic', 'South Atlantic', 'East North Central']:
        if div not in prices:
            prices[div] = {}
        for fuel in ['fuelOil', 'propane']:
            prices[div][fuel] = {}
            for scenario in ['No Inflation Reduction Act', 'AEO2023 Reference Case']:
                prices[div][fuel][scenario] = {}
                for year in range(2024, 2040):
                    prices[div][fuel][scenario][year] = 0.15 + (year - 2024) * 0.005

    return prices
