"""Tests for calculate_lifetime_fuel_costs (private_impact/calculate_lifetime_fuel_costs.py).

The order dependence flagged here previously (a KeyError on 'waterHeating' when
the private_impact folder ran on its own) was fixed on 12 Aug 2026. The cause
was not a leaking fixture: the module under test and validation_framework.py
each copy EQUIPMENT_SPECS into their own namespace at import time, so patching
only cmu_tare_model.constants left those copies alone, and the outcome depended
on which test file imported them first. mock_constants below now patches every
copy.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from cmu_tare_model.tests.conftest import (
    FULL_EQUIPMENT_SPECS, FULL_UPGRADE_COLUMNS, FULL_FUEL_MAPPING,
    create_sample_homes_df, BASE_YEAR,
)

MODULE = 'cmu_tare_model.private_impact.calculate_lifetime_fuel_costs'


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def mock_constants(monkeypatch):
    """Mock constants with full production lifetimes.

    Both the module under test and validation_framework.py copy these names
    into their own namespace with `from cmu_tare_model.constants import ...`
    when they are first imported, so patching cmu_tare_model.constants alone
    leaves those copies pointing at the real two-category spec. Every copy is
    patched here so the result does not depend on which test file imported
    these modules first.
    """
    monkeypatch.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', FULL_EQUIPMENT_SPECS)
    monkeypatch.setattr('cmu_tare_model.constants.UPGRADE_COLUMNS', FULL_UPGRADE_COLUMNS)
    monkeypatch.setattr('cmu_tare_model.constants.FUEL_MAPPING', FULL_FUEL_MAPPING)
    monkeypatch.setattr('cmu_tare_model.constants.VERBOSE', False)
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', FULL_EQUIPMENT_SPECS)
    monkeypatch.setattr(f'{MODULE}.FUEL_MAPPING', FULL_FUEL_MAPPING)
    monkeypatch.setattr(
        'cmu_tare_model.utils.validation_framework.EQUIPMENT_SPECS',
        FULL_EQUIPMENT_SPECS)
    monkeypatch.setattr(
        'cmu_tare_model.utils.validation_framework.UPGRADE_COLUMNS',
        FULL_UPGRADE_COLUMNS)


@pytest.fixture
def fuel_cost_df():
    """DataFrame with all required columns for fuel cost calculation."""
    n = 8
    np.random.seed(42)
    data = {
        'state': np.random.choice(['CA', 'TX', 'NY', 'FL'], n),
        'census_division': np.random.choice(['Pacific', 'West South Central', 'Middle Atlantic', 'South Atlantic'], n),
        'include_heating': [True, True, False, True, True, False, True, True],
        'include_waterHeating': [True, False, True, True, False, True, True, False],
        'include_clothesDrying': [True, True, True, False, True, True, False, True],
        'include_cooking': [True, True, True, True, False, True, True, False],
        'valid_fuel_heating': [True, True, False, True, True, False, True, True],
        'base_heating_fuel': ['Natural Gas', 'Electricity', 'Natural Gas', 'Propane', 'Electricity', 'Natural Gas', 'Fuel Oil', 'Electricity'],
        'base_waterHeating_fuel': ['Electricity', 'Natural Gas', 'Electricity', 'Natural Gas', 'Propane', 'Electricity', 'Natural Gas', 'Electricity'],
        'base_clothesDrying_fuel': ['Electricity', 'Natural Gas', 'Propane', 'Electricity', 'Natural Gas', 'Electricity', 'Natural Gas', 'Propane'],
        'base_cooking_fuel': ['Natural Gas', 'Propane', 'Natural Gas', 'Propane', 'Natural Gas', 'Propane', 'Natural Gas', 'Propane'],
        'upgrade_hvac_heating_efficiency': ['ASHP', 'MSHP', None, 'ASHP', None, None, 'ASHP', 'MSHP'],
        'upgrade_water_heater_efficiency': ['HP', None, 'HP', None, None, 'HP', None, None],
        'upgrade_clothes_dryer': [None, 'HP', None, None, 'HP', None, None, 'HP'],
        'upgrade_cooking_range': [None, 'Electric', None, 'Electric', None, None, 'Electric', None],
    }

    for cat, lifetime in FULL_EQUIPMENT_SPECS.items():
        for year in range(BASE_YEAR, BASE_YEAR + lifetime):
            data[f'baseline_{year}_{cat}_consumption'] = np.random.uniform(800, 2500, n)
            for mp in [0, 3, 4, 8]:
                if mp > 0:
                    data[f'mp{mp}_{year}_{cat}_consumption'] = np.random.uniform(300, 1200, n)
                    data[f'mp{mp}_{cat}_consumption'] = np.random.uniform(300, 1200, n)

        for fuel_key in FULL_FUEL_MAPPING.values():
            data[f'base_{fuel_key}_{cat}_consumption'] = np.random.uniform(500, 2000, n)

    return pd.DataFrame(data)


@pytest.fixture
def fuel_prices():
    """Nested fuel price lookup with all states, fuels, scenarios, and years."""
    prices = {}
    states = ['CA', 'TX', 'NY', 'FL']
    divisions = ['Pacific', 'West South Central', 'Middle Atlantic', 'South Atlantic']

    for loc in states + divisions:
        prices[loc] = {}
        for fuel in ['electricity', 'naturalGas', 'fuelOil', 'propane']:
            prices[loc][fuel] = {}
            for scenario in ['2025 Reference Case']:
                prices[loc][fuel][scenario] = {}
                for year in range(BASE_YEAR, BASE_YEAR + 16):
                    prices[loc][fuel][scenario][year] = 0.10 + (year - BASE_YEAR) * 0.002
    return prices


# =============================================================================
# PARAMETER VALIDATION
# =============================================================================

def test_fuel_costs_invalid_policy_scenario(fuel_cost_df):
    """Raises ValueError for invalid policy_scenario."""
    from cmu_tare_model.private_impact.calculate_lifetime_fuel_costs import calculate_lifetime_fuel_costs

    with pytest.raises((ValueError, RuntimeError)):
        calculate_lifetime_fuel_costs(fuel_cost_df, menu_mp=0, policy_scenario='Invalid')


def test_fuel_costs_empty_dataframe():
    """Returns empty DataFrames for empty input."""
    from cmu_tare_model.private_impact.calculate_lifetime_fuel_costs import calculate_lifetime_fuel_costs

    df_main, df_detailed = calculate_lifetime_fuel_costs(
        pd.DataFrame(), menu_mp=0, policy_scenario='2025 Reference Case', verbose=False
    )
    assert df_main.empty
    assert df_detailed.empty


# =============================================================================
# BASELINE CALCULATION (menu_mp=0)
# =============================================================================

def test_baseline_fuel_costs_output_structure(fuel_cost_df, fuel_prices):
    """Baseline calculation produces df_main and df_detailed with correct structure."""
    from cmu_tare_model.private_impact.calculate_lifetime_fuel_costs import calculate_lifetime_fuel_costs

    with patch(f'{MODULE}.define_scenario_params') as mock_params, \
         patch(f'{MODULE}.get_hdd_adjusted_consumption') as mock_hdd:
        mock_params.return_value = ('baseline_', 'MidCase', {}, {}, fuel_prices)
        mock_hdd.return_value = pd.Series(1000.0, index=fuel_cost_df.index)

        df_main, df_detailed = calculate_lifetime_fuel_costs(
            fuel_cost_df, menu_mp=0, policy_scenario='2025 Reference Case', verbose=False
        )

    assert isinstance(df_main, pd.DataFrame)
    assert isinstance(df_detailed, pd.DataFrame)
    assert len(df_main) == len(fuel_cost_df)

    for cat in FULL_EQUIPMENT_SPECS:
        lifetime_col = f'baseline_{cat}_lifetime_fuel_cost'
        assert lifetime_col in df_main.columns, f"Missing {lifetime_col} in df_main"


def test_baseline_invalid_homes_get_nan(fuel_cost_df, fuel_prices):
    """Invalid homes get NaN in lifetime fuel cost columns after masking."""
    from cmu_tare_model.private_impact.calculate_lifetime_fuel_costs import calculate_lifetime_fuel_costs

    with patch(f'{MODULE}.define_scenario_params') as mock_params, \
         patch(f'{MODULE}.get_hdd_adjusted_consumption') as mock_hdd:
        mock_params.return_value = ('baseline_', 'MidCase', {}, {}, fuel_prices)
        mock_hdd.return_value = pd.Series(1000.0, index=fuel_cost_df.index)

        df_main, _ = calculate_lifetime_fuel_costs(
            fuel_cost_df, menu_mp=0, policy_scenario='2025 Reference Case', verbose=False
        )

    for cat in FULL_EQUIPMENT_SPECS:
        col = f'baseline_{cat}_lifetime_fuel_cost'
        if col in df_main.columns:
            invalid_mask = ~fuel_cost_df[f'include_{cat}']
            if invalid_mask.any():
                assert df_main.loc[invalid_mask, col].isna().all(), \
                    f"Invalid homes should have NaN in {col}"


def test_baseline_lifetime_is_sum_of_yearly(fuel_cost_df, fuel_prices):
    """Lifetime fuel cost equals sum of annual costs (skipna=False)."""
    from cmu_tare_model.private_impact.calculate_lifetime_fuel_costs import calculate_lifetime_fuel_costs

    with patch(f'{MODULE}.define_scenario_params') as mock_params, \
         patch(f'{MODULE}.get_hdd_adjusted_consumption') as mock_hdd:
        mock_params.return_value = ('baseline_', 'MidCase', {}, {}, fuel_prices)
        mock_hdd.return_value = pd.Series(1000.0, index=fuel_cost_df.index)

        df_main, df_detailed = calculate_lifetime_fuel_costs(
            fuel_cost_df, menu_mp=0, policy_scenario='2025 Reference Case', verbose=False
        )

    for cat, lifetime in FULL_EQUIPMENT_SPECS.items():
        lifetime_col = f'baseline_{cat}_lifetime_fuel_cost'
        if lifetime_col not in df_main.columns:
            continue

        annual_cols = [f'baseline_{BASE_YEAR + y}_{cat}_fuel_cost'
                       for y in range(lifetime)
                       if f'baseline_{BASE_YEAR + y}_{cat}_fuel_cost' in df_detailed.columns]

        if annual_cols:
            valid_mask = fuel_cost_df[f'include_{cat}']
            expected_sum = df_detailed.loc[valid_mask, annual_cols].sum(axis=1, skipna=False)
            actual = df_main.loc[valid_mask, lifetime_col]

            for idx in valid_mask[valid_mask].index:
                if not np.isnan(actual.loc[idx]):
                    assert actual.loc[idx] == pytest.approx(expected_sum.loc[idx], rel=0.01), \
                        f"Lifetime != sum of annual for {cat} at idx {idx}"


# =============================================================================
# MEASURE PACKAGE CALCULATION (menu_mp=8)
# =============================================================================

def test_mp_fuel_costs_uses_scenario_prefix(fuel_cost_df, fuel_prices):
    """Measure package columns use the correct scenario prefix."""
    from cmu_tare_model.private_impact.calculate_lifetime_fuel_costs import calculate_lifetime_fuel_costs

    with patch(f'{MODULE}.define_scenario_params') as mock_params, \
         patch(f'{MODULE}.get_hdd_adjusted_consumption') as mock_hdd:
        mock_params.return_value = ('ref2025_mp8_', 'MidCase', {}, {}, fuel_prices)
        mock_hdd.return_value = pd.Series(1000.0, index=fuel_cost_df.index)

        df_main, _ = calculate_lifetime_fuel_costs(
            fuel_cost_df, menu_mp=8, policy_scenario='2025 Reference Case', verbose=False
        )

    for cat in FULL_EQUIPMENT_SPECS:
        expected_col = f'ref2025_mp8_{cat}_lifetime_fuel_cost'
        assert expected_col in df_main.columns, f"Missing {expected_col}"


# =============================================================================
# COLUMN TRACKING (regression test)
# =============================================================================

def test_all_lifetime_columns_are_tracked(fuel_cost_df, fuel_prices):
    """Lifetime columns are tracked in all_columns_to_mask (regression test)."""
    from cmu_tare_model.private_impact.calculate_lifetime_fuel_costs import calculate_lifetime_fuel_costs

    with patch(f'{MODULE}.define_scenario_params') as mock_params, \
         patch(f'{MODULE}.get_hdd_adjusted_consumption') as mock_hdd:
        mock_params.return_value = ('baseline_', 'MidCase', {}, {}, fuel_prices)
        mock_hdd.return_value = pd.Series(1000.0, index=fuel_cost_df.index)

        df_main, _ = calculate_lifetime_fuel_costs(
            fuel_cost_df, menu_mp=0, policy_scenario='2025 Reference Case', verbose=False
        )

    for cat in FULL_EQUIPMENT_SPECS:
        col = f'baseline_{cat}_lifetime_fuel_cost'
        if col in df_main.columns:
            invalid = ~fuel_cost_df[f'include_{cat}']
            if invalid.any():
                assert df_main.loc[invalid, col].isna().all(), \
                    f"Lifetime col {col} not masked for invalid homes — column tracking regression"


# =============================================================================
# MISSING COLUMNS
# =============================================================================

def test_missing_state_column_raises(fuel_cost_df, fuel_prices):
    """Raises error when 'state' column is missing."""
    from cmu_tare_model.private_impact.calculate_lifetime_fuel_costs import calculate_lifetime_fuel_costs

    df_no_state = fuel_cost_df.drop(columns=['state'])
    with pytest.raises((KeyError, RuntimeError)):
        with patch(f'{MODULE}.define_scenario_params') as mock_params:
            mock_params.return_value = ('baseline_', 'MidCase', {}, {}, fuel_prices)
            calculate_lifetime_fuel_costs(
                df_no_state, menu_mp=0, policy_scenario='2025 Reference Case', verbose=False
            )


# =============================================================================
# FULL LIFETIME COVERAGE
# =============================================================================

def test_all_years_processed_for_full_lifetime(fuel_cost_df, fuel_prices):
    """Every year of the full production lifetime produces an annual column."""
    from cmu_tare_model.private_impact.calculate_lifetime_fuel_costs import calculate_lifetime_fuel_costs

    with patch(f'{MODULE}.define_scenario_params') as mock_params, \
         patch(f'{MODULE}.get_hdd_adjusted_consumption') as mock_hdd:
        mock_params.return_value = ('baseline_', 'MidCase', {}, {}, fuel_prices)
        mock_hdd.return_value = pd.Series(1000.0, index=fuel_cost_df.index)

        _, df_detailed = calculate_lifetime_fuel_costs(
            fuel_cost_df, menu_mp=0, policy_scenario='2025 Reference Case', verbose=False
        )

    for cat, lifetime in FULL_EQUIPMENT_SPECS.items():
        for year in range(BASE_YEAR, BASE_YEAR + lifetime):
            annual_col = f'baseline_{year}_{cat}_fuel_cost'
            assert annual_col in df_detailed.columns, \
                f"Missing annual column {annual_col} — not all {lifetime} years processed"
