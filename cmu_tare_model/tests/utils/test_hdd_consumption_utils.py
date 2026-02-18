"""Tests for cmu_tare_model.utils.hdd_consumption_utils module.

Verifies HDD-adjusted consumption calculation including category-specific
rules (only heating gets HDD adjustment), fail-fast validation, and
baseline vs retrofit consumption paths.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

MODULE = 'cmu_tare_model.utils.hdd_consumption_utils'


@pytest.fixture
def consumption_df():
    """DataFrame with consumption columns and census_division."""
    n = 5
    return pd.DataFrame({
        'census_division': ['Pacific', 'Pacific', 'Mountain', 'Mountain', 'Pacific'],
        'base_electricity_heating_consumption': [1000.0, 1200.0, 800.0, 900.0, 1100.0],
        'base_naturalGas_heating_consumption': [2000.0, 2200.0, 1800.0, 1900.0, 2100.0],
        'base_propane_heating_consumption': [500.0, 600.0, 400.0, 450.0, 550.0],
        'base_fuelOil_heating_consumption': [300.0, 350.0, 250.0, 280.0, 320.0],
        'base_electricity_waterHeating_consumption': [800.0, 900.0, 700.0, 750.0, 850.0],
        'base_electricity_cooking_consumption': [200.0, 220.0, 180.0, 190.0, 210.0],
        'base_naturalGas_cooking_consumption': [300.0, 320.0, 280.0, 290.0, 310.0],
        'base_propane_cooking_consumption': [100.0, 110.0, 90.0, 95.0, 105.0],
        'mp8_heating_consumption': [600.0, 700.0, 500.0, 550.0, 650.0],
        'mp8_waterHeating_consumption': [400.0, 450.0, 350.0, 375.0, 425.0],
    })


# ── apply_hdd_adjustment ────────────────────────────────────────────────────

def test_apply_hdd_adjustment_heating_multiplies():
    from cmu_tare_model.utils.hdd_consumption_utils import apply_hdd_adjustment
    consumption = pd.Series([1000.0, 2000.0])
    hdd_factor = pd.Series([1.1, 0.9])
    result = apply_hdd_adjustment(consumption, 'heating', hdd_factor)
    expected = pd.Series([1100.0, 1800.0])
    pd.testing.assert_series_equal(result, expected)


def test_apply_hdd_adjustment_non_heating_unchanged():
    from cmu_tare_model.utils.hdd_consumption_utils import apply_hdd_adjustment
    consumption = pd.Series([1000.0, 2000.0])
    hdd_factor = pd.Series([1.5, 1.5])
    for cat in ['waterHeating', 'clothesDrying', 'cooking']:
        result = apply_hdd_adjustment(consumption, cat, hdd_factor)
        pd.testing.assert_series_equal(result, consumption)


# ── get_hdd_factor_for_year ──────────────────────────────────────────────────

def test_get_hdd_factor_missing_census_division_raises():
    from cmu_tare_model.utils.hdd_consumption_utils import get_hdd_factor_for_year
    df = pd.DataFrame({'other_col': [1, 2]})
    with pytest.raises(KeyError, match="census_division"):
        get_hdd_factor_for_year(df, 2024)


def test_get_hdd_factor_invalid_year_raises():
    from cmu_tare_model.utils.hdd_consumption_utils import get_hdd_factor_for_year
    df = pd.DataFrame({'census_division': ['Pacific']})
    with pytest.raises(ValueError, match="Invalid year_label"):
        get_hdd_factor_for_year(df, 2019)


def test_get_hdd_factor_invalid_year_type_raises():
    from cmu_tare_model.utils.hdd_consumption_utils import get_hdd_factor_for_year
    df = pd.DataFrame({'census_division': ['Pacific']})
    with pytest.raises(ValueError, match="Invalid year_label"):
        get_hdd_factor_for_year(df, 2024.5)


def test_get_hdd_factor_returns_series(consumption_df):
    """Even with empty lookup, returns Series of 1.0 (default fallback)."""
    from cmu_tare_model.utils.hdd_consumption_utils import get_hdd_factor_for_year
    result = get_hdd_factor_for_year(consumption_df, 2024)
    assert isinstance(result, pd.Series)
    assert len(result) == len(consumption_df)


# ── get_electricity_consumption_for_year ─────────────────────────────────────

def test_electricity_consumption_baseline_heating(consumption_df, monkeypatch):
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {
        'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15
    })
    from cmu_tare_model.utils.hdd_consumption_utils import get_electricity_consumption_for_year
    result = get_electricity_consumption_for_year(consumption_df, 'heating', 2024, menu_mp=0)
    assert isinstance(result, pd.Series)
    assert len(result) == len(consumption_df)


def test_electricity_consumption_retrofit(consumption_df, monkeypatch):
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {
        'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15
    })
    from cmu_tare_model.utils.hdd_consumption_utils import get_electricity_consumption_for_year
    result = get_electricity_consumption_for_year(consumption_df, 'heating', 2024, menu_mp=8)
    pd.testing.assert_series_equal(result, consumption_df['mp8_heating_consumption'], check_names=False)


def test_electricity_consumption_invalid_category_raises(consumption_df, monkeypatch):
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {'heating': 15})
    from cmu_tare_model.utils.hdd_consumption_utils import get_electricity_consumption_for_year
    with pytest.raises(ValueError, match="Invalid category"):
        get_electricity_consumption_for_year(consumption_df, 'invalid', 2024, 0)


def test_electricity_consumption_missing_column_raises(consumption_df, monkeypatch):
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {
        'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15
    })
    from cmu_tare_model.utils.hdd_consumption_utils import get_electricity_consumption_for_year
    with pytest.raises(ValueError, match="Required column"):
        get_electricity_consumption_for_year(consumption_df, 'heating', 2024, menu_mp=99)


# ── get_total_baseline_consumption ───────────────────────────────────────────

def test_total_baseline_consumption_heating_sums_all_fuels(consumption_df, monkeypatch):
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {
        'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15
    })
    from cmu_tare_model.utils.hdd_consumption_utils import get_total_baseline_consumption
    result = get_total_baseline_consumption(consumption_df, 'heating', 2024)
    assert isinstance(result, pd.Series)
    # Should be non-zero (sum of all fuels)
    assert (result > 0).all()


def test_total_baseline_consumption_cooking_excludes_fuelOil(consumption_df, monkeypatch):
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {
        'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15
    })
    from cmu_tare_model.utils.hdd_consumption_utils import get_total_baseline_consumption
    result = get_total_baseline_consumption(consumption_df, 'cooking', 2024)
    # Should be electricity + naturalGas + propane (no fuelOil for cooking)
    expected = (consumption_df['base_electricity_cooking_consumption'] +
                consumption_df['base_naturalGas_cooking_consumption'] +
                consumption_df['base_propane_cooking_consumption'])
    pd.testing.assert_series_equal(result, expected)


def test_total_baseline_consumption_invalid_category_raises(consumption_df, monkeypatch):
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {'heating': 15})
    from cmu_tare_model.utils.hdd_consumption_utils import get_total_baseline_consumption
    with pytest.raises(ValueError, match="Unknown fuel pattern"):
        get_total_baseline_consumption(consumption_df, 'invalid', 2024)


# ── get_hdd_adjusted_consumption ─────────────────────────────────────────────

def test_hdd_adjusted_consumption_baseline_vs_retrofit(consumption_df, monkeypatch):
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {
        'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15
    })
    from cmu_tare_model.utils.hdd_consumption_utils import get_hdd_adjusted_consumption
    baseline = get_hdd_adjusted_consumption(consumption_df, 'heating', 2024, menu_mp=0)
    retrofit = get_hdd_adjusted_consumption(consumption_df, 'heating', 2024, menu_mp=8)
    # Baseline sums all fuels; retrofit uses electricity only
    assert isinstance(baseline, pd.Series)
    assert isinstance(retrofit, pd.Series)


def test_hdd_adjusted_consumption_invalid_category_raises(consumption_df, monkeypatch):
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {'heating': 15})
    from cmu_tare_model.utils.hdd_consumption_utils import get_hdd_adjusted_consumption
    with pytest.raises(ValueError, match="Invalid category"):
        get_hdd_adjusted_consumption(consumption_df, 'invalid', 2024, 0)
