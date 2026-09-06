"""Tests for cmu_tare_model.utils.degree_day_consumption_utils module.

Verifies degree-day (HDD + CDD) adjusted consumption including the new
cooling category CDD adjustments and category-specific rules.
"""

import pytest
import pandas as pd
import numpy as np

from cmu_tare_model.constants import ANCHOR_YEAR

MODULE = 'cmu_tare_model.utils.degree_day_consumption_utils'


@pytest.fixture
def consumption_df():
    """DataFrame with consumption columns, census_division, and cooling data."""
    n = 5
    return pd.DataFrame({
        'census_division': ['Pacific', 'Pacific', 'Mountain', 'Mountain', 'Pacific'],
        'base_electricity_heating_consumption': [1000.0, 1200.0, 800.0, 900.0, 1100.0],
        'base_naturalGas_heating_consumption': [2000.0, 2200.0, 1800.0, 1900.0, 2100.0],
        'base_propane_heating_consumption': [500.0, 600.0, 400.0, 450.0, 550.0],
        'base_fuelOil_heating_consumption': [300.0, 350.0, 250.0, 280.0, 320.0],
        'base_electricity_cooling_consumption': [600.0, 700.0, 500.0, 550.0, 650.0],
        'base_electricity_waterHeating_consumption': [800.0, 900.0, 700.0, 750.0, 850.0],
        'base_electricity_cooking_consumption': [200.0, 220.0, 180.0, 190.0, 210.0],
        'base_naturalGas_cooking_consumption': [300.0, 320.0, 280.0, 290.0, 310.0],
        'base_propane_cooking_consumption': [100.0, 110.0, 90.0, 95.0, 105.0],
        'mp8_heating_consumption': [600.0, 700.0, 500.0, 550.0, 650.0],
        'mp8_cooling_consumption': [300.0, 350.0, 250.0, 275.0, 325.0],
    })


# ── apply_degree_day_adjustment ──────────────────────────────────────────────

def test_apply_degree_day_heating_uses_hdd():
    from cmu_tare_model.utils.degree_day_consumption_utils import apply_degree_day_adjustment
    consumption = pd.Series([1000.0, 2000.0])
    hdd_factor = pd.Series([1.1, 0.9])
    result = apply_degree_day_adjustment(consumption, 'heating', hdd_factor=hdd_factor)
    expected = pd.Series([1100.0, 1800.0])
    pd.testing.assert_series_equal(result, expected)


def test_apply_degree_day_cooling_uses_cdd():
    from cmu_tare_model.utils.degree_day_consumption_utils import apply_degree_day_adjustment
    consumption = pd.Series([1000.0, 2000.0])
    cdd_factor = pd.Series([1.2, 0.8])
    result = apply_degree_day_adjustment(consumption, 'cooling', cdd_factor=cdd_factor)
    expected = pd.Series([1200.0, 1600.0])
    pd.testing.assert_series_equal(result, expected)


def test_apply_degree_day_other_categories_unchanged():
    from cmu_tare_model.utils.degree_day_consumption_utils import apply_degree_day_adjustment
    consumption = pd.Series([1000.0, 2000.0])
    hdd_factor = pd.Series([1.5, 1.5])
    cdd_factor = pd.Series([1.5, 1.5])
    for cat in ['waterHeating', 'clothesDrying', 'cooking']:
        result = apply_degree_day_adjustment(consumption, cat, hdd_factor=hdd_factor, cdd_factor=cdd_factor)
        pd.testing.assert_series_equal(result, consumption)


def test_apply_degree_day_heating_no_hdd_unchanged():
    from cmu_tare_model.utils.degree_day_consumption_utils import apply_degree_day_adjustment
    consumption = pd.Series([1000.0, 2000.0])
    result = apply_degree_day_adjustment(consumption, 'heating', hdd_factor=None)
    pd.testing.assert_series_equal(result, consumption)


# ── get_hdd_factor_for_year ──────────────────────────────────────────────────

def test_get_hdd_factor_missing_census_division_raises():
    from cmu_tare_model.utils.degree_day_consumption_utils import get_hdd_factor_for_year
    df = pd.DataFrame({'other_col': [1, 2]})
    with pytest.raises(KeyError, match="census_division"):
        get_hdd_factor_for_year(df, ANCHOR_YEAR)


def test_get_hdd_factor_invalid_year_raises():
    from cmu_tare_model.utils.degree_day_consumption_utils import get_hdd_factor_for_year
    df = pd.DataFrame({'census_division': ['Pacific']})
    with pytest.raises(ValueError, match="Invalid year_label"):
        get_hdd_factor_for_year(df, 2019)


def test_get_hdd_factor_returns_series(consumption_df):
    from cmu_tare_model.utils.degree_day_consumption_utils import get_hdd_factor_for_year
    result = get_hdd_factor_for_year(consumption_df, ANCHOR_YEAR)
    assert isinstance(result, pd.Series)
    assert len(result) == len(consumption_df)


# ── get_cdd_factor_for_year ──────────────────────────────────────────────────

def test_get_cdd_factor_missing_census_division_raises():
    from cmu_tare_model.utils.degree_day_consumption_utils import get_cdd_factor_for_year
    df = pd.DataFrame({'other_col': [1, 2]})
    with pytest.raises(KeyError, match="census_division"):
        get_cdd_factor_for_year(df, ANCHOR_YEAR)


def test_get_cdd_factor_invalid_year_raises():
    from cmu_tare_model.utils.degree_day_consumption_utils import get_cdd_factor_for_year
    df = pd.DataFrame({'census_division': ['Pacific']})
    with pytest.raises(ValueError, match="Invalid year_label"):
        get_cdd_factor_for_year(df, 2070)


def test_get_cdd_factor_returns_series(consumption_df):
    from cmu_tare_model.utils.degree_day_consumption_utils import get_cdd_factor_for_year
    result = get_cdd_factor_for_year(consumption_df, ANCHOR_YEAR)
    assert isinstance(result, pd.Series)
    assert len(result) == len(consumption_df)


# ── get_total_baseline_consumption ───────────────────────────────────────────

def test_total_baseline_cooling_uses_electricity_only(consumption_df, monkeypatch):
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {
        'heating': 15, 'cooling': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15
    })
    from cmu_tare_model.utils.degree_day_consumption_utils import get_total_baseline_consumption
    result = get_total_baseline_consumption(consumption_df, 'cooling', ANCHOR_YEAR)
    assert isinstance(result, pd.Series)
    assert len(result) == len(consumption_df)


def test_total_baseline_cooking_includes_three_fuels(consumption_df, monkeypatch):
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {
        'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15
    })
    from cmu_tare_model.utils.degree_day_consumption_utils import get_total_baseline_consumption
    result = get_total_baseline_consumption(consumption_df, 'cooking', ANCHOR_YEAR)
    expected = (consumption_df['base_electricity_cooking_consumption'] +
                consumption_df['base_naturalGas_cooking_consumption'] +
                consumption_df['base_propane_cooking_consumption'])
    pd.testing.assert_series_equal(result, expected)


def test_total_baseline_invalid_category_raises(consumption_df, monkeypatch):
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {'heating': 15})
    from cmu_tare_model.utils.degree_day_consumption_utils import get_total_baseline_consumption
    with pytest.raises(ValueError, match="Unknown fuel pattern"):
        get_total_baseline_consumption(consumption_df, 'invalid', ANCHOR_YEAR)


# ── get_electricity_consumption_for_year ─────────────────────────────────────

def test_electricity_consumption_baseline(consumption_df, monkeypatch):
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {
        'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15
    })
    from cmu_tare_model.utils.degree_day_consumption_utils import get_electricity_consumption_for_year
    result = get_electricity_consumption_for_year(consumption_df, 'heating', ANCHOR_YEAR, menu_mp=0)
    assert isinstance(result, pd.Series)


def test_electricity_consumption_retrofit(consumption_df, monkeypatch):
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {
        'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15
    })
    from cmu_tare_model.utils.degree_day_consumption_utils import get_electricity_consumption_for_year
    result = get_electricity_consumption_for_year(consumption_df, 'heating', ANCHOR_YEAR, menu_mp=8)
    pd.testing.assert_series_equal(result, consumption_df['mp8_heating_consumption'], check_names=False)


def test_electricity_consumption_invalid_category_raises(consumption_df, monkeypatch):
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {'heating': 15})
    from cmu_tare_model.utils.degree_day_consumption_utils import get_electricity_consumption_for_year
    with pytest.raises(ValueError, match="Invalid category"):
        get_electricity_consumption_for_year(consumption_df, 'invalid', ANCHOR_YEAR, 0)


# ── get_hdd_adjusted_consumption ─────────────────────────────────────────────

def test_hdd_adjusted_consumption_baseline(consumption_df, monkeypatch):
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {
        'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15
    })
    from cmu_tare_model.utils.degree_day_consumption_utils import get_hdd_adjusted_consumption
    result = get_hdd_adjusted_consumption(consumption_df, 'heating', ANCHOR_YEAR, menu_mp=0)
    assert isinstance(result, pd.Series)
    assert len(result) == len(consumption_df)


def test_hdd_adjusted_consumption_retrofit(consumption_df, monkeypatch):
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {
        'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15
    })
    from cmu_tare_model.utils.degree_day_consumption_utils import get_hdd_adjusted_consumption
    result = get_hdd_adjusted_consumption(consumption_df, 'heating', ANCHOR_YEAR, menu_mp=8)
    assert isinstance(result, pd.Series)


def test_hdd_adjusted_consumption_invalid_category_raises(consumption_df, monkeypatch):
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {'heating': 15})
    from cmu_tare_model.utils.degree_day_consumption_utils import get_hdd_adjusted_consumption
    with pytest.raises(ValueError, match="Invalid category"):
        get_hdd_adjusted_consumption(consumption_df, 'invalid', ANCHOR_YEAR, 0)
