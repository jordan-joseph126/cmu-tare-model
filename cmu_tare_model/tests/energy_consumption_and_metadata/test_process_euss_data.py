"""Tests for cmu_tare_model.energy_consumption_and_metadata.process_euss_data module.

Verifies pure utility functions: extract_city_name, map_metro_status,
standardize_fuel_name, and preprocess_fuel_data.
"""

import pytest
import pandas as pd
import numpy as np


# ── extract_city_name ────────────────────────────────────────────────────────

def test_extract_city_name_standard_format():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import extract_city_name
    assert extract_city_name('CA, Los Angeles') == 'Los Angeles'


def test_extract_city_name_two_word_city():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import extract_city_name
    assert extract_city_name('NY, New York') == 'New York'


def test_extract_city_name_no_match_returns_original():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import extract_city_name
    assert extract_city_name('Los Angeles') == 'Los Angeles'


def test_extract_city_name_lowercase_state_returns_original():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import extract_city_name
    assert extract_city_name('ca, Los Angeles') == 'ca, Los Angeles'


def test_extract_city_name_non_string_returns_input():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import extract_city_name
    assert extract_city_name(42) == 42
    assert extract_city_name(None) is None


# ── map_metro_status ─────────────────────────────────────────────────────────

def test_map_metro_status_urban():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import map_metro_status
    assert map_metro_status('In metro area, principal city') == 'Urban'


def test_map_metro_status_suburban():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import map_metro_status
    assert map_metro_status('In metro area, not/partially in principal city') == 'Suburban'


def test_map_metro_status_rural():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import map_metro_status
    assert map_metro_status('Not/partially in metro area') == 'Rural'


def test_map_metro_status_unrecognized_returns_original():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import map_metro_status
    assert map_metro_status('Unknown area') == 'Unknown area'


def test_map_metro_status_non_string():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import map_metro_status
    assert map_metro_status(None) is None
    assert map_metro_status(42) == 42


def test_map_metro_status_strips_whitespace():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import map_metro_status
    assert map_metro_status('  In metro area, principal city  ') == 'Urban'


# ── standardize_fuel_name ────────────────────────────────────────────────────

def test_standardize_fuel_name_electricity():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import standardize_fuel_name
    assert standardize_fuel_name('Electric Heater') == 'Electricity'
    assert standardize_fuel_name('electric') == 'Electricity'


def test_standardize_fuel_name_natural_gas():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import standardize_fuel_name
    assert standardize_fuel_name('Gas Furnace') == 'Natural Gas'
    assert standardize_fuel_name('Natural Gas') == 'Natural Gas'


def test_standardize_fuel_name_propane():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import standardize_fuel_name
    assert standardize_fuel_name('Propane Heater') == 'Propane'


def test_standardize_fuel_name_fuel_oil():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import standardize_fuel_name
    assert standardize_fuel_name('Fuel Oil Boiler') == 'Fuel Oil'
    assert standardize_fuel_name('Oil Furnace') == 'Fuel Oil'


def test_standardize_fuel_name_unrecognized():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import standardize_fuel_name
    assert standardize_fuel_name('Wood Stove') is None


def test_standardize_fuel_name_nan():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import standardize_fuel_name
    assert standardize_fuel_name(np.nan) is None


def test_standardize_fuel_name_non_string():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import standardize_fuel_name
    assert standardize_fuel_name(42) is None


def test_standardize_fuel_name_case_insensitive():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import standardize_fuel_name
    assert standardize_fuel_name('ELECTRIC') == 'Electricity'
    assert standardize_fuel_name('PROPANE') == 'Propane'


# ── preprocess_fuel_data ─────────────────────────────────────────────────────

def test_preprocess_fuel_data_standardizes():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import preprocess_fuel_data
    df = pd.DataFrame({'fuel_col': ['Electric Heater', 'Gas Range', 'Propane', np.nan]})
    result = preprocess_fuel_data(df, 'fuel_col')
    assert result['fuel_col'].iloc[0] == 'Electricity'
    assert result['fuel_col'].iloc[1] == 'Natural Gas'
    assert result['fuel_col'].iloc[2] == 'Propane'
    assert pd.isna(result['fuel_col'].iloc[3])


def test_preprocess_fuel_data_missing_column_raises():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import preprocess_fuel_data
    df = pd.DataFrame({'other_col': [1, 2]})
    with pytest.raises(KeyError, match="Column"):
        preprocess_fuel_data(df, 'nonexistent_col')


def test_preprocess_fuel_data_non_dataframe_raises():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import preprocess_fuel_data
    with pytest.raises(TypeError, match="pandas DataFrame"):
        preprocess_fuel_data("not_a_df", 'col')
