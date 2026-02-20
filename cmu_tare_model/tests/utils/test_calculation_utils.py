"""Tests for calculation utilities (calculation_utils.py)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from cmu_tare_model.tests.conftest import FULL_EQUIPMENT_SPECS, FULL_UPGRADE_COLUMNS, FULL_FUEL_MAPPING


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def mock_constants(monkeypatch):
    """Mock constants with full production lifetimes."""
    monkeypatch.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', FULL_EQUIPMENT_SPECS)
    monkeypatch.setattr('cmu_tare_model.constants.VALID_CATEGORIES', list(FULL_EQUIPMENT_SPECS.keys()))
    monkeypatch.setattr('cmu_tare_model.constants.UPGRADE_COLUMNS', FULL_UPGRADE_COLUMNS)
    monkeypatch.setattr('cmu_tare_model.constants.FUEL_MAPPING', FULL_FUEL_MAPPING)
    monkeypatch.setattr('cmu_tare_model.constants.VERBOSE', False)
    monkeypatch.setattr('cmu_tare_model.constants.ALLOWED_TECHNOLOGIES', {
        'heating': ['Natural Gas Fuel Furnace', 'Electricity ASHP'],
        'cooling': ['Central AC'],
    })
    # Also patch the import locations where these modules bind the constants
    monkeypatch.setattr('cmu_tare_model.utils.calculation_utils.EQUIPMENT_SPECS', FULL_EQUIPMENT_SPECS)
    monkeypatch.setattr('cmu_tare_model.utils.calculation_utils.FUEL_MAPPING', FULL_FUEL_MAPPING)
    monkeypatch.setattr('cmu_tare_model.utils.calculation_utils.ALLOWED_TECHNOLOGIES', {
        'heating': ['Natural Gas Fuel Furnace', 'Electricity ASHP'],
        'cooling': ['Central AC'],
    })


@pytest.fixture
def sample_df():
    """DataFrame with consumption data for calculation_utils tests."""
    n = 5
    data = {
        'include_heating': [True, True, False, True, False],
        'include_waterHeating': [True, False, True, True, False],
        'include_clothesDrying': [True, True, True, False, False],
        'include_cooking': [True, True, True, True, False],
        'valid_fuel_heating': [True, True, False, True, False],
        'valid_tech_heating': [True, True, False, True, False],
        'base_heating_fuel': ['Natural Gas', 'Electricity', 'Natural Gas', 'Propane', 'Fuel Oil'],
        'base_waterHeating_fuel': ['Electricity', 'Natural Gas', 'Electricity', 'Natural Gas', 'Propane'],
        'base_clothesDrying_fuel': ['Electricity', 'Natural Gas', 'Propane', 'Electricity', 'Natural Gas'],
        'base_cooking_fuel': ['Natural Gas', 'Propane', 'Natural Gas', 'Propane', 'Natural Gas'],
        'heating_type': ['Natural Gas Fuel Furnace', 'Electricity ASHP', 'Natural Gas Fuel Furnace', 'Natural Gas Fuel Furnace', 'Natural Gas Fuel Furnace'],
    }

    # Fuel-specific consumption columns
    for fuel_key in FULL_FUEL_MAPPING.values():
        for cat in FULL_EQUIPMENT_SPECS:
            data[f'base_{fuel_key}_{cat}_consumption'] = np.random.uniform(500, 2000, n)

    # Baseline total and MP consumption
    for cat in FULL_EQUIPMENT_SPECS:
        data[f'baseline_{cat}_consumption'] = np.random.uniform(1000, 3000, n)
        for mp in [3, 4, 8]:
            data[f'mp{mp}_{cat}_consumption'] = np.random.uniform(400, 1500, n)

    return pd.DataFrame(data)


# =============================================================================
# validate_common_parameters
# =============================================================================

def test_validate_common_parameters_valid():
    """Accepts valid menu_mp and policy_scenario."""
    from cmu_tare_model.utils.calculation_utils import validate_common_parameters

    menu_mp, policy = validate_common_parameters(8, 'AEO2023 Reference Case')
    assert menu_mp == 8
    assert policy == 'AEO2023 Reference Case'


def test_validate_common_parameters_string_menu_mp():
    """Converts string menu_mp to integer."""
    from cmu_tare_model.utils.calculation_utils import validate_common_parameters

    menu_mp, _ = validate_common_parameters('0', 'AEO2023 Reference Case')
    assert menu_mp == 0
    assert isinstance(menu_mp, int)


def test_validate_common_parameters_invalid_menu_mp():
    """Raises ValueError for non-numeric menu_mp."""
    from cmu_tare_model.utils.calculation_utils import validate_common_parameters

    with pytest.raises(ValueError, match="Invalid menu_mp"):
        validate_common_parameters('abc', 'AEO2023 Reference Case')


@pytest.mark.parametrize("scenario", ['No Inflation Reduction Act', 'AEO2023 Reference Case'])
def test_validate_common_parameters_valid_scenarios(scenario):
    """Both valid policy scenarios are accepted."""
    from cmu_tare_model.utils.calculation_utils import validate_common_parameters

    _, result_scenario = validate_common_parameters(0, scenario)
    assert result_scenario == scenario


def test_validate_common_parameters_invalid_scenario():
    """Raises ValueError for invalid policy_scenario."""
    from cmu_tare_model.utils.calculation_utils import validate_common_parameters

    with pytest.raises(ValueError, match="Invalid policy_scenario"):
        validate_common_parameters(0, 'Invalid Scenario')


# =============================================================================
# get_all_possible_fuel_columns
# =============================================================================

def test_get_all_possible_fuel_columns_heating():
    """Heating has all four fuel type columns."""
    from cmu_tare_model.utils.calculation_utils import get_all_possible_fuel_columns

    result = get_all_possible_fuel_columns('heating')
    assert len(result) == 4
    assert 'base_electricity_heating_consumption' in result
    assert 'base_fuelOil_heating_consumption' in result


def test_get_all_possible_fuel_columns_clothes_drying():
    """Clothes drying excludes fuel oil."""
    from cmu_tare_model.utils.calculation_utils import get_all_possible_fuel_columns

    result = get_all_possible_fuel_columns('clothesDrying')
    assert all('fuelOil' not in col for col in result)
    assert len(result) == 3


def test_get_all_possible_fuel_columns_cooking():
    """Cooking excludes both electricity and fuel oil."""
    from cmu_tare_model.utils.calculation_utils import get_all_possible_fuel_columns

    result = get_all_possible_fuel_columns('cooking')
    assert all('electricity' not in col for col in result)
    assert all('fuelOil' not in col for col in result)
    assert len(result) == 2


def test_get_all_possible_fuel_columns_invalid_category():
    """Raises ValueError for invalid category."""
    from cmu_tare_model.utils.calculation_utils import get_all_possible_fuel_columns

    with pytest.raises(ValueError):
        get_all_possible_fuel_columns('invalid')


# =============================================================================
# get_post_retrofit_columns
# =============================================================================

def test_get_post_retrofit_columns():
    """Returns correct column name format."""
    from cmu_tare_model.utils.calculation_utils import get_post_retrofit_columns

    result = get_post_retrofit_columns('heating', menu_mp=8)
    assert result == ['mp8_heating_consumption']


def test_get_post_retrofit_columns_invalid_category():
    """Raises ValueError for invalid category."""
    from cmu_tare_model.utils.calculation_utils import get_post_retrofit_columns

    with pytest.raises(ValueError):
        get_post_retrofit_columns('invalid', menu_mp=8)


# =============================================================================
# apply_temporary_validation_and_mask
# =============================================================================

def test_apply_temporary_validation_and_mask_basic():
    """Joins new columns to df_copy with masking applied."""
    from cmu_tare_model.utils.calculation_utils import apply_temporary_validation_and_mask

    df_copy = pd.DataFrame({
        'include_heating': [True, False, True],
        'valid_fuel_heating': [True, False, True],
        'existing_col': [1, 2, 3],
    })
    df_new = pd.DataFrame({
        'new_col': [10.0, 20.0, 30.0],
    })
    all_cols = {
        'heating': ['new_col'],
        'waterHeating': [],
        'clothesDrying': [],
        'cooking': [],
    }
    result = apply_temporary_validation_and_mask(df_copy, df_new, all_cols, verbose=False)

    assert 'new_col' in result.columns
    assert 'existing_col' in result.columns
    assert result.loc[0, 'new_col'] == 10.0
    assert np.isnan(result.loc[1, 'new_col'])  # Masked — include_heating is False
    assert result.loc[2, 'new_col'] == 30.0


def test_apply_temporary_validation_and_mask_removes_temp_columns():
    """Temporary validation columns are not present in the output."""
    from cmu_tare_model.utils.calculation_utils import apply_temporary_validation_and_mask

    df_copy = pd.DataFrame({
        'include_heating': [True, False],
        'valid_fuel_heating': [True, False],
        'valid_tech_heating': [True, False],
    })
    df_new = pd.DataFrame({'result': [1.0, 2.0]})
    all_cols = {'heating': ['result'], 'waterHeating': [], 'clothesDrying': [], 'cooking': []}

    result = apply_temporary_validation_and_mask(df_copy, df_new, all_cols, verbose=False)

    # The original df_copy validation columns should survive, but they shouldn't be duplicated
    assert 'include_heating' in result.columns


def test_apply_temporary_validation_and_mask_overlapping_columns():
    """Overlapping columns in df_new replace those in df_copy."""
    from cmu_tare_model.utils.calculation_utils import apply_temporary_validation_and_mask

    df_copy = pd.DataFrame({
        'include_heating': [True, True],
        'overlap_col': [1.0, 2.0],
    })
    df_new = pd.DataFrame({'overlap_col': [10.0, 20.0]})
    all_cols = {'heating': ['overlap_col'], 'waterHeating': [], 'clothesDrying': [], 'cooking': []}

    result = apply_temporary_validation_and_mask(df_copy, df_new, all_cols, verbose=False)
    assert result.loc[0, 'overlap_col'] == 10.0


# =============================================================================
# identify_valid_homes
# =============================================================================

def test_identify_valid_homes_creates_include_flags(sample_df):
    """Creates include_{category} flags for all categories in EQUIPMENT_SPECS."""
    from cmu_tare_model.utils.calculation_utils import identify_valid_homes

    result = identify_valid_homes(sample_df, verbose=False)
    for cat in FULL_EQUIPMENT_SPECS:
        assert f'include_{cat}' in result.columns
    assert 'include_all' in result.columns


def test_identify_valid_homes_include_all_is_intersection(sample_df):
    """include_all is the intersection of all category include flags."""
    from cmu_tare_model.utils.calculation_utils import identify_valid_homes

    result = identify_valid_homes(sample_df, verbose=False)
    expected = pd.Series(True, index=result.index)
    for cat in FULL_EQUIPMENT_SPECS:
        expected = expected & result[f'include_{cat}']
    pd.testing.assert_series_equal(result['include_all'], expected, check_names=False)


# =============================================================================
# filter_valid_tech_homes
# =============================================================================

def test_filter_valid_tech_homes_basic():
    """Filters to homes with both valid data and known technology."""
    from cmu_tare_model.utils.calculation_utils import filter_valid_tech_homes

    df = pd.DataFrame({'a': [1, 2, 3, 4]})
    valid_mask = pd.Series([True, True, False, True], index=df.index)
    tech = np.array(['ASHP', 'unknown', 'ASHP', 'MSHP'])
    eff = np.array([10.0, 8.0, 12.0, 9.0])

    df_valid, indices, tech_out, eff_out = filter_valid_tech_homes(
        df, valid_mask, tech, eff, default_value='unknown'
    )
    # Home 0: valid + known tech → included
    # Home 1: valid + unknown tech → excluded
    # Home 2: invalid → excluded
    # Home 3: valid + known tech → included
    assert len(df_valid) == 2
    assert 0 in indices
    assert 3 in indices
    assert list(tech_out) == ['ASHP', 'MSHP']


def test_filter_valid_tech_homes_all_unknown():
    """Returns empty results when all homes have unknown technology."""
    from cmu_tare_model.utils.calculation_utils import filter_valid_tech_homes

    df = pd.DataFrame({'a': [1, 2]})
    valid_mask = pd.Series([True, True], index=df.index)
    tech = np.array(['unknown', 'unknown'])
    eff = np.array([10.0, 8.0])

    df_valid, indices, tech_out, eff_out = filter_valid_tech_homes(
        df, valid_mask, tech, eff, default_value='unknown'
    )
    assert len(df_valid) == 0
    assert len(tech_out) == 0
