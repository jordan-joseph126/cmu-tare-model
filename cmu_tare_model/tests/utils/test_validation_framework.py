"""Tests for the core validation framework (validation_framework.py)."""

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
    monkeypatch.setattr('cmu_tare_model.constants.UPGRADE_COLUMNS', FULL_UPGRADE_COLUMNS)
    monkeypatch.setattr('cmu_tare_model.constants.VERBOSE', False)


@pytest.fixture
def basic_df():
    """Minimal DataFrame for validation framework testing."""
    n = 8
    data = {
        'include_heating': [True, True, False, True, False, True, True, False],
        'include_waterHeating': [True, False, True, True, False, True, False, True],
        'include_clothesDrying': [True, True, True, False, False, True, True, False],
        'include_cooking': [True, True, True, True, False, False, True, True],
        'upgrade_hvac_heating_efficiency': ['ASHP', 'MSHP', None, 'ASHP', None, None, 'ASHP', None],
        'upgrade_water_heater_efficiency': ['Electric Heat Pump', None, 'Electric Heat Pump', None, None, 'Electric Heat Pump', None, None],
        'upgrade_clothes_dryer': ['Electric, Premium, Heat Pump, Ventless', None, None, None, None, 'Electric, Premium, Heat Pump, Ventless', None, None],
        'upgrade_cooking_range': [None, 'Electric', None, 'Electric', None, None, 'Electric', None],
    }
    return pd.DataFrame(data)


@pytest.fixture
def all_valid_df():
    """DataFrame where all homes are valid for all categories."""
    n = 5
    data = {'upgrade_hvac_heating_efficiency': ['ASHP'] * n}
    for cat in FULL_EQUIPMENT_SPECS:
        data[f'include_{cat}'] = [True] * n
    for col in FULL_UPGRADE_COLUMNS.values():
        if col not in data:
            data[col] = ['SomeTech'] * n
    return pd.DataFrame(data)


@pytest.fixture
def all_invalid_df():
    """DataFrame where all homes are invalid for all categories."""
    n = 5
    data = {}
    for cat in FULL_EQUIPMENT_SPECS:
        data[f'include_{cat}'] = [False] * n
    for col in FULL_UPGRADE_COLUMNS.values():
        data[col] = [None] * n
    return pd.DataFrame(data)


# =============================================================================
# STEP 1: initialize_validation_tracking
# =============================================================================

def test_initialize_validation_tracking_returns_correct_structure(basic_df):
    """Returns a 4-tuple of (DataFrame, Series, dict, list)."""
    from cmu_tare_model.utils.validation_framework import initialize_validation_tracking

    df_ref, valid_mask, all_cols, cat_cols = initialize_validation_tracking(
        basic_df, 'heating', menu_mp=0, verbose=False
    )
    assert isinstance(df_ref, pd.DataFrame)
    assert isinstance(valid_mask, pd.Series)
    assert isinstance(all_cols, dict)
    assert isinstance(cat_cols, list)
    assert set(all_cols.keys()) == set(FULL_EQUIPMENT_SPECS.keys())
    assert len(cat_cols) == 0


def test_initialize_validation_tracking_no_copy_by_default(basic_df):
    """Default copy=False returns the same object, not a copy."""
    from cmu_tare_model.utils.validation_framework import initialize_validation_tracking

    df_ref, _, _, _ = initialize_validation_tracking(
        basic_df, 'heating', menu_mp=0, verbose=False, copy=False
    )
    assert df_ref is basic_df


def test_initialize_validation_tracking_with_copy(basic_df):
    """copy=True returns a new DataFrame object."""
    from cmu_tare_model.utils.validation_framework import initialize_validation_tracking

    df_ref, _, _, _ = initialize_validation_tracking(
        basic_df, 'heating', menu_mp=0, verbose=False, copy=True
    )
    assert df_ref is not basic_df
    pd.testing.assert_frame_equal(df_ref, basic_df)


@pytest.mark.parametrize("category", ['heating', 'waterHeating', 'clothesDrying', 'cooking'])
def test_initialize_validation_tracking_all_categories(basic_df, category):
    """Mask initialization works for each equipment category."""
    from cmu_tare_model.utils.validation_framework import initialize_validation_tracking

    _, valid_mask, _, _ = initialize_validation_tracking(
        basic_df, category, menu_mp=0, verbose=False
    )
    expected = basic_df[f'include_{category}']
    pd.testing.assert_series_equal(valid_mask, expected, check_names=False)


def test_initialize_validation_tracking_baseline_uses_include_only(basic_df):
    """For baseline (menu_mp=0), the mask is just the include flag."""
    from cmu_tare_model.utils.validation_framework import initialize_validation_tracking

    _, valid_mask, _, _ = initialize_validation_tracking(
        basic_df, 'heating', menu_mp=0, verbose=False
    )
    expected = basic_df['include_heating']
    pd.testing.assert_series_equal(valid_mask, expected, check_names=False)


def test_initialize_validation_tracking_measure_package_combines_masks(basic_df):
    """For measure packages (menu_mp!=0), mask combines include flag AND retrofit status."""
    from cmu_tare_model.utils.validation_framework import initialize_validation_tracking

    _, valid_mask, _, _ = initialize_validation_tracking(
        basic_df, 'heating', menu_mp=8, verbose=False
    )
    include = basic_df['include_heating']
    retrofit = basic_df['upgrade_hvac_heating_efficiency'].notna()
    expected = include & retrofit
    pd.testing.assert_series_equal(valid_mask, expected, check_names=False)


# =============================================================================
# get_valid_calculation_mask
# =============================================================================

def test_get_valid_calculation_mask_missing_include_col():
    """Raises ValueError when inclusion flag column is missing."""
    from cmu_tare_model.utils.validation_framework import get_valid_calculation_mask

    df = pd.DataFrame({'some_col': [1, 2, 3]})
    with pytest.raises(ValueError, match="Inclusion flag"):
        get_valid_calculation_mask(df, 'heating', menu_mp=0, verbose=False)


def test_get_valid_calculation_mask_baseline_string():
    """menu_mp='baseline' is treated the same as menu_mp=0."""
    from cmu_tare_model.utils.validation_framework import get_valid_calculation_mask

    df = pd.DataFrame({
        'include_heating': [True, False, True],
        'upgrade_hvac_heating_efficiency': ['ASHP', None, 'ASHP'],
    })
    mask_0 = get_valid_calculation_mask(df, 'heating', menu_mp=0, verbose=False)
    mask_baseline = get_valid_calculation_mask(df, 'heating', menu_mp='baseline', verbose=False)
    pd.testing.assert_series_equal(mask_0, mask_baseline)


def test_get_valid_calculation_mask_mp_all_excluded_raises(basic_df):
    """Raises ValueError when all homes are excluded for a measure package."""
    from cmu_tare_model.utils.validation_framework import get_valid_calculation_mask

    # Make all homes invalid
    basic_df['include_heating'] = False
    basic_df['upgrade_hvac_heating_efficiency'] = None
    with pytest.raises(ValueError, match="All homes excluded"):
        get_valid_calculation_mask(basic_df, 'heating', menu_mp=8, verbose=False)


# =============================================================================
# get_valid_fuel_types
# =============================================================================

def test_get_valid_fuel_types_heating():
    """Heating includes all four fuel types."""
    from cmu_tare_model.utils.validation_framework import get_valid_fuel_types

    result = get_valid_fuel_types('heating')
    assert set(result) == {'Electricity', 'Natural Gas', 'Propane', 'Fuel Oil'}


def test_get_valid_fuel_types_water_heating():
    """Water heating includes all four fuel types."""
    from cmu_tare_model.utils.validation_framework import get_valid_fuel_types

    result = get_valid_fuel_types('waterHeating')
    assert set(result) == {'Electricity', 'Natural Gas', 'Propane', 'Fuel Oil'}


def test_get_valid_fuel_types_clothes_drying_excludes_fuel_oil():
    """Clothes drying excludes fuel oil."""
    from cmu_tare_model.utils.validation_framework import get_valid_fuel_types

    result = get_valid_fuel_types('clothesDrying')
    assert 'Fuel Oil' not in result
    assert 'Electricity' in result


def test_get_valid_fuel_types_cooking_excludes_electricity_and_fuel_oil():
    """Cooking excludes both electricity and fuel oil."""
    from cmu_tare_model.utils.validation_framework import get_valid_fuel_types

    result = get_valid_fuel_types('cooking')
    assert 'Electricity' not in result
    assert 'Fuel Oil' not in result
    assert set(result) == {'Natural Gas', 'Propane'}


def test_get_valid_fuel_types_invalid_category():
    """Raises ValueError for unknown category."""
    from cmu_tare_model.utils.validation_framework import get_valid_fuel_types

    with pytest.raises(ValueError, match="Invalid category"):
        get_valid_fuel_types('invalid_category')


# =============================================================================
# get_retrofit_homes_mask
# =============================================================================

def test_get_retrofit_homes_mask_baseline_all_true(basic_df):
    """For baseline (menu_mp=0), all homes are marked as retrofit."""
    from cmu_tare_model.utils.validation_framework import get_retrofit_homes_mask

    mask = get_retrofit_homes_mask(basic_df, 'heating', menu_mp=0, verbose=False)
    assert mask.all()


def test_get_retrofit_homes_mask_mp_uses_upgrade_column(basic_df):
    """For measure packages, mask is based on upgrade column being non-null."""
    from cmu_tare_model.utils.validation_framework import get_retrofit_homes_mask

    mask = get_retrofit_homes_mask(basic_df, 'heating', menu_mp=8, verbose=False)
    expected = basic_df['upgrade_hvac_heating_efficiency'].notna()
    pd.testing.assert_series_equal(mask, expected, check_names=False)


# =============================================================================
# STEP 2: create_retrofit_only_series
# =============================================================================

def test_create_retrofit_only_series_with_mask():
    """Valid homes get 0.0, invalid homes get NaN."""
    from cmu_tare_model.utils.validation_framework import create_retrofit_only_series

    df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
    mask = pd.Series([True, False, True, False, True], index=df.index)

    result = create_retrofit_only_series(df, retrofit_mask=mask)
    assert result.loc[0] == 0.0
    assert np.isnan(result.loc[1])
    assert result.loc[2] == 0.0
    assert np.isnan(result.loc[3])
    assert result.loc[4] == 0.0


def test_create_retrofit_only_series_all_valid():
    """When all homes are valid, all get 0.0."""
    from cmu_tare_model.utils.validation_framework import create_retrofit_only_series

    df = pd.DataFrame({'a': range(5)})
    mask = pd.Series([True] * 5, index=df.index)
    result = create_retrofit_only_series(df, retrofit_mask=mask)
    assert (result == 0.0).all()


def test_create_retrofit_only_series_all_invalid():
    """When all homes are invalid, all get NaN."""
    from cmu_tare_model.utils.validation_framework import create_retrofit_only_series

    df = pd.DataFrame({'a': range(5)})
    mask = pd.Series([False] * 5, index=df.index)
    result = create_retrofit_only_series(df, retrofit_mask=mask)
    assert result.isna().all()


def test_create_retrofit_only_series_no_mask_raises():
    """Raises ValueError when neither mask nor category/menu_mp provided."""
    from cmu_tare_model.utils.validation_framework import create_retrofit_only_series

    df = pd.DataFrame({'a': range(5)})
    with pytest.raises(ValueError):
        create_retrofit_only_series(df)


def test_create_retrofit_only_series_with_category_and_mp(basic_df):
    """Can derive mask from category and menu_mp arguments."""
    from cmu_tare_model.utils.validation_framework import create_retrofit_only_series

    result = create_retrofit_only_series(basic_df, category='heating', menu_mp=0)
    assert len(result) == len(basic_df)
    assert result.dtype == float


# =============================================================================
# STEP 5: apply_final_masking
# =============================================================================

def test_apply_final_masking_masks_invalid_homes():
    """Invalid homes get NaN in all tracked columns."""
    from cmu_tare_model.utils.validation_framework import apply_final_masking

    df = pd.DataFrame({
        'include_heating': [True, False, True, False],
        'include_waterHeating': [True, True, False, False],
        'heating_result': [1.0, 2.0, 3.0, 4.0],
        'waterHeating_result': [10.0, 20.0, 30.0, 40.0],
    })
    all_cols = {
        'heating': ['heating_result'],
        'waterHeating': ['waterHeating_result'],
        'clothesDrying': [],
        'cooking': [],
    }
    result = apply_final_masking(df, all_cols, verbose=False)

    # Home 1 (idx=1): invalid for heating → heating_result should be NaN
    assert np.isnan(result.loc[1, 'heating_result'])
    # Home 2 (idx=2): invalid for waterHeating → waterHeating_result should be NaN
    assert np.isnan(result.loc[2, 'waterHeating_result'])
    # Home 0: valid for both → values preserved
    assert result.loc[0, 'heating_result'] == 1.0
    assert result.loc[0, 'waterHeating_result'] == 10.0


def test_apply_final_masking_nonexistent_columns_ignored():
    """Columns listed in all_columns_to_mask but not in df are silently ignored."""
    from cmu_tare_model.utils.validation_framework import apply_final_masking

    df = pd.DataFrame({
        'include_heating': [True, False],
        'heating_result': [1.0, 2.0],
    })
    all_cols = {
        'heating': ['heating_result', 'nonexistent_col'],
        'waterHeating': [],
        'clothesDrying': [],
        'cooking': [],
    }
    result = apply_final_masking(df, all_cols, verbose=False)
    assert 'nonexistent_col' not in result.columns


def test_apply_final_masking_empty_tracking():
    """No columns to mask — df returned unchanged."""
    from cmu_tare_model.utils.validation_framework import apply_final_masking

    df = pd.DataFrame({
        'include_heating': [True, False],
        'some_col': [1.0, 2.0],
    })
    all_cols = {cat: [] for cat in FULL_EQUIPMENT_SPECS}
    result = apply_final_masking(df, all_cols, verbose=False)
    assert result.loc[1, 'some_col'] == 2.0  # No masking applied


# =============================================================================
# mask_category_specific_data
# =============================================================================

def test_mask_category_specific_data_basic():
    """Masks specified columns where include flag is False."""
    from cmu_tare_model.utils.validation_framework import mask_category_specific_data

    df = pd.DataFrame({
        'include_heating': [True, False, True],
        'cost': [100.0, 200.0, 300.0],
        'savings': [10.0, 20.0, 30.0],
    })
    result = mask_category_specific_data(df, ['cost', 'savings'], 'heating', verbose=False, inplace=False)
    assert result.loc[0, 'cost'] == 100.0
    assert np.isnan(result.loc[1, 'cost'])
    assert np.isnan(result.loc[1, 'savings'])
    assert result.loc[2, 'savings'] == 30.0


def test_mask_category_specific_data_missing_include_raises():
    """Raises ValueError if the inclusion flag column is missing."""
    from cmu_tare_model.utils.validation_framework import mask_category_specific_data

    df = pd.DataFrame({'cost': [100.0, 200.0]})
    with pytest.raises(ValueError, match="Inclusion flag"):
        mask_category_specific_data(df, ['cost'], 'heating', verbose=False)


# =============================================================================
# apply_new_columns_to_dataframe
# =============================================================================

def test_apply_new_columns_to_dataframe_tracking():
    """New columns are properly tracked in all_columns_to_mask."""
    from cmu_tare_model.utils.validation_framework import apply_new_columns_to_dataframe

    df_orig = pd.DataFrame({'include_heating': [True, False], 'existing': [1, 2]})
    df_new = pd.DataFrame({'new_col_1': [10, 20], 'new_col_2': [30, 40]})
    cat_cols = []
    all_cols = {cat: [] for cat in FULL_EQUIPMENT_SPECS}

    result_df, result_all_cols = apply_new_columns_to_dataframe(
        df_orig, df_new, 'heating', cat_cols, all_cols
    )
    assert 'new_col_1' in result_df.columns
    assert 'new_col_2' in result_df.columns
    assert 'new_col_1' in result_all_cols['heating']
    assert 'new_col_2' in result_all_cols['heating']


def test_apply_new_columns_handles_overlapping_columns():
    """Overlapping columns in df_new replace those in df_original."""
    from cmu_tare_model.utils.validation_framework import apply_new_columns_to_dataframe

    df_orig = pd.DataFrame({'overlap_col': [1, 2], 'keep_col': [3, 4]})
    df_new = pd.DataFrame({'overlap_col': [10, 20], 'new_col': [30, 40]})
    cat_cols = []
    all_cols = {cat: [] for cat in FULL_EQUIPMENT_SPECS}

    result_df, _ = apply_new_columns_to_dataframe(
        df_orig, df_new, 'heating', cat_cols, all_cols
    )
    assert result_df.loc[0, 'overlap_col'] == 10
    assert result_df.loc[0, 'keep_col'] == 3


# =============================================================================
# replace_small_values_with_nan
# =============================================================================

def test_replace_small_values_with_nan_series():
    """Values at or below threshold become NaN; values above are preserved."""
    from cmu_tare_model.utils.validation_framework import replace_small_values_with_nan

    s = pd.Series([1e-11, -1e-11, 1e-10, -1e-10, 1e-9, -1e-9, 0.0, 5.0])
    result = replace_small_values_with_nan(s, threshold=1e-10)

    # abs <= 1e-10 should be NaN
    assert np.isnan(result.iloc[0])  # 1e-11
    assert np.isnan(result.iloc[1])  # -1e-11
    assert np.isnan(result.iloc[2])  # 1e-10 (at threshold, not > threshold)
    assert np.isnan(result.iloc[3])  # -1e-10
    # abs > 1e-10 should be preserved
    assert result.iloc[4] == 1e-9
    assert result.iloc[5] == -1e-9
    assert np.isnan(result.iloc[6])  # 0.0
    assert result.iloc[7] == 5.0


def test_replace_small_values_with_nan_dataframe():
    """Works on DataFrames column by column."""
    from cmu_tare_model.utils.validation_framework import replace_small_values_with_nan

    df = pd.DataFrame({'a': [1e-11, 5.0], 'b': [0.0, -1e-9]})
    result = replace_small_values_with_nan(df, threshold=1e-10)
    assert np.isnan(result.loc[0, 'a'])
    assert result.loc[1, 'a'] == 5.0
    assert np.isnan(result.loc[0, 'b'])
    assert result.loc[1, 'b'] == -1e-9


def test_replace_small_values_with_nan_dict():
    """Works on dictionaries of Series."""
    from cmu_tare_model.utils.validation_framework import replace_small_values_with_nan

    d = {'x': pd.Series([1e-11, 5.0]), 'y': pd.Series([0.0, -1e-9])}
    result = replace_small_values_with_nan(d, threshold=1e-10)
    assert isinstance(result, dict)
    assert np.isnan(result['x'].iloc[0])
    assert result['y'].iloc[1] == -1e-9


def test_replace_small_values_with_nan_invalid_type():
    """Raises TypeError for unsupported input types."""
    from cmu_tare_model.utils.validation_framework import replace_small_values_with_nan

    with pytest.raises(TypeError):
        replace_small_values_with_nan([1, 2, 3])


# =============================================================================
# calculate_avoided_values
# =============================================================================

def test_calculate_avoided_values_basic():
    """Avoided = baseline - measure for retrofit homes, NaN for others."""
    from cmu_tare_model.utils.validation_framework import calculate_avoided_values

    baseline = pd.Series([100.0, 200.0, 300.0, 400.0])
    measure = pd.Series([80.0, 150.0, 250.0, 350.0])
    mask = pd.Series([True, True, False, True])

    result = calculate_avoided_values(baseline, measure, retrofit_mask=mask)
    assert result.iloc[0] == pytest.approx(20.0)
    assert result.iloc[1] == pytest.approx(50.0)
    assert np.isnan(result.iloc[2])
    assert result.iloc[3] == pytest.approx(50.0)


def test_calculate_avoided_values_no_mask():
    """Without mask (baseline scenario), calculates for all homes."""
    from cmu_tare_model.utils.validation_framework import calculate_avoided_values

    baseline = pd.Series([100.0, 200.0])
    measure = pd.Series([80.0, 150.0])

    result = calculate_avoided_values(baseline, measure, retrofit_mask=None)
    assert result.iloc[0] == pytest.approx(20.0)
    assert result.iloc[1] == pytest.approx(50.0)


def test_calculate_avoided_values_nan_propagation():
    """NaN in baseline or measure propagates to result."""
    from cmu_tare_model.utils.validation_framework import calculate_avoided_values

    baseline = pd.Series([100.0, np.nan, 300.0])
    measure = pd.Series([80.0, 150.0, np.nan])
    mask = pd.Series([True, True, True])

    result = calculate_avoided_values(baseline, measure, retrofit_mask=mask)
    assert result.iloc[0] == pytest.approx(20.0)
    assert np.isnan(result.iloc[1])
    assert np.isnan(result.iloc[2])


# =============================================================================
# apply_validation_mask_vectorized
# =============================================================================

def test_apply_validation_mask_baseline_returns_copy():
    """For menu_mp=0, returns a copy without masking."""
    from cmu_tare_model.utils.validation_framework import apply_validation_mask_vectorized

    values = pd.Series([1.0, 2.0, 3.0])
    mask = pd.Series([True, False, True])
    result = apply_validation_mask_vectorized(values, mask, menu_mp=0)
    pd.testing.assert_series_equal(result, values)
    assert result is not values


def test_apply_validation_mask_mp_masks_invalid():
    """For menu_mp!=0, invalid homes get NaN."""
    from cmu_tare_model.utils.validation_framework import apply_validation_mask_vectorized

    values = pd.Series([1.0, 2.0, 3.0])
    mask = pd.Series([True, False, True])
    result = apply_validation_mask_vectorized(values, mask, menu_mp=8)
    assert result.iloc[0] == 1.0
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 3.0
