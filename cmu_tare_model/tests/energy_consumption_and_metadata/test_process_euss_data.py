"""Tests for cmu_tare_model.energy_consumption_and_metadata.process_euss_data module.

Verifies pure utility functions: extract_city_name, map_metro_status,
standardize_fuel_name, and preprocess_fuel_data; and the savings check,
check_savings_against_resstock.
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


# -- check_savings_against_resstock -------------------------------------------

@pytest.fixture
def savings_check_inputs():
    """Three retrofitted homes with ResStock's 2022.1.1 column names. Homes 1
    and 2 heated with natural gas; home 3 heated with electricity and uses no
    natural gas. No home uses propane or fuel oil."""
    bldg_ids = pd.Index([1, 2, 3], name='bldg_id')
    energy = 'energy_consumption.kwh'
    df_mp = pd.DataFrame({
        # Retrofit use, then ResStock's savings (baseline minus retrofit).
        f'out.electricity.heating.{energy}': [6000.0, 4000.0, 5000.0],
        f'out.electricity.heating.{energy}.savings': [-6000.0, -4000.0, 7000.0],
        f'out.electricity.cooling.{energy}': [2500.0, 1500.0, 2000.0],
        f'out.electricity.cooling.{energy}.savings': [500.0, 300.0, 400.0],
        f'out.natural_gas.heating.{energy}': [0.0, 0.0, 0.0],
        f'out.natural_gas.heating.{energy}.savings': [20000.0, 15000.0, 0.0],
        # Side effects ResStock reports outside heating and cooling.
        f'out.electricity.hot_water.{energy}.savings': [0.0, 50.0, -30.0],
        f'out.electricity.refrigerator.{energy}.savings': [10.0, 0.0, 0.0],
        f'out.natural_gas.hot_water.{energy}.savings': [0.0, 0.0, 0.0],
        # Whole home: heating + cooling savings plus the side effects.
        f'out.electricity.total.{energy}': [18000.0, 12000.0, 9000.0],
        f'out.electricity.total.{energy}.savings': [-5490.0, -3650.0, 7370.0],
        f'out.natural_gas.total.{energy}': [5000.0, 4000.0, 0.0],
        f'out.natural_gas.total.{energy}.savings': [20000.0, 15000.0, 0.0],
        f'out.propane.total.{energy}': [0.0] * 3,
        f'out.propane.total.{energy}.savings': [0.0] * 3,
        f'out.fuel_oil.total.{energy}': [0.0] * 3,
        f'out.fuel_oil.total.{energy}.savings': [0.0] * 3,
    }, index=bldg_ids)
    tare_savings_by_fuel = {
        'electricity': pd.Series([-5500.0, -3700.0, 7400.0], index=bldg_ids),
        'naturalGas': pd.Series([20000.0, 15000.0, 0.0], index=bldg_ids),
        'propane': pd.Series(0.0, index=bldg_ids),
        'fuelOil': pd.Series(0.0, index=bldg_ids),
    }
    return tare_savings_by_fuel, df_mp, bldg_ids


def test_savings_check_passes(savings_check_inputs):
    """Matching savings pass; hot water and refrigerator changes are set aside."""
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import (
        check_savings_against_resstock)
    tare_savings_by_fuel, df_mp, bldg_ids = savings_check_inputs
    df_check = check_savings_against_resstock(
        tare_savings_by_fuel, df_mp, bldg_ids, menu_mp=3, release='2022.1.1')
    assert list(df_check['status']) == ['[OK]'] * 4
    assert df_check['rdu_failed'].sum() == 0


def test_savings_check_left_out_part_raises(savings_check_inputs):
    """Energy the parts do not count shows up against the whole home."""
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import (
        check_savings_against_resstock)
    tare_savings_by_fuel, df_mp, bldg_ids = savings_check_inputs
    # Home 1 used 800 kWh more electricity than its heating and cooling parts
    # explain, as if backup heat had been left out.
    df_mp.loc[1, 'out.electricity.total.energy_consumption.kwh.savings'] -= 800.0
    with pytest.raises(ValueError, match='electricity'):
        check_savings_against_resstock(
            tare_savings_by_fuel, df_mp, bldg_ids, menu_mp=3, release='2022.1.1')


def test_savings_check_part_mismatch_raises(savings_check_inputs):
    """TARE's savings must equal ResStock's part savings within 1 kWh."""
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import (
        check_savings_against_resstock)
    tare_savings_by_fuel, df_mp, bldg_ids = savings_check_inputs
    # 5 kWh is well inside the whole-home limit, so only the part test fails.
    tare_savings_by_fuel['naturalGas'].loc[2] += 5.0
    with pytest.raises(ValueError, match='naturalGas'):
        check_savings_against_resstock(
            tare_savings_by_fuel, df_mp, bldg_ids, menu_mp=3, release='2022.1.1')


def test_savings_check_home_without_the_fuel(savings_check_inputs):
    """A home that does not use a fuel passes at zero and fails on anything else."""
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import (
        check_savings_against_resstock)
    tare_savings_by_fuel, df_mp, bldg_ids = savings_check_inputs
    # Home 3 uses no natural gas, and no home uses propane or fuel oil.
    check_savings_against_resstock(
        tare_savings_by_fuel, df_mp, bldg_ids, menu_mp=3, release='2022.1.1')
    # 0.5 kWh is inside the part test's 1 kWh limit, but 1% of zero use is zero.
    tare_savings_by_fuel['naturalGas'].loc[3] = 0.5
    with pytest.raises(ValueError, match='naturalGas'):
        check_savings_against_resstock(
            tare_savings_by_fuel, df_mp, bldg_ids, menu_mp=3, release='2022.1.1')


def test_savings_check_blank_raises(savings_check_inputs):
    """A sample home with a blank savings value fails; it is not counted as zero."""
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import (
        check_savings_against_resstock)
    tare_savings_by_fuel, df_mp, bldg_ids = savings_check_inputs
    df_mp.loc[2, 'out.electricity.cooling.energy_consumption.kwh.savings'] = np.nan
    with pytest.raises(ValueError, match='electricity'):
        check_savings_against_resstock(
            tare_savings_by_fuel, df_mp, bldg_ids, menu_mp=3, release='2022.1.1')


def test_savings_check_quiet_pass_prints_one_line(savings_check_inputs, capsys):
    """A passing check with verbose off prints one [OK] line and no table."""
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import (
        check_savings_against_resstock)
    tare_savings_by_fuel, df_mp, bldg_ids = savings_check_inputs
    capsys.readouterr()  # drop anything printed while importing
    check_savings_against_resstock(
        tare_savings_by_fuel, df_mp, bldg_ids, menu_mp=3, release='2022.1.1',
        verbose=False)
    printed_lines = capsys.readouterr().out.strip().splitlines()
    assert len(printed_lines) == 1
    assert printed_lines[0].startswith('[OK] MP3')
    assert '3 rdu, 4 fuels' in printed_lines[0]


def test_savings_check_quiet_failure_prints_table(savings_check_inputs, capsys):
    """A failing check prints the full table even with verbose off."""
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import (
        check_savings_against_resstock)
    tare_savings_by_fuel, df_mp, bldg_ids = savings_check_inputs
    # Same 5 kWh mismatch as test_savings_check_part_mismatch_raises.
    tare_savings_by_fuel['naturalGas'].loc[2] += 5.0
    with pytest.raises(ValueError, match='naturalGas'):
        check_savings_against_resstock(
            tare_savings_by_fuel, df_mp, bldg_ids, menu_mp=3, release='2022.1.1',
            verbose=False)
    printed = capsys.readouterr().out
    assert 'rdu_failed' in printed
    assert '[WARNING]' in printed


# -- select_resstock_2025_1_columns (column-limited 2025.1 read) ---------------

def _made_up_2025_1_file_columns():
    """A small made-up 2025.1 column list: names the pipeline reads, and names
    it must leave out (per-square-foot, pre-weighted, unrelated end uses)."""
    from cmu_tare_model.utils.resstock_schema import RESSTOCK_COLUMN_MAP
    mapped = list(RESSTOCK_COLUMN_MAP['2025.1'].values())
    extra = [
        'completed_status',
        'in.hvac_has_ducts',
        'out.electricity.heating_hp_bkup_fa.energy_consumption..kwh',
        'out.electricity.heating_hp_bkup_fa.energy_savings..kwh',
        'out.electricity.heating.energy_savings..kwh',
        'out.natural_gas.heating.energy_savings..kwh',
        'out.electricity.cooling.energy_savings..kwh',
        'out.electricity.hot_water.energy_savings..kwh',
        'out.electricity.hot_water_solar_th.energy_savings..kwh',
        'out.electricity.refrigerator.energy_savings..kwh',
        'out.electricity.total.energy_savings..kwh',
        'out.site_energy.total.energy_savings..kwh',
        # Must be left out:
        'out.electricity.heating.energy_consumption_intensity..kwh_per_ft2',
        'out.electricity.total.energy_savings_intensity..kwh_per_ft2',
        'calc.weighted.electricity.total.energy_savings..tbtu',
        'out.electricity.lighting_interior.energy_consumption..kwh',
        'out.electricity.lighting_interior.energy_savings..kwh',
        'out.electricity.pool_heater.energy_savings..kwh',
    ]
    return mapped + extra


def test_select_2025_1_columns_keeps_what_the_pipeline_reads():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import (
        select_resstock_2025_1_columns,
    )
    file_columns = _made_up_2025_1_file_columns()
    selected = select_resstock_2025_1_columns(file_columns)
    for column in (
            'bldg_id', 'weight', 'applicability', 'in.hvac_has_ducts',
            'out.electricity.heating_hp_bkup_fa.energy_consumption..kwh',
            'out.electricity.heating_hp_bkup_fa.energy_savings..kwh',
            'out.natural_gas.heating.energy_savings..kwh',
            'out.electricity.hot_water.energy_savings..kwh',
            'out.electricity.hot_water_solar_th.energy_savings..kwh',
            'out.electricity.refrigerator.energy_savings..kwh',
            'out.site_energy.total.energy_savings..kwh'):
        assert column in selected, column


def test_select_2025_1_columns_leaves_out_weighted_and_unrelated():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import (
        select_resstock_2025_1_columns,
    )
    selected = select_resstock_2025_1_columns(_made_up_2025_1_file_columns())
    for column in (
            'completed_status',
            'out.electricity.heating.energy_consumption_intensity..kwh_per_ft2',
            'out.electricity.total.energy_savings_intensity..kwh_per_ft2',
            'calc.weighted.electricity.total.energy_savings..tbtu',
            'out.electricity.lighting_interior.energy_consumption..kwh',
            'out.electricity.lighting_interior.energy_savings..kwh',
            'out.electricity.pool_heater.energy_savings..kwh'):
        assert column not in selected, column


def test_select_2025_1_columns_keeps_file_order():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import (
        select_resstock_2025_1_columns,
    )
    file_columns = _made_up_2025_1_file_columns()[::-1]
    selected = select_resstock_2025_1_columns(file_columns)
    assert selected == [c for c in file_columns if c in set(selected)]


def test_select_2025_1_columns_needs_bldg_id_and_weight():
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import (
        select_resstock_2025_1_columns,
    )
    file_columns = [c for c in _made_up_2025_1_file_columns() if c != 'weight']
    with pytest.raises(ValueError, match='weight'):
        select_resstock_2025_1_columns(file_columns)
