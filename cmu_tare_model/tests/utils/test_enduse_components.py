"""Tests for finding and reading heating/cooling parts from ResStock columns
(find_enduse_columns, get_consumption_component_columns, and
get_resstock_savings_column, calculation_utils.py)."""

import pytest

from cmu_tare_model.utils.calculation_utils import (
    find_enduse_columns,
    get_consumption_component_columns,
    get_resstock_savings_column,
)

# Real column names from each release's heating/cooling end uses, plus
# look-alikes that must never match.
COLUMNS_2022 = [
    'out.electricity.heating.energy_consumption.kwh',
    'out.electricity.heating_fans_pumps.energy_consumption.kwh',
    'out.electricity.heating_hp_bkup.energy_consumption.kwh',
    'out.natural_gas.heating.energy_consumption.kwh',
    'out.natural_gas.heating_hp_bkup.energy_consumption.kwh',
    'out.electricity.cooling.energy_consumption.kwh',
    'out.electricity.cooling_fans_pumps.energy_consumption.kwh',
    'out.electricity.heating.energy_consumption.kwh.savings',
    'out.electricity.pool_heater.energy_consumption.kwh',
    'out.natural_gas.hot_water.energy_consumption.kwh',
]
COLUMNS_2025 = [
    'out.electricity.heating.energy_consumption..kwh',
    'out.electricity.heating_hp_bkup_fa.energy_consumption..kwh',
    'out.natural_gas.heating_hp_bkup.energy_consumption..kwh',
]


def test_finds_every_heating_part_2022():
    found = find_enduse_columns(COLUMNS_2022, 'heating')
    assert sorted((fuel, part) for fuel, part, _ in found) == [
        ('electricity', 'fansPumps'), ('electricity', 'hpBackup'),
        ('electricity', 'primary_system'),
        ('naturalGas', 'hpBackup'), ('naturalGas', 'primary_system'),
    ]


def test_cooling_and_look_alikes_kept_apart():
    found = find_enduse_columns(COLUMNS_2022, 'cooling')
    assert sorted(part for _, part, _ in found) == ['fansPumps', 'primary_system']
    all_matched = {col for category in ('heating', 'cooling')
                   for _, _, col in find_enduse_columns(COLUMNS_2022, category)}
    assert not any(col.endswith('.savings') or 'pool_heater' in col
                   or 'hot_water' in col for col in all_matched)


def test_2025_naming_and_backup_fans():
    found = find_enduse_columns(COLUMNS_2025, 'heating')
    assert ('electricity', 'hpBackupFans') in [(fuel, part) for fuel, part, _ in found]
    assert ('naturalGas', 'hpBackup') in [(fuel, part) for fuel, part, _ in found]


def test_unknown_part_or_fuel_raises():
    with pytest.raises(ValueError, match='no TARE name'):
        find_enduse_columns(COLUMNS_2022 + [
            'out.electricity.heating_new_part.energy_consumption.kwh'], 'heating')
    with pytest.raises(ValueError, match='not one TARE prices'):
        find_enduse_columns(COLUMNS_2022 + [
            'out.wood.heating.energy_consumption.kwh'], 'heating')


def test_missing_primary_system_raises():
    with pytest.raises(ValueError, match='primary-system'):
        find_enduse_columns(
            ['out.electricity.heating_fans_pumps.energy_consumption.kwh'], 'heating')


def test_reader_lists_only_parts_in_the_frame_in_fixed_order():
    frame_columns = [
        'mp3_heating_consumption',
        'mp3_naturalGas_heating_hpBackup_consumption',
        'mp3_electricity_heating_hpBackup_consumption',
        'mp3_electricity_heating_fansPumps_consumption',
        'mp3_electricity_cooling_fansPumps_consumption',
    ]
    pairs = get_consumption_component_columns('heating', 3, columns=frame_columns)
    assert pairs == [
        ('electricity', 'mp3_heating_consumption'),
        ('electricity', 'mp3_electricity_heating_fansPumps_consumption'),
        ('electricity', 'mp3_electricity_heating_hpBackup_consumption'),
        ('naturalGas', 'mp3_naturalGas_heating_hpBackup_consumption'),
    ]


def test_savings_column_name_2022():
    """2022.1.1 adds '.savings' to the energy column name."""
    assert get_resstock_savings_column(
        'out.electricity.heating.energy_consumption.kwh'
    ) == 'out.electricity.heating.energy_consumption.kwh.savings'


def test_savings_column_name_2025():
    """2025.1 renames 'energy_consumption' to 'energy_savings'."""
    assert get_resstock_savings_column(
        'out.natural_gas.heating_hp_bkup.energy_consumption..kwh'
    ) == 'out.natural_gas.heating_hp_bkup.energy_savings..kwh'


def test_savings_column_bad_name_raises():
    with pytest.raises(ValueError, match='not a ResStock energy consumption'):
        get_resstock_savings_column('out.load.heating.energy_delivered.kbtu')
