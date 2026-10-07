"""Tests for cmu_tare_model.utils.column_names module.

Verifies that all column name builder functions produce correctly formatted
strings matching the TARE model's naming conventions for REMDB v3/v4 scenarios.
"""

import pytest
from cmu_tare_model.utils.column_names import (
    create_fuel_cost_col,
    create_cost_col,
    create_rebate_col,
    create_capital_col,
    create_npv_col,
    create_enclosure_cost_col,
    create_weatherization_rebate_col,
    create_installation_premium_col,
    create_combined_heating_cooling_col,
    create_climate_npv_col,
    create_lifetime_damages_col,
    create_avoided_damages_col,
    create_adoption_col,
    create_total_npv_col,
)


# ── Fuel cost column ─────────────────────────────────────────────────────────

def test_fuel_cost_col_baseline():
    result = create_fuel_cost_col('baseline_', 'year1', 'heating')
    assert result == 'baseline_year1_heating_fuel_cost'


def test_fuel_cost_col_measure_package():
    result = create_fuel_cost_col('iraRef_mp3_', 'year5', 'waterHeating')
    assert result == 'iraRef_mp3_year5_waterHeating_fuel_cost'


# ── Installed cost column ────────────────────────────────────────────────────

def test_cost_col_upgrade_v3():
    result = create_cost_col(3, 'heating', 'upgrade', 'v3')
    assert result == 'mp3_heating_upgrade_installed_cost_v3'


def test_cost_col_replacement_v4mid():
    result = create_cost_col(8, 'waterHeating', 'replacement', 'v4MID')
    assert result == 'mp8_waterHeating_replacement_installed_cost_v4MID'


def test_cost_col_all_cost_scenarios():
    for cs in ['v3', 'v4LOW', 'v4MID', 'v4HIGH']:
        result = create_cost_col(3, 'cooking', 'upgrade', cs)
        assert result.endswith(f'_{cs}')


# ── Rebate column ────────────────────────────────────────────────────────────

def test_rebate_col():
    result = create_rebate_col(3, 'heating', 'v3')
    assert result == 'mp3_heating_rebate_amount_v3'


def test_rebate_col_v4():
    result = create_rebate_col(8, 'waterHeating', 'v4MID')
    assert result == 'mp8_waterHeating_rebate_amount_v4MID'


# ── Capital cost column ──────────────────────────────────────────────────────

def test_capital_col_total():
    result = create_capital_col('iraRef_mp3_', 'heating', net=False, cost_scenario='v3')
    assert result == 'iraRef_mp3_heating_total_capital_cost_v3'


def test_capital_col_net():
    result = create_capital_col('iraRef_mp3_', 'heating', net=True, cost_scenario='v4MID')
    assert result == 'iraRef_mp3_heating_net_capital_cost_v4MID'


# ── NPV column ───────────────────────────────────────────────────────────────

def test_npv_col_v3():
    result = create_npv_col('iraRef_mp3_', 'heating', 'lessWTP', cost_scenario='v3', method_suffix='_fixed_low')
    assert result == 'iraRef_mp3_heating_private_npv_lessWTP_v3_fixed_low'


def test_npv_col_v4mid():
    result = create_npv_col('iraRef_mp3_', 'heating', 'moreWTP', cost_scenario='v4MID', method_suffix='_fixed_high')
    assert result == 'iraRef_mp3_heating_private_npv_moreWTP_v4MID_fixed_high'


# ── Enclosure cost column ────────────────────────────────────────────────────

def test_enclosure_cost_col_v3():
    result = create_enclosure_cost_col(9, 'v3')
    assert result == 'mp9_enclosure_upgrade_installed_cost_v3'


def test_enclosure_cost_col_v4():
    result = create_enclosure_cost_col(10, 'v4MID')
    assert result == 'mp10_enclosure_upgrade_installed_cost_v4MID'


# ── Weatherization rebate column ─────────────────────────────────────────────

def test_weatherization_rebate_col_v3():
    result = create_weatherization_rebate_col('v3')
    assert result == 'weatherization_rebate_amount_v3'


def test_weatherization_rebate_col_v4():
    result = create_weatherization_rebate_col('v4MID')
    assert result == 'weatherization_rebate_amount_v4MID'


# ── Installation premium column ──────────────────────────────────────────────

def test_installation_premium_col():
    result = create_installation_premium_col(3, 'heating')
    assert result == 'mp3_heating_installation_premium'


# ── Combined heating/cooling column ──────────────────────────────────────────

def test_combined_heating_cooling_col():
    result = create_combined_heating_cooling_col(3, 'replacement_installed_cost', 'v4MID')
    assert result == 'mp3_heating_and_cooling_replacement_installed_cost_v4MID'


# ── Climate NPV column ───────────────────────────────────────────────────────

def test_climate_npv_col():
    result = create_climate_npv_col('iraRef_mp3_', 'heating', 'central')
    assert result == 'iraRef_mp3_heating_climate_npv_central'


def test_climate_npv_col_all_scc():
    for scc in ['lower', 'central', 'upper']:
        result = create_climate_npv_col('baseline_', 'cooking', scc)
        assert scc in result


# ── Lifetime damages column ──────────────────────────────────────────────────

def test_lifetime_damages_col_climate():
    result = create_lifetime_damages_col('baseline_', 'heating', 'climate', 'lrmer', 'central')
    assert result == 'baseline_heating_lifetime_damages_climate_lrmer_central'


# ── Avoided damages column ───────────────────────────────────────────────────

def test_avoided_damages_col_climate():
    result = create_avoided_damages_col('iraRef_mp3_', 'heating', 'climate', 'lrmer', 'central')
    assert result == 'iraRef_mp3_heating_avoided_damages_climate_lrmer_central'


# ── Adoption column (economic-adopter, one per NPV case) ──────────────────────

def test_adoption_col_heatingSavings_coolingLCC_sub():
    result = create_adoption_col(
        'ref2025_mp3_', 'heatingSavings_coolingLCC_sub',
        method_suffix='_fixed_base',
    )
    assert result == 'ref2025_mp3_heatingSavings_coolingLCC_sub_econ_adopter_fixed_base'


def test_adoption_col_heatingSavings_coolingLCC_unsub():
    result = create_adoption_col(
        'ref2025_mp3_', 'heatingSavings_coolingLCC_unsub',
        method_suffix='_fixed_base',
    )
    assert result == 'ref2025_mp3_heatingSavings_coolingLCC_unsub_econ_adopter_fixed_base'


def test_adoption_col_heatingLCC_coolingLCC_sub():
    result = create_adoption_col(
        'ref2025_mp4_', 'heatingLCC_coolingLCC_sub',
        method_suffix='_fixed_base',
    )
    assert result == 'ref2025_mp4_heatingLCC_coolingLCC_sub_econ_adopter_fixed_base'


def test_adoption_col_heatingLCC_coolingLCC_unsub():
    result = create_adoption_col(
        'ref2025_mp4_', 'heatingLCC_coolingLCC_unsub',
        method_suffix='_fixed_base',
    )
    assert result == 'ref2025_mp4_heatingLCC_coolingLCC_unsub_econ_adopter_fixed_base'


def test_adoption_col_invalid_npv_case_raises():
    with pytest.raises(ValueError, match="Invalid npv_case"):
        create_adoption_col(
            'ref2025_mp3_', 'not_a_case',
            method_suffix='_fixed_base',
        )


# ── Total NPV column (climate-only) ──────────────────────────────────────────

def test_total_npv_col_climate_only():
    result = create_total_npv_col(
        'ref2025_mp3_', 'heating',
        cost_scenario='v4MID', method_suffix='_fixed_base',
        scc_assumption='central', climate_only=True,
    )
    assert result == 'ref2025_mp3_heating_total_npv_climateOnly_central_v4MID_fixed_base'


def test_total_npv_col_requires_climate_only():
    with pytest.raises(ValueError, match="climate_only=True"):
        create_total_npv_col(
            'ref2025_mp3_', 'heating',
            cost_scenario='v4MID', method_suffix='_fixed_base',
            scc_assumption='central', climate_only=False,
        )


# ── Cross-cutting: prefix consistency ─────────────────────────────────────────

def test_baseline_prefix_produces_baseline_columns():
    """All column builders using scenario_prefix accept 'baseline_'."""
    assert create_fuel_cost_col('baseline_', 'year1', 'heating').startswith('baseline_')
    assert create_capital_col('baseline_', 'heating', net=False, cost_scenario='v3').startswith('baseline_')
    assert create_climate_npv_col('baseline_', 'heating', 'central').startswith('baseline_')
    assert create_lifetime_damages_col('baseline_', 'heating', 'climate', 'lrmer', 'central').startswith('baseline_')


def test_all_categories_in_fuel_cost_col():
    """fuel_cost_col works for all four equipment categories."""
    for cat in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        result = create_fuel_cost_col('baseline_', 'year1', cat)
        assert cat in result


# -- Peak electric demand names follow the release ----------------------------

def test_peak_electricity_names_2022_are_unchanged():
    """2022.1.1 keeps its heating and cooling peak names."""
    from cmu_tare_model.utils.column_names import create_peak_electricity_col
    assert create_peak_electricity_col('base_', 'heating', '2022.1.1') == (
        'base_peak_electricity_heating_kw')
    assert create_peak_electricity_col('mp4_', 'cooling', '2022.1.1', savings=True) == (
        'mp4_peak_electricity_cooling_kw_savings')


def test_peak_electricity_names_2025_say_winter_and_summer():
    """2025.1 names say winter and summer; an unknown release or end use stops."""
    from cmu_tare_model.utils.column_names import create_peak_electricity_col
    assert create_peak_electricity_col('base_', 'heating', '2025.1') == (
        'base_peak_electricity_winter_kw')
    assert create_peak_electricity_col('base_', 'cooling', '2025.1') == (
        'base_peak_electricity_summer_kw')
    assert create_peak_electricity_col('mp5_', 'heating', '2025.1', savings=True) == (
        'mp5_peak_electricity_winter_kw_savings')
    with pytest.raises(KeyError):
        create_peak_electricity_col('base_', 'heating', '2030.1')
    with pytest.raises(KeyError, match='waterHeating'):
        create_peak_electricity_col('base_', 'waterHeating', '2025.1')


def test_tepper_list_uses_the_release_peak_names():
    """Under the default release (2022.1.1) the export keeps its peak names."""
    from cmu_tare_model.utils.export_tepper_csv import build_household_column_list
    columns = build_household_column_list(4, summary_columns=[])
    assert 'base_peak_electricity_heating_kw' in columns
    assert 'mp4_peak_electricity_cooling_kw_savings' in columns
    assert not any(
        'winter' in column or 'summer' in column for column in columns)


def test_tepper_list_adds_furnace_and_dual_fuel_columns_only_for_dual_fuel(
        monkeypatch):
    """Only a dual-fuel package gets the furnace cost and dual-fuel ratings."""
    from cmu_tare_model.utils import export_tepper_csv
    columns_mp4 = export_tepper_csv.build_household_column_list(
        4, summary_columns=[])
    assert not any('backupFurnace' in column for column in columns_mp4)
    assert 'upgrade_backup_afue' not in columns_mp4
    # The same list for a package treated as dual fuel.
    monkeypatch.setattr(
        export_tepper_csv, 'is_dual_fuel_package', lambda menu_mp: True)
    columns_dual = export_tepper_csv.build_household_column_list(
        4, summary_columns=[])
    assert 'mp4_heating_backupFurnace_installed_cost_v4MID' in columns_dual
    for column in ('upgrade_hp_seer2', 'upgrade_hp_seer1', 'upgrade_backup_afue',
                   'size_heat_pump_backup_primary_k_btu_h'):
        assert column in columns_dual
    # The furnace cost sits beside the heat pump's upgrade cost.
    position = columns_dual.index('mp4_heating_backupFurnace_installed_cost_v4MID')
    assert columns_dual[position - 1] == 'mp4_heating_upgrade_installed_cost_v4MID'


def test_tepper_list_holds_the_panel_columns_only_for_2025(monkeypatch):
    """A 2025.1 run's list holds the four electric panel columns; 2022.1.1 none."""
    from cmu_tare_model.utils import export_tepper_csv
    panel_columns = [
        'panel_service_rating_amps',
        'mp4_panel_constraint_overall',
        'mp4_panel_constraint_capacity',
        'mp4_panel_constraint_breaker_space',
    ]
    monkeypatch.setattr(export_tepper_csv, 'RESSTOCK_RELEASE_THIS_RUN', '2022.1.1')
    columns_2022 = export_tepper_csv.build_household_column_list(
        4, summary_columns=[])
    assert not any('panel' in column for column in columns_2022)
    # The same package number as a 2025.1 run: the panel columns follow the
    # release, not the package, so package 4 can stand in for any package.
    monkeypatch.setattr(export_tepper_csv, 'RESSTOCK_RELEASE_THIS_RUN', '2025.1')
    columns_2025 = export_tepper_csv.build_household_column_list(
        4, summary_columns=[])
    for column in panel_columns:
        assert column in columns_2025
    # They sit together, in this order.
    position = columns_2025.index('panel_service_rating_amps')
    assert columns_2025[position:position + 4] == panel_columns
