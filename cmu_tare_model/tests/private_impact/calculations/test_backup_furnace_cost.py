"""Tests for the dual-fuel backup furnace cost (G7).

add_backup_furnace_metrics (utils/remdb_v4_installed_cost_utils.py) prepares
the furnace's REMDB inputs and calculate_backup_furnace_installed_cost
(private_impact/calculations/calculate_equipment_installation_costs.py) prices
it. The REMDB row below is written out by hand with the real
'furnaces_gas_furnace' coefficients, so no data file is read.
"""

import numpy as np
import pandas as pd
import pytest

from cmu_tare_model.utils.remdb_v4_installed_cost_utils import (
    add_backup_furnace_metrics,
)

# The real REMDB v4 'furnaces_gas_furnace' row (2023 dollars).
_FURNACE_ROW = {
    'pm1_metric': 'Heating Capacity', 'pm1_unit': 'BTU/Hr',
    'pm1_coef_low': 0.004744, 'pm1_coef_mid': 0.00645, 'pm1_coef_high': 0.01105,
    'pm1_lower_bound': 30000.0, 'pm1_upper_bound': 156250.0,
    'pm2_metric': 'AFUE', 'pm2_unit': 'Unitless',
    'pm2_coef_low': 3309.827024, 'pm2_coef_mid': 4325.000025,
    'pm2_coef_high': 4862.6875,
    'pm2_lower_bound': 0.8, 'pm2_upper_bound': 0.97,
    'intercept_low': -2009.224953, 'intercept_mid': -2780.000019,
    'intercept_high': -2910.150001,
    'multiplier_retrofit': 1.0, 'adder_retrofit': 2217.753,
}


@pytest.fixture
def remdb_v4_costs() -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        {'furnaces_gas_furnace': _FURNACE_ROW}, orient='index')


@pytest.fixture
def df_dual_fuel() -> pd.DataFrame:
    """Three sample homes with a gas backup and one home outside the package."""
    return pd.DataFrame({
        'size_heat_pump_backup_primary_k_btu_h': [63.2, 80.0, 40.0, 50.0],
        'upgrade_backup_fuel': ['Natural Gas', 'Natural Gas', 'Natural Gas', None],
        'upgrade_backup_afue': [0.925, 0.95, 0.925, np.nan],
        'upgrade_hvac_heating_efficiency': ['dual fuel'] * 3 + [None],
        'include_heating': [True, True, True, False],
        'include_sample': [True, True, True, False],
    }, index=pd.Index([1, 2, 3, 4], name='bldg_id'))


def _expected_2023_cost(size_kbtu_h: float, afue: float) -> float:
    """The REMDB formula, written out: (pm1 c1 + pm2 c2 + b) x m + a."""
    material = (size_kbtu_h * 1000 * _FURNACE_ROW['pm1_coef_mid']
                + afue * _FURNACE_ROW['pm2_coef_mid']
                + _FURNACE_ROW['intercept_mid'])
    return (material * _FURNACE_ROW['multiplier_retrofit']
            + _FURNACE_ROW['adder_retrofit'])


def test_metrics_use_the_backup_size_and_afue_fraction(df_dual_fuel, remdb_v4_costs):
    df_main, df_detailed = add_backup_furnace_metrics(
        df_dual_fuel, remdb_v4_costs, verbose=False)
    pm1 = df_main['heating_backupFurnace_pm1_euss']
    pm2 = df_main['heating_backupFurnace_pm2_euss']
    assert pm1.iloc[:3].tolist() == pytest.approx([63200.0, 80000.0, 40000.0])
    # The AFUE reaches the regression as the fraction, not divided again.
    assert pm2.iloc[:3].tolist() == pytest.approx([0.925, 0.95, 0.925])
    # The home outside the package gets no row and no metrics.
    assert pd.isna(df_main['row_id_heating_backupFurnace'].iloc[3])
    assert np.isnan(pm1.iloc[3]) and np.isnan(pm2.iloc[3])
    assert 'heating_backupFurnace_intercept_mid' in df_detailed.columns


@pytest.mark.parametrize('bad_afue', [0.00925, 92.5])
def test_metrics_stop_on_an_afue_read_wrongly(df_dual_fuel, remdb_v4_costs, bad_afue):
    df_bad = df_dual_fuel.copy()
    df_bad.loc[1, 'upgrade_backup_afue'] = bad_afue
    with pytest.raises(ValueError, match='AFUE'):
        add_backup_furnace_metrics(df_bad, remdb_v4_costs, verbose=False)


def test_metrics_stop_on_a_backup_fuel_with_no_row(df_dual_fuel, remdb_v4_costs):
    df_propane = df_dual_fuel.copy()
    df_propane.loc[2, 'upgrade_backup_fuel'] = 'Propane'
    with pytest.raises(ValueError, match='Propane'):
        add_backup_furnace_metrics(df_propane, remdb_v4_costs, verbose=False)


def test_furnace_cost_follows_the_remdb_formula(
        df_dual_fuel, remdb_v4_costs, monkeypatch):
    from cmu_tare_model.private_impact.calculations import (
        calculate_equipment_installation_costs as installation,
    )
    from cmu_tare_model.utils.inflation_adjustment import cpi_ratio_2025_2023
    # Run as the 2025.1 dual-fuel package, whatever release the tests run as.
    monkeypatch.setattr(installation, 'VALID_MENU_MPS', [0, 5])
    monkeypatch.setattr(installation, 'is_dual_fuel_package', lambda mp: mp == 5)

    df_main, df_detailed = add_backup_furnace_metrics(
        df_dual_fuel, remdb_v4_costs, verbose=False)
    df_out, _ = installation.calculate_backup_furnace_installed_cost(
        df_main, df_detailed, menu_mp=5, cost_scenario='v4MID', verbose=False)

    cost = df_out['mp5_heating_backupFurnace_installed_cost_v4MID']
    expected = [
        round(_expected_2023_cost(size, afue) * cpi_ratio_2025_2023, 2)
        for size, afue in ((63.2, 0.925), (80.0, 0.95), (40.0, 0.925))]
    assert cost.iloc[:3].tolist() == pytest.approx(expected, abs=0.006)
    # About $3,846 in 2023 dollars for the average home's 63.2 kBtu/h furnace.
    assert _expected_2023_cost(63.2, 0.925) == pytest.approx(3846.02, abs=0.01)
    # Outside the study sample the cost is blank.
    assert np.isnan(cost.iloc[3])


@pytest.mark.parametrize('menu_mp', [3, 4])
def test_no_furnace_cost_for_a_heat_pump_only_package(
        df_dual_fuel, remdb_v4_costs, menu_mp):
    # Run as the tests' default release (2022.1.1): MP3 and MP4 are heat
    # pumps with no backup furnace, so pricing one must stop.
    from cmu_tare_model.private_impact.calculations import (
        calculate_equipment_installation_costs as installation,
    )
    df_main, df_detailed = add_backup_furnace_metrics(
        df_dual_fuel, remdb_v4_costs, verbose=False)
    with pytest.raises(ValueError, match='not a dual-fuel package'):
        installation.calculate_backup_furnace_installed_cost(
            df_main, df_detailed, menu_mp=menu_mp, cost_scenario='v4MID',
            verbose=False)
