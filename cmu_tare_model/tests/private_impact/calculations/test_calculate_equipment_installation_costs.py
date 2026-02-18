"""Tests for cmu_tare_model.private_impact.calculations.calculate_equipment_installation_costs module.

Verifies input validation, v4 regression calculation, and the public API
for upgrade installed cost computation across REMDB v3/v4 scenarios.
"""

import pytest
import pandas as pd
import numpy as np

MODULE = 'cmu_tare_model.private_impact.calculations.calculate_equipment_installation_costs'


@pytest.fixture
def mock_constants(monkeypatch):
    monkeypatch.setattr(f'{MODULE}.VALID_MENU_MPS', [0, 3, 4, 8, 9, 10])
    monkeypatch.setattr(f'{MODULE}.VALID_CATEGORIES', ['heating', 'waterHeating', 'clothesDrying', 'cooking'])
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {
        'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15
    })
    monkeypatch.setattr(f'{MODULE}.REMDB_COST_SCENARIO_KEYS', ['v3', 'v4LOW', 'v4MID', 'v4HIGH'])


# ── _validate_inputs ─────────────────────────────────────────────────────────

def test_validate_inputs_valid(mock_constants):
    from cmu_tare_model.private_impact.calculations.calculate_equipment_installation_costs import (
        _validate_inputs,
    )
    # Should not raise
    _validate_inputs(menu_mp=3, end_use='heating', cost_scenario='v3')
    _validate_inputs(menu_mp=8, end_use='waterHeating', cost_scenario='v4MID')


def test_validate_inputs_invalid_menu_mp(mock_constants):
    from cmu_tare_model.private_impact.calculations.calculate_equipment_installation_costs import (
        _validate_inputs,
    )
    with pytest.raises(ValueError, match="valid measure package"):
        _validate_inputs(menu_mp=99, end_use='heating', cost_scenario='v3')


def test_validate_inputs_invalid_end_use(mock_constants):
    from cmu_tare_model.private_impact.calculations.calculate_equipment_installation_costs import (
        _validate_inputs,
    )
    with pytest.raises(ValueError, match="Invalid end_use"):
        _validate_inputs(menu_mp=3, end_use='invalid', cost_scenario='v3')


def test_validate_inputs_invalid_cost_scenario(mock_constants):
    from cmu_tare_model.private_impact.calculations.calculate_equipment_installation_costs import (
        _validate_inputs,
    )
    with pytest.raises(ValueError, match="Invalid cost_scenario"):
        _validate_inputs(menu_mp=3, end_use='heating', cost_scenario='invalid')


# ── _calculate_v4_upgrade ────────────────────────────────────────────────────

def test_v4_upgrade_regression_formula(mock_constants):
    """Verify REMDB v4 regression: Material = pm1*coef + pm2*coef + intercept, then installed = material*mult + adder."""
    from cmu_tare_model.private_impact.calculations.calculate_equipment_installation_costs import (
        _calculate_v4_upgrade,
    )
    n = 3
    prefix = 'heating_upgrade_'
    df_detailed = pd.DataFrame({
        f'{prefix}pm1_euss': [36.0, 48.0, 60.0],
        f'{prefix}pm2_euss': [15.0, 18.0, 21.0],
        f'{prefix}pm1_coef_mid': [10.0, 10.0, 10.0],
        f'{prefix}pm2_coef_mid': [50.0, 50.0, 50.0],
        f'{prefix}intercept_mid': [100.0, 100.0, 100.0],
        f'{prefix}multiplier_retrofit': [1.5, 1.5, 1.5],
        f'{prefix}adder_retrofit': [200.0, 200.0, 200.0],
    })
    result = _calculate_v4_upgrade(df_detailed, 'heating', 'mid')
    # Row 0: material = 36*10 + 15*50 + 100 = 360+750+100 = 1210
    #         installed = 1210*1.5 + 200 = 2015
    assert result.iloc[0] == pytest.approx(2015.0)


def test_v4_upgrade_missing_columns_raises(mock_constants):
    from cmu_tare_model.private_impact.calculations.calculate_equipment_installation_costs import (
        _calculate_v4_upgrade,
    )
    df_detailed = pd.DataFrame({'other_col': [1, 2, 3]})
    with pytest.raises(KeyError, match="Missing columns"):
        _calculate_v4_upgrade(df_detailed, 'heating', 'mid')


# ── calculate_upgrade_installed_cost (public API) ────────────────────────────

def test_upgrade_installed_cost_v4_output(mock_constants):
    """Integration test for v4 upgrade cost calculation."""
    from cmu_tare_model.private_impact.calculations.calculate_equipment_installation_costs import (
        calculate_upgrade_installed_cost,
    )
    n = 5
    prefix = 'heating_upgrade_'
    df = pd.DataFrame({
        'include_heating': [True, True, True, False, True],
        'base_heating_fuel': ['Natural Gas'] * 5,
        'upgrade_hvac_heating_efficiency': ['ASHP', 'ASHP', 'ASHP', None, 'ASHP'],
    })
    df_detailed = pd.DataFrame({
        f'{prefix}pm1_euss': np.random.uniform(30, 60, n),
        f'{prefix}pm2_euss': np.random.uniform(14, 22, n),
        f'{prefix}pm1_coef_mid': [10.0] * n,
        f'{prefix}pm2_coef_mid': [50.0] * n,
        f'{prefix}intercept_mid': [100.0] * n,
        f'{prefix}multiplier_retrofit': [1.5] * n,
        f'{prefix}adder_retrofit': [200.0] * n,
    })
    df_copy, df_det_out = calculate_upgrade_installed_cost(
        df, df_detailed, menu_mp=8, end_use='heating', cost_scenario='v4MID', verbose=False)
    cost_col = 'mp8_heating_upgrade_installed_cost_v4MID'
    assert cost_col in df_copy.columns
    # Invalid home (index 3) should have NaN cost
    assert pd.isna(df_copy[cost_col].iloc[3])
    # Valid homes should have positive costs
    valid_costs = df_copy[cost_col][df['include_heating']]
    assert (valid_costs.dropna() > 0).all()


def test_upgrade_installed_cost_v3_requires_cost_dict(mock_constants):
    """v3 without cost_dict should raise ValueError."""
    from cmu_tare_model.private_impact.calculations.calculate_equipment_installation_costs import (
        calculate_upgrade_installed_cost,
    )
    df = pd.DataFrame({
        'include_heating': [True],
        'base_heating_fuel': ['Natural Gas'],
        'upgrade_hvac_heating_efficiency': ['ASHP'],
    })
    df_detailed = pd.DataFrame({'col': [1]})
    with pytest.raises(ValueError, match="cost_dict is required"):
        calculate_upgrade_installed_cost(
            df, df_detailed, menu_mp=8, end_use='heating', cost_scenario='v3', verbose=False)


# ── obtain_heating_system_specs ──────────────────────────────────────────────

def test_obtain_heating_system_specs_extracts_afue(mock_constants):
    from cmu_tare_model.private_impact.calculations.calculate_equipment_installation_costs import (
        obtain_heating_system_specs,
    )
    df = pd.DataFrame({
        'hvac_heating_efficiency': ['80% AFUE', '95% AFUE', '100% Efficiency'],
        'upgrade_hvac_heating_efficiency': ['SEER 18, 9.3 HSPF'] * 3,
    })
    result = obtain_heating_system_specs(df)
    assert result['baseline_AFUE'].iloc[0] == 80.0
    assert result['baseline_AFUE'].iloc[1] == 95.0


def test_obtain_heating_system_specs_extracts_seer(mock_constants):
    from cmu_tare_model.private_impact.calculations.calculate_equipment_installation_costs import (
        obtain_heating_system_specs,
    )
    df = pd.DataFrame({
        'hvac_heating_efficiency': ['SEER 15, 8.8 HSPF'],
        'upgrade_hvac_heating_efficiency': ['SEER 18, 9.3 HSPF'],
    })
    result = obtain_heating_system_specs(df)
    assert result['baseline_SEER'].iloc[0] == 15.0
    assert result['baseline_HSPF'].iloc[0] == 8.8


def test_obtain_heating_system_specs_missing_columns_raises(mock_constants):
    from cmu_tare_model.private_impact.calculations.calculate_equipment_installation_costs import (
        obtain_heating_system_specs,
    )
    df = pd.DataFrame({'other_col': [1]})
    with pytest.raises(ValueError, match="necessary columns"):
        obtain_heating_system_specs(df)


# ── calculate_heating_installation_premium ───────────────────────────────────

def test_heating_installation_premium_no_ac(mock_constants):
    from cmu_tare_model.private_impact.calculations.calculate_equipment_installation_costs import (
        calculate_heating_installation_premium,
    )
    df = pd.DataFrame({
        'hvac_cooling_type': ['None', 'Central AC', 'None'],
        'heating_type': ['Natural Gas Fuel Furnace', 'Natural Gas Fuel Furnace', 'Natural Gas Fuel Boiler'],
    })
    result = calculate_heating_installation_premium(df, menu_mp=3, cpi_ratio_2023_2013=1.3)
    col = 'mp3_heating_installation_premium'
    assert col in result.columns
    # Row 0: No AC + Furnace -> 400 * CPI
    assert result[col].iloc[0] == pytest.approx(400 * 1.3, rel=0.01)
    # Row 1: Has central AC -> 0
    assert result[col].iloc[1] == 0.0
    # Row 2: No AC + Boiler -> 1500 * CPI
    assert result[col].iloc[2] == pytest.approx(1500 * 1.3, rel=0.01)


def test_heating_installation_premium_missing_columns_raises(mock_constants):
    from cmu_tare_model.private_impact.calculations.calculate_equipment_installation_costs import (
        calculate_heating_installation_premium,
    )
    df = pd.DataFrame({'other_col': [1]})
    with pytest.raises(ValueError, match="necessary columns"):
        calculate_heating_installation_premium(df, menu_mp=3, cpi_ratio_2023_2013=1.0)
