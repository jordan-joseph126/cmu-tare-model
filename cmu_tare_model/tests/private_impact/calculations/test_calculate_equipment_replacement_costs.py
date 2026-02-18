"""Tests for cmu_tare_model.private_impact.calculations.calculate_equipment_replacement_costs module.

Verifies input validation, v4 regression calculation, and the public API
for replacement installed cost computation (like-for-like counterfactual).
"""

import pytest
import pandas as pd
import numpy as np

MODULE = 'cmu_tare_model.private_impact.calculations.calculate_equipment_replacement_costs'


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
    from cmu_tare_model.private_impact.calculations.calculate_equipment_replacement_costs import (
        _validate_inputs,
    )
    _validate_inputs(menu_mp=3, end_use='heating', cost_scenario='v4MID')


def test_validate_inputs_invalid_menu_mp(mock_constants):
    from cmu_tare_model.private_impact.calculations.calculate_equipment_replacement_costs import (
        _validate_inputs,
    )
    with pytest.raises(ValueError, match="valid measure package"):
        _validate_inputs(menu_mp=99, end_use='heating', cost_scenario='v3')


def test_validate_inputs_invalid_end_use(mock_constants):
    from cmu_tare_model.private_impact.calculations.calculate_equipment_replacement_costs import (
        _validate_inputs,
    )
    with pytest.raises(ValueError, match="Invalid end_use"):
        _validate_inputs(menu_mp=3, end_use='invalid', cost_scenario='v3')


def test_validate_inputs_cooling_allowed(mock_constants):
    """Replacement costs allow 'cooling' even if not in standard VALID_CATEGORIES."""
    from cmu_tare_model.private_impact.calculations.calculate_equipment_replacement_costs import (
        _validate_inputs,
    )
    # Should not raise — cooling is allowed for replacement
    _validate_inputs(menu_mp=3, end_use='cooling', cost_scenario='v4MID')


def test_validate_inputs_invalid_cost_scenario(mock_constants):
    from cmu_tare_model.private_impact.calculations.calculate_equipment_replacement_costs import (
        _validate_inputs,
    )
    with pytest.raises(ValueError, match="Invalid cost_scenario"):
        _validate_inputs(menu_mp=3, end_use='heating', cost_scenario='invalid')


# ── _calculate_v4_replacement ────────────────────────────────────────────────

def test_v4_replacement_regression_formula(mock_constants):
    """Verify REMDB v4 regression: Material = pm1*coef + pm2*coef + intercept, then installed = material*mult + adder."""
    from cmu_tare_model.private_impact.calculations.calculate_equipment_replacement_costs import (
        _calculate_v4_replacement,
    )
    prefix = 'heating_replacement_'
    df_detailed = pd.DataFrame({
        f'{prefix}pm1_euss': [36.0, 48.0],
        f'{prefix}pm2_euss': [0.80, 0.95],
        f'{prefix}pm1_coef_mid': [10.0, 10.0],
        f'{prefix}pm2_coef_mid': [100.0, 100.0],
        f'{prefix}intercept_mid': [50.0, 50.0],
        f'{prefix}multiplier_retrofit': [1.5, 1.5],
        f'{prefix}adder_retrofit': [200.0, 200.0],
    })
    result = _calculate_v4_replacement(df_detailed, 'heating', 'mid')
    # Row 0: material = 36*10 + 0.80*100 + 50 = 360+80+50 = 490
    #         installed = 490*1.5 + 200 = 935
    assert result.iloc[0] == pytest.approx(935.0)


def test_v4_replacement_missing_columns_raises(mock_constants):
    from cmu_tare_model.private_impact.calculations.calculate_equipment_replacement_costs import (
        _calculate_v4_replacement,
    )
    df_detailed = pd.DataFrame({'other_col': [1, 2]})
    with pytest.raises(KeyError, match="Missing columns"):
        _calculate_v4_replacement(df_detailed, 'heating', 'mid')


# ── calculate_replacement_installed_cost (public API) ────────────────────────

def test_replacement_installed_cost_v4_output(mock_constants):
    from cmu_tare_model.private_impact.calculations.calculate_equipment_replacement_costs import (
        calculate_replacement_installed_cost,
    )
    n = 5
    prefix = 'heating_replacement_'
    df = pd.DataFrame({
        'include_heating': [True, True, True, False, True],
        'base_heating_fuel': ['Natural Gas', 'Propane', 'Fuel Oil', 'Electricity', 'Natural Gas'],
        'upgrade_hvac_heating_efficiency': ['ASHP', 'ASHP', 'ASHP', None, 'ASHP'],
    })
    df_detailed = pd.DataFrame({
        f'{prefix}pm1_euss': np.random.uniform(30, 60, n),
        f'{prefix}pm2_euss': np.random.uniform(0.8, 0.95, n),
        f'{prefix}pm1_coef_mid': [10.0] * n,
        f'{prefix}pm2_coef_mid': [100.0] * n,
        f'{prefix}intercept_mid': [50.0] * n,
        f'{prefix}multiplier_retrofit': [1.5] * n,
        f'{prefix}adder_retrofit': [200.0] * n,
    })
    df_copy, df_det_out = calculate_replacement_installed_cost(
        df, df_detailed, menu_mp=8, end_use='heating', cost_scenario='v4MID', verbose=False)
    cost_col = 'mp8_heating_replacement_installed_cost_v4MID'
    assert cost_col in df_copy.columns
    assert pd.isna(df_copy[cost_col].iloc[3])  # Invalid home
    valid_costs = df_copy[cost_col][df['include_heating']]
    assert (valid_costs.dropna() >= 0).all()


def test_replacement_installed_cost_v3_requires_cost_dict(mock_constants):
    from cmu_tare_model.private_impact.calculations.calculate_equipment_replacement_costs import (
        calculate_replacement_installed_cost,
    )
    df = pd.DataFrame({
        'include_heating': [True],
        'base_heating_fuel': ['Natural Gas'],
        'upgrade_hvac_heating_efficiency': ['ASHP'],
    })
    df_detailed = pd.DataFrame({'col': [1]})
    with pytest.raises(ValueError, match="cost_dict is required"):
        calculate_replacement_installed_cost(
            df, df_detailed, menu_mp=8, end_use='heating', cost_scenario='v3', verbose=False)


def test_replacement_cost_clipped_to_zero(mock_constants):
    """Costs should never go negative (clipped to 0)."""
    from cmu_tare_model.private_impact.calculations.calculate_equipment_replacement_costs import (
        calculate_replacement_installed_cost,
    )
    n = 3
    prefix = 'heating_replacement_'
    df = pd.DataFrame({
        'include_heating': [True, True, True],
        'base_heating_fuel': ['Natural Gas'] * n,
        'upgrade_hvac_heating_efficiency': ['ASHP'] * n,
    })
    # Use negative intercepts to force negative costs
    df_detailed = pd.DataFrame({
        f'{prefix}pm1_euss': [1.0] * n,
        f'{prefix}pm2_euss': [0.8] * n,
        f'{prefix}pm1_coef_mid': [0.1] * n,
        f'{prefix}pm2_coef_mid': [0.1] * n,
        f'{prefix}intercept_mid': [-10000.0] * n,
        f'{prefix}multiplier_retrofit': [1.0] * n,
        f'{prefix}adder_retrofit': [0.0] * n,
    })
    df_copy, _ = calculate_replacement_installed_cost(
        df, df_detailed, menu_mp=8, end_use='heating', cost_scenario='v4MID', verbose=False)
    cost_col = 'mp8_heating_replacement_installed_cost_v4MID'
    assert (df_copy[cost_col].dropna() >= 0).all()
