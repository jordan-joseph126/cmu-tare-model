"""Tests for cmu_tare_model.constants module.

Verifies that all critical constants have the expected types, values,
and relationships needed by the rest of the TARE model.
"""

import pytest
from cmu_tare_model.constants import (
    EQUIPMENT_SPECS,
    VALID_CATEGORIES,
    FUEL_MAPPING,
    UPGRADE_COLUMNS,
    POLLUTANTS,
    MER_TYPES,
    SCC_ASSUMPTIONS,
    CR_FUNCTIONS,
    RCM_MODELS,
    VALID_MENU_MPS,
    TD_LOSSES,
    TD_LOSSES_MULTIPLIER,
    PUBLIC_DISCOUNT_RATE,
    PRIVATE_FIXED_RATE_LOW,
    PRIVATE_FIXED_RATE_BASE,
    PRIVATE_FIXED_RATE_HIGH,
    VARIABLE_RATE_MIN,
    VARIABLE_RATE_MAX,
    AMI_THRESHOLD,
    PRIVATE_DISCOUNT_RATE_SHORT_KEYS,
    PRIVATE_DISCOUNT_RATE_COLS,
    REMDB_COST_SCENARIO_KEYS,
    EFFICIENCY_FLOORS_PM2,
)


# ── EQUIPMENT_SPECS ──────────────────────────────────────────────────────────

def test_equipment_specs_is_dict():
    assert isinstance(EQUIPMENT_SPECS, dict)


def test_equipment_specs_has_heating():
    """Heating should always be present (core model requirement)."""
    assert 'heating' in EQUIPMENT_SPECS


def test_equipment_specs_values_are_positive_integers():
    for cat, lifetime in EQUIPMENT_SPECS.items():
        assert isinstance(lifetime, int), f"{cat} lifetime should be int"
        assert lifetime > 0, f"{cat} lifetime should be positive"


def test_valid_categories_matches_equipment_specs():
    assert VALID_CATEGORIES == list(EQUIPMENT_SPECS.keys())


# ── FUEL_MAPPING ─────────────────────────────────────────────────────────────

def test_fuel_mapping_has_all_four_fuels():
    expected = {'Electricity', 'Natural Gas', 'Fuel Oil', 'Propane'}
    assert set(FUEL_MAPPING.keys()) == expected


def test_fuel_mapping_values_are_camelcase():
    for key, val in FUEL_MAPPING.items():
        assert isinstance(val, str)
        assert '_' not in val, f"FUEL_MAPPING values should be camelCase, got '{val}'"


# ── UPGRADE_COLUMNS ──────────────────────────────────────────────────────────

def test_upgrade_columns_keys_subset_of_equipment_specs():
    """Every key in UPGRADE_COLUMNS should correspond to an EQUIPMENT_SPECS category."""
    for cat in UPGRADE_COLUMNS:
        assert cat in EQUIPMENT_SPECS, f"UPGRADE_COLUMNS key '{cat}' not in EQUIPMENT_SPECS"


def test_upgrade_columns_values_are_strings():
    for cat, col in UPGRADE_COLUMNS.items():
        assert isinstance(col, str)
        assert col.startswith('upgrade_'), f"Upgrade column for {cat} should start with 'upgrade_'"


# ── Pollutants and sensitivity parameters ────────────────────────────────────

def test_pollutants_contains_four():
    assert set(POLLUTANTS) == {'so2', 'nox', 'pm25', 'co2e'}


def test_mer_types_contains_expected():
    assert 'lrmer' in MER_TYPES
    assert 'srmer' in MER_TYPES


def test_scc_assumptions_contains_expected():
    assert set(SCC_ASSUMPTIONS) == {'lower', 'central', 'upper'}


def test_cr_functions_contains_expected():
    assert set(CR_FUNCTIONS) == {'acs', 'h6c'}


def test_rcm_models_contains_expected():
    assert set(RCM_MODELS) == {'ap2', 'easiur', 'inmap'}


# ── VALID_MENU_MPS ───────────────────────────────────────────────────────────

def test_valid_menu_mps_contains_baseline():
    assert 0 in VALID_MENU_MPS


def test_valid_menu_mps_are_non_negative():
    for mp in VALID_MENU_MPS:
        assert mp >= 0


# ── Discount rates ───────────────────────────────────────────────────────────

def test_public_discount_rate_value():
    assert PUBLIC_DISCOUNT_RATE == 0.02


def test_private_fixed_rates_ordering():
    assert PRIVATE_FIXED_RATE_LOW <= PRIVATE_FIXED_RATE_BASE <= PRIVATE_FIXED_RATE_HIGH


def test_variable_rate_bounds():
    assert VARIABLE_RATE_MIN < VARIABLE_RATE_MAX
    assert VARIABLE_RATE_MIN > 0
    assert VARIABLE_RATE_MAX < 1.0  # Should be a decimal, not percentage


def test_ami_threshold_positive():
    assert AMI_THRESHOLD > 0


def test_discount_rate_short_keys_count():
    assert len(PRIVATE_DISCOUNT_RATE_SHORT_KEYS) == 4


def test_discount_rate_cols_count():
    assert len(PRIVATE_DISCOUNT_RATE_COLS) == 4


# ── TD_LOSSES ────────────────────────────────────────────────────────────────

def test_td_losses_multiplier_formula():
    """TD_LOSSES_MULTIPLIER should equal 1/(1-TD_LOSSES)."""
    expected = 1 / (1 - TD_LOSSES)
    assert TD_LOSSES_MULTIPLIER == pytest.approx(expected)


def test_td_losses_between_0_and_1():
    assert 0 < TD_LOSSES < 1


# ── REMDB cost scenarios ─────────────────────────────────────────────────────

def test_remdb_cost_scenario_keys_are_strings():
    for key in REMDB_COST_SCENARIO_KEYS:
        assert isinstance(key, str)


# ── Efficiency floors ────────────────────────────────────────────────────────

def test_efficiency_floors_values_positive():
    for tech, floor in EFFICIENCY_FLOORS_PM2.items():
        assert floor > 0, f"Floor for {tech} should be positive"
