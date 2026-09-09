"""Tests for cmu_tare_model.utils.remdb_v4_installed_cost_utils.

Focused on one thing: add_remdb_metrics() must price the REPLACEMENT cost
(the avoided cost of replacing the home's OLD system) off the old system's
own size, and price the UPGRADE cost (the new heat pump) off the heat
pump's own size. Before the 20 Aug 2026 fix, both cases used the heat
pump's size, which understated or overstated the replacement credit
depending on the home. See docs/SESSION_CHANGELOG_2026-08-20.md.

This test builds a tiny, hand-written REMDB v4 cost table instead of
loading the real one, so it runs fast and does not depend on external
data files. The three homes below are picked so that the old system's
size and the heat pump's size are clearly different numbers, making it
obvious which one the function actually used.
"""

import pandas as pd
import pytest

from cmu_tare_model.utils.remdb_v4_installed_cost_utils import add_remdb_metrics


@pytest.fixture
def remdb_v4_costs() -> pd.DataFrame:
    """A minimal REMDB v4 cost table covering one heating and one cooling row.

    Only the columns add_remdb_metrics() actually reads are included. Real
    dollar coefficients are not needed here -- this test checks which size
    column feeds the calculation, not the resulting price.
    """
    rows = {
        # Old gas furnace, priced as a replacement. REMDB stores furnace
        # capacity in BTU/hr, so the kBtu/h input gets multiplied by 1000.
        'furnaces_gas_furnace': {
            'pm1_metric': 'capacity', 'pm1_unit': 'BTU/hr',
            'pm2_metric': 'AFUE', 'pm2_unit': 'fraction',
        },
        # New heat pump, priced as an upgrade (heating) or as a replacement
        # for the old central AC's slot (cooling never uses 'upgrade' -- see
        # the module docstring). REMDB stores heat pump / AC capacity in
        # tons, so the kBtu/h input gets divided by 12.
        'air_source_heat_pump_centrally_ducted': {
            'pm1_metric': 'capacity', 'pm1_unit': 'Tons',
            'pm2_metric': 'SEER', 'pm2_unit': 'SEER',
        },
        'air_conditioner_centrally_ducted': {
            'pm1_metric': 'capacity', 'pm1_unit': 'Tons',
            'pm2_metric': 'SEER', 'pm2_unit': 'SEER',
        },
    }
    return pd.DataFrame.from_dict(rows, orient='index')


@pytest.fixture
def df_home() -> pd.DataFrame:
    """One home with a gas furnace and a central AC.

    The old-system sizes (base_size_*) and the heat pump's size (size_*)
    are set to clearly different numbers so a test failure is obvious: if
    the function reads the wrong column, the result lands on the wrong
    one of these two values instead of somewhere in between.
    """
    return pd.DataFrame({
        'base_heating_fuel': ['Natural Gas'],
        'heating_type': ['Natural Gas Fuel Furnace'],
        'hvac_has_ducts': ['Yes'],
        'hvac_cooling_type': ['Central AC'],
        # Old furnace: 60 kBtu/h. New heat pump: 20 kBtu/h.
        'base_size_heating_system_primary_k_btu_h': [60.0],
        'size_heating_system_primary_k_btu_h': [20.0],
        'base_heating_efficiency': ['80% AFUE'],
        'upgrade_hvac_heating_efficiency': ['SEER 15'],
        # Old central AC: 36 kBtu/h. New heat pump: 24 kBtu/h.
        'base_size_cooling_system_primary_k_btu_h': [36.0],
        'size_cooling_system_primary_k_btu_h': [24.0],
        'base_cooling_efficiency': ['SEER 16'],
    })


def test_heating_replacement_uses_old_furnace_size(df_home, remdb_v4_costs):
    """The heating replacement cost must be sized off the old furnace, not the new heat pump."""
    df_main, _ = add_remdb_metrics(
        df_home, remdb_v4_costs, 'heating', 'replacement', verbose=False)
    # 60 kBtu/h old furnace -> 60,000 BTU/hr after the unit conversion.
    assert df_main['heating_replacement_pm1_euss'].iloc[0] == pytest.approx(60_000.0)


def test_heating_upgrade_uses_new_heat_pump_size(df_home, remdb_v4_costs):
    """The heat pump's own upgrade cost must still be sized off the heat pump, unchanged by this fix."""
    df_main, _ = add_remdb_metrics(
        df_home, remdb_v4_costs, 'heating', 'upgrade', verbose=False)
    # 20 kBtu/h heat pump -> 1.6667 tons after the unit conversion.
    assert df_main['heating_upgrade_pm1_euss'].iloc[0] == pytest.approx(20.0 / 12.0)


def test_cooling_replacement_uses_old_ac_size(df_home, remdb_v4_costs):
    """The cooling replacement cost must be sized off the old air conditioner, not the new heat pump."""
    df_main, _ = add_remdb_metrics(
        df_home, remdb_v4_costs, 'cooling', 'replacement', verbose=False)
    # 36 kBtu/h old AC -> 3.0 tons after the unit conversion.
    assert df_main['cooling_replacement_pm1_euss'].iloc[0] == pytest.approx(36.0 / 12.0)
