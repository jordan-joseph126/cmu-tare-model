"""Tests for compute_scenario_demand and build_demand_breakdown
(adoption_kpis/demand.py): only the study sample is counted, every change is
read from ResStock's savings columns, and the breakdown adds up.

The adoption_kpis package imports its geospatial module on load, so these tests
need geopandas and are skipped where it is not installed.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("geopandas")

from cmu_tare_model.adoption_kpis.data_loading import (
    COOLING_ELEC_SAVINGS_COLS,
    COUNTY_COL,
    ELEC_TOTAL_COL,
    ELEC_TOTAL_SAVINGS_COL,
    HEATING_ELEC_SAVINGS_COLS,
    HOT_WATER_ELEC_SAVINGS_COL,
    SITE_ENERGY_TOTAL_COL,
    SITE_ENERGY_TOTAL_SAVINGS_COL,
)
from cmu_tare_model.adoption_kpis.demand import (
    build_demand_breakdown,
    compute_scenario_demand,
)

WEIGHT = 242.131013


@pytest.fixture
def euss_pair():
    """Five homes. Each retrofit adds 100 x bldg_id kWh of electricity (heating
    up by 150 x bldg_id, cooling down by 40 x bldg_id, hot water down by
    10 x bldg_id) and cuts site energy (all fuels) by 5,000 kWh. Homes 1-3
    heated with natural gas, homes 4-5 with electricity."""
    bldg_ids = pd.Index([1, 2, 3, 4, 5], name='bldg_id')
    per_home = bldg_ids.to_numpy().astype(float)
    df_baseline = pd.DataFrame({
        'in.state': ['PA'] * 5,
        COUNTY_COL: ['G4200030'] * 5,
        'in.heating_fuel': ['Natural Gas'] * 3 + ['Electricity'] * 2,
        'weight': [WEIGHT] * 5,
        ELEC_TOTAL_COL: [1000.0] * 5,
        SITE_ENERGY_TOTAL_COL: [20000.0] * 5,
    }, index=bldg_ids)
    heating_col, backup_col, heating_fans_col = HEATING_ELEC_SAVINGS_COLS
    cooling_col, cooling_fans_col = COOLING_ELEC_SAVINGS_COLS
    df_upgrade = pd.DataFrame({
        ELEC_TOTAL_COL: 1000.0 + 100.0 * per_home,
        SITE_ENERGY_TOTAL_COL: [15000.0] * 5,
        # ResStock's savings: baseline minus upgrade, so a rise is negative.
        ELEC_TOTAL_SAVINGS_COL: -100.0 * per_home,
        SITE_ENERGY_TOTAL_SAVINGS_COL: [5000.0] * 5,
        heating_col: -100.0 * per_home,
        backup_col: -30.0 * per_home,
        heating_fans_col: -20.0 * per_home,
        cooling_col: 30.0 * per_home,
        cooling_fans_col: 10.0 * per_home,
        HOT_WATER_ELEC_SAVINGS_COL: 10.0 * per_home,
    }, index=bldg_ids)
    return df_baseline, df_upgrade


def test_only_sample_homes_counted(euss_pair):
    df_b, df_u = euss_pair
    sample = pd.Index([1, 3, 5])
    result = compute_scenario_demand(df_b, df_u, sample_bldg_ids=sample)
    assert sorted(result.index) == [1, 3, 5]


def test_kept_homes_values_unchanged(euss_pair):
    df_b, df_u = euss_pair
    sample = pd.Index([2, 4])
    result = compute_scenario_demand(df_b, df_u, sample_bldg_ids=sample)
    assert np.allclose(result.loc[[2, 4], 'elec_demand_change_kwh'], [200.0, 400.0])


def test_missing_sample_home_raises(euss_pair):
    df_b, df_u = euss_pair
    sample = pd.Index([1, 6])
    with pytest.raises(ValueError, match='missing'):
        compute_scenario_demand(df_b, df_u, sample_bldg_ids=sample)


def test_bad_sample_ids_raise(euss_pair):
    df_b, df_u = euss_pair
    with pytest.raises(ValueError, match='empty'):
        compute_scenario_demand(df_b, df_u, sample_bldg_ids=pd.Index([]))
    with pytest.raises(TypeError, match='pd.Index'):
        compute_scenario_demand(df_b, df_u, sample_bldg_ids=[1, 2])


def test_changes_read_from_savings_columns(euss_pair):
    """Each change is minus ResStock's savings; 'other' is what is left."""
    df_b, df_u = euss_pair
    result = compute_scenario_demand(df_b, df_u, sample_bldg_ids=df_b.index)
    per_home = df_b.index.to_numpy().astype(float)
    assert np.allclose(result['elec_demand_change_kwh'], 100.0 * per_home)
    assert np.allclose(result['heating_elec_change_kwh'], 150.0 * per_home)
    assert np.allclose(result['cooling_elec_change_kwh'], -40.0 * per_home)
    assert np.allclose(result['hot_water_elec_change_kwh'], -10.0 * per_home)
    assert np.allclose(result['other_elec_change_kwh'], 0.0)


def test_site_energy_is_its_own_all_fuel_change(euss_pair):
    """Site energy falls while electricity rises; it is not a copy."""
    df_b, df_u = euss_pair
    result = compute_scenario_demand(df_b, df_u, sample_bldg_ids=df_b.index)
    assert np.allclose(result['site_energy_change_kwh'], -5000.0)
    assert (result['elec_demand_change_kwh'] > 0).all()


def test_savings_disagreeing_with_totals_raises(euss_pair):
    """ResStock's savings column must agree with retrofit minus baseline."""
    df_b, df_u = euss_pair
    df_u.loc[3, ELEC_TOTAL_SAVINGS_COL] += 50.0
    with pytest.raises(ValueError, match='elec_demand_change_kwh'):
        compute_scenario_demand(df_b, df_u, sample_bldg_ids=df_b.index)


def test_blank_in_sample_home_raises(euss_pair):
    """A blank in a sample home stops the run; outside the sample it does not."""
    df_b, df_u = euss_pair
    df_u.loc[2, HOT_WATER_ELEC_SAVINGS_COL] = np.nan
    with pytest.raises(ValueError, match='blank'):
        compute_scenario_demand(df_b, df_u, sample_bldg_ids=df_b.index)
    result = compute_scenario_demand(df_b, df_u, sample_bldg_ids=pd.Index([1, 3]))
    assert sorted(result.index) == [1, 3]


def test_breakdown_adds_up(euss_pair):
    """Baseline plus the net changes is post-retrofit; fuel rows add to heating."""
    df_b, df_u = euss_pair
    result = compute_scenario_demand(df_b, df_u, sample_bldg_ids=df_b.index)
    breakdown = build_demand_breakdown(result)
    gwh_per_kwh = WEIGHT / 1e6
    assert np.isclose(breakdown['Net change'], 1500.0 * gwh_per_kwh)
    assert np.isclose(breakdown['Net heating'], 2250.0 * gwh_per_kwh)
    assert np.isclose(
        breakdown['Baseline electricity'] + breakdown['Net heating']
        + breakdown['Net cooling'] + breakdown['Net hot water']
        + breakdown['Other end uses'],
        breakdown['Post-retrofit electricity'])
    assert np.isclose(
        breakdown['Net heating, Natural Gas homes']
        + breakdown['Net heating, Electricity homes'],
        breakdown['Net heating'])
    assert np.isclose(breakdown['Site energy change (%)'], -25.0)
