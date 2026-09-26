"""Tests that compute_scenario_demand counts only the study sample
(adoption_kpis/demand.py).

The adoption_kpis package imports its geospatial module on load, so these tests
need geopandas and are skipped where it is not installed.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("geopandas")

from cmu_tare_model.adoption_kpis.data_loading import COUNTY_COL, ELEC_TOTAL_COL
from cmu_tare_model.adoption_kpis.demand import compute_scenario_demand


@pytest.fixture
def euss_pair():
    """Five homes; each retrofit adds 100 x bldg_id kWh of electricity."""
    bldg_ids = pd.Index([1, 2, 3, 4, 5], name='bldg_id')
    df_baseline = pd.DataFrame({
        'in.state': ['PA'] * 5,
        COUNTY_COL: ['G4200030'] * 5,
        'in.heating_fuel': ['Natural Gas'] * 5,
        'weight': [242.131013] * 5,
        ELEC_TOTAL_COL: [1000.0] * 5,
    }, index=bldg_ids)
    df_upgrade = pd.DataFrame({
        ELEC_TOTAL_COL: 1000.0 + 100.0 * bldg_ids.to_numpy(),
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
