"""Tests for build_projected_consumption (projected_consumption.py)."""

import numpy as np
import pandas as pd
import pytest

from cmu_tare_model.constants import ANCHOR_YEAR

MODULE = 'cmu_tare_model.energy_consumption_and_metadata.projected_consumption'


@pytest.fixture
def df_baseline_homes(monkeypatch):
    """Three baseline homes, heating only. Homes 1 and 2 are in the calculation
    (natural gas heat and electric heat). Home 3 is outside the study sample, so
    its energy columns are blank, as df_enduse_refactored leaves them."""
    # Heating alone keeps the frame small.
    monkeypatch.setattr(f'{MODULE}.EQUIPMENT_SPECS', {'heating': 15})
    df_homes = pd.DataFrame({
        'bldg_id': [1, 2, 3],
        'census_division': ['Pacific', 'Mountain', 'Pacific'],
        'include_heating': [True, True, False],
        'include_sample': [True, True, False],
        'base_electricity_heating_consumption': [500.0, 9000.0, np.nan],
        'base_naturalGas_heating_consumption': [20000.0, 0.0, np.nan],
        'base_propane_heating_consumption': [0.0, 0.0, np.nan],
        'base_fuelOil_heating_consumption': [0.0, 0.0, np.nan],
    })
    return df_homes.set_index('bldg_id')


def test_home_outside_the_calculation_stays_blank(df_baseline_homes):
    """Covered homes get a number for every fuel and year. A home outside the
    calculation stays blank and does not stop the run."""
    from cmu_tare_model.energy_consumption_and_metadata.projected_consumption import (
        build_projected_consumption)
    df_table = build_projected_consumption(
        df_baseline_homes, menu_mp=0, verbose=False)

    assert df_table.loc[[1, 2]].notna().all().all()
    assert df_table.loc[3].isna().all()
    # The anchor year is unscaled, and an unused fuel is 0, not blank.
    assert df_table.loc[
        1, f'baseline_{ANCHOR_YEAR}_heating_consumption'] == 20500.0
    assert df_table.loc[
        2, f'baseline_{ANCHOR_YEAR}_heating_naturalGas_consumption'] == 0.0


def test_blank_in_covered_home_raises(df_baseline_homes):
    """A blank fuel in a home the calculation covers stops the run. It is not
    counted as zero use."""
    from cmu_tare_model.energy_consumption_and_metadata.projected_consumption import (
        build_projected_consumption)
    df_baseline_homes.loc[2, 'base_naturalGas_heating_consumption'] = np.nan
    with pytest.raises(ValueError, match='blank energy use'):
        build_projected_consumption(df_baseline_homes, menu_mp=0, verbose=False)
