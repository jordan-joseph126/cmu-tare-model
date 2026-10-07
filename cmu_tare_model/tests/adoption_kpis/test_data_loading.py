"""Tests for adoption_kpis/data_loading.py and its use in demand.py.

The KPI loaders and column names follow this run's ResStock release: every
column name comes from the column map, and both loaders apply the model run's
own scope filters through load_and_filter_upgrade. The column names are fixed
when the module is imported, so the 2025.1 cases run in a fresh Python
process with TARE_RESSTOCK_RELEASE set. All data is made up in the test.
"""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("geopandas")

from cmu_tare_model.adoption_kpis import data_loading

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_in_release(release: str, code: str) -> str:
    """Runs code in a new Python process for one release; returns its output.

    Args:
        release: Value for TARE_RESSTOCK_RELEASE.
        code: Python source to run.

    Returns:
        The process's standard output.
    """
    env = dict(os.environ, TARE_RESSTOCK_RELEASE=release)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + [p for p in [env.get("PYTHONPATH")] if p])
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], env=env,
        capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


# -- column names -------------------------------------------------------------

def test_no_resstock_column_name_is_typed_into_the_module():
    source = Path(data_loading.__file__).read_text(encoding="utf-8")
    code_lines = [
        line for line in source.splitlines()
        if not line.strip().startswith("#")]
    typed_names = re.findall(
        r"""["'](?:in|out|upgrade|calc)\.[^"']*["']|["']applicability["']""",
        "\n".join(code_lines))
    assert typed_names == []


def test_2022_names_are_unchanged():
    # The default release; these are the names the module used to type in.
    assert data_loading.ELEC_TOTAL_COL == (
        "out.electricity.total.energy_consumption.kwh")
    assert data_loading.ELEC_TOTAL_SAVINGS_COL == (
        "out.electricity.total.energy_consumption.kwh.savings")
    assert data_loading.HEATING_ELEC_SAVINGS_COLS == [
        "out.electricity.heating.energy_consumption.kwh.savings",
        "out.electricity.heating_hp_bkup.energy_consumption.kwh.savings",
        "out.electricity.heating_fans_pumps.energy_consumption.kwh.savings",
    ]
    assert data_loading.COUNTY_COL == "in.county"
    assert data_loading.mp_to_upgrade(4) == "upgrade04"


def test_2025_names_follow_the_release():
    output = _run_in_release("2025.1", """
        from cmu_tare_model.adoption_kpis import data_loading as dl
        print(dl.ELEC_TOTAL_COL)
        print(dl.SITE_ENERGY_TOTAL_SAVINGS_COL)
        print('|'.join(dl.HEATING_ELEC_SAVINGS_COLS))
        print(dl.mp_to_upgrade(5))
    """)
    lines = output.strip().splitlines()[-4:]
    assert lines[0] == "out.electricity.total.energy_consumption..kwh"
    assert lines[1] == "out.site_energy.total.energy_savings..kwh"
    # Every electric part of heating, including the backup's own fans.
    assert lines[2].split("|") == [
        "out.electricity.heating.energy_savings..kwh",
        "out.electricity.heating_hp_bkup.energy_savings..kwh",
        "out.electricity.heating_hp_bkup_fa.energy_savings..kwh",
        "out.electricity.heating_fans_pumps.energy_savings..kwh",
    ]
    assert lines[3] == "upgrade5"


# -- loaders --------------------------------------------------------------------

@pytest.fixture
def recorded_loads(monkeypatch):
    """Replaces the model run's loader with one that records its calls."""
    from cmu_tare_model.energy_consumption_and_metadata import process_euss_data
    calls = []

    def fake_load_and_filter_upgrade(menu_mp, verbose=False, release=None, **_):
        calls.append((menu_mp, release))
        df = pd.DataFrame({"weight": [1.0]}, index=pd.Index([7], name="bldg_id"))
        return df, pd.DataFrame({"stage": ["load"], "rdu_count": [1]})

    monkeypatch.setattr(
        process_euss_data, "load_and_filter_upgrade", fake_load_and_filter_upgrade)
    return calls


def test_baseline_uses_the_model_run_loader(recorded_loads):
    df = data_loading.load_euss_baseline()
    assert recorded_loads == [(0, "2022.1.1")]
    assert list(df.index) == [7]


def test_upgrade_uses_the_model_run_loader(recorded_loads):
    data_loading.load_euss_upgrade(data_loading.mp_to_upgrade(4))
    assert recorded_loads == [(4, "2022.1.1")]


def test_upgrade_rejects_another_release_name(recorded_loads):
    with pytest.raises(ValueError, match="upgrade5"):
        data_loading.load_euss_upgrade("upgrade5")
    assert recorded_loads == []


def test_baseline_rejects_another_file(recorded_loads):
    with pytest.raises(ValueError, match="filename"):
        data_loading.load_euss_baseline("some_other_file.csv")


# -- demand on a 2025.1-named frame (G10) ---------------------------------------

def test_demand_runs_on_2025_column_names():
    # Two made-up homes in one county, named the way the 2025.1 files name
    # their columns. The demand code must find every column it reads.
    output = _run_in_release("2025.1", """
        import pandas as pd
        from cmu_tare_model.adoption_kpis import data_loading as dl
        from cmu_tare_model.adoption_kpis.demand import (
            aggregate_demand, compute_scenario_demand)

        ids = pd.Index([1, 2], name='bldg_id')
        weight = 253.90367272727272
        baseline = pd.DataFrame({
            dl.STATE_COL: ['PA', 'PA'],
            dl.COUNTY_COL: ['G4200030', 'G4200030'],
            dl.HEATING_FUEL_COL: ['Natural Gas', 'Electricity'],
            dl.DWELLING_UNIT_WEIGHT: [weight, weight],
            dl.ELEC_TOTAL_COL: [1000.0, 1000.0],
            dl.SITE_ENERGY_TOTAL_COL: [20000.0, 20000.0],
        }, index=ids)
        upgrade = pd.DataFrame({
            dl.ELEC_TOTAL_COL: [1500.0, 900.0],
            dl.SITE_ENERGY_TOTAL_COL: [15000.0, 19900.0],
            dl.ELEC_TOTAL_SAVINGS_COL: [-500.0, 100.0],
            dl.SITE_ENERGY_TOTAL_SAVINGS_COL: [5000.0, 100.0],
            dl.HOT_WATER_ELEC_SAVINGS_COL: [0.0, 0.0],
        }, index=ids)
        for column in dl.HEATING_ELEC_SAVINGS_COLS:
            upgrade[column] = [-125.0, 25.0]
        for column in dl.COOLING_ELEC_SAVINGS_COLS:
            upgrade[column] = [0.0, 0.0]

        df_demand = compute_scenario_demand(baseline, upgrade, ids)
        county = aggregate_demand(df_demand, geo_level='county')
        print(round(df_demand['heating_elec_change_kwh'].sum(), 3))
        print(round(float(county['home_count'].iloc[0]), 6))
        print(round(float(county['elec_change_gwh'].iloc[0] * 1e6 / weight), 3))
    """)
    lines = output.strip().splitlines()[-3:]
    # Four heating parts in 2025.1: -(4 x -125) - (4 x 25) = 400 kWh
    assert float(lines[0]) == pytest.approx(400.0)
    assert float(lines[1]) == pytest.approx(2 * 253.90367272727272)
    # GWh is rounded to 0.01 inside aggregate_demand; 400 kWh x weight is far
    # below that, so only check the sign and size loosely.
    assert float(lines[2]) == pytest.approx(400.0, abs=40.0)
