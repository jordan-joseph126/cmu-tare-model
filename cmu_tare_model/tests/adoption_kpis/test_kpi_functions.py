"""Tests for cmu_tare_model.adoption_kpis.kpi_functions module.

Tests all public functions using synthetic data — no EUSS files or
AWS connections required.  File-loading functions are tested with
mocked I/O.
"""

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cmu_tare_model.adoption_kpis.kpi_functions import (
    BTU_PER_CF_NATURAL_GAS,
    BTU_PER_KWH,
    COP_BENCHMARK_RANGES,
    DWELLING_UNIT_WEIGHT,
    HEATING_FUEL_COLS,
    HEATING_LOAD_COL,
    HP_BACKUP_ELEC_COL,
    HP_FANS_PUMPS_COL,
    KBTU_PER_KWH,
    KWH_PER_MMBTU,
    NG_CONVERSION_FACTOR,
    STATE_NAMES,
    aggregate_demand_by_state,
    calculate_price_ratios,
    compute_breakeven_cop,
    compute_scenario_demand,
    compute_spark_gap_metrics,
    compute_thermal_cop,
    compute_thermal_cop_by_state,
    iecc_to_cz_group,
    load_euss_baseline,
    load_euss_upgrade,
    mp_to_upgrade,
)


# ============================================================================
# mp_to_upgrade
# ============================================================================


class TestMpToUpgrade:
    def test_single_digit(self):
        assert mp_to_upgrade(4) == "upgrade04"

    def test_double_digit(self):
        assert mp_to_upgrade(10) == "upgrade10"

    def test_zero(self):
        assert mp_to_upgrade(0) == "upgrade00"


# ============================================================================
# iecc_to_cz_group
# ============================================================================


class TestIeccToCzGroup:
    @pytest.mark.parametrize(
        "zone,expected",
        [
            ("1A", "1-3"),
            ("2B", "1-3"),
            ("3C", "1-3"),
            ("4A", "4-5"),
            ("5B", "4-5"),
            ("6A", "6-7"),
            ("7A", "6-7"),
        ],
    )
    def test_valid_zones(self, zone, expected):
        assert iecc_to_cz_group(zone) == expected

    def test_none_returns_unknown(self):
        assert iecc_to_cz_group(None) == "unknown"

    def test_nan_returns_unknown(self):
        assert iecc_to_cz_group(float("nan")) == "unknown"

    def test_invalid_prefix_returns_unknown(self):
        assert iecc_to_cz_group("XY") == "unknown"

    def test_zone_8_raises(self):
        with pytest.raises(ValueError, match="outside 1-7"):
            iecc_to_cz_group("8A")


# ============================================================================
# Constants
# ============================================================================


class TestConstants:
    def test_ng_conversion_factor(self):
        expected = BTU_PER_KWH / (1000 * BTU_PER_CF_NATURAL_GAS)
        assert np.isclose(NG_CONVERSION_FACTOR, expected)

    def test_dwelling_unit_weight(self):
        assert DWELLING_UNIT_WEIGHT == 242

    def test_kwh_per_mmbtu(self):
        assert np.isclose(KWH_PER_MMBTU, 293.07107, rtol=1e-4)

    def test_kbtu_per_kwh(self):
        assert np.isclose(KBTU_PER_KWH, 3.412, rtol=1e-4)

    def test_state_names_count(self):
        # 50 states + DC
        assert len(STATE_NAMES) == 51

    def test_cop_benchmark_range_keys(self):
        assert set(COP_BENCHMARK_RANGES.keys()) == {"1-3", "4-5", "6-7"}


# ============================================================================
# load_euss_baseline (mocked I/O)
# ============================================================================


class TestLoadEussBaseline:
    @pytest.fixture()
    def mock_csv_data(self):
        return pd.DataFrame(
            {
                "bldg_id": [1, 2, 3, 4],
                "in.vacancy_status": [
                    "Occupied", "Occupied", "Vacant", "Occupied"
                ],
                "in.geometry_building_type_recs": [
                    "Single-Family Detached",
                    "Single-Family Attached",
                    "Single-Family Detached",
                    "Multi-Family with 5+ Units",
                ],
            }
        )

    @patch("cmu_tare_model.adoption_kpis.kpi_functions.pd.read_csv")
    def test_filters_vacant(self, mock_read, mock_csv_data):
        mock_read.return_value = mock_csv_data.set_index("bldg_id")
        df = load_euss_baseline()
        # bldg 3 is Vacant
        assert 3 not in df.index

    @patch("cmu_tare_model.adoption_kpis.kpi_functions.pd.read_csv")
    def test_filters_non_sf(self, mock_read, mock_csv_data):
        mock_read.return_value = mock_csv_data.set_index("bldg_id")
        df = load_euss_baseline()
        # bldg 4 is Multi-Family 5+
        assert 4 not in df.index

    @patch("cmu_tare_model.adoption_kpis.kpi_functions.pd.read_csv")
    def test_keeps_valid_sf(self, mock_read, mock_csv_data):
        mock_read.return_value = mock_csv_data.set_index("bldg_id")
        df = load_euss_baseline()
        assert set(df.index) == {1, 2}


# ============================================================================
# load_euss_upgrade (mocked I/O)
# ============================================================================


class TestLoadEussUpgrade:
    @pytest.fixture()
    def mock_csv_data(self):
        return pd.DataFrame(
            {
                "bldg_id": [1, 2, 3],
                "in.vacancy_status": ["Occupied", "Occupied", "Occupied"],
                "in.geometry_building_type_recs": [
                    "Single-Family Detached",
                    "Single-Family Detached",
                    "Single-Family Detached",
                ],
                "applicability": [True, False, True],
            }
        )

    @patch("cmu_tare_model.adoption_kpis.kpi_functions.pd.read_csv")
    def test_filters_non_applicable(self, mock_read, mock_csv_data):
        mock_read.return_value = mock_csv_data.set_index("bldg_id")
        df = load_euss_upgrade("upgrade04")
        assert 2 not in df.index
        assert set(df.index) == {1, 3}


# ============================================================================
# calculate_price_ratios
# ============================================================================


class TestCalculatePriceRatios:
    @pytest.fixture()
    def fuel_prices_csv(self, tmp_path):
        """Create a minimal fuel_prices CSV file."""
        data = pd.DataFrame(
            {
                "fuel_type": [
                    "electricity", "electricity", "electricity",
                    "naturalGas", "naturalGas", "naturalGas",
                ],
                "state_region": ["PA", "CA", "National", "PA", "CA", "National"],
                "2024_nominal_unit_price": [
                    15.0, 25.0, 20.0,  # electricity: cents/kWh
                    12.0, 18.0, 15.0,  # NG: $/1000cf
                ],
            }
        )
        fpath = tmp_path / "fuel_prices.csv"
        data.to_csv(fpath, index=False)
        return str(fpath)

    def test_output_columns(self, fuel_prices_csv):
        df = calculate_price_ratios(fuel_prices_csv)
        expected_cols = [
            "state", "state_name", "elec_price_kwh", "gas_price_kwh",
            "elec_price_mmbtu", "gas_price_mmbtu", "spark_gap",
        ]
        assert list(df.columns) == expected_cols

    def test_excludes_national(self, fuel_prices_csv):
        df = calculate_price_ratios(fuel_prices_csv)
        assert "National" not in df["state"].values

    def test_state_count(self, fuel_prices_csv):
        df = calculate_price_ratios(fuel_prices_csv)
        assert len(df) == 2  # PA + CA

    def test_spark_gap_positive(self, fuel_prices_csv):
        df = calculate_price_ratios(fuel_prices_csv)
        assert (df["spark_gap"] > 0).all()

    def test_elec_price_conversion(self, fuel_prices_csv):
        df = calculate_price_ratios(fuel_prices_csv)
        pa = df[df["state"] == "PA"].iloc[0]
        # 15 cents/kWh → 0.15 $/kWh
        assert np.isclose(pa["elec_price_kwh"], 0.15, atol=0.001)

    def test_missing_year_raises(self, fuel_prices_csv):
        with pytest.raises(KeyError, match="not found"):
            calculate_price_ratios(fuel_prices_csv, year=1999)

    def test_sorted_by_spark_gap_descending(self, fuel_prices_csv):
        df = calculate_price_ratios(fuel_prices_csv)
        assert list(df["spark_gap"]) == sorted(df["spark_gap"], reverse=True)


# ============================================================================
# compute_thermal_cop
# ============================================================================


def _make_euss_pair(
    n_homes: int = 5,
    heating_load_kbtu: float = 50000.0,
    gas_kwh: float = 15000.0,
    hp_elec_kwh: float = 5000.0,
    hp_bkup_kwh: float = 500.0,
    hp_fans_kwh: float = 200.0,
    fuel: str = "Natural Gas",
    state: str = "PA",
    cz: str = "5A",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build synthetic baseline + upgrade DataFrames for COP/demand tests."""
    bldg_ids = list(range(1, n_homes + 1))
    baseline = pd.DataFrame(
        {
            "in.state": [state] * n_homes,
            "in.heating_fuel": [fuel] * n_homes,
            "in.vacancy_status": ["Occupied"] * n_homes,
            "in.geometry_building_type_recs": ["Single-Family Detached"] * n_homes,
            "in.ashrae_iecc_climate_zone_2004": [cz] * n_homes,
            "in.county": ["G4200030"] * n_homes,
            "weight": [242] * n_homes,
            HEATING_LOAD_COL: [heating_load_kbtu] * n_homes,
            "out.natural_gas.heating.energy_consumption.kwh": [gas_kwh] * n_homes,
            "out.electricity.heating.energy_consumption.kwh": [hp_elec_kwh] * n_homes,
            "out.fuel_oil.heating.energy_consumption.kwh": [0] * n_homes,
            "out.propane.heating.energy_consumption.kwh": [0] * n_homes,
            HP_BACKUP_ELEC_COL: [0] * n_homes,
            HP_FANS_PUMPS_COL: [0] * n_homes,
        },
        index=bldg_ids,
    )
    baseline.index.name = "bldg_id"

    upgrade = pd.DataFrame(
        {
            "in.state": [state] * n_homes,
            "in.heating_fuel": [fuel] * n_homes,
            "in.vacancy_status": ["Occupied"] * n_homes,
            "in.geometry_building_type_recs": ["Single-Family Detached"] * n_homes,
            "in.ashrae_iecc_climate_zone_2004": [cz] * n_homes,
            "in.county": ["G4200030"] * n_homes,
            "weight": [242] * n_homes,
            "applicability": [True] * n_homes,
            HEATING_LOAD_COL: [heating_load_kbtu] * n_homes,
            "out.electricity.heating.energy_consumption.kwh": [hp_elec_kwh] * n_homes,
            HP_BACKUP_ELEC_COL: [hp_bkup_kwh] * n_homes,
            HP_FANS_PUMPS_COL: [hp_fans_kwh] * n_homes,
        },
        index=bldg_ids,
    )
    upgrade.index.name = "bldg_id"

    return baseline, upgrade


class TestComputeThermalCop:
    @pytest.fixture()
    def euss_pair(self):
        return _make_euss_pair()

    def test_output_columns(self, euss_pair):
        df_b, df_u = euss_pair
        result = compute_thermal_cop(df_b, df_u)
        expected = [
            "state", "Q_upgrade_total_kbtu", "hp_total_elec_kbtu",
            "hp_fans_pumps_total_kbtu", "Q_baseline_total_kbtu",
            "gas_consumed_total_kbtu", "home_count",
            "thermal_cop", "baseline_afue", "fans_pumps_pct",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_cop_value(self, euss_pair):
        """COP = Q_delivered / (elec + bkup + fans) in kBtu."""
        df_b, df_u = euss_pair
        result = compute_thermal_cop(df_b, df_u)
        # Per home: Q = 50000 kBtu, HP total = (5000 + 500 + 200) * 3.412 kBtu
        hp_total_kbtu = (5000 + 500 + 200) * KBTU_PER_KWH
        expected_cop = 50000.0 / hp_total_kbtu
        assert np.isclose(result["thermal_cop"].iloc[0], expected_cop, rtol=1e-3)

    def test_afue_value(self, euss_pair):
        """AFUE = Q_delivered / gas_consumed in kBtu."""
        df_b, df_u = euss_pair
        result = compute_thermal_cop(df_b, df_u)
        gas_kbtu = 15000.0 * KBTU_PER_KWH
        expected_afue = 50000.0 / gas_kbtu
        assert np.isclose(result["baseline_afue"].iloc[0], expected_afue, rtol=1e-3)

    def test_home_count(self, euss_pair):
        df_b, df_u = euss_pair
        result = compute_thermal_cop(df_b, df_u)
        assert result["home_count"].iloc[0] == 5

    def test_fuel_filter_excludes(self):
        df_b, df_u = _make_euss_pair(fuel="Fuel Oil")
        result = compute_thermal_cop(df_b, df_u, fuel_filter="Natural Gas")
        assert len(result) == 0

    def test_fuel_filter_none_includes_all(self):
        df_b, df_u = _make_euss_pair(fuel="Fuel Oil")
        result = compute_thermal_cop(df_b, df_u, fuel_filter=None)
        assert len(result) == 1

    def test_require_baseline_heating_excludes_zero(self):
        df_b, df_u = _make_euss_pair(heating_load_kbtu=0.0)
        result = compute_thermal_cop(df_b, df_u, require_baseline_heating=True)
        assert len(result) == 0

    def test_require_baseline_heating_false_includes_zero(self):
        df_b, df_u = _make_euss_pair(heating_load_kbtu=0.0)
        result = compute_thermal_cop(
            df_b, df_u, require_baseline_heating=False
        )
        # COP is 0/nonzero = 0 (or np.nan) — just check the row exists
        assert len(result) == 1

    def test_group_by_cz_group(self):
        df_b, df_u = _make_euss_pair(cz="4A")
        result = compute_thermal_cop(df_b, df_u, group_cols=["cz_group"])
        assert "cz_group" in result.columns
        assert result["cz_group"].iloc[0] == "4-5"

    def test_missing_column_raises(self):
        df_b, df_u = _make_euss_pair()
        df_b = df_b.drop(columns=[HEATING_LOAD_COL])
        with pytest.raises(KeyError, match="Missing column"):
            compute_thermal_cop(df_b, df_u)


class TestComputeThermalCopByState:
    def test_delegates_to_compute_thermal_cop(self):
        df_b, df_u = _make_euss_pair()
        result = compute_thermal_cop_by_state(df_b, df_u)
        assert "state" in result.columns
        assert len(result) == 1


# ============================================================================
# compute_breakeven_cop
# ============================================================================


class TestComputeBreakevenCop:
    @pytest.fixture()
    def price_df(self):
        return pd.DataFrame(
            {
                "state": ["PA", "CA"],
                "state_name": ["Pennsylvania", "California"],
                "spark_gap": [3.5, 5.0],
            }
        )

    def test_default_afue_columns(self, price_df):
        result = compute_breakeven_cop(price_df)
        for afue_pct in [80, 90, 95, 100]:
            assert f"breakeven_cop_{afue_pct}" in result.columns

    def test_breakeven_formula(self, price_df):
        result = compute_breakeven_cop(price_df)
        pa = result[result["state"] == "PA"].iloc[0]
        # breakeven_cop_90 = spark_gap * 0.90 = 3.5 * 0.9 = 3.15
        assert np.isclose(pa["breakeven_cop_90"], 3.15, atol=0.01)

    def test_custom_afue_scenarios(self, price_df):
        result = compute_breakeven_cop(price_df, afue_scenarios=[0.70])
        assert "breakeven_cop_70" in result.columns
        assert "breakeven_cop_80" not in result.columns

    def test_state_name_preserved(self, price_df):
        result = compute_breakeven_cop(price_df)
        assert "state_name" in result.columns


# ============================================================================
# compute_spark_gap_metrics
# ============================================================================


class TestComputeSparkGapMetrics:
    @pytest.fixture()
    def price_df(self):
        return pd.DataFrame(
            {
                "state": ["PA"],
                "state_name": ["Pennsylvania"],
                "elec_price_kwh": [0.15],
                "gas_price_kwh": [0.04],
                "spark_gap": [3.5],
            }
        )

    @pytest.fixture()
    def cop_df(self):
        return pd.DataFrame(
            {
                "state": ["PA"],
                "thermal_cop": [2.5],
                "baseline_afue": [0.90],
                "fans_pumps_pct": [5.0],
                "home_count": [100],
            }
        )

    def test_merged_output(self, price_df, cop_df):
        result = compute_spark_gap_metrics(price_df, cop_df)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["state"] == "PA"
        assert np.isclose(row["thermal_cop"], 2.5)
        assert np.isclose(row["spark_gap"], 3.5)

    def test_inline_breakeven_when_none(self, price_df, cop_df):
        result = compute_spark_gap_metrics(price_df, cop_df, df_breakeven=None)
        assert "breakeven_cop_90" in result.columns
        # 3.5 * 0.90 = 3.15
        assert np.isclose(result["breakeven_cop_90"].iloc[0], 3.15, atol=0.01)

    def test_explicit_breakeven_merge(self, price_df, cop_df):
        be = pd.DataFrame(
            {"state": ["PA"], "breakeven_cop_90": [3.20]}
        )
        result = compute_spark_gap_metrics(price_df, cop_df, df_breakeven=be)
        assert np.isclose(result["breakeven_cop_90"].iloc[0], 3.20)


# ============================================================================
# compute_scenario_demand
# ============================================================================


class TestComputeScenarioDemand:
    @pytest.fixture()
    def euss_pair(self):
        return _make_euss_pair(
            n_homes=3,
            hp_elec_kwh=2000.0,
            hp_bkup_kwh=300.0,
            hp_fans_kwh=100.0,
            gas_kwh=5000.0,
        )

    def test_output_columns(self, euss_pair):
        df_b, df_u = euss_pair
        result = compute_scenario_demand(df_b, df_u)
        for col in [
            "baseline_electric_kwh",
            "retrofit_electric_kwh",
            "elec_demand_change_kwh",
            "site_energy_change_kwh",
            "weighted_elec_demand_change_kwh",
        ]:
            assert col in result.columns, f"Missing: {col}"

    def test_elec_demand_change(self, euss_pair):
        df_b, df_u = euss_pair
        result = compute_scenario_demand(df_b, df_u)
        # baseline electric: 2000 kWh
        # retrofit electric: 2000 + 300 + 100 = 2400 kWh
        # change: +400 kWh per home
        assert np.allclose(result["elec_demand_change_kwh"], 400.0)

    def test_site_energy_change(self, euss_pair):
        df_b, df_u = euss_pair
        result = compute_scenario_demand(df_b, df_u)
        # baseline total = elec (2000) + gas (5000) + oil (0) + propane (0) = 7000
        # retrofit total = 2400 kWh
        # site change = 2400 - 7000 = -4600
        assert np.allclose(result["site_energy_change_kwh"], -4600.0)

    def test_weighted_values(self, euss_pair):
        df_b, df_u = euss_pair
        result = compute_scenario_demand(df_b, df_u)
        assert np.allclose(
            result["weighted_elec_demand_change_kwh"],
            result["elec_demand_change_kwh"] * 242,
        )

    def test_fuel_filter(self, euss_pair):
        df_b, df_u = euss_pair
        result = compute_scenario_demand(df_b, df_u, fuel_filter="Propane")
        assert len(result) == 0

    def test_fuel_filter_match(self, euss_pair):
        df_b, df_u = euss_pair
        result = compute_scenario_demand(df_b, df_u, fuel_filter="Natural Gas")
        assert len(result) == 3


# ============================================================================
# aggregate_demand_by_state
# ============================================================================


class TestAggregateDemandByState:
    @pytest.fixture()
    def demand_df(self):
        return pd.DataFrame(
            {
                "in.state": ["PA", "PA", "CA"],
                "weight": [242, 242, 242],
                "baseline_electric_kwh": [2000, 3000, 4000],
                "baseline_heating_total_kwh": [7000, 8000, 9000],
                "retrofit_electric_kwh": [2400, 3400, 4400],
                "elec_demand_change_kwh": [400, 400, 400],
                "site_energy_change_kwh": [-4600, -4600, -4600],
                "weighted_baseline_electric_kwh": [
                    2000 * 242, 3000 * 242, 4000 * 242
                ],
                "weighted_baseline_heating_total_kwh": [
                    7000 * 242, 8000 * 242, 9000 * 242
                ],
                "weighted_retrofit_electric_kwh": [
                    2400 * 242, 3400 * 242, 4400 * 242
                ],
                "weighted_elec_demand_change_kwh": [
                    400 * 242, 400 * 242, 400 * 242
                ],
                "weighted_site_energy_change_kwh": [
                    -4600 * 242, -4600 * 242, -4600 * 242
                ],
            }
        )

    def test_state_count(self, demand_df):
        result = aggregate_demand_by_state(demand_df)
        assert len(result) == 2  # PA + CA

    def test_pa_home_count(self, demand_df):
        result = aggregate_demand_by_state(demand_df)
        pa = result[result["state"] == "PA"].iloc[0]
        assert pa["home_count"] == 2

    def test_elec_change_gwh(self, demand_df):
        result = aggregate_demand_by_state(demand_df)
        pa = result[result["state"] == "PA"].iloc[0]
        expected = (400 * 242 * 2) / 1e6
        assert np.isclose(pa["elec_change_gwh"], expected, atol=0.01)

    def test_pct_columns_present(self, demand_df):
        result = aggregate_demand_by_state(demand_df)
        assert "pct_elec_demand_change" in result.columns
        assert "pct_site_energy_change" in result.columns

    def test_sorted_by_elec_change_descending(self, demand_df):
        result = aggregate_demand_by_state(demand_df)
        assert list(result["elec_change_gwh"]) == sorted(
            result["elec_change_gwh"], reverse=True
        )

    def test_accounting_check_passes(self, demand_df):
        """No warning should be printed if groupby sums match."""
        result = aggregate_demand_by_state(demand_df, verbose=True)
        assert len(result) > 0
