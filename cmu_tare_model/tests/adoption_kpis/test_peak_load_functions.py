"""Tests for cmu_tare_model.adoption_kpis.peak_load_functions module.

Tests all four public functions using synthetic data — no AWS/BSQ
connection required.
"""

import numpy as np
import pandas as pd
import pytest

from cmu_tare_model.grid_impact.peak_load_functions import (
    BLDG_ID_COL,
    BSQ_ELEC_COL,
    compute_county_scenario_profile,
    extract_adopter_ids,
    find_adoption_column,
    gisjoin_to_fips,
)


# ============================================================================
# gisjoin_to_fips
# ============================================================================


class TestGisjoinToFips:
    """Tests for gisjoin_to_fips()."""

    def test_standard_allegheny(self):
        assert gisjoin_to_fips("G4200030") == "42003"

    def test_standard_los_angeles(self):
        assert gisjoin_to_fips("G0600370") == "06037"

    def test_leading_zero_state(self):
        # Alabama (01) county 001
        assert gisjoin_to_fips("G0100010") == "01001"

    def test_seven_character_minimum(self):
        assert gisjoin_to_fips("G420003") == "42003"

    def test_eight_character_standard(self):
        assert gisjoin_to_fips("G4200030") == "42003"

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            gisjoin_to_fips("G42")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="too short"):
            gisjoin_to_fips("")

    def test_six_chars_raises(self):
        with pytest.raises(ValueError, match="too short"):
            gisjoin_to_fips("G42003")


# ============================================================================
# find_adoption_column
# ============================================================================


class TestFindAdoptionColumn:
    """Tests for find_adoption_column()."""

    @pytest.fixture()
    def expected_col_name(self):
        """The column name that find_adoption_column should produce for defaults."""
        return (
            "ref2025_mp3_heatingLCC_coolingLCC_unsub_econ_adopter_fixed_base"
        )

    @pytest.fixture()
    def df_with_adoption(self, expected_col_name):
        """DataFrame containing the expected economic-adopter column."""
        return pd.DataFrame({expected_col_name: [1.0, 0.0]})

    def test_exact_match(self, df_with_adoption, expected_col_name):
        result = find_adoption_column(df_with_adoption, mp=3, cost_scenario="v4MID")
        assert result == expected_col_name

    def test_missing_column_with_candidates(self):
        df = pd.DataFrame({"some_econ_adopter_col": [1.0]})
        with pytest.raises(KeyError, match="Candidates containing 'econ_adopter'"):
            find_adoption_column(df, mp=3, cost_scenario="v4MID")

    def test_missing_column_no_candidates(self):
        df = pd.DataFrame({"unrelated_col": [1]})
        with pytest.raises(KeyError, match="no columns containing 'econ_adopter'"):
            find_adoption_column(df, mp=3, cost_scenario="v4MID")

    def test_custom_mp_and_cost_scenario(self):
        from cmu_tare_model.utils.column_names import create_adoption_col

        expected = create_adoption_col(
            scenario_prefix="ref2025_mp4_",
            npv_case="heatingLCC_coolingLCC_unsub",
            method_suffix="_fixed_base",
        )
        df = pd.DataFrame({expected: [1.0, 0.0, 1.0]})
        result = find_adoption_column(df, mp=4, cost_scenario="v4HIGH")
        assert result == expected


# ============================================================================
# extract_adopter_ids
# ============================================================================


class TestExtractAdopterIds:
    """Tests for extract_adopter_ids()."""

    @pytest.fixture()
    def tare_df(self):
        """Synthetic TARE DataFrame with two counties and mixed tiers."""
        data = {
            "county": [
                "G4200030",  # Allegheny (42003)
                "G4200030",
                "G4200030",
                "G0600370",  # LA (06037)
                "G0600370",
            ],
            "adoption_tier": [
                "Tier 1: Feasible",
                "Tier 2: Feasible vs. Alternative",
                "Not Feasible",
                "Tier 1: Feasible",
                "Tier 1: Feasible",
            ],
        }
        return pd.DataFrame(data, index=[101, 102, 103, 201, 202])

    def test_county_keys(self, tare_df):
        result = extract_adopter_ids(tare_df, "adoption_tier")
        assert set(result.keys()) == {"42003", "06037"}

    def test_tier_1_ids(self, tare_df):
        result = extract_adopter_ids(tare_df, "adoption_tier")
        assert result["42003"]["tier1"] == [101]
        assert result["06037"]["tier1"] == [201, 202]

    def test_tier_2_ids(self, tare_df):
        result = extract_adopter_ids(tare_df, "adoption_tier")
        assert result["42003"]["tier2"] == [102]
        assert result["06037"]["tier2"] == []

    def test_constrained_is_tier1_plus_tier2(self, tare_df):
        result = extract_adopter_ids(tare_df, "adoption_tier")
        assert result["42003"]["constrained"] == [101, 102]

    def test_all_filtered_includes_all(self, tare_df):
        result = extract_adopter_ids(tare_df, "adoption_tier")
        assert sorted(result["42003"]["all_filtered"]) == [101, 102, 103]

    def test_in_county_column_fallback(self, tare_df):
        """Should work with 'in.county' column name too."""
        tare_df = tare_df.rename(columns={"county": "in.county"})
        result = extract_adopter_ids(tare_df, "adoption_tier")
        assert "42003" in result

    def test_missing_county_column_raises(self, tare_df):
        tare_df = tare_df.rename(columns={"county": "region"})
        with pytest.raises(KeyError, match="Neither 'county' nor 'in.county'"):
            extract_adopter_ids(tare_df, "adoption_tier")

    def test_empty_county(self):
        """County with no buildings of any tier still gets entry."""
        df = pd.DataFrame(
            {
                "county": ["G4200030"],
                "adoption_tier": ["Not Feasible"],
            },
            index=[999],
        )
        result = extract_adopter_ids(df, "adoption_tier")
        assert result["42003"]["tier1"] == []
        assert result["42003"]["tier2"] == []
        assert result["42003"]["constrained"] == []
        assert result["42003"]["all_filtered"] == [999]


# ============================================================================
# compute_county_scenario_profile
# ============================================================================


def _make_hourly_df(
    bldg_ids: list[int],
    kwh_col: str,
    value: float,
    n_hours: int = 8760,
) -> pd.DataFrame:
    """Helper: create a synthetic hourly DataFrame for one county.

    Every building gets the same constant value for every hour,
    making the expected aggregation straightforward to compute.
    """
    rows = []
    for bid in bldg_ids:
        for h in range(n_hours):
            rows.append({BLDG_ID_COL: bid, "hour": h, kwh_col: value})
    return pd.DataFrame(rows)


class TestComputeCountyScenarioProfile:
    """Tests for compute_county_scenario_profile()."""

    @pytest.fixture()
    def baseline_df(self):
        """3 buildings, constant 100 kWh per hour."""
        return _make_hourly_df([1, 2, 3], "baseline_kwh", 100.0)

    @pytest.fixture()
    def upgrade_df(self):
        """2 buildings (1, 2) with upgrade data at 80 kWh per hour."""
        return _make_hourly_df([1, 2], "retrofit_kwh", 80.0)

    def test_profile_length(self, baseline_df, upgrade_df):
        df_profile, _ = compute_county_scenario_profile(
            baseline_df, upgrade_df, adopter_bldg_ids=[1, 2]
        )
        assert len(df_profile) == 8760

    def test_profile_columns(self, baseline_df, upgrade_df):
        df_profile, _ = compute_county_scenario_profile(
            baseline_df, upgrade_df, adopter_bldg_ids=[1]
        )
        assert list(df_profile.columns) == [
            "hour", "baseline_mw", "scenario_mw", "delta_mw"
        ]

    def test_baseline_aggregation(self, baseline_df, upgrade_df):
        """3 buildings × 100 kWh / 1000 = 0.3 MW baseline."""
        df_profile, _ = compute_county_scenario_profile(
            baseline_df, upgrade_df, adopter_bldg_ids=[1]
        )
        expected_baseline_mw = (3 * 100.0) / 1000.0
        assert np.isclose(df_profile["baseline_mw"].iloc[0], expected_baseline_mw)

    def test_scenario_with_one_adopter(self, baseline_df, upgrade_df):
        """1 adopter switches 100→80, 2 non-adopters stay at 100. Total = 280/1000."""
        df_profile, _ = compute_county_scenario_profile(
            baseline_df, upgrade_df, adopter_bldg_ids=[1]
        )
        expected_scenario_mw = (80.0 + 100.0 + 100.0) / 1000.0
        assert np.isclose(df_profile["scenario_mw"].iloc[0], expected_scenario_mw)

    def test_scenario_with_two_adopters(self, baseline_df, upgrade_df):
        """2 adopters switch 100→80, 1 non-adopter stays at 100. Total = 260/1000."""
        df_profile, _ = compute_county_scenario_profile(
            baseline_df, upgrade_df, adopter_bldg_ids=[1, 2]
        )
        expected_scenario_mw = (80.0 + 80.0 + 100.0) / 1000.0
        assert np.isclose(df_profile["scenario_mw"].iloc[0], expected_scenario_mw)

    def test_delta_is_scenario_minus_baseline(self, baseline_df, upgrade_df):
        df_profile, _ = compute_county_scenario_profile(
            baseline_df, upgrade_df, adopter_bldg_ids=[1, 2]
        )
        expected_delta = ((80 + 80 + 100) - (100 + 100 + 100)) / 1000.0
        assert np.isclose(df_profile["delta_mw"].iloc[0], expected_delta)

    def test_peak_dict_keys(self, baseline_df, upgrade_df):
        _, peak_dict = compute_county_scenario_profile(
            baseline_df, upgrade_df, adopter_bldg_ids=[1]
        )
        expected_keys = {
            "peak_hour_baseline",
            "peak_hour_scenario",
            "baseline_peak_mw",
            "scenario_peak_mw",
            "delta_mw",
            "n_adopters",
            "n_total_buildings",
        }
        assert set(peak_dict.keys()) == expected_keys

    def test_n_adopters(self, baseline_df, upgrade_df):
        _, peak_dict = compute_county_scenario_profile(
            baseline_df, upgrade_df, adopter_bldg_ids=[1, 2]
        )
        assert peak_dict["n_adopters"] == 2

    def test_n_total_buildings(self, baseline_df, upgrade_df):
        _, peak_dict = compute_county_scenario_profile(
            baseline_df, upgrade_df, adopter_bldg_ids=[1]
        )
        assert peak_dict["n_total_buildings"] == 3

    def test_adopter_without_upgrade_falls_back_to_baseline(
        self, baseline_df, upgrade_df, capsys
    ):
        """Building 3 is listed as adopter but has no upgrade data.
        It should fall back to baseline and a warning should be printed."""
        df_profile, peak_dict = compute_county_scenario_profile(
            baseline_df, upgrade_df, adopter_bldg_ids=[1, 3]
        )
        # bldg 1: 80 (upgrade), bldg 2: 100 (non-adopter), bldg 3: 100 (fallback)
        expected_scenario_mw = (80.0 + 100.0 + 100.0) / 1000.0
        assert np.isclose(df_profile["scenario_mw"].iloc[0], expected_scenario_mw)
        # Only bldg 1 is an effective adopter
        assert peak_dict["n_adopters"] == 1
        captured = capsys.readouterr()
        assert "have no upgrade data" in captured.out

    def test_no_adopters(self, baseline_df, upgrade_df):
        """With no adopters, scenario equals baseline."""
        df_profile, peak_dict = compute_county_scenario_profile(
            baseline_df, upgrade_df, adopter_bldg_ids=[]
        )
        assert np.isclose(
            df_profile["baseline_mw"].iloc[0], df_profile["scenario_mw"].iloc[0]
        )
        assert np.isclose(df_profile["delta_mw"].iloc[0], 0.0)
        assert peak_dict["n_adopters"] == 0

    def test_wrong_hour_count_raises(self):
        """Profile with != 8760 hours should raise ValueError."""
        df_b = _make_hourly_df([1], "baseline_kwh", 100.0, n_hours=24)
        df_u = _make_hourly_df([1], "retrofit_kwh", 80.0, n_hours=24)
        with pytest.raises(ValueError, match="Expected 8,760"):
            compute_county_scenario_profile(df_b, df_u, adopter_bldg_ids=[1])
