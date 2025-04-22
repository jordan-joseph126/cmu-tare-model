# =====================================================================================================================================================================================
# test_calculate_lifetime_climate_impacts_sensitivity.py
# =====================================================================================================================================================================================
# Test cases for the calculate_lifetime_climate_impacts_sensitivity module.
#   calculate lifetime climate impacts (emissions and damages) based on various scenarios and equipment categories.
#
# WORKING AS OF 2025-04-03 @ 7:40 PM EST
#
# AUTHOR: JORDAN M. JOSEPH
# =====================================================================================================================================================================================

import pytest
import pandas as pd
# from pandas.testing import assert_frame_equal

# Import functions from the updated script
from cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity import (
    calculate_lifetime_climate_impacts,
    calculate_climate_emissions_and_damages,
)
from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import calculate_fossil_fuel_emissions
from cmu_tare_model.constants import MER_TYPES, EQUIPMENT_SPECS, SCC_ASSUMPTIONS

# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def sample_df():
    """
    Create a DataFrame that has *all* the needed baseline and measure-package
    consumption columns for each category and year up to its lifetime.
    """
    data = {
        'county_fips': ['12345'],
        'state': ['XX'],
        'year': [2023],
        'cambium_gea_region': ['Region1'],
        'census_division': ['National'],
        'gea_region': ['Region1'],
    }
    df = pd.DataFrame(data)

    # 1) Provide baseline columns for each category and fuel
    fuels = ['electricity', 'naturalGas', 'propane', 'fuelOil']
    for category in EQUIPMENT_SPECS.keys():
        for fuel in fuels:
            fuel_consumption_col = f"base_{fuel}_{category}_consumption"
            df[fuel_consumption_col] = 10.0  # dummy value
        
        consumption_col = f"baseline_{category}_consumption"
        df[consumption_col] = 15.0  # dummy value

    # 2) Provide measure-package columns for each year from 2024..(2024 + lifetime)
    #    This ensures we have mp8_<year>_<category>_consumption for each year.
    for category, lifetime in EQUIPMENT_SPECS.items():
        for year_offset in range(1, lifetime + 1):
            year_label = 2023 + year_offset  # e.g. 2024..2024 + lifetime
            col_name = f"mp8_{year_label}_{category}_consumption"
            df[col_name] = 12.0  # dummy value

    return df


@pytest.fixture
def dummy_define_scenario_settings():
    """
    Dummy replacement for define_scenario_params.
    Returns a function that, given menu_mp and policy_scenario, returns a tuple:
      (scenario_prefix, cambium_scenario,
       dummy_lookup_emissions_fossil_fuel,
       dummy_lookup_emissions_electricity_climate,
       dummy_lookup_emissions_electricity_health)

    The dummy lookups return trivial values for fossil fuel and electricity
    climate emissions, matching the dictionary shape expected in the original program.
    """
    def dummy(menu_mp, policy_scenario):
        if menu_mp == 0:
            scenario_prefix = "baseline_"
        elif policy_scenario == "No Inflation Reduction Act":
            scenario_prefix = f"preIRA_mp{menu_mp}_"
        elif policy_scenario == "AEO2023 Reference Case":
            scenario_prefix = f"iraRef_mp{menu_mp}_"
        else:
            raise ValueError("Invalid Policy Scenario!")

        cambium_scenario = "MidCase"

        # --------------------------------------------------------------------
        # 1) Provide emission factors for each year from 2024 to 2040.
        #    This fixes KeyError for 2025+ in the tests.
        # --------------------------------------------------------------------
        dummy_emission_factors_by_year = {
            year: {
                'lrmer_mt_per_kWh_co2e': 0.02,
                'srmer_mt_per_kWh_co2e': 0.03
            }
            for year in range(2024, 2041)  # 2024 ... 2040 inclusive
        }

        dummy_lookup_emissions_electricity_climate = {
            (cambium_scenario, "Region1"): dummy_emission_factors_by_year
        }

        # CHANGED: Use nested dict for fossil fuel lookups
        dummy_lookup_emissions_fossil_fuel = {
            'naturalGas': {'so2': 0.0, 'nox': 0.0, 'pm25': 0.0, 'co2e': 0.0},
            'propane':    {'so2': 0.0, 'nox': 0.0, 'pm25': 0.0, 'co2e': 0.0},
            'fuelOil':    {'so2': 0.0, 'nox': 0.0, 'pm25': 0.0, 'co2e': 0.0},
        }

        dummy_lookup_emissions_electricity_health = {}

        dummy_lookup_fuel_prices = {}

        return (
            scenario_prefix,
            cambium_scenario,
            dummy_lookup_emissions_fossil_fuel,
            dummy_lookup_emissions_electricity_climate,
            dummy_lookup_emissions_electricity_health,
            dummy_lookup_fuel_prices
        )

    return dummy


# =========================================================================
# Tests for calculate_climate_emissions_and_damages
# =========================================================================

def test_get_scc_value_correct_retrieval():
    """
    Verify that the get_scc_value function raises KeyError if the exact year
    is not found, and returns the correct value for exact matches.
    """
    dummy_lookup_climate_impact_scc = {
        "lower":   {2023: 10.0, 2024: 12.5, 2030: 15.0},
        "central": {2023: 50.0, 2024: 55.0, 2030: 60.0},
        "upper":   {2023: 100.0, 2024: 110.0, 2030: 120.0}
    }

    def get_scc_value(year_label: int, scc_assumption: str, scc_lookup: dict) -> float:
        """
        Retrieve the SCC value for the given year and assumption ('lower', 'central', 'upper').
        Raises KeyError if the year is not present in the lookup.
        """
        if year_label not in scc_lookup[scc_assumption]:
            raise KeyError(
                f"SCC value for year {year_label} with assumption '{scc_assumption}' not found."
            )
        return scc_lookup[scc_assumption][year_label]

    # 1) Exact year matches
    assert get_scc_value(2023, "lower", dummy_lookup_climate_impact_scc) == 10.0
    assert get_scc_value(2024, "central", dummy_lookup_climate_impact_scc) == 55.0
    assert get_scc_value(2024, "upper", dummy_lookup_climate_impact_scc) == 110.0

    # 2) Missing year => KeyError
    with pytest.raises(KeyError, match="SCC value for year 2040 with assumption 'lower' not found."):
        get_scc_value(2040, "lower", dummy_lookup_climate_impact_scc)


def test_calculate_climate_emissions_and_damages_success(sample_df, dummy_define_scenario_settings):
    """
    Verify that calculate_climate_emissions_and_damages returns expected keys
    for annual emissions/damages, given a valid DataFrame & dummy scenario.
    """
    category = 'heating'
    year_label = 2024
    adjusted_hdd_factor = pd.Series(1.0, index=sample_df.index)

    scenario_prefix, cambium_scenario, dummy_lookup_emissions_fossil_fuel, dummy_lookup_emissions_electricity_climate, _, _ = dummy_define_scenario_settings(0, "No Inflation Reduction Act")

    total_fossil_fuel_emissions = calculate_fossil_fuel_emissions(
        sample_df, category, adjusted_hdd_factor, dummy_lookup_emissions_fossil_fuel, 0
    )

    climate_results, annual_emissions, annual_damages = calculate_climate_emissions_and_damages(
        df=sample_df,
        category=category,
        year_label=year_label,
        adjusted_hdd_factor=adjusted_hdd_factor,
        lookup_emissions_electricity_climate=dummy_lookup_emissions_electricity_climate,
        cambium_scenario=cambium_scenario,
        total_fossil_fuel_emissions=total_fossil_fuel_emissions,
        scenario_prefix=scenario_prefix,
        menu_mp=0
    )

    for mer_type in MER_TYPES:
        emissions_key = f"{scenario_prefix}{year_label}_{category}_mt_co2e_{mer_type}"
        assert emissions_key in climate_results, f"Missing {emissions_key}"

        for scc_assumption in SCC_ASSUMPTIONS:
            damages_key = f"{scenario_prefix}{year_label}_{category}_damages_climate_{mer_type}_{scc_assumption}"
            assert damages_key in climate_results, f"Missing {damages_key}"

    for mer_type in MER_TYPES:
        assert mer_type in annual_emissions, f"Missing {mer_type} in annual_emissions"

    for key in annual_damages.keys():
        # Should be (mer_type, scc_assumption)
        assert isinstance(key, tuple) and len(key) == 2, "annual_damages key format mismatch"


# =========================================================================
# Tests for calculate_lifetime_climate_impacts
# =========================================================================

@pytest.mark.parametrize("menu_mp", [0, 8])
@pytest.mark.parametrize("policy_scenario", ["No Inflation Reduction Act", "AEO2023 Reference Case"])
def test_calculate_lifetime_climate_impacts_success(sample_df, dummy_define_scenario_settings, monkeypatch, menu_mp, policy_scenario):
    """
    Test that calculate_lifetime_climate_impacts returns two DataFrames with the lifetime
    climate impact columns for each equipment category, scenario, and SCC assumption.
    """
    monkeypatch.setattr(
        "cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.define_scenario_params",
        dummy_define_scenario_settings
    )

    df_main, df_detailed = calculate_lifetime_climate_impacts(
        df=sample_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        df_baseline_damages=None
    )

    assert isinstance(df_main, pd.DataFrame), "df_main not a DataFrame"
    assert isinstance(df_detailed, pd.DataFrame), "df_detailed not a DataFrame"

    scenario_prefix = dummy_define_scenario_settings(menu_mp, policy_scenario)[0]

    # Check for lifetime emissions/damages columns for each equipment category
    for category in EQUIPMENT_SPECS.keys():
        for mer in MER_TYPES:
            emissions_col = f"{scenario_prefix}{category}_lifetime_mt_co2e_{mer}"
            assert emissions_col in df_main.columns, f"Missing col {emissions_col}"
            for scc_assumption in SCC_ASSUMPTIONS:
                damages_col = f"{scenario_prefix}{category}_lifetime_damages_climate_{mer}_{scc_assumption}"
                assert damages_col in df_main.columns, f"Missing col {damages_col}"

    assert not df_detailed.empty, "df_detailed should contain annual/lifetime detail"


def test_calculate_lifetime_climate_impacts_empty_df(dummy_define_scenario_settings, monkeypatch):
    """
    Test that an empty DataFrame triggers an exception when passed to calculate_lifetime_climate_impacts.
    """
    empty_df = pd.DataFrame()

    monkeypatch.setattr(
        "cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.define_scenario_params",
        dummy_define_scenario_settings
    )

    with pytest.raises(Exception):
        calculate_lifetime_climate_impacts(
            df=empty_df,
            menu_mp=0,
            policy_scenario="No Inflation Reduction Act",
            df_baseline_damages=None
        )


def test_calculate_lifetime_climate_impacts_missing_column(sample_df, dummy_define_scenario_settings, monkeypatch):
    """
    Test that a missing required column (e.g., 'census_division') triggers an exception.
    """
    df_missing = sample_df.drop(columns=['census_division'])

    monkeypatch.setattr(
        "cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.define_scenario_params",
        dummy_define_scenario_settings
    )

    with pytest.raises(Exception):
        calculate_lifetime_climate_impacts(
            df=df_missing,
            menu_mp=0,
            policy_scenario="No Inflation Reduction Act",
            df_baseline_damages=None
        )


def test_calculate_lifetime_climate_impacts_boundary_lifetime(sample_df, dummy_define_scenario_settings, monkeypatch):
    """
    Test boundary condition for equipment lifetime by temporarily overriding EQUIPMENT_SPECS
    to have a single category with lifetime=1.
    """
    test_category = "test_cat"
    original_specs = EQUIPMENT_SPECS.copy()
    try:
        # Provide needed columns so the code doesn't fail looking for them.
        sample_df["base_electricity_test_cat_consumption"] = 99.0
        sample_df["base_naturalGas_test_cat_consumption"] = 15.0
        sample_df["base_propane_test_cat_consumption"] = 10.0
        sample_df["base_fuelOil_test_cat_consumption"] = 5.0

        EQUIPMENT_SPECS.clear()
        EQUIPMENT_SPECS.update({test_category: 1})  # Set the test category's lifetime to 1

        monkeypatch.setattr(
            "cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.define_scenario_params",
            dummy_define_scenario_settings
        )

        df_main, _ = calculate_lifetime_climate_impacts(
            df=sample_df,
            menu_mp=0,
            policy_scenario="No Inflation Reduction Act",
            df_baseline_damages=None
        )

        scenario_prefix = dummy_define_scenario_settings(0, "No Inflation Reduction Act")[0]
        for mer in MER_TYPES:
            emissions_col = f"{scenario_prefix}{test_category}_lifetime_mt_co2e_{mer}"
            assert emissions_col in df_main.columns, f"Missing col {emissions_col}"
    finally:
        # Restore the original specs to avoid side effects for other tests
        EQUIPMENT_SPECS.clear()
        EQUIPMENT_SPECS.update(original_specs)


# =========================================================================
# NEW TESTS FOR ADDITIONAL ERROR HANDLING & COVERAGE
# =========================================================================

@pytest.mark.parametrize("invalid_scenario", ["SomeUnknownPolicy", "InvalidScenario"])
def test_calculate_lifetime_climate_impacts_invalid_policy_scenario(sample_df, dummy_define_scenario_settings, monkeypatch, invalid_scenario):
    """
    Test that an invalid or unrecognized policy_scenario raises ValueError from define_scenario_params.
    """
    monkeypatch.setattr(
        "cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.define_scenario_params",
        dummy_define_scenario_settings
    )

    with pytest.raises(ValueError, match="Invalid Policy Scenario"):
        calculate_lifetime_climate_impacts(
            df=sample_df,
            menu_mp=8,
            policy_scenario=invalid_scenario,
            df_baseline_damages=None
        )


def test_calculate_lifetime_climate_impacts_missing_region_factor(sample_df, dummy_define_scenario_settings, monkeypatch):
    """
    Test that if a needed GEA region is not present in the electricity climate lookup,
    a KeyError or RuntimeError is raised when calculating annual emissions.
    """
    # CHANGED: Modify the sample DF so that gea_region is something not in the dummy lookup
    sample_df['gea_region'] = "UnlistedRegion"

    monkeypatch.setattr(
        "cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.define_scenario_params",
        dummy_define_scenario_settings
    )

    # We expect a KeyError from the emission factor retrieval
    with pytest.raises(Exception):
        calculate_lifetime_climate_impacts(
            df=sample_df,
            menu_mp=0,
            policy_scenario="No Inflation Reduction Act",
            df_baseline_damages=None
        )


def test_calculate_lifetime_climate_impacts_missing_hdd_factor_year(sample_df, dummy_define_scenario_settings, monkeypatch):
    """
    Test that if the precomputed HDD factors do not contain an expected year,
    a RuntimeError is raised. We can do this by mocking 'precompute_hdd_factors'
    to return data missing a necessary year.
    """
    # Mock precompute_hdd_factors to return a DataFrame missing year 2024
    def mock_precompute_hdd_factors(_):
        # Return DF with no columns for year 2024 => KeyError in code
        return pd.DataFrame(index=sample_df.index, columns=[2025, 2026]).fillna(1.0)

    # Monkeypatch the function used by the code
    monkeypatch.setattr(
        "cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.precompute_hdd_factors",
        mock_precompute_hdd_factors
    )
    monkeypatch.setattr(
        "cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.define_scenario_params",
        dummy_define_scenario_settings
    )

    with pytest.raises(RuntimeError, match="Error processing year 2024"):
        calculate_lifetime_climate_impacts(
            df=sample_df,
            menu_mp=0,
            policy_scenario="No Inflation Reduction Act",
            df_baseline_damages=None
        )


def test_calculate_lifetime_climate_impacts_with_baseline_damages(sample_df, dummy_define_scenario_settings, monkeypatch):
    """
    Test coverage for scenario where menu_mp != 0 and df_baseline_damages is provided.
    Verifies that avoided emissions/damages columns are created in df_main and that
    df_detailed includes both annual and lifetime breakdown columns for all equipment categories,
    including heating, waterHeating, clothesDrying, and cooking.
    """
    # Build dummy baseline data for all categories
    baseline_data = {}
    for category in EQUIPMENT_SPECS.keys():
        for mer in MER_TYPES:
            baseline_data[f'baseline_{category}_lifetime_mt_co2e_{mer}'] = [10.0]
            for scc in SCC_ASSUMPTIONS:
                baseline_data[f'baseline_{category}_lifetime_damages_climate_{mer}_{scc}'] = [100.0]
    df_baseline = pd.DataFrame(baseline_data, index=sample_df.index)

    monkeypatch.setattr(
        "cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity.define_scenario_params",
        dummy_define_scenario_settings
    )

    df_main, df_detailed = calculate_lifetime_climate_impacts(
        df=sample_df,
        menu_mp=8,
        policy_scenario="No Inflation Reduction Act",
        df_baseline_damages=df_baseline
    )

    scenario_prefix = dummy_define_scenario_settings(8, "No Inflation Reduction Act")[0]

    # Verify lifetime avoided columns in df_main for each category and MER type
    for category in EQUIPMENT_SPECS.keys():
        for mer in MER_TYPES:
            avoided_emissions_col = f"{scenario_prefix}{category}_avoided_mt_co2e_{mer}"
            assert avoided_emissions_col in df_main.columns, f"Missing {avoided_emissions_col}"
            for scc in SCC_ASSUMPTIONS:
                avoided_damages_col = f"{scenario_prefix}{category}_avoided_damages_climate_{mer}_{scc}"
                assert avoided_damages_col in df_main.columns, f"Missing {avoided_damages_col}"

    # Verify that df_detailed contains at least one annual breakdown column per category
    for category in EQUIPMENT_SPECS.keys():
        # Expect at least a column for the first year (2024) for each MER type
        for mer in MER_TYPES:
            annual_emissions_col = f"{scenario_prefix}2024_{category}_mt_co2e_{mer}"
            assert annual_emissions_col in df_detailed.columns, f"Missing {annual_emissions_col} in detailed output"

    # Also check that the lifetime columns appear in df_detailed
    for category in EQUIPMENT_SPECS.keys():
        for mer in MER_TYPES:
            lifetime_emissions_col = f"{scenario_prefix}{category}_lifetime_mt_co2e_{mer}"
            assert lifetime_emissions_col in df_detailed.columns, f"Missing {lifetime_emissions_col} in detailed output"
