# =======================================================================================================================
# test_calculate_lifetime_health_impacts_sensitivity.py
# =======================================================================================================================
# Test file for the updated health impacts functions in calculate_lifetime_health_impacts.py,
# with scenario params updated to reflect real naming conventions.
#
# Context:
#     The functions calculate_lifetime_health_impacts and calculate_health_damages_for_pair compute lifetime and
#     annual health damages from fossil fuel and electricity consumption. They rely on the define_scenario_params
#     function to determine scenario-specific lookup dictionaries and a scenario prefix, and on precompute_hdd_factors
#     to obtain HDD adjustment factors. The main function returns a tuple of two DataFrames:
#       - df_main: the input DataFrame updated with aggregated (lifetime) health damage columns.
#       - df_detailed: a detailed DataFrame with annual breakdowns.
#
# Objectives:
#     - Test successful execution with valid input using dummy scenario params.
#     - Test edge cases: empty DataFrame, missing required columns.
#     - Test boundary conditions (e.g., equipment with minimal lifetime).
#     - Verify proper exception handling using pytest.raises.
#
# Constraints/Requirements:
#     - Use monkeypatch to override define_scenario_params with a dummy function.
#     - Use pytest.raises for exception testing.
#     - No actual I/O (all data is created in-memory).
#     - Follow naming conventions: "electricity" and "fossil fuel" (not "elec"/"fossil") are used.
# 
# WORKING AS OF 2025-04-03 @ 7:40 PM EST
#
# AUTHOR: JORDAN M. JOSEPH
# =======================================================================================================================

import pytest
import pandas as pd
# from pandas.testing import assert_frame_equal

# Import functions to test (adjust import paths as needed)
from cmu_tare_model.public_impact.calculate_lifetime_health_impacts_sensitivity import (
    calculate_lifetime_health_impacts,
    calculate_health_damages_for_pair,
)
from cmu_tare_model.constants import EQUIPMENT_SPECS, POLLUTANTS, CR_FUNCTIONS, RCM_MODELS, SCC_ASSUMPTIONS

# =========================================================================
# Fixtures
# =========================================================================
@pytest.fixture
def dummy_define_scenario_settings():
    """
    Dummy replacement for define_scenario_params.
    Returns a function that, given menu_mp and policy_scenario, returns a tuple:
      (scenario_prefix, cambium_scenario,
       dummy_lookup_emissions_fossil_fuel,
       dummy_lookup_emissions_electricity_climate,
       dummy_lookup_emissions_electricity_health)

    The dummy lookups return trivial values for fossil fuel and electricity health emissions,
    matching the dictionary shape expected in the original program.
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
                'delta_egrid_so2': 0.01,
                'delta_egrid_nox': 0.01,
                'delta_egrid_pm25': 0.01,
                'delta_egrid_co2e': 0.0,
            }
            for year in range(2024, 2041)  # 2024 ... 2040 inclusive
        }

        # Dummy lookup for electricity health damage factors.
        # Key: (year, region), Value: dict of pollutant factors.
        dummy_lookup_emissions_electricity_health = {
            (2024, 'Region1'): dummy_emission_factors_by_year
        }

        # CHANGED: Use nested dict for fossil fuel lookups
        dummy_lookup_emissions_fossil_fuel = {
            'naturalGas': {'so2': 0.0, 'nox': 0.0, 'pm25': 0.0, 'co2e': 0.0},
            'propane':    {'so2': 0.0, 'nox': 0.0, 'pm25': 0.0, 'co2e': 0.0},
            'fuelOil':    {'so2': 0.0, 'nox': 0.0, 'pm25': 0.0, 'co2e': 0.0},
        }

        # Dummy lookup for electricity climate factors (not used in health damages)
        dummy_lookup_emissions_electricity_climate = {}

        return (
            scenario_prefix,
            cambium_scenario,
            dummy_lookup_emissions_fossil_fuel,
            dummy_lookup_emissions_electricity_climate,
            dummy_lookup_emissions_electricity_health,
        )

    return dummy

@pytest.fixture
def sample_dfs():
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

    # Baseline DataFrame
    df_baseline_damages = pd.DataFrame(data)

    base_year = 2024
    for category in EQUIPMENT_SPECS.keys():
        for year in range(1, lifetime + 1):
            year_label = year + (base_year - 1)

            for scc in SCC_ASSUMPTIONS:
                base_climate_col = f'baseline_{year_label}_{category}_damages_climate_lrmer_{scc}'
                df_baseline_damages[base_climate_col] = 0.0  # dummy value
                
            for rcm in RCM_MODELS:
                for cr in CR_FUNCTIONS:
                    base_health_col = f'baseline_{year_label}_{category}_damages_health_{rcm}_{cr}'
                    df_baseline_damages[base_health_col] = 0.0  # dummy value

    # Measure-Package DataFrame
    df_mp_damages = df_baseline_damages.copy()

    # Call the dummy function to get the scenario prefix
    scenario_prefix, _, _, _, _ = dummy_define_scenario_settings(8, "AEO2023 Reference Case")

    for category, lifetime in EQUIPMENT_SPECS.items():
        for year in range(1, lifetime + 1):
            for scc in SCC_ASSUMPTIONS:
                mp_climate_col = f'{scenario_prefix}{year_label}_{category}_damages_climate_lrmer_{scc}'
                df_mp_damages[mp_climate_col] = 0.0  # dummy value
                
            for rcm in RCM_MODELS:
                for cr in CR_FUNCTIONS:
                    mp_health_col = f'{scenario_prefix}{year_label}_{category}_damages_health_{rcm}_{cr}'
                    df_mp_damages[mp_health_col] = 0.0  # dummy value

    # Main DataFrame
    df_main = df_mp_damages.copy()

    return df_main, df_baseline_damages, df_mp_damages

# =========================================================================
# Tests for calculate_health_damages_for_pair
# =========================================================================

def test_vectorized_lookup_missing_value(sample_df, dummy_define_scenario_settings, monkeypatch):
    """
    Test that the vectorized lookup correctly handles missing lookup values by returning np.nan.
    """
    # Set county_fips to a value that is not in the dummy lookup
    sample_df['county_fips'] = ['00000']
    monkeypatch.setattr(
        "cmu_tare_model.utils.modeling_params.define_scenario_params",
        dummy_define_scenario_settings
    )
    df_main, _ = calculate_lifetime_health_impacts(
        df=sample_df,
        menu_mp=0,
        policy_scenario="No Inflation Reduction Act",
        df_baseline_damages=None
    )
    # Check one of the damage columns; assume at least one exists
    damage_col = [col for col in df_main.columns if "damages_health" in col][0]
    # The result should be np.nan for the row with missing lookup
    assert pd.isna(df_main.loc[sample_df.index[0], damage_col]), "Expected np.nan for missing lookup value"
    

def test_calculate_health_damages_for_pair_success(sample_df, dummy_define_scenario_settings):
    """
    Verify that calculate_health_damages_for_pair returns expected keys and series for a valid input.
    """
    # Use category 'heating' and year_label corresponding to 2024.
    category = 'heating'
    year_label = 2024
    adjusted_hdd_factor = pd.Series(1.0, index=sample_df.index)
    # Dummy fossil fuel emissions: zero for all pollutants.
    total_fossil_fuel_emissions = {p: pd.Series(0.0, index=sample_df.index) for p in POLLUTANTS}
    rcm = RCM_MODELS[0]  # e.g., 'AP2'
    cr = CR_FUNCTIONS[0]  # e.g., 'acs'
    # Get dummy scenario params.
    scenario_prefix, _, _, _, dummy_lookup_emissions_electricity_health = dummy_define_scenario_settings(0, "No Inflation Reduction Act")
    
    result = calculate_health_damages_for_pair(
        df=sample_df,
        category=category,
        year_label=year_label,
        adjusted_hdd_factor=adjusted_hdd_factor,
        lookup_emissions_electricity_health=dummy_lookup_emissions_electricity_health,
        scenario_prefix=scenario_prefix,
        total_fossil_fuel_emissions=total_fossil_fuel_emissions,
        menu_mp=0,
        rcm=rcm,
        cr=cr
    )
    # Expected keys for each pollutant (except 'co2e') and an overall key.
    for pollutant in POLLUTANTS:
        if pollutant == 'co2e':
            continue
        key = f"{scenario_prefix}{year_label}_{category}_damages_{pollutant}_{rcm}_{cr}"
        assert key in result, f"Missing key {key} in result."
    overall_key = f"{scenario_prefix}{year_label}_{category}_damages_health_{rcm}_{cr}"
    assert overall_key in result, f"Missing overall key {overall_key} in result."
    pd.testing.assert_index_equal(result[overall_key].index, sample_df.index)

# =========================================================================
# Tests for calculate_lifetime_health_impacts
# =========================================================================

@pytest.mark.parametrize("menu_mp", [0, 8])
@pytest.mark.parametrize("policy_scenario", ["No Inflation Reduction Act", "AEO2023 Reference Case"])
def test_calculate_lifetime_health_impacts_success(sample_df, dummy_define_scenario_settings, monkeypatch, menu_mp, policy_scenario):
    """
    Test that calculate_lifetime_health_impacts returns a tuple of DataFrames with correct lifetime columns.
    This test overrides define_scenario_params to return dummy lookup dictionaries and scenario prefix.
    """
    # Monkeypatch define_scenario_params with our dummy function.
    from cmu_tare_model.utils.modeling_params import define_scenario_params    monkeypatch.setattr(
        "cmu_tare_model.utils.modeling_params.define_scenario_params",
        dummy_define_scenario_settings
    )
    
    df_main, df_detailed = calculate_lifetime_health_impacts(
        df=sample_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        df_baseline_damages=None
    )
    assert isinstance(df_main, pd.DataFrame)
    assert isinstance(df_detailed, pd.DataFrame)
    
    # For each equipment category, check lifetime columns in df_main.
    # Since sample_df only provides consumption for 'heating', only check for that category.
    category = "heating"
    scenario_prefix, _, _, _, _ = dummy_define_scenario_settings(menu_mp, policy_scenario)
    for rcm in RCM_MODELS:
        for cr in CR_FUNCTIONS:
            col_name = f"{scenario_prefix}{category}_lifetime_damages_health_{rcm}_{cr}"
            assert col_name in df_main.columns, f"Expected column {col_name} not found in df_main."
    
    assert not df_detailed.empty, "df_detailed should contain annual breakdown details."


def test_calculate_lifetime_health_impacts_empty_df(dummy_define_scenario_settings, monkeypatch):
    """
    Test that calculate_lifetime_health_impacts raises an exception when provided with an empty DataFrame.
    """
    empty_df = pd.DataFrame()
    monkeypatch.setattr(
        "cmu_tare_model.utils.modeling_params.define_scenario_params",
        dummy_define_scenario_settings
    )
    with pytest.raises(Exception) as excinfo:
        calculate_lifetime_health_impacts(
            df=empty_df,
            menu_mp=0,
            policy_scenario="No Inflation Reduction Act",
            df_baseline_damages=None
        )
    # Expect an error message indicating missing required columns (e.g., 'census_division')
    assert "census_division" in str(excinfo.value).lower() or "missing" in str(excinfo.value).lower()


def test_calculate_lifetime_health_impacts_missing_column(sample_df, dummy_define_scenario_settings, monkeypatch):
    """
    Test that calculate_lifetime_health_impacts raises an exception if a required column (e.g., 'county_fips') is missing.
    """
    df_missing = sample_df.drop(columns=['county_fips'])
    monkeypatch.setattr(
        "cmu_tare_model.utils.modeling_params.define_scenario_params",
        dummy_define_scenario_settings
    )
    # Set up the test to expect a RuntimeError due to missing 'county_fips' column.
    with pytest.raises(RuntimeError) as excinfo:
        calculate_lifetime_health_impacts(
            df=df_missing,
            menu_mp=0,
            policy_scenario="No Inflation Reduction Act",
            df_baseline_damages=None
        )
    assert "'county_fips'" in str(excinfo.value), "Expected error message to mention 'county_fips'"

def test_calculate_lifetime_health_impacts_boundary_lifetime(sample_df, dummy_define_scenario_settings, monkeypatch):
    """
    Test the boundary condition for equipment lifetime by temporarily overriding EQUIPMENT_SPECS.
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
            "cmu_tare_model.utils.modeling_params.define_scenario_params",
            dummy_define_scenario_settings
        )

        df_main, _ = calculate_lifetime_health_impacts(
            df=sample_df,
            menu_mp=0,
            policy_scenario="No Inflation Reduction Act",
            df_baseline_damages=None
        )

        for rcm in RCM_MODELS:
            for cr in CR_FUNCTIONS:
                col_name = f"{dummy_define_scenario_settings(0, 'No Inflation Reduction Act')[0]}{test_category}_lifetime_damages_health_{rcm}_{cr}"
                assert col_name in df_main.columns, f"Missing lifetime column {col_name}"
    finally:
        EQUIPMENT_SPECS.clear()
        EQUIPMENT_SPECS.update(original_specs)
