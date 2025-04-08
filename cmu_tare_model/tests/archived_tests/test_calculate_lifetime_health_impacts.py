"""
Test file for the updated health impacts functions in calculate_lifetime_health_impacts.py,
with scenario params updated to reflect real naming conventions.

Context:
    The functions calculate_health_impacts and calculate_health_damages_for_pair compute lifetime and
    annual health damages from fossil fuel and electricity consumption. They rely on the define_scenario_params
    function to determine scenario-specific lookup dictionaries and a scenario prefix, and on precompute_hdd_factors
    to obtain HDD adjustment factors. The main function returns a tuple of two DataFrames:
      - df_main: the input DataFrame updated with aggregated (lifetime) health damage columns.
      - df_detailed: a detailed DataFrame with annual breakdowns.

Objectives:
    - Test successful execution with valid input using dummy scenario params.
    - Test edge cases: empty DataFrame, missing required columns.
    - Test boundary conditions (e.g., equipment with minimal lifetime).
    - Verify proper exception handling using pytest.raises.

Constraints/Requirements:
    - Use monkeypatch to override define_scenario_params with a dummy function.
    - Use pytest.raises for exception testing.
    - No actual I/O (all data is created in-memory).
    - Follow naming conventions: "electricity" and "fossil fuel" (not "elec"/"fossil") are used.
"""

import pytest
import pandas as pd
# from pandas.testing import assert_frame_equal

# Import functions to test (adjust import paths as needed)
from cmu_tare_model.public_impact.calculate_lifetime_health_impacts import (
    calculate_health_impacts,
    calculate_health_damages_for_pair,
)
from cmu_tare_model.constants import EQUIPMENT_SPECS, POLLUTANTS, CR_FUNCTIONS, RCM_MODELS

# --------------------------
# Fixtures for sample inputs and dummy scenario params
# --------------------------

@pytest.fixture
def sample_df():
    """
    Create a minimal valid DataFrame with required columns.
    Required columns include:
      - 'fips', 'state', 'year', 'gea_region', 'census_division'
      - Consumption columns for category 'heating': 'base_electricity_heating_consumption'
    """
    data = {
        'fips': ['12345'],
        'state': ['XX'],
        'year': [2023],
        'gea_region': ['Region1'],
        'census_division': ['National'],  # precompute_hdd_factors uses this
        'base_electricity_heating_consumption': [100.0],
        # For non-baseline case:
        'mp1_2024_heating_consumption': [110.0],
    }
    return pd.DataFrame(data)

@pytest.fixture
def dummy_define_scenario_settings():
    """
    Dummy replacement for define_scenario_params.

    Returns a function that, given menu_mp and policy_scenario, returns a tuple:
      (scenario_prefix, cambium_scenario, dummy_lookup_emissions_fossil_fuel, 
       dummy_lookup_emissions_electricity_climate, dummy_lookup_emissions_electricity_health)
    
    The dummy lookup dictionaries return 0 for fossil fuel emissions and simple fixed factors for
    electricity health damages.
    """
    def dummy(menu_mp, policy_scenario):
        # Determine scenario_prefix based on inputs.
        if menu_mp == 0:
            scenario_prefix = "baseline_"
        elif policy_scenario == "No Inflation Reduction Act":
            scenario_prefix = f"preIRA_mp{menu_mp}_"
        elif policy_scenario == "AEO2023 Reference Case":
            scenario_prefix = f"iraRef_mp{menu_mp}_"
        else:
            raise ValueError("Invalid Policy Scenario!")
        
        cambium_scenario = "MidCase"
        
        # Dummy lookup for fossil fuel emissions factors: returns 0.0 for any key.
        dummy_lookup_fossil_fuel = {
            ('naturalGas', 'so2'): 0.0,
            ('naturalGas', 'nox'): 0.0,
            ('naturalGas', 'pm25'): 0.0,
            ('naturalGas', 'co2e'): 0.0,
            ('propane', 'so2'): 0.0,
            ('propane', 'nox'): 0.0,
            ('propane', 'pm25'): 0.0,
            ('propane', 'co2e'): 0.0,
            ('fuelOil', 'so2'): 0.0,
            ('fuelOil', 'nox'): 0.0,
            ('fuelOil', 'pm25'): 0.0,
            ('fuelOil', 'co2e'): 0.0,
        }
        
        # Dummy lookup for electricity climate factors (not used in health damages)
        dummy_lookup_electricity_climate = {}
        
        # Dummy lookup for electricity health damage factors.
        # Key: (year, region), Value: dict of pollutant factors.
        dummy_lookup_electricity_health = {
            (2024, 'Region1'): {
                'delta_egrid_so2': 0.01,
                'delta_egrid_nox': 0.01,
                'delta_egrid_pm25': 0.01,
                'delta_egrid_co2e': 0.0,
            }
        }
        return scenario_prefix, cambium_scenario, dummy_lookup_fossil_fuel, dummy_lookup_electricity_climate, dummy_lookup_electricity_health
    return dummy

# --------------------------
# Tests for calculate_health_damages_for_pair
# --------------------------

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
    scenario_prefix, _, _, _, dummy_lookup_electricity_health = dummy_define_scenario_settings(0, "No Inflation Reduction Act")
    
    result = calculate_health_damages_for_pair(
        df=sample_df,
        category=category,
        year_label=year_label,
        adjusted_hdd_factor=adjusted_hdd_factor,
        lookup_emissions_electricity_health=dummy_lookup_electricity_health,
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
        key = f"{scenario_prefix}{year_label}_{category}_damages_{pollutant}_{cr}_{rcm}"
        assert key in result, f"Missing key {key} in result."
    overall_key = f"{scenario_prefix}{year_label}_{category}_damages_health_{cr}_{rcm}"
    assert overall_key in result, f"Missing overall key {overall_key} in result."
    pd.testing.assert_index_equal(result[overall_key].index, sample_df.index)

# --------------------------
# Tests for calculate_health_impacts
# --------------------------

@pytest.mark.parametrize("menu_mp", [0, 1])
@pytest.mark.parametrize("policy_scenario", ["No Inflation Reduction Act", "AEO2023 Reference Case"])
def test_calculate_health_impacts_success(sample_df, dummy_define_scenario_settings, monkeypatch, menu_mp, policy_scenario):
    """
    Test that calculate_health_impacts returns a tuple of DataFrames with correct lifetime columns.
    This test overrides define_scenario_params to return dummy lookup dictionaries and scenario prefix.
    """
    # Monkeypatch define_scenario_params with our dummy function.
    from cmu_tare_model.utils.modeling_params import define_scenario_params    monkeypatch.setattr(
        "cmu_tare_model.utils.modeling_params.define_scenario_params",
        dummy_define_scenario_settings
    )
    
    df_main, df_detailed = calculate_health_impacts(
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
            col_name = f"{scenario_prefix}{category}_lifetime_damages_health_{cr}_{rcm}"
            assert col_name in df_main.columns, f"Expected column {col_name} not found in df_main."
    
    assert not df_detailed.empty, "df_detailed should contain annual breakdown details."

def test_calculate_health_impacts_empty_df(dummy_define_scenario_settings, monkeypatch):
    """
    Test that calculate_health_impacts raises an exception when provided with an empty DataFrame.
    """
    empty_df = pd.DataFrame()
    monkeypatch.setattr(
        "cmu_tare_model.utils.modeling_params.define_scenario_params",
        dummy_define_scenario_settings
    )
    with pytest.raises(Exception) as excinfo:
        calculate_health_impacts(
            df=empty_df,
            menu_mp=0,
            policy_scenario="No Inflation Reduction Act",
            df_baseline_damages=None
        )
    # Expect an error message indicating missing required columns (e.g., 'census_division')
    assert "census_division" in str(excinfo.value).lower() or "missing" in str(excinfo.value).lower()

def test_calculate_health_impacts_missing_column(sample_df, dummy_define_scenario_settings, monkeypatch):
    """
    Test that calculate_health_impacts raises an exception if a required column (e.g., 'fips') is missing.
    """
    df_missing = sample_df.drop(columns=['fips'])
    monkeypatch.setattr(
        "cmu_tare_model.utils.modeling_params.define_scenario_params",
        dummy_define_scenario_settings
    )
    with pytest.raises(KeyError):
        calculate_health_impacts(
            df=df_missing,
            menu_mp=0,
            policy_scenario="No Inflation Reduction Act",
            df_baseline_damages=None
        )

def test_calculate_health_impacts_boundary_lifetime(sample_df, dummy_define_scenario_settings, monkeypatch):
    """
    Test the boundary condition for equipment lifetime by temporarily overriding EQUIPMENT_SPECS.
    """
    test_category = "test_cat"
    original_specs = EQUIPMENT_SPECS.copy()
    try:
        EQUIPMENT_SPECS.clear()
        EQUIPMENT_SPECS.update({test_category: 1})
        monkeypatch.setattr(
            "cmu_tare_model.utils.modeling_params.define_scenario_params",
            dummy_define_scenario_settings
        )
        df_main, df_detailed = calculate_health_impacts(
            df=sample_df,
            menu_mp=0,
            policy_scenario="No Inflation Reduction Act",
            df_baseline_damages=None
        )
        for rcm in RCM_MODELS:
            for cr in CR_FUNCTIONS:
                col_name = f"{dummy_define_scenario_settings(0, 'No Inflation Reduction Act')[0]}{test_category}_lifetime_damages_health_{cr}_{rcm}"
                assert col_name in df_main.columns, f"Missing lifetime column {col_name}"
    finally:
        EQUIPMENT_SPECS.clear()
        EQUIPMENT_SPECS.update(original_specs)
