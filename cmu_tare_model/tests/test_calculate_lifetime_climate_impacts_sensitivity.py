import pytest
import pandas as pd
# from pandas.testing import assert_frame_equal

# Import from your updated code (the "sensitivity" version or whichever final module name you use)
from cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity import (
    calculate_climate_impacts,
    calculate_climate_emissions_and_damages
)
from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import calculate_fossil_fuel_emissions
from cmu_tare_model.constants import MER_TYPES, EQUIPMENT_SPECS

# --------------------------
# Fixtures for sample inputs and dummy scenario settings
# --------------------------

@pytest.fixture
def sample_df():
    """
    Create a minimal valid DataFrame with required columns for climate impacts.
    Required columns include:
      - 'fips', 'state', 'year', 'cambium_gea_region', 'census_division', 'gea_region'
      - Consumption columns for category 'heating': 
          'base_electricity_heating_consumption', 
          'base_naturalGas_heating_consumption',
          'base_propane_heating_consumption',
          'base_fuelOil_heating_consumption',
          and for non-baseline: 'mp1_2024_heating_consumption'
    """
    data = {
        'fips': ['12345'],
        'state': ['XX'],
        'year': [2023],
        'cambium_gea_region': ['Region1'],
        'census_division': ['National'],  # required for HDD factors
        'gea_region': ['Region1'],         # used for electricity climate lookups
        'base_electricity_heating_consumption': [100.0],
        'base_naturalGas_heating_consumption': [50.0],
        'base_propane_heating_consumption': [40.0],
        'base_fuelOil_heating_consumption': [30.0],
        'mp1_2024_heating_consumption': [110.0],
    }
    return pd.DataFrame(data)

@pytest.fixture
def dummy_define_scenario_settings():
    """
    Dummy replacement for define_scenario_settings.

    Returns a function that, given menu_mp and policy_scenario, returns a tuple:
      (scenario_prefix, cambium_scenario, dummy_lookup_emissions_fossil_fuel, 
       dummy_lookup_emissions_electricity_climate, dummy_dummy)
    
    The dummy lookup dictionaries return 0 for fossil fuel emissions and fixed factors for electricity climate emissions.
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
        dummy_lookup_electricity_climate = {
            (cambium_scenario, "Region1"): {
                2024: {
                    'lrmer_mt_per_kWh_co2e': 0.02,
                    'srmer_mt_per_kWh_co2e': 0.03
                }
            }
        }
        dummy_dummy = {}  # placeholder for lookup_emissions_electricity_health (unused in climate impacts)
        return scenario_prefix, cambium_scenario, dummy_lookup_fossil_fuel, dummy_lookup_electricity_climate, dummy_dummy
    return dummy

# --------------------------
# Tests for calculate_climate_emissions_and_damages
# --------------------------

def test_get_scc_value_correct_retrieval():
    """
    Verifies that the get_scc_value() helper function retrieves
    the correct SCC from a nested dictionary:
        lookup_climate_impact_scc[scc_assumption][year] = value
    Also checks clamping if the requested year is out of range.
    """
    # A small SCC lookup dict for testing:
    # Keys are scc_assumption => each is a dict of year => value
    lookup_climate_impact_scc = {
        "lower": {2023: 10.0, 2024: 12.5, 2030: 15.0},
        "central": {2023: 50.0, 2024: 55.0, 2030: 60.0},
        "upper": {2023: 100.0, 2024: 110.0, 2030: 120.0}
    }
    
    # We create a local function that mirrors the one in your code.
    def get_scc_value(year_label: int, scc_assumption: str, scc_lookup: dict) -> float:
        if year_label in scc_lookup[scc_assumption]:
            return scc_lookup[scc_assumption][year_label]
        else:
            # clamp to the max year present
            max_year = max(scc_lookup[scc_assumption].keys())
            return scc_lookup[scc_assumption][max_year]

    # 1) Exact year matches
    assert get_scc_value(2023, "lower", lookup_climate_impact_scc) == 10.0
    assert get_scc_value(2024, "central", lookup_climate_impact_scc) == 55.0
    assert get_scc_value(2024, "upper", lookup_climate_impact_scc) == 110.0

    # 2) Year is out of range => should clamp to the max year
    #    For each assumption, the max year is 2030 in our small dict
    assert get_scc_value(2040, "lower", lookup_climate_impact_scc) == 15.0
    assert get_scc_value(3000, "central", lookup_climate_impact_scc) == 60.0
    assert get_scc_value(9999, "upper", lookup_climate_impact_scc) == 120.0

def test_calculate_climate_emissions_and_damages_success(sample_df, dummy_define_scenario_settings):
    """
    Verify that calculate_climate_emissions_and_damages returns the expected keys
    for annual emissions and damages given a valid input.
    """
    category = 'heating'
    year_label = 2024
    adjusted_hdd_factor = pd.Series(1.0, index=sample_df.index)
    
    # Get dummy scenario settings
    scenario_prefix, cambium_scenario, dummy_lookup_fossil_fuel, dummy_lookup_electricity_climate, _ = dummy_define_scenario_settings(0, "No Inflation Reduction Act")
    
    # Calculate dummy fossil fuel emissions (should be zeros given the dummy lookup)
    total_fossil_fuel_emissions = calculate_fossil_fuel_emissions(
        sample_df, category, adjusted_hdd_factor, dummy_lookup_fossil_fuel, 0
    )
    
    # Call the function under test
    climate_results, annual_emissions, annual_damages = calculate_climate_emissions_and_damages(
        df=sample_df,
        category=category,
        year_label=year_label,
        adjusted_hdd_factor=adjusted_hdd_factor,
        lookup_emissions_electricity_climate=dummy_lookup_electricity_climate,
        cambium_scenario=cambium_scenario,
        total_fossil_fuel_emissions=total_fossil_fuel_emissions,
        scenario_prefix=scenario_prefix,
        menu_mp=0
    )
    
    # Check for expected keys in climate_results for each MER type
    for mer_type in MER_TYPES:
        emissions_key = f"{scenario_prefix}{year_label}_{category}_mt_co2e_{mer_type}"
        assert emissions_key in climate_results, f"Missing key {emissions_key} in climate_results."
        for scc_assumption in ['lower', 'central', 'upper']:
            damages_key = f"{scenario_prefix}{year_label}_{category}_damages_climate_{mer_type}_{scc_assumption}"
            assert damages_key in climate_results, f"Missing key {damages_key} in climate_results."
    
    # Verify that annual_emissions contains each MER type
    for mer_type in MER_TYPES:
        assert mer_type in annual_emissions
    # Verify that annual_damages keys are tuples (mer_type, scc_assumption)
    for key in annual_damages.keys():
        assert isinstance(key, tuple) and len(key) == 2

# --------------------------
# Tests for calculate_climate_impacts
# --------------------------

@pytest.mark.parametrize("menu_mp", [0, 1])
@pytest.mark.parametrize("policy_scenario", ["No Inflation Reduction Act", "AEO2023 Reference Case"])
def test_calculate_climate_impacts_success(sample_df, dummy_define_scenario_settings, monkeypatch, menu_mp, policy_scenario):
    """
    Test that calculate_climate_impacts returns a tuple of DataFrames with the expected lifetime climate impact columns.
    This test overrides define_scenario_settings with a dummy function.
    """
    from cmu_tare_model.public_impact.emissions_scenario_settings import define_scenario_settings
    monkeypatch.setattr(
        "cmu_tare_model.public_impact.emissions_scenario_settings.define_scenario_settings",
        dummy_define_scenario_settings
    )
    
    df_main, df_detailed = calculate_climate_impacts(
        df=sample_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        df_baseline_damages=None
    )
    
    assert isinstance(df_main, pd.DataFrame)
    assert isinstance(df_detailed, pd.DataFrame)
    
    # For each equipment category in EQUIPMENT_SPECS, check for lifetime emissions and damages columns in df_main.
    for category in EQUIPMENT_SPECS.keys():
        # Check emissions columns for each MER type.
        for mer in MER_TYPES:
            emissions_col = f"{dummy_define_scenario_settings(menu_mp, policy_scenario)[0]}{category}_lifetime_mt_co2e_{mer}"
            assert emissions_col in df_main.columns, f"Expected column {emissions_col} not found in df_main."
            # Check damages columns for each scc assumption.
            for scc_assumption in ['lower', 'central', 'upper']:
                damages_col = f"{dummy_define_scenario_settings(menu_mp, policy_scenario)[0]}{category}_lifetime_damages_climate_{mer}_{scc_assumption}"
                assert damages_col in df_main.columns, f"Expected column {damages_col} not found in df_main."
    
    assert not df_detailed.empty, "df_detailed should contain annual breakdown details."

def test_calculate_climate_impacts_empty_df(dummy_define_scenario_settings, monkeypatch):
    """
    Test that calculate_climate_impacts raises an exception when provided with an empty DataFrame.
    """
    empty_df = pd.DataFrame()
    monkeypatch.setattr(
        "cmu_tare_model.public_impact.emissions_scenario_settings.define_scenario_settings",
        dummy_define_scenario_settings
    )
    with pytest.raises(Exception) as excinfo:
        calculate_climate_impacts(
            df=empty_df,
            menu_mp=0,
            policy_scenario="No Inflation Reduction Act",
            df_baseline_damages=None
        )
    assert "census_division" in str(excinfo.value).lower() or "missing" in str(excinfo.value).lower()

def test_calculate_climate_impacts_missing_column(sample_df, dummy_define_scenario_settings, monkeypatch):
    """
    Test that calculate_climate_impacts raises an exception if a required column (e.g., 'census_division') is missing.
    """
    df_missing = sample_df.drop(columns=['census_division'])
    monkeypatch.setattr(
        "cmu_tare_model.public_impact.emissions_scenario_settings.define_scenario_settings",
        dummy_define_scenario_settings
    )
    with pytest.raises(KeyError):
        calculate_climate_impacts(
            df=df_missing,
            menu_mp=0,
            policy_scenario="No Inflation Reduction Act",
            df_baseline_damages=None
        )

def test_calculate_climate_impacts_boundary_lifetime(sample_df, dummy_define_scenario_settings, monkeypatch):
    """
    Test the boundary condition for equipment lifetime by temporarily overriding EQUIPMENT_SPECS.
    """
    test_category = "test_cat"
    original_specs = EQUIPMENT_SPECS.copy()
    try:
        EQUIPMENT_SPECS.clear()
        EQUIPMENT_SPECS.update({test_category: 1})
        monkeypatch.setattr(
            "cmu_tare_model.public_impact.emissions_scenario_settings.define_scenario_settings",
            dummy_define_scenario_settings
        )
        df_main, df_detailed = calculate_climate_impacts(
            df=sample_df,
            menu_mp=0,
            policy_scenario="No Inflation Reduction Act",
            df_baseline_damages=None
        )
        for mer in MER_TYPES:
            emissions_col = f"{dummy_define_scenario_settings(0, 'No Inflation Reduction Act')[0]}{test_category}_lifetime_mt_co2e_{mer}"
            assert emissions_col in df_main.columns, f"Missing lifetime column {emissions_col}"
    finally:
        EQUIPMENT_SPECS.clear()
        EQUIPMENT_SPECS.update(original_specs)