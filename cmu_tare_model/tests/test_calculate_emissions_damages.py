# Filename: test_calculate_emissions_damages.py
"""
Pytest suite for testing the refactored emissions and damages calculation functions.
Covers:
- Successful end-to-end calculations on typical data.
- Edge cases (empty DataFrame, missing columns, negative/large values, invalid scenarios, etc.).
"""

import pytest
import pandas as pd
import numpy as np

# 1) Import from the refactored modules
from cmu_tare_model.public_impact.calculate_emissions_damages import (
    calculate_marginal_damages,
    calculate_damages_grid_scenario
)

from cmu_tare_model.public_impact.precompute_hdd_factors import precompute_hdd_factors
from cmu_tare_model.public_impact.calculate_fossil_fuel_emissions import calculate_fossil_fuel_emissions
from cmu_tare_model.public_impact.calculate_electricity_emissions import (
    calculate_climate_emissions_and_damages,
    calculate_health_damages
)
from cmu_tare_model.constants import define_scenario_params

# 2) Import constants and global lookups (to patch)
from cmu_tare_model.constants import EPA_SCC_USD2023_PER_MT, POLLUTANTS

# Suppose these are the real global dictionaries your code imports
from cmu_tare_model.public_impact.create_lookup_emissions_fossil_fuel import lookup_emissions_fossil_fuel as dummy_fossil
from cmu_tare_model.public_impact.create_lookup_emissions_electricity_climate import (
    lookup_emissions_electricity_climate_preIRA as dummy_co2e_preIRA,
    lookup_emissions_electricity_climate_IRA as dummy_co2e_IRA
)
from cmu_tare_model.public_impact.lookup_emissions_electricity_health import (
    lookup_health_damages_electricity_preIRA as dummy_health_preIRA,
    lookup_health_damages_electricity_iraRef as dummy_health_iraRef
)
from cmu_tare_model.energy_consumption_and_metadata.project_future_energy_consumption import (
    lookup_hdd_factor as dummy_hdd
)


@pytest.fixture
def mock_lookup_emis_fossil_fuel():
    """
    Returns a minimal but valid dictionary simulating
    fossil fuel emissions factors for pollutants in ton/unit.
    """
    return {
        ('naturalGas', 'so2'): 0.0001,
        ('naturalGas', 'nox'): 0.001,
        ('naturalGas', 'pm25'): 0.00005,
        ('naturalGas', 'co2e'): 0.06,
        ('propane', 'so2'): 0.0002,
        ('propane', 'nox'): 0.002,
        ('propane', 'pm25'): 0.00008,
        ('propane', 'co2e'): 0.10,
        ('fuelOil', 'so2'): 0.0005,
        ('fuelOil', 'nox'): 0.003,
        ('fuelOil', 'pm25'): 0.0001,
        ('fuelOil', 'co2e'): 0.12
    }

@pytest.fixture
def mock_lookup_co2e_emis_electricity_preIRA():
    """
    Returns a minimal dictionary for pre-IRA electricity CO2e factors
    under a "MidCase" scenario, for a couple of years.
    """
    return {
        ('MidCase', 'R1'): {
            2024: {'lrmer_mt_per_kWh_co2e': 0.0005, 'srmer_mt_per_kWh_co2e': 0.0007},
            2025: {'lrmer_mt_per_kWh_co2e': 0.0004, 'srmer_mt_per_kWh_co2e': 0.0006},
        }
    }

@pytest.fixture
def mock_lookup_co2e_emis_electricity_IRA():
    """
    Returns a minimal dictionary for IRA electricity CO2e factors
    under a "MidCase" scenario, for a couple of years.
    """
    return {
        ('MidCase', 'R1'): {
            2024: {'lrmer_mt_per_kWh_co2e': 0.0003, 'srmer_mt_per_kWh_co2e': 0.0005},
            2025: {'lrmer_mt_per_kWh_co2e': 0.0002, 'srmer_mt_per_kWh_co2e': 0.0004},
        }
    }

@pytest.fixture
def mock_lookup_health_damages_electricity_preIRA():
    """
    Returns minimal health damage factors per kWh by pollutant,
    adjusted for a region and year, pre-IRA scenario.
    """
    return {
        ('MidCase', 'R1'): {
            2024: {
                'so2_dollarPerkWh_adjustVSL': 0.0001,
                'nox_dollarPerkWh_adjustVSL': 0.0002,
                'pm25_dollarPerkWh_adjustVSL': 0.0003
            },
            2025: {
                'so2_dollarPerkWh_adjustVSL': 0.00008,
                'nox_dollarPerkWh_adjustVSL': 0.00018,
                'pm25_dollarPerkWh_adjustVSL': 0.00028
            }
        }
    }

@pytest.fixture
def mock_lookup_health_damages_electricity_iraRef():
    """
    Returns minimal health damage factors per kWh by pollutant,
    adjusted for a region and year, IRA scenario.
    """
    return {
        ('MidCase', 'R1'): {
            2024: {
                'so2_dollarPerkWh_adjustVSL': 0.00005,
                'nox_dollarPerkWh_adjustVSL': 0.0001,
                'pm25_dollarPerkWh_adjustVSL': 0.0002
            },
            2025: {
                'so2_dollarPerkWh_adjustVSL': 0.00004,
                'nox_dollarPerkWh_adjustVSL': 0.00009,
                'pm25_dollarPerkWh_adjustVSL': 0.00015
            }
        }
    }

@pytest.fixture
def mock_lookup_hdd_factor():
    """
    Returns a minimal dictionary for HDD adjustments by region and year.
    """
    return {
        'National': {
            2024: 1.0,
            2025: 1.0
        },
        'R1': {
            2024: 1.2,
            2025: 1.1
        }
    }


@pytest.fixture
def patch_global_lookups(
    mock_lookup_emis_fossil_fuel,
    mock_lookup_co2e_emis_electricity_preIRA,
    mock_lookup_co2e_emis_electricity_IRA,
    mock_lookup_health_damages_electricity_preIRA,
    mock_lookup_health_damages_electricity_iraRef,
    mock_lookup_hdd_factor
):
    """
    Fixture to patch the global dictionaries (dummy_fossil, dummy_co2e_preIRA, etc.)
    with our mock data. This ensures each test that depends on these lookups
    gets consistent data without manually updating them.
    """
    old_fossil = dummy_fossil.copy()
    old_co2e_preIRA = dummy_co2e_preIRA.copy()
    old_co2e_IRA = dummy_co2e_IRA.copy()
    old_health_preIRA = dummy_health_preIRA.copy()
    old_health_iraRef = dummy_health_iraRef.copy()
    old_hdd = dummy_hdd.copy()
    
    # Overwrite with mock data
    dummy_fossil.update(mock_lookup_emis_fossil_fuel)
    dummy_co2e_preIRA.update(mock_lookup_co2e_emis_electricity_preIRA)
    dummy_co2e_IRA.update(mock_lookup_co2e_emis_electricity_IRA)
    dummy_health_preIRA.update(mock_lookup_health_damages_electricity_preIRA)
    dummy_health_iraRef.update(mock_lookup_health_damages_electricity_iraRef)
    dummy_hdd.update(mock_lookup_hdd_factor)
    
    yield  # Let the test(s) run
    
    # Restore originals
    dummy_fossil.clear()
    dummy_fossil.update(old_fossil)
    dummy_co2e_preIRA.clear()
    dummy_co2e_preIRA.update(old_co2e_preIRA)
    dummy_co2e_IRA.clear()
    dummy_co2e_IRA.update(old_co2e_IRA)
    dummy_health_preIRA.clear()
    dummy_health_preIRA.update(old_health_preIRA)
    dummy_health_iraRef.clear()
    dummy_health_iraRef.update(old_health_iraRef)
    dummy_hdd.clear()
    dummy_hdd.update(old_hdd)


@pytest.fixture
def sample_df():
    """
    Returns a typical DataFrame with minimal columns needed
    for calculations, representing two rows (households).
    """
    data = {
        'gea_region': ['R1', 'R1'],
        'census_division': ['R1', 'R1'],
        'base_naturalGas_heating_consumption': [100.0, 200.0],
        'base_electricity_heating_consumption': [300.0, 400.0],
        'marginal_damages_so2': [0.01, 0.02],
        'marginal_damages_nox': [0.04, 0.05],
        'marginal_damages_pm25': [0.06, 0.07],
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_baseline_damages():
    """
    Returns a baseline damage DataFrame with columns expected
    for the scenario to compare avoided emissions/damages.
    """
    data = {
        'baseline_heating_lifetime_mt_co2e_lrmer': [50.0, 60.0],
        'baseline_heating_lifetime_damages_climate_lrmer': [100.0, 120.0],
        'baseline_heating_lifetime_mt_co2e_srmer': [70.0, 80.0],
        'baseline_heating_lifetime_damages_climate_srmer': [140.0, 160.0],
        'baseline_heating_lifetime_damages_health': [20.0, 25.0]
    }
    return pd.DataFrame(data)


def test_define_scenario_settings_valid():
    """
    Test that define_scenario_params returns expected tuple
    for baseline, pre-IRA, and IRA scenarios.
    """
    prefix, scenario, _, _, _ = define_scenario_params(0, 'No IRA')
    assert prefix == "baseline_"
    assert scenario == "MidCase"

    prefix, scenario, _, _, _ = define_scenario_params(1, 'No IRA')
    assert prefix.startswith("preIRA_mp1_")
    assert scenario == "MidCase"

    prefix, scenario, _, _, _ = define_scenario_params(2, 'AEO2023 Reference Case')
    assert prefix.startswith("iraRef_mp2_")
    assert scenario == "MidCase"


def test_define_scenario_settings_invalid():
    """
    Test that define_scenario_params raises ValueError
    for invalid policy scenario strings.
    """
    with pytest.raises(ValueError):
        define_scenario_params(1, 'Invalid Policy')


def test_precompute_hdd_factors_minimal(sample_df):
    """
    Test precompute_hdd_factors with a minimal sample df
    to ensure columns are as expected.
    """
    result = precompute_hdd_factors(sample_df)
    assert not result.empty
    # Should have columns for 2024, 2025, etc. 
    assert 2024 in result.columns
    assert 2025 in result.columns


def test_fossil_fuel_emissions_basic(sample_df, mock_lookup_emis_fossil_fuel):
    """
    Test calculate_fossil_fuel_emissions for baseline (menu_mp=0).
    """
    adjusted_hdd_factor = pd.Series(1.0, index=sample_df.index)
    result = calculate_fossil_fuel_emissions(
        df=sample_df,
        category='heating',
        adjusted_hdd_factor=adjusted_hdd_factor,
        lookup_emissions_fossil_fuel=mock_lookup_emis_fossil_fuel,
        menu_mp=0
    )
    assert set(result.keys()) == {'so2', 'nox', 'pm25', 'co2e'}
    # Natural gas consumption: [100,200] -> so2 factor=0.0001
    #  => row0 so2=0.01, row1 so2=0.02
    assert abs(result['so2'].iloc[1] - 0.02) < 1e-8


def test_climate_emissions_and_damages_preIRA(sample_df, mock_lookup_co2e_emis_electricity_preIRA):
    """
    Verify that calculate_climate_emissions_and_damages
    produces correct columns and values for a normal scenario (pre-IRA).
    """
    category = 'heating'
    year_label = 2024
    adjusted_hdd_factor = pd.Series(1.2, index=sample_df.index)
    td_losses_multiplier = 1.06
    total_fossil_emissions = {'co2e': pd.Series([10.0, 20.0], index=sample_df.index)}

    results, annual_climate_emissions, annual_climate_damages = calculate_climate_emissions_and_damages(
        df=sample_df,
        category=category,
        year_label=year_label,
        adjusted_hdd_factor=adjusted_hdd_factor,
        td_losses_multiplier=td_losses_multiplier,
        lookup_emissions_electricity_climate=mock_lookup_co2e_emis_electricity_preIRA,
        cambium_scenario='MidCase',
        EPA_SCC_USD2023_PER_MT=EPA_SCC_USD2023_PER_MT,
        total_fossil_emissions=total_fossil_emissions,
        scenario_prefix='preIRA_mp1_',
        menu_mp=1
    )
    # We expect columns like 'preIRA_mp1_2024_heating_mt_co2e_lrmer', etc.
    expected_cols = [
        'preIRA_mp1_2024_heating_mt_co2e_lrmer',
        'preIRA_mp1_2024_heating_mt_co2e_srmer',
        'preIRA_mp1_2024_heating_damages_climate_lrmer',
        'preIRA_mp1_2024_heating_damages_climate_srmer'
    ]
    for col in expected_cols:
        assert col in results


def test_health_damages_basic(sample_df, mock_lookup_health_damages_electricity_preIRA):
    """
    Test calculate_health_damages to ensure pollutant damages and total are computed.
    """
    category = 'heating'
    year_label = 2024
    adjusted_hdd_factor = pd.Series(1.0, index=sample_df.index)
    td_losses_multiplier = 1.06
    total_fossil_emissions = {
        'so2': pd.Series([5.0, 5.0], index=sample_df.index),
        'nox': pd.Series([5.0, 5.0], index=sample_df.index),
        'pm25': pd.Series([5.0, 5.0], index=sample_df.index),
        'co2e': pd.Series([0.0, 0.0], index=sample_df.index)
    }
    results, annual_health_damages = calculate_health_damages(
        df=sample_df,
        category=category,
        year_label=year_label,
        adjusted_hdd_factor=adjusted_hdd_factor,
        td_losses_multiplier=td_losses_multiplier,
        lookup_emissions_electricity_health=mock_lookup_health_damages_electricity_preIRA,
        cambium_scenario='MidCase',
        scenario_prefix='baseline_',
        total_fossil_emissions=total_fossil_emissions,
        menu_mp=0
    )
    # Check columns
    assert f'baseline_{year_label}_heating_damages_so2' in results
    assert f'baseline_{year_label}_heating_damages_health' in results
    assert annual_health_damages.sum() > 0


def test_calculate_damages_grid_scenario_empty_df():
    """
    Test that calculate_damages_grid_scenario fails with empty df.
    """
    empty_df = pd.DataFrame()
    df_baseline = pd.DataFrame()
    df_details = pd.DataFrame()

    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        calculate_damages_grid_scenario(
            df_copy=empty_df,
            df_baseline_damages_copy=df_baseline,
            df_detailed_damages=df_details,
            menu_mp=0,
            td_losses_multiplier=1.06,
            lookup_emissions_electricity_climate={},
            cambium_scenario='MidCase',
            scenario_prefix='baseline_',
            hdd_factors_df=pd.DataFrame(),
            lookup_emissions_fossil_fuel={},
            lookup_emissions_electricity_health={},
            EPA_SCC_USD2023_PER_MT=100.0,
            equipment_specs={'heating': 15}
        )


def test_calculate_marginal_damages_invalid_scenario(sample_df):
    """
    Test that calculate_marginal_damages raises ValueError
    for an invalid policy_scenario.
    """
    with pytest.raises(ValueError):
        calculate_marginal_damages(
            df=sample_df,
            menu_mp=1,
            policy_scenario="Invalid Scenario"
        )


@pytest.mark.usefixtures("patch_global_lookups")
def test_calculate_marginal_damages_valid(sample_df, sample_baseline_damages):
    """
    End-to-end test of calculate_marginal_damages with valid data
    for a pre-IRA scenario. 
    """
    df_out, df_details = calculate_marginal_damages(
        df=sample_df.copy(),
        menu_mp=1,
        policy_scenario='No IRA',
        df_baseline_damages=sample_baseline_damages.copy()
    )
    # Check for a lifetime column in df_details
    expected_col_substring = 'preIRA_mp1_heating_lifetime_mt_co2e_lrmer'
    matched_cols = [c for c in df_details.columns if expected_col_substring in c]
    assert matched_cols, "Expected new lifetime columns with preIRA_mp1_ prefix in df_details"
    assert not df_details.empty


@pytest.mark.usefixtures("patch_global_lookups")
def test_calculate_marginal_damages_missing_baseline_cols(sample_df):
    """
    Tests handling of missing baseline columns in df_baseline_damages,
    ensuring no avoided columns are added.
    """
    partial_baseline = pd.DataFrame({'baseline_heating_lifetime_damages_health': [10.0, 12.0]})
    df_out, df_details = calculate_marginal_damages(
        df=sample_df.copy(),
        menu_mp=1,
        policy_scenario='No IRA',
        df_baseline_damages=partial_baseline
    )
    avoided_cols = [c for c in df_out.columns if '_avoided_' in c]
    assert not avoided_cols, "No avoided columns should be computed if baseline is incomplete"


def test_negative_consumption(sample_df):
    """
    Tests negative consumption values, verifying the code can handle them
    without crashing.
    """
    sample_df['base_naturalGas_heating_consumption'] = [-50.0, -10.0]
    df_out, df_details = calculate_marginal_damages(
        df=sample_df,
        menu_mp=0,
        policy_scenario='No IRA'
    )
    assert not df_out.empty


def test_large_consumption(sample_df):
    """
    Test very large consumption values to check function handles
    large-scale data without overflow or performance issues.
    """
    sample_df['base_naturalGas_heating_consumption'] = [1e9, 2e9]
    df_out, df_details = calculate_marginal_damages(
        df=sample_df,
        menu_mp=0,
        policy_scenario='No IRA'
    )
    assert not df_out.empty