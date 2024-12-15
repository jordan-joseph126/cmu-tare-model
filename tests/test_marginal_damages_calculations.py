import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from unittest.mock import patch

# Updated import statement:
from tare_model_functions_v1_5_1 import (
    calculate_marginal_damages,
    define_scenario_settings,
    precompute_hdd_factors,
    calculate_damages_grid_scenario,
    calculate_fossil_fuel_emissions,
    calculate_climate_emissions_and_damages,
    calculate_health_damages,
    TD_LOSSES,
    TD_LOSSES_MULTIPLIER,
    EQUIPMENT_SPECS,
    POLLUTANTS
)

#### FIXTURES ####

@pytest.fixture
def mock_emis_fossil_fuel_lookup():
    """
    Provides a minimal mock dictionary for fossil fuel emission factors.
    Keys are tuples (fuel, pollutant), values are emission factors.
    """
    return {
        ('naturalGas', 'so2'): 0.0001,
        ('naturalGas', 'nox'): 0.0002,
        ('naturalGas', 'pm25'): 0.00005,
        ('naturalGas', 'co2e'): 0.05,
        ('propane', 'so2'): 0.0001,
        ('propane', 'nox'): 0.0002,
        ('propane', 'pm25'): 0.00005,
        ('propane', 'co2e'): 0.05,
        ('fuelOil', 'so2'): 0.0001,
        ('fuelOil', 'nox'): 0.0002,
        ('fuelOil', 'pm25'): 0.00005,
        ('fuelOil', 'co2e'): 0.05,
    }

@pytest.fixture
def mock_emis_preIRA_co2e_cambium21_lookup():
    """
    Mock lookup for pre-IRA electricity emission factors by region and year.
    Here we only define 'MidCase' scenario and 'region1' for a couple of years.
    """
    return {
        ('MidCase', 'region1'): {
            2024: {'lrmer_ton_per_kWh_co2e': 0.0005, 'srmer_ton_per_kWh_co2e': 0.0004},
            2025: {'lrmer_ton_per_kWh_co2e': 0.0004, 'srmer_ton_per_kWh_co2e': 0.0003}
        }
    }

@pytest.fixture
def mock_emis_IRA_co2e_cambium22_lookup():
    """
    Mock lookup for IRA scenario electricity emission factors.
    """
    return {
        ('MidCase', 'region1'): {
            2024: {'lrmer_ton_per_kWh_co2e': 0.0003, 'srmer_ton_per_kWh_co2e': 0.0002},
            2025: {'lrmer_ton_per_kWh_co2e': 0.0002, 'srmer_ton_per_kWh_co2e': 0.0001}
        }
    }

@pytest.fixture
def mock_damages_preIRA_health_damages_lookup():
    """
    Mock lookup for health damages (so2, nox, pm25) in pre-IRA scenario.
    """
    return {
        ('MidCase', 'region1'): {
            2024: {'so2_dollarPerkWh_adjustVSL': 0.001, 'nox_dollarPerkWh_adjustVSL': 0.002, 'pm25_dollarPerkWh_adjustVSL': 0.003},
            2025: {'so2_dollarPerkWh_adjustVSL': 0.001, 'nox_dollarPerkWh_adjustVSL': 0.002, 'pm25_dollarPerkWh_adjustVSL': 0.003}
        }
    }

@pytest.fixture
def mock_damages_iraRef_health_damages_lookup():
    """
    Mock lookup for health damages in IRA scenario.
    """
    return {
        ('MidCase', 'region1'): {
            2024: {'so2_dollarPerkWh_adjustVSL': 0.0008, 'nox_dollarPerkWh_adjustVSL': 0.0015, 'pm25_dollarPerkWh_adjustVSL': 0.0025},
            2025: {'so2_dollarPerkWh_adjustVSL': 0.0008, 'nox_dollarPerkWh_adjustVSL': 0.0015, 'pm25_dollarPerkWh_adjustVSL': 0.0025}
        }
    }

@pytest.fixture
def mock_hdd_factor_lookup():
    """
    Mock HDD (Heating Degree Day) factor lookup. 
    Here we use a simple scenario where all years have a factor of 1.0.
    """
    return {
        'National': {
            2024: 1.0,
            2025: 1.0
        }
    }

@pytest.fixture
def mock_df():
    """
    Provide a minimal input DataFrame expected by calculate_marginal_damages.
    Includes columns for baseline fossil fuel and electricity consumption, 
    and marginal damages for pollutants. Two rows simulate two different sites/regions.
    """
    data = {
        'census_division': ['National', 'National'],
        'gea_region': ['region1', 'region1'],
        'base_naturalGas_heating_consumption': [100, 200],
        'base_fuelOil_heating_consumption': [50, 100],
        'base_propane_heating_consumption': [20, 40],
        'base_electricity_heating_consumption': [500, 1000],
        'marginal_damages_so2': [10, 10],
        'marginal_damages_nox': [20, 20],
        'marginal_damages_pm25': [30, 30]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_df_baseline_damages():
    """
    Provide baseline damages DataFrame, used when menu_mp != 0.
    Contains baseline lifetime emissions and damages so that we can compute avoided damages.
    """
    data = {
        'baseline_heating_lifetime_tons_co2e_lrmer': [50, 100],
        'baseline_heating_lifetime_damages_climate_lrmer': [5000, 10000],
        'baseline_heating_lifetime_tons_co2e_srmer': [40, 80],
        'baseline_heating_lifetime_damages_climate_srmer': [4000, 8000],
        'baseline_heating_lifetime_damages_health': [3000, 6000],
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_epa_scc():
    """
    Mock EPA social cost of carbon per ton for 2023.
    Assume a CPI ratio and compute a mock SCC value.
    """
    cpi_ratio_2023_2020 = 1.1
    EPA_SCC_USD2023_PER_TON = 190 * cpi_ratio_2023_2020
    return EPA_SCC_USD2023_PER_TON

#### TESTS ####

def test_calculate_marginal_damages_no_baseline(mock_df, mock_epa_scc, mock_hdd_factor_lookup,
                                                mock_emis_fossil_fuel_lookup,
                                                mock_emis_preIRA_co2e_cambium21_lookup,
                                                mock_emis_IRA_co2e_cambium22_lookup,
                                                mock_damages_preIRA_health_damages_lookup,
                                                mock_damages_iraRef_health_damages_lookup):
    """
    Test the main function with menu_mp=0 and no baseline damages, for two possible policy scenarios:
    'No Inflation Reduction Act' and 'AEO2023 Reference Case'.

    This test uses pytest parameterization to run the same test twice with different scenarios.
    """
    from tare_model_functions_v1_5_1 import (hdd_factor_lookup, emis_fossil_fuel_lookup, 
                                             emis_preIRA_co2e_cambium21_lookup, damages_preIRA_health_damages_lookup, 
                                             emis_IRA_co2e_cambium22_lookup, damages_iraRef_health_damages_lookup, EPA_SCC_USD2023_PER_TON)

    # Update global lookups with mock data
    hdd_factor_lookup.clear()
    hdd_factor_lookup.update(mock_hdd_factor_lookup)

    emis_fossil_fuel_lookup.clear()
    emis_fossil_fuel_lookup.update(mock_emis_fossil_fuel_lookup)

    emis_preIRA_co2e_cambium21_lookup.clear()
    emis_preIRA_co2e_cambium21_lookup.update(mock_emis_preIRA_co2e_cambium21_lookup)

    damages_preIRA_health_damages_lookup.clear()
    damages_preIRA_health_damages_lookup.update(mock_damages_preIRA_health_damages_lookup)

    emis_IRA_co2e_cambium22_lookup.clear()
    emis_IRA_co2e_cambium22_lookup.update(mock_emis_IRA_co2e_cambium22_lookup)

    damages_iraRef_health_damages_lookup.clear()
    damages_iraRef_health_damages_lookup.update(mock_damages_iraRef_health_damages_lookup)

    EPA_SCC_USD2023_PER_TON = mock_epa_scc

    # Test both scenarios
    for policy_scenario in ["No Inflation Reduction Act", "AEO2023 Reference Case"]:
        df_copy, df_detailed = calculate_marginal_damages(
            df=mock_df,
            menu_mp=0,
            policy_scenario=policy_scenario,
            df_baseline_damages=None,
            df_detailed_damages=None
        )

        # Basic checks
        assert isinstance(df_copy, pd.DataFrame)
        assert isinstance(df_detailed, pd.DataFrame)

        # Since menu_mp=0 and no baseline, we don't expect avoided columns
        assert not any('avoided' in c for c in df_copy.columns)

        # Check that lifetime damage columns appear
        # For the baseline scenario, columns typically start with "baseline_"
        # For menu_mp=0, scenario_prefix would be "baseline_"
        expected_col = 'baseline_heating_lifetime_damages_health'
        assert expected_col in df_detailed.columns or expected_col in df_copy.columns, \
            f"Expected column {expected_col} not found in output data."

def test_calculate_marginal_damages_with_baseline(mock_df, mock_df_baseline_damages, mock_epa_scc,
                                                  mock_hdd_factor_lookup, mock_emis_fossil_fuel_lookup,
                                                  mock_emis_preIRA_co2e_cambium21_lookup,
                                                  mock_damages_preIRA_health_damages_lookup):
    """
    Test calculate_marginal_damages with menu_mp=1 (not baseline) and baseline damages provided.
    This should produce 'avoided' emissions and damages columns in the results.
    """
    from tare_model_functions_v1_5_1 import (hdd_factor_lookup, emis_fossil_fuel_lookup, emis_preIRA_co2e_cambium21_lookup, 
                                             damages_preIRA_health_damages_lookup, EPA_SCC_USD2023_PER_TON)

    # Set up scenario lookups for "No Inflation Reduction Act"
    hdd_factor_lookup.clear()
    hdd_factor_lookup.update(mock_hdd_factor_lookup)

    emis_fossil_fuel_lookup.clear()
    emis_fossil_fuel_lookup.update(mock_emis_fossil_fuel_lookup)

    emis_preIRA_co2e_cambium21_lookup.clear()
    emis_preIRA_co2e_cambium21_lookup.update(mock_emis_preIRA_co2e_cambium21_lookup)

    damages_preIRA_health_damages_lookup.clear()
    damages_preIRA_health_damages_lookup.update(mock_damages_preIRA_health_damages_lookup)

    EPA_SCC_USD2023_PER_TON = mock_epa_scc

    # Running with menu_mp=1 to simulate a scenario with different measures applied
    df_copy, df_detailed = calculate_marginal_damages(
        df=mock_df,
        menu_mp=1,
        policy_scenario="No Inflation Reduction Act",
        df_baseline_damages=mock_df_baseline_damages,
        df_detailed_damages=None
    )

    assert isinstance(df_copy, pd.DataFrame)
    assert isinstance(df_detailed, pd.DataFrame)

    # Check that avoided columns are created when baseline damages are provided
    avoided_cols = [c for c in df_copy.columns if 'avoided' in c]
    assert len(avoided_cols) > 0, "Expected avoided emissions/damages columns not found."

def test_invalid_policy_scenario(mock_df, mock_emis_fossil_fuel_lookup, mock_emis_preIRA_co2e_cambium21_lookup, 
                                 mock_damages_preIRA_health_damages_lookup, mock_hdd_factor_lookup, mock_epa_scc):
    """
    Test that passing an invalid policy scenario to calculate_marginal_damages 
    raises a ValueError as expected.
    """
    from tare_model_functions_v1_5_1 import (hdd_factor_lookup, emis_fossil_fuel_lookup, emis_preIRA_co2e_cambium21_lookup, 
                                             damages_preIRA_health_damages_lookup, EPA_SCC_USD2023_PER_TON)

    hdd_factor_lookup.clear()
    hdd_factor_lookup.update(mock_hdd_factor_lookup)

    emis_fossil_fuel_lookup.clear()
    emis_fossil_fuel_lookup.update(mock_emis_fossil_fuel_lookup)

    emis_preIRA_co2e_cambium21_lookup.clear()
    emis_preIRA_co2e_cambium21_lookup.update(mock_emis_preIRA_co2e_cambium21_lookup)

    damages_preIRA_health_damages_lookup.clear()
    damages_preIRA_health_damages_lookup.update(mock_damages_preIRA_health_damages_lookup)

    EPA_SCC_USD2023_PER_TON = mock_epa_scc

    # Try an invalid policy scenario
    with pytest.raises(ValueError):
        calculate_marginal_damages(
            df=mock_df,
            menu_mp=0,
            policy_scenario="Invalid Scenario",
            df_baseline_damages=None,
            df_detailed_damages=None
        )