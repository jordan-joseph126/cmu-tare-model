"""Tests for cmu_tare_model.public_impact.data_processing.create_lookup_emissions_fossil_fuel module.

Verifies the emission factor calculation function and the resulting lookup
dictionary structure used throughout the public impact calculations.
"""

import pytest


# ── calculate_fossil_fuel_emission_factor ────────────────────────────────────

def test_natural_gas_emission_factor_keys():
    from cmu_tare_model.public_impact.data_processing.create_lookup_emissions_fossil_fuel import (
        calculate_fossil_fuel_emission_factor,
    )
    result = calculate_fossil_fuel_emission_factor(
        fuel_type="naturalGas",
        so2_factor=0.0006,
        nox_factor=0.0922,
        pm25_factor=1.9,
        conversion_factor1=1_000_000,
        conversion_factor2=1039,
    )
    assert 'naturalGas_so2' in result
    assert 'naturalGas_nox' in result
    assert 'naturalGas_pm25' in result
    assert 'naturalGas_co2e' in result


def test_fuel_oil_emission_factor_keys():
    from cmu_tare_model.public_impact.data_processing.create_lookup_emissions_fossil_fuel import (
        calculate_fossil_fuel_emission_factor,
    )
    result = calculate_fossil_fuel_emission_factor(
        fuel_type="fuelOil",
        so2_factor=0.0015,
        nox_factor=0.1300,
        pm25_factor=0.83,
        conversion_factor1=1000,
        conversion_factor2=138_500,
    )
    assert 'fuelOil_so2' in result
    assert 'fuelOil_co2e' in result


def test_propane_emission_factor_keys():
    from cmu_tare_model.public_impact.data_processing.create_lookup_emissions_fossil_fuel import (
        calculate_fossil_fuel_emission_factor,
    )
    result = calculate_fossil_fuel_emission_factor(
        fuel_type="propane",
        so2_factor=0.0002,
        nox_factor=0.1421,
        pm25_factor=0.17,
        conversion_factor1=1000,
        conversion_factor2=91_452,
    )
    assert 'propane_so2' in result
    assert 'propane_co2e' in result


def test_emission_factors_are_positive():
    from cmu_tare_model.public_impact.data_processing.create_lookup_emissions_fossil_fuel import (
        calculate_fossil_fuel_emission_factor,
    )
    result = calculate_fossil_fuel_emission_factor(
        fuel_type="naturalGas",
        so2_factor=0.0006,
        nox_factor=0.0922,
        pm25_factor=1.9,
        conversion_factor1=1_000_000,
        conversion_factor2=1039,
    )
    for key, value in result.items():
        assert value > 0, f"Emission factor {key} should be positive, got {value}"


def test_co2e_factor_natural_gas():
    """Natural gas CO2e should be 228.5 kg/MWh converted to mt/kWh."""
    from cmu_tare_model.public_impact.data_processing.create_lookup_emissions_fossil_fuel import (
        calculate_fossil_fuel_emission_factor,
    )
    result = calculate_fossil_fuel_emission_factor(
        fuel_type="naturalGas",
        so2_factor=0.0006,
        nox_factor=0.0922,
        pm25_factor=1.9,
        conversion_factor1=1_000_000,
        conversion_factor2=1039,
    )
    expected_co2e = 228.5 * (1 / 1000) * (1 / 1000)  # kg/MWh -> mt/kWh
    assert result['naturalGas_co2e'] == pytest.approx(expected_co2e)


def test_co2e_factor_fuel_oil():
    """Fuel oil CO2e should be 303.9 kg/MWh converted to mt/kWh."""
    from cmu_tare_model.public_impact.data_processing.create_lookup_emissions_fossil_fuel import (
        calculate_fossil_fuel_emission_factor,
    )
    result = calculate_fossil_fuel_emission_factor(
        fuel_type="fuelOil",
        so2_factor=0.0015,
        nox_factor=0.1300,
        pm25_factor=0.83,
        conversion_factor1=1000,
        conversion_factor2=138_500,
    )
    expected_co2e = 303.9 * (1 / 1000) * (1 / 1000)
    assert result['fuelOil_co2e'] == pytest.approx(expected_co2e)


# ── Module-level lookup dictionary ───────────────────────────────────────────

def test_lookup_emissions_fossil_fuel_structure():
    """The module-level lookup_emissions_fossil_fuel dict should have correct structure."""
    from cmu_tare_model.public_impact.data_processing.create_lookup_emissions_fossil_fuel import (
        lookup_emissions_fossil_fuel,
    )
    assert isinstance(lookup_emissions_fossil_fuel, dict)
    expected_fuels = ['fuelOil', 'naturalGas', 'propane']
    for fuel in expected_fuels:
        assert fuel in lookup_emissions_fossil_fuel, f"Missing fuel: {fuel}"
        for pollutant in ['so2', 'nox', 'pm25', 'co2e']:
            assert pollutant in lookup_emissions_fossil_fuel[fuel], \
                f"Missing pollutant {pollutant} for {fuel}"
            assert lookup_emissions_fossil_fuel[fuel][pollutant] > 0


def test_lookup_emissions_fuel_oil_higher_co2e_than_natural_gas():
    """Fuel oil should have higher CO2e per kWh than natural gas."""
    from cmu_tare_model.public_impact.data_processing.create_lookup_emissions_fossil_fuel import (
        lookup_emissions_fossil_fuel,
    )
    assert lookup_emissions_fossil_fuel['fuelOil']['co2e'] > \
           lookup_emissions_fossil_fuel['naturalGas']['co2e']


def test_so2_nox_pm25_in_metric_tons():
    """SO2, NOx, PM2.5 values should be very small (metric tons/kWh)."""
    from cmu_tare_model.public_impact.data_processing.create_lookup_emissions_fossil_fuel import (
        lookup_emissions_fossil_fuel,
    )
    for fuel in ['naturalGas', 'propane', 'fuelOil']:
        for pollutant in ['so2', 'nox', 'pm25']:
            val = lookup_emissions_fossil_fuel[fuel][pollutant]
            assert val < 1e-6, f"{fuel} {pollutant} = {val} seems too large for mt/kWh"
