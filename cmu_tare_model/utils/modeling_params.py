# import functions.tare_setup as tare_setup
from cmu_tare_model.public_impact.data_processing.create_lookup_emissions_fossil_fuel import lookup_emissions_fossil_fuel
from cmu_tare_model.public_impact.data_processing.create_lookup_emissions_electricity_climate import lookup_emissions_electricity_climate
from cmu_tare_model.public_impact.data_processing.create_lookup_emissions_electricity_health import lookup_emissions_electricity_health
from cmu_tare_model.private_impact.data_processing.create_lookup_fuel_prices import lookup_fuel_prices_aeo2026
from typing import Tuple, Dict

from cmu_tare_model.constants import VERBOSE

def define_scenario_params(
    menu_mp: int,
    policy_scenario: str = "2025 Reference Case",
    verbose: bool = VERBOSE
) -> Tuple[str, str, Dict, Dict, Dict, Dict]:
    """
    Define scenario-specific params based on a measure package.

    There is one policy scenario: 2025 Reference Case.
    The policy_scenario argument is accepted for API compatibility but not used
    for scenario routing; it is passed through as the lookup key so callers
    must supply the exact string "2025 Reference Case".

    Args:
        menu_mp (int): Measure package identifier (0 indicates baseline).
        policy_scenario (str): Must equal "2025 Reference Case".
            Accepted for backward-compatibility; routing is single-scenario only.
        verbose (bool): Whether to print scenario configuration details.

    Returns:
        tuple:
            str: scenario_prefix  ('baseline_' for mp=0, 'ref2025_mp{mp}_' otherwise)
            str: cambium_scenario ('MidCase')
            dict: lookup_emissions_fossil_fuel
            dict: lookup_emissions_electricity_climate
            dict: lookup_emissions_electricity_health
            dict: lookup_fuel_prices_aeo2026

    Raises:
        ValueError: If menu_mp is not an integer.
    """
    if menu_mp == 0:
        scenario_prefix = "baseline_"
        if verbose:
            print(f"-- Scenario: Baseline (mp{menu_mp}) --")
    else:
        scenario_prefix = f"ref2025_mp{menu_mp}_"
        if verbose:
            print(f"-- Scenario: 2025 Reference Case (mp{menu_mp}) --")

    return (
        scenario_prefix,
        "MidCase",
        lookup_emissions_fossil_fuel,
        lookup_emissions_electricity_climate,
        lookup_emissions_electricity_health,
        lookup_fuel_prices_aeo2026,
    )

