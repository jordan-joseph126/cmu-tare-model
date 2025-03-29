# import functions.tare_setup as tare_setup
from cmu_tare_model.public_impact.data_processing.create_lookup_emissions_fossil_fuel import lookup_emissions_fossil_fuel
from cmu_tare_model.public_impact.data_processing.create_lookup_emissions_electricity_climate import lookup_emissions_electricity_climate_preIRA, lookup_emissions_electricity_climate_IRA
from cmu_tare_model.public_impact.data_processing.create_lookup_emissions_electricity_health import lookup_emissions_electricity_health
from typing import Tuple, Dict

def define_scenario_settings(menu_mp: int, policy_scenario: str) -> Tuple[str, str, Dict, Dict, Dict]:
    """
    Define scenario-specific settings based on a measure package and policy scenario.

    Args:
        menu_mp (int): Measure package identifier (0 indicates baseline).
        policy_scenario (str): Policy scenario name ('No Inflation Reduction Act' or 'AEO2023 Reference Case').

    Returns:
        tuple:
            str: `scenario_prefix` used in output column naming.
            str: `cambium_scenario` describing the chosen Cambium scenario (e.g., 'MidCase').
            dict: `lookup_emissions_fossil_fuel` for fossil fuel emission factors.
            dict: `lookup_emissions_electricity_climate` for electricity climate factors.
            dict: `lookup_emissions_electricity_health` for electricity health damage factors.

    Raises:
        ValueError: If the policy_scenario is invalid.
    """

    if menu_mp == 0:
        print(f"""-- Scenario: Baseline -- 
              scenario_prefix: 'baseline_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health'
              """)
        return "baseline_", "MidCase", lookup_emissions_fossil_fuel, lookup_emissions_electricity_climate_preIRA, lookup_emissions_electricity_health

    if policy_scenario == 'No Inflation Reduction Act':
        print(f"""-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp{menu_mp}_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health'
              """)
        return f"preIRA_mp{menu_mp}_", "MidCase", lookup_emissions_fossil_fuel, lookup_emissions_electricity_climate_preIRA, lookup_emissions_electricity_health

    if policy_scenario == 'AEO2023 Reference Case':
        print(f"""-- Scenario: Inflation Reduction Act (IRA) Reference -- 
              scenario_prefix: 'iraRef_mp{menu_mp}_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel', 
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health'
              """)
        return f"iraRef_mp{menu_mp}_", "MidCase", lookup_emissions_fossil_fuel, lookup_emissions_electricity_climate_IRA, lookup_emissions_electricity_health

    raise ValueError("Invalid Policy Scenario! Choose 'No Inflation Reduction Act' or 'AEO2023 Reference Case'.")