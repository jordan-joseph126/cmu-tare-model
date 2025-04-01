import pandas as pd
from cmu_tare_model.constants import POLLUTANTS
from typing import Dict

def calculate_fossil_fuel_emissions(
    df: pd.DataFrame,
    category: str,
    adjusted_hdd_factor: float,
    lookup_emissions_fossil_fuel: Dict[tuple, float],
    menu_mp: int
) -> Dict[str, pd.Series]:
    """
    Calculate fossil fuel emissions (SO2, NOx, PM2.5, and CO2e) for a given category and scenario.

    Args:
        df (pd.DataFrame): Main DataFrame containing region- and fuel-specific consumption data.
        category (str): Category of energy use (e.g., 'heating', 'cooking').
        adjusted_hdd_factor (float): Adjustment factor for heating degree days.
        lookup_emissions_fossil_fuel (dict): Dictionary mapping (fuel, pollutant) → emission factor.
        menu_mp (int): Measure package identifier (0 indicates baseline).

    Returns:
        dict: A dictionary where each key is a pollutant (str) and each value is a pd.Series
              representing emissions for that pollutant.

    Raises:
        None

    Note 03/31/2025:
    Fixed TARE.V2 code now uses the nested dictionary structure and properly retrieves emission factors based on the fuel type and pollutant.
    Previously, the code was trying to access the emissions factors using a tuple
        (fuel, pollutant) as a single key, which would not work thus returning 0.

    Structure of the lookup_emissions_fossil_fuel dictionary: lookup_emissions_fossil_fuel[fuel][pollutant]

    So: lookup_emissions_fossil_fuel.get(fuel, {}).get(pollutant, 0)
        - first key is the fuel type (e.g., 'naturalGas', 'fuelOil', 'propane')
        - second key is the pollutant (e.g., 'so2', 'nox', 'pm25', 'co2e').
    """
    total_fossil_emissions = {p: pd.Series(0.0, index=df.index) for p in POLLUTANTS}

    if menu_mp == 0:
        fuels = ['naturalGas', 'propane']
        if category not in ['cooking', 'clothesDrying']:
            fuels.append('fuelOil')

        for fuel in fuels:
            consumption_col = f'base_{fuel}_{category}_consumption'
            fuel_consumption = df.get(consumption_col, pd.Series(0.0, index=df.index)).fillna(0) * adjusted_hdd_factor

            # Fixed issue with the emissions factor lookup
            for pollutant in POLLUTANTS:
                emis_factor = lookup_emissions_fossil_fuel.get(fuel, {}).get(pollutant, 0)
                total_fossil_emissions[pollutant] += fuel_consumption * emis_factor

    return total_fossil_emissions
