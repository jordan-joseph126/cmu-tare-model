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
    """
    total_fossil_emissions = {p: pd.Series(0.0, index=df.index) for p in POLLUTANTS}

    if menu_mp == 0:
        fuels = ['naturalGas', 'propane']
        if category not in ['cooking', 'clothesDrying']:
            fuels.append('fuelOil')

        for fuel in fuels:
            consumption_col = f'base_{fuel}_{category}_consumption'
            fuel_consumption = df.get(consumption_col, pd.Series(0.0, index=df.index)).fillna(0) * adjusted_hdd_factor

            for pollutant in POLLUTANTS:
                emis_factor = lookup_emissions_fossil_fuel.get((fuel, pollutant), 0)
                total_fossil_emissions[pollutant] += fuel_consumption * emis_factor

    return total_fossil_emissions

# def calculate_fossil_fuel_emissions(df, category, adjusted_hdd_factor, lookup_emissions_fossil_fuel, menu_mp):
#     """
#     Calculate fossil fuel emissions for a given row and category.
#     """

#     total_fossil_emissions = {pollutant: pd.Series(0.0, index=df.index) for pollutant in POLLUTANTS}

#     if menu_mp == 0:
#         fuels = ['naturalGas', 'propane']
#         if category not in ['cooking', 'clothesDrying']:
#             fuels.append('fuelOil')

#         for fuel in fuels:
#             consumption_col = f'base_{fuel}_{category}_consumption'
#             fuel_consumption = df.get(consumption_col, pd.Series(0.0, index=df.index)).fillna(0) * adjusted_hdd_factor

#             for pollutant in total_fossil_emissions.keys():
#                 emis_factor = lookup_emissions_fossil_fuel.get((fuel, pollutant), 0)
#                 total_fossil_emissions[pollutant] += fuel_consumption * emis_factor

#     return total_fossil_emissions