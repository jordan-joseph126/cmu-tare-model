import pandas as pd
from typing import Dict, Optional

from cmu_tare_model.constants import POLLUTANTS, EQUIPMENT_SPECS, VERBOSE
from cmu_tare_model.utils.column_names import (
    create_annual_consumption_col,
    create_annual_fuel_consumption_col,
)
from cmu_tare_model.utils.validation_framework import (
    get_retrofit_homes_mask,
    create_retrofit_only_series,
)

# Fuels burned on site. Electricity is covered separately, by grid emission factors.
FOSSIL_FUELS = ('naturalGas', 'propane', 'fuelOil')


def calculate_fossil_fuel_emissions(
    df: pd.DataFrame,
    category: str,
    year_label: int,
    lookup_emissions_fossil_fuel: Dict[str, Dict[str, float]],
    menu_mp: int,
    df_consumption: pd.DataFrame,
    scenario_prefix: str,
    retrofit_mask: Optional[pd.Series] = None,
    verbose: bool = VERBOSE
) -> Dict[str, pd.Series]:
    """
    Calculate fossil fuel emissions (SO2, NOx, PM2.5, and CO2e) for a given category and scenario.

    Reads this scenario's natural gas, propane, and fuel oil use from
    df_consumption (already degree-day adjusted, every component counted), for
    the baseline and measure packages alike: a dual-fuel heat pump's backup
    furnace still burns gas after the retrofit. For an all-electric package
    these columns are zero.

    Args:
        df (pd.DataFrame): DataFrame with the inclusion flags used for masking.
        category (str): Category of energy use (e.g., 'heating').
        year_label (int): The year for calculation (e.g., 2025).
        lookup_emissions_fossil_fuel (Dict[str, Dict[str, float]]):
            Dictionary mapping fuel -> (pollutant -> emission factor).
        menu_mp (int): Measure package identifier (0 indicates baseline).
        df_consumption (pd.DataFrame): This scenario's table from
            build_projected_consumption, indexed like df.
        scenario_prefix (str): Scenario prefix of the table's columns
            (e.g. 'baseline_', 'ref2025_mp3_').
        retrofit_mask (Optional[pd.Series]): Pre-computed retrofit mask. If None, it will be calculated.
        verbose (bool): Whether to print detailed information.

    Returns:
        dict: A dictionary where each key is a pollutant (str) and each value is a pd.Series
              representing emissions for that pollutant.

    Raises:
        ValueError: If category is invalid, menu_mp is negative, or
            df_consumption has no columns for this scenario, category and year.
    """
    # Validate inputs
    if category not in EQUIPMENT_SPECS:
        raise ValueError(f"Invalid category: {category}. Must be one of {list(EQUIPMENT_SPECS.keys())}")
    if not isinstance(menu_mp, int) or menu_mp < 0:
        raise ValueError(f"Invalid menu_mp: {menu_mp}. Must be a non-negative integer.")
    total_col = create_annual_consumption_col(scenario_prefix, year_label, category)
    if total_col not in df_consumption.columns:
        raise ValueError(
            f"df_consumption has no '{total_col}' column; build it with "
            f"build_projected_consumption for menu_mp={menu_mp}.")

    # Determine retrofit mask
    if retrofit_mask is None:
        retrofit_mask = get_retrofit_homes_mask(df, category, menu_mp, verbose=verbose)

    # Initialize emissions series: zeros for retrofit homes, NaN elsewhere
    total_fossil_emissions = {
        pollutant: create_retrofit_only_series(df, retrofit_mask, verbose=verbose)
        for pollutant in POLLUTANTS
    }

    # A category's table has a column only for fuels it can use (cooling has
    # electricity only), so a missing fossil column means zero use.
    for fuel in FOSSIL_FUELS:
        consumption_col = create_annual_fuel_consumption_col(
            scenario_prefix, year_label, category, fuel)
        if consumption_col not in df_consumption.columns:
            continue
        fuel_consumption = df_consumption[consumption_col].fillna(0)
        for pollutant in POLLUTANTS:
            emis_factor = lookup_emissions_fossil_fuel.get(fuel, {}).get(pollutant, 0)
            total_fossil_emissions[pollutant] += fuel_consumption * emis_factor

    return total_fossil_emissions
