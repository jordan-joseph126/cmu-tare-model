import pandas as pd
from typing import Dict, Optional

from cmu_tare_model.constants import POLLUTANTS, EQUIPMENT_SPECS
from cmu_tare_model.utils.hdd_consumption_utils import (
    get_hdd_factor_for_year,
    apply_hdd_adjustment
)
from cmu_tare_model.utils.validation_framework import (
    get_retrofit_homes_mask,
    create_retrofit_only_series,
)


def calculate_fossil_fuel_emissions(
    df: pd.DataFrame,
    category: str,
    year_label: int,
    lookup_emissions_fossil_fuel: Dict[str, Dict[str, float]],
    menu_mp: int,
    retrofit_mask: Optional[pd.Series] = None,
    verbose: bool = False
) -> Dict[str, pd.Series]:
    """
    Calculate fossil fuel emissions (SO2, NOx, PM2.5, and CO2e) for a given category and scenario.

    Args:
        df (pd.DataFrame): DataFrame containing region- and fuel-specific consumption data.
        category (str): Category of energy use (e.g., 'heating', 'cooking').
        year_label (int): The year for calculation (e.g., 2024).
        lookup_emissions_fossil_fuel (Dict[str, Dict[str, float]]): 
            Dictionary mapping fuel → (pollutant → emission factor).
        menu_mp (int): Measure package identifier (0 indicates baseline).
        retrofit_mask (Optional[pd.Series]): Pre-computed retrofit mask. If None, it will be calculated.
        verbose (bool): Whether to print detailed information.

    Returns:
        dict: A dictionary where each key is a pollutant (str) and each value is a pd.Series
              representing emissions for that pollutant.

    Raises:
        ValueError: If category is invalid or menu_mp is negative.
        KeyError: If required consumption columns are missing.
    """
    # Validate inputs
    if category not in EQUIPMENT_SPECS:
        raise ValueError(f"Invalid category: {category}. Must be one of {list(EQUIPMENT_SPECS.keys())}")
    if not isinstance(menu_mp, int) or menu_mp < 0:
        raise ValueError(f"Invalid menu_mp: {menu_mp}. Must be a non-negative integer.")

    # Determine retrofit mask
    if retrofit_mask is None:
        retrofit_mask = get_retrofit_homes_mask(df, category, menu_mp, verbose=verbose)

    # Initialize emissions series: zeros for retrofit homes, NaN elsewhere
    total_fossil_emissions = {
        pollutant: create_retrofit_only_series(df, retrofit_mask, verbose=False)
        for pollutant in POLLUTANTS
    }

    # Only calculate baseline fossil fuel emissions (menu_mp=0)
    if menu_mp == 0:
        # Prepare HDD factor for the year
        hdd_factor = get_hdd_factor_for_year(df, year_label)

        # Determine applicable fuels for this category
        fuels = ['naturalGas', 'propane']
        if category not in ['cooking', 'clothesDrying']:
            fuels.append('fuelOil')

        for fuel in fuels:
            consumption_col = f'base_{fuel}_{category}_consumption'
            if consumption_col not in df.columns:
                raise KeyError(f"Required column '{consumption_col}' not found in DataFrame")

            # Base consumption, filling missing values with zero
            fuel_consumption = df[consumption_col].fillna(0)

            # Apply HDD adjustment (only affects 'heating' category)
            fuel_consumption = apply_hdd_adjustment(fuel_consumption, category, hdd_factor)

            # Compute emissions for each pollutant
            for pollutant in POLLUTANTS:
                emis_factor = lookup_emissions_fossil_fuel.get(fuel, {}).get(pollutant, 0)
                total_fossil_emissions[pollutant] += fuel_consumption * emis_factor

    return total_fossil_emissions
