"""
Stored per-home energy use, by fuel and year, for one scenario.

The single source for every later step that needs energy use: fuel costs,
emissions, and the whole-home savings fraction. Each value counts a fuel's full
use for heating or cooling -- primary energy plus fans, pumps, and heat-pump
backup -- scaled to each year by the heating or cooling degree-day factor.
"""

import pandas as pd

from cmu_tare_model.constants import ANCHOR_YEAR, EQUIPMENT_SPECS, VERBOSE
from cmu_tare_model.utils.column_names import (
    create_annual_consumption_col,
    create_annual_fuel_consumption_col,
)
from cmu_tare_model.utils.degree_day_consumption_utils import (
    get_degree_day_adjusted_consumption_by_fuel,
)
from cmu_tare_model.utils.modeling_params import define_scenario_params
from cmu_tare_model.utils.validation_framework import get_valid_calculation_mask


def build_projected_consumption(
    df: pd.DataFrame,
    menu_mp: int,
    verbose: bool = VERBOSE,
) -> pd.DataFrame:
    """Builds the per-home, per-fuel, per-year consumption table for one scenario.

    For each category in EQUIPMENT_SPECS and each year of its equipment lifetime
    (ANCHOR_YEAR onward, the same years fuel costs and emissions use), writes:
      - one column per fuel, from create_annual_fuel_consumption_col, and
      - a total column, from create_annual_consumption_col (the name fuel costs
        already writes, so existing readers keep working).

    Homes outside the calculation for a category (get_valid_calculation_mask:
    invalid data, or not retrofitted in a measure package) are NaN in every
    column for that category. Valid homes have a number in every column; a fuel
    the home does not use is 0.

    Args:
        df: Scenario DataFrame from df_enduse_refactored (baseline) or
            df_enduse_compare (measure package), with census_division and every
            component column get_consumption_component_columns names.
        menu_mp: Measure package number; 0 for the baseline.
        verbose: Whether to print progress.

    Returns:
        DataFrame indexed like df, one column per (year, category, fuel) plus
        one total per (year, category), in kWh.

    Raises:
        TypeError: If menu_mp is not an integer.
        ValueError: If menu_mp is negative.
        KeyError: If a component column or census_division is missing.
    """
    if not isinstance(menu_mp, int) or isinstance(menu_mp, bool):
        raise TypeError(f"menu_mp must be an integer, got {type(menu_mp).__name__}")
    if menu_mp < 0:
        raise ValueError(f"menu_mp must be 0 or greater, got {menu_mp}")

    scenario_prefix = define_scenario_params(menu_mp, verbose=False)[0]
    columns = {}

    for category, lifetime in EQUIPMENT_SPECS.items():
        valid_mask = get_valid_calculation_mask(df, category, menu_mp, verbose=verbose)
        last_year = ANCHOR_YEAR + lifetime - 1
        if verbose:
            print(f"Projecting {category} consumption {ANCHOR_YEAR}-{last_year} "
                  f"for {int(valid_mask.sum()):,} valid homes")

        for year_label in range(ANCHOR_YEAR, last_year + 1):
            consumption_by_fuel = get_degree_day_adjusted_consumption_by_fuel(
                df, category, year_label, menu_mp)

            # A valid home's missing value means that fuel is unused, so it
            # counts as 0; homes outside the calculation stay NaN.
            total = pd.Series(0.0, index=df.index)
            for fuel, fuel_consumption in consumption_by_fuel.items():
                fuel_consumption = fuel_consumption.fillna(0.0)
                total += fuel_consumption
                fuel_col = create_annual_fuel_consumption_col(
                    scenario_prefix, year_label, category, fuel)
                columns[fuel_col] = fuel_consumption.where(valid_mask)

            total_col = create_annual_consumption_col(
                scenario_prefix, year_label, category)
            columns[total_col] = total.where(valid_mask)

    return pd.DataFrame(columns, index=df.index)
