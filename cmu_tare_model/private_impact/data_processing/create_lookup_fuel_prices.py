import os
import pandas as pd

from config import PROJECT_ROOT
from cmu_tare_model.constants import ANCHOR_YEAR, PROJECTION_END_YEAR
from cmu_tare_model.utils.data_visualization import print_truncated_dict

# ====================================================================================================
# Set print_verbose to True for detailed output, or False for minimal output
print_verbose = False
# ====================================================================================================

SCENARIO_STRING = "2025 Reference Case"   # must byte-match CSV policy_scenario column

# Paths to the two new CSV artifacts
_PATH_ANCHOR  = os.path.join(PROJECT_ROOT, "cmu_tare_model", "data", "fuel_prices",
                              "eia_fuel_price_data_2025_usd2025.csv")
_PATH_FACTORS = os.path.join(PROJECT_ROOT, "cmu_tare_model", "data", "projections",
                              "aeo2026_fuel_price_factors_2025_2050.csv")

# ====================================================================================================
# LOAD ANCHOR PRICES â€” already in USD2025/kWh, no unit conversion or CPI deflation needed
# ====================================================================================================
df_anchor = pd.read_csv(_PATH_ANCHOR)

# ====================================================================================================
# LOAD PROJECTION FACTORS; cast year-header strings to int (Constraint 4 analogue for this module)
# ====================================================================================================
df_factors = pd.read_csv(_PATH_FACTORS)
df_factors.columns = [int(c) if isinstance(c, str) and c.isdigit() else c
                      for c in df_factors.columns]

_year_cols = sorted([c for c in df_factors.columns if isinstance(c, int)])


def _validate_factor_years(df_factors_checked, year_cols) -> None:
    """Check the projection file covers the expected years and is anchored at 1.0.

    Two things must hold for the fuel-price projection file, and if either
    fails the model would still run while producing quietly wrong costs, so
    both are checked when this module is imported:

    1. The year columns are exactly ANCHOR_YEAR through PROJECTION_END_YEAR,
       with no gaps and no year earlier than ANCHOR_YEAR. A year before
       ANCHOR_YEAR has no meaning here -- every factor is measured relative to
       ANCHOR_YEAR -- and a gap would silently shorten a cost stream.
    2. Every ANCHOR_YEAR factor is exactly 1.0, for every region and fuel.
       That is what makes ANCHOR_YEAR the anchor: prices in that year are the
       observed prices, unprojected.

    Args:
        df_factors_checked: Projection factors table, year columns already
            cast to int.
        year_cols: Sorted list of the integer year columns found in the table.

    Raises:
        ValueError: If the year coverage is wrong or any ANCHOR_YEAR factor is
            not exactly 1.0.
    """
    expected_years = list(range(ANCHOR_YEAR, PROJECTION_END_YEAR + 1))
    if year_cols != expected_years:
        missing = sorted(set(expected_years) - set(year_cols))
        unexpected = sorted(set(year_cols) - set(expected_years))
        raise ValueError(
            f"Fuel-price factor years must be exactly {ANCHOR_YEAR}-"
            f"{PROJECTION_END_YEAR} with no gaps. "
            f"Missing: {missing}. Unexpected: {unexpected}. "
            f"File: {_PATH_FACTORS}")

    off_anchor = df_factors_checked[df_factors_checked[ANCHOR_YEAR] != 1.0]
    if not off_anchor.empty:
        offenders = [
            f"{row['region']}/{row['fuel_type']}={row[ANCHOR_YEAR]}"
            for _, row in off_anchor.iterrows()
        ]
        raise ValueError(
            f"Every {ANCHOR_YEAR} fuel-price factor must be exactly 1.0 "
            f"({ANCHOR_YEAR} is the anchor year, so its prices are "
            f"unprojected). Offending rows: {offenders}. "
            f"File: {_PATH_FACTORS}")


_validate_factor_years(df_factors, _year_cols)

# Internal factor lookup: {(region, fuel_type): {year: factor}}
_factor_lookup: dict = {}
for _, _row in df_factors.iterrows():
    _factor_lookup[(_row['region'], _row['fuel_type'])] = {yr: _row[yr] for yr in _year_cols}


def create_projection_factors_dict(df_projection_factors):
    """
    Transforms a projection-factors DataFrame into a nested dict.

    Key: (region, fuel_type, policy_scenario) â†’ {year: factor}
    Year columns must already be int before calling this function.
    Retained for any external callers; internal code uses _factor_lookup.
    """
    out = {}
    year_columns = [c for c in df_projection_factors.columns
                    if isinstance(c, int) or (isinstance(c, str) and c.isdigit())]
    for _, row in df_projection_factors.iterrows():
        key = (row['region'], row['fuel_type'], row['policy_scenario'])
        out.setdefault(key, {})
        for yc in year_columns:
            out[key][int(yc) if isinstance(yc, str) else yc] = row[yc]
    return out


# ====================================================================================================
# BUILD LOOKUP DICT
#   Structure: lookup[location_key][fuel_type][SCENARIO_STRING][year] â†’ USD2025/kWh
#
#   location_key:
#     electricity / naturalGas  â†’ state abbreviation  (e.g. 'PA')
#     fuelOil / propane          â†’ census division name (e.g. 'Middle Atlantic')
#     'National'                 â†’ National fallback for all four fuels
# ====================================================================================================
def _build_lookup() -> dict:
    lookup: dict = {}

    # â”€â”€ electricity and naturalGas: state-level anchor prices â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _state_fuels = df_anchor[df_anchor['fuel_type'].isin(['electricity', 'naturalGas'])]
    for _, row in _state_fuels.iterrows():
        loc    = row['state']
        fuel   = row['fuel_type']
        anchor = row['price_usd2025_per_kwh']
        cdiv   = row['census_division']

        fk = (cdiv, fuel)
        if fk not in _factor_lookup:
            fk = ('National', fuel)
        if fk not in _factor_lookup:
            continue

        yearly = {yr: anchor * _factor_lookup[fk][yr] for yr in _year_cols}
        lookup.setdefault(loc, {}).setdefault(fuel, {})[SCENARIO_STRING] = yearly

    # â”€â”€ fuelOil and propane: census-division-level averaged anchor prices â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _cdiv_fuels = (
        df_anchor[
            df_anchor['fuel_type'].isin(['fuelOil', 'propane']) &
            (df_anchor['state'] != 'National')
        ]
        .groupby(['census_division', 'fuel_type'], as_index=False)['price_usd2025_per_kwh']
        .mean()
    )
    for _, row in _cdiv_fuels.iterrows():
        loc    = row['census_division']
        fuel   = row['fuel_type']
        anchor = row['price_usd2025_per_kwh']

        fk = (loc, fuel)
        if fk not in _factor_lookup:
            fk = ('National', fuel)
        if fk not in _factor_lookup:
            continue

        yearly = {yr: anchor * _factor_lookup[fk][yr] for yr in _year_cols}
        lookup.setdefault(loc, {}).setdefault(fuel, {})[SCENARIO_STRING] = yearly

    # â”€â”€ National fallback for all four fuels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    _national = df_anchor[df_anchor['state'] == 'National']
    for _, row in _national.iterrows():
        fuel   = row['fuel_type']
        anchor = row['price_usd2025_per_kwh']

        fk = ('National', fuel)
        if fk not in _factor_lookup:
            continue

        yearly = {yr: anchor * _factor_lookup[fk][yr] for yr in _year_cols}
        lookup.setdefault('National', {}).setdefault(fuel, {})[SCENARIO_STRING] = yearly

    return lookup


lookup_fuel_prices_aeo2026 = _build_lookup()

if print_verbose:
    print(f"Fuel-price lookup built: {len(lookup_fuel_prices_aeo2026)} location keys")
    print_truncated_dict(lookup_fuel_prices_aeo2026, n=5)
