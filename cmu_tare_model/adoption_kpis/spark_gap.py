"""
State-level electricity-to-natural-gas spark gap computation.

Reads nominal EIA fuel prices from CSV, converts to $/kWh and $/MMBTU,
and computes the spark gap (electricity price / gas price on a $/MMBTU basis)
for each state.

Location: cmu_tare_model/adoption_kpis/spark_gap.py
"""

from typing import List, Union

import pandas as pd

from cmu_tare_model.adoption_kpis.data_loading import (
    KWH_PER_MMBTU,
    NG_CONVERSION_FACTOR,
    STATE_NAMES,
)


def calculate_spark_gap(
    filepath: str,
    year: Union[int, List[int]] = 2022,
) -> pd.DataFrame:
    """Compute state-level electricity-to-natural-gas price ratios (spark gap).

    Reads nominal EIA residential fuel prices from a CSV file, converts
    electricity (cents/kWh) and natural gas ($/1000 cf) to a common
    $/MMBTU basis, and returns the per-state spark gap.

    Spark gap = P_elec_mmbtu / P_gas_mmbtu.  A higher spark gap means
    electricity is relatively expensive versus gas, making heat pumps
    less competitive on operating cost alone.

    Args:
        filepath: Path to the EIA fuel prices CSV
            (e.g., ``FUEL_PRICES_PATH``).  The file must contain a
            ``fuel_type`` column, a ``state_region`` column, and at
            least one column named ``{year}_nominal_unit_price``.
        year: Single calendar year or list of years to average.
            Defaults to 2022.  Use ``list(range(2020, 2025))`` for a
            5-year average.

    Returns:
        DataFrame with one row per state (51 rows: 50 states + DC),
        sorted by ``spark_gap`` descending, with columns:

        - ``state``: 2-letter state abbreviation
        - ``state_name``: full state name
        - ``elec_price_kwh``: electricity price in $/kWh (4 dp)
        - ``gas_price_kwh``: natural gas price in $/kWh (4 dp)
        - ``elec_price_mmbtu``: electricity price in $/MMBTU (2 dp)
        - ``gas_price_mmbtu``: natural gas price in $/MMBTU (2 dp)
        - ``spark_gap``: elec_price_mmbtu / gas_price_mmbtu (2 dp)

    Raises:
        KeyError: If any requested year column is not present in the CSV.

    Notes:
        Gas heat content assumption: 1036 BTU/cf (EIA average).
        Jenkins et al. use 1020 BTU/cf, causing an ~1.8% discrepancy
        in spark gap and break-even COP values.

    Verification:
        - 51 rows (50 states + DC)
        - FL spark gap ≈ 1.61 (±0.05)
        - AK spark gap ≈ 6.35 (±0.05)
        - National mean spark gap ≈ 3.15 (±0.05)
    """
    df = pd.read_csv(filepath)

    years = [year] if isinstance(year, int) else list(year)
    price_cols = [f"{y}_nominal_unit_price" for y in years]
    missing = [c for c in price_cols if c not in df.columns]
    if missing:
        available = [c for c in df.columns if "nominal_unit_price" in c]
        raise KeyError(
            f"Column(s) {missing} not found in {filepath}. "
            f"Available year columns: {available}"
        )

    # Average across requested years
    df["_avg_price"] = df[price_cols].mean(axis=1)

    # Natural gas prices — state-level rows only (2-char state code, not 'National')
    df_ng = df[
        (df["fuel_type"] == "naturalGas")
        & (df["state_region"].str.len() == 2)
        & (df["state_region"] != "National")
    ][["state_region", "_avg_price"]].copy()
    df_ng.columns = ["state", "ng_price_per_1000cf"]

    # Electricity prices — state-level rows only
    df_elec = df[
        (df["fuel_type"] == "electricity")
        & (df["state_region"].str.len() == 2)
        & (df["state_region"] != "National")
    ][["state_region", "_avg_price"]].copy()
    df_elec.columns = ["state", "elec_price_cents_kwh"]

    df_merged = df_elec.merge(df_ng, on="state", how="inner")

    # Unit conversions
    df_merged["elec_price_kwh"] = df_merged["elec_price_cents_kwh"] / 100
    df_merged["gas_price_kwh"] = df_merged["ng_price_per_1000cf"] * NG_CONVERSION_FACTOR
    df_merged["elec_price_mmbtu"] = df_merged["elec_price_kwh"] * KWH_PER_MMBTU
    df_merged["gas_price_mmbtu"] = df_merged["gas_price_kwh"] * KWH_PER_MMBTU
    df_merged["spark_gap"] = df_merged["elec_price_mmbtu"] / df_merged["gas_price_mmbtu"]
    df_merged["state_name"] = df_merged["state"].map(STATE_NAMES)

    result = df_merged[[
        "state", "state_name",
        "elec_price_kwh", "gas_price_kwh",
        "elec_price_mmbtu", "gas_price_mmbtu",
        "spark_gap",
    ]].copy()

    for col in ["elec_price_kwh", "gas_price_kwh"]:
        result[col] = result[col].round(4)
    for col in ["elec_price_mmbtu", "gas_price_mmbtu", "spark_gap"]:
        result[col] = result[col].round(2)

    return result.sort_values("spark_gap", ascending=False).reset_index(drop=True)
