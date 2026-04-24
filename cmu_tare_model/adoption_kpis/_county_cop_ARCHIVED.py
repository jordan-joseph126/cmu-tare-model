"""
ARCHIVED — County-level COP aggregation helpers.

Preserved for potential future use. Not part of the current analysis
pipeline. The paper (Joseph et al. 2026) uses state-level metrics only.

State-level EIA fuel prices are broadcast to counties here as the best
available proxy; genuine county-level utility rates are not in this
dataset and would require EIA-861 integration.

See thermal_cop.py for the active state-level implementation.

Location: cmu_tare_model/adoption_kpis/_county_cop_ARCHIVED.py
"""

# NOTE: COUNTY_SHAPEFILE_PATH is a county-level constant retained here
# for context; it is not imported by any active module.

import os
import pandas as pd
from config import PROJECT_ROOT

COUNTY_SHAPEFILE_PATH: str = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "electricity_ng_price_ratio",
    "tl_2025_us_county", "tl_2025_us_county.shp"
)
"""Path to the county-level shapefile (not used in current pipeline)."""

_COUNTY_COL = "in.county"
_STATE_COL = "in.state"


def broadcast_prices_to_counties(
    df_prices: pd.DataFrame,
    df_baseline: pd.DataFrame,
    county_col: str = _COUNTY_COL,
    state_col: str = _STATE_COL,
) -> pd.DataFrame:
    """Broadcast state-level EIA fuel prices to county level by joining on state.

    ARCHIVED — not called by any active module.

    NOTE: This is NOT county-specific rate data — state-level EIA rates are
    broadcast to counties as the best available proxy. Future work: integrate
    utility-level rates from EIA-861.

    Args:
        df_prices: State-level prices from calculate_spark_gap(). Must
            contain a ``state`` column with 2-letter state abbreviations.
        df_baseline: EUSS baseline DataFrame (indexed by bldg_id) with
            county (GISJOIN) and state abbreviation columns.
        county_col: Column name in df_baseline for the GISJOIN county code
            (default: ``'in.county'``).
        state_col: Column name in df_baseline for the state abbreviation
            (default: ``'in.state'``).

    Returns:
        DataFrame with one row per unique (state, county) pair in
        df_baseline, with all columns from df_prices plus a ``county``
        column (GISJOIN format, e.g. ``'G4200030'`` for Allegheny County PA).

    Raises:
        KeyError: If county_col or state_col are not found in df_baseline.
    """
    if county_col not in df_baseline.columns:
        raise KeyError(
            f"Column '{county_col}' not found in df_baseline. "
            f"Available: {list(df_baseline.columns[:10])}"
        )
    if state_col not in df_baseline.columns:
        raise KeyError(
            f"Column '{state_col}' not found in df_baseline. "
            f"Available: {list(df_baseline.columns[:10])}"
        )

    counties = (
        df_baseline[[state_col, county_col]]
        .drop_duplicates()
        .rename(columns={state_col: "state", county_col: "county"})
    )
    return counties.merge(df_prices, on="state", how="inner").reset_index(drop=True)
