"""
EIA API utilities for fetching AEO fuel price and degree-day data.

This module contains the constants, geographic lookup tables, and helper functions
that fetch_aeo_data_and_project.py uses to produce three CSV artifacts from the EIA
API v2 using AEO2026 Counterfactual Baseline data:

    data/fuel_prices/eia_fuel_price_data_2025_usd2025.csv
        2025 state-level residential price anchor in USD2025/kWh -- one row per
        state x fuel combination, with PADD-broadcast provenance recorded in
        source_region and source_method.

    data/projections/aeo2026_fuel_price_factors_2025_2050.csv
        Annual price-projection factors (2025 = 1.0) per census division and fuel,
        derived from AEO Table 3 real 2025-dollar series.

    data/projections/aeo2026_degree_day_factors_2025_2050.csv
        Annual HDD/CDD projection factors (2025 = 1.0) per census division,
        derived from AEO Table 4 via backward-compatible v1 series IDs.

The EIA API key is never stored as a module-level variable. Pass it explicitly as
the `api_key` argument to any function that makes HTTP requests. The key is obtained
via interactive input() in the calling notebook at runtime.
"""

import time

import pandas as pd
import requests


# ----------------------------------------------------------------------
# GEOGRAPHIC LOOKUP TABLES
# ----------------------------------------------------------------------

ALL_STATES = [
    'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'ID',
    'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO',
    'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA',
    'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
]  # 51 locations: 50 states + DC

# Census division --> member states. Used to JOIN each state's anchor price to its AEO
# projection factor (factors are published per census division, not per state). Same
# partition used in create_lookup_fuel_prices.py -- keep both in sync.
DIVISION_TO_STATES = {
    "New England": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "Middle Atlantic": ["NJ", "NY", "PA"],
    "East North Central": ["IN", "IL", "MI", "OH", "WI"],
    "West North Central": ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South Atlantic": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"],
    "East South Central": ["AL", "KY", "MS", "TN"],
    "West South Central": ["AR", "LA", "OK", "TX"],
    "Mountain": ["AZ", "CO", "ID", "NM", "MT", "UT", "NV", "WY"],
    "Pacific": ["AK", "CA", "HI", "OR", "WA"],
}

# Inverted: state abbreviation --> its census division, for O(1) lookup.
STATE_TO_DIVISION = {s: d for d, states in DIVISION_TO_STATES.items() for s in states}

# PADD / sub-PADD --> member states, at the finest EIA petroleum granularity available.
# EIA `duoarea` codes returned by petroleum/pri/wfr:
#   R1X = PADD 1A (New England)      R30 = PADD 3 (Gulf Coast)
#   R1Y = PADD 1B (Central Atlantic) R40 = PADD 4 (Rocky Mountain)
#   R1Z = PADD 1C (Lower Atlantic)   R50 = PADD 5 (West Coast)
#   R20 = PADD 2 (Midwest)       R10 = PADD 1 (East Coast; parent of 1A/1B/1C)
#   NUS = U.S. National
# Every state in a PADD inherits that PADD's oil/propane price -- states within the same
# PADD carry IDENTICAL oil/propane anchors. This is expected and recorded in the
# `source_region` column so it is auditable, not silent.
PADD_TO_STATES = {
    "R1X": ["CT", "ME", "MA", "NH", "RI", "VT"],  # PADD 1A
    "R1Y": ["DE", "DC", "MD", "NJ", "NY", "PA"],  # PADD 1B
    "R1Z": ["FL", "GA", "NC", "SC", "VA", "WV"],  # PADD 1C
    "R20": ["IL", "IN", "IA", "KS", "KY", "MI", "MN", "MO", "NE", "ND", "OH",
            "OK", "SD", "TN", "WI"],  # PADD 2
    "R30": ["AL", "AR", "LA", "MS", "NM", "TX"],  # PADD 3
    "R40": ["CO", "ID", "MT", "UT", "WY"],  # PADD 4
    "R50": ["AK", "AZ", "CA", "HI", "NV", "OR", "WA"],  # PADD 5
}

# Inverted: state abbreviation --> PADD duoarea code, for O(1) lookup.
STATE_TO_PADD = {s: r for r, states in PADD_TO_STATES.items() for s in states}

# PADD 1 sub-regions. These fall back to R10 when sub-PADD data is absent.
# EIA sometimes reports 1A/1B/1C individually; other times only the PADD-1 aggregate.
PADD1_SUBREGIONS = {"R1X", "R1Y", "R1Z"}

# Human-readable labels for the source_region_label provenance column in the anchor CSV.
REGION_LABELS = {
    "R1X": "PADD 1A New England",
    "R1Y": "PADD 1B Central Atlantic",
    "R1Z": "PADD 1C Lower Atlantic",
    "R10": "PADD 1 East Coast",
    "R20": "PADD 2 Midwest",
    "R30": "PADD 3 Gulf Coast",
    "R40": "PADD 4 Rocky Mountain",
    "R50": "PADD 5 West Coast",
    "NUS": "U.S. National",
}

# AEO regionName (from Table 3 API response) --> census-division string used in
# factor tables. AEO uses "United States" for the national aggregate; all other
# names match DIVISION_TO_STATES keys exactly, but they map to "National" here
# to keep a consistent sentinel value downstream.
REGION_MAP = {
    "United States": "National",
    "New England": "New England",
    "Middle Atlantic": "Middle Atlantic",
    "East North Central": "East North Central",
    "West North Central": "West North Central",
    "South Atlantic": "South Atlantic",
    "East South Central": "East South Central",
    "West South Central": "West South Central",
    "Mountain": "Mountain",
    "Pacific": "Pacific",
}

# AEO Table 3 seriesName --> codebase fuel_type string. AEO publishes both nominal
# and real series under subtly different names; both variants map to the same
# fuel_type so the notebook filter works regardless of which unit variant the
# API returns first.
SERIES_TO_FUEL = {
    "Energy Prices : Residential : Electricity": "electricity",
    "Energy Prices : Residential : Natural Gas": "naturalGas",
    "Energy Prices : Residential : Distillate Fuel Oil": "fuelOil",
    "Energy Prices : Residential : Propane": "propane",
    "Energy Prices : Nominal : Residential : Electricity": "electricity",
    "Energy Prices : Nominal : Residential : Natural Gas": "naturalGas",
    "Energy Prices : Nominal : Residential : Distillate Fuel Oil": "fuelOil",
    "Energy Prices : Nominal : Residential : Propane": "propane",
}


# ----------------------------------------------------------------------
# EIA API HELPERS
# ----------------------------------------------------------------------


def eia_get(route: str, api_key: str, **params) -> list:
    """Paginated EIA API v2 GET for /data endpoints. Returns all rows across all pages.

    AEO datasets can exceed 5 000 rows, so this function paginates via offset until
    the accumulated count reaches the `total` the API reports. Backs off on HTTP 429
    rate-limit responses before raising on other HTTP errors. For small datasets that
    fit in one response, use eia_fetch instead.

    Args:
        route: API route after the v2 base URL, e.g. 'aeo/2026/data'.
        api_key: EIA API key obtained at notebook runtime.
        **params: Extra query params passed through (facets, frequency, start, end).
            Do NOT include api_key, data[], length, or offset -- those are managed
            internally.

    Returns:
        A flat list of row dicts across all pages, in API-returned order.

    Raises:
        requests.HTTPError: On any non-429 HTTP error, or after four consecutive 429s.
    """
    url = f"https://api.eia.gov/v2/{route}"
    all_rows, offset = [], 0

    while True:
        # ===== STEP 1: Fetch one page, with exponential back-off on rate limits =====
        for attempt in range(4):
            r = requests.get(
                url,
                params={"api_key": api_key,
                        "data[]": "value", "length": 5000, "offset": offset,
                        **params},
                timeout=30,
            )
            if r.status_code == 429:
                wait = 30 * (attempt + 1)  # 30s back-off per attempt: 30, 60, 90, 120s
                print(f"Rate limited -- waiting {wait}s...")
                time.sleep(wait)
            else:
                r.raise_for_status()
                break

        # ===== STEP 2: Accumulate rows; stop when total is reached =====
        body = r.json()["response"]
        rows = body.get("data", [])
        all_rows.extend(rows)
        if len(all_rows) >= int(body.get("total", len(all_rows))):
            break
        offset += 5000
        time.sleep(2)  # avoid triggering EIA rate limits between pages

    return all_rows


def eia_fetch(route: str, api_key: str, **params) -> list:
    """Single-request EIA API v2 GET for small state/regional datasets.

    Appropriate for electricity, natural gas, and petroleum price endpoints where the
    full dataset fits in one response and no offset pagination is needed. For large
    AEO datasets that may span multiple pages, use eia_get instead.

    Args:
        route: API route after the v2 base URL, e.g. 'electricity/retail-sales/data'.
        api_key: EIA API key obtained at notebook runtime.
        **params: Query params passed directly to the request (data[], facets,
            frequency, start, end, length).

    Returns:
        A list of row dicts from the API response body.

    Raises:
        requests.HTTPError: If the API returns a non-2xx status.
    """
    r = requests.get(
        f"https://api.eia.gov/v2/{route}",
        params={"api_key": api_key, **params},
        timeout=30,
    )
    r.raise_for_status()
    body = r.json()["response"]
    return body.get("data", [])


def fetch_seriesid(v1_series_id: str, api_key: str, anchor_year: int) -> list:
    """Fetch one series via the v2 SeriesID backward-compatibility path.

    AEO Table 4 (HDD/CDD) is not reachable through facets[tableId][] in the standard
    v2 /data route. The SeriesID path accepts a v1-style ID and returns data for the
    given year range without requiring data[], facets, or offset parameters.

    Args:
        v1_series_id: A v1-style series ID, e.g.
            'AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_NENGL_HDD.A'.
        api_key: EIA API key obtained at notebook runtime.
        anchor_year: First year to fetch (inclusive). Data is returned through 2050.

    Returns:
        A list of row dicts for that series from anchor_year through 2050.

    Raises:
        requests.HTTPError: If the API returns a non-2xx status.
    """
    r = requests.get(
        f"https://api.eia.gov/v2/seriesid/{v1_series_id}",
        params={"api_key": api_key,
                "start": str(anchor_year), "end": "2050"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["response"]["data"]


# ----------------------------------------------------------------------
# UNIT CONVERSION
# ----------------------------------------------------------------------


def to_usd2025_per_kwh(value: float, fuel_type: str) -> float:
    """Convert a 2025 residential fuel price from its native EIA unit to USD2025/kWh.

    Conversion factors match process_fuel_price_data() in create_lookup_fuel_prices.py
    exactly. No CPI adjustment is applied: 2025 actuals are already in 2025 dollars
    (CPI ratio 2025/2025 = 1.0).

    Base units by fuel, as published by EIA:
        electricity : cents per kWh
        naturalGas  : dollars per MCF (1 MCF = 1 000 cf; 1 cf ~= 1 039 BTU)
        fuelOil     : dollars per gallon (1 gal heating oil ~= 138 500 BTU)
        propane     : dollars per gallon (1 gal propane ~= 91 452 BTU)

    Args:
        value: Price in the fuel's native EIA unit (2025 nominal = 2025 real).
        fuel_type: One of {'electricity', 'naturalGas', 'fuelOil', 'propane'}.

    Returns:
        Price in USD2025 per kWh.

    Raises:
        ValueError: If fuel_type is not one of the four recognized strings.
    """
    if fuel_type == "electricity":
        # cents/kWh --> $/kWh
        return value / 100
    if fuel_type == "naturalGas":
        # $/MCF --> $/cf --> $/BTU --> $/kWh  (1 cf ~= 1 039 BTU; 1 kWh = 3 412 BTU)
        return value * (1 / 1000) * (1 / 1039) * 3412
    if fuel_type == "fuelOil":
        # $/gal --> $/BTU --> $/kWh  (1 gal heating oil ~= 138 500 BTU)
        return value * (1 / 138500) * 3412
    if fuel_type == "propane":
        # $/gal --> $/BTU --> $/kWh  (1 gal propane ~= 91 452 BTU)
        return value * (1 / 91452) * 3412
    raise ValueError(
        f"Unknown fuel_type: {fuel_type!r}. "
        f"Expected one of: 'electricity', 'naturalGas', 'fuelOil', 'propane'."
    )


# ----------------------------------------------------------------------
# FUEL PRICE PROJECTION
# ----------------------------------------------------------------------


def fetch_petroleum_region_prices(
    product_code: str, api_key: str
) -> tuple[dict, float]:
    """Fetch 2025 residential petroleum prices and average across months by EIA region.

    EIA publishes heating oil and propane prices at the PADD / sub-PADD level (duoarea),
    not per state. Prices are monthly; this function averages all 2025 months that
    reported to produce a single annual representative price per region. Regions that
    reported zero months are excluded from the returned dict rather than stored as NaN,
    so downstream fallback logic can treat absence as "no data."

    Args:
        product_code: EIA product code -- 'EPD2F' for heating oil, 'EPLLPA' for propane.
        api_key: EIA API key obtained at notebook runtime.

    Returns:
        (region_price, national_price):
            region_price: {duoarea_code: mean 2025 $/gal} for every region that
                reported at least one month. Keys follow EIA duoarea notation
                (R1X, R1Y, R1Z, R20, R30, R40, R50, R10, NUS).
            national_price: The 'NUS' mean $/gal, or float('nan') if NUS is absent.

    Raises:
        requests.HTTPError: If the EIA API returns a non-2xx status.
    """
    petro_params = {
        "data[]": "value",
        "facets[product][]": product_code,
        "facets[process][]": "PRS",  # PRS = price to residential
        "frequency": "monthly",
        "start": "2025-01",
        "end": "2025-12",
        "length": 5000,  # full pull -- no discovery cap
    }
    rows_pet = eia_fetch("petroleum/pri/wfr/data", api_key=api_key, **petro_params)
    df_pet = pd.DataFrame(rows_pet)
    df_pet["value"] = pd.to_numeric(df_pet["value"], errors="coerce")
    region_mean = df_pet.dropna(subset=["value"]).groupby("duoarea")["value"].mean()
    region_price = region_mean.to_dict()
    return region_price, float(region_price.get("NUS", float("nan")))


def resolve_state_petroleum_price(
    state: str, region_price: dict, national: float
) -> tuple:
    """Resolve one state's petroleum price through the PADD fallback chain.

    EIA does not publish per-state heating oil or propane prices. The resolution order
    is: finest available sub-PADD --> PADD-1 aggregate (R10) if applicable -->
    U.S. National.
    PADD-1 states (1A/1B/1C) get a second chance at R10 before falling back to NUS
    because EIA sometimes reports the PADD-1 aggregate when the sub-PADDs are absent.

    Args:
        state: 2-letter state/DC abbreviation (must be a key in STATE_TO_PADD).
        region_price: {duoarea_code: price $/gal} from fetch_petroleum_region_prices.
        national: National (NUS) price $/gal used as last-resort fallback.

    Returns:
        (price, source_region_code, source_method) where:
            price: Resolved $/gal for this state.
            source_region_code: The duoarea code that supplied the price (e.g. 'R1X').
            source_method: 'padd_broadcast_monthly_mean_2025' or 'national_fallback'.
    """
    # Build the fallback chain: own PADD --> R10 parent (PADD-1 only) --> NUS.
    chain = [STATE_TO_PADD[state]]
    if STATE_TO_PADD[state] in PADD1_SUBREGIONS:
        chain.append("R10")
    chain.append("NUS")

    for code in chain:
        if code in region_price and pd.notna(region_price[code]):
            method = (
                "national_fallback" if code == "NUS"
                else "padd_broadcast_monthly_mean_2025"
            )
            return region_price[code], code, method
    return national, "NUS", "national_fallback"


def build_petroleum_fuel_df(
    product_code: str, fuel_type: str, api_key: str
) -> tuple[pd.DataFrame, float]:
    """Assemble a state-level DataFrame for one fuel by broadcasting PADD prices.

    EIA's finest petroleum geography is PADD / sub-PADD, so every state in the same PADD
    receives the identical price. This is expected (not a data error) and is recorded in
    the source_region and source_method columns for auditability rather than silently
    duplicating the value with no provenance.

    Args:
        product_code: EIA product code -- 'EPD2F' (heating oil) or 'EPLLPA' (propane).
        fuel_type: Codebase fuel string -- 'fuelOil' or 'propane'.
        api_key: EIA API key obtained at notebook runtime.

    Returns:
        (df_fuel, national_price):
            df_fuel: DataFrame with columns
                [state, base_value, source_region, source_method, fuel_type, base_unit].
                One row per state in ALL_STATES. base_value is in $/gal.
            national_price: NUS mean $/gal, used by the caller to build the
                National row.

    Raises:
        requests.HTTPError: Propagated from fetch_petroleum_region_prices.
    """
    # ===== STEP 1: Fetch and average PADD-level prices for the year =====
    region_price, national = fetch_petroleum_region_prices(product_code, api_key)

    # ===== STEP 2: Broadcast each state's price via the PADD fallback chain =====
    records = []
    for state in ALL_STATES:
        price, code, method = resolve_state_petroleum_price(
            state, region_price, national
        )
        records.append({"state": state, "base_value": price,
                        "source_region": code, "source_method": method})

    # ===== STEP 3: Add fuel_type and unit columns; print a coverage summary =====
    df_fuel = pd.DataFrame(records)
    df_fuel["fuel_type"] = fuel_type
    df_fuel["base_unit"] = "$/gal"

    n_fallback = int((df_fuel["source_method"] == "national_fallback").sum())
    reported = sorted(r for r in region_price if r != "NUS")
    reporting = [REGION_LABELS.get(r, r) for r in reported]
    print(f"{fuel_type}: regions reporting = {reporting}")
    print(f"{fuel_type}: {n_fallback} state(s) --> National fallback "
          f"(no PADD-level residential {fuel_type}); National = {national:.3f} $/gal")
    return df_fuel, national


# ----------------------------------------------------------------------
# DEGREE-DAY PROJECTION
# ----------------------------------------------------------------------


def fetch_degree_day_rows(
    series_ids: dict, dd_type: str, api_key: str, anchor_year: int
) -> list:
    """Fetch every series in a degree-day dict and tag each row with division and type.

    AEO Table 4 (HDD/CDD) returns regionName = 'No Regional Tables' for every row --
    the API does not embed the census division in the response body. The dict key is
    therefore the authoritative source of the census_division label. A wrong series ID
    would silently produce data under the correct key name with no error, which is why
    CDD_SERIES_IDS carries an explicit guard comment on the West South Central entry
    (the HDD and CDD series IDs for WSC differ only in the final suffix).

    Args:
        series_ids: {census_division_label: v1_series_id} mapping, e.g. HDD_SERIES_IDS.
        dd_type: 'hdd' or 'cdd', written verbatim into each row's dd_type field.
        api_key: EIA API key obtained at notebook runtime.
        anchor_year: First year to include; passed through to fetch_seriesid.

    Returns:
        A flat list of row dicts. Each dict contains all fields from the raw API
        response plus 'census_division' (from the dict key) and 'dd_type'.

    Raises:
        requests.HTTPError: Propagated from fetch_seriesid on any bad series ID.
    """
    out = []
    for division, series_id in series_ids.items():
        # ===== STEP 1: Fetch one series; tag with division label from dict key =====
        for row in fetch_seriesid(series_id, api_key=api_key, anchor_year=anchor_year):
            row["census_division"] = division
            row["dd_type"] = dd_type
            out.append(row)
        time.sleep(0.3)  # brief pause to avoid EIA rate limits between series fetches
    return out


def degree_day_factor_table(
    df_raw: pd.DataFrame, dd_type: str, anchor_year: int
) -> pd.DataFrame:
    """Pivot one degree-day type to wide format and normalize so anchor_year = 1.0.

    The normalization converts raw degree-day counts (dimensioned) into dimensionless
    factors that scale heating/cooling loads relative to the anchor year. A value of
    0.90 in a given year means 10% fewer degree-days than the anchor, implying
    proportionally lower energy demand for that fuel in that division.

    Args:
        df_raw: Tagged raw rows as returned by fetch_degree_day_rows (or a concat of
            multiple calls). Must contain columns:
            census_division, dd_type, period, value.
        dd_type: 'hdd' or 'cdd' -- selects the subset to pivot.
        anchor_year: Year whose raw value becomes the denominator (factor = 1.0).

    Returns:
        DataFrame with a 'census_division' column and one integer column per year
        (from anchor_year through 2050). Each cell is the ratio
        year_value / anchor_value.

    Raises:
        ValueError: If anchor_year is absent from the pivoted data.
    """
    # ===== STEP 1: Filter to the requested type and coerce column dtypes =====
    d = df_raw[df_raw["dd_type"] == dd_type].copy()
    d["year"] = d["period"].astype(int)
    d["value"] = pd.to_numeric(d["value"], errors="coerce")

    # ===== STEP 2: Pivot to census_division x year wide format =====
    pivot = d.pivot_table(index="census_division", columns="year", values="value")
    if anchor_year not in pivot.columns:
        raise ValueError(
            f"Anchor year {anchor_year} not found in {dd_type.upper()} data. "
            f"Available years: {sorted(pivot.columns.tolist())}"
        )

    # ===== STEP 3: Normalize each row so anchor_year = 1.0 =====
    return pivot.div(pivot[anchor_year], axis=0).reset_index()


# ----------------------------------------------------------------------
# SERIES ID TABLES
# ----------------------------------------------------------------------

# AEO2026 Counterfactual Baseline (CB2026) series IDs for residential HDD and CDD,
# fetched via the v2 SeriesID backward-compatibility path. These are v1-style IDs;
# they embed the scenario (CB2026) and geographic suffix directly in the string, so
# they do NOT respond to facets[scenario][] filtering in the standard v2 /data route.
HDD_SERIES_IDS = {
    "New England": "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_NENGL_HDD.A",
    "Middle Atlantic": "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_MATL_HDD.A",
    "East North Central": "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_ENC_HDD.A",
    "West North Central": "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_WNC_HDD.A",
    "South Atlantic": "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_SATL_HDD.A",
    "East South Central": "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_ESC_HDD.A",
    "West South Central": "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_WSC_HDD.A",
    "Mountain": "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_MTN_HDD.A",
    "Pacific": "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_PCF_HDD.A",
    "National": "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_USA_HDD.A",
}
CDD_SERIES_IDS = {
    "New England": "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_NENGL_CDD.A",
    "Middle Atlantic": "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_MATL_CDD.A",
    "East North Central": "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_ENC_CDD.A",
    "West North Central": "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_WNC_CDD.A",
    "South Atlantic": "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_SATL_CDD.A",
    "East South Central": "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_ESC_CDD.A",
    # NOT _WSC_HDD -- WSC HDD and CDD series IDs differ only in the final segment
    "West South Central": "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_WSC_CDD.A",
    "Mountain": "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_MTN_CDD.A",
    "Pacific": "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_PCF_CDD.A",
    "National": "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_USA_CDD.A",
}
