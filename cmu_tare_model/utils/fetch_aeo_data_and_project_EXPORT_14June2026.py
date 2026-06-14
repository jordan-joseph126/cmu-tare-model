# %% [markdown]
# # Fetch EIA Fuel Price Data using API and Project Fuel Prices
#  
# Replaces the manually maintained Excel workbook `aeo_projections_2022_2050.xlsx`
# (three sheets) with three on-disk CSV artifacts, all sourced live from the EIA API v2
# using AEO2026 Counterfactual Baseline data:
#  
#     Old Excel sheet                  →  New CSV artifact (this file writes it)
#     ───────────────────────────────────────────────────────────────────────────────────
#     fuel_price_factors_2022_2050     →  data/projections/aeo2026_fuel_price_factors_2025_2050.csv
#     hdd_factors_2022_2050 +          →  data/projections/aeo2026_degree_day_factors_2025_2050.csv
#     cdd_factors_2022_2050               (combined, with a `dd_type` column: 'hdd' / 'cdd')
#     (none — new this refactor)       →  data/fuel_prices/eia_fuel_price_data_2025_usd2025.csv
#                                         (the 2025 state-level price ANCHOR in USD2025/kWh)
#  
# Two distinct things are produced:
#   1. ANCHOR PRICES (the price LEVEL): 2025 residential prices per state, in USD2025/kWh.
#   2. PROJECTION FACTORS (the curve SHAPE): each year's price ÷ the 2025 price, so 2025 = 1.0.
# A future-year price = anchor (state level) × factor (its census division, that year).
#  
# Anchor year = 2025. Single scenario: AEO2026 Counterfactual Baseline.
#  
# Run order matters — each section defines variables the next one consumes

# %% [markdown]
# ## Imports, configuration, and shared EIA API helpers

# %%
import os
import time
import requests
import pandas as pd
 
from config import PROJECT_ROOT   # repo-root anchor used for all data paths
 
AEO_YEAR      = 2026
ANCHOR_YEAR   = 2025
SCENARIO_ID   = "cb2026"                          # Counterfactual Baseline
SCENARIO_NAME = "AEO2026 Counterfactual Baseline"

# AEO region IDs (confirmed by API probe): 1-0 = National, 1-1 = New England, … 1-9 = Pacific
REGION_IDS = ["1-0", "1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8", "1-9"]

EIA_API_KEY = input("Enter your EIA API key: ")

def _eia_get(route: str, **params) -> list:
    """Paginated EIA API v2 GET for `/data` endpoints. Returns all rows.
 
    Handles AEO datasets large enough to require offset pagination, and backs off
    on HTTP 429 rate limits.
 
    Args:
        route: API route after the v2 base, e.g. 'aeo/2026/data'.
        **params: Extra query params (facets, frequency, start, end).
 
    Returns:
        A list of row dicts across all pages.
    """
    url = f"https://api.eia.gov/v2/{route}"
    all_rows, offset = [], 0
    while True:
        for attempt in range(4):
            r = requests.get(
                url,
                params={"api_key": EIA_API_KEY,
                        "data[]": "value", "length": 5000, "offset": offset,
                        **params},
                timeout=30,
            )
            if r.status_code == 429:
                wait = 30 * (attempt + 1)          # 30s, 60s, 90s, 120s
                print(f"Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                r.raise_for_status()
                break
        body = r.json()["response"]
        rows = body.get("data", [])
        all_rows.extend(rows)
        if len(all_rows) >= int(body.get("total", len(all_rows))):
            break
        offset += 5000
        time.sleep(2)                              # polite gap between pages
    return all_rows
 
 
def _eia_fetch(route: str, **params) -> list:
    """Single-request EIA API v2 GET for small state/regional datasets.
 
    Appropriate for electricity, natural gas, and petroleum price endpoints where the
    full dataset fits in one response (no pagination).
 
    Args:
        route: API route after the v2 base, e.g. 'electricity/retail-sales/data'.
        **params: Query params (data[], facets, frequency, start, end, length).
 
    Returns:
        A list of row dicts.
    """
    r = requests.get(
        f"https://api.eia.gov/v2/{route}",
        params={"api_key": EIA_API_KEY, **params},
        timeout=30,
    )
    r.raise_for_status()
    body = r.json()["response"]
    return body.get("data", [])
 
 
def _fetch_seriesid(v1_series_id: str) -> list:
    """Fetch one series via the v2 SeriesID backward-compatibility path.
 
    Needed for AEO Table 4 (HDD/CDD), which is not reachable through `facets[tableId][]`.
    Takes only start/end — no data[], facets, or pagination params.
 
    Args:
        v1_series_id: A v1-style series ID, e.g. 'AEO.2026.CB2026.KEI_HDD_..._HDD.A'.
 
    Returns:
        A list of row dicts for that series (ANCHOR_YEAR through 2050).
    """
    r = requests.get(
        f"https://api.eia.gov/v2/seriesid/{v1_series_id}",
        params={"api_key": EIA_API_KEY,
                "start": str(ANCHOR_YEAR), "end": "2050"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["response"]["data"]

# %% [markdown]
# ## Geographic mapping tables and the unit-conversion helper

# %%
ALL_STATES = [
    'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN',
    'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH',
    'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
    'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
]   # 51 locations: 50 states + DC
 
# Census division → member states. Used to JOIN each state's anchor price to its AEO
# projection factor (factors are published per census division). Same partition as
# create_lookup_fuel_prices.py.
DIVISION_TO_STATES = {
    "New England":        ["CT", "ME", "MA", "NH", "RI", "VT"],
    "Middle Atlantic":    ["NJ", "NY", "PA"],
    "East North Central": ["IN", "IL", "MI", "OH", "WI"],
    "West North Central": ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South Atlantic":     ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"],
    "East South Central": ["AL", "KY", "MS", "TN"],
    "West South Central": ["AR", "LA", "OK", "TX"],
    "Mountain":           ["AZ", "CO", "ID", "NM", "MT", "UT", "NV", "WY"],
    "Pacific":            ["AK", "CA", "HI", "OR", "WA"],
}
STATE_TO_DIVISION = {s: d for d, states in DIVISION_TO_STATES.items() for s in states}
 
# PADD / sub-PADD → member states, at the FINEST EIA petroleum granularity available.
# These EIA `duoarea` codes are what petroleum/pri/wfr returns:
#   R1X = PADD 1A (New England)      R30 = PADD 3 (Gulf Coast)
#   R1Y = PADD 1B (Central Atlantic) R40 = PADD 4 (Rocky Mountain)
#   R1Z = PADD 1C (Lower Atlantic)   R50 = PADD 5 (West Coast)
#   R20 = PADD 2 (Midwest)           R10 = PADD 1 (East Coast aggregate; parent of 1A/1B/1C)
#   NUS = U.S. National
# BROADCAST: every state in a PADD inherits that PADD's oil/propane price, so states in the
# same PADD carry IDENTICAL oil/propane anchors. This is expected and recorded in the
# `source_region` column so it is auditable, not silent.
PADD_TO_STATES = {
    "R1X": ["CT", "ME", "MA", "NH", "RI", "VT"],                                  # PADD 1A
    "R1Y": ["DE", "DC", "MD", "NJ", "NY", "PA"],                                  # PADD 1B
    "R1Z": ["FL", "GA", "NC", "SC", "VA", "WV"],                                  # PADD 1C
    "R20": ["IL", "IN", "IA", "KS", "KY", "MI", "MN", "MO", "NE", "ND", "OH",
            "OK", "SD", "TN", "WI"],                                              # PADD 2
    "R30": ["AL", "AR", "LA", "MS", "NM", "TX"],                                  # PADD 3
    "R40": ["CO", "ID", "MT", "UT", "WY"],                                        # PADD 4
    "R50": ["AK", "AZ", "CA", "HI", "NV", "OR", "WA"],                            # PADD 5
}
STATE_TO_PADD = {s: r for r, states in PADD_TO_STATES.items() for s in states}
PADD1_SUBREGIONS = {"R1X", "R1Y", "R1Z"}          # these fall back to R10 (PADD 1) if needed
 
# Human-readable region labels for the provenance column.
REGION_LABELS = {
    "R1X": "PADD 1A New England", "R1Y": "PADD 1B Central Atlantic",
    "R1Z": "PADD 1C Lower Atlantic", "R10": "PADD 1 East Coast",
    "R20": "PADD 2 Midwest", "R30": "PADD 3 Gulf Coast",
    "R40": "PADD 4 Rocky Mountain", "R50": "PADD 5 West Coast",
    "NUS": "U.S. National",
}
 
# AEO regionName (from Table 3) → census-division string the factor table uses.
REGION_MAP = {
    "United States":      "National",
    "New England":        "New England",
    "Middle Atlantic":    "Middle Atlantic",
    "East North Central": "East North Central",
    "West North Central": "West North Central",
    "South Atlantic":     "South Atlantic",
    "East South Central": "East South Central",
    "West South Central": "West South Central",
    "Mountain":           "Mountain",
    "Pacific":            "Pacific",
}
 
# AEO Table 3 series name → codebase fuel_type string (both nominal and real variants).
SERIES_TO_FUEL = {
    "Energy Prices : Residential : Electricity":                   "electricity",
    "Energy Prices : Residential : Natural Gas":                   "naturalGas",
    "Energy Prices : Residential : Distillate Fuel Oil":          "fuelOil",
    "Energy Prices : Residential : Propane":                       "propane",
    "Energy Prices : Nominal : Residential : Electricity":         "electricity",
    "Energy Prices : Nominal : Residential : Natural Gas":         "naturalGas",
    "Energy Prices : Nominal : Residential : Distillate Fuel Oil": "fuelOil",
    "Energy Prices : Nominal : Residential : Propane":             "propane",
}
 
 
def to_usd2025_per_kwh(value: float, fuel_type: str) -> float:
    """Convert a 2025 residential fuel price from its base unit to USD2025 per kWh.
 
    Conversion factors are identical to process_fuel_price_data() in
    create_lookup_fuel_prices.py — the math has just moved into this notebook so the
    exported CSV is already in one consistent unit. No CPI step: 2025 actuals are already
    in 2025 dollars (the CPI ratio 2025/2025 = 1.0).
 
    Base units by fuel:
        electricity : cents per kWh
        naturalGas  : dollars per MCF (thousand cubic feet)
        fuelOil     : dollars per gallon
        propane     : dollars per gallon
 
    Args:
        value: Price in the fuel's base unit (2025 nominal = 2025 real).
        fuel_type: One of {'electricity', 'naturalGas', 'fuelOil', 'propane'}.
 
    Returns:
        Price in USD2025 per kWh.
 
    Raises:
        ValueError: If fuel_type is unrecognized.
    """
    if fuel_type == "electricity":      # cents/kWh → $/kWh
        return value / 100
    if fuel_type == "naturalGas":       # $/MCF → $/cf → $/BTU → $/kWh  (1 cf ≈ 1039 BTU)
        return value * (1 / 1000) * (1 / 1039) * 3412
    if fuel_type == "fuelOil":          # $/gal → $/BTU → $/kWh  (1 gal heating oil ≈ 138,500 BTU)
        return value * (1 / 138500) * 3412
    if fuel_type == "propane":          # $/gal → $/BTU → $/kWh  (1 gal propane ≈ 91,452 BTU)
        return value * (1 / 91452) * 3412
    raise ValueError(f"Unknown fuel_type: {fuel_type!r}")
 
 
# Output paths (all relative to the repo root).
PATH_FUEL_PRICE_ANCHOR = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "fuel_prices",
    "eia_fuel_price_data_2025_usd2025.csv")
PATH_FUEL_PRICE_FACTORS = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "projections",
    "aeo2026_fuel_price_factors_2025_2050.csv")
PATH_DEGREE_DAY_FACTORS = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "projections",
    "aeo2026_degree_day_factors_2025_2050.csv")

# %% [markdown]
# ## AEO2026 PROJECTION FACTORS: Fuel Prices

# %%
# Fetch AEO2026 Table 3 (Energy Prices), residential series, all 10 regions, 2025–2050.
rows = _eia_get(
    f"aeo/{AEO_YEAR}/data",
    **{
        "facets[scenario][]": SCENARIO_ID,
        "facets[tableId][]":  "3",
        "facets[regionId][]": REGION_IDS,
        "frequency":          "annual",
        "start":              str(ANCHOR_YEAR),
        "end":                "2050",
    },
)
df_aeo_prices_raw = pd.DataFrame(rows)
 
# Keep only residential energy-price series.
mask_residential = (
    df_aeo_prices_raw["seriesName"].str.contains("Residential", na=False) &
    df_aeo_prices_raw["seriesName"].str.contains("Energy Prices", na=False)
)
df_aeo_prices_raw = df_aeo_prices_raw[mask_residential].copy()
print(f"AEO Table 3 residential rows: {len(df_aeo_prices_raw)} (expect 2080 = 8 series × 10 regions × 26 yr)")
print(f"  Units: {df_aeo_prices_raw['unit'].unique()}")
 
# Tidy: map to codebase fuel_type / census-division / price_type, coerce numerics.
df = df_aeo_prices_raw.copy()
df["fuel_type"]  = df["seriesName"].map(SERIES_TO_FUEL)
df["region"]     = df["regionName"].map(REGION_MAP)
df["price_type"] = df["unit"].map({"2025 $/MMBtu": "real_2025", "nom $/MMBtu": "nominal"})
df["year"]       = df["period"].astype(int)
df["value"]      = pd.to_numeric(df["value"], errors="coerce")
 
df_aeo_prices = (
    df[df["fuel_type"].notna() & df["region"].notna() & df["price_type"].notna()]
    [["region", "fuel_type", "price_type", "year", "value"]]
    .dropna()
    .sort_values(["fuel_type", "region", "price_type", "year"])
    .reset_index(drop=True)
)
print(f"  After tidy: {len(df_aeo_prices)} rows | "
      f"fuels={sorted(df_aeo_prices['fuel_type'].unique())} | "
      f"regions={df_aeo_prices['region'].nunique()} | "
      f"price_types={list(df_aeo_prices['price_type'].unique())}")
 
# Build projection factors from the REAL 2025-dollar series (units cancel in the ratio,
# but real series keeps the factor interpretation clean). Each year ÷ the 2025 value.
df_real = df_aeo_prices[df_aeo_prices["price_type"] == "real_2025"].copy()
pivot = df_real.pivot_table(index=["region", "fuel_type"], columns="year", values="value")
 
if ANCHOR_YEAR not in pivot.columns:
    raise ValueError(
        f"Anchor year {ANCHOR_YEAR} missing from AEO prices. "
        f"Available: {sorted(pivot.columns.tolist())}"
    )
 
df_projection_factors = pivot.div(pivot[ANCHOR_YEAR], axis=0).reset_index()
df_projection_factors["policy_scenario"] = SCENARIO_NAME
 
# Invariant: every factor at the anchor year must be exactly 1.0.
anchor_vals = df_projection_factors[ANCHOR_YEAR].round(9).unique()
assert set(anchor_vals) == {1.0}, f"Anchor-year factors must all be 1.0; got {anchor_vals}"
print(f"  Projection factors: shape {df_projection_factors.shape}; all {ANCHOR_YEAR} factors = 1.0 ✓")
 
# Export. create_projection_factors_dict() already casts string year headers back to int,
# so this CSV round-trips safely on the consumer side.
os.makedirs(os.path.dirname(PATH_FUEL_PRICE_FACTORS), exist_ok=True)
df_projection_factors.to_csv(PATH_FUEL_PRICE_FACTORS, index=False)
print(f"  [PASS] Wrote fuel-price factors → {os.path.relpath(PATH_FUEL_PRICE_FACTORS, PROJECT_ROOT)}")
 

# %% [markdown]
# ## 2025 nominal ANCHOR prices, per state, by fuel 

# %% [markdown]
# ### Electricity — state-level residential retail price (cents/kWh) 

# %%
# The /data suffix is required to get records (the bare route returns a schema dict).
rows_elec = _eia_fetch(
    "electricity/retail-sales/data",
    **{
        "data[]":             "price",       # cents/kWh
        "facets[sectorid][]": "RES",         # residential only
        "frequency":          "annual",
        "start":              "2025",
        "end":                "2025",
        "length":             100,           # headroom (raw returns ~62: states + US + aggregates)
    },
)
df_elec_raw = pd.DataFrame(rows_elec)
df_elec_raw["price"] = pd.to_numeric(df_elec_raw["price"], errors="coerce")
 
# Keep the 51 state/DC rows; capture 'US' separately as the National anchor.
elec_states = (
    df_elec_raw[df_elec_raw["stateid"].isin(ALL_STATES)][["stateid", "price"]]
    .rename(columns={"stateid": "state", "price": "base_value"})
    .copy()
)
elec_states["source_region"] = elec_states["state"]      # measured per state
elec_states["source_method"] = "state_annual_2025"
elec_states["fuel_type"]     = "electricity"
elec_states["base_unit"]     = "cents/kWh"
df_elec_final = elec_states
 
elec_us = df_elec_raw[df_elec_raw["stateid"] == "US"]["price"]
elec_national = float(elec_us.iloc[0]) if len(elec_us) else float("nan")
print(f"Electricity: {len(df_elec_final)} state rows (expect 51); National = {elec_national} cents/kWh")

# %% [markdown]
# ### Natural gas — state-level residential price ($/MCF), with monthly fallback

# %%
ng_kwargs = {
    "data[]":            "value",
    "facets[process][]": "PRS",              # Price delivered to Residential consumers
    "facets[product][]": "EPG0",             # natural gas (excludes NGLs, LNG)
    "frequency":         "annual",
    "start":             "2025",
    "end":               "2025",
    "length":            200,
}
df_ng_annual = pd.DataFrame(_eia_fetch("natural-gas/pri/sum/data", **ng_kwargs))
df_ng_annual["value"] = pd.to_numeric(df_ng_annual["value"], errors="coerce")
 
# State rows: duoarea is 'S' + 2-letter abbrev (exactly 3 chars). 'NUS' etc. excluded.
state_mask = (
    df_ng_annual["duoarea"].str.startswith("S") &
    (df_ng_annual["duoarea"].str.len() == 3)
)
df_ng_states = df_ng_annual[state_mask].copy()
df_ng_states["state_abbr"] = df_ng_states["duoarea"].str[1:]          # 'SAK' → 'AK'
df_ng_nonstate = df_ng_annual[~state_mask]
 
states_with_data = set(df_ng_states[df_ng_states["value"].notna()]["state_abbr"])
missing_annual = [s for s in ALL_STATES if s not in states_with_data]
print(f"Natural gas: {len(states_with_data)} states with annual 2025; "
      f"{len(missing_annual)} missing → {missing_annual}")

# %%
# Monthly fallback for states without an annual figure (EIA publishes with a ~2-month lag).
if missing_annual:
    df_ng_monthly = pd.DataFrame(_eia_fetch(
        "natural-gas/pri/sum/data",
        **{**ng_kwargs, "frequency": "monthly", "start": "2025-01", "end": "2025-12",
           "length": 700},                                          # 51 × 12 = 612 max
    ))
    df_ng_monthly["value"] = pd.to_numeric(df_ng_monthly["value"], errors="coerce")
    missing_duoareas = {f"S{s}" for s in missing_annual}
    df_ng_monthly = df_ng_monthly[df_ng_monthly["duoarea"].isin(missing_duoareas)].copy()
    df_ng_monthly["state_abbr"] = df_ng_monthly["duoarea"].str[1:]
 
    ng_fill = (
        df_ng_monthly.groupby("state_abbr")["value"]
        .agg(value="mean", months_used="count")
        .reset_index()
    )
    ng_fill["source_method"] = "monthly_mean_2025"
 
    df_ng_states["source_method"] = "annual_2025"
    df_ng_combined = pd.concat([
        df_ng_states[["state_abbr", "value", "source_method"]],
        ng_fill[["state_abbr", "value", "source_method"]],
    ], ignore_index=True)
else:
    df_ng_states["source_method"] = "annual_2025"
    df_ng_combined = df_ng_states[["state_abbr", "value", "source_method"]].copy()
 

# %%
# National row: prefer the non-state 'U.S.' row from the annual fetch; else the series ID.
nat_row = df_ng_nonstate[df_ng_nonstate["area-name"].str.contains("U.S.", case=False, na=False)]
if len(nat_row):
    ng_national = float(nat_row.iloc[0]["value"])
    ng_nat_method = "annual_PRS_fetch"
else:
    df_nat = pd.DataFrame(_fetch_seriesid("NG.N3010US3.A"))
    df_nat["value"] = pd.to_numeric(df_nat["value"], errors="coerce")
    nat_2025 = df_nat[df_nat["period"].astype(str) == "2025"]
    if len(nat_2025):
        ng_national = float(nat_2025.iloc[0]["value"])
        ng_nat_method = "series_id_NG.N3010US3.A"
    else:
        ng_national = float(df_nat.dropna(subset=["value"]).iloc[0]["value"])
        ng_nat_method = f"series_id_NG.N3010US3.A_latest({df_nat.iloc[0]['period']})"
 
df_ng_combined = pd.concat([
    df_ng_combined,
    pd.DataFrame({"state_abbr": ["National"], "value": [ng_national],
                  "source_method": [ng_nat_method]}),
], ignore_index=True)

# %%
# Drop NaN rows first, then keep one row per state.
df_ng = (
    df_ng_combined.dropna(subset=["value"])
    .drop_duplicates(subset=["state_abbr"], keep="first")
    .copy()
)
df_ng_final = (
    df_ng[df_ng["state_abbr"].isin(ALL_STATES)]
    .rename(columns={"state_abbr": "state", "value": "base_value"})
    .copy()
)
df_ng_final["fuel_type"]     = "naturalGas"
df_ng_final["base_unit"]     = "$/MCF"
df_ng_final["source_region"] = df_ng_final["state"]
print(f"Natural gas: {len(df_ng_final)} state rows (expect 51); National = {ng_national:.2f} $/MCF")

# %% [markdown]
# ### Heating oil + propane — PADD broadcast to states ($/gal)
# 
# EIA reports these residential prices only at PADD / sub-PADD level. Fetch the full 2025 monthly series, average available months per region, then broadcast to member states.
# 
# Coverage is incomplete BY DESIGN (heating oil is a NE/Midwest fuel): a state whose PADD did not report falls back to the National price, flagged in source_method.

# %%
def fetch_petroleum_region_prices(product_code: str) -> tuple[dict, float]:
    """Fetch 2025 residential petroleum prices and average by EIA region (duoarea).
 
    Args:
        product_code: EIA product code — 'EPD2F' (heating oil) or 'EPLLPA' (propane).
 
    Returns:
        (region_price, national_price):
            region_price: {duoarea_code: mean 2025 $/gal} for regions that reported.
            national_price: the 'NUS' mean, or NaN if absent.
    """
    rows_pet = _eia_fetch(
        "petroleum/pri/wfr/data",
        **{
            "data[]":            "value",
            "facets[product][]": product_code,
            "facets[process][]": "PRS",
            "frequency":         "monthly",
            "start":             "2025-01",
            "end":               "2025-12",
            "length":            5000,          # full pull — no discovery cap
        },
    )
    df_pet = pd.DataFrame(rows_pet)
    df_pet["value"] = pd.to_numeric(df_pet["value"], errors="coerce")
    region_mean = df_pet.dropna(subset=["value"]).groupby("duoarea")["value"].mean()
    region_price = region_mean.to_dict()
    return region_price, float(region_price.get("NUS", float("nan")))
 
 
def resolve_state_petroleum_price(state: str, region_price: dict, national: float) -> tuple:
    """Resolve one state's petroleum price: finest region → PADD-1 parent → National.
 
    Args:
        state: 2-letter state/DC abbreviation.
        region_price: {duoarea_code: price $/gal} for regions that reported.
        national: National (NUS) price, used as last-resort fallback.
 
    Returns:
        (price, source_region_code, source_method).
    """
    chain = [STATE_TO_PADD[state]]
    if STATE_TO_PADD[state] in PADD1_SUBREGIONS:
        chain.append("R10")                     # PADD-1 states fall back to the PADD-1 aggregate
    chain.append("NUS")
 
    for code in chain:
        if code in region_price and pd.notna(region_price[code]):
            method = "national_fallback" if code == "NUS" else "padd_broadcast_monthly_mean_2025"
            return region_price[code], code, method
    return national, "NUS", "national_fallback"
 
 
def build_petroleum_fuel_df(product_code: str, fuel_type: str) -> tuple[pd.DataFrame, float]:
    """Build a state-level DataFrame for one petroleum fuel by broadcasting PADD prices.
 
    Args:
        product_code: 'EPD2F' (heating oil) or 'EPLLPA' (propane).
        fuel_type: 'fuelOil' or 'propane' (the codebase fuel string).
 
    Returns:
        (df_fuel, national_price): df_fuel columns
        [state, base_value, source_region, source_method, fuel_type, base_unit].
    """
    region_price, national = fetch_petroleum_region_prices(product_code)
    records = []
    for state in ALL_STATES:
        price, code, method = resolve_state_petroleum_price(state, region_price, national)
        records.append({"state": state, "base_value": price,
                        "source_region": code, "source_method": method})
    df_fuel = pd.DataFrame(records)
    df_fuel["fuel_type"] = fuel_type
    df_fuel["base_unit"] = "$/gal"
 
    n_fallback = int((df_fuel["source_method"] == "national_fallback").sum())
    reported = sorted(r for r in region_price if r != "NUS")
    print(f"{fuel_type}: regions reporting = {[REGION_LABELS.get(r, r) for r in reported]}")
    print(f"{fuel_type}: {n_fallback} state(s) → National fallback "
          f"(no PADD-level residential {fuel_type}); National = {national:.3f} $/gal")
    return df_fuel, national
 
 
df_oil_final,     oil_national     = build_petroleum_fuel_df("EPD2F",  "fuelOil")
df_propane_final, propane_national = build_petroleum_fuel_df("EPLLPA", "propane")

# %% [markdown]
# ### Combine all four fuels --> Normalize units --> export CSV

# %%
common_cols = ["state", "fuel_type", "base_value", "base_unit", "source_region", "source_method"]
df_all = pd.concat(
    [df_elec_final[common_cols], df_ng_final[common_cols],
     df_oil_final[common_cols],  df_propane_final[common_cols]],
    ignore_index=True,
)
 
df_all["census_division"]     = df_all["state"].map(STATE_TO_DIVISION)
df_all["source_region_label"] = df_all["source_region"].map(lambda c: REGION_LABELS.get(c, c))
df_all["price_usd2025_per_kwh"] = df_all.apply(
    lambda r: to_usd2025_per_kwh(r["base_value"], r["fuel_type"]), axis=1
)
 
# National anchor rows (one per fuel) so downstream lookups have a National fallback.
national_base = {
    "electricity": (elec_national,    "cents/kWh"),
    "naturalGas":  (ng_national,      "$/MCF"),
    "fuelOil":     (oil_national,     "$/gal"),
    "propane":     (propane_national, "$/gal"),
}
national_rows = [{
    "state": "National", "fuel_type": f, "base_value": v, "base_unit": u,
    "source_region": "NUS", "source_method": "national_2025",
    "census_division": "National", "source_region_label": "U.S. National",
    "price_usd2025_per_kwh": to_usd2025_per_kwh(v, f),
} for f, (v, u) in national_base.items()]
df_all = pd.concat([df_all, pd.DataFrame(national_rows)], ignore_index=True)
 
df_fuel_prices_2025 = df_all[[
    "state", "census_division", "fuel_type",
    "price_usd2025_per_kwh", "base_value", "base_unit",
    "source_region", "source_region_label", "source_method",
]].sort_values(["fuel_type", "state"]).reset_index(drop=True)
 
# Validation: 51 states × 4 fuels + 4 National = 208 rows; no missing prices.
expected_state_rows = len(ALL_STATES) * 4
n_state_rows = int((df_fuel_prices_2025["state"] != "National").sum())
assert n_state_rows == expected_state_rows, (
    f"Expected {expected_state_rows} state×fuel rows, got {n_state_rows}."
)
missing_price = df_fuel_prices_2025[df_fuel_prices_2025["price_usd2025_per_kwh"].isna()]
if len(missing_price):
    print(f"[WARNING] {len(missing_price)} row(s) have no price:")
    print(missing_price[["state", "fuel_type", "source_method"]].to_string(index=False))
 
# Make the broadcast auditable: which states share an oil/propane price?
print("\nShared petroleum prices by source region (same PADD ⇒ identical price):")
for fuel in ["fuelOil", "propane"]:
    grp = (df_fuel_prices_2025[df_fuel_prices_2025["fuel_type"] == fuel]
           .groupby("source_region_label")["state"].apply(list))
    print(f"  {fuel}:")
    for region, states in grp.items():
        print(f"    {region}: {states}")
 
# National anchors — spot-check against EIA's published 2025 residential figures.
print("\nNational anchors (USD2025/kWh):")
print(df_fuel_prices_2025[df_fuel_prices_2025["state"] == "National"]
      [["fuel_type", "price_usd2025_per_kwh", "base_value", "base_unit"]].to_string(index=False))
 
os.makedirs(os.path.dirname(PATH_FUEL_PRICE_ANCHOR), exist_ok=True)
df_fuel_prices_2025.to_csv(PATH_FUEL_PRICE_ANCHOR, index=False)
print(f"\n[PASS] Wrote {len(df_fuel_prices_2025)} anchor rows → "
      f"{os.path.relpath(PATH_FUEL_PRICE_ANCHOR, PROJECT_ROOT)}")

# %%
display(df_fuel_prices_2025)

# %% [markdown]
# ## AEO2026 API --> CDD/HDD Projection Factors --> Export CSV

# %%
# Degree-day Series IDs
HDD_SERIES_IDS = {
    "New England":        "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_NENGL_HDD.A",
    "Middle Atlantic":    "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_MATL_HDD.A",
    "East North Central": "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_ENC_HDD.A",
    "West North Central": "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_WNC_HDD.A",
    "South Atlantic":     "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_SATL_HDD.A",
    "East South Central": "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_ESC_HDD.A",
    "West South Central": "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_WSC_HDD.A",
    "Mountain":           "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_MTN_HDD.A",
    "Pacific":            "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_PCF_HDD.A",
    "National":           "AEO.2026.CB2026.KEI_HDD_RESD_NA_NA_NA_USA_HDD.A",
}
CDD_SERIES_IDS = {
    "New England":        "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_NENGL_CDD.A",
    "Middle Atlantic":    "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_MATL_CDD.A",
    "East North Central": "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_ENC_CDD.A",
    "West North Central": "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_WNC_CDD.A",
    "South Atlantic":     "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_SATL_CDD.A",
    "East South Central": "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_ESC_CDD.A",
    "West South Central": "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_WSC_CDD.A",   # NOT _WSC_HDD (silent-corruption guard)
    "Mountain":           "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_MTN_CDD.A",
    "Pacific":            "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_PCF_CDD.A",
    "National":           "AEO.2026.CB2026.KEI_CDD_RESD_NA_NA_NA_USA_CDD.A",
}

# Fetch Degree-day Rows from AEO2026 API
def fetch_degree_day_rows(series_ids: dict, dd_type: str) -> list:
    """Fetch every series in a degree-day dict, tagging each row with its division and type.
 
    Degree-day rows return regionName = 'No Regional Tables', so the census_division label
    MUST come from the dict key — a wrong series ID corrupts silently.
 
    Args:
        series_ids: {census_division: v1_series_id}.
        dd_type: 'hdd' or 'cdd'.
 
    Returns:
        A flat list of row dicts, each tagged with census_division and dd_type.
    """
    out = []
    for division, series_id in series_ids.items():
        for row in _fetch_seriesid(series_id):
            row["census_division"] = division
            row["dd_type"] = dd_type
            out.append(row)
        time.sleep(0.3)
    return out

df_dd_raw = pd.DataFrame(
    fetch_degree_day_rows(HDD_SERIES_IDS, "hdd") +
    fetch_degree_day_rows(CDD_SERIES_IDS, "cdd")
)
print(f"Degree days: {len(df_dd_raw)} rows (expect 520 = 10 divisions × 2 types × 26 yr)")

# %%
# Degree-day Projection factor table
def degree_day_factor_table(df_raw: pd.DataFrame, dd_type: str) -> pd.DataFrame:
    """Pivot one degree-day type to wide format and normalize so ANCHOR_YEAR = 1.0.
 
    Args:
        df_raw: Tagged raw rows (must contain census_division, dd_type, period, value).
        dd_type: 'hdd' or 'cdd'.
 
    Returns:
        DataFrame with a census_division column and integer year columns (factors).
 
    Raises:
        ValueError: If the anchor year is absent.
    """
    d = df_raw[df_raw["dd_type"] == dd_type].copy()
    d["year"]  = d["period"].astype(int)
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    pivot = d.pivot_table(index="census_division", columns="year", values="value")
    if ANCHOR_YEAR not in pivot.columns:
        raise ValueError(f"Anchor year {ANCHOR_YEAR} not found. Available: {sorted(pivot.columns)}")
    return pivot.div(pivot[ANCHOR_YEAR], axis=0).reset_index()

df_hdd_factors = degree_day_factor_table(df_dd_raw, "hdd")
df_cdd_factors = degree_day_factor_table(df_dd_raw, "cdd")

# In-memory lookups (int year keys, built from the live DataFrames) for verification and
# optional direct wiring into degree_day_consumption_utils.py.
lookup_hdd_factor = df_hdd_factors.set_index("census_division").to_dict("index")
lookup_cdd_factor = df_cdd_factors.set_index("census_division").to_dict("index")

# Hard invariant: anchor-year factor == 1.0 for every division.
for label, lookup in [("HDD", lookup_hdd_factor), ("CDD", lookup_cdd_factor)]:
    bad = {d: v[ANCHOR_YEAR] for d, v in lookup.items() if round(v[ANCHOR_YEAR], 6) != 1.0}
    assert not bad, f"{label} anchor-year factor != 1.0 for {bad}"

# %%
# Directional sanity across ALL divisions (catches the WSC-typo class of bug):
# warming ⇒ HDD factor < 1.0 and CDD factor > 1.0 by 2040. Report violators, don't crash.
CHECK_YEAR = 2040
hdd_viol = [d for d, v in lookup_hdd_factor.items() if v.get(CHECK_YEAR, 1.0) >= 1.0]
cdd_viol = [d for d, v in lookup_cdd_factor.items() if v.get(CHECK_YEAR, 1.0) <= 1.0]
print(f"Anchor-year factors = 1.0 for all divisions ✓")
if hdd_viol:
    print(f"[CHECK] HDD {CHECK_YEAR} not < 1.0 for {hdd_viol} — verify HDD series IDs")
if cdd_viol:
    print(f"[CHECK] CDD {CHECK_YEAR} not > 1.0 for {cdd_viol} — verify CDD series IDs")
if not hdd_viol and not cdd_viol:
    print(f"Directional check passed: all HDD {CHECK_YEAR} < 1.0 and all CDD {CHECK_YEAR} > 1.0 ✓")

# Combine into one CSV with a dd_type column (replaces the two Excel sheets).
df_dd_factors = pd.concat([
    df_hdd_factors.assign(dd_type="hdd"),
    df_cdd_factors.assign(dd_type="cdd"),
], ignore_index=True)
# Put identifier columns first, year columns after.
id_cols = ["census_division", "dd_type"]
year_cols = [c for c in df_dd_factors.columns if c not in id_cols]
df_dd_factors = df_dd_factors[id_cols + year_cols]

display(df_dd_factors)

# %%
os.makedirs(os.path.dirname(PATH_DEGREE_DAY_FACTORS), exist_ok=True)
df_dd_factors.to_csv(PATH_DEGREE_DAY_FACTORS, index=False)
print(f"[PASS] Wrote degree-day factors → {os.path.relpath(PATH_DEGREE_DAY_FACTORS, PROJECT_ROOT)}")

# ── READ-BACK NOTE for degree_day_consumption_utils.py ──────────────────────────────────
# CSV headers are strings, but the lookups do `.get(year_label, 1.0)` with INT years.
# Cast year columns back to int on read, or every lookup silently misses and returns 1.0:
#
#   df = pd.read_csv(PATH_DEGREE_DAY_FACTORS)
#   df.columns = [int(c) if str(c).isdigit() else c for c in df.columns]
#   lookup_hdd_factor = (df[df.dd_type == "hdd"].drop(columns="dd_type")
#                          .set_index("census_division").to_dict("index"))
#   lookup_cdd_factor = (df[df.dd_type == "cdd"].drop(columns="dd_type")
#                          .set_index("census_division").to_dict("index"))

# %%



