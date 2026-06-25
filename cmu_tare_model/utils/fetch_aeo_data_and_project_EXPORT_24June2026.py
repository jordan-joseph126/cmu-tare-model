# %% [markdown]
# # Fetch EIA Fuel Price Data using API and Project Fuel Prices
#  
# Replaces the manually maintained Excel workbook `aeo_projections_2022_2050.xlsx`
# (three sheets) with three on-disk CSV artifacts, all sourced live from the EIA API v2
# using AEO2026 Counterfactual Baseline data:
#  
#     Old Excel sheet                  -->  New CSV artifact (this file writes it)
#     ---------------------------------------------------------------------------------
#     fuel_price_factors_2022_2050     -->  data/projections/aeo2026_fuel_price_factors_2025_2050.csv
#     hdd_factors_2022_2050 +          -->  data/projections/aeo2026_degree_day_factors_2025_2050.csv
#     cdd_factors_2022_2050               (combined, with a `dd_type` column: 'hdd' / 'cdd')
#     (none -- new this refactor)      -->  data/fuel_prices/eia_fuel_price_data_2025_usd2025.csv
#                                         (the 2025 state-level price ANCHOR in USD2025/kWh)
#  
# Two distinct things are produced:
#   1. ANCHOR PRICES (the price LEVEL): 2025 residential prices per state, in USD2025/kWh.
#   2. PROJECTION FACTORS (the curve SHAPE): each year's price / the 2025 price, so 2025 = 1.0.
# A future-year price = anchor (state level) x factor (its census division, that year).
#  
# Anchor year = 2025. Single scenario: AEO2026 Counterfactual Baseline.
#  
# Run order matters -- each section defines variables the next one consumes

# %% [markdown]
# ## Imports, configuration, and shared EIA API helpers

# %%
import os
import pandas as pd

from config import PROJECT_ROOT   # repo-root anchor used for all data paths
from cmu_tare_model.constants import PRINT_VERBOSE_DATAFRAMES

from cmu_tare_model.utils.eia_api_utils import (
    ALL_STATES, STATE_TO_DIVISION,
    REGION_LABELS, REGION_MAP, SERIES_TO_FUEL,
    HDD_SERIES_IDS, CDD_SERIES_IDS,
    eia_get, eia_fetch, fetch_seriesid,
    to_usd2025_per_kwh,
    fetch_degree_day_rows, degree_day_factor_table,
    build_petroleum_fuel_df,
)

AEO_YEAR = 2026
ANCHOR_YEAR = 2025
SCENARIO_ID = "cb2026"  # Counterfactual Baseline
SCENARIO_NAME = "AEO2026 Counterfactual Baseline"

# AEO region IDs (confirmed by API probe): 1-0 = National, 1-1 = New England, ... 1-9 = Pacific
REGION_IDS = ["1-0", "1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8", "1-9"]

EIA_API_KEY = input("Enter your EIA API key: ")

# Output paths (all relative to the repo root).
PATH_FUEL_PRICE_ANCHOR = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "fuel_prices",
    "eia_fuel_price_data_2025_usd2025.csv")
print(f"Fuel price anchor data will be saved to: {PATH_FUEL_PRICE_ANCHOR}")

PATH_FUEL_PRICE_FACTORS = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "projections",
    "aeo2026_fuel_price_factors_2025_2050.csv")
print(f"Fuel price factors will be saved to: {PATH_FUEL_PRICE_FACTORS}")

PATH_DEGREE_DAY_FACTORS = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "projections",
    "aeo2026_degree_day_factors_2025_2050.csv")
print(f"Degree day factors will be saved to: {PATH_DEGREE_DAY_FACTORS}")

# %% [markdown]
# ## AEO2026 PROJECTION FACTORS: Fuel Prices

# %%
# Fetch AEO2026 Table 3 (Energy Prices), residential series, all 10 regions, 2025-2050.
aeo_price_params = {
    "facets[scenario][]": SCENARIO_ID,
    "facets[tableId][]": "3",
    "facets[regionId][]": REGION_IDS,
    "frequency": "annual",
    "start": str(ANCHOR_YEAR),
    "end": "2050",
}
rows = eia_get(f"aeo/{AEO_YEAR}/data", api_key=EIA_API_KEY, **aeo_price_params)
df_aeo_prices_raw = pd.DataFrame(rows)

# Keep only residential energy-price series. AEO Table 3 returns all sectors
# (residential, commercial, industrial) in a single response; we need only the
# residential rows to match the end-use prices paid by homeowners in the model.
mask_residential = (
    df_aeo_prices_raw["seriesName"].str.contains("Residential", na=False) &
    df_aeo_prices_raw["seriesName"].str.contains("Energy Prices", na=False)
)
df_aeo_prices_raw = df_aeo_prices_raw[mask_residential].copy()
print(
    f"AEO Table 3 residential rows: {len(df_aeo_prices_raw)}"
    f" (expect 2080 = 8 series x 10 regions x 26 yr)"
)
print(f"  Units: {df_aeo_prices_raw['unit'].unique()}")

# Tidy: map to codebase fuel_type / census-division / price_type, coerce numerics.
df_tidy = df_aeo_prices_raw.copy()
df_tidy["fuel_type"] = df_tidy["seriesName"].map(SERIES_TO_FUEL)
df_tidy["region"] = df_tidy["regionName"].map(REGION_MAP)
df_tidy["price_type"] = df_tidy["unit"].map(
    {"2025 $/MMBtu": "real_2025", "nom $/MMBtu": "nominal"}
)
df_tidy["year"] = df_tidy["period"].astype(int)
df_tidy["value"] = pd.to_numeric(df_tidy["value"], errors="coerce")

# SERIES_TO_FUEL only maps the residential energy-price series; rows for other
# series types in Table 3 return NaN for fuel_type and are dropped here.
df_aeo_prices = (
    df_tidy[
        df_tidy["fuel_type"].notna()
        & df_tidy["region"].notna()
        & df_tidy["price_type"].notna()
    ]
    [["region", "fuel_type", "price_type", "year", "value"]]
    .dropna()
    .sort_values(["fuel_type", "region", "price_type", "year"])
    .reset_index(drop=True)
)
print(f"  After tidy: {len(df_aeo_prices)} rows | "
      f"fuels={sorted(df_aeo_prices['fuel_type'].unique())} | "
      f"regions={df_aeo_prices['region'].nunique()} | "
      f"price_types={list(df_aeo_prices['price_type'].unique())}")

# %%
# Build projection factors using the real (2025-dollar) series rather than nominal.
# The ratio cancels units either way, but real prices make the factor easier to
# interpret: a factor > 1.0 means the fuel is genuinely more expensive in the model
# year, not just because of general inflation. Each factor = year price / 2025 price.
df_real = df_aeo_prices[df_aeo_prices["price_type"] == "real_2025"].copy()
pivot = df_real.pivot_table(
    index=["region", "fuel_type"], columns="year", values="value"
)

if ANCHOR_YEAR not in pivot.columns:
    raise ValueError(
        f"Anchor year {ANCHOR_YEAR} missing from AEO prices. "
        f"Available: {sorted(pivot.columns.tolist())}"
    )

df_projection_factors = pivot.div(pivot[ANCHOR_YEAR], axis=0).reset_index()
df_projection_factors["policy_scenario"] = SCENARIO_NAME

# Every projection factor at the anchor year must equal exactly 1.0.
anchor_vals = df_projection_factors[ANCHOR_YEAR].round(9).unique()
assert set(anchor_vals) == {1.0}, (
    f"Anchor-year factors must all be 1.0; got {anchor_vals}"
)
print(
    f"  Projection factors: shape {df_projection_factors.shape};"
    f" all {ANCHOR_YEAR} factors = 1.0 [OK]"
)

# Read this CSV back and casts year column headers from strings to integers.
# See the read-back note at the end of this notebook for the exact pattern.
os.makedirs(os.path.dirname(PATH_FUEL_PRICE_FACTORS), exist_ok=True)
df_projection_factors.to_csv(PATH_FUEL_PRICE_FACTORS, index=False)
print(f"""\
  [PASS] Fuel-price factors written
         {os.path.relpath(PATH_FUEL_PRICE_FACTORS, PROJECT_ROOT)}
         Shape: {df_projection_factors.shape} | All {ANCHOR_YEAR} factors = 1.0""")

# %% [markdown]
# ## 2025 nominal fuel prices, per state, by fuel 

# %% [markdown]
# ### Electricity: State-level residential retail price (cents/kWh) 

# %%
# The /data suffix is required to get records (the bare route returns a schema dict).
print("\nObtaining electricity data...")
elec_params = {
    "data[]": "price",
    "facets[sectorid][]": "RES",
    "frequency": "annual",
    "start": "2025",
    "end": "2025",
    "length": 100,  # headroom -- raw returns ~62 rows (states + US + aggregates)
}
rows_elec = eia_fetch(
    "electricity/retail-sales/data", api_key=EIA_API_KEY, **elec_params
)
df_elec_raw = pd.DataFrame(rows_elec)
df_elec_raw["price"] = pd.to_numeric(df_elec_raw["price"], errors="coerce")

# Keep the 51 state/DC rows. The U.S. national figure is captured separately as a
# fallback price for any state that ends up with a missing value after the PADD
# broadcast (heating oil and propane only -- electricity and natural gas have full
# state coverage).
elec_states = (
    df_elec_raw[df_elec_raw["stateid"].isin(ALL_STATES)][["stateid", "price"]]
    .rename(columns={"stateid": "state", "price": "base_value"})
    .copy()
)
elec_states["source_region"] = elec_states["state"]      # measured per state
elec_states["source_method"] = "state_annual_2025"
elec_states["fuel_type"] = "electricity"
elec_states["base_unit"] = "cents/kWh"
df_elec_final = elec_states

elec_us = df_elec_raw[df_elec_raw["stateid"] == "US"]["price"]
elec_national = float(elec_us.iloc[0]) if len(elec_us) else float("nan")
print(
    f"Electricity: {len(df_elec_final)} state rows (expect 51);"
    f" National = {elec_national} cents/kWh"
)

if PRINT_VERBOSE_DATAFRAMES:
    display(df_elec_final)

# %% [markdown]
# ### Natural Gas: State-level residential price ($/MCF), with monthly fallback

# %%
print("\nObtaining natural gas data...")
ng_kwargs = {
    "data[]": "value",
    "facets[process][]": "PRS",  # Price delivered to Residential consumers
    "facets[product][]": "EPG0",  # natural gas (excludes NGLs, LNG)
    "frequency": "annual",
    "start": "2025",
    "end": "2025",
    "length": 200,
}
df_ng_annual = pd.DataFrame(
    eia_fetch("natural-gas/pri/sum/data", api_key=EIA_API_KEY, **ng_kwargs)
)
df_ng_annual["value"] = pd.to_numeric(df_ng_annual["value"], errors="coerce")

# State rows: duoarea is 'S' + 2-letter abbrev (exactly 3 chars). 'NUS' etc. excluded.
state_mask = (
    df_ng_annual["duoarea"].str.startswith("S") &
    (df_ng_annual["duoarea"].str.len() == 3)
)

# 'SAK' --> 'AK'
df_ng_states = df_ng_annual[state_mask].copy()
df_ng_states["state_abbr"] = df_ng_states["duoarea"].str[1:]
df_ng_nonstate = df_ng_annual[~state_mask]

states_with_data = set(df_ng_states[df_ng_states["value"].notna()]["state_abbr"])
missing_annual = [s for s in ALL_STATES if s not in states_with_data]
print(f"Natural gas: {len(states_with_data)} states with annual 2025; "
      f"{len(missing_annual)} missing --> {missing_annual}")

# Monthly fallback for states without an annual figure (EIA publishes with a ~2-month lag).
if missing_annual:
    ng_monthly_params = {
        **ng_kwargs,
        "frequency": "monthly",
        "start": "2025-01",
        "end": "2025-12",
        "length": 700,  # 51 states x 12 months = 612 max
    }
    df_ng_monthly = pd.DataFrame(
        eia_fetch("natural-gas/pri/sum/data", api_key=EIA_API_KEY, **ng_monthly_params)
    )
    df_ng_monthly["value"] = pd.to_numeric(df_ng_monthly["value"], errors="coerce")
    missing_duoareas = {f"S{s}" for s in missing_annual}
    df_ng_monthly = df_ng_monthly[
        df_ng_monthly["duoarea"].isin(missing_duoareas)
    ].copy()
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
    print(f"Natural gas: {len(ng_fill)} states filled with averaged monthly data")

    if PRINT_VERBOSE_DATAFRAMES:
        display(df_ng_combined)

else:
    df_ng_states["source_method"] = "annual_2025"
    df_ng_combined = df_ng_states[["state_abbr", "value", "source_method"]].copy()
    print(f"Natural gas: {len(df_ng_combined)} states with annual 2025")
    
    if PRINT_VERBOSE_DATAFRAMES:
        display(df_ng_combined)


# %%
# National row: prefer the non-state 'U.S.' row from the annual fetch; else the series ID.
nat_row = df_ng_nonstate[
    df_ng_nonstate["area-name"].str.contains("U.S.", na=False)
]
if len(nat_row):
    ng_national = float(nat_row.iloc[0]["value"])
    ng_nat_method = "annual_PRS_fetch"
else:
    df_nat = pd.DataFrame(fetch_seriesid(
        "NG.N3010US3.A", api_key=EIA_API_KEY, anchor_year=ANCHOR_YEAR
    ))
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

# keep="first" gives priority to annual rows over monthly fallback rows, because
# annual data was concatenated before monthly data in df_ng_combined.
df_ng_final = (
    df_ng_combined
    .dropna(subset=["value"])
    .drop_duplicates(subset=["state_abbr"], keep="first")
    .loc[lambda d: d["state_abbr"].isin(ALL_STATES)]
    .rename(columns={"state_abbr": "state", "value": "base_value"})
    .copy()
)
df_ng_final["fuel_type"] = "naturalGas"
df_ng_final["base_unit"] = "$/MCF"
df_ng_final["source_region"] = df_ng_final["state"]
print(
    f"Natural gas: {len(df_ng_final)} state rows (expect 51);"
    f" National = {ng_national:.2f} $/MCF"
)

if PRINT_VERBOSE_DATAFRAMES:
    display(df_ng_final)

# %% [markdown]
# ### Heating oil + propane -- PADD broadcast to states ($/gal)
# 
# EIA reports these residential prices only at PADD / sub-PADD level. Fetch the full
# 2025 monthly series, average available months per region, then broadcast to member
# states.
# 
# Coverage is incomplete BY DESIGN (heating oil is a NE/Midwest fuel): a state whose
# PADD did not report falls back to the National price, flagged in source_method.

# %%
# EPD2F = EIA product code for No. 2 Fuel Oil (residential heating oil).
# EPLLPA = EIA product code for propane (liquefied petroleum gas).
print("Obtaining fuel oil data...")
df_oil_final, oil_national = build_petroleum_fuel_df(
    "EPD2F", "fuelOil", api_key=EIA_API_KEY
)
# Print the final DataFrame if verbose output is enabled
if PRINT_VERBOSE_DATAFRAMES:
    display(df_oil_final)

print("\nObtaining propane data...")
df_propane_final, propane_national = build_petroleum_fuel_df(
    "EPLLPA", "propane", api_key=EIA_API_KEY
)
# Print the final DataFrame if verbose output is enabled
if PRINT_VERBOSE_DATAFRAMES:
    display(df_propane_final)

# %% [markdown]
# ### Combine all four fuels --> Normalize units --> export CSV

# %%
print("\nCombining all fuel data...")

common_cols = [
    "state", "fuel_type", "base_value", "base_unit", "source_region", "source_method"
]
df_all = pd.concat(
    [df_elec_final[common_cols], df_ng_final[common_cols],
     df_oil_final[common_cols],  df_propane_final[common_cols]],
    ignore_index=True,
)

df_all["census_division"] = df_all["state"].map(STATE_TO_DIVISION)
df_all["source_region_label"] = df_all["source_region"].map(
    lambda c: REGION_LABELS.get(c, c)
)
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
    "fuel_type", "source_region", "source_region_label", "source_method",
    "base_value", "base_unit", "state", "census_division",
    "price_usd2025_per_kwh",
]].sort_values(["fuel_type", "state"]).reset_index(drop=True)

# Validation: 51 states x 4 fuels + 4 National = 208 rows; no missing prices.
expected_state_rows = len(ALL_STATES) * 4
n_state_rows = int((df_fuel_prices_2025["state"] != "National").sum())
assert n_state_rows == expected_state_rows, (
    f"Expected {expected_state_rows} statexfuel rows, got {n_state_rows}."
)
missing_price = df_fuel_prices_2025[df_fuel_prices_2025["price_usd2025_per_kwh"].isna()]
if len(missing_price):
    print(f"[WARNING] {len(missing_price)} row(s) have no price:")
    print(missing_price[["state", "fuel_type", "source_method"]].to_string(index=False))

# Make the broadcast auditable: which states share an oil/propane price?
print("\nShared petroleum prices by source region (same PADD --> identical price):")
for fuel in ["fuelOil", "propane"]:
    grp = (df_fuel_prices_2025[df_fuel_prices_2025["fuel_type"] == fuel]
           .groupby("source_region_label")["state"].apply(list))
    print(f"  {fuel}:")
    for region, states in grp.items():
        print(f"    {region}: {states}")

# National anchors -- spot-check against EIA's published 2025 residential figures.
print("\nNational anchors (USD2025/kWh):")
print(
    df_fuel_prices_2025[df_fuel_prices_2025["state"] == "National"]
    [["fuel_type", "price_usd2025_per_kwh", "base_value", "base_unit"]]
    .to_string(index=False)
)

os.makedirs(os.path.dirname(PATH_FUEL_PRICE_ANCHOR), exist_ok=True)
df_fuel_prices_2025.to_csv(PATH_FUEL_PRICE_ANCHOR, index=False)
print(f"\n[PASS] Wrote {len(df_fuel_prices_2025)} anchor rows --> "
      f"{os.path.relpath(PATH_FUEL_PRICE_ANCHOR, PROJECT_ROOT)}")

display(df_fuel_prices_2025)

# %% [markdown]
# ## AEO2026 API --> CDD/HDD Projection Factors --> Export CSV

# %%
print("\nFetching degree day data...")

# Fetch degree day data for heating and cooling degree days.
hdd_rows = fetch_degree_day_rows(
    HDD_SERIES_IDS, "hdd", api_key=EIA_API_KEY, anchor_year=ANCHOR_YEAR
)
cdd_rows = fetch_degree_day_rows(
    CDD_SERIES_IDS, "cdd", api_key=EIA_API_KEY, anchor_year=ANCHOR_YEAR
)
df_dd_raw = pd.DataFrame(hdd_rows + cdd_rows)
print(
    f"Degree days: {len(df_dd_raw)} rows"
    f" (expect 520 = 10 divisions x 2 types x 26 yr)"
)

df_hdd_factors = degree_day_factor_table(df_dd_raw, "hdd", anchor_year=ANCHOR_YEAR)
df_cdd_factors = degree_day_factor_table(df_dd_raw, "cdd", anchor_year=ANCHOR_YEAR)

# In-memory lookups (int year keys, built from the live DataFrames) for verification and
# optional direct wiring into degree_day_consumption_utils.py.
lookup_hdd_factor = df_hdd_factors.set_index("census_division").to_dict("index")
lookup_cdd_factor = df_cdd_factors.set_index("census_division").to_dict("index")

# Every census division must have a factor of 1.0 for the anchor year.
for label, lookup in [("HDD", lookup_hdd_factor), ("CDD", lookup_cdd_factor)]:
    bad = {
        d: v[ANCHOR_YEAR]
        for d, v in lookup.items()
        if round(v[ANCHOR_YEAR], 6) != 1.0
    }
    assert not bad, f"{label} anchor-year factor != 1.0 for {bad}"

# %%
# Sanity check across all divisions: a warming climate means fewer heating days (HDD
# factor < 1.0) and more cooling days (CDD factor > 1.0) by 2040. Report any division
# that fails this check rather than raising an error.
CHECK_YEAR = 2040
hdd_viol = [d for d, v in lookup_hdd_factor.items() if v.get(CHECK_YEAR, 1.0) >= 1.0]
cdd_viol = [d for d, v in lookup_cdd_factor.items() if v.get(CHECK_YEAR, 1.0) <= 1.0]
print(f"Anchor-year factors = 1.0 for all divisions [OK]")
if hdd_viol:
    print(f"[CHECK] HDD {CHECK_YEAR} not < 1.0 for {hdd_viol} -- verify HDD series IDs")
if cdd_viol:
    print(f"[CHECK] CDD {CHECK_YEAR} not > 1.0 for {cdd_viol} -- verify CDD series IDs")
if not hdd_viol and not cdd_viol:
    print(
        f"Directional check passed: all HDD {CHECK_YEAR} < 1.0"
        f" and all CDD {CHECK_YEAR} > 1.0 [OK]"
    )

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
print(f"""\
  [PASS] Degree-day factors written
         {os.path.relpath(PATH_DEGREE_DAY_FACTORS, PROJECT_ROOT)}""")

# -- READ-BACK NOTE for degree_day_consumption_utils.py ---------------------------------
# CSV headers are strings, but the lookups do `.get(year_label, 1.0)` with INT years.
# Cast year columns back to int on read, or every lookup silently misses and returns 1.0:
#
#   df = pd.read_csv(PATH_DEGREE_DAY_FACTORS)
#   df.columns = [int(c) if str(c).isdigit() else c for c in df.columns]
#   lookup_hdd_factor = (df[df.dd_type == "hdd"].drop(columns="dd_type")
#                          .set_index("census_division").to_dict("index"))
#   lookup_cdd_factor = (df[df.dd_type == "cdd"].drop(columns="dd_type")
#                          .set_index("census_division").to_dict("index"))


