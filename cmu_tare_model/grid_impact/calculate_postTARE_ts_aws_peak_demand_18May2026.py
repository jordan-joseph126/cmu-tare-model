# %% [markdown]
# ---
# # Post-TARE Model Run: Timeseries Peak Load Analysis (AWS / BuildStockQuery)
# ---
# 
# **Author:** Jordan M. Joseph, PhD — Carnegie Mellon University
# 
# Queries the AWS-hosted ResStock EUSS 2022.1.1 timeseries database via BuildStockQuery
# to compute **county-level peak load changes** under two adoption scenarios:
# - **(a) 100% adoption counterfactual** — all filtered building IDs adopt (upper bound)
# - **(b) Economically-constrained adoption** — only Tier 1 + Tier 2 adopters (TARE output)
# 
# **Prerequisite:** Steps 0–0c must be run first (TARE data loaded, MP selected).
# 
# **Dataset scope:** ResStock 2022.1.1 (EUSS, AMY2018). ResStock 2025.1 is future work.
# 
# **Primary test case:** Allegheny County, PA (FIPS 42003) — validate here before scaling.
# 
# See methodology notes in `peak_load_methodology.md`.

# %% [markdown]
# ---
# ## Step 0: Imports and Configuration
# ---

# %%
import logging
import os
import time
from typing import Any, Final

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from config import PROJECT_ROOT
from cmu_tare_model.constants import (
    ALLOWED_HOUSING_TYPES,
    VALID_MENU_MPS,
    VERBOSE,
    REMDB_COST_SCENARIO_KEYS,
    PRIVATE_DISCOUNT_RATE_SHORT_KEYS,
    BLDG_ID_COL,
    TIMESTAMP_COL,
    ELEC_TOTAL_COL,
    BSQ_ELEC_COL,
    METADATA_TABLE,
    COUNTY_COL,
    STATE_COL,
    WEIGHT_COL,
    TEST_FIPS,
    TEST_GISJOIN,
)

from cmu_tare_model.utils.load_exported_results_to_df import load_measure_package_data

from cmu_tare_model.adoption_kpis import (
    load_euss_baseline, load_euss_upgrade, mp_to_upgrade,
    compute_scenario_demand, aggregate_demand_by_state,
)
from cmu_tare_model.adoption_kpis.data_loading import (
    FUEL_PRICES_PATH,
    SHAPEFILE_PATH,
    HEATING_FUEL_COLS,
    HP_BACKUP_ELEC_COL,
    HP_FANS_PUMPS_COL,
)
from cmu_tare_model.grid_impact.peak_load_functions import (
    gisjoin_to_fips,
    find_adoption_column,
    extract_adopter_ids,
    compute_county_scenario_profile,
)
from cmu_tare_model.adoption_kpis.visualize_geospatial_data import (
    prepare_state_geodataframe,
    create_choropleth_map,
)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 60)

# Suppress verbose BSQ/botocore logging — only show ERROR and above
logging.getLogger("buildstock_query").setLevel(logging.ERROR)

print("✓ Imports loaded")

# %% [markdown]
# ---
# ## Step 0b: Measure Package Selection
# ---

# %%
# =============================================================================
# STEP 0b: Measure Package Selection (hardcoded for reproducibility)
# =============================================================================
# Previously used input() — replaced with constants for non-interactive runs.
# Change selected_mps to run different measure packages.

SELECTABLE_MPS = [mp for mp in VALID_MENU_MPS if mp != 0]

# selected_mps: list[int] = [3]  # MP3 = ducted ASHP (primary analysis)
selected_mps: list[int] = [3, 4]
batch_mode: bool = False

print(f"Selected measure packages: {selected_mps}")
print(f"Available: {SELECTABLE_MPS}")

# %% [markdown]
# ---
# ## Step 0c: Load TARE Model Data (Measure Packages 3, 4)
# ---
# 
# Load pre-computed TARE model outputs for the selected measure packages.
# Required for Step 4d (Private NPV extraction).
# 
# If the top-section data loading cells have already been run, this will reuse `DATAFRAMES_BY_MP`.

# %%
# =============================================================================
# STEP 0c: LOAD TARE MODEL DATA (for NPV extraction)
# =============================================================================
# Check if DATAFRAMES_BY_MP was already loaded by the top-section cells.
# If not, load using hardcoded constants for reproducibility.

LOCATION_ID: str = "National"           # Hardcoded for reproducibility
MODEL_RUN_DATE_TIME: str = "2026-04-10_00-05"  # Hardcoded for reproducibility

try:
    _ = DATAFRAMES_BY_MP
    print(f"DATAFRAMES_BY_MP already loaded: {list(DATAFRAMES_BY_MP.keys())}")
except NameError:
    print("DATAFRAMES_BY_MP not found — loading TARE model outputs...")

    try:
        _ = output_folder_path
        print(f"  Using existing output_folder_path: {output_folder_path}")
    except NameError:
        output_folder_path = os.path.join(PROJECT_ROOT, "cmu_tare_model", "output_results")

    print(f"  output_folder_path   : {output_folder_path}")
    print(f"  location_id          : {LOCATION_ID}")
    print(f"  model_run_date_time  : {MODEL_RUN_DATE_TIME}")

    DATAFRAMES_BY_MP = {}
    for mp in selected_mps:
        DATAFRAMES_BY_MP[mp] = load_measure_package_data(
            mp, output_folder_path, LOCATION_ID, MODEL_RUN_DATE_TIME
        )

    print(f"\n✓ Loaded TARE data for MPs: {list(DATAFRAMES_BY_MP.keys())}")

# %% [markdown]
# ---
# ## Step 1: BuildStockQuery Initialization
# ---
# 
# Initializes BuildStockQuery (NREL SWR-23-58) to connect to the OEDI data lake
# via AWS Athena. BSQ handles all Athena infrastructure (workgroup, Glue catalog,
# partition registration, query execution) internally.
# 
# ### Prerequisites
# 1. `pip install git+https://github.com/NREL/buildstock-query.git`
# 2. `aws configure` with valid credentials (region: `us-west-2`)
# 3. IAM policies: `AmazonAthenaFullAccess`, `AmazonS3ReadOnlyAccess`
# 
# ### BSQ Parameters (EUSS 2022.1.1)
# | Parameter | Value |
# |-----------|-------|
# | `workgroup` | `resstock-euss` |
# | `db_name` | `euss-oedi` |
# | `table_name` | `resstock_amy2018_release_1_1` |
# | `db_schema` | `resstock_oedi` |
# | `buildstock_type` | `resstock` |
# 
# ### Weight note
# All 548,916 EUSS 2022.1.1 buildings have **uniform weight = 242.131013**.
# BSQ applies this automatically via `SUM(enduse × weight)` in generated SQL.
# Previous hardcode of 240.0 has been removed (0.9% error).
# 
# ### Known issue
# `split_enduses=True` triggers a Pydantic `ValidationError` in BSQ's batch query
# path. Use `split_enduses=False` with single enduse (works correctly).

# %%
# =============================================================================
# STEP 1: BuildStockQuery Initialization
# =============================================================================

# ========== 1. Import BuildStockQuery ==========
from buildstock_query import BuildStockQuery  # type: ignore[import-untyped]
from buildstock_query.schema.query_params import TSQuery
print(f"✓ BuildStockQuery imported")

# ========== 2. Verify AWS credentials ==========
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

session = boto3.session.Session()
aws_region: str | None = session.region_name

try:
    sts = session.client("sts")
    aws_identity: dict = sts.get_caller_identity()
    print(f"""
          ✓ AWS credentials valid
            Account : {aws_identity['Account']}
            ARN     : {aws_identity['Arn']}
            Region  : {aws_region}
          """)

except NoCredentialsError:
    raise RuntimeError(
        "AWS credentials not found. Run `aws configure` or set "
        "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY environment variables."
    )
except ClientError as e:
    raise RuntimeError(f"AWS STS call failed: {e}")

# ========== 3. Initialize BuildStockQuery ==========
my_run = BuildStockQuery(
    workgroup='resstock-euss',
    db_name='euss-oedi',
    table_name='resstock_amy2018_release_1_1',
    db_schema='resstock_oedi',
    buildstock_type='resstock',
    skip_reports=True,
)

print(f"""
✓ BuildStockQuery initialized: {type(my_run).__name__}

Metadata table : {my_run.bs_table}
Timeseries table: {my_run.ts_table}
Building ID col : {my_run.building_id_column_name}

✓ Step 1 COMPLETE
""")

# %%
# =============================================================================
# Step 2: Column Name Constants (imported from cmu_tare_model.constants)
# =============================================================================
# BSQ column names and Allegheny County reference values are imported from
# cmu_tare_model.constants (see Step 0 imports). Weights are handled by BSQ
# internally via SUM(enduse × baseline.weight) in generated SQL.

print(f"""
✓ Step 2: Column constants (imported from constants.py)

BLDG_ID_COL    : {BLDG_ID_COL}
ELEC_TOTAL_COL : {ELEC_TOTAL_COL}
BSQ_ELEC_COL   : {BSQ_ELEC_COL}
TEST_FIPS      : {TEST_FIPS}
""")

# %% [markdown]
# ---
# ## Step 3: County Geography Mapping — FIPS / GISJOIN to Shapefile
# ---
# 
# ResStock EUSS uses either FIPS codes or GISJOIN identifiers for county-level geography.
# We need to confirm which identifier is present and map it to standard Census TIGER/Line
# shapefiles for choropleth visualization.
# 
# **GISJOIN format:** `G` + state FIPS (2 digits, zero-padded) + `0` + county FIPS
# (3 digits, zero-padded). Example: Allegheny County PA = `G4200030`.
# 
# **Standard FIPS format:** 5-digit string. Example: Allegheny County PA = `42003`.
# 
# **Goal of this step:** Build a lookup table `county_geo_df` that maps ResStock's
# county identifier → FIPS → county name → state, so all downstream results can be
# joined to shapefiles.
# 
# **Test case:** Confirm that Allegheny County, PA appears with FIPS `42003`.

# %%
# County shapefile path (separate from the state-level SHAPEFILE_PATH)
COUNTY_SHAPEFILE_PATH: str = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "shapefiles",
    "tl_2025_us_county", "tl_2025_us_county.shp"
)

# TIGER/Line county shapefiles: GEOID = 5-digit FIPS, NAME = county name,
# STATEFP = 2-digit state FIPS (no state abbreviation in county shapefiles).
TIGER_FIPS_COL: Final[str] = "GEOID"
TIGER_NAME_COL: Final[str] = "NAME"
TIGER_STATEFP_COL: Final[str] = "STATEFP"

# Load the county shapefile
gdf_counties_raw: gpd.GeoDataFrame = gpd.read_file(COUNTY_SHAPEFILE_PATH)

# Validate expected columns exist
_expected_cols = {TIGER_FIPS_COL, TIGER_NAME_COL, TIGER_STATEFP_COL}
_missing = _expected_cols - set(gdf_counties_raw.columns)
if _missing:
    raise KeyError(
        f"TIGER county shapefile missing expected columns: {_missing}\n"
        f"  Available: {sorted(gdf_counties_raw.columns.tolist())}\n"
        f"  Path: {COUNTY_SHAPEFILE_PATH}"
    )

# Build the plain lookup DataFrame
county_geo_df: pd.DataFrame = gdf_counties_raw[
    [TIGER_FIPS_COL, TIGER_NAME_COL, TIGER_STATEFP_COL]
].rename(
    columns={
        TIGER_FIPS_COL: "fips_5digit",
        TIGER_NAME_COL: "county_name",
        TIGER_STATEFP_COL: "state_fips",
    }
).copy()
county_geo_df["fips_5digit"] = county_geo_df["fips_5digit"].astype(str).str.zfill(5)

# GeoDataFrame with renamed columns for downstream merges
gdf_counties: gpd.GeoDataFrame = gdf_counties_raw.rename(
    columns={
        TIGER_FIPS_COL: "fips_5digit",
        TIGER_NAME_COL: "county_name",
        TIGER_STATEFP_COL: "state_fips",
    }
)[["fips_5digit", "county_name", "state_fips", "geometry"]]
gdf_counties["fips_5digit"] = gdf_counties["fips_5digit"].astype(str).str.zfill(5)

# Validation: Allegheny County, PA (FIPS 42003) must be present
test_row = county_geo_df[county_geo_df["fips_5digit"] == TEST_FIPS]
if test_row.empty:
    raise ValueError(
        f"Test county FIPS {TEST_FIPS} (Allegheny, PA) not found in shapefile.\n"
        f"  This blocks the Step 7 validation case. Check COUNTY_SHAPEFILE_PATH."
    )
print(f"✓ county_geo_df: {len(county_geo_df):,d} counties")
print(f"  Test county: {test_row.iloc[0].to_dict()}")
print(f"✓ gdf_counties CRS: {gdf_counties.crs}")

# %% [markdown]
# ---
# ## Step 4: Extract Adopter Building IDs from TARE Results
# ---
# 
# The TARE model assigns each building (`bldg_id`) to an adoption tier:
# - **Tier 1** — Private NPV positive (saves money without incentives)
# - **Tier 2** — Private NPV positive only with IRA rebates
# - **Tier 3** — Requires public benefits (social cost of carbon) to be justified
# - **Tier 4** — No adoption pathway under current conditions
# 
# For the **economically-constrained scenario**, adopters = Tier 1 + Tier 2.
# For the **100% adoption counterfactual**, adopters = all filtered building IDs.
# 
# This step extracts both sets of building IDs, organized by county (FIPS), for use
# in the timeseries queries in Steps 5–6.
# 
# **Output structure:**
# ```python
# adopter_ids_by_county = {
#     "42003": {                     # FIPS for Allegheny County, PA
#         "tier1": [101, 204, ...],
#         "tier2": [305, 412, ...],
#         "constrained": [101, 204, 305, 412, ...],  # tier1 + tier2
#         "all_filtered": [101, 204, ..., 99999],    # 100% adoption
#     }
# }
# ```

# %%
# =============================================================================
# STEP 4: Extract Adopter Building IDs from TARE Results — all selected MPs
# =============================================================================

# ========== Configuration ==========
DISCOUNT_RATE_KEY: str = "fixed_base"

print(f"Discount rate key: {DISCOUNT_RATE_KEY}")

# ========== Per-MP loop: build adopter_ids_by_mp and adoption_col_by_mp ==========
adopter_ids_by_mp: dict[int, dict[str, dict[str, list[int]]]] = {}
adoption_col_by_mp: dict[int, str] = {}

for mp in selected_mps:
    # DATAFRAMES_BY_MP[mp] is keyed by discount rate; each value is a DataFrame.
    df_tare: pd.DataFrame = DATAFRAMES_BY_MP[mp][DISCOUNT_RATE_KEY]
    print(f"\nMP{mp}: DATAFRAMES_BY_MP[{mp}]['{DISCOUNT_RATE_KEY}']  shape={df_tare.shape}")

    # Find the adoption column — try each cost scenario in priority order
    adoption_col: str | None = None
    for cs in REMDB_COST_SCENARIO_KEYS:
        try:
            adoption_col = find_adoption_column(df_tare, mp, cs)
            print(f"  ✓ Adoption column (cost_scenario='{cs}'): {adoption_col}")
            break
        except KeyError:
            continue

    if adoption_col is None:
        # Final attempt — will raise with full diagnostics
        adoption_col = find_adoption_column(df_tare, mp, REMDB_COST_SCENARIO_KEYS[0])

    adoption_col_by_mp[mp] = adoption_col

    # Print tier distribution for this MP
    tier_counts = df_tare[adoption_col].value_counts()
    print(f"\n  ========== Tier distribution (MP{mp}) ==========")
    for tier_val, count in tier_counts.items():
        print(f"    {tier_val:<45s}  {count:>7,d}")
    print(f"    {'TOTAL':<45s}  {tier_counts.sum():>7,d}")

    # Extract adopter IDs by county
    adopter_ids_by_mp[mp] = extract_adopter_ids(df_tare, adoption_col)

    n_counties = len(adopter_ids_by_mp[mp])
    total_constrained = sum(len(v["constrained"]) for v in adopter_ids_by_mp[mp].values())
    total_all = sum(len(v["all_filtered"]) for v in adopter_ids_by_mp[mp].values())
    print(f"\n  ========== Adopter summary (MP{mp}) ==========")
    print(f"    Counties           : {n_counties:,d}")
    print(f"    Constrained (T1+T2): {total_constrained:,d}")
    print(f"    All filtered       : {total_all:,d}")

    # Test case: Allegheny County, PA (FIPS 42003)
    if TEST_FIPS in adopter_ids_by_mp[mp]:
        ac = adopter_ids_by_mp[mp][TEST_FIPS]
        print(f"\n  ========== Allegheny County (MP{mp}, FIPS {TEST_FIPS}) ==========")
        print(f"    Tier 1       : {len(ac['tier1']):,d}")
        print(f"    Tier 2       : {len(ac['tier2']):,d}")
        print(f"    Constrained  : {len(ac['constrained']):,d}")
        print(f"    All filtered : {len(ac['all_filtered']):,d}")
    else:
        print(f"\n  ⚠ MP{mp}: FIPS {TEST_FIPS} (Allegheny County, PA) not found in results.")
        sample_fips = list(adopter_ids_by_mp[mp].keys())[:10]
        print(f"    Available FIPS (first 10): {sample_fips}")

print(f"\n✓ Step 4 COMPLETE — adopter_ids_by_mp.keys() = {list(adopter_ids_by_mp.keys())}")

# %% [markdown]
# ---
# ## Step 5: Baseline Timeseries — Allegheny County (BSQ)
# ---
# 
# Queries baseline (upgrade=0) hourly electricity consumption for all buildings
# in Allegheny County via `BuildStockQuery.agg.aggregate_timeseries()`.
# 
# BSQ generates SQL that:
# - Joins timeseries ← metadata on `bldg_id`
# - Aggregates 15-min intervals to hourly via `date_trunc('hour', ...)`
# - Applies `SUM(enduse × weight)` — values are **weight-applied**
# - Returns per-building data when `group_by=['bldg_id']`
# 
# **Note:** `split_enduses=False` required (True triggers Pydantic bug in BSQ batch path).

# %%
# =============================================================================
# STEP 5: Baseline Timeseries — Allegheny County (BSQ)
# =============================================================================
# Baseline is MP-independent — query once using the union of all MPs' bldg_ids.
allegheny_bldg_ids: list[int] = sorted(set().union(*[
    adopter_ids_by_mp[mp][TEST_FIPS]["all_filtered"] for mp in selected_mps
]))
print(f"✓ Allegheny bldg_ids (union across MPs {selected_mps}): {len(allegheny_bldg_ids):,d}")

# ========== Query baseline timeseries via BSQ ==========
print(f"\n Querying baseline timeseries (upgrade=0, {len(allegheny_bldg_ids)} buildings)...")
t_start = time.perf_counter()

ts_query_baseline = TSQuery(
    enduses=[ELEC_TOTAL_COL],
    restrict=[('bldg_id', allegheny_bldg_ids)],
    upgrade_id='0',
    timestamp_grouping_func='hour',
    group_by=[BLDG_ID_COL],
    split_enduses=False,
)

df_ts_baseline_allegheny: pd.DataFrame = my_run.agg.aggregate_timeseries(
    params=ts_query_baseline
)
query_time_s: float = time.perf_counter() - t_start

# ========== Rename and downcast ==========
df_ts_baseline_allegheny = df_ts_baseline_allegheny.rename(
    columns={BSQ_ELEC_COL: 'baseline_kwh'}
)
# float32 is sufficient for kWh values and halves memory at national scale
df_ts_baseline_allegheny['baseline_kwh'] = df_ts_baseline_allegheny['baseline_kwh'].astype(np.float32)

# Deterministic hour index: sort by (bldg_id, timestamp) before assigning ordinals
df_ts_baseline_allegheny = df_ts_baseline_allegheny.sort_values(
    [BLDG_ID_COL, TIMESTAMP_COL]
).reset_index(drop=True)
df_ts_baseline_allegheny['hour'] = (
    df_ts_baseline_allegheny.groupby(BLDG_ID_COL).cumcount() + 1
)

# ========== Summary ==========
n_bldgs: int = df_ts_baseline_allegheny[BLDG_ID_COL].nunique()
n_hours_per_bldg = df_ts_baseline_allegheny.groupby(BLDG_ID_COL).size()
weight_val: float = df_ts_baseline_allegheny['units_count'].iloc[0]

print(f"\n========== df_ts_baseline_allegheny summary ==========")
print(f"  Rows            : {len(df_ts_baseline_allegheny):,d}")
print(f"  Buildings       : {n_bldgs:,d}")
print(f"  Hours/bldg      : {n_hours_per_bldg.min():,d} - {n_hours_per_bldg.max():,d}")
print(f"  kWh range (wtd) : {df_ts_baseline_allegheny['baseline_kwh'].min():.3f}"
      f" to {df_ts_baseline_allegheny['baseline_kwh'].max():.3f}")
print(f"  BSQ weight      : {weight_val:.6f}")
print(f"  Query time (s)  : {query_time_s:.2f}")

assert n_hours_per_bldg.min() == 8760, f"Min hours/bldg = {n_hours_per_bldg.min()}, expected 8760"
assert n_hours_per_bldg.max() == 8760, f"Max hours/bldg = {n_hours_per_bldg.max()}, expected 8760"
print(f"\n✓ Step 5 PASSED — {n_bldgs} buildings × 8,760 hours")
display(df_ts_baseline_allegheny.head())

# %% [markdown]
# ---
# ## Step 6: Upgrade Timeseries — All Selected MPs — Allegheny County (BSQ)
# ---
# 
# Queries post-retrofit timeseries for the same county and building IDs, for each selected MP.
# Same BSQ approach as Step 5 with `upgrade_id=str(mp)`.
# 
# Only buildings where `applicability == True` have valid upgrade data.
# Buildings in `all_filtered` with no upgrade data will fall back to baseline in Step 7.

# %%
# =============================================================================
# STEP 6: Upgrade Timeseries — Allegheny County (BSQ), all selected MPs
# =============================================================================
df_ts_upgrade_allegheny_by_mp: dict[int, pd.DataFrame] = {}

for mp in selected_mps:
    print(f"\n Querying upgrade timeseries (upgrade={mp})...")
    t_start = time.perf_counter()

    ts_query_upgrade = TSQuery(
        enduses=[ELEC_TOTAL_COL],
        restrict=[('bldg_id', allegheny_bldg_ids)],
        upgrade_id=str(mp),
        timestamp_grouping_func='hour',
        group_by=[BLDG_ID_COL],
        split_enduses=False,
    )

    df_ts_upgrade: pd.DataFrame = my_run.agg.aggregate_timeseries(
        params=ts_query_upgrade
    )
    upgrade_query_time_s: float = time.perf_counter() - t_start

    # Rename and downcast
    df_ts_upgrade = df_ts_upgrade.rename(columns={BSQ_ELEC_COL: 'retrofit_kwh'})
    df_ts_upgrade['retrofit_kwh'] = df_ts_upgrade['retrofit_kwh'].astype(np.float32)

    # Deterministic hour index
    df_ts_upgrade = df_ts_upgrade.sort_values(
        [BLDG_ID_COL, TIMESTAMP_COL]
    ).reset_index(drop=True)
    df_ts_upgrade['hour'] = (
        df_ts_upgrade.groupby(BLDG_ID_COL).cumcount() + 1
    )

    # Schema parity check
    baseline_bldgs: set[int] = set(df_ts_baseline_allegheny[BLDG_ID_COL].unique())
    upgrade_bldgs: set[int] = set(df_ts_upgrade[BLDG_ID_COL].unique())
    only_in_baseline: set[int] = baseline_bldgs - upgrade_bldgs
    only_in_upgrade: set[int] = upgrade_bldgs - baseline_bldgs

    n_hours_up = df_ts_upgrade.groupby(BLDG_ID_COL).size()

    print(f"\n========== df_ts_upgrade_allegheny (MP{mp}) summary ==========")
    print(f"  Rows            : {len(df_ts_upgrade):,d}")
    print(f"  Buildings       : {len(upgrade_bldgs):,d}")
    print(f"  Hours/bldg      : {n_hours_up.min()} - {n_hours_up.max()}")
    print(f"  Baseline bldgs  : {len(baseline_bldgs):,d}")
    print(f"  Only in baseline: {len(only_in_baseline):,d}")
    print(f"  Only in upgrade : {len(only_in_upgrade):,d}")
    print(f"  kWh range (wtd) : {df_ts_upgrade['retrofit_kwh'].min():.3f}"
          f" to {df_ts_upgrade['retrofit_kwh'].max():.3f}")
    print(f"  Query time (s)  : {upgrade_query_time_s:.2f}")

    if only_in_baseline:
        print(f"\n  Note: {len(only_in_baseline):,d} buildings have no MP{mp} upgrade data — "
              f"will use baseline in Step 7.")
    if only_in_upgrade:
        raise ValueError(
            f"MP{mp}: {len(only_in_upgrade)} bldg_ids in upgrade but not baseline. Investigate."
        )

    df_ts_upgrade_allegheny_by_mp[mp] = df_ts_upgrade
    display(df_ts_upgrade.head())

print(f"\n✓ Step 6 PASSED — df_ts_upgrade_allegheny_by_mp.keys() = "
      f"{list(df_ts_upgrade_allegheny_by_mp.keys())}")

# %% [markdown]
# ---
# ## Step 7: Compute Scenario Demand Profile — Allegheny County (Test Case)
# ---
# 
# Applies the adopter/non-adopter mask to produce a scenario hourly demand profile.
# 
# **Logic (per building, per hour):**
# - If `bldg_id` is in `adopter_ids` → use `retrofit_kwh` (post-HP electricity)
# - Otherwise → use `baseline_kwh` (existing equipment electricity)
# 
# **Two profiles produced:**
# 1. **100% adoption:** all filtered buildings adopt → all use `retrofit_kwh`
# 2. **Constrained adoption:** Tier 1 + Tier 2 adopt; Tier 3 + Tier 4 use baseline
# 
# **Peak load change** = `max(scenario_profile_mw)` − `max(baseline_profile_mw)`
# 
# **Units note:** BSQ returns weight-applied kWh (= raw kWh × 242.131013).
# Dividing by 1000 converts to MW. No separate sampling weight multiplication needed.

# %%
# --- Compute profiles for Allegheny County, per MP ---
peak_results_allegheny_by_mp: dict[int, dict[str, dict]] = {}
df_profiles_by_mp: dict[int, dict[str, pd.DataFrame]] = {}

for mp in selected_mps:
    df_ts_upgrade_allegheny = df_ts_upgrade_allegheny_by_mp[mp]
    adopter_ids_allegheny = adopter_ids_by_mp[mp][TEST_FIPS]

    print(f"\n Computing profiles for MP{mp}...")
    df_profile_100pct, peak_100pct = compute_county_scenario_profile(
        df_ts_baseline_allegheny,
        df_ts_upgrade_allegheny,
        adopter_bldg_ids=adopter_ids_allegheny["all_filtered"],
    )

    df_profile_constrained, peak_constrained = compute_county_scenario_profile(
        df_ts_baseline_allegheny,
        df_ts_upgrade_allegheny,
        adopter_bldg_ids=adopter_ids_allegheny["constrained"],
    )

    peak_results_allegheny_by_mp[mp] = {
        "100pct": peak_100pct,
        "constrained": peak_constrained,
    }
    df_profiles_by_mp[mp] = {
        "100pct": df_profile_100pct,
        "constrained": df_profile_constrained,
    }

    print(f"\n Allegheny peak results (MP{mp})")
    for scenario, p in peak_results_allegheny_by_mp[mp].items():
        print(f"\n  [{scenario}]")
        print(f"    adopters      : {p['n_adopters']:,d} / {p['n_total_buildings']:,d}")
        print(f"    baseline peak : {p['baseline_peak_mw']:.2f} MW @ hour {p['peak_hour_baseline']}")
        print(f"    scenario peak : {p['scenario_peak_mw']:.2f} MW @ hour {p['peak_hour_scenario']}")
        print(f"    delta         : {p['delta_mw']:+.2f} MW")

    assert len(df_profile_100pct) == 8760, f"MP{mp}: 100pct profile not 8760 rows!"
    assert len(df_profile_constrained) == 8760, f"MP{mp}: constrained profile not 8760 rows!"

print(f"\n✓ Step 7 PASSED — peak_results_allegheny_by_mp.keys() = "
      f"{list(peak_results_allegheny_by_mp.keys())}")

# %%
# =============================================================================
# STEP 7b: Helper — plot_demand_panel
# =============================================================================

def plot_demand_panel(
    ax: "plt.Axes",
    df_profile: pd.DataFrame,
    peak_result: dict[str, Any],
    mp: int,
    scenario_label: str,
    county_name: str = "Allegheny County, PA",
) -> None:
    """Plot baseline and scenario demand timeseries on a single axes panel.

    Draws two line series (baseline in red, scenario in blue), vertical dashed
    lines at the peak hours for each series, and a text annotation for each
    peak value. All peak hours are read dynamically from ``peak_result`` — no
    values are hardcoded.

    Args:
        ax: Matplotlib Axes to draw on.
        df_profile: DataFrame output of ``compute_county_scenario_profile``.
            Must contain columns ``hour``, ``baseline_mw``, and ``scenario_mw``.
        peak_result: Dict output of ``compute_county_scenario_profile``.
            Must contain keys ``baseline_peak_mw``, ``scenario_peak_mw``,
            ``peak_hour_baseline``, and ``peak_hour_scenario``.
        mp: Integer measure package number (e.g. 3 or 4). Used in line label.
        scenario_label: Human-readable scenario label for the legend/title.
        county_name: County name string shown in axis title.

    Returns:
        None. Modifies ``ax`` in place.

    Notes:
        Units: ``baseline_mw`` and ``scenario_mw`` are in MW (kWh ÷ 1000,
        weight already applied by BSQ). No additional scaling is applied here.
        Peak hours are 1-indexed (hour 1 = first hour of the year).
    """
    hours = df_profile["hour"]

    # ── Demand lines ──────────────────────────────────────────────────────────
    ax.plot(hours, df_profile["baseline_mw"],
            color="tab:red", linewidth=0.8, alpha=0.5, label="Baseline")
    ax.plot(hours, df_profile["scenario_mw"],
            color="tab:blue", linewidth=0.8, alpha=0.5, label=f"Scenario (MP{mp})")

    # ── Peak vertical lines — read dynamically from peak_result ───────────────
    peak_hr_base = peak_result["peak_hour_baseline"]
    peak_mw_base = peak_result["baseline_peak_mw"]
    peak_hr_scen = peak_result["peak_hour_scenario"]
    peak_mw_scen = peak_result["scenario_peak_mw"]

    ax.axvline(x=peak_hr_base, color="black", linestyle="--", linewidth=2.0,
               alpha=0.85, label=f"Baseline peak hr {peak_hr_base}")
    ax.axvline(x=peak_hr_scen, color="black", linestyle="--", linewidth=2.0,
               alpha=0.85, label=f"Scenario peak hr {peak_hr_scen}")

    # ── Annotations at top of axes ────────────────────────────────────────────
    y_top = ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else peak_mw_base * 1.05
    ax_ymin, ax_ymax = ax.get_ylim()
    # Use axes-fraction coordinates so annotations are always near the top
    ax.annotate(
        f"Base peak\n{peak_mw_base:.1f} MW\n(hr {peak_hr_base})",
        xy=(peak_hr_base, ax_ymax),
        xycoords=("data", "axes fraction"),
        xytext=(peak_hr_base, 0.97),
        textcoords=("data", "axes fraction"),
        fontsize=7, color="tab:red",
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tab:red", alpha=0.7),
    )
    ax.annotate(
        f"Scenario peak\n{peak_mw_scen:.1f} MW\n(hr {peak_hr_scen})",
        xy=(peak_hr_scen, ax_ymax),
        xycoords=("data", "axes fraction"),
        xytext=(peak_hr_scen, 0.83),
        textcoords=("data", "axes fraction"),
        fontsize=7, color="tab:blue",
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tab:blue", alpha=0.7),
    )

    # ── Labels ────────────────────────────────────────────────────────────────
    ax.set_xlabel("Hour of Year", fontsize=9)
    ax.set_ylabel("Demand (MW)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=8)

# %%
# =============================================================================
# STEP 7c: 2×2 Demand Timeseries Visualization — Allegheny County, PA
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
scenarios = ["100pct", "constrained"]
scenario_labels = ["100% Adoption", "Constrained (Tier 1+2)"]
mp_labels = {3: "Standard ASHP (15 SEER1, 9 HSPF1)", 4: "High-Efficiency ASHP (24 SEER1, 14 HSPF1)"}

for row_idx, (scenario, scenario_label) in enumerate(zip(scenarios, scenario_labels)):
    for col_idx, mp in enumerate(selected_mps):
        ax = axes[row_idx, col_idx]
        df_profile = df_profiles_by_mp[mp][scenario]
        peak_result = peak_results_allegheny_by_mp[mp][scenario]
        plot_demand_panel(ax, df_profile, peak_result, mp, scenario_label)
        ax.set_title(f"{mp_labels[mp]}\n{scenario_label}", fontsize=11)

plt.suptitle("Hourly Electricity Demand — Allegheny County, PA (AMY2018)", fontsize=13)
plt.tight_layout()

output_path = os.path.join(PROJECT_ROOT, "outputs",
    f"allegheny_demand_profiles_MP{'_'.join(str(m) for m in selected_mps)}.png")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"✓ Figure saved: {output_path}")
plt.show()

# %%
# =============================================================================
# STEP 7c: 2×2 Demand Timeseries Visualization — Allegheny County, PA
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
scenarios = ["100pct", "constrained"]
scenario_labels = ["100% Adoption", "Constrained (Tier 1+2)"]
mp_labels = {3: "Standard ASHP (15 SEER1, 9 HSPF1)", 4: "High-Efficiency ASHP (24 SEER1, 14 HSPF1)"}

for row_idx, (scenario, scenario_label) in enumerate(zip(scenarios, scenario_labels)):
    for col_idx, mp in enumerate(selected_mps):
        ax = axes[row_idx, col_idx]
        df_profile = df_profiles_by_mp[mp][scenario]
        peak_result = peak_results_allegheny_by_mp[mp][scenario]
        plot_demand_panel(ax, df_profile, peak_result, mp, scenario_label)
        ax.set_title(f"{mp_labels[mp]}\n{scenario_label}", fontsize=11)

plt.suptitle("Hourly Electricity Demand — Allegheny County, PA (AMY2018)", fontsize=13)
plt.tight_layout()

output_path = os.path.join(PROJECT_ROOT, "outputs",
    f"allegheny_demand_profiles_MP{'_'.join(str(m) for m in selected_mps)}.png")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"✓ Figure saved: {output_path}")
plt.show()

# %%


# %% [markdown]
# ---
# ## Step 8: Validate — Compare Aggregated Profile Against EUSS Peak Load Columns
# ---
# 
# EUSS annual/metadata files include individual household peak load values (kW).
# These provide a within-dataset sanity check before comparing to external benchmarks.
# 
# **Validation approach:**
# - Sum the EUSS individual peak load values for Allegheny County buildings → county peak (naive sum)
# - Compare to the profile-derived peak from Step 7 (baseline, 100% adoption)
# - These will not be identical (naive sum ≠ coincident peak), but should be same order of magnitude
# 
# **External benchmark (after internal check passes):**
# - Compare PA statewide peak load change to Maxim et al. (2024) 2018 vs. 2050 scenarios
# - Reference OEDI URL from notebook stub:
#   `https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F2022%2Fresstock_amy2018_release_1.1%2F`
# 
# **Flag:** If profile-derived peak differs from EUSS column sum by more than 20%, investigate
# before scaling to national loop.

# %%
# Step 8 stub — see post_tare_review_cleanup_plan_v2.md for spec.
# Validate aggregated profile against EUSS annual peak load columns.
raise NotImplementedError("Step 8 — validate against EUSS metadata peak load columns")

# %% [markdown]
# ---
# ## Step 9: National Loop — County-Level Peak Load Table
# ---
# 
# ⚠️ **Run Step 7 validation on Allegheny County BEFORE running this step.**
# 
# Scales the Allegheny County pipeline across all counties present in `adopter_ids_by_county`.
# 
# **Design decisions to resolve before running:**
# 1. **Query batching:** Query all buildings for a county in one Athena call, or batch by
#    state first (reduces number of Athena connections)?
# 2. **Aggregation location:** Push the hourly sum to Athena SQL (cheaper, faster) or pull
#    building-level data and aggregate in pandas (more flexible for adopter mask)?
# 3. **Checkpoint saving:** Save intermediate results per state to disk in case the loop fails
#    mid-run (strongly recommended for a national run).
# 
# **Expected output:** `df_peak_results_national` with one row per county:
# `[fips, county_name, state, n_adopters_constrained, n_all_filtered,
#   baseline_peak_mw, scenario_100pct_peak_mw, scenario_constrained_peak_mw,
#   delta_100pct_mw, delta_constrained_mw, peak_hour_100pct, peak_hour_constrained]`

# %%
# Step 9 stub — see post_tare_review_cleanup_plan_v2.md for spec.
# Scale county-level pipeline nationally with per-state checkpointing.

def run_national_peak_load_loop(
    bsq: "BuildStockQuery",
    adopter_ids_by_mp: dict[int, dict[str, dict[str, list[int]]]],
    county_geo_df: pd.DataFrame,
    selected_mps: list[int],
    project_root: str,
) -> pd.DataFrame:
    """Scale Steps 5-7 across all counties with per-state checkpointing.

    See post_tare_review_cleanup_plan_v2.md Step 9 spec for design decisions
    (query batching strategy, aggregation location, checkpoint format).

    Args:
        bsq: Initialized BuildStockQuery object.
        adopter_ids_by_mp: Per-MP, per-county adopter IDs from Step 4.
            Keyed ``{mp: {fips: {"tier1": [...], "tier2": [...],
            "constrained": [...], "all_filtered": [...]}}}``
        county_geo_df: County FIPS -> name -> state lookup from Step 3.
        selected_mps: List of integer measure package numbers to process
            (e.g. ``[3, 4]``).
        project_root: Project root for checkpoint file paths.

    Returns:
        DataFrame with one row per county and peak load results.
    """
    raise NotImplementedError("Step 9 — implement after performance profiling (Task 5)")

# %% [markdown]
# ---
# ## Step 10: Export Results for Paper Figures
# ---
# 
# Exports the national county-level peak load table for use in:
# - **Figure XX (paper Section 3.6):** County choropleth of peak load change under
#   100% adoption and economically-constrained adoption, by technology scenario (MP3/MP4)
# - **Allegheny County case study panel:** Bar chart comparing baseline vs scenario peak
#   across all technology scenarios
# 
# Files exported:
# - `peak_load_results_MP{mp}_national.csv` — full national county table
# - `peak_load_results_MP{mp}_allegheny.csv` — Allegheny County only (for case study)

# %%
# Step 10 stub — see post_tare_review_cleanup_plan_v2.md for spec.
# Export national peak load results as CSV for paper figures.
raise NotImplementedError("Step 10 — implement after Step 9 completes")


