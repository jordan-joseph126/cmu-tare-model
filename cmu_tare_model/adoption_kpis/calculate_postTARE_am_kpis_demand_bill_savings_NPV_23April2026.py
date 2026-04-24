# %% [markdown]
# ---
# # Post-TARE Adoption KPIs: Bill Savings, Demand Change, NPV
# ---
# 
# **Author:** Jordan M. Joseph, PhD — Carnegie Mellon University
# 
# Computes adoption metrics that depend on EUSS building-level data and (optionally)
# TARE model run outputs: actual bill savings, electricity demand change, and site
# energy change under various adoption scenarios.
# 
# **Prerequisite:** Run the preTARE notebook first (or ensure EUSS data is loaded).
# 
# See `README_adoption_kpis.md` for methodology notes and design decisions.

# %% [markdown]
# ---
# ## Step 0: Imports and Configuration
# ---

# %%
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from config import PROJECT_ROOT
from cmu_tare_model.constants import (
    ALLOWED_HOUSING_TYPES,
    VALID_MENU_MPS,
    VERBOSE,
    REMDB_COST_SCENARIO_KEYS,
    RCM_MODELS,
    PRIVATE_DISCOUNT_RATE_SHORT_KEYS,
)

from cmu_tare_model.utils.column_names import (
    create_npv_col,
    create_capital_col,
)

from cmu_tare_model.utils.load_exported_results_to_df import load_measure_package_data

from cmu_tare_model.adoption_kpis.kpi_functions import (
    mp_to_upgrade,
    load_euss_baseline,
    load_euss_upgrade,
    calculate_price_ratios,
    compute_thermal_cop_by_state,
    compute_spark_gap_metrics,
    compute_scenario_demand,
    aggregate_demand_by_state,
    FUEL_PRICES_PATH,
    SHAPEFILE_PATH,
    HEATING_FUEL_COLS,
    HP_BACKUP_ELEC_COL,
    HP_FANS_PUMPS_COL,
)
from cmu_tare_model.adoption_kpis.visualize_geospatial_data import (
    prepare_state_geodataframe,
    create_choropleth_map,
)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 60)

print("[OK] Imports loaded")

# %% [markdown]
# ---
# ## Step 0b: Measure Package Selection
# ---

# %%
SELECTABLE_MPS = [mp for mp in VALID_MENU_MPS if mp != 0]

try:
    _ = input_measure_package
    batch_mode = True
    selected_mps = [int(input_measure_package)]
    print(f"BATCH MODE: Running for MP{selected_mps[0]}")
except NameError:
    batch_mode = False
    print(f"Available measure packages: {SELECTABLE_MPS}")
    mp_input = input("Enter MP numbers (comma-separated, or 'all'): ").strip()
    if mp_input.lower() == 'all':
        selected_mps = SELECTABLE_MPS
    else:
        selected_mps = [int(x.strip()) for x in mp_input.split(',') if x.strip().isdigit()]
        selected_mps = [mp for mp in selected_mps if mp in SELECTABLE_MPS]
    if not selected_mps:
        selected_mps = [4]
        print("No valid MPs selected. Defaulting to MP4.")

print(f"\nSelected measure packages: {selected_mps}")

# %% [markdown]
# ---
# ## Step 0c: Load TARE Model Data (Measure Packages 3, 4)
# ---
# 
# Load pre-computed TARE model outputs for the selected measure packages.
# Required for Step 2d (Private NPV extraction).
# 
# If the top-section data loading cells have already been run, this will reuse `DATAFRAMES_BY_MP`.

# %%
# =============================================================================
# STEP 0c: LOAD TARE MODEL DATA (for NPV extraction)
# =============================================================================
# Check if DATAFRAMES_BY_MP was already loaded by the top-section cells.
# If not, prompt for the output folder and load for selected MPs.

try:
    _ = DATAFRAMES_BY_MP
    print(f"DATAFRAMES_BY_MP already loaded: {list(DATAFRAMES_BY_MP.keys())}")
except NameError:
    print("DATAFRAMES_BY_MP not found — loading TARE model outputs...")

    # Check if output_folder_path is already defined (from top section)
    try:
        _ = output_folder_path
        print(f"  Using existing output_folder_path: {output_folder_path}")
    except NameError:
        output_folder_path = os.path.join(PROJECT_ROOT, "cmu_tare_model", "output_results")
        location_id = input("Enter location ID (e.g., 'National' or 'PA'): ").strip()
        model_run_date_time = input("Enter model run timestamp (YYYY-MM-DD_HH-MM): ").strip()
        print(f"  output_folder_path: {output_folder_path}")
        print(f"  location_id: {location_id}")
        print(f"  model_run_date_time: {model_run_date_time}")

    DATAFRAMES_BY_MP = {}
    for mp in selected_mps:
        DATAFRAMES_BY_MP[mp] = load_measure_package_data(
            mp, output_folder_path, location_id, model_run_date_time
        )

    print(f"\n[OK] Loaded TARE data for MPs: {list(DATAFRAMES_BY_MP.keys())}")

# %% [markdown]
# ---
# ## Step 1: Load EUSS Data
# ---

# %%
print("=" * 80)
print("STEP 1: LOAD EUSS DATA")
print("=" * 80)

df_baseline = load_euss_baseline()
print(f"  Baseline: {len(df_baseline):,} occupied SF homes")

upgrade_data = {}
for mp in selected_mps:
    upgrade_name = mp_to_upgrade(mp)
    print(f"\nLoading MP{mp} ({upgrade_name})...")
    upgrade_data[mp] = load_euss_upgrade(upgrade_name)
    print(f"  MP{mp}: {len(upgrade_data[mp]):,} applicable homes")

print(f"\n[OK] STEP 1 COMPLETE")

# %% [markdown]
# ---
# ## Step 2: Bill Savings Ratio (Per-Building → State Median)
# ---
# 
# Computes the actual bill savings ratio per building:
# `ratio = retrofit_elec_cost / baseline_heating_cost`
# 
# Unlike the analytical `bill_impact_ratio` (spark_gap × AFUE / COP), this uses
# actual per-building energy consumption and fuel-specific state prices.
# Ratio < 1 = HP saves money; ratio > 1 = HP costs more.

# %%
df_tare_mp3_results = DATAFRAMES_BY_MP[3]['fixed_base']['inmap']
mp3_fuel_cost_cols = [col for col in df_tare_mp3_results.columns if 'fuel_cost' in col]
print(f"MP3 fuel cost columns: {mp3_fuel_cost_cols}")

# %%
df_tare_mp4_results = DATAFRAMES_BY_MP[4]['fixed_base']['inmap']
mp4_fuel_cost_cols = [col for col in df_tare_mp4_results.columns if 'fuel_cost' in col]
print(f"MP4 fuel cost columns: {mp4_fuel_cost_cols}")

# %%
# ============================================================================
# Step 2: BILL SAVINGS RATIO — Per-Building → State Median
# UPDATE SO THAT IT LOOPS OVER SELECTED MPs AND USES ACTUAL POST-TARE LIFETIME FUEL COST DATA INSTEAD OF PLACEHOLDER CALCULATIONS
# ============================================================================

print(f"===== Step 2: BILL SAVINGS RATIO (MP{mp}, NG homes) =====")

# ============================================================================
# PLACEHOLDER: NEED TO UPDATE SO USES ACTUAL POST-TARE LIFETIME FUEL COST DATA
# MP3 fuel cost columns: ['preIRA_mp3_heating_lifetime_fuel_cost', 'preIRA_mp3_heating_lifetime_savings_fuel_cost', 'iraRef_mp3_heating_lifetime_fuel_cost', 'iraRef_mp3_heating_lifetime_savings_fuel_cost', 'baseline_heating_lifetime_fuel_cost']
# MP4 fuel cost columns: ['preIRA_mp4_heating_lifetime_fuel_cost', 'preIRA_mp4_heating_lifetime_savings_fuel_cost', 'iraRef_mp4_heating_lifetime_fuel_cost', 'iraRef_mp4_heating_lifetime_savings_fuel_cost', 'baseline_heating_lifetime_fuel_cost']
# ============================================================================

baseline_lifetime_fuel_cost_col = 'baseline_heating_lifetime_fuel_cost'
retrofit_lifetime_fuel_cost_col = f'iraRef_mp{mp}_heating_lifetime_fuel_cost'

# # Bill savings ratio: retrofit / baseline (< 1 = savings)
# # Guard against zero baseline cost
# df_ratio['bill_savings_ratio'] = np.where(
#     df_ratio['baseline_annual_cost'] > 0,
#     df_ratio['retrofit_annual_cost'] / df_ratio['baseline_annual_cost'],
#     np.nan,
# )

# print(f"Per-building records: {len(df_ratio):,}")
# print(f"Median bill savings ratio: {df_ratio['bill_savings_ratio'].median():.3f}")
# print(f"Mean bill savings ratio:   {df_ratio['bill_savings_ratio'].mean():.3f}")

# # Aggregate: state-level MEDIAN bill savings ratio and MEDIAN annual savings
# df_ratio_state = df_ratio.groupby('state').agg(
#     home_count=('weight', 'size'),
#     median_bill_savings_ratio=('bill_savings_ratio', 'median'),
#     mean_bill_savings_ratio=('bill_savings_ratio', 'mean'),
#     median_annual_savings=('annual_bill_savings', 'median'),
#     mean_annual_savings=('annual_bill_savings', 'mean'),
#     total_baseline_cost=('baseline_annual_cost', lambda x: (x * df_ratio.loc[x.index, 'weight']).sum()),
#     total_retrofit_cost=('retrofit_annual_cost', lambda x: (x * df_ratio.loc[x.index, 'weight']).sum()),
# ).reset_index()

# # Weighted aggregate ratio as cross-check
# df_ratio_state['weighted_ratio'] = np.where(
#     df_ratio_state['total_baseline_cost'] > 0,
#     df_ratio_state['total_retrofit_cost'] / df_ratio_state['total_baseline_cost'],
#     np.nan,
# )

# # Round for display
# for col in ['median_bill_savings_ratio', 'mean_bill_savings_ratio', 'weighted_ratio']:
#     df_ratio_state[col] = df_ratio_state[col].round(3)
# for col in ['median_annual_savings', 'mean_annual_savings']:
#     df_ratio_state[col] = df_ratio_state[col].round(2)

# n_savings = (df_ratio_state['median_bill_savings_ratio'] < 1.0).sum()
# print(f"\nStates where median home saves money (ratio < 1): {n_savings} / {len(df_ratio_state)}")

# print(f"\n--- Top 5 States (Best for Electrification — lowest ratio) ---")
# top5 = df_ratio_state.nsmallest(5, 'median_bill_savings_ratio')
# print(top5[['state', 'median_bill_savings_ratio', 'median_annual_savings',
#             'mean_annual_savings', 'home_count']].to_string(index=False))

# print(f"\n--- Bottom 5 States (Worst for Electrification — highest ratio) ---")
# bot5 = df_ratio_state.nlargest(5, 'median_bill_savings_ratio')
# print(bot5[['state', 'median_bill_savings_ratio', 'median_annual_savings',
#             'mean_annual_savings', 'home_count']].to_string(index=False))

# print("\n[OK] Step 2 COMPLETE")

# %% [markdown]
# ---
# ## Step 3: Demand Change Under Adoption Scenario
# ---
# 
# Two metrics: **electricity demand change** (grid impact) and **site energy change** (efficiency).
# 
# # TODO: UPDATE TO ALSO DO COUNTY LEVEL RESULTS. RIGHT NOW ONLY STATE.

# %%
print(f"===== Step 3a: SCENARIO DEMAND (MP{primary_mp}, 100% adoption, all fuels) =====")
df_demand = compute_scenario_demand(df_baseline, df_upgrade_primary, fuel_filter=None, verbose=True)

print(f"\n--- Sample: gas homes ---")
gas_sample = df_demand[df_demand['in.heating_fuel'] == 'Natural Gas'].head(3)
print(gas_sample[['in.state', 'in.heating_fuel', 'baseline_electric_kwh',
                   'baseline_heating_total_kwh', 'retrofit_electric_kwh',
                   'elec_demand_change_kwh', 'site_energy_change_kwh']].to_string())

print(f"\n--- Sample: electric baseboard homes ---")
elec_sample = df_demand[df_demand['in.heating_fuel'] == 'Electricity'].head(3)
print(elec_sample[['in.state', 'in.heating_fuel', 'baseline_electric_kwh',
                    'baseline_heating_total_kwh', 'retrofit_electric_kwh',
                    'elec_demand_change_kwh', 'site_energy_change_kwh']].to_string())
print("\n[OK] Step 3a COMPLETE")

# %%
print("===== Step 3b: AGGREGATE DEMAND BY STATE =====")
df_demand_state = aggregate_demand_by_state(df_demand, verbose=True)

print(f"\n--- Top 5 (largest elec demand increase) ---")
print(df_demand_state[['state', 'elec_change_gwh', 'pct_elec_demand_change',
                        'site_energy_change_gwh', 'pct_site_energy_change']].head(5).to_string(index=False))
print(f"\n--- Bottom 5 ---")
print(df_demand_state[['state', 'elec_change_gwh', 'pct_elec_demand_change',
                        'site_energy_change_gwh', 'pct_site_energy_change']].tail(5).to_string(index=False))
print("\n[OK] Step 3b COMPLETE")

# %% [markdown]
# ---
# ## Step 4: Geospatial Visualization
# ---
# 
# # TODO: UPDATE TO ALSO DO COUNTY LEVEL RESULTS. RIGHT NOW ONLY STATE.

# %%
gdf_conus = None
gdf_alaska = None

try:
    gdf_states_raw = gpd.read_file(SHAPEFILE_PATH)
    _, gdf_conus, gdf_alaska = prepare_state_geodataframe(gdf_states_raw, df_spark, merge_col='state')
    print(f"[OK] Geodataframe prepared: CONUS={len(gdf_conus)}, AK={len(gdf_alaska)}")
except Exception as e:
    print(f"[WARN] Shapefile not loaded: {e} — skipping maps")

# %%
# Demand change map (diverging)
if gdf_conus is not None and gdf_alaska is not None:
    _, gdf_demand_conus, gdf_demand_alaska = prepare_state_geodataframe(
        gdf_states_raw, df_demand_state, merge_col='state'
    )
    create_choropleth_map(
        gdf_demand_conus, gdf_demand_alaska,
        column='elec_change_gwh',
        title='Electricity Demand Change Under 100% HP Adoption by State (2022)',
        cbar_label='Electricity Demand Change (GWh)\n(positive = more grid electricity needed)',
        output_path=os.path.join(PROJECT_ROOT, "state_elec_demand_change_map_2022.png"),
        cmap='coolwarm', show_plot=True,
    )
    print("[OK] Demand map generated")
else:
    print("[WARN] Maps skipped")

# %%
# ============================================================================
# Step 4b: BILL SAVINGS RATIO CHOROPLETH MAP
# TODO: 
# - UPDATE THIS TO USE ACTUAL POST-TARE LIFETIME FUEL COST DATA INSTEAD OF PLACEHOLDER CALCULATIONS
# - REMOVE ALL NPV CALCULATIONS/COLUMNS FROM THIS ANALYSIS AND MAPS
# - ENSURE THAT THE MAPS SHARE A COLOR SCALE
# ============================================================================

if gdf_conus is not None and gdf_alaska is not None:
    print("Generating bill savings and NPV choropleth maps...")

    # --- Merge bill savings ratio data with geodata ---
    _, gdf_ratio_conus, gdf_ratio_alaska = prepare_state_geodataframe(
        gdf_states_raw, df_ratio_state, merge_col='state'
    )

    # --- Map 1: Median Bill Savings Ratio (diverging, centered at 1.0) ---
    ratio_vals = pd.concat([
        gdf_ratio_conus['median_bill_savings_ratio'],
        gdf_ratio_alaska['median_bill_savings_ratio']
    ]).dropna()

    ratio_map_path = os.path.join(
        PROJECT_ROOT, f"state_bill_savings_ratio_map_2022_MP{primary_mp}.png"
    )
    create_choropleth_map(
        gdf_ratio_conus, gdf_ratio_alaska,
        column='median_bill_savings_ratio',
        title=(
            f'Median Bill Savings Ratio by State (MP{primary_mp}, 2022)\n'
            '(ratio < 1 = HP saves money; ratio > 1 = HP costs more)'
        ),
        cbar_label='Bill Savings Ratio\n(retrofit cost / baseline cost)',
        output_path=ratio_map_path,
        cmap='RdBu_r',
        norm=mcolors.TwoSlopeNorm(
            vmin=ratio_vals.min(),
            vcenter=1.0,
            vmax=ratio_vals.max(),
        ),
        show_plot=True,
    )

    print("[OK] Bill savings ratio maps generated")
else:
    print("[WARN] Maps skipped — geodataframe not available")

# %% [markdown]
# ---
# ## Display Results
# ---

# %%
# UPDATE THIS TO 2024 PRICES
print("===== PRICE RATIOS (2022 nominal) =====\n")
display(df_prices_csv)

for mp in selected_mps:
    print(f"\n===== THERMAL COP & AFUE (MP{mp}, NG homes) =====\n")
    display(cop_results[mp].sort_values('thermal_cop', ascending=False)[
        ['state', 'thermal_cop', 'baseline_afue', 'fans_pumps_pct', 'home_count']
    ])

# ============================================================================
# DISPLAY: BILL SAVINGS RATIO
# ============================================================================

print(f"\n===== BILL SAVINGS RATIO (MP{primary_mp}, NG homes, median per state) =====\n")

print(f"  States where median home saves money (ratio < 1): "
      f"{(df_ratio_state['median_bill_savings_ratio'] < 1.0).sum()} / {len(df_ratio_state)}")
print(f"  National median ratio: {df_ratio_state['median_bill_savings_ratio'].median():.3f}")
print(f"  Range: {df_ratio_state['median_bill_savings_ratio'].min():.3f} - "
      f"{df_ratio_state['median_bill_savings_ratio'].max():.3f}")

print(f"\n--- Top 5 States (Best for Electrification — lowest bill savings ratio) ---")
top5_ratio = df_ratio_state.nsmallest(5, 'median_bill_savings_ratio')
print(top5_ratio[['state', 'median_bill_savings_ratio', 'median_annual_savings',
                   'weighted_ratio', 'home_count']].to_string(index=False))

print(f"\n--- Bottom 5 States (Worst for Electrification — highest bill savings ratio) ---")
bot5_ratio = df_ratio_state.nlargest(5, 'median_bill_savings_ratio')
print(bot5_ratio[['state', 'median_bill_savings_ratio', 'median_annual_savings',
                   'weighted_ratio', 'home_count']].to_string(index=False))

# ============================================================================
# DISPLAY: DEMAND CHANGE
# ============================================================================

print(f"\n===== DEMAND CHANGE (MP{primary_mp}, GWh, all fuels, 100% adoption) =====\n")
display(df_demand_state[['state', 'home_count', 'elec_change_gwh',
                          'pct_elec_demand_change', 'site_energy_change_gwh',
                          'pct_site_energy_change']])

print(f"\n[OK] DISPLAY COMPLETE")


