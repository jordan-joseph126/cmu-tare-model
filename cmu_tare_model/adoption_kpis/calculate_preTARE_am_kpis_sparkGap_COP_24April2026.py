# %% [markdown]
# ---
# # Pre-TARE Adoption KPIs: Spark Gap, Thermal COP, Break-Even COP
# ---
# 
# **Author:** Jordan M. Joseph, PhD — Carnegie Mellon University
# 
# Computes adoption economics metrics that can be calculated directly from EUSS
# data and EIA fuel prices, without requiring a TARE model run.
# 
# **Workflow:** Load EUSS data → spark gap → thermal COP & AFUE → break-even COP → maps
# 
# **Batch Mode:** Set `input_measure_package` before calling via `%run -i` to skip prompts.
# 
# See `README_adoption_kpis.md` for methodology notes and design decisions.

# %%
import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from config import PROJECT_ROOT
from cmu_tare_model.constants import VALID_MENU_MPS, PA_COP_RANGES
from cmu_tare_model.adoption_kpis import (
    load_euss_baseline, load_euss_upgrade, mp_to_upgrade,
    calculate_spark_gap, compute_thermal_cop, compute_breakeven_cop,
)
from cmu_tare_model.adoption_kpis.thermal_cop import COP_BENCHMARK_RANGES
from cmu_tare_model.adoption_kpis.visualize_geospatial_data import (
    prepare_state_geodataframe, create_choropleth_map,
)
from cmu_tare_model.adoption_kpis.data_loading import FUEL_PRICES_PATH, SHAPEFILE_PATH

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 60)
print("[OK] Imports loaded")
print(f"Project root: {PROJECT_ROOT}")


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
# ## Step 2: Fuel Price Ratios (Spark Gap)
# ---

# %%
print("=" * 60)
print("STEP 2: SPARK GAP (2024 EIA nominal prices)")
print("=" * 60)

df_spark_gap = calculate_spark_gap(FUEL_PRICES_PATH, year=2024)
print(f"[OK] {len(df_spark_gap)} states loaded")
print(f"  Mean: {df_spark_gap['spark_gap'].mean():.2f}, "
      f"Range: {df_spark_gap['spark_gap'].min():.2f}"
      f"–{df_spark_gap['spark_gap'].max():.2f}")
fl = float(df_spark_gap[df_spark_gap['state'] == 'FL']['spark_gap'].values[0])
ak = float(df_spark_gap[df_spark_gap['state'] == 'AK']['spark_gap'].values[0])
print(f"  FL: {fl:.2f}, AK: {ak:.2f}")
print("[OK] STEP 2 COMPLETE")


# %% [markdown]
# ---
# ## Step 3: Thermal COP and Baseline AFUE
# ---

# %%
print("=" * 60)
print("STEP 3: THERMAL COP (Natural Gas homes)")
print("=" * 60)

cop_results = {}
for mp in selected_mps:
    print(f"\nMP{mp}:")
    cop_results[mp] = compute_thermal_cop(
        df_baseline, upgrade_data[mp],
        fuel_filter='Natural Gas', verbose=True,
    )
    print(f"  Mean COP: {cop_results[mp]['thermal_cop'].mean():.3f}")

primary_mp = selected_mps[0]
df_cop = cop_results[primary_mp]
print(f"\n[OK] STEP 3 COMPLETE — Primary MP: {primary_mp}")


# %% [markdown]
# ---
# ### Climate Zone Benchmark Validation
# ---
# 
# Aggregates thermal COP by IECC climate zone group and validates against
# literature-derived benchmark ranges (`COP_BENCHMARK_RANGES`). Also
# produces a state × climate zone cross-tab with a PA spot check.

# %%
print("=" * 60)
print("CLIMATE ZONE BENCHMARK VALIDATION")
print("=" * 60)

for mp in selected_mps:
    mp_key = f'mp{mp}'
    print(f"\n--- MP{mp} Climate Zone Group Aggregation ---")

    df_cop_cz = compute_thermal_cop(
        df_baseline, upgrade_data[mp],
        group_cols=['cz_group'],
        fuel_filter='Natural Gas',
        verbose=True,
    )

    df_cop_state_cz = compute_thermal_cop(
        df_baseline, upgrade_data[mp],
        group_cols=['state', 'cz_group'],
        fuel_filter='Natural Gas',
        verbose=False,
    )

    print(f"\n--- Benchmark Validation (MP{mp}) ---")
    print(f"{'CZ Group':<10} {'Label':<18} {'COP':>6} {'Expected':>14} {'Pass':>6}")
    print("-" * 58)
    n_checked = 0
    n_passed = 0
    for cz_grp, bench in COP_BENCHMARK_RANGES.items():
        row = df_cop_cz[df_cop_cz['cz_group'] == cz_grp]
        if len(row) == 0:
            print(f"{cz_grp:<10} {bench['label']:<18} {'N/A':>6} {'':<14} {'SKIP':>6}")
            continue
        cop_val = row['thermal_cop'].values[0]
        if mp_key in bench:
            lo, hi = bench[mp_key]
            in_range = lo <= cop_val <= hi
            n_checked += 1
            n_passed += int(in_range)
            range_str = f"[{lo:.1f}, {hi:.1f}]"
            print(f"{cz_grp:<10} {bench['label']:<18} {cop_val:>6.2f} {range_str:>14} "
                  f"{'[OK]' if in_range else '[FAIL]':>6}")
        else:
            print(f"{cz_grp:<10} {bench['label']:<18} {cop_val:>6.2f} {'no bench':>14} {'—':>6}")
    print(f"\nOverall: {n_passed}/{n_checked} climate zone groups within expected range")

    pa_rows = df_cop_state_cz[df_cop_state_cz['state'] == 'PA']
    if len(pa_rows) > 0:
        print(f"\n--- PA Spot Check (MP{mp}) ---")
        for _, pa_row in pa_rows.iterrows():
            cop_val = pa_row['thermal_cop']
            cz = pa_row['cz_group']
            if mp_key in PA_COP_RANGES:
                lo, hi = PA_COP_RANGES[mp_key]
                ok = lo <= cop_val <= hi
                print(f"  PA (CZ {cz}): COP = {cop_val:.2f}, "
                      f"expected [{lo:.1f}, {hi:.1f}] -> {'[OK] PASS' if ok else '[FAIL] FAIL'}")
            else:
                print(f"  PA (CZ {cz}): COP = {cop_val:.2f} (no PA benchmark for {mp_key})")
    else:
        print("\n[WARN] PA not found in state x CZ cross-tab")

print(f"\n[OK] VALIDATION COMPLETE")


# %% [markdown]
# ---
# ## Step 4: Break-Even COP and Merged Metrics
# ---

# %%
from cmu_tare_model.constants import JENKINS_BREAKEVEN_REF_90

print("=" * 60)
print("STEP 4: BREAK-EVEN COP")
print("=" * 60)

df_breakeven = compute_breakeven_cop(df_spark_gap, df_cop)
be90 = df_breakeven['breakeven_cop_90']
print(f"[OK] BE @90% AFUE — Mean: {be90.mean():.2f}, "
      f"Range: {be90.min():.2f}–{be90.max():.2f}")

n_pass, n_total = 0, len(JENKINS_BREAKEVEN_REF_90)
for st, ref in JENKINS_BREAKEVEN_REF_90.items():
    row = df_breakeven[df_breakeven['state'] == st]
    if len(row) > 0 and abs(float(row['breakeven_cop_90'].values[0]) - ref) <= 0.50:
        n_pass += 1
print(f"Jenkins validation (+-0.50): {n_pass}/{n_total} states")
print("[OK] STEP 4 COMPLETE")


# %% [markdown]
# ---
# ## Step 5: Geospatial Visualization
# ---

# %%
gdf_conus = None
gdf_alaska = None

# Merge all state-level metrics for map rendering
df_map = df_spark_gap.merge(
    df_cop[['state', 'thermal_cop', 'baseline_afue']], on='state', how='inner'
).merge(
    df_breakeven[['state', 'breakeven_cop_90']], on='state', how='inner'
)

try:
    print(f"Loading shapefile from: {SHAPEFILE_PATH}")
    gdf_states_raw = gpd.read_file(SHAPEFILE_PATH)
    _, gdf_conus, gdf_alaska = prepare_state_geodataframe(
        gdf_states_raw, df_map, merge_col='state'
    )
    print(f"  CONUS: {len(gdf_conus)}, Alaska: {len(gdf_alaska)}")
    print("[OK] Geodataframe prepared")
except FileNotFoundError:
    print(f"[WARN] Shapefile not found at {SHAPEFILE_PATH} — skipping maps")
except Exception as e:
    print(f"[WARN] Could not load shapefile: {e}")


# %%
if gdf_conus is not None and gdf_alaska is not None:
    print("Generating choropleth maps...")
    for column, title, cmap, cbar, fname in [
        ('spark_gap',
         'Electricity-to-Natural Gas Spark Gap by State (2024)',
         'Blues', 'Spark Gap (electricity price / gas price, $/MMBTU)',
         'state_spark_gap_map_2024.png'),
        ('thermal_cop',
         f'Heat Pump Thermal COP by State (MP{primary_mp}, 2024)',
         'Greens', 'Thermal COP (heat delivered / electricity consumed)',
         f'state_thermal_cop_mp{primary_mp}_map_2024.png'),
        ('breakeven_cop_90',
         'Break-Even COP by State at 90% AFUE (2024)',
         'Oranges', 'Break-Even COP (COP threshold to match furnace fuel cost)',
         'state_breakeven_cop_90_map_2024.png'),
    ]:
        create_choropleth_map(
            gdf_conus, gdf_alaska, column=column, title=title,
            cbar_label=cbar, cmap=cmap, show_plot=True, dpi=600,
            output_path=os.path.join(PROJECT_ROOT, fname),
        )
    print("[OK] Maps generated")
else:
    print("[WARN] Maps skipped — no geodataframe available")


# %%
print("===== SPARK GAP (2024 nominal) =====\n")
print(df_spark_gap[['state', 'state_name', 'spark_gap',
                     'elec_price_mmbtu', 'gas_price_mmbtu']].to_string())

for mp in selected_mps:
    print(f"\n===== THERMAL COP & AFUE (MP{mp}, NG homes) =====\n")
    print(cop_results[mp].sort_values('thermal_cop', ascending=False)[
        ['state', 'thermal_cop', 'baseline_afue', 'fans_pumps_pct', 'home_count']
    ].to_string())

# --- Comparison table: spark gap, break-even COP, effective COP ---
print(f"\n===== SPARK GAP & BREAK-EVEN COP vs EFFECTIVE COP (MP{primary_mp}) =====\n")

df_display = df_breakeven.copy()
df_display['thermal_cop'] = (
    df_cop.set_index('state')['thermal_cop']
    .reindex(df_display['state'].values)
    .values
)
df_display['hp_beats_breakeven_90'] = (
    df_display['thermal_cop'] > df_display['breakeven_cop_90']
)

print(df_display[[
    'state', 'state_name', 'spark_gap',
    'breakeven_cop_80', 'breakeven_cop_90', 'breakeven_cop_95', 'breakeven_cop_100',
    'thermal_cop', 'baseline_afue', 'hp_beats_breakeven_90',
]].sort_values('spark_gap', ascending=False).reset_index(drop=True).to_string())


# %%



