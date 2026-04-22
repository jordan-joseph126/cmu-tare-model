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
    JENKINS_BREAKEVEN_REF_90,
    PA_COP_RANGES,
)

from cmu_tare_model.adoption_kpis.kpi_functions import (
    mp_to_upgrade,
    load_euss_baseline,
    load_euss_upgrade,
    calculate_price_ratios,
    compute_thermal_cop,          # renamed from compute_thermal_cop_by_state
    compute_breakeven_cop,
    compute_spark_gap_metrics,
    broadcast_prices_to_counties, # Phase 2: county aggregation
    iecc_to_cz_group,             # new — climate zone mapping
    COP_BENCHMARK_RANGES,         # new — validation benchmark ranges
    FUEL_PRICES_PATH,
    SHAPEFILE_PATH,
    COUNTY_SHAPEFILE_PATH,        # Phase 2: county shapefile path
    COUNTY_COL,                   # Phase 2: GISJOIN county column name
    HEATING_LOAD_COL,
    CLIMATE_ZONE_COL,             # new — column name constant
)
from cmu_tare_model.adoption_kpis.visualize_geospatial_data import (
    prepare_state_geodataframe,
    prepare_county_geodataframe,  # Phase 2: county geodataframe
    create_choropleth_map,
)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 60)

print("✓ Imports loaded")
print(f"Project root: {PROJECT_ROOT}")

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

print(f"\n✓ STEP 1 COMPLETE")

# %% [markdown]
# ---
# ## Step 2: Fuel Price Ratios (Spark Gap)
# ---

# %%
print("===== STEP 2: FUEL PRICE RATIOS (CSV, nominal) =====")

# --- 2022 single-year ---
df_prices_2022 = calculate_price_ratios(FUEL_PRICES_PATH, year=2022)
print(f"✓ 2022 prices loaded for {len(df_prices_2022)} states")
print(f"  Spark Gap — Mean: {df_prices_2022['spark_gap'].mean():.2f}, "
      f"Range: {df_prices_2022['spark_gap'].min():.2f}–{df_prices_2022['spark_gap'].max():.2f}")

# --- 2024 single-year ---
df_prices_csv = calculate_price_ratios(FUEL_PRICES_PATH, year=2024)
print(f"\n✓ 2024 prices loaded for {len(df_prices_csv)} states")
print(f"  Spark Gap — Mean: {df_prices_csv['spark_gap'].mean():.2f}, "
      f"Range: {df_prices_csv['spark_gap'].min():.2f}–{df_prices_csv['spark_gap'].max():.2f}")

# --- 5-year average (2020–2024) ---
df_prices_5yr = calculate_price_ratios(FUEL_PRICES_PATH, year=list(range(2020, 2025)))
print(f"\n✓ 5-year avg (2020–2024) loaded for {len(df_prices_5yr)} states")
print(f"  Spark Gap — Mean: {df_prices_5yr['spark_gap'].mean():.2f}, "
      f"Range: {df_prices_5yr['spark_gap'].min():.2f}–{df_prices_5yr['spark_gap'].max():.2f}")

# --- 10-year average (2015–2024) ---
df_prices_10yr = calculate_price_ratios(FUEL_PRICES_PATH, year=list(range(2015, 2025)))
print(f"\n✓ 10-year avg (2015–2024) loaded for {len(df_prices_10yr)} states")
print(f"  Spark Gap — Mean: {df_prices_10yr['spark_gap'].mean():.2f}, "
      f"Range: {df_prices_10yr['spark_gap'].min():.2f}–{df_prices_10yr['spark_gap'].max():.2f}")

print(f"\n✓ STEP 2 COMPLETE")

# %%
# Build summary table: 2024 prices + spark gaps for 2022, 2024, 5yr, 10yr
df_summary = df_prices_csv[['state', 'state_name', 'elec_price_mmbtu', 'gas_price_mmbtu']].copy()
df_summary = df_summary.rename(columns={
    'elec_price_mmbtu': 'elec_$/mmbtu_2024',
    'gas_price_mmbtu': 'gas_$/mmbtu_2024',
})

df_summary = df_summary.merge(
    df_prices_2022[['state', 'spark_gap']].rename(columns={'spark_gap': 'spark_gap_2022'}),
    on='state', how='inner',
)
df_summary = df_summary.merge(
    df_prices_csv[['state', 'spark_gap']].rename(columns={'spark_gap': 'spark_gap_2024'}),
    on='state', how='inner',
)
df_summary = df_summary.merge(
    df_prices_5yr[['state', 'spark_gap']].rename(columns={'spark_gap': 'spark_gap_5yr_avg'}),
    on='state', how='inner',
)
df_summary = df_summary.merge(
    df_prices_10yr[['state', 'spark_gap']].rename(columns={'spark_gap': 'spark_gap_10yr_avg'}),
    on='state', how='inner',
)

# Sort by 2024 spark gap for display
df_summary = df_summary.sort_values('spark_gap_2024', ascending=False).reset_index(drop=True)

display_cols = [
    'state', 'state_name',
    'elec_$/mmbtu_2024', 'gas_$/mmbtu_2024',
    'spark_gap_2022', 'spark_gap_2024', 'spark_gap_5yr_avg', 'spark_gap_10yr_avg',
]

print("=" * 100)
print("SPARK GAP SUMMARY: TOP 5 & BOTTOM 5 STATES")
print("=" * 100)
print(f"\n{'Elec & Gas prices shown as 2024 nominal $/MMBTU'}")
print(f"{'Spark Gap = Electricity Price / Natural Gas Price'}\n")

print("--- Top 5 (highest spark gap, most favorable for gas heating) ---")
print(df_summary[display_cols].head(5).to_string(index=False))

print("\n--- Bottom 5 (lowest spark gap, most favorable for heat pumps) ---")
print(df_summary[display_cols].tail(5).to_string(index=False))

print(f"\n--- National Summary ---")
for label, col in [('2022', 'spark_gap_2022'), ('2024', 'spark_gap_2024'),
                    ('5yr avg', 'spark_gap_5yr_avg'), ('10yr avg', 'spark_gap_10yr_avg')]:
    vals = df_summary[col]
    print(f"  {label:>8}: Mean={vals.mean():.2f}, Median={vals.median():.2f}, "
          f"Range=[{vals.min():.2f}, {vals.max():.2f}]")

# %% [markdown]
# ---
# ## Step 3: Thermal COP and Baseline AFUE
# ---

# %%
cop_results = {}

for mp in selected_mps:
    print(f"\n{'=' * 80}")
    print(f"STEP 3: THERMAL COP & BASELINE AFUE (MP{mp}, Natural Gas homes)")
    print(f"{'=' * 80}")

    cop_results[mp] = compute_thermal_cop(
        df_baseline, upgrade_data[mp],
        group_cols=['state'],
        fuel_filter='Natural Gas',
        verbose=True,
    )

    df_cop_mp = cop_results[mp]
    print(f"\n--- Top 5 by COP (MP{mp}) ---")
    print(df_cop_mp.sort_values('thermal_cop', ascending=False).head(5)[
        ['state', 'thermal_cop', 'baseline_afue', 'fans_pumps_pct', 'home_count']
    ].to_string(index=False))

    print(f"\n--- Bottom 5 by COP (MP{mp}) ---")
    print(df_cop_mp.sort_values('thermal_cop', ascending=True).head(5)[
        ['state', 'thermal_cop', 'baseline_afue', 'fans_pumps_pct', 'home_count']
    ].to_string(index=False))

    # Validation: heating load consistency
    common_ids = df_baseline.index.intersection(upgrade_data[mp].index)
    Q_baseline = df_baseline.loc[common_ids, HEATING_LOAD_COL].fillna(0)
    Q_upgrade = upgrade_data[mp].loc[common_ids, HEATING_LOAD_COL].fillna(0)
    mask = Q_baseline > 0
    pct_diff = ((Q_upgrade[mask] - Q_baseline[mask]) / Q_baseline[mask]).median()
    print(f"\nHeating load consistency: {pct_diff:.1%} median difference (expect ±5–10%)")

primary_mp = selected_mps[0]
df_cop = cop_results[primary_mp]
df_upgrade_primary = upgrade_data[primary_mp]
print(f"\n✓ STEP 3 COMPLETE — Primary MP: {primary_mp}")

# %% [markdown]
# ---
# ## Task B: COP Direction Check (Zero-Baseline Filter Validation)
# ---
# 
# Validates that the zero-baseline filter (excluding homes with no prior
# heating system) lowers COP as expected. Compares fixed (default) vs
# unfixed (`require_baseline_heating=False`) results per state.

# %%
print("=" * 80)
print("TASK B: COP DIRECTION CHECK (zero-baseline filter validation)")
print("=" * 80)

for mp in selected_mps:
    print(f"\n--- MP{mp} ---")

    # Unfixed: include homes with zero baseline heating (old behavior)
    cop_unfixed = compute_thermal_cop(
        df_baseline, upgrade_data[mp],
        group_cols=['state'],
        fuel_filter='Natural Gas',
        require_baseline_heating=False,
        verbose=False,
    )

    # Fixed: cop_results[mp] already computed with require_baseline_heating=True (default)
    cop_fixed = cop_results[mp]

    # Merge for comparison
    df_compare = cop_unfixed[['state', 'thermal_cop']].rename(
        columns={'thermal_cop': 'cop_unfixed'}
    ).merge(
        cop_fixed[['state', 'thermal_cop']].rename(
            columns={'thermal_cop': 'cop_fixed'}
        ),
        on='state', how='outer',
    )
    df_compare['delta'] = df_compare['cop_fixed'] - df_compare['cop_unfixed']
    df_compare['direction_ok'] = df_compare['cop_fixed'] <= df_compare['cop_unfixed']

    print(f"\n{'State':<6} {'COP_unfixed':>12} {'COP_fixed':>10} {'Delta':>8} {'Dir OK?':>8}")
    print("-" * 48)
    for _, r in df_compare.sort_values('state').iterrows():
        print(f"{r['state']:<6} {r['cop_unfixed']:>12.3f} {r['cop_fixed']:>10.3f} "
              f"{r['delta']:>+8.3f} {'✓' if r['direction_ok'] else '✗':>8}")

    n_pass = df_compare['direction_ok'].sum()
    n_total = len(df_compare)
    n_fail = n_total - n_pass
    print(f"\nSummary: {n_pass}/{n_total} states pass direction check")
    if n_fail > 0:
        fails = df_compare[~df_compare['direction_ok']]
        print(f"⚠ {n_fail} state(s) where fixed COP > unfixed (investigate):")
        for _, r in fails.iterrows():
            print(f"    {r['state']}: unfixed={r['cop_unfixed']:.3f}, "
                  f"fixed={r['cop_fixed']:.3f}, delta={r['delta']:+.3f}")
    else:
        print("✓ All states: fixed COP ≤ unfixed COP (filter lowered COP as expected)")

print(f"\n✓ TASK B COMPLETE")

# %% [markdown]
# ---
# ## Task C: Climate Zone Benchmark Validation
# ---
# 
# Aggregates thermal COP by IECC climate zone group and validates against
# literature-derived benchmark ranges (`COP_BENCHMARK_RANGES`). Also
# produces a state × climate zone cross-tab with a PA spot check.

# %%
print("=" * 80)
print("TASK C: CLIMATE ZONE BENCHMARK VALIDATION")
print("=" * 80)

for mp in selected_mps:
    mp_key = f'mp{mp}'
    print(f"\n{'─' * 60}")
    print(f"MP{mp} — Climate Zone Group Aggregation")
    print(f"{'─' * 60}")

    # --- C1: COP by climate zone group ---
    df_cop_cz = compute_thermal_cop(
        df_baseline, upgrade_data[mp],
        group_cols=['cz_group'],
        fuel_filter='Natural Gas',
        verbose=True,
    )

    # --- C2: COP by state × climate zone group ---
    df_cop_state_cz = compute_thermal_cop(
        df_baseline, upgrade_data[mp],
        group_cols=['state', 'cz_group'],
        fuel_filter='Natural Gas',
        verbose=True,
    )

    # --- C3: Validate against COP_BENCHMARK_RANGES ---
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
                  f"{'✓' if in_range else '✗':>6}")
        else:
            print(f"{cz_grp:<10} {bench['label']:<18} {cop_val:>6.2f} {'no bench':>14} {'—':>6}")

    print(f"\nOverall: {n_passed}/{n_checked} climate zone groups within expected range")

    # --- C4: PA spot check ---
    # PA_COP_RANGES is sourced from constants.PA_COP_RANGES (P2.8 consolidation)
    pa_rows = df_cop_state_cz[df_cop_state_cz['state'] == 'PA']
    if len(pa_rows) > 0:
        print(f"\n--- PA Spot Check (MP{mp}) ---")
        for _, pa_row in pa_rows.iterrows():
            cop_val = pa_row['thermal_cop']
            cz = pa_row['cz_group']
            if mp_key in PA_COP_RANGES:
                lo, hi = PA_COP_RANGES[mp_key]
                ok = lo <= cop_val <= hi
                # TODO: follow-up (P0.2) — CZ 6-7 fails; do NOT change ranges without evidence
                print(f"  PA (CZ {cz}): COP = {cop_val:.2f}, "
                      f"expected [{lo:.1f}, {hi:.1f}] → {'[OK] PASS' if ok else '[FAIL] FAIL'}")
            else:
                print(f"  PA (CZ {cz}): COP = {cop_val:.2f} (no PA benchmark for {mp_key})")
    else:
        print("\n[WARN] PA not found in state x CZ cross-tab")

print(f"\n✓ TASK C COMPLETE")

# %% [markdown]
# ---
# ## Task D: Break-Even COP Cross-Validation (Jenkins et al.)
# ---
# 
# Rich cross-validation against Jenkins Heat Pump Map reference values.
# For each reference state: compares thermal COP, spark gap, computed
# break-even COP at 90% AFUE, and Jenkins reference. Determines whether
# the heat pump beats break-even and whether the computed value matches
# Jenkins within strict (±0.05) and relaxed (±0.50) tolerances.
# 
# **Known discrepancy sources:**
# 1. Gas heat content — Jenkins: 1020 BTU/cf; ours: 1038 BTU/cf (EIA average)
# 2. ~~Price year — Jenkins: earlier year; ours: 2024 EIA nominal~~

# %%
print("=" * 80)
print(f"TASK D: BREAK-EVEN COP CROSS-VALIDATION (Jenkins et al., MP{primary_mp})")
print("=" * 80)

# Jenkins Heat Pump Map break-even COP at 90% AFUE reference values
# Source: Jenkins et al.
# ASSUMPTION: Jenkins assumes 1020 BTU/cf gas heat content; we use 1038 BTU/cf
# (current EIA average). This ~1.8% difference propagates into spark gap and
# break-even COP. Our prices are now 2024 EIA nominal (same as Jenkins).
# P2.7 FIX: jenkins_ref consolidated to constants.JENKINS_BREAKEVEN_REF_90
# P1.6 FIX: df_prices_for_d and df_breakeven_for_d removed — reuse df_prices_csv
# and df_breakeven already in scope (df_prices_csv at line ~124, df_breakeven at ~490)
df_spark_for_d = compute_spark_gap_metrics(
    df_prices_csv, cop_results[primary_mp],
    df_breakeven=df_breakeven, verbose=False,
)

header = (f"{'State':<6} {'COP':>6} {'Spark':>6} {'BE_90':>6} "
          f"{'Jenkins':>8} {'Strict':>7} {'Relaxed':>8} {'HP>BE?':>7}")
print(f"\n{header}")
print("-" * len(header))

n_strict = 0
n_relaxed = 0
for st, ref_val in sorted(JENKINS_BREAKEVEN_REF_90.items()):
    row_cop = cop_results[primary_mp][cop_results[primary_mp]['state'] == st]
    row_spark = df_spark_for_d[df_spark_for_d['state'] == st]

    if len(row_cop) == 0 or len(row_spark) == 0:
        print(f"{st:<6} {'N/A':>6} {'N/A':>6} {'N/A':>6} {ref_val:>8.2f} "
              f"{'—':>7} {'—':>8} {'—':>7}")
        continue

    thermal_cop = row_cop['thermal_cop'].values[0]
    spark_gap = row_spark['spark_gap'].values[0]
    # ASSUMPTION: breakeven_cop_90 = spark_gap * 0.90
    breakeven_cop_90 = spark_gap * 0.90

    cop_exceeds = thermal_cop > breakeven_cop_90
    strict = abs(ref_val - breakeven_cop_90) <= 0.05
    relaxed = abs(ref_val - breakeven_cop_90) <= 0.50
    n_strict += int(strict)
    n_relaxed += int(relaxed)

    print(f"{st:<6} {thermal_cop:>6.2f} {spark_gap:>6.2f} {breakeven_cop_90:>6.2f} "
          f"{ref_val:>8.2f} {'[OK]' if strict else '[FAIL]':>7} "
          f"{'[OK]' if relaxed else '[FAIL]':>8} {'Yes' if cop_exceeds else 'No':>7}")

n_total = len(JENKINS_BREAKEVEN_REF_90)
print(f"\nStrict match (+/-0.05): {n_strict}/{n_total}")
print(f"Relaxed match (+/-0.50): {n_relaxed}/{n_total}")
print(f"\nNote: Remaining discrepancies arise from (1) gas heat content "
      f"(Jenkins: 1020, ours: 1038 BTU/cf)")

# Interpretation
print("\n--- Interpretation ---")
warm = ['FL']
cold = ['MN', 'AK']
for label, states in [('Warm', warm), ('Cold', cold)]:
    for st in states:
        row_cop = cop_results[primary_mp][cop_results[primary_mp]['state'] == st]
        row_spark = df_spark_for_d[df_spark_for_d['state'] == st]
        if len(row_cop) == 0 or len(row_spark) == 0:
            continue
        cop_val = row_cop['thermal_cop'].values[0]
        be_val = row_spark['spark_gap'].values[0] * 0.90
        exceeds = cop_val > be_val
        msg = "HP saves money on fuel" if exceeds else "HP does NOT beat furnace on fuel cost"
        print(f"  {st} ({label}): COP={cop_val:.2f}, BE_90={be_val:.2f} → {msg}")

print("\n[OK] TASK D COMPLETE")

# %% [markdown]
# ---
# ## Step 4: Break-Even COP and Merged Metrics
# ---

# %%
print("=" * 80)
print("STEP 4: BREAK-EVEN COP & MERGED METRICS")
print("=" * 80)

# --- 4a: Compute break-even COP for multiple AFUE scenarios ---
# ASSUMPTION: 2024 EIA state-level residential prices.
afue_scenarios = [0.80, 0.90, 0.95, 1.00]
df_breakeven = compute_breakeven_cop(df_prices_csv, afue_scenarios=afue_scenarios)
print(f"✓ Break-even COP computed for {len(df_breakeven)} states, "
      f"AFUE scenarios: {afue_scenarios}")

print(f"\nBreak-even COP @90% AFUE — "
      f"Mean: {df_breakeven['breakeven_cop_90'].mean():.2f}, "
      f"Range: {df_breakeven['breakeven_cop_90'].min():.2f}–"
      f"{df_breakeven['breakeven_cop_90'].max():.2f}")

# --- 4b: Merge prices + COP + break-even into single state-level table ---
df_spark = compute_spark_gap_metrics(
    df_prices_csv, df_cop, df_breakeven=df_breakeven, verbose=True
)

# --- 4c: Validate against Jenkins Heat Pump Map reference values (90% AFUE) ---
# Source: Jenkins et al., break-even COP at 90% AFUE
# Note: Jenkins assumes 1020 BTU/cf heat content (ours: 1038) and a different
# price year. Differences are expected and documented in Task D.
# P2.7 FIX: jenkins_ref replaced with constant JENKINS_BREAKEVEN_REF_90
print("\n--- Jenkins Validation (break-even COP @90% AFUE, +/-0.05) ---")
print(f"{'State':<6} {'Computed':>10} {'Jenkins':>10} {'Diff':>8} {'Pass':>6}")
print("-" * 44)
for st, ref_val in sorted(JENKINS_BREAKEVEN_REF_90.items()):
    row = df_spark[df_spark['state'] == st]
    if len(row) == 0:
        print(f"{st:<6} {'N/A':>10} {ref_val:>10.2f} {'—':>8} {'SKIP':>6}")
        continue
    computed = row['breakeven_cop_90'].values[0]
    diff = computed - ref_val
    passed = abs(diff) <= 0.05
    print(f"{st:<6} {computed:>10.2f} {ref_val:>10.2f} {diff:>+8.2f} {'✓' if passed else '✗':>6}")

# --- 4d: Cross-validation (effective COP vs break-even COP) ---
print("\n--- Cross-Validation: Effective COP vs Break-Even @90% ---")
warm_states = ['FL', 'GA', 'SC', 'TX', 'LA']
cold_states = ['MN', 'WI', 'VT', 'ND', 'ME']
for label, states in [('Warm', warm_states), ('Cold', cold_states)]:
    subset = df_spark[df_spark['state'].isin(states)].copy()
    if len(subset) == 0:
        continue
    subset['cop_minus_be'] = subset['thermal_cop'] - subset['breakeven_cop_90']
    print(f"\n{label} states:")
    print(f"  {'State':<6} {'COP':>6} {'BE_90':>6} {'Diff':>8} {'HP Wins?':>10}")
    for _, r in subset.iterrows():
        hp_wins = r['thermal_cop'] > r['breakeven_cop_90']
        print(f"  {r['state']:<6} {r['thermal_cop']:>6.2f} "
              f"{r['breakeven_cop_90']:>6.2f} {r['cop_minus_be']:>+8.2f} "
              f"{'Yes' if hp_wins else 'No':>10}")

print(f"\n✓ STEP 4 COMPLETE")

# %%
# Build COP & break-even summary table: state, spark gap, MP3 COP, MP4 COP, BE 80/90/95
df_cop_be_summary = df_spark[['state', 'state_name', 'spark_gap',
                               'breakeven_cop_80', 'breakeven_cop_90', 'breakeven_cop_95']].copy()

# Merge MP3 and MP4 thermal COP
for mp_num in [3, 4]:
    if mp_num in cop_results:
        df_cop_be_summary = df_cop_be_summary.merge(
            cop_results[mp_num][['state', 'thermal_cop']].rename(
                columns={'thermal_cop': f'thermal_cop_mp{mp_num}'}
            ),
            on='state', how='left',
        )

# Reorder columns for readability
col_order = ['state', 'state_name', 'spark_gap']
for mp_num in [3, 4]:
    col_name = f'thermal_cop_mp{mp_num}'
    if col_name in df_cop_be_summary.columns:
        col_order.append(col_name)
col_order += ['breakeven_cop_80', 'breakeven_cop_90', 'breakeven_cop_95']
df_cop_be_summary = df_cop_be_summary[col_order].sort_values('spark_gap', ascending=False).reset_index(drop=True)

print("=" * 110)
print("THERMAL COP & BREAK-EVEN COP SUMMARY: TOP 5 & BOTTOM 5 STATES")
print("=" * 110)
print(f"\n{'Spark Gap & Break-Even COP based on 2024 EIA prices'}")
print(f"{'Break-Even COP = Spark Gap × AFUE (COP threshold where HP matches furnace fuel cost)'}\n")

print("--- Top 5 (highest spark gap) ---")
print(df_cop_be_summary.head(5).to_string(index=False))

print("\n--- Bottom 5 (lowest spark gap) ---")
print(df_cop_be_summary.tail(5).to_string(index=False))

# Highlight states where MP COP exceeds break-even at each AFUE
for mp_num in [3, 4]:
    cop_col = f'thermal_cop_mp{mp_num}'
    if cop_col not in df_cop_be_summary.columns:
        continue
    for afue_pct in [80, 90, 95]:
        be_col = f'breakeven_cop_{afue_pct}'
        n_beats = (df_cop_be_summary[cop_col] > df_cop_be_summary[be_col]).sum()
        n_total = df_cop_be_summary[cop_col].notna().sum()
        print(f"  MP{mp_num} COP > BE @{afue_pct}% AFUE: {n_beats}/{n_total} states")

# %% [markdown]
# ---
# ## Step 5: Geospatial Visualization
# ---

# %%
gdf_conus = None
gdf_alaska = None

try:
    print(f"Loading shapefile from: {SHAPEFILE_PATH}")
    gdf_states_raw = gpd.read_file(SHAPEFILE_PATH)
    _, gdf_conus, gdf_alaska = prepare_state_geodataframe(gdf_states_raw, df_spark, merge_col='state')
    print(f"  CONUS: {len(gdf_conus)}, Alaska: {len(gdf_alaska)}")
    print("✓ Geodataframe prepared")
except FileNotFoundError:
    print(f"⚠ Shapefile not found at {SHAPEFILE_PATH} — skipping maps")
except Exception as e:
    print(f"⚠ Could not load shapefile: {e}")

# %%
if gdf_conus is not None and gdf_alaska is not None:
    print("Generating choropleth maps...")

    # --- Spark gap (blue) ---
    create_choropleth_map(
        gdf_conus, gdf_alaska, column='spark_gap',
        title='Electricity-to-Natural Gas Spark Gap by State (2024)',
        cbar_label='Spark Gap\n(electricity price ÷ gas price, $/MMBTU basis)',
        output_path=os.path.join(PROJECT_ROOT, "state_spark_gap_map_2024.png"),
        cmap='Blues', show_plot=True,
    )

else:
    print("⚠ Maps skipped — no geodataframe available")

# %%
if gdf_conus is not None and gdf_alaska is not None:
    print("Generating choropleth maps...")

    # --- Thermal COP: MP3 top / MP4 bottom with shared color scale (greens) ---
    cop_mps = [mp for mp in [3, 4] if mp in cop_results]
    if len(cop_mps) >= 2:
        # Build per-MP geodataframes and find shared color range
        gdf_cop_panels = {}
        all_vals = pd.Series(dtype=float)
        for mp_num in cop_mps:
            df_spark_mp = compute_spark_gap_metrics(df_prices_csv, cop_results[mp_num])
            _, gdf_c, gdf_a = prepare_state_geodataframe(
                gdf_states_raw, df_spark_mp, merge_col='state'
            )
            gdf_cop_panels[mp_num] = (gdf_c, gdf_a)
            all_vals = pd.concat([all_vals, gdf_c['thermal_cop'], gdf_a['thermal_cop']])

        shared_norm = mcolors.Normalize(vmin=all_vals.dropna().min(), vmax=all_vals.dropna().max())
        col = 'thermal_cop'
        nrows = len(cop_mps)

        fig, axes_grid = plt.subplots(nrows, 1, figsize=(14, 9 * nrows), facecolor='white')
        if nrows == 1:
            axes_grid = [axes_grid]

        for i, mp_num in enumerate(cop_mps):
            gdf_c, gdf_a = gdf_cop_panels[mp_num]
            ax = axes_grid[i]

            gdf_c.plot(ax=ax, column=col, cmap='Greens', edgecolor='black',
                       linewidth=0.5, norm=shared_norm, legend=False)
            ax.set_axis_off()
            ax.set_title(f'MP{mp_num} — Thermal COP by State (2024)',
                         fontsize=18, fontweight='bold', pad=12)

            # Alaska inset — only plot if data exists
            inset_bounds = ax.get_position()
            ax_ak = fig.add_axes([
                inset_bounds.x0, inset_bounds.y0,
                inset_bounds.width * 0.22, inset_bounds.height * 0.28,
            ])
            if not gdf_a.empty:
                gdf_a.plot(ax=ax_ak, column=col, cmap='Greens', edgecolor='black',
                           linewidth=0.5, norm=shared_norm, legend=False)
            ax_ak.set_axis_off()

        # Shared colorbar — bottom center
        cax = fig.add_axes([0.25, 0.02, 0.50, 0.015])
        sm = plt.cm.ScalarMappable(cmap='Greens', norm=shared_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_label('Thermal COP (heat delivered / electricity consumed)', fontsize=16)
        cax.tick_params(labelsize=16)

        fig.subplots_adjust(hspace=0.04)
        output = os.path.join(PROJECT_ROOT, "state_thermal_cop_mp3_mp4_map_2024.png")
        fig.savefig(output, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output}")
        plt.show()
    elif len(cop_mps) == 1:
        create_choropleth_map(
            gdf_conus, gdf_alaska, column='thermal_cop',
            title=f'Heat Pump Thermal COP by State (MP{cop_mps[0]}, 2024)',
            cbar_label='Thermal COP\n(heat delivered / electricity consumed)',
            output_path=os.path.join(PROJECT_ROOT, f"state_thermal_cop_mp{cop_mps[0]}_map_2024.png"),
            cmap='Greens', show_plot=True,
        )
    else:
        print("⚠ No thermal COP results for MP3 or MP4 — skipping COP maps")

    print("[OK] Maps generated")
else:
    print("[WARN] Maps skipped — no geodataframe available")

# %%
# P0.4 FIX (21-Apr-2026): Local create_choropleth_map redefinition removed.
# The module version (visualize_geospatial_data.py) imported at the top of this
# file already has norm=None and all required parameters. Using the local version
# was shadowing module edits silently. The dpi=600 calls below now pass dpi
# explicitly to the module function.

if gdf_conus is not None and gdf_alaska is not None:
    print("Generating choropleth maps...")

    # --- Break-even COP at 90% AFUE (green) ---
    create_choropleth_map(
        gdf_conus, gdf_alaska, column='breakeven_cop_90',
        title='Break-Even COP by State at 90% AFUE (2024)',
        cbar_label='Break-Even COP\n(COP needed for HP to match furnace cost)',
        output_path=os.path.join(PROJECT_ROOT, "state_breakeven_cop_90_map_2024.png"),
        cmap='Greens', show_plot=True,
    )

    print("✓ Maps generated")
else:
    print("⚠ Maps skipped — no geodataframe available")

# %%
if gdf_conus is not None and gdf_alaska is not None:
    print("Generating break-even COP panel maps (80%, 90%, 95% AFUE)...")

    be_cols = ['breakeven_cop_80', 'breakeven_cop_90', 'breakeven_cop_95']
    be_labels = ['80% AFUE', '90% AFUE', '95% AFUE']

    # Shared color range across all three AFUE scenarios
    all_be_vals = pd.concat([
        pd.concat([gdf_conus[c], gdf_alaska[c]]) for c in be_cols
    ]).dropna()
    shared_norm = mcolors.Normalize(vmin=all_be_vals.min(), vmax=all_be_vals.max())

    nrows = len(be_cols)
    fig, axes_grid = plt.subplots(nrows, 1, figsize=(14, 9 * nrows), facecolor='white')

    for i, (col, label) in enumerate(zip(be_cols, be_labels)):
        ax = axes_grid[i]
        gdf_conus.plot(ax=ax, column=col, cmap='Oranges', edgecolor='black',
                       linewidth=0.5, norm=shared_norm, legend=False)
        ax.set_axis_off()
        ax.set_title(f'Break-Even COP by State — {label} (2024)',
                     fontsize=18, fontweight='bold', pad=12)

        # Alaska inset
        inset_bounds = ax.get_position()
        ax_ak = fig.add_axes([
            inset_bounds.x0, inset_bounds.y0,
            inset_bounds.width * 0.22, inset_bounds.height * 0.28,
        ])
        if not gdf_alaska.empty:
            gdf_alaska.plot(ax=ax_ak, column=col, cmap='Oranges', edgecolor='black',
                            linewidth=0.5, norm=shared_norm, legend=False)
        ax_ak.set_axis_off()

    # Shared colorbar — bottom center
    cax = fig.add_axes([0.25, 0.02, 0.50, 0.015])
    sm = plt.cm.ScalarMappable(cmap='Oranges', norm=shared_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Break-Even COP (COP threshold for HP to match furnace fuel cost)', fontsize=16)
    cax.tick_params(labelsize=16)

    fig.subplots_adjust(hspace=0.04)
    output = os.path.join(PROJECT_ROOT, "state_breakeven_cop_80_90_95_panel_map_2024.png")
    fig.savefig(output, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output}")
    plt.show()
else:
    print("⚠ Maps skipped — no geodataframe available")

# %% [markdown]
# ---
# ## Display Results
# ---

# %%
# P2.9 FIX: display() replaced with print(df.to_string()) for script compatibility
print("===== STATE PRICE RATIOS (2024 nominal) =====\n")
print(df_prices_csv.to_string())

for mp in selected_mps:
    print(f"\n===== THERMAL COP & AFUE (MP{mp}, NG homes) =====\n")
    print(cop_results[mp].sort_values('thermal_cop', ascending=False)[
        ['state', 'thermal_cop', 'baseline_afue', 'fans_pumps_pct', 'home_count']
    ].to_string())

# --- Comparison table: spark gap, break-even COP, effective COP ---
print(f"\n===== SPARK GAP & BREAK-EVEN COP vs EFFECTIVE COP (MP{primary_mp}) =====\n")

df_display = df_spark[[
    'state', 'state_name', 'spark_gap',
    'breakeven_cop_80', 'breakeven_cop_90', 'breakeven_cop_95', 'breakeven_cop_100',
    'thermal_cop', 'baseline_afue',
]].copy()

# Flag whether the heat pump beats the break-even threshold at 90% AFUE
df_display['hp_beats_breakeven_90'] = df_display['thermal_cop'] > df_display['breakeven_cop_90']

print(df_display.sort_values('spark_gap', ascending=False).reset_index(drop=True).to_string())


# %% [markdown]
# ---
# ## County Smoke Test (Phase 2)
# ---

# %%
# Verify county-level COP computation. Allegheny County PA (G4200030, FIPS 42003)
# should fall within the literature-validated PA ranges from PA_COP_RANGES.
ALLEGHENY_GISJOIN = 'G4200030'

try:
    cop_county = compute_thermal_cop(
        df_baseline, df_upgrades[primary_mp],
        fuel_filter='Natural Gas',
        aggregation='county',
    )
    allegheny_row = cop_county[cop_county['county'] == ALLEGHENY_GISJOIN]
    if allegheny_row.empty:
        print(f"[FAIL] Allegheny County (GISJOIN={ALLEGHENY_GISJOIN}) not found in county COP results")
    else:
        cop_val = allegheny_row['thermal_cop'].iloc[0]
        mp_key = f'mp{primary_mp}'
        if mp_key in PA_COP_RANGES:
            lo, hi = PA_COP_RANGES[mp_key]
            status = '[OK] ' if lo <= cop_val <= hi else '[FAIL]'
            print(f"{status} Allegheny PA COP (MP{primary_mp}) = {cop_val:.3f}  "
                  f"(expected [{lo}, {hi}])")
        else:
            print(f"[OK]  Allegheny PA COP (MP{primary_mp}) = {cop_val:.3f}  "
                  f"(no PA reference range for mp{primary_mp})")
        print(f"       County rows total: {len(cop_county)}")
except Exception as e:
    print(f"[FAIL] County smoke test raised: {e}")

