# %% [markdown]
# # Post-TARE Adoption KPIs: Bill Savings & Demand Change
# **Author:** Jordan M. Joseph, PhD â€” Carnegie Mellon University
# 
# Computes bill savings ratio and electricity demand change metrics from TARE model outputs and EUSS data.

# %%
import os
import importlib
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from config import PROJECT_ROOT
from cmu_tare_model.constants import VALID_MENU_MPS
from cmu_tare_model.utils.load_exported_results_to_df import load_measure_package_data

from cmu_tare_model.adoption_kpis import (
    load_euss_baseline, load_euss_upgrade, mp_to_upgrade,
    compute_scenario_demand, aggregate_demand,
    compute_bill_savings_ratio, aggregate_bill_savings,
    compute_adoption_rate,
)
from cmu_tare_model.grid_impact.peak_load_functions import find_adoption_column

from cmu_tare_model.adoption_kpis.visualize_geospatial_data import (
    prepare_state_geodataframe, plot_combined_choropleth,
)
from cmu_tare_model.adoption_kpis.data_loading import (
    SHAPEFILE_PATH, COUNTY_SHAPEFILE_PATH,
)

# Adoption potential: multi-index DataFrame and dotplot
from cmu_tare_model.adoption_potential.data_processing.visuals_adoption_potential import (
    build_adoption_scenario_names,
    create_multiIndex_adoption_df,
    print_adoption_decision_percentages,
    subplot_grid_adoption_vBar
)

from cmu_tare_model.adoption_potential.data_processing.visuals_adoption_dotplot import (
    prepare_plot_data,
    plot_adoption_panel,
    _build_legend_handles,
    GROUPING_ORDER,
)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 60)
SAVE_FIGURES = False  # Set to True to save figure files to disk

print("[OK] Imports loaded")


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

print(f"Selected measure packages: {selected_mps}")

# %%
# Load TARE model outputs for selected measure packages.
# These contain per-building lifetime fuel costs used for bill savings ratio.

try:
    _ = DATAFRAMES_BY_MP
    print(f"DATAFRAMES_BY_MP already loaded: {list(DATAFRAMES_BY_MP.keys())}")
except NameError:
    try:
        _ = output_folder_path
        print(f"Using existing output_folder_path: {output_folder_path}")
    except NameError:
        output_folder_path = os.path.join(PROJECT_ROOT, "cmu_tare_model", "output_results")
        location_id = input("Enter location ID (e.g., 'National' or 'PA'): ").strip()
        model_run_date_time = input("Enter model run timestamp (YYYY-MM-DD_HH-MM): ").strip()

    DATAFRAMES_BY_MP = {}
    for mp in selected_mps:
        DATAFRAMES_BY_MP[mp] = load_measure_package_data(
            mp, output_folder_path, location_id, model_run_date_time
        )
    print(f"[OK] Loaded TARE data for MPs: {list(DATAFRAMES_BY_MP.keys())}")

# %%
df_baseline = load_euss_baseline()
print(f"Baseline: {len(df_baseline):,} occupied SF homes")

upgrade_data = {}
for mp in selected_mps:
    upgrade_name = mp_to_upgrade(mp)
    print(f"\nLoading MP{mp} ({upgrade_name})...")
    upgrade_data[mp] = load_euss_upgrade(upgrade_name)
    print(f"  MP{mp}: {len(upgrade_data[mp]):,} applicable homes")

print(f"\n[OK] EUSS data loaded")

# %%
bill_savings_geo_level = 'county'

bill_savings_results = {}
for mp in selected_mps:
    print(f"\n{'='*60}")
    print(f"BILL SAVINGS RATIO — MP{mp}, all fuels")
    print(f"{'='*60}")
    df_tare = DATAFRAMES_BY_MP[mp]['fixed_base']['inmap']
    df_ratio = compute_bill_savings_ratio(
        df_tare, mp=mp, policy_scenario='iraRef',
        fuel_filter=None, verbose=True,
        df_euss=df_baseline,
    )
    df_bill_savings_county = aggregate_bill_savings(
        df_ratio, geo_level=bill_savings_geo_level, verbose=True
    )
    bill_savings_results[mp] = df_bill_savings_county

print(f"\n[OK] Bill savings ratio complete ({bill_savings_geo_level}-level)")


# %%
if bill_savings_geo_level == 'state':
    for mp, df_state in bill_savings_results.items():
        print(f"\n===== BILL SAVINGS RATIO (MP{mp}, NG homes) =====")
        n_savings = (df_state['median_bill_savings_ratio'] < 1.0).sum()
        print(f"States where median home saves money: {n_savings} / {len(df_state)}")
        print(f"National median: {df_state['median_bill_savings_ratio'].median():.3f}")
        print(f"\nTop 5 (best for electrification):")
        print(df_state.nsmallest(5, 'median_bill_savings_ratio')[
            ['state', 'median_bill_savings_ratio', 'pct_bill_change', 'home_count']
        ].to_string(index=False))
        print(f"\nBottom 5 (worst for electrification):")
        print(df_state.nlargest(5, 'median_bill_savings_ratio')[
            ['state', 'median_bill_savings_ratio', 'pct_bill_change', 'home_count']
        ].to_string(index=False))


# %%
demand_geo_level = 'county'

demand_results = {}
for mp in selected_mps:
    print(f"\n{'='*60}")
    print(f"DEMAND CHANGE — MP{mp}, all fuels, 100% adoption")
    print(f"{'='*60}")
    df_demand = compute_scenario_demand(
        df_baseline, upgrade_data[mp], fuel_filter=None, verbose=True,
    )
    df_demand_county = aggregate_demand(
        df_demand, geo_level=demand_geo_level, verbose=True
    )
    demand_results[mp] = df_demand_county

print(f"\n[OK] Demand change complete ({demand_geo_level}-level)")


# %%
# ============================================================
# TASK 1 DIAGNOSTICS — County sample size & coverage
# ============================================================

print("=" * 65)
print("DIAGNOSTIC: Homes entering aggregation")
print("=" * 65)
print(f"fuel_filter=None confirmed: {len(df_ratio):,} homes "
      f"(expected ~331K, NOT ~181K)")
print(f"Unique counties (bill savings): {df_ratio['in.county'].nunique():,}")
print(f"Homes entering demand:          {len(df_demand):,}")
print(f"Unique counties (demand):       {df_demand['in.county'].nunique():,}")

print("\n--- County sample size distribution (demand, last MP) ---")
county_counts = df_demand.groupby('in.county').size()
print(f"Counties total:             {len(county_counts):,}")
print(f"Min / Median / Mean / Max:  "
      f"{county_counts.min()} / {county_counts.median():.0f} / "
      f"{county_counts.mean():.1f} / {county_counts.max()}")
for thresh in [5, 10, 15, 20, 30]:
    below = (county_counts < thresh).sum()
    above = (county_counts >= thresh).sum()
    print(f"  below {thresh:2d}: {below:4d}   |   at or above {thresh:2d}: {above:4d}")

print("\n--- Aggregated output coverage ---")
from cmu_tare_model.adoption_kpis import aggregate_bill_savings, aggregate_demand
for t in [5, 10, 15, 20, 30]:
    d = aggregate_demand(df_demand, geo_level='county', min_home_count=t)
    n_data = d['pct_elec_demand_change'].notna().sum()
    print(f"  min_home_count={t:2d}: {n_data:4d} counties with data")


# %%
if demand_geo_level == 'state':
    for mp, df_state in demand_results.items():
        print(f"\n===== DEMAND CHANGE (MP{mp}, GWh, all fuels, 100% adoption) =====")
        print(f"Total elec demand change: {df_state['elec_change_gwh'].sum():+.1f} GWh")
        print(f"Total site energy change: {df_state['site_energy_change_gwh'].sum():+.1f} GWh")
        print(f"\nTop 5 (largest elec demand increase):")
        print(df_state[['state', 'elec_change_gwh', 'pct_elec_demand_change',
                        'site_energy_change_gwh', 'pct_site_energy_change']].head(5).to_string(index=False))
        print(f"\nBottom 5:")
        print(df_state[['state', 'elec_change_gwh', 'pct_elec_demand_change',
                        'site_energy_change_gwh', 'pct_site_energy_change']].tail(5).to_string(index=False))

# %%
from matplotlib.colors import Normalize

gdf_states_raw = None
gdf_counties_raw = None

try:
    gdf_states_raw = gpd.read_file(SHAPEFILE_PATH)
    print(f"[OK] State shapefile loaded: {len(gdf_states_raw)} features")
except Exception as e:
    print(f"[WARN] State shapefile not loaded: {e}")

try:
    gdf_counties_raw = gpd.read_file(COUNTY_SHAPEFILE_PATH)
    print(f"[OK] County shapefile loaded: {len(gdf_counties_raw)} features")
except Exception as e:
    print(f"[WARN] County shapefile not loaded: {e} — skipping county maps")

if gdf_counties_raw is not None:
    # pct_bill_change is now computed in aggregate_bill_savings() as:
    #   (median_bill_savings_ratio - 1) × 100
    # No post-hoc derivation needed here.

    # ---- Compute shared norms BEFORE any per-MP map calls ----
    # Symmetric Normalize: keeps white at the meaningful center (0 or 1.0)
    # and makes the colorbar evenly spaced on both sides.
    # Clip to 2nd/98th percentile to avoid extreme outliers compressing the scale.

    # 1. Bill savings ratio — center at 1.0
    all_ratio_vals = pd.concat([
        bill_savings_results[mp]['median_bill_savings_ratio']
        for mp in selected_mps
    ]).dropna()
    _q_low = all_ratio_vals.quantile(0.02)
    _q_high = all_ratio_vals.quantile(0.98)
    _dev = max(abs(_q_low - 1.0), abs(_q_high - 1.0))
    shared_ratio_norm = Normalize(vmin=1.0 - _dev, vmax=1.0 + _dev)
    print(f"Bill savings ratio norm: [{1.0 - _dev:.3f}, 1.0 (center), {1.0 + _dev:.3f}]")

    # 2. Bill savings percent change — center at 0
    # Derived from median ratio so spatial pattern is identical to ratio map.
    all_pct_bill_vals = pd.concat([
        bill_savings_results[mp]['pct_bill_change']
        for mp in selected_mps
    ]).dropna()
    _q_low = all_pct_bill_vals.quantile(0.02)
    _q_high = all_pct_bill_vals.quantile(0.98)
    _abs = max(abs(_q_low), abs(_q_high))
    shared_pct_bill_norm = Normalize(vmin=-_abs, vmax=_abs)
    print(f"Bill savings % norm: [{-_abs:.1f}, 0 (center), {_abs:.1f}]%")

    # 3. Electricity demand change GWh — center at 0
    all_demand_vals = pd.concat([
        demand_results[mp]['elec_change_gwh']
        for mp in selected_mps
    ]).dropna()
    _q_low = all_demand_vals.quantile(0.02)
    _q_high = all_demand_vals.quantile(0.98)
    _abs = max(abs(_q_low), abs(_q_high))
    shared_demand_norm = Normalize(vmin=-_abs, vmax=_abs)
    print(f"Demand GWh norm: [{-_abs:.1f}, 0 (center), {_abs:.1f}] GWh")

    # 4. Electricity demand percent change — center at 0
    all_pct_demand_vals = pd.concat([
        demand_results[mp]['pct_elec_demand_change']
        for mp in selected_mps
    ]).dropna()
    _q_low = all_pct_demand_vals.quantile(0.02)
    _q_high = all_pct_demand_vals.quantile(0.98)
    _abs = max(abs(_q_low), abs(_q_high))
    shared_pct_demand_norm = Normalize(vmin=-_abs, vmax=_abs)
    print(f"Demand % norm: [{-_abs:.1f}, 0 (center), {_abs:.1f}]%")

    # ---- Map 1: Bill savings ratio (county-level) ----
    print("\n--- Summary: median_bill_savings_ratio ---")
    for mp in selected_mps:
        _v = bill_savings_results[mp]['median_bill_savings_ratio'].dropna()
        _pct = (_v < 1.0).mean() * 100
        print(f"  MP{mp}: n={len(_v):,} counties | "
              f"min={_v.min():.3f} | med={_v.median():.3f} | "
              f"mean={_v.mean():.3f} | max={_v.max():.3f} | "
              f"{_pct:.1f}% of counties HP saves money (ratio < 1)")
        
    plot_combined_choropleth(
        gdf_counties_raw, bill_savings_results,
        column='median_bill_savings_ratio',
        title_template='Median Bill Savings Ratio — MP{mp}\n(< 1 = HP saves money)',
        cbar_label='Bill Savings Ratio (retrofit cost / baseline cost)',
        cmap='RdBu_r', norm=shared_ratio_norm, selected_mps=selected_mps,
        geo_level='county',
        save_figure=SAVE_FIGURES,
        output_path=os.path.join(PROJECT_ROOT, 'county_bill_savings_ratio_combined.png'),
    )

    # ---- Map 2: Bill savings percent change (county-level) ----
    # pct_bill_change = (median_bill_savings_ratio - 1) × 100
    # Spatial pattern is identical to Map 1; only the scale changes.
    print("\n--- Summary: pct_bill_change ---")
    for mp in selected_mps:
        _v = bill_savings_results[mp]['pct_bill_change'].dropna()
        _pct = (_v < 0).mean() * 100
        print(f"  MP{mp}: n={len(_v):,} counties | "
              f"min={_v.min():.1f}% | med={_v.median():.1f}% | "
              f"mean={_v.mean():.1f}% | max={_v.max():.1f}% | "
              f"{_pct:.1f}% of counties HP saves money (< 0%)")
        
    plot_combined_choropleth(
        gdf_counties_raw, bill_savings_results,
        column='pct_bill_change',
        title_template='Heating Bill Change vs Baseline — MP{mp}\n(negative = HP saves money)',
        cbar_label='Median Bill Change (%)',
        cmap='RdBu_r', norm=shared_pct_bill_norm, selected_mps=selected_mps,
        geo_level='county',
        save_figure=SAVE_FIGURES,
        output_path=os.path.join(PROJECT_ROOT, 'county_bill_pct_change_combined.png'),
    )

    # ---- Map 3: Electricity demand change GWh (county-level) ----
    print("\n--- Summary: elec_change_gwh ---")
    for mp in selected_mps:
        _v = demand_results[mp]['elec_change_gwh'].dropna()
        _total = demand_results[mp]['elec_change_gwh'].sum()
        _pct = (_v > 0).mean() * 100
        print(f"  MP{mp}: n={len(_v):,} counties | "
              f"min={_v.min():.1f} | med={_v.median():.1f} | "
              f"mean={_v.mean():.1f} | max={_v.max():.1f} GWh | "
              f"total={_total:+.1f} GWh | {_pct:.1f}% of counties increase")
        
    plot_combined_choropleth(
        gdf_counties_raw, demand_results,
        column='elec_change_gwh',
        title_template='Electricity Demand Change — MP{mp}\n(100% HP adoption, all fuels)',
        cbar_label='Electricity Demand Change (GWh)',
        cmap='coolwarm', norm=shared_demand_norm, selected_mps=selected_mps,
        geo_level='county',
        save_figure=SAVE_FIGURES,
        output_path=os.path.join(PROJECT_ROOT, 'county_elec_demand_gwh_combined.png'),
    )

    # ---- Map 4: Electricity demand percent change (county-level) ----
    print("\n--- Summary: pct_elec_demand_change ---")
    for mp in selected_mps:
        _v = demand_results[mp]['pct_elec_demand_change'].dropna()
        _pct = (_v > 0).mean() * 100
        print(f"  MP{mp}: n={len(_v):,} counties | "
              f"min={_v.min():.1f}% | med={_v.median():.1f}% | "
              f"mean={_v.mean():.1f}% | max={_v.max():.1f}% | "
              f"{_pct:.1f}% of counties increase demand")
        
    plot_combined_choropleth(
        gdf_counties_raw, demand_results,
        column='pct_elec_demand_change',
        title_template='Electricity Demand % Change — MP{mp}\n(100% HP adoption, all fuels)',
        cbar_label='Electricity Demand Change (%)',
        cmap='coolwarm', norm=shared_pct_demand_norm, selected_mps=selected_mps,
        geo_level='county',
        save_figure=SAVE_FIGURES,
        output_path=os.path.join(PROJECT_ROOT, 'county_elec_demand_pct_combined.png'),
    )

    print("[OK] All county-level maps generated")
else:
    print("[WARN] County maps skipped — county shapefile not available")


# %%
print("=" * 70)
print("SUMMARY")
print("=" * 70)

for mp in selected_mps:
    df_bs = bill_savings_results[mp]
    df_dem = demand_results[mp]
    n_savings = (df_bs['median_bill_savings_ratio'] < 1.0).sum()
    total_states = len(df_bs)
    nat_median_ratio = df_bs['median_bill_savings_ratio'].median()
    nat_median_pct = df_bs['pct_bill_change'].median()
    total_elec_gwh = df_dem['elec_change_gwh'].sum()
    median_pct_demand = df_dem['pct_elec_demand_change'].median()
    print(f"\n--- MP{mp} ---")
    print(f"Bill savings: {n_savings}/{total_states} states save money "
          f"| median ratio: {nat_median_ratio:.3f} "
          f"| median bill change: {nat_median_pct:+.1f}%")
    print(f"Demand change (all fuels): {total_elec_gwh:+.1f} GWh total "
          f"| median state % change: {median_pct_demand:+.1f}%")

print("\n[DONE]")


# %% [markdown]
# ---
# # ADOPTION POTENTIAL
# ---

# %% [markdown]
# ## Choropleth maps for county-level adoption rate

# %%
# ============================================================
# ADOPTION RATE — compute county-level adoption rate
# ============================================================
# Uses Tier 1 + Tier 2 buildings (total adoption potential).
# adoption_rate_pct = n_adopters / n_total × 100  (uniform weights cancel)
# min_home_count defaults to MIN_HOME_COUNT (constants.py) = 1 — all
# counties included; sparsely populated areas have fewer samples by nature.
# df_euss supplies 'weight' when absent from older TARE CSVs;
# county/state column aliases ('county' vs 'in.county') are auto-detected.

_ADOPTION_COST_SCENARIO = 'v4MID'   # REMDB base case
_ADOPTION_GEO_LEVEL = 'county'

adoption_rate_results = {}
for mp in selected_mps:
    print(f"\n{'='*60}")
    print(f"ADOPTION RATE — MP{mp} (Tier 1 + Tier 2, IRA-Ref)")
    print(f"{'='*60}")
    df_tare = DATAFRAMES_BY_MP[mp]['fixed_base']['inmap']
    adoption_col = find_adoption_column(
        df_tare, mp=mp,
        cost_scenario=_ADOPTION_COST_SCENARIO,
        discount_rate_key='fixed_base',
        rcm_model_key='inmap',
    )
    print(f"  Adoption column: {adoption_col}")
    df_adopt = compute_adoption_rate(
        df_tare,
        adoption_col=adoption_col,
        geo_level=_ADOPTION_GEO_LEVEL,
        df_euss=df_baseline,
        verbose=True,
    )
    adoption_rate_results[mp] = df_adopt

print(f"\n[OK] Adoption rate complete ({_ADOPTION_GEO_LEVEL}-level)")


# %%
if gdf_counties_raw is not None:
    from matplotlib.colors import Normalize

    # Sequential monochromatic colormap — adoption rate is always 0–100%
    _adopt_cmap = 'Greens'
    _adopt_norm = Normalize(vmin=0, vmax=100)
    _adopt_cbar_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    print("\n--- Summary: adoption_rate_pct ---")
    for mp in selected_mps:
        _v = adoption_rate_results[mp]['adoption_rate_pct'].dropna()
        _pct_high = (_v >= 50).mean() * 100
        print(f"  MP{mp}: n={len(_v):,} counties | "
              f"min={_v.min():.1f}% | med={_v.median():.1f}% | "
              f"mean={_v.mean():.1f}% | max={_v.max():.1f}% | "
              f"{_pct_high:.1f}% of counties ≥ 50% adoption")

    plot_combined_choropleth(
        gdf_counties_raw, adoption_rate_results,
        column='adoption_rate_pct',
        title_template='Adoption Potential — MP{mp}\n(Tier 1 + Tier 2, IRA-Ref)',
        cbar_label='Weighted Adoption Rate (%)',
        cmap=_adopt_cmap, norm=_adopt_norm,
        cbar_ticks=_adopt_cbar_ticks,
        selected_mps=selected_mps,
        geo_level='county',
        save_figure=SAVE_FIGURES,
        output_path=os.path.join(PROJECT_ROOT, 'county_adoption_rate_combined.png'),
    )

    print("[OK] Adoption choropleth generated")
else:
    print("[WARN] Adoption choropleth skipped — county shapefile not available")


# %%
if gdf_counties_raw is not None:
    from matplotlib.colors import Normalize

    # Sequential monochromatic colormap — adoption rate is always 0–100%
    _adopt_cmap = 'Greens'
    _adopt_norm = Normalize(vmin=0, vmax=100)
    # _adopt_cbar_ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    print("\n--- Summary: adoption_rate_pct ---")
    for mp in selected_mps:
        _v = adoption_rate_results[mp]['adoption_rate_pct'].dropna()
        _pct_high = (_v >= 50).mean() * 100
        print(f"  MP{mp}: n={len(_v):,} counties | "
              f"min={_v.min():.1f}% | med={_v.median():.1f}% | "
              f"mean={_v.mean():.1f}% | max={_v.max():.1f}% | "
              f"{_pct_high:.1f}% of counties ≥ 50% adoption")

    plot_combined_choropleth(
        gdf_counties_raw, adoption_rate_results,
        column='adoption_rate_pct',
        title_template='Adoption Potential — MP{mp}\n(Tier 1 + Tier 2, IRA-Ref)',
        cbar_label='Weighted Adoption Rate (%)',
        cmap=_adopt_cmap, norm=_adopt_norm,
        # cbar_ticks=_adopt_cbar_ticks,
        selected_mps=selected_mps,
        geo_level='county',
        save_figure=SAVE_FIGURES,
        output_path=os.path.join(PROJECT_ROOT, 'county_adoption_rate_combined.png'),
    )

    print("[OK] Adoption choropleth generated")
else:
    print("[WARN] Adoption choropleth skipped — county shapefile not available")


# %% [markdown]
# ## Adoption Potential Dotplot Maps

# %%
# =============================================================================
# ADOPTION POTENTIAL PARAMETERS (base-case defaults)
# =============================================================================
# These mirror the upstream TARE pipeline settings. The column name convention
# encodes all parameters: e.g. 'iraRef_mp3_heating_adoption_central_inmap_acs_v4MID_fixed_base'
#
# Only change these if running a sensitivity analysis. For the paper's
# primary results, these are the correct values.

scc = 'central'                          # Social cost of carbon estimate
rcm_model = 'inmap'                      # Reduced-complexity model (InMAP)
cr_function = 'acs'                      # Concentration-response function (ACS)
cost_scenario = 'v4MID'                  # REMDB v4 midpoint cost scenario
discount_rate = 'fixed_base'             # 7% fixed discount rate
hvac_replacement_scenario = 'heating'    # Case A: Heating Only

# Map selected_mps to the naming convention used by the dotplot
HEATING_MEASURE_PACKAGES = selected_mps

HEATING_MP_SUBTITLES = {
    3: 'ASHP (MP3 - Min Efficiency)',
    4: 'ASHP (MP4 - High Efficiency)',
    8: 'Cold Climate HP (MP8)',
}

print(f"Sensitivity parameters set:")
print(f"  SCC: {scc}, RCM: {rcm_model}, CR: {cr_function}")
print(f"  Cost: {cost_scenario}, Discount: {discount_rate}")
print(f"  HVAC scenario: {hvac_replacement_scenario}")
print(f"  Measure packages: {HEATING_MEASURE_PACKAGES}")

# %%
# =============================================================================
# ADOPTION POTENTIAL DOTPLOT: N-row × 1-col (one row per MP, shared x-axis)
# =============================================================================

CASE_LABELS = {
    'heating': 'Heating Replacement Only',
    'heating_and_cooling': 'Heating & Cooling Replacement',
}

category = 'heating'
case_label = CASE_LABELS.get(hvac_replacement_scenario, hvac_replacement_scenario)

# Fuel counts for y-axis labels (compute once from first MP)
_first_mp = HEATING_MEASURE_PACKAGES[0]
_src = DATAFRAMES_BY_MP[_first_mp][discount_rate][rcm_model]
fuel_counts_millions = {
    str(fuel): int(n) * 242 / 1_000_000
    for fuel, n in _src.groupby('base_heating_fuel', observed=True).size().items()
}

n_mps = len(HEATING_MEASURE_PACKAGES)

# --- N rows × 1 column layout ---
# CHANGE: figsize width 12 -> 16 to give labels room at annotation_fontsize=12.
# Below width=16 at this font size, edge labels (x=0%, x=100%) clip the axis
# and adjacent labels collide. See sweep_both.py for the full margin x width
# grid that determined this value.
fig, axes = plt.subplots(
    n_mps, 1,
    figsize=(16, 8 * n_mps),
    sharex=True,
    sharey=True,
)
if n_mps == 1:
    axes = [axes]

for row_idx, mp in enumerate(HEATING_MEASURE_PACKAGES):
    ax = axes[row_idx]
    panel_title = f'{HEATING_MP_SUBTITLES.get(mp, f"MP{mp}")} — {case_label}'

    # Build the multi-index adoption DataFrame from TARE output
    source_df = DATAFRAMES_BY_MP[mp][discount_rate][rcm_model]

    mi_df = create_multiIndex_adoption_df(
        df=source_df,
        menu_mp=mp,
        category=category,
        scc=scc,
        rcm_model=rcm_model,
        cr_function=cr_function,
        cost_scenario=cost_scenario,
        discount_rate=discount_rate,
        hvac_replacement_scenario=hvac_replacement_scenario,
    )

    if mi_df is None:
        ax.set_title(panel_title, fontsize=16, fontweight='bold')
        ax.text(0.5, 0.5, 'No data\n(adoption columns missing)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, color='gray')
        # CHANGE: match the populated panels' xlim so an empty row aligns
        # visually with its neighbors. Cosmetic only.
        ax.set_xlim(-14, 114)
        ax.set_xticks(range(0, 101, 20))
        y_order = list(reversed(GROUPING_ORDER))
        ax.set_ylim(-0.5, len(y_order) - 0.5)
        ax.set_yticks(range(len(y_order)))
        continue

    scenario_names = build_adoption_scenario_names(
        mp, category, scc, rcm_model, cr_function,
        cost_scenario, discount_rate,
        hvac_replacement_scenario=hvac_replacement_scenario,
    )
    preira_col = scenario_names[0]
    iraref_col = scenario_names[1]

    plot_df = prepare_plot_data(
        mi_df, source_df,
        preira_col=preira_col,
        iraref_col=iraref_col,
        income_groups=['LMI'],
    )

    # CHANGE: pass xlim_margin=14 to scale axis margin to the larger annotation
    # font. Default 12 works for fontsize=7; 14 is needed for fontsize=12.
    plot_adoption_panel(
        plot_df, ax, title=panel_title,
        title_fontsize=16,
        ytick_fontsize=14,
        annotation_fontsize=14,
        xlim_margin=14,
        fuel_counts_millions=fuel_counts_millions,
    )
    ax.tick_params(axis='both', labelsize=14)

    # Only bottom panel gets x-axis label
    if row_idx < n_mps - 1:
        ax.set_xlabel('')

# --- Shared legend below figure ---
legend_handles = _build_legend_handles()
fig.legend(
    handles=legend_handles,
    loc='lower center',
    ncol=len(legend_handles),
    fontsize=14,
    frameon=True,
    bbox_to_anchor=(0.5, 0.0),
)

fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])

# --- Save ---
out_dir = os.path.join('.', 'figures')
os.makedirs(out_dir, exist_ok=True)
case_tag = 'caseA' if hvac_replacement_scenario == 'heating' else 'caseB'
for ext in ('png', 'pdf'):
    fig.savefig(
        os.path.join(out_dir, f'figure5_adoption_dotplot_{case_tag}.{ext}'),
        dpi=600, bbox_inches='tight',
    )
print(f"Saved to {out_dir}/figure5_adoption_dotplot_{case_tag}.{{png,pdf}}")
plt.show()

# %%
CASE_LABELS = {
    'heating': 'Heating Replacement Only',
    'heating_and_cooling': 'Heating & Cooling Replacement',
}

if not HEATING_MEASURE_PACKAGES:
    print("No active heating measure packages — skipping dotplot.")
else:
    category = 'heating'
    case_label = CASE_LABELS.get(hvac_replacement_scenario, hvac_replacement_scenario)

    # Compute national fuel counts in millions (scaling_factor = 242)
    _src = DATAFRAMES_BY_MP[HEATING_MEASURE_PACKAGES[0]][discount_rate][rcm_model]
    fuel_counts_millions = {
        str(fuel): int(n) * 242 / 1_000_000
        for fuel, n in _src.groupby('base_heating_fuel', observed=True).size().items()
    }

    # --- Print figure header before creating the figure ---
    print(f"Heat Pump Adoption Potential — {case_label}")
    print(f"Discount Rate: {discount_rate} | Cost Scenario: {cost_scenario}")
    print()
    print("Fuel sample counts (national, approx.):")
    for fuel, count in sorted(fuel_counts_millions.items()):
        print(f"  {fuel}: {count:.1f}M homes")
    print()

    n_mps = len(HEATING_MEASURE_PACKAGES)

    # figsize width 16 (was 12) gives labels room at annotation_fontsize=12.
    # Below width=16 at this font size, edge labels (x=0%, x=100%) clip the
    # axis and adjacent labels collide.
    fig, axes = plt.subplots(
        n_mps, 1,
        figsize=(16, 7.5 * n_mps),
        sharex=True,
        sharey=True,
    )
    if n_mps == 1:
        axes = [axes]

    for row_idx, mp in enumerate(HEATING_MEASURE_PACKAGES):
        ax = axes[row_idx]
        panel_title = f'{HEATING_MP_SUBTITLES.get(mp, f"MP{mp}")} — {case_label}'

        # Build the multi-index adoption DataFrame from TARE output
        source_df = DATAFRAMES_BY_MP[mp][discount_rate][rcm_model]

        mi_df = create_multiIndex_adoption_df(
            df=source_df,
            menu_mp=mp,
            category=category,
            scc=scc,
            rcm_model=rcm_model,
            cr_function=cr_function,
            cost_scenario=cost_scenario,
            discount_rate=discount_rate,
            hvac_replacement_scenario=hvac_replacement_scenario,
        )

        if mi_df is None:
            ax.set_title(panel_title, fontsize=16, fontweight='bold')
            ax.text(0.5, 0.5, 'No data\n(adoption columns missing)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=14, color='gray')
            # Match populated panels' xlim for visual consistency
            ax.set_xlim(-14, 114)
            ax.set_xticks(range(0, 101, 20))
            y_order = list(reversed(GROUPING_ORDER))
            ax.set_ylim(-0.5, len(y_order) - 0.5)
            ax.set_yticks(range(len(y_order)))
            continue

        scenario_names = build_adoption_scenario_names(
            mp, category, scc, rcm_model, cr_function,
            cost_scenario, discount_rate,
            hvac_replacement_scenario=hvac_replacement_scenario,
        )
        preira_col = scenario_names[0]
        iraref_col = scenario_names[1]

        plot_df = prepare_plot_data(
            mi_df, source_df,
            preira_col=preira_col,
            iraref_col=iraref_col,
            income_groups=['LMI'],
        )

        # --- Print sample stats for this panel ---
        print(f"--- MP{mp} sample stats ---")
        sample_info = (
            plot_df[['grouping', 'pct_of_sample', 'weighted_homes_millions']]
            .drop_duplicates('grouping')
            .reset_index(drop=True)
        )
        for _, r in sample_info.iterrows():
            print(f"  {r['grouping']}: {r['pct_of_sample']:.1f}% of sample, "
                  f"{r['weighted_homes_millions']:.1f}M homes")
        print()

        # xlim_margin=14 scales the axis margin to the larger annotation font.
        # Default 12 works for fontsize=7; 14 is needed for fontsize=12.
        plot_adoption_panel(
            plot_df, ax, title=panel_title,
            title_fontsize=16,
            ytick_fontsize=14,
            annotation_fontsize=14,
            xlim_margin=14,
            fuel_counts_millions=fuel_counts_millions,
        )
        ax.tick_params(axis='both', labelsize=14)

        # Only bottom panel gets x-axis label
        if row_idx < n_mps - 1:
            ax.set_xlabel('')

    # --- Shared legend below figure ---
    legend_handles = _build_legend_handles()
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=len(legend_handles),
        fontsize=14,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])

    # --- Save ---
    out_dir = os.path.join('.', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    case_tag = 'caseA' if hvac_replacement_scenario == 'heating' else 'caseB'
    for ext in ('png', 'pdf'):
        fig.savefig(
            os.path.join(out_dir, f'figure5_adoption_dotplot_{case_tag}_{location_id}.{ext}'),
            dpi=600, bbox_inches='tight',
        )
    print(f"Saved to {out_dir}/figure5_adoption_dotplot_{case_tag}_{location_id}.{{png,pdf}}")
    plt.show()

# %%



