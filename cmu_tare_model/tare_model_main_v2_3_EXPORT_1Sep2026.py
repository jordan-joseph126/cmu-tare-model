# %% [markdown]
# -------------------------------------------------------------------------------------------------------
# # EUSS Post-Retrofit Measure Packages: MP3, MP4, MP8, MP9, MP10
# -------------------------------------------------------------------------------------------------------
# - MP3: Min-efficiency, single-stage ASHP (15 SEER1, 9 HSPF1) --> (16 SEER1, 9.5 HSPF1) for ENERGY STAR
# - MP4: High-efficiency, variable-speed ASHP (24-29.3 SEER1, 14 HSPF1)
# - MP8: Whole Home Electrification (MP4 + High Efficiency End-uses)
# - MP9: Whole-Home Electrification (MP8) + Basic Enclosure Upgrade (MP1)
# - MP10: Whole-Home Electrification (MP8) + Enhanced Enclosure Upgrade (MP2)
# 
# -------------------------------------------------------------------------------------------------------
# # TARE MODEL SCENARIO: 2025 Reference Case
# -------------------------------------------------------------------------------------------------------
# - AEO2026 fuel price projections
# - AEO2026 degree-day factors
# - Cambium MidCase electricity grid
# - Single scenario: '2025 Reference Case'

# %%
# =============================================================================
# IMPORTS
# =============================================================================
import os
from IPython import get_ipython
from datetime import datetime
import logging
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# Project configuration
from config import PROJECT_ROOT

# Model constants - explicit imports for clarity
from cmu_tare_model.constants import (
    VERBOSE,
    SCC_ASSUMPTIONS,
    REMDB_COST_SCENARIO_KEYS,
    VALID_MENU_MPS,
    VALID_CATEGORIES,
    PRINT_DEBUG,
    PRINT_VERBOSE_DATAFRAMES,
    BLDG_ID_COL,
    TIMESTAMP_COL,
    ELEC_TOTAL_COL,
    BSQ_ELEC_COL,
    TEST_FIPS,
)
from cmu_tare_model.constants import (
    PRIVATE_DISCOUNT_RATE_COLS,
    PRIVATE_DISCOUNT_RATE_SHORT_KEYS,
)

# Column name builders
from cmu_tare_model.utils.column_names import (
    NPV_CASE_CATEGORIES,
    BASE_CASE_NPV_CASE,
    create_cost_col,
    create_capital_col,
    create_npv_col,
    create_npv_case_col,
    create_rebate_col,
    create_total_npv_col,
    create_climate_npv_col,
)
from cmu_tare_model.grid_impact.peak_load_functions import (
    find_adoption_column,
    extract_adopter_ids,
    compute_county_scenario_profile,
    plot_demand_panel,
)

# Data loading utility
from cmu_tare_model.utils.load_exported_results_to_df import load_model_run_output, load_measure_package_data

# =============================================================================
# MATPLOTLIB/SEABORN CONFIGURATION
# =============================================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.close('all')
%matplotlib inline

sns.set_theme(font='sans-serif', style='white')

# =============================================================================
# PROJECT ROOT AND TIMESTAMP SETUP
# =============================================================================
# Get the current datetime
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Format the name of the exported results file using the location ID
result_export_time = datetime.now()
model_run_date_time = result_export_time.strftime("%Y-%m-%d_%H-%M")

print(f"""
PROJECT_ROOT: {PROJECT_ROOT}

Start Time: {start_time}
Model Run Timestamp: {model_run_date_time}

Active Measure Packages: {VALID_MENU_MPS}
Active Capital Cost Scenarios: {REMDB_COST_SCENARIO_KEYS}
Active SCC Assumptions: {SCC_ASSUMPTIONS}
Active Discount Rates: {PRIVATE_DISCOUNT_RATE_SHORT_KEYS}

Note: DataFrames contain columns for ALL active cost scenarios.

""")

# %%
# Select whether to begin new run or visualize existing model outputs
while True:
    try:
        start_new_model_run = str(input("""
Would you like to begin a new simulation or visualize output results from a previous model run? Please enter one of the following:
Y. I'd like to start a new model run.
N. I'd like to visualize output results from a previous model run.""")).upper()

        print(f"Enter the following input: {start_new_model_run}")

        if start_new_model_run == 'Y':
            print(f"Formatted date for use in file name: {model_run_date_time}")

            # Relative path to the file from the project root
            relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_run_simulation_v2_3.ipynb")

            # Construct the absolute path to the file
            file_path = os.path.join(PROJECT_ROOT, relative_path)
            print(f"File path: {file_path}")

            # Storing Result Outputs in output_results folder
            output_folder_path = os.path.join(PROJECT_ROOT, "cmu_tare_model", "output_results")
            print(f"Result outputs will be exported here: {output_folder_path}")

            # On Windows, to avoid any path-escape quirks, convert backslashes to forward slashes
            file_path = file_path.replace("\\", "/")

            print(f"Running file: {file_path}")

            # iPthon magic command to run a .py file and import variables into the current IPython session
            if os.path.exists(file_path):
                get_ipython().run_line_magic('run', f'-i {file_path}')  # If your path has NO spaces, no quotes needed.
            else:
                print(f"File not found: {file_path}")

            break  # Exit the loop if input is 'Y'
            
        elif start_new_model_run == 'N':
            # Enter the date time of the model run in the following format: YYYY-MM-DD_HH-MM
            model_run_date_time = str(input("Enter the date time of the model run in the following format YYYY-MM-DD_HH-MM: "))
            location_id = str(input("Enter the location ID used in the model run (e.g., 'National' or 'PA'): "))
            
            # Load model run results
            print(f"Loading model run results for location ID: {location_id} and timestamp: {model_run_date_time}")

            # Storing Result Outputs in output_results folder
            output_folder_path = os.path.join(PROJECT_ROOT, "cmu_tare_model", "output_results")
            print(f"Past model run results will be loaded from here: {output_folder_path}")
            
            break  # Exit the loop if input is 'N'
        
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")
    
    except Exception as e:
        print("An error occurred:", e)
        print("Please try again.")

# %%
if VERBOSE:
    print(f"""
    ====================================================================================================================================================================
    LOAD SCENARIO DATA
    ====================================================================================================================================================================
    The load_model_run_output function loads scenario data from a specified folder and date. Additional details are provided below:
        
    Documentation for the load_model_run_output function:
    {load_model_run_output.__doc__}

    -----------------------------------------------------------------------------------------------
    LOADING SCENARIO DATA ...

    These parameters are common to all function calls:
    Output folder path: {output_folder_path}
    Model run date time: {model_run_date_time}
    """)

# %% [markdown]
# -------------------------------------------------------------------------------------------------------
# # LOAD MODEL RUN OUTPUT FOR ANALYSIS + VISUALS
# -------------------------------------------------------------------------------------------------------

# %%
# =======================================================================================================
# Baseline Scenario: Measure Package 0 (MP0)
# =======================================================================================================
menu_mp = 0

df_outputs_baseline_home = load_model_run_output(
    results_category='summary_baseline',
    menu_mp=menu_mp,
    output_folder_path=output_folder_path,
    location_id=location_id,
    results_export_formatted_date=model_run_date_time,
    use_chunked_loading=True,
    chunk_size=10000
)

# %%
# =============================================================================
# LOAD MODEL RESULTS: Based on VALID_MENU_MPS
# =============================================================================
# Only load measure packages that are in VALID_MENU_MPS.
# MP0 (baseline) is loaded separately above.
NON_BASELINE_MPS = [mp for mp in VALID_MENU_MPS if mp != 0]

# Convenience mapping for downstream code
DATAFRAMES_BY_MP = {}

for mp in NON_BASELINE_MPS:
    DATAFRAMES_BY_MP[mp] = load_measure_package_data(
        mp, output_folder_path, location_id, model_run_date_time
    )

print(f"\nLoaded measure packages: {list(DATAFRAMES_BY_MP.keys())}")

# %%
# # Load the raw MP3 or MP4 EUSS CSV and print its column names
# # Set to 3 or 4 to pick which measure package's raw file to inspect.
# menu_mp = 3

# MP_TO_UPGRADE = {3: "upgrade03", 4: "upgrade04"}
# input_mp = MP_TO_UPGRADE[menu_mp]

# filename = f"{input_mp}_metadata_and_annual_results.csv"
# relative_path = os.path.join(
#     "cmu_tare_model", "data", "euss_data", "resstock_amy2018_release_1.1",
#     "national", "csv", filename,
# )
# file_path = os.path.join(PROJECT_ROOT, relative_path)

# print(f"Retrieved data for filename: {filename}")
# print(f"Located at filepath: {file_path}")
# print("\n")

# # low_memory=False reads the entire file before inferring dtypes,
# # so mixed-type columns are automatically cast to object (str) without warnings.
# df_euss_am = pd.read_csv(file_path, low_memory=False, index_col="bldg_id")
# print(f"DATAFRAME SIZE before applying any filters: {df_euss_am.shape}")

# # Filter for occupied homes
# occupancy_filter = df_euss_am["in.vacancy_status"] == "Occupied"
# df_euss_am = df_euss_am.loc[occupancy_filter]
# print(f"DATAFRAME SIZE after filtering for 'Occupied' homes: {df_euss_am.shape}")

# # Filter for allowed housing types
# house_type_filter = df_euss_am["in.geometry_building_type_recs"].isin(ALLOWED_HOUSING_TYPES)
# df_euss_am = df_euss_am.loc[house_type_filter]
# print(f"Allowed housing types: {ALLOWED_HOUSING_TYPES}")
# print(f"DATAFRAME SIZE after filtering for allowed housing types: {df_euss_am.shape}")

# print("\nCOLUMN NAMES:")
# for col in df_euss_am.columns:
#     print(col)

# %%
# print("\nCOLUMN NAMES CONTAINING 'btu' in df_euss_am_baseline:")
# for col in df_euss_am_baseline.columns:
#     if "btu" in col.lower():
#         print(col)

# %%
# print("\nCOLUMN NAMES CONTAINING 'btu' in df_euss_am_mpX:")
# for col in df_euss_am.columns:
#     if "btu" in col.lower():
#         print(col)

# %% [markdown]
# -------------------------------------------------------------------------------------------------------
# # ECONOMIC ADOPTION POTENTIAL
# -------------------------------------------------------------------------------------------------------

# %% [markdown]
# ## Does the heat pump pay for itself?
# 
# An **economic adopter** is a home where the heat pump's extra upfront cost is
# recovered from energy-bill savings alone -- no climate or health benefit is
# needed to justify the investment.
# 
# **The rule:** a home is an economic adopter if its incremental private
# NPV >= 0. Break-even counts as adoption.
# 
# | Value | Meaning |
# |-------|---------|
# | `True`  | Heat pump covers its incremental cost (or better) from bill savings |
# | `False` | Valid home that cannot recover the incremental cost from savings alone |
# | `NaN`   | Excluded: invalid baseline fuel/tech or not in this measure package |
# 
# Climate and health damages are computed elsewhere and reported as outcomes,
# not as inputs to this decision.

# %%
# =============================================================================
# ECONOMIC ADOPTION -- setup: imports, parameters, and shared inputs
# =============================================================================
# Transported from calculate_postTARE_am_kpis_demand_bill_savings. Column names
# use the nine-case NPV_CASE_CATEGORIES scheme; there is no WTP or cost-scenario
# token in adopter column names.
import geopandas as gpd
import matplotlib.lines as mlines
from matplotlib.colors import Normalize

from cmu_tare_model.adoption_potential.determine_economic_adoption_potential import (
    economic_adoption_decision,
)
from cmu_tare_model.utils.modeling_params import define_scenario_params
from cmu_tare_model.utils.column_names import create_adoption_col
from cmu_tare_model.adoption_kpis import load_euss_baseline, compute_adoption_rate
from cmu_tare_model.adoption_kpis.visualize_geospatial_data import (
    plot_combined_choropleth,
)
from cmu_tare_model.adoption_kpis.data_loading import COUNTY_SHAPEFILE_PATH
from cmu_tare_model.adoption_potential.data_processing.visuals_adoption_dotplot import (
    plot_adoption_panel,
    build_econ_plot_df,
    REPLACEMENT_CREDIT_CASES,
    REPLACEMENT_CREDIT_MARKERS,
    NATIONAL_FUEL_GROUPING_ORDER,
    build_replacement_credit_legend_handles,
    build_rebate_policy_scenario_legend_handles,
    REBATE_POLICY_SCENARIO_MARKERS,
)

# Base-case parameters. The economic-adopter column name encodes these:
# e.g. 'ref2025_mp3_heatingLCC_coolingLCC_sub_econ_adopter_fixed_base'.
_POLICY = '2025 Reference Case'
_DISCOUNT_COL = 'private_discount_rate_fixed_base'
_COST = 'v4MID'                 # REMDB v4 midpoint (retained for API compat)
discount_rate = 'fixed_base'    # 7% fixed discount rate
SAVE_FIGURES = False            # Set True to write figure files to disk
FIGURE_DPI = 600                # Resolution for saved figures (matches other savefig calls)
GRID_IMPACT_ANALYSIS = True     # Set True to run BSQ-based grid impact analysis

# The adoption analysis runs on every loaded non-baseline measure package.
selected_mps = NON_BASELINE_MPS
HEATING_MEASURE_PACKAGES = selected_mps

# Equipment subtitles used as panel/map titles. Covers this notebook's MP set.
# TODO: Consider updating the specs to be SEER2/HSPF2 and MP3 as min rebate eligible
HEATING_MP_SUBTITLES = {
    3: 'Single-stage, min-efficiency ASHP (16 SEER1, 9.5 HSPF1)',
    4: 'Variable-speed, high-efficiency ASHP (24-29.3 SEER1, 13-14 HSPF1)',
    8: 'Whole-Home Electrification (High Efficiency)',
    9: 'Whole-Home Electrification + Basic Enclosure Upgrade',
    10: 'Whole-Home Electrification + Enhanced Enclosure Upgrade',
}

# EUSS baseline provides household weights for county adoption-rate weighting.
df_baseline = load_euss_baseline()
print(f"Baseline: {len(df_baseline):,} occupied SF homes")

# County shapefile for the adoption-rate choropleth. Missing shapefile is a
# warning, not an error -- the choropleth cell is skipped if it is unavailable.
gdf_counties_raw = None
try:
    gdf_counties_raw = gpd.read_file(COUNTY_SHAPEFILE_PATH)
    print(f"[OK] County shapefile loaded: {len(gdf_counties_raw)} features")
except Exception as e:
    print(f"[WARN] County shapefile not loaded: {e}")

# %%
# Generate the NINE economic-adopter columns (one per NPV case) for each measure
# package. After this cell, every case column exists in the 'fixed_base' frame
# (three scopes x three rebate policy scenarios: unsub, sub, sub_june2026):
#   ref2025_mp{mp}_heatingSavings_coolingLCC_{unsub,sub,sub_june2026}_econ_adopter_fixed_base
#   ref2025_mp{mp}_heatingLCC_coolingSavings_{unsub,sub,sub_june2026}_econ_adopter_fixed_base
#   ref2025_mp{mp}_heatingLCC_coolingLCC_{unsub,sub,sub_june2026}_econ_adopter_fixed_base

for mp in selected_mps:
    df_tare = DATAFRAMES_BY_MP[mp]['fixed_base']
    scenario_prefix = define_scenario_params(mp, _POLICY)[0]
    expected_adopter_cols = [
        create_adoption_col(scenario_prefix, npv_case, '_fixed_base')
        for npv_case in NPV_CASE_CATEGORIES
    ]
    missing_cols = [c for c in expected_adopter_cols if c not in df_tare.columns]
    if missing_cols:
        df_econ = economic_adoption_decision(
            df_tare,
            menu_mp=mp,
            policy_scenario=_POLICY,
            discount_rate_col_name=_DISCOUNT_COL,
            cost_scenario=_COST,
            verbose=False,
        )
        # Copy only newly created columns back into the canonical frame.
        new_cols = [c for c in df_econ.columns if c not in df_tare.columns]
        for col in new_cols:
            DATAFRAMES_BY_MP[mp]['fixed_base'][col] = df_econ[col]
        print(f"[OK] MP{mp}: {len(new_cols)} econ-adopter columns added")
    else:
        print(f"[SKIP] MP{mp}: all economic-adopter columns already present")

print("\n[OK] Economic-adopter columns present for all selected MPs")

# %%
# County-level economic adoption rate. Uses BASE_CASE_NPV_CASE
# ('heatingLCC_coolingLCC_unsub' -- unsubsidized, both avoided replacements
# credited) as the headline adoption metric.
print(f"\n{'='*60}")
print("Economic Adoption Rate -- 2025 Reference Case")
print(f"{'='*60}")

econ_adoption_rate_results = {}
for mp in selected_mps:
    print(f"\n===== {HEATING_MP_SUBTITLES.get(mp, f'MP{mp}')} =====")
    df_tare = DATAFRAMES_BY_MP[mp]['fixed_base']
    prefix = define_scenario_params(mp, _POLICY)[0]
    adoption_col = create_adoption_col(
        scenario_prefix=prefix,
        npv_case=BASE_CASE_NPV_CASE,
        method_suffix='_fixed_base',
    )
    print(f'  Adoption column: {adoption_col}')
    # adopter_tiers=[True] counts 1.0 (adopter) vs 0.0; NaN (excluded) rows are
    # ignored automatically by compute_adoption_rate.
    df_adopt = compute_adoption_rate(
        df_tare,
        adoption_col=adoption_col,
        adopter_tiers=[True],
        geo_level='county',
        df_euss=df_baseline,
        verbose=True,
    )
    econ_adoption_rate_results[mp] = df_adopt

print("\n[OK] Economic adoption rate complete (county-level)")


# %%
# Visualize the county-level economic adoption rate as a choropleth. 
# Colorbase 0 to 100% adoption potential. The choropleth is skipped if the county shapefile is unavailable.

if gdf_counties_raw is not None:
    _adopt_cmap = 'Greens'
    _adopt_norm = Normalize(vmin=0, vmax=100)

    print("\n--- Summary: adoption_rate_pct ---")
    for mp in selected_mps:
        _v = econ_adoption_rate_results[mp]['adoption_rate_pct'].dropna()
        _pct_high = (_v >= 50).mean() * 100
        print(f"  MP{mp}: n={len(_v):,} counties | "
              f"min={_v.min():.1f}% | med={_v.median():.1f}% | "
              f"mean={_v.mean():.1f}% | max={_v.max():.1f}% | "
              f"{_pct_high:.1f}% of counties >= 50% adoption potential")

    plot_combined_choropleth(
        gdf_counties_raw, econ_adoption_rate_results,
        column='adoption_rate_pct',
        title_template=HEATING_MP_SUBTITLES,
        cbar_label='Share of households recovering electrification premium through discounted operational savings (%)',
        cmap=_adopt_cmap, norm=_adopt_norm,
        selected_mps=selected_mps,
        geo_level='county',
        save_figure=SAVE_FIGURES,
        output_path=os.path.join(PROJECT_ROOT, 'county_econ_adoption_rate_combined.png'),
    )
    print("[OK] Economic adoption choropleth generated")
else:
    print("[WARN] Adoption choropleth skipped -- county shapefile not available")

# %% [markdown]
# ## Economic Adoption Potential Dotplot
# 
# Three markers per row, one per replacement-credit scope, plotting the June
# 2026 subsidized adoption rate:
# - circle = heating replacement credit only (`heatingLCC_coolingSavings`)
# - triangle = cooling replacement credit only (`heatingSavings_coolingLCC`)
# - square = heating + cooling replacement credit
# (`heatingLCC_coolingLCC`) -- the headline pick
# 
# The annotation shows the June 2026 subsidized minus unsubsidized
# adoption-rate delta for the same scope.
# 

# %%
sns.set_theme(font='sans-serif', style='whitegrid')

# %%
if not HEATING_MEASURE_PACKAGES:
    print("No active heating measure packages -- skipping economic adoption dotplot.")
else:
    # National fuel counts, weighted to homes (same method as the tier dotplot).
    _src = DATAFRAMES_BY_MP[HEATING_MEASURE_PACKAGES[0]][discount_rate]
    fuel_counts_millions = {
        str(fuel): weighted_homes / 1_000_000
        for fuel, weighted_homes in _src.groupby(
            'base_heating_fuel', observed=True)['weight'].sum().items()
    }

    n_mps = len(HEATING_MEASURE_PACKAGES)
    fig, axes = plt.subplots(
        n_mps, 1, figsize=(12, 6 * n_mps), sharex=True, sharey=True,
    )
    if n_mps == 1:
        axes = [axes]

    for row_idx, mp in enumerate(HEATING_MEASURE_PACKAGES):
        ax = axes[row_idx]
        panel_title = f'{HEATING_MP_SUBTITLES.get(mp, f"MP{mp}")}'
        source_df = DATAFRAMES_BY_MP[mp][discount_rate]

        # Base case: this figure now plots the UNSUBSIDIZED adoption rate for
        # each replacement-credit scope (it used to plot the June 2026
        # subsidized rate). The headline scope stays heatingLCC_coolingLCC
        # (both heating + cooling replacement credited), drawn with the square
        # marker. The vintage is passed as a plain argument so it is not buried
        # in a hardcoded column name.
        plot_df = build_econ_plot_df(
            source_df, mp, cost_scenario=_COST, discount_rate=discount_rate,
            rebate_vintage='unsub',
        )

        # Print the National unsubsidized rate per scope so the plotted values
        # are visible on every run.
        print(f"--- MP{mp} economic adoption summary (National, unsubsidized) ---")
        for case_label in REPLACEMENT_CREDIT_CASES:
            nat_row = plot_df[
                (plot_df['grouping'] == 'National -- Overall') &
                (plot_df['tier_label'] == case_label)
            ]
            if not nat_row.empty:
                print(f"  {case_label}: {nat_row.iloc[0]['case_b_pct']:.1f}%")
        print()

        plot_adoption_panel(
            plot_df, ax,
            grouping_order=NATIONAL_FUEL_GROUPING_ORDER,
            title=panel_title,
            title_fontsize=16,
            ytick_fontsize=14,
            annotation_fontsize=14,
            annotation_x_offset_pts=26,
            annotation_y_offset_pts=8,
            xlim_margin=20,
            fuel_counts_millions=fuel_counts_millions,
            custom_tier_markers=REPLACEMENT_CREDIT_MARKERS,
            fill_markers=False,
        )
        ax.tick_params(axis='both', labelsize=14)

        ax.legend(
            handles=build_replacement_credit_legend_handles(
                fill_markers=False),
            loc='upper right', fontsize=14, frameon=True)

        if row_idx < n_mps - 1:
            ax.set_xlabel('')

    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.96])

    if SAVE_FIGURES:
        out_dir = os.path.join(PROJECT_ROOT, 'figures')
        os.makedirs(out_dir, exist_ok=True)
        for ext in ('png', 'pdf'):
            fig.savefig(
                os.path.join(out_dir, f'figure6_econ_adoption_dotplot_{location_id}.{ext}'),
                dpi=FIGURE_DPI, bbox_inches='tight',
            )
        print(f"Saved econ adoption dotplot to {out_dir}")
    plt.show()


# %%
# Hold this replacement-credit scenario fixed; the three markers vary the rebate
# policy scenario. Switch to 'heatingLCC_coolingSavings' for the heating-only view.
_FIXED_CREDIT = 'heatingLCC_coolingLCC'

if not HEATING_MEASURE_PACKAGES:
    print("No active heating measure packages -- skipping rebate-policy dotplot.")
else:
    _src = DATAFRAMES_BY_MP[HEATING_MEASURE_PACKAGES[0]][discount_rate]
    fuel_counts_millions = {
        str(fuel): weighted_homes / 1_000_000
        for fuel, weighted_homes in _src.groupby(
            'base_heating_fuel', observed=True)['weight'].sum().items()
    }

    n_mps = len(HEATING_MEASURE_PACKAGES)
    fig, axes = plt.subplots(
        n_mps, 1, figsize=(12, 6 * n_mps), sharex=True, sharey=True,
    )
    if n_mps == 1:
        axes = [axes]

    for row_idx, mp in enumerate(HEATING_MEASURE_PACKAGES):
        ax = axes[row_idx]
        panel_title = f'{HEATING_MP_SUBTITLES.get(mp, f"MP{mp}")}'
        source_df = DATAFRAMES_BY_MP[mp][discount_rate]

        plot_df = build_econ_plot_df(
            source_df, mp, cost_scenario=_COST, discount_rate=discount_rate,
            shape_by='rebate_policy_scenario',
            fixed_replacement_credit_scenario=_FIXED_CREDIT,
        )

        # Per-panel national summary (headline adoption rate per rebate policy scenario).
        print(f"--- MP{mp} adoption by rebate policy scenario ({_FIXED_CREDIT}) ---")
        nat = plot_df[plot_df['grouping'] == 'National -- Overall']
        for _, r in nat.iterrows():
            print(f"  {r['tier_label']}: {r['case_b_pct']:.1f}%")
        print()

        plot_adoption_panel(
            plot_df, ax,
            grouping_order=NATIONAL_FUEL_GROUPING_ORDER,
            title=panel_title,
            title_fontsize=16,
            ytick_fontsize=14,
            annotation_fontsize=14,
            annotation_x_offset_pts=26,
            annotation_y_offset_pts=8,
            xlim_margin=20,
            fuel_counts_millions=fuel_counts_millions,
            custom_tier_markers=REBATE_POLICY_SCENARIO_MARKERS,
            fill_markers=False,
        )
        ax.tick_params(axis='both', labelsize=14)
        ax.legend(
            handles=build_rebate_policy_scenario_legend_handles(
                fill_markers=False),
            loc='upper right', fontsize=14, frameon=True,
        )
        if row_idx < n_mps - 1:
            ax.set_xlabel('')

    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.96])

    if SAVE_FIGURES:
        out_dir = os.path.join(PROJECT_ROOT, 'figures')
        os.makedirs(out_dir, exist_ok=True)
        stem = f'econ_adoption_dotplot_rebate_policy_scenario_{_FIXED_CREDIT}'
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(out_dir, f'{stem}.{ext}'),
                        bbox_inches='tight', dpi=FIGURE_DPI)
        print(f"[OK] Saved {stem}.png / .pdf")
    plt.show()


# %% [markdown]
# ## County Level Maps

# %%
# =============================================================================
# Retrofit impact on electricity demand and operating cost (county choropleths)
# =============================================================================
# Transported from calculate_postTARE_am_kpis_demand_bill_savings. Produces the
# three county maps: operating-cost % change, electricity demand change (GWh),
# and demand % change. Reuses the already-loaded df_baseline and
# gdf_counties_raw; it does not re-read any shapefile.
from cmu_tare_model.adoption_kpis import (
    load_euss_upgrade,
    mp_to_upgrade,
    compute_scenario_demand,
    aggregate_demand,
)


# These three helpers have no module home yet; they are defined inline here.
# If they gain wider use, move them into an adoption_kpis viz-helper module.
def pct_change(new, old):
    """Per-element percent change (new - old) / old * 100.

    Returns NaN wherever old <= 0 (invalid baseline) or either input is NaN,
    so homes with zero or negative baseline cost are excluded rather than
    producing infinite or misleading values.
    """
    old_safe = old.where(old > 0, other=np.nan)
    return (new - old_safe) / old_safe * 100


def make_symmetric_norm(values, center=0.0, low_q=0.02, high_q=0.98):
    """Symmetric Normalize centered at ``center``.

    Clips to the [low_q, high_q] percentiles before computing the symmetric
    deviation, so a single extreme county cannot compress the colormap.
    """
    v = values.dropna()
    q_low = v.quantile(low_q)
    q_high = v.quantile(high_q)
    dev = max(abs(q_low - center), abs(q_high - center))
    return Normalize(vmin=center - dev, vmax=center + dev)


def print_column_summary(results, column, label, selected_mps, mp_subtitles,
                         positive_direction="increase"):
    """Print per-MP min/median/mean/max summary for a county-level column."""
    print(f"\n--- Summary: {column} ---")
    for mp in selected_mps:
        _v = results[mp][column].dropna()
        if positive_direction == "increase":
            _pct = (_v > 0).mean() * 100
        else:
            _pct = (_v < 0).mean() * 100
        print(f"  MP{mp}: n={len(_v):,} counties | "
              f"min={_v.min():.1f} | med={_v.median():.1f} | "
              f"mean={_v.mean():.1f} | max={_v.max():.1f} | "
              f"{_pct:.1f}% of counties {positive_direction}")


# %%
# Step 1 -- load EUSS upgrade energy for each measure package.
upgrade_data = {}
for mp in selected_mps:
    upgrade_name = mp_to_upgrade(mp)
    print(f"Loading MP{mp} ({upgrade_name})...")
    upgrade_data[mp] = load_euss_upgrade(upgrade_name)
    print(f"  MP{mp}: {len(upgrade_data[mp]):,} applicable homes")

# Step 2 -- operating-cost % change (county median of per-home percent change).
print(f"\n{'='*60}")
print("OPERATING COST % CHANGE -- average annual, all fuels, 100% adoption")
print(f"{'='*60}")

bill_savings_results = {}
for mp in selected_mps:
    print(f"\n===== {HEATING_MP_SUBTITLES.get(mp, f'MP{mp}')} =====")
    df_tare = DATAFRAMES_BY_MP[mp][discount_rate]
    scenario_prefix = define_scenario_params(mp, _POLICY)[0]
    pct_col = f"{scenario_prefix}heating_avg_annual_fuel_cost_pct_change"
    if pct_col not in df_tare.columns:
        raise KeyError(
            f"Average-annual operating-cost column '{pct_col}' not found for "
            f"MP{mp}. Re-run the fuel-cost pipeline "
            "(calculate_lifetime_fuel_costs) so the average-annual columns are "
            "materialized, then reload DATAFRAMES_BY_MP."
        )
    # County median of the materialized per-home percent change.
    df_county = (
        pd.DataFrame({'county': df_tare['county'],
                      'operating_cost_pct_change': df_tare[pct_col]})
        .groupby('county')['operating_cost_pct_change']
        .median()
        .reset_index()
    )
    print(f"  Per-home valid records: {df_tare[pct_col].notna().sum():,} | "
          f"Counties: {len(df_county):,}")
    bill_savings_results[mp] = df_county

# Step 3 -- ANNUAL electricity demand change in 2025 (county-level GWh and percent). Both
# elec_change_gwh and pct_elec_demand_change come straight from aggregate_demand.
print(f"\n{'='*60}")
print("DEMAND CHANGE -- all fuels, 100% adoption")
print(f"{'='*60}")

demand_results = {}
for mp in selected_mps:
    print(f"\n===== {HEATING_MP_SUBTITLES.get(mp, f'MP{mp}')} =====")
    df_demand = compute_scenario_demand(
        df_baseline, upgrade_data[mp], fuel_filter=None, verbose=True,
    )
    demand_results[mp] = aggregate_demand(
        df_demand, geo_level='county', verbose=True,
    )


# %%
# Step 4 -- shared symmetric norms (centered at 0) then the three choropleths.
# make_symmetric_norm clips to the 2nd/98th percentile across all MPs so a
# single extreme county cannot compress the colormap.
if gdf_counties_raw is not None:
    _all_pct_bill = pd.concat(
        [bill_savings_results[mp]['operating_cost_pct_change']
         for mp in selected_mps]
    )
    shared_pct_bill_norm = make_symmetric_norm(_all_pct_bill)
    print(f"\nOperating cost % norm: [{shared_pct_bill_norm.vmin:.1f}, 0, "
          f"{shared_pct_bill_norm.vmax:.1f}]%")

    _all_demand_gwh = pd.concat(
        [demand_results[mp]['elec_change_gwh'] for mp in selected_mps]
    )
    shared_demand_norm = make_symmetric_norm(_all_demand_gwh)
    print(f"Demand GWh norm: [{shared_demand_norm.vmin:.1f}, 0, "
          f"{shared_demand_norm.vmax:.1f}] GWh")

    _all_pct_demand = pd.concat(
        [demand_results[mp]['pct_elec_demand_change'] for mp in selected_mps]
    )
    shared_pct_demand_norm = make_symmetric_norm(_all_pct_demand)
    print(f"Demand % norm: [{shared_pct_demand_norm.vmin:.1f}, 0, "
          f"{shared_pct_demand_norm.vmax:.1f}]%")

    # ---- Operating cost percent change (county-level) ----
    print_column_summary(
        bill_savings_results, 'operating_cost_pct_change',
        'Operating cost % change', selected_mps, HEATING_MP_SUBTITLES,
        positive_direction='HP saves money (< 0)',
    )
    plot_combined_choropleth(
        gdf_counties_raw, bill_savings_results,
        column='operating_cost_pct_change',
        title_template=HEATING_MP_SUBTITLES,
        cbar_label='Post-retrofit change in average annual operating cost, relative to baseline equipment (%)',
        cmap='RdBu_r', norm=shared_pct_bill_norm, selected_mps=selected_mps,
        geo_level='county',
        save_figure=SAVE_FIGURES,
        output_path=os.path.join(
            PROJECT_ROOT, 'county_bill_pct_change_combined.png'),
    )

    # ---- Electricity demand change (GWh, county-level) ----
    print_column_summary(
        demand_results, 'elec_change_gwh',
        'Elec demand change (GWh)', selected_mps, HEATING_MP_SUBTITLES,
        positive_direction='increase',
    )
    plot_combined_choropleth(
        gdf_counties_raw, demand_results,
        column='elec_change_gwh',
        title_template=HEATING_MP_SUBTITLES,
        cbar_label='Post-retrofit change in 2025 annual electricity demand, relative to baseline (GWh)',
        cmap='coolwarm', norm=shared_demand_norm, selected_mps=selected_mps,
        geo_level='county',
        save_figure=SAVE_FIGURES,
        output_path=os.path.join(
            PROJECT_ROOT, 'county_elec_demand_gwh_combined.png'),
    )

    # ---- Electricity demand percent change (county-level) ----
    print_column_summary(
        demand_results, 'pct_elec_demand_change',
        'Elec demand % change', selected_mps, HEATING_MP_SUBTITLES,
        positive_direction='increase',
    )
    plot_combined_choropleth(
        gdf_counties_raw, demand_results,
        column='pct_elec_demand_change',
        title_template=HEATING_MP_SUBTITLES,
        cbar_label='Post-retrofit change in 2025 annual electricity demand, relative to baseline (%)',
        cmap='coolwarm', norm=shared_pct_demand_norm, selected_mps=selected_mps,
        geo_level='county',
        save_figure=SAVE_FIGURES,
        output_path=os.path.join(
            PROJECT_ROOT, 'county_elec_demand_pct_combined.png'),
    )
    print("\n[OK] All county-level demand maps generated")
else:
    print("[WARN] County maps skipped -- county shapefile not available")


# %% [markdown]
# ## Export County Level and Household Level CSVs for Case-Studies

# %%
# ============================================================
# Tepper CSV exports -- one household + one county CSV per export scope
# ============================================================
# Reuses objects already in memory from this notebook run:
#   DATAFRAMES_BY_MP[mp]['fixed_base']  -- household frame (per MP)
#   econ_adoption_rate_results[mp]      -- county adoption table
#   bill_savings_results[mp]            -- county operating-cost table
#   demand_results[mp]                  -- county demand table
# and the run identifiers output_folder_path / location_id / model_run_date_time.
#
# The per-year consumption columns are not in the household summary frame.
# They live in the supplemental fuel-cost file written earlier in this run,
# which is loaded once per measure package below.

from cmu_tare_model.utils.export_model_run_results import export_model_run_output
from cmu_tare_model.utils.load_exported_results_to_df import load_model_run_output
from cmu_tare_model.utils.export_tepper_csv import (
    export_source_data_copies,
    filter_to_export_scope,
)

# Reconcile the per-county home tallies before exporting. The adoption and
# demand tables sum household weights independently, so their totals drift by a
# hair (float precision) for almost every county. One home's weight -- read from
# the data, not hardcoded -- is the tolerance: a gap smaller than a single home
# is that precision noise (or a lone edge-case building only one path counts)
# and is ignored; a larger gap means the two tables saw a different home set.
one_home_weight = df_baseline["weight"].median()  # ~242, from the weight column

for mp in selected_mps:
    _hc = econ_adoption_rate_results[mp][["county", "home_count"]].merge(
        demand_results[mp][["county", "home_count"]], on="county",
        how="outer", suffixes=("_adoption", "_demand"),
    )
    _hc["delta"] = (_hc["home_count_adoption"] - _hc["home_count_demand"]).abs()
    _disagree = _hc[_hc["delta"] > one_home_weight]
    if len(_disagree):
        print(f"[WARN] MP{mp}: home_count differs by more than one home "
              f"({one_home_weight:,.2f} weighted) for {len(_disagree)} "
              f"county(ies):")
        for _, _row in _disagree.iterrows():
            print(f"       {_row['county']}: adoption="
                  f"{_row['home_count_adoption']} demand="
                  f"{_row['home_count_demand']} (delta={_row['delta']:.2f})")
    else:
        print(f"[OK] MP{mp}: home_count agrees within one home "
              f"({one_home_weight:,.2f} weighted).")


# Export scopes. Each entry writes one household CSV and one county CSV per
# measure package. Filtering happens here, not at model run scope, so one
# national run can produce a national file plus any number of state or county
# files.
#
#   label                -- goes in the filename
#   column               -- any column shared by the household frame and the
#                           county tables: 'state' (two-letter code) or
#                           'county' (Census GISJOIN code). None keeps every row.
#   value                -- what that column must equal
#   drop_not_applicable  -- drop homes the model could not evaluate
#                           (include_heating = False) from the household file.
#                           Leave False for the full-scope master and True for
#                           a filtered subset: Excel treats a blank cell as zero
#                           in arithmetic, so averaging an NPV column over rows
#                           that are not applicable gives a wrong answer with no
#                           warning.
#
# To add a scope, copy a line. A whole state is
# {"label": "PA", "column": "state", "value": "PA"}; another county is its
# Census GISJOIN code, e.g. "G4200030" for Allegheny County, PA.
TEPPER_EXPORT_SCOPES = [
    {"label": location_id, "column": None, "value": None,
     "drop_not_applicable": False},
    {"label": "Allegheny", "column": "county", "value": "G4200030",
     "drop_not_applicable": True},
]

for mp in selected_mps:
    # Per-year consumption for this measure package. The file also holds the
    # per-year fuel costs; the export selects only the consumption columns.
    df_annual_consumption = load_model_run_output(
        results_category='fuel_costs_ref2025',
        menu_mp=mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time,
    )

    df_household = DATAFRAMES_BY_MP[mp]['fixed_base']
    county_tables_full = {
        'adoption': econ_adoption_rate_results[mp],
        'bill_savings': bill_savings_results[mp],
        'demand': demand_results[mp],
    }

    for scope in TEPPER_EXPORT_SCOPES:
        scope_label = scope["label"]
        scope_column = scope["column"]
        scope_value = scope["value"]

        # Skip a scope that this run does not cover, rather than failing. A
        # single-state run will not contain most counties.
        if scope_column is not None:
            if not (df_household[scope_column] == scope_value).any():
                print(f"[INFO] MP{mp}: no homes match {scope_column} = "
                      f"{scope_value!r}; skipping the {scope_label} files.")
                continue

        df_scope = filter_to_export_scope(
            df_household, scope_column, scope_value)

        if scope["drop_not_applicable"]:
            applicable = df_scope['include_heating'].fillna(False).astype(bool)
            dropped_count = int((~applicable).sum())
            df_scope = df_scope.loc[applicable]
            print(f"[INFO] MP{mp} {scope_label}: {dropped_count:,} of "
                  f"{dropped_count + len(df_scope):,} homes dropped for "
                  f"include_heating = False; {len(df_scope):,} exported.")

        # Household CSV (one row per home).
        export_model_run_output(
            df_results_export=df_scope,
            results_category='tepper_household',
            menu_mp=mp,
            output_folder_path=output_folder_path,
            location_id=scope_label,
            results_export_formatted_date=model_run_date_time,
            df_annual_consumption=df_annual_consumption,
        )

        # County CSV (one row per county) -- three tables assembled on 'county'.
        county_tables_scope = {}
        for table_name, df_county_table in county_tables_full.items():
            county_tables_scope[table_name] = filter_to_export_scope(
                df_county_table, scope_column, scope_value)

        export_model_run_output(
            df_results_export=None,  # not used by the county export
            results_category='tepper_county',
            menu_mp=mp,
            output_folder_path=output_folder_path,
            location_id=scope_label,
            results_export_formatted_date=model_run_date_time,
            county_tables=county_tables_scope,
        )

print(f"\nTepper exports written to: {os.path.join(output_folder_path, 'tepper_export')}")

# The three vendored inputs, copied unchanged, so a reader can redo the
# fuel-price lookup themselves.
export_source_data_copies(output_folder_path)


# %%
# ============================================================================
# VERIFY: peak-load + whole-home electricity columns in the Tepper household CSV
# ============================================================================
# Reads the actual exported files and confirms the 14 retained columns are
# present, populated, and internally consistent. Nothing in-memory is trusted --
# this reads the CSVs that Tamar will pull. There is now one file per export
# scope per measure package, so every scope is checked, not just the newest.
import pandas as pd
from pathlib import Path

try:
    from config import PROJECT_ROOT
    search_root = Path(PROJECT_ROOT)
except Exception:
    search_root = Path.cwd()

def expected_new_columns(mp):
    """The 14 columns this change adds for one measure package."""
    base = [
        "base_peak_electricity_cooling_kw", "base_peak_electricity_heating_kw",
        "base_peak_load_cooling_kbtu_hr", "base_peak_load_heating_kbtu_hr",
        "base_total_electricity_consumption",
    ]
    mp_cols = [f"mp{mp}_{s}" for s in [
        "peak_electricity_cooling_kw", "peak_electricity_heating_kw",
        "peak_electricity_cooling_kw_savings", "peak_electricity_heating_kw_savings",
        "peak_load_cooling_kbtu_hr", "peak_load_heating_kbtu_hr",
        "peak_load_cooling_kbtu_hr_savings", "peak_load_heating_kbtu_hr_savings",
        "total_electricity_consumption",
    ]]
    return base, mp_cols

# Find the newest exported household CSV for each measure package AND scope.
# Filenames read tepper_household_mp{mp}_{scope}_{date}.csv.
all_files = sorted(search_root.rglob("tepper_household_mp*.csv"),
                   key=lambda p: p.stat().st_mtime)
by_file = {}
for p in all_files:
    token = p.name.split("_mp", 1)[1]           # e.g. "3_National_2026-08-18.csv"
    parts = token.split("_")
    mp = int(parts[0])
    scope = parts[1]                             # 'National', 'Allegheny', ...
    by_file[(mp, scope)] = p                     # keep the newest of each

if not by_file:
    print("No tepper_household_mp*.csv found under", search_root)
else:
    for mp, scope in sorted(by_file):
        path = by_file[(mp, scope)]
        df = pd.read_csv(path, index_col="bldg_id")
        base_cols, mp_cols = expected_new_columns(mp)
        need = base_cols + mp_cols
        missing = [c for c in need if c not in df.columns]

        print("=" * 78)
        print(f"MP{mp} {scope}: {path.name}")
        print(f"  rows={len(df):,}  total_columns={df.shape[1]}")
        print(f"  new columns present: {len(need) - len(missing)}/{len(need)}"
              f"   missing={missing}")

        if not missing:
            for c in need:
                print(f"    {c:<50} non_null={int(df[c].notna().sum()):,}")

            # Consistency checks (do not assume -- verify against the data):
            # whole-home electricity delta = baseline total - retrofit total.
            elec_delta = (df["base_total_electricity_consumption"]
                          - df[f"mp{mp}_total_electricity_consumption"])
            print(f"  whole-home elec change (base - retrofit) kWh:"
                  f" mean={elec_delta.mean():,.0f}"
                  f" min={elec_delta.min():,.0f} max={elec_delta.max():,.0f}")
            print(f"  heating peak-demand savings (kW):"
                  f" mean={df[f'mp{mp}_peak_electricity_heating_kw_savings'].mean():.3f}")
    print("=" * 78)
    print("[OK] Verification complete.")


# %%
# ============================================================================
# VERIFY (in-memory): the loaded export frames vs the export column list
# ============================================================================
# The export now draws from two frames: the household summary frame supplies
# 94 columns, and the supplemental fuel-cost frame supplies the 60 per-year
# consumption columns. This checks each column against the frame it comes
# from, which is exactly what export_tepper_household does -- if this succeeds,
# the real export will not raise KeyError.
from cmu_tare_model.utils.export_tepper_csv import (
    build_household_column_list,
    build_annual_consumption_column_list,
)
from cmu_tare_model.utils.load_exported_results_to_df import load_model_run_output

if "DATAFRAMES_BY_MP" in dir():
    for mp in sorted(DATAFRAMES_BY_MP):
        df = DATAFRAMES_BY_MP[mp]["fixed_base"]
        wanted = build_household_column_list(mp)
        annual_columns = build_annual_consumption_column_list(mp)
        annual_column_set = set(annual_columns)

        summary_columns = []
        for column in wanted:
            if column not in annual_column_set:
                summary_columns.append(column)

        missing_from_frame = []
        for column in summary_columns:
            if column not in df.columns:
                missing_from_frame.append(column)

        print(f"MP{mp}: frame_cols={df.shape[1]}  export_list={len(wanted)}  "
              f"(summary {len(summary_columns)} + annual {len(annual_columns)})")
        print(f"  summary columns missing from frame: "
              f"{missing_from_frame or 'none'}")
        _ = df.loc[:, summary_columns]
        print(f"  [OK] the {len(summary_columns)} summary columns select cleanly")

        # The 60 annual consumption columns live in the supplemental
        # fuel-cost file, so check them there.
        df_annual = load_model_run_output(
            results_category='fuel_costs_ref2025',
            menu_mp=mp,
            output_folder_path=output_folder_path,
            location_id=location_id,
            results_export_formatted_date=model_run_date_time,
        )
        missing_from_annual = []
        for column in annual_columns:
            if column not in df_annual.columns:
                missing_from_annual.append(column)
        print(f"  annual columns missing from supplemental frame: "
              f"{missing_from_annual or 'none'}")
        _ = df_annual.loc[:, annual_columns]
        print(f"  [OK] the {len(annual_columns)} annual columns select cleanly")
else:
    print("DATAFRAMES_BY_MP not in session -- run the export cells first, "
          "or rely on Cell 1 (reads the written CSVs).")


# %% [markdown]
# # GRID IMPACT ANALYSIS
# - TODO: Update to prompt the user for the FIPS Code and County Name OR simply request a custom set of bldg_id values like Tamar has.

# %%
# =============================================================================
# GRID IMPACT -- build adopter building IDs by measure package and county
# =============================================================================
# Produces adopter_ids_by_mp, consumed by the BSQ timeseries and county-profile
# cells below. Two adopter sets per county:
#   all_filtered -- every filtered building in the county (100% adoption bound)
#   constrained  -- buildings that are economic adopters (econ_adopter == 1.0),
#                   i.e. the heat pump pays for itself at NPV >= 0. This matches
#                   the economic-adoption definition used throughout the
#                   notebook; it is NOT the deprecated Tier 1+2 tiered split.
#
# The constrained set uses BASE_CASE_NPV_CASE (the study base case, defined once
# in column_names.py: 'heatingLCC_coolingLCC_unsub' -- unsubsidized, both the
# heating and cooling replacement costs credited in the NPV). npv_case is passed
# by keyword so the adopter column can never silently fall back to
# find_adoption_column's default (a positional call previously routed this
# figure to the wrong case).
from cmu_tare_model.grid_impact.peak_load_functions import gisjoin_to_fips
from cmu_tare_model.utils.column_names import BASE_CASE_NPV_CASE

adopter_ids_by_mp = {}
adoption_col_by_mp = {}

for mp in selected_mps:
    df_tare = DATAFRAMES_BY_MP[mp][discount_rate]

    # Derive the economic-adopter column via the helper -- no hardcoded prefix.
    # The cost token does not change the column name (it was dropped from output
    # names in the July 2026 refactor), so any key in REMDB_COST_SCENARIO_KEYS
    # resolves the same column; the loop just takes the first one that succeeds.
    # All arguments are passed by keyword so a positional slip cannot silently
    # change the case.
    adoption_col = None
    for cost_scenario in REMDB_COST_SCENARIO_KEYS:
        try:
            adoption_col = find_adoption_column(
                df=df_tare,
                mp=mp,
                cost_scenario=cost_scenario,
                discount_rate_key=discount_rate,
                npv_case=BASE_CASE_NPV_CASE,
            )
            break
        except KeyError:
            continue
    if adoption_col is None:
        # Re-raise with full diagnostics using the first cost scenario key.
        adoption_col = find_adoption_column(
            df=df_tare,
            mp=mp,
            cost_scenario=REMDB_COST_SCENARIO_KEYS[0],
            discount_rate_key=discount_rate,
            npv_case=BASE_CASE_NPV_CASE,
        )
    adoption_col_by_mp[mp] = adoption_col

    # Group buildings by 5-digit county FIPS (from the GISJOIN county code) and
    # split each county into the 100% set and the economic-adopter set. A NaN
    # adopter value (excluded home) is not equal to 1.0, so it is left out of
    # constrained -- the intended behavior.
    county_fips = df_tare['county'].apply(gisjoin_to_fips)
    is_adopter = df_tare[adoption_col] == 1.0

    adopter_ids_by_mp[mp] = {}
    for fips, idx in df_tare.groupby(county_fips).groups.items():
        adopter_mask = is_adopter.loc[idx].to_numpy()
        adopter_ids_by_mp[mp][str(fips)] = {
            "all_filtered": list(idx),
            "constrained": list(idx[adopter_mask]),
        }

    n_counties = len(adopter_ids_by_mp[mp])
    n_all = sum(len(v["all_filtered"]) for v in adopter_ids_by_mp[mp].values())
    n_con = sum(len(v["constrained"]) for v in adopter_ids_by_mp[mp].values())
    print(f"[OK] MP{mp}: adoption column {adoption_col}")
    print(f"     Counties: {n_counties:,} | all_filtered: {n_all:,} | "
          f"constrained (NPV>=0): {n_con:,}")

print(f"\n[OK] adopter_ids_by_mp built for MPs: {list(adopter_ids_by_mp.keys())}")


# %%
if GRID_IMPACT_ANALYSIS:
    from buildstock_query import BuildStockQuery  # type: ignore[import-untyped]
    from buildstock_query.schema.query_params import TSQuery
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    logging.getLogger("buildstock_query").setLevel(logging.ERROR)

    # ---------- AWS Credentials and BuildStockQuery ----------
    session = boto3.session.Session()
    aws_region = session.region_name
    try:
        sts = session.client("sts")
        aws_identity = sts.get_caller_identity()
        print(f"""
          [OK] AWS credentials valid
            Account : {aws_identity['Account']}
            ARN     : {aws_identity['Arn']}
            Region  : {aws_region}
          """)
    except NoCredentialsError:
        raise RuntimeError(
            "AWS credentials not found. Run `aws configure` or set "
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY."
        )
    except ClientError as e:
        raise RuntimeError(f"AWS STS call failed: {e}")

    my_run = BuildStockQuery(
        workgroup="resstock-euss",
        db_name="euss-oedi",
        table_name="resstock_amy2018_release_1_1",
        db_schema="resstock_oedi",
        buildstock_type="resstock",
        skip_reports=True,
    )
    print(f"[OK] BuildStockQuery initialized: {type(my_run).__name__}")

    # ---------- Allegheny County test case ----------
    allegheny_buildings_by_mp = []
    for mp in selected_mps:
        if TEST_FIPS not in adopter_ids_by_mp[mp]:
            raise KeyError(
                f"MP{mp}: Allegheny County FIPS {TEST_FIPS} not found in "
                "adopter_ids_by_mp."
            )
        allegheny_buildings_by_mp.append(
            adopter_ids_by_mp[mp][TEST_FIPS]["all_filtered"]
        )
    allegheny_bldg_ids = sorted(set().union(*allegheny_buildings_by_mp))

    print(
        f"[OK] Allegheny County baseline bldg_ids (union across MPs "
        f"{selected_mps}): {len(allegheny_bldg_ids):,d}"
    )

    # ---------- Step 5: Baseline timeseries ----------
    print("\nQuerying baseline timeseries (upgrade=0)...")
    t_start = time.perf_counter()
    ts_query_baseline = TSQuery(
        enduses=[ELEC_TOTAL_COL],
        restrict=[("bldg_id", allegheny_bldg_ids)],
        upgrade_id="0",
        timestamp_grouping_func="hour",
        group_by=[BLDG_ID_COL],
        split_enduses=False,
    )

    df_ts_baseline_allegheny = my_run.agg.aggregate_timeseries(
        params=ts_query_baseline
    )
    query_time_s = time.perf_counter() - t_start

    df_ts_baseline_allegheny = df_ts_baseline_allegheny.rename(
        columns={BSQ_ELEC_COL: "baseline_kwh"}
    )
    df_ts_baseline_allegheny["baseline_kwh"] = (
        df_ts_baseline_allegheny["baseline_kwh"].astype(np.float32)
    )
    df_ts_baseline_allegheny = df_ts_baseline_allegheny.sort_values(
        [BLDG_ID_COL, TIMESTAMP_COL]
    ).reset_index(drop=True)
    df_ts_baseline_allegheny["hour"] = (
        df_ts_baseline_allegheny.groupby(BLDG_ID_COL).cumcount() + 1
    )

    n_bldgs = df_ts_baseline_allegheny[BLDG_ID_COL].nunique()
    n_hours_per_bldg = df_ts_baseline_allegheny.groupby(BLDG_ID_COL).size()
    weight_val = df_ts_baseline_allegheny["units_count"].iloc[0]

    print(f"\n========== df_ts_baseline_allegheny summary ==========")
    print(f"  Rows       : {len(df_ts_baseline_allegheny):,d}")
    print(f"  Buildings  : {n_bldgs:,d}")
    print(
        f"  Hours/bldg : {n_hours_per_bldg.min():,d} - "
        f"{n_hours_per_bldg.max():,d}"
    )
    print(
        f"  kWh range  : {df_ts_baseline_allegheny['baseline_kwh'].min():.3f} "
        f"to {df_ts_baseline_allegheny['baseline_kwh'].max():.3f}"
    )
    print(f"  BSQ weight : {weight_val:.6f}")
    print(f"  Query time : {query_time_s:.2f} s")

    assert n_hours_per_bldg.min() == 8760
    assert n_hours_per_bldg.max() == 8760
    print("[OK] Step 5 PASSED")

    # ---------- Step 6: Upgrade timeseries for each selected MP ----------
    df_ts_upgrade_allegheny_by_mp = {}
    for mp in selected_mps:
        print(f"\nQuerying upgrade timeseries (upgrade={mp})...")
        t_start = time.perf_counter()
        ts_query_upgrade = TSQuery(
            enduses=[ELEC_TOTAL_COL],
            restrict=[("bldg_id", allegheny_bldg_ids)],
            upgrade_id=str(mp),
            timestamp_grouping_func="hour",
            group_by=[BLDG_ID_COL],
            split_enduses=False,
        )

        df_ts_upgrade = my_run.agg.aggregate_timeseries(params=ts_query_upgrade)
        query_time_s = time.perf_counter() - t_start

        df_ts_upgrade = df_ts_upgrade.rename(
            columns={BSQ_ELEC_COL: "retrofit_kwh"}
        )
        df_ts_upgrade["retrofit_kwh"] = df_ts_upgrade[
            "retrofit_kwh"
        ].astype(np.float32)
        df_ts_upgrade = df_ts_upgrade.sort_values(
            [BLDG_ID_COL, TIMESTAMP_COL]
        ).reset_index(drop=True)
        df_ts_upgrade["hour"] = (
            df_ts_upgrade.groupby(BLDG_ID_COL).cumcount() + 1
        )

        baseline_bldgs = set(
            df_ts_baseline_allegheny[BLDG_ID_COL].unique()
        )
        upgrade_bldgs = set(df_ts_upgrade[BLDG_ID_COL].unique())
        only_in_baseline = baseline_bldgs - upgrade_bldgs
        only_in_upgrade = upgrade_bldgs - baseline_bldgs

        print(f"\n========== df_ts_upgrade_allegheny (MP{mp}) summary ==========")
        print(f"  Rows            : {len(df_ts_upgrade):,d}")
        print(f"  Buildings       : {len(upgrade_bldgs):,d}")
        print(
            f"  Hours/bldg      : {df_ts_upgrade.groupby(BLDG_ID_COL).size().min():,d} "
            f"- {df_ts_upgrade.groupby(BLDG_ID_COL).size().max():,d}"
        )
        print(
            f"  kWh range (wtd) : {df_ts_upgrade['retrofit_kwh'].min():.3f} "
            f"to {df_ts_upgrade['retrofit_kwh'].max():.3f}"
        )
        print(f"  Query time (s)  : {query_time_s:.2f}")
        if only_in_baseline:
            print(
                f"  Note: {len(only_in_baseline):,d} buildings have no MP{mp} "
                "upgrade data and will use baseline."
            )
        if only_in_upgrade:
            raise ValueError(
                f"MP{mp}: {len(only_in_upgrade):,d} upgrade buildings "
                "missing baseline."
            )

        df_ts_upgrade_allegheny_by_mp[mp] = df_ts_upgrade

    print("[OK] Step 6 PASSED")


# %% [markdown]
# ### Visuals - Retrofit Impact on County Peak Load

# %%
if GRID_IMPACT_ANALYSIS:

    # ---------- Step 7: Compute scenario profiles ----------
    peak_results_allegheny_by_mp = {}
    df_profiles_by_mp = {}

    # Base case being modeled for the constrained (economic-adopter) scenario.
    # adoption_col_by_mp was built in the adopter-IDs cell above from
    # BASE_CASE_NPV_CASE; echo it here so the peak numbers below are traceable
    # to the exact NPV case that defined the adopters. (The 100pct scenario uses
    # all filtered buildings and does not depend on the NPV case.)

    from cmu_tare_model.utils.column_names import BASE_CASE_NPV_CASE
    print(f"Grid impact base case (constrained adopters): {BASE_CASE_NPV_CASE}")

    for mp in selected_mps:
        print(
            f"\nComputing county profiles for MP{mp} "
            f"(adopter column: {adoption_col_by_mp[mp]})..."
        )
        adopter_ids_allegheny = adopter_ids_by_mp[mp][TEST_FIPS]

        df_profile_100pct, peak_100pct = compute_county_scenario_profile(
            df_ts_baseline_allegheny,
            df_ts_upgrade_allegheny_by_mp[mp],
            adopter_bldg_ids=adopter_ids_allegheny["all_filtered"],
        )

        df_profile_constrained, peak_constrained = compute_county_scenario_profile(
            df_ts_baseline_allegheny,
            df_ts_upgrade_allegheny_by_mp[mp],
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

        print(f"\nAllegheny peak results (MP{mp})")
        for scenario, p in peak_results_allegheny_by_mp[mp].items():
            print(f"  [{scenario}] adopters: {p['n_adopters']:,d} / {p['n_total_buildings']:,d}")
            print(f"    baseline peak : {p['baseline_peak_mw']:.2f} MW @ hour {p['peak_hour_baseline']}")
            print(f"    scenario peak : {p['scenario_peak_mw']:.2f} MW @ hour {p['peak_hour_scenario']}")
            print(f"    delta         : {p['delta_mw']:+.2f} MW")

        assert len(df_profile_100pct) == 8760
        assert len(df_profile_constrained) == 8760

    print(f"\n[OK] Step 7 PASSED -- peak_results_allegheny_by_mp.keys() = {list(peak_results_allegheny_by_mp.keys())}")

    # ---------- Optional visualization: 2x2, white background (scoped locally) ----------
    # Scope the white style to THIS figure only via sns.axes_style, independent
    # of the global sns.set_theme(style='white') set at the top of the notebook,
    # so this figure's styling stays fixed even if that global changes later.
    scenarios = ["100pct", "constrained"]
    scenario_labels = ["100% Adoption", "ONLY Economic Adopters"]
    mp_labels = {
        3: "MIN-efficiency ASHP Retrofit",
        4: "HIGH-efficiency ASHP Retrofit",
    }
    subplot_title_fontsize = 14   # bump this to taste
    tick_label_fontsize = 13      # x/y tick font

    # Month x-axis: hour is hour-of-year. Put one tick at the MIDDLE of each
    # month's block so the label reads as "which month", not "start of month".
    # Non-leap year (matches the 8,760-row profile). Hours are cumulative.
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    _starts_days, _running = [], 0
    for _d in days_in_month:
        _starts_days.append(_running)
        _running += _d
    # First hour of each month (cumulative day count x 24). Ticks land at the
    # left edge of each month rather than its center.
    month_start_hours = [s * 24 for s in _starts_days]

    with sns.axes_style("white"):
        # sharey='all' -> all four panels share a single y-scale, so peak MW is
        # directly comparable across both scenarios (100% vs economic adopters)
        # and both measure packages.
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey='all')
        fig.patch.set_facecolor("white")

        for row_idx, (scenario, scenario_label) in enumerate(
                zip(scenarios, scenario_labels)):
            for col_idx, mp in enumerate(selected_mps):
                ax = axes[row_idx, col_idx]
                ax.set_facecolor("white")
                df_profile = df_profiles_by_mp[mp][scenario]
                peak_result = peak_results_allegheny_by_mp[mp][scenario]
                plot_demand_panel(ax, df_profile, peak_result, mp, scenario_label)
                ax.set_title(
                    f"{mp_labels.get(mp, f'MP{mp}')} | {scenario_label}",
                    fontsize=subplot_title_fontsize,
                )

                # --- Override x-axis to months + enlarge tick labels ---
                h0 = df_profile["hour"].min()
                ax.set_xticks([h0 + m for m in month_start_hours])
                ax.set_xticklabels(month_labels)
                ax.set_xlim(h0, h0 + 8760)
                ax.set_xlabel("Month", fontsize=14)
                ax.tick_params(labelsize=tick_label_fontsize)

        plt.tight_layout()
        if SAVE_FIGURES:
            out_path = os.path.join(
                PROJECT_ROOT,
                "outputs",
                f"allegheny_demand_profiles_MP{'_'.join(str(m) for m in selected_mps)}.png",
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
            print(f"[OK] Figure saved: {out_path}")
        plt.show()


# %%


# %%
import importlib
from cmu_tare_model.grid_impact import peak_load_functions

importlib.reload(peak_load_functions)

# Re-bind the names pulled in at the top of the notebook so the plotting
# cell below picks up the reloaded plot_demand_panel (larger axis/tick
# fonts) without restarting the kernel or re-running the model.
from cmu_tare_model.grid_impact.peak_load_functions import (
    find_adoption_column,
    extract_adopter_ids,
    compute_county_scenario_profile,
    plot_demand_panel,
)


# %%
# ---------- Optional visualization: 2x2, white background (scoped locally) ----------
# Scope the white style to THIS figure only via sns.axes_style, independent
# of the global sns.set_theme(style='white') set at the top of the notebook,
# so this figure's styling stays fixed even if that global changes later.
from matplotlib.lines import Line2D

scenarios = ["constrained", "100pct"]
scenario_labels = ["ONLY Economic Adopters", "100% Adoption"]
mp_labels = {
    3: "MIN-efficiency ASHP Retrofit",
    4: "HIGH-efficiency ASHP Retrofit",
}
subplot_title_fontsize = 18   # bump this to taste
tick_label_fontsize = 16      # x/y tick font

# Month x-axis: hour is hour-of-year. Put one tick at the MIDDLE of each
# month's block so the label reads as "which month", not "start of month".
# Non-leap year (matches the 8,760-row profile). Hours are cumulative.
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_starts_days, _running = [], 0
for _d in days_in_month:
    _starts_days.append(_running)
    _running += _d
# First hour of each month (cumulative day count x 24). Ticks land at the
# left edge of each month rather than its center.
month_start_hours = [s * 24 for s in _starts_days]

with sns.axes_style("white"):
    # sharey='all' -> all four panels share a single y-scale, so peak MW is
    # directly comparable across both scenarios (economic adopters vs 100%)
    # and both measure packages.
    #
    # Rows are measure packages (MP3 top, MP4 bottom); columns are scenarios
    # (economic adopters left, 100% adoption right).
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey='all')
    fig.patch.set_facecolor("white")

    for row_idx, mp in enumerate(selected_mps):
        for col_idx, (scenario, scenario_label) in enumerate(
                zip(scenarios, scenario_labels)):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("white")
            df_profile = df_profiles_by_mp[mp][scenario]
            peak_result = peak_results_allegheny_by_mp[mp][scenario]
            plot_demand_panel(ax, df_profile, peak_result, mp, scenario_label)
            ax.set_title(
                f"{mp_labels.get(mp, f'MP{mp}')} | {scenario_label}",
                fontsize=subplot_title_fontsize,
            )

            # --- Override x-axis to months + enlarge tick labels ---
            h0 = df_profile["hour"].min()
            ax.set_xticks([h0 + m for m in month_start_hours])
            ax.set_xticklabels(month_labels)
            ax.set_xlim(h0, h0 + 8760)
            ax.set_xlabel("Month", fontsize=17)
            ax.tick_params(labelsize=tick_label_fontsize)

    # --- Shared legend, bottom center, drawn as a fancy box ---
    # Proxy handles only (no data) -- the real lines/markers are drawn per
    # panel by plot_demand_panel. The black peak-X marker is left out of the
    # legend on purpose (self-evident on the panels, and a fifth entry would
    # add a row and compress the figure) -- only solid vs. dashed and
    # red vs. blue are explained here.
    # Order is [solid_red, dashed_red, solid_blue, dashed_blue] so that
    # matplotlib's column-major legend fill (with ncol=2) lays them out as
    # two rows -- row 1 solid red/blue, row 2 dashed red/blue -- matching
    # "Solid Red | Solid Blue" then "Dashed Red | Dashed Blue".
    legend_handles = [
        Line2D([0], [0], color="tab:red", linewidth=2.5, linestyle="-"),
        Line2D([0], [0], color="tab:red", linewidth=2.5, linestyle="--"),
        Line2D([0], [0], color="tab:blue", linewidth=2.5, linestyle="-"),
        Line2D([0], [0], color="tab:blue", linewidth=2.5, linestyle="--"),
    ]
    legend_labels = [
        "Existing HVAC",
        "Peak Existing HVAC",
        "Post-Retrofit",
        "Peak Post-Retrofit",
    ]
    fig_legend = fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=2,
        fontsize=16,
        title="Residential Electricity Load (MW) for Allegheny County, PA",
        title_fontsize=17,
        frameon=True,
        fancybox=True,
        shadow=True,
        facecolor="white",
        edgecolor="0.3",
        framealpha=0.95,
        borderpad=1.1,
        labelspacing=0.9,
        handlelength=2.5,
    )
    fig_legend.get_title().set_fontweight("bold")

    # Extra bottom margin so the legend box has room below the panels.
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    if SAVE_FIGURES:
        out_path = os.path.join(
            PROJECT_ROOT,
            "outputs",
            f"allegheny_demand_profiles_MP{'_'.join(str(m) for m in selected_mps)}.png",
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"[OK] Figure saved: {out_path}")
    plt.show()


# %%


# %%
# =============================================================================
# GRID IMPACT -- Allegheny County baseline heating-fuel distribution table
# =============================================================================
# Baseline heating-fuel breakdown for the four combinations:
#   MP3 / MP4 x constrained (economic adopters) / 100% adoption.
# Each cell shows count and percentage within the scenario; percentages sum to
# 100% per column. adopter_ids_by_mp already holds only Allegheny County IDs
# per FIPS, so they are used directly. Note: main keys DATAFRAMES_BY_MP by
# discount rate only, so the frame is DATAFRAMES_BY_MP[mp]['fixed_base'].
if GRID_IMPACT_ANALYSIS:
    # Collect fuel counts and percentages for all four combinations.
    _fuel_results = {}
    for _mp in selected_mps:
        _df_tare = DATAFRAMES_BY_MP[_mp]['fixed_base']
        if TEST_FIPS not in adopter_ids_by_mp[_mp]:
            raise KeyError(
                f"MP{_mp}: Allegheny FIPS {TEST_FIPS} not in adopter_ids_by_mp."
            )
        for _scenario, _key in [("constrained", "constrained"),
                                ("100pct", "all_filtered")]:
            _ids = set(adopter_ids_by_mp[_mp][TEST_FIPS][_key])
            _counts = (
                _df_tare.loc[_df_tare.index.isin(_ids), 'base_heating_fuel']
                .value_counts()
                .sort_index()
            )
            _pcts = _counts / _counts.sum() * 100
            _fuel_results[(_mp, _scenario)] = {
                "n": len(_ids), "counts": _counts, "pcts": _pcts,
            }

    # All fuel categories seen across the four combinations.
    _all_fuels = sorted(
        set().union(*[v["counts"].index for v in _fuel_results.values()])
    )

    _col_width = 26
    _fuel_width = 20
    print(f"Allegheny County (FIPS {TEST_FIPS}) -- Baseline Heating Fuel "
          f"Distribution")
    print("=" * (_fuel_width + _col_width * 4 + 3))

    # Header row -- one Constrained and one 100% column per measure package.
    _headers = []
    for _mp in selected_mps:
        _n_con = _fuel_results[(_mp, "constrained")]["n"]
        _n_all = _fuel_results[(_mp, "100pct")]["n"]
        _headers.append(f"MP{_mp} Constrained (n={_n_con:,})")
        _headers.append(f"MP{_mp} 100% (n={_n_all:,})")

    print(f"{'Fuel':<{_fuel_width}}", end="")
    for _h in _headers:
        print(f"  {_h:>{_col_width - 2}}", end="")
    print()

    print(f"{'-'*_fuel_width}", end="")
    for _ in _headers:
        print(f"  {'-'*(_col_width-2)}", end="")
    print()

    # Data rows.
    for _fuel in _all_fuels:
        print(f"{_fuel:<{_fuel_width}}", end="")
        for _mp in selected_mps:
            for _scenario in ("constrained", "100pct"):
                _c = _fuel_results[(_mp, _scenario)]["counts"].get(_fuel, 0)
                _p = _fuel_results[(_mp, _scenario)]["pcts"].get(_fuel, 0.0)
                print(f"  {f'{_c:,} ({_p:.1f}%)':>{_col_width - 2}}", end="")
        print()

    # Total row.
    print(f"{'-'*_fuel_width}", end="")
    for _ in _headers:
        print(f"  {'-'*(_col_width-2)}", end="")
    print()
    print(f"{'TOTAL':<{_fuel_width}}", end="")
    for _mp in selected_mps:
        for _scenario in ("constrained", "100pct"):
            _n = _fuel_results[(_mp, _scenario)]["n"]
            print(f"  {f'{_n:,} (100.0%)':>{_col_width - 2}}", end="")
    print()

    print("\n[OK] Baseline heating fuel distribution table complete")


# %% [markdown]
# # Model Runtime

# %%
# Get the current datetime again
end_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Calculate the elapsed time
elapsed_time = datetime.strptime(end_time, "%Y-%m-%d_%H-%M-%S") - datetime.strptime(start_time, "%Y-%m-%d_%H-%M-%S")

# Format the elapsed time
elapsed_seconds = elapsed_time.total_seconds()
elapsed_minutes = int(elapsed_seconds // 60)
elapsed_seconds = int(elapsed_seconds % 60)

# Print the elapsed time
print(f"The code took {elapsed_minutes} minutes and {elapsed_seconds} seconds to execute.")


