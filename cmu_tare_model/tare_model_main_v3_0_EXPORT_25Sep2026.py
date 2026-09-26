# %% [markdown]
# -------------------------------------------------------------------------------------------------------
# # TARE MODEL SCENARIO: 2025 Reference Case
# -------------------------------------------------------------------------------------------------------
# ResStock 2022 Release (EUSS) Post-Retrofit Measure Packages, Heat Pump Adoption: MP3 and MP4
# - MP3: Min-efficiency, single-stage ASHP (15 SEER1, 9 HSPF1) --> (16 SEER1, 9.5 HSPF1) for ENERGY STAR
# - MP4: High-efficiency, variable-speed ASHP (24-29.3 SEER1, 14 HSPF1)
# 
# Reference Case Data and Assumptions:
# - AEO2026 fuel price projections
# - AEO2026 degree-day factors
# - Cambium 2024 MidCase electricity grid
# - Single scenario: '2025 Reference Case'
# 

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
import seaborn as sns

# Project configuration
from config import PROJECT_ROOT

# Model constants - explicit imports for clarity
from cmu_tare_model.constants import (
    VERBOSE,
    SCC_ASSUMPTIONS,
    REMDB_COST_SCENARIO_KEYS,
    VALID_MENU_MPS,
    BLDG_ID_COL,
    TIMESTAMP_COL,
    ELEC_TOTAL_COL,
    BSQ_ELEC_COL,
    TEST_FIPS,
)
from cmu_tare_model.constants import PRIVATE_DISCOUNT_RATE_SHORT_KEYS

# Column name builders
from cmu_tare_model.utils.column_names import (
    NPV_CASE_CATEGORIES,
    BASE_CASE_NPV_CASE
)
from cmu_tare_model.grid_impact.peak_load_functions import (
    find_adoption_column,
    compute_county_scenario_profile,
    plot_county_demand_grid
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
            relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_run_simulation_v3_0.ipynb")

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
# ========== Baseline Scenario: Measure Package 0 (MP0) ==========
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

# The study sample, rebuilt from the include_sample flag the run saved. County
# demand, peaks, and custom weighting use it, since they read ResStock directly.
from cmu_tare_model.constants import RESSTOCK_RELEASE_THIS_RUN
from cmu_tare_model.energy_consumption_and_metadata.study_sample import (
    build_tare_sample_ids,
)
TARE_SAMPLE_IDS = build_tare_sample_ids(
    df_outputs_baseline_home, RESSTOCK_RELEASE_THIS_RUN)
print(f"Study sample: {len(TARE_SAMPLE_IDS['all']):,} rdu in "
      f"{len(TARE_SAMPLE_IDS['by_county']):,} counties")

# ========== Load Measure Packages in VALID_MENU_MPS ==========
NON_BASELINE_MPS = [mp for mp in VALID_MENU_MPS if mp != 0]

# Convenience mapping for downstream code
DATAFRAMES_BY_MP = {}

for mp in NON_BASELINE_MPS:
    DATAFRAMES_BY_MP[mp] = load_measure_package_data(
        mp, output_folder_path, location_id, model_run_date_time
    )

print(f"\nLoaded measure packages: {list(DATAFRAMES_BY_MP.keys())}")

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
# | `True` or 1  | Heat pump covers its incremental cost (or better) from bill savings |
# | `False` or 0 | Valid home that cannot recover the incremental cost from savings alone |
# | `NaN`   | Excluded: invalid baseline fuel/tech or not in this measure package |
# 
# Climate emissions and damages are computed elsewhere and reported as outcomes,
# not as inputs to this decision.

# %%
# =============================================================================
# ECONOMIC ADOPTION -- setup: imports, parameters, and shared inputs
# =============================================================================
# Transported from calculate_postTARE_am_kpis_demand_bill_savings. Column names
# use the nine-case NPV_CASE_CATEGORIES scheme; there is no WTP or cost-scenario
# token in adopter column names.
import geopandas as gpd
from matplotlib.colors import Normalize
from cmu_tare_model.constants import CAPITAL_COST_VALIDATION, GRID_IMPACT_ANALYSIS
from cmu_tare_model.adoption_potential.determine_economic_adoption_potential import (
    economic_adoption_decision,
)
from cmu_tare_model.utils.modeling_params import define_scenario_params
from cmu_tare_model.utils.column_names import create_adoption_col
from cmu_tare_model.adoption_kpis.data_loading import load_euss_baseline
from cmu_tare_model.adoption_kpis.compute_adoption_rate import compute_adoption_rate
from cmu_tare_model.adoption_kpis.visualize_geospatial_data import (
    plot_national_county_choropleth,
    plot_national_county_change_map,
)
from cmu_tare_model.adoption_kpis.data_loading import COUNTY_SHAPEFILE_PATH
from cmu_tare_model.adoption_potential.data_processing.visuals_adoption_dotplot import (
    plot_econ_adoption_dotplot_figure,
    REPLACEMENT_CREDIT_MARKERS,
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

# The adoption analysis runs on every loaded non-baseline measure package.
selected_mps = NON_BASELINE_MPS
HEATING_MEASURE_PACKAGES = selected_mps

# Equipment subtitles used as panel/map titles. Covers this notebook's MP set.
HEATING_MP_SUBTITLES = {
    # 3: 'Single-stage, min-efficiency ASHP (16 SEER1, 9.5 HSPF1)',
    # 4: 'Variable-speed, high-efficiency ASHP (24-29.3 SEER1, 13-14 HSPF1)',
    3: 'Single-stage, minimum-efficiency heat pump',
    4: 'Variable-speed, high-efficiency heat pump',
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
# =============================================================================
# CAPITAL COST VALIDATION
# =============================================================================

if CAPITAL_COST_VALIDATION:
    # Perform capital cost validation using the validate_capital_costs module.
    # Builds the ASHP/Central AC/Furnace consumption-and-size distribution
    # figure and the REMDB v4MID disaggregation workbook (ASHP upgrade,
    # Central AC replacement, NG Furnace replacement) for MP3 and MP4.
    from cmu_tare_model.utils.validate_capital_costs import (
        build_capital_cost_disaggregation_workbook,
        build_capital_cost_distribution_figure,
        run_capital_cost_validation,
    )

    df_mp3_ccv = DATAFRAMES_BY_MP[3]['fixed_base']
    df_mp4_ccv = DATAFRAMES_BY_MP[4]['fixed_base']

    # Consumption/size distributions, color-coded by baseline heating fuel.
    fig_capital_cost_dist = build_capital_cost_distribution_figure(df_mp3_ccv, df_mp4_ccv)
    if SAVE_FIGURES:
        fig_path_base = os.path.join(PROJECT_ROOT, 'figures', 'capital_cost_consumption_size_distributions')
        fig_capital_cost_dist.savefig(f'{fig_path_base}.png', dpi=FIGURE_DPI, bbox_inches='tight')
        fig_capital_cost_dist.savefig(f'{fig_path_base}.pdf', dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {fig_path_base}.png / .pdf")
    plt.show()

    # REMDB v4MID disaggregation workbook -- matches the Equipment_Installed_TARE
    # reference workbook's sheet names and block layout, minus its v3 columns.
    results_mp3_ccv = run_capital_cost_validation(df=df_mp3_ccv, menu_mp=3, cost_scenarios=['v4MID'])
    results_mp4_ccv = run_capital_cost_validation(df=df_mp4_ccv, menu_mp=4, cost_scenarios=['v4MID'])
    wb_capital_cost = build_capital_cost_disaggregation_workbook(results_mp3_ccv, results_mp4_ccv)

    workbook_path = (
        r'C:\Users\jorda\Desktop\CMU\Scott-Trane Externship\Cost Data'
        r'\Equipment_Installed_TARE_v4MID_current_2026-09-16.xlsx'
    )
    wb_capital_cost.save(workbook_path)
    print(f"Saved: {workbook_path}")


# %%
# County-level economic adoption rate. Uses BASE_CASE_NPV_CASE
# ('heatingLCC_coolingLCC_unsub' -- unsubsidized, both avoided replacements credited).
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
    # adopter_tiers=[True] counts 1.0 (adopter) vs 0.0
    # NaN (excluded) rows are ignored automatically by compute_adoption_rate.
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
# =============================================================================
# ECONOMIC ADOPTION (Choropleth Map) -- Unsub, heating and cooling LCC
# =============================================================================

# The choropleth is skipped if the county shapefile is unavailable.
if gdf_counties_raw is not None:
    _adopt_cmap = 'Greens'

    # Colorbase 0 to 100% adoption potential. 
    _adopt_norm = Normalize(vmin=0, vmax=100)

    print("\n--- Summary: adoption_rate_pct ---")
    for mp in selected_mps:
        _v = econ_adoption_rate_results[mp]['adoption_rate_pct'].dropna()
        _pct_high = (_v >= 50).mean() * 100
        print(f"  MP{mp}: n={len(_v):,} counties | "
              f"min={_v.min():.1f}% | med={_v.median():.1f}% | "
              f"mean={_v.mean():.1f}% | max={_v.max():.1f}% | "
              f"{_pct_high:.1f}% of counties >= 50% adoption potential")

    # Visualize the county-level economic adoption rate as a choropleth. 
    plot_national_county_choropleth(
        gdf_counties_raw, econ_adoption_rate_results,
        column='adoption_rate_pct',
        title_template=HEATING_MP_SUBTITLES,
        cbar_label='Share of households recovering electrification premium through discounted operational savings (%)',
        cmap=_adopt_cmap, norm=_adopt_norm,
        selected_mps=selected_mps,
        save_figure=SAVE_FIGURES,
        output_filename='county_econ_adoption_rate_combined.png',
    )

    print("[OK] Economic adoption choropleth generated")
else:
    print("[WARN] Adoption choropleth skipped -- county shapefile not available")

# %% [markdown]
# ## Economic Adoption Potential Dotplot
# ### Sensitivity of Heat Pump Adoption to Replacement Cost Offset Assumptions

# %%
# =============================================================================
# ECONOMIC ADOPTION (Dotplot) -- Replacement Cost Offset Sensitivity
# =============================================================================

if not HEATING_MEASURE_PACKAGES:
    print("No active heating measure packages -- skipping economic adoption dotplot.")
else:
    # Set Seaborn theme for consistent styling across all dotplot figures. 
    sns.set_theme(font='sans-serif', style='whitegrid')

    plot_econ_adoption_dotplot_figure(
        HEATING_MEASURE_PACKAGES, DATAFRAMES_BY_MP, discount_rate, _COST,
        HEATING_MP_SUBTITLES,
        build_df_kwargs=dict(rebate_vintage='unsub'),
        custom_tier_markers=REPLACEMENT_CREDIT_MARKERS,
        legend_handles=build_replacement_credit_legend_handles(fill_markers=False),
        summary_header='economic adoption summary (National, unsubsidized)',
        annotation_x_offset_pts=14.0,
        annotation_y_offset_pts=8.0,
        save_figure=SAVE_FIGURES,
        output_dir=PROJECT_ROOT,
        output_filename=f'figure6_econ_adoption_dotplot_{location_id}',
        figure_dpi=FIGURE_DPI,
    )


# %% [markdown]
# ### Sensitivity of Heat Pump Adoption to Rebate Policy Scenario

# %%
# =============================================================================
# ECONOMIC ADOPTION (Dotplot) -- Rebate Policy Scenario Sensitivity
# =============================================================================
# Hold this replacement-credit scenario fixed; the three markers vary the rebate
# policy scenario. Switch to 'heatingLCC_coolingSavings' for the heating-only view.
_FIXED_CREDIT = 'heatingLCC_coolingLCC'

if not HEATING_MEASURE_PACKAGES:
    print("No active heating measure packages -- skipping rebate-policy dotplot.")
else:
    # Set Seaborn theme for consistent styling across all dotplot figures. 
    sns.set_theme(font='sans-serif', style='whitegrid')

    plot_econ_adoption_dotplot_figure(
        HEATING_MEASURE_PACKAGES, DATAFRAMES_BY_MP, discount_rate, _COST,
        HEATING_MP_SUBTITLES,
        build_df_kwargs=dict(
            shape_by='rebate_policy_scenario',
            fixed_replacement_credit_scenario=_FIXED_CREDIT,
        ),
        custom_tier_markers=REBATE_POLICY_SCENARIO_MARKERS,
        legend_handles=build_rebate_policy_scenario_legend_handles(fill_markers=False),
        summary_header=f'adoption by rebate policy scenario ({_FIXED_CREDIT})',
        annotation_x_offset_pts=1.0,
        annotation_y_offset_pts=8.0,
        save_figure=SAVE_FIGURES,
        output_dir=PROJECT_ROOT,
        output_filename=f'econ_adoption_dotplot_rebate_policy_scenario_{_FIXED_CREDIT}',
        figure_dpi=FIGURE_DPI,
    )


# %% [markdown]
# ## County Level Maps

# %%
# =============================================================================
# Retrofit impact on electricity demand and operating cost (county choropleths)
# =============================================================================
from cmu_tare_model.adoption_kpis.data_loading import load_euss_upgrade, mp_to_upgrade
from cmu_tare_model.adoption_kpis.demand import compute_scenario_demand, aggregate_demand

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
        pd.DataFrame({'county': df_tare['county'],'operating_cost_pct_change': df_tare[pct_col]})
        .groupby('county')['operating_cost_pct_change']
        .median()
        .reset_index()
    )
    print(f"  Per-home valid records: {df_tare[pct_col].notna().sum():,} | Counties: {len(df_county):,}")
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
        df_baseline=df_baseline, df_upgrade=upgrade_data[mp],
        sample_bldg_ids=TARE_SAMPLE_IDS['all'], fuel_filter=None, verbose=True,
    )
    
    demand_results[mp] = aggregate_demand(
        df_demand=df_demand, geo_level='county', verbose=True,
    )


# %%
# =============================================================================
# Operating cost percent change (county-level) -- separate function call
# =============================================================================
if gdf_counties_raw is not None:
    # ---- Operating cost percent change (county-level) ----
    plot_national_county_change_map(
        gdf_counties_raw, bill_savings_results,
        column='operating_cost_pct_change',
        cbar_label='Post-retrofit change in average annual operating cost, relative to baseline equipment (%)',
        cmap='RdBu_r',
        title_template=HEATING_MP_SUBTITLES,
        selected_mps=selected_mps,
        positive_direction='HP saves money (< 0)',
        norm_label='Operating cost %',
        norm_unit='%',
        save_figure=SAVE_FIGURES,
        output_filename='county_bill_pct_change_combined.png',
    )
    

# %%
# =============================================================================
# Electricity demand change (GWh, county-level) -- separate function call
# =============================================================================
if gdf_counties_raw is not None:
    # ---- Electricity demand change (GWh, county-level) ----
    plot_national_county_change_map(
        gdf_counties_raw, demand_results,
        column='elec_change_gwh',
        cbar_label='Post-retrofit change in 2025 annual electricity demand, relative to baseline (GWh)',
        cmap='coolwarm',
        title_template=HEATING_MP_SUBTITLES,
        selected_mps=selected_mps,
    positive_direction='increase',
    norm_label='Demand GWh',
    norm_unit=' GWh',
    save_figure=SAVE_FIGURES,
    output_filename='county_elec_demand_gwh_combined.png',
)


# %%
# =============================================================================
# Electricity demand percent change (county-level) -- separate function call
# =============================================================================
if gdf_counties_raw is not None:
    # ---- Electricity demand percent change (county-level) ----
    plot_national_county_change_map(
        gdf_counties_raw, demand_results,
        column='pct_elec_demand_change',
        cbar_label='Post-retrofit change in 2025 annual electricity demand, relative to baseline (%)',
        cmap='coolwarm',
        title_template=HEATING_MP_SUBTITLES,
        selected_mps=selected_mps,
        positive_direction='increase',
        norm_label='Demand %',
        norm_unit='%',
        save_figure=SAVE_FIGURES,
        output_filename='county_elec_demand_pct_combined.png',
    )


# %%
if CAPITAL_COST_VALIDATION:
    # Perform capital cost validation using the validate_capital_costs module.
    # Builds the ASHP/Central AC/Furnace consumption-and-size distribution
    # figure and the REMDB v4MID disaggregation workbook (ASHP upgrade,
    # Central AC replacement, NG Furnace replacement) for MP3 and MP4.
    from cmu_tare_model.utils.validate_capital_costs import (
        build_capital_cost_disaggregation_workbook,
        build_capital_cost_distribution_figure,
        run_capital_cost_validation,
    )

    df_mp3_ccv = DATAFRAMES_BY_MP[3]['fixed_base']
    df_mp4_ccv = DATAFRAMES_BY_MP[4]['fixed_base']

    # Consumption/size distributions, color-coded by baseline heating fuel.
    fig_capital_cost_dist = build_capital_cost_distribution_figure(df_mp3_ccv, df_mp4_ccv)
    if SAVE_FIGURES:
        fig_path_base = os.path.join(PROJECT_ROOT, 'figures', 'capital_cost_consumption_size_distributions')
        fig_capital_cost_dist.savefig(f'{fig_path_base}.png', dpi=FIGURE_DPI, bbox_inches='tight')
        fig_capital_cost_dist.savefig(f'{fig_path_base}.pdf', dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved: {fig_path_base}.png / .pdf")
    plt.show()

    # REMDB v4MID disaggregation workbook -- matches the Equipment_Installed_TARE
    # reference workbook's sheet names and block layout, minus its v3 columns.
    results_mp3_ccv = run_capital_cost_validation(df=df_mp3_ccv, menu_mp=3, cost_scenarios=['v4MID'])
    results_mp4_ccv = run_capital_cost_validation(df=df_mp4_ccv, menu_mp=4, cost_scenarios=['v4MID'])
    wb_capital_cost = build_capital_cost_disaggregation_workbook(results_mp3_ccv, results_mp4_ccv)

    workbook_path = (
        r'C:\Users\jorda\Desktop\CMU\Scott-Trane Externship\Cost Data'
        r'\Equipment_Installed_TARE_v4MID_current_2026-09-16.xlsx'
    )
    wb_capital_cost.save(workbook_path)
    print(f"Saved: {workbook_path}")


# %%
import importlib
import cmu_tare_model.utils.data_visualization_histograms
import cmu_tare_model.utils.validate_capital_costs

importlib.reload(cmu_tare_model.utils.data_visualization_histograms)
importlib.reload(cmu_tare_model.utils.validate_capital_costs)

from cmu_tare_model.utils.validate_capital_costs import (
    build_capital_cost_disaggregation_workbook,
    build_capital_cost_distribution_figure,
    run_capital_cost_validation,
    build_furnace_ashp_consumption_comparison_figure,
    build_furnace_ashp_metric_comparison_figure,
)


# %%
fig_capacity = build_furnace_ashp_metric_comparison_figure(
    df_mp3_ccv, df_mp4_ccv,
    baseline_col='base_size_heating_system_primary_k_btu_h',
    mp3_col='size_heating_system_primary_k_btu_h',
    mp4_col='size_heating_system_primary_k_btu_h',
    x_label='Heating Capacity (kBTU/h)',
    metric_label='Heating Capacity',
)
plt.show()


# %%
# fig_consumption = build_furnace_ashp_consumption_comparison_figure(
#     df_mp3_ccv, df_mp4_ccv,
#     baseline_col='base_annual_energy_consumption_kwh',

# %%
fig_cost = build_furnace_ashp_metric_comparison_figure(
    df_mp3_ccv, df_mp4_ccv,
    baseline_col='baseline_heating_avg_annual_fuel_cost',
    mp3_col='ref2025_mp3_heating_avg_annual_fuel_cost',
    mp4_col='ref2025_mp4_heating_avg_annual_fuel_cost',
    x_label='Annual Heating Cost ($)',
    metric_label='Annual Heating Cost',
)
plt.show()


# %%
import importlib
import cmu_tare_model.utils.validate_capital_costs as validate_capital_costs

importlib.reload(validate_capital_costs)
from cmu_tare_model.utils.validate_capital_costs import (
    build_ashp_primary_vs_total_consumption_figure,
)

fig_primary_vs_total = build_ashp_primary_vs_total_consumption_figure(
    df_mp3_ccv, df_mp4_ccv,
)
plt.show()


# %%
import importlib
import cmu_tare_model.utils.validate_capital_costs as validate_capital_costs

importlib.reload(validate_capital_costs)
from cmu_tare_model.utils.validate_capital_costs import (
    build_ashp_primary_vs_total_consumption_figure,
)

fig_primary_vs_total = build_ashp_primary_vs_total_consumption_figure(
    df_mp3=df_mp3_ccv,
    df_mp4=df_mp4_ccv,
    sharex=True,
    sharey=True,
)
plt.show()


# %% [markdown]
# # GRID IMPACT ANALYSIS

# %%
# =============================================================================
# GRID IMPACT -- build adopter building IDs by measure package and county
# =============================================================================
# Produces adopter_ids_by_mp, consumed by the BSQ timeseries and county-profile
# cells below. Two adopter sets per county:
#   all_filtered -- every study-sample home in the county (100% adoption bound)
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

    # Split each county's sample homes into the 100% set (every sample home)
    # and the economic-adopter set. A NaN adopter value is not 1.0, so it is
    # left out of constrained.
    is_adopter = df_tare[adoption_col] == 1.0

    adopter_ids_by_mp[mp] = {}
    for fips, county_sample_ids in TARE_SAMPLE_IDS['by_county'].items():
        county_is_adopter = is_adopter.loc[county_sample_ids].to_numpy()
        adopter_ids_by_mp[mp][fips] = {
            "all_filtered": list(county_sample_ids),
            "constrained": [
                bldg_id
                for bldg_id, is_econ_adopter in zip(county_sample_ids, county_is_adopter)
                if is_econ_adopter
            ],
        }

    # Every adopter must be a sample home; a mismatch means the frames and
    # the saved sample come from different runs.
    n_adopters_in_sample = sum(
        len(county_ids["constrained"]) for county_ids in adopter_ids_by_mp[mp].values())
    n_adopters_total = int(is_adopter.sum())
    if n_adopters_in_sample != n_adopters_total:
        raise ValueError(f"MP{mp}: {n_adopters_total:,} adopters but "
                         f"{n_adopters_in_sample:,} in the sample")

print(f"\n[OK] adopter_ids_by_mp built for MPs: {list(adopter_ids_by_mp.keys())}")


# %% [markdown]
# ### Grid Impact Analysis -- AWS troubleshooting (optional)

# %%
# ============================================================================
# BSQ / AWS troubleshooting diagnostic (optional)
# ----------------------------------------------------------------------------
# Set RUN_BSQ_DIAGNOSTIC = True to check the AWS setup before running the
# grid-impact cells below. It runs seven checks -- buildstock_query install,
# AWS credentials, the Athena workgroup, the query-result bucket, the Glue
# tables, BuildStockQuery initialization, and one real end-to-end query -- and
# prints what to fix for whichever one fails first.
#
# Leave it False for normal runs. It issues two small Athena queries, so it
# takes a few seconds and a fraction of a cent.
#
# The same checks from a terminal:
#     python -m cmu_tare_model.grid_impact.diagnose_bsq_aws
#
# What each check means: cmu_tare_model/docs/BSQ_AWS_SETUP.md
# ============================================================================
RUN_BSQ_DIAGNOSTIC = True

if RUN_BSQ_DIAGNOSTIC:
    from cmu_tare_model.grid_impact.diagnose_bsq_aws import run_diagnostic

    bsq_diagnostic_exit_code = run_diagnostic()

    if bsq_diagnostic_exit_code != 0:
        print(
            "\n[ACTION NEEDED] A check above did not pass. The grid-impact "
            "cells below will not run until it is resolved.\n"
            "                See cmu_tare_model/docs/BSQ_AWS_SETUP.md, Part 2."
        )
else:
    print(
        "[SKIP] BSQ / AWS diagnostic not run. "
        "Set RUN_BSQ_DIAGNOSTIC = True above to troubleshoot the AWS setup."
    )


# %% [markdown]
# ### Grid Impact Analysis using BuildStockQuery and AWS-hosted ResStock data

# %%
# custom_weighting picks which of the two weighting paths this section runs.
# False keeps today's behavior (one county, BSQ's own uniform weight); True
# switches to Tamar's matched tax-parcel weights. See
# compute_county_scenario_profile for how each path is actually applied.
CUSTOM_WEIGHTING = False

if GRID_IMPACT_ANALYSIS:
    from cmu_tare_model.grid_impact.peak_load_functions import (
        print_heating_fuel_distribution_table,
    )

    # TODO: ADD ANOTHER LAYER FOR FEEDERS
    # DICTIONARY: FEEDER --> BUILDING IDS --> WEIGHT PER BUILDING ID
    if CUSTOM_WEIGHTING:
        # build_weight_dict_from_mapping turns Tamar's parcel-match CSV into
        # a {bldg_id: weight} lookup -- the weight is how many real tax
        # parcels matched to that representative building.
        from cmu_tare_model.grid_impact.build_parcel_frame import (
            build_weight_dict_from_mapping,
        )

        # This file lives outside version control (see .gitignore) because
        # it is Tamar's working data, not a TARE model output.
        tamar_mapping_path = os.path.join(
            PROJECT_ROOT,
            "cmu_tare_model",
            "tamar_grid_impact",
            "PSM_output_buildYear07_09_2026.csv",
        )
        df_tamar_mapping = pd.read_csv(tamar_mapping_path)
        case_study_weight_dict = build_weight_dict_from_mapping(df_tamar_mapping)

        # Keep only matched buildings in the study sample, so the case study
        # describes the same homes as every other result.
        n_matched_buildings = len(case_study_weight_dict)
        sample_bldg_id_set = set(TARE_SAMPLE_IDS['all'])
        case_study_weight_dict = {
            bldg_id: parcel_weight
            for bldg_id, parcel_weight in case_study_weight_dict.items()
            if bldg_id in sample_bldg_id_set
        }
        print(f"[OK] Custom weighting: {len(case_study_weight_dict):,d} of "
              f"{n_matched_buildings:,d} matched buildings are in the study sample.")

        # The building scope for this weighting mode is simply every building
        # the match dict covers -- there is no county filter to apply.
        case_study_bldg_ids = sorted(case_study_weight_dict.keys())

        print(
            f"[OK] Custom weighting: {len(case_study_bldg_ids):,d} representative "
            f"buildings matched from Tamar's tax-parcel data "
            f"({sum(case_study_weight_dict.values()):,.0f} real parcels total)."
        )
    else:
        # Prompted here instead of imported as a fixed constant (TEST_FIPS),
        # so this section can be re-run for any county, not only the
        # Allegheny County case study used in the paper.
        case_study_fips = str(
            input(
                "Enter the 5-digit county FIPS code for the grid-impact case "
                "study (e.g. 42003 for Allegheny County, PA): "
            )
        ).strip()

        # case_study_weight_dict stays defined (as None) even in this branch,
        # so downstream cells can check CUSTOM_WEIGHTING without also having
        # to guard against a missing variable.
        case_study_weight_dict = None

        # Union the county's buildings across every selected measure package,
        # since the baseline query below only needs to run once per building,
        # not once per MP.
        case_study_buildings_by_mp = []
        for mp in selected_mps:
            if case_study_fips not in adopter_ids_by_mp[mp]:
                raise KeyError(
                    f"MP{mp}: county FIPS {case_study_fips} not found in "
                    "adopter_ids_by_mp."
                )
            case_study_buildings_by_mp.append(
                adopter_ids_by_mp[mp][case_study_fips]["all_filtered"]
            )
        case_study_bldg_ids = sorted(set().union(*case_study_buildings_by_mp))

        print(
            f"[OK] County FIPS {case_study_fips} baseline bldg_ids (union "
            f"across MPs {selected_mps}): {len(case_study_bldg_ids):,d}"
        )

        # Only meaningful for one county -- print the baseline heating-fuel
        # breakdown for this case study now, before the BSQ queries below.
        print_heating_fuel_distribution_table(
            {mp: DATAFRAMES_BY_MP[mp]["fixed_base"] for mp in selected_mps},
            adopter_ids_by_mp,
            selected_mps,
            case_study_fips,
        )


# %%
if GRID_IMPACT_ANALYSIS:
    from buildstock_query import BuildStockQuery  # type: ignore[import-untyped]
    from buildstock_query.schema.query_params import TSQuery
    from cmu_tare_model.grid_impact.peak_load_functions import (
        prepare_bsq_timeseries,
        summarize_hourly_timeseries,
        check_upgrade_building_coverage,
    )

    logging.getLogger("buildstock_query").setLevel(logging.ERROR)

    # sample_weight_override=1 returns RAW kWh -- required for custom
    # weighting, where the per-building weight is applied in Python later.
    # None (default) keeps BSQ's own uniform weighting, as before.
    #
    # Note: BSQ 0.2.0 (pinned here) has no query_unload_s3_bucket parameter
    # -- a colleague's 0.3.0 does. Don't add it below until this project
    # upgrades, or it will raise TypeError.
    #
    # AWS credentials and the Athena bucket are checked by the
    # RUN_BSQ_DIAGNOSTIC cell above -- run that first, not repeated here.
    #
    # TODO: CHANGE THIS TO TAKE IN USER INPUT FOR BSQ INITIALIZATION PARAMETERS 
    # (WORKGROUP, DB_NAME, TABLE_NAME, DB_SCHEMA)
    my_run = BuildStockQuery(
        workgroup="resstock-euss",
        db_name="euss-oedi",
        table_name="resstock_amy2018_release_1_1",
        db_schema="resstock_oedi",
        buildstock_type="resstock",
        sample_weight_override=1 if CUSTOM_WEIGHTING else None,
        skip_reports=True,
    )
    print(f"[OK] BuildStockQuery initialized: {type(my_run).__name__}")

    # ---------- Step 5: Baseline timeseries ----------
    print("\nQuerying baseline timeseries (upgrade=0)...")
    t_start = time.perf_counter()
    # upgrade_id="0" is ResStock's baseline (pre-retrofit) scenario.
    ts_query_baseline = TSQuery(
        enduses=[ELEC_TOTAL_COL],
        restrict=[("bldg_id", case_study_bldg_ids)],
        upgrade_id="0",
        timestamp_grouping_func="hour",
        group_by=[BLDG_ID_COL],
        split_enduses=False,
    )
    df_ts_baseline_case_study = my_run.agg.aggregate_timeseries(
        params=ts_query_baseline
    )
    query_time_s = time.perf_counter() - t_start

    # Rename, downcast, add the hour index -- see prepare_bsq_timeseries.
    df_ts_baseline_case_study = prepare_bsq_timeseries(
        df_ts_baseline_case_study, "baseline_kwh"
    )

    # Prints the summary; raises if any building is missing hours.
    summarize_hourly_timeseries(
        df_ts_baseline_case_study,
        "baseline_kwh",
        "df_ts_baseline_case_study",
        query_time_s,
    )
    print("[OK] Step 5 PASSED")

    # ---------- Step 6: Upgrade timeseries for each selected MP ----------
    df_ts_upgrade_case_study_by_mp = {}
    for mp in selected_mps:
        print(f"\nQuerying upgrade timeseries (upgrade={mp})...")
        t_start = time.perf_counter()
        # upgrade_id=str(mp) selects that MP's retrofit scenario.
        ts_query_upgrade = TSQuery(
            enduses=[ELEC_TOTAL_COL],
            restrict=[("bldg_id", case_study_bldg_ids)],
            upgrade_id=str(mp),
            timestamp_grouping_func="hour",
            group_by=[BLDG_ID_COL],
            split_enduses=False,
        )
        df_ts_upgrade = my_run.agg.aggregate_timeseries(params=ts_query_upgrade)
        query_time_s = time.perf_counter() - t_start

        # Same fixes as the baseline query above.
        df_ts_upgrade = prepare_bsq_timeseries(df_ts_upgrade, "retrofit_kwh")

        summarize_hourly_timeseries(
            df_ts_upgrade,
            "retrofit_kwh",
            f"df_ts_upgrade_case_study (MP{mp})",
            query_time_s,
        )

        # Confirms baseline and upgrade scoped the same buildings.
        check_upgrade_building_coverage(
            set(df_ts_baseline_case_study[BLDG_ID_COL].unique()),
            set(df_ts_upgrade[BLDG_ID_COL].unique()),
            mp,
        )

        df_ts_upgrade_case_study_by_mp[mp] = df_ts_upgrade

    print("[OK] Step 6 PASSED")

    # ---------- Save the query cache to disk ----------
    # save_cache() writes the query results to
    # cmu_tare_model/grid_impact/.bsq_cache/ for the next run to reuse.
    # Avoids long re-queries for the same buildings and MPs.
    my_run.save_cache()
    print("[OK] Query cache saved to disk (.bsq_cache/)")


# %% [markdown]
# ### Visuals - Retrofit Impact on County Peak Load

# %%
# ============================================================================
# GRID IMPACT -- plot county-level demand for the selected measure packages
# ============================================================================
if GRID_IMPACT_ANALYSIS:

    # ---------- Step 7: Compute scenario profiles ----------
    peak_results_case_study_by_mp = {}
    df_profiles_by_mp = {}

    # Echoed here so the printed peak numbers below are traceable to exactly
    # which NPV case and weighting mode produced them.
    print(f"Grid impact base case (constrained adopters): {BASE_CASE_NPV_CASE}")
    print(f"Custom weighting: {CUSTOM_WEIGHTING}")        

    for mp in selected_mps:
        print(
            f"\nComputing county profiles for MP{mp} "
            f"(adopter column: {adoption_col_by_mp[mp]})..."
        )

        # TODO: For loop for different NPV cases
        # NPV_CASES_CATEGORIES defined in column_names.py
        # NPV_CASE_CATEGORIES = (
        #     "heatingSavings_coolingLCC_sub",
        #     "heatingSavings_coolingLCC_unsub",
        #     "heatingSavings_coolingLCC_sub_june2026",
        #     "heatingLCC_coolingSavings_sub",
        #     "heatingLCC_coolingSavings_unsub",
        #     "heatingLCC_coolingSavings_sub_june2026",
        #     "heatingLCC_coolingLCC_sub",
        #     "heatingLCC_coolingLCC_unsub",
        #     "heatingLCC_coolingLCC_sub_june2026",
        # )
        # for 

        if CUSTOM_WEIGHTING:
            # The matched subset can span more than one real county (it
            # currently spans two, in Colorado), so adopter IDs are gathered
            # across every county in adopter_ids_by_mp rather than one FIPS
            # lookup, then narrowed to just the buildings the weight dict
            # covers.
            case_study_bldg_id_set = set(case_study_weight_dict.keys())
            constrained_ids = set()
            for county_adopters in adopter_ids_by_mp[mp].values():
                constrained_ids |= (
                    set(county_adopters["constrained"]) & case_study_bldg_id_set
                )
            case_study_adopter_ids = {
                "all_filtered": sorted(case_study_bldg_id_set),
                "constrained": sorted(constrained_ids),
            }
        else:
            # Default mode: the one county chosen earlier already has its
            # adopter IDs precomputed in adopter_ids_by_mp.
            case_study_adopter_ids = adopter_ids_by_mp[mp][case_study_fips]

        # 100pct: every building in scope adopts the retrofit -- the upper
        # bound on the demand-side impact.
        df_profile_100pct, peak_100pct = compute_county_scenario_profile(
            df_ts_baseline_case_study,
            df_ts_upgrade_case_study_by_mp[mp],
            adopter_bldg_ids=case_study_adopter_ids["all_filtered"],
            custom_weighting=CUSTOM_WEIGHTING,
            weight_dict=case_study_weight_dict,
        )

        # constrained: only the buildings where the heat pump is an economic
        # adopter (NPV >= 0) actually switch.
        df_profile_constrained, peak_constrained = compute_county_scenario_profile(
            df_ts_baseline_case_study,
            df_ts_upgrade_case_study_by_mp[mp],
            adopter_bldg_ids=case_study_adopter_ids["constrained"],
            custom_weighting=CUSTOM_WEIGHTING,
            weight_dict=case_study_weight_dict,
        )

        peak_results_case_study_by_mp[mp] = {
            "100pct": peak_100pct,
            "constrained": peak_constrained,
        }
        df_profiles_by_mp[mp] = {
            "100pct": df_profile_100pct,
            "constrained": df_profile_constrained,
        }

        print(f"\nCase-study peak results (MP{mp})")
        for scenario, peak_summary in peak_results_case_study_by_mp[mp].items():
            print(
                f"  [{scenario}] adopters: {peak_summary['n_adopters']:,d} / "
                f"{peak_summary['n_total_buildings']:,d}"
            )
            print(
                f"    baseline peak : {peak_summary['baseline_peak_mw']:.2f} MW "
                f"@ hour {peak_summary['peak_hour_baseline']}"
            )
            print(
                f"    scenario peak : {peak_summary['scenario_peak_mw']:.2f} MW "
                f"@ hour {peak_summary['peak_hour_scenario']}"
            )
            print(f"    delta         : {peak_summary['delta_mw']:+.2f} MW")

        # Both profiles must cover a full year of hours -- guards against a
        # silent partial-year query slipping through to the figure below.
        assert len(df_profile_100pct) == 8760
        assert len(df_profile_constrained) == 8760

    print(
        f"\n[OK] Step 7 PASSED -- peak_results_case_study_by_mp.keys() = "
        f"{list(peak_results_case_study_by_mp.keys())}"
    )

    # ---------- Demand-profile grid figure ----------
    # plot_county_demand_grid itself is unchanged this session (see Current
    # state) -- only the renamed dict it is passed here differs from before.
    plot_county_demand_grid(
        df_profiles_by_mp,
        peak_results_case_study_by_mp,
        selected_mps,
        save_figure=SAVE_FIGURES,
        output_dir=PROJECT_ROOT,
        figure_dpi=FIGURE_DPI,
    )


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


