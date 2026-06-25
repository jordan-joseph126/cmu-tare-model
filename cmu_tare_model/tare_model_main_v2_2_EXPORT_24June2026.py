# %% [markdown]
# -------------------------------------------------------------------------------------------------------
# # EUSS Post-Retrofit Measure Packages: MP8, MP9, MP10
# -------------------------------------------------------------------------------------------------------
# - MP8: Whole Home Electrification (High Efficiency)
# - MP9: Whole-Home Electrification + Basic Enclosure Upgrade
# - MP10: Whole-Home Electrification + Enhanced Enclosure Upgrade
# 
# -------------------------------------------------------------------------------------------------------
# # TARE MODEL SCENARIOS
# -------------------------------------------------------------------------------------------------------
# - Pre-IRA Scenario:
#     - NREL End-Use Savings Shapes Database: Measure Package 8/9/10
#     - AEO2023 No Inflation Reduction Act
#     - Cambium 2021 MidCase
#       
# - IRA-Reference Scenario:
#     - NREL End-Use Savings Shapes Database: Measure Package 8/9/10
#     - AEO2023 REFERENCE CASE - HDD and Fuel Price Projections
#     - Cambium 2022 and 2023 MidCase

# %%
# =============================================================================
# IMPORTS
# =============================================================================
import os
from IPython import get_ipython
from datetime import datetime

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
    RCM_MODELS, 
    CR_FUNCTIONS,
    SCC_ASSUMPTIONS,
    REMDB_COST_SCENARIO_KEYS,
    VALID_MENU_MPS,
    VALID_CATEGORIES,
    PRINT_DEBUG,
    PRINT_VERBOSE_DATAFRAMES,
    VALID_HVAC_REPLACEMENT_SCENARIOS
)
from cmu_tare_model.constants import (
    PRIVATE_DISCOUNT_RATE_COLS, 
    PRIVATE_DISCOUNT_RATE_SHORT_KEYS
)

# Column name builders
from cmu_tare_model.utils.column_names import (
    create_cost_col,
    create_capital_col,
    create_npv_col,
    create_rebate_col,
    create_adoption_col,
    create_total_npv_col,
    create_health_npv_col,
    create_climate_npv_col
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

sns.set_theme(font='sans-serif', style='darkgrid')

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
Active RCM Models: {RCM_MODELS}
Active CR Functions: {CR_FUNCTIONS}
Active Discount Rates: {PRIVATE_DISCOUNT_RATE_SHORT_KEYS}

Note: DataFrames contain columns for ALL active cost scenarios.
Visualizations default to 'v4MID' with comparative sections for other scenarios.

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
            relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_run_simulation_v2_2.ipynb")

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
# # Baseline Scenario: Measure Package 0 (MP0)
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

# %% [markdown]
# -------------------------------------------------------------------------------------------------------
# # Basic Retrofit: Measure Package 8 (MP8)
# -------------------------------------------------------------------------------------------------------

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

# %% [markdown]
# # CLIMATE CHANGE AND PUBLIC HEALTH IMPACTS

# %%
from cmu_tare_model.utils.data_visualization import print_summary_stats
from cmu_tare_model.utils.data_visualization_boxplots import create_subplot_grid_boxplot
from cmu_tare_model.utils.data_visualization_histograms import create_subplot_grid_histogram, print_positive_percentages_complete

if VERBOSE:
    print(f"""  
    ====================================================================================================================================================================
    UNCERTAINTY ANALYSIS VISUALIZATION
    ====================================================================================================================================================================

    --------------------------------------------------------
    SUMMARY STATISTICS TABLE
    --------------------------------------------------------
    data_visualization.py file contains the documentation for the print_summary_stats function.

    --------------------------------------------------------
    SUBPLOT GRID OF BOXPLOTS
    --------------------------------------------------------
    data_visualization_boxplots.py file contains the documentation for the create_subplot_grid_boxplot function.
        
    --------------------------------------------------------
    SUBPLOT GRID OF HISTOGRAMS
    --------------------------------------------------------
    data_visualization_histograms.py file contains the documentation for the create_subplot_grid_histogram function.
        
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    """)

# %% [markdown]
# ## HEALTH IMPACT: 3 Reduced Complexity Models x 2 CR Functions

# %%
# =============================================================================
# HEALTH IMPACT VISUALIZATIONS: RCM × CR-Function Sensitivity
# =============================================================================
if 8 not in DATAFRAMES_BY_MP:
    print("MP8 not in VALID_MENU_MPS — skipping health impact visualizations.")
else:
    scenario_prefix = 'iraRef_mp8_'
    category = 'heating'
    lower_percentile = 0.5
    upper_percentile = 99.5
    discount_rate = 'fixed_base'

    # Store figures for later reference
    health_npv_figures = {}
    n_rcm = len(RCM_MODELS)

    for cr_function in CR_FUNCTIONS:
        print(f"\n{'='*60}")
        print(f"FIGURE: MONETIZED HEALTH IMPACT ({cr_function.upper()} CR-FUNCTION)")
        print(f"Active RCM Models: {[r.upper() for r in RCM_MODELS]}")
        print(f"Discount Rate: {discount_rate}")
        print(f"{'='*60}")

        # Build dynamic lists based on active RCM models
        rcm_dataframes = [DATAFRAMES_BY_MP[8][discount_rate][rcm] for rcm in RCM_MODELS]
        rcm_y_cols = [f'{scenario_prefix}{category}_health_npv_{rcm}_{cr_function}' for rcm in RCM_MODELS]
        rcm_titles = [f'{rcm.upper()} ({cr_function.upper()})' for rcm in RCM_MODELS]

        fig = create_subplot_grid_boxplot(
            dataframes=rcm_dataframes,
            subplot_positions=[(0, i) for i in range(n_rcm)],
            y_cols=rcm_y_cols,
            hue_col=f'base_{category}_fuel',
            sharex=True,
            sharey=True,
            subplot_titles=rcm_titles,
            x_labels=[''] * n_rcm,
            y_labels=['Health NPV [2023 $USD]'] + [''] * (n_rcm - 1),
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            figure_size=(max(8, 6 * n_rcm), 6),
            show_outliers=False,
            show_xtick_labels=False
        )

        # Print summary statistics for each active RCM model
        for rcm in RCM_MODELS:
            print_summary_stats(
                dataframes=[DATAFRAMES_BY_MP[8][discount_rate][rcm]],
                column_names=[f'{scenario_prefix}{category}_health_npv_{rcm}_{cr_function}'],
                subplot_titles=[f'{rcm.upper()} with {cr_function.upper()} CR-Function']
            )

        # Print positive percentage statistics
        print_positive_percentages_complete(
            dataframes=rcm_dataframes,
            column_names=rcm_y_cols,
            subplot_titles=rcm_titles,
            fuel_column=f'base_{category}_fuel'
        )

        # Store figure
        health_npv_figures[cr_function] = fig
        
        # Display
        display(fig)

# %% [markdown]
# ## Climate Change Impact (SCC) and Tier 3 Adopters

# %% [markdown]
# ### Space Heating - Progressive Impact of Climate Benefit Valuation

# %%
if 8 not in DATAFRAMES_BY_MP:
    print("MP8 not in VALID_MENU_MPS — skipping climate SCC visualization.")
else:
    scenario_prefix = 'iraRef_mp8_'
    category = 'heating'
    rcm_model = 'inmap'
    discount_rate = 'fixed_base'
    cost_scenario = 'v4MID'  # Default cost scenario for visualization
    lower_percentile = 0.5
    upper_percentile = 99.5

    # Build column names using centralized builders
    private_npv_col = create_npv_col(scenario_prefix, category, 'moreWTP', cost_scenario, f'_{discount_rate}')
    climate_npv_lower = create_total_npv_col(scenario_prefix, category, cost_scenario=cost_scenario,
                                              method_suffix=f'_{discount_rate}', scc_assumption='lower', climate_only=True)
    climate_npv_central = create_total_npv_col(scenario_prefix, category, cost_scenario=cost_scenario,
                                                method_suffix=f'_{discount_rate}', scc_assumption='central', climate_only=True)
    climate_npv_upper = create_total_npv_col(scenario_prefix, category, cost_scenario=cost_scenario,
                                              method_suffix=f'_{discount_rate}', scc_assumption='upper', climate_only=True)

    print(f"""
===== FIGURE 7: CLIMATE BENEFIT IMPACT ON RETROFIT ADOPTION POTENTIAL (TIER 3) =====
- Retrofit Scenarios: {scenario_prefix} 
- Discount Rate: {discount_rate}
- RCM Model where the dataframe is stored (not included in climate NPV): {rcm_model}
- Cost Scenario: {cost_scenario}
- Categories: {category}

Valid Range: {lower_percentile}th to {upper_percentile}th Percentile

Column names:
  Private NPV: {private_npv_col}
  Climate Lower: {climate_npv_lower}
  Climate Central: {climate_npv_central}
  Climate Upper: {climate_npv_upper}
""")

    fig_heating_climate_scc_FIXED_BASE = create_subplot_grid_histogram(
        dataframes=[
            # Private NPV
            DATAFRAMES_BY_MP[8][discount_rate][rcm_model], 
            # Climate NPV Lower (Total NPV with climate benefits only) 
            DATAFRAMES_BY_MP[8][discount_rate][rcm_model],
            # Climate NPV Central (Total NPV with climate benefits only)
            DATAFRAMES_BY_MP[8][discount_rate][rcm_model],
            # Climate NPV Upper (Total NPV with climate benefits only)
            DATAFRAMES_BY_MP[8][discount_rate][rcm_model]
            ],
        subplot_positions=[(0, 0), (0, 1), (0, 2), (0, 3)],  # 1x4 grid
        x_cols=[
            # Private NPV
            private_npv_col,
            # Climate NPV Lower (Total NPV with climate benefits only)
            climate_npv_lower,
            # Climate NPV Central (Total NPV with climate benefits only)
            climate_npv_central,
            # Climate NPV Upper (Total NPV with climate benefits only)
            climate_npv_upper
        ],
        x_labels=['Private NPV [2023 $USD]'] + ['Total NPV [2023 $USD]'] * 3,
        y_labels=['Dwelling Units', '', '', ''],
        bin_number=40,  # Optional: number of bins for histogram
        lower_percentile=lower_percentile,    # Show nearly full range
        upper_percentile=upper_percentile,   # Show nearly full range
        subplot_titles=[
            'Private NPV Only\n37% Positive NPV',
            'SCC Lower Bound\n56% Positive NPV',
            'SCC Central Estimate\n78% Positive NPV',
            'SCC Upper Bound\n83% Positive NPV' 
        ],
        # suptitle=f'{category.title()}: Progressive Impact of Climate Benefit Valuation',
        figure_size=(20, 10),  # Wide format for 4 panels
        sharex=False,  # Keep different scales to show full distributions
        sharey=True,   # Same y-scale for comparison
        color_code=f'base_{category}_fuel'
    )

    print_positive_percentages_complete(
        df=DATAFRAMES_BY_MP[8][discount_rate][rcm_model], 
        column_names=[
            private_npv_col,
            climate_npv_lower,
            climate_npv_central,
            climate_npv_upper
        ],
        subplot_titles=[
            f'Private NPV Only (Baseline), Discount Rate: {discount_rate}', 
            f'Lower Bound SCC (+ Climate), Discount Rate: {discount_rate}', 
            f'Central Estimate SCC (+ Climate), Discount Rate: {discount_rate}', 
            f'Upper Bound SCC (+ Climate), Discount Rate: {discount_rate}'
        ],
        fuel_column=f'base_{category}_fuel'
    )

    print("""\n===== IMPORTANT: UPDATE THE OVERALL POSITIVE NPV VALUES IN THE TITLES! =====\n""")
    display(fig_heating_climate_scc_FIXED_BASE)
    print("""\n===== IMPORTANT: UPDATE THE OVERALL POSITIVE NPV VALUES IN THE TITLES! =====\n""")

# %% [markdown]
# # Adoption Rate Scenario Comparison

# %%
from cmu_tare_model.adoption_potential.data_processing.visuals_adoption_potential import (
    build_adoption_scenario_names,
    create_multiIndex_adoption_df,
    print_adoption_decision_percentages,
    subplot_grid_adoption_vBar
)

if VERBOSE:

    print(f"""  
    ====================================================================================================================================================================
    ADOPTION POTENTIAL VISUALIZATION
    ====================================================================================================================================================================

    --------------------------------------------------------
    CREATE MULTI-INDEX DF FOR ADOPTION POTENTIAL
    --------------------------------------------------------
    visuals_adoption_potential.py file contains the documentation for the create_multiIndex_adoption_df function.

    --------------------------------------------------------
    VISUALIZE ADOPTION POTENTIAL SUBPLOT GRID
    --------------------------------------------------------
    visuals_adoption_potential.py file contains the documentation for the subplot_grid_adoption_vBar function.
        
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------

    """)

# %% [markdown]
# ## Space Heating - Basic (MP8), Moderate (MP9), Advanced (MP10) Retrofit
# 

# %%
# =============================================================================
# CREATE ADOPTION POTENTIAL DATAFRAMES (ALL COMBINATIONS)
# =============================================================================
# Creates DataFrames for all sensitivity combinations upfront.
# Structure: ALL_HEATING_ADOPTION_MI[mp][hvac_scenario][cost_scenario][discount_rate][rcm][crf] = DataFrame
# The hvac_scenario dimension supports Case A ('heating') and Case B ('heating_and_cooling').
# Only creates DataFrames for MPs that are in VALID_MENU_MPS.
# Combinations whose adoption columns are missing in source_df are skipped (left as None).

scc = 'central'

# Derive heating measure packages from VALID_MENU_MPS (exclude baseline MP0)
HEATING_MEASURE_PACKAGES = [mp for mp in VALID_MENU_MPS if mp != 0]

if not HEATING_MEASURE_PACKAGES:
    print("No non-baseline measure packages in VALID_MENU_MPS — skipping adoption potential.")
    ALL_HEATING_ADOPTION_MI = {}
else:
    # Master dictionary to store all results
    # Structure: [mp][hvac_scenario][cost_scenario][discount_rate][rcm][crf]
    ALL_HEATING_ADOPTION_MI = {
        mp: {
            hvac_scenario: {
                cost_scenario: {
                    discount_rate: {
                        rcm: {crf: None for crf in CR_FUNCTIONS}
                        for rcm in RCM_MODELS
                    }
                    for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS
                }
                for cost_scenario in REMDB_COST_SCENARIO_KEYS
            }
            for hvac_scenario in VALID_HVAC_REPLACEMENT_SCENARIOS
        }
        for mp in HEATING_MEASURE_PACKAGES
    }

    print("Creating adoption potential DataFrames...")
    print(f"Active Measure Packages: {HEATING_MEASURE_PACKAGES}")
    print(f"HVAC replacement scenarios: {VALID_HVAC_REPLACEMENT_SCENARIOS}")
    print(f"Cost scenarios: {REMDB_COST_SCENARIO_KEYS}")

    _skipped = []  # track skipped combos for summary

    for menu_mp in HEATING_MEASURE_PACKAGES:
        print(f"\n{'='*80}")
        print(f"MEASURE PACKAGE {menu_mp}")
        print(f"{'='*80}")

        for hvac_replacement_scenario in VALID_HVAC_REPLACEMENT_SCENARIOS:
            print(f"\n  HVAC Scenario: {hvac_replacement_scenario}")

            for cost_scenario in REMDB_COST_SCENARIO_KEYS:
                print(f"    Cost Scenario: {cost_scenario}")

                for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS:
                    print(f"      Discount Rate: {discount_rate}")

                    for rcm_model in RCM_MODELS:
                        for cr_function in CR_FUNCTIONS:
                            # Direct dictionary access with short keys
                            source_df = DATAFRAMES_BY_MP[menu_mp][discount_rate][rcm_model]

                            # Build expected column names to verify they exist
                            expected_cols = build_adoption_scenario_names(
                                menu_mp, 'heating', scc, rcm_model, cr_function,
                                cost_scenario, discount_rate,
                                hvac_replacement_scenario=hvac_replacement_scenario,
                            )
                            missing = [c for c in expected_cols if c not in source_df.columns]
                            if missing:
                                tag = f"MP{menu_mp}/{hvac_replacement_scenario}/{cost_scenario}/{discount_rate}"
                                _skipped.append(tag)
                                print(f"        ⚠ SKIPPED ({tag}): columns not found — {missing}")
                                continue

                            df_mi = create_multiIndex_adoption_df(
                                df=source_df,
                                menu_mp=menu_mp,
                                category='heating',
                                scc=scc,
                                rcm_model=rcm_model,
                                cr_function=cr_function,
                                cost_scenario=cost_scenario,
                                discount_rate=discount_rate,
                                hvac_replacement_scenario=hvac_replacement_scenario,
                            )

                            ALL_HEATING_ADOPTION_MI[menu_mp][hvac_replacement_scenario][cost_scenario][discount_rate][rcm_model][cr_function] = df_mi

    _created = (len(VALID_HVAC_REPLACEMENT_SCENARIOS) * len(REMDB_COST_SCENARIO_KEYS) *
                len(PRIVATE_DISCOUNT_RATE_SHORT_KEYS) * len(HEATING_MEASURE_PACKAGES) *
                len(RCM_MODELS) * len(CR_FUNCTIONS)) - len(_skipped)
    print(f"\n✓ Created {_created} DataFrames, skipped {len(_skipped)} (missing columns)")
    if _skipped:
        print(f"  Skipped combos: {', '.join(sorted(set(_skipped)))}")

# %%
# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================
# Edit these values, then run the next cell to create the visualization.

# Discount rate: 'fixed_low', 'fixed_base', 'fixed_high', 'variable'
discount_rate = 'fixed_base'

# Capital cost scenario: 'v3', 'v4MID', etc.
cost_scenario = 'v4MID'

# Health model parameters (typically keep these fixed)
scc = 'central'
rcm_model = 'inmap'
cr_function = 'acs'

# HVAC replacement scenario for bar-chart visualization (Case A default)
hvac_replacement_scenario = 'heating'

# =============================================================================
# ADOPTION POTENTIAL VISUALIZATION
# =============================================================================
if not HEATING_MEASURE_PACKAGES:
    print("No active heating measure packages — skipping adoption visualization.")
else:
    category = 'heating'
    
    # Subplot titles and labels for each measure package
    HEATING_MP_SUBTITLES = {
        3: "ASHP (MP3 - Min Efficiency):\nNo IRA vs. IRA-Reference",
        4: "ASHP (MP4 - High Efficiency):\nNo IRA vs. IRA-Reference",
        8: "ASHP (MP4) + No Enclosure:\nNo IRA vs. IRA-Reference",
        9: "ASHP (MP4) + Basic Enclosure:\nNo IRA vs. IRA-Reference",
        10: "ASHP (MP4) + Enhanced Enclosure:\nNo IRA vs. IRA-Reference"
    }

    print(f"""
================================================================================
ADOPTION POTENTIAL VISUALIZATION
================================================================================
Active Measure Packages: {HEATING_MEASURE_PACKAGES}
Discount Rate: {discount_rate}
Cost Scenario: {cost_scenario}
HVAC Scenario: {hvac_replacement_scenario}
SCC: {scc} | RCM: {rcm_model} | CRF: {cr_function}
""")

    n_panels = len(HEATING_MEASURE_PACKAGES)

    fig_adoption = subplot_grid_adoption_vBar(
        dataframes=[
            ALL_HEATING_ADOPTION_MI[mp][hvac_replacement_scenario][cost_scenario][discount_rate][rcm_model][cr_function]
            for mp in HEATING_MEASURE_PACKAGES
        ],
        scenarios_list=[
            build_adoption_scenario_names(mp, category, scc, rcm_model, cr_function, cost_scenario, discount_rate, hvac_replacement_scenario)
            for mp in HEATING_MEASURE_PACKAGES
        ],
        subplot_positions=[(0, i) for i in range(n_panels)],
        filter_fuel=['Electricity', 'Natural Gas', 'Fuel Oil', 'Propane'],
        x_labels=[""] * (n_panels // 2) + ["Fuel Type and Income Group (LMI: Low-to-Moderate-Income, MUI: Middle-to-Upper-Income)"] + [""] * (n_panels - n_panels // 2 - 1),
        plot_titles=[HEATING_MP_SUBTITLES.get(mp, f"MP{mp}:\nNo IRA vs. IRA-Reference") for mp in HEATING_MEASURE_PACKAGES],
        y_labels=["Retrofit Adoption Potential (%)"] + [""] * (n_panels - 1),
        figure_size=(6 * n_panels, 12),
        sharey=True,
        x_tick_format="all"
    )

    # =======================================================================================================
    # PRINT ADOPTION DECISION PERCENTAGES
    # =======================================================================================================
    for i, menu_mp in enumerate(HEATING_MEASURE_PACKAGES):
        scenario_names = build_adoption_scenario_names(menu_mp, category, scc, rcm_model, cr_function, cost_scenario, discount_rate, hvac_replacement_scenario)
        print_adoption_decision_percentages(
                dataframes=[
                    ALL_HEATING_ADOPTION_MI[menu_mp][hvac_replacement_scenario][cost_scenario][discount_rate][rcm_model][cr_function],
                    ALL_HEATING_ADOPTION_MI[menu_mp][hvac_replacement_scenario][cost_scenario][discount_rate][rcm_model][cr_function],
                    ],
                scenario_names=scenario_names,
                source_dataframes=[
                    DATAFRAMES_BY_MP[menu_mp][discount_rate][rcm_model],
                    DATAFRAMES_BY_MP[menu_mp][discount_rate][rcm_model],
                ],
                category='heating',
                title=f"SPACE HEATING ADOPTION POTENTIAL: {discount_rate.upper()} | Cost: {cost_scenario}", 
                subtitle=HEATING_MP_SUBTITLES.get(menu_mp, f"MP{menu_mp}"),
                print_header_key=True,
            )

    display(fig_adoption)

# %%
## Adoption Potential Dot Plot: MP3 vs. MP4 — Case A (Heating) vs. Case B (Heating & Cooling)

# %%
# =============================================================================
# ADOPTION POTENTIAL DOT PLOT: 1-row × N-col (one col per MP, shared x + y)
# =============================================================================
import importlib
import cmu_tare_model.adoption_potential.data_processing.visuals_adoption_dotplot as _vdp
importlib.reload(_vdp)
from cmu_tare_model.adoption_potential.data_processing.visuals_adoption_dotplot import (
    prepare_plot_data,
    plot_adoption_panel,
    _build_legend_handles,
    GROUPING_ORDER,
)

CASE_LABELS = {
    'heating': 'Case A: Heating Only',
    'heating_and_cooling': 'Case B: Heating & Cooling',
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

    # Print figure title and fuel counts before creating the figure
    print(f"Heat Pump Adoption Potential — {case_label}")
    print(f"Discount Rate: {discount_rate} | Cost Scenario: {cost_scenario}")
    print()
    print("Fuel sample counts (national, approx.):")
    for fuel, count in sorted(fuel_counts_millions.items()):
        print(f"  {fuel}: {count:.1f}M homes")
    print()

    n_cols = len(HEATING_MEASURE_PACKAGES)

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(10 * n_cols, 9),
        sharex=True,
        sharey=True,
    )
    # Normalise to list
    if n_cols == 1:
        axes = [axes]

    for col_idx, mp in enumerate(HEATING_MEASURE_PACKAGES):
        ax = axes[col_idx]
        panel_title = f'{HEATING_MP_SUBTITLES.get(mp, f"MP{mp}")} — {case_label}'

        mi_df = ALL_HEATING_ADOPTION_MI[mp][hvac_replacement_scenario][cost_scenario][discount_rate][rcm_model][cr_function]

        if mi_df is None:
            ax.set_title(panel_title, fontsize=16, fontweight='bold')
            ax.text(0.5, 0.5, 'No data\n(adoption columns missing\nfor this scenario)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=14, color='gray')
            ax.set_xlim(0, 100)
            ax.set_xticks(range(0, 101, 20))
            y_order = list(reversed(GROUPING_ORDER))
            ax.set_ylim(-0.5, len(y_order) - 0.5)
            ax.set_yticks(range(len(y_order)))
            if col_idx > 0:
                ax.set_yticklabels([])
            continue

        scenario_names = build_adoption_scenario_names(
            mp, category, scc, rcm_model, cr_function,
            cost_scenario, discount_rate,
            hvac_replacement_scenario=hvac_replacement_scenario,
        )
        preira_col = scenario_names[0]
        iraref_col = scenario_names[1]

        source_df = DATAFRAMES_BY_MP[mp][discount_rate][rcm_model]

        plot_df = prepare_plot_data(
            mi_df,
            source_df,
            preira_col=preira_col,
            iraref_col=iraref_col,
            income_groups=['LMI'],
        )

        # Print sample stats for this panel
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

        plot_adoption_panel(
            plot_df, ax, title=panel_title,
            title_fontsize=16,
            ytick_fontsize=14,
            annotation_fontsize=14,
            fuel_counts_millions=fuel_counts_millions,
        )
        ax.tick_params(axis='both', labelsize=14)

        # Only leftmost panel keeps y-tick labels; sharey hides the rest automatically
        # but suppress the y-axis label on non-first panels to avoid duplication
        if col_idx > 0:
            ax.set_ylabel('')

    # Legend below figure
    legend_handles = _build_legend_handles()
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=len(legend_handles),
        fontsize=14,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])

    # Save — case tag in filename
    out_dir = os.path.join('.', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    case_tag = 'caseA' if hvac_replacement_scenario == 'heating' else 'caseB'
    for ext in ('png', 'pdf'):
        fig.savefig(
            os.path.join(out_dir, f'figure5_adoption_dotplot_{case_tag}_{location_id}.{ext}'),
            dpi=300,
            bbox_inches='tight',
        )
    print(f"Saved to {out_dir}/figure5_adoption_dotplot_{case_tag}_{location_id}.{{png,pdf}}")
    plt.show()


# %%
# =============================================================================
# ADOPTION POTENTIAL DOT PLOT: 1-row × N-col (one col per MP, shared x + y)
# =============================================================================
import importlib
import cmu_tare_model.adoption_potential.data_processing.visuals_adoption_dotplot as _vdp
importlib.reload(_vdp)
from cmu_tare_model.adoption_potential.data_processing.visuals_adoption_dotplot import (
    prepare_plot_data,
    plot_adoption_panel,
    _build_legend_handles,
    GROUPING_ORDER,
)

CASE_LABELS = {
    'heating': 'Case A: Heating Only',
    'heating_and_cooling': 'Case B: Heating & Cooling',
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

    # Print figure title and fuel counts before creating the figure
    print(f"Heat Pump Adoption Potential — {case_label}")
    print(f"Discount Rate: {discount_rate} | Cost Scenario: {cost_scenario}")
    print()
    print("Fuel sample counts (national, approx.):")
    for fuel, count in sorted(fuel_counts_millions.items()):
        print(f"  {fuel}: {count:.1f}M homes")
    print()

    n_cols = len(HEATING_MEASURE_PACKAGES)

    fig, axes = plt.subplots(
        1, n_cols,
        figsize=(10 * n_cols, 9),
        sharex=True,
        sharey=True,
    )
    # Normalise to list
    if n_cols == 1:
        axes = [axes]

    for col_idx, mp in enumerate(HEATING_MEASURE_PACKAGES):
        ax = axes[col_idx]
        panel_title = f'{HEATING_MP_SUBTITLES.get(mp, f"MP{mp}")} — {case_label}'

        mi_df = ALL_HEATING_ADOPTION_MI[mp][hvac_replacement_scenario][cost_scenario][discount_rate][rcm_model][cr_function]

        if mi_df is None:
            ax.set_title(panel_title, fontsize=16, fontweight='bold')
            ax.text(0.5, 0.5, 'No data\n(adoption columns missing\nfor this scenario)',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=14, color='gray')
            ax.set_xlim(0, 100)
            ax.set_xticks(range(0, 101, 20))
            y_order = list(reversed(GROUPING_ORDER))
            ax.set_ylim(-0.5, len(y_order) - 0.5)
            ax.set_yticks(range(len(y_order)))
            if col_idx > 0:
                ax.set_yticklabels([])
            continue

        scenario_names = build_adoption_scenario_names(
            mp, category, scc, rcm_model, cr_function,
            cost_scenario, discount_rate,
            hvac_replacement_scenario=hvac_replacement_scenario,
        )
        preira_col = scenario_names[0]
        iraref_col = scenario_names[1]

        source_df = DATAFRAMES_BY_MP[mp][discount_rate][rcm_model]

        plot_df = prepare_plot_data(
            mi_df,
            source_df,
            preira_col=preira_col,
            iraref_col=iraref_col,
            income_groups=['LMI'],
        )

        # Print sample stats for this panel
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

        plot_adoption_panel(
            plot_df, ax, title=panel_title,
            title_fontsize=16,
            ytick_fontsize=14,
            annotation_fontsize=12,
            fuel_counts_millions=fuel_counts_millions,
        )
        ax.tick_params(axis='both', labelsize=14)

        # Only leftmost panel keeps y-tick labels; sharey hides the rest automatically
        # but suppress the y-axis label on non-first panels to avoid duplication
        if col_idx > 0:
            ax.set_ylabel('')

    # Legend below figure
    legend_handles = _build_legend_handles()
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=len(legend_handles),
        fontsize=14,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])

    # Save — case tag in filename
    out_dir = os.path.join('.', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    case_tag = 'caseA' if hvac_replacement_scenario == 'heating' else 'caseB'
    for ext in ('png', 'pdf'):
        fig.savefig(
            os.path.join(out_dir, f'figure5_adoption_dotplot_{case_tag}_{location_id}.{ext}'),
            dpi=300,
            bbox_inches='tight',
        )
    print(f"Saved to {out_dir}/figure5_adoption_dotplot_{case_tag}_{location_id}.{{png,pdf}}")
    plt.show()


# %% [markdown]
# ## Water Heating, Clothes Drying, and Cooking - Basic Retrofit (MP8)

# %%
# =============================================================================
# CREATE ADOPTION POTENTIAL DFs FOR NON-HVAC CATEGORIES (ALL DISCOUNT RATES)
# =============================================================================

NONHVAC_CATEGORIES = ['waterHeating', 'clothesDrying', 'cooking']
ACTIVE_NONHVAC = [c for c in NONHVAC_CATEGORIES if c in VALID_CATEGORIES]

if not ACTIVE_NONHVAC:
    print(f"""
=======================================================================================================
BASIC RETROFIT: MEASURE PACKAGE {menu_mp} (MP{menu_mp}) - NON-HVAC CATEGORIES
=======================================================================================================
NOTE: No non-HVAC categories are currently active in EQUIPMENT_SPECS.
Active categories: {VALID_CATEGORIES}
Skipping non-HVAC adoption analysis.
=======================================================================================================
""")
    
    MP8_NONHVAC_ADOPTION_MI = {}
else:
    menu_mp = 8
    scc = 'central'
    cost_scenario = 'v4MID'  # Default cost scenario
    
    print(f"""
=======================================================================================================
BASIC RETROFIT: MEASURE PACKAGE {menu_mp} (MP{menu_mp}) - NON-HVAC CATEGORIES
=======================================================================================================

Creating Multi-Index DataFrames for:
- Categories: {ACTIVE_NONHVAC}
- Cost Scenarios: {REMDB_COST_SCENARIO_KEYS}
- Discount Rates: {PRIVATE_DISCOUNT_RATE_SHORT_KEYS}
- RCM Models: {RCM_MODELS}
- CR Functions: {CR_FUNCTIONS}

""")
    # Initialize nested dictionary to store results
    # Structure: [category][cost_scenario][discount_rate][rcm][crf]
    MP8_NONHVAC_ADOPTION_MI = {
        category: {
            cost_scenario: {
                discount_rate: {
                    rcm: {crf: None for crf in CR_FUNCTIONS}
                    for rcm in RCM_MODELS
                }
                for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS
            }
            for cost_scenario in REMDB_COST_SCENARIO_KEYS
        }
        for category in ACTIVE_NONHVAC
    }

    CATEGORY_NAMES = {
        'waterHeating': 'Water Heating',
        'clothesDrying': 'Clothes Drying',
        'cooking': 'Cooking'
    }

    for category in ACTIVE_NONHVAC:
        print(f"\n{'='*80}")
        print(f"CATEGORY: {CATEGORY_NAMES.get(category, category).upper()}")
        print(f"{'='*80}")

        for cost_scenario in REMDB_COST_SCENARIO_KEYS:
            print(f"\n  Cost Scenario: {cost_scenario}")

            for discount_rate_short in PRIVATE_DISCOUNT_RATE_SHORT_KEYS:
                print(f"    Discount Rate: {discount_rate_short}")

                for rcm_model in RCM_MODELS:
                    for cr_function in CR_FUNCTIONS:
                        source_df = DATAFRAMES_BY_MP[8][discount_rate_short][rcm_model]

                        df_mi = create_multiIndex_adoption_df(
                            df=source_df,
                            menu_mp=menu_mp,
                            category=category,
                            scc=scc,
                            rcm_model=rcm_model,
                            cr_function=cr_function,
                            cost_scenario=cost_scenario,
                            discount_rate=discount_rate_short
                        )

                        MP8_NONHVAC_ADOPTION_MI[category][cost_scenario][discount_rate_short][rcm_model][cr_function] = df_mi

    print(f"\n{'='*80}")
    print(f"COMPLETE: Created adoption DataFrames for {len(ACTIVE_NONHVAC)} non-HVAC categories")
    print(f"{'='*80}\n")

# %%
# ====================================================================
# VISUALIZATION: Water Heating, Clothes Drying, Cooking - MP8
# ====================================================================

if not ACTIVE_NONHVAC:
    print(f"""
=======================================================================================================
BASIC RETROFIT: MEASURE PACKAGE {menu_mp} (menu_mp{menu_mp}) - NON-HVAC CATEGORIES
=======================================================================================================
NOTE: No non-HVAC categories are currently active in EQUIPMENT_SPECS.
Active categories: {VALID_CATEGORIES}
Skipping non-HVAC adoption analysis.
=======================================================================================================
""")
    
    MP8_NONHVAC_ADOPTION_MI = {}
else:
    menu_mp = 8
    scc = 'central'
    cost_scenario = 'v4MID'  # Default cost scenario    
    rcm_model = 'inmap'
    cr_function = 'acs'
    discount_rate = 'fixed_base'  # Or 'fixed_low', 'fixed_high', 'variable'

    print(f"""
================================================================================
ADOPTION POTENTIAL OF NON-HVAC RETROFITS: MEASURE PACKAGE {menu_mp} (menu_mp{menu_mp})
================================================================================
Categories: {ACTIVE_NONHVAC}
Discount Rate: {discount_rate}
Cost Scenario: {cost_scenario}
SCC: {scc} | RCM: {rcm_model} | CRF: {cr_function}
""")

    # Subplot titles and labels for each measure package
    NONHVAC_CATEGORY_SUBTITLES = {
        'waterHeating': "Heat Pump Water Heater:\nNo IRA vs. IRA-Reference",
        'clothesDrying': "Heat Pump Clothes Dryer:\nNo IRA vs. IRA-Reference",
        'cooking': "Electric Resistance Range:\nNo IRA vs. IRA-Reference"
    }

    n_panels = len(ACTIVE_NONHVAC)

    # Structure: MP8_NONHVAC_ADOPTION_MI[category][cost_scenario][discount_rate][rcm][crf]
    fig_adoption_nonHVAC = subplot_grid_adoption_vBar(
        dataframes=[
            MP8_NONHVAC_ADOPTION_MI[category][cost_scenario][discount_rate][rcm_model][cr_function]
            for category in ACTIVE_NONHVAC
        ],
        scenarios_list=[
            build_adoption_scenario_names(menu_mp, category, scc, rcm_model, cr_function, cost_scenario, discount_rate)
            for category in ACTIVE_NONHVAC
        ],
        subplot_positions=[(0, i) for i in range(n_panels)],
        filter_fuel=['Electricity', 'Natural Gas', 'Fuel Oil', 'Propane'],
        x_labels=[""] * (n_panels // 2) + ["Fuel Type and Income Group (LMI: Low-to-Moderate-Income, MUI: Middle-to-Upper-Income)"] + [""] * (n_panels - n_panels // 2 - 1),
        plot_titles=[NONHVAC_CATEGORY_SUBTITLES.get(category, f"{category}") for category in ACTIVE_NONHVAC],
        y_labels=["Retrofit Adoption Potential (%)"] + [""] * (n_panels - 1),
        # suptitle=f"Space Heating Air-Source Heat Pump (ASHP) Retrofit Scenario Comparison\nClimate Sensitivity: SCC-{scc.upper()} | Health Sensitivity: {rcm_model.upper()}-{cr_function.upper()}",
        figure_size=(6 * n_panels, 12),
        sharey=True,
        x_tick_format="all"  # Use LMI/MUI classification for x-ticks
    )

    # =======================================================================================================
    # PRINT ADOPTION DECISION PERCENTAGES
    # =======================================================================================================
    for i, category in enumerate(ACTIVE_NONHVAC):
        scenario_names = build_adoption_scenario_names(menu_mp, category, scc, rcm_model, cr_function, cost_scenario, discount_rate)
        print_adoption_decision_percentages(
                dataframes=[
                    MP8_NONHVAC_ADOPTION_MI[category][cost_scenario][discount_rate][rcm_model][cr_function],
                    MP8_NONHVAC_ADOPTION_MI[category][cost_scenario][discount_rate][rcm_model][cr_function],
                    ],
                scenario_names=scenario_names,
                source_dataframes=[
                    DATAFRAMES_BY_MP[menu_mp][discount_rate][rcm_model],
                    DATAFRAMES_BY_MP[menu_mp][discount_rate][rcm_model],
                ],
                category=category,
                title=f"NON-HVAC ADOPTION POTENTIAL: {discount_rate.upper()} | Cost: {cost_scenario}", 
                subtitle=NONHVAC_CATEGORY_SUBTITLES.get(category, f"category{category}"),
                print_header_key=True,
            )

    display(fig_adoption_nonHVAC)

# %% [markdown]
# # SENSITIVITY ANALYSIS: Private Discount Rate and Adoption Feasibility (Retrofit Lifecycle Cost)

# %%
# Select MP for discount rate sensitivity analysis (typically MP8 for heating, but can be adapted for other categories/MPs as needed).
menu_mp_input = int(input(f"Enter the measure package number for discount rate sensitivity analysis (e.g., 8 for heating): "))

# Discount Rate Sensitivity Analysis
if int(menu_mp_input) not in DATAFRAMES_BY_MP:
    print(f"MP{menu_mp_input} not in VALID_MENU_MPS — skipping IRA-Reference discount rate sensitivity.")
else:
    category = 'heating'
    rcm_model = 'inmap' if 'inmap' in RCM_MODELS else RCM_MODELS[0]
    cost_scenario = 'v4MID' if 'v4MID' in REMDB_COST_SCENARIO_KEYS else REMDB_COST_SCENARIO_KEYS[0]
    policy_scenario = 'iraRef'  # Focus on IRA-Reference scenario for discount rate sensitivity
    lower_percentile = 0.5
    upper_percentile = 99.5

    # Human-readable labels for each discount rate key
    DISCOUNT_RATE_LABELS = {
        'fixed_low': 'Fixed Discount Rate\n Low (2%)',
        'fixed_base': 'Fixed Discount Rate\n Base (7%)',
        'fixed_high': 'Fixed Discount Rate\n High (12%)',
        'variable': 'Variable Discount Rate\n Inverse to % AMI (7% to 45%)'
    }

    n_rates = len(PRIVATE_DISCOUNT_RATE_SHORT_KEYS)

    # Build NPV column names dynamically using centralized builders
    npv_cols = {
        discount_rate: create_npv_col(f'{policy_scenario}_mp{menu_mp_input}_', category, 'moreWTP', cost_scenario, f'_{discount_rate}')
        for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS
    }

    fig_discount_rate_sensitivity = create_subplot_grid_histogram(
        dataframes=[
            DATAFRAMES_BY_MP[menu_mp_input][discount_rate][rcm_model] for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS
        ],
        subplot_positions=[(0, i) for i in range(n_rates)],
        x_cols=[npv_cols[discount_rate] for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS],
        x_labels=['Private NPV [$USD2023]'] * n_rates,
        y_labels=['Dwelling units in IRA-Reference Scenario'] + [''] * (n_rates - 1),
        bin_number='auto',
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        subplot_titles=[DISCOUNT_RATE_LABELS.get(discount_rate, discount_rate) for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS],
        figure_size=(5 * n_rates, 10),
        sharex=False,
        sharey=True,
        color_code=f'base_{category}_fuel',
        show_legend=True
    )

    # Print comparison statistics
    print("="*60)
    print("IRA-Reference Scenario\nAdoption Feasibility under Different Discount Rate Assumptions")
    print(f"Cost Scenario: {cost_scenario} | RCM Model: {rcm_model}")
    print("="*60)

    print_positive_percentages_complete(
        dataframes=[DATAFRAMES_BY_MP[menu_mp_input][discount_rate][rcm_model] for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS],
        column_names=[npv_cols[discount_rate] for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS],
        subplot_titles=[DISCOUNT_RATE_LABELS.get(discount_rate, discount_rate) for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS],
        fuel_column=f'base_{category}_fuel'
    )

    display(fig_discount_rate_sensitivity)

# %% [markdown]
# ## Capital Cost Scenario Sensitivity: Disaggregate by Housing Type, Fuel and Technology (Efficiency), and Geographic Region

# %%
# =============================================================================
# TEST 9: Equipment-Level Capital Cost Disaggregation
# =============================================================================
# Validates installed costs disaggregated by equipment type, SEER/AFUE, 
# capacity, and fuel type. Reports P10/P50/P90 for each cost scenario.
#
# Binning approach (floor-based rounding):
#   Capacity: 1.5→2.4 = 2 tons, 2.5→3.4 = 3 tons, ... up to 10 tons
#   SEER: 12.5→13.4 = 13, 13.5→14.4 = 14, ... up to 25
#   AFUE: 77.5→78.4 = 78, 78.5→79.4 = 79, ... up to 98
#   kBTU/h: 35→44 = 40, 45→54 = 50, ... up to 200
# Outlier counts reported for homes outside bin ranges.
#
# Equipment types to compare in Trane data:
#   - ASHP (heating replacement): SEER {15,16,18,19,21} × {2,3,4,5} tons
#   - Central AC (cooling replacement): SEER {15,16,18,19,21} × {2,3,4,5} tons
#   - Gas Furnace (heating replacement): AFUE {80,96} × {80,120} kBTU/h
#   - Propane Furnace (heating replacement): AFUE {80,96} × {80,120} kBTU/h
#
# Results are stored in:
#   CAPITAL_COSTS_SENSITIVITY[menu_mp][enduse][cost_type]
#
#   enduse keys: 'ashp', 'central_ac', 'gas_furnace', 'propane_furnace'
#   cost_type keys: 'heating_replacement', 'cooling_replacement',
#                   'heating_upgrade', 'cooling_upgrade'
# =============================================================================

PRINT_DEBUG = True  # Set to True to run capital cost validation and print summary
if PRINT_DEBUG:
        
    import importlib
    import cmu_tare_model.utils.validate_capital_costs as vcv
    importlib.reload(vcv)
    from cmu_tare_model.utils.validate_capital_costs import run_capital_cost_validation

    # Dictionary to store all capital cost sensitivity results
    # Structure: CAPITAL_COSTS_SENSITIVITY[menu_mp][category][technology][cost_type] = DataFrame
    CAPITAL_COSTS_SENSITIVITY = {}

    # Use HEATING_MEASURE_PACKAGES (already defined, excludes baseline MP0)
    for menu_mp in HEATING_MEASURE_PACKAGES:
        CAPITAL_COSTS_SENSITIVITY[menu_mp] = run_capital_cost_validation(
            df=DATAFRAMES_BY_MP[menu_mp][discount_rate][rcm_model],
            menu_mp=menu_mp,
            # capital_costs_mpx=CAPITAL_COSTS_MPX,
            cost_scenarios=list(REMDB_COST_SCENARIO_KEYS),
        )

    # Print summary of stored results
    print(f"\n{'='*80}")
    print(f"CAPITAL_COSTS_SENSITIVITY dictionary populated:")
    for mp, categories in CAPITAL_COSTS_SENSITIVITY.items():
        for category, technologies in categories.items():
            for technology, cost_types in technologies.items():
                for cost_type, df in cost_types.items():
                    rows = len(df) if not df.empty else 0
                    print(f"  CAPITAL_COSTS_SENSITIVITY[{mp}]['{category}']['{technology}']['{cost_type}']  →  {rows} rows")
    print(f"{'='*80}")

# %%
technology = 'gas_furnace'
category = 'heating'

for menu_mp in HEATING_MEASURE_PACKAGES:
    for cost_type in ['replacement']:
        print(f"\nSummary for MP{menu_mp} - {technology.upper()} - {cost_type.capitalize()}:")
        df = CAPITAL_COSTS_SENSITIVITY[menu_mp][category][technology][cost_type]
        display(df)
        print(f"MP {menu_mp}, {technology}, {cost_type}: {len(df)} rows")

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


