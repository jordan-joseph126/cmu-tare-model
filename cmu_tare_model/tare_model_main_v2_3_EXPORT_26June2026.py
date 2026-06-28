# %% [markdown]
# -------------------------------------------------------------------------------------------------------
# # EUSS Post-Retrofit Measure Packages: MP8, MP9, MP10
# -------------------------------------------------------------------------------------------------------
# - MP8: Whole Home Electrification (High Efficiency)
# - MP9: Whole-Home Electrification + Basic Enclosure Upgrade
# - MP10: Whole-Home Electrification + Enhanced Enclosure Upgrade
# 
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
)
from cmu_tare_model.constants import (
    PRIVATE_DISCOUNT_RATE_COLS, 
    PRIVATE_DISCOUNT_RATE_SHORT_KEYS
)

# Column name builders
from cmu_tare_model.utils.column_names import (
    NPV_CASE_CATEGORIES,
    create_cost_col,
    create_capital_col,
    create_npv_col,
    create_npv_case_col,
    create_rebate_col,
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
    mp = 8
    scenario_prefix = f'ref2025_mp{mp}_'
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
    mp = 8
    scenario_prefix = f'ref2025_mp{mp}_'
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
# ## [TO-DO] Space Heating - Basic (MP8), Moderate (MP9), Advanced (MP10) Retrofit
# 
# ## UPDATE THE SIMPLIFIED DOTPLOT CODE FROM THE ADOPTION KPIS 

# %%


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


