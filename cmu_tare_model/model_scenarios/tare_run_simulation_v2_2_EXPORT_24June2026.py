# %%
"""
This file runs all of the code responsible for generating the output CSVs but does not visualize the data.
Data visualization is done in the main program.
"""

# %%
import os
import pandas as pd

# Format the name of the exported results file using the location ID
from datetime import datetime
result_export_time = datetime.now()
model_run_date_time = result_export_time.strftime("%Y-%m-%d_%H-%M")

from config import PROJECT_ROOT
from cmu_tare_model.constants import RCM_MODELS, REMDB_COST_SCENARIO_KEYS, VALID_MENU_MPS, CR_FUNCTIONS, EQUIPMENT_SPECS
from cmu_tare_model.constants import PRIVATE_DISCOUNT_RATE_COLS, PRIVATE_DISCOUNT_RATE_SHORT_KEYS
from cmu_tare_model.utils.export_model_run_results import export_model_run_output
from cmu_tare_model.utils.column_names import create_cost_col, create_npv_col, create_adoption_col

print(f"""
Running the model for the following measure packages:

VALID_MENU_MPS = {VALID_MENU_MPS}
      
The results will be exported using the export_model_run_output function.
Documentation for this function:
{export_model_run_output.__doc__}

Active Capital Cost Scenarios: {REMDB_COST_SCENARIO_KEYS}
Note: Each exported CSV contains columns for ALL active cost scenarios.

""")

# %% [markdown]
# # EUSS Baseline Measure Package (MP0)

# %%
# ===================================================================================================================================================================================
# Measure Package 0: Baseline
# ===================================================================================================================================================================================
# BASELINE Pre-IRA Scenario:
menu_mp = 0
input_mp = 'baseline'

print(f"PROJECT_ROOT (from config.py): {PROJECT_ROOT}")

# Construct the absolute path to the .py file
relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_baseline_v2_2.ipynb")
file_path = os.path.join(PROJECT_ROOT, relative_path)

# On Windows, to avoid any path-escape quirks, convert backslashes to forward slashes
file_path = file_path.replace("\\", "/")

print(f"Running file: {file_path}")

# iPthon magic command to run a .py file and import variables into the current IPython session
get_ipython().run_line_magic('run', f'-i {file_path}')  # If your path has NO spaces, no quotes needed.

print(f"Model Run Complete for Baseline (MP{menu_mp})")

# %%
# ===================================================================================================================================================================================
# EXPORT RESULTS TO CSV
# ===================================================================================================================================================================================

# ===== DAMAGES RESULTS =====
export_model_run_output(
    df_results_export=df_baseline_damages_climate,
    results_category='damages_climate_baseline',
    menu_mp=menu_mp,
    output_folder_path=output_folder_path,
    location_id=location_id,
    results_export_formatted_date=model_run_date_time
    )

export_model_run_output(
    df_results_export=df_baseline_damages_health,
    results_category='damages_health_baseline',
    menu_mp=menu_mp,
    output_folder_path=output_folder_path,
    location_id=location_id,
    results_export_formatted_date=model_run_date_time
    )

# ===== FUEL COSTS RESULTS =====
export_model_run_output(
    df_results_export=df_baseline_fuel_costs,
    results_category='fuel_costs_baseline',
    menu_mp=menu_mp,
    output_folder_path=output_folder_path,
    location_id=location_id,
    results_export_formatted_date=model_run_date_time
    )

# ===== SUMMARY RESULTS =====
export_model_run_output(
    df_results_export=df_euss_am_baseline_home,
    results_category='summary_baseline',
    menu_mp=menu_mp,
    output_folder_path=output_folder_path,
    location_id=location_id,
    results_export_formatted_date=model_run_date_time
    )


# %% [markdown]
# -------------------------------------------------------------------------------------------------------
# # EUSS Post-Retrofit Measure Packages
# -------------------------------------------------------------------------------------------------------
# ## No IRA Scenario:
# - AEO2023 No Inflation Reduction Act
# - Cambium 2021 MidCase
# ## IRA-Reference Scenario:
# - AEO2023 REFERENCE CASE - HDD and Fuel Price Projections
# - Cambium 2022 and 2023 MidCase
# -------------------------------------------------------------------------------------------------------

# %%
print(f"""
-------------------------------------------------------------------------------------------------------
EUSS Post-Retrofit Measure Packages
-------------------------------------------------------------------------------------------------------
Running the model for the following measure packages:

VALID_MENU_MPS = {VALID_MENU_MPS}

-------------------------------------------------------------------------------------------------------
TARE MODEL SCENARIOS
-------------------------------------------------------------------------------------------------------
- Pre-IRA Scenario:
    - NREL End-Use Savings Shapes Database: Measure Package 8/9/10
    - AEO2023 No Inflation Reduction Act
    - Cambium 2021 MidCase
      
- IRA-Reference Scenario:
    - NREL End-Use Savings Shapes Database: Measure Package 8/9/10
    - AEO2023 REFERENCE CASE - HDD and Fuel Price Projections
    - Cambium 2022 and 2023 MidCase
""")

# %% [markdown]
# ## Air Source Heat Pump (ASHP) - Standard Efficiency

# %% [markdown]
# ## MP3: ASHP Retrofit Only (Standard Efficiency)

# %%
if 3 in VALID_MENU_MPS:

    # Measure Package 3
    menu_mp = 3
    input_mp = 'upgrade03'

    # Pre-set measure package for batch mode
    input_measure_package = '3'

    print(f"PROJECT_ROOT (from config.py): {PROJECT_ROOT}")

    # Construct the absolute path to the unified scenarios file
    relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_scenarios_v2_2.ipynb")
    file_path = os.path.join(PROJECT_ROOT, relative_path)

    # On Windows, to avoid any path-escape quirks, convert backslashes to forward slashes
    file_path = file_path.replace("\\", "/")
    print(f"Running file: {file_path}")

    # iPython magic command to run notebook and import variables into current session
    get_ipython().run_line_magic('run', f'-i {file_path}')

    print(f"Model Run Complete for EUSS Measure Package: MP{menu_mp}")

# %%
if 3 in VALID_MENU_MPS:

    # PRESERVE RESULTS AND PREVENT OVERWRITING OF PREVIOUS MODEL RUN DATA
    print(f"""
    Preserving MP{menu_mp} results by copying dataframe variables and re-assigning to MP-specific names.
    This allows the scenarios file to be re-run for MP{menu_mp} without overwriting previous model run data""")

    # Supplemental DataFrames
    df_mp3_noIRA_damages_climate = df_mpX_noIRA_damages_climate.copy()
    df_mp3_IRA_damages_climate = df_mpX_IRA_damages_climate.copy()
    df_mp3_noIRA_damages_health = df_mpX_noIRA_damages_health.copy()
    df_mp3_IRA_damages_health = df_mpX_IRA_damages_health.copy()
    df_mp3_noIRA_fuel_costs = df_mpX_noIRA_fuel_costs.copy()
    df_mp3_IRA_fuel_costs = df_mpX_IRA_fuel_costs.copy()

    # Summary results dictionary (nested: discount rate -> RCM model -> DataFrame)
    # Structure matches new organization: [discount_rate][rcm_model]
    DATAFRAMES_MP3_RCM_DISCOUNT_RATE_RESULTS = {
        discount_rate: {
            rcm_model: df.copy()
            for rcm_model, df in rcm_dict.items()
        }
        for discount_rate, rcm_dict in DATAFRAMES_MPX_RCM_DISCOUNT_RATE.items()
    }

    # Clear the batch mode trigger
    input_measure_package = None

    print(f"MP{menu_mp} results preserved to MP-specific variable names.")


    # =========================================================================================================
    # EXPORT RESULTS TO CSV - SUPPLEMENTAL DATA
    # =========================================================================================================
    print(f"Exporting MP{menu_mp} Supplemental Data...")

    # ===== DAMAGES RESULTS =====
    export_model_run_output(
        df_results_export=df_mp3_noIRA_damages_climate,
        results_category='damages_climate_noIRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp3_IRA_damages_climate,
        results_category='damages_climate_IRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp3_noIRA_damages_health,
        results_category='damages_health_noIRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp3_IRA_damages_health,
        results_category='damages_health_IRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # ===== FUEL COSTS RESULTS =====
    export_model_run_output(
        df_results_export=df_mp3_noIRA_fuel_costs,
        results_category='fuel_costs_noIRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp3_IRA_fuel_costs,
        results_category='fuel_costs_IRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # =========================================================================================================
    # EXPORT RESULTS TO CSV - SUMMARY RESULTS FOR RCM MODELS AND DISCOUNT RATES SENSITIVITY ANALYSIS
    # =========================================================================================================

    # Process each discount rate, then each RCM model (matches dictionary structure)
    for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS:
        print(f"Exporting SUMMARY RESULTS for discount rate: {discount_rate}")
        
        # Process each RCM model for this discount rate
        for rcm_model in RCM_MODELS:
            print(f"  RCM Model: {rcm_model.upper()}")
            
            # Get the specific DataFrame for this discount rate × RCM combination
            df_results_export = DATAFRAMES_MP3_RCM_DISCOUNT_RATE_RESULTS[discount_rate][rcm_model]
            
            # Export summary results with explicit sensitivity parameters (using short keys)
            export_model_run_output(
                df_results_export=df_results_export,
                results_category='summary',
                menu_mp=menu_mp,
                output_folder_path=output_folder_path,
                location_id=location_id,
                results_export_formatted_date=model_run_date_time,
                rcm_model=rcm_model,
                discount_rate=discount_rate
            )

# %%
if 3 in VALID_MENU_MPS:
    
    # =============================================================================
    # VERIFICATION: Cost Scenario Column Presence in MP3 Results
    # =============================================================================
    print(f"{'='*80}")
    print(f"VERIFICATION: Cost Scenario Columns in MP3 Exported DataFrames")
    print(f"{'='*80}")
    print(f"Active cost scenarios: {REMDB_COST_SCENARIO_KEYS}\n")

    for rcm_model in RCM_MODELS:
        print(f"Checking RCM model: {rcm_model.upper()}")    
        # Check a representative DataFrame (fixed_base × first active RCM)
        df_check = DATAFRAMES_MP3_RCM_DISCOUNT_RATE_RESULTS['fixed_base'][rcm_model]


        for cost_scenario in REMDB_COST_SCENARIO_KEYS:
            # Check installed cost columns
            cost_col = create_cost_col(menu_mp=3, category='heating', cost_type='upgrade', cost_scenario=cost_scenario)
            cost_present = cost_col in df_check.columns
            
            # Check NPV columns  
            npv_col = create_npv_col(scenario_prefix='iraRef_mp3_', category='heating', wtp='moreWTP', 
                                    cost_scenario=cost_scenario, method_suffix='_fixed_base')
            npv_present = npv_col in df_check.columns
            
            for cr_function in CR_FUNCTIONS:
                # Check adoption columns
                adopt_col = create_adoption_col(scenario_prefix='iraRef_mp3_', category='heating', column_type='adoption',
                                                cost_scenario=cost_scenario, method_suffix='_fixed_base',
                                                scc_assumption='central', rcm_model=rcm_model, cr_function=cr_function)
                adopt_present = adopt_col in df_check.columns
                
                status = "PASS" if (cost_present and npv_present and adopt_present) else "WARN"
                print(f"  [{status}] {cost_scenario}:")
                print(f"    Cost column   ({cost_col}): {'✓' if cost_present else '✗ MISSING'}")
                print(f"    NPV column    ({npv_col}): {'✓' if npv_present else '✗ MISSING'}")
                print(f"    Adoption col  ({adopt_col}): {'✓' if adopt_present else '✗ MISSING'}")

    # Count v4-suffixed columns
    v4_cols = [c for c in df_check.columns if any(f'_{cs}' in c for cs in REMDB_COST_SCENARIO_KEYS if cs != 'v3')]
    print(f"\nTotal v4-suffixed columns: {len(v4_cols)}")
    print(f"Total columns in DataFrame: {len(df_check.columns)}")
    print(f"{'='*80}")

# %% [markdown]
# ## Air Source Heat Pump (ASHP) - High Efficiency

# %% [markdown]
# ## MP4: ASHP Retrofit Only (High Efficiency)

# %%
if 4 in VALID_MENU_MPS:

    # Measure Package 4
    menu_mp = 4
    input_mp = 'upgrade04'

    # Pre-set measure package for batch mode
    input_measure_package = '4'

    print(f"PROJECT_ROOT (from config.py): {PROJECT_ROOT}")

    # Construct the absolute path to the unified scenarios file
    relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_scenarios_v2_2.ipynb")
    file_path = os.path.join(PROJECT_ROOT, relative_path)

    # On Windows, to avoid any path-escape quirks, convert backslashes to forward slashes
    file_path = file_path.replace("\\", "/")
    print(f"Running file: {file_path}")

    # iPython magic command to run notebook and import variables into current session
    get_ipython().run_line_magic('run', f'-i {file_path}')

    print(f"Model Run Complete for EUSS Measure Package: MP{menu_mp}")

# %%
if 4 in VALID_MENU_MPS:

    # PRESERVE RESULTS AND PREVENT OVERWRITING OF PREVIOUS MODEL RUN DATA
    print(f"""
    Preserving MP{menu_mp} results by copying dataframe variables and re-assigning to MP-specific names.
    This allows the scenarios file to be re-run for MP{menu_mp} without overwriting previous model run data""")

    # Supplemental DataFrames
    df_mp4_noIRA_damages_climate = df_mpX_noIRA_damages_climate.copy()
    df_mp4_IRA_damages_climate = df_mpX_IRA_damages_climate.copy()
    df_mp4_noIRA_damages_health = df_mpX_noIRA_damages_health.copy()
    df_mp4_IRA_damages_health = df_mpX_IRA_damages_health.copy()
    df_mp4_noIRA_fuel_costs = df_mpX_noIRA_fuel_costs.copy()
    df_mp4_IRA_fuel_costs = df_mpX_IRA_fuel_costs.copy()

    # Summary results dictionary (nested: discount rate -> RCM model -> DataFrame)
    # Structure matches new organization: [discount_rate][rcm_model]
    DATAFRAMES_MP4_RCM_DISCOUNT_RATE_RESULTS = {
        discount_rate: {
            rcm_model: df.copy()
            for rcm_model, df in rcm_dict.items()
        }
        for discount_rate, rcm_dict in DATAFRAMES_MPX_RCM_DISCOUNT_RATE.items()
    }

    # Clear the batch mode trigger
    input_measure_package = None

    print(f"MP{menu_mp} results preserved to MP-specific variable names.")


    # =========================================================================================================
    # EXPORT RESULTS TO CSV - SUPPLEMENTAL DATA
    # =========================================================================================================
    print(f"Exporting MP{menu_mp} Supplemental Data...")

    # ===== DAMAGES RESULTS =====
    export_model_run_output(
        df_results_export=df_mp4_noIRA_damages_climate,
        results_category='damages_climate_noIRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp4_IRA_damages_climate,
        results_category='damages_climate_IRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp4_noIRA_damages_health,
        results_category='damages_health_noIRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp4_IRA_damages_health,
        results_category='damages_health_IRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # ===== FUEL COSTS RESULTS =====
    export_model_run_output(
        df_results_export=df_mp4_noIRA_fuel_costs,
        results_category='fuel_costs_noIRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp4_IRA_fuel_costs,
        results_category='fuel_costs_IRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # =========================================================================================================
    # EXPORT RESULTS TO CSV - SUMMARY RESULTS FOR RCM MODELS AND DISCOUNT RATES SENSITIVITY ANALYSIS
    # =========================================================================================================

    # Process each discount rate, then each RCM model (matches dictionary structure)
    for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS:
        print(f"Exporting SUMMARY RESULTS for discount rate: {discount_rate}")
        
        # Process each RCM model for this discount rate
        for rcm_model in RCM_MODELS:
            print(f"  RCM Model: {rcm_model.upper()}")
            
            # Get the specific DataFrame for this discount rate × RCM combination
            df_results_export = DATAFRAMES_MP4_RCM_DISCOUNT_RATE_RESULTS[discount_rate][rcm_model]
            
            # Export summary results with explicit sensitivity parameters (using short keys)
            export_model_run_output(
                df_results_export=df_results_export,
                results_category='summary',
                menu_mp=menu_mp,
                output_folder_path=output_folder_path,
                location_id=location_id,
                results_export_formatted_date=model_run_date_time,
                rcm_model=rcm_model,
                discount_rate=discount_rate
            )

# %%
if 4 in VALID_MENU_MPS:
    
    # =============================================================================
    # VERIFICATION: Cost Scenario Column Presence in MP4 Results
    # =============================================================================
    print(f"{'='*80}")
    print(f"VERIFICATION: Cost Scenario Columns in MP4 Exported DataFrames")
    print(f"{'='*80}")
    print(f"Active cost scenarios: {REMDB_COST_SCENARIO_KEYS}\n")

    for rcm_model in RCM_MODELS:
        print(f"Checking RCM model: {rcm_model.upper()}")    
        # Check a representative DataFrame (fixed_base × first active RCM)
        df_check = DATAFRAMES_MP4_RCM_DISCOUNT_RATE_RESULTS['fixed_base'][rcm_model]


        for cost_scenario in REMDB_COST_SCENARIO_KEYS:
            # Check installed cost columns
            cost_col = create_cost_col(menu_mp=4, category='heating', cost_type='upgrade', cost_scenario=cost_scenario)
            cost_present = cost_col in df_check.columns
            
            # Check NPV columns  
            npv_col = create_npv_col(scenario_prefix='iraRef_mp4_', category='heating', wtp='moreWTP', 
                                    cost_scenario=cost_scenario, method_suffix='_fixed_base')
            npv_present = npv_col in df_check.columns
            
            for cr_function in CR_FUNCTIONS:
                # Check adoption columns
                adopt_col = create_adoption_col(scenario_prefix='iraRef_mp4_', category='heating', column_type='adoption',
                                                cost_scenario=cost_scenario, method_suffix='_fixed_base',
                                                scc_assumption='central', rcm_model=rcm_model, cr_function=cr_function)
                adopt_present = adopt_col in df_check.columns
                
                status = "PASS" if (cost_present and npv_present and adopt_present) else "WARN"
                print(f"  [{status}] {cost_scenario}:")
                print(f"    Cost column   ({cost_col}): {'✓' if cost_present else '✗ MISSING'}")
                print(f"    NPV column    ({npv_col}): {'✓' if npv_present else '✗ MISSING'}")
                print(f"    Adoption col  ({adopt_col}): {'✓' if adopt_present else '✗ MISSING'}")

    # Count v4-suffixed columns
    v4_cols = [c for c in df_check.columns if any(f'_{cs}' in c for cs in REMDB_COST_SCENARIO_KEYS if cs != 'v4')]
    print(f"\nTotal v4-suffixed columns: {len(v4_cols)}")
    print(f"Total columns in DataFrame: {len(df_check.columns)}")
    print(f"{'='*80}")

# %% [markdown]
# ## MP8: Whole Home Electrification (High Efficiency)

# %%
if 8 in VALID_MENU_MPS:

    # Measure Package 8
    menu_mp = 8
    input_mp = 'upgrade08'

    # Pre-set measure package for batch mode
    input_measure_package = '8'

    print(f"PROJECT_ROOT (from config.py): {PROJECT_ROOT}")

    # Construct the absolute path to the unified scenarios file
    relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_scenarios_v2_2.ipynb")
    file_path = os.path.join(PROJECT_ROOT, relative_path)

    # On Windows, to avoid any path-escape quirks, convert backslashes to forward slashes
    file_path = file_path.replace("\\", "/")
    print(f"Running file: {file_path}")

    # iPython magic command to run notebook and import variables into current session
    get_ipython().run_line_magic('run', f'-i {file_path}')

    print(f"Model Run Complete for EUSS Measure Package: MP{menu_mp}")

# %%
if 8 in VALID_MENU_MPS:

    # PRESERVE RESULTS AND PREVENT OVERWRITING OF PREVIOUS MODEL RUN DATA
    print(f"""
    Preserving MP{menu_mp} results by copying dataframe variables and re-assigning to MP-specific names.
    This allows the scenarios file to be re-run for MP{menu_mp} without overwriting previous model run data""")

    # Supplemental DataFrames
    df_mp8_noIRA_damages_climate = df_mpX_noIRA_damages_climate.copy()
    df_mp8_IRA_damages_climate = df_mpX_IRA_damages_climate.copy()
    df_mp8_noIRA_damages_health = df_mpX_noIRA_damages_health.copy()
    df_mp8_IRA_damages_health = df_mpX_IRA_damages_health.copy()
    df_mp8_noIRA_fuel_costs = df_mpX_noIRA_fuel_costs.copy()
    df_mp8_IRA_fuel_costs = df_mpX_IRA_fuel_costs.copy()

    # Summary results dictionary (nested: discount rate -> RCM model -> DataFrame)
    # Structure matches new organization: [discount_rate][rcm_model]
    DATAFRAMES_MP8_RCM_DISCOUNT_RATE_RESULTS = {
        discount_rate: {
            rcm_model: df.copy()
            for rcm_model, df in rcm_dict.items()
        }
        for discount_rate, rcm_dict in DATAFRAMES_MPX_RCM_DISCOUNT_RATE.items()
    }

    # Clear the batch mode trigger
    input_measure_package = None

    print(f"MP{menu_mp} results preserved to MP-specific variable names.")


    # =========================================================================================================
    # EXPORT RESULTS TO CSV - SUPPLEMENTAL DATA
    # =========================================================================================================
    print(f"Exporting MP{menu_mp} Supplemental Data...")

    # ===== DAMAGES RESULTS =====
    export_model_run_output(
        df_results_export=df_mp8_noIRA_damages_climate,
        results_category='damages_climate_noIRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp8_IRA_damages_climate,
        results_category='damages_climate_IRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp8_noIRA_damages_health,
        results_category='damages_health_noIRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp8_IRA_damages_health,
        results_category='damages_health_IRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # ===== FUEL COSTS RESULTS =====
    export_model_run_output(
        df_results_export=df_mp8_noIRA_fuel_costs,
        results_category='fuel_costs_noIRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp8_IRA_fuel_costs,
        results_category='fuel_costs_IRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # =========================================================================================================
    # EXPORT RESULTS TO CSV - SUMMARY RESULTS FOR RCM MODELS AND DISCOUNT RATES SENSITIVITY ANALYSIS
    # =========================================================================================================

    # Process each discount rate, then each RCM model (matches dictionary structure)
    for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS:
        print(f"Exporting SUMMARY RESULTS for discount rate: {discount_rate}")
        
        # Process each RCM model for this discount rate
        for rcm_model in RCM_MODELS:
            print(f"  RCM Model: {rcm_model.upper()}")
            
            # Get the specific DataFrame for this discount rate × RCM combination
            df_results_export = DATAFRAMES_MP8_RCM_DISCOUNT_RATE_RESULTS[discount_rate][rcm_model]
            
            # Export summary results with explicit sensitivity parameters (using short keys)
            export_model_run_output(
                df_results_export=df_results_export,
                results_category='summary',
                menu_mp=menu_mp,
                output_folder_path=output_folder_path,
                location_id=location_id,
                results_export_formatted_date=model_run_date_time,
                rcm_model=rcm_model,
                discount_rate=discount_rate
            )

# %%
if 8 in VALID_MENU_MPS:
    
    # =============================================================================
    # VERIFICATION: Cost Scenario Column Presence in MP8 Results
    # =============================================================================
    print(f"{'='*80}")
    print(f"VERIFICATION: Cost Scenario Columns in MP8 Exported DataFrames")
    print(f"{'='*80}")
    print(f"Active cost scenarios: {REMDB_COST_SCENARIO_KEYS}\n")

    for rcm_model in RCM_MODELS:
        print(f"Checking RCM model: {rcm_model.upper()}")    
        # Check a representative DataFrame (fixed_base × first active RCM)
        df_check = DATAFRAMES_MP8_RCM_DISCOUNT_RATE_RESULTS['fixed_base'][rcm_model]


        for cost_scenario in REMDB_COST_SCENARIO_KEYS:
            # Check installed cost columns
            cost_col = create_cost_col(menu_mp=8, category='heating', cost_type='upgrade', cost_scenario=cost_scenario)
            cost_present = cost_col in df_check.columns
            
            # Check NPV columns  
            npv_col = create_npv_col(scenario_prefix='iraRef_mp8_', category='heating', wtp='moreWTP', 
                                    cost_scenario=cost_scenario, method_suffix='_fixed_base')
            npv_present = npv_col in df_check.columns
            
            for cr_function in CR_FUNCTIONS:
                # Check adoption columns
                adopt_col = create_adoption_col(scenario_prefix='iraRef_mp8_', category='heating', column_type='adoption',
                                                cost_scenario=cost_scenario, method_suffix='_fixed_base',
                                                scc_assumption='central', rcm_model=rcm_model, cr_function=cr_function)
                adopt_present = adopt_col in df_check.columns
                
                status = "PASS" if (cost_present and npv_present and adopt_present) else "WARN"
                print(f"  [{status}] {cost_scenario}:")
                print(f"    Cost column   ({cost_col}): {'✓' if cost_present else '✗ MISSING'}")
                print(f"    NPV column    ({npv_col}): {'✓' if npv_present else '✗ MISSING'}")
                print(f"    Adoption col  ({adopt_col}): {'✓' if adopt_present else '✗ MISSING'}")

    # Count v4-suffixed columns
    v4_cols = [c for c in df_check.columns if any(f'_{cs}' in c for cs in REMDB_COST_SCENARIO_KEYS if cs != 'v8')]
    print(f"\nTotal v4-suffixed columns: {len(v4_cols)}")
    print(f"Total columns in DataFrame: {len(df_check.columns)}")
    print(f"{'='*80}")

# %% [markdown]
# ## MP9: Whole Home Electrification + Basic Enclosure Upgrade
# 

# %%
if 9 in VALID_MENU_MPS:

    # Measure Package 9
    menu_mp = 9
    input_mp = 'upgrade09'

    # Pre-set measure package for batch mode
    input_measure_package = '9'

    print(f"PROJECT_ROOT (from config.py): {PROJECT_ROOT}")

    # Construct the absolute path to the unified scenarios file
    relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_scenarios_v2_2.ipynb")
    file_path = os.path.join(PROJECT_ROOT, relative_path)

    # On Windows, to avoid any path-escape quirks, convert backslashes to forward slashes
    file_path = file_path.replace("\\", "/")
    print(f"Running file: {file_path}")

    # iPython magic command to run notebook and import variables into current session
    get_ipython().run_line_magic('run', f'-i {file_path}')

    print(f"Model Run Complete for EUSS Measure Package: MP{menu_mp}")

    # PRESERVE RESULTS AND PREVENT OVERWRITING OF PREVIOUS MODEL RUN DATA
    print(f"""
    Preserving MP{menu_mp} results by copying dataframe variables and re-assigning to MP-specific names.
    This allows the scenarios file to be re-run for MP{menu_mp} without overwriting previous model run data""")

    # Supplemental DataFrames
    df_mp9_noIRA_damages_climate = df_mpX_noIRA_damages_climate.copy()
    df_mp9_IRA_damages_climate = df_mpX_IRA_damages_climate.copy()
    df_mp9_noIRA_damages_health = df_mpX_noIRA_damages_health.copy()
    df_mp9_IRA_damages_health = df_mpX_IRA_damages_health.copy()
    df_mp9_noIRA_fuel_costs = df_mpX_noIRA_fuel_costs.copy()
    df_mp9_IRA_fuel_costs = df_mpX_IRA_fuel_costs.copy()

    # Summary results dictionary (nested: discount rate -> RCM model -> DataFrame)
    # Structure matches new organization: [discount_rate][rcm_model]
    DATAFRAMES_MP9_RCM_DISCOUNT_RATE_RESULTS = {
        discount_rate: {
            rcm_model: df.copy()
            for rcm_model, df in rcm_dict.items()
        }
        for discount_rate, rcm_dict in DATAFRAMES_MPX_RCM_DISCOUNT_RATE.items()
    }

    # Clear the batch mode trigger
    input_measure_package = None

    print(f"MP{menu_mp} results preserved to MP-specific variable names.")

# %%
if 9 in VALID_MENU_MPS:

    # =========================================================================================================
    # EXPORT RESULTS TO CSV - SUPPLEMENTAL DATA
    # =========================================================================================================
    print(f"Exporting MP{menu_mp} Supplemental Data...")

    # ===== DAMAGES RESULTS =====
    export_model_run_output(
        df_results_export=df_mp9_noIRA_damages_climate,
        results_category='damages_climate_noIRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp9_IRA_damages_climate,
        results_category='damages_climate_IRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp9_noIRA_damages_health,
        results_category='damages_health_noIRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp9_IRA_damages_health,
        results_category='damages_health_IRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # ===== FUEL COSTS RESULTS =====
    export_model_run_output(
        df_results_export=df_mp9_noIRA_fuel_costs,
        results_category='fuel_costs_noIRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp9_IRA_fuel_costs,
        results_category='fuel_costs_IRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # =========================================================================================================
    # EXPORT RESULTS TO CSV - SUMMARY RESULTS FOR RCM MODELS AND DISCOUNT RATES SENSITIVITY ANALYSIS
    # =========================================================================================================

    # Process each discount rate, then each RCM model (matches dictionary structure)
    for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS:
        print(f"Exporting SUMMARY RESULTS for discount rate: {discount_rate}")
        
        # Process each RCM model for this discount rate
        for rcm_model in RCM_MODELS:
            print(f"  RCM Model: {rcm_model.upper()}")
            
            # Get the specific DataFrame for this discount rate × RCM combination
            df_results_export = DATAFRAMES_MP9_RCM_DISCOUNT_RATE_RESULTS[discount_rate][rcm_model]
            
            # Export summary results with explicit sensitivity parameters (using short keys)
            export_model_run_output(
                df_results_export=df_results_export,
                results_category='summary',
                menu_mp=menu_mp,
                output_folder_path=output_folder_path,
                location_id=location_id,
                results_export_formatted_date=model_run_date_time,
                rcm_model=rcm_model,
                discount_rate=discount_rate
            )

# %%
if 9 in VALID_MENU_MPS:
    
    # =============================================================================
    # VERIFICATION: Cost Scenario Column Presence in MP9 Results
    # =============================================================================
    print(f"{'='*80}")
    print(f"VERIFICATION: Cost Scenario Columns in MP9 Exported DataFrames")
    print(f"{'='*80}")
    print(f"Active cost scenarios: {REMDB_COST_SCENARIO_KEYS}\n")

    for rcm_model in RCM_MODELS:
        print(f"Checking RCM model: {rcm_model.upper()}")    
        # Check a representative DataFrame (fixed_base × first active RCM)
        df_check = DATAFRAMES_MP9_RCM_DISCOUNT_RATE_RESULTS['fixed_base'][rcm_model]


        for cost_scenario in REMDB_COST_SCENARIO_KEYS:
            # Check installed cost columns
            cost_col = create_cost_col(menu_mp=9, category='heating', cost_type='upgrade', cost_scenario=cost_scenario)
            cost_present = cost_col in df_check.columns
            
            # Check NPV columns  
            npv_col = create_npv_col(scenario_prefix='iraRef_mp9_', category='heating', wtp='moreWTP', 
                                    cost_scenario=cost_scenario, method_suffix='_fixed_base')
            npv_present = npv_col in df_check.columns
            
            for cr_function in CR_FUNCTIONS:
                # Check adoption columns
                adopt_col = create_adoption_col(scenario_prefix='iraRef_mp9_', category='heating', column_type='adoption',
                                                cost_scenario=cost_scenario, method_suffix='_fixed_base',
                                                scc_assumption='central', rcm_model=rcm_model, cr_function=cr_function)
                adopt_present = adopt_col in df_check.columns
                
                status = "PASS" if (cost_present and npv_present and adopt_present) else "WARN"
                print(f"  [{status}] {cost_scenario}:")
                print(f"    Cost column   ({cost_col}): {'✓' if cost_present else '✗ MISSING'}")
                print(f"    NPV column    ({npv_col}): {'✓' if npv_present else '✗ MISSING'}")
                print(f"    Adoption col  ({adopt_col}): {'✓' if adopt_present else '✗ MISSING'}")

    # Count v4-suffixed columns
    v4_cols = [c for c in df_check.columns if any(f'_{cs}' in c for cs in REMDB_COST_SCENARIO_KEYS if cs != 'v9')]
    print(f"\nTotal v4-suffixed columns: {len(v4_cols)}")
    print(f"Total columns in DataFrame: {len(df_check.columns)}")
    print(f"{'='*80}")

# %% [markdown]
# ## MP10: Whole Home Electrification + Enhanced Enclosure Upgrade
# 

# %%
# Measure Package 10
if 10 in VALID_MENU_MPS:

    menu_mp = 10
    input_mp = 'upgrade10'

    # Pre-set measure package for batch mode
    input_measure_package = '10'

    print(f"PROJECT_ROOT (from config.py): {PROJECT_ROOT}")

    # Construct the absolute path to the unified scenarios file
    relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_scenarios_v2_2.ipynb")
    file_path = os.path.join(PROJECT_ROOT, relative_path)

    # On Windows, to avoid any path-escape quirks, convert backslashes to forward slashes
    file_path = file_path.replace("\\", "/")
    print(f"Running file: {file_path}")

    # iPython magic command to run notebook and import variables into current session
    get_ipython().run_line_magic('run', f'-i {file_path}')

    print(f"Model Run Complete for EUSS Measure Package: MP{menu_mp}")

    # PRESERVE RESULTS AND PREVENT OVERWRITING OF PREVIOUS MODEL RUN DATA
    print(f"""
    Preserving MP{menu_mp} results by copying dataframe variables and re-assigning to MP-specific names.
    This allows the scenarios file to be re-run for MP{menu_mp} without overwriting previous model run data""")

    # Supplemental DataFrames
    df_mp10_noIRA_damages_climate = df_mpX_noIRA_damages_climate.copy()
    df_mp10_IRA_damages_climate = df_mpX_IRA_damages_climate.copy()
    df_mp10_noIRA_damages_health = df_mpX_noIRA_damages_health.copy()
    df_mp10_IRA_damages_health = df_mpX_IRA_damages_health.copy()
    df_mp10_noIRA_fuel_costs = df_mpX_noIRA_fuel_costs.copy()
    df_mp10_IRA_fuel_costs = df_mpX_IRA_fuel_costs.copy()

    # Summary results dictionary (nested: discount rate -> RCM model -> DataFrame)
    # Structure matches new organization: [discount_rate][rcm_model]
    DATAFRAMES_MP10_RCM_DISCOUNT_RATE_RESULTS = {
        discount_rate: {
            rcm_model: df.copy()
            for rcm_model, df in rcm_dict.items()
        }
        for discount_rate, rcm_dict in DATAFRAMES_MPX_RCM_DISCOUNT_RATE.items()
    }

    # Clear the batch mode trigger
    input_measure_package = None

    print(f"MP{menu_mp} results preserved to MP-specific variable names.")

# %%
# Measure Package 10
if 10 in VALID_MENU_MPS:

    # =========================================================================================================
    # EXPORT RESULTS TO CSV - SUPPLEMENTAL DATA
    # =========================================================================================================
    print(f"Exporting MP{menu_mp} Supplemental Data...")

    # ===== DAMAGES RESULTS =====
    export_model_run_output(
        df_results_export=df_mp10_noIRA_damages_climate,
        results_category='damages_climate_noIRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp10_IRA_damages_climate,
        results_category='damages_climate_IRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp10_noIRA_damages_health,
        results_category='damages_health_noIRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp10_IRA_damages_health,
        results_category='damages_health_IRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # ===== FUEL COSTS RESULTS =====
    export_model_run_output(
        df_results_export=df_mp10_noIRA_fuel_costs,
        results_category='fuel_costs_noIRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    export_model_run_output(
        df_results_export=df_mp10_IRA_fuel_costs,
        results_category='fuel_costs_IRA',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # =========================================================================================================
    # EXPORT RESULTS TO CSV - SUMMARY RESULTS FOR RCM MODELS AND DISCOUNT RATES SENSITIVITY ANALYSIS
    # =========================================================================================================

    # Process each discount rate, then each RCM model (matches dictionary structure)
    for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS:
        print(f"Exporting SUMMARY RESULTS for discount rate: {discount_rate}")
        
        # Process each RCM model for this discount rate
        for rcm_model in RCM_MODELS:
            print(f"  RCM Model: {rcm_model.upper()}")
            
            # Get the specific DataFrame for this discount rate × RCM combination
            df_results_export = DATAFRAMES_MP10_RCM_DISCOUNT_RATE_RESULTS[discount_rate][rcm_model]
            
            # Export summary results with explicit sensitivity parameters (using short keys)
            export_model_run_output(
                df_results_export=df_results_export,
                results_category='summary',
                menu_mp=menu_mp,
                output_folder_path=output_folder_path,
                location_id=location_id,
                results_export_formatted_date=model_run_date_time,
                rcm_model=rcm_model,
                discount_rate=discount_rate
            )

# %%
if 10 in VALID_MENU_MPS:
    
    # =============================================================================
    # VERIFICATION: Cost Scenario Column Presence in MP10 Results
    # =============================================================================
    print(f"{'='*80}")
    print(f"VERIFICATION: Cost Scenario Columns in MP10 Exported DataFrames")
    print(f"{'='*80}")
    print(f"Active cost scenarios: {REMDB_COST_SCENARIO_KEYS}\n")

    for rcm_model in RCM_MODELS:
        print(f"Checking RCM model: {rcm_model.upper()}")    
        # Check a representative DataFrame (fixed_base × first active RCM)
        df_check = DATAFRAMES_MP10_RCM_DISCOUNT_RATE_RESULTS['fixed_base'][rcm_model]


        for cost_scenario in REMDB_COST_SCENARIO_KEYS:
            # Check installed cost columns
            cost_col = create_cost_col(menu_mp=10, category='heating', cost_type='upgrade', cost_scenario=cost_scenario)
            cost_present = cost_col in df_check.columns
            
            # Check NPV columns  
            npv_col = create_npv_col(scenario_prefix='iraRef_mp10_', category='heating', wtp='moreWTP', 
                                    cost_scenario=cost_scenario, method_suffix='_fixed_base')
            npv_present = npv_col in df_check.columns
            
            for cr_function in CR_FUNCTIONS:
                # Check adoption columns
                adopt_col = create_adoption_col(scenario_prefix='iraRef_mp10_', category='heating', column_type='adoption',
                                                cost_scenario=cost_scenario, method_suffix='_fixed_base',
                                                scc_assumption='central', rcm_model=rcm_model, cr_function=cr_function)
                adopt_present = adopt_col in df_check.columns
                
                status = "PASS" if (cost_present and npv_present and adopt_present) else "WARN"
                print(f"  [{status}] {cost_scenario}:")
                print(f"    Cost column   ({cost_col}): {'✓' if cost_present else '✗ MISSING'}")
                print(f"    NPV column    ({npv_col}): {'✓' if npv_present else '✗ MISSING'}")
                print(f"    Adoption col  ({adopt_col}): {'✓' if adopt_present else '✗ MISSING'}")

    # Count v4-suffixed columns
    v4_cols = [c for c in df_check.columns if any(f'_{cs}' in c for cs in REMDB_COST_SCENARIO_KEYS if cs != 'v3')]
    print(f"\nTotal v4-suffixed columns: {len(v4_cols)}")
    print(f"Total columns in DataFrame: {len(df_check.columns)}")
    print(f"{'='*80}")


    # =============================================================================
    # FINAL SUMMARY
    # =============================================================================
    print(f"\n{'='*80}")
    print(f"ALL EXPORTS COMPLETE")
    print(f"{'='*80}")
    print(f"Cost scenarios exported: {REMDB_COST_SCENARIO_KEYS}")
    print(f"Discount rates: {PRIVATE_DISCOUNT_RATE_SHORT_KEYS}")
    print(f"RCM models: {RCM_MODELS}")
    print(f"CSVs per MP: {len(PRIVATE_DISCOUNT_RATE_SHORT_KEYS)} × {len(RCM_MODELS)} = {len(PRIVATE_DISCOUNT_RATE_SHORT_KEYS) * len(RCM_MODELS)}")
    print(f"Total CSVs: 3 MPs × {len(PRIVATE_DISCOUNT_RATE_SHORT_KEYS) * len(RCM_MODELS)} = {3 * len(PRIVATE_DISCOUNT_RATE_SHORT_KEYS) * len(RCM_MODELS)}")
    print(f"{'='*80}")


