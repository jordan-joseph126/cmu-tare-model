# %%
import os
import pandas as pd
from typing import Dict, List, Optional

# Format the name of the exported results file using the location ID
from datetime import datetime
result_export_time = datetime.now()
model_run_date_time = result_export_time.strftime("%Y-%m-%d_%H-%M")

from config import PROJECT_ROOT
from cmu_tare_model.constants import (
    REMDB_COST_SCENARIO_KEYS, VALID_MENU_MPS,
    PRIVATE_DISCOUNT_RATE_COLS, PRIVATE_DISCOUNT_RATE_SHORT_KEYS
)
from cmu_tare_model.utils.export_model_run_results import export_model_run_output
from cmu_tare_model.utils.column_names import (
    create_cost_col,
    create_npv_case_col,
    NPV_CASE_CATEGORIES,
)

print(f"""
Running the model for the following measure packages:

VALID_MENU_MPS = {VALID_MENU_MPS}
      
The results will be exported using the export_model_run_output function.
Documentation for this function:
{export_model_run_output.__doc__}

Active Capital Cost Scenarios: {REMDB_COST_SCENARIO_KEYS}
Note: Each exported CSV contains columns for ALL active cost scenarios.

""")

# %%
from typing import Dict, List, Optional

def verify_cost_scenario_columns(
    menu_mp: int,
    dataframes_by_discount_rate: Dict[str, pd.DataFrame],
    discount_rate_key: str = 'fixed_base',
    category: str = 'heating',
    cost_type: str = 'upgrade',
    cost_scenarios: Optional[List[str]] = None,
    npv_cases: Optional[List[str]] = None,
) -> None:
    """Print a verification report for cost and NPV columns in an MP export.

    Checks, for each active cost scenario, whether the expected installed
    cost column and each NPV-case column are present in the target
    DataFrame, then lists every economic-adopter column found. This is a
    read-only diagnostic -- it does not modify any DataFrame.

    Args:
        menu_mp: Measure package number (e.g. 3, 4).
        dataframes_by_discount_rate: Dict keyed by discount-rate method
            (e.g. 'fixed_base', 'fixed_low') mapping to the exported
            DataFrame for that method, for this menu_mp.
        discount_rate_key: Which entry of dataframes_by_discount_rate to
            check. Defaults to 'fixed_base'.
        category: Equipment category used to build the cost column name
            ('heating' or 'cooling'). Defaults to 'heating'.
        cost_type: Cost type used to build the cost column name
            ('upgrade' or 'replacement'). Defaults to 'upgrade'.
        cost_scenarios: Cost scenario keys to check. Defaults to
            REMDB_COST_SCENARIO_KEYS.
        npv_cases: NPV case categories to check. Defaults to
            NPV_CASE_CATEGORIES.

    Returns:
        None. Results are printed directly.

    Raises:
        ValueError: If category is not 'heating' or 'cooling'.
        KeyError: If discount_rate_key is not a key in
            dataframes_by_discount_rate.
    """
    # Validate inputs early so a typo produces a clear error, not a
    # silent empty report further down.
    if category not in ('heating', 'cooling'):
        raise ValueError(
            f"Invalid category: '{category}'. Must be 'heating' or "
            f"'cooling'."
        )
    if discount_rate_key not in dataframes_by_discount_rate:
        raise KeyError(
            f"discount_rate_key '{discount_rate_key}' not found. "
            f"Available keys: {list(dataframes_by_discount_rate.keys())}"
        )

    if cost_scenarios is None:
        cost_scenarios = list(REMDB_COST_SCENARIO_KEYS)
    if npv_cases is None:
        npv_cases = list(NPV_CASE_CATEGORIES)

    print(f"{'=' * 80}")
    print(f"VERIFICATION: Cost Scenario Columns in MP{menu_mp} Exported DataFrames")
    print(f"{'=' * 80}")
    print(f"Active cost scenarios: {cost_scenarios}\n")

    scenario_prefix = f'ref2025_mp{menu_mp}_'
    df_check = dataframes_by_discount_rate[discount_rate_key]
    # method_suffix is derived from discount_rate_key, not hardcoded, so
    # this stays correct for fixed_low / fixed_high / variable too.
    method_suffix = f'_{discount_rate_key}'

    for cost_scenario in cost_scenarios:
        cost_col = create_cost_col(
            menu_mp=menu_mp, category=category,
            cost_type=cost_type, cost_scenario=cost_scenario,
        )
        cost_present = cost_col in df_check.columns

        npv_results = {}
        for npv_case in npv_cases:
            npv_col = create_npv_case_col(
                scenario_prefix, npv_case, method_suffix=method_suffix,
            )
            npv_results[npv_case] = (npv_col, npv_col in df_check.columns)

        all_ok = cost_present and all(v[1] for v in npv_results.values())
        status = "PASS" if all_ok else "WARN"
        print(f"  [{status}] {cost_scenario}:")
        flag = "[OK]" if cost_present else "[MISSING]"
        print(f"    Cost column   ({cost_col}): {flag}")
        for npv_case, (col, present) in npv_results.items():
            flag = "[OK]" if present else "[MISSING]"
            print(f"    NPV {npv_case} ({col}): {flag}")

    econ_adopter_cols = sorted(
        c for c in df_check.columns if 'econ_adopter' in c
    )
    print(f"\nEconomic-adopter columns ({len(econ_adopter_cols)}):")
    if econ_adopter_cols:
        for col in econ_adopter_cols:
            print(f"    {col}")
    else:
        print("    (none found)")

    print(f"\nTotal columns in DataFrame: {len(df_check.columns)}")
    print(f"{'=' * 80}")

# %% [markdown]
# # EUSS Baseline Measure Package (MP0)

# %%
# ===================================================================================================================================================================================
# Measure Package 0: Baseline
# ===================================================================================================================================================================================
# BASELINE 2025 Reference Case:
menu_mp = 0
input_mp = 'baseline'

print(f"PROJECT_ROOT (from config.py): {PROJECT_ROOT}")

# Construct the absolute path to the .py file
relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_baseline_v3_0.ipynb")
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

# ===== STUDY SAMPLE FUNNEL (SI table) =====
sample_funnel_dir = os.path.join(
    output_folder_path, "baseline_summary", "sample_funnel")
os.makedirs(sample_funnel_dir, exist_ok=True)
sample_funnel_path = os.path.join(
    sample_funnel_dir, f"sample_funnel_{location_id}_{model_run_date_time}.csv")
df_sample_funnel.to_csv(sample_funnel_path, index=False)
print(f"Saved sample funnel: {sample_funnel_path}")

# %% [markdown]
# -------------------------------------------------------------------------------------------------------
# # EUSS Post-Retrofit Measure Packages
# -------------------------------------------------------------------------------------------------------
# ## 2025 Reference Case:
# - AEO2026 fuel price projections
# - AEO2026 degree-day factors
# - Cambium MidCase electricity grid
# -------------------------------------------------------------------------------------------------------

# %%
print(f"""
-------------------------------------------------------------------------------------------------------
EUSS Post-Retrofit Measure Packages
-------------------------------------------------------------------------------------------------------
Running the model for the following measure packages:

VALID_MENU_MPS = {VALID_MENU_MPS}

-------------------------------------------------------------------------------------------------------
TARE MODEL SCENARIO: 2025 Reference Case
-------------------------------------------------------------------------------------------------------
- AEO2026 fuel price projections
- AEO2026 degree-day factors
- Cambium MidCase electricity grid
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
    relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_scenarios_v3_0.ipynb")
    file_path = os.path.join(PROJECT_ROOT, relative_path)

    # On Windows, to avoid any path-escape quirks, convert backslashes to forward slashes
    file_path = file_path.replace("\\", "/")
    print(f"Running file: {file_path}")

    # iPython magic command to run notebook and import variables into current session
    get_ipython().run_line_magic('run', f'-i {file_path}')

    print(f"Model Run Complete for EUSS Measure Package: MP{menu_mp}")

    # PRESERVE RESULTS AND PREVENT OVERWRITING OF PREVIOUS MODEL RUN DATA
    # Moved this block from the second cell in the section to the first cell
    # This ensures that the results are preserved immediately after the model run.
    print(f"""
    Preserving MP{menu_mp} results by copying dataframe variables and re-assigning to MP-specific names.
    This allows the scenarios file to be re-run for MP{menu_mp} without overwriting previous model run data""")

    # Supplemental DataFrames (single scenario: 2025 Reference Case)
    df_mp3_ref2025_damages_climate = df_mpX_ref2025_damages_climate.copy()
    df_mp3_ref2025_fuel_costs = df_mpX_ref2025_fuel_costs.copy()

    # Summary results dictionary keyed by discount rate: [discount_rate] -> DataFrame
    DATAFRAMES_MP3_RCM_DISCOUNT_RATE_RESULTS = {
        discount_rate: df.copy()
        for discount_rate, df in DATAFRAMES_MPX_RCM_DISCOUNT_RATE.items()
    }

    # Clear the batch mode trigger
    input_measure_package = None

    print(f"MP{menu_mp} results preserved to MP-specific variable names.")


# %%
if 3 in VALID_MENU_MPS:

    # =========================================================================================================
    # EXPORT RESULTS TO CSV - SUPPLEMENTAL DATA
    # =========================================================================================================
    print(f"Exporting MP{menu_mp} Supplemental Data...")

    # ===== DAMAGES RESULTS =====
    export_model_run_output(
        df_results_export=df_mp3_ref2025_damages_climate,
        results_category='damages_climate_ref2025',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # ===== FUEL COSTS RESULTS =====
    export_model_run_output(
        df_results_export=df_mp3_ref2025_fuel_costs,
        results_category='fuel_costs_ref2025',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # =========================================================================================================
    # EXPORT RESULTS TO CSV - SUMMARY RESULTS FOR DISCOUNT RATE SENSITIVITY ANALYSIS
    # =========================================================================================================

    # Process each discount rate (matches dictionary structure)
    for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS:
        print(f"Exporting SUMMARY RESULTS for discount rate: {discount_rate}")

        # Get the DataFrame for this discount rate
        df_results_export = DATAFRAMES_MP3_RCM_DISCOUNT_RATE_RESULTS[discount_rate]

        # Export summary results with the discount rate key (short key)
        export_model_run_output(
            df_results_export=df_results_export,
            results_category='summary',
            menu_mp=menu_mp,
            output_folder_path=output_folder_path,
            location_id=location_id,
            results_export_formatted_date=model_run_date_time,
            discount_rate=discount_rate
        )

    # =============================================================================
    # VERIFICATION: Cost Scenario Column Presence in MP3 Results
    # =============================================================================
    verify_cost_scenario_columns(
        menu_mp=menu_mp,
        dataframes_by_discount_rate=DATAFRAMES_MP3_RCM_DISCOUNT_RATE_RESULTS,
        )

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
    relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_scenarios_v3_0.ipynb")
    file_path = os.path.join(PROJECT_ROOT, relative_path)

    # On Windows, to avoid any path-escape quirks, convert backslashes to forward slashes
    file_path = file_path.replace("\\", "/")
    print(f"Running file: {file_path}")

    # iPython magic command to run notebook and import variables into current session
    get_ipython().run_line_magic('run', f'-i {file_path}')

    print(f"Model Run Complete for EUSS Measure Package: MP{menu_mp}")

    # PRESERVE RESULTS AND PREVENT OVERWRITING OF PREVIOUS MODEL RUN DATA
    # Moved this block from the second cell in the section to the first cell
    # This ensures that the results are preserved immediately after the model run.
    print(f"""
    Preserving MP{menu_mp} results by copying dataframe variables and re-assigning to MP-specific names.
    This allows the scenarios file to be re-run for MP{menu_mp} without overwriting previous model run data""")

    # Supplemental DataFrames (single scenario: 2025 Reference Case)
    df_mp4_ref2025_damages_climate = df_mpX_ref2025_damages_climate.copy()
    df_mp4_ref2025_fuel_costs = df_mpX_ref2025_fuel_costs.copy()

    # Summary results dictionary keyed by discount rate: [discount_rate] -> DataFrame
    DATAFRAMES_MP4_RCM_DISCOUNT_RATE_RESULTS = {
        discount_rate: df.copy()
        for discount_rate, df in DATAFRAMES_MPX_RCM_DISCOUNT_RATE.items()
    }

    # Clear the batch mode trigger
    input_measure_package = None

    print(f"MP{menu_mp} results preserved to MP-specific variable names.")


# %%
if 4 in VALID_MENU_MPS:

    # =========================================================================================================
    # EXPORT RESULTS TO CSV - SUPPLEMENTAL DATA
    # =========================================================================================================
    print(f"Exporting MP{menu_mp} Supplemental Data...")

    # ===== DAMAGES RESULTS =====
    export_model_run_output(
        df_results_export=df_mp4_ref2025_damages_climate,
        results_category='damages_climate_ref2025',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # ===== FUEL COSTS RESULTS =====
    export_model_run_output(
        df_results_export=df_mp4_ref2025_fuel_costs,
        results_category='fuel_costs_ref2025',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # =========================================================================================================
    # EXPORT RESULTS TO CSV - SUMMARY RESULTS FOR DISCOUNT RATE SENSITIVITY ANALYSIS
    # =========================================================================================================

    # Process each discount rate (matches dictionary structure)
    for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS:
        print(f"Exporting SUMMARY RESULTS for discount rate: {discount_rate}")

        # Get the DataFrame for this discount rate
        df_results_export = DATAFRAMES_MP4_RCM_DISCOUNT_RATE_RESULTS[discount_rate]

        # Export summary results with the discount rate key (short key)
        export_model_run_output(
            df_results_export=df_results_export,
            results_category='summary',
            menu_mp=menu_mp,
            output_folder_path=output_folder_path,
            location_id=location_id,
            results_export_formatted_date=model_run_date_time,
            discount_rate=discount_rate
        )
    
    # =============================================================================
    # VERIFICATION: Cost Scenario Column Presence in MP4 Results
    # =============================================================================
    verify_cost_scenario_columns(
        menu_mp=menu_mp,
        dataframes_by_discount_rate=DATAFRAMES_MP4_RCM_DISCOUNT_RATE_RESULTS,
        )

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
    relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_scenarios_v3_0.ipynb")
    file_path = os.path.join(PROJECT_ROOT, relative_path)

    # On Windows, to avoid any path-escape quirks, convert backslashes to forward slashes
    file_path = file_path.replace("\\", "/")
    print(f"Running file: {file_path}")

    # iPython magic command to run notebook and import variables into current session
    get_ipython().run_line_magic('run', f'-i {file_path}')

    print(f"Model Run Complete for EUSS Measure Package: MP{menu_mp}")

    # PRESERVE RESULTS AND PREVENT OVERWRITING OF PREVIOUS MODEL RUN DATA
    # Moved this block from the second cell in the section to the first cell
    # This ensures that the results are preserved immediately after the model run.
    print(f"""
    Preserving MP{menu_mp} results by copying dataframe variables and re-assigning to MP-specific names.
    This allows the scenarios file to be re-run for MP{menu_mp} without overwriting previous model run data""")

    # Supplemental DataFrames (single scenario: 2025 Reference Case)
    df_mp8_ref2025_damages_climate = df_mpX_ref2025_damages_climate.copy()
    df_mp8_ref2025_fuel_costs = df_mpX_ref2025_fuel_costs.copy()

    # Summary results dictionary keyed by discount rate: [discount_rate] -> DataFrame
    DATAFRAMES_MP8_RCM_DISCOUNT_RATE_RESULTS = {
        discount_rate: df.copy()
        for discount_rate, df in DATAFRAMES_MPX_RCM_DISCOUNT_RATE.items()
    }

    # Clear the batch mode trigger
    input_measure_package = None

    print(f"MP{menu_mp} results preserved to MP-specific variable names.")


# %%
if 8 in VALID_MENU_MPS:

    # =========================================================================================================
    # EXPORT RESULTS TO CSV - SUPPLEMENTAL DATA
    # =========================================================================================================
    print(f"Exporting MP{menu_mp} Supplemental Data...")

    # ===== DAMAGES RESULTS =====
    export_model_run_output(
        df_results_export=df_mp8_ref2025_damages_climate,
        results_category='damages_climate_ref2025',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # ===== FUEL COSTS RESULTS =====
    export_model_run_output(
        df_results_export=df_mp8_ref2025_fuel_costs,
        results_category='fuel_costs_ref2025',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # =========================================================================================================
    # EXPORT RESULTS TO CSV - SUMMARY RESULTS FOR DISCOUNT RATE SENSITIVITY ANALYSIS
    # =========================================================================================================

    # Process each discount rate (matches dictionary structure)
    for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS:
        print(f"Exporting SUMMARY RESULTS for discount rate: {discount_rate}")

        # Get the DataFrame for this discount rate
        df_results_export = DATAFRAMES_MP8_RCM_DISCOUNT_RATE_RESULTS[discount_rate]

        # Export summary results with the discount rate key (short key)
        export_model_run_output(
            df_results_export=df_results_export,
            results_category='summary',
            menu_mp=menu_mp,
            output_folder_path=output_folder_path,
            location_id=location_id,
            results_export_formatted_date=model_run_date_time,
            discount_rate=discount_rate
        )
    
    # =============================================================================
    # VERIFICATION: Cost Scenario Column Presence in MP8 Results
    # =============================================================================
    verify_cost_scenario_columns(
        menu_mp=menu_mp,
        dataframes_by_discount_rate=DATAFRAMES_MP8_RCM_DISCOUNT_RATE_RESULTS,
        )

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
    relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_scenarios_v3_0.ipynb")
    file_path = os.path.join(PROJECT_ROOT, relative_path)

    # On Windows, to avoid any path-escape quirks, convert backslashes to forward slashes
    file_path = file_path.replace("\\", "/")
    print(f"Running file: {file_path}")

    # iPython magic command to run notebook and import variables into current session
    get_ipython().run_line_magic('run', f'-i {file_path}')

    print(f"Model Run Complete for EUSS Measure Package: MP{menu_mp}")

    # PRESERVE RESULTS AND PREVENT OVERWRITING OF PREVIOUS MODEL RUN DATA
    # Moved this block from the second cell in the section to the first cell
    # This ensures that the results are preserved immediately after the model run.
    print(f"""
    Preserving MP{menu_mp} results by copying dataframe variables and re-assigning to MP-specific names.
    This allows the scenarios file to be re-run for MP{menu_mp} without overwriting previous model run data""")

    # Supplemental DataFrames (single scenario: 2025 Reference Case)
    df_mp9_ref2025_damages_climate = df_mpX_ref2025_damages_climate.copy()
    df_mp9_ref2025_fuel_costs = df_mpX_ref2025_fuel_costs.copy()

    # Summary results dictionary keyed by discount rate: [discount_rate] -> DataFrame
    DATAFRAMES_MP9_RCM_DISCOUNT_RATE_RESULTS = {
        discount_rate: df.copy()
        for discount_rate, df in DATAFRAMES_MPX_RCM_DISCOUNT_RATE.items()
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
        df_results_export=df_mp9_ref2025_damages_climate,
        results_category='damages_climate_ref2025',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # ===== FUEL COSTS RESULTS =====
    export_model_run_output(
        df_results_export=df_mp9_ref2025_fuel_costs,
        results_category='fuel_costs_ref2025',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # =========================================================================================================
    # EXPORT RESULTS TO CSV - SUMMARY RESULTS FOR DISCOUNT RATE SENSITIVITY ANALYSIS
    # =========================================================================================================

    # Process each discount rate (matches dictionary structure)
    for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS:
        print(f"Exporting SUMMARY RESULTS for discount rate: {discount_rate}")

        # Get the DataFrame for this discount rate
        df_results_export = DATAFRAMES_MP9_RCM_DISCOUNT_RATE_RESULTS[discount_rate]

        # Export summary results with the discount rate key (short key)
        export_model_run_output(
            df_results_export=df_results_export,
            results_category='summary',
            menu_mp=menu_mp,
            output_folder_path=output_folder_path,
            location_id=location_id,
            results_export_formatted_date=model_run_date_time,
            discount_rate=discount_rate
        )

    # =============================================================================
    # VERIFICATION: Cost Scenario Column Presence in MP9 Results
    # =============================================================================
    verify_cost_scenario_columns(
        menu_mp=menu_mp,
        dataframes_by_discount_rate=DATAFRAMES_MP9_RCM_DISCOUNT_RATE_RESULTS,
        )

# %% [markdown]
# ## MP10: Whole Home Electrification + Enhanced Enclosure Upgrade
# 

# %%
if 10 in VALID_MENU_MPS:

    menu_mp = 10
    input_mp = 'upgrade10'

    # Pre-set measure package for batch mode
    input_measure_package = '10'

    print(f"PROJECT_ROOT (from config.py): {PROJECT_ROOT}")

    # Construct the absolute path to the unified scenarios file
    relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_scenarios_v3_0.ipynb")
    file_path = os.path.join(PROJECT_ROOT, relative_path)

    # On Windows, to avoid any path-escape quirks, convert backslashes to forward slashes
    file_path = file_path.replace("\\", "/")
    print(f"Running file: {file_path}")

    # iPython magic command to run notebook and import variables into current session
    get_ipython().run_line_magic('run', f'-i {file_path}')

    print(f"Model Run Complete for EUSS Measure Package: MP{menu_mp}")

    # PRESERVE RESULTS AND PREVENT OVERWRITING OF PREVIOUS MODEL RUN DATA
    # Moved this block from the second cell in the section to the first cell
    # This ensures that the results are preserved immediately after the model run.
    print(f"""
    Preserving MP{menu_mp} results by copying dataframe variables and re-assigning to MP-specific names.
    This allows the scenarios file to be re-run for MP{menu_mp} without overwriting previous model run data""")

    # Supplemental DataFrames (single scenario: 2025 Reference Case)
    df_mp10_ref2025_damages_climate = df_mpX_ref2025_damages_climate.copy()
    df_mp10_ref2025_fuel_costs = df_mpX_ref2025_fuel_costs.copy()

    # Summary results dictionary keyed by discount rate: [discount_rate] -> DataFrame
    DATAFRAMES_MP10_RCM_DISCOUNT_RATE_RESULTS = {
        discount_rate: df.copy()
        for discount_rate, df in DATAFRAMES_MPX_RCM_DISCOUNT_RATE.items()
    }

    # Clear the batch mode trigger
    input_measure_package = None

    print(f"MP{menu_mp} results preserved to MP-specific variable names.")


# %%
if 10 in VALID_MENU_MPS:

    # =========================================================================================================
    # EXPORT RESULTS TO CSV - SUPPLEMENTAL DATA
    # =========================================================================================================
    print(f"Exporting MP{menu_mp} Supplemental Data...")

    # ===== DAMAGES RESULTS =====
    export_model_run_output(
        df_results_export=df_mp10_ref2025_damages_climate,
        results_category='damages_climate_ref2025',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # ===== FUEL COSTS RESULTS =====
    export_model_run_output(
        df_results_export=df_mp10_ref2025_fuel_costs,
        results_category='fuel_costs_ref2025',
        menu_mp=menu_mp,
        output_folder_path=output_folder_path,
        location_id=location_id,
        results_export_formatted_date=model_run_date_time
    )

    # =========================================================================================================
    # EXPORT RESULTS TO CSV - SUMMARY RESULTS FOR DISCOUNT RATE SENSITIVITY ANALYSIS
    # =========================================================================================================

    # Process each discount rate (matches dictionary structure)
    for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS:
        print(f"Exporting SUMMARY RESULTS for discount rate: {discount_rate}")

        # Get the DataFrame for this discount rate
        df_results_export = DATAFRAMES_MP10_RCM_DISCOUNT_RATE_RESULTS[discount_rate]

        # Export summary results with the discount rate key (short key)
        export_model_run_output(
            df_results_export=df_results_export,
            results_category='summary',
            menu_mp=menu_mp,
            output_folder_path=output_folder_path,
            location_id=location_id,
            results_export_formatted_date=model_run_date_time,
            discount_rate=discount_rate
        )

    # =============================================================================
    # VERIFICATION: Cost Scenario Column Presence in MP10 Results
    # =============================================================================
    verify_cost_scenario_columns(
        menu_mp=menu_mp,
        dataframes_by_discount_rate=DATAFRAMES_MP10_RCM_DISCOUNT_RATE_RESULTS,
        )

# %% [markdown]
# ## FINAL SUMMARY

# %%
# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n{'='*80}")
print(f"ALL EXPORTS COMPLETE")
print(f"{'='*80}")
print(f"Cost scenarios exported: {REMDB_COST_SCENARIO_KEYS}")
print(f"Discount rates: {PRIVATE_DISCOUNT_RATE_SHORT_KEYS}")
print(f"CSVs per MP: {len(PRIVATE_DISCOUNT_RATE_SHORT_KEYS)}")
print(f"Total CSVs: {len(VALID_MENU_MPS)} MPs x {len(PRIVATE_DISCOUNT_RATE_SHORT_KEYS)} = {len(VALID_MENU_MPS) * len(PRIVATE_DISCOUNT_RATE_SHORT_KEYS)}")
print(f"{'='*80}")


