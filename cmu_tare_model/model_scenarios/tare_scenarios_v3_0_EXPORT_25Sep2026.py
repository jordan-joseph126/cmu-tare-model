# %%
# Run the baseline notebook only when this notebook is run on its own.
# tare_run_simulation has already run the baseline (and asked the location
# questions) before it runs this notebook for each package, so re-running
# it here would repeat the whole baseline and ask the questions again.
if 'df_euss_am_baseline_home' not in globals():
    import os
    from config import PROJECT_ROOT

    # Measure Package 0: Baseline
    menu_mp = 0
    input_mp = 'baseline'
    print(f"PROJECT_ROOT (from config.py): {PROJECT_ROOT}")

    # Construct the absolute path to the .py file
    relative_path = os.path.join("cmu_tare_model", "model_scenarios", "tare_baseline_v3_0.ipynb")
    file_path = os.path.join(PROJECT_ROOT, relative_path)

    # On Windows, to avoid any path-escape quirks, convert backslashes to forward slashes
    file_path = file_path.replace("\\", "/")
    print(f"Running file: {file_path}")


    %run -i {file_path}
    print("Baseline Scenario - Model Run Complete")

    # Flag to prevent excessive output in other scenario files
    individual_scenario_run = True

    na = df_euss_am_baseline_home['gea_region'].isna()
    print("homes with no GEA region:", int(na.sum()))

    print("states affected:", sorted(df_euss_am_baseline_home.loc[na, 'state'].dropna().unique()))

    print(
        df_euss_am_baseline_home.loc[na, ['state', 'county', 'county_fips']]
        .drop_duplicates()
        .sort_values('county_fips')
        .head(40)
        .to_string(index=False)
    )


# %% [markdown]
# ## MEASURE PACKAGE X (MPX): Metadata and Annual Enduse Energy Consumption

# %%
# TODO: Update to provide explicit imports and add techfilter to the table (should show 0% ASHP after that stage)

from cmu_tare_model.constants import RESSTOCK_RELEASE_THIS_RUN, VALID_MENU_MPS
from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import (
    load_and_filter_upgrade,
)

# Packages this release can run; the baseline (MP0) is not selectable.
SELECTABLE_MPS = [mp for mp in VALID_MENU_MPS if mp != 0]
SELECTABLE_MPS_STRINGS = [str(mp) for mp in SELECTABLE_MPS]


def mp_to_upgrade_name(mp_num: int) -> str:
    """ResStock upgrade name: 2022.1.1 zero-pads ('upgrade03'), 2025.1 does not."""
    if RESSTOCK_RELEASE_THIS_RUN == '2022.1.1':
        return f"upgrade{mp_num:02d}"
    return f"upgrade{mp_num}"


# Batch mode: tare_run_simulation sets input_measure_package before running
# this notebook, then clears it to None. Otherwise, ask which package to run.
_batch_mode = (
    'input_measure_package' in dir() and input_measure_package is not None)
if _batch_mode:
    input_measure_package = str(input_measure_package).strip()
    if input_measure_package not in SELECTABLE_MPS_STRINGS:
        raise ValueError(
            f"Invalid pre-set measure package: {input_measure_package}. Must "
            f"be one of {SELECTABLE_MPS_STRINGS} for ResStock "
            f"{RESSTOCK_RELEASE_THIS_RUN}.")
    menu_mp = int(input_measure_package)
    print(f"[BATCH MODE] Using pre-set Measure Package {menu_mp}.")
else:
    while True:
        input_measure_package = input(
            f"Please enter the measure package to run "
            f"({', '.join(SELECTABLE_MPS_STRINGS)}): ").strip()
        if input_measure_package in SELECTABLE_MPS_STRINGS:
            menu_mp = int(input_measure_package)
            break
        print(f"Invalid measure package. Must be one of {SELECTABLE_MPS_STRINGS}.")

input_mp = mp_to_upgrade_name(menu_mp)
cost_scenario = 'BAU Costs'
grid_scenario = 'Current Electricity Grid'

print(f"""
====================================================================================================================================================================
MODEL SCENARIO -- ResStock {RESSTOCK_RELEASE_THIS_RUN}
====================================================================================================================================================================
Measure Package {menu_mp} ({input_mp})
{cost_scenario}
{grid_scenario}
====================================================================================================================================================================
""")

# Same scope filters as the baseline (applicability, occupancy, housing type,
# Alaska/Hawaii). state/city reuse the choices made once in the baseline
# notebook, so this cell never asks the location questions again.
state_arg = None if menu_state == 'N' else input_state
city_arg = input_cityFilter if (menu_state == 'Y' and menu_city == 'Y') else None
df_euss_am_mpX, df_funnel_mpX = load_and_filter_upgrade(
    menu_mp=menu_mp, state=state_arg, city=city_arg, verbose=True,
    release=RESSTOCK_RELEASE_THIS_RUN)

# Display the filtered dataframe
print(f"DATAFRAME SIZE after all filters: {df_euss_am_mpX.shape}")
print(df_euss_am_mpX)
print(f"""
FILTER FUNNEL:
{df_funnel_mpX}
""")


# %% [markdown]
# # Project Future Energy Consumption

# %%
from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import df_enduse_compare

print(F"""
====================================================================================================================================================================
LOAD RESSTOCK DATA FOR MEASURE PACKAGE {menu_mp} (MP{menu_mp}) -- ResStock {RESSTOCK_RELEASE_THIS_RUN}
====================================================================================================================================================================
df_euss_am_mpX_home will be created by running the df_enduse_compare function (contains post-retrofit consumption data for the entire home).
process_euss_data.py file contains the function definition.
      
""")

# df_cooking_range is omitted (defaults to None): cooking is not in
# EQUIPMENT_SPECS today, so it is not needed on this path.
df_euss_am_mpX_home = df_enduse_compare(
    df_mp=df_euss_am_mpX,
    input_mp=input_mp,
    menu_mp=menu_mp,
    df_baseline=df_euss_am_baseline_home,
    release=RESSTOCK_RELEASE_THIS_RUN,
    )


# %% [markdown]
# # PUBLIC IMPACTS: Climate Damages
# ## Scenario: 2025 Reference Case

# %%
from cmu_tare_model.public_impact.calculate_lifetime_climate_impacts_sensitivity import calculate_lifetime_climate_impacts

print(f"""
====================================================================================================================================================================
PUBLIC IMPACTS: DAMAGES FROM CLIMATE-RELATED EMISSIONS
====================================================================================================================================================================

""")

# Make copies from scenario consumption to keep df smaller
print("\n", "Creating dataframe to store marginal damages calculations ...")

# Damage DataFrames: 2025 Reference Case
df_mpX_ref2025_damages_climate = df_euss_am_mpX_home.copy()

# %%
print("""
========== SCENARIO: 2025 Reference Case ==========
""")
df_euss_am_mpX_home, df_mpX_ref2025_damages_climate = calculate_lifetime_climate_impacts(
    df=df_euss_am_mpX_home,
    menu_mp=menu_mp,
    policy_scenario='2025 Reference Case',
    df_baseline_damages=df_baseline_damages_climate,
    verbose=VERBOSE
    )

print(f"""
====================================================================================================================================================================
Post-Retrofit (MP{menu_mp}) Marginal Damages: WHOLE-HOME
Scenario: 2025 Reference Case
====================================================================================================================================================================

CLIMATE DAMAGES (2025 Reference Case): df_mpX_ref2025_damages_climate
{df_mpX_ref2025_damages_climate}

SUMMARY DATAFRAME FOR MP{menu_mp}: df_euss_am_mp{menu_mp}_home
{df_euss_am_mpX_home}
====================================================================================================================================================================
""")

# %% [markdown]
# # PRIVATE IMPACTS: FUEL COSTS
# ## Scenario: 2025 Reference Case

# %%
from cmu_tare_model.private_impact.calculate_lifetime_fuel_costs import calculate_lifetime_fuel_costs

print(f"""
====================================================================================================================================================================
PRIVATE IMPACTS: OVERVIEW
====================================================================================================================================================================
Step 1: Calculate annual operating (fuel) costs
Step 2: Calculate equipment capital costs (For space heating, include ductwork and weatherization (MP9 and MP10))
Step 3: Calculate replacement cost (replacing existing piece of equipment with similar technology)
Step 4: Calculate net equipment capital costs and private NPV (less WTP and more WTP)

----------------------------------------------------------------------------------------------------------------------
Step 1: Calculate annual operating (fuel) costs
----------------------------------------------------------------------------------------------------------------------

====================================================================================================================================================================
FUEL COSTS RESULTS: 2025 Reference Case

""")

# %%
print("""
========== SCENARIO: 2025 Reference Case ==========
""")
print("Creating dataframe to store annual fuel cost calculations ...")
df_euss_am_mpX_home, df_mpX_ref2025_fuel_costs = calculate_lifetime_fuel_costs(
    df=df_euss_am_mpX_home,
    menu_mp=menu_mp,
    policy_scenario='2025 Reference Case',
    df_baseline_costs=df_baseline_fuel_costs
    )


print(f"""
====================================================================================================================================================================
Lifetime Fuel Costs: 2025 Reference Case

FUEL COSTS (2025 Reference Case): df_mpX_ref2025_fuel_costs
{df_mpX_ref2025_fuel_costs}

SUMMARY DATAFRAME FOR MP{menu_mp}: df_euss_am_mp{menu_mp}_home
{df_euss_am_mpX_home}

====================================================================================================================================================================
""")

# %% [markdown]
# # PRIVATE IMPACTS: CAPITAL COSTS
# ## Scenarios: 2025 Reference Case

# %%
from cmu_tare_model.utils.inflation_adjustment import *
from cmu_tare_model.utils.column_names import create_cost_col

# # ============================================================================
# COST MODULES: REMDB v4 regression-based installed costs.
# (These unified modules retain a legacy v3 probabilistic path, but this
# pipeline runs v4 only -- see REMDB_COST_SCENARIO_KEYS in constants.py.)
# ============================================================================
from cmu_tare_model.private_impact.calculations.calculate_equipment_installation_costs import (
    calculate_upgrade_installed_cost,
    obtain_heating_system_specs,
    calculate_heating_installation_premium
)
from cmu_tare_model.private_impact.calculations.calculate_equipment_replacement_costs import (
    calculate_replacement_installed_cost
)

# Enclosure costs use a separate probabilistic sampling path. This is only
# exercised for MP9/MP10 enclosure measures, not the MP3/MP4 HVAC models in
# this study, so it is kept as-is rather than migrated to the v4 regression.
from cmu_tare_model.private_impact.calculations.calculate_enclosure_upgrade_costs import (
    calculate_enclosure_retrofit_upgrade_costs
)

# REMDB v4 utilities (data loading & metric preparation)
from cmu_tare_model.constants import REMDB_COST_SCENARIO_KEYS
from cmu_tare_model.utils.remdb_v4_installed_cost_utils import (
    load_remdb_v4_data,
    add_remdb_metrics
)

if VERBOSE:
    print(f"""
    ====================================================================================================================================================================
    PRIVATE IMPACTS: NET CAPITAL COSTS AND TOTAL CAPITAL COSTS
    ====================================================================================================================================================================
    Completed Steps:
    1. Calculate annual operating (fuel) costs                                                                  [COMPLETED]

    REMAINING STEPS:
    Step 2: Calculate equipment capital costs (For space heating, include ductwork)
    Step 3: Calculate replacement cost (replacing existing piece of eqipment with similar technology)

    ----------------------------------------------------------------------------------------------------------------------
    Step 4 (MP9 AND MP10 SPACE HEATING ONLY): 
        Calculate Enclosure Upgrade Costs
        - calculate_enclosure_upgrade_costs.py file contains the definition for the calculate_enclosure_upgrade_costs function.
    ----------------------------------------------------------------------------------------------------------------------

    Cost Databases:
    - REMDB v4: Regression-based deterministic cost calculations
    - Cost scenarios: {REMDB_COST_SCENARIO_KEYS}

    ====================================================================================================================================================================
    LIFETIME CAPITAL COSTS RESULTS: 2025 Reference Case (With and Without Rebates)

    """)


# %%
print("\n" + "="*80)
print("LOADING CAPITAL COST DATABASES")
print("="*80)

# ============================================================================
# REMDB v4: Regression-based cost database
# ============================================================================
print("\nREMDB v4 (Regression):")
remdb_v4_costs = load_remdb_v4_data()
print(f"  Loaded {len(remdb_v4_costs)} equipment types from REMDB v4")

print("\n" + "="*80 + "\n")

# %% [markdown]
# ## REMDB v4: Capital Costs (Regression-Based)
# ### Heating only - additional end-uses to be added in future versions

# %%
# ============================================================================
# REMDB v4: CAPITAL COST SCENARIO LOOP
# ============================================================================
# Calculates installed costs using the REMDB v4 regression methodology.
# Results stored in a nested dict for cross-scenario comparison.
#
# v4 workflow (two-step per end-use):
#   1. add_remdb_metrics() - assigns row_id, maps coefficients, converts units
#   2. calculate_*_installed_cost() - applies regression formula (unified)
#
# Currently implemented for heating only. Other end-uses (waterHeating,
# clothesDrying, cooking) will be added when REMDB v4 supports them.
# ============================================================================
VERBOSE = True

# Initialize nested dictionary: CAPITAL_COSTS_MPX[end_use][cost_type][scenario_key]
CAPITAL_COSTS_MPX = {
    end_use: {'replacement': {}, 'upgrade': {}}
    for end_use in VALID_CATEGORIES
}

print("="*80)
print(f"CALCULATING CAPITAL COSTS - MEASURE PACKAGE {menu_mp}")
print("="*80)

# Loop over REMDB v4 cost scenarios (low, mid, high percentiles)
for scenario_key in REMDB_COST_SCENARIO_KEYS:
    # All active scenarios use the REMDB v4 regression. The percentile token
    # is the scenario suffix: v4LOW -> 'low', v4MID -> 'mid', v4HIGH -> 'high'.
    method = 'remdb_v4'
    percentile = scenario_key[2:].lower()

    print(f"\nScenario: {scenario_key} | Method: {method} | Percentile: {percentile}")
    
    # Work from a clean copy of the base DataFrame
    df_scenario = df_euss_am_mpX_home.copy()
    
    for end_use in ['heating']:
        # STEP 1: Prepare replacement metrics (assigns row_id, converts units)
        df_scenario, df_detailed_repl = add_remdb_metrics(
            df=df_scenario,
            remdb_v4_costs=remdb_v4_costs,
            end_use=end_use,
            metric_type='replacement',
            percentile=percentile,
            verbose=VERBOSE
        )
        
        # STEP 2: Calculate replacement installed costs
        df_scenario, df_detailed_repl = calculate_replacement_installed_cost(
            df=df_scenario,
            df_detailed=df_detailed_repl,
            menu_mp=menu_mp,
            end_use=end_use,
            cost_scenario=scenario_key
        )
        
        # STEP 3: Prepare upgrade metrics
        df_scenario, df_detailed_upgr = add_remdb_metrics(
            df=df_scenario,
            remdb_v4_costs=remdb_v4_costs,
            end_use=end_use,
            metric_type='upgrade',
            percentile=percentile,
            verbose=VERBOSE
        )
        
        # STEP 4: Calculate upgrade installed costs
        df_scenario, df_detailed_upgr = calculate_upgrade_installed_cost(
            df=df_scenario,
            df_detailed=df_detailed_upgr,
            menu_mp=menu_mp,
            end_use=end_use,
            cost_scenario=scenario_key
        )
        
        # Also calculate cooling replacement costs (metadata for net cost calculation)
        df_scenario, df_detailed_cool = add_remdb_metrics(
            df=df_scenario,
            remdb_v4_costs=remdb_v4_costs,
            end_use='cooling',
            metric_type='replacement',
            percentile=percentile,
            verbose=VERBOSE
        )
        
        df_scenario, df_detailed_cool = calculate_replacement_installed_cost(
            df=df_scenario,
            df_detailed=df_detailed_cool,
            menu_mp=menu_mp,
            end_use='cooling',
            cost_scenario=scenario_key
        )
    
    # Store results for this scenario
    CAPITAL_COSTS_MPX['heating']['replacement'][scenario_key] = df_scenario.copy()
    CAPITAL_COSTS_MPX['heating']['upgrade'][scenario_key] = df_scenario.copy()
    
    replacement_cost_col_name = create_cost_col(menu_mp=menu_mp, category=end_use, cost_type='replacement', cost_scenario=scenario_key)
    upgrade_cost_col_name = create_cost_col(menu_mp=menu_mp, category=end_use, cost_type='upgrade', cost_scenario=scenario_key)

    if replacement_cost_col_name in df_scenario.columns:
        valid_repl = df_scenario[replacement_cost_col_name].notna().sum()
        mean_repl = df_scenario[replacement_cost_col_name].mean()
        print(f"  Replacement: {valid_repl:,} valid homes, mean=${mean_repl:,.2f}")
    if upgrade_cost_col_name in df_scenario.columns:
        valid_upgr = df_scenario[upgrade_cost_col_name].notna().sum()
        mean_upgr = df_scenario[upgrade_cost_col_name].mean()
        print(f"  Upgrade: {valid_upgr:,} valid homes, mean=${mean_upgr:,.2f}")

print(f"\nCalculated {len(REMDB_COST_SCENARIO_KEYS)} scenarios: {REMDB_COST_SCENARIO_KEYS}")
print("="*80)

# %%
# ============================================================================
# MERGE v4 COST COLUMNS INTO df_euss_am_mpX_home
# ============================================================================
# The v4 cost loop above computed installed costs on per-scenario DataFrame
# copies stored in CAPITAL_COSTS_MPX. The rebate calculation below runs on
# df_euss_am_mpX_home and expects all cost scenario columns to be present.
# This block merges only the final cost columns (not REMDB intermediates)
# from each v4 scenario back onto df_euss_am_mpX_home so that
# calculate_rebate_program() can find them.
# ============================================================================

v4_columns_merged = []

for scenario_key in REMDB_COST_SCENARIO_KEYS:
    # The 'upgrade' DataFrame contains heating upgrade, heating replacement,
    # AND cooling replacement cost columns (all computed in the v4 loop)
    df_v4_source = CAPITAL_COSTS_MPX['heating']['upgrade'][scenario_key]

    # Build the list of final cost columns to merge for this scenario
    cost_columns_to_merge = []
    for end_use, cost_type in [('heating', 'replacement'), ('heating', 'upgrade'), ('cooling', 'replacement')]:
        col_name = create_cost_col(menu_mp=menu_mp, category=end_use, cost_type=cost_type, cost_scenario=scenario_key)
        if col_name in df_v4_source.columns:
            cost_columns_to_merge.append(col_name)

    # Merge via column assignment (aligned by index - same row order as source)
    for col in cost_columns_to_merge:
        df_euss_am_mpX_home[col] = df_v4_source[col].values

    v4_columns_merged.extend(cost_columns_to_merge)

# ============================================================================
# NOTE ON v4 MONOTONICITY — DO NOT ENFORCE
# ============================================================================
# Per the REMDB Machine Readable Guidance Document, low/mid/high represent
# 10th, 50th, and 90th percentile quantile regressions fitted independently.
# Coefficient-level non-monotonicity is BY DESIGN — the guidance document
# itself shows examples where coefficients decrease from low→high (e.g.,
# Water Heater PM2 coefficients: 28.81 → 19.33 → 8.39; intercepts:
# 155.30 → 436.45 → -651.90). For certain input value combinations,
# quantile regression crossings are expected and should NOT be "corrected."
# ============================================================================

# Diagnostic output
print("\n" + "=" * 80)
print(f"MERGED v4 COST COLUMNS INTO df_euss_am_mpX_home ({len(v4_columns_merged)} columns)")
print("=" * 80)
for col in sorted(v4_columns_merged):
    print(f"  {col}")
print(f"\ndf_euss_am_mpX_home shape: {df_euss_am_mpX_home.shape}")
print("=" * 80)

# %% [markdown]
#  ## Calculate Rebate Amounts

# %%
from cmu_tare_model.private_impact.data_processing.determine_rebate_eligibility_and_amount import (
    calculate_percent_AMI,
    calculate_rebate_program,
)
from cmu_tare_model.utils.discounting import prepare_discount_rates
from cmu_tare_model.constants import (
    PRIVATE_DISCOUNT_RATE_SHORT_KEYS,
    REBATE_POLICY_SCENARIOS,
)

print(f"""
====================================================================================================================================================================
CALCULATE HOUSEHOLD PERCENT AREA MEDIAN INCOME (%AMI) AND REBATE ELIGIBILITY/AMOUNTS
====================================================================================================================================================================
determine_rebate_eligibility_and_amount.py file contains the function definitions for calculating rebate amounts and determining household %AMI.
process_income_data_for_rebates.py file contains additional information on data sources and procedures used to process data for determine_rebate_eligibility_and_amount.py file.

----------------------------------------------------------------------------------------------------------------------

""")

# Determine Percent AMI and Rebate Amounts
# This needs to be done before running the calculate_percent_AMI function
df_euss_am_mpX_home = df_euss_am_mpX_home.copy()

print("Calculating Percent AMI for each household ...")
df_euss_am_mpX_home = calculate_percent_AMI(df_results_IRA=df_euss_am_mpX_home)

# New function that prepares discount rates (e.g., variable) for NPV calculations and prints the discount rates used if verbose=True
print("Preparing discount rates for NPV calculations ...")
df_euss_am_mpX_home = prepare_discount_rates(df=df_euss_am_mpX_home,
                                             verbose=VERBOSE)

for end_use in VALID_CATEGORIES:
    print(VALID_CATEGORIES)
    for cost_scenario in REMDB_COST_SCENARIO_KEYS:
        # One central rebate function per guidance vintage. REBATE_POLICY_SCENARIOS
        # is [REBATE_GUIDANCE_IRA2024, REBATE_GUIDANCE_JUNE2026]: the December 2024
        # Home Energy Rebate guidance and the updated June 2026 Program-notice
        # guidance. Each vintage writes its own rebate amount + eligibility
        # columns (2024 uses the guidance-less amount name for byte-identity).
        for guidance in REBATE_POLICY_SCENARIOS:
            print(f"\nCalculating {guidance} rebate amounts for "
                  f"{end_use} ({cost_scenario}) ...")
            df_euss_am_mpX_home = calculate_rebate_program(
                df_results_IRA=df_euss_am_mpX_home,
                category=end_use,
                menu_mp=menu_mp,
                cost_scenario=cost_scenario,
                guidance=guidance)

print(f"""
====================================================================================================================================================================
DATAFRAME: df_euss_am_mpX_home AFTER CALCULATING REBATE AMOUNTS
{df_euss_am_mpX_home}

====================================================================================================================================================================
""")

# %% [markdown]
# # SCENARIO ANALYSIS: 2025 Reference Case
# ## Public Impact, Private Impact and Adoption Potential

# %%
from cmu_tare_model.private_impact.calculate_lifetime_private_impact import calculate_private_npv
from cmu_tare_model.adoption_potential.determine_economic_adoption_potential import (
    economic_adoption_decision
)

# Create one DataFrame copy per discount rate.
DATAFRAMES_MPX_RCM_DISCOUNT_RATE = {
    discount_rate: df_euss_am_mpX_home.copy()
    for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS
}

# ============================================================================
# Merge REMAINING v4 columns into each scenario DataFrame.
# NOTE: Final cost columns (installed_cost_{v4LOW/MID/HIGH}) were already
# merged into df_euss_am_mpX_home BEFORE the rebate calculation cell.
# Since DATAFRAMES_MPX_RCM_DISCOUNT_RATE was built from df_euss_am_mpX_home
# (which now includes v4 cost columns), those columns are already present.
# This block propagates any remaining columns (REMDB intermediates like
# row_id_*, *_pm1_euss, *_pm2_euss) into DATAFRAMES_MPX_RCM_DISCOUNT_RATE
# so they are available for downstream diagnostic/analysis if needed.
# ============================================================================
v4_cost_columns_added = []
for scenario_key in REMDB_COST_SCENARIO_KEYS:
    # Get v4 DataFrame that contains the scenario-specific columns
    # Use 'upgrade' since it contains both upgrade and replacement columns
    df_v4_source = CAPITAL_COSTS_MPX['heating']['upgrade'][scenario_key]

    # Identify new columns not yet on df_euss_am_mpX_home (cost columns
    # are already merged; this picks up REMDB intermediate columns only)
    base_cols = set(df_euss_am_mpX_home.columns)
    new_cols = [col for col in df_v4_source.columns if col not in base_cols]
    v4_cost_columns_added.extend(new_cols)

    # Add these remaining columns to every DataFrame in the dictionary.
    for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS:
        for col in new_cols:
            DATAFRAMES_MPX_RCM_DISCOUNT_RATE[discount_rate][col] = df_v4_source[col].values

print(f"  v4 cost columns merged into dictionary: {sorted(set(v4_cost_columns_added))}")

print(f"""  
========================================================================================================
SCENARIO ANALYSIS: CLIMATE IMPACT
    - calculate_lifetime_climate_impacts_sensitivity.py contains the definition for the calculate_lifetime_climate_impacts function.
    - Additional information on emissions/damage factor lookups as well as marginal damages calculation methods can be found in the public_impact folder.
========================================================================================================

Completed Steps:
1. Calculate the baseline marginal damages for climate-related emissions                                    [COMPLETED]
2. Calculate the post-retrofit marginal damages for climate-related emissions                               [COMPLETED]

========================================================================================================
SCENARIO ANALYSIS: PRIVATE IMPACT
    - calculate_lifetime_private_impact.py file contains the definition for the calculate_private_npv function.
    - Additional information on fuel price lookups as well as capital costs calculation methods can be found in the private_impact folder.
========================================================================================================

Completed Steps:
1. Calculate annual operating (fuel) costs                                                                  [COMPLETED]
2. Calculate equipment capital costs (For space heating, include ductwork and weatherization (MP9-10))      [COMPLETED]
3. Calculate replacement cost (replacing existing piece of eqipment with similar technology)                [COMPLETED]

REMAINING STEP:
Step 4: Calculate net equipment capital costs and private NPV
------------------------------------------------------------------------------------------------------

========================================================================================================
SCENARIO ANALYSIS: ADOPTION POTENTIAL
    determine_economic_adoption_potential.py defines economic_adoption_decision.
    A home is an economic adopter if its private incremental NPV >= 0.
    Climate damages are computed and stored but do not enter the adoption decision.
    Nine adopter columns are produced per call, one per NPV case.
========================================================================================================

Economic adopter condition (NPV >= 0) applied across nine NPV cases
(three replacement-credit scopes x three rebate-policy scenarios).
All nine cases credit BOTH heating and cooling operating savings; the scope
token controls which end-use's avoided-replacement capital (LCC) is credited:
    heatingSavings_coolingLCC --> cooling replacement credited; no heating LCC
    heatingLCC_coolingSavings --> heating replacement credited; no cooling LCC
    heatingLCC_coolingLCC     --> both heating and cooling replacement credited
    each x rebate-policy scenario --> _unsub, _sub, _sub_june2026 

------------------------------------------------------------------------------------------------------

Cost scenarios to process: {REMDB_COST_SCENARIO_KEYS}
      
""")


# %% [markdown]
# # MEASURE PACKAGE (MPX): 2025 REFERENCE CASE

# %%
policy_scenario = '2025 Reference Case'

print(f"""
====================================================================================================================================================================
MODEL SCENARIO
====================================================================================================================================================================
EUSS Measure Package {menu_mp}
Policy Scenario: {policy_scenario}
====================================================================================================================================================================
""")

# %%
print(f"""
====================================================================================================================================================================
SCENARIO ANALYSIS ({policy_scenario.upper()}): PRIVATE IMPACT
====================================================================================================================================================================
""")

# Process each cost scenario then discount rate.
print("Calculating Private NPV for all cost scenarios and discount methods ...")

for cost_scenario_key in REMDB_COST_SCENARIO_KEYS:
    print(f"\n--- Cost Scenario: {cost_scenario_key} ---")

    for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS:
        # Create full discount rate column name.
        discount_rate_col_name = f'private_discount_rate_{discount_rate}'
        print(f"  Discount Rate: {discount_rate}, Column: {discount_rate_col_name}")

        # Get the DataFrame for this discount rate.
        df = DATAFRAMES_MPX_RCM_DISCOUNT_RATE[discount_rate]

        # One call per (cost_scenario, discount_rate) combination.
        # calculate_private_npv produces all nine NPV case columns in a single call.
        df = calculate_private_npv(
            df=df,
            df_fuel_costs=df_mpX_ref2025_fuel_costs,
            df_baseline_costs=df_baseline_fuel_costs,
            menu_mp=menu_mp,
            input_mp=input_mp,
            policy_scenario=policy_scenario,
            discount_rate_col_name=discount_rate_col_name,
            cost_scenario=cost_scenario_key,
            verbose=VERBOSE,
        )

        # Update the DataFrame back in the dictionary.
        DATAFRAMES_MPX_RCM_DISCOUNT_RATE[discount_rate] = df

if PRINT_VERBOSE_DATAFRAMES:
    print(f"\n{'='*100}")
    print(f"DATAFRAME FOR MP{menu_mp} AFTER CALCULATING PRIVATE NPV ({policy_scenario.upper()})")
    print(f"{'='*100}")
    print(DATAFRAMES_MPX_RCM_DISCOUNT_RATE['fixed_base'])
    print()

# %%
print(f"""
====================================================================================================
SCENARIO ANALYSIS ({policy_scenario.upper()}): ADOPTION POTENTIAL
====================================================================================================
""")

# Process each cost scenario then discount rate.
print("Determining Economic Adoption Potential for all cost scenarios and discount methods ...")

for cost_scenario_key in REMDB_COST_SCENARIO_KEYS:
    print(f"\n--- Cost Scenario: {cost_scenario_key} ---")

    for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS:
        # Create full discount rate column name.
        discount_rate_col_name = f'private_discount_rate_{discount_rate}'
        print(f"  Discount Rate: {discount_rate}, Column: {discount_rate_col_name}")

        # Get the DataFrame for this discount rate.
        df = DATAFRAMES_MPX_RCM_DISCOUNT_RATE[discount_rate]

        duplicate_mask = df.columns.duplicated(keep='first')
        duplicate_count = duplicate_mask.sum()

        # Diagnostic check BEFORE processing.
        if duplicate_count > 0:
            duplicate_cols = df.columns[duplicate_mask].unique().tolist()
            print(f"\n{discount_rate}: {duplicate_count} duplicates")
            print(f"  Columns: {duplicate_cols[:5]}")

        # One call per (cost_scenario, discount_rate) combination.
        # economic_adoption_decision applies NPV >= 0 across all nine NPV cases
        # in a single call. Climate damages remain in the DataFrame for sensitivity
        # analysis but do not enter the adoption decision.
        df = economic_adoption_decision(
            df=df,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            discount_rate_col_name=discount_rate_col_name,
            cost_scenario=cost_scenario_key,
            verbose=VERBOSE,
        )

        # Update the DataFrame back in the dictionary.
        DATAFRAMES_MPX_RCM_DISCOUNT_RATE[discount_rate] = df

if PRINT_VERBOSE_DATAFRAMES:
    print(f"\n{'='*100}")
    print(f"DATAFRAME FOR MP{menu_mp} AFTER DETERMINING ECONOMIC ADOPTION FEASIBILITY")
    print("Nine adopter columns produced "
          "(three replacement-credit scopes x three rebate-policy scenarios):")
    print("  heatingSavings_coolingLCC, heatingLCC_coolingSavings, heatingLCC_coolingLCC")
    print("  each x {_unsub, _sub, _sub_june2026}")
    print(f"{'='*100}")
    print(DATAFRAMES_MPX_RCM_DISCOUNT_RATE['fixed_base'])
    print()


# %% [markdown]
# # Model Runtime

# %%
# Flag to prevent excessive output in other scenario files
individual_scenario_run = False

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

# %%
# =====================================================================
# Economic adopter analysis -- results check
# Adopter cols: float 0.0/1.0 (1 = NPV >= 0 adopter, NaN = excluded)
# Nine NPV cases per MP: three replacement-credit scopes
# (heatingSavings_coolingLCC | heatingLCC_coolingSavings | heatingLCC_coolingLCC)
# x three rebate-policy scenarios (_unsub | _sub | _sub_june2026)
# =====================================================================
import pandas as pd

def summarize_econ_adopters(df, weight_col="weight"):
    cols = sorted(c for c in df.columns if "econ_adopter" in c)
    if not cols:
        print("  (no econ_adopter columns in this DataFrame)")
        return None
    has_w = weight_col in df.columns
    rows = []
    for c in cols:
        s = df[c]
        appl = s.notna()                      # excluded homes are NaN
        n_appl = int(appl.sum())
        n_adopt = int((s == 1).sum())
        rate = n_adopt / n_appl if n_appl else float("nan")
        if has_w:
            w = df.loc[appl, weight_col]
            wrate = (df.loc[appl & (s == 1), weight_col].sum() / w.sum()
                     if w.sum() else float("nan"))
        else:
            wrate = float("nan")
        rows.append({
            "column": c,
            "n_applicable": n_appl,
            "n_adopters": n_adopt,
            "rate_%": round(100 * rate, 1),
            "weighted_rate_%": round(100 * wrate, 1) if has_w else None,
        })
    out = pd.DataFrame(rows)
    with pd.option_context("display.max_colwidth", None, "display.width", 200):
        print(out.to_string(index=False))
    return out

# Auto-find every DataFrame in the session that carries econ_adopter columns.
_dfs = {n: o for n, o in list(globals().items())
        if isinstance(o, pd.DataFrame)
        and any("econ_adopter" in c for c in o.columns)}
print("DataFrames with econ_adopter columns:", list(_dfs) or "(none found)")
for _name, _df in _dfs.items():
    print(f"\n=== {_name} ===")
    summarize_econ_adopters(_df)


# %%
# County-level adoption rates for the adopter columns in `df`.
import pandas as pd

def county_adoption_rates(df, adopter_col, county_col="county_fips",
                          weight_col="weight"):
    if county_col not in df.columns:
        print(f"  (no '{county_col}' column in this DataFrame)"); return None
    sub = df.loc[df[adopter_col].notna(), [county_col, adopter_col]].copy()
    if weight_col in df.columns:
        sub["_w"] = df.loc[sub.index, weight_col]
        sub["_wa"] = sub["_w"] * sub[adopter_col]
        g = sub.groupby(county_col)[["_wa", "_w"]].sum()
        rates = 100 * g["_wa"] / g["_w"]
    else:
        rates = 100 * sub.groupby(county_col)[adopter_col].mean()
    rates = rates.rename("adoption_rate_%")
    print(f"{adopter_col}\n  counties={rates.size}  mean={rates.mean():.1f}%  "
          f"median={rates.median():.1f}%  min={rates.min():.1f}%  "
          f"max={rates.max():.1f}%")
    return rates

adopter_cols = sorted(c for c in df.columns if "econ_adopter" in c)
print("Adopter columns in df:")
for c in adopter_cols:
    print("  ", c)
print()

county_rate_series = {}
for col in adopter_cols:
    county_rate_series[col] = county_adoption_rates(df, col)
    print()


# %%
"""Spec-driven verification: June 2026 rebate fuel gate (per measure package).

Runs inside each retrofit MP's pass and checks the CURRENT MP (menu_mp), so it
never references an MP whose columns have not been built yet.

What it asserts (current code state):
  - HEEHR June 2026 is electric-gated: a rebate may not fund removing a fossil
    heating system, so fossil baselines (Natural Gas, Propane, Fuel Oil) get $0.
  - HOMES June 2026 is ALSO electric-gated *for now* -- this is a temporary
    byte-identity hold, NOT the final rule. HOMES is adjudicated fuel-neutral;
    the 2026-HOMES fuel-neutral fix is DEFERRED (it would move _sub_june2026).
    While that hold stands, every fossil baseline is still $0 under June 2026,
    so the whole-fuel check below is valid.

  ==> WHEN THE DEFERRED 2026-HOMES FUEL-NEUTRAL FIX LANDS, fossil HOMES becomes
      > $0 and Spec check 1 will (correctly) start failing. At that point make
      the check HEEHR-only, which needs a program x fuel split. The authoritative
      program x fuel version already exists at
      scripts/verify_june2026_rebate_fossil_gate.py -- prefer updating there.

Reported figures: 'total_eligible' is the uncapped potential (no funding cap is
modeled); 'adopters_only' (economic adopters) is the figure to compare against
the ~$8-9B appropriation.
"""

from cmu_tare_model.private_impact.data_processing.determine_rebate_eligibility_and_amount import (
    summarize_rebate_funding,
)
from cmu_tare_model.utils.column_names import create_adoption_col
from cmu_tare_model.utils.modeling_params import define_scenario_params

_MP = menu_mp  # the MP just processed this pass (3, then 4) -- never hardcode 4
_COST = 'v4MID'
_METHOD_SUFFIX = '_fixed_base'
_WEIGHT_COL = 'weight'  # adjust if the frame's household-weight column differs

_df = DATAFRAMES_MPX_RCM_DISCOUNT_RATE['fixed_base']  # current-MP frame

# The fuel gate is applied to the rebate itself, so 'total_eligible' by fuel is
# identical across NPV scopes; one representative adopter scope drives the
# informational 'adopters_only' column.
_prefix = define_scenario_params(_MP)[0]
_adopter_col = create_adoption_col(
    _prefix, 'heatingLCC_coolingLCC_sub_june2026', _METHOD_SUFFIX)

# Guard: only run once this MP's June 2026 rebate + adopter columns exist.
# Skips the baseline pass (MP0) and any pass where the columns are not built yet,
# so this verification can never crash a full run on a missing-column KeyError.
_amount_col = f'mp{_MP}_heating_rebate_amount_june2026_{_COST}'
_elig_col = f'mp{_MP}_rebate_eligibility_june2026'
_required = [_amount_col, _elig_col, _adopter_col]
_missing = [c for c in _required if c not in _df.columns]

if _MP == 0 or _missing:
    print(f'[SKIP] MP{_MP} June 2026 fuel-gate check: '
          f'{"baseline pass" if _MP == 0 else f"columns not present {_missing}"}.')
else:
    by_program, by_fuel = summarize_rebate_funding(
        _df,
        menu_mp=_MP,
        cost_scenario=_COST,
        guidance='june2026',
        weight_col=_WEIGHT_COL,
        adopter_col=_adopter_col,
    )

    print(f'MP{_MP} adopter column:', _adopter_col)
    print('\n--- June 2026 rebate funding by program (weighted $) ---')
    print(by_program.round(0))
    print('\n--- June 2026 rebate funding by baseline fuel (weighted $) ---')
    print(by_fuel.round(0))

    # Spec check 1 -- no fossil baseline may receive June 2026 rebate dollars.
    # Valid while the 2026-HOMES electric-gate hold stands (see docstring). When
    # the deferred fuel-neutral fix lands, convert this to a HEEHR-only check.
    _fossil = by_fuel.drop(index='Electricity', errors='ignore')
    _fossil_nonzero = _fossil[(_fossil != 0).any(axis=1)]
    assert _fossil_nonzero.empty, (
        f"MP{_MP} June 2026 fuel-gate regression: non-electric baselines received "
        "rebate dollars (fossil-system removal must not be funded under the current "
        "electric-gate hold):\n"
        f"{_fossil_nonzero.round(2)}"
    )

    # Spec check 2 -- electric-resistance baselines must still be funded; guards
    # against a bug that zeroes every rebate and would pass check 1 trivially.
    assert 'Electricity' in by_fuel.index and by_fuel.loc['Electricity', 'total_eligible'] > 0, (
        f"MP{_MP} June 2026 gate too strict: Electricity baselines received $0 "
        "total_eligible; expected electric-resistance homes to qualify for HEEHR/HOMES."
    )

    print(f'\n[PASS] MP{_MP} June 2026 fuel gate holds: fossil baselines $0, '
          'electric-resistance baselines funded.')


