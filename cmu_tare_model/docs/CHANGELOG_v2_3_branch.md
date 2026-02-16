# CHANGELOG: cmu-tare-model-v2-3 Branch

**Branch:** `cmu-tare-model-v2-3`  
**Base:** `main`  
**Date:** February 12, 2026  
**Author:** Jordan  

---

## Overview

The v2.3 branch introduces **multi-capital-cost-estimation-method support** via REMDB v4 quantile regression, alongside the existing v3 probabilistic (Excel-based CPI-adjusted) approach. This adds `cost_scenario` as a new sensitivity dimension throughout the model pipeline — from capital cost calculation through NPV, adoption potential, export/load, and visualization.

---

## 1. New Sensitivity Dimension: Capital Cost Scenarios (`cost_scenario`)

### 1.1 Constants (`constants.py`)
- **Added `REMDB_COST_SCENARIO_KEYS`** — controls which cost methodologies are run:
  - Active: `['v3', 'v4MID']`
  - Available (commented out): `v4LOW`, `v4HIGH`
- When all 4 are enabled, the full sensitivity matrix becomes:
  - 3 MPs × **4 cost** × 4 discount × 3 RCM × 2 CR × 3 SCC × 2 policy = **1,728 combinations** (vs. 864 currently)

### 1.2 Column Naming Convention (`utils/column_names.py`)
- **New module** providing centralized column name builders — single source of truth for all column names
- `cost_scenario` is embedded in all output column names:
  - `create_cost_col()` → `mp8_heating_upgrade_installed_cost_v3` / `..._v4MID`
  - `create_capital_col()` → `preIRA_mp8_heating_total_capital_cost_v3`
  - `create_npv_col()` → `preIRA_mp8_heating_private_npv_lessWTP_v3_fixed_base`
  - `create_rebate_col()` → `mp8_heating_rebate_amount_v3`
  - `create_adoption_col()` → `preIRA_mp8_heating_adoption_central_inmap_acs_v3_fixed_base`
  - `create_total_npv_col()` → `preIRA_mp8_heating_total_npv_climateOnly_central_v3_fixed_base`

---

## 2. REMDB v4 Regression Engine

### 2.1 New Utility: `utils/remdb_v4_installed_cost_utils.py`
- `load_remdb_v4_data()` — loads REMDB v4 regression coefficients from `data/retrofit_costs/remdb_v4_tare_retrofit_costs.csv`
- `add_remdb_metrics()` — assigns `row_id`, maps coefficients, converts units (EUSS → REMDB)
- Quantile regression formula: `Installed_Cost = Material_Price × multiplier + adder`
- Three percentiles: LOW (25th), MID (50th), HIGH (75th)

### 2.2 Monotonicity Decision
- Monotonicity enforcement between v4LOW/v4MID/v4HIGH was **removed** per REMDB guidance
- Independent quantile regressions can produce crossing estimates — this is expected behavior
- Documented in `docs/troubleshooting_summary_2026-02-11.md`

---

## 3. Unified Cost Calculation Modules

### 3.1 `private_impact/calculations/calculate_equipment_installation_costs.py`
- Unified v3 + v4 interface via `cost_scenario` parameter
- When `cost_scenario='v3'`: uses Excel-based CPI-adjusted probabilistic sampling
- When `cost_scenario='v4*'`: uses REMDB v4 regression-derived costs
- Legacy v3-only version preserved as `calculate_equipment_installation_costs_v3.py`

### 3.2 `private_impact/calculations/calculate_equipment_replacement_costs.py`
- Same unified interface pattern
- Legacy preserved as `calculate_equipment_replacement_costs_v3.py`

### 3.3 `private_impact/calculations/calculate_enclosure_upgrade_costs.py`
- Enclosure costs for MP9 (basic) and MP10 (enhanced)
- Currently v3 only (REMDB v4 does not cover enclosure upgrades)

---

## 4. Scenario Notebook Consolidation (`model_scenarios/tare_scenarios_v2_2.ipynb`)

### 4.1 Four-to-One Consolidation
- Previous: 4 separate notebooks (`tare_basic_v2_1.ipynb`, `tare_moderate_v2_1.ipynb`, `tare_advanced_v2_1.ipynb`, `tare_scenarios_v2_1.ipynb`)
- Current: Single unified `tare_scenarios_v2_2.ipynb` (69 cells)
- Parameterized by `menu_mp` variable set externally by run simulation or interactively

### 4.2 New Data Structures
- **`CAPITAL_COSTS_MPX`** — nested dict `[end_use][cost_type][scenario_key] → DataFrame`
  - Stores per-scenario capital cost DataFrames (v3 + each v4 variant)
  - Each DataFrame is a full copy of `df_euss_am_mpX_home` with scenario-specific cost columns
  - Used for sensitivity analysis; v4 columns merged back into main DataFrame
- **`DATAFRAMES_MPX_RCM_DISCOUNT_RATE`** — nested dict `[discount_rate][rcm_model] → DataFrame`
  - Each DataFrame contains columns for ALL active cost scenarios
  - Structure: 4 discount rates × 3 RCM models = 12 DataFrames per MP

### 4.3 Triple-Nested Loop Architecture
- NPV and adoption now loop over `cost_scenario × discount_rate × rcm_model`:
  ```python
  for cost_scenario_key in REMDB_COST_SCENARIO_KEYS:     # v3, v4MID
      for discount_rate in PRIVATE_DISCOUNT_RATE_SHORT_KEYS:  # 4 methods
          for rcm_model in RCM_MODELS:                        # 3 models
              calculate_private_npv(df, ..., cost_scenario=cost_scenario_key)
              adoption_decision(df, ..., cost_scenario=cost_scenario_key)
  ```

### 4.4 Rebate Calculation per Cost Scenario
- `calculate_rebateIRA()` now runs for each `cost_scenario` in `REMDB_COST_SCENARIO_KEYS`
- Produces separate rebate columns: `mp8_heating_rebate_amount_v3`, `mp8_heating_rebate_amount_v4MID`

### 4.5 Capital Cost Sensitivity Analysis (Cells 56–69)
- 9 diagnostic tests added post-model-run:
  
  | Test | Description |
  |------|-------------|
  | TEST 1 | Data integrity — column existence, NaN%, negative values, mean/median |
  | TEST 2 | v4 quantile regression crossing diagnostic (informational only) |
  | TEST 3 | Cross-scenario summary statistics — P5/P25/P50/P75/P95 + v3 vs v4MID pairwise |
  | TEST 4a/b/c | 2×2 visualizations — heating upgrade, heating replacement, cooling replacement |
  | TEST 5 | NPV consistency — capital + NPV columns exist, `moreWTP ≥ lessWTP` validation |
  | TEST 6 | v4 column propagation count in main vs post-NPV DataFrames |
  | TEST 7 | Regional disaggregation — cost by census_division × scenario |
  | TEST 8 | Fuel type disaggregation — cost by base_heating_fuel × scenario |
  | TEST 9 | Cooling replacement deep dive — row_id mapping, coefficient monotonicity |

---

## 5. Private Impact Updates

### 5.1 `private_impact/calculate_lifetime_private_impact.py`
- `calculate_private_npv()` — added `cost_scenario` parameter
- `calculate_capital_costs()` — builds scenario-specific column lookups via `create_cost_col()`, `create_capital_col()`, `create_rebate_col()`
- All NPV columns include `cost_scenario` in name: `{prefix}heating_private_npv_{wtp}_{cost_scenario}{method_suffix}`

### 5.2 `adoption_potential/determine_adoption_potential_sensitivity.py`
- `adoption_decision()` — added `cost_scenario` parameter
- Adoption tier columns include `cost_scenario`: `{prefix}heating_adoption_{scc}_{rcm}_{crf}_{cost_scenario}{method_suffix}`
- `calculate_climate_only_adoption_robust()` and `calculate_health_only_adoption_robust()` updated similarly
- Total NPV columns: `{prefix}heating_total_npv_climateOnly_{scc}_{cost_scenario}{method_suffix}`

---

## 6. Export/Load Infrastructure

### 6.1 `utils/export_model_run_results.py`
- `export_model_run_output()` — exports full DataFrames containing all cost_scenario columns
- Directory structure unchanged: `retrofit_mp{X}_results/summary_mp{X}_{rcm}_{discount_rate}/`
- Cost scenario is NOT a directory dimension — all cost_scenario columns are in one CSV per discount_rate × rcm_model

### 6.2 `utils/load_exported_results_to_df.py`
- `load_model_run_output()` — loads single CSV (all cost_scenario columns included)
- `load_measure_package_data()` — returns `{discount_rate: {rcm_model: DataFrame}}`
- Chunked loading with configurable `chunk_size` for large national datasets

---

## 7. Memory Optimization
- ~4.9 GB memory savings through:
  - Shared public impact calculations across discount rates
  - Selective DataFrame copying (only when mutation needed)
  - Garbage collection after intermediate calculations

---

## 8. Miscellaneous Updates

### 8.1 Parameter Updates
- T&D losses: 6% → 5% (`TD_LOSSES = 0.05`, `TD_LOSSES_MULTIPLIER = 1.0526`)
- Equipment specs: Currently only `heating` active (15-year life); `waterHeating`, `clothesDrying`, `cooking` commented out

### 8.2 Discount Rate System
- 4 private methods: `fixed_low` (2%), `fixed_base` (7%), `fixed_high` (12%), `variable` (7–45% inverse to AMI)
- 1 public method: 2% flat
- AMI-based variable rate: 45% at 0% AMI → 7% at 150%+ AMI

### 8.3 IRA Rebate Logic
- Rebates now calculated per `cost_scenario` (ensures correct upgrade cost is used for eligibility)
- HVAC heating rebate: $8,000 for ASHP/MSHP systems
- Weatherization rebate: $1,600 for enclosure upgrades (MP9/MP10)

### 8.4 Validation Framework (`utils/validation_framework.py`)
- 5-step validation framework for verifying calculation pipeline integrity
- Step 1: Initialize tracking; Step 2: Pre-calculation checks; Step 3: Post-calculation checks; Step 4: Cross-validation; Step 5: Summary report

### 8.5 Documentation
- `private_impact/CHANGELOG.md` — detailed function-level changelog
- `private_impact/TARE_Private_Capital_Cost_Estimation_Documentation.md` — methodology doc
- `private_impact/Integrate_REMDB_v4_Cost_Estimation.md` — integration guide
- `private_impact/REMDB_v4_Refactoring_Documentation.md` — refactoring notes
- `docs/capital_cost_sensitivity_analysis.md` — single-market (PA) results
- `docs/capital_cost_sensitivity_analysis_national.md` — national results
- `docs/troubleshooting_summary_2026-02-11.md` — debugging notes
- `docs/SENSITIVITY_ANALYSIS_IMPLEMENTATION_PLAN.md` — implementation plan

### 8.6 Archived Files
- Previous v2.1 versions of all notebooks archived in `archived_files/`
- Legacy model_scenarios archived in `model_scenarios/model_scenarios_LEGACY/`

---

## 9. Known Issues & Pending Work

### 9.1 Critical: Visualization Column Name Mismatch (FIXED in v2.3 refactoring)
- `create_multiIndex_adoption_df` in `visuals_adoption_potential.py` was constructing adoption column names **without** `cost_scenario`
- Actual columns: `preIRA_mp8_heating_adoption_central_inmap_acs_v3_fixed_base`
- Function expected: `preIRA_mp8_heating_adoption_central_inmap_acs_fixed_base`
- **Fix:** Added `cost_scenario` parameter to `create_multiIndex_adoption_df`

### 9.2 Pending: Run Simulation Notebook
- Needs `REMDB_COST_SCENARIO_KEYS` import for documentation/verification
- Needs post-export verification of v4 column presence

### 9.3 Pending: Main Model Notebook
- NPV column references need `cost_scenario` inserted (e.g., `_private_npv_moreWTP_v3_fixed_base`)
- Adoption analysis needs `cost_scenario` parameter in `create_multiIndex_adoption_df` calls
- New visualization sections needed for v3 vs v4MID comparison
- Cross-MP sensitivity analysis section needed

### 9.4 Pending: REMDB v4 Cost Discrepancy
- Order-of-magnitude discrepancy between replacement and retrofit estimates under investigation
- v4LOW/v4HIGH remain commented out in `REMDB_COST_SCENARIO_KEYS`

### 9.5 Pending: Non-HVAC End Uses
- Water heating, clothes drying, cooking v4 costs commented out
- Equipment specs for these categories commented out in `EQUIPMENT_SPECS`
