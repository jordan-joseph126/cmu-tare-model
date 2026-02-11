# TARE Model Troubleshooting Summary — February 11, 2026

## Notebook

`cmu_tare_model/model_scenarios/tare_scenarios_v2_2.ipynb`

Running for **PA only** while troubleshooting. Tests are located in cells 58–69 of the notebook.

---

## Issues Investigated

Three test failures were analyzed. Two were test-code bugs and one was a codebase issue requiring a data-pipeline fix.

---

## Test 2: v4 Quantile Regression Crossing Diagnostic (INFORMATIONAL)

### Status: **RESOLVED — not a bug; expected behavior per REMDB guidance**

### Previous Symptom

- **Heating replacement**: 1,492 rows where `v4LOW > v4MID`
- **Cooling replacement**: 37,385 rows where `v4MID > v4HIGH`; 21,963 rows where `v4LOW > v4HIGH`
- Heating upgrade: no crossings

### Understanding (Updated Feb 11, 2026)

Per the **REMDB Machine Readable Guidance Document (Dec 2023)**, low/mid/high represent **10th, 50th, and 90th percentile quantile regressions fitted independently**. The guidance document itself contains explicit examples where coefficients are non-monotonic across percentiles:

| Component | Parameter | Low | Mid | High | Behavior |
|-----------|-----------|-----|-----|------|----------|
| Water Heater HP Tank | PM2 coef (Nominal volume) | 28.81 | 19.33 | **8.39** | **decreasing** |
| Water Heater HP Tank | Intercept | 155.30 | 436.45 | **-651.90** | **non-monotonic** |

This is **by design** of quantile regression — different quantiles are fitted independently and can "cross" for specific input value combinations. The guidance document demonstrates this methodology in its worked examples (Example 1: ASHP, Example 2: Attic Insulation).

Additional REMDB coefficient examples from our data:

| Row ID | Parameter | Low | Mid | High |
|--------|-----------|-----|-----|------|
| `air_conditioner_room_ac_window_or_through_wall` | pm1_coef | 350.16 | 364.33 | 311.64 |
| `furnaces_gas_furnace` | intercept | -2009.22 | -2780.00 | -2910.15 |
| `air_source_heat_pump_centrally_ducted` | intercept | -1374.60 | -2291.00 | -3207.40 |

### Resolution

- **Monotonicity enforcement was REMOVED** from cell 39 (the v4 merge cell). The previous `np.sort(arr, axis=1)` workaround incorrectly treated crossings as errors.
- **Test 2 was converted** from a pass/fail test to an **informational diagnostic** that reports crossing rates without treating them as failures.
- Quantile regression crossings are expected for certain performance metric input combinations and should be preserved in the data — they represent the actual statistical uncertainty distribution.

---

## Test 5: NPV Consistency — Column Name Mismatch

### Status: **FIX IMPLEMENTED — needs re-run verification**

### Symptom

Four v4MID NPV columns reported as `NOT FOUND`:
```
✗ NPV lessWTP (v4MID)  | Column 'preIRA_mp3_heating_private_npv_lessWTP_v4MID' NOT FOUND
✗ NPV moreWTP (v4MID)  | Column 'preIRA_mp3_heating_private_npv_moreWTP_v4MID' NOT FOUND
✗ NPV lessWTP (v4MID)  | Column 'iraRef_mp3_heating_private_npv_lessWTP_v4MID' NOT FOUND
✗ NPV moreWTP (v4MID)  | Column 'iraRef_mp3_heating_private_npv_moreWTP_v4MID' NOT FOUND
```

### Root Cause

**Test bug** — the test used `method_suffix=''` for v4MID NPV lookups but the actual column names include the discount method suffix.

The NPV column naming pattern from `create_npv_col()` in `column_names.py` is:

```
{scenario_prefix}{category}_private_npv_{wtp}_{cost_scenario}{method_suffix}
```

The reference DataFrame is `DATAFRAMES_MPX_RCM_DISCOUNT_RATE['fixed_base']['ap2']`, which uses discount rate `fixed_base`. The actual columns therefore are:

```
preIRA_mp3_heating_private_npv_lessWTP_v4MID_fixed_base   ← actual
preIRA_mp3_heating_private_npv_lessWTP_v4MID              ← what the test looked for
```

The v3 lookups in the same test already correctly used `method_suffix='_fixed_base'`.

### Fix Applied (Cell 64 — `#VSC-1288fb61`)

Added `NPV_METHOD_SUFFIX = '_fixed_base'` and updated all `create_npv_col()` calls (both v4MID and v3) to use it consistently:

```python
NPV_METHOD_SUFFIX = '_fixed_base'
# ...
col_npv = create_npv_col(..., cost_scenario='v4MID', method_suffix=NPV_METHOD_SUFFIX)
col_npv_v3 = create_npv_col(..., cost_scenario='v3', method_suffix=NPV_METHOD_SUFFIX)
```

### Key File: `cmu_tare_model/utils/column_names.py`

The `create_npv_col` function signature:
```python
def create_npv_col(scenario_prefix, category, wtp, cost_scenario, method_suffix) -> str:
    return f'{scenario_prefix}{category}_private_npv_{wtp}_{cost_scenario}{method_suffix}'
```

The `method_suffix` values come from `PRIVATE_DISCOUNTING_METHOD_SUFFIXES` in `cmu_tare_model/utils/discounting.py`:
- `'_fixed_low'`
- `'_fixed_base'`
- `'_fixed_high'`
- `'_variable'`

---

## Test 6: v4 Column Propagation — Suffix Mismatch

### Status: **FIX IMPLEMENTED — needs re-run verification**

### Symptom

Test reported only 2 "v4-suffixed" columns across the entire DataFrame, and those turned out to be `private_discount_rate_fixed_low` and `private_discount_rate_fixed_high` — discount rate columns, not REMDB cost columns.

### Root Cause

**Test bug** — the test searched for column suffixes `_low`, `_mid`, `_high` but the actual REMDB v4 cost column suffixes are `_v4LOW`, `_v4MID`, `_v4HIGH`.

The discount rate columns `private_discount_rate_fixed_low` and `private_discount_rate_fixed_high` end in `_low`/`_high`, causing false positive matches.

### Fix Applied (Cell 65 — `#VSC-bdbfbfd4`)

Changed the suffix list to use the actual REMDB cost scenario suffixes:

```python
# Before (wrong):
v4_suffixes = ['_low', '_mid', '_high']

# After (correct):
v4_suffixes = ['_v4LOW', '_v4MID', '_v4HIGH']
```

---

## Cooling Replacement Cost — Deep Dive Diagnostic (NEW)

### Status: **New diagnostic cell added (Cell 68) — not yet executed**

### Concern

Cooling replacement costs should be a similar order of magnitude to heating upgrade costs. A comprehensive diagnostic cell (TEST 9) was added after Test 8 to investigate:

1. **Cooling type distribution** — `hvac_cooling_type` value counts and `hvac_has_ducts` distribution
2. **Row ID mapping** — how cooling types map to REMDB row_ids (via `_assign_replacement_row_id`)
3. **Coefficient monotonicity** — per-row-id check of whether REMDB coefficients are monotonic
4. **Per-row-id cost statistics** — mean/median/P5/P95 broken out by row_id for each v4 scenario
5. **Magnitude comparison** — heating upgrade vs heating replacement vs cooling replacement (v4MID)
6. **Post-enforcement violations** — confirms monotonicity enforcement worked

### Cooling Row ID Mapping (from `remdb_v4_installed_cost_utils.py`)

```python
# _assign_replacement_row_id, end_use='cooling':
conditions = [
    (df['hvac_cooling_type'] == 'Room AC'),         → 'air_conditioner_room_ac_window_or_through_wall'
    (df['hvac_cooling_type'] == 'Central AC'),       → 'air_conditioner_centrally_ducted'
    (df['hvac_cooling_type'] == 'Heat Pump') & ducts → 'air_source_heat_pump_centrally_ducted'
    (df['hvac_cooling_type'] == 'Heat Pump') & !ducts → 'air_source_heat_pump_non_ducted_multi_zone'
]
```

### Known REMDB Coefficient Issues for Cooling

| row_id | pm1_coef | intercept | Notes |
|--------|----------|-----------|-------|
| `air_conditioner_room_ac_window_or_through_wall` | high (311.64) < low (350.16) | non-monotonic | Primary cause of cooling violations |
| `air_conditioner_centrally_ducted` | OK (monotonic) | non-monotonic: low=-3533, mid=-5889, high=-8245 | Intercepts decrease |
| `air_source_heat_pump_centrally_ducted` | OK (monotonic) | non-monotonic: low=-1375, mid=-2291, high=-3207 | Intercepts decrease |
| `air_source_heat_pump_non_ducted_multi_zone` | OK (monotonic) | non-monotonic: low=-806, mid=-1343, high=-1881 | Intercepts decrease |

### Areas for Future Investigation

- Verify that cooling costs are in the right ballpark compared to heating costs
- Check if room AC replacements are dominating the violation count
- Consider whether the `multiplier_retrofit` (1.5 for heat pumps and central ACs, 1.0 for room ACs) and `adder_retrofit` values are reasonable
- The room AC row has `adder_retrofit=352.61` while heat pumps have `adder_retrofit=-1384` or `0` — this asymmetry may affect cost magnitude

---

## Key File Locations

| File | Purpose |
|------|---------|
| `cmu_tare_model/model_scenarios/tare_scenarios_v2_2.ipynb` | Main notebook |
| `cmu_tare_model/utils/column_names.py` | All column name builder functions (`create_cost_col`, `create_capital_col`, `create_npv_col`, etc.) |
| `cmu_tare_model/utils/remdb_v4_installed_cost_utils.py` | REMDB v4 metric prep: row_id assignment, coefficient mapping, unit conversion, `add_remdb_metrics()` |
| `cmu_tare_model/private_impact/calculations/calculate_equipment_replacement_costs.py` | Unified replacement cost calculator (v3 + v4) |
| `cmu_tare_model/private_impact/calculations/calculate_equipment_installation_costs.py` | Unified upgrade cost calculator (v3 + v4) |
| `cmu_tare_model/private_impact/calculate_lifetime_private_impact.py` | Private NPV: `calculate_private_npv()`, `calculate_capital_costs()`, `calculate_and_update_npv()` |
| `cmu_tare_model/utils/discounting.py` | Discount rate constants: `PRIVATE_DISCOUNT_RATE_SHORT_KEYS`, `PRIVATE_DISCOUNTING_METHOD_SUFFIXES` |
| `cmu_tare_model/data/retrofit_costs/remdb_v4_tare_retrofit_costs.csv` | REMDB v4 regression coefficients (34 rows) |
| `cmu_tare_model/constants.py` | `REMDB_COST_SCENARIO_KEYS = ['v3', 'v4LOW', 'v4MID', 'v4HIGH']`, `EQUIPMENT_SPECS`, etc. |

---

## REMDB v4 Cost Formula Reference

```
Material_Price = (pm1 * pm1_coef_{pct}) + (pm2 * pm2_coef_{pct}) + intercept_{pct}
Installed_Cost = (Material_Price * multiplier_retrofit) + adder_retrofit
```

Where `{pct}` is `low`, `mid`, or `high`.

- `pm1` = capacity (converted to REMDB units: Tons for heat pumps/ACs, BTU/hr for furnaces)
- `pm2` = efficiency (SEER for ACs/HPs, AFUE÷100 for furnaces)
- Costs are in 2023$ (no CPI adjustment needed)

---

## Column Naming Conventions

### Input cost columns (from data pipeline)

```
mp{menu_mp}_{category}_{cost_type}_installed_cost_{cost_scenario}
```
Examples:
- `mp3_heating_upgrade_installed_cost_v3`
- `mp3_heating_replacement_installed_cost_v4MID`
- `mp3_cooling_replacement_installed_cost_v4HIGH`

### Output capital cost columns

```
{scenario_prefix}{category}_{total|net}_capital_cost_{cost_scenario}
```
Examples:
- `preIRA_mp3_heating_total_capital_cost_v3`
- `iraRef_mp3_heating_net_capital_cost_v4MID`

### Output NPV columns

```
{scenario_prefix}{category}_private_npv_{wtp}_{cost_scenario}{method_suffix}
```
Examples:
- `preIRA_mp3_heating_private_npv_lessWTP_v3_fixed_base`
- `iraRef_mp3_heating_private_npv_moreWTP_v4MID_fixed_base`

### Rebate columns

```
mp{menu_mp}_{category}_rebate_amount_{cost_scenario}
```

---

## Notebook Execution Flow (Key Cells)

| Cell # | Description | Exec Count (last run) |
|--------|-------------|----------------------|
| 38 | REMDB v4 capital cost scenario loop (heating + cooling) | 26 |
| 39 | **Merge v4 columns + monotonicity enforcement** | 27 |
| 41 | Rebate calculations (IRA) | 28 |
| 43 | Build `DATAFRAMES_MPX_RCM_DISCOUNT_RATE` dictionary | 29 |
| 46 | Public NPV (No IRA) | 31 |
| 47 | **Private NPV (No IRA)** — loops over all cost scenarios × discount rates × RCM models | 32 |
| 48 | Adoption potential (No IRA) | 33 |
| 50–52 | IRA scenario (public, private, adoption) | 34–36 |
| 58 | TEST 1: Data integrity | — |
| 59 | TEST 2: v4 Monotonicity | — |
| 60 | TEST 3: Cross-scenario summary stats | — |
| 64 | TEST 5: NPV consistency | — |
| 65 | TEST 6: v4 column propagation | — |
| 68 | **TEST 9: Cooling replacement deep dive (NEW)** | — |

---

## Data Flow Summary

```
df_euss_am_mpX_home
  │
  ├── v3 costs calculated directly (cells 26, etc.)
  │
  ├── v4 loop (cell 38): For each scenario_key in [v4LOW, v4MID, v4HIGH]:
  │     ├── add_remdb_metrics() → assigns row_id, maps coefficients, converts units
  │     ├── calculate_replacement_installed_cost() → applies regression formula
  │     ├── calculate_upgrade_installed_cost() → applies regression formula
  │     └── Also: cooling replacement via same two-step process
  │     Results stored in CAPITAL_COSTS_MPX[end_use][cost_type][scenario_key]
  │
  ├── Merge v4 columns back (cell 39) + MONOTONICITY ENFORCEMENT
  │
  ├── Rebate calculation (cell 41)
  │
  └── DATAFRAMES_MPX_RCM_DISCOUNT_RATE (cell 43)
        Built from df_euss_am_mpX_home.copy() — inherits enforced monotonicity
        Structure: [discount_rate_key][rcm_model] → DataFrame
```

---

## Open Questions / Next Steps

1. **Re-run cells 39+ after kernel restart** to verify all three test fixes produce PASS results
2. **Review TEST 9 output** for cooling replacement cost diagnostics — check magnitude vs heating
3. **Consider whether monotonicity enforcement is the right long-term approach** vs fixing REMDB regressions upstream
4. **Confirm v4MID NPV columns now appear** in Test 5 after the `method_suffix` fix
5. **Check if cooling replacement costs lack a v3 baseline** for comparison (v3 is intentionally skipped for cooling)
6. **Investigate whether `air_conditioner_room_ac_window_or_through_wall` costs** are reasonable — `multiplier_retrofit=1.0` vs heat pump's `1.5` could explain magnitude differences
