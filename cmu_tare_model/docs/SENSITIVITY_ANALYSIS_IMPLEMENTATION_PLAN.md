# Implementation Plan: Enhanced Capital Cost Sensitivity Analysis

**Date:** February 10, 2026  
**Notebook:** `cmu_tare_model/model_scenarios/tare_scenarios_v2_2.ipynb`  
**Status:** Ready to implement — notebook is at committed state (55 cells), needs full re-run before adding new cells.

---

## Context

The notebook currently has 55 cells (last commit `85c0a29`). The sensitivity analysis section needs to be added after cell 55. The notebook must be fully executed (cells 1–55) before adding the new cells so that key variables are populated:
- `df_euss_am_mpX_home` — main DataFrame with all cost columns, metadata
- `CAPITAL_COSTS_MPX` — dictionary of cost DataFrames by end_use/cost_type/scenario
- `DATAFRAMES_MPX_RCM_DISCOUNT_RATE` — post-NPV DataFrames keyed by discount rate and RCM model
- `menu_mp` — measure package number (e.g., 8)
- `location_id` — location identifier string
- `REMDB_COST_SCENARIO_KEYS` — `['v3', 'v4LOW', 'remdb_v4_mid', 'remdb_v4_high']`

## Key Data Columns

| Column | Source | Notes |
|--------|--------|-------|
| Cost columns | `create_cost_col(menu_mp, category, cost_type, scenario)` | e.g. `mp8_heating_upgrade_installed_cost_mid` |
| Cooling v3 | **Does not exist** | Only v4_low/mid/high for cooling replacement |
| Square footage | `square_footage` | In `df_euss_am_mpX_home`, some extreme values exist |
| Region | `census_division_recs` | Standard 9 U.S. Census divisions |
| Fuel type | `base_heating_fuel` | Natural Gas, Electricity, Fuel Oil, Propane |

### Column Name Construction

```python
from cmu_tare_model.utils.column_names import create_cost_col
# Pattern: mp{menu_mp}_{category}_{cost_type}_installed_cost{sfx}
# sfx = '' for v3, '_low'/'_mid'/'_high' for v4
create_cost_col(8, 'heating', 'upgrade', 'remdb_v4_mid')  # → 'mp8_heating_upgrade_installed_cost_mid'
create_cost_col(8, 'cooling', 'replacement', 'remdb_v4_mid')  # → 'mp8_cooling_replacement_installed_cost_mid'
```

### Scenario Applicability

| Cost Metric | v3 | v4LOW | remdb_v4_mid | remdb_v4_high |
|-------------|----------|--------------|--------------|---------------|
| Heating Upgrade | Yes | Yes | Yes | Yes |
| Heating Replacement | Yes | Yes | Yes | Yes |
| Cooling Replacement | **No** | Yes | Yes | Yes |

## Cell-by-Cell Plan (12 new cells: 56–67)

### Cell 56 — Markdown: Section Header
```markdown
# Capital Cost Sensitivity Analysis: REMDB v3 vs v4 (Low / Mid / High)
Compares installed cost estimates across the 4 cost scenarios for heating upgrade, heating replacement, and cooling replacement.
Tests data integrity, monotonicity (low < mid < high), and cross-scenario reasonableness.
```

### Cell 57 — Code: TEST 1: Data Integrity
- Loop over 3 cost metrics: `('heating', 'upgrade')`, `('heating', 'replacement')`, `('cooling', 'replacement')`
- For each metric, loop over applicable scenarios (all 4 for heating, only v4_low/mid/high for cooling)
- For each: check column existence, NaN rate (<50%), no negatives, positive mean
- Print PASS/FAIL table with valid count, NaN%, neg count, mean, median
- Cooling replacement `v3` gets a printed "SKIP (no v3 data)" note

### Cell 58 — Code: TEST 2: v4 Monotonicity
- Check `low ≤ mid ≤ high` for all 3 cost metrics
- For each metric: count violations for low>mid, mid>high, low>high
- Print violations count + example rows (head 5) if any violations exist
- Print PASS/FAIL summary

### Cell 59 — Code: TEST 3: Cross-Scenario Summary Stats
- Descriptive stats table for all 3 metrics × applicable scenarios
- Columns: Cost Type, Scenario, N Valid, Mean, Std, P5, P25, Median, P75, P95, Min, Max
- Plus v3 vs v4_mid pairwise comparison for heating upgrade and heating replacement only:
  - Mean/median of each, difference, pct difference, ratio (mean, median, P5, P95)

### Cell 60 — Code: TEST 4a: Heating Upgrade Visualization
**2×2 figure (16×10):**
- Top-left: Boxplot of heating upgrade cost by 4 scenarios (v3, v4_low, v4_mid, v4_high)
- Top-right: Overlaid histogram (1st–99th percentile) by 4 scenarios
- Bottom-left: Boxplot of `square_footage` for valid heating upgrade homes (single distribution)
- Bottom-right: Histogram of `square_footage` (1st–99th percentile)
- Save as `docs/capital_cost_sensitivity_heating_upgrade_mp{X}_{location}.png`

**Style:**
- Colors: `['#4C72B0', '#55A868', '#C44E52', '#8172B2']` for v3/v4_low/v4_mid/v4_high
- Sqft bottom row uses neutral color `#666666`
- Dollar axes: `$` with commas via `mticker.FuncFormatter`
- Outlier-free boxplots (`showfliers=False`), means shown as red diamonds
- `fig.suptitle(f'Heating Upgrade Cost Sensitivity (MP{menu_mp}) — {n_valid:,} valid homes')`

### Cell 61 — Code: TEST 4b: Heating Replacement Visualization
- Same 2×2 layout as Cell 60 but for heating replacement
- 4 scenarios (v3, v4_low, v4_mid, v4_high)
- Save as `docs/capital_cost_sensitivity_heating_replacement_mp{X}_{location}.png`

### Cell 62 — Code: TEST 4c: Cooling Replacement Visualization
- Same 2×2 layout but **only 3 scenarios** (v4_low, v4_mid, v4_high — no v3)
- Use 3 colors: `['#55A868', '#C44E52', '#8172B2']`
- Save as `docs/capital_cost_sensitivity_cooling_replacement_mp{X}_{location}.png`

### Cell 63 — Code: TEST 5: NPV Consistency
- Use `DATAFRAMES_MPX_RCM_DISCOUNT_RATE['fixed_base']['ap2']` as reference DataFrame
- For each policy (No IRA, IRA):
  - Check total/net capital cost column existence (with `_mid` suffix)
  - Check lessWTP/moreWTP NPV column existence and stats
  - Verify moreWTP ≥ lessWTP

### Cell 64 — Code: TEST 6: v4 Column Propagation
- Count v4-suffixed columns in `df_euss_am_mpX_home`
- Count v4-suffixed columns in post-NPV `DATAFRAMES_MPX_RCM_DISCOUNT_RATE['fixed_base']['ap2']`
- Print `CAPITAL_COSTS_MPX` structure: end_use.cost_type → list of scenario keys

### Cell 65 — Code: Regional Disaggregation
- Group by `census_division_recs` × scenario
- For each of the 3 cost metrics:
  - Print table: Region | Scenario | N | Mean | Median | P25 | P75
- 9 census divisions × 4 scenarios for heating (3 for cooling)
- Use `pd.DataFrame` for formatted output

### Cell 66 — Code: Fuel Type Disaggregation
- Group by `base_heating_fuel` × scenario
- For heating upgrade & replacement:
  - Print table: Fuel | Scenario | N | Mean | Median | P25 | P75
- For cooling replacement: note that cooling fuel is always Electricity, so disaggregate by `base_heating_fuel` as a proxy for home characteristics
- ~4 fuel types × 4 scenarios (3 for cooling)

### Cell 67 — Code: Per-Scenario Summary + Save Figures
- Compact overall summary for all 3 cost metrics (one line per scenario per metric)
- Print all saved figure paths
- Print total model sample size and valid counts

## Visualization Layout (each of the 3 figures)

```
┌─────────────────────────────┬─────────────────────────────┐
│  Cost Metric — Boxplot      │  Cost Metric — Histogram    │
│  (scenarios on x-axis)      │  (overlaid, 1st-99th pctl)  │
├─────────────────────────────┼─────────────────────────────┤
│  Square Footage — Boxplot   │  Square Footage — Histogram │
│  (single distribution,      │  (1st-99th pctl, neutral    │
│   neutral color)            │   color)                    │
└─────────────────────────────┴─────────────────────────────┘
```

## Implementation Notes

1. All cells use `from cmu_tare_model.constants import REMDB_COST_SCENARIO_KEYS, parse_cost_scenario` and `from cmu_tare_model.utils.column_names import create_cost_col`
2. Define a `COST_METRICS` list at the top of cell 57 for reuse:
   ```python
   COST_METRICS = [
       ('heating', 'upgrade', REMDB_COST_SCENARIO_KEYS),           # all 4 scenarios
       ('heating', 'replacement', REMDB_COST_SCENARIO_KEYS),       # all 4 scenarios
       ('cooling', 'replacement', [k for k in REMDB_COST_SCENARIO_KEYS if k != 'v3']),  # v4 only
   ]
   ```
3. Square footage clipping for visualization: use 1st–99th percentile to handle extreme values
4. `location_id` variable should be available from earlier cells for figure naming
5. All figures saved to `cmu_tare_model/docs/` directory
6. After all cells run, update the national markdown report (`capital_cost_sensitivity_analysis_national.md`) with new data including cooling, regional tables, fuel tables, and updated visualizations
