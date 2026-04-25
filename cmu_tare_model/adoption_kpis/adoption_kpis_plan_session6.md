# Adoption KPIs — Session 6: Demand Fix, County Aggregation, Categorical Map

> **Version notes.** Session 6. Fixes demand electricity column, adds county-level
> aggregation for bill savings and demand, builds categorical break-even map.
> Sessions 3–5 (Tasks 1–3) are complete.

## Your Role

Senior Python developer and research software engineer for the TARE model.
You understand ResStock data columns, matplotlib categorical mapping,
geopandas choropleth rendering, and the paper's analytical framework.

## Project Context

Sessions 3–5 successfully split `kpi_functions.py` into dedicated modules
and fixed multi-MP shared-scale maps in the preTARE notebook (Tasks 1–3).
Three issues remain, plus two new analytical requirements:

**Bug — demand uses wrong electricity column.** `demand.py` currently uses
`out.electricity.heating.energy_consumption.kwh` (heating only). This
produces misleading percent changes (+1,285% for NY) because the
denominator is near-zero for gas homes. Must use
`out.electricity.total.energy_consumption` (total residential) for both
baseline and retrofit. The ResStock upgrade file already contains
whole-house electricity after the HP is installed.

**Design change — NG filter scope.** The NG filter was inherited from the
retired `bill_impact_ratio` workflow. It is only needed for COP and
break-even COP (gas-to-HP economics). Bill savings and demand should use
the full ~331K sample (all fuel types). Propane and fuel oil homes often
have the strongest economic case for electrification.

**New — county-level aggregation.** Re-enable county-level aggregation for
bill savings and demand only. COP/break-even stay state-level. County
aggregation does NOT require county-level fuel prices — bill savings uses
TARE lifetime costs (already building-specific), and demand uses total
electricity consumption. Apply `min_home_count=30` masking.

**New — categorical break-even map.** Replace continuous choropleth with
Jenkins-style 4-category map based on boolean break-even columns:
- Very favorable: HP beats break-even at 95% AFUE
- Favorable: beats 90% but not 95%
- Marginal: beats 80% but not 90%
- Unfavorable: doesn't beat 80%

## What Was Done Before

### Sessions 3–4
- Created `data_loading.py`, `spark_gap.py`, `thermal_cop.py`, `demand.py`, `bill_savings.py`
- Cleaned both notebooks, retired bill_impact_ratio

### Session 5 (Tasks 1–3 complete)
- Fixed break-even for all MPs (`breakeven_results` dict)
- Fixed shared-scale COP and break-even maps
- Fixed comparison tables for all MPs
- Removed `primary_mp` pattern

### This session (Session 6)
- Fix `demand.py` electricity column + percent change formula
- Remove NG filter default from demand and bill savings
- Add county-level aggregation to `bill_savings.py` and `demand.py`
- Build categorical break-even map in preTARE
- Fix postTARE maps (shared scales, updated demand calls)
- Session 5 cleanup (mid-notebook import, dead imports)

## Key Column Reference

```python
# Total residential electricity (correct — use this for demand)
ELEC_TOTAL_COL = "out.electricity.total.energy_consumption"

# Heating electricity only (wrong for demand — do NOT use)
ELEC_HEATING_COL = "out.electricity.heating.energy_consumption.kwh"
```

The ResStock upgrade file's `out.electricity.total.energy_consumption` already
reflects the full whole-house electricity AFTER the HP is installed — it is
not just the heating component. So demand change = `upgrade_total - baseline_total`.

## Percent Change Formula

At the aggregate (state or county) level:
```python
pct_change = (aggregate_retrofit_total - aggregate_baseline_total) / aggregate_baseline_total * 100
```

No `abs()` on the denominator. Baseline total residential electricity is always
a large positive number (10,000–15,000 kWh/home for lighting, appliances,
cooling, etc.). Expected percent changes: +20–60% for gas-heavy states,
small negative for electric-resistance-heavy states.

## NG Filter Rules

| Metric | NG Filter | Why |
|--------|-----------|-----|
| Thermal COP | Yes | COP compared against gas break-even thresholds |
| Break-even COP | Yes | Formula uses spark_gap (elec/gas) × AFUE (gas furnace) |
| Categorical break-even map | Yes | Derived from break-even COP |
| Bill savings ratio | **No** | TARE costs are fuel-specific per building |
| Electricity demand | **No** | Grid impact includes all fuel-to-HP conversions |

## County-Level Aggregation Design

Both `bill_savings.py` and `demand.py` get a `geo_level: str = 'state'`
parameter. When `geo_level='county'`, aggregate by `in.county` (FIPS code)
instead of `in.state`. Add a `state` column derived from the EUSS `in.state`
for downstream merges. Apply `min_home_count` threshold — set metric
columns to NaN for counties below threshold.

The county column in EUSS is `in.county` (GISJOIN format).

## Tasks

### Task 1 — Fix `demand.py`: total electricity column

**Changes to `compute_scenario_demand()`:**
1. Replace `elec_col = 'out.electricity.heating.energy_consumption.kwh'`
   with the total electricity column
2. Baseline: `df_baseline[ELEC_TOTAL_COL]`
3. Retrofit: `df_upgrade[ELEC_TOTAL_COL]`
4. Remove the manual sum of heating fuel columns — no longer needed
5. Remove HP backup and fans/pumps column references — the total column
   already includes them
6. Keep `site_energy_change_kwh` if still useful, but reconsider: with
   total electricity, the "site energy" concept changes (now includes
   non-heating loads). May want to drop this column or rename.

**Changes to `aggregate_demand_by_state()`:**
1. Fix percent change: `(retrofit - baseline) / baseline * 100`
2. Ensure denominator is `aggregate_baseline_total` (weighted sum across
   homes), NOT per-building baseline

**Import `ELEC_TOTAL_COL` from `data_loading.py`** (add constant there if
not already present).

### Task 2 — Add county-level aggregation to `demand.py`

Add `geo_level: str = 'state'` parameter to `aggregate_demand_by_state()`.

When `geo_level='county'`:
- Group by `in.county` instead of `in.state`
- Include `in.state` in the output for downstream merges
- Apply `min_home_count` masking (default 30)
- Rename function to `aggregate_demand()` (drop `_by_state` since it
  now supports both levels)

### Task 3 — Add county-level aggregation to `bill_savings.py`

Add `geo_level: str = 'state'` parameter to `aggregate_bill_savings_by_state()`.

Same pattern as Task 2:
- Group by `in.county` when county-level
- Include `in.state` in output
- `min_home_count` masking
- Rename to `aggregate_bill_savings()`

### Task 4 — Update `__init__.py`

Update exports to reflect renamed functions:
```python
from .demand import compute_scenario_demand, aggregate_demand
from .bill_savings import compute_bill_savings_ratio, aggregate_bill_savings
```

### Task 5 — Build categorical break-even map (preTARE)

Create a categorical map using the boolean break-even columns. The map
assigns each state to one of 4 categories based on whether the effective
annual COP exceeds break-even thresholds at 80%, 90%, and 95% AFUE.

**Category logic (per state, per MP):**
```python
if hp_beats_breakeven_95:
    category = 'Very Favorable'    # HP wins even vs high-eff furnace
elif hp_beats_breakeven_90:
    category = 'Favorable'         # HP wins vs typical furnace
elif hp_beats_breakeven_80:
    category = 'Marginal'          # HP wins only vs old/inefficient furnace
else:
    category = 'Unfavorable'       # Gas furnace cheaper at any efficiency
```

**Color scheme (inspired by Jenkins but grounded in modeled performance):**
- Very Favorable: dark green
- Favorable: light green / yellow-green
- Marginal: orange / yellow
- Unfavorable: dark brown / red

**Implementation:** Use `ListedColormap` + `BoundaryNorm` with integer
category codes (0–3), NOT a continuous colormap. The legend should show
category labels, not a continuous colorbar.

Generate one map per MP with a shared categorical scale (same 4 colors,
same legend). The visual comparison shows how upgrading from standard
(MP3) to high-efficiency (MP4) shifts states from unfavorable to favorable.

### Task 6 — Fix postTARE notebook

1. Update import block (renamed functions)
2. Update `compute_scenario_demand()` call — confirm `fuel_filter=None` (already correct)
3. Update demand display — percent changes should now be plausible (20–60% range)
4. Shared-scale maps for bill savings ratio and demand change across MPs
5. Compute shared norms BEFORE the per-MP loop

### Task 7 — Cleanup

1. Move `from cmu_tare_model.constants import JENKINS_BREAKEVEN_REF_90`
   to preTARE top import cell (if not already done)
2. Remove `calculate_spark_gap` dead import from postTARE (if present)
3. Add `ELEC_TOTAL_COL` to `data_loading.py` constants
4. Verify no `primary_mp` references remain in computation/visualization cells
5. Verify `fuel_filter=None` for demand and bill savings calls

## Reference Values

| Metric | Current (broken) | Expected (fixed) |
|--------|-----------------|-------------------|
| NY elec demand pct change | +1,285% | ~+30–50% |
| FL elec demand pct change | -72% | ~-5–10% |
| Total homes in demand calc | 180,939 (NG only) | ~331,526 (all fuels) |
| MP3 bill savings states < 1.0 | 39/49 | TBD (all fuels, will change) |
| MP4 bill savings states < 1.0 | 49/49 | TBD (all fuels, may change) |

COP and break-even values are UNCHANGED — those computations are correct.

## Known Anti-Patterns

| Anti-pattern | Why it's wrong |
|-------------|----------------|
| Using `out.electricity.heating.energy_consumption.kwh` for demand | Heating-only; produces misleading % changes |
| Applying NG filter to demand or bill savings | Only COP/break-even need NG filter |
| Per-building percent change then averaging | Must use aggregate formula: (Σretrofit - Σbaseline) / Σbaseline |
| Using `abs()` in percent change denominator | Baseline total elec is always large positive |
| Continuous choropleth for break-even map | Categorical map with 4 bins is more policy-readable |
| Computing norm inside per-MP loop | Gives each MP its own scale |
| Using `Normalize` for demand change map | Need `TwoSlopeNorm(vcenter=0)` for diverging palette |
| Separate state/county aggregation functions | Use `geo_level` parameter |

## Session Summary Template

1. Files modified with change descriptions
2. New demand percent change values (should be 20–60% range, not 1,000%+)
3. Categorical map category counts per MP
4. Confirm all fuel_filter settings are correct per metric
5. Confirm county-level aggregation works for bill savings and demand
6. Updated golden values table
