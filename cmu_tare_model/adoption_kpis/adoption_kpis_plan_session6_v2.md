# Adoption KPIs — Session 6 Plan v2 (Post-Audit)

> **Version notes.** v2 incorporates findings from the pre-Session 6 codebase
> audit. Original Session 6 tasks 1–4 (demand column fix, county aggregation,
> `__init__.py`, fuel_filter) are COMPLETE — the audit confirmed correct
> implementations already in place. Session 6 now focuses on visualization
> consolidation, county-level plotting, and notebook cleanup.

## Your Role

Senior Python developer and research software engineer for the TARE model.
You understand ResStock data columns, matplotlib categorical mapping,
geopandas choropleth rendering, and the paper's analytical framework.

## Project Context

Sessions 3–5 split `kpi_functions.py` into dedicated modules (`data_loading.py`,
`spark_gap.py`, `thermal_cop.py`, `demand.py`, `bill_savings.py`) and fixed
shared-scale maps. The pre-Session 6 audit confirmed that demand columns,
aggregation functions, `__init__.py` exports, and `fuel_filter` settings are
all correct.

However, the audit surfaced a new class of issues: the preTARE notebook
contains **four inline function definitions** that shadow the module imports
from `visualize_geospatial_data.py`. One of these — `plot_combined_choropleth`
— has **diverged layout parameters** from the module version (different font
sizes, colorbar positioning). Additionally, there is no county-level
choropleth *plotting* function, and the categorical break-even map logic
exists only inline.

## Scope Constraint — CRITICAL

**In scope:** Visualization consolidation, county-level plotting, notebook
cleanup, extracting inline logic to modules.

**Out of scope — Session 7:**
- Adoption potential computation and visualization (Tier 1/Tier 2 county maps,
  monochromatic color scales, label overlap fixes)
- `visualize_tabular_data.py` implementation
- Any changes to `thermal_cop.py` or `spark_gap.py` module logic

## What Was Done Before

### Sessions 3–4
- Created all five dedicated modules from monolithic `kpi_functions.py`
- Cleaned both notebooks, retired bill_impact_ratio

### Session 5
- Fixed break-even for all MPs (`breakeven_results` dict)
- Fixed shared-scale COP and break-even maps
- Removed `primary_mp` pattern

### Pre-Session 6 (user-completed)
- Fixed demand.py: uses `ELEC_TOTAL_COL`, correct aggregate formula, no `abs()`
- Added `geo_level` + `min_home_count` to `aggregate_demand` and `aggregate_bill_savings`
- Updated `__init__.py` with backward-compatible aliases
- Set `fuel_filter=None` for bill savings and demand in postTARE
- Built categorical break-even map inline in preTARE (ListedColormap + BoundaryNorm)

## Current Implementation Status

| Item | Status | Notes |
|------|--------|-------|
| Demand column (`ELEC_TOTAL_COL`) | ✅ Done | Correct in demand.py |
| Aggregate percent change formula | ✅ Done | No `abs()`, weighted aggregate |
| County aggregation (demand) | ✅ Done | `geo_level` param works |
| County aggregation (bill savings) | ✅ Done | `geo_level` param works |
| `__init__.py` exports | ✅ Done | Backward-compatible aliases |
| fuel_filter settings | ✅ Done | Correctly overridden in postTARE |
| Categorical break-even map | ✅ Inline | Exists in preTARE, not modularized |
| Inline function duplicates in preTARE | ⬜ Needs fix | 4 functions shadow module imports |
| `plot_combined_choropleth` divergence | ⬜ Needs fix | Layout params differ between module and inline |
| County choropleth plotting function | ⬜ Missing | `prepare_county_geodataframe` exists, no plotter |
| postTARE county-level maps | ⬜ Not done | Functions support it, notebooks don't use it yet |
| Unused imports in postTARE | ⬜ Needs cleanup | `create_choropleth_map`, `FUEL_PRICES_PATH` |
| `ELEC_TOTAL_COL` in `BASELINE_USECOLS` | ⬜ Cosmetic | No runtime impact |

## Tasks

### Task 1 — Reconcile `plot_combined_choropleth` and delete inline duplicates

**Problem:** preTARE notebook has a large cell (execution count 32) that
redefines `prepare_state_geodataframe`, `create_choropleth_map`,
`plot_combined_choropleth`, and `prepare_county_geodataframe`. These shadow
the module imports. `plot_combined_choropleth` has diverged layout params.

**Diverged parameters:**

| Parameter | Module | Notebook inline |
|-----------|--------|-----------------|
| `map_bottom` | 0.18 | 0.15 |
| `map_height` | 0.78 | 0.81 |
| colorbar axes | [0.30, 0.06, 0.40, 0.035] | [0.25, 0.04, 0.50, 0.035] |
| title fontsize | 16 | 20 |
| cbar label fontsize | 13 | 18 |
| tick labelsize | 12 | 16 |

**Decision required:** Which layout is preferred? The notebook inline version
has larger fonts and wider colorbar — likely tuned for higher-DPI output or
presentation slides. The developer should compare outputs from both before
committing.

**Steps:**
1. Update `plot_combined_choropleth` in `visualize_geospatial_data.py` to
   use the preferred layout parameters (likely the notebook inline version,
   since it was tuned more recently)
2. Delete the entire inline cell from preTARE notebook
3. Verify preTARE imports resolve correctly and maps render identically
4. Verify `prepare_county_geodataframe` is still importable from the module

**Validation:** Re-run preTARE COP and break-even maps. Visual output should
match pre-edit appearance.

### Task 2 — Build county-level choropleth plotting

**Problem:** `prepare_county_geodataframe` exists in the module but there is
no corresponding plotting function. County-level maps are needed for bill
savings and demand in the postTARE notebook.

**Design:** Extend `plot_combined_choropleth` to accept a `geo_level` parameter,
OR create a parallel `plot_combined_choropleth_county` function. The function
needs to:

1. Accept county-level GeoDataFrame (FIPS-merged via `prepare_county_geodataframe`)
2. Handle the larger number of geometries (3,000+ counties vs 49 states)
3. Render CONUS + Alaska inset (same layout pattern as state maps)
4. Suppress county borders or use very thin linewidths (too many polygons
   for thick borders)
5. Accept shared `cmap` and `norm` (same pattern as state maps)

**Recommended approach:** Add `geo_level: str = 'state'` parameter to
`plot_combined_choropleth`. When `'county'`, use `prepare_county_geodataframe`
internally and adjust linewidth. This keeps one function with consistent
behavior rather than a separate function.

**Validation:** Generate a test county-level map using existing `aggregate_demand`
output with `geo_level='county'`. Visual check: counties should be colored,
borders thin, no missing geometries.

### Task 3 — Extract categorical break-even map logic to module

**Problem:** The categorical break-even map in preTARE is implemented inline
with `np.select`, `ListedColormap`, and `BoundaryNorm`. This should be a
reusable function.

**Steps:**
1. Add `assign_breakeven_category(df: pd.DataFrame) -> pd.Series` to
   `thermal_cop.py` (since it operates on break-even COP results). Uses
   `np.select` with the existing 4-category logic.
2. Add `BREAKEVEN_COLORS` and `BREAKEVEN_LABELS` constants to
   `visualize_geospatial_data.py` (or keep with the plotting function).
3. Add a `plot_categorical_breakeven_map` function to
   `visualize_geospatial_data.py` that:
   - Takes `breakeven_results` dict and `selected_mps` list
   - Creates `ListedColormap` + `BoundaryNorm`
   - Generates one panel per MP with shared categorical scale
   - Renders a legend (not a colorbar)
4. Replace the inline preTARE cell with imports and function calls.

**Validation:** preTARE categorical map output should be identical to
pre-edit version.

### Task 4 — Update postTARE to use county-level maps

**Steps:**
1. Change `aggregate_bill_savings` call to pass `geo_level='county'`
2. Change `aggregate_demand` call to pass `geo_level='county'`
3. Update map cells to use county-level plotting (from Task 2)
4. Verify shared norms are computed BEFORE the per-MP loop
5. Bill savings map: `TwoSlopeNorm(vcenter=1.0)`, shared across MPs
6. Demand map: `TwoSlopeNorm(vcenter=0)`, shared across MPs

**Validation:**
- Total homes should be ~331K (all fuels)
- Demand percent changes should be 20–60% range
- Maps should show county-level variation within states

### Task 5 — Cleanup

1. Remove unused imports from postTARE: `create_choropleth_map`, `FUEL_PRICES_PATH`
2. Add `ELEC_TOTAL_COL` to `BASELINE_USECOLS` in `data_loading.py`
3. Remove any lingering `primary_mp` references
4. Verify `fuel_filter` settings are correct per metric:
   - COP: `fuel_filter='Natural Gas'` ✅
   - Demand: `fuel_filter=None` ✅
   - Bill savings: `fuel_filter=None` ✅
5. Grep for any remaining inline `def` statements in both notebooks

## Reference Values (golden)

| Metric | Expected |
|--------|----------|
| Total homes in demand/bill savings | ~331K (all fuels) |
| Demand % change range | +20–60% (gas-heavy states), small negative (electric states) |
| NY demand % change | ~30–50% (NOT 751% or 1,285%) |
| FL demand % change | ~-5–10% (NOT -72%) |
| Categorical map MP3 | Mostly orange/brown (few green states) |
| Categorical map MP4 | Mostly green (HP favorable in most states) |
| Break-even COP values | UNCHANGED from Session 5 |

## Known Anti-Patterns (do NOT suggest)

| Anti-pattern | Why it's wrong |
|-------------|----------------|
| Defining visualization functions inline in notebooks | Creates shadow/divergence problems; use module imports |
| Separate `_by_state` and `_by_county` plotting functions | Use `geo_level` parameter on a single function |
| Thick county borders on choropleth | 3,000+ polygons; use `linewidth=0.1` or suppress |
| Computing `TwoSlopeNorm` inside per-MP loop | Gives each MP its own scale — compute before loop |
| Continuous colormap for break-even | Categorical with 4 bins is policy-readable |
| Using `Normalize` for demand change | Need `TwoSlopeNorm(vcenter=0)` for diverging palette |

## Session Summary Template

1. Files modified with change descriptions
2. `plot_combined_choropleth` layout resolution (which version won)
3. County choropleth plotting — confirmed working
4. Inline functions removed from preTARE — count and names
5. postTARE maps now county-level — confirmed
6. Categorical map extracted to module — confirmed
7. Unused imports cleaned — list
8. Remaining items for Session 7 (adoption potential)
