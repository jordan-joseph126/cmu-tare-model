# Adoption KPIs — Session 7 Plan (v1)

> **Version notes.** New session. Three tasks: (1) derive bill % change from
> ratio to eliminate inconsistency, (2) add adoption potential choropleth map
> at county/state level, (3) update dotplot layout and styling.

## Your Role

Senior Python developer and research software engineer for the TARE model.
You understand ResStock data, matplotlib visualization, geopandas choropleth
rendering, the adoption potential tier framework, and the paper's analytical
structure.

## Project Context

Sessions 3–6 refactored KPI modules, fixed demand columns, added county-level
aggregation, built categorical break-even maps, and consolidated visualization
functions. The postTARE notebook now generates county-level maps for bill
savings ratio, bill % change, and demand % change with dark gray base layers,
state borders, and explicit colorbar ticks.

Two issues remain from Session 6, plus new adoption potential work:

**Bill % change inconsistency.** The bill savings ratio map uses a weighted
median of per-building ratios, while the bill % change map uses
`(total_cost_ratio - 1) × 100` (ratio of aggregate weighted totals). These
produce different spatial patterns even though they should agree. The fix
is to derive % change directly from the median ratio: `(median_ratio - 1) × 100`.

**Adoption potential choropleth.** The paper needs county-level and state-level
maps showing the percentage of homes classified as adopters (Tier 1 + Tier 2).
The per-building adoption tier columns already exist in the TARE output
DataFrames — they just need to be aggregated to county/state and visualized
as monochromatic choropleths, similar to the spark gap and effective annual
COP maps.

**Dotplot updates.** The adoption potential dotplot works well but needs minor
styling and layout changes: national grouping color from gray to black,
2-row × 1-column layout with shared x-axis, and label overlap mitigation.

## Scope Constraint — CRITICAL

**In scope:** Bill % change derivation fix, adoption potential choropleth map
(county and state), dotplot layout/styling changes, integration into postTARE
notebook.

**Out of scope:** Peak load analysis, dual-fuel/cold-climate scenarios,
discount rate sensitivity visualization, changes to TARE model computation,
changes to preTARE notebook.

## What Was Done Before

### Sessions 3–5
- Module split, shared-scale maps, break-even COP, `primary_mp` removal

### Session 6
- Visualization consolidation (inline functions → module imports)
- County-level `plot_combined_choropleth` with `geo_level` parameter
- `cbar_ticks` parameter for explicit colorbar tick control
- Dark gray base layer + state borders for county maps
- Symmetric `Normalize` for colorbars
- Alaska inset removed (CONUS only)
- `aggregate_bill_savings` and `aggregate_demand` with `geo_level='county'`
- Weighted median for bill savings ratio
- `weighted_ratio` renamed → `total_cost_ratio`
- `min_home_count=10` (84% county coverage)

## Attached Files

- `visuals_adoption_dotplot.py` — dotplot module with `prepare_plot_data()`,
  `plot_adoption_panel()`, `_build_legend_handles()`, `FUEL_COLORS`, `GROUPING_ORDER`
- `visuals_adoption_potential.py` — adoption potential module with
  `create_multiIndex_adoption_df()`, `plot_adoption_rate_bar()`
- Notebook cell showing dotplot function call (pasted in user message)

## Current Implementation Status

| Item | Status |
|------|--------|
| Bill savings ratio (weighted median) | ✅ Done |
| Bill % change (total_cost_ratio) | ⚠️ Inconsistent with ratio |
| Adoption potential choropleth | ⬜ Not started |
| Dotplot national color | ⬜ Not started |
| Dotplot 2-row layout | ⬜ Not started |
| Dotplot label overlap | ⬜ Not started |

## Tasks

### Task 1 — Derive bill % change from median ratio

**Problem:** `median_bill_savings_ratio` and `pct_bill_change` show different
spatial patterns. The ratio uses weighted median of per-building ratios.
The % change uses `(total_cost_ratio - 1) × 100` where `total_cost_ratio =
Σ(w × retrofit_cost) / Σ(w × baseline_cost)`.

**Fix:** In `aggregate_bill_savings()` in `bill_savings.py`, compute
`pct_bill_change` from the median ratio instead of from the aggregate:

```python
grouped['pct_bill_change'] = (grouped['median_bill_savings_ratio'] - 1) * 100
```

Remove the `total_cost_ratio` column and its intermediate computation. The
two maps should now show identical spatial patterns: ratio < 1 ↔ negative %,
ratio > 1 ↔ positive %.

**Validation:** Compare MP3 maps side by side — every county should be the
same color on both maps. The % change ticks should correspond exactly:
ratio 0.6 = -40%, ratio 1.0 = 0%, ratio 1.2 = +20%.

### Task 2 — Adoption potential choropleth map

**Goal:** Visualize the percentage of adopters (Tier 1 + Tier 2) at county
and state level as a monochromatic choropleth. One map per MP, shared scale.

**Data source:** The per-building TARE output DataFrames contain adoption
decision columns for each building. These columns classify each building
into one of four tiers. A building is an "adopter" if it is Tier 1 or Tier 2.

**Computation steps:**
1. For each building, determine if it is Tier 1 or Tier 2 (adopter = True)
2. Group by county (using `in.county` from the EUSS baseline), weight by
   `DWELLING_UNIT_WEIGHT`
3. Compute `adoption_rate = Σ(w × adopter) / Σ(w) × 100` for each county
4. Apply `min_home_count` masking (sample count, not weighted)
5. Generate monochromatic choropleth using `plot_combined_choropleth`
   with `geo_level='county'` and a sequential colormap (e.g., `'Greens'`
   or `'YlGn'`)

**Where to put the function:** Add `compute_adoption_rate()` to a new file
or to `bill_savings.py` / an appropriate module. The function should accept
the TARE output DataFrame, the adoption column name, and return a DataFrame
with county-level adoption rates.

**Audit first:** Copilot should search the existing codebase (especially the
peak demand notebook and any files referencing `DEFAULT_ADOPTER_TIERS`) to
identify how adoption tier columns are currently named and where county-level
adoption has been partially implemented.

### Task 3 — Update dotplot styling and layout

**3a. National color:** In `visuals_adoption_dotplot.py`, change
`FUEL_COLORS['National']` from `'#7f7f7f'` (gray) to `'#000000'` (black).

**3b. Layout change:** In the notebook cell that creates the dotplot figure,
change from `1 row × N cols` to `N rows × 1 col` with a shared x-axis:

```python
fig, axes = plt.subplots(
    n_cols, 1,           # was (1, n_cols)
    figsize=(10, 9 * n_cols),  # taller, not wider
    sharex=True,
    sharey=True,
)
```

Adjust `tight_layout` and legend positioning for vertical stacking.

**3c. Label overlap:** After generating the new layout, pause for user
feedback. Known overlap issues: markers at 0% or 100% have annotations
clipped by axis edges; markers close together have overlapping text. Possible
mitigations include adjusting annotation positions, using `adjustText`, or
truncating labels.

### Task 4 — Integrate into postTARE notebook

Add adoption potential choropleth and dotplot cells to the postTARE notebook
after the existing bill savings and demand map cells. Import functions from
the appropriate modules.

## Reference Values (golden)

| Metric | Expected |
|--------|----------|
| Total homes (bill savings / demand) | ~331K |
| County coverage at min_home_count=10 | ~2,601 counties (84%) |
| Bill ratio → % change correspondence | ratio 0.6 = -40%, ratio 1.0 = 0% |
| National adoption rate (MP3, IRA-Ref) | ~55% (T1+T2+T3), ~17% (T1+T2) |
| National adoption rate (MP4, IRA-Ref) | ~60% (T1+T2+T3), ~17% (T1+T2) |

## Known Anti-Patterns

| Anti-pattern | Why it's wrong |
|-------------|----------------|
| Computing % change from `total_cost_ratio` when ratio uses weighted median | Produces inconsistent maps |
| Using `Normalize` for adoption rate choropleth | Use sequential colormap with `Normalize(vmin=0, vmax=100)` |
| Defining dotplot functions inline in notebook | Already modularized; import only |
| Using `RdBu_r` for adoption rate | Monochromatic (Greens, YlGn) is appropriate — there's no diverging center point |
| Auto-computing colorbar ticks | Pass `cbar_ticks` explicitly |

## Session Summary Template

1. Bill % change now derived from median ratio — maps match
2. Adoption potential choropleth working at county/state level
3. Dotplot updated: black national color, 2-row layout
4. Label overlap status (may need iteration)
5. All new code in modules, not inline
6. postTARE notebook updated with new cells
