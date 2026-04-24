# Adoption KPIs — Session 5: Multi-MP Shared-Scale Visualization Fix

> **Version notes.** Session 5. Fixes P0 bugs from Session 3/4 refactoring
> where multi-MP map generation and shared color scales were lost.

## Your Role

Senior Python developer and research software engineer. You understand
matplotlib color normalization (`Normalize`, `TwoSlopeNorm`), geopandas
choropleth rendering, and the TARE model's adoption KPI pipeline.

## Project Context

Sessions 3–4 successfully extracted all KPI functions into dedicated
modules (`spark_gap.py`, `thermal_cop.py`, `demand.py`, `bill_savings.py`).
Both notebooks run end-to-end and produce correct numerical results.
However, the refactoring introduced visualization regressions:

1. **preTARE:** only generates maps for MP3 (the first selected MP). MP4
   COP, break-even COP, and comparison table are never computed/displayed.
2. **Both notebooks:** maps for different MPs use independent color scales,
   making visual comparison impossible.

The fix pattern: compute metrics for ALL selected MPs, find the global
min/max across MPs for each metric, then render maps with a shared norm.

## What Was Done Before

### Session 3 (April 24, 2026)
- Split `kpi_functions.py` into `data_loading.py`, `spark_gap.py`, `thermal_cop.py`
- Cleaned up preTARE notebook — but introduced single-MP regression for maps

### Session 4 (April 24, 2026)
- Created `demand.py` and `bill_savings.py`
- Cleaned up postTARE notebook — but maps use per-MP independent scales

### This session (Session 5)
- Fix preTARE to compute and visualize all selected MPs
- Fix both notebooks to use shared color scales across MPs
- Remove the `primary_mp` vestige from preTARE

## Scope Constraint — CRITICAL

**In scope:**
- preTARE notebook: Steps 4–5 (break-even COP, maps) and display section
- postTARE notebook: maps cell only
- Both: the looping/normalization pattern for shared scales

**Out of scope:**
- Module code (`spark_gap.py`, `thermal_cop.py`, etc.) — these are correct
- Bill savings ratio computation — correct, just needs shared-scale maps
- Demand computation — correct, just needs shared-scale maps
- Climate zone validation section in preTARE — already loops over MPs correctly
- Step 1–3 of preTARE — already correct (COP computed for all MPs)

## Attached Files

- `calculate_preTARE_am_kpis_sparkGap_COP_24April2026.py` — preTARE notebook
- `calculate_preTARE_am_kpis_sparkGap_COP_24April2026.pdf` — preTARE output
- `calculate_postTARE_am_kpis_demand_bill_savings_24April2026.py` — postTARE notebook
- `calculate_postTARE_am_kpis_demand_bill_savings_24April2026.pdf` — postTARE output

## Root Cause Analysis

### preTARE — why only MP3 maps appear

```python
# Line 108-110: sets a single primary_mp
primary_mp = selected_mps[0]   # always MP3
df_cop = cop_results[primary_mp]  # only MP3 COP data

# Line 214: break-even only for MP3
df_breakeven = compute_breakeven_cop(df_spark_gap, df_cop)

# Lines 237-241: map merge only includes MP3 COP/break-even
df_map = df_spark_gap.merge(
    df_cop[['state', 'thermal_cop', 'baseline_afue']], on='state'
).merge(
    df_breakeven[['state', 'breakeven_cop_90']], on='state'
)

# Lines 260-278: loop over METRICS, not MPs
for column, title, cmap, cbar, fname in [...]:
    create_choropleth_map(...)  # generates one map per metric, always MP3
```

The `cop_results` dict correctly contains both MPs (computed in Step 3).
But `df_cop`, `df_breakeven`, `df_map`, and the maps loop all reference
only `primary_mp`. The fix: compute break-even for each MP, then loop
over MPs within the map generation.

### Both notebooks — why scales don't match

Each map call computes its own `vmin`/`vmax` from the data for that single
MP. When MP3 COP ranges 1.53–3.05 and MP4 ranges 2.15–5.72, the same
shade of green means different COP values. The fix: compute global
min/max across all MPs before any map call, pass a shared `norm`.

## Tasks

### Task 1 — Fix preTARE: compute break-even for all MPs

**Current (broken):**
```python
df_breakeven = compute_breakeven_cop(df_spark_gap, df_cop)
```

**Fix:** Compute break-even for each MP, store in a dict:
```python
breakeven_results = {}
for mp in selected_mps:
    breakeven_results[mp] = compute_breakeven_cop(
        df_spark_gap, cop_results[mp]
    )
```

Remove the `primary_mp` and `df_cop` single-MP variables from Steps 4–5.
Keep `primary_mp` ONLY in Step 3 output line (harmless informational print).

### Task 2 — Fix preTARE: shared-scale maps for all MPs

**Pattern:** For each metric that varies by MP (thermal_cop, breakeven_cop_90),
compute global min/max across all MPs, then render one map per MP with a
shared norm.

Spark gap is MP-independent (only depends on fuel prices), so it gets
one map with its own scale.

```python
# Spark gap — single map (MP-independent)
create_choropleth_map(..., column='spark_gap', ...)

# COP maps — shared scale across MPs
cop_vals = pd.concat([
    cop_results[mp]['thermal_cop'] for mp in selected_mps
])
cop_norm = mcolors.Normalize(vmin=cop_vals.min(), vmax=cop_vals.max())

for mp in selected_mps:
    # merge this MP's COP into geodataframe
    df_map_mp = df_spark_gap.merge(
        cop_results[mp][['state', 'thermal_cop', 'baseline_afue']], on='state'
    )
    _, gdf_conus_mp, gdf_alaska_mp = prepare_state_geodataframe(
        gdf_states_raw, df_map_mp, merge_col='state'
    )
    create_choropleth_map(
        gdf_conus_mp, gdf_alaska_mp,
        column='thermal_cop',
        title=f'Heat Pump Thermal COP by State (MP{mp}, 2024)',
        norm=cop_norm, ...
    )

# Break-even maps — shared scale across MPs
be_vals = pd.concat([
    breakeven_results[mp]['breakeven_cop_90'] for mp in selected_mps
])
be_norm = mcolors.Normalize(vmin=be_vals.min(), vmax=be_vals.max())

for mp in selected_mps:
    ...  # same pattern
```

### Task 3 — Fix preTARE: display tables for all MPs

The comparison table at the end (lines 291–315) currently only shows MP3.
Loop over `selected_mps` and produce one comparison table per MP.

### Task 4 — Fix postTARE: shared-scale bill savings maps

**Current (broken):** Each MP computes its own `r_min`/`r_max` and
`TwoSlopeNorm` independently.

**Fix:** First pass computes global min/max, then second pass renders:

```python
# First pass: global min/max for bill savings ratio
all_ratio_vals = pd.concat([
    bill_savings_results[mp]['median_bill_savings_ratio']
    for mp in selected_mps
]).dropna()
r_min = min(all_ratio_vals.min(), 0.999)
r_max = max(all_ratio_vals.max(), 1.001)
shared_ratio_norm = mcolors.TwoSlopeNorm(vmin=r_min, vcenter=1.0, vmax=r_max)

# First pass: global min/max for demand change
all_demand_vals = pd.concat([
    demand_results[mp]['elec_change_gwh'] for mp in selected_mps
]).dropna()
d_max = max(abs(all_demand_vals.min()), abs(all_demand_vals.max()))
shared_demand_norm = mcolors.TwoSlopeNorm(
    vmin=-d_max, vcenter=0, vmax=d_max
)

# Second pass: render all maps with shared norms
for mp in selected_mps:
    create_choropleth_map(..., norm=shared_ratio_norm, ...)
    create_choropleth_map(..., norm=shared_demand_norm, ...)
```

### Task 5 — Clean up preTARE: remove primary_mp vestige

- Remove `primary_mp = selected_mps[0]` from Step 3
- Remove `df_cop = cop_results[primary_mp]`
- Move `from cmu_tare_model.constants import JENKINS_BREAKEVEN_REF_90`
  to the top import block
- Remove `calculate_spark_gap` dead import from postTARE (if present)

## Reference Values (from PDF, 2024 prices)

| Metric | MP3 | MP4 |
|--------|-----|-----|
| COP mean | 2.21 | 3.89 |
| COP range | 1.53–3.05 | 2.15–5.72 |
| Break-even COP @90% mean | 3.15 | (should be same — price-derived) |
| Bill savings median (national) | 0.771 | 0.520 |
| Bill savings states < 1.0 | 39/49 | 49/49 |
| Demand elec change (GWh) | +451,076 | +186,791 |

## Known Anti-Patterns

| Anti-pattern | Why it's wrong |
|-------------|----------------|
| Using `primary_mp` for anything beyond informational print | Creates single-MP regression |
| Computing norm inside the per-MP loop | Gives each MP its own scale |
| Using `Normalize` for bill savings ratio | Need `TwoSlopeNorm` centered at 1.0 |
| Using `Normalize` for demand change | Need `TwoSlopeNorm` centered at 0 |
| Skipping spark gap shared scale | Spark gap is MP-independent; one map is correct |
| Generating separate map cells per MP | Use one cell with a loop and shared norm |
