# TARE Pre-TARE KPI Extension — County Aggregation & Visual Updates (v1.1)

> **Version notes.**
> - **v1** (21 April 2026). Initial plan built from the audit of `calculate_preTARE_am_kpis_sparkGap_COP_20April2026.py`.
> - **v1.1** (21 April 2026). Incorporated user clarifications: (1) Figure 5 is a **dot plot** (2×2 grid of MP × Case panels), not a Shafiee-style stacked bar chart — Task 4 and Appendix B rewritten accordingly; (2) `Already Upgraded` tier removed from `TIER_MARKERS` and `TIER_LABELS_SHORT`; (3) existing-ASHP count comes from `ALLOWED_TECHNOLOGIES['heating']` filter (already excludes heat pumps from baseline eligibility).

---

## Your Role

You are a senior research engineer on the CMU TARE Model project. Your job is to extend the pre-TARE KPI pipeline — which today computes spark gap, thermal COP, break-even COP, and bill impact ratio at the **state** level — to also support **county-level** aggregation, and to update two manuscript visuals to the specifications below. You work to research-code standards (Google/NumPy docstrings, explicit type hints, validation gates) and flag every assumption.

---

## Project Context

The Joseph et al. 2026 paper (target journal: **Energy Policy**, submission late April 2026) uses the CMU TARE model with the EUSS (End-Use Saturation Survey) and ResStock 2025.1 data to produce a technology-differentiated national assessment of residential heat pump adoption. The pre-TARE KPI notebook (`calculate_preTARE_am_kpis_sparkGap_COP_20April2026.py`) is an upstream, non-TARE stage that derives adoption-economics metrics directly from EUSS + EIA fuel prices. These metrics (spark gap, thermal COP, break-even COP, bill impact ratio) are manuscript-ready screening tools and feed two headline Figures:

- **Figure 2b** — Bill impact ratio choropleth for standard and high-efficiency ASHP (diverging color ramp around 1.0).
- **Figure 5** — Adoption potential stacked bar chart by tier and fuel type, side-by-side percentage and millions-of-households view.

County-level aggregation is needed because (a) the Pittsburgh/Allegheny County case study is the primary applied demonstration in Section 4.5, and (b) census-tract-level analysis enabling program targeting (Section 4.3) rolls up to county boundaries in the peak-load pipeline.

---

## Scope Constraint — CRITICAL

**In scope for this session:**
- Extend `compute_thermal_cop`, `compute_breakeven_cop`, `compute_spark_gap_metrics` to accept an `aggregation: Literal['state', 'county']` parameter.
- Add new county-level helpers for data that doesn't decompose cleanly (state→county price broadcast, county geodataframe prep).
- Update the adoption potential Figure (Figure 5) per specs below.
- Update the bill savings ratio map (Figure 2b) per specs below.
- Fix P0 items from the audit that touch code paths the new work modifies (function shadowing, `jenkins_ref` deduplication, PA ranges consolidation).

**Out of scope for this session:**
- Investigating MP4 warm-state COP anomaly (P0.1) — a separate audit task; do not silence the warning.
- Resolving Alaska exclusion (P0.3) — document but don't fix here.
- Investigating PA CZ 6-7 spot-check failure (P0.2) — flag but don't modify benchmark ranges without evidence.
- Rewriting the TARE model itself, or anything in the `cmu_tare_model.core` package.
- Touching ResStock 2025.1 integration (still pending per outline).

**If out-of-scope material appears:** add a `# TODO: investigate in follow-up (P0.X)` marker and continue. Do not expand scope mid-session.

---

## Non-Negotiable Rules

1. **Google/NumPy docstrings on every new or modified function.** Include `Args`, `Returns`, `Raises` sections. Research code without docstrings gets bit-rot fast.
2. **Type hints on all function signatures.** Use `Literal['state', 'county']` for the aggregation parameter — not a plain `str`. This catches typos at static-analysis time.
3. **Validate inputs early, fail clear.** Unknown aggregation values → `ValueError("aggregation must be 'state' or 'county', got {value}")`, not a silent fallback.
4. **Preserve state-level reference values exactly** (see Reference Values table). County aggregation is additive — existing state outputs must not drift by so much as 0.01 on any metric.
5. **No hardcoded constants in function bodies.** Gas heat content, fuel price columns, benchmark ranges, reference dicts all live in `constants.py` or `kpi_functions.py` module-level.
6. **Comments explain WHY, not WHAT.** If a line needs a comment to say what it does, rename variables until the what is obvious and keep the comment for why.
7. **No silent `try/except`.** If you catch an exception, either re-raise with context or log a structured warning. Swallowing errors breaks the audit trail.
8. **Every new function gets at least one smoke-test cell** in the notebook that calls it and verifies the output shape + a reference value.

---

## What Was Done Before

### Prior session 1 — Pre-TARE KPI notebook built (March 2026)
Refactored `compute_thermal_cop` to accept `group_cols` as a list, added zero-baseline-heating filter with validation (Task B), added IECC climate zone aggregation with benchmark validation (Task C), added Jenkins cross-validation (Task D). State-level results frozen against Jenkins reference for FL, PA, MN, MA, CA (AK missing from shapefile).

### Prior session 2 — Audit (21 April 2026)
See `notebook_audit_preTARE_kpis_20April2026.md`. Key findings:
- P0.4: `create_choropleth_map` double-defined (imported + redefined locally). **This blocks safe county visual work** until resolved.
- P1.5–P1.6: redundant frames + hardcoded benchmark ranges. Cleanup before county expansion.
- P0.1–0.3: MP4 warm-state COP, PA CZ 6-7, Alaska — flagged but out of scope for this session.

---

## Attached Files

- `calculate_preTARE_am_kpis_sparkGap_COP_20April2026.py` — the notebook under extension.
- `calculate_preTARE_am_kpis_sparkGap_COP_20April2026.pdf` — matching PDF export with observed outputs.
- `notebook_audit_preTARE_kpis_20April2026.md` — the audit (P0–P3 issue list).
- `cmu_tare_model/adoption_kpis/kpi_functions.py` — functions to extend (`compute_thermal_cop`, `compute_breakeven_cop`, `compute_spark_gap_metrics`, `calculate_price_ratios`).
- `cmu_tare_model/adoption_kpis/visualize_geospatial_data.py` — `prepare_state_geodataframe`, `create_choropleth_map`.
- `cmu_tare_model/constants.py` — `COP_BENCHMARK_RANGES`, `HEATING_LOAD_COL`, fuel-price column maps, `ALLOWED_TECHNOLOGIES` (confirmed to exclude existing ASHP from baseline eligibility — this is the source of the excluded-ASHP count for the Figure 5 subtitle).
- `cmu_tare_model/adoption_potential/data_processing/visuals_adoption_dotplot.py` — module containing `TIER_MARKERS`, `TIER_LABELS_SHORT`, `GROUPING_ORDER`, `prepare_plot_data`, `plot_adoption_panel`, `_build_legend_handles`. This is the file to edit for Figure 5 tier changes.
- `visuals_adoption_dotplot.py` (notebook cell, user-attached) — the outer cell code that orchestrates the 2×2 grid and calls the module functions. This is where font sizes, x-tick spacing, secondary axis, and subtitle changes land.

---

## Current Implementation Status

| Area | Status | Notes |
|---|---|---|
| `compute_thermal_cop` | ✅ state; ⬜ county | Already takes `group_cols`; add `aggregation` wrapper + county-column handling |
| `calculate_price_ratios` | ✅ state; ⬜ county | State-level EIA data; county requires state→county broadcast |
| `compute_breakeven_cop` | ✅ state; ⬜ county | Derived from prices; inherits state/county support |
| `compute_spark_gap_metrics` | ✅ state; ⬜ county | Merges prices + COP; add group-key generalization |
| `prepare_state_geodataframe` | ✅ | Needs sibling `prepare_county_geodataframe` |
| Adoption potential dot plot (Fig 5) | ⚠️ | 2×2 grid exists; surgical edits only — tier cleanup, fonts, x-ticks, subtitle, y-label augmentation |
| Bill savings ratio map (Fig 2b) | ⚠️ | See Appendix C spec |
| `create_choropleth_map` | 🔴 | Double-defined — MUST fix before visual work |

---

## Required First Action

Before editing any code, do all four:

1. Open `calculate_preTARE_am_kpis_sparkGap_COP_20April2026.py`, `kpi_functions.py`, `constants.py`, `visualize_geospatial_data.py`.
2. Read `notebook_audit_preTARE_kpis_20April2026.md` end to end.
3. Read this plan end to end.
4. Grep the codebase for current usages of `group_cols=` and `aggregation=` to check for any existing county conventions (e.g., `in.county`, `county_id`, `GEOID`, `FIPS`) you need to be consistent with. Report what you find before drafting changes.

---

## Tasks

### Task 1 — Orient and review

**Goal.** Ground yourself in the codebase before writing. Build a shared reference frame.

**Steps.**
1. Read the audit, this plan, and the four source files listed above.
2. Grep for `in.county`, `county_id`, `GEOID`, `FIPS`, `puma` in the `cmu_tare_model` package. Determine the canonical county-key column name. If more than one exists, surface the inconsistency.
3. Grep for `group_cols=` usages outside this notebook. If other notebooks rely on the current return-column names, document those call sites — any new columns must be additive, not renames.
4. Confirm the EUSS baseline and upgrade DataFrames have a county column (check `df_baseline.columns` for something like `in.county` or `in.county_fips`).
5. Verify the shapefile used for state maps has a county-level companion (e.g., `tl_2015_county` or equivalent). If not, flag missing data and stop.

**Validation gate.** Produce a short report (5–10 lines) summarizing: canonical county-key column name, number of distinct counties in EUSS, presence/absence of county shapefile, and any naming inconsistencies.

🛑 Present the report and get confirmation before proceeding.

### Task 2 — Iterate the plan based on findings

**Goal.** If Task 1 surfaced anything that invalidates an assumption in this plan, patch the plan before coding.

**Steps.**
1. For each finding in the Task 1 report, decide: does it change scope, reference values, or task order?
2. Propose a plan diff (not a code diff) — added tasks, removed tasks, revised reference values.
3. Get explicit approval on the diff before proceeding.

**Validation gate.** Plan diff approved or explicitly marked "no changes needed."

🛑 Do not proceed to Task 3 without this confirmation.

### Task 3 — Add county aggregation

**Goal.** Extend the KPI functions to support county-level aggregation via an `aggregation: Literal['state', 'county']` parameter, and add new functions where state→county broadcast is required.

#### 3a. Add `aggregation` parameter to existing functions

For each of `compute_thermal_cop`, `compute_breakeven_cop`, `compute_spark_gap_metrics`:

1. Add `aggregation: Literal['state', 'county'] = 'state'` as a new keyword-only argument.
2. Internally, map `aggregation='state'` → `group_cols=['state']` (default behavior — unchanged).
3. Map `aggregation='county'` → `group_cols=[<canonical state column>, <canonical county column>]`. Rationale: counties are unique *within* states, so the state column stays in the grouping to preserve ambiguity-free joins.
4. Validate the input: raise `ValueError` on any other value.
5. Ensure returned DataFrames carry both keys (`state`, `county`) as columns when `aggregation='county'`.

#### 3b. Add county-specific helpers

State EIA prices cannot be meaningfully computed per county (EIA publishes state-level residential rates). So county-level price frames are **state prices broadcast to counties via a state→county crosswalk**:

1. Create `broadcast_prices_to_counties(df_state_prices: pd.DataFrame, county_crosswalk: pd.DataFrame) -> pd.DataFrame`. The crosswalk maps each county to its parent state; merging on state gives every county its state's price. Document clearly that **this is NOT county-specific rate data** — it's a necessary approximation in absence of tariff-level data.
2. Create `prepare_county_geodataframe(gdf_counties: gpd.GeoDataFrame, df_analysis: pd.DataFrame, merge_col: str = 'GEOID') -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]`. Mirror the state version: return `(gdf_all, gdf_conus, gdf_alaska)` in US Albers Equal Area (ESRI:102003).
3. Update `create_choropleth_map` to accept county-level geometries (it already takes arbitrary gdfs, so this is likely a no-op with a metadata kwarg).

#### 3c. Clean up `create_choropleth_map` shadowing (P0.4)

Remove the in-notebook redefinition at line 713. If the module version lacks a `norm` kwarg, add it there. Re-run all state maps; they must render identically.

#### 3d. Consolidate duplicated constants (P1.5, P2.7–8)

1. Move `jenkins_ref` dict to `constants.JENKINS_BREAKEVEN_REF_90`.
2. Move inline PA ranges (line 368) into `COP_BENCHMARK_RANGES` keyed by `(state, cz_group, mp_key)`.
3. Remove the duplicate `jenkins_ref` at Step 4c.

**Validation gate.** Re-run the notebook with `aggregation='state'` (default); ALL reference values below must match exactly. Then re-run with `aggregation='county'` for PA only; confirm Allegheny County (FIPS 42003) appears with a plausible thermal COP (~1.8–2.5 range for MP3).

🛑 Present state-level re-run output side-by-side with frozen reference values. Do not proceed until match is confirmed.

### Task 4 — Update adoption potential dot plot (Figure 5)

**Goal.** Make targeted, minimal edits to the existing 2×2 dot plot (MP3/MP4 rows × Case A Heating-Only / Case B Heating-and-Cooling columns). This is a **dot plot**, not a stacked bar chart — the Shafiee & Schrag reference in v1 of this plan was a misread and has been superseded.

**Where the code lives.**
- Module: `cmu_tare_model/adoption_potential/data_processing/visuals_adoption_dotplot.py` — contains `TIER_MARKERS`, `TIER_LABELS_SHORT`, `GROUPING_ORDER`, `prepare_plot_data`, `plot_adoption_panel`, `_build_legend_handles`.
- Outer cell (user-attached): orchestrates the 2×2 `plt.subplots` grid, calls `plot_adoption_panel` per panel, sets figure-level legend and title, saves PNG + PDF.

**What the dots encode.**
- X-axis: adoption percentage (% of eligible homes within each fuel row).
- Y-axis: fuel-type categories (`GROUPING_ORDER`, reversed).
- Marker shape: adoption tier (circle = T1 Feasible, triangle = T1+T2 Total Adoption Potential, diamond = T1+T2+T3 With Subsidy). **The `Already Upgraded` square marker is being removed (see 4a below).**
- Marker position: IRA-Reference value.
- Delta annotation `(+X.X)`: change from Pre-IRA → IRA-Reference, shown next to each IRA-Reference marker. **Keep this convention.**

#### 4a. Remove 'Already Upgraded' from tier dictionaries

Edit `visuals_adoption_dotplot.py` (module):

1. Remove the `'Already Upgraded': 's'` entry from `TIER_MARKERS`.
2. Remove the `'Already Upgraded': 'Already Upgraded'` entry from `TIER_LABELS_SHORT`.
3. Verify `plot_adoption_panel` iterates over `TIER_MARKERS.items()` (or equivalent dict traversal) so deletion propagates cleanly. If the function explicitly references the string `'Already Upgraded'` anywhere, that reference must be removed too.
4. Verify `_build_legend_handles` only produces handles for keys that remain in `TIER_MARKERS`.

**Rationale.** The outline (Section 4.2) specifies that existing heat pump homes should be excluded from the adoption-potential framing entirely — they're not candidates for adoption, they're already adopters. The cleaner visual choice is to drop the category and state the count in a subtitle (see 4d).

#### 4b. Add absolute-millions information to the Y-axis labels

The outer cell currently shows only percentages on the x-axis. To satisfy "absolute units (millions of households), not just percentages" for a dot plot (where the y-axis is categorical, not numeric), augment each fuel-type label with the eligible-home count:

```
Natural Gas (62.3M)
Electric Resistance (8.4M)
Fuel Oil (4.1M)
...
```

Implementation:

1. In the outer cell, compute per-fuel eligible-home counts from the baseline ResStock sample using the `ALLOWED_TECHNOLOGIES['heating']` list — sum the weighted home counts per `in.heating_fuel` for rows where `in.hvac_heating_type_and_fuel` is in the allowed list.
2. Pass the per-fuel count dict into `plot_adoption_panel` (add a new optional kwarg `fuel_counts_millions: Optional[Dict[str, float]] = None`) so the function can produce `f"{fuel} ({count:.1f}M)"` y-tick labels.
3. Fallback: if `fuel_counts_millions` is `None`, keep existing label behavior (avoids breaking other callers).

**Why this approach.** A secondary x-axis showing millions doesn't work cleanly for a dot plot because each fuel row has a different 100%-equivalent absolute count. Y-tick-label augmentation gives the reader the denominator they need to convert any percentage on the chart to an absolute count themselves.

#### 4c. Increase font sizes in the outer cell

Current values → new values:

| Element | Current | New |
|---|---|---|
| `fig.suptitle` | `fontsize=13` | `fontsize=18`, `fontweight='bold'` |
| Panel title | `fontsize=11` | `fontsize=16`, `fontweight='bold'` |
| Axis labels (x and y) | default | `fontsize=16` |
| Tick labels | default | `fontsize=14` |
| Legend | `fontsize=9` | `fontsize=14` |
| "No data" annotation | `fontsize=10` | `fontsize=13` |

These changes live in the outer cell (and in `plot_adoption_panel` if it sets font sizes internally — audit the function signature and body in Task 1 to confirm where each setting lives).

#### 4d. Reduce x-axis ticks to 0/20/40/60/80/100

In the outer cell, line `ax.set_xticks(range(0, 101, 10))` (appears in the "No data" guard branch). Change to `ax.set_xticks(range(0, 101, 20))`. Also set this in the data-panel branch if `plot_adoption_panel` sets x-ticks internally — check and update the module if so. Rationale: the delta annotations and marker-position labels already encode numeric values; denser ticks are visual noise.

#### 4e. Add subtitle stating excluded-ASHP count

Currently `fig.suptitle` reads:

```
Heat Pump Adoption Potential — Case A vs. Case B
Discount Rate: {discount_rate} | Cost Scenario: {cost_scenario}
```

Change to:

```
Heat Pump Adoption Potential — Case A vs. Case B
Discount Rate: {discount_rate} | Cost Scenario: {cost_scenario}
Excludes {n_existing_ashp:.1f}M homes with existing ASHP systems
```

The excluded-ASHP count comes from summing ResStock weighted homes where the baseline heating category is `'Electricity ASHP'` (or whichever enumeration value represents existing heat pumps — confirm in Task 1 via the `in.hvac_heating_type_and_fuel` column). Compute once, before constructing the title string.

#### 4f. Export at 300 DPI PDF

Current code saves both PNG and PDF at `dpi=600`. Update to:

- PDF at 300 DPI (Energy Policy standard) — keep as primary submission artifact.
- PNG at 300 DPI — kept as preview.

Change the output directory from `./figures` to the project `figures/` directory (confirm path in Task 1). New file name: `figure5_adoption_dotplot_caseAB_{location_id}.pdf`.

**Validation gate.**

1. Render the updated figure. Inspect visually: markers should be circle / triangle / diamond only (no squares); y-labels should read `"Natural Gas (XX.XM)"` etc.; x-ticks at 0/20/40/60/80/100.
2. Compute the sum of eligible-home counts across all fuel rows; this should match the total eligible population excluding existing ASHP. The excluded-ASHP count in the subtitle, added to this sum, should equal the total `ALLOWED_TECHNOLOGIES['heating']` homes in the weighted baseline.
3. Report: updated figure path, excluded-ASHP count, per-fuel millions-of-households dict.

🛑 Present the rendered figure and the computed counts before committing.

### Task 5 — Update bill savings ratio map

**Goal.** Produce a two-panel (min-efficiency / high-efficiency) bill savings ratio choropleth per the specs below.

**Spec.**

1. **Shared color scale across both panels.** Compute `vmin, vmax` from the combined values; use `TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)` with diverging palette (`RdBu_r` — blue = bills savings, red = bills increase). The `1.0` center is the break-even point for the bill impact ratio (< 1 → savings, > 1 → increase).
2. **Metadata banner on both maps.** Include:
   - Sample size (n = X,XXX homes)
   - Efficiency tier (Min-Efficiency: 15 SEER1 ducted ASHP / High-Efficiency: 24 SEER1 ducted or 29.3 SEER1 ductless)
   - Ducted/ductless assumption used in the COP computation
   - Fuel price year (2024 EIA nominal)
3. **Title dates must read 2024**, not 2022. (Legacy labels exist in older notebook outputs; verify no "(2022)" string remains in titles or captions.)

**Validation gate.** Render both panels. Verify for PA specifically: min-efficiency ratio ≈ 1.28 (bills increase, red), high-efficiency ratio ≈ 0.90 (bill savings, blue). These are outline-cited values; mismatch is a P0.

🛑 Present both panels before committing.

---

## Reference Values (golden)

These values must hold EXACTLY after any refactor. If any drift by > 0.01, the refactor has broken state-level behavior.

### Spark gap (2024 EIA nominal, state-level)

| Metric | Value |
|---|---|
| National mean | 3.55 |
| National median | 3.41 |
| Min (FL) | 1.70 |
| Max (AK) | 6.44 |
| N states | 51 |

### Thermal COP (MP3, Natural Gas homes, state-level)

| State | Thermal COP | Home count |
|---|---|---|
| CA | 3.055 | 24,571 |
| FL | 2.893 | 1,084 |
| PA | 2.020 | 8,594 |
| MN | 1.590 | 4,687 |
| ND | 1.527 | 402 |

### Thermal COP (MP4, Natural Gas homes, state-level) — flagged

| State | Thermal COP | Flag |
|---|---|---|
| CA | 5.720 | ⚠️ exceeds literature ceiling |
| FL | 5.697 | ⚠️ exceeds literature ceiling |
| PA | 3.342 | within expected range |
| MN | 2.360 | within expected range |
| ND | 2.149 | within expected range |

### Break-even COP @ 90% AFUE (2024 prices)

| State | BE @90% |
|---|---|
| CA | 4.57 |
| FL | 1.53 |
| PA | 3.57 |
| MN | 3.97 |
| MI | 4.91 |

### Bill impact ratio (outline-cited)

| State / Tech | Ratio | Interpretation |
|---|---|---|
| PA — 15 SEER1 standard ASHP | 1.28 | bills increase |
| PA — 24 SEER1 high-efficiency ASHP | 0.90 | bill savings |

### Adoption totals (outline)

| Scenario | Total adoption |
|---|---|
| S1 + S2 combined | 13.3–13.7M homes |
| Electric-heated subpopulation | 9.8–10.7M homes |
| Natural gas subpopulation (with IRA) | 0.2–0.4M homes |

---

## Code Standards

- **Docstrings:** Google/NumPy style on every function. Sections: one-line summary, longer description, `Args`, `Returns`, `Raises`.
- **Type hints:** required on all signatures. Use `Literal`, `Optional`, `Union`, `Tuple`, `Dict`, `List` from `typing`. Use `from __future__ import annotations` at top of module for forward refs.
- **Error handling:** validate inputs at function entry; raise `ValueError`/`TypeError` with a message that names the parameter and the received value.
- **Comments:** explain WHY. Stale what-comments get deleted.
- **Imports:** top-of-module only. No `import` statements inside cells.
- **Naming:** `aggregation` parameter (not `level` or `agg`); `county` not `cnty`; match the canonical column name found in Task 1.
- **Cell-level print gates:** every step ends with a clearly-labeled "✓ STEP N COMPLETE" or equivalent. Replace Unicode glyphs (`✓ ⚠ ✗`) with ASCII (`[OK] [WARN] [FAIL]`) for terminal portability.
- **File paths:** use `pathlib.Path` for new code; legacy `os.path.join` may remain in untouched code paths.

---

## Known Anti-Patterns (do NOT suggest)

| ❌ Wrong thing | Why it's wrong |
|---|---|
| Adding a separate `compute_thermal_cop_by_county` that duplicates `compute_thermal_cop` logic | The function already takes `group_cols`; duplication violates DRY. Use the `aggregation` parameter pattern. |
| Treating county EIA prices as a data source | EIA publishes state-level residential rates; county tariffs are not public. Use state→county broadcast with explicit documentation. |
| Silencing the MP4 warm-state COP warning | The warning is informative; the underlying issue is a real audit finding (P0.1). Leave it visible. |
| "Fixing" the PA CZ 6-7 benchmark range to make it pass | The benchmark may be wrong OR the data may be off. Fixing ranges without evidence hides the real issue. |
| Using `dict` for the aggregation parameter instead of `Literal` | `Literal['state', 'county']` gives static-analysis safety. `str` does not. |
| Removing Alaska from the Jenkins reference to avoid the N/A line | Hides a real data-coverage problem. Leave AK in and log the skip. |
| Using `display()` in new code | Notebook-only; breaks `.py` script runs. Use `print(df.to_string())`. |
| Rewriting `visuals_adoption_dotplot.py` from scratch to "clean it up" | Task 4 is surgical — 6 targeted edits. A rewrite introduces drift risk and invalidates the stable dot plot layout. |
| Replacing the dot plot with a stacked bar chart | The user explicitly confirmed the dot plot format. The v1 Shafiee reference was a misread. |
| Changing break-even COP formula from `spark_gap × AFUE` | This is the canonical definition from Jenkins. Do not re-derive. |
| Running `calculate_price_ratios` multiple times per cell | Cache once; reuse. |

---

## Appendix A — State → county price broadcast design

Because EIA does not publish county-level residential fuel prices, county-level spark gap and bill impact ratio metrics must use state-level prices broadcast to counties. The design decision: **every county inherits its parent state's residential electricity and natural gas rate.** This introduces a known limitation:

- Counties in the same state share identical spark gap values.
- Intra-state price variation (different utilities, municipal rates) is not captured.
- Bill impact ratio variation within a state comes entirely from COP variation (climate, ResStock sample), not price variation.

Document this prominently in the paper's Methods section and in the function's docstring. Future work could integrate utility-level rate data from EIA-861 to decompose within-state variation.

**Crosswalk table:** Use the 5-digit FIPS county code (2-digit state + 3-digit county). Canonical source: `cmu_tare_model/data/geo/county_to_state_crosswalk.csv` (confirm path in Task 1; create if missing from Census TIGER 2015 shapefile).

---

## Appendix B — Adoption potential dot plot: detailed spec

**Not a stacked bar chart.** The v1 version of this plan referenced Shafiee & Schrag 2026 Fig. 3 as the template. That was a misread. The actual figure is a **2×2 grid of dot plots** already built in `cmu_tare_model/adoption_potential/data_processing/visuals_adoption_dotplot.py` and orchestrated by an outer notebook cell. Changes to this visual are surgical edits to existing code, not a rewrite.

### Grid layout

```
                Case A: Heating Only      Case B: Heating & Cooling
MP3 (Min-Eff):  [dot plot panel]          [dot plot panel]
MP4 (High-Eff): [dot plot panel]          [dot plot panel]
```

Each panel:
- Y-axis: fuel-type categories from `GROUPING_ORDER` (reversed so Natural Gas is on top).
- X-axis: adoption percentage (0–100%).
- One row per fuel type; multiple markers per row (one per tier).

### Marker encoding (after 4a changes)

| Tier | Marker | Short label | Notes |
|---|---|---|---|
| T1: Feasible | circle (`'o'`) | T1: Feasible | Cost-positive without any rebates |
| T1+T2: Total Adoption Potential | triangle (`'^'`) | T1+T2: Adoption Potential | Cost-positive with IRA-Reference rebates |
| T1+T2+T3: With Subsidy | diamond (`'D'`) | T1+T2+T3: With Subsidy | Additional subsidy above IRA |
| ~~Already Upgraded~~ | ~~square~~ | ~~Already Upgraded~~ | **REMOVED in 4a** |

Marker position = IRA-Reference adoption percentage. Delta annotation `(+X.X)` next to each marker = change from Pre-IRA → IRA-Reference. **Keep both conventions.**

### Label augmentation (4b)

Y-tick labels get the eligible-home count appended:

```
Natural Gas (62.3M)
Electric Resistance (8.4M)
Fuel Oil (4.1M)
Propane (3.2M)
```

Counts come from `ALLOWED_TECHNOLOGIES['heating']` filtered rows in the baseline, weighted by `weight`.

### Typography (4c)

- Figure suptitle: 18pt bold.
- Panel titles: 16pt bold.
- Axis labels: 16pt.
- Tick labels (both axes): 14pt.
- Legend: 14pt.
- "No data" placeholder annotation: 13pt.

### X-axis ticks (4d)

Change `range(0, 101, 10)` → `range(0, 101, 20)`. Final ticks: 0, 20, 40, 60, 80, 100.

### Subtitle (4e)

Three-line `fig.suptitle`:

```
Heat Pump Adoption Potential — Case A vs. Case B
Discount Rate: {discount_rate} | Cost Scenario: {cost_scenario}
Excludes {n_existing_ashp:.1f}M homes with existing ASHP systems
```

Compute `n_existing_ashp` from weighted homes where `in.hvac_heating_type_and_fuel == 'Electricity ASHP'` (verify exact enumeration string in Task 1).

### Export (4f)

- Primary: `figure5_adoption_dotplot_caseAB_{location_id}.pdf` at 300 DPI.
- Preview: same filename, `.png`, at 300 DPI.
- Location: project `figures/` directory (path confirmed in Task 1).

### What NOT to change

- The 2×2 grid layout (MPs × Cases).
- The LMI income filter (`income_groups=['LMI']` in the `prepare_plot_data` call).
- The delta-in-parentheses annotation convention.
- The "No data" guard branch for panels where `mi_df is None`.
- `GROUPING_ORDER` contents or ordering.
- `prepare_plot_data` logic.
- Case A / Case B distinction.

---

## Appendix C — Bill savings ratio map: detailed spec

Metric definition (outline, Section 1.3):

```
Bill Impact Ratio = Spark Gap / Efficiency Ratio
Efficiency Ratio = Effective Annual COP / Gas AFUE
```

< 1 → bills decrease with electrification (heat pump wins).
> 1 → bills increase with electrification (furnace wins).

Panel layout: two maps side-by-side (or stacked), titled:
- (A) Min-Efficiency ASHP (15 SEER1, ducted) — Bill Impact Ratio by State, 2024.
- (B) High-Efficiency ASHP (24 SEER1 ducted / 29.3 SEER1 ductless) — Bill Impact Ratio by State, 2024.

Color scale (shared across both panels):
- Palette: `RdBu_r` (reversed so blue = savings, red = increase).
- Normalization: `mcolors.TwoSlopeNorm(vmin=combined_min, vcenter=1.0, vmax=combined_max)`. Centering at 1.0 is the crux — it makes the interpretation visually instant.
- Colorbar label: "Bill Impact Ratio (electrification cost / baseline cost)".
- Colorbar ticks: include 1.0 explicitly with the label "break-even".

Metadata banner (below each panel title, above the map):
- Font size: 10 pt (small but legible).
- Content:
  - `n = XX,XXX homes (Natural Gas baseline)` — the home count from the COP computation.
  - `Efficiency tier: [Min-Efficiency 15 SEER1 / High-Efficiency 24 SEER1 ducted — 29.3 SEER1 ductless]`.
  - `Baseline AFUE: 0.80 (data-derived mean; state-level range 0.75–0.82)`.
  - `Fuel prices: 2024 EIA nominal, state residential`.
- Use a single multi-line annotation anchored below the title with `ha='left', va='top'`.

Ducted/ductless handling:
- For MP3 (standard ASHP): ducted only. State in banner.
- For MP4 (high-efficiency): mixed fleet per EUSS measure package. State the assumption explicitly in the banner.

Title year:
- Must read "(2024)" — not "(2022)". Audit all title strings in the notebook for legacy "(2022)" labels and fix.

---

## Session Summary Template

After Task 5 completes, produce a summary covering:

1. Task 1 orientation findings (county-key column name, county count, shapefile availability).
2. Plan diffs applied in Task 2 (if any).
3. State-level reference-value parity check results (Task 3).
4. County-level smoke-test results (Allegheny County PA).
5. Figure 5 rendered output — file path, size, key values.
6. Figure 2b rendered output — file paths, min/max bill impact ratio, PA specifics.
7. Any P0/P1 items from the audit that surfaced during implementation and were handled, deferred, or need escalation.
8. Files modified — full list with before/after line counts.
9. New files created — with purpose.
10. Open follow-ups (P0.1, P0.2, P0.3 still pending from the audit).
