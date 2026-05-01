# Post-TARE Peak Demand Notebook — Refactoring Implementation Plan (v1.0)

> **Version notes.** v1.0 — Initial plan based on cross-audit of 30Apr2026 (current) vs.
> 21Apr2026_MP3_MP4 (older, with multi-MP fixes). Covers the full refactoring session
> including multi-MP loop restoration, visualization implementation, and cleanup.

---

## Your Role

You are a research software engineer embedded in the Joseph et al. 2026 (Energy Policy) project.
You are refactoring a Jupyter notebook that computes county-level peak electricity demand
changes under residential heat pump adoption scenarios. Your work must be numerically
reproducible — every output you produce must be verifiable against the golden values table.

---

## Project Context

The TARE model assigns ~330K U.S. residential buildings to heat pump adoption tiers (1–4)
based on private NPV under two measure packages (MPs):

- **MP3:** Standard ducted ASHP (15 SEER1)
- **MP4:** High-efficiency ASHP (24 SEER1 ducted / 29.3 SEER1 ductless)

This notebook queries the AWS-hosted ResStock EUSS 2022.1.1 timeseries via BuildStockQuery
(BSQ) to compute hourly electricity demand profiles for:
- **(a) 100% adoption counterfactual** — all filtered buildings retrofit
- **(b) Economically-constrained adoption** — only Tier 1 + Tier 2 adopters

Primary test case: **Allegheny County, PA (FIPS 42003)**. Validate here before national scaling.

The canonical reference for multi-MP behavior is `calculate_postTARE_ts_aws_peak_demand_21April2026_MP3_MP4.py/.pdf`.
The target file to edit is `calculate_postTARE_ts_aws_peak_demand.ipynb`.

---

## Scope Constraint — CRITICAL

**Do NOT edit:**
- `calculate_postTARE_ts_aws_peak_demand_30April2026.py` (reference only)
- `calculate_postTARE_ts_aws_peak_demand_21April2026_MP3_MP4.py` (reference only)
- Any `.pdf` files
- Any files in `cmu_tare_model/` package (modules are read-only during this session)

**All edits go to:** `calculate_postTARE_ts_aws_peak_demand.ipynb` only.

---

## No Hardcoded Weights — Non-Negotiable

BSQ applies `SUM(enduse × weight)` internally in generated SQL. The weight (242.131013) is
already embedded in all kWh values returned by BSQ. **Never multiply by the weight again.**
The only conversion needed is `÷ 1000` to convert kWh to MW at the county aggregate level.

This error appeared in a prior version (240.0 hardcode, 0.9% error) and has been removed.
Do not reintroduce it in any form — not in visualization code, not in comments.

---

## What Was Done Before

### Prior session (30Apr2026 — reference version)
- Confirmed BSQ connection (AWS credentials valid, region us-west-2)
- Step 3 county geography working (3,235 counties, TIGER/Line 2025 shapefile)
- Steps 5–7 working for **MP3 only** (primary_mp regression introduced)
- Weight corrected from 240.0 to 242.131013
- Import paths updated from `kpi_functions` to `adoption_kpis` / `adoption_kpis.data_loading`
- Steps 8–10 remain as `NotImplementedError` stubs

### Prior session (21Apr2026 — multi-MP reference)
- Full multi-MP loop implemented in Steps 4, 6, 7
- Correct `adopter_ids_by_mp`, `df_ts_upgrade_allegheny_by_mp`, `peak_results_allegheny_by_mp` structures
- Both MP3 and MP4 peak profiles validated for Allegheny County
- Golden values confirmed (see table below)

---

## Attached Reference Files

| File | Purpose |
|------|---------|
| `calculate_postTARE_ts_aws_peak_demand_30April2026.py` | Current state of target notebook (as .py export) |
| `calculate_postTARE_ts_aws_peak_demand_30April2026.pdf` | 30Apr runtime outputs for Steps 0–7 (MP3 only) |
| `calculate_postTARE_ts_aws_peak_demand_21April2026_MP3_MP4.py` | Multi-MP reference implementation |
| `calculate_postTARE_ts_aws_peak_demand_21April2026_MP3_MP4.pdf` | Multi-MP runtime outputs (golden values) |

---

## Current Implementation Status

| Step | Status | Time (PDF) | Notes |
|------|--------|-----------|-------|
| 0 | ✅ | — | Imports correct for 30Apr module structure |
| 0b | ✅ | — | `selected_mps = [3, 4]` |
| 0c | ✅ | — | Loads both MPs |
| 1 | ✅ | — | BSQ initialized |
| 2 | ✅ | — | Constants printed |
| 3 | ✅ | — | Shapefile loaded, Allegheny confirmed |
| 4 | 🔴 | — | Regression: single-MP only; must port 21Apr loop |
| 5 | ⚠️ | 396 s | Uses wrong bldg_id source; fix after Step 4 |
| 6 | 🔴 | 266 s | Single MP only; no `_by_mp` dict |
| 7 | 🔴 | — | Wrong variable names; no `_by_mp` storage |
| 8 | ❌ | — | Stub — NOT in scope for this session |
| 9 | ❌ | — | Stub — NOT in scope for this session |
| 10 | ❌ | — | Stub — NOT in scope for this session |

---

## Required First Action

Before writing any code, read the full contents of:
1. `calculate_postTARE_ts_aws_peak_demand_21April2026_MP3_MP4.py` — Steps 4, 5, 6, 7
2. The golden values table below

Do NOT start editing the notebook until you have verified which cells correspond to each step.

---

## Tasks

### Task 1 — Port Step 4: Multi-MP Adopter ID Loop

**Goal:** Replace the single-MP `primary_mp` logic with the multi-MP loop from the 21Apr reference.

**Steps:**
1. Remove the `primary_mp: int = selected_mps[0]` line entirely.
2. Replace the Step 4 cell body with the loop from 21Apr Step 4 (lines 384–451).
3. The output variables must be:
   - `adopter_ids_by_mp: dict[int, dict[str, dict[str, list[int]]]]`
   - `adoption_col_by_mp: dict[int, str]`
4. Retain the Allegheny County test print block inside the loop.
5. Final print: `✓ Step 4 COMPLETE — adopter_ids_by_mp.keys() = [3, 4]`

**Validation gate:**
- `list(adopter_ids_by_mp.keys())` → `[3, 4]`
- MP3 Allegheny: Tier 1=64, Tier 2=29, Constrained=93, All filtered=1,610
- MP4 Allegheny: Tier 1=190, Tier 2=328, Constrained=518, All filtered=1,610

---

### Task 2 — Fix Step 5: bldg_id Union Across MPs

**Goal:** Replace the single-MP bldg_id source with a union across all selected MPs.

**Steps:**
1. Replace line:
   ```python
   allegheny_bldg_ids: list[int] = adopter_ids_by_county[TEST_FIPS]["all_filtered"]
   ```
   With:
   ```python
   allegheny_bldg_ids: list[int] = sorted(set().union(*[
       adopter_ids_by_mp[mp][TEST_FIPS]["all_filtered"] for mp in selected_mps
   ]))
   ```
2. Update the print statement to reference `selected_mps` not a single MP.
3. Rest of Step 5 (BSQ query, rename, sort, assertions) is unchanged.

**Validation gate:**
- `len(allegheny_bldg_ids)` → 1,610
- `n_bldgs` → 1,610
- `n_hours_per_bldg.min()` == `n_hours_per_bldg.max()` == 8760
- `weight_val` → 242.131013

---

### Task 3 — Fix Step 6: Multi-MP Upgrade Loop

**Goal:** Replace the single-MP upgrade query with a loop over `selected_mps`, producing `df_ts_upgrade_allegheny_by_mp`.

**Steps:**
1. Replace the Step 6 cell body with the loop from 21Apr Step 6 (lines 547–610).
2. The output variable must be: `df_ts_upgrade_allegheny_by_mp: dict[int, pd.DataFrame]`
3. Each iteration must:
   - Query `upgrade_id=str(mp)` for the current loop variable `mp`
   - Rename column to `'retrofit_kwh'`
   - Downcast to `np.float32`
   - Sort and assign deterministic `hour` index
   - Run schema parity check against `df_ts_baseline_allegheny`
   - Store in `df_ts_upgrade_allegheny_by_mp[mp]`
4. Final print outside loop: `✓ Step 6 PASSED — df_ts_upgrade_allegheny_by_mp.keys() = [3, 4]`
5. Update Step 6 markdown heading from "MP3 or MP4" to "All Selected MPs".

**Validation gate (per MP):**
- Rows: 14,103,600 (1,610 × 8,760)
- Buildings: 1,610
- Hours/bldg: 8760 – 8760
- Only in baseline: 0; Only in upgrade: 0
- MP3 kWh range: 32.930 to 25086.951
- MP4 kWh range: 32.930 to 23417.943

---

### Task 4 — Fix Step 7: Multi-MP Profile Loop + Profile Storage

**Goal:** Replace Step 7 with a loop over `selected_mps` that produces `peak_results_allegheny_by_mp` and `df_profiles_by_mp` (both indexed by MP).

**Steps:**
1. Initialize storage dicts before the loop:
   ```python
   peak_results_allegheny_by_mp: dict[int, dict[str, dict]] = {}
   df_profiles_by_mp: dict[int, dict[str, pd.DataFrame]] = {}
   ```
2. Loop over `selected_mps`. For each `mp`:
   - Pull `df_ts_upgrade_allegheny = df_ts_upgrade_allegheny_by_mp[mp]`
   - Pull `adopter_ids_allegheny = adopter_ids_by_mp[mp][TEST_FIPS]`
   - Compute 100pct profile: `df_profile_100pct, peak_100pct = compute_county_scenario_profile(..., adopter_ids_allegheny["all_filtered"])`
   - Compute constrained profile: `df_profile_constrained, peak_constrained = compute_county_scenario_profile(..., adopter_ids_allegheny["constrained"])`
   - Store: `peak_results_allegheny_by_mp[mp] = {"100pct": peak_100pct, "constrained": peak_constrained}`
   - Store: `df_profiles_by_mp[mp] = {"100pct": df_profile_100pct, "constrained": df_profile_constrained}`
   - Print peak results for both scenarios
   - Assert both profiles have 8760 rows
3. Final print: `✓ Step 7 PASSED — peak_results_allegheny_by_mp.keys() = [3, 4]`

**Validation gate — golden values:**
| MP | Scenario | Baseline Peak (MW) | Scenario Peak (MW) | Peak Hr (baseline) | Peak Hr (scenario) | Delta (MW) |
|----|----------|--------------------|--------------------|--------------------|--------------------|------------|
| 3 | 100% | 862.51 | 6629.87 | 4433 | 152 | +5767.36 |
| 3 | Constrained | 862.51 | 885.63 | 4433 | 116 | +23.12 |
| 4 | 100% | 862.51 | 5364.10 | 4433 | 152 | +4501.59 |
| 4 | Constrained | 862.51 | 2016.92 | 4433 | 152 | +1154.41 |

---

### Task 5 — Implement 2×2 Demand Timeseries Visualization

**Goal:** Produce a 2×2 matplotlib figure showing baseline and scenario demand profiles with
peak-hour dashed vertical lines. The figure must be dynamically computed from `df_profiles_by_mp`
and `peak_results_allegheny_by_mp` — no hardcoded hour values.

**Figure layout:**
- Rows: Scenario (100% adoption / Constrained adoption)
- Columns: MP3 (Standard ASHP) / MP4 (High-efficiency ASHP)
- X-axis: Hour of year (1–8760)
- Y-axis: MW (county aggregate, weight-applied)

**Per-panel content:**
- Red line: `baseline_mw` (hourly baseline electricity demand)
- Blue line: `scenario_mw` (hourly scenario demand post-retrofit)
- Vertical dashed black line at baseline peak hour (from `p["peak_hour_baseline"]`)
- Vertical dashed black line at scenario peak hour (from `p["peak_hour_scenario"]`)
- Annotation showing peak MW values and delta at the scenario peak line
- Legend: "Baseline", "Scenario (MPX)", peak hour labels
- Title per panel: e.g. "MP3 — 100% Adoption" / "MP4 — Constrained"

**Implementation requirements:**
- Use a helper function `plot_demand_panel(ax, df_profile, peak_result, mp, scenario_label)` with full Google/NumPy docstring and type hints
- Peak hour vertical lines must be drawn using `p["peak_hour_baseline"]` and `p["peak_hour_scenario"]` from `peak_results_allegheny_by_mp` — never hardcoded
- The function must work for any county, not just Allegheny
- Figure size: 16×10 inches; tight layout
- Save figure to `outputs/allegheny_demand_profiles_MP{selected_mps}.png` with `dpi=150`

**Column labels (in MP order):**
- MP3: "S1: Standard ASHP (15 SEER₁)"
- MP4: "S2: High-Efficiency ASHP (24 SEER₁)"

**Validation gate:**
- Figure renders with 4 panels (no empty axes, no missing lines)
- Baseline peak line falls at hour 4433 in all 4 panels (same baseline)
- Scenario peak lines match golden hour values per panel (see table above)
- No hardcoded hour numbers in any cell code

---

### Task 6 — Notebook Cleanup

**Goal:** Remove dead code, fix stale comments, consolidate imports.

**Steps:**
1. Remove `primary_mp: int = selected_mps[0]` if not already done in Task 1.
2. Remove unused imports: `matplotlib.colors as mcolors`, `create_npv_col`, `create_capital_col`.
3. Replace `input()` calls in Step 0c with hardcoded constants at top of cell, with comment:
   ```python
   # Hardcoded for reproducibility — change these for different runs.
   LOCATION_ID: str = "National"
   MODEL_RUN_DATE_TIME: str = "2026-04-10_00-05"
   ```
4. Add `logging.getLogger("buildstock_query").setLevel(logging.ERROR)` after imports in Step 0 to suppress recurring WARNING messages.
5. Update Step 6 markdown heading to "All Selected MPs".
6. Update Step 9 stub function signature: `primary_mp: int` → `selected_mps: list[int]`. Update docstring accordingly.
7. Remove bare `df_profile_100pct` / `df_profile_constrained` display cells (replaced by visualization in Task 5).

**Validation gate:**
- Notebook runs from top to bottom without errors up to Step 7 (Steps 8–10 still raise `NotImplementedError`)
- No `primary_mp` references remain outside of Step 9 stub docstring history note
- `input()` calls are gone

---

## Reference Values (Golden)

### Allegheny County Peak Results (FIPS 42003, AMY2018)

| MP | Scenario | Baseline Peak (MW) | Scenario Peak (MW) | Peak Hr Baseline | Peak Hr Scenario | Delta (MW) | Adopters | Total Bldgs |
|----|----------|--------------------|--------------------|-----------------|-----------------|------------|----------|------------|
| 3 | 100% | 862.51 | 6629.87 | 4433 | 152 | +5767.36 | 1,610 | 1,610 |
| 3 | Constrained | 862.51 | 885.63 | 4433 | 116 | +23.12 | 93 | 1,610 |
| 4 | 100% | 862.51 | 5364.10 | 4433 | 152 | +4501.59 | 1,610 | 1,610 |
| 4 | Constrained | 862.51 | 2016.92 | 4433 | 152 | +1154.41 | 518 | 1,610 |

### Allegheny County Adopter Tier Counts

| MP | Tier 1 | Tier 2 | Constrained | All Filtered |
|----|--------|--------|-------------|-------------|
| 3 | 64 | 29 | 93 | 1,610 |
| 4 | 190 | 328 | 518 | 1,610 |

### Baseline Timeseries

| Metric | Value |
|--------|-------|
| Buildings | 1,610 |
| Hours/building | 8,760 |
| Total rows | 14,103,600 |
| BSQ weight | 242.131013 |
| kWh range (wtd) | 31.961 to 10401.222 |

---

## Code Standards

- Google/NumPy docstrings on all functions introduced in this session
- Type hints required on all function parameters and return values
- No hardcoded magic numbers — use named constants from `cmu_tare_model.constants` or defined at top of cell
- `np.float32` for all kWh/MW timeseries arrays (memory constraint at national scale)
- All variable names in `snake_case`
- All `_by_mp` dicts are `dict[int, ...]` keyed by measure package integer
- Deterministic sort before `.cumcount()` — always `sort_values([BLDG_ID_COL, TIMESTAMP_COL])`

---

## Known Anti-Patterns (Do NOT Suggest)

| Anti-Pattern | Why Wrong |
|---|---|
| `weight = 242.131013; df['kwh'] = df['kwh'] * weight` | BSQ already applies weight; this double-counts |
| `primary_mp = selected_mps[0]` | Silently drops all MPs except the first |
| `adopter_ids_by_county[TEST_FIPS]["all_filtered"]` | Wrong variable name — must be `adopter_ids_by_mp[mp][...]` |
| Hardcoding peak hours (e.g., `ax.axvline(x=152)`) | Breaks for other counties; must come from `peak_results` dict |
| `split_enduses=True` in TSQuery | Triggers Pydantic ValidationError in BSQ batch path |
| `df['kwh'] / 1000 * 242.131013` | Weight already applied; divide by 1000 only |
| Editing `.py` or `.pdf` reference files | Out of scope — these are read-only timestamps |

---

## Appendix A — Variable Name Mapping (Old → New)

| Old (30Apr regression) | New (correct multi-MP) |
|---|---|
| `adopter_ids_by_county` | `adopter_ids_by_mp[mp]` |
| `df_ts_upgrade_allegheny` | `df_ts_upgrade_allegheny_by_mp[mp]` |
| `peak_results_allegheny` | `peak_results_allegheny_by_mp[mp]` |
| `df_profile_100pct` | `df_profiles_by_mp[mp]["100pct"]` |
| `df_profile_constrained` | `df_profiles_by_mp[mp]["constrained"]` |
| `primary_mp` | *(remove entirely)* |

---

## Appendix B — Plot Helper Function Signature (Target)

```python
def plot_demand_panel(
    ax: "matplotlib.axes.Axes",
    df_profile: pd.DataFrame,
    peak_result: dict[str, Any],
    mp: int,
    scenario_label: str,
    county_name: str = "Allegheny County, PA",
) -> None:
    """Plot baseline and scenario demand timeseries on a single axes panel.

    Draws the baseline (red) and scenario (blue) hourly demand profiles,
    with vertical dashed black lines at each series' peak hour.
    Peak hour positions are read dynamically from peak_result — never hardcoded.

    Args:
        ax: Matplotlib axes to draw on.
        df_profile: 8760-row DataFrame with columns 'hour', 'baseline_mw', 'scenario_mw'.
        peak_result: Dict from peak_results_allegheny_by_mp[mp][scenario] containing
            'baseline_peak_mw', 'scenario_peak_mw', 'peak_hour_baseline',
            'peak_hour_scenario', 'delta_mw'.
        mp: Measure package integer (e.g. 3 or 4), used in legend label.
        scenario_label: Human-readable scenario name (e.g. "100% Adoption").
        county_name: County name for axis title. Defaults to Allegheny County, PA.

    Returns:
        None. Modifies ax in place.
    """
```

---

## Session Summary Template

After completing all tasks, produce a summary covering:
1. Which tasks were completed vs. deferred
2. Final golden-value verification table (copy from Task 4 gate, with actual vs. expected)
3. Any discrepancies found during the run (and disposition)
4. Current state of Steps 8–10 stubs (should still be `NotImplementedError`)
5. Recommended next session focus (Step 8 validation vs. Step 9 national loop design)
