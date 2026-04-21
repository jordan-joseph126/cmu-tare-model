# Post-TARE Peak Load Pipeline — Implementation Plan (v1)

> **Version notes.** Created April 21, 2026 after two rounds of notebook audit (original April 20 MP3-only, then April 21 MP3+MP4 refactor). This plan consolidates all outstanding P0/P1 findings into an executable sequence. The audit transcript references are in `pre-post-tare-adoption-kpis` branch session history.

---

## Your Role

You are a senior Python engineer working on the TARE model's grid-impact pipeline inside the `cmu-tare-model` repo, branch `pre-post-tare-adoption-kpis`. The notebook you are fixing produces county-level peak load numbers for the Joseph et al. 2026 Energy Policy paper (submission deadline: late April 2026). The primary author is a researcher, not a senior engineer — prioritize readable, well-tested, well-commented code over clever one-liners.

---

## Project Context

The notebook `calculate_postTARE_ts_aws_peak_demand_20April2026_MP3_MP4.py` queries the AWS-hosted ResStock EUSS 2022.1.1 timeseries database via BuildStockQuery (BSQ) to compute county-level peak load changes under two adoption scenarios per measure package (MP): (a) 100% adoption counterfactual, (b) economically-constrained adoption (TARE Tier 1 + Tier 2 only). The current test case is Allegheny County, PA (FIPS 42003) for MP3 (standard ASHP) and MP4 (high-efficiency ASHP).

Test-county runs (Steps 5–7) produced plausible numbers, but two correctness gates are stubbed (Step 8 EUSS validation) or unresolved (NaN-tier handling in `extract_adopter_ids`), and the national loop (Step 9) cannot run in its current form both because its signature is stale *and* because the per-building data download pattern would take ~18 days sequentially.

---

## Scope Constraint — CRITICAL

**In scope:**
- The notebook `calculate_postTARE_ts_aws_peak_demand_20April2026_MP3_MP4.py`
- Helper functions in `cmu_tare_model/grid_impact/peak_load_functions.py`
- Constants in `cmu_tare_model/constants.py` (read-only reference)
- Any new helper modules you create in `cmu_tare_model/grid_impact/`

**Out of scope:**
- The TARE model itself (`cmu_tare_model/adoption_kpis/`, model core, REMDB, ACS integration)
- The ResStock data pipeline (ResStock 2025.1 integration is flagged as future work and is not your concern)
- Paper text, figures, or the outline (`paper_outline_v3.md`)
- Any other notebook in the repo

If you find yourself wanting to edit files outside `grid_impact/` or outside the notebook under review, stop and ask.

---

## Key Principle: Validate Before Scaling (Non-Negotiable)

This pipeline is cheap to run wrong and expensive to run right. A single national loop takes an estimated 12–18 days of Athena queries and ~7 TB of S3 transfer. **No scaling work happens until the test county is validated.** That means:

1. NaN/adopter logic must be understood and correct (Task 1) before Step 8 runs.
2. Step 8 must pass on Allegheny (Task 2) before Step 9 is implemented.
3. Step 9 performance must be proven on one state (Task 4 checkpoint) before the full national run.

**Do not trust "it ran and produced a number."** Trust only "it ran and produced the number that matches the independent oracle."

---

## What Was Done Before

### Prior session 1 — Original audit (April 20, MP3-only notebook)
- Identified two P0 correctness issues (NaN-tier handling, Step 8 stub)
- Identified two P1 scaling issues (per-building download volume, stubbed national loop)
- Catalogued ~13 P2/P3 code-quality items
- Flagged that MP4 was loaded into memory but never processed (`selected_mps=[3,4]` with `primary_mp=selected_mps[0]`)

### Prior session 2 — MP3+MP4 refactor (April 21)
- Added per-MP loops in Steps 4, 6, 7
- Changed Step 5 to use the **union** of bldg_ids across MPs (so baseline queries once, not per-MP)
- Verified MP3 results reproduce bit-for-bit from session 1
- Generated MP4 Allegheny results: 100% peak = 5,364 MW (vs MP3: 6,630 MW), constrained peak = 2,017 MW with 518 adopters (vs MP3: 886 MW with 93 adopters)

### Not yet done (this session's work)
- P0-1: NaN-tier / already-upgraded question
- P0-2: Step 8 EUSS validation
- P1-2: Step 9 stale function signature
- P1-1 + P1-3: Step 9 redesign (Athena aggregation + per-MP output schema)
- Step 10 CSV export
- Selected P2 cleanups

---

## Attached Files

| File | Purpose |
|---|---|
| `calculate_postTARE_ts_aws_peak_demand_20April2026_MP3_MP4.py` | Notebook script (current state — will be your main edit target) |
| `calculate_postTARE_ts_aws_peak_demand_20April2026_MP3_MP4.pdf` | Last known good PDF export of the notebook outputs — your regression oracle for MP3 and MP4 Allegheny results |
| `cmu_tare_model/grid_impact/peak_load_functions.py` | Helper module (`extract_adopter_ids`, `compute_county_scenario_profile`, `find_adoption_column`, `gisjoin_to_fips`) |
| `cmu_tare_model/constants.py` | Read-only constants (column names, TEST_FIPS=42003, BSQ weight, etc.) |

---

## Current Implementation Status

| Step | Status | Last exec time | Notes |
|---|---|---|---|
| 0 — Imports | ✅ Done (with ~25 dead imports) | — | P2 cleanup candidate |
| 0b — MP selection | ✅ Done | — | Hardcoded `selected_mps=[3,4]` |
| 0c — Load TARE data | ⚠️ Partial | — | Still uses `input()` despite refactor comment; dual `try/except NameError` is fragile |
| 1 — BSQ init | ✅ Done | — | |
| 2 — Column constants | ✅ Done | — | Print-only cell |
| 3 — County geography | ✅ Done | — | 3,235 counties; Allegheny 42003 validated |
| 4 — Extract adopters | ⚠️ Partial | — | **P0-1**: 39,973 NaN-tier buildings unexplained (same count for MP3/MP4, which conflicts with the "already upgraded" hypothesis — see Task 1) |
| 5 — Baseline ts query | ✅ Done | 166 s / 1.18 GB | Union across MPs |
| 6 — Upgrade ts query | ✅ Done (per MP) | 152–213 s / 1.18 GB per MP | |
| 7 — Scenario profile | ✅ Done (per MP) | — | **Numbers unverified** until Task 2 passes |
| 8 — EUSS validation | ❌ Stub | — | `NotImplementedError` — Task 2 |
| 9 — National loop | ❌ Stub + stale sig | — | `run_national_peak_load_loop` signature references old `adopter_ids_by_county` and `primary_mp`; neither exists post-refactor. Task 3 (signature) + Task 4 (redesign). |
| 10 — CSV export | ❌ Stub | — | Task 5 |

---

## Required First Action (Task 0 — do this before anything else)

Before writing any code:

1. **Read this entire plan end to end.** You will iterate on it in Task 0 — that's intentional.
2. **Read the notebook script** `calculate_postTARE_ts_aws_peak_demand_20April2026_MP3_MP4.py` in full.
3. **Read `cmu_tare_model/grid_impact/peak_load_functions.py`** to understand how `extract_adopter_ids`, `compute_county_scenario_profile`, and `find_adoption_column` work today. Pay special attention to how `extract_adopter_ids` handles rows where the adoption column is NaN.
4. **Grep the repo for "Already Upgraded", "already_upgraded", "ALREADY", or similar labels** that might explain the 39,973-row gap between `df_tare.shape[0]=331,531` and `tier_counts.sum()=291,558`.
5. **Produce a short audit note (markdown, ~1 page)** covering:
   - Whether the plan's task list matches what the codebase actually needs
   - Any tasks that should be added, removed, or re-ordered
   - Any assumptions in the plan that turn out to be wrong
   - Specific file paths and function names you'll be touching (so we can confirm scope before you edit)
6. **Update this plan file in place** (`cmu_tare_model/grid_impact/peak_load_implementation_plan.md`) with your findings. Mark the version as v2 and describe what changed.

**Do not start Task 1 until the audit note is presented and confirmed.**

---

## Tasks

### Task 1 — Resolve the NaN-tier / Already-Upgraded question (P0-1)

**Goal:** Determine with evidence (not inference) what the 39,973 NaN-tier rows in `df_tare` represent, and whether they should be in the `all_filtered` set used for the 100% adoption counterfactual.

**Background:** `df_tare.shape[0]` is 331,531 for both MP3 and MP4. The `value_counts()` of the adoption tier column sums to 291,558 for both MPs, leaving 39,973 rows with NaN adoption tier. The user hypothesis is that these represent "already upgraded" homes — buildings with an existing heat pump at or above the measure package's target efficiency. **This hypothesis is suspect** because:

- The NaN count is **identical** for MP3 and MP4 (39,973 each), but MP3 (SEER 15 target) and MP4 (SEER 24 ducted / SEER 29.3 ductless target) have very different efficiency thresholds, so "already-upgraded" counts *should* differ between them.
- Meanwhile, the `"N/A: Invalid Baseline Fuel/Tech"` tier count **does** differ: MP3 = 183, MP4 = 1,273. This is more consistent with a threshold-dependent category.

**Working hypothesis to verify:** The 39,973 NaN rows are filtered out *upstream* (before the adoption decision is made) for an MP-independent reason — probably non-residential, multifamily-with-shared-HVAC, or some other global applicability filter. Separately, the `"N/A: Invalid Baseline Fuel/Tech"` tier captures the MP-dependent "already-upgraded" category. If this is correct, the 39,973 probably should NOT be in `all_filtered`, because they aren't meaningful study-scope buildings.

**Steps:**

1. In a Jupyter cell or a standalone script in `grid_impact/`, load the TARE MP3 DataFrame and examine the NaN-adoption rows:
   ```python
   nan_mask = df_tare[adoption_col].isna()
   df_nan = df_tare[nan_mask]
   df_not_nan = df_tare[~nan_mask]
   ```
2. Compare distributions of `in.heating_fuel`, `in.hvac_heating_type_and_fuel`, `in.hvac_heating_efficiency`, `in.geometry_building_type_recs`, and `in.vacancy_status` between the two groups. Document the differences.
3. Check whether NaN rows are consistent between MP3 and MP4 (same bldg_ids have NaN in both, or different). Compute:
   ```python
   nan_bldg_ids_mp3 = set(df_tare_mp3.index[df_tare_mp3[adoption_col_mp3].isna()])
   nan_bldg_ids_mp4 = set(df_tare_mp4.index[df_tare_mp4[adoption_col_mp4].isna()])
   overlap = len(nan_bldg_ids_mp3 & nan_bldg_ids_mp4)
   ```
4. Read `extract_adopter_ids` in `peak_load_functions.py` and determine explicitly what it does with NaN rows. Does it include them in `all_filtered`? In the tier buckets? Dropped entirely?
5. Cross-reference: look at the Tier 4 ("Averse") counts. MP3 has 151,340 and MP4 has 62,856 — very different. Tier 4 likely includes "adoption not economically rational," which *should* differ between MPs. Confirm that Tier 4 is not the "already-upgraded" category by checking the HVAC efficiency distribution within it.
6. Write a short finding note (~1 page markdown) in `grid_impact/nan_tier_investigation.md` with:
   - What the 39,973 NaN rows actually represent
   - What the 183 (MP3) / 1,273 (MP4) `N/A` tier rows represent
   - Whether `extract_adopter_ids` handles each correctly
   - Recommended fix (if any) — exact function signatures, expected behavior

**Verification contract:**
- The finding note explains both the NaN and the `N/A` categories with distribution evidence, not inference.
- Any recommended fix is specified with before/after function signatures and expected `all_filtered` / `constrained` counts for Allegheny after the fix.
- The recommended fix does NOT change the MP3 constrained result (93 adopters, +23.12 MW delta) unless evidence explicitly justifies it. Tier 1 + Tier 2 counts should remain 16,650 + 16,453 = 33,103 nationally for MP3.

🛑 **Present the finding note and proposed fix before editing any code.** If the fix alters MP3 golden values, we need to confirm the justification together.

---

### Task 2 — Implement Step 8: EUSS Peak Load Validation (P0-2)

**Goal:** Provide an independent oracle for the baseline peak computed from timeseries data, using the pre-computed EUSS per-building peak load columns.

**Background:** EUSS metadata includes two columns: `out.electricity.winter.peak.kw` and `out.electricity.summer.peak.kw`. These are per-building annual peak values. Summed naively across a county's buildings, they give an upper bound on the county's coincident peak (since coincident peak ≤ sum of individual peaks). If the profile-derived baseline peak from Step 7 (862.51 MW for Allegheny) is within 20% of this naive sum, the pipeline is internally consistent. More than 20% difference requires investigation before scaling.

**Steps:**

1. Add a new Jupyter cell for Step 8 (replacing the `raise NotImplementedError` line).
2. Query the EUSS baseline metadata for Allegheny County buildings using BSQ or direct Athena — pull `out.electricity.winter.peak.kw`, `out.electricity.summer.peak.kw`, and the BSQ `weight` column for the `allegheny_bldg_ids` list.
3. Compute the **weighted naive sum**:
   ```python
   naive_winter_peak_mw = (df_meta['out.electricity.winter.peak.kw'] * df_meta['weight']).sum() / 1000
   naive_summer_peak_mw = (df_meta['out.electricity.summer.peak.kw'] * df_meta['weight']).sum() / 1000
   ```
4. Compare to the profile-derived baseline peak from `peak_results_allegheny_by_mp[3]['100pct']['baseline_peak_mw']`:
   ```python
   profile_baseline_mw = 862.51  # from Step 7
   ratio_winter = profile_baseline_mw / naive_winter_peak_mw
   ratio_summer = profile_baseline_mw / naive_summer_peak_mw
   ```
5. **Expected:** the profile peak (coincident across buildings) should be *less than* either naive sum (since naive sums assume all buildings peak at the same hour, which overstates true coincident peak). Ratio should be in the range ~0.3–0.8.
6. If ratio > 1.0, something is wrong — the coincident peak cannot exceed the naive sum of individual peaks. Raise a clear `AssertionError` explaining the discrepancy.
7. If ratio < 0.2, the timeseries aggregation is suspiciously low — also flag.
8. Print a summary table: profile_baseline_mw, naive_winter_peak_mw, naive_summer_peak_mw, ratio_winter, ratio_summer, decision (PASS/INVESTIGATE).

**Verification contract:**
- Running Step 8 produces a clear PASS or INVESTIGATE decision for Allegheny.
- If PASS: baseline is validated, and Task 4 (national loop) can proceed.
- If INVESTIGATE: present the numbers in chat before moving on. Do not silence the check or relax the 20% threshold without explicit approval.

---

### Task 3 — Fix the stale Step 9 signature (P1-2)

**Goal:** Update `run_national_peak_load_loop` signature to match the post-refactor data shape, without implementing the body yet.

**Current (stale) signature:**
```python
def run_national_peak_load_loop(
    bsq: "BuildStockQuery",
    adopter_ids_by_county: dict[str, dict[str, list[int]]],  # ← no longer exists
    county_geo_df: pd.DataFrame,
    primary_mp: int,                                          # ← no longer exists
    project_root: str,
) -> pd.DataFrame:
```

**New signature:**
```python
def run_national_peak_load_loop(
    bsq: "BuildStockQuery",
    adopter_ids_by_mp: dict[int, dict[str, dict[str, list[int]]]],  # from Step 4
    county_geo_df: pd.DataFrame,
    selected_mps: list[int],
    project_root: str,
    checkpoint_dir: str | None = None,
) -> pd.DataFrame:
    """Scale Steps 5–7 across all counties and selected measure packages.

    Parameters
    ----------
    bsq : BuildStockQuery
        Initialized BSQ client.
    adopter_ids_by_mp : dict[int, dict[str, dict[str, list[int]]]]
        Nested mapping {mp: {fips: {"tier1": [...], "tier2": [...],
        "constrained": [...], "all_filtered": [...]}}}.
    county_geo_df : pd.DataFrame
        Lookup of fips_5digit -> county_name -> state_fips.
    selected_mps : list[int]
        Measure packages to process (e.g., [3, 4]).
    project_root : str
        Path to repo root for checkpoint file resolution.
    checkpoint_dir : str | None
        If provided, save per-state checkpoints here. If None, defaults
        to `{project_root}/cmu_tare_model/output_results/checkpoints/`.

    Returns
    -------
    pd.DataFrame
        One row per (fips, mp) with columns:
        [fips, county_name, state_fips, mp, n_adopters_constrained,
         n_all_filtered, baseline_peak_mw, scenario_100pct_peak_mw,
         scenario_constrained_peak_mw, delta_100pct_mw,
         delta_constrained_mw, peak_hour_100pct, peak_hour_constrained]
    """
    raise NotImplementedError("Task 4 — implement body")
```

**Steps:**
1. Update the signature and docstring only. Keep the `NotImplementedError` body.
2. Update the Step 9 markdown description in the notebook to reflect the per-MP output shape.
3. Verify nothing in the notebook still references `adopter_ids_by_county` or `primary_mp`.

**Verification contract:**
- Running the notebook through Step 8 still produces MP3 and MP4 golden values unchanged.
- Running the Step 9 cell produces the expected `NotImplementedError`, not a `NameError` about stale variables.

---

### Task 4 — Redesign Step 9 for Athena-side aggregation (P1-1 + P1-3)

**Goal:** Implement `run_national_peak_load_loop` such that the national run completes in < 24 hours, not 18 days.

**Background:** The current Steps 5–6 pattern downloads per-building hourly data (~14 M rows, 1.18 GB per county per upgrade). The peak load computation only needs the per-hour county sum — 8,760 numbers. Pushing the aggregation into Athena SQL reduces the download ~1,600× per query.

**Design:**

Instead of one query per (county × upgrade) that groups by `bldg_id`, issue **two queries per (county × upgrade × scenario)** that group only by hour:

1. **Adopter-only sum:** `SUM(elec_total × weight)` for bldg_ids in the scenario's adopter set, grouped by hour.
2. **Non-adopter-only sum:** Same, for bldg_ids NOT in the adopter set (baseline values only, so query against upgrade=0).

Combine: hourly county scenario profile = adopter_hourly_sum(upgrade=MP) + non_adopter_hourly_sum(upgrade=0).

**Variants to consider in this task:**

- **Variant A (simplest):** Two queries per scenario → 4 queries per (county × MP) for constrained and 2 queries for 100% (since 100% has no non-adopter set). Estimated runtime: ~200 queries/s at Athena concurrency limits → hours, not days.
- **Variant B (one query per scenario):** Use a `CASE WHEN bldg_id IN (adopter_list) THEN upgrade_elec ELSE baseline_elec END` expression inside a single query. Fewer queries but more complex SQL. Requires two BSQ tables joined (or one BSQ query parameterized by upgrade_id with an adopter filter).
- **Variant C (batch by state):** Process all counties of a state in one query, using `GROUP BY in.county, hour` and building a `CASE WHEN bldg_id IN (state_adopter_union) ...` expression. Fewest queries but most complex.

**Recommended starting point:** Variant A, because it's the easiest to verify against the known Allegheny golden values.

**Steps:**

1. Implement Variant A as the initial design. Validate it against Allegheny Step 7 output — the hourly profile it produces should match `df_profile_100pct` and `df_profile_constrained` within float32 rounding.
2. Add per-state checkpoint saving: after each state completes, write a CSV to `{checkpoint_dir}/peak_load_MP{mp}_{state_abbr}.csv`. On resume, skip states whose checkpoint already exists.
3. Add progress logging: print a summary line per state (e.g., "PA (67 counties) — 420 s, peak delta range +12 to +5,767 MW").
4. Add a dry-run mode: if `dry_run=True`, print the estimated number of queries and data volume without executing.
5. Wrap the BSQ call in a retry decorator (3 attempts, exponential backoff) to handle transient Athena failures.
6. **Do not attempt the full national run from Claude Code.** Run it on PA only (67 counties) as the scaling test. Once PA finishes in < 3 hours, the design is validated for national scale.

**Verification contract:**
- Allegheny (FIPS 42003) results from Step 9 match the Step 7 results exactly for both MP3 and MP4 (baseline peak, scenario peaks, deltas, adopter counts).
- PA (state FIPS 42) completes in < 3 hours with all 67 counties producing one row each.
- Checkpoint files resume correctly: delete the last state's checkpoint, re-run, and verify only that state's counties re-execute.
- Running with `dry_run=True` prints the plan without executing any Athena queries.

🛑 **Present the Allegheny re-validation before attempting PA.** If the per-MP Allegheny numbers don't match Step 7 exactly, stop and investigate.

---

### Task 5 — Implement Step 10: CSV export for paper figures

**Goal:** Export the national results DataFrame in the exact shape needed for Figure XX (county choropleth by scenario × MP) and the Allegheny case study panel.

**Steps:**
1. Export `df_peak_results_national` (the output of Task 4) to `{project_root}/cmu_tare_model/output_results/peak_load_results_{timestamp}_national.csv`.
2. Export the Allegheny-only subset to `peak_load_results_{timestamp}_allegheny.csv` for the case study.
3. Include a README.txt next to the CSVs describing column meanings and units (especially that `baseline_peak_mw` is in MW, `peak_hour` is 1-indexed 1..8760, etc.).

**Verification contract:**
- The national CSV has one row per (county × MP); no missing counties from `adopter_ids_by_mp`.
- Column names match what paper Figure XX scripts expect (if known — ask if unclear).

---

### Task 6 — Selected P2 cleanup (optional, only if time permits)

Priority within P2:
1. Extract the per-MP loop bodies in Steps 4, 6, and 7 into a `process_mp()` helper (removes ~80 lines of duplication).
2. Remove the ~25 dead imports in Step 0.
3. Replace `input()` calls in Step 0c with explicit configuration (dict or dataclass).
4. Fix the fragile `try/except NameError` pattern in Step 0c (guard all three variables, not just one).

Skip anything not in this list.

---

## Reference Values (golden)

These are the numbers your work must preserve. If they change, stop and explain why.

### Allegheny County (FIPS 42003) — Step 7 baseline oracle

| Metric | MP3 | MP4 |
|---|---|---|
| Baseline peak | 862.51 MW @ hour 4433 | 862.51 MW @ hour 4433 |
| 100% adoption scenario peak | 6,629.87 MW @ hour 152 | 5,364.10 MW @ hour 152 |
| 100% adoption delta | +5,767.36 MW | +4,501.59 MW |
| Constrained adopters | 93 | 518 |
| Constrained scenario peak | 885.63 MW @ hour 116 | 2,016.92 MW @ hour 152 |
| Constrained delta | +23.12 MW | +1,154.41 MW |
| All filtered bldg_ids | 1,610 | 1,610 |

### National-level adopter counts (before any Task 1 fix)

| | MP3 | MP4 |
|---|---|---|
| Tier 1 (Feasible) | 16,650 | 52,407 |
| Tier 2 (Feasible vs. Alternative) | 16,453 | 60,466 |
| Tier 3 (Subsidy-Dependent) | 106,932 | 114,556 |
| Tier 4 (Averse) | 151,340 | 62,856 |
| N/A: Invalid Baseline Fuel/Tech | 183 | 1,273 |
| **Tier total** | **291,558** | **291,558** |
| df_tare total rows | 331,531 | 331,531 |
| **NaN-tier gap** | **39,973** | **39,973** |
| Counties with adopters | 3,098 | 3,098 |
| Constrained total (T1+T2) | 33,103 | 112,873 |

### Infrastructure constants

| Constant | Value |
|---|---|
| BSQ sample weight (EUSS 2022.1.1) | 242.131013 |
| Hours per year (query assertion) | 8,760 |
| Test FIPS | 42003 (Allegheny, PA) |
| ResStock release | `resstock_amy2018_release_1_1` |
| Athena workgroup | `resstock-euss` |

---

## Code Standards

1. **Docstrings:** Google/NumPy style on every function. Include `Parameters`, `Returns`, and `Raises` sections.
2. **Type hints:** Required on all function signatures. Use `from __future__ import annotations` for forward references. Use `dict[...]` / `list[...]` syntax (3.9+), not `typing.Dict` / `typing.List`.
3. **Error handling:** Fail fast with specific exception types. Validate inputs early. Never silently swallow exceptions.
4. **Comments:** Explain *why*, not *what*. Domain-specific logic (TARE tiers, BSQ weight, coincident peak) should have a one-line rationale.
5. **Test values:** When writing a new helper, add an assertion against the Allegheny golden value. Example:
   ```python
   assert peak_100pct["baseline_peak_mw"] == pytest.approx(862.51, rel=0.001)
   ```
6. **Float precision:** Use `np.float32` for timeseries kWh values (halves memory at national scale). Use `np.float64` for aggregated MW values (no precision concern, easier debugging).
7. **Logging:** Use `print()` inside the notebook for user-visible progress. Use the `logging` module for helper functions that might be called outside the notebook.
8. **Commits:** Keep Task 1's investigation commit separate from Task 2's Step 8 implementation. One task per commit.

---

## Known Anti-Patterns (do NOT suggest)

| ❌ Anti-pattern | Why it's wrong |
|---|---|
| Silently dropping NaN-tier rows with `df.dropna(subset=[adoption_col])` before Task 1 investigation completes | We don't yet know what NaN represents; dropping them could change MP3 golden values. |
| Hardcoding the BSQ weight as 240.0 or any other number | The weight is 242.131013 and BSQ applies it internally. Hardcoding introduces 0.9% error, silently. |
| Using `split_enduses=True` in `TSQuery` | Triggers a Pydantic ValidationError in BSQ's batch query path. Use `split_enduses=False` with a single enduse. |
| Applying weights in pandas after a BSQ query | BSQ already applies `SUM(enduse × weight)` in SQL. Multiplying again double-counts by 242.131013×. |
| Summing `out.electricity.winter.peak.kw` across buildings and reporting it as the county coincident peak | This is the naive sum used for the upper-bound sanity check in Step 8 — it is NOT the coincident peak. |
| Writing the national loop without checkpointing | An 18-hour job that fails at hour 14 with no checkpoint = start over. |
| Running the national loop from a Claude Code session | Long-running AWS queries from an interactive agent is a bad pattern. Validate on PA, then let the user run national separately. |
| Aggregating in pandas when Athena can aggregate in SQL | This is the core P1-1 issue from the audit. If you write `.groupby("hour").sum()` on a pandas DataFrame that came from BSQ, ask whether you should have pushed that group-by into SQL. |
| Adding helper functions to `cmu_tare_model/adoption_kpis/` | Scope boundary — new grid-impact helpers go in `cmu_tare_model/grid_impact/`. |
| Modifying the TARE model's adoption column logic to "fix" NaN handling | That's upstream of your scope. Fix the downstream handling in `extract_adopter_ids` instead. |

---

## Appendix A — NaN vs N/A taxonomy (context for Task 1)

There are two distinct "missing" categories in the TARE output, easily confused:

1. **NaN in the adoption column** (39,973 rows per MP, same count, same bldg_ids likely). These are rows where `find_adoption_column(df_tare, mp, 'v3')` returns a Series with `NaN` values. `value_counts()` silently drops these. Hypothesis: upstream filter (non-residential, invalid geometry, etc.), MP-independent.
2. **`"N/A: Invalid Baseline Fuel/Tech"` as a string tier value** (183 MP3, 1,273 MP4). These appear in `value_counts()` as an explicit tier. Hypothesis: MP-dependent applicability (existing HVAC already matches or exceeds target).

Task 1 must establish which is which with evidence. The technical descriptions from ResStock EUSS documentation:

> **MP3**: Applies to dwellings with ducts and (no HP, or HP with SEER 10/13/15 or HSPF 6.2/7.7/8.5) OR dwellings without ducts and (no HP, or MSHP SEER 14.5 / HSPF 8.2). Target: SEER 15, HSPF 9.

> **MP4**: Applies to dwellings with ducts and (no HP, or HP with SEER < 24 / HSPF < 13) OR dwellings without ducts and (no HP, or MSHP SEER 14.5 / HSPF 8.2, or MSHP SEER 29.3 / HSPF 14 not sized to max load). Target: SEER 24 ducted / SEER 29.3 ductless.

Note that MP4's applicability is **wider** than MP3's (more baseline HP types qualify for upgrade). This predicts MP4 should have **fewer** "already-upgraded" buildings than MP3, which is the **opposite** of the observed 183 → 1,273 pattern. This contradiction is why Task 1 is an investigation, not a rote fix.

---

## Appendix B — Step 8 EUSS validation math

The naive sum of individual building winter peaks overstates the true county coincident peak because buildings don't all peak at the same hour. The ratio of coincident to naive sum is the **coincidence factor** (sometimes "diversity factor" inverted), typically 0.3–0.7 for residential aggregates at county scale.

The validation check asks: is the profile-derived coincident peak within this physically reasonable range relative to the naive sum? If yes, the pipeline is internally consistent. If no, one of:

- Timeseries aggregation is wrong (e.g., weight applied twice, hour indexing misaligned)
- Metadata peak columns use different units than assumed
- BSQ is returning unweighted values somewhere

A ratio > 1.0 is physically impossible and indicates a unit or aggregation bug.

---

## Appendix C — Step 9 Athena aggregation SQL sketch

Variant A template (adopter-only sum for county X, MP Y):

```sql
SELECT
    date_trunc('hour', timestamp) AS hour_ts,
    SUM("electricity.total.energy_consumption" * baseline.weight) AS adopter_kwh
FROM {timeseries_table}
JOIN {metadata_table} AS baseline
    ON timeseries.bldg_id = baseline.bldg_id
WHERE baseline.upgrade = {mp_id}
  AND baseline.bldg_id IN ({adopter_list})
  AND baseline."in.county_fips" = '{county_fips}'
GROUP BY date_trunc('hour', timestamp)
ORDER BY hour_ts
```

BSQ should be able to generate this with `TSQuery(group_by=['hour'], restrict=[('bldg_id', adopter_list), ('county_fips', county)])` — verify before hand-writing SQL.

---

## Session Summary Template

At the end of this session, produce a summary covering:

1. **Task 0 (plan audit):** What changed in the plan, what assumptions held, what surprised you.
2. **Task 1 (NaN investigation):** The evidence-based answer and any code changes to `extract_adopter_ids`.
3. **Task 2 (Step 8):** PASS/INVESTIGATE decision for Allegheny + the numbers.
4. **Task 3 (signature fix):** Before/after signature and any renamed variables elsewhere.
5. **Task 4 (Step 9 redesign):** Chosen variant (A/B/C), Allegheny re-validation result, PA scaling test result.
6. **Task 5 (Step 10):** Output CSV paths and column schema.
7. **Task 6 (cleanup):** What was done, what was deferred.
8. **Open items / residual risks:** Anything deferred to a future session or to the user for manual action.
9. **Updated golden values table** if any official numbers changed (with justification).
10. **Suggested v2 plan edits** if the structure of this plan should change for a future session.
