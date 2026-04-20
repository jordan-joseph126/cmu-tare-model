# Post-TARE Peak Demand — Implementation Plan (v1)

> **Version notes.** v1 created April 20, 2026 following the notebook audit
> (`notebook_audit_postTARE_peak_demand.md`). This plan bundles the MP-loop fix,
> Step 8 validation, Step 9 national scaling, and Step 10 export into a single
> coordinated implementation effort.

---

## Your Role

You are an expert Python developer and research engineer supporting Jordan M. Joseph's
Energy Policy submission (Joseph et al. 2026). You understand ResStock EUSS data,
BuildStockQuery (BSQ), Athena/S3 cost drivers, and the technology-differentiated framing
that anchors the paper. You prioritize correctness and scaling safety over speed of
implementation; when a change could alter a numerical result, you validate against a
reference value before declaring done.

## Project Context

The notebook `calculate_postTARE_ts_aws_peak_demand_20April2026` computes county-level
peak electricity demand changes from residential heat pump adoption, for two measure
packages (MP3 = standard ducted ASHP, MP4 = high-efficiency ASHP). Inputs are TARE model
adoption tiers (per bldg_id) joined against ResStock EUSS 2022.1.1 AMY2018 hourly
timeseries on AWS. The test case is Allegheny County, PA (FIPS 42003). The national
target is ~3,098 counties × 2 MPs. Results feed paper Figure XX (Section 3.6 choropleth)
and the Pittsburgh case-study panel.

Four issues block the paper:

1. **MP4 is loaded then silently discarded.** `primary_mp = selected_mps[0]` collapses
   the MP list to a scalar, so only MP3 flows through Steps 4–7. This defeats the
   paper's technology-differentiation argument.
2. **Peak magnitudes are unvalidated.** Step 7's 100% adoption peak of 6.6 GW on 1,610
   buildings looks extreme. Step 8 (EUSS metadata cross-check) is a stub.
3. **Step 9 (national loop) is a stub.** Current pull-and-aggregate pattern moves
   1.18 GB per county per query → ~10 TB of S3 egress at national scale. Infeasible
   without a design change.
4. **Step 10 (CSV export) is a stub.**

## Scope Constraint — CRITICAL

**In scope:** the five tasks below (MP loop fix, Step 5/6 helper extraction, Step 8,
Step 9, Step 10) plus cleanup items documented in Appendix B. The goal is a working,
validated national peak-load pipeline for MP3 and MP4.

**Out of scope:**
- Refactoring TARE model internals (`cmu_tare_model/` modules beyond
  `cmu_tare_model/grid_impact/peak_load_functions.py`)
- Implementing MP5/MP6/MP7 (cold-climate HP, dual-fuel, DR) — these are paper
  extensions, not this session's work
- Touching Steps 0 through 3 functionality (shapefile load works; import path
  update in Task 0 is the one exception)
- ResStock 2025.1 migration (EUSS 2022.1.1 is what's in scope)

If out-of-scope material surfaces (e.g., "while we're at it, let's also add MP5"),
redirect: "Note this in the follow-up list; don't change the scope of this session."

## Philosophy — Non-Negotiable

**No hardcoded weights, ever.** BSQ applies the EUSS sampling weight (`242.131013`)
internally via `SUM(enduse × weight)` in generated SQL. Downstream code that multiplies
again double-counts by a factor of 242. A previous bug hardcoded `240.0` and produced a
0.9% error; the fix was to delete the hardcode, not to update it. If you see a numeric
literal near 240/242 in this file, treat it as a bug until proven otherwise.

**Validate before scaling.** Step 8 exists because Step 9 costs ~300 hours of Athena
compute and >10 TB of S3 egress. Running Step 9 before Step 8 passes is a fireable
offense. The order is: fix MP loop → Step 8 → design decision → Step 9 → Step 10.

## What Was Done Before

### Prior session — Cleanup refactor (through ~April 19, 2026)
- Replaced Step 0b interactive `input()` with `selected_mps = [3, 4]` constant
- Migrated TARE data loading to `load_measure_package_data()`
- Adopted BSQ as the AWS interface (was raw pyathena previously)
- Established the float32 downcast + deterministic hour-index ordering in Steps 5/6
- Wrote the TSQuery pattern that works around the `split_enduses=True` Pydantic bug
- Implemented Steps 1–7 for the Allegheny test case with MP3; Steps 8–10 are stubs

### This session — the plan below

## Attached Files

All peak-load work is colocated under `cmu_tare_model/grid_impact/`:

- `cmu_tare_model/grid_impact/calculate_postTARE_ts_aws_peak_demand_20April2026.py` —
  notebook as executable script (the file being modified)
- `cmu_tare_model/grid_impact/calculate_postTARE_ts_aws_peak_demand_20April2026.pdf` —
  last successful run (MP3 on Allegheny County), reference for what "working"
  looks like
- `cmu_tare_model/grid_impact/calculate_postTARE_ts_aws_peak_demand.ipynb` —
  source Jupyter notebook
- `cmu_tare_model/grid_impact/peak_load_functions.py` — existing helpers
  (`find_adoption_column`, `extract_adopter_ids`, `compute_county_scenario_profile`,
  `gisjoin_to_fips`); new helpers added in Tasks 2 and 4 go here
- `cmu_tare_model/grid_impact/notebook_audit_postTARE_peak_demand.md` — the audit
  that motivated this plan
- `cmu_tare_model/grid_impact/PERF_PROFILE_peak_load.md` — performance profile
  notes (user-maintained; reference for scaling decisions in Task 4)
- `cmu_tare_model/grid_impact/archived_files/` — older notebook revisions
  (e.g., `calculate_postTARE_ts_aws_peak_demand_15April2026.pdf`); do not edit
- `cmu_tare_model/constants.py` — constants (BSQ column names, FIPS, etc.)

**Import path caveat:** because `peak_load_functions.py` moved from
`cmu_tare_model/adoption_kpis/` into `cmu_tare_model/grid_impact/`, the notebook's
existing import (`from cmu_tare_model.adoption_kpis.peak_load_functions import ...`)
no longer resolves. Task 0 in the Copilot prompt fixes this before any other work.

## Current Implementation Status

| Step | Status | Notes |
|---|---|---|
| 0–3 | ✅ Done | Imports, MP selection, TARE load, BSQ init, column constants, county shapefile |
| 4 | ⚠️ Only MP3 processed | `primary_mp = selected_mps[0]` collapses MPs to scalar |
| 5 | ✅ Done, MP3 only | 174.54 s, 1,610 bldgs, 14.1M rows, 1.18 GB S3 |
| 6 | ✅ Done, MP3 only | 175.95 s, identical shape |
| 7 | ✅ Done, MP3 only | Magnitudes unvalidated |
| 8 | ❌ Stub | `NotImplementedError` |
| 9 | ❌ Stub | Function skeleton only |
| 10 | ❌ Stub | `NotImplementedError` |

## Required First Action

Before touching any code, re-read:
1. `notebook_audit_postTARE_peak_demand.md` (the audit) — especially the P0 findings
2. Step 5 and Step 6 in the `.py` file — they are the pattern that Steps 8 and 9 mirror
3. The TSQuery params: `split_enduses=False`, `timestamp_grouping_func='hour'`,
   `group_by=[BLDG_ID_COL]` — Step 9 may change the last one; confirm which
   `group_by` BSQ supports.

## Tasks

### Task 0 — Repoint imports to `grid_impact/` (blocker)

`peak_load_functions.py` has moved from `cmu_tare_model/adoption_kpis/` into
`cmu_tare_model/grid_impact/`. The notebook still imports from the old location,
so a fresh kernel will `ModuleNotFoundError` on cell 0.

**Strategy:** update *only* the import statements, not the code they reference.
Do not move any files; the on-disk layout is the source of truth.

**Implementation:**

1. In the notebook (`.py` export), change:
   ```python
   from cmu_tare_model.adoption_kpis.peak_load_functions import (
       gisjoin_to_fips, find_adoption_column, extract_adopter_ids,
       compute_county_scenario_profile,
   )
   ```
   to:
   ```python
   from cmu_tare_model.grid_impact.peak_load_functions import (
       gisjoin_to_fips, find_adoption_column, extract_adopter_ids,
       compute_county_scenario_profile,
   )
   ```
2. Ensure `cmu_tare_model/grid_impact/__init__.py` exists; create an empty file if
   not, or the package is not importable.
3. Other `cmu_tare_model.adoption_kpis.*` imports (`kpi_functions`,
   `visualize_geospatial_data`): leave untouched **unless** those files have also
   moved. Do not move files; only update imports to match reality.

**Verification contract:**
- Notebook cell 0 runs in a fresh kernel without `ModuleNotFoundError`
- No residual `adoption_kpis.peak_load_functions` references in the `.py` file

This is a blocker for Tasks 1–5.

### Task 1 — Fix the MP loop (P0)

**Goal:** process both MP3 and MP4 end-to-end, not just MP3.

**Strategy:** the cleanest path is to extract Steps 4–7 into a function and call it
for each MP. Baseline (Step 5) is MP-independent and must stay *outside* the loop to
avoid re-querying 1.18 GB of identical baseline data.

**Implementation:**

1. Keep Steps 0–3 unchanged.
2. After Step 3, extract adopter IDs **per MP** into a dict-of-dicts:
   ```python
   adopter_ids_by_mp: dict[int, dict[str, dict[str, list[int]]]] = {}
   for mp in selected_mps:
       df_tare = resolve_tare_dataframe(DATAFRAMES_BY_MP, mp, DISCOUNT_RATE_KEY, RCM_MODEL_KEY)
       adoption_col = find_adoption_column_any_cost_scenario(df_tare, mp)
       adopter_ids_by_mp[mp] = extract_adopter_ids(df_tare, adoption_col)
   ```
3. Query baseline (upgrade=0) **once**, using the union of all bldg_ids across MPs for
   the test county. Store as `df_ts_baseline_allegheny` (name unchanged for continuity).
4. For each MP, query the upgrade timeseries and compute the scenario profile:
   ```python
   peak_results_allegheny_by_mp: dict[int, dict[str, dict]] = {}
   for mp in selected_mps:
       df_ts_upgrade = query_county_hourly_electricity(
           my_run, bldg_ids, upgrade_id=str(mp), output_col='retrofit_kwh'
       )
       profile_100pct, peak_100pct = compute_county_scenario_profile(
           df_ts_baseline_allegheny, df_ts_upgrade,
           adopter_bldg_ids=adopter_ids_by_mp[mp][TEST_FIPS]["all_filtered"],
       )
       profile_constrained, peak_constrained = compute_county_scenario_profile(
           df_ts_baseline_allegheny, df_ts_upgrade,
           adopter_bldg_ids=adopter_ids_by_mp[mp][TEST_FIPS]["constrained"],
       )
       peak_results_allegheny_by_mp[mp] = {"100pct": peak_100pct, "constrained": peak_constrained}
   ```
5. Delete the `primary_mp = selected_mps[0]` line.

**Verification contract:**
- Notebook prints separate tier-distribution tables for MP3 and MP4
- Notebook prints separate peak-delta blocks for MP3 and MP4
- MP3 reference values (golden table below) unchanged
- `peak_results_allegheny_by_mp.keys() == {3, 4}`

### Task 2 — Extract the Step 5/6 helper (P2, prerequisite for Task 4)

**Goal:** one function that runs both Step 5 (baseline) and Step 6 (upgrade); removes
duplication and is the core of Step 9's national loop.

**Signature (add to `cmu_tare_model/grid_impact/peak_load_functions.py`):**

```python
def query_county_hourly_electricity(
    bsq: "BuildStockQuery",
    bldg_ids: list[int],
    upgrade_id: str,
    output_col: str,
) -> pd.DataFrame:
    """Query BSQ for hourly electricity for a set of buildings under one upgrade.

    Returns a DataFrame with columns [bldg_id, timestamp, output_col, hour, units_count].
    hour is a deterministic 1-based ordinal within each bldg_id, after sorting by
    (bldg_id, timestamp). Values are weight-applied kWh (BSQ handles the weight
    internally).
    """
```

**Verification:** Step 5 and Step 6 cells reduce to one function call each; the
pre/post row counts, hour ranges, and kWh-range summaries match the reference PDF.

### Task 3 — Implement Step 8 validation (P0)

**Goal:** cross-check Step 7's profile-derived baseline peak against EUSS metadata's
`out.electricity.winter.peak.kw` summed over the same bldg_ids for Allegheny County.
If the two disagree by >20%, we do not proceed.

**Implementation sketch:**

1. Pull EUSS metadata for the Allegheny bldg_id set:
   ```python
   df_meta = bsq.get_buildings_by_ids(
       bldg_ids=adopter_ids_by_mp[3][TEST_FIPS]["all_filtered"],
       columns=["out.electricity.winter.peak.kw", "in.county", "weight"],
   )
   ```
   (Adjust to the actual BSQ API — if `get_buildings_by_ids` doesn't exist, use
   `bsq.get_results_csv(...)` or a direct metadata query. Verify the method name
   against BSQ docs.)
2. Compute the weighted sum:
   ```python
   euss_peak_kw = (df_meta["out.electricity.winter.peak.kw"] * df_meta["weight"]).sum()
   euss_peak_mw = euss_peak_kw / 1000.0
   ```
3. Compare against `peak_results_allegheny_by_mp[3]["100pct"]["baseline_peak_mw"]`
   (baseline is MP-independent so MP3's is fine).
4. Print both values, their ratio, and a pass/fail flag at the 20% threshold.
5. If the check passes for MP3 baseline, declare Step 8 complete. If it fails,
   raise a clear diagnostic error and do NOT proceed to Task 4.

**Reference (expected from Step 7):** baseline peak = 862.51 MW. The EUSS sum of
per-building winter peaks is a naive sum (not a coincident peak), so the EUSS number
should be *higher* than 862.51 MW (naive sum ≥ coincident peak). If EUSS sum comes
out *lower*, something is wrong.

**Verification contract:**
- Step 8 prints `profile_baseline_peak_mw`, `euss_metadata_sum_peak_mw`, and the ratio
- Ratio falls in [0.8, 1.5] (EUSS sum can exceed profile peak; the floor of 0.8 guards
  against the profile being too high from a unit bug)
- Step 8 does not raise NotImplementedError

### Task 4 — Resolve the Step 9 design decision then implement

**Decision required before code:** Option A (Athena-side aggregation), Option B
(state-batched pull), or Option C (BSQ native county group-by). See audit for
trade-offs. Default recommendation: **Option A.**

**Option A sketch:**

1. For each (county, upgrade, scenario) query, push the hourly SUM into Athena:
   ```python
   ts_query = TSQuery(
       enduses=[ELEC_TOTAL_COL],
       restrict=[('bldg_id', adopter_ids_subset)],
       upgrade_id=str(upgrade_id),
       timestamp_grouping_func='hour',
       group_by=[],  # aggregate across all restricted buildings
       split_enduses=False,
   )
   # Returns 8760 rows instead of 8760 × n_bldgs
   ```
2. Query pattern per (county, mp, scenario):
   - Baseline-adopters (upgrade=0, bldg_ids=adopter set) — for the portion of baseline
     that will be *replaced* by the upgrade
   - Baseline-non-adopters (upgrade=0, bldg_ids=non-adopter set)
   - Upgrade-adopters (upgrade=mp, bldg_ids=adopter set)
   - Scenario profile = (non-adopter baseline) + (adopter upgrade)
   - Reference baseline profile = (non-adopter baseline) + (adopter baseline) = same
     as full baseline with no mask
3. Implement per-state parallelism using `concurrent.futures.ThreadPoolExecutor`
   with `max_workers` set conservatively (start at 8; Athena default concurrent-query
   limit is often 10 per workgroup).
4. Per-state checkpointing: after each state completes, write
   `checkpoints/peak_load_MP{mp}_state{state_abbr}.csv`. On restart, skip states
   whose checkpoints exist.

**Verification contract:**
- Function runs end-to-end for a single state (e.g., PA) and produces a DataFrame
  with one row per county in that state
- Re-running after Ctrl+C skips states with existing checkpoints
- For Allegheny County (FIPS 42003), the Step 9 peak_baseline_mw matches Step 7's
  Allegheny peak_baseline_mw within 0.5% (floating-point tolerance)
- Total S3 egress per state logged so you can verify the 1.18 GB → ~8.76 MB
  reduction held

### Task 5 — Implement Step 10 export

**Goal:** write the national peak-load results as CSVs, one per MP, plus the
Allegheny County subset for the case study figure.

**Implementation:**

1. For each mp in `selected_mps`, collect the per-county results into
   `df_peak_results_national_mp{mp}` with the schema from the stub comment:
   ```
   [fips, county_name, state, n_adopters_constrained, n_all_filtered,
    baseline_peak_mw, scenario_100pct_peak_mw, scenario_constrained_peak_mw,
    delta_100pct_mw, delta_constrained_mw, peak_hour_100pct, peak_hour_constrained]
   ```
2. Write to `cmu_tare_model/grid_impact/output_csv/peak_load_results_MP{mp}_national.csv`
3. Filter to FIPS == 42003 and write `peak_load_results_MP{mp}_allegheny.csv`
4. Print row counts and output file sizes for each CSV

**Verification contract:**
- Two CSVs per MP exist at the expected paths
- Row counts: national ≈ 3,098, allegheny == 1
- CSV opens cleanly in pandas; no NaN in `baseline_peak_mw` or `delta_*_mw` columns
- Spot-check: MP3 Allegheny row matches Step 7 reference values

## Reference Values (golden)

These are the MP3 Allegheny County outputs from the April 20, 2026 run. Any refactor
that changes them is a regression unless the change is intentional and documented.

| Metric | Value |
|---|---|
| Allegheny FIPS | 42003 |
| all_filtered buildings | 1,610 |
| Tier 1 + Tier 2 (constrained) | 93 (64 + 29) |
| Tier 1 count | 64 |
| Tier 2 count | 29 |
| MP3 tier distribution: Feasible / Feasible-vs-Alt / Subsidy-Dep / Averse / N-A | 16,650 / 16,453 / 106,932 / 151,340 / 183 |
| Total MP3 TARE rows | 291,558 (+ 39,973 N/A in full 331,531) |
| National counties with adopters | 3,098 |
| National constrained adopters (MP3) | 33,103 |
| BSQ sampling weight | 242.131013 |
| Allegheny baseline peak | 862.51 MW @ hour 4433 |
| Allegheny 100% adoption peak (MP3) | 6,629.87 MW @ hour 152 |
| Allegheny constrained peak (MP3) | 885.63 MW @ hour 116 |
| Step 5 query time | 174.54 s |
| Step 6 query time | 175.95 s |
| S3 bytes per TS query (MP3 Allegheny) | 1,184,663,251 (~1.18 GB) |

MP4 values are **unknown** until Task 1 completes — producing them is the purpose of
the task.

## Code Standards

- Python ≥ 3.10 syntax (`list[int]`, `dict[str, list[int]]` — no `typing.List`)
- Type hints on every new function signature
- Google/NumPy docstring style, as already used in `peak_load_functions.py`
- No bare `except:` — catch specific exceptions with informative messages
- Prefer named constants over magic strings (e.g., `SCENARIO_100PCT = "100pct"`)
- Lines ≤ 100 chars where practical
- Run-to-completion determinism: any `.groupby().cumcount()` must be preceded by
  an explicit `.sort_values()` on the grouping and ordering keys
- No new `input()` calls — use env vars or module-level constants
- No hardcoded weights, ever (see Philosophy)

## Known Anti-Patterns (do NOT suggest)

| Anti-pattern | Why it's wrong |
|---|---|
| Keep `primary_mp = selected_mps[0]` "for convenience" | This is the bug. Deleting it is the fix. |
| Multiply any query result by `240.0` or `242.131013` to "apply the weight" | BSQ already applied the weight. Double-counting. |
| Set `split_enduses=True` in TSQuery | Triggers a Pydantic ValidationError in BSQ's batch path. Known BSQ bug. |
| Skip Step 8; Step 7 output "looks fine" | The 6.6 GW peak on 1,610 buildings is exactly the shape of a units bug. Don't scale what you haven't validated. |
| Implement Step 9 as a sequential county loop | 300+ hours of sequential Athena time. Parallelize at state level with checkpointing. |
| Re-query baseline inside the MP loop | Baseline is MP-independent. Query once, reuse. |
| Build Step 9 around pulling 14M rows per county | That's 10+ TB of S3 egress nationally. Aggregate in Athena. |
| Use `df.groupby().cumcount()` without a prior `sort_values` | Order-nondeterministic; will silently shuffle the hour index. |
| Paraphrase `list[int]` as `List[int]` from `typing` | The repo is on modern Python type syntax. |

## Appendix A — Step 9 design decision notes

See audit section "Design decision required before Step 9 is written" for full detail.
Recommendation: Option A (Athena-side aggregation, `group_by=['time']` with WHERE-clause
adopter restriction). Produces 8,760 rows per county per query instead of 14M.
Trade-off is 4 queries per county per MP (baseline-adopters, baseline-non-adopters,
upgrade-adopters, and a whole-county baseline if needed for total sanity-check)
instead of 2.

If BSQ's `group_by=[]` doesn't drop bldg_id (Option C), check docs for
`group_by=['time']` or verify via a test query on Allegheny before redesigning.

## Appendix B — Cleanup items (P2, do last)

Only address after Tasks 1–5 pass their verification contracts:

1. Prune unused imports (matplotlib, most of `kpi_functions`, unused `constants`).
   Add `# for Step 10` comments if they'll be needed for visualization.
2. Replace `input()` fallback in Step 0c with env-var resolution behind an explicit
   `INTERACTIVE_MODE = False` flag.
3. Suppress BSQ metadata SQL dump in Step 1; replace with column count.
4. Name the scenario constants: `SCENARIO_100PCT` and `SCENARIO_CONSTRAINED`.
5. Suppress repeated botocore credential-found INFO logs.
6. Add a module docstring to the top of the `.py` file.
7. Prettier scenario labels in Step 7 print output for paper figures.

## Session Summary Template

When the session ends, produce a summary covering:

1. **Which tasks completed** (Task 1 / 2 / 3 / 4 / 5) and verification outcomes
2. **Golden-value drift**, if any — any deviation from the reference table and why
3. **New reference values** produced for MP4 (tier distribution, Allegheny peak, etc.)
4. **Step 8 result** — profile-peak vs EUSS-metadata-sum ratio, pass/fail
5. **Step 9 design choice** actually used (A/B/C) and rationale
6. **Remaining work** — open items for a follow-up session
7. **Any new anti-patterns discovered** for future plan versions
