# Notebook Audit — `calculate_postTARE_ts_aws_peak_demand`

> **Files reviewed:**
> - **Newer (current):** `calculate_postTARE_ts_aws_peak_demand_30April2026.py` + `.pdf`
> - **Older (with MP fixes):** `calculate_postTARE_ts_aws_peak_demand_21April2026_MP3_MP4.py` + `.pdf`
>
> The 30April version is the target notebook for refactoring. The 21April version is the reference
> for the multi-MP fixes that must be ported forward.

---

## Per-Step Status

| Step | Name | Status (30Apr) | Exec time (PDF) | Notes |
|------|------|---------------|-----------------|-------|
| 0 | Imports | ✅ Done | — | Import path changed vs. 21Apr; see P2-1 |
| 0b | MP Selection | ✅ Done | — | `selected_mps = [3, 4]`; `primary_mp` still set below |
| 0c | Load TARE Data | ✅ Done | ~few s | Loads both MP3 and MP4 correctly |
| 1 | BSQ Init | ✅ Done | — | Credentials valid; BSQ confirms weight 242.131013 |
| 2 | Column Constants | ✅ Done | — | Print-only; all constants imported from `constants.py` |
| 3 | County Geography | ✅ Done | — | 3,235 counties; Allegheny FIPS 42003 confirmed |
| 4 | Extract Adopter IDs | 🔴 Broken | — | **Reverted to single-MP `primary_mp` logic** — see P0-1 |
| 5 | Baseline Timeseries | ⚠️ Partial | 396 s | Runs but uses only MP3 bldg_ids — see P0-2 |
| 6 | Upgrade Timeseries | 🔴 Broken | 266 s | Single MP only; no `_by_mp` dict — see P0-3 |
| 7 | Scenario Profiles | 🔴 Broken | — | References wrong variable names; no `_by_mp` storage — see P0-4 |
| 8 | Validate vs EUSS | ❌ Stub | — | `NotImplementedError` raised explicitly |
| 9 | National Loop | ❌ Stub | — | `NotImplementedError`; signature still uses `primary_mp` — see P1-2 |
| 10 | Export Results | ❌ Stub | — | `NotImplementedError`; blocked by Step 9 |

> ⚠️ Steps 5–7 all show `query_time > 30s`. Each county takes ~400s for baseline + ~270s per upgrade. At 3,098 counties this is a serious national-scale constraint — flagged under P1.

---

## Library Log Audit

### `WARNING:buildstock_query.query_core:Column bldg_id found in multiple tables`
**Appears in:** Steps 5 and 6 (both baseline and upgrade queries), all runs.
**What it means:** BSQ resolves the `bldg_id` column ambiguity — it appears in the timeseries table, the baseline alias, and the upgrade alias. BSQ picks the timeseries table automatically.
**Problem?** No. This is BSQ's normal join-disambiguation. The correct table is selected (`resstock_amy2018_release_1_1_by_state`).
**Action needed:** None. Consider suppressing with `logging.getLogger("buildstock_query").setLevel(logging.ERROR)` at the top of the notebook if it's distracting.

### `INFO:buildstock_query.aggregate_query:Restricting query to Upgrade N`
**Appears in:** Steps 5 and 6.
**What it means:** BSQ confirms the upgrade_id filter is applied before executing SQL. This is informational.
**Problem?** No.
**Action needed:** None.

### `INFO:pyathena.pandas.result_set:Reading N bytes from S3 in full mode using c engine`
**Appears in:** Steps 5 and 6.
**What it means:** PyAthena is downloading the full result set from S3 in one pass using the C parquet engine. "Full mode" means no streaming/chunking — the entire result lands in memory before pandas sees it.
**Problem?** At county scale (1,184 MB for Allegheny baseline) this is acceptable. At national scale (3,098 counties × ~1 GB each), full-mode S3 reads are the binding resource constraint. This should inform the Step 9 batching strategy.
**Action needed:** Flag in Step 9 design: batch by state to reduce round-trips and memory pressure.

### `INFO:botocore.credentials:Found credentials in shared credentials file: ~/.aws/credentials`
**Appears in:** Step 1 (multiple times during BSQ init).
**What it means:** boto3 is reading AWS credentials from the shared credential file at each client instantiation.
**Problem?** No. Normal credential resolution behavior.
**Action needed:** None.

---

## Cross-Reference Findings (Code vs. PDF Output)

### Finding 1 — Step 4: Regression from multi-MP to single-MP
**Code (30Apr Step 4):** Sets `primary_mp = selected_mps[0]`, then processes only `DATAFRAMES_BY_MP[primary_mp]`, producing `adopter_ids_by_county` (singular flat dict).

**Code (21Apr Step 4):** Loops over `selected_mps`, producing `adopter_ids_by_mp: dict[int, dict[str, dict[str, list[int]]]]` and `adoption_col_by_mp: dict[int, str]` — both indexed by MP.

**PDF output (21Apr):** Confirms the loop ran for both MP3 and MP4 with correct tier distributions and Allegheny County counts:
- MP3 constrained: 93 buildings; all filtered: 1,610
- MP4 constrained: 518 buildings; all filtered: 1,610

**PDF output (30Apr):** Only shows MP3 tier distribution and counts. MP4 adopter data is never extracted.

**Consequence:** Steps 5–7 in the 30Apr notebook cannot produce MP4 peak profiles. The 2×2 visualization cannot be built.

### Finding 2 — Step 5: bldg_id union regression
**Code (30Apr):** `allegheny_bldg_ids = adopter_ids_by_county[TEST_FIPS]["all_filtered"]` — draws from the single-MP `adopter_ids_by_county` (MP3 only).

**Code (21Apr):** `allegheny_bldg_ids = sorted(set().union(*[adopter_ids_by_mp[mp][TEST_FIPS]["all_filtered"] for mp in selected_mps]))` — correctly takes the union across all MPs before querying.

**Why the union matters:** The baseline query is MP-independent (upgrade_id='0'), so it should cover every building that could adopt under *any* scenario. In Allegheny County both MPs happen to share 1,610 buildings, so the 30Apr numeric outputs are accidentally correct here — but in counties where MPs have different building populations this will silently drop buildings.

**PDF evidence:** Both versions show 1,610 buildings × 8,760 hours = 14,103,600 rows. The 30Apr happened to produce the same result for Allegheny by coincidence.

### Finding 3 — Step 6: Single-MP upgrade query regression
**Code (30Apr):** One `TSQuery` with `upgrade_id=str(primary_mp)` → one result stored as `df_ts_upgrade_allegheny` (single DataFrame).

**Code (21Apr):** Loops over `selected_mps`, stores `df_ts_upgrade_allegheny_by_mp: dict[int, pd.DataFrame]`.

**PDF (30Apr):** Shows only one upgrade query (MP3, 265 s). MP4 upgrade is never queried.

**PDF (21Apr):** Shows both MP3 (213 s) and MP4 (152 s) upgrade queries, both passing schema parity checks.

### Finding 4 — Step 7: Wrong variable names and no `_by_mp` storage
**Code (30Apr):** References `adopter_ids_by_county[TEST_FIPS]` (which no longer exists as structured — Step 4 never creates this) and `primary_mp`. Stores results as `peak_results_allegheny` (flat dict) and `df_profile_100pct`, `df_profile_constrained` (single variables).

**Code (21Apr):** Loops over `selected_mps`, references `adopter_ids_by_mp[mp][TEST_FIPS]`, stores all results in `peak_results_allegheny_by_mp: dict[int, dict[str, dict]]`.

**Critical consequence:** The 2×2 visualization requires `df_profile_by_mp[mp]["100pct"]` and `df_profile_by_mp[mp]["constrained"]` for mp ∈ {3, 4}. The 30Apr structure cannot feed this.

**PDF (21Apr) golden values — confirmed correct:**
| MP | Scenario | Baseline Peak (MW) | Scenario Peak (MW) | Peak Hr (baseline) | Peak Hr (scenario) | Delta (MW) |
|----|----------|--------------------|--------------------|--------------------|--------------------|------------|
| 3 | 100% | 862.51 | 6629.87 | 4433 | 152 | +5767.36 |
| 3 | Constrained | 862.51 | 885.63 | 4433 | 116 | +23.12 |
| 4 | 100% | 862.51 | 5364.10 | 4433 | 152 | +4501.59 |
| 4 | Constrained | 862.51 | 2016.92 | 4433 | 152 | +1154.41 |

### Finding 5 — Step 9 stub: `primary_mp` in function signature
**Code (both versions):** `run_national_peak_load_loop(..., primary_mp: int, ...)` — still uses `primary_mp` even in the 21Apr version.

**Consequence:** When Step 9 is implemented, the signature must be updated to accept `selected_mps: list[int]` and loop internally. The current stub docstring also references `primary_mp` as if single-MP.

---

## Structural Findings

- **Dead code — `primary_mp` variable**: Set at the top of Step 4 in the 30Apr version. After porting the 21Apr loop, `primary_mp` should be removed entirely. It resurfaces in Step 7 print statements and Step 9 stub.
- **Unused imports**: `matplotlib.colors as mcolors` — imported but not referenced in any current cell. Can be removed or moved to the future visualization step.
- **Unused imports**: `create_npv_col`, `create_capital_col` from `column_names` — not used in peak demand analysis. Likely carried over from the parent notebook.
- **Stale Step 6 markdown heading**: "MP3 or MP4" in the 30Apr markdown should read "all selected MPs" once the multi-MP loop is ported.
- **Import module path divergence**: 30Apr splits `kpi_functions` imports into `adoption_kpis` (top-level) and `adoption_kpis.data_loading`. The 21Apr uses `adoption_kpis.kpi_functions`. This reflects a real module refactor. The 30Apr import paths should be treated as authoritative for the live codebase, but must be verified against `__init__.py` before accepting.
- **`input()` calls remain in Step 0c**: The `input()` fallback for `location_id` and `model_run_date_time` is still present. These should be documented constants at the top of the notebook for full reproducibility.
- **Missing `df_profile_by_mp` storage in Step 7**: The 30Apr version displays `df_profile_100pct` and `df_profile_constrained` as bare `display()` calls but never stores them in a dict. The visualization requires these to be indexed by MP.

---

## Performance Profile

| Step | Operation | Time (30Apr PDF) | Time (21Apr PDF) | Binding Resource | Notes |
|------|-----------|-----------------|-----------------|-----------------|-------|
| 5 | Baseline BSQ query (1,610 bldgs) | 396 s | 166 s | Network I/O / S3 read (1.18 GB) | 30Apr was cold cache; 21Apr likely warm |
| 6 | Upgrade BSQ query MP3 (1,610 bldgs) | 266 s | 213 s | Network I/O / S3 read (1.18 GB) | |
| 6 | Upgrade BSQ query MP4 (1,610 bldgs) | N/A (not run) | 152 s | Network I/O / S3 read (1.18 GB) | |

**National-scale projection:** At 3,098 counties, even assuming 2 min/county (optimistic with state-level batching), total runtime ≈ 103 hours for one MP. Two MPs = ~206 hours. State-level batching in Step 9 is not optional — it is the critical path item.

---

## Prioritized Issue List

### P0 — Correctness

**P0-1: Step 4 reverted to single-MP logic (critical)**
- **What:** 30Apr Step 4 uses `primary_mp` and creates `adopter_ids_by_county` (flat dict). The 21Apr multi-MP loop creating `adopter_ids_by_mp` was not ported forward.
- **Why it matters:** MP4 adoption data is never extracted. All downstream steps silently produce MP3-only results. The 2×2 visualization is structurally impossible with 30Apr code.
- **Where:** Step 4, lines 369–444 (30Apr .py)
- **How to verify:** After fix, print `list(adopter_ids_by_mp.keys())` → should be `[3, 4]`. Check Allegheny constrained counts: MP3=93, MP4=518.

**P0-2: Step 5 bldg_id union uses single-MP variable**
- **What:** `allegheny_bldg_ids = adopter_ids_by_county[TEST_FIPS]["all_filtered"]` references the now-incorrect single-MP variable instead of `sorted(set().union(*[...]))` across all MPs.
- **Why it matters:** Silently drops buildings unique to one MP's filtered set in counties where MPs differ. Will produce wrong baseline profiles at national scale.
- **Where:** Step 5, line 466 (30Apr .py)
- **How to verify:** Print `len(allegheny_bldg_ids)` → 1,610 for Allegheny (happens to be correct here, but verify with union logic).

**P0-3: Step 6 single-MP upgrade query, no `_by_mp` storage**
- **What:** Only MP3 upgrade is queried. No `df_ts_upgrade_allegheny_by_mp` dict is created.
- **Why it matters:** Cannot compute MP4 scenario profiles. Visualization Step requires per-MP DataFrames.
- **Where:** Step 6, lines 532–593 (30Apr .py)
- **How to verify:** After fix, `list(df_ts_upgrade_allegheny_by_mp.keys())` → `[3, 4]`. Both should show 1,610 buildings × 8,760 hours.

**P0-4: Step 7 wrong variable names and no `_by_mp` profile storage**
- **What:** References `adopter_ids_by_county` (wrong) instead of `adopter_ids_by_mp`. Stores profiles as `df_profile_100pct`/`df_profile_constrained` (single variables) rather than in a dict indexed by MP.
- **Why it matters:** Step 7 will crash on `adopter_ids_by_county` KeyError (variable doesn't exist after P0-1 fix). Even if patched, visualization cannot index profiles by MP.
- **Where:** Step 7, lines 616–654 (30Apr .py)
- **How to verify:** After fix, `list(peak_results_allegheny_by_mp.keys())` → `[3, 4]`. Confirm golden values from table above.

### P1 — Scaling

**P1-1: Step 9 stub signature uses `primary_mp: int`**
- **What:** `run_national_peak_load_loop` takes `primary_mp: int`. When implemented, must take `selected_mps: list[int]` and loop internally per MP.
- **Why it matters:** Embedding a single-MP assumption in the function signature will propagate the regression into the national loop.
- **Where:** Step 9 stub function definition (both versions)
- **How to verify:** Update signature; update docstring `Args` section.

**P1-2: National loop — no state-level batching strategy decided**
- **What:** Step 9 design decisions (query batching, aggregation location, checkpointing) are unresolved. At ~1 GB per county query, the naive per-county loop will be prohibitively slow.
- **Why it matters:** Without state-level batching, the national run is likely 100+ hours and will fail mid-run without checkpointing.
- **Where:** Step 9 (stub)
- **How to verify:** Design decision must be documented before implementation begins.

### P2 — Code Quality

- **P2-1:** Import path divergence between 30Apr (`adoption_kpis` + `adoption_kpis.data_loading`) and 21Apr (`adoption_kpis.kpi_functions`). Verify against live module `__init__.py` before proceeding.
- **P2-2:** `primary_mp` variable should be fully removed after the multi-MP loop is ported. It currently appears in Step 4 (set), Step 7 print (referenced), and Step 9 stub signature.
- **P2-3:** Step 6 markdown heading says "MP3 or MP4" — update to "all selected MPs" to match loop behavior.
- **P2-4:** `df_profile_100pct` and `df_profile_constrained` display cells in Step 7 should be updated to display per-MP profiles (e.g., loop over `df_profiles_by_mp[mp]["100pct"]`).
- **P2-5:** `input()` calls in Step 0c should be replaced with hardcoded constants at the top of the notebook with a clear comment explaining the values. Reproducibility requirement.

### P3 — Nice-to-Have

- **P3-1:** Remove unused imports: `matplotlib.colors as mcolors`, `create_npv_col`, `create_capital_col`. These are not needed in this notebook.
- **P3-2:** Add `logging.getLogger("buildstock_query").setLevel(logging.ERROR)` to suppress the recurring `bldg_id` WARNING during BSQ queries.
- **P3-3:** The 30Apr Step 6 query time (396 s) vs. 21Apr (166 s) suggests cold vs. warm Athena cache. Add a note that first runs will be significantly slower.

---

## Recommended Next Steps

1. **Port P0-1 first** (Step 4 multi-MP loop from 21Apr). Everything else is blocked on having `adopter_ids_by_mp` correctly populated.
2. **Fix P0-2** (Step 5 union) immediately after — it's a one-liner but correctness-critical at national scale.
3. **Fix P0-3 and P0-4 together** (Steps 6 and 7 loops + `_by_mp` dicts) — they are tightly coupled and should be ported as a unit from the 21Apr version.
4. **Run to Step 7 and validate golden values** before proceeding to visualization.
5. **Implement visualization** (2×2 grid) using `df_profiles_by_mp` and `peak_results_allegheny_by_mp`.
6. **Defer Steps 8–10** — Step 8 validation and Step 9 national loop are separate workstreams.
