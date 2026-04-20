# Notebook Audit — `calculate_postTARE_ts_aws_peak_demand_20April2026`

**Auditor:** Claude (notebook-refactor-audit skill)
**Date:** April 20, 2026
**Artifacts reviewed:** `.py` export + `.pdf` cell-output export
**Prior context:** Post-TARE peak load analysis for Joseph et al. 2026, Energy Policy submission

---

## Per-step status

| Step | Status | Exec time | Notes |
|---|---|---|---|
| 0 — Imports | ✅ Done | n/a | Large block of unused imports (see P2) |
| 0b — MP selection | ✅ Done | n/a | `selected_mps = [3, 4]` — both MPs selected |
| 0c — Load TARE data | ✅ Done | n/a | PDF confirms **both** MP3 and MP4 loaded: `Loaded TARE data for MPs: [3, 4]` |
| 1 — BSQ init | ✅ Done | ~2 s | Init succeeds; long metadata SQL dump is style-noise |
| 2 — Column constants | ✅ Done | n/a | Imported cleanly |
| 3 — County shapefile | ✅ Done | ~2 s | 3,235 counties; Allegheny (42003) confirmed |
| 4 — Extract adopter IDs | ⚠️ **Partial — silently drops MP4** | n/a | `primary_mp = selected_mps[0]` = 3 only |
| 5 — Baseline TS query | ✅ Done | **174.54 s** | 1,610 bldgs × 8,760 hr = 14.1M rows; 1.18 GB from S3 |
| 6 — Upgrade TS query | ✅ Done | **175.95 s** | Same shape; MP3 only (driven by `primary_mp`) |
| 7 — Compute scenario profile | ✅ Done | <1 s | MP3 output only; magnitudes warrant scrutiny |
| 8 — EUSS validation | ❌ Stub | n/a | `NotImplementedError` fires |
| 9 — National loop | ❌ Stub | n/a | Function skeleton only |
| 10 — Export | ❌ Stub | n/a | `NotImplementedError` fires |

---

## Library log audit

**1. `INFO:botocore.credentials:Found credentials in shared credentials file`** (×5+)
AWS SDK discovering local creds; repeated because BSQ creates multiple boto3 sessions during init. Benign. Suppress via `logging.getLogger('botocore').setLevel(logging.WARNING)` for cleaner output.

**2. `INFO:buildstock_query.query_core:Loading resstock_amy2018_release_1_1 …`**
BSQ loading Glue catalog metadata. Benign.

**3. `WARNING:buildstock_query.query_core:Column bldg_id found in multiple tables [...]. Using ..._by_state`** (×2 per query)
BSQ resolving an ambiguous column reference across joined aliases and picking the timeseries table. This is BSQ's normal behavior — the timeseries table is the correct choice. No action needed; worth a one-line reassuring comment in Step 5.

**4. `INFO:buildstock_query.aggregate_query:Restricting query to Upgrade 0.` / `Upgrade 3.`**
BSQ confirming the scope filter. Benign and useful for provenance.

**5. `INFO:pyathena.pandas.result_set:Reading 1184663251 bytes from S3 in full mode using c engine`**
**The most important log line in the PDF.** 1.18 GB of S3 data transferred per query, per county, per upgrade. At national scale this is 7–15 TB — see P1 scaling.

---

## Cross-reference findings

### 1. 🔴 MP4 is loaded but never used — root cause of the "only MP3 results" symptom

Causal chain:

- Line 110: `selected_mps: list[int] = [3, 4]` ✓
- Lines 151–155: loop correctly loads both `DATAFRAMES_BY_MP[3]` and `DATAFRAMES_BY_MP[4]`. PDF confirms `Loaded TARE data for MPs: [3, 4]`.
- **Line 373: `primary_mp: int = selected_mps[0]`** collapses the list to a scalar `3`.
- Every reference from Step 4 onward uses `primary_mp`:
  - Line 382: `DATAFRAMES_BY_MP[primary_mp]` → MP3 only
  - Line 405: `find_adoption_column(df_tare, primary_mp, cs)` → MP3's adoption column
  - Line 546: `upgrade_id=str(primary_mp)` → Athena queries upgrade=3 only
  - Line 640: `f"Allegheny peak results (MP{primary_mp})"` → prints "MP3"

There is **no loop over `selected_mps` after Step 0c**. MP4 is loaded to memory, then silently abandoned.

**Impact:** The paper's central thesis is *technology-differentiated* electrification. Comparing standard ASHP (MP3) against high-efficiency ASHP (MP4) is the mechanism that drives the bill impact ratio finding. Producing only MP3 output defeats the differentiation argument.

**Fix (architectural):** Wrap Steps 4–7 in a per-MP function, or iterate `for mp in selected_mps:` over the relevant cells. Baseline (Step 5) should be pulled *outside* the MP loop since it's MP-independent.

### 2. 🟡 Peak magnitudes merit a sanity check before scaling

Step 7 output:
- Baseline peak: **862.51 MW** (1,610 buildings)
- 100% adoption peak: **6,629.87 MW** (+5,767 MW delta)
- Step 6 `retrofit_kwh` max per bldg-hour bucket: **25,086.95 kWh**

That max-hour value / BSQ weight (242.13) = ~103.6 kWh real consumption per building in a single hour, i.e., ~103 kW peak per home. Even a fully-cold-stuck heat pump on 100% resistance backup tops out around 15–25 kW per home.

Possibilities:
- (a) This bldg_id bucket is a manufactured-home outlier with dramatic resistance spikes
- (b) Units/aggregation issue (e.g., 15-min-to-hour rollup double-counting)
- (c) Real extreme-cold-snap coincident peak

**You can't tell without Step 8.** Flagging as P0/P1 hybrid: only a bug if Step 8 disagrees, but scaling nationally before Step 8 runs would propagate a potentially-wrong number across 3,098 counties.

### 3. ⚠️ Peak-hour drift between scenarios (constrained hour 116 vs 100pct hour 152)

Both are in the first week of January — heating-driven winter peak. Normal small-signal behavior: when the adopter set shrinks ~17× (1,610 → 93), the argmax location can move by tens of hours even when magnitudes are consistent. **Not a bug.** Worth noting in methods that peak-hour is unstable but peak-magnitude is the reportable quantity.

### 4. ⚠️ Baseline redundantly queried per MP under any naive refactor

Step 5 queries upgrade=0, which is MP-independent. When Steps 4–7 are wrapped in the MP loop to fix finding #1, baseline must stay *outside* the loop — otherwise ~175 s and 1.18 GB are wasted per extra MP.

---

## Structural findings

1. **Unused imports** — `matplotlib.pyplot`, `matplotlib.colors`, most of `kpi_functions` (`load_euss_baseline`, `compute_spark_gap_metrics`, `aggregate_demand_by_state`, `FUEL_PRICES_PATH`, etc.), and many `constants` imports (`ALLOWED_HOUSING_TYPES`, `VERBOSE`, `COUNTY_COL`, `STATE_COL`, `WEIGHT_COL`, `TEST_GISJOIN`, `RCM_MODELS`, `PRIVATE_DISCOUNT_RATE_SHORT_KEYS`). Some are plausibly needed for Step 10 (visualization). Trim or comment-mark `# for Step 10`.

2. **`input()` fallback in Step 0c (lines 145–146)** contradicts Step 0b's own comment ("replaced with constants for non-interactive runs"). Make it fully non-interactive via env vars or a config dict; keep `input()` only behind an explicit `INTERACTIVE_MODE = True` flag.

3. **Step 5 and Step 6 are near-duplicates** — identical TSQuery shape, differing only in `upgrade_id` and output column name. Extract a helper — you will need it for Step 9 anyway.

4. **Scenario keys `"100pct"` and `"constrained"` are magic strings.** Minor now; will bite at Step 10 when the export has to reference them in filenames and figure captions.

5. **BSQ metadata SQL dump** (~200 lines of column names printed in Step 1) is visual noise. Replace with a column-count print.

6. **Type annotations missing** on `test_row`, `df_tare_nested`, `tier_counts`, `n_counties`, `n_hours_up`. Notebook-level so low-priority, but matters if any of this migrates to a module.

---

## Performance profile

| Step | Operation | Time (s) | Naive national projection | Binding resource |
|---|---|---|---|---|
| 0c | Load TARE CSVs | not measured | Once per MP | Disk I/O |
| 1 | BSQ init | ~2 | Once | Network (AWS) |
| 3 | Shapefile load | ~2 | Once | Disk I/O |
| 4 | Extract adopter IDs | not measured | Once per MP | CPU (pandas) |
| **5** | Baseline TS query | **174.54** | Once (MP-independent): ~175 s × 3,098 counties ≈ **150 hours** sequential | **Network I/O + Athena compute**; 1.18 GB S3 read |
| **6** | Upgrade TS query | **175.95** | Per MP: ~150 hours × 2 MPs ≈ **300 hours** sequential | Same; 1.18 GB × 3,098 × 2 ≈ **7.3 TB of S3 reads** |
| 7 | Pandas mask + max | <1 | Trivial per county | CPU + memory |

**The pull-everything-and-aggregate-in-pandas pattern does not scale.** Naive sequential runtime is ~450 hours for a national 2-MP run with ~10+ TB of S3 egress.

### Design decision required before Step 9 is written

- **Option A (recommended): push aggregation into Athena.** Change the TSQuery to `group_by=['time']` (drop `bldg_id`) with the adopter set passed as a `WHERE bldg_id IN (...)` restriction. Athena returns 8,760 rows per county instead of 14M. Trade-off: separate queries for adopter/non-adopter subsets per county per scenario (so the mask is applied in SQL). That's 4 queries per county per MP instead of 2, but each is 1/1600 the data.
- **Option B: batch by state, post-filter in pandas.** One state's timeseries per query, filter per-county in memory. Reduces query count from ~3,098 to ~50 but keeps multi-TB egress.
- **Option C: check BSQ's native county group-by.** If BSQ supports `group_by=['in.county']` with a bldg_id restrict list for the adopter subset, that's the cleanest path. Verify before designing around A or B.

Regardless of A/B/C: **checkpointing per state is non-negotiable** — a single network blip restarts from zero otherwise.

---

## Prioritized issue list

### P0 — Correctness

1. **MP4 silently discarded after Step 0c.** Paper's technology-differentiated framing requires both MPs. Verify fix: notebook prints separate tier distributions for MP3 and MP4, separate peak deltas, and writes two output CSVs (or one CSV with an `mp` column).
2. **Peak magnitudes unvalidated.** Retrofit peak of 6.6 GW and per-bldg max of ~103 kW/hour are plausible but unverified. Verify: implement Step 8 comparing profile-derived Allegheny baseline peak against `SUM(out.electricity.winter.peak.kw)` from EUSS metadata for the same bldg_ids; agreement within 20% clears it.

### P1 — Scaling

3. **Pull-and-aggregate moves 1.18 GB per county per query.** At national scale: 7–15 TB of S3 egress for a single MP pair. Redesign Step 9 to aggregate in Athena (preferred), batch by state, or verify BSQ has a native county group-by.
4. **~175 s per query, sequential.** Even with Option A's 8,760-row response, 3,098 counties × 4 queries × N seconds is the dominant cost. Parallelize at the *state* level (respecting Athena concurrent-query limits, typically 10–50).
5. **Baseline will be redundantly queried per MP** once the MP loop is added. Query baseline *once* outside the loop.
6. **No checkpointing.** Step 9 must write per-state intermediate CSVs so a crash doesn't restart the full run.

### P2 — Code Quality

7. Step 5/6 near-duplicates — extract `query_county_hourly_electricity` helper (also required by Step 9).
8. Large block of unused imports.
9. `input()` fallback in Step 0c contradicts the non-interactive-runs comment.
10. `primary_mp = selected_mps[0]` idiom hides the multi-MP bug; delete once the loop is in.
11. Scenario keys `"100pct"` / `"constrained"` are magic strings.
12. BSQ metadata column dump in Step 1 is ~200 lines of noise.
13. Type hints missing on several locals.

### P3 — Nice-to-have

14. Suppress repeated botocore credential-found INFO logs.
15. Prettier display labels for paper figures (`"100% Adoption"`, `"Tier 1+2 Only"`).
16. Add a module docstring to the `.py` export.

---

## Recommended next steps (ordered)

1. **Fix the MP loop first.** Highest-value change; structurally prerequisite to everything else.
2. **Implement Step 8 against the MP3 test case.** If Step 8 fails the 20% agreement check, all subsequent effort is moot.
3. **Resolve the Step 9 aggregation-location decision (A/B/C)** before writing Step 9 code — it changes the function signature.
4. **Then Step 9 (with checkpointing) and Step 10.**
5. **Cleanup pass last** — unused imports, dead comments, duplicate cells. Don't refactor working code mid-implementation.
