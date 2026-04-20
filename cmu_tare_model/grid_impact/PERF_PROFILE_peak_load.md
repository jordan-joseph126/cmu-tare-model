# Task 5: Performance Profiling — Peak Load Analysis Pipeline

**Scope:** Steps 0–7 of `calculate_postTARE_ts_aws_peak_demand.ipynb`
**Date:** 2026-07-10
**Golden reference:** Allegheny County (FIPS 42003), MP3

---

## 5A. Timing Table — Allegheny County Single-County Run

| Step | Description | Wall-clock (s) | Rows returned | Bottleneck |
|------|-------------|---------------:|:--------------|------------|
| 0    | Imports + data load | ~5 | — | Disk I/O |
| 1    | BSQ init + AWS auth | ~3 | — | STS call |
| 2    | Constants (now imports) | <0.01 | — | — |
| 3    | County shapefile load | ~2 | 3,235 counties | `gpd.read_file()` |
| 4    | Adoption extraction | ~1 | 3,098 counties | `gisjoin_to_fips()` vectorized |
| **5** | **Baseline timeseries (BSQ)** | **149.54** | **14,103,600** | **Athena query + transfer** |
| **6** | **Upgrade timeseries (BSQ)** | **132.63** | **14,103,600** | **Athena query + transfer** |
| 7    | `compute_county_scenario_profile()` ×2 | ~2 (est.) | 8,760 per profile | merge + groupby |
| | **Total** | **~295** | | |

**Dominant cost:** Steps 5+6 are **95.5%** of total wall-clock time.
Pandas computation in Step 7 is negligible (~0.7%).

---

## 5B. BSQ Aggregation Options

### Current approach
```python
TSQuery(
    enduses=[ELEC_TOTAL_COL],
    restrict=[('bldg_id', allegheny_bldg_ids)],  # WHERE bldg_id IN (1610 values)
    group_by=[BLDG_ID_COL],                       # Returns per-building rows
    ...
)
```
- Returns: `1,610 × 8,760 = 14,103,600` rows
- Each row: `[bldg_id, timestamp, sample_count, units_count, rows_per_sample, electricity...]`

### Can we skip `group_by=[BLDG_ID_COL]`?

**No** — for the constrained scenario, we need building-level data to apply the
adopter mask (only Tier 1+2 buildings get retrofit kwh). Pre-aggregating in SQL
would lose the ability to distinguish adopters from non-adopters.

For the 100% adoption scenario, pre-aggregation WOULD work
(`group_by=[]` → SQL SUM → 8,760 rows), but this is only useful for one of
two scenarios per county.

### Critical architectural insight

The EUSS timeseries table is **`resstock_amy2018_release_1_1_by_state`** — the data
is **partitioned by state** in S3. The current `restrict=[('bldg_id', [...1610 ids...])]`
generates a `WHERE bldg_id IN (...)` clause that forces Athena to scan partitions
without pruning. State-level queries should enable partition pruning.

---

## 5C. National Loop — Path Analysis

### Baseline: Path A (per-county queries)
- **3,098 counties** × 2 queries × ~141 s/query = **873,636 s ≈ 242 hours ≈ 10 days**
- **INFEASIBLE.** Also: SQL `WHERE bldg_id IN (...)` clause would hit Athena limits
  for large counties.

### Recommended: Path B (per-state queries)

**Strategy:** Query all buildings in a state at once, then split/mask in pandas.

```
States with adopter data: ~50
Queries: 50 states × 2 (baseline + upgrade) = 100 Athena queries
```

**Per-state workflow:**
1. Collect all `bldg_id`s from `adopter_ids_by_county` for counties in this state
2. `TSQuery(restrict=[('bldg_id', state_bldg_ids)], group_by=[BLDG_ID_COL], ...)`
3. BSQ returns (n_state_buildings × 8,760) rows for baseline and upgrade
4. In pandas: add county FIPS via `gisjoin_to_fips()`, then per-county:
   - mask adopters with `compute_county_scenario_profile()`
   - aggregate to county-level peak results
5. Checkpoint state results to Parquet

**Estimated timing:**

| Metric | Allegheny (actual) | Avg state (projected) |
|--------|-------------------:|----------------------:|
| Buildings | 1,610 | ~6,631 |
| Rows per query | 14.1M | ~58.1M |
| Query time (est.) | 141 s | ~580 s (linear scale) |
| Pandas compute | ~2 s | ~8 s |
| Per-state total | — | ~1,168 s |
| **National total** | — | **~58,400 s ≈ 16.2 hours** |

**Improvement over Path A: ~15× faster** (16 hrs vs 242 hrs).

### Path C (alternative — metadata-based restrict)

Instead of enumerating `bldg_id`s in SQL, use:
```python
restrict=[('in.county', county_gisjoin_list)]
```
BSQ would filter via metadata JOIN, and Athena's optimizer may push the state
partition filter down. This eliminates the huge `IN (...)` clause.

**Risk:** Untested — needs BSQ behavior verification. If BSQ can filter by
`in.county` on the timeseries query, this could enable even more efficient
per-state batching with `restrict=[('in.state', state_code)]`.

---

## 5D. Cross-Cutting Optimizations

### D1. Drop BSQ bookkeeping columns immediately
```python
# After BSQ returns, drop columns not needed downstream
df.drop(columns=['sample_count', 'units_count', 'rows_per_sample'],
        inplace=True, errors='ignore')
```
**Impact:** ~37.5% memory reduction per DataFrame (3 of 8 columns).

### D2. Parquet checkpoints per state
```python
checkpoint_path = f"checkpoints/peak_load_state_{state_code}.parquet"
df_state_results.to_parquet(checkpoint_path, index=False)
```
**Why:** If the national loop fails at state 35/50, resume from the
checkpoint instead of re-running 35 states (~11 hours of queries).

### D3. Pre-compute adopter mask as a column
Instead of calling `compute_county_scenario_profile()` with separate adopter
lists per county, add the mask to the DataFrame once per state:
```python
df['is_adopter_constrained'] = df[BLDG_ID_COL].isin(all_state_constrained_ids)
df['is_adopter_100pct'] = df[BLDG_ID_COL].isin(all_state_filtered_ids)
```
Then groupby `(county, hour)` with conditional aggregation:
```python
df.groupby(['county_fips', 'hour']).agg(
    baseline_kwh=('baseline_kwh', 'sum'),
    scenario_100pct_kwh=('scenario_kwh_100pct', 'sum'),
    scenario_constrained_kwh=('scenario_kwh_constrained', 'sum'),
)
```
**Impact:** Eliminates the per-county merge; county-level aggregation happens
in a single `groupby` pass over the state data. Reduces pandas compute from
~8 s/state to ~2 s/state.

### D4. Memory: process one state at a time, then discard

```python
for state_code in states:
    df_baseline = query_baseline(state_code)  # ~58M rows
    df_upgrade = query_upgrade(state_code)    # ~58M rows
    results = compute_all_counties(df_baseline, df_upgrade, ...)
    save_checkpoint(results, state_code)
    del df_baseline, df_upgrade   # Free ~4.5 GB
    gc.collect()
```
**Peak memory per state:** ~4.5 GB (two DataFrames × 58M rows × 4 bytes × ~5 cols).
Manageable on a 16 GB machine.

### D5. float32 already applied ✓
Steps 5 and 6 now downcast to float32 immediately after BSQ returns.
Annual savings at national scale: ~2.3 GB vs float64.

---

## 5E. Recommendation Summary

| # | Recommendation | Priority | Impact |
|---|----------------|----------|--------|
| **R1** | **Per-state batching** (Path B) | **MUST** | 15× faster (16 hrs vs 242 hrs) |
| **R2** | Parquet checkpoints per state | MUST | Crash recovery; resume from last state |
| **R3** | Drop BSQ bookkeeping columns | SHOULD | 37.5% less memory per DF |
| **R4** | Pre-compute adopter mask at state level | SHOULD | Fewer merge ops, single groupby |
| **R5** | Test `restrict=[('in.county', ...)]` | COULD | Avoid huge `IN (...)` clauses |
| **R6** | Test `restrict=[('in.state', ...)]` | COULD | Partition pruning → faster queries |
| **R7** | Add timing instrumentation to Step 7 | SHOULD | Verify compute is negligible at scale |

### Implementation order for Step 9

1. Build state → county mapping from `adopter_ids_by_county + county_geo_df`
2. Loop over states (sorted alphabetically for progress tracking)
3. Query baseline + upgrade timeseries per state (R1)
4. Add county FIPS column via `gisjoin_to_fips()` on BSQ metadata
5. Apply adopter masks (R4) and compute county profiles in one `groupby`
6. Save state-level checkpoint (R2)
7. Concat all state checkpoints → `df_peak_results_national`

### Data volume sanity check

| Metric | Value |
|--------|-------|
| Total EUSS buildings | 548,916 |
| Buildings with adopter data | 331,531 |
| Counties with adopters | 3,098 |
| States (est.) | ~50 |
| Avg buildings/state | ~6,631 |
| Avg rows/state query | ~58.1M |
| Bytes per state (baseline, float32, 5 cols) | ~1.1 GB |
| Bytes per state (baseline + upgrade) | ~2.3 GB |
| National total data transferred | ~115 GB |
| Output table | 3,098 rows × 12 cols |

---

## Appendix: Allegheny County Golden Values (Post-Edit Verification Pending)

| Metric | Expected |
|--------|----------|
| Baseline peak | 862.51 MW @ hour 4433 |
| 100% adoption peak | 6,629.87 MW @ hour 152 |
| 100% delta | +5,767.36 MW |
| Constrained peak | 885.63 MW @ hour 116 |
| Constrained delta | +23.12 MW |
| Row count (per query) | 14,103,600 |
| Buildings | 1,610 |
| Hours/building | 8,760 |

**Note:** float32 downcast may shift peak values by ±0.01 MW.
Re-run Steps 5–7 to verify post-edit golden values.
